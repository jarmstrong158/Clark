"""ClarkActorCritic: Transformer + LSTM hybrid actor-critic for variable N workers, M tasks.

Architecture overview:
  Encoder (per-step, no temporal memory):
    W = WorkerLinear(worker_feats) + RoleEmbed(role_ids)       → (N, d_model)
    T = TaskLinear(task_feats)     + TaskTypeEmbed(type_ids)   → (M, d_model)
    E = EnvLinear(env_feats)                                   → (d_model,)
    W = W + E                                                   broadcast env into workers
    W = SelfAttentionLayer(W) × n_sa_layers                    workers attend to each other
    W = CrossAttentionLayer(W, T)  × n_ca_layers               workers attend to tasks

  Temporal memory:
    g = W.mean(dim=0)                                          pool across workers → (d_model,)
    h_t, c_t = LSTM(g, (h_prev, c_prev))                      temporal context
    W_final = W + h_t                                          broadcast temporal into workers

  Outputs:
    assignment_logits = W_final @ T.T                          (N, M) — pre-softmax
    hustle_logits     = HustleHead(W_final)                    (N, 2) — pre-softmax
    value             = ValueHead(h_t)                         scalar
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional

ARCH_VERSION = "clark-v1"

# ── Hyperparameter defaults ───────────────────────────────────────────────────

_DEFAULT_D_MODEL = 256
_DEFAULT_N_HEADS = 8
_DEFAULT_N_SA_LAYERS = 2
_DEFAULT_N_CA_LAYERS = 1
_DEFAULT_LSTM_HIDDEN = 256
_DEFAULT_DROPOUT = 0.1

# Vocab sizes
_NUM_ROLES = 4       # manager=0, assistant_manager=1, lead=2, warehouse=3
_MAX_TASKS = 20      # len(STANDARD_VOCAB)=12 + 5 custom + 3 buffer

# Feature dimensions (mirrors FacilityEnv constants)
_WORKER_FEAT_DIM = 13
_TASK_FEAT_DIM = 3   # demand_signal, availability_flag, task_type_id
_ENV_FEAT_DIM = 15


# ─────────────────────────────────────────────────────────────────────────────
# Attention building blocks
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with pre-LN residual."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_scale = math.sqrt(self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, d_model). Returns (N, d_model)."""
        residual = x
        x = self.norm(x)             # pre-LN

        N, D = x.shape
        qkv = self.qkv(x)            # (N, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)  # each (N, D)

        # Reshape to (n_heads, N, d_head)
        q = q.view(N, self.n_heads, self.d_head).transpose(0, 1)  # (H, N, dh)
        k = k.view(N, self.n_heads, self.d_head).transpose(0, 1)
        v = v.view(N, self.n_heads, self.d_head).transpose(0, 1)

        scores = torch.bmm(q, k.transpose(1, 2)) / self.attn_scale  # (H, N, N)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        attended = torch.bmm(weights, v)  # (H, N, dh)
        attended = attended.transpose(0, 1).contiguous().view(N, D)  # (N, D)

        out = self.out_proj(attended)
        out = self.dropout(out)
        return residual + out


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention: Q from workers, K/V from tasks. Pre-LN residual."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm_q = nn.LayerNorm(d_model)  # pre-LN on queries
        self.dropout = nn.Dropout(dropout)
        self.attn_scale = math.sqrt(self.d_head)

    def forward(self, workers: torch.Tensor, tasks: torch.Tensor) -> torch.Tensor:
        """
        workers: (N, d_model) — queries
        tasks:   (M, d_model) — keys / values
        Returns: (N, d_model)
        """
        residual = workers
        workers_ln = self.norm_q(workers)

        N, D = workers_ln.shape
        M = tasks.shape[0]

        q = self.q_proj(workers_ln).view(N, self.n_heads, self.d_head).transpose(0, 1)  # (H, N, dh)
        k = self.k_proj(tasks).view(M, self.n_heads, self.d_head).transpose(0, 1)       # (H, M, dh)
        v = self.v_proj(tasks).view(M, self.n_heads, self.d_head).transpose(0, 1)       # (H, M, dh)

        scores = torch.bmm(q, k.transpose(1, 2)) / self.attn_scale  # (H, N, M)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        attended = torch.bmm(weights, v)  # (H, N, dh)
        attended = attended.transpose(0, 1).contiguous().view(N, D)  # (N, D)

        out = self.out_proj(attended)
        out = self.dropout(out)
        return residual + out


class FeedForward(nn.Module):
    """Position-wise feed-forward: d_model → 4*d_model → d_model. Pre-LN residual."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class TransformerBlock(nn.Module):
    """Self-attention + feed-forward (one full transformer encoder layer)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.sa = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sa(x)
        x = self.ff(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# ClarkActorCritic
# ─────────────────────────────────────────────────────────────────────────────

class ClarkActorCritic(nn.Module):
    """
    Transformer + LSTM hybrid actor-critic.

    Handles variable N workers and M tasks at runtime via dynamic attention.
    The LSTM provides temporal memory across the full shift.

    forward() takes a single-step state dict and returns raw logits (no softmax).
    select_action() samples from those logits and returns log-probs for PPO.
    """

    def __init__(
        self,
        worker_feat_dim: int = _WORKER_FEAT_DIM,
        task_feat_dim: int = _TASK_FEAT_DIM,
        env_feat_dim: int = _ENV_FEAT_DIM,
        num_roles: int = _NUM_ROLES,
        max_task_types: int = _MAX_TASKS,
        d_model: int = _DEFAULT_D_MODEL,
        n_heads: int = _DEFAULT_N_HEADS,
        n_sa_layers: int = _DEFAULT_N_SA_LAYERS,
        n_ca_layers: int = _DEFAULT_N_CA_LAYERS,
        lstm_hidden: int = _DEFAULT_LSTM_HIDDEN,
        dropout: float = _DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden

        # ── Input projections ─────────────────────────────────────────────────
        # task_feat_dim includes the task_type_id column; continuous part is dim-1
        self.worker_linear = nn.Linear(worker_feat_dim, d_model)
        self.task_linear = nn.Linear(task_feat_dim - 1, d_model)  # exclude task_type_id column
        self.env_linear = nn.Linear(env_feat_dim, d_model)

        # ── Embeddings ────────────────────────────────────────────────────────
        # Fixed vocab sizes — config-agnostic so weights transfer across facilities
        self.role_embed = nn.Embedding(num_roles, d_model)
        self.task_type_embed = nn.Embedding(max_task_types, d_model)

        # ── Encoder: Self-Attention layers ────────────────────────────────────
        self.sa_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_sa_layers)
        ])

        # ── Encoder: Cross-Attention layers ───────────────────────────────────
        self.ca_layers = nn.ModuleList([
            MultiHeadCrossAttention(d_model, n_heads, dropout)
            for _ in range(n_ca_layers)
        ])
        # Feed-forward after each cross-attention
        self.ca_ff = nn.ModuleList([
            FeedForward(d_model, dropout)
            for _ in range(n_ca_layers)
        ])

        # ── Temporal memory ───────────────────────────────────────────────────
        self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers=1, batch_first=False)

        # Project lstm_hidden → d_model if sizes differ (they're equal by default,
        # but keep this for safety if someone passes different values)
        if lstm_hidden != d_model:
            self.lstm_proj = nn.Linear(lstm_hidden, d_model)
        else:
            self.lstm_proj = nn.Identity()

        # ── Output heads ──────────────────────────────────────────────────────
        self.hustle_head = nn.Linear(d_model, 2)
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.Tanh(),
            nn.Linear(lstm_hidden // 2, 1),
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "lstm" in name:
                if "weight" in name:
                    nn.init.orthogonal_(p, gain=1.0)
                elif "bias" in name:
                    nn.init.constant_(p, 0.0)
            elif "embed" in name:
                nn.init.normal_(p, std=0.02)
            elif isinstance(p, nn.Parameter) and p.dim() >= 2:
                nn.init.orthogonal_(p, gain=math.sqrt(2))
        # Small init for output heads to keep initial policy close to uniform
        for module in [self.hustle_head]:
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0.0)
        for module in self.value_head:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    # ── Hidden state ─────────────────────────────────────────────────────────

    def init_hidden(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zeroed (h0, c0) for a fresh episode."""
        return (
            torch.zeros(1, batch_size, self.lstm_hidden),
            torch.zeros(1, batch_size, self.lstm_hidden),
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        state_dict: dict,
        action_mask: Optional[np.ndarray] = None,
        lstm_hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """
        Single-step forward pass.

        Args:
            state_dict:   Output of StateBuilder.build() — numpy arrays.
            action_mask:  (N, M) bool ndarray. True = valid action.
                          If None, all actions are valid.
            lstm_hidden:  (h, c) tuple from previous step. If None, zeros used.

        Returns:
            assignment_logits: (N, M)  — raw logits, masked but NOT softmaxed
            hustle_logits:     (N, 2)  — raw logits, NOT softmaxed
            value:             ()      — scalar
            new_hidden:        (h, c)  LSTM state after this step
        """
        if lstm_hidden is None:
            lstm_hidden = self.init_hidden(batch_size=1)

        device = next(self.parameters()).device

        # ── Unpack inputs ─────────────────────────────────────────────────────
        worker_feats = torch.tensor(
            state_dict["worker_feats"], dtype=torch.float32, device=device
        )  # (N, 13)
        task_feats_raw = torch.tensor(
            state_dict["task_feats"], dtype=torch.float32, device=device
        )  # (M, 3)
        env_feats = torch.tensor(
            state_dict["env_feats"], dtype=torch.float32, device=device
        )  # (15,)
        worker_role_ids = torch.tensor(
            state_dict["worker_role_ids"], dtype=torch.long, device=device
        )  # (N,)
        task_type_ids = torch.tensor(
            state_dict["task_type_ids"], dtype=torch.long, device=device
        )  # (M,)

        # Separate task continuous features from the type-id column
        # task_feats columns: [demand_signal, availability_flag, task_type_id(float)]
        task_feats_cont = task_feats_raw[:, :2]  # (M, 2) — continuous only

        # ── Encoder ───────────────────────────────────────────────────────────
        # Worker tokens
        W = self.worker_linear(worker_feats)              # (N, d_model)
        W = W + self.role_embed(worker_role_ids)          # (N, d_model) sum

        # Task tokens — clamp type_ids to vocab size
        task_type_ids_safe = task_type_ids.clamp(
            0, self.task_type_embed.num_embeddings - 1
        )
        T = self.task_linear(task_feats_cont)             # (M, d_model)
        T = T + self.task_type_embed(task_type_ids_safe)  # (M, d_model) sum

        # Env broadcast into worker tokens
        E = self.env_linear(env_feats)                    # (d_model,)
        W = W + E.unsqueeze(0)                            # (N, d_model)

        # Self-attention layers — workers attend to each other
        for sa_layer in self.sa_layers:
            W = sa_layer(W)                               # (N, d_model)

        # Cross-attention layers — workers attend to tasks
        for ca_attn, ca_ff in zip(self.ca_layers, self.ca_ff):
            W = ca_attn(W, T)                             # (N, d_model)
            W = ca_ff(W)                                  # (N, d_model)

        # ── Temporal memory (LSTM) ────────────────────────────────────────────
        g = W.mean(dim=0, keepdim=True)                   # (1, d_model)
        # LSTM expects (seq_len, batch, input_size) — seq_len=1, batch=1
        g_lstm = g.unsqueeze(0)                           # (1, 1, d_model)
        lstm_out, new_hidden = self.lstm(g_lstm, lstm_hidden)
        h_t = lstm_out.squeeze(0).squeeze(0)              # (lstm_hidden,)

        # Broadcast temporal context back into worker tokens
        h_proj = self.lstm_proj(h_t)                      # (d_model,)
        W_final = W + h_proj.unsqueeze(0)                 # (N, d_model)

        # ── Outputs ───────────────────────────────────────────────────────────
        # Task assignment logits: (N, M)
        assignment_logits = W_final @ T.t()               # (N, M)

        # Apply action mask: invalid actions get -1e9
        if action_mask is not None:
            mask_t = torch.tensor(action_mask, dtype=torch.bool, device=device)
            assignment_logits = assignment_logits.masked_fill(~mask_t, -1e9)

        # Hustle logits: (N, 2)
        hustle_logits = self.hustle_head(W_final)

        # Value: scalar
        value = self.value_head(h_t.unsqueeze(0)).squeeze()  # ()

        return assignment_logits, hustle_logits, value, new_hidden

    # ── Action selection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        state_dict: dict,
        action_mask: Optional[np.ndarray] = None,
        lstm_hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[list[int], list[int], torch.Tensor, torch.Tensor, tuple]:
        """
        Sample actions for one environment step.

        Args:
            state_dict:   Output of StateBuilder.build().
            action_mask:  (N, M) bool ndarray. True = valid task.
            lstm_hidden:  Previous LSTM hidden state.

        Returns:
            task_actions:   list[int] length N — task index per worker
            hustle_actions: list[int] length N — 0 or 1 per worker
            log_probs:      scalar tensor — sum of log-probs across all workers (for PPO)
            value:          scalar tensor
            new_hidden:     (h, c) updated LSTM state
        """
        assignment_logits, hustle_logits, value, new_hidden = self.forward(
            state_dict, action_mask, lstm_hidden
        )

        task_actions = []
        hustle_actions = []
        log_prob_sum = torch.tensor(0.0, device=assignment_logits.device)

        N = assignment_logits.shape[0]
        for i in range(N):
            # Task assignment
            task_dist = Categorical(logits=assignment_logits[i])
            task_idx = task_dist.sample()
            task_actions.append(task_idx.item())
            log_prob_sum = log_prob_sum + task_dist.log_prob(task_idx)

            # Hustle decision
            hustle_dist = Categorical(logits=hustle_logits[i])
            hustle_flag = hustle_dist.sample()
            hustle_actions.append(hustle_flag.item())
            log_prob_sum = log_prob_sum + hustle_dist.log_prob(hustle_flag)

        return task_actions, hustle_actions, log_prob_sum, value, new_hidden

    # ── Entropy (for PPO entropy bonus) ───────────────────────────────────────

    def entropy(
        self,
        assignment_logits: torch.Tensor,
        hustle_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total entropy across all workers (task + hustle distributions).

        Args:
            assignment_logits: (N, M)
            hustle_logits:     (N, 2)

        Returns:
            scalar tensor — sum of entropies across all workers
        """
        N = assignment_logits.shape[0]
        total_entropy = torch.tensor(0.0, device=assignment_logits.device)
        for i in range(N):
            total_entropy = total_entropy + Categorical(logits=assignment_logits[i]).entropy()
            total_entropy = total_entropy + Categorical(logits=hustle_logits[i]).entropy()
        return total_entropy
