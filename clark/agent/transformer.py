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

ARCH_VERSION = "clark-v2.5"
# v2.5 (2026-05): added env_feat 17 = mgmt_backlog_norm. The v2.7
# reward-shaping iteration (per_ot_hour -1.5 -> -5.0) successfully
# eliminated OT-driven downgrades but exposed an invisible failure
# mode: 80% of resulting C-grade days were hit by management_backlog
# demerit, which is a multi-day rolling state the policy literally
# could not observe in env_feats. v2.5 adds a normalized backlog
# observation (clipped [0,1], 1.0 = demerit threshold) so the policy
# can reason about and prevent the multi-day mgmt failure. env_feat_dim
# 17 -> 18; only env_linear's input width changes, every other
# parameter is bit-identical to v2/v2.7, so trained weights warm-start
# with a zero-init 18th column. See tools/transplant_obs_extension.py.
# All v2.6 mask gates and v2.7 reward weights are PRESERVED — v2.5 is
# additive to the existing structural interventions, not a replacement.

# ── Hyperparameter defaults ───────────────────────────────────────────────────
# v2 bump (2026-04): d_model 256→512, n_sa_layers 2→4, lstm_hidden 256→512.
# At v1 the model was 2.9M params and the GPU was kernel-launch-bound for a
# batched-forward workload — compute per kernel was so small that launch
# overhead dominated. v2 is ~18M params: forward compute now dominates on
# modern GPUs and the Tier-2 batched path finally gets its full speedup.

_DEFAULT_D_MODEL = 512
_DEFAULT_N_HEADS = 8
_DEFAULT_N_SA_LAYERS = 4
_DEFAULT_N_CA_LAYERS = 1
_DEFAULT_LSTM_HIDDEN = 512
_DEFAULT_DROPOUT = 0.0  # See PPO compatibility note below.
# Dropout is disabled because PPO's importance-sampling correction relies on
# old_log_prob and new_log_prob being computed from the SAME stochastic policy.
# With dropout > 0, every forward call resamples masks, so the ratio
# exp(new_log_prob - old_log_prob) drifts on the order of clip_epsilon (0.2)
# even when the underlying weights have not changed — saturating PPO's clip
# threshold and silencing the learning signal. The model is small and
# regularised by the diversity of synthetic facility configs during pretrain;
# explicit dropout buys nothing here and breaks PPO correctness.

# Vocab sizes
_NUM_ROLES = 4       # manager=0, assistant_manager=1, lead=2, warehouse=3
_MAX_TASKS = 20      # len(STANDARD_VOCAB)=12 + 5 custom + 3 buffer
                    # If you bump this, retrain — it changes the
                    # task_type_embed shape and invalidates checkpoints.
                    # FacilityConfig.validate() rejects configs with
                    # num_tasks > _MAX_TASKS to prevent silent embedding
                    # collisions at the clamp in forward().

# Feature dimensions — must match WORKER_STATE_SCALARS and ENV_STATE_SIZE in facility_env.py
_WORKER_FEAT_DIM = 14  # 13 base scalars + task_oph_normalized (index 13)
_TASK_FEAT_DIM = 3     # demand_signal, availability_flag, task_type_id
_ENV_FEAT_DIM = 18     # 17 prior + mgmt_backlog_norm (index 17, v2.5)


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

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (N, d_model) or (B, N, d_model). Returns same shape.

        key_padding_mask: optional (B, N) bool tensor. True = VALID, False = pad.
        Pad positions are excluded from attention via -inf fill before softmax.
        Only meaningful when x has a batch dimension.
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)           # → (1, N, D)

        B, N, D = x.shape
        H, dh = self.n_heads, self.d_head

        residual = x
        x = self.norm(x)                 # (B, N, D)
        qkv = self.qkv(x)               # (B, N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, N, D)

        # Reshape to (B*H, N, dh) for batched matmul
        q = q.view(B, N, H, dh).permute(0, 2, 1, 3).reshape(B * H, N, dh)
        k = k.view(B, N, H, dh).permute(0, 2, 1, 3).reshape(B * H, N, dh)
        v = v.view(B, N, H, dh).permute(0, 2, 1, 3).reshape(B * H, N, dh)

        scores = torch.bmm(q, k.transpose(1, 2)) / self.attn_scale  # (B*H, N, N)

        if key_padding_mask is not None:
            # Mask shape: (B, N). Broadcast to (B*H, 1, N) so padded *keys* are
            # excluded from attention for every query in that batch element.
            # True = valid → keep; False = pad → mask to -inf.
            m = key_padding_mask.unsqueeze(1).unsqueeze(1)              # (B, 1, 1, N)
            m = m.expand(B, H, 1, N).reshape(B * H, 1, N)               # (B*H, 1, N)
            scores = scores.masked_fill(~m, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        # If an ENTIRE row was masked (a padded query attending to all-padded
        # keys) softmax produces NaNs. Clean them up — those rows will be
        # discarded by downstream masking anyway.
        if key_padding_mask is not None:
            weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)

        attended = torch.bmm(weights, v)                              # (B*H, N, dh)
        attended = attended.reshape(B, H, N, dh).permute(0, 2, 1, 3).reshape(B, N, D)

        out = self.out_proj(attended)
        out = self.dropout(out)
        result = residual + out
        return result.squeeze(0) if squeeze else result


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

    def forward(
        self,
        workers: torch.Tensor,
        tasks: torch.Tensor,
        task_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        workers: (N, d_model) or (B, N, d_model) — queries
        tasks:   (M, d_model) or (B, M, d_model) — keys / values

        task_key_padding_mask: optional (B, M) bool tensor. True = valid task,
        False = pad. Padded tasks are excluded from cross-attention.

        Returns: same shape as workers
        """
        squeeze = workers.dim() == 2
        if squeeze:
            workers = workers.unsqueeze(0)
            tasks = tasks.unsqueeze(0)

        B, N, D = workers.shape
        M = tasks.shape[1]
        H, dh = self.n_heads, self.d_head

        residual = workers
        workers_ln = self.norm_q(workers)

        q = self.q_proj(workers_ln).view(B, N, H, dh).permute(0, 2, 1, 3).reshape(B * H, N, dh)
        k = self.k_proj(tasks).view(B, M, H, dh).permute(0, 2, 1, 3).reshape(B * H, M, dh)
        v = self.v_proj(tasks).view(B, M, H, dh).permute(0, 2, 1, 3).reshape(B * H, M, dh)

        scores = torch.bmm(q, k.transpose(1, 2)) / self.attn_scale  # (B*H, N, M)

        if task_key_padding_mask is not None:
            m = task_key_padding_mask.unsqueeze(1).unsqueeze(1)         # (B, 1, 1, M)
            m = m.expand(B, H, 1, M).reshape(B * H, 1, M)               # (B*H, 1, M)
            scores = scores.masked_fill(~m, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        if task_key_padding_mask is not None:
            weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)

        attended = torch.bmm(weights, v)                              # (B*H, N, dh)
        attended = attended.reshape(B, H, N, dh).permute(0, 2, 1, 3).reshape(B, N, D)

        out = self.out_proj(attended)
        out = self.dropout(out)
        result = residual + out
        return result.squeeze(0) if squeeze else result


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


# NOTE: PopArtValueHead was tried earlier in development and removed —
# the multi-task design didn't fit our single-policy / per-day update /
# single-trajectory-buffer setting (μ/σ oscillated, value head chased
# moving targets). Replaced by EMA running mean/var return normalization
# in ppo.py (_ret_mean / _ret_var). The legacy checkpoint migration in
# ClarkAgent.load() handles loading any old PopArt-shaped state dicts.

class TransformerBlock(nn.Module):
    """Self-attention + feed-forward (one full transformer encoder layer)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.sa = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.sa(x, key_padding_mask=key_padding_mask)
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
        # Assignment logits come from W_final @ T.T. With both factors at
        # gain=√2 orthogonal init, the raw matmul output magnitude scales
        # with √d_model (~22 at d_model=512) — producing near-deterministic
        # softmax at init and immediate entropy collapse.
        # Fix: divide by √d_model (same trick attention uses for its scores)
        # so initial logits are O(1), softmax is well-tempered, and the
        # model has room to commit OR back off as training progresses.
        # No learnable scaler — that just throttled gradients in either
        # direction (small init = frozen, large init = saturated).
        self._assign_scale: float = 1.0 / math.sqrt(d_model)
        self.hustle_head = nn.Linear(d_model, 2)
        # Simple 2-layer MLP value head. PopArt was tried and removed
        # (see note above the class). Return normalization for the value
        # loss is handled by an EMA running mean/var in ppo.py
        # (_ret_mean / _ret_var) — stable across the per-day correlated
        # TBPTT chunks where per-batch standardization was failing.
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
        """Return zeroed (h0, c0) for a fresh episode, on the model's device."""
        # Resolve device from the model's own parameters so callers don't have
        # to pass it in. Works for CPU, CUDA, and any future backends.
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return (
            torch.zeros(1, batch_size, self.lstm_hidden, device=device),
            torch.zeros(1, batch_size, self.lstm_hidden, device=device),
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        state_dict: dict,
        action_mask: Optional[np.ndarray] = None,
        lstm_hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        hustle_mask: Optional[np.ndarray] = None,
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
        # Small-init scale (0.05) keeps initial logits ~0.5–1.5, so softmax
        # is near-uniform at epoch 0 and entropy starts high. The policy
        # learns to grow this scalar as it commits to specific assignments.
        assignment_logits = self._assign_scale * (W_final @ T.t())   # (N, M)

        # Apply action mask: invalid actions get -1e9
        if action_mask is not None:
            mask_t = torch.tensor(action_mask, dtype=torch.bool, device=device)
            assignment_logits = assignment_logits.masked_fill(~mask_t, -1e9)

        # Hustle logits: (N, 2)
        hustle_logits = self.hustle_head(W_final)
        if hustle_mask is not None:
            hmask_t = torch.tensor(hustle_mask, dtype=torch.bool, device=device)
            hustle_logits = hustle_logits.masked_fill(~hmask_t, -1e9)

        # Value: scalar.
        value = self.value_head(h_t.unsqueeze(0)).squeeze()  # ()

        return assignment_logits, hustle_logits, value, new_hidden

    # ── Batched sequence forward (PPO update path) ────────────────────────────

    def forward_sequence(
        self,
        state_dicts: list,
        action_masks: list,
        lstm_hidden: tuple[torch.Tensor, torch.Tensor],
        hustle_masks: Optional[list] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """
        Batched forward over T sequential steps for the PPO update.

        The encoder (SA + CA) processes all T steps simultaneously by treating T
        as a batch dimension. The LSTM processes the T pooled tokens as a single
        sequence in one call — mathematically identical to T sequential forward()
        calls, but far faster because PyTorch's LSTM kernel is optimised for
        sequences and the attention ops are batched.

        Args:
            state_dicts:  list[dict] of length T — StateBuilder.build() outputs
            action_masks: list[ndarray|None] of length T — (N, M) bool per step
            lstm_hidden:  (h, c) at the start of this chunk

        Returns:
            assign_logits: (T, N, M) — masked, pre-softmax
            hustle_logits: (T, N, 2) — pre-softmax
            values:        (T,)
            entropies:     (T,) — summed over workers, for PPO entropy bonus
            new_hidden:    (h, c) after processing all T steps
        """
        T = len(state_dicts)
        device = next(self.parameters()).device

        # Stack all T state dicts into batch tensors
        worker_feats = torch.tensor(
            np.stack([sd["worker_feats"] for sd in state_dicts]),
            dtype=torch.float32, device=device,
        )  # (T, N, worker_feat_dim)
        task_feats_raw = torch.tensor(
            np.stack([sd["task_feats"] for sd in state_dicts]),
            dtype=torch.float32, device=device,
        )  # (T, M, 3)
        env_feats = torch.tensor(
            np.stack([sd["env_feats"] for sd in state_dicts]),
            dtype=torch.float32, device=device,
        )  # (T, env_feat_dim)
        worker_role_ids = torch.tensor(
            np.stack([sd["worker_role_ids"] for sd in state_dicts]),
            dtype=torch.long, device=device,
        )  # (T, N)
        task_type_ids = torch.tensor(
            np.stack([sd["task_type_ids"] for sd in state_dicts]),
            dtype=torch.long, device=device,
        )  # (T, M)

        task_feats_cont = task_feats_raw[:, :, :2]  # (T, M, 2) — drop task_type_id col

        # ── Encoder (all T steps batched) ─────────────────────────────────────
        W = self.worker_linear(worker_feats)                          # (T, N, d_model)
        W = W + self.role_embed(worker_role_ids)                      # (T, N, d_model)

        task_type_ids_safe = task_type_ids.clamp(
            0, self.task_type_embed.num_embeddings - 1
        )
        T_tokens = self.task_linear(task_feats_cont)                  # (T, M, d_model)
        T_tokens = T_tokens + self.task_type_embed(task_type_ids_safe)  # (T, M, d_model)

        E = self.env_linear(env_feats)                                # (T, d_model)
        W = W + E.unsqueeze(1)                                        # (T, N, d_model)

        for sa_layer in self.sa_layers:
            W = sa_layer(W)                                           # (T, N, d_model)

        for ca_attn, ca_ff in zip(self.ca_layers, self.ca_ff):
            W = ca_attn(W, T_tokens)                                  # (T, N, d_model)
            W = ca_ff(W)                                              # (T, N, d_model)

        # ── LSTM as a sequence (one call for all T steps) ─────────────────────
        g = W.mean(dim=1)                                             # (T, d_model)
        # nn.LSTM batch_first=False expects (seq_len, batch, input_size)
        # g is (T, d_model); unsqueeze(1) → (T, 1, d_model): seq_len=T, batch=1
        lstm_out, new_hidden = self.lstm(g.unsqueeze(1), lstm_hidden)  # (T, 1, lstm_hidden)
        h_t = lstm_out.squeeze(1)                                     # (T, lstm_hidden)

        # Broadcast temporal context into worker tokens
        W_final = W + self.lstm_proj(h_t).unsqueeze(1)               # (T, N, d_model)

        # ── Outputs ───────────────────────────────────────────────────────────
        # Assignment logits via batched matmul: (T, N, d) @ (T, d, M) → (T, N, M)
        assign_logits = self._assign_scale * torch.bmm(
            W_final, T_tokens.transpose(1, 2)
        )  # (T, N, M) — small-init scale; see forward() for rationale.

        # Apply per-step action masks
        N = W_final.shape[1]
        M = T_tokens.shape[1]
        full_mask = torch.ones(T, N, M, dtype=torch.bool, device=device)
        for t_idx, mask in enumerate(action_masks):
            if mask is not None:
                full_mask[t_idx] = torch.tensor(mask, dtype=torch.bool, device=device)
        assign_logits = assign_logits.masked_fill(~full_mask, -1e9)

        hustle_logits = self.hustle_head(W_final)                     # (T, N, 2)

        # Apply per-step hustle masks (only steps that have one)
        if hustle_masks is not None:
            hfull = torch.ones(T, N, 2, dtype=torch.bool, device=device)
            for t_idx, hm in enumerate(hustle_masks):
                if hm is not None:
                    hfull[t_idx] = torch.tensor(hm, dtype=torch.bool, device=device)
            hustle_logits = hustle_logits.masked_fill(~hfull, -1e9)

        # value_head returns (T, 1); use .reshape(-1) instead of .squeeze(-1)
        # so a single-step chunk (T==1) doesn't collapse to a 0-d tensor
        # which would break torch.cat downstream.
        values = self.value_head(h_t).reshape(-1)                     # (T,)

        # Vectorised entropy: MEAN over workers (not sum) so the entropy
        # bonus is invariant to N. With sum-over-workers, entropy term
        # scaled with N and dwarfed the policy gradient (entropy ~7 vs
        # policy_loss ~0.005), pushing the optimizer toward "spread the
        # policy" instead of "improve it". Per-worker mean keeps the
        # bonus magnitude consistent across N=3 vs N=10 facilities.
        task_lp = F.log_softmax(assign_logits, dim=-1)                # (T, N, M)
        task_ent = -(task_lp.exp() * task_lp).sum(dim=-1)            # (T, N)
        hustle_lp = F.log_softmax(hustle_logits, dim=-1)              # (T, N, 2)
        hustle_ent = -(hustle_lp.exp() * hustle_lp).sum(dim=-1)      # (T, N)
        entropies = (task_ent + hustle_ent).mean(dim=-1)              # (T,) — per-worker mean

        return assign_logits, hustle_logits, values, entropies, new_hidden

    # ── Batched multi-env forward (Tier-2 fast rollout) ──────────────────────

    def forward_batched(
        self,
        state_dicts: list,
        action_masks: list,
        lstm_hidden: tuple[torch.Tensor, torch.Tensor],
        hustle_masks: Optional[list] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, list, tuple]:
        """
        Single-step forward over B parallel environments with variable N_i, M_i.

        This is the Tier-2 hot path: one batched kernel launch feeds data from
        all envs through the model at once. Per-env shapes are padded to
        `max_N × max_M` and key_padding_masks exclude the pad positions from
        attention.

        Args:
            state_dicts: list[dict] of length B — one StateBuilder output per env
            action_masks: list[ndarray|None] length B — (N_i, M_i) valid-action masks
            lstm_hidden: (h, c) each shaped (1, B, lstm_hidden) — one slot per env

        Returns:
            assign_logits: (B, max_N, max_M) — pad positions set to -1e9
            hustle_logits: (B, max_N, 2)
            values:        (B,)
            n_workers_list: list[int] length B — per-env real worker count
            n_tasks_list:   list[int] length B — per-env real task count
            new_hidden:    (h, c) each (1, B, lstm_hidden)
        """
        device = next(self.parameters()).device
        B = len(state_dicts)

        # Extract real per-env sizes.
        n_workers_list = [int(sd["worker_feats"].shape[0]) for sd in state_dicts]
        n_tasks_list   = [int(sd["task_feats"].shape[0])   for sd in state_dicts]
        max_N = max(n_workers_list)
        max_M = max(n_tasks_list)

        worker_feat_dim = state_dicts[0]["worker_feats"].shape[1]
        # Task continuous features exclude the task_type_id column (see forward())
        task_feat_cont_dim = state_dicts[0]["task_feats"].shape[1] - 1
        env_feat_dim = state_dicts[0]["env_feats"].shape[0]

        # Padded host-side buffers; one H→D transfer per tensor.
        wf_buf = np.zeros((B, max_N, worker_feat_dim), dtype=np.float32)
        tf_buf = np.zeros((B, max_M, task_feat_cont_dim), dtype=np.float32)
        ef_buf = np.zeros((B, env_feat_dim), dtype=np.float32)
        rid_buf = np.zeros((B, max_N), dtype=np.int64)
        ttid_buf = np.zeros((B, max_M), dtype=np.int64)
        worker_pad_np = np.zeros((B, max_N), dtype=bool)
        task_pad_np = np.zeros((B, max_M), dtype=bool)

        for b, sd in enumerate(state_dicts):
            Ni = n_workers_list[b]
            Mi = n_tasks_list[b]
            wf_buf[b, :Ni]   = sd["worker_feats"]
            # Drop the task_type_id column (index 2) from the continuous slice
            tf_buf[b, :Mi]   = sd["task_feats"][:, :task_feat_cont_dim]
            ef_buf[b]        = sd["env_feats"]
            rid_buf[b, :Ni]  = sd["worker_role_ids"]
            ttid_buf[b, :Mi] = sd["task_type_ids"]
            worker_pad_np[b, :Ni] = True
            task_pad_np[b, :Mi]   = True

        worker_feats = torch.from_numpy(wf_buf).to(device, non_blocking=True)
        task_feats_c = torch.from_numpy(tf_buf).to(device, non_blocking=True)
        env_feats    = torch.from_numpy(ef_buf).to(device, non_blocking=True)
        worker_role_ids = torch.from_numpy(rid_buf).to(device, non_blocking=True)
        task_type_ids   = torch.from_numpy(ttid_buf).to(device, non_blocking=True)
        worker_pad_mask = torch.from_numpy(worker_pad_np).to(device, non_blocking=True)
        task_pad_mask   = torch.from_numpy(task_pad_np).to(device, non_blocking=True)

        # ── Encoder ───────────────────────────────────────────────────────────
        W = self.worker_linear(worker_feats)                        # (B, max_N, d)
        W = W + self.role_embed(worker_role_ids)

        task_type_ids_safe = task_type_ids.clamp(
            0, self.task_type_embed.num_embeddings - 1
        )
        T_tokens = self.task_linear(task_feats_c)                   # (B, max_M, d)
        T_tokens = T_tokens + self.task_type_embed(task_type_ids_safe)

        # Zero out task embeddings at padded positions so they can't leak
        # contribution into the mean pool or downstream ops.
        T_tokens = T_tokens * task_pad_mask.unsqueeze(-1).to(T_tokens.dtype)

        E = self.env_linear(env_feats)                              # (B, d)
        W = W + E.unsqueeze(1)                                      # broadcast

        for sa_layer in self.sa_layers:
            W = sa_layer(W, key_padding_mask=worker_pad_mask)

        for ca_attn, ca_ff in zip(self.ca_layers, self.ca_ff):
            W = ca_attn(W, T_tokens, task_key_padding_mask=task_pad_mask)
            W = ca_ff(W)

        # Mean-pool workers with mask so padded slots don't bias the pool.
        wmask = worker_pad_mask.to(W.dtype).unsqueeze(-1)           # (B, max_N, 1)
        denom = wmask.sum(dim=1).clamp(min=1.0)                     # (B, 1)
        g = (W * wmask).sum(dim=1) / denom                          # (B, d)

        # LSTM expects (seq_len, batch, input_size) — seq_len=1, batch=B.
        lstm_out, new_hidden = self.lstm(g.unsqueeze(0), lstm_hidden)  # (1, B, lstm_hidden)
        h_t = lstm_out.squeeze(0)                                      # (B, lstm_hidden)

        W_final = W + self.lstm_proj(h_t).unsqueeze(1)              # (B, max_N, d)

        # Logits via batched matmul (B, max_N, d) @ (B, d, max_M) → (B, max_N, max_M)
        assign_logits = self._assign_scale * torch.bmm(
            W_final, T_tokens.transpose(1, 2)
        )  # small-init scale (see forward())

        # Apply combined mask: per-env action_mask (if given) AND exclude
        # padded N or M positions. Start from pad mask, then AND the action mask.
        valid = worker_pad_mask.unsqueeze(-1) & task_pad_mask.unsqueeze(1)  # (B,max_N,max_M)
        for b, am in enumerate(action_masks):
            if am is None:
                continue
            Ni = n_workers_list[b]
            Mi = n_tasks_list[b]
            am_t = torch.as_tensor(am, dtype=torch.bool, device=device)
            valid[b, :Ni, :Mi] &= am_t

        assign_logits = assign_logits.masked_fill(~valid, -1e9)

        hustle_logits = self.hustle_head(W_final)                   # (B, max_N, 2)

        # Apply per-env hustle masks. Padded worker rows stay unmasked since
        # they're never sampled (the per-env loop uses Ni from n_workers_list).
        if hustle_masks is not None:
            hvalid = torch.ones(B, max_N, 2, dtype=torch.bool, device=device)
            for b, hm in enumerate(hustle_masks):
                if hm is None:
                    continue
                Ni = n_workers_list[b]
                hm_t = torch.as_tensor(hm, dtype=torch.bool, device=device)
                hvalid[b, :Ni] = hm_t
            hustle_logits = hustle_logits.masked_fill(~hvalid, -1e9)

        values = self.value_head(h_t).reshape(-1)                   # (B,)

        return assign_logits, hustle_logits, values, n_workers_list, n_tasks_list, new_hidden

    # ── Batched action selection ─────────────────────────────────────────────

    @torch.no_grad()
    def select_action_batched(
        self,
        state_dicts: list,
        action_masks: list,
        lstm_hidden: tuple[torch.Tensor, torch.Tensor],
        hustle_masks: Optional[list] = None,
    ):
        """
        Sample actions for B parallel envs in one forward pass.

        Returns PER-WORKER log-probs as a list of per-env tensors — see
        select_action() docstring for why the joint sum-over-workers ratio is
        wrong for PPO.

        Returns:
            task_actions_list:    list[list[int]] length B
            hustle_actions_list:  list[list[int]] length B
            task_log_probs_list:  list[(N_b,) tensor] length B
            hustle_log_probs_list:list[(N_b,) tensor] length B
            values:               (B,) tensor
            new_hidden:           updated (h, c) LSTM state
        """
        assign_logits, hustle_logits, values, n_w_list, n_t_list, new_hidden = (
            self.forward_batched(
                state_dicts, action_masks, lstm_hidden, hustle_masks=hustle_masks,
            )
        )
        B = assign_logits.shape[0]

        task_actions_list: list[list[int]] = []
        hustle_actions_list: list[list[int]] = []
        task_log_probs_list: list[torch.Tensor] = []
        hustle_log_probs_list: list[torch.Tensor] = []

        # Vectorized sample over the full padded logits in ONE batched
        # Categorical, then ONE GPU->CPU transfer per head. The previous
        # per-env Python loop did 2*B blocking `.tolist()` GPU syncs per
        # tick (~235ms/tick vs 27ms of actual GPU compute — the dominant
        # training-throughput bottleneck). Categorical batches over the
        # leading (B, max_N) dims independently, so this is
        # distribution-identical to the per-env loop for real worker rows;
        # padded rows / padded task columns are -1e9-masked by
        # forward_batched (probability underflows to exactly 0, so they
        # can never be sampled and contribute nothing to logsumexp) and
        # are sliced off per env below. Equivalence is pinned by
        # tests/test_sampler_equivalence.py (exact log_prob match + matched
        # empirical distribution).
        task_dist = Categorical(logits=assign_logits)              # batch (B, max_N) over max_M
        task_idx_full = task_dist.sample()                         # (B, max_N)
        task_lp_full = task_dist.log_prob(task_idx_full)           # (B, max_N), on device

        hustle_dist = Categorical(logits=hustle_logits)            # batch (B, max_N) over 2
        hustle_full = hustle_dist.sample()                         # (B, max_N)
        hustle_lp_full = hustle_dist.log_prob(hustle_full)         # (B, max_N), on device

        # The only forced syncs: actions must reach the CPU env workers.
        # One transfer each instead of 2*B.
        task_idx_cpu = task_idx_full.tolist()                      # [[...max_N...] x B]
        hustle_cpu = hustle_full.tolist()

        for b in range(B):
            Ni = n_w_list[b]
            task_actions_list.append(task_idx_cpu[b][:Ni])
            hustle_actions_list.append(hustle_cpu[b][:Ni])
            # log-probs stay on device, sliced to real workers — same
            # shape/contract the per-env loop produced.
            task_log_probs_list.append(task_lp_full[b, :Ni])
            hustle_log_probs_list.append(hustle_lp_full[b, :Ni])

        return (
            task_actions_list, hustle_actions_list,
            task_log_probs_list, hustle_log_probs_list,
            values, new_hidden,
        )

    # ── Action selection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        state_dict: dict,
        action_mask: Optional[np.ndarray] = None,
        lstm_hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        hustle_mask: Optional[np.ndarray] = None,
    ) -> tuple[list[int], list[int], torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """
        Sample actions for one environment step.

        Returns PER-WORKER log-probs separately for the two action heads — NOT
        summed. PPO must compute the importance-sampling ratio per worker (and
        clip per worker), otherwise the joint ratio variance scales with N
        and PPO's clip threshold saturates regardless of weight changes.

        Returns:
            task_actions:    list[int] length N — task index per worker
            hustle_actions:  list[int] length N — 0 or 1 per worker
            task_log_probs:  (N,) tensor — per-worker log P(task_i)
            hustle_log_probs:(N,) tensor — per-worker log P(hustle_i)
            value:           scalar tensor
            new_hidden:      (h, c) updated LSTM state
        """
        assignment_logits, hustle_logits, value, new_hidden = self.forward(
            state_dict, action_mask, lstm_hidden, hustle_mask=hustle_mask,
        )

        task_dist = Categorical(logits=assignment_logits)   # batch_shape (N,)
        task_idx = task_dist.sample()                       # (N,)
        task_log_probs = task_dist.log_prob(task_idx)       # (N,)

        hustle_dist = Categorical(logits=hustle_logits)     # batch_shape (N,)
        hustle_flag = hustle_dist.sample()                  # (N,)
        hustle_log_probs = hustle_dist.log_prob(hustle_flag)  # (N,)

        task_actions = task_idx.tolist()
        hustle_actions = hustle_flag.tolist()

        return task_actions, hustle_actions, task_log_probs, hustle_log_probs, value, new_hidden

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
        # Vectorized: one Categorical per distribution with batch_shape (N,).
        task_entropy = Categorical(logits=assignment_logits).entropy().sum()
        hustle_entropy = Categorical(logits=hustle_logits).entropy().sum()
        return task_entropy + hustle_entropy
