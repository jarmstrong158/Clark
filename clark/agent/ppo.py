"""PPO training loop for Clark — variable-dimension transformer actor-critic.

Key differences from Jack's LSTM-only PPO:
  - ClarkActorCritic replaces ActorCritic
  - Config-agnostic model: same weights work across different facility configs
  - Two action heads: task assignment (N, M) and hustle (N, 2), both variable-dim
  - RolloutBuffer stores structured state dicts instead of flat numpy arrays
  - TBPTT with chunk_size=16 for gradient truncation through the LSTM
  - evaluate_sequence replays stored state_dicts step-by-step (can't vectorize
    across steps like Jack because N/M may vary, though within one episode N/M
    are constant)
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import asdict
from typing import TYPE_CHECKING, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.distributions import Categorical
from torch.optim import Adam

from clark.agent.transformer import ClarkActorCritic, ARCH_VERSION

if TYPE_CHECKING:
    from clark.config.schema import FacilityConfig


# ── Hyperparameter defaults ───────────────────────────────────────────────────

PPO_DEFAULTS: dict = {
    "lr": 5e-5,
    # gamma 0.999 + gae_lambda 0.98 give effective horizons of ~1000 and
    # ~50 steps respectively — sized for the 13050-step year. Default 0.99
    # / 0.95 left the agent blind to end-of-day rewards (decay to ~0.6 by
    # 9 AM choices) and effectively invisible to year-end ones (γ^13300 ≈
    # 10⁻⁵⁸). Standard PPO tuning recommendation for sparse-terminal RL.
    "gamma": 0.999,
    "gae_lambda": 0.98,
    "clip_epsilon": 0.2,
    # entropy_coeff bumped 0.02 → 0.10 to compensate for the entropy
    # sum→mean change in transformer.py. Old code summed per-worker entropy
    # so the bonus magnitude scaled with N (~8× at average N=8); new mean-
    # over-workers makes the bonus N-invariant but ~8× smaller in absolute
    # terms. Without this bump the policy committed to near-deterministic
    # actions within ~250 PPO updates and stopped exploring.
    "entropy_coeff": 0.10,
    "value_loss_coeff": 0.5,
    "max_grad_norm": 0.5,
    "epochs_per_update": 4,
    # chunk_size 16 → 64: a "day" in this sim spans many ticks. With chunk=16
    # the LSTM gradient was truncated 3-4 times per day, so the agent could
    # not learn that an 8AM pick-heavy assignment causes packer-starvation
    # at 11AM. 64 covers ~half a day's gradient horizon and reduces the
    # per-update batch correlation that was poisoning value-loss stats.
    "tbptt_chunk_size": 64,
}


# ─────────────────────────────────────────────────────────────────────────────
# RolloutBuffer
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores one rollout's worth of transitions for a sequential PPO update.

    Stores structured state_dicts rather than flat arrays, because the
    transformer takes dict inputs and the shapes (N, M) vary per config.
    """

    def __init__(self):
        self.states: list[dict] = []            # list of state_dicts
        self.task_actions: list[list[int]] = [] # list of len-N int lists
        self.hustle_actions: list[list[int]] = []
        self.action_masks: list[np.ndarray | None] = []  # (N, M) or None
        self.hustle_masks: list[np.ndarray | None] = []  # (N, 2) or None
        # PER-WORKER old log-probs: stored separately for the two action heads.
        # PPO computes the policy ratio per (step, worker, head) — summing
        # log-probs across workers (joint ratio) makes the variance scale with
        # N and saturates the clip threshold structurally.
        self.task_log_probs: list = []   # list of (N_t,) torch.Tensor
        self.hustle_log_probs: list = [] # list of (N_t,) torch.Tensor
        # Critic estimate is per-step (one scalar).
        self.values: list = []      # list of 0-D torch.Tensor
        self.rewards: list[float] = []
        self.dones: list[bool] = []

        # LSTM hidden state at the START of this rollout, for evaluate_sequence.
        self.entry_hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    def set_entry_hidden(self, hidden: tuple[torch.Tensor, torch.Tensor]):
        self.entry_hidden = (hidden[0].detach().clone(),
                             hidden[1].detach().clone())

    def add(
        self,
        state_dict: dict,
        task_actions: list[int],
        hustle_actions: list[int],
        task_log_prob,    # (N_t,) torch.Tensor — per-worker log P(task_i)
        hustle_log_prob,  # (N_t,) torch.Tensor — per-worker log P(hustle_i)
        value,            # 0-D torch.Tensor OR python float
        reward: float,
        done: bool,
        action_mask: np.ndarray | None = None,
        hustle_mask: np.ndarray | None = None,
    ):
        self.states.append(state_dict)
        self.task_actions.append(task_actions)
        self.hustle_actions.append(hustle_actions)
        self.task_log_probs.append(task_log_prob)
        self.hustle_log_probs.append(hustle_log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_masks.append(action_mask)
        self.hustle_masks.append(hustle_mask)

    def compute_returns(
        self, gamma: float, gae_lambda: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAE advantage and return computation (same as Jack's).

        Values may be stored as 0-D torch tensors on GPU. We do ONE bulk
        D→H sync here at flush time rather than one per step in the rollout.
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        # Clamp rewards before GAE. Loosened from ±5000 → ±20000 because end-
        # of-day single-step penalties (per_order_incomplete = -10 × N_unshipped)
        # can legitimately hit -7000 on N=4 disasters with 700 unshipped. The
        # tighter clip was masking the catastrophic-day signal, making the
        # value head think bad days were ordinary.
        rewards = [float(max(-20_000.0, min(20_000.0, r))) for r in self.rewards]

        # Batch-sync values: if they are tensors, stack them once and
        # convert to a single numpy array. If they are already floats, this
        # still works via np.array of a Python list.
        first = self.values[0] if self.values else 0.0
        if isinstance(first, torch.Tensor):
            # .float() up-casts bf16/fp16 to fp32 before the numpy conversion,
            # which doesn't support bf16 natively.
            values_np = torch.stack(self.values).detach().float().cpu().numpy()
        else:
            values_np = np.asarray(self.values, dtype=np.float32)

        for t in reversed(range(n)):
            next_value = 0.0 if t == n - 1 else float(values_np[t + 1])
            non_terminal = 1.0 - float(self.dones[t])
            delta = rewards[t] + gamma * next_value * non_terminal - float(values_np[t])
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values_np
        return advantages, returns

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self):
        self.states.clear()
        self.task_actions.clear()
        self.hustle_actions.clear()
        self.action_masks.clear()
        self.hustle_masks.clear()
        self.task_log_probs.clear()
        self.hustle_log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        # entry_hidden intentionally not cleared — overwritten by set_entry_hidden


# ─────────────────────────────────────────────────────────────────────────────
# ClarkAgent
# ─────────────────────────────────────────────────────────────────────────────

class ClarkAgent:
    """
    PPO agent wrapping ClarkActorCritic.

    The model is config-agnostic (weights work across different FacilityConfigs).
    Facility-specific details (N workers, M tasks, action masks) are provided
    at inference time through state_dicts and action_mask arrays.

    Usage:
        agent = ClarkAgent()
        # Per episode:
        agent.reset_hidden()
        agent.buffer.set_entry_hidden(agent.hidden)
        state_dict = builder.build(env.day_env)
        mask = get_action_mask(env.day_env)
        task_actions, hustle_actions, log_prob, value = agent.select_action_from_dict(state_dict, mask)
        actions = [(i, ta, bool(ha)) for i, (ta, ha) in enumerate(zip(task_actions, hustle_actions))]
        _, reward, done, _ = env.step(actions)
        agent.store_transition(state_dict, task_actions, hustle_actions, log_prob, value, reward, done, mask)
        if len(agent.buffer) >= agent.tbptt_chunk_size:
            metrics = agent.update()
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        compile_model: bool = False,
        **hparams,
    ):
        """
        Args:
            device:        'cpu', 'cuda', 'cuda:0', or a torch.device.
            use_amp:       When True and device is CUDA, wrap forward/backward
                           passes in `torch.autocast` for bf16 speedup.
            amp_dtype:     dtype for autocast (bf16 preferred on Blackwell/Ampere+).
            compile_model: When True and device is CUDA, wrap the model in
                           torch.compile. First call has a ~30s compile cost.
            **hparams:     PPO hyperparameter overrides.
        """
        self.hparams: dict = {**PPO_DEFAULTS, **hparams}
        self.device: torch.device = torch.device(device)

        self.model = ClarkActorCritic().to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.hparams["lr"])
        self.buffer = RolloutBuffer()

        # Running return statistics for value-loss normalization. Per-batch
        # standardization on TBPTT chunks of 64 correlated steps was producing
        # tiny / unstable σ → value-loss explosion + value head chasing
        # moving targets across chunks. A single running mean/σ updated by
        # EMA gives a stable normalizer the value head can actually learn
        # against. Welford-style would be more accurate but EMA tracks
        # distribution drift better as the policy improves.
        self._ret_mean: float = 0.0
        self._ret_var: float = 1.0
        self._ret_initialized: bool = False
        self._ret_beta: float = 0.01  # ~100-update half-life

        # AMP config — only meaningful on CUDA. CPU autocast works for bf16 too
        # but the RL hot path is tiny per-step and AMP overhead dominates there.
        # Also gate bf16 on hardware support — older GPUs (Pascal, original
        # Volta) lack native bf16 ops and would fall back to fp32 silently or
        # raise mid-run; explicit fallback to fp16 keeps behavior predictable.
        self._use_amp: bool = bool(use_amp) and self.device.type == "cuda"
        if (
            self._use_amp
            and amp_dtype == torch.bfloat16
            and not torch.cuda.is_bf16_supported()
        ):
            print("  [AMP] GPU does not support bf16; falling back to fp16.")
            amp_dtype = torch.float16
        self._amp_dtype: torch.dtype = amp_dtype

        if compile_model and self.device.type == "cuda":
            # Compile the two hot paths. Bound-method reassignment on an
            # nn.Module instance is supported: __call__ resolves `forward` via
            # instance dict first, so this plugs the compiled version into the
            # normal call path without breaking state_dict / parameters().
            self.model.forward = torch.compile(  # type: ignore[method-assign]
                self.model.forward, dynamic=True
            )
            self.model.forward_sequence = torch.compile(  # type: ignore[method-assign]
                self.model.forward_sequence, dynamic=True
            )

        # Live LSTM hidden state — persists across all steps within a year.
        self.hidden: tuple[torch.Tensor, torch.Tensor] = self.model.init_hidden()

    # ── Episode management ─────────────────────────────────────────────────────

    def _amp_ctx(self):
        """Context manager for AMP autocast — no-op on CPU or when disabled."""
        if self._use_amp:
            return torch.autocast(device_type="cuda", dtype=self._amp_dtype)
        return contextlib.nullcontext()

    def reset_hidden(self):
        """Reset LSTM memory. Call at the start of each new episode."""
        self.hidden = self.model.init_hidden()

    # ── Batched (multi-env) hidden-state management ──────────────────────────

    def init_hidden_batched(self, n_envs: int) -> None:
        """Initialize a batched LSTM hidden state of shape (1, n_envs, hidden).

        Populates `self.batched_hidden` and creates per-env rollout buffers
        at `self.buffers`. Call once before entering a batched training loop.
        """
        self.n_envs: int = n_envs
        self.batched_hidden: tuple[torch.Tensor, torch.Tensor] = (
            torch.zeros(1, n_envs, self.model.lstm_hidden, device=self.device),
            torch.zeros(1, n_envs, self.model.lstm_hidden, device=self.device),
        )
        self.buffers: list[RolloutBuffer] = [
            RolloutBuffer() for _ in range(n_envs)
        ]
        for buf in self.buffers:
            buf.set_entry_hidden(self.model.init_hidden())

    def reset_hidden_slots(self, slots: list[bool]) -> None:
        """Zero LSTM hidden state for any slot where `slots[b]` is True.

        Used after an env's year completes to give it a fresh memory for
        the next year. Hidden shape is (1, n_envs, hidden); we zero the
        middle-axis slices for the flagged batch indices.
        """
        if not any(slots):
            return
        idx = torch.as_tensor(
            [b for b, flag in enumerate(slots) if flag],
            dtype=torch.long, device=self.device,
        )
        with torch.no_grad():
            self.batched_hidden[0][:, idx, :] = 0.0
            self.batched_hidden[1][:, idx, :] = 0.0

    def snapshot_entry_hidden_for_slot(self, b: int) -> None:
        """Cache slot b's CURRENT hidden state as the entry hidden for its
        next rollout chunk. Call right after a slot's buffer is flushed
        (update completed) so the next chunk continues from where this one
        ended."""
        h = self.batched_hidden[0][:, b:b+1, :].detach().clone()
        c = self.batched_hidden[1][:, b:b+1, :].detach().clone()
        self.buffers[b].entry_hidden = (h, c)

    # ── Action selection ───────────────────────────────────────────────────────

    def select_action_from_dict(
        self,
        state_dict: dict,
        action_mask: np.ndarray | None = None,
        hustle_mask: np.ndarray | None = None,
    ):
        """
        Sample actions from a structured state dict.

        Args:
            state_dict:   Output of StateBuilder.build().
            action_mask:  (N, M) bool ndarray from get_action_mask(). Optional.

        Returns:
            task_actions:   list[int] length N
            hustle_actions: list[int] length N (0 or 1)
            log_prob:       0-D tensor on model device — sum of log-probs
            value:          0-D tensor on model device — critic estimate

        log_prob and value are returned as DETACHED tensors, not Python floats.
        This avoids a GPU→CPU sync on every rollout step. The buffer stores
        them as-is and flushes them in one batched sync at update time.
        """
        # eval() to disable dropout during rollout; cuDNN LSTM backward in the
        # PPO update later requires train mode.
        self.model.eval()
        try:
            with self._amp_ctx():
                (task_actions, hustle_actions,
                 task_lp, hustle_lp, value_t, new_hidden) = (
                    self.model.select_action(
                        state_dict, action_mask, self.hidden, hustle_mask=hustle_mask,
                    )
                )
        finally:
            self.model.train()
        self.hidden = new_hidden
        # fp32 cast: bf16 stored old_log_probs would round-trip through PPO
        # update with error comparable to clip_epsilon, drifting the ratio
        # even with frozen weights.
        return (
            task_actions, hustle_actions,
            task_lp.detach().float(), hustle_lp.detach().float(),
            value_t.detach().float(),
        )

    def store_transition(
        self,
        state_dict: dict,
        task_actions: list[int],
        hustle_actions: list[int],
        task_log_prob,    # (N,) torch.Tensor
        hustle_log_prob,  # (N,) torch.Tensor
        value,            # 0-D tensor or float
        reward: float,
        done: bool,
        action_mask: np.ndarray | None = None,
        hustle_mask: np.ndarray | None = None,
    ):
        self.buffer.add(
            state_dict, task_actions, hustle_actions,
            task_log_prob, hustle_log_prob,
            value, reward, done, action_mask, hustle_mask,
        )

    # ── Batched action selection ─────────────────────────────────────────────

    def select_action_batched(
        self,
        state_dicts: list,
        action_masks: list,
        hustle_masks: list | None = None,
    ):
        """
        Sample actions for self.n_envs envs in one batched forward.

        Returns:
            task_actions_list:   list[list[int]] length B — per-env task idx
            hustle_actions_list: list[list[int]] length B — per-env hustle flag
            log_probs:           (B,) detached tensor on model device
            values:              (B,) detached tensor on model device
        """
        # eval() to silence dropout during rollout; cuDNN LSTM backward in the
        # PPO update later requires train mode.
        self.model.eval()
        try:
            with self._amp_ctx():
                (ta, ha, t_lps, h_lps, v, new_hidden) = (
                    self.model.select_action_batched(
                        state_dicts, action_masks, self.batched_hidden,
                        hustle_masks=hustle_masks,
                    )
                )
        finally:
            self.model.train()
        self.batched_hidden = new_hidden
        # fp32 cast: see select_action_from_dict.
        t_lps = [t.detach().float() for t in t_lps]
        h_lps = [t.detach().float() for t in h_lps]
        return ta, ha, t_lps, h_lps, v.detach().float()

    def store_transition_batched(
        self,
        state_dicts: list,
        task_actions_list: list,
        hustle_actions_list: list,
        task_log_probs_list: list,   # list of (N_b,) per-env tensors
        hustle_log_probs_list: list, # list of (N_b,) per-env tensors
        values: torch.Tensor,        # (B,)
        rewards: list,               # list[float] length B
        dones: list,                 # list[bool] length B
        action_masks: list,
        hustle_masks: list | None = None,
    ) -> None:
        """Append one tick's transitions across all env slots into per-env buffers."""
        B = len(state_dicts)
        values = values.detach()
        for b in range(B):
            self.buffers[b].add(
                state_dicts[b],
                task_actions_list[b],
                hustle_actions_list[b],
                task_log_probs_list[b],
                hustle_log_probs_list[b],
                values[b],
                rewards[b],
                dones[b],
                action_masks[b],
                hustle_masks[b] if hustle_masks is not None else None,
            )

    # ── Batched PPO update ───────────────────────────────────────────────────

    def update_batched(self) -> dict:
        """Run PPO updates across all per-env buffers, accumulating metrics.

        Each env's buffer is processed by the existing sequential update code
        (via `_update_single_buffer`). This preserves per-env TBPTT correctness
        (the LSTM entry hidden state is snapshotted per env) while letting each
        env contribute to the learning signal.

        After every buffer is flushed, the per-env entry-hidden snapshot is
        refreshed to match that env's current `batched_hidden` slice so the
        next rollout chunk continues from where this one ended.

        Returns: averaged metrics dict across all env buffers.
        """
        totals = {
            "policy_loss": 0.0, "value_loss": 0.0,
            "entropy": 0.0, "clip_fraction": 0.0, "n": 0,
        }

        for b, buf in enumerate(self.buffers):
            if len(buf) == 0:
                continue
            m = self._update_single_buffer(buf)
            if m["n_updates"] == 0:
                continue
            totals["policy_loss"]  += m["policy_loss"]
            totals["value_loss"]   += m["value_loss"]
            totals["entropy"]      += m["entropy"]
            totals["clip_fraction"] += m["clip_fraction"]
            totals["n"] += 1

            # After flushing buf[b], snapshot the NEW entry hidden for its
            # next rollout chunk (continues from current live hidden state).
            self.snapshot_entry_hidden_for_slot(b)

        n = max(1, totals["n"])
        return {
            "policy_loss":   totals["policy_loss"]  / n,
            "value_loss":    totals["value_loss"]   / n,
            "entropy":       totals["entropy"]      / n,
            "clip_fraction": totals["clip_fraction"] / n,
            "n_updates":     totals["n"],
        }

    # ── PPO update ─────────────────────────────────────────────────────────────

    def update(self) -> dict:
        """PPO update over self.buffer. Thin wrapper for single-env callers."""
        return self._update_single_buffer(self.buffer)

    def _update_single_buffer(self, buf: RolloutBuffer) -> dict:
        """
        PPO update over a given rollout buffer.

        Processes steps sequentially to maintain LSTM temporal order.
        Uses TBPTT to bound gradient length through the LSTM.
        Returns dict of loss metrics.
        """
        if len(buf) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                    "clip_fraction": 0.0, "n_updates": 0}

        if buf.entry_hidden is None:
            buf.entry_hidden = self.model.init_hidden()

        advantages, returns = buf.compute_returns(
            self.hparams["gamma"], self.hparams["gae_lambda"]
        )

        # Defense-in-depth: clamp returns. Raised again from ±50k → ±500k.
        # An N=4 catastrophic year accumulates rw_per_worker around -700k.
        # At ±50k the value head saw "moderate failure" instead of the real
        # catastrophe, so the policy gradient never received the signal
        # "N=4 → AVOID this assignment pattern". The 500k bound just guards
        # against a numerical blow-up while letting the legitimate worst-
        # case signal through.
        returns = np.clip(returns, -500_000.0, 500_000.0)

        # Normalize advantages (per-batch).
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - advantages.mean()) / adv_std

        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # Update running return stats ONCE per update (not per epoch). On
        # the very first call, snap to batch stats so the value head isn't
        # asked to predict against μ=0/σ=1 when returns are at totally
        # different scale. Subsequent calls EMA toward batch stats.
        batch_mean = float(returns.mean())
        batch_var  = float(returns.var() + 1e-8)
        if not self._ret_initialized:
            self._ret_mean = batch_mean
            self._ret_var  = batch_var
            self._ret_initialized = True
        else:
            b = self._ret_beta
            self._ret_mean = (1.0 - b) * self._ret_mean + b * batch_mean
            self._ret_var  = (1.0 - b) * self._ret_var  + b * batch_var

        # Old per-worker log-probs are already fp32 tensors on device (cast at
        # storage time in select_action_*). We keep them as a list of per-step
        # variable-N tensors and align with new log-probs step-by-step below.
        old_task_lps: list[torch.Tensor] = [
            t.to(device=self.device, dtype=torch.float32).detach()
            for t in buf.task_log_probs
        ]
        old_hustle_lps: list[torch.Tensor] = [
            t.to(device=self.device, dtype=torch.float32).detach()
            for t in buf.hustle_log_probs
        ]

        clip_eps = self.hparams["clip_epsilon"]
        ent_coeff = self.hparams["entropy_coeff"]
        val_coeff = self.hparams["value_loss_coeff"]
        max_norm = self.hparams["max_grad_norm"]
        chunk_size = self.hparams["tbptt_chunk_size"]
        epochs = self.hparams["epochs_per_update"]

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
            "n_updates": 0,
        }

        for _ in range(epochs):
            with self._amp_ctx():
                # _evaluate_sequence returns PER-WORKER per-step log-probs as
                # parallel lists (one tensor per step, shape (N_t,)) for both
                # action heads. Variable N across steps is the reason — peak
                # staffing can change worker count mid-day.
                new_task_lps, new_hustle_lps, values, entropies = (
                    self._evaluate_sequence(
                        buf.states,
                        buf.task_actions,
                        buf.hustle_actions,
                        buf.action_masks,
                        buf.entry_hidden,
                        chunk_size,
                        hustle_masks=buf.hustle_masks,
                    )
                )

                values_f32 = values.float()
                entropies_f32 = entropies.float()

                # Per-(step, worker, head) policy loss — fully vectorized.
                # Computing the ratio per worker (instead of summing log-probs
                # across all workers and exponentiating) keeps the ratio
                # variance from scaling with N. Without this, joint ratio for
                # N=25 workers has stdev ~0.35 from harmless replay drift,
                # which alone saturates the clip threshold (clip_eps=0.2).
                #
                # The previous implementation was a Python `for t in range(T)`
                # loop that called `.item()` per step (forcing GPU→CPU sync
                # every step × every PPO epoch × every day-boundary). That
                # was the dominant runtime cost. Flattening to single 1D
                # tensors removes the per-step Python overhead and lets us
                # sync once at the end.
                #
                # Math note: this averages the surrogate over all
                # (step × worker × head) elements, vs the old per-step mean.
                # With variable N per step, this gives proportional weight
                # to busier steps — which matches standard PPO and is
                # arguably more correct than equal-weighting tiny steps
                # against full-staffed ones.
                new_t_flat = torch.cat([lp.float() for lp in new_task_lps])
                new_h_flat = torch.cat([lp.float() for lp in new_hustle_lps])
                old_t_flat = torch.cat(old_task_lps)
                old_h_flat = torch.cat(old_hustle_lps)

                # Broadcast per-step advantage to every worker in that step.
                n_workers_per_step = torch.tensor(
                    [lp.shape[0] for lp in new_task_lps],
                    dtype=torch.long, device=advantages_t.device,
                )
                adv_per_worker = advantages_t.repeat_interleave(
                    n_workers_per_step
                )                                                  # (sum N,)

                r_task = torch.exp(new_t_flat - old_t_flat)
                r_hustle = torch.exp(new_h_flat - old_h_flat)

                # Stack task + hustle decisions into one big batch and apply
                # the same advantage to both heads of each worker.
                r = torch.cat([r_task, r_hustle])                  # (2*sum N,)
                adv = torch.cat([adv_per_worker, adv_per_worker])  # (2*sum N,)

                surr1 = r * adv
                surr2 = torch.clamp(
                    r, 1.0 - clip_eps, 1.0 + clip_eps
                ) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Single GPU→CPU sync at the very end (was per-step before).
                clip_fraction = (
                    (r - 1.0).abs() > clip_eps
                ).float().mean().item()

                # Running-stats return standardization for value loss.
                # Per-batch standardization on a 64-step correlated chunk
                # was the previous approach and destabilized the value head:
                # σ within a chunk is tiny (correlated samples), so MSE
                # divided by tiny σ blew up; μ moves chunk-to-chunk, so the
                # value head chased a moving target. We instead maintain a
                # single global EMA of return mean/var (updated below per
                # update call, NOT per epoch) — a stable normalizer the
                # value head can actually learn against.
                ret_mean = float(self._ret_mean)
                ret_std  = float(self._ret_var) ** 0.5 + 1e-8
                value_loss = F.mse_loss(
                    (values_f32 - ret_mean) / ret_std,
                    (returns_t  - ret_mean) / ret_std,
                )
                entropy_loss = -entropies_f32.mean()

                loss = policy_loss + val_coeff * value_loss + ent_coeff * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                self.optimizer.zero_grad()
                continue
            self.optimizer.step()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += -entropy_loss.item()
            metrics["clip_fraction"] += clip_fraction
            metrics["n_updates"] += 1

        n = max(1, metrics["n_updates"])
        for k in ("policy_loss", "value_loss", "entropy", "clip_fraction"):
            metrics[k] /= n

        buf.clear()
        return metrics

    def _evaluate_sequence(
        self,
        states: list[dict],
        task_actions: list[list[int]],
        hustle_actions: list[list[int]],
        action_masks: list[np.ndarray | None],
        entry_hidden: tuple[torch.Tensor, torch.Tensor],
        chunk_size: int,
        hustle_masks: list[np.ndarray | None] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Replay the stored sequence through the model to get per-worker log-probs,
        values, and entropies for the PPO loss.

        Uses forward_sequence() to process each TBPTT chunk in one batched call.
        Chunks are split at every N-change so np.stack of state dicts succeeds.
        TBPTT: hidden state is detached at each chunk boundary.

        Returns:
            new_task_log_probs:   list[(N_t,) tensor] length T — per-step per-worker
            new_hustle_log_probs: list[(N_t,) tensor] length T — per-step per-worker
            values:               (T,) tensor
            entropies:            (T,) tensor — summed over workers, for entropy bonus
        """
        n = len(states)
        hidden = (entry_hidden[0].clone(), entry_hidden[1].clone())

        all_task_lps: list[torch.Tensor] = []
        all_hustle_lps: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []
        all_entropies: list[torch.Tensor] = []

        # Chunk boundaries: split at every point where the worker count N
        # changes (e.g. peak-staffing arrival/departure mid-day) AND at every
        # `chunk_size` steps. Without the N-split, forward_sequence's np.stack
        # over heterogeneous-N state dicts would raise.
        def _next_chunk_end(start: int) -> int:
            N0 = states[start]["worker_feats"].shape[0]
            limit = min(start + chunk_size, n)
            end = start + 1
            while end < limit:
                if states[end]["worker_feats"].shape[0] != N0:
                    break
                end += 1
            return end

        chunk_start = 0
        while chunk_start < n:
            chunk_end = _next_chunk_end(chunk_start)
            T = chunk_end - chunk_start

            hm_chunk = (
                hustle_masks[chunk_start:chunk_end]
                if hustle_masks is not None else None
            )
            assign_logits, hustle_logits, values, entropies, hidden = (
                self.model.forward_sequence(
                    states[chunk_start:chunk_end],
                    action_masks[chunk_start:chunk_end],
                    hidden,
                    hustle_masks=hm_chunk,
                )
            )
            # assign_logits: (T, N, M), hustle_logits: (T, N, 2)
            # values: (T,),  entropies: (T,)

            # Vectorised log-prob computation — no Python loop over workers
            device = assign_logits.device
            ta = torch.tensor(
                [task_actions[chunk_start + t] for t in range(T)],
                dtype=torch.long, device=device,
            )  # (T, N)
            ha = torch.tensor(
                [hustle_actions[chunk_start + t] for t in range(T)],
                dtype=torch.long, device=device,
            )  # (T, N)

            task_lp = F.log_softmax(assign_logits, dim=-1)            # (T, N, M)
            taken_task_lp = task_lp.gather(
                dim=2, index=ta.unsqueeze(-1)
            ).squeeze(-1)                                              # (T, N)

            hustle_lp = F.log_softmax(hustle_logits, dim=-1)          # (T, N, 2)
            taken_hustle_lp = hustle_lp.gather(
                dim=2, index=ha.unsqueeze(-1)
            ).squeeze(-1)                                              # (T, N)

            # Append per-step per-worker tensors. Steps inside this chunk all
            # share N (chunker guarantees), but across chunks N can differ —
            # so we keep them as a list of variable-N tensors, not a stacked
            # (T_total, N) array.
            for t in range(T):
                all_task_lps.append(taken_task_lp[t])      # (N,)
                all_hustle_lps.append(taken_hustle_lp[t])  # (N,)
            all_values.append(values)
            all_entropies.append(entropies)

            # TBPTT: detach hidden state at chunk boundary
            hidden = (hidden[0].detach(), hidden[1].detach())

            chunk_start = chunk_end

        return (
            all_task_lps,
            all_hustle_lps,
            torch.cat(all_values),
            torch.cat(all_entropies),
        )

    # ── Checkpoint ─────────────────────────────────────────────────────────────

    def save(self, path: str, episode: int, config: "FacilityConfig | None" = None):
        """Save model checkpoint with arch version, config, and optimizer state.

        Builds a CPU snapshot WITHOUT mutating the live model or optimizer.
        `optimizer.state_dict()` returns references to the live state tensors,
        so rewriting entries in that dict in-place would move the real
        optimizer state onto CPU and the next optimizer.step() would crash
        with a mixed-device error. Instead we build a fresh nested copy with
        all tensors detach-cloned to CPU.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        config_yaml: str | None = None
        if config is not None:
            try:
                config_yaml = yaml.dump(asdict(config))
            except Exception:
                config_yaml = None

        def _tensor_to_cpu(v):
            if isinstance(v, torch.Tensor):
                return v.detach().cpu()
            return v

        def _deep_cpu(obj):
            """Recursively copy a container, putting every tensor on CPU.

            Handles the optimizer state_dict shape: {"state": {param_id: {k: v}},
            "param_groups": [...]}. Also handles plain model state_dicts
            (flat {name: tensor}) for completeness.
            """
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu()
            if isinstance(obj, dict):
                return {k: _deep_cpu(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_deep_cpu(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_deep_cpu(v) for v in obj)
            return obj

        # CRITICAL: build an independent snapshot. Do NOT mutate self.optimizer
        # state_dict in place — that dict holds references to the live optimizer
        # state tensors, and moving them to CPU would corrupt the live optimizer.
        model_sd_cpu = _deep_cpu(self.model.state_dict())
        opt_sd_cpu   = _deep_cpu(self.optimizer.state_dict())

        torch.save(
            {
                "arch_version": ARCH_VERSION,
                "episode": episode,
                "facility_config": config_yaml,
                "model_state_dict": model_sd_cpu,
                "optimizer_state_dict": opt_sd_cpu,
                "hparams": self.hparams,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        facility_config: "FacilityConfig | None" = None,
        device: Union[str, torch.device] = "cpu",
        **agent_kwargs,
    ) -> "ClarkAgent":
        """
        Load a checkpoint and return a ClarkAgent.

        If facility_config is None, the config embedded in the checkpoint is
        used (informational only — the model itself is config-agnostic).
        Raises ValueError if arch_version does not match ARCH_VERSION.

        `device` selects where the loaded model lives. Additional kwargs are
        forwarded to ClarkAgent.__init__ (use_amp, compile_model, etc.).
        """
        dev = torch.device(device)
        data = torch.load(path, map_location=dev, weights_only=False)

        saved_version = data.get("arch_version", "unknown")
        if saved_version != ARCH_VERSION:
            raise ValueError(
                f"Checkpoint architecture mismatch: "
                f"saved={saved_version!r}, current={ARCH_VERSION!r}. "
                f"Cannot load this checkpoint."
            )

        hparams = data.get("hparams", {})
        agent = cls(device=dev, **agent_kwargs, **hparams)

        # Migrate legacy PopArt-era checkpoints (had value_head.body.* /
        # value_head.out.*) back to the simple nn.Sequential layout
        # (value_head.0.* / value_head.2.*). PopArt buffers (mu/sigma/
        # initialized) are dropped — they no longer exist in the model.
        msd = dict(data["model_state_dict"])
        if "value_head.body.0.weight" in msd and "value_head.0.weight" not in msd:
            msd["value_head.0.weight"] = msd.pop("value_head.body.0.weight")
            msd["value_head.0.bias"] = msd.pop("value_head.body.0.bias")
            msd["value_head.2.weight"] = msd.pop("value_head.out.weight")
            msd["value_head.2.bias"] = msd.pop("value_head.out.bias")
            # Drop the now-defunct PopArt buffers if present.
            for k in list(msd.keys()):
                if k.startswith("value_head.") and k.split(".")[-1] in ("mu", "sigma", "initialized"):
                    msd.pop(k)
        agent.model.load_state_dict(msd, strict=False)
        opt_sd = data.get("optimizer_state_dict")
        if opt_sd is not None:
            agent.optimizer.load_state_dict(opt_sd)
            # load_state_dict keeps optimizer tensors on CPU if the checkpoint
            # was saved from CPU — move them to the agent's device so they
            # match the model parameters they refer to.
            for state in agent.optimizer.state.values():
                for k, v in list(state.items()):
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(dev)
        # else: keep fresh optimizer (used when loading salvaged checkpoints)
        return agent
