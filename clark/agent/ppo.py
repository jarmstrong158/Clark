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

import os
from dataclasses import asdict
from typing import TYPE_CHECKING

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
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coeff": 0.02,
    "value_loss_coeff": 0.5,
    "max_grad_norm": 0.5,
    "epochs_per_update": 4,
    "tbptt_chunk_size": 16,
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
        self.log_probs: list[float] = []        # scalar per step
        self.values: list[float] = []           # scalar per step
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
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        action_mask: np.ndarray | None = None,
    ):
        self.states.append(state_dict)
        self.task_actions.append(task_actions)
        self.hustle_actions.append(hustle_actions)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def compute_returns(
        self, gamma: float, gae_lambda: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAE advantage and return computation (same as Jack's)."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            next_value = 0.0 if t == n - 1 else self.values[t + 1]
            non_terminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self):
        self.states.clear()
        self.task_actions.clear()
        self.hustle_actions.clear()
        self.action_masks.clear()
        self.log_probs.clear()
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

    def __init__(self, **hparams):
        self.hparams: dict = {**PPO_DEFAULTS, **hparams}

        self.model = ClarkActorCritic()
        self.optimizer = Adam(self.model.parameters(), lr=self.hparams["lr"])
        self.buffer = RolloutBuffer()

        # Live LSTM hidden state — persists across all steps within a year.
        self.hidden: tuple[torch.Tensor, torch.Tensor] = self.model.init_hidden()

    # ── Episode management ─────────────────────────────────────────────────────

    def reset_hidden(self):
        """Reset LSTM memory. Call at the start of each new episode."""
        self.hidden = self.model.init_hidden()

    # ── Action selection ───────────────────────────────────────────────────────

    def select_action_from_dict(
        self,
        state_dict: dict,
        action_mask: np.ndarray | None = None,
    ) -> tuple[list[int], list[int], float, float]:
        """
        Sample actions from a structured state dict.

        Args:
            state_dict:   Output of StateBuilder.build().
            action_mask:  (N, M) bool ndarray from get_action_mask(). Optional.

        Returns:
            task_actions:   list[int] length N
            hustle_actions: list[int] length N (0 or 1)
            log_prob:       float — sum of log-probs across all workers
            value:          float — critic estimate
        """
        task_actions, hustle_actions, log_prob_t, value_t, new_hidden = (
            self.model.select_action(state_dict, action_mask, self.hidden)
        )
        self.hidden = new_hidden
        return task_actions, hustle_actions, log_prob_t.item(), value_t.item()

    def store_transition(
        self,
        state_dict: dict,
        task_actions: list[int],
        hustle_actions: list[int],
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        action_mask: np.ndarray | None = None,
    ):
        self.buffer.add(
            state_dict, task_actions, hustle_actions,
            log_prob, value, reward, done, action_mask
        )

    # ── PPO update ─────────────────────────────────────────────────────────────

    def update(self) -> dict:
        """
        PPO update over the current rollout buffer.

        Processes steps sequentially to maintain LSTM temporal order.
        Uses TBPTT to bound gradient length through the LSTM.
        Returns dict of loss metrics.
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_updates": 0}

        if self.buffer.entry_hidden is None:
            self.buffer.entry_hidden = self.model.init_hidden()

        advantages, returns = self.buffer.compute_returns(
            self.hparams["gamma"], self.hparams["gae_lambda"]
        )

        # Normalize advantages
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - advantages.mean()) / adv_std

        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)
        old_log_probs_t = torch.FloatTensor(self.buffer.log_probs)

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
            log_probs, values, entropies = self._evaluate_sequence(
                self.buffer.states,
                self.buffer.task_actions,
                self.buffer.hustle_actions,
                self.buffer.action_masks,
                self.buffer.entry_hidden,
                chunk_size,
            )

            ratio = torch.exp(log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Fraction of steps where the ratio hit the clip boundary.
            # Healthy range ≈ 5–20%. Below 5% = updates too timid; above 20% = steps too large.
            clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

            # Normalize returns to keep value_loss scale stable regardless of reward scale
            ret_mean = returns_t.mean()
            ret_std = returns_t.std() + 1e-8
            value_loss = F.mse_loss(
                (values - ret_mean) / ret_std,
                (returns_t - ret_mean) / ret_std,
            )
            entropy_loss = -entropies.mean()

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

        self.buffer.clear()
        return metrics

    def _evaluate_sequence(
        self,
        states: list[dict],
        task_actions: list[list[int]],
        hustle_actions: list[list[int]],
        action_masks: list[np.ndarray | None],
        entry_hidden: tuple[torch.Tensor, torch.Tensor],
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Replay the stored sequence through the model to get log-probs, values,
        and entropies for the PPO loss.

        TBPTT: hidden state is detached at each chunk boundary.

        Returns:
            log_probs:  (N_steps,) tensor
            values:     (N_steps,) tensor
            entropies:  (N_steps,) tensor
        """
        n = len(states)
        hidden = (entry_hidden[0].clone(), entry_hidden[1].clone())

        all_log_probs: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []
        all_entropies: list[torch.Tensor] = []

        for chunk_start in range(0, n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n)

            for t in range(chunk_start, chunk_end):
                sd = states[t]
                mask = action_masks[t]

                # Forward through model — with grad for loss computation
                assign_logits, hustle_logits, value, hidden = self.model.forward(
                    sd, mask, hidden
                )

                N = assign_logits.shape[0]

                # Compute log-prob of the stored actions
                step_log_prob = torch.tensor(0.0, device=assign_logits.device)
                step_entropy = torch.tensor(0.0, device=assign_logits.device)

                ta_t = torch.tensor(task_actions[t], dtype=torch.long,
                                    device=assign_logits.device)
                ha_t = torch.tensor(hustle_actions[t], dtype=torch.long,
                                    device=hustle_logits.device)

                for i in range(N):
                    # Task assignment
                    task_dist = Categorical(logits=assign_logits[i])
                    step_log_prob = step_log_prob + task_dist.log_prob(ta_t[i])
                    step_entropy = step_entropy + task_dist.entropy()

                    # Hustle decision
                    hustle_dist = Categorical(logits=hustle_logits[i])
                    step_log_prob = step_log_prob + hustle_dist.log_prob(ha_t[i])
                    step_entropy = step_entropy + hustle_dist.entropy()

                all_log_probs.append(step_log_prob)
                all_values.append(value)
                all_entropies.append(step_entropy)

            # TBPTT: detach hidden state at chunk boundary
            hidden = (hidden[0].detach(), hidden[1].detach())

        return (
            torch.stack(all_log_probs),
            torch.stack(all_values),
            torch.stack(all_entropies),
        )

    # ── Checkpoint ─────────────────────────────────────────────────────────────

    def save(self, path: str, episode: int, config: "FacilityConfig | None" = None):
        """Save model checkpoint with arch version, config, and optimizer state."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        config_yaml: str | None = None
        if config is not None:
            try:
                config_yaml = yaml.dump(asdict(config))
            except Exception:
                config_yaml = None

        torch.save(
            {
                "arch_version": ARCH_VERSION,
                "episode": episode,
                "facility_config": config_yaml,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "hparams": self.hparams,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        facility_config: "FacilityConfig | None" = None,
    ) -> "ClarkAgent":
        """
        Load a checkpoint and return a ClarkAgent.

        If facility_config is None, the config embedded in the checkpoint is
        used (informational only — the model itself is config-agnostic).
        Raises ValueError if arch_version does not match ARCH_VERSION.
        """
        data = torch.load(path, weights_only=False)

        saved_version = data.get("arch_version", "unknown")
        if saved_version != ARCH_VERSION:
            raise ValueError(
                f"Checkpoint architecture mismatch: "
                f"saved={saved_version!r}, current={ARCH_VERSION!r}. "
                f"Cannot load this checkpoint."
            )

        hparams = data.get("hparams", {})
        agent = cls(**hparams)
        agent.model.load_state_dict(data["model_state_dict"])
        agent.optimizer.load_state_dict(data["optimizer_state_dict"])
        return agent
