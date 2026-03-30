"""Fine-tuning from a foundation checkpoint on a specific facility config.

Fine-tuning uses a lower learning rate (default 5e-5 vs pre-train 3e-4) and
optionally freezes the encoder (self-attention, cross-attention) to prevent
catastrophic forgetting of the general representation learned during pre-training.

Only the task-assignment head, hustle head, value head, and LSTM adapt to the
specific facility — the shared backbone stays frozen.

Usage:
    python -m clark.training.finetune \\
        --config clark/data/configs/my_facility.yaml \\
        --base-model clark/data/checkpoints/clark_foundation.pt \\
        --output clark/data/checkpoints/my_facility_agent.pt
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from torch.optim import Adam

from clark.agent.actions import get_action_mask
from clark.agent.ppo import ClarkAgent
from clark.agent.state import StateBuilder
from clark.config.schema import FacilityConfig
from clark.env.year_env import YearEnv
from clark.sim_logging.episode_logger import EpisodeLogger


def finetune(
    config_path: str,
    base_model_path: str,
    output_path: str,
    n_episodes: int = 500,
    lr: float = 5e-5,
    freeze_encoder: bool = False,
    log_interval: int = 10,
    save_interval: int = 50,
    log_dir: str = None,
) -> None:
    """
    Fine-tune a foundation checkpoint on a specific facility.

    Args:
        config_path:      Path to facility YAML (e.g. my_facility.yaml).
        base_model_path:  Path to foundation checkpoint (.pt file).
        output_path:      Where to save the fine-tuned checkpoint.
        n_episodes:       Number of fine-tuning episodes.
        lr:               Learning rate (default 5e-5, lower than pre-train).
        freeze_encoder:   If True, freeze worker/task linear projections and all
                          self-attention + cross-attention layers. Only the LSTM,
                          value head, and output heads will train.
        log_interval:     Print progress every N episodes.
        save_interval:    Save checkpoint every N episodes.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    config = FacilityConfig.from_yaml(config_path)

    # Default log dir: next to checkpoint, named after the facility
    if log_dir is None:
        safe_name = config.name.lower().replace(" ", "_").replace("/", "_")
        log_dir = str(Path(output_path).parent / f"logs_{safe_name}")

    errors, warnings = config.validate()
    if errors:
        raise ValueError(f"Invalid facility config {config_path}: {errors}")
    if warnings:
        for w in warnings:
            print(f"  [Warning] {w}")

    # Load foundation model
    agent = ClarkAgent.load(base_model_path)
    agent.hparams["lr"] = lr

    if freeze_encoder:
        # Freeze input projection layers and attention layers.
        # The LSTM, value head, hustle head, and task_type_embed remain trainable
        # so the model can adapt its temporal memory and output distributions
        # to this specific facility's dynamics.
        layers_to_freeze = (
            list(agent.model.worker_linear.parameters())
            + list(agent.model.task_linear.parameters())
            + list(agent.model.env_linear.parameters())
            + list(agent.model.role_embed.parameters())
        )
        for layer in agent.model.sa_layers:
            layers_to_freeze.extend(layer.parameters())
        for layer in agent.model.ca_layers:
            layers_to_freeze.extend(layer.parameters())
        for layer in agent.model.ca_ff:
            layers_to_freeze.extend(layer.parameters())

        for param in layers_to_freeze:
            param.requires_grad = False

        # Rebuild optimizer with only unfrozen parameters
        trainable = [p for p in agent.model.parameters() if p.requires_grad]
        agent.optimizer = Adam(trainable, lr=lr)
        n_frozen = sum(p.numel() for p in agent.model.parameters() if not p.requires_grad)
        n_trainable = sum(p.numel() for p in agent.model.parameters() if p.requires_grad)
        print(f"  Encoder frozen: {n_frozen:,} params frozen, {n_trainable:,} trainable.")
    else:
        # Rebuild optimizer at fine-tune LR even with all params unfrozen
        agent.optimizer = Adam(agent.model.parameters(), lr=lr)

    # Full-detail logger — includes step assignments and year snapshots for dashboard
    logger = EpisodeLogger(log_dir=log_dir, mode="finetune", facility_config=config)

    print(f"Fine-tuning on: {config.name} ({config_path})")
    print(f"Base model:     {base_model_path}")
    print(f"Output:         {output_path}")
    print(f"Logs:           {log_dir}")
    print(f"LR: {lr}  |  freeze_encoder: {freeze_encoder}  |  episodes: {n_episodes}")
    print()

    start_time = time.time()
    episode_rewards: list[float] = []

    for ep in range(1, n_episodes + 1):
        env = YearEnv(config)
        builder = StateBuilder(config)

        env.reset()
        agent.reset_hidden()
        agent.buffer.set_entry_hidden(agent.hidden)

        episode_reward = 0.0
        done = False
        steps = 0

        while not done:
            state_dict = builder.build(env.day_env)
            mask = get_action_mask(env.day_env)

            task_actions, hustle_actions, log_prob, value = (
                agent.select_action_from_dict(state_dict, mask)
            )

            actions = [
                (i, task_actions[i], bool(hustle_actions[i]))
                for i in range(config.num_workers)
            ]

            _, reward, done, info = env.step(actions)
            episode_reward += reward
            steps += 1

            agent.store_transition(
                state_dict, task_actions, hustle_actions,
                log_prob, value, reward, done, mask
            )

            if info.get("new_day") or done:
                if len(agent.buffer) > 0:
                    agent.update()
                agent.buffer.set_entry_hidden(agent.hidden)

        episode_rewards.append(episode_reward)

        # Log all daily summaries for this year — full detail for dashboard
        for day_idx, summary in enumerate(env.daily_summaries):
            ep_day_num = (ep - 1) * env.total_work_days + day_idx + 1
            logger.log_episode(summary, ep_day_num, write=False)
        logger._write_log()  # single disk write per year

        # Year snapshot — dashboard reads this for the grade history panel
        year_summary = env._get_year_summary()
        logger.write_year_snapshot(
            n_days=env.total_work_days,
            year_summary=year_summary,
        )

        if ep % log_interval == 0:
            recent = episode_rewards[-log_interval:]
            avg_reward = sum(recent) / len(recent)
            elapsed = time.time() - start_time
            stats = logger.get_training_stats()
            print(
                f"  Ep {ep:5d}/{n_episodes} | "
                f"AvgReward: {avg_reward:8.1f} | "
                f"WinRate: {stats['win_rate_last_window']:.1%} | "
                f"Steps: {steps:4d} | "
                f"{elapsed:6.0f}s"
            )

        if ep % save_interval == 0:
            agent.save(output_path, ep, config)
            print(f"  [Checkpoint saved → {output_path}]")

    # Final save
    agent.save(output_path, n_episodes, config)
    print(f"\nFine-tuning complete. {n_episodes} episodes in {time.time() - start_time:.0f}s")
    print(f"Fine-tuned model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Clark on a specific facility.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to facility YAML config.")
    parser.add_argument("--base-model", type=str, required=True,
                        help="Path to foundation checkpoint (.pt).")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for fine-tuned checkpoint.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder layers (SA/CA/input projections).")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    args = parser.parse_args()

    finetune(
        config_path=args.config,
        base_model_path=args.base_model,
        output_path=args.output,
        n_episodes=args.episodes,
        lr=args.lr,
        freeze_encoder=args.freeze_encoder,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
