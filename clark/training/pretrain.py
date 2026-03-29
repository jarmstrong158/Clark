"""Foundation model pre-training on synthetic facility configs.

Pre-training strategy:
  - ONE ClarkAgent (foundation model) is kept throughout all episodes.
  - Each episode samples a new random FacilityConfig (domain randomization).
  - The transformer weights are config-agnostic (fixed input/output dims),
    so the same model trains on facilities with 5-50 workers and 3-15 tasks.
  - Updates happen after every TBPTT chunk within the year (not at year-end),
    matching Jack's daily-update strategy.

Usage:
    python -m clark.training.pretrain
    python -m clark.training.pretrain --episodes 5000 --output path/to/out.pt
"""
from __future__ import annotations

import argparse
import os
import time

from clark.agent.actions import get_action_mask
from clark.agent.ppo import ClarkAgent
from clark.agent.state import StateBuilder
from clark.env.year_env import YearEnv
from clark.training.synthetic_gen import generate_random_facility


def pretrain(
    n_episodes: int = 10000,
    output_path: str = "clark/data/checkpoints/clark_foundation.pt",
    save_interval: int = 100,
    log_interval: int = 10,
    device: str = "cpu",
) -> None:
    """
    Pre-train the Clark foundation model on randomized facility configs.

    Args:
        n_episodes:    Total number of year-episodes to train.
        output_path:   Where to save the foundation checkpoint.
        save_interval: Save checkpoint every N episodes.
        log_interval:  Print progress every N episodes.
        device:        Torch device ("cpu" or "cuda"). Model stays on CPU by
                       default since env ops are CPU-bound anyway.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # One foundation agent, no config dependency in the model weights.
    agent = ClarkAgent()

    print(f"Pre-training Clark foundation model for {n_episodes} episodes.")
    print(f"Output: {output_path}")
    print()

    start_time = time.time()
    episode_rewards: list[float] = []

    for ep in range(1, n_episodes + 1):
        config = generate_random_facility()
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

            # Build action list: (worker_id, task_idx, hustle_bool)
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

            # Update on new day or end of year — mirrors Jack's daily update cadence
            if info.get("new_day") or done:
                if len(agent.buffer) > 0:
                    agent.update()
                # Snapshot hidden for next day's rollout
                agent.buffer.set_entry_hidden(agent.hidden)

        episode_rewards.append(episode_reward)

        if ep % log_interval == 0:
            recent = episode_rewards[-log_interval:]
            avg_reward = sum(recent) / len(recent)
            elapsed = time.time() - start_time
            n_workers = config.num_workers
            n_tasks = config.num_tasks
            print(
                f"  Ep {ep:6d}/{n_episodes} | "
                f"AvgReward: {avg_reward:8.1f} | "
                f"N={n_workers:2d} M={n_tasks:2d} | "
                f"Steps: {steps:4d} | "
                f"{elapsed:6.0f}s"
            )

        if ep % save_interval == 0:
            agent.save(output_path, ep, config)
            print(f"  [Checkpoint saved: ep {ep} → {output_path}]")

    # Final save
    final_config = generate_random_facility()  # dummy config for metadata
    agent.save(output_path, n_episodes, final_config)
    print(f"\nPre-training complete. {n_episodes} episodes in {time.time() - start_time:.0f}s")
    print(f"Foundation model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-train Clark foundation model.")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--output", type=str,
                        default="clark/data/checkpoints/clark_foundation.pt")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    pretrain(
        n_episodes=args.episodes,
        output_path=args.output,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        device=args.device,
    )


if __name__ == "__main__":
    main()
