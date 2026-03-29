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
from clark.training.synthetic_gen import generate_random_facility, CURRICULUM_STAGES
from clark.sim_logging.episode_logger import EpisodeLogger


def pretrain(
    n_episodes: int = 10000,
    output_path: str = "clark/data/checkpoints/clark_foundation.pt",
    log_dir: str = "clark/data/logs/pretrain",
    years_per_config: int = 5,
    save_interval: int = 50,
    log_interval: int = 10,
    device: str = "cpu",
) -> None:
    """
    Pre-train the Clark foundation model on randomized facility configs.

    Args:
        n_episodes:      Total number of year-episodes to train.
        output_path:     Where to save the foundation checkpoint.
        log_dir:         Directory for training logs.
        years_per_config: Train this many years on each facility config before
                         sampling a new one.
        save_interval:   Save checkpoint every N episodes.
        log_interval:    Print progress every N episodes.
        device:          Torch device ("cpu" or "cuda"). Model stays on CPU by
                         default since env ops are CPU-bound anyway.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    total_configs = max(1, n_episodes // years_per_config)
    stage1_end = max(1, int(total_configs * 0.15))   # first 15%
    stage2_end = max(1, int(total_configs * 0.45))   # first 45%

    def _get_stage(configs_seen: int) -> int:
        if configs_seen < stage1_end:
            return 1
        if configs_seen < stage2_end:
            return 2
        return 3

    # One foundation agent, no config dependency in the model weights.
    agent = ClarkAgent()

    # Lightweight logger — pretrain mode strips step data to control log size
    logger = EpisodeLogger(log_dir=log_dir, mode="pretrain")

    print(f"Pre-training Clark foundation model")
    print(f"  Episodes (years): {n_episodes}")
    print(f"  Years per config: {years_per_config}")
    print(f"  Total configs:    {total_configs}")
    print(f"  Stage 1 ends at config: {stage1_end} (simple: 2-10 workers, 3-5 tasks)")
    print(f"  Stage 2 ends at config: {stage2_end} (intermediate: 5-25 workers, 3-10 tasks)")
    print(f"  Stage 3 starts:         configs {stage2_end+1}+ (full complexity)")
    print(f"  Output: {output_path}")
    print(f"  Logs:   {log_dir}")
    print()
    print(
        f"  {'Grade':<6} {'Episode':<14} {'Stage':<7} {'Config':<20} "
        f"{'Size':<10} {'R/W':>8} {'Win':>6} {'OT':>6} {'Cmp%':>6} "
        f"{'P-loss':>8} {'V-loss':>8} {'Entr':>6} {'Clip':>6} {'Time':>7}"
    )
    print("  " + "-" * 130)

    configs_seen: int = 0
    years_on_current_config: int = 0
    current_config = None
    current_stage: int = 1

    start_time = time.time()

    # Per-episode tracking (raw, for window math)
    episode_rewards: list[float] = []           # raw reward
    episode_reward_per_worker: list[float] = [] # reward / num_workers
    episode_grades: list[str] = []
    episode_ot_flags: list[bool] = []
    episode_ord_pct: list[float] = []           # orders_completed / total_orders

    # Per-window loss accumulation (reset every log_interval)
    loss_accum: dict[str, float] = {
        "policy_loss": 0.0, "value_loss": 0.0,
        "entropy": 0.0, "clip_fraction": 0.0, "n": 0,
    }

    for ep in range(1, n_episodes + 1):
        # Rotate to a new config every `years_per_config` years
        if years_on_current_config == 0 or years_on_current_config >= years_per_config:
            current_stage = _get_stage(configs_seen)
            current_config = generate_random_facility(stage=current_stage)
            configs_seen += 1
            years_on_current_config = 0

        years_on_current_config += 1

        # Heartbeat: print a dot every episode so the terminal isn't silent during long episodes.
        # Overwrite the same line to avoid flooding the log.
        print(f"  ... running ep {ep}/{n_episodes} (stg {current_stage}, cfg {configs_seen}/{total_configs})", end="\r", flush=True)

        env = YearEnv(current_config)
        builder = StateBuilder(current_config)

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
            # Use len(task_actions) — temp peak-staffing workers push N above config.num_workers
            actions = [
                (i, task_actions[i], bool(hustle_actions[i]))
                for i in range(len(task_actions))
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
                    metrics = agent.update()
                    loss_accum["policy_loss"] += metrics["policy_loss"]
                    loss_accum["value_loss"]  += metrics["value_loss"]
                    loss_accum["entropy"]     += metrics["entropy"]
                    loss_accum["clip_fraction"] += metrics["clip_fraction"]
                    loss_accum["n"] += 1
                # Snapshot hidden for next day's rollout
                agent.buffer.set_entry_hidden(agent.hidden)

        # ── Episode stats ──────────────────────────────────────────────────────
        episode_rewards.append(episode_reward)
        episode_reward_per_worker.append(episode_reward / max(1, current_config.num_workers))

        # Pull grade + completion stats from last day's summary
        last_grade = "?"
        last_ord_pct = 0.0
        had_ot = False
        if env.daily_summaries:
            last_summary = env.daily_summaries[-1]
            footer = last_summary.get("footer", {})
            last_grade = footer.get("grade", "?")
            total_orders = last_summary.get("header", {}).get("total_orders", 1) or 1
            orders_remaining = footer.get("orders_remaining", 0)
            last_ord_pct = max(0.0, (total_orders - orders_remaining) / total_orders)
            had_ot = footer.get("ot_hours", 0.0) > 0.0

        episode_grades.append(last_grade)
        episode_ot_flags.append(had_ot)
        episode_ord_pct.append(last_ord_pct)

        # Log last day's summary (lightweight in pretrain mode)
        if env.daily_summaries:
            logger.log_episode(
                env.daily_summaries[-1], ep,
                facility_config=current_config,
                write=(ep % log_interval == 0),
            )

        # Write year snapshot at checkpoint intervals
        if ep % save_interval == 0:
            year_summary = env._get_year_summary()
            logger.write_year_snapshot(
                n_days=env.total_work_days,
                year_summary=year_summary,
                facility_config=current_config,
            )

        # ── Progress line ──────────────────────────────────────────────────────
        if ep % log_interval == 0:
            print(" " * 80, end="\r")  # clear heartbeat line
            w = log_interval  # window size

            recent_rw   = episode_reward_per_worker[-w:]
            prev_rw     = episode_reward_per_worker[-2*w:-w]
            avg_rw      = sum(recent_rw) / len(recent_rw)

            # Trend arrow: compare this window to the previous one
            if prev_rw:
                prev_avg = sum(prev_rw) / len(prev_rw)
                delta_pct = (avg_rw - prev_avg) / (abs(prev_avg) + 1e-8) * 100
                if delta_pct > 2.0:
                    trend = "+"
                elif delta_pct < -2.0:
                    trend = "-"
                else:
                    trend = "="
            else:
                trend = " "

            recent_grades = episode_grades[-w:]
            win_rate = sum(1 for g in recent_grades if g in ("A", "B")) / max(1, len(recent_grades))

            ot_rate  = sum(episode_ot_flags[-w:]) / w
            ord_pct  = sum(episode_ord_pct[-w:]) / w

            # Loss averages over the window (reset for next window)
            n_upd = max(1, loss_accum["n"])
            pl   = loss_accum["policy_loss"]   / n_upd
            vl   = loss_accum["value_loss"]    / n_upd
            ent  = loss_accum["entropy"]       / n_upd
            clip = loss_accum["clip_fraction"] / n_upd
            loss_accum = {"policy_loss": 0.0, "value_loss": 0.0,
                          "entropy": 0.0, "clip_fraction": 0.0, "n": 0}

            elapsed = time.time() - start_time
            n_workers = current_config.num_workers
            n_tasks   = current_config.num_tasks

            # Grade badge — last episode's grade
            grade_badge = f"[{last_grade}]"

            cfg_str = f"{configs_seen:4d}/{total_configs} (yr {years_on_current_config}/{years_per_config})"
            size_str = f"N={n_workers:2d} M={n_tasks:2d}"

            print(
                f"  {grade_badge:<6} "
                f"Ep {ep:6d}/{n_episodes} | "
                f"Stg {current_stage} | "
                f"Cfg {cfg_str:<20} | "
                f"{size_str:<10} | "
                f"R/W {avg_rw:7.1f}{trend} | "
                f"Win {win_rate:4.0%} | "
                f"OT {ot_rate:4.0%} | "
                f"Cmp% {ord_pct:4.0%} | "
                f"P:{pl:.3f} V:{vl:.3f} H:{ent:.3f} Clip:{clip:.0%} | "
                f"{elapsed:6.0f}s"
            )

        if ep % save_interval == 0:
            agent.save(output_path, ep, current_config)
            print(f"  [Checkpoint saved -> {output_path}]")

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
    parser.add_argument("--years-per-config", type=int, default=5,
                        help="Years to train on each facility config before sampling a new one.")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--log-dir", type=str, default="clark/data/logs/pretrain")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    pretrain(
        n_episodes=args.episodes,
        output_path=args.output,
        log_dir=args.log_dir,
        years_per_config=args.years_per_config,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        device=args.device,
    )


if __name__ == "__main__":
    main()
