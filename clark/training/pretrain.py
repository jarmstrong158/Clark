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

from clark.agent.actions import get_action_mask, get_hustle_mask
from clark.agent.ppo import ClarkAgent
from clark.sim_logging.training_metrics_logger import TrainingMetricsLogger
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
    use_amp: bool = True,
    compile_model: bool = False,
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

    # Resume from checkpoint if one exists at the output path, otherwise start fresh.
    start_episode = 1
    agent_kwargs = dict(device=device, use_amp=use_amp, compile_model=compile_model)
    if os.path.exists(output_path):
        try:
            import torch as _torch
            _ckpt = _torch.load(output_path, map_location="cpu", weights_only=False)
            agent = ClarkAgent.load(output_path, **agent_kwargs)
            start_episode = int(_ckpt.get("episode", 0)) + 1
            # Force live PPO_DEFAULTS LR onto the resumed optimizer. Without
            # this the saved hparams (which include the old LR) silently
            # override the current default, so changing PPO_DEFAULTS["lr"]
            # has no effect on resumed runs.
            from clark.agent.ppo import PPO_DEFAULTS as _PPO_DEFAULTS
            new_lr = float(_PPO_DEFAULTS["lr"])
            if abs(agent.hparams.get("lr", new_lr) - new_lr) > 1e-12:
                old_lr = agent.hparams["lr"]
                agent.hparams["lr"] = new_lr
                for g in agent.optimizer.param_groups:
                    g["lr"] = new_lr
                print(f"  [Resume] LR override: {old_lr:.1e} -> {new_lr:.1e}")
            print(f"  [Resume] Loaded checkpoint — resuming at episode {start_episode}/{n_episodes}")
        except Exception as e:
            print(f"  [Resume] Could not load checkpoint ({e}) — starting fresh.")
            agent = ClarkAgent(**agent_kwargs)
    else:
        agent = ClarkAgent(**agent_kwargs)

    # Report runtime config so the user knows what's actually active.
    print(f"  [Device] model on {agent.device} "
          f"(amp={'bf16' if agent._use_amp else 'off'}, "
          f"compile={'on' if compile_model else 'off'})")

    # Lightweight logger — pretrain mode strips step data to control log size
    logger = EpisodeLogger(log_dir=log_dir, mode="pretrain")

    print(f"Pre-training Clark foundation model")
    print(f"  Episodes (years): {n_episodes}  (start: {start_episode})")
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

    # Fast-forward curriculum counters to match resumed episode
    configs_seen: int = (start_episode - 1) // years_per_config
    years_on_current_config: int = (start_episode - 1) % years_per_config
    current_config = None
    current_stage: int = _get_stage(configs_seen)

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

    for ep in range(start_episode, n_episodes + 1):
      try:
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
            hmask = get_hustle_mask(env.day_env)

            (task_actions, hustle_actions,
             task_lp, hustle_lp, value) = (
                agent.select_action_from_dict(state_dict, mask, hustle_mask=hmask)
            )

            # Build action list: (worker_id, task_idx, hustle_bool)
            # Use len(task_actions) — temp peak-staffing workers push N above config.num_workers
            actions = [
                (i, task_actions[i], bool(hustle_actions[i]))
                for i in range(len(task_actions))
            ]

            _, reward, done, info = env.step(actions)
            # Reward clip kept as safety ceiling; per-N normalization removed
            # (squashed the positive learning signal — see batched path).
            reward = float(max(-5000.0, min(5000.0, reward)))
            episode_reward += reward
            steps += 1

            agent.store_transition(
                state_dict, task_actions, hustle_actions,
                task_lp, hustle_lp, value, reward, done, mask, hmask,
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

      except Exception as exc:
        import traceback
        print(f"\n\n  [CRASH] Episode {ep} failed: {exc}")
        traceback.print_exc()
        # Emergency save so we don't lose all progress
        try:
            emergency_ep = max(1, ep - 1)
            agent.save(output_path, emergency_ep, current_config)
            print(f"  [Emergency checkpoint saved at ep {emergency_ep} -> {output_path}]")
        except Exception as save_err:
            print(f"  [Emergency save ALSO failed: {save_err}]")
        raise  # re-raise so caller sees the error

    # Final save
    final_config = generate_random_facility()  # dummy config for metadata
    agent.save(output_path, n_episodes, final_config)
    print(f"\nPre-training complete. {n_episodes} episodes in {time.time() - start_time:.0f}s")
    print(f"Foundation model saved to: {output_path}")


def pretrain_batched(
    n_episodes: int = 10000,
    n_envs: int = 8,
    output_path: str = "clark/data/checkpoints/clark_foundation.pt",
    log_dir: str = "clark/data/logs/pretrain",
    years_per_config: int = 5,
    save_interval: int = 50,
    log_interval: int = 10,
    device: str = "cpu",
    use_amp: bool = True,
    compile_model: bool = False,
    use_mp: bool = False,
) -> None:
    """Tier-2 batched pretraining: N parallel envs × single batched model forward.

    Semantics match `pretrain()` from the logging user's perspective — episodes
    are counted globally across all env slots, curriculum stage progresses as
    new configs are minted, and the same checkpoint/log layout is produced.

    The difference: each tick advances N envs simultaneously and the model
    runs ONE batched forward over all of them. Updates are per-env (triggered
    on that env's day boundary), which matches the single-env cadence.
    """
    import os
    from clark.training.batched_runner import BatchedRunner
    from clark.training.mp_runner import MultiprocessRunner
    from clark.agent.actions import get_action_mask

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    total_configs = max(1, n_episodes // years_per_config)
    stage1_end = max(1, int(total_configs * 0.15))
    stage2_end = max(1, int(total_configs * 0.45))

    # Shared curriculum counter — one integer across all env slots.
    configs_seen_state = {"count": 0}

    def _get_stage(count: int) -> int:
        if count < stage1_end:
            return 1
        if count < stage2_end:
            return 2
        return 3

    def _config_factory() -> "FacilityConfig":
        stage = _get_stage(configs_seen_state["count"])
        cfg = generate_random_facility(stage=stage)
        configs_seen_state["count"] += 1
        return cfg

    # Agent init / resume (reuses single-env save format)
    start_episode = 1
    agent_kwargs = dict(device=device, use_amp=use_amp, compile_model=compile_model)
    if os.path.exists(output_path):
        try:
            import torch as _torch
            _ckpt = _torch.load(output_path, map_location="cpu", weights_only=False)
            agent = ClarkAgent.load(output_path, **agent_kwargs)
            start_episode = int(_ckpt.get("episode", 0)) + 1
            # Force live PPO_DEFAULTS LR onto the resumed optimizer. Without
            # this the saved hparams (which include the old LR) silently
            # override the current default, so changing PPO_DEFAULTS["lr"]
            # has no effect on resumed runs.
            from clark.agent.ppo import PPO_DEFAULTS as _PPO_DEFAULTS
            new_lr = float(_PPO_DEFAULTS["lr"])
            if abs(agent.hparams.get("lr", new_lr) - new_lr) > 1e-12:
                old_lr = agent.hparams["lr"]
                agent.hparams["lr"] = new_lr
                for g in agent.optimizer.param_groups:
                    g["lr"] = new_lr
                print(f"  [Resume] LR override: {old_lr:.1e} -> {new_lr:.1e}")
            print(f"  [Resume] Loaded checkpoint — resuming at episode {start_episode}/{n_episodes}")
        except Exception as e:
            print(f"  [Resume] Could not load checkpoint ({e}) — starting fresh.")
            agent = ClarkAgent(**agent_kwargs)
    else:
        agent = ClarkAgent(**agent_kwargs)

    # Fast-forward the curriculum counter to match the resumed episode.
    # WITHOUT this, every restart resets configs_seen_state to 0 — meaning
    # the curriculum thinks it's back in stage 1, even at episode 1500+.
    # The single-env path does this at line ~122; the batched path was
    # missing the parallel logic, silently capping training at stage 1
    # across all resumes. Verified by inspecting episode metrics: 520
    # post-resume episodes all tagged stage=1 despite ep ranging up to
    # 1695 (well past stage1_end of 1497 episodes-equivalent).
    if start_episode > 1:
        resumed_configs = (start_episode - 1) // max(1, years_per_config)
        configs_seen_state["count"] = resumed_configs
        print(f"  [Resume] Fast-forward curriculum counter to {resumed_configs} configs (stage {_get_stage(resumed_configs)})")

    agent.init_hidden_batched(n_envs=n_envs)
    if use_mp:
        runner = MultiprocessRunner(
            n_envs=n_envs,
            config_factory=_config_factory,
            years_per_config=years_per_config,
        )
    else:
        runner = BatchedRunner(
            n_envs=n_envs,
            config_factory=_config_factory,
            years_per_config=years_per_config,
        )

    logger = EpisodeLogger(log_dir=log_dir, mode="pretrain")
    metrics = TrainingMetricsLogger(log_dir=log_dir)
    metrics.update_status(
        n_episodes_target=n_episodes,
        current_episode=start_episode - 1,
        alive=True,
    )

    runner_kind = "MP" if use_mp else "in-proc"
    print(f"Pre-training Clark foundation model — BATCHED (n_envs={n_envs}, runner={runner_kind})")
    print(f"  Episodes (years): {n_episodes}  (start: {start_episode})")
    print(f"  Years per config: {years_per_config}")
    print(f"  Total configs:    {total_configs}")
    print(f"  Device: {agent.device} (amp={'bf16' if agent._use_amp else 'off'}, compile={'on' if compile_model else 'off'})")
    print()
    print(
        f"  {'Grade':<6} {'Episode':<14} {'Stage':<7} {'Cfgs':<10} "
        f"{'Size':<10} {'R/W':>8} {'Win':>6} {'OT':>6} {'Cmp%':>6} "
        f"{'P-loss':>8} {'V-loss':>8} {'Entr':>6} {'Clip':>6} {'Time':>7}"
    )
    print("  " + "-" * 130)

    ep = start_episode - 1
    start_time = time.time()

    episode_reward_per_worker: list[float] = []
    episode_grades: list[str] = []           # last day's grade per episode (badge)
    episode_year_win_rates: list[float] = [] # full-year win rate per episode (real signal)
    episode_year_grades_flat: list[str] = [] # all 261 day-grades per episode, flat
    episode_ot_flags: list[bool] = []
    episode_ord_pct: list[float] = []
    # Reward-component aggregator — summed across all episodes since the last
    # progress print, so we can see WHICH reward keys are dominating the
    # signal the agent is being trained on. Reset every log_interval window.
    reward_breakdown_window: dict[str, float] = {}
    last_grade = "?"
    last_cfg = None

    loss_accum = {"policy_loss": 0.0, "value_loss": 0.0,
                  "entropy": 0.0, "clip_fraction": 0.0, "n": 0}

    tick = 0
    last_heartbeat = 0.0
    heartbeat_interval_s = 10.0  # one terse in-place line every 10s
    # Rolling buffer of completed-day digests for the periodic detailed report.
    intra_year_days: list[dict] = []
    last_intra_report = 0.0
    intra_year_report_interval_s = 60.0
    last_metrics_write = 0.0
    metrics_write_interval_s = 30.0
    # Tick-rate computation (rolling)
    tick_rate_window_start = 0.0
    tick_rate_window_ticks = 0
    last_tick_rate = 0.0
    # Last seen PPO health (for heartbeat one-liner)
    last_clip = 0.0; last_vloss = 0.0; last_entropy = 0.0

    # ── Pipelined warmup ──────────────────────────────────────────────────────
    # Pre-fill prev_* with the very first tick's data and kick the workers off
    # so the loop's first step_recv has something to drain. This lets the loop
    # body run bookkeeping for tick N in parallel with workers stepping tick N+1.
    reset_flags = runner.pop_just_reset_flags()
    agent.reset_hidden_slots(reset_flags)
    for b, flag in enumerate(reset_flags):
        if flag:
            agent.snapshot_entry_hidden_for_slot(b)
    # Snapshot the runner's cached state lists. runner.states() returns the
    # underlying _states list by reference; step_recv mutates it next iter,
    # which would silently re-point prev_states[b] to the post-step state and
    # blow up the store_transition / PPO update later. list(...) is enough —
    # the dicts themselves are reassigned (not mutated) by the runner.
    prev_states = list(runner.states())
    prev_masks = list(runner.masks())
    prev_hmasks = list(runner.hustle_masks())
    prev_ta, prev_ha, prev_t_lps, prev_h_lps, prev_v = agent.select_action_batched(
        prev_states, prev_masks, hustle_masks=prev_hmasks,
    )
    runner.step_send(prev_ta, prev_ha)

    while ep < n_episodes:
      try:
        tick += 1

        # Heartbeat so the terminal isn't silent for ~9 minutes between
        # episode completions. Overwrites itself in-place so the log stays
        # readable once real progress lines appear.
        now = time.time()
        if now - last_heartbeat >= heartbeat_interval_s:
            elapsed = now - start_time
            day_idxs = runner.current_day_idxs()
            avg_day = sum(day_idxs) / max(1, len(day_idxs))
            max_day = max(day_idxs) if day_idxs else 0
            total_days = runner.total_work_days
            # Tick rate over this heartbeat window.
            if tick_rate_window_start > 0:
                window_dt = now - tick_rate_window_start
                if window_dt > 0:
                    last_tick_rate = (tick - tick_rate_window_ticks) / window_dt
            tick_rate_window_start = now
            tick_rate_window_ticks = tick
            # Quick-glance window for the heartbeat — last 100 days from
            # this run's intra_year buffer.
            recent_days = intra_year_days[-100:]
            if recent_days:
                wins = sum(1 for d in recent_days if d.get("grade") in ("A", "B"))
                ots = sum(1 for d in recent_days if d.get("ot"))
                cmps = sum(d.get("completion", 0.0) for d in recent_days) / len(recent_days)
                last_summary = (f"last{len(recent_days):>3}: "
                                f"win {wins*100//len(recent_days):>3d}% "
                                f"OT {ots*100//len(recent_days):>3d}% "
                                f"Cmp {int(cmps*100):>3d}%")
            else:
                last_summary = "last  0: (waiting for first day)"
            msg = (f"  >> tick {tick:>6} | "
                   f"day {avg_day:4.1f}/{total_days} "
                   f"| {last_tick_rate:4.1f}t/s "
                   f"| clip {last_clip*100:4.1f}% V {last_vloss:5.2f} H {last_entropy:5.2f} "
                   f"| {last_summary} "
                   f"| {int(elapsed/60):>2}m")
            print(msg + " " * 10, end="\r", flush=True)
            last_heartbeat = now

            # Update status every heartbeat (cheap, just dict assignment),
            # but throttle the on-disk write so we're not re-serializing
            # the entire metrics file every 2s. Dashboard polls every 5s
            # so 30s write cadence is plenty for "live" feel.
            metrics.update_status(
                current_episode=ep,
                ticks=tick,
                elapsed_s=elapsed,
                alive=True,
            )
            if now - last_metrics_write >= metrics_write_interval_s:
                metrics.write()
                last_metrics_write = now

            # Drain every heartbeat (cheap) so the buffer doesn't grow
            # unbounded between intra-year reports.
            drained = runner.drain_recent_days() if hasattr(runner, "drain_recent_days") else []
            intra_year_days.extend(drained)
            # Also push them into the metrics file so the dashboard can
            # show day-level grade trend long before any year completes.
            if drained:
                metrics.record_day_grades(drained)

            # Intra-year grade report — fires every ~60s with three
            # fixed-size sliding windows so we can see whether the recent
            # trend is up, flat, or spiraling. Older window on the left,
            # newer window on the right; eyeball direction = improvement.
            if (now - last_intra_report >= intra_year_report_interval_s
                    and intra_year_days):

                def _window_summary(days_slice: list[dict]) -> str:
                    if not days_slice:
                        return "n=0"
                    gd = {g: 0 for g in "ABCDF"}
                    ot_count = 0
                    cmp_sum = 0.0
                    for d in days_slice:
                        g = d.get("grade", "?")
                        gd[g] = gd.get(g, 0) + 1
                        if d.get("ot"): ot_count += 1
                        cmp_sum += d.get("completion", 0.0)
                    n = len(days_slice)
                    wins = gd.get("A", 0) + gd.get("B", 0)
                    return (f"n={n:>3} win {wins*100//n:>3d}% "
                            f"OT {ot_count*100//n:>3d}% "
                            f"Cmp {int(cmp_sum/n*100):>3d}% "
                            f"A{gd.get('A',0)}/B{gd.get('B',0)}/"
                            f"C{gd.get('C',0)}/D{gd.get('D',0)}/F{gd.get('F',0)}")

                w100 = intra_year_days[-100:]
                w50 = intra_year_days[-50:]
                w25 = intra_year_days[-25:]
                # Newline first so we don't clobber the \r heartbeat line.
                print(
                    f"\n  ... [trend] 100d: {_window_summary(w100)}"
                    f"\n  ... [trend]  50d: {_window_summary(w50)}"
                    f"\n  ... [trend]  25d: {_window_summary(w25)}"
                    f"\n  ... [ppo]   clip {last_clip*100:5.2f}% "
                    f"V {last_vloss:6.3f} H {last_entropy:5.3f} "
                    f"tick/s {last_tick_rate:5.1f}",
                    flush=True,
                )
                last_intra_report = now
                # Trim to keep memory bounded but keep enough for 100d window.
                if len(intra_year_days) > 400:
                    intra_year_days = intra_year_days[-200:]

        # Block until workers finish the previously-sent step.
        results = runner.step_recv()

        # Zero LSTM slots for any env that just reset (its just_reset flag
        # was set by the worker when its episode ended). Must happen BEFORE
        # the forward that consumes its now-fresh state.
        reset_flags = runner.pop_just_reset_flags()
        agent.reset_hidden_slots(reset_flags)
        for b, flag in enumerate(reset_flags):
            if flag:
                agent.snapshot_entry_hidden_for_slot(b)

        # Read fresh states (post-step, possibly post-reset, possibly post-SWAP
        # if MP runner rotated to a new config inside step_recv). The action
        # we're about to compute is sampled FROM these states and SENT to the
        # workers — so it's correctly aligned with whatever config they're now
        # on. Snapshot the runner's cached lists so the next step_recv's
        # in-place reassignment doesn't silently rewrite our prev_* on the
        # rotate at the bottom.
        states = list(runner.states())
        masks = list(runner.masks())
        hmasks = list(runner.hustle_masks())
        task_a, hustle_a, t_lps, h_lps, values = agent.select_action_batched(
            states, masks, hustle_masks=hmasks,
        )

        # Send actions to workers IMMEDIATELY so they start stepping in the
        # background while the main thread does bookkeeping below. This is
        # the pipelining win: the next ~200ms of CPU work overlaps with the
        # ~30-100ms workers spend on env.step + state-build per tick.
        runner.step_send(task_a, hustle_a)

        # ── Bookkeeping for the PREVIOUS tick (overlaps with worker stepping) ──
        # Store transitions for the prev (state, action, log_prob, value, reward, done).
        # Per-N normalization REMOVED — Jack doesn't do it and works; for us
        # it preferentially squashed the +per_order_shipped positive signal
        # while leaving fixed event penalties relatively dominant.
        # Reward clip loosened ±5000 → ±20000: end-of-day single-step
        # penalties (per_order_incomplete = -10 × N_unshipped) can hit
        # -7000 on N=4 disasters with 700 unshipped. Tighter clip was
        # masking the catastrophic-day signal so the value head couldn't
        # distinguish a normal failure from a 4×-worse one.
        rewards = [
            float(max(-20_000.0, min(20_000.0, r["reward"])))
            for r in results
        ]
        dones = [r["done"] for r in results]
        agent.store_transition_batched(
            prev_states, prev_ta, prev_ha,
            prev_t_lps, prev_h_lps, prev_v,
            rewards, dones, prev_masks, hustle_masks=prev_hmasks,
        )

        # Per-env day-boundary PPO updates (matches single-env cadence).
        for b, r in enumerate(results):
            info = r["info"]
            if (info.get("new_day") or r["done"]) and len(agent.buffers[b]) > 0:
                m = agent._update_single_buffer(agent.buffers[b])
                if m["n_updates"] > 0:
                    loss_accum["policy_loss"]  += m["policy_loss"]
                    loss_accum["value_loss"]   += m["value_loss"]
                    loss_accum["entropy"]      += m["entropy"]
                    loss_accum["clip_fraction"]+= m["clip_fraction"]
                    loss_accum["n"] += 1
                    # Cache for terminal heartbeat.
                    last_clip = m["clip_fraction"]
                    last_vloss = m["value_loss"]
                    last_entropy = m["entropy"]
                    # Per-update PPO metrics for the dashboard's PPO-health
                    # panel. Recording every update would be too noisy; we
                    # record one in 16 to keep the file small while still
                    # tracking trends over thousands of updates.
                    if tick % 16 == 0:
                        metrics.record_ppo_update(
                            episode=ep,
                            clip_fraction=m["clip_fraction"],
                            policy_loss=m["policy_loss"],
                            value_loss=m["value_loss"],
                            entropy=m["entropy"],
                            n_updates=m["n_updates"],
                        )
                        # PopArt removed — only call record_popart if a future
                        # value head re-introduces .mu / .sigma attributes.
                        if hasattr(agent.model.value_head, "mu"):
                            metrics.record_popart(
                                episode=ep,
                                mu=float(agent.model.value_head.mu.item()),
                                sigma=float(agent.model.value_head.sigma.item()),
                            )
                agent.snapshot_entry_hidden_for_slot(b)

        # Handle episode completions: log them and advance ep counter.
        for b, r in enumerate(results):
            if not r["episode_done"]:
                continue
            fin = r["finished_episode"]
            ep += 1
            cfg = fin["config"]
            last_cfg = cfg

            # Aggregate ALL 261 days of this year, not just the last day.
            # Previously we only displayed the last day's grade — throwing
            # away 260 days of signal per episode. The year-level win rate
            # (grades A or B) and average completion are far more
            # representative of how the agent is actually doing.
            daily = fin["daily_summaries"]
            year_grades: list[str] = []
            year_ord_pcts: list[float] = []
            year_ot_count = 0
            for day_summary in daily:
                d_footer = day_summary.get("footer", {})
                d_header = day_summary.get("header", {})
                year_grades.append(d_footer.get("grade", "?"))
                d_total = d_header.get("total_orders", 1) or 1
                d_remaining = d_footer.get("orders_remaining", 0)
                year_ord_pcts.append(
                    max(0.0, (d_total - d_remaining) / d_total)
                )
                if d_footer.get("ot_hours", 0.0) > 0.0:
                    year_ot_count += 1
            n_days = max(1, len(daily))
            year_win = sum(1 for g in year_grades if g in ("A", "B")) / n_days
            year_ot_rate = year_ot_count / n_days
            year_ord_pct = sum(year_ord_pcts) / max(1, len(year_ord_pcts))

            # Bookkeeping: store year-level metrics, not last-day metrics.
            # `last_grade` becomes the most-recent day's grade for the
            # progress-line badge (still useful as a "what just happened"
            # cue) but the windowed averages are now over 261-day years.
            if daily:
                last_grade = daily[-1].get("footer", {}).get("grade", "?")
                footer = daily[-1].get("footer", {})  # for reward breakdown
            else:
                last_grade = "?"
                footer = {}

            episode_grades.append(last_grade)
            episode_ot_flags.append(year_ot_rate > 0)
            episode_ord_pct.append(year_ord_pct)
            # Track year-level win rate separately for a more honest summary
            # in the windowed display.
            episode_year_win_rates.append(year_win)
            episode_year_grades_flat.extend(year_grades)

            # Per-episode metrics for the dashboard (curriculum scatter,
            # per-config performance, cross-config trend).
            metrics.record_episode(
                episode=ep,
                stage=_get_stage(configs_seen_state["count"]),
                n_workers=cfg.num_workers,
                n_tasks=cfg.num_tasks,
                win_rate_year=year_win,
                ot_rate_year=year_ot_rate,
                completion_rate_year=year_ord_pct,
                reward_per_worker=fin["reward"] / max(1, cfg.num_workers),
                last_grade=last_grade,
                config_name=cfg.name,
            )
            episode_reward_per_worker.append(
                fin["reward"] / max(1, cfg.num_workers)
            )
            # Aggregate this episode's reward components into the window total.
            # The display below pulls the top 3 contributors so we can see
            # whether the gradient signal is dominated by the components we
            # think it is (ship/management) vs the per-step bleed.
            ep_breakdown = footer.get("reward_breakdown", {}) if footer else {}
            for k, v in ep_breakdown.items():
                reward_breakdown_window[k] = reward_breakdown_window.get(k, 0.0) + v

            if daily:
                logger.log_episode(
                    daily[-1], ep,
                    facility_config=cfg,
                    write=(ep % log_interval == 0),
                )

            # Windowed progress line + checkpoint on the same cadence as single-env.
            if ep % log_interval == 0:
                w = log_interval
                recent_rw = episode_reward_per_worker[-w:]
                avg_rw = sum(recent_rw) / max(1, len(recent_rw))
                prev_rw = episode_reward_per_worker[-2*w:-w]
                if prev_rw:
                    prev_avg = sum(prev_rw) / len(prev_rw)
                    delta_pct = (avg_rw - prev_avg) / (abs(prev_avg) + 1e-8) * 100
                    trend = "+" if delta_pct > 2.0 else ("-" if delta_pct < -2.0 else "=")
                else:
                    trend = " "
                # Year-level win rate (fraction of A/B days across each year,
                # then averaged over the window). Far more representative
                # than the old "did the last day of each year happen to be
                # an A/B" measurement.
                recent_year_wins = episode_year_win_rates[-w:]
                win_rate = sum(recent_year_wins) / max(1, len(recent_year_wins))
                ot_rate = sum(episode_ot_flags[-w:]) / w if episode_ot_flags else 0.0
                avg_ord_pct = sum(episode_ord_pct[-w:]) / max(1, len(episode_ord_pct[-w:]))
                # Also surface the per-day grade distribution in this window
                # so we can see "we got 17 A's, 32 B's, 60 C's, 80 D's, 70 F's
                # over the last 4 years" instead of guessing.
                window_day_grades = episode_year_grades_flat[-w*262:]
                grade_counts = {g: window_day_grades.count(g) for g in "ABCDF"}

                n_upd = max(1, loss_accum["n"])
                pl  = loss_accum["policy_loss"]   / n_upd
                vl  = loss_accum["value_loss"]    / n_upd
                ent = loss_accum["entropy"]       / n_upd
                clip = loss_accum["clip_fraction"] / n_upd
                loss_accum = {"policy_loss": 0.0, "value_loss": 0.0,
                              "entropy": 0.0, "clip_fraction": 0.0, "n": 0}

                elapsed = time.time() - start_time
                stage = _get_stage(configs_seen_state["count"])
                cfg_str = f"{configs_seen_state['count']:4d}/{total_configs}"
                size_str = f"N={cfg.num_workers:2d} M={cfg.num_tasks:2d}"
                # Clear the heartbeat line so it doesn't collide with the
                # persistent progress line.
                print(" " * 120, end="\r", flush=True)
                print(
                    f"  [{last_grade}]    "
                    f"Ep {ep:6d}/{n_episodes} | "
                    f"Stg {stage} | "
                    f"Cfg {cfg_str:<10} | "
                    f"{size_str:<10} | "
                    f"R/W {avg_rw:7.1f}{trend} | "
                    f"Win {win_rate:4.0%} | "
                    f"OT {ot_rate:4.0%} | "
                    f"Cmp% {avg_ord_pct:4.0%} | "
                    f"P:{pl:.3f} V:{vl:.3f} H:{ent:.3f} Clip:{clip:.0%} | "
                    f"{elapsed:6.0f}s",
                    flush=True,
                )
                # Record this window's metrics for the dashboard. Includes
                # avg reward per worker, win/OT/cmp rates, top reward
                # contributors, and per-day grade distribution across the
                # window's full years.
                metrics.record_window(
                    episode=ep,
                    avg_reward_per_worker=avg_rw,
                    win_rate=win_rate,
                    ot_rate=ot_rate,
                    completion_rate=avg_ord_pct,
                    reward_dominance=reward_breakdown_window,
                    grade_distribution=grade_counts,
                )
                metrics.update_status(
                    current_episode=ep,
                    current_stage=stage,
                    ticks=tick,
                    elapsed_s=elapsed,
                    alive=True,
                )
                metrics.write()

                # Per-day grade distribution across this window's full years
                # (not just the last day). One line per window — gives a far
                # more honest view of "is the model learning" than a single
                # cherry-picked end-of-year grade.
                gd_total = sum(grade_counts.values())
                if gd_total > 0:
                    gd_parts = [
                        f"{g}={grade_counts[g]}({grade_counts[g]*100//gd_total}%)"
                        for g in "ABCDF" if grade_counts[g] > 0
                    ]
                    print(f"           year-day grades: {' '.join(gd_parts)}",
                          flush=True)

                # Reward dominance line — top 3 components (by absolute value)
                # in the window. Helps diagnose whether the agent is being
                # pushed toward the right behaviors.
                if reward_breakdown_window:
                    top = sorted(
                        reward_breakdown_window.items(),
                        key=lambda kv: -abs(kv[1]),
                    )[:3]
                    parts = [f"{k}={v:+.0f}" for k, v in top]
                    print(f"           reward dominance: {' | '.join(parts)}",
                          flush=True)
                reward_breakdown_window = {}
                last_heartbeat = 0.0  # force an immediate heartbeat next tick

            if ep % save_interval == 0:
                agent.save(output_path, ep, cfg)
                print(f"  [Checkpoint saved -> {output_path}]")

        # Rotate state for next iter — current step's data becomes "prev" for
        # the next iter's bookkeeping. Workers are already stepping with
        # task_a/hustle_a, so this just hands the labels forward.
        prev_states = states
        prev_masks = masks
        prev_hmasks = hmasks
        prev_ta = task_a
        prev_ha = hustle_a
        prev_t_lps = t_lps
        prev_h_lps = h_lps
        prev_v = values

      except Exception as exc:
        import traceback
        print(f"\n\n  [CRASH] tick {tick} ep~{ep} failed: {exc}")
        traceback.print_exc()
        try:
            emergency_ep = max(1, ep)
            agent.save(output_path, emergency_ep, last_cfg if last_cfg else generate_random_facility())
            print(f"  [Emergency checkpoint saved at ep {emergency_ep} -> {output_path}]")
        except Exception as save_err:
            print(f"  [Emergency save ALSO failed: {save_err}]")
        # Mark training as not alive so the dashboard can show it.
        try:
            metrics.update_status(alive=False); metrics.write()
        except Exception:
            pass
        raise

    agent.save(output_path, n_episodes,
               last_cfg if last_cfg else generate_random_facility())
    print(f"\nBatched pre-training complete. "
          f"{n_episodes} episodes in {time.time() - start_time:.0f}s "
          f"({tick} ticks, n_envs={n_envs})")
    print(f"Foundation model saved to: {output_path}")

    # Mark complete in the dashboard status.
    metrics.update_status(alive=False, current_episode=n_episodes); metrics.write()

    # Clean up worker procs if we used the MP runner.
    close = getattr(runner, "close", None)
    if callable(close):
        close()


def main():
    parser = argparse.ArgumentParser(description="Pre-train Clark foundation model.")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--output", type=str,
                        default="clark/data/checkpoints/clark_foundation.pt")
    parser.add_argument("--years-per-config", type=int, default=5,
                        help="Years to train on each facility config before sampling a new one.")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=0,
                        help="Episodes between progress lines. 0 = auto "
                             "(4 for batched, 10 for single-env).")
    parser.add_argument("--log-dir", type=str, default="clark/data/logs/pretrain")
    parser.add_argument("--device", type=str, default="cpu",
                        help="torch device: 'cpu', 'cuda', 'cuda:0', etc.")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable bf16 autocast on CUDA. Default: enabled.")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model. First call has a ~30s compile cost.")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel envs for Tier-2 batched pretrain. "
                             "1 = use legacy single-env loop; >1 = batched loop. "
                             "Default 8. On GPU this gives ~2.6x over single-env; "
                             "on CPU it's roughly neutral.")
    args = parser.parse_args()

    # Auto log-interval: batched runs complete episodes in chunks of n_envs at
    # a time, so a lower number keeps the screen updating. Single-env gets the
    # original cadence.
    log_interval = args.log_interval
    if log_interval == 0:
        log_interval = 4 if args.n_envs > 1 else 10

    if args.n_envs > 1:
        pretrain_batched(
            n_episodes=args.episodes,
            n_envs=args.n_envs,
            output_path=args.output,
            log_dir=args.log_dir,
            years_per_config=args.years_per_config,
            save_interval=args.save_interval,
            log_interval=log_interval,
            device=args.device,
            use_amp=not args.no_amp,
            compile_model=args.compile,
        )
    else:
        pretrain(
            n_episodes=args.episodes,
            output_path=args.output,
            log_dir=args.log_dir,
            years_per_config=args.years_per_config,
            save_interval=args.save_interval,
            log_interval=log_interval,
            device=args.device,
            use_amp=not args.no_amp,
            compile_model=args.compile,
        )


if __name__ == "__main__":
    main()
