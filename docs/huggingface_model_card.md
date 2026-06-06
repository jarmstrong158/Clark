---
license: other
license_name: polyform-noncommercial-1.0.0
license_link: https://github.com/jarmstrong158/Clark/blob/master/LICENSE
tags:
  - reinforcement-learning
  - ppo
  - operations-research
  - scheduling
  - warehouse
  - transformer
pipeline_tag: reinforcement-learning
---

# Clark — Foundation RL Model for Warehouse Workforce Scheduling

**Clark is a transformer + LSTM PPO agent that schedules a warehouse workforce.** It pre-trains once on thousands of synthetic facilities, then fine-tunes to a specific facility in ~50 episodes (~3 h on a consumer GPU). One foundation model, many facilities — variable numbers of workers and tasks, no per-site retrain from scratch.

- **Code, docs, CLI, setup wizard, operations dashboard:** https://github.com/jarmstrong158/Clark
- **Architecture (`clark-v2.5`, ~18M params):** per step, workers and tasks are tokenized separately; workers self-attend, then cross-attend to tasks; an LSTM carries state across the simulated year; per-worker assignment + hustle heads sample under action masks. Trained with PPO using per-worker importance-sampling ratios (IPPO-style), symlog value targets (DreamerV3), and a completion-dominant order reward.
- **This checkpoint:** refinement level **v2.11** — carries the full v2.5→v2.11 chain (multi-gate filler mask, restock/OT/management tuning, management-backlog observation, and the task-flow ramp + minimum-dwell cadence work that makes workers hold tasks in realistic ~30–60-min blocks).

## Intended use

Generate daily shift plans (per-worker, per-10-min task assignments), project a day's outcome (grade + completion distribution), and answer staffing questions for a configured facility. **Inputs are out-of-distribution beyond the trained bounds** (3–50 workers, 3–15 tasks, see `clark/config/clark_limits.yaml`); expanding them requires retraining.

## Usage

```bash
git clone https://github.com/jarmstrong158/Clark.git && cd Clark
pip install -e .

# download this checkpoint to where Clark expects it
hf download jarmstrong158/clark-foundation clark_foundation.pt \
  --local-dir clark/data/checkpoints

# serve it (deploy at temperature tau ~ 1.0 — argmax catastrophically underperforms)
clark serve   --model clark/data/checkpoints/clark_foundation.pt --facilities-dir clark/data/configs
# or fine-tune to your facility
clark finetune --config my_warehouse.yaml --base clark/data/checkpoints/clark_foundation.pt --episodes 50
```

> **Serve-time temperature matters.** This policy is trained in a distribution-mixing regime (PPO entropy bonus). Argmax inference drops to ~13% ship-win on stage-3 configs vs ~93% at `tau ≈ 1.0`. Deploy at `tau ≈ 1.0`.

## Performance

Validated on the predecessor "[Jack](https://github.com/jarmstrong158/Jack)" single-facility setup, translated faithfully to a Clark config and simulated over a full work-year:

| Metric | v2.10 foundation alone (no per-facility training) | + 50 fine-tune episodes |
|---|---|---|
| A-grade days | 57.5% (matches Jack-from-scratch's 58%) | 62.1% |
| A + B days | 85.1% | 95.8% |
| F-grade days | 15.0% | 4.2% |
| Per-facility training | none | ~50 episodes (~0.2 sim years) vs Jack's ~9.4 |

v2.11 additionally makes the per-worker schedule realistic — task switching drops from ~29/worker/day (a flip every ~10 min) to ~14 (every ~38 min) via a structural minimum-dwell mask, with grades held (A+B ~90%, 100% completion on the validation facility). Full methodology, the honest F-rate read, and the v2.5→v2.11 iteration log are in the [repo README](https://github.com/jarmstrong158/Clark#readme) and [CHANGELOG](https://github.com/jarmstrong158/Clark/blob/master/CHANGELOG.md).

## Limitations

- Numbers above are on synthetic facilities + the Jack-translated config; real facilities will vary, and a per-facility fine-tune is recommended.
- A small irreducible F-rate (~2–4%) on the hardest stress configs is by design (days that exceed rescue capacity), not reward-hacked away.
- Trained for the bounds in `clark_limits.yaml`; out-of-distribution facilities need retraining.

## License

Released under **[PolyForm Noncommercial 1.0.0](https://github.com/jarmstrong158/Clark/blob/master/LICENSE)** — the same license as the source. Free to download, run, study, evaluate, and fine-tune for any **noncommercial** purpose (research, personal, educational, journalism). Running Clark in production for a for-profit operation, or selling a product/service built on it, requires a separate commercial agreement — open a `commercial-access` issue on the [GitHub repo](https://github.com/jarmstrong158/Clark).

## Citation

```bibtex
@software{armstrong_clark,
  author = {Armstrong, Jonathan},
  title  = {Clark: a foundation reinforcement learning model for warehouse workforce scheduling},
  url    = {https://github.com/jarmstrong158/Clark}
}
```
