# tools/archive/

One-shot scripts from the v2.0 → v2.10 iteration chain. Kept for posterity / future
"what did we already try" lookups, but **not part of the active toolkit.** The
preferred way to evaluate a checkpoint is now `tools/clark_eval.py` (full year,
production grader, multi-stage / multi-seed).

| script | what it did | iteration |
|---|---|---|
| `ab_v2_vs_v21.py` | head-to-head probe between v2 baseline and the failed v2.1 dvc-observation prototype | v2.1 |
| `ab_three_way.py` | three-way A/B between v2 baseline, v2.5 mask, and v2.6 5-gate mask | v2.5 / v2.6 |
| `quick_v2_baseline_metrics.py` | quick baseline metrics on v2 weights | v2 |
| `audit_dvc_weights.py` | sanity-check on the v2.1 demand-vs-capacity observation column | v2.1 |
| `v25_stats_audit.py` | statistical cross-check on the v2.5 mask fine-tune trajectory | v2.5 |
| `probe_queue_pressure.py` | scan the queue_pressure observation channel for v2.2 | v2.2 |
| `probe_task_churn.py` | per-worker distinct-tasks-per-day diagnostic at multiple temperatures | v2.10 |
| `probe_v210_breakdown.py` | training-run dashboard chart for the v2.10 fine-tune | v2.10 |
| `probe_v210_trend.py` | early-vs-mid-vs-late thirds slope-fit on the v2.10 run | v2.10 |
| `probe_head_to_head.py` | **flawed** head-to-head between v2.8 and v2.10 — single-day episodes, 3-grade rule (A/C/D/F, no B). Kept as a cautionary example: it reported "essentially tied" when the production grader showed real promotion. Use `clark_eval.py` instead. | v2.10 |
| `audit_probe.py` | methodology audit that caught the `probe_head_to_head.py` flaw | v2.10 |

If you're tempted to write a new ad-hoc probe to compare two checkpoints, **use
`tools/clark_eval.py` instead** — full year, real grader, repeatable. That's the
lesson these scripts collectively taught.
