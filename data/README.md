# Benchmark artifacts

- `performance_benchmark.csv/json`: runtime over 70, 140 and 280 signals; windows 7, 14, 28, 42. Standard workloads have three runs (median and maximum). All-pairs history is one measured run per tested 280-series window. Process peak RSS includes the interpreter and every earlier workload in that worker, not only the final algorithm.
- `event_suite_cadence_runtime.csv`: three repeat full recomputations, including synthetic input generation, all fourteen score arrays, thresholds and reporting for the selected policies. Reproduce with `python benchmark_detectors.py` after selecting settings.
- `event_suite_family_runtime.csv`: three repeat stage timings; per-method stages that use learned peers additionally require shared peer fitting. Reproduce with `python benchmark_detectors.py`.
- `event_suite_runs.csv`: raw TP/FP/FN/TN, event counts, matched-control flags and delay totals for every setting, noise, scenario and seed. Tuning and held-out splits are explicit.
- `event_suite_pooled.csv`: all settings, pooled over scenarios/noise by summing counts. This is exploratory output; do not select on held-out rows.
- `event_suite_selected_settings.csv`: settings chosen only on tuning seeds, maximizing weekly-label F1 under a 5% matched-control per-signal weekly-flag budget. If a family cannot meet that budget, its least-false-alert setting is retained and explicitly marked `tuning_budget_met=False`; this is a diagnostic, not an acceptable policy.
- `event_suite_summary.csv`: selected settings, held-out pooled errors and event delays.
- `event_suite_by_scenario.csv`: selected settings, held-out metrics by scenario and noise.
- `event_suite_runtime.csv`: total fourteen-configuration score generation for every dataset and control. Different datasets provide a runtime distribution, not repeated identical-input trials.
- `event_suite_metadata.json`: seeds, size, elapsed full sweep and process high-water memory.
- `suite_mixed_signals.csv`: 180 × 280 mixed-event dataset for seed 42/noise 0.8. All values synthetic.
- `suite_series_metadata.csv`: integer series index maps to LOB/geography and explicitly defined synthetic test cohorts. Series indices correspond to signal columns and event labels.
- `suite_mixed_events.csv`: series index, type, inclusive start/end day. Index zero is the first observation. Each row is a series/event instance; random spikes each have one-day labels. Mixed events may overlap and a single report can cover multiple labels.

Weekly label denominators are signal/report instances, not independent observations. Event recall is series/event-instance recall; larger targeted cohorts contribute more instances. Noise levels and event types are pooled by counts, not equally weighted macro averages. Matched-control counts repeat across scenarios because each seed/noise uses the same no-event control; do not treat those repeats as independent trials. All detection metrics describe temporal association to injected events, not causal identification.

The other CSV files are snapshots and results from the earlier two-series experiment.

## Ensemble / aggregation / smoothing extension

- `extensions_runs.csv.gz`: raw counts for every setting/seed/scenario/noise/split; gzip-compressed CSV.
- `extensions_catalog.csv`: score candidates, group, weekly/daily frequency, and ensemble strategy.
- `extensions_pooled.csv`: all settings on tuning and held-out seeds; select on tuning only.
- `extensions_selected.csv`: tuning-selected settings for six main groups, five fusion rules and four LOESS spans; budget failures remain explicit.
- `extensions_selected_runs.csv`: held-out per-seed counts for those selected policies.
- `extensions_summary.csv` / `extensions_by_scenario.csv`: held-out metrics.
- `extensions_lift.csv`: paired F1, seven-day event-recall and control-flag differences; 2,000 seed-cluster bootstrap draws. Intervals are exploratory and not multiple-comparison-adjusted.
- `extensions_runtime.csv` / `extensions_metadata.json`: timing and protocol metadata.

Extension tuning seeds are 142–147 and held-out seeds are 20042–20049, independent of the original benchmark seeds. All algorithms in a comparison share matched signals/events. Six-month histories produce only eight weekly training and eight weekly calibration observations. Weekly means are dated at bin completion. LOESS is trailing local-linear tricube smoothing without robust reweighting, evaluated at the latest observation. Weighted detector scores are calibration-scaled evidence, not probabilities.
