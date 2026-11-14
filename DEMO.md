# Guided experiment

Use defaults: seed 42, 540 days, noise 0.8, event strength 12, trend/weekday removal, 28-day windows and 3-day allowed lag.

## Premiere: discover geography-dependent responses

In Motifs & anomalies, choose TV+ · US and compare with TV+ · UK. Inspect window start 150, which surrounds the planted premiere at day 162. The normalized profile distance is approximately **0.222**. Canada is similarly close (**0.230**), while India (**0.849**) and China (**1.079**) have weaker shape matches. These numbers are computed from the seeded data and preprocessing, not real business observations.

In Groups & departures, set the baseline to 150 and inspect the current window around 430. Compare peer turnover across signals, then switch off the India departure and examine which changes remain. Adjust the group threshold to see how fragile or stable the grouping is. Seasonal adjustment and noise may change which series group together; the planted scenario is ground truth, not a guarantee every parameter setting recovers it.

## Recurring patterns and unusual windows

Choose a self-join. Low points identify a window with a similar shape elsewhere; high points identify a relatively unusual shape. Use the window slider to inspect the actual nearest match. A discord is not automatically a business incident: events, noise and preprocessing edges can all explain it.

## Calendar alignment matters

Set lag to zero to require same-date matching, then increase it to allow a delayed response. For cross-series joins the chart is a lag-constrained AB matrix profile. For groups the heatmap is a local symmetric subsequence distance, not a full-history correlation matrix.
