# Weekly departure detection

## Goal and experiment

Can a weekly VP review flag one member of a previously related pair within seven or fourteen days? We compare causal shape joins, peer residual means and two-sided residual CUSUM. The controlled pair is labeled US TV+ and US Fitness+ but is a separate illustrative scenario, not an assertion about the original synthetic series or real Apple data.

Training uses days 0–83 to fit a frozen peer regression (intercept, peer signal and weekday effects). Days 84–167 calibrate empirical score thresholds. Reporting starts at day 174 and repeats every seven days. A departure begins at day 210. Each report flags if any daily threshold exceedance was confirmed in its preceding seven days. Persistence is consecutive daily exceedances, not consecutive weekly reports. All scoring uses data available at the report date; candidate shape matches only look backward. The first scheduled report after departure is day 216, so the first opportunity in this particular calendar is six days after onset. Other report-day phases would change delays.

We sweep 7/14/28/42-day windows, zero/three-day backward matching lag, 95th/99th calibration quantiles and one/three-day persistence. CUSUM has no finite window and uses a fixed 0.5-sigma allowance; it is displayed with Window=0. Lag applies only to shape joins. Defaults use eight tuning seeds (42–49) and eight separate held-out seeds (10042–10049). Pick the setting on tuning seeds, then report held-out performance without reselecting on held-out results. Repeated inspection and retuning would make those seeds tuning data too.

A matched no-departure control uses identical random noise and common history. Its weekly flag rate is the false-alert metric, across the full reporting interval. Repeated weekly flags can be the same persistent episode. Detection rate means the first flagged weekly report following onset, not proof an alert was caused by that event. Pre-event false alerts and post-event misses must also be examined. The tests establish software properties, not statistical reliability.

## What to try beyond matrix profiles

| Approach | Historical learning | Recent evidence | Fit for weekly peer departure |
|---|---|---|---|
| Frozen peer model + residual mean | Learn peer membership and expected relationship over months | Last 7–14 days | Strong first baseline for level/growth departure; preserve absolute residual magnitude. Implemented here. |
| Residual CUSUM | Fit expected relationship and noise first | Accumulates evidence each day | Useful for sustained small shifts; false alarms depend on calibration and baseline stability. Implemented here. |
| Rolling residual correlation + clustering/graph | Learn stable peer groups over longer history | Update similarities weekly | Avoid estimating a large fresh correlation matrix from only seven points. Use shrinkage, historical priors and stable peers; clustering alone is not a departure detector. |
| Dynamic factor / robust peer forecast | Learn shared drivers and expected loading over history | Recent forecast errors | A practical route for many KPIs, with robust peer aggregation and historical seasonal baselines. Fit monitoring and multiple-testing control are needed. |
| ADWIN on peer residuals | Ongoing residual stream | Adaptive evidence window | Detects changes in a statistic; not a clustering system and no seven-day guarantee. Requires suitable bounded/scaled inputs and error assumptions. |
| Bayesian online change points | Priors plus past residual distribution | Sequential posterior | Can express uncertainty about a change; hazard and observation model matter. Not implemented in this experiment. |
| FLOSS / FLUSS matrix-profile segmentation | Time-series subsequence neighborhood structure | Streaming / retrospective segmentation | Finds regime boundaries in a series. It does not directly identify departures from cross-series peer groups, and subsequence length still controls sensitivity. |

Recommended architecture: learn a stable peer graph from a longer clean baseline, freeze that graph for a monitoring epoch, forecast each KPI from robust peers, and monitor daily residuals with seven-day summaries plus a calibrated sequential detector. Report evidence once a week: which KPI changed, peers that stayed stable, magnitude, onset interval, confidence and false-alert history. Relearn peer groups deliberately rather than letting a changed series immediately redefine its own expected behavior. When scaling to hundreds of series, calibrate report-level or family-level alert rates; pairwise false-alert rates do not transfer automatically.

## Source references

- STUMPY FLOSS/FLUSS implementation and literature links: https://github.com/TDAmeritrade/stumpy/blob/main/stumpy/floss.py
- River ADWIN algorithm/API: https://riverml.xyz/latest/api/drift/ADWIN/
- Scikit-learn clustering methods: https://scikit-learn.org/stable/modules/clustering.html
- Ruptures offline change-point methods: https://centre-borelli.github.io/ruptures-docs/ (offline methods require rolling re-evaluation; do not present them as online alerts without checking lookahead)
- Bayesian Online Changepoint Detection, Adams & MacKay: https://arxiv.org/abs/0710.3742

These are method references, not benchmark evidence that any method meets a seven-day service level on business data. Only the first three detectors are evaluated here. The standalone report contains measured synthetic results and their limits.

## Reproduce the offline report

```bash
python build_report.py
```

This writes `reports/weekly_detection.html` with embedded Plotly JavaScript; it needs no server, Streamlit or network connection to view. CSV sweeps are saved in `data/`. Open the HTML directly in your browser.
