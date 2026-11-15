# Signal Atlas — KPI Matrix Lab

An interactive matrix-profile laboratory for exploring business signals across products and countries. **All data is synthetic**, not Apple operating data.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Python 3.11+ recommended.

## Explore

1. Start in **Signal explorer**: 50 daily signals across five businesses and ten countries, with trends, weekly/yearly seasonality, noise and planted events.
2. In **Motifs & anomalies**, select a self-join for recurring motifs and unusual windows, or a cross-series join to find shared patterns within a configurable lag. Inspect aligned nearest matches.
3. In **Groups & departures**, compare the premiere baseline with the period following the price change. The distance heatmap and peer-turnover table reveal evolving local shape groups. Toggle the injected India TV+ departure to compare scenarios.
4. Adjust seed, history, countries, noise, event strength, preprocessing, window length, lag and grouping threshold. Export signals, ground truth and distances.

## Method and interpretation

Exact NumPy z-normalized Euclidean subsequence search, divided by sqrt(window length). For nonconstant windows this is sqrt(2-2*Pearson correlation). Constants match constants at distance zero and nonconstants at distance one. Self-joins exclude ceil(window/4) temporal neighbors. Cross-joins optionally constrain calendar lag.

The full matrix profile is the nearest-neighbor distance and index for every window of a selected series. Grouping is a separate local analysis: average the two directional minimum AB distances at the selected date, threshold the resulting symmetric matrix and compute connected components. Groups may contain indirectly connected members. Peer turnover is Jaccard distance between baseline and current peer sets.

Centered trend removal and weekday adjustment are retrospective, not suitable for leakage-free real-time evaluation. Z-normalization discards absolute magnitude. Shared seasonality, threshold sensitivity and noise can create apparent relationships. No causal claims or statistical significance estimates are made. Exact full joins cost O(n²m); this implementation suits demo histories, not hundreds of multi-year high-frequency series. STUMPY/scalable joins and out-of-sample validation are next steps for production.

## Test

```bash
pip install pytest
python -m pytest -q
```

A seeded snapshot is included in `data/`. See [DEMO.md](DEMO.md) for a reproducible walkthrough with measured example distances.

Docker alternative:

```bash
docker build -t kpi-matrix-lab .
docker run --rm -p 8501:8501 kpi-matrix-lab
```

For a hosted UI, deploy `app.py` with Streamlit Community Cloud using this repository (private repository access depends on your Streamlit plan/account). The repository itself does not provision paid hosting.

## Weekly detection experiment — standalone report

Open [`reports/weekly_detection.html`](reports/weekly_detection.html) in a browser (download the file from GitHub first). It embeds all charts and works offline. It includes the original US pairs, controlled level/growth/shape departures, tuning and independent held-out results, false-alert estimates and alternative methods for large KPI universes.

Regenerate with `python build_report.py`. See [WEEKLY_DETECTION.md](WEEKLY_DETECTION.md) for experiment definitions and references. The optional **Weekly detection lab** tab exposes the same causal benchmark. Exploratory tabs now label scores by **window-ending date**; they remain retrospective, as disclosed in the UI.

## 280-series scale and event suite

The generator and app now support **seven LOBs × forty geographies**. Run these scripts for six-month runtime and quality benchmarks:

```bash
python benchmark_scale.py
python run_suite.py
```

Open `reports/performance_280.html` for measured runtime/memory and `reports/event_suite_280.html` for held-out error metrics over level shifts, random spikes, temporary seasonality, gradual trends, rise/recovery, LOB offers, developing-market offers, English-market offers, and their mixed overlap. Both reports embed charts for offline viewing. On GitHub Pages, the root landing page links to all reports.

The event suite evaluates five detector families: frozen-history matrix profiles, learned-peer shape distances, peer residual means, peer residual spikes and residual CUSUM. Training and calibration each use 56 days of the 180-day history. Four seeds tune settings; eight separate seeds validate, at two noise levels. Counts and settings are exported in `data/event_suite_*.csv`; the reproducible mixed dataset and exact event ledger are included. Full methodology and limitations appear in the report.

## Ensembles, weekly aggregation and trailing LOESS

Run `python run_extensions.py` to compare raw daily detectors, equal/weighted score fusion, OR/voting, completed seven-day aggregation, causal trailing LOESS (7/14/28/42 days), and raw-plus-transformed ensembles. The extension uses fresh tuning seeds 142–147 and fresh held-out seeds 20042–20049, the same nine event scenarios and two noise levels. It evaluates 96 score candidates / 552 settings per dataset. Weights and other settings are selected only on tuning results under the existing 5% per-signal control-alert budget.

The event HTML includes a dedicated section with held-out comparisons, paired seed-bootstrap lift intervals, weight grids, smoothing and aggregation illustrations, and scenario-level errors. Raw extension counts are compressed in `data/extensions_runs.csv.gz`. Weekly methods use only completed bins and trailing LOESS never uses future observations. The report explains the reduced weekly sample size, effective smoothing history and false-alert trade-offs.

To regenerate only the HTML from saved results, run `python render_event_report.py`; no simulation rerun is needed.
