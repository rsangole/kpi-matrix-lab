# Signal Atlas — KPI Matrix Lab

An interactive matrix-profile laboratory for exploring business signals across products and countries. **All data is synthetic**, not Apple operating data.

## Run

```bash
uv sync
uv run streamlit run app.py
```

Open http://localhost:8501. Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+ (pinned to 3.13 via `.python-version`).

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
uv run pytest -q
```

A seeded snapshot is included in `data/`. See [DEMO.md](DEMO.md) for a reproducible walkthrough with measured example distances.

Docker alternative:

```bash
docker build -t kpi-matrix-lab .
docker run --rm -p 8501:8501 kpi-matrix-lab
```

For a hosted UI, deploy `app.py` with Streamlit Community Cloud using this repository (private repository access depends on your Streamlit plan/account). The repository itself does not provision paid hosting.
