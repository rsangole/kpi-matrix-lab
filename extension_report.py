"""Render ensemble, weekly aggregation and causal smoothing results into the report."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from suite import generate_suite
from extensions import trailing_loess,WEIGHTS

GROUPS=['Daily single detector','Daily ensemble','Weekly aggregation','Trailing LOESS','Daily + weekly ensemble','Daily + LOESS ensemble']

def section(chart):
    if not Path('data/extensions_summary.csv').exists():return ''
    summary=pd.read_csv('data/extensions_summary.csv')
    selected=pd.read_csv('data/extensions_selected.csv')
    detail=pd.read_csv('data/extensions_by_scenario.csv')
    lifts=pd.read_csv('data/extensions_lift.csv')
    metadata=json.loads(Path('data/extensions_metadata.json').read_text())
    def table(df):return df.to_html(index=False,border=0,float_format=lambda v:f'{v:.3f}')
    parts=['<section id="extensions"><h2>Do ensembles, weekly aggregation or trailing LOESS add lift?</h2><p>These additions are evaluated on <b>new simulations</b>: six tuning seeds (142–147), eight fresh held-out seeds (20042–20049), both noise levels and all nine event scenarios. Every method sees the same 280 signals and event truth. Daily single-detector baselines are rerun on those same seeds. Compare the results inside this section, rather than subtracting the older benchmark table with its different seeds.</p>']
    parts.append('<nav class="anomaly-nav"><a href="#extension-results">Results and lift</a><a href="#ensemble-details">Ensemble rules and weights</a><a href="#weekly-details">Weekly aggregation</a><a href="#loess-details">Trailing LOESS</a><a href="#extension-errors">Event-specific errors</a></nav>')
    cols=['model','precision','recall','f1','false_positive_rate','event_recall_7d','event_recall_14d','miss_rate_14d','control_weekly_flag_rate','control_family_report_rate','mean_delay_detected_days']
    main=summary.set_index('model').reindex(GROUPS).reset_index()
    ref=main[main.model=='Daily single detector'].iloc[0]
    parts.append('<h3 id="extension-results">Fresh held-out comparison</h3><p>One configuration per group is selected by tuning F1 under the same 5% per-signal matched-control weekly flag budget. If a group cannot meet the budget, its least-false-alert setting remains visible and marked as a failure. Selection includes weights, smoothing span, detector window, threshold and persistence where applicable; held-out results never choose those settings. Rates are fractions (0.5 = 50%).</p>')
    parts.append(table(main[cols]))
    for group in GROUPS[1:]:
        row=main[main.model==group].iloc[0]
        delta=lifts[(lifts.model==group)&(lifts.reference=='Daily single detector')].iloc[0]
        policy=selected[selected.model==group].iloc[0]
        supported=delta.f1_low>0
        verdict='Positive F1 lift across the seed-bootstrap interval.' if supported else ('F1 is lower across the seed-bootstrap interval.' if delta.f1_high<0 else 'The seed-bootstrap interval includes no F1 improvement.')
        parts.append(f'<article><h3>{group}</h3><p>F1 change versus the paired daily baseline: <b>{delta.f1_lift*100:+.1f} percentage points</b> (95% seed-bootstrap interval {delta.f1_low*100:+.1f} to {delta.f1_high*100:+.1f}). Seven-day event recall: <b>{row.event_recall_7d:.1%}</b>, compared with {ref.event_recall_7d:.1%} for the daily baseline.</p><p>{verdict} Unchanged-control flag rate: {row.control_weekly_flag_rate:.1%} per signal/report; {row.control_family_report_rate:.1%} of reports contain at least one false flag. Tuning budget {"met" if policy.tuning_budget_met else "FAILED"}.</p></article>')
    parts.append('<p>Bootstrap intervals resample the eight held-out seed clusters 2,000 times, keeping correlated signals, event scenarios, noise levels and weekly reports together inside each seed. They describe variability within this synthetic model; they do not prove improvement on real business data. Intervals are exploratory and are not adjusted for multiple method comparisons. A positive F1 lift alone does not establish an acceptable false-alert rate or seven-day detection rate.</p>')
    fig=px.bar(main,x='model',y='f1',color='model',title='Held-out F1: selected daily baseline versus additions',labels={'f1':'Weekly signal/report F1','model':'Method group'})
    fig.update_layout(showlegend=False,height=480)
    parts.append(chart(fig))
    parts.append('<h3>Exactly which settings were selected?</h3>'+table(selected[selected.model.isin(GROUPS)]))
    parts.append('<h3>Paired lift, including false-alert cost</h3>'+table(lifts[lifts.model.isin(GROUPS)]))
    parts.append('<h3 id="ensemble-details">Combining detectors: equal weights, learned grid weights, OR and voting</h3><p>The five fixed daily components are historical matrix-profile novelty (7 days), learned-peer shape distance (7 days), peer residual mean (7 days), daily peer residual magnitude and peer residual CUSUM. Each component is divided by its own per-signal calibration-period 95th percentile so that incompatible raw units do not dominate. This scaling produces relative scores, not calibrated probabilities.</p><p>For equal/weighted means and calibrated maximum, the combined score gets its own calibration threshold. Weights are selected from the finite grid below on tuning simulations, not fitted to held-out labels or continuously optimized. All weights are positive and sum to one. The same calibration history sets component scales and fused-score thresholds, so independent held-out error estimates remain essential.</p><p><b>Naive OR</b> fires when any component exceeds its own chosen threshold; <b>2-of-5 voting</b> requires two components on the same day. They use member quantiles 95%/99%/99.5%, with no claim that their combined false-alert rate equals a member rate. Daily confirmation requires one or three consecutive exceedances; a weekly report flags if any confirmed day occurred in its preceding seven days.</p>')
    weighttable=pd.DataFrame(WEIGHTS,index=['Historical profile','Peer shape','Residual mean','Residual magnitude','Residual CUSUM']).T.reset_index(names='Weight grid')
    parts.append(table(weighttable))
    fusion=summary[summary.model.str.startswith('Fusion:')]
    parts.append(table(fusion[cols]))
    parts.append(table(selected[selected.model.str.startswith('Fusion:')]))
    weighted=lifts[(lifts.model=='Fusion: Tuned weighted mean')&(lifts.reference=='Fusion: Equal-weight mean')]
    parts.append('<p>Direct weighted-versus-equal comparison, with thresholds independently selected on tuning data:</p>'+table(weighted))
    parts.append('<p>Correlated detectors do not provide independent votes. OR can multiply the alert burden, whereas voting can discard a one-day spike detected by only one component. Weighted means can dilute a specialist detector. These measured comparisons include those costs rather than assuming fusion helps.</p>')
    parts.append('<h3 id="weekly-details">Daily observations versus completed weekly means</h3><p>Downsample to nonoverlapping seven-day means aligned to the existing reporting cadence. A weekly observation is available only at its bin end. No partial-week or future weekly value is used. Training has eight weekly observations, calibration has eight, and evaluation has nine report dates; the final incomplete week is discarded. These are fixed seven-day bins, not a separate calendar-week convention.</p><p>Weekly models relearn peers and residual regressions on the eight training weeks. Test absolute weekly residuals, two/three-week residual means, three/four-week historical shape profiles, three/four-week peer shape distances, and weekly CUSUM. A three-week shape profile already summarizes 21 days, not seven. One/two consecutive weekly exceedances are tested; confirmation is evaluated at the same report dates as daily methods. Weekly sums would differ by a constant factor of seven in this experiment, so means are used to retain the KPI index units.</p><p>The daily-plus-weekly ensemble blends an equal-weight five-detector daily score with the mean of normalized weekly residual magnitude and three-week historical-profile novelty. Completed weekly evidence is held forward until the next bin finishes; it is never backfilled. Weekly weight is selected from 25% or 50% on tuning data.</p>')
    x,_,e=generate_suite(seed=142,scenario='Random spikes');b,_,_=generate_suite(seed=142,scenario='Random spikes',changed=False)
    effect=(x-b).iloc[:,0];n=len(effect)//7;week=effect.values[:n*7].reshape(n,7).mean(axis=1)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=x.index,y=effect,name='Daily injected spike',line=dict(color='#c26323')))
    fig.add_trace(go.Scatter(x=x.index[np.arange(n)*7+6],y=week,mode='markers',name='Completed weekly mean injection',marker=dict(size=9,color='#087f70')))
    fig.update_layout(title='Averaging can attenuate or cancel short signed spikes',xaxis_title='Date available',yaxis_title='Exact injected KPI change',height=480)
    fig.update_xaxes(range=[x.index[115],x.index[165]])
    parts.append(chart(fig))
    parts.append('<p>This chart isolates the injected effect using the matched no-event baseline. A single five-unit one-day spike contributes only five-sevenths of a unit to its week’s mean; opposite signs can cancel. Aggregation can reduce independent noise, but it also removes high-frequency evidence. The short eight-week baseline limits peer and threshold estimation.</p>')
    parts.append('<h3 id="loess-details">Trailing local-linear LOESS, not centered smoothing</h3><p>For each date, fit a local linear regression to only the previous 7, 14, 28 or 42 observations, including today, with tricube distance weights, and evaluate the fitted line at today’s endpoint. This is non-robust LOESS (no residual reweighting iterations). Early dates use expanding available history. Future data never enters a fitted past point.</p><p>Rerun all daily detector families on each smoothed dataset, relearning peers, regression and thresholds within that dataset’s original training/calibration periods. Thus the comparison tests the complete smoothing-plus-detection pipeline. A 42-day smoother followed by a 42-day profile can depend on as many as 83 recent daily observations; a short reporting cadence does not remove that longer evidence footprint. Endpoint fits can overshoot and use negative effective coefficients.</p>')
    spans=summary[summary.model.str.startswith('LOESS span')]
    parts.append(table(spans[cols]))
    parts.append(table(selected[selected.model.str.startswith('LOESS span')]))
    x,_,_=generate_suite(seed=142,scenario='Gradual trend');b,_,_=generate_suite(seed=142,scenario='Gradual trend',changed=False)
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=['Noisy observed series and trailing smoothers','Exact injected effect after the same smoothing'])
    fig.add_trace(go.Scatter(x=x.index,y=x.iloc[:,0],name='Raw observed',line=dict(color='#aab4c4',width=1)),row=1,col=1)
    fig.add_trace(go.Scatter(x=x.index,y=(x-b).iloc[:,0],name='Raw injected effect',line=dict(color='#aab4c4',dash='dash')),row=2,col=1)
    for span,color in zip([7,14,28,42],['#167f70','#3557ad','#b46328','#985ba8']):
        smooth=trailing_loess(x,span);control=trailing_loess(b,span)
        fig.add_trace(go.Scatter(x=x.index,y=smooth.iloc[:,0],name=f'Trailing LOESS {span}d',legendgroup=str(span),line=dict(color=color)),row=1,col=1)
        fig.add_trace(go.Scatter(x=x.index,y=(smooth-control).iloc[:,0],name=f'LOESS {span}d effect',legendgroup=str(span),showlegend=False,line=dict(color=color)),row=2,col=1)
    for row in [1,2]:fig.add_vline(x=x.index[126].timestamp()*1000,line_dash='dot',row=row,col=1)
    fig.update_xaxes(range=[x.index[100],x.index[-1]])
    fig.update_layout(height=690,title='Smoothing a gradual departure: noise reduction versus onset distortion',legend=dict(orientation='h',y=-.12))
    parts.append(chart(fig))
    parts.append('<p>Daily-plus-LOESS fusion averages the five normalized detectors on smoothed data, then blends that score with its raw-daily counterpart. Test smoothing spans 14/28 days and smoothed weights 25%/50%; choose on tuning seeds. This preserves a raw-data path that could otherwise be erased by smoothing.</p>')
    parts.append('<h3 id="extension-errors">Where does each addition help or hurt?</h3><p>The following errors use the globally selected settings, not per-event oracle settings. Read random spikes separately from sustained trends and broad offers. Precision/F1 use signal/report labels; event recall uses series/event instances. Overlapping mixed events can share a detected report.</p>')
    d=detail[detail.model.isin(GROUPS)]
    fig=px.bar(d,x='scenario',y='event_recall_7d',color='model',facet_row='noise',barmode='group',title='Seven-day event recall by anomaly type and noise',labels={'event_recall_7d':'Event recall within seven days'})
    fig.update_layout(height=800);parts.append(chart(fig))
    parts.append('<details><summary>All scenario/noise errors for the six main comparisons</summary>'+table(d[['scenario','noise']+cols])+'</details>')
    parts.append('<details><summary>All selected configurations, including fusion rules and individual LOESS spans</summary>'+table(selected)+'</details>')
    times=pd.read_csv('data/extensions_runtime.csv')
    parts.append(f"<h3>Cost and reproducibility</h3><p>The extension sweep evaluates {metadata['candidates']} score candidates and {metadata['settings_per_dataset']} parameter settings per dataset across {metadata['datasets']} event datasets. The complete simulation/scoring/metrics sweep before summary rendering took {metadata['full_sweep_seconds']:.1f} seconds. Median time to build all candidate score arrays for one 280-series, 180-day dataset: {times.seconds.median():.3f} seconds; this excludes threshold/report evaluation, file output and rendering. Single numerical worker thread; runtime includes rebuilding peers and all smoothing variants, rather than an incremental production update.</p>")
    if Path('data/extensions_cadence.csv').exists():
        cadence=pd.read_csv('data/extensions_cadence.csv')
        parts.append(f"<p>Three full recomputations including all score candidates and reporting for the selected policies took median {cadence.seconds.median():.3f}s, maximum {cadence.seconds.max():.3f}s. This includes synthetic input generation and excludes data fetching, scheduling, network and page rendering.</p>")
    parts.append('<p>Run <code>python run_extensions.py</code> to reproduce the score/metric sweep, the event report is regenerated from the saved results. Use <code>python render_event_report.py</code> to update only the HTML without rerunning simulations. Raw counts are compressed in <code>data/extensions_runs.csv.gz</code>; catalog, selected settings, held-out summary, scenario errors and paired bootstrap lifts are adjacent CSV files. Bootstrap intervals use eight seeds and do not account for every possible synthetic-model misspecification. Tuning across many candidates can overfit; the fresh holdout limits that risk but must not be reused for subsequent selection.</p><p><a href="#anomaly-examples">Back to anomaly illustrations</a></p></section>')
    return ''.join(parts)
