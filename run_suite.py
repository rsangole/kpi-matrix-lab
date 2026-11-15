"""Execute tuning/held-out 280-series event benchmarks and produce offline HTML."""
import os
# Set numerical threading before importing NumPy; this is a single-process benchmark.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import time
import json
import resource
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from suite import generate_suite,scores,flags,measure,report_truth,SCENARIOS

KEY=['method','window','quantile','persistence']
COUNTS=['tp','fp','fn','tn','event_count','events_7','events_14','control_flags','control_opportunities','control_family_reports','reports','delay_sum','detected_events']

def aggregate(df,keys):
    out=df.groupby(keys,dropna=False)[COUNTS].sum().reset_index()
    divide=lambda a,b:np.divide(a,b,out=np.zeros(len(out),dtype=float),where=np.asarray(b)>0)
    out['precision']=divide(out.tp,out.tp+out.fp)
    out['recall']=divide(out.tp,out.tp+out.fn)
    out['f1']=divide(2*out.tp,2*out.tp+out.fp+out.fn)
    out['false_positive_rate']=divide(out.fp,out.fp+out.tn)
    out['event_recall_7d']=divide(out.events_7,out.event_count)
    out['event_recall_14d']=divide(out.events_14,out.event_count)
    out['miss_rate_14d']=1-out.event_recall_14d
    out['control_weekly_flag_rate']=divide(out.control_flags,out.control_opportunities)
    out['control_family_report_rate']=divide(out.control_family_reports,out.reports)
    out['mean_delay_detected_days']=divide(out.delay_sum,out.detected_events)
    out.loc[out.detected_events==0,'mean_delay_detected_days']=np.nan
    return out

def select_settings(pooled,budget=.05):
    selected=[]
    for method in pooled.method.unique():
        family=pooled[(pooled.split=='tuning')&(pooled.method==method)]
        eligible=family[family.control_weekly_flag_rate<=budget]
        passed=not eligible.empty
        if passed:
            best=eligible.sort_values(['f1','event_recall_7d','control_weekly_flag_rate'],ascending=[False,False,True]).iloc[0]
        else:
            # Keep every family's errors visible; failed budget is explicitly labeled.
            best=family.sort_values(['control_weekly_flag_rate','f1'],ascending=[True,False]).iloc[0]
        row={k:best[k] for k in KEY}
        row['tuning_budget_met']=passed
        row['tuning_control_flag_rate']=float(best.control_weekly_flag_rate)
        selected.append(row)
    return pd.DataFrame(selected)


def main():
    start=time.perf_counter(); rows=[]; timing=[]
    for split,seeds in [('tuning',range(42,46)),('heldout',range(10042,10050))]:
        for noise in [.8,1.6]:
            for seed in seeds:
                baseline,_,_=generate_suite(seed,noise,changed=False)
                t=time.perf_counter(); controls=scores(baseline)
                timing.append({'split':split,'noise':noise,'seed':seed,'scenario':'No event','stage':'14 detector configurations','seconds':time.perf_counter()-t})
                # Cache control flags once per seed/setting; independent of scenario.
                controlflags={(method,m,q,p):flags(v,q,p) for (method,m),v in controls.items() for q in [.95,.99,.995] for p in [1,3]}
                for scenario in SCENARIOS:
                    data,meta,events=generate_suite(seed,noise,scenario)
                    t=time.perf_counter(); values=scores(data)
                    timing.append({'split':split,'noise':noise,'seed':seed,'scenario':scenario,'stage':'14 detector configurations','seconds':time.perf_counter()-t})
                    context=report_truth(events)
                    for (method,m),score in values.items():
                        for q in [.95,.99,.995]:
                            for persistence in [1,3]:
                                pred=flags(score,q,persistence)
                                metrics=measure(pred,controlflags[(method,m,q,persistence)],events,context)
                                rows.append(dict(split=split,seed=seed,noise=noise,scenario=scenario,method=method,window=m,quantile=q,persistence=persistence,**metrics))
                    if seed==42 and noise==.8 and scenario=='Mixed events':
                        Path('data').mkdir(exist_ok=True)
                        data.to_csv('data/suite_mixed_signals.csv',index_label='date')
                        meta.to_csv('data/suite_series_metadata.csv',index_label='series')
                        events.to_csv('data/suite_mixed_events.csv',index=False)
                print(f'{split} seed {seed}, noise {noise}: completed all nine event scenarios',flush=True)
    df=pd.DataFrame(rows); Path('data').mkdir(exist_ok=True); Path('reports').mkdir(exist_ok=True)
    df.to_csv('data/event_suite_runs.csv',index=False)
    pooled=aggregate(df, ['split']+KEY)
    pooled.to_csv('data/event_suite_pooled.csv',index=False)
    # Select within each family on tuning data only, pooling both noise levels/scenarios.
    chosen=select_settings(pooled)
    chosen.to_csv('data/event_suite_selected_settings.csv',index=False)
    from benchmark_detectors import cadence
    cadence(chosen)
    held=df[df.split=='heldout'].merge(chosen,on=KEY)
    summary=aggregate(held,['method'])
    details=aggregate(held,['scenario','noise','method'])
    summary.to_csv('data/event_suite_summary.csv',index=False)
    details.to_csv('data/event_suite_by_scenario.csv',index=False)
    pd.DataFrame(timing).to_csv('data/event_suite_runtime.csv',index=False)
    elapsed=time.perf_counter()-start
    runmeta={'elapsed_seconds_before_report':elapsed,'peak_process_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'replicates':216,'settings_per_replicate':84,'tuning_seeds':[42,43,44,45],'heldout_seeds':list(range(10042,10050)),'noise':[.8,1.6],'amplitude':5,'days':180,'series':280,'train_days':56,'calibration_days':56,'numpy':np.__version__}
    Path('data/event_suite_metadata.json').write_text(json.dumps(runmeta,indent=2))
    report(df,chosen,summary,details,pd.DataFrame(timing),runmeta)
    print(summary.to_string(index=False)); print('Wrote reports/event_suite_280.html',flush=True)

def report(runs,chosen,summary,details,timing,meta):
    parts=[]; first=True
    def chart(fig):
        nonlocal first
        fig.update_layout(template='plotly_white',height=fig.layout.height or 450)
        html=fig.to_html(full_html=False,include_plotlyjs=True if first else False); first=False
        return html
    def table(x): return x.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    parts.append('''<h1>Weekly event detection across 280 business signals</h1><p class="lead">Seven LOBs · forty geographies · six months of daily data. Runtime and error metrics, measured separately.</p><p>All signals, offer targeting and event labels are synthetic. This report uses controlled event truth to evaluate detection, not visual judgment. No production service level is implied.</p>''')
    from anomaly_examples import visual_examples
    parts.append(visual_examples(chart))
    parts.append('<h2>Held-out results</h2><p>One setting per detector family is chosen only on tuning seeds, maximizing weekly-label F1 under a 5% per-signal unchanged-control weekly flag budget. If no setting meets that budget, show the least-false-alert setting explicitly as a budget failure rather than hiding the family. The following results use separate held-out seeds. Rates are fractions: 0.5 = 50%. The budget is not a guarantee on held-out data or across all 280 signals.</p>')
    cols=['method','precision','recall','f1','false_positive_rate','event_recall_7d','event_recall_14d','miss_rate_14d','control_weekly_flag_rate','control_family_report_rate','mean_delay_detected_days']
    parts.append(table(summary[cols]))
    for row in summary.itertuples():
        policy=chosen[chosen.method==row.method].iloc[0]
        label='Met tuning false-alert budget' if policy.tuning_budget_met else 'FAILED tuning false-alert budget'
        parts.append(f"<article><h3>{row.method}</h3><p><b>{row.event_recall_7d:.1%}</b> of held-out events flagged within 7 days; <b>{row.event_recall_14d:.1%}</b> within 14 days. Weekly precision {row.precision:.1%}, recall {row.recall:.1%}.</p><p>Unchanged-control signal/report flag rate: <b>{row.control_weekly_flag_rate:.1%}</b>. Weekly reports with at least one false flag across 280 signals: <b>{row.control_family_report_rate:.1%}</b>. {label}.</p></article>")
    parts.append('<p><b>Reading the trade-off:</b> high recall from an unstable detector is not a usable weekly policy if false flags dominate. Historical/peer shape methods have low recall here; residual methods retain more level information. The pooled optimum can sacrifice single-day spikes: three-day persistence rejects isolated impulses. These are globally selected settings across all event types, not a separately optimized detector for each scenario. Use the full sweep counts to study specialized policies, and validate any combined policy separately.</p>')

    parts.append('<h3>Selected settings</h3>'+table(chosen))
    missing=chosen.loc[~chosen.tuning_budget_met,'method'].tolist()
    if missing: parts.append('<p>No tuning setting met the 5% per-signal control budget for: '+', '.join(missing)+'. The least-false-alert tuning setting is shown for each of these families; this is a diagnostic comparison, not an acceptable alert policy. Full sweep counts remain in the CSV files.</p>')
    parts.append('<details><summary>Held-out confusion counts and event totals</summary>'+table(summary[['method','tp','fp','fn','tn','event_count','events_7','events_14']])+'</details>')
    parts.append('<p>Window=0 denotes cumulative CUSUM, not zero observations. Persistence counts consecutive daily exceedances. With 280 signals, even modest per-signal error rates can make almost every weekly review include a false alert; compare the family-report rate above. False alarms cannot be summarized by the per-signal rate alone.</p>')
    fig=px.bar(details,x='scenario',y='event_recall_7d',color='method',facet_row='noise',barmode='group',labels={'event_recall_7d':'Event recall within 7 days'},title='Which events can be reported within a week?')
    fig.update_layout(height=780); parts.append(chart(fig))
    fig=px.scatter(details,x='false_positive_rate',y='recall',color='method',symbol='noise',hover_data=['scenario','precision','f1'],title='Weekly label recall versus false positive rate')
    parts.append(chart(fig))
    data,metadata,events=generate_suite()
    fig=px.line(data.iloc[:,[0,11,22,33,44,55,66,77]],labels={'index':'Date','value':'Synthetic KPI index'},title='Selected signals from the mixed-event dataset')
    parts.append(chart(fig))
    parts.append('<h2>What was simulated</h2><table><tr><th>Scenario</th><th>Injection</th><th>Target</th></tr><tr><td>Level shift</td><td>Persistent +5 units</td><td>14 signals</td></tr><tr><td>Random spikes</td><td>Three random signed one-day ±5-unit impulses per target</td><td>14 signals, 42 event instances</td></tr><tr><td>Temporary seasonality</td><td>Additional 3.5-day oscillation, amplitude 5, for 21 days</td><td>14 signals</td></tr><tr><td>Gradual trend</td><td>New slope +5 units per 21 days</td><td>14 signals</td></tr><tr><td>Rise and recovery</td><td>28-day smooth rise and return, peak 5</td><td>14 signals</td></tr><tr><td>LOB offer</td><td>14-day +5-unit promotion</td><td>Only TV+ and App Store</td></tr><tr><td>Developing-market offer</td><td>14-day +5-unit promotion</td><td>13 explicitly listed test geographies, all LOBs</td></tr><tr><td>English-market offer</td><td>14-day +5-unit promotion</td><td>6 explicitly listed test geographies, all LOBs</td></tr><tr><td>Mixed events</td><td>All eight types overlap; onset staggered by up to 14 days</td><td>Targets overlap</td></tr></table><p>Common background includes trend, weekly/annual seasonality, correlated global and LOB factors, and independent Gaussian noise (standard deviation 0.8 or 1.6). Seasonal changes are temporary and spikes have random signs and dates. The economic/language groupings are explicit synthetic test cohorts, not authoritative country classifications; metadata lists every membership. Mixed-event performance cannot uniquely attribute an alert to one overlapping event.</p>')
    parts.append('<h2>Algorithms evaluated</h2><ul><li><b>Historical matrix profile:</b> exact z-normalized trailing-window nearest-neighbor distance against all frozen training-period windows of the same signal. Run across all 280 series. This is a causal baseline AB profile; no later normal windows are added to the reference set.</li><li><b>Peer shape distance:</b> median same-date z-normalized window distance to five peers learned on training data, not an all-pairs full-history matrix profile.</li><li><b>Peer residual mean:</b> predict each signal from a robust median of its five standardized peers plus weekday effects; inspect absolute rolling standardized residual mean.</li><li><b>Peer residual spike:</b> absolute daily standardized peer residual; intended to retain short impulses.</li><li><b>Peer residual CUSUM:</b> accumulate standardized residual shifts with a fixed 0.5-sigma allowance.</li></ul><p>Peers are chosen by correlation of training residuals after fitting trend and weekday effects. Peer membership, loadings and training residual scale remain frozen. A shared offer can move an entire peer group together and be invisible to a relative detector; the historical matrix profile gives complementary absolute-history evidence. None of these methods explains cause.</p>')
    parts.append('<h2>Evaluation protocol</h2><p>Train on days 0–55, calibrate per-series empirical thresholds on days 56–111, and evaluate weekly reports ending on days 118, 125, …, 174. Isolated events start on day 126, except random spikes distributed across the next 28 days; mixed onsets are 126, 133 or 140. The first scheduled report after a day-126 onset is day 132: six days later. The final five days after the last weekly report are outside report-level scoring.</p><p>Windows are 7/14/28/42 days, quantiles 95%/99%/99.5%, persistence 1/3 days. Five detector families yield fourteen family/window configurations and 84 settings. Nine scenarios × two noise levels × twelve independent seeds = 216 dataset replicates and 18,144 evaluated setting/replicate combinations. Four seeds (42–45) tune settings; eight different seeds (10042–10049) validate. Changing settings after inspecting held-out results would consume that holdout.</p><p><b>Weekly truth:</b> a signal/report is positive if at least one labeled event intersects the report’s preceding seven days. Precision, recall, F1 and false-positive rate use these signal/report labels, pooled by counts. An alert in a late recovery week can count as a false positive if no labeled event still intersects that week. All repeated weekly flags count toward report burden.</p><p><b>Event recall:</b> each series/event instance is detected if that signal flags at a report ending within seven or fourteen days after onset, while the report still intersects the event. Miss rate is one minus fourteen-day event recall. Mean delay is over detected events only, so always read it alongside misses. One report can cover several overlapping events; mixed-event recall may be optimistic about attribution. A post-onset flag is temporal detection, not proof of causal attribution.</p><p><b>Controls:</b> identical background and noise with injections disabled. Per-signal control rate counts flagged signal/reports; family control rate counts weekly reports containing at least one flag among 280 signals. Control counts are repeated across scenarios in the pooled table and therefore must not be treated as independent samples. Signals and weekly reports are correlated. No independent-trial confidence intervals are asserted.</p><p>With only 56 calibration days, a 99.5th percentile is largely determined by the largest few observations. Shape windows may overlap training and calibration boundaries, but every scored window ends before the observation date; no future samples enter. Long windows make the effective calibration sample even smaller. This short history is a genuine statistical constraint.</p>')
    parts.append('<h2>Detection by scenario and noise</h2>'+table(details[['scenario','noise']+cols]))
    parts.append('<h2>Compute time and regular cadence</h2>')
    stats=timing.seconds.describe(percentiles=[.5,.95]).to_frame('seconds').reset_index()
    parts.append('<p>These timings measure building all fourteen detector score arrays across all 280 series and all 180 days, including peer learning and regression. They exclude the 84 threshold/report evaluations, file output and HTML rendering. This is a full six-month recomputation, not an incremental daily update.</p>'+table(stats))
    parts.append(f"<p>The complete quality sweep before HTML rendering took <b>{meta['elapsed_seconds_before_report']:.1f}s</b> on this machine, including controls, scoring, report evaluation and CSV output. Peak suite-process RSS: {meta['peak_process_RSS_MiB']:.1f} MiB. One numerical worker thread was configured. The runtime distribution is measured across different synthetic datasets, not repeated latency trials of identical inputs.</p>")
    parts.append('<p>See the companion <a href="performance_280.html">performance benchmark</a> for three-run timings over 70/140/280 series and four windows: all self-profiles, latest peer matrices, weekly history, and one full-history AB join across every one of 39,060 pairs. Recomputing just the latest peer snapshot is substantially less work than recomputing all pair histories. These measurements support a cadence decision on this hardware; they exclude scheduler queues, cloud cold starts, ingestion and browser rendering.</p>')
    cadence_times=pd.read_csv('data/event_suite_cadence_runtime.csv')
    parts.append(f"<p><b>Operational cadence example:</b> recomputing the entire 180-day dataset for all 280 signals, learning peers, building all fourteen score arrays and applying the selected policies to weekly flags took a median of {cadence_times.seconds.median():.3f}s (maximum {cadence_times.seconds.max():.3f}s) across three runs. This is an intentionally simple full recomputation, not a streaming optimization. It includes synthetic input generation; excludes real data fetching, export, scheduler and user-interface latency. The much larger quality sweep is research validation and need not run on every weekly refresh.</p>")
    stage=pd.read_csv('data/event_suite_family_runtime.csv').groupby('workload').seconds.agg(['median','max']).reset_index()
    parts.append('<h3>Per-stage full-history detector timings</h3><p>Three repeats on the mixed dataset (seed 42, noise 0.8), one thread. Values are seconds. Peer-based methods additionally require the shared peer-learning/residual-fitting stage. Stage timing excludes threshold calibration and reporting. The family timing script is <code>benchmark_detectors.py</code>.</p>'+table(stage))
    perf=pd.read_csv('data/performance_benchmark.csv')
    parts.append(table(perf[perf.series==280][['window','self_profiles_all_series_median_s','latest_peer_matrix_median_s','weekly_peer_history_median_s','all_pair_history_once_s','peak_process_RSS_MiB']]))
    parts.append('<h2>Limits and next decisions</h2><p>This benchmark tests one controlled amplitude with two noise levels, 180 days, complete observations and fixed targeting definitions. It does not prove detection across real-world data, missing values, structural zeros, heterogeneous count distributions or unstable baseline relationships. Detection of broad common-mode changes and single-day spikes requires different evidence. A seven-day reporting cadence cannot itself supply seven days of statistical evidence for a spike that lasted one day.</p><p>A production weekly report should combine recent residual spikes, sustained residual change and historical shape novelty, group related alerts, and explicitly calibrate the total alert burden across the KPI universe. Do not automatically OR these detectors together: the combined policy needs its own held-out calibration and error metrics.</p><h2>Reproduce</h2><p>Run <code>python benchmark_scale.py</code>, <code>python benchmark_detectors.py</code> and then <code>python run_suite.py</code>. Scripts, event ledgers, metadata, full tuning/held-out counts and selected settings are committed with this report. Other methods discussed previously, including ADWIN and Bayesian change points, are not implemented or benchmarked here.</p>')
    style='body{font:17px/1.65 system-ui;max-width:1160px;margin:40px auto;padding:0 20px;color:#203047;background:#f5f7fb}h1{font-size:40px;line-height:1.2}h2{margin-top:40px}.lead{font-size:22px}table{display:block;overflow:auto;max-width:100%;font-size:13px;border-collapse:collapse;background:white;margin:20px 0}td,th{padding:9px;border-bottom:1px solid #d9e2eb;text-align:left;vertical-align:top}th{background:#e5ebf3}code{background:#e2e8f0;padding:3px}a{color:#087366}article{padding:16px 22px;margin:16px 0;background:#e7f0ed;border-radius:12px}article h3{margin-top:0}.anomaly-nav{display:flex;flex-wrap:wrap;gap:8px}.anomaly-nav a{background:#e2ece8;padding:6px 12px;border-radius:8px;text-decoration:none}.example-back{font-size:14px}section[id]{scroll-margin-top:20px}@media(max-width:600px){body{padding:0 12px}h1{font-size:30px}}'
    Path('reports/event_suite_280.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>280-signal event detection suite</title><style>'+style+'</style></head><body>'+''.join(parts)+'</body></html>')

if __name__=='__main__': main()
