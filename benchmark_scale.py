"""Measured scale benchmarks. Each configuration runs in a fresh single-thread worker."""
import os
import sys
import json
import subprocess
from pathlib import Path


def worker(n,m,repeats):
    import time, resource, platform
    import numpy as np
    import pandas as pd
    from analysis import generate, profile, relationships, relationships_fast
    timings={}
    def timed(name,fn):
        values=[]
        for _ in range(repeats):
            start=time.perf_counter(); fn(); values.append(time.perf_counter()-start)
        timings[name+'_median_s']=float(np.median(values))
        timings[name+'_max_s']=max(values)
    geos=n//7
    data,_=generate(days=180,countries=geos,lobs=7)
    timed('generate',lambda:generate(days=180,countries=geos,lobs=7))
    timed('self_profiles_all_series',lambda:[profile(data[c].values,m) for c in data])
    timed('latest_peer_matrix',lambda:relationships_fast(data,m,180-m,3,.75,causal=True))
    report_ends=list(range(83,180,7))
    timed('weekly_peer_history',lambda:[relationships_fast(data,m,end-m+1,3,.75,causal=True) for end in report_ends])
    # Pairwise full AB profiles are distinct from the cheap local peer matrix.
    # Measure the entire all-pairs workload, not extrapolation from one pair.
    def all_pair_profiles():
        cols=list(data)
        for i in range(n):
            for j in range(i): profile(data[cols[i]].values,m,data[cols[j]].values,3)
    # Run full-history all-pairs exactly once at the requested scale for each window.
    if n==280:
        start=time.perf_counter(); all_pair_profiles()
        timings['all_pair_history_once_s']=time.perf_counter()-start
        start=time.perf_counter(); relationships(data,m,180-m,3,.75)
        timings['legacy_latest_once_s']=time.perf_counter()-start
        start=time.perf_counter(); relationships_fast(data,m,180-m,3,.75)
        timings['fast_retrospective_latest_once_s']=time.perf_counter()-start
    return dict(series=n,days=180,window=m,pairs=n*(n-1)//2,repeats=repeats,weekly_reports=len(report_ends),input_MiB=data.values.nbytes/2**20,peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),cpu_available=os.cpu_count(),**timings)


def main():
    import pandas as pd
    import plotly.express as px
    from html import escape
    rows=[]
    for n in [70,140,280]:
        for m in [7,14,28,42]:
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            result=subprocess.run([sys.executable,__file__,'worker',str(n),str(m),'3'],env=env,text=True,capture_output=True,check=True)
            row=json.loads(result.stdout); rows.append(row)
            print(f"Measured {n} series / {m} days: peer snapshot {row['latest_peer_matrix_median_s']:.4f}s",flush=True)
    df=pd.DataFrame(rows); Path('reports').mkdir(exist_ok=True); Path('data').mkdir(exist_ok=True)
    df.to_csv('data/performance_benchmark.csv',index=False)
    Path('data/performance_benchmark.json').write_text(json.dumps(rows,indent=2))
    target=df[(df.series==280)&(df.window==7)].iloc[0]
    html=['<h1>280 business signals: measured performance</h1><p>Seven lines of business × forty geographic regions × 180 daily observations. All values below are measurements on the execution machine, not latency promises for your hardware.</p>']
    html.append(f"<div class='callout'>For a seven-day window: all 280 self-profiles <b>{target.self_profiles_all_series_median_s:.3f}s</b>; latest causal 280×280 peer distance matrix plus groups <b>{target.latest_peer_matrix_median_s:.3f}s</b>; {int(target.weekly_reports)} weekly peer snapshots <b>{target.weekly_peer_history_median_s:.3f}s</b>. Full-history AB profiles for all 39,060 pairs, one measured run: <b>{target.all_pair_history_once_s:.2f}s</b>.</div>")
    html.append('<h2>These workloads answer different questions</h2><ul><li>Self-profiles: recurring/unusual shapes within every series across its full history.</li><li>Latest peer matrix: compare the latest window to each peer’s latest or up-to-three-days-earlier window. Returns distances and connected-component groups. This is a local subsequence similarity analysis, not all full-history AB matrix profiles.</li><li>Weekly peer history: rebuild that causal snapshot at each of 14 weekly dates from day 83 onward. Window normalization is recomputed for each snapshot; no incremental cache is assumed.</li><li>All-pairs history: run the existing exact AB profile for every unordered pair, with a ±3-day retrospective lag. This includes future matches within the lag and is not a live detector. The implementation computes the full distance matrix before masking lag.</li></ul>')
    html.append(f"<h2>Optimization and correctness</h2><p>The original loop takes {target.legacy_latest_once_s:.3f}s for one retrospective snapshot; the equivalent vectorized version takes {target.fast_retrospective_latest_once_s:.3f}s in a single comparison ({target.legacy_latest_once_s/target.fast_retrospective_latest_once_s:.1f}×). This single-run comparison is indicative. Automated checks compare scores and groups against the original loop, including constant signals and boundary windows. A separate causal mode allows only already-observed matching windows.</p>")
    first=True
    for metric,title in [('self_profiles_all_series_median_s','All-series self-profiles'),('latest_peer_matrix_median_s','Latest causal peer distance matrix + groups'),('weekly_peer_history_median_s','14 weekly causal peer snapshots')]:
        fig=px.line(df,x='series',y=metric,color='window',markers=True,title=title,labels={'series':'Number of series',metric:'Median seconds','window':'Window (days)'})
        fig.update_layout(template='plotly_white')
        html.append(fig.to_html(full_html=False,include_plotlyjs=True if first else False));first=False
    html.append('<h2>Measured timings</h2>'+df[['series','days','window','pairs','self_profiles_all_series_median_s','latest_peer_matrix_median_s','weekly_peer_history_median_s','all_pair_history_once_s','peak_process_RSS_MiB']].to_html(index=False,float_format=lambda x:f'{x:.4f}'))
    html.append('<h2>Measurement protocol and limits</h2><p>Each series-count/window configuration runs in a new worker process, with BLAS/OpenMP threads set to one. Three sequential runs per standard workload; report median and maximum in the downloadable CSV. No explicit warm-up; imports and startup are outside workload timing. All-pairs and legacy comparison run once per window at 280 series. Timings include Python grouping and window construction but exclude imports, disk exports, HTML generation, network, Streamlit/browser rendering and deployment latency. Single-machine results are descriptive; three repetitions do not establish a p95 service-level bound.</p>')
    html.append(f"<p>Environment: Python {escape(str(target.python))}, NumPy {escape(str(target.numpy))}, {escape(str(target.platform))}; {int(target.cpu_available)} logical CPUs visible, one numerical worker thread configured. Input float64 array: {target.input_MiB:.3f} MiB; peak worker process RSS: {target.peak_process_RSS_MiB:.1f} MiB for the 280/7 configuration. RSS includes interpreter, libraries, allocations and previous stages, not isolated algorithm memory. Peak RSS is sampled via Linux resource high-water mark. Similarity matrices contain 78,400 cells; unordered pairs total 39,060.</p>")
    html.append('<h2>What this establishes</h2><p>This scale is practical for the measured numerical workloads. The vectorized snapshot avoids tens of thousands of Python pair loops and avoids materializing a four-dimensional all-pairs subsequence tensor. A readable VP report should rank changed signals and show selected peer groups rather than render all 280 lines at once.</p><p>This is a performance benchmark, not a validation of departure detection accuracy at 280 signals. Six months leaves less room for training and calibration than the earlier 336-day controlled experiment; its thresholds and detection rates cannot be copied unchanged. Evaluate historical peer stability, pooled or shrinkage estimates, multiple-alert burden and held-out departures before making weekly accuracy claims. The larger synthetic dataset extends the existing generator; it does not establish realistic cross-LOB cluster structure or real operating volumes.</p>')
    html.append('<h2>Reproduce</h2><p>Run <code>python benchmark_scale.py</code>. Full timings and environment metadata are in <code>data/performance_benchmark.csv</code> and the adjacent JSON. The optional app now supports seven LOBs and forty geographies and uses the vectorized peer analysis.</p>')
    style='body{font:17px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 20px;background:#f5f7fb;color:#203047}h1{font-size:40px}h2{margin-top:35px}.callout{padding:24px;background:#dff2eb;border-left:5px solid #21876e}table{display:block;overflow:auto;font-size:13px;border-collapse:collapse;background:white}td,th{padding:9px;border:1px solid #dae1eb}code{background:#e5eaf0}'
    Path('reports/performance_280.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>280-series performance benchmark</title><style>'+style+'</style></head><body>'+''.join(html)+'</body></html>')
    print('Wrote reports/performance_280.html')

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker': print(json.dumps(worker(*map(int,sys.argv[2:]))))
    else: main()
