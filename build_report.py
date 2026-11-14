"""Run reproducible simulations and write a self-contained HTML report."""
from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
import plotly.express as px
from analysis import generate
from detection import pair, sweep, evaluate, score_series, ONSET, METHODS

OUT=Path('reports'); OUT.mkdir(exist_ok=True)
Path('data').mkdir(exist_ok=True)
parts=[]; first=True

def chart(fig):
    global first
    fig.update_layout(template='plotly_white',height=420,margin=dict(t=45,b=50,l=55,r=35))
    html=fig.to_html(full_html=False,include_plotlyjs=True if first else False)
    first=False
    return html

def table(df): return df.to_html(index=False,float_format=lambda x:f'{x:.3f}',classes='results',border=0,escape=True)

def wilson(k,n):
    z=1.96; p=k/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den; half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return f'{max(0,mid-half):.0%}–{min(1,mid+half):.0%}'

parts.append('''<h1>Can a weekly review spot a KPI departure?</h1>
<p class="lead">A reproducible experiment with matrix-profile shape matching and peer-based change detectors.</p>
<p>All data is synthetic. This report tests whether recent evidence can support weekly alerts; it does not establish production accuracy.</p>
<h2>How to read this report</h2><p>First see the original US example series. Then examine three controlled departures with a known start: a level shift, a growth change and a shape change. Settings are selected using tuning simulations, then checked on independent held-out simulations. A matched unchanged control estimates false alerts.</p>
<p><b>Timing:</b> every score is placed on its window-ending date. At a weekly report, only data available by that date can be used. A seven-day window means the last seven observed days; it does not force a seven-day delay after a change.</p>''')
raw,events=generate()
cols=['TV+ · US','Fitness+ · US','App Store · US']
parts.append('<h2>The original example series</h2><p>These retain the original synthetic premieres and promotions. Their smooth Gaussian event responses begin before their peaks, so the peak day is not a clean change onset. We do not assign a detection-delay claim to those peaks.</p>')
fig=px.line(raw[cols],labels={'index':'Date','value':'Synthetic KPI index'})
for event in events.itertuples(): fig.add_vline(x=raw.index[event.day].timestamp()*1000,line_dash='dot')
parts.append(chart(fig))
parts.append('<h3>Run the short-window detectors on the original pairs</h3><p>For each original pair, fit on days 0–83 and calibrate a 99th-percentile threshold on days 84–139, before the premiere peak at day 162. Show seven-day shape distance and seven-day peer residual mean, divided by their own threshold. Values above one exceed that threshold. These are exploratory alerts, not verified errors: the original signals contain several genuine, smooth events and no clean binary departure onset.</p>')
for target in ['Fitness+ · US','App Store · US']:
    original=raw[['TV+ · US',target]]
    records=[]
    for method in ['Shape AB join','Peer residual mean']:
        score=score_series(original,7,0,method)
        cutoff=max(float(np.nanquantile(score[84:140],.99)),1e-9)
        records.extend({'Date':date,'Detector':method,'Score / threshold':float(value/cutoff)} for date,value in zip(raw.index[140:],score[140:]))
    records=pd.DataFrame(records)
    records.to_csv('data/original_'+('fitness' if target.startswith('Fitness') else 'appstore')+'_scores.csv',index=False)
    fig=px.line(records,x='Date',y='Score / threshold',facet_row='Detector',title='Original TV+ · US vs '+target)
    fig.update_yaxes(matches=None)
    fig.add_hline(y=1,line_dash='dash')
    for event in events.itertuples(): fig.add_vline(x=raw.index[event.day].timestamp()*1000,line_dash='dot')
    parts.append(chart(fig))
parts.append('''<h2>A fair weekly test</h2><p>The controlled pair shares trend, annual/weekly seasonality and a common pulse, with independent noise. Fitness+ departs at day 210. Noise standard deviation is 0.8 KPI units; departure amplitude is 3 units (growth: 3 per week; shape: a 9-day sinusoid with amplitude 3). Product labels are illustrative, not a fitted replay of the original pair.</p>
<ul><li>First 84 days: fit a frozen peer model. Next 84 days: calibrate thresholds.</li><li>Weekly reporting begins on day 174. First post-change report: day 216, six days after onset.</li><li>Eight tuning seeds: 42–49. Eight separate held-out seeds: 10042–10049.</li><li>Windows: 7, 14, 28, 42 days; shape lag: 0 or 3 backward days; threshold quantiles: 95% or 99%; daily persistence: 1 or 3 days.</li><li>Pick the highest tuning detection rate within seven days under a 10% unchanged-control weekly-alert budget. Break ties with 14-day detection and then false alerts.</li></ul>
<p>The peer residual model predicts Fitness+ from TV+ and weekday effects using the training period. Its residual mean preserves level/growth departures. CUSUM accumulates residual changes since day 84 with a fixed 0.5-sigma allowance; it is not a seven-day-only method. Shape matching normalizes every window separately, potentially hiding a lasting level difference. No detector receives the change date.</p>''')
summary=[]
for kind in ['Level shift','Growth change','Shape change']:
    tuning=sweep(kind=kind); validation=sweep(kind=kind,heldout=True)
    slug=kind.lower().replace(' ','_')
    tuning.to_csv(f'data/weekly_{slug}_tuning.csv',index=False)
    validation.to_csv(f'data/weekly_{slug}_validation.csv',index=False)
    eligible=tuning[tuning['No-change weekly alert rate']<=.1].sort_values(['Detected within 7d','Detected within 14d','No-change weekly alert rate'],ascending=[False,False,True])
    parts.append(f'<h2>{escape(kind)}</h2>')
    if eligible.empty:
        parts.append('<p>No tested setting met the tuning false-alert budget.</p>'); continue
    best=eligible.iloc[0]; idx=eligible.index[0]; held=validation.loc[idx]
    summary.append({'Scenario':kind,'Selected detector':best['Method'],'Window':int(best['Window']),'Within 7d (held-out)':held['Detected within 7d'],'Within 14d (held-out)':held['Detected within 14d'],'Miss rate (held-out)':held['Miss rate'],'Null weekly alert rate':held['No-change weekly alert rate']})
    parts.append(f"<p><b>Selected setting:</b> {best['Method']}; window {int(best['Window']) or 'cumulative'}, backward lag {int(best['Lag'])}, quantile {best['Quantile']}, {int(best['Persistence'])} consecutive daily exceedances.</p>")
    perseed=[]
    for seed in range(10042,10050):
        result=evaluate(seed,int(best['Window']) or 7,int(best['Lag']),best['Method'],best['Quantile'],int(best['Persistence']),.8,3.,kind)
        delay,false,total,*_=result
        perseed.append({'Held-out seed':seed,'First post-onset flag delay (days)':delay,'No-change flagged reports':false,'No-change reports':total})
    perseed=pd.DataFrame(perseed)
    k=int((perseed.iloc[:,1]<=7).sum())
    parts.append(f"<p class='callout'><b>{k}/8</b> held-out runs flagged within seven days ({held['Detected within 7d']:.0%}); within fourteen days: <b>{held['Detected within 14d']:.0%}</b>. Missed through day 335: {held['Miss rate']:.0%}. Matched no-change weekly alert rate: <b>{held['No-change weekly alert rate']:.1%}</b>.</p>")
    parts.append(f'<p>The approximate 95% Wilson interval for the seven-day rate is {wilson(k,8)}. Eight runs are a small sample. Weekly reports within a run are dependent, so their alert counts are not independent trials.</p>')
    sample=pair(10042,.8,3.,kind)
    example=evaluate(10042,int(best['Window']) or 7,int(best['Lag']),best['Method'],best['Quantile'],int(best['Persistence']),.8,3.,kind)
    delay,false,total,score,cutoff,rd,flags=example
    fig=px.line(sample,labels={'index':'Date','value':'Synthetic KPI index'},title='Held-out example: seed 10042')
    fig.add_vline(x=sample.index[ONSET].timestamp()*1000,line_dash='dot')
    parts.append(chart(fig))
    fig=px.line(x=sample.index,y=score,labels={'x':'Evidence available through','y':'Departure score'},title='Daily evidence and weekly report flags')
    fig.add_hline(y=cutoff,line_dash='dash'); fig.add_vline(x=sample.index[ONSET].timestamp()*1000,line_dash='dot')
    fig.add_scatter(x=sample.index[rd[flags]],y=score[rd[flags]],mode='markers',name='Weekly report flag')
    parts.append(chart(fig))
    parts.append('<p>Dotted vertical line: known departure. Dashed horizontal line: threshold fixed from calibration. Markers: weekly reports including a confirmed alert in the preceding seven days. An alert can persist over many reports.</p>')
    # Show the tuning-selected setting within each detector family, then its held-out result.
    family=[]
    for method in METHODS:
        candidates=eligible[eligible.Method==method]
        if candidates.empty:
            family.append({'Detector':method,'Setting':'No setting meets tuning budget'}); continue
        r=candidates.iloc[0]; v=validation.loc[candidates.index[0]]
        family.append({'Detector':method,'Setting':f"m={int(r.Window)}, lag={int(r.Lag)}, q={r.Quantile}, persistence={int(r.Persistence)}",'Held-out <=7d':v['Detected within 7d'],'Held-out <=14d':v['Detected within 14d'],'Held-out miss rate':v['Miss rate'],'Held-out null alerts':v['No-change weekly alert rate']})
    parts.append('<h3>Compare detector families</h3>'+table(pd.DataFrame(family)))
    parts.append('<details><summary>Held-out run details</summary>'+table(perseed)+'</details>')
    parts.append('<details><summary>All tuning settings</summary>'+table(tuning)+'</details>')
    parts.append('<details><summary>All held-out settings — do not use these to reselect the winner</summary>'+table(validation)+'</details>')
parts.insert(1,'<h2>Measured results at a glance</h2>'+table(pd.DataFrame(summary))+'<p>Rates are fractions (1.0 = 100%). These results use a fixed reporting calendar and small synthetic samples. They are not production guarantees.</p>')
parts.append('''<h2>What this means for a weekly VP update</h2><p>In these simulations, seven-day peer residual summaries can identify level and growth departures sooner and more reliably than independently normalized shape matching. Shape-only changes remain harder. The answer depends on effect size, noise, change type and false-alert tolerance; there is no universal seven-day guarantee.</p>
<p>A longer historical baseline and seven-day recent evidence serve different purposes. Use history to learn who normally behaves together, then use the latest week to test whether that relationship still holds. Learning all peer groups afresh from seven daily samples is unstable.</p>
<h2>Approaches for a larger universe of KPIs</h2>
<table><tr><th>Approach</th><th>How it helps</th><th>Weekly constraint</th></tr>
<tr><td>Stable peer graph + residual monitoring</td><td>Learn peer groups on historical adjusted signals; predict each KPI from robust peers and flag departures.</td><td>Recommended first baseline. Freeze peer groups during evaluation; seven-day recent evidence need not mean seven-day training.</td></tr>
<tr><td>Dynamic factors / robust peer forecasts</td><td>Model common market and product drivers, monitor unusual forecast errors.</td><td>Requires sufficient historical training and monitoring of model fit.</td></tr>
<tr><td>CUSUM / sequential residual tests</td><td>Accumulate sustained weak deviations.</td><td>Evidence can span more than seven days; calibrate alert frequency and delay.</td></tr>
<tr><td>ADWIN / Bayesian online change points</td><td>Adaptively detect distribution shifts or estimate change-point probabilities.</td><td>Not clustering engines; choose residual statistics and assumptions carefully. Not benchmarked here.</td></tr>
<tr><td>Rolling similarity and clustering</td><td>Track peer graph changes over time.</td><td>Seven samples are too few for reliable unrestricted large correlation matrices; use historical priors and regularization. Not benchmarked here.</td></tr>
<tr><td>FLOSS / FLUSS</td><td>Matrix-profile-based regime segmentation within a series.</td><td>Not directly a cross-series peer departure detector; window sensitivity remains. Not benchmarked here.</td></tr></table>
<h2>Limits and next checks</h2><p>Try more seeds, different reporting weekdays, correlated noise, smaller effects, seasonal changes and missing observations before trusting an alert policy. Quantile thresholds are estimated, not guaranteed false-alarm rates. First post-onset flags may be chance alerts. Calibrating many settings on eight seeds can overfit; independent held-out seeds reduce that risk but do not eliminate it. For many KPIs, calibrate the combined alert burden across the universe, not only a single pair.</p>
<p>Peer residual mean can miss zero-mean shape changes; CUSUM can accumulate baseline bias. Shape joins discard level and scale. Fixed baselines can age, while automatic adaptation can absorb the departure you want to find. Report uncertainty and magnitude alongside an alert.</p>
<h2>Reproduce and inspect</h2><p>Run <code>python build_report.py</code> after installing <code>requirements.txt</code>. This HTML embeds charts and JavaScript and opens offline. CSV sweeps are in <code>data/weekly_*.csv</code>. Algorithms are in <code>detection.py</code>; causal-prefix and calibration tests are in <code>test_detection.py</code>. The optional Streamlit tab exposes further controls.</p>
<h2>Method references</h2><ul>
<li><a href="https://github.com/TDAmeritrade/stumpy/blob/main/stumpy/floss.py">STUMPY FLOSS / FLUSS implementation</a></li>
<li><a href="https://riverml.xyz/latest/api/drift/ADWIN/">River ADWIN</a></li>
<li><a href="https://scikit-learn.org/stable/modules/clustering.html">Scikit-learn clustering methods</a></li>
<li><a href="https://centre-borelli.github.io/ruptures-docs/">Ruptures offline change-point detection</a></li>
<li><a href="https://arxiv.org/abs/0710.3742">Adams & MacKay: Bayesian Online Changepoint Detection</a></li></ul>''')
style='''body{font:17px/1.65 system-ui,sans-serif;color:#1b2940;background:#f5f7fb;max-width:1120px;margin:40px auto;padding:0 24px}h1{font-size:44px;line-height:1.15}h2{margin-top:48px}h3{margin-top:28px}.lead{font-size:22px;color:#456}.callout{background:#e0f3ed;border-left:5px solid #14846b;padding:20px}table{border-collapse:collapse;display:block;overflow-x:auto;max-width:100%;background:white;font-size:14px;margin:20px 0}th,td{padding:10px;border-bottom:1px solid #dce2eb;text-align:left;vertical-align:top}th{background:#e9eef5}details{background:white;padding:14px;margin:12px 0}summary{cursor:pointer;font-weight:600}a{color:#087566}.plotly-graph-div{background:white}code{background:#e6eaf0;padding:2px 5px}'''
html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Weekly KPI departure detection</title><style>'+style+'</style></head><body>'+''.join(parts)+'</body></html>'
(OUT/'weekly_detection.html').write_text(html)
print('Wrote',OUT/'weekly_detection.html',len(html),'characters')
