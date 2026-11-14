import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from analysis import generate, preprocess, profile, relationships
from detection import pair, sweep, evaluate, ONSET, TRAIN, CALIBRATION

st.set_page_config(page_title='Signal Atlas · KPI Matrix Lab',page_icon='◈',layout='wide')
st.markdown('''<style>.stApp{background:#0b1020}h1{letter-spacing:-1.5px}div[data-testid="stMetric"]{background:#141d32;padding:18px;border-radius:12px} .block-container{padding-top:2rem}</style>''',unsafe_allow_html=True)
st.caption('SIGNAL ATLAS / BUSINESS INTELLIGENCE LAB')
st.title('Find the patterns behind your KPIs')
st.write('Explore recurring shapes, unusual events, and markets that move together. All data is synthetic; product names illustrate a business scenario.')
with st.sidebar:
    st.header('Experiment controls')
    seed=st.number_input('Random seed',0,100000,42)
    days=st.slider('History (daily observations)',180,900,540,30)
    countries=st.slider('Countries per business',5,10,10)
    noise=st.slider('Noise amplitude',0.,4.,.8,.1)
    strength=st.slider('Event strength',0.,30.,12.,1.)
    drift=st.toggle('Inject India TV+ group departure',True)
    st.divider()
    mode=st.selectbox('Preprocessing',['Remove trend + weekly seasonality','Raw','First difference'])
    m=st.slider('Subsequence window (days)',7,90,28)
    lag=st.slider('Allowed matching lag (days)',0,28,3)
    threshold=st.slider('Group distance threshold',.1,2.,.75,.05)
    st.caption('Distances use z-normalized shapes. Lower is closer. Groups use connected components, so members can be linked indirectly.')

@st.cache_data
def dataset(*args): return generate(*args)
raw,events=dataset(days,countries,seed,noise,strength,drift)
data=preprocess(raw,mode)
a,b,c=st.columns(3)
a.metric('Business signals',len(data.columns)); b.metric('Daily observations',days); c.metric('Planted events',len(events))
tab1,tab2,tab3,tab4,tab5=st.tabs(['Signal explorer','Motifs & anomalies','Groups & departures','Method & data','Weekly detection lab'])
with tab1:
    selected=st.multiselect('Signals',list(raw),default=[n for n in raw if n.startswith('TV+')][:5])
    processed=st.toggle('Show preprocessed values',False)
    fig=px.line((data if processed else raw)[selected],labels={'value':'Synthetic KPI index','index':'Date','variable':'Signal'},template='plotly_dark')
    for e in events.itertuples(): fig.add_vline(x=raw.index[e.day].timestamp()*1000,line_dash='dot',line_color='#f4b860')
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Dotted lines mark planted events. Read the event ledger in Method & data.')
with tab2:
    left,right=st.columns(2)
    focus=left.selectbox('Reference signal',list(data))
    other=right.selectbox('Cross-series comparison',['Self-join (recurring patterns)']+list(data),index=2)
    y=None if other.startswith('Self-join') else data[other].values
    p,idx=profile(data[focus].values,m,y,None if y is None else lag)
    st.plotly_chart(px.line(x=data.index[m-1:],y=p,labels={'x':'Window ending date','y':'Normalized matrix profile distance'},template='plotly_dark'),use_container_width=True)
    st.caption('Self-join: low values indicate recurring motifs; high values indicate discords. Cross-series: low values indicate a similar window in the comparison signal within the allowed lag. Each point describes a full window.')
    ending=st.slider('Inspect window ending day',m-1,days-1,int(np.argmax(p))+m-1)
    start=ending-m+1
    match=int(idx[start]); target=focus if y is None else other
    w1=data[focus].values[start:start+m]; w2=data[target].values[match:match+m]
    z=lambda w:(w-w.mean())/(w.std() if w.std()>1e-10 else 1)
    st.plotly_chart(px.line(pd.DataFrame({'Reference':z(w1),'Nearest match':z(w2)}),labels={'index':'Day within window','value':'Standard deviations'},template='plotly_dark'),use_container_width=True)
    st.write(f'Reference ends {data.index[ending].date()}; nearest match: {target}, ending {data.index[match+m-1].date()}. Distance: {p[start]:.3f}.')
    st.caption('This tab is retrospective: centered preprocessing and full-history self-joins can use future data; cross-joins may match ahead by the lag. Ending dates alone do not make these scores live alerts. Use Weekly detection lab for causal evaluation.')
with tab3:
    anchor=st.slider('Current window ending day',m-1,days-1,min(int(days*.8)+m-1,days-1))-m+1
    baseline=st.slider('Baseline window ending day',m-1,days-1,min(int(days*.3)+m-1,days-1))-m+1
    scores,groups=relationships(data,m,anchor,lag,threshold)
    previous,oldgroups=relationships(data,m,baseline,lag,threshold)
    st.caption(f'Current: {data.index[anchor].date()} → {data.index[anchor+m-1].date()} · baseline: {data.index[baseline].date()}')
    st.plotly_chart(px.imshow(scores,zmin=0,zmax=2,color_continuous_scale='Tealrose',aspect='auto',labels={'color':'Distance'},template='plotly_dark'),use_container_width=True)
    st.write('Current shape groups')
    st.dataframe(pd.DataFrame([{'Group':i+1,'Size':len(g),'Signals':', '.join(g)} for i,g in enumerate(groups)]),hide_index=True,use_container_width=True)
    rows=[]
    for name in data:
        old=set(next(g for g in oldgroups if name in g))-{name}
        new=set(next(g for g in groups if name in g))-{name}
        union=old|new
        rows.append({'Signal':name,'Peer turnover':1-len(old&new)/len(union) if union else 0,'Lost peers':len(old-new),'New peers':len(new-old),'Lost peer names':', '.join(sorted(old-new))})
    st.write('Group membership changes')
    st.dataframe(pd.DataFrame(rows).sort_values('Peer turnover',ascending=False),hide_index=True,use_container_width=True)
    st.caption('Peer turnover compares sets of peers, not arbitrary group numbers. A change is an exploratory clue; threshold and noise can also change group membership.')
with tab4:
    st.subheader('What this experiment demonstrates')
    st.write('TV+ premieres affect Western markets together, with a small UK delay and weaker responses in other markets. Services promotions span products. A price change creates shared disruption. India TV+ then acquires an independent pattern when the departure toggle is on.')
    ledger=events.copy(); ledger['Date']=[raw.index[d].date() for d in ledger.day]
    st.dataframe(ledger,hide_index=True)
    st.write('The matrix profile is an exact nearest-neighbor subsequence search. Each window is z-normalized. Self-joins exclude neighbors within ceil(window/4). Cross-joins restrict matching to a calendar lag. Grouping uses symmetric local AB distances at a selected window, an extension of the same subsequence distance rather than a claim that matrix profiles directly produce clusters.')
    st.write('Normalization removes level and scale: signals can match despite different business volumes. Seasonal adjustment uses a centered rolling trend and weekday averages, appropriate for retrospective exploration but not causal or live forecasting. Similarity does not establish causality. Connected-component groups can chain through intermediate signals. This exact NumPy implementation is intended for small experiments; use STUMPY or scalable joins for production histories.')
    st.download_button('Download synthetic signals (CSV)',raw.to_csv(index_label='date'),'synthetic_kpis.csv','text/csv')
    st.download_button('Download event ground truth (CSV)',ledger.to_csv(index=False),'events.csv','text/csv')
    st.download_button('Download current distance matrix',scores.to_csv(),'distances.csv','text/csv')

with tab5:
    st.subheader('Can the weekly review catch a departure?')
    st.write('A controlled two-series experiment: US TV+ and US Fitness+ share a trend and seasonality, then Fitness+ departs on a known date. This is a new counterfactual scenario, separate from the original event data. Ground truth defines evaluation only; detectors do not receive the departure date.')
    st.caption('These controls are independent of the exploratory sidebar. Train: first 84 days. Calibrate thresholds: next 84 days. Evaluate from day 168; departure: day 210. Reports occur every seven days and include alerts from that week. All detector inputs end on or before the evaluation day.')
    c1,c2,c3=st.columns(3)
    kind=c1.selectbox('Departure type',['Level shift','Growth change','Shape change'])
    effect=c2.slider('Departure amplitude (KPI units)',0.,10.,3.,.5)
    labnoise=c3.slider('Experiment noise',.1,4.,.8,.1)
    c1,c2,c3=st.columns(3)
    labseed=c1.number_input('Experiment seed',0,99999,42)
    repeats=c2.slider('Replicates per setting',3,20,8)
    budget=c3.slider('Maximum no-change weekly alert rate',0.,.5,.1,.01)
    st.caption('A 10% budget means at most one in ten weekly reports flags an unchanged control series in the simulation. Rates are empirical, not guaranteed production error bounds.')
    @st.cache_data
    def benchmark(*args): return sweep(*args)
    with st.spinner('Evaluating settings on tuning and separate held-out seeds…'):
        tuning=benchmark(labseed,repeats,labnoise,effect,kind,False)
        validation=benchmark(labseed,repeats,labnoise,effect,kind,True)
    eligible=tuning[tuning['No-change weekly alert rate']<=budget].sort_values(['Detected within 7d','Detected within 14d','No-change weekly alert rate'],ascending=[False,False,True])
    st.write('Parameter sweep — tuning seeds')
    st.dataframe(tuning,hide_index=True,use_container_width=True)
    if eligible.empty:
        st.info('No tested setting meets the false-alert budget. Reduce noise, increase effect size, or relax the budget; a weekly result is not guaranteed.')
    else:
        best=eligible.iloc[0]; bestindex=eligible.index[0]; held=validation.loc[bestindex]
        st.write(f"Best tuning setting within budget: {best['Method']}, window {int(best['Window']) or 'cumulative'}, lag {int(best['Lag'])}, calibration quantile {best['Quantile']}, persistence {int(best['Persistence'])} days.")
        a,b,c=st.columns(3)
        a.metric('Held-out detection within 7 days',f"{held['Detected within 7d']:.0%}")
        b.metric('Held-out detection within 14 days',f"{held['Detected within 14d']:.0%}")
        c.metric('Held-out no-change weekly alerts',f"{held['No-change weekly alert rate']:.1%}")
        st.caption(f"Held-out seeds {labseed+10000}–{labseed+10000+repeats-1} were not used to select this setting. Miss rate: {held['Miss rate']:.0%}. Small replicate counts carry substantial uncertainty. CUSUM uses all residuals since calibration began, not a seven-day-only evidence window.")
    st.write('Inspect one run (illustration, not a reliability estimate)')
    c1,c2,c3=st.columns(3)
    inspectmethod=c1.selectbox('Detector',['Shape AB join','Peer residual mean','Residual CUSUM'])
    inspectm=c2.selectbox('Detection window',[7,14,28,42])
    inspectlag=c3.selectbox('Backward matching lag',[0,3])
    c1,c2=st.columns(2)
    q=c1.selectbox('Calibration quantile',[.95,.99],index=1)
    persistence=c2.selectbox('Consecutive daily exceedances',[1,3])
    delay,false,total,score,cutoff,reportdays,flags=evaluate(labseed,inspectm,inspectlag,inspectmethod,q,persistence,labnoise,effect,kind)
    sample=pair(labseed,labnoise,effect,kind)
    fig=px.line(sample,template='plotly_dark',labels={'index':'Date','value':'Synthetic KPI index'})
    fig.add_vline(x=sample.index[ONSET].timestamp()*1000,line_dash='dot')
    st.plotly_chart(fig,use_container_width=True)
    fig=px.line(x=sample.index,y=score,template='plotly_dark',labels={'x':'Evidence available through','y':'Departure score'})
    fig.add_hline(y=cutoff,line_dash='dash')
    fig.add_vline(x=sample.index[ONSET].timestamp()*1000,line_dash='dot')
    fig.add_scatter(x=sample.index[reportdays[flags]],y=score[reportdays[flags]],mode='markers',name='Weekly report flags')
    st.plotly_chart(fig,use_container_width=True)
    st.write('First post-departure weekly flag: '+(f'{int(delay)} days after onset.' if np.isfinite(delay) else 'not detected during follow-up.'))
    st.write(f'Matched no-change control: {false}/{total} weekly reports flagged. Threshold: {cutoff:.3f}.')
    st.caption('Shape join compares independently normalized windows and can miss a persistent level shift. Peer residual mean uses a frozen baseline regression and preserves departures in level. Two-sided CUSUM accumulates residual changes with a fixed 0.5-sigma allowance. Lag applies only to shape joins; window applies to shape and residual mean. Calibration estimates thresholds from finite data; no method is guaranteed to meet its nominal quantile out of sample.')
    st.download_button('Download tuning sweep',tuning.to_csv(index=False),'weekly_tuning.csv','text/csv')
    st.download_button('Download held-out sweep',validation.to_csv(index=False),'weekly_validation.csv','text/csv')
    st.markdown('For a larger KPI universe, learn stable peer groups from historical residuals, freeze those peers for each evaluation period, then score each signal against a robust peer prediction. See [the approaches and experiment notes](https://github.com/rsangole/kpi-matrix-lab/blob/main/WEEKLY_DETECTION.md).')
