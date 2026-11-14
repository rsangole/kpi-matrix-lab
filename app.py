import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from analysis import generate, preprocess, profile, relationships

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
tab1,tab2,tab3,tab4=st.tabs(['Signal explorer','Motifs & anomalies','Groups & departures','Method & data'])
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
    st.plotly_chart(px.line(x=data.index[:len(p)],y=p,labels={'x':'Window start','y':'Normalized matrix profile distance'},template='plotly_dark'),use_container_width=True)
    st.caption('Self-join: low values indicate recurring motifs; high values indicate discords. Cross-series: low values indicate a similar window in the comparison signal within the allowed lag. Each point describes a full window.')
    start=st.slider('Inspect window start',0,len(p)-1,int(np.argmax(p)))
    match=int(idx[start]); target=focus if y is None else other
    w1=data[focus].values[start:start+m]; w2=data[target].values[match:match+m]
    z=lambda w:(w-w.mean())/(w.std() if w.std()>1e-10 else 1)
    st.plotly_chart(px.line(pd.DataFrame({'Reference':z(w1),'Nearest match':z(w2)}),labels={'index':'Day within window','value':'Standard deviations'},template='plotly_dark'),use_container_width=True)
    st.write(f'Reference starts {data.index[start].date()}; nearest match: {target}, {data.index[match].date()}. Distance: {p[start]:.3f}.')
with tab3:
    anchor=st.slider('Current window start',0,days-m,min(int(days*.8),days-m))
    baseline=st.slider('Baseline window start',0,days-m,int(days*.3))
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
