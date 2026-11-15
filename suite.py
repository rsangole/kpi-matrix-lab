"""Six-month synthetic event suite and causal multi-series detector evaluation."""
import time
import numpy as np
import pandas as pd
from analysis import COUNTRIES, PRODUCTS

TRAIN=56
CAL=112
DAYS=180
SCENARIOS=['Level shift','Random spikes','Temporary seasonality','Gradual trend','Rise and recovery','LOB offer','Developing-market offer','English-market offer','Mixed events']
# Explicit synthetic cohorts for testing, not an economic/geopolitical taxonomy.
DEVELOPING={'India','Brazil','China','Mexico','Argentina','Chile','Colombia','Malaysia','Thailand','Indonesia','Philippines','Vietnam','South Africa'}
ENGLISH={'US','UK','Canada','Australia','Ireland','New Zealand'}

def generate_suite(seed=42,noise=.8,scenario='Mixed events',amplitude=5.,changed=True):
    rng=np.random.default_rng(seed); t=np.arange(DAYS)
    meta=pd.DataFrame([{'lob':p,'geo':c,'developing_test_cohort':c in DEVELOPING,'english_test_cohort':c in ENGLISH} for p in PRODUCTS for c in COUNTRIES])
    n=len(meta)
    ar=rng.normal(size=(DAYS,7))
    for d in range(1,DAYS): ar[d]=.85*ar[d-1]+ar[d]
    common=rng.normal(size=DAYS)
    for d in range(1,DAYS): common[d]=.9*common[d-1]+common[d]
    values=np.empty((DAYS,n))
    for i,row in meta.iterrows():
        lob=i//40
        values[:,i]=100+20*lob+(i%40)*.7+(.02+.004*lob)*t+2*np.sin(2*np.pi*t/7+.1*lob)+3*np.sin(2*np.pi*t/365)+.7*common+.7*ar[:,lob]+rng.normal(0,noise,DAYS)
    events=[]; effect=np.zeros_like(values)
    types=SCENARIOS[:-1] if scenario=='Mixed events' else [scenario]
    # Deterministic targets: paired controls retain exactly identical noise histories.
    spike_rng=np.random.default_rng(seed+900000)
    for k,kind in enumerate(types):
        onset=126+(k%3)*7 if scenario=='Mixed events' else 126
        after=np.maximum(t-onset,0)
        mask=np.zeros(n,dtype=bool)
        if kind=='LOB offer': mask=meta.lob.isin(PRODUCTS[:2]).values
        elif kind=='Developing-market offer': mask=meta.developing_test_cohort.values
        elif kind=='English-market offer': mask=meta.english_test_cohort.values
        else: mask[np.arange(k*11,k*11+14)%n]=True
        end=DAYS-1
        if kind=='Level shift': shape=amplitude*(t>=onset)
        elif kind=='Gradual trend': shape=amplitude*after/21
        elif kind=='Temporary seasonality':
            end=onset+20; shape=amplitude*np.sin(2*np.pi*after/3.5)*((t>=onset)&(t<=end))
        elif kind=='Rise and recovery':
            end=onset+27; shape=amplitude*np.sin(np.pi*np.clip(after/28,0,1))
        elif kind=='Random spikes':
            # Three isolated signed impulses per selected series, not broad 3-week labels.
            for i in np.flatnonzero(mask):
                for day in sorted(spike_rng.choice(np.arange(onset,onset+28),3,replace=False)):
                    sign=spike_rng.choice([-1,1])
                    if changed: effect[day,i]+=amplitude*sign
                    events.append({'series':int(i),'type':kind,'start':int(day),'end':int(day)})
            continue
        else:
            end=onset+13; shape=amplitude*((t>=onset)&(t<=end))
        if changed: effect[:,mask]+=shape[:,None]
        events.extend({'series':int(i),'type':kind,'start':onset,'end':end} for i in np.flatnonzero(mask))
    names=[f'{r.lob} · {r.geo}' for r in meta.itertuples()]
    data=pd.DataFrame(values+effect,index=pd.date_range('2025-01-01',periods=DAYS),columns=names)
    return data,meta,pd.DataFrame(events)

def normalized_windows(values,m):
    w=np.lib.stride_tricks.sliding_window_view(values.T,m,axis=1)
    w=w-w.mean(axis=-1,keepdims=True)
    norm=np.linalg.norm(w,axis=-1,keepdims=True)
    return np.divide(w,norm,out=np.zeros_like(w),where=norm>1e-10)

def frozen_peers(data):
    x=data.values; t=np.arange(len(x)); weekday=np.eye(7)[t%7][:,1:]
    design=np.column_stack([np.ones(len(x)),t,weekday])
    beta=np.linalg.lstsq(design[:TRAIN],x[:TRAIN],rcond=None)[0]
    detrended=x[:TRAIN]-design[:TRAIN]@beta
    corr=np.corrcoef(detrended.T); np.fill_diagonal(corr,-np.inf)
    peers=np.argsort(corr,axis=1)[:,-5:]
    mu=x[:TRAIN].mean(axis=0); sigma=np.maximum(x[:TRAIN].std(axis=0),1e-8)
    z=(x-mu)/sigma
    median=np.median(z[:,peers],axis=-1)
    residual=np.empty_like(x)
    for i in range(x.shape[1]):
        # Freeze loadings, weekday effects and scale; no event labels or future fit.
        X=np.column_stack([np.ones(len(x)),median[:,i],weekday])
        coef=np.linalg.lstsq(X[:TRAIN],x[:TRAIN,i],rcond=None)[0]
        r=x[:,i]-X@coef
        residual[:,i]=r/max(np.std(r[:TRAIN],ddof=X.shape[1]),1e-8)
    return residual,peers

def scores(data,peers=None,timings=None):
    started=time.perf_counter()
    x=data.values; r,p=frozen_peers(data)
    if timings is not None: timings["Peer learning and residual fitting"]=time.perf_counter()-started
    if peers is not None: raise ValueError('Peers are learned inside each independent replicate')
    output={}
    for m in [7,14,28,42]:
        started=time.perf_counter()
        w=normalized_windows(x,m)
        refs=w[:,:TRAIN-m+1]
        # Frozen historical AB matrix profile: all baseline windows, every series.
        dot=np.matmul(w,refs.transpose(0,2,1))
        mp=np.sqrt(np.maximum(0,2-2*dot.max(axis=2))).T
        output[('Historical matrix profile',m)]=np.vstack([np.full((m-1,x.shape[1]),np.nan),mp])
        if timings is not None: timings[f"Historical matrix profile / {m}d"]=time.perf_counter()-started
        started=time.perf_counter()
        # Median of distances to five historically learned peers, calendar-aligned.
        similarities=np.sum(w[:,:,None,:]*w[p].transpose(0,2,1,3),axis=-1)
        ps=np.median(np.sqrt(np.maximum(0,2-2*similarities)),axis=-1).T
        output[('Peer shape distance',m)]=np.vstack([np.full((m-1,x.shape[1]),np.nan),ps])
        if timings is not None: timings[f"Peer shape distance / {m}d"]=time.perf_counter()-started
        started=time.perf_counter()
        mean=pd.DataFrame(r).rolling(m).mean().values
        output[('Peer residual mean',m)]=np.abs(mean)*np.sqrt(m)
        if timings is not None: timings[f"Peer residual mean / {m}d"]=time.perf_counter()-started
    started=time.perf_counter()
    output[('Peer residual spike',1)]=np.abs(r)
    if timings is not None: timings["Peer residual spike"]=time.perf_counter()-started
    started=time.perf_counter()
    cusum=np.full_like(r,np.nan); up=np.zeros(x.shape[1]); down=up.copy()
    for d in range(TRAIN,len(x)):
        up=np.maximum(0,up+r[d]-.5); down=np.maximum(0,down-r[d]-.5)
        cusum[d]=np.maximum(up,down)
    output[('Peer residual CUSUM',0)]=cusum
    if timings is not None: timings["Peer residual CUSUM"]=time.perf_counter()-started
    return output

def report_truth(events,n=280):
    ends=np.arange(CAL+6,DAYS,7)
    truth=np.zeros((len(ends),n),dtype=bool)
    for e in events.itertuples():
        truth[(ends>=e.start)&(ends-6<=e.end),e.series]=True
    return ends,truth

def flags(score,q,persistence):
    threshold=np.nanquantile(score[TRAIN:CAL],q,axis=0)
    exceed=(score>threshold)&np.isfinite(score)
    run=pd.DataFrame(exceed).rolling(persistence).sum().values>=persistence
    ends=np.arange(CAL+6,DAYS,7)
    return np.stack([run[d-6:d+1].any(axis=0) for d in ends])

def measure(pred,control,events,context=None):
    if context is None:
        ends,truth=report_truth(events,pred.shape[1])
    else:
        ends,truth=context
    tp=int((pred&truth).sum()); fp=int((pred&~truth).sum())
    fn=int((~pred&truth).sum()); tn=int((~pred&~truth).sum())
    starts=events.start.values
    stops=events.end.values
    allowed=(ends[:,None]>=starts)&(ends[:,None]<=np.minimum(starts+14,stops+6))
    detected=pred[:,events.series.values]&allowed
    ds=np.min(np.where(detected,ends[:,None]-starts,np.inf),axis=0).astype(float)
    ds[~np.isfinite(ds)]=np.nan
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,event_count=len(ds),events_7=int(np.sum(ds<=7)),events_14=int(np.isfinite(ds).sum()),control_flags=int(control.sum()),control_opportunities=control.size,control_family_reports=int(control.any(axis=1).sum()),reports=len(ends),delay_sum=float(np.nansum(ds)),detected_events=int(np.isfinite(ds).sum()))
