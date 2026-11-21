"""Causal weekly detection benchmark with separate calibration and held-out seeds."""
import numpy as np
import pandas as pd
from analysis import windows

TRAIN, CALIBRATION, ONSET, LENGTH = 84, 168, 210, 336
METHODS = ['Shape AB join', 'Peer residual mean', 'Residual CUSUM']

def pair(seed=42, noise=.8, effect=3., kind='Level shift', changed=True):
    rng=np.random.default_rng(seed); t=np.arange(LENGTH)
    shared=100+.025*t+3*np.sin(t*2*np.pi/365)+2*np.sin(t*2*np.pi/7)
    shared+=4*np.exp(-.5*((t-130)/12)**2)
    x=shared+rng.normal(0,noise,LENGTH)
    y=25+1.2*shared+rng.normal(0,noise,LENGTH)
    after=np.maximum(t-ONSET+1,0)
    if kind=='Level shift': change=effect*(t>=ONSET)
    elif kind=='Growth change': change=effect*after/7
    else: change=effect*np.sin(after*2*np.pi/9)*(t>=ONSET)
    if changed: y+=change
    return pd.DataFrame({'TV+ · US':x,'Fitness+ · US':y},index=pd.date_range('2024-01-01',periods=LENGTH))

def residuals(data):
    # Frozen pre-event regression. Weekday effects and peer loading are fit only to training data.
    t=np.arange(len(data)); weekday=np.eye(7)[t%7][:,1:]
    X=np.column_stack([np.ones(len(t)),data.iloc[:,0],weekday])
    beta=np.linalg.lstsq(X[:TRAIN],data.iloc[:TRAIN,1],rcond=None)[0]
    r=data.iloc[:,1].values-X@beta
    return r/max(np.std(r[:TRAIN],ddof=X.shape[1]),1e-8)

def score_series(data,m,lag,method):
    if method=='Shape AB join':
        a=windows(data.iloc[:,0],m); b=windows(data.iloc[:,1],m)
        best=np.full(len(a),np.inf)
        # Both directions use only windows whose ends have already arrived.
        for shift in range(lag+1):
            n=len(a)-shift
            def row_distance(u,v):
                val=np.sqrt(np.maximum(0,2-2*np.sum(u*v,axis=1)))
                ca=np.linalg.norm(u,axis=1)<1e-10; cb=np.linalg.norm(v,axis=1)<1e-10
                val[ca|cb]=1; val[ca&cb]=0
                return val
            forward=row_distance(a[shift:],b[:n])
            reverse=row_distance(b[shift:],a[:n])
            # Conservative symmetric distance: both directional scores must agree.
            best[shift:]=np.minimum(best[shift:],(forward+reverse)/2)
        return np.r_[np.full(m-1,np.nan),best]
    r=residuals(data)
    if method=='Peer residual mean':
        return np.abs(pd.Series(r).rolling(m).mean().values)*np.sqrt(m)
    out=np.full(len(r),np.nan); up=down=0.
    for t in range(TRAIN,len(r)):
        up=max(0,up+r[t]-.5); down=max(0,down-r[t]-.5)
        out[t]=max(up,down)
    return out

def reports(score,threshold,persistence):
    exceed=np.isfinite(score)&(score>threshold)
    confirmed=pd.Series(exceed).rolling(persistence).sum().values>=persistence
    # Report any newly confirmed or continuing alert in the preceding seven days.
    report_days=np.arange(CALIBRATION+6,len(score),7)
    flags=np.array([confirmed[t-6:t+1].any() for t in report_days])
    return report_days,flags

def evaluate(seed,m,lag,method,quantile,persistence,noise,effect,kind):
    changed=pair(seed,noise,effect,kind,True); control=pair(seed,noise,effect,kind,False)
    score=score_series(changed,m,lag,method)
    threshold=float(np.nanquantile(score[TRAIN:CALIBRATION],quantile))
    days,flags=reports(score,threshold,persistence)
    _,nullflags=reports(score_series(control,m,lag,method),threshold,persistence)
    hits=days[flags&(days>=ONSET)]
    delay=float(hits[0]-ONSET) if len(hits) else np.nan
    return delay,int(np.sum(nullflags)),len(nullflags),score,threshold,days,flags


def sweep(seed=42,repeats=8,noise=.8,effect=3.,kind='Level shift',heldout=False):
    rows=[]
    for method in METHODS:
        for m in ([7,14,28,42] if method!='Residual CUSUM' else [7]):
            for lag in ([0,3] if method=='Shape AB join' else [0]):
                for quantile in [.95,.99]:
                    for persistence in [1,3]:
                        delays=[]; false=total=0
                        for r in range(repeats):
                            s=seed+r+(10000 if heldout else 0)
                            d,f,n,*_=evaluate(s,m,lag,method,quantile,persistence,noise,effect,kind)
                            delays.append(d); false+=f; total+=n
                        ds=np.array(delays)
                        rows.append({'Method':method,'Window':m if method!='Residual CUSUM' else 0,'Lag':lag,'Quantile':quantile,'Persistence':persistence,'Detected within 7d':float(np.mean(ds<=7)),'Detected within 14d':float(np.mean(ds<=14)),'Miss rate':float(np.mean(np.isnan(ds))),'Median delay (detected only)':float(np.nanmedian(ds)) if np.isfinite(ds).any() else np.nan,'No-change weekly alert rate':false/total})
    return pd.DataFrame(rows)
