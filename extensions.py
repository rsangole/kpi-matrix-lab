"""Causal detector fusion, completed-week aggregation and trailing LOESS variants."""
import numpy as np
import pandas as pd
from suite import TRAIN, CAL, DAYS, normalized_windows, scores

MEMBERS=[('Historical matrix profile',7),('Peer shape distance',7),('Peer residual mean',7),('Peer residual spike',1),('Peer residual CUSUM',0)]
WEIGHTS={
    'equal':[.2,.2,.2,.2,.2],
    'residual-balanced':[.1,.1,.35,.35,.1],
    'shape-heavy':[.35,.35,.1,.1,.1],
    'spike-heavy':[.05,.05,.15,.7,.05],
    'sustained-heavy':[.05,.05,.55,.15,.2],
}

def trailing_loess(data,span):
    """Local linear regression with tricube weights, evaluated at the right endpoint.

    No robust reweighting iterations. Early windows use available past observations.
    Endpoint kernels can have negative coefficients; overshoot is possible.
    """
    x=data.values; out=np.empty_like(x)
    for width in range(1,span+1):
        t=np.arange(width,dtype=float)-(width-1)
        if width<3:
            kernel=np.zeros(width);kernel[-1]=1
        else:
            weights=(1-(np.abs(t)/width)**3)**3
            design=np.column_stack([np.ones(width),t])
            kernel=(np.linalg.pinv(design.T@(weights[:,None]*design))@design.T*weights)[0]
        if width<span:
            out[width-1]=kernel@x[:width]
        else:
            windows=np.lib.stride_tricks.sliding_window_view(x,width,axis=0)
            out[width-1:]=np.einsum('tnw,w->tn',windows,kernel)
    return pd.DataFrame(out,index=data.index,columns=data.columns)

def weekly_scores(data):
    """Nonoverlapping complete seven-day means; each value is dated at bin end."""
    count=len(data)//7
    x=data.values[:count*7].reshape(count,7,-1).mean(axis=1)
    ntrain=TRAIN//7
    mu=x[:ntrain].mean(axis=0);sd=np.maximum(x[:ntrain].std(axis=0),1e-8)
    z=(x-mu)/sd
    time=np.arange(count)
    design=np.column_stack([np.ones(count),time])
    coef=np.linalg.lstsq(design[:ntrain],z[:ntrain],rcond=None)[0]
    corr=np.corrcoef((z[:ntrain]-design[:ntrain]@coef).T)
    np.fill_diagonal(corr,-np.inf)
    peers=np.argsort(corr,axis=1)[:,-5:]
    peer=np.median(z[:,peers],axis=-1)
    residual=np.empty_like(z)
    for i in range(x.shape[1]):
        X=np.column_stack([np.ones(count),peer[:,i]])
        beta=np.linalg.lstsq(X[:ntrain],z[:ntrain,i],rcond=None)[0]
        r=z[:,i]-X@beta
        residual[:,i]=r/max(np.std(r[:ntrain],ddof=2),1e-8)
    output={'weekly residual / 1w':np.abs(residual)}
    for m in [2,3]:output[f'weekly residual mean / {m}w']=np.abs(pd.DataFrame(residual).rolling(m).mean().values)*np.sqrt(m)
    for m in [3,4]:
        w=normalized_windows(x,m)
        refs=w[:,:ntrain-m+1]
        novelty=np.sqrt(np.maximum(0,2-2*np.matmul(w,refs.transpose(0,2,1)).max(axis=2))).T
        output[f'weekly historical profile / {m}w']=np.vstack([np.full((m-1,x.shape[1]),np.nan),novelty])
        peer_distance=np.median(np.sqrt(np.maximum(0,2-2*np.sum(w[:,:,None,:]*w[peers].transpose(0,2,1,3),axis=-1))),axis=-1).T
        output[f'weekly peer shape / {m}w']=np.vstack([np.full((m-1,x.shape[1]),np.nan),peer_distance])
    c=np.full_like(residual,np.nan);up=np.zeros(x.shape[1]);down=up.copy()
    for d in range(ntrain,count):
        up=np.maximum(0,up+residual[d]-.5);down=np.maximum(0,down-residual[d]-.5)
        c[d]=np.maximum(up,down)
    output['weekly residual CUSUM']=c
    return output

def normalize(score,weekly=False):
    block=score[8:16] if weekly else score[TRAIN:CAL]
    scale=np.maximum(np.quantile(block,.95,axis=0),1e-8)
    return score/scale

def completed_week_hold(values):
    out=np.full((DAYS,values.shape[1]),np.nan)
    for week in range(len(values)):
        end=7*week+6
        out[end:min(end+7,DAYS)]=values[week]
    return out

def candidate_scores(data):
    daily=scores(data)
    result={};definitions={}
    def add(name,group,value,weekly=False):
        result[name]=value;definitions[name]={'candidate':name,'group':group,'weekly':weekly}
    for (method,m),v in daily.items():add(f'Daily | {method} | {m}d','Daily single detector',v)
    normalized=np.stack([normalize(daily[k]) for k in MEMBERS])
    for name,weights in WEIGHTS.items():
        value=np.einsum('k,ktn->tn',np.array(weights),normalized)
        add(f'Ensemble | weighted {name}','Daily ensemble',value)
    add('Ensemble | calibrated max','Daily ensemble',np.max(normalized,axis=0))
    # OR/vote use member thresholds, without pretending their combined rate is calibrated.
    for vote in [1,2]:
        for q in [.95,.99,.995]:
            thresholds=np.quantile(np.stack([daily[k][TRAIN:CAL] for k in MEMBERS]),q,axis=1)
            votes=np.sum(np.stack([daily[k] for k in MEMBERS])>thresholds[:,None,:],axis=0)
            name=f'Ensemble | {"OR" if vote==1 else "2-of-5 vote"} | member q={q}'
            add(name,'Daily ensemble',votes>=vote)
            definitions[name]['binary']=True
    week=weekly_scores(data)
    for name,v in week.items():add(name,'Weekly aggregation',v,True)
    week_fusion=(normalize(week['weekly residual / 1w'],True)+normalize(week['weekly historical profile / 3w'],True))/2
    raw_equal=np.mean(normalized,axis=0)
    for weight in [.25,.5]:
        combined=(1-weight)*raw_equal+weight*completed_week_hold(week_fusion)
        add(f'Daily + weekly | weekly weight {weight}','Daily + weekly ensemble',combined)
    for span in [7,14,28,42]:
        smoothed=scores(trailing_loess(data,span))
        for (method,m),v in smoothed.items():
            add(f'LOESS {span}d | {method} | {m}d','Trailing LOESS',v)
        if span in [14,28]:
            fusion=np.mean(np.stack([normalize(smoothed[k]) for k in MEMBERS]),axis=0)
            for weight in [.25,.5]:
                add(f'Daily + LOESS {span}d | smoothed weight {weight}','Daily + LOESS ensemble',(1-weight)*raw_equal+weight*fusion)
    return result,definitions

def candidate_flags(value,definition,quantile,persistence,cutoff=None):
    weekly=definition['weekly']
    cal=value[8:16] if weekly else value[TRAIN:CAL]
    exceed=value if definition.get('binary',False) else value>(np.quantile(cal,quantile,axis=0) if cutoff is None else cutoff)
    # Vectorized causal consecutive-exceedance confirmation; no pandas per-column loop.
    exceed=np.asarray(exceed,dtype=bool)&np.isfinite(value)
    accum=np.vstack([np.zeros((1,value.shape[1]),dtype=int),np.cumsum(exceed,axis=0)])
    counts=accum[persistence:]-accum[:-persistence]
    confirmed=np.vstack([np.zeros((persistence-1,value.shape[1]),dtype=bool),counts>=persistence])
    if weekly:return confirmed[16:25]
    ends=np.arange(CAL+6,DAYS,7)
    return np.stack([confirmed[t-6:t+1].any(axis=0) for t in ends])

def settings(definition):
    qs=[0.] if definition.get('binary',False) else [.95,.99,.995]
    return [(q,p) for q in qs for p in ([1,2] if definition['weekly'] else [1,3])]
