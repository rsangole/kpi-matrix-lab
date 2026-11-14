"""Exact, small-data matrix profiles and reproducible synthetic business signals."""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

COUNTRIES = ['US', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Japan', 'India', 'Brazil', 'China']
PRODUCTS = ['TV+', 'App Store', 'Fitness+', 'News+', 'Apple TV hardware']

def generate(days=540, countries=10, seed=42, noise=0.8, strength=12., drift=True):
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    dates = pd.date_range('2024-01-01', periods=days)
    launch, promo, price = int(days*.3), int(days*.58), int(days*.77)
    events = pd.DataFrame({'day': [launch,promo,price], 'event': ['TV+ premiere', 'Services promotion', 'Price change / India regime shift']})
    signals = {}
    for p, product in enumerate(PRODUCTS):
        for c, country in enumerate(COUNTRIES[:countries]):
            base = 90 + p*23 + c*7
            organic = .025*t + 3*np.sin(2*np.pi*t/365) + 2*np.sin(2*np.pi*t/7+p*.3)
            pulse = lambda center, width: np.exp(-.5*((t-center)/width)**2)
            effect = strength*pulse(promo,12)*(1 if p<4 else .3)
            if p==0:
                effect += strength*1.8*pulse(launch+(2 if c==1 else 0),9)*(1 if c<6 else .12)
            if p==2:
                effect += strength*.6*pulse(launch+3,9)*(1 if c<4 else .1)
            effect -= strength*.7*(t>=price)*(1-np.exp(-np.maximum(t-price,0)/10))*(p<4)
            if drift and country=='India' and p==0:
                effect += strength*1.8*pulse(price+22,8) + 8*np.sin((t-price)/5)*(t>=price)
            signals[f'{product} · {country}'] = np.maximum(1,base+organic+effect+rng.normal(0,noise,days))
    return pd.DataFrame(signals,index=dates), events

def preprocess(data, mode):
    if mode=='First difference': return data.diff().fillna(0)
    if mode=='Remove trend + weekly seasonality':
        trend=data.rolling(35,center=True,min_periods=1).mean()
        residual=data-trend
        seasonal=residual.groupby(residual.index.dayofweek).transform('mean')
        return residual-seasonal
    return data.copy()

def windows(x,m):
    x=np.asarray(x,dtype=float)
    if not 4<=m<=len(x)//2: raise ValueError('Window must be between 4 and half the series length')
    w=sliding_window_view(x,m)
    centered=w-w.mean(axis=1,keepdims=True)
    norm=np.linalg.norm(centered,axis=1,keepdims=True)
    return np.divide(centered,norm,out=np.zeros_like(centered),where=norm>1e-10)

def distances(a,b,m):
    # z-normalized Euclidean distance / sqrt(m), equivalently sqrt(2-2r).
    d=np.sqrt(np.maximum(0,2-2*(a@b.T)))
    ca=np.linalg.norm(a,axis=1)<1e-10; cb=np.linalg.norm(b,axis=1)<1e-10
    d[ca,:]=1; d[:,cb]=1; d[np.ix_(ca,cb)]=0
    return d

def profile(x,m,y=None,lag=None):
    a=windows(x,m); b=a if y is None else windows(y,m)
    d=distances(a,b,m)
    delta=np.abs(np.arange(len(a))[:,None]-np.arange(len(b))[None,:])
    if y is None: d[delta<=int(np.ceil(m/4))]=np.inf
    elif lag is not None: d[delta>lag]=np.inf
    idx=np.argmin(d,axis=1)
    return d[np.arange(len(a)),idx],idx

def relationships(data,m,anchor,lag,threshold):
    names=list(data.columns); ws=[windows(data[n].values,m) for n in names]
    anchor=min(anchor,len(ws[0])-1)
    scores=np.zeros((len(names),len(names)))
    for i in range(len(names)):
        for j in range(i):
            lo=max(0,anchor-lag); hi=min(len(ws[j]),anchor+lag+1)
            # Symmetric local AB matching, allowing bounded calendar lag.
            d1=distances(ws[i][anchor:anchor+1],ws[j][lo:hi],m).min()
            d2=distances(ws[j][anchor:anchor+1],ws[i][lo:hi],m).min()
            scores[i,j]=scores[j,i]=(d1+d2)/2
    unseen=set(range(len(names))); groups=[]
    while unseen:
        todo=[min(unseen)]; group=[]
        while todo:
            i=todo.pop()
            if i not in unseen: continue
            unseen.remove(i); group.append(names[i])
            todo.extend(j for j in sorted(unseen) if scores[i,j]<=threshold)
        groups.append(group)
    return pd.DataFrame(scores,index=names,columns=names),groups
