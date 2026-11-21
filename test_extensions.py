import numpy as np
import pandas as pd
from suite import generate_suite,measure
from extensions import trailing_loess,weekly_scores,completed_week_hold,candidate_scores,candidate_flags

def test_trailing_loess_is_causal_and_reproduces_linear_trends():
    x=pd.DataFrame({'a':np.arange(180,dtype=float),'b':np.ones(180)})
    for span in [7,14,28,42]:
        s=trailing_loess(x,span)
        assert np.allclose(s,x,atol=1e-8)
        y=x.copy();y.iloc[150:]+=1000
        assert np.allclose(s.iloc[:150],trailing_loess(y,span).iloc[:150])

def test_complete_weeks_never_arrive_early():
    v=np.arange(25)[:,None].astype(float)
    held=completed_week_hold(v)
    assert np.isnan(held[:6]).all()
    assert np.all(held[6:13]==0)
    assert held[13,0]==1
    score=np.zeros((25,1));score[18,0]=100
    p=candidate_flags(score,{'weekly':True},.99,1)
    assert p.shape==(9,1) and p[2,0] and p.sum()==1

def test_all_variant_scores_ignore_future_data():
    a,_,_=generate_suite();b=a.copy();b.iloc[150:]+=100
    av,definitions=candidate_scores(a);bv,_=candidate_scores(b)
    for key in av:
        cutoff=150//7 if definitions[key]['weekly'] else 150
        assert np.allclose(av[key][:cutoff],bv[key][:cutoff],equal_nan=True,atol=1e-6),key

def test_persistence_and_weekly_metrics_share_original_dates():
    score=np.zeros((180,1));score[126:]=10
    pred=candidate_flags(score,{'weekly':False},.99,3)
    events=pd.DataFrame([dict(series=0,type='shift',start=126,end=179)])
    r=measure(pred,np.zeros_like(pred),events)
    assert r['events_7']==1 and r['delay_sum']==6
