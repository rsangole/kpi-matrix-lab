import numpy as np
import pandas as pd
from suite import generate_suite,scores,flags,measure,report_truth,SCENARIOS

def test_suite_shape_cohorts_and_matched_controls():
    a,meta,e=generate_suite()
    b,_,f=generate_suite(changed=False)
    assert a.shape==(180,280) and len(meta)==280
    assert e.equals(f)
    assert a.iloc[:126].equals(b.iloc[:126])
    assert len(e.type.unique())==8
    assert meta.developing_test_cohort.sum()==13*7
    assert meta.english_test_cohort.sum()==6*7

def test_targeted_offers_have_no_effect_outside_cohort():
    for scenario,column in [('Developing-market offer','developing_test_cohort'),('English-market offer','english_test_cohort')]:
        a,meta,_=generate_suite(scenario=scenario)
        b,_,_=generate_suite(scenario=scenario,changed=False)
        assert np.allclose(a.values[:,~meta[column].values],b.values[:,~meta[column].values])
        assert not np.allclose(a,b)

def test_all_detectors_do_not_use_future_samples():
    a,_,_=generate_suite()
    b=a.copy(); b.iloc[150:]+=100
    sa=scores(a); sb=scores(b)
    assert len(sa)==14
    for key in sa:
        assert np.allclose(sa[key][:150],sb[key][:150],equal_nan=True,atol=1e-7),key

def test_exact_historical_matrix_profile_against_brute_force():
    a,_,_=generate_suite()
    score=scores(a)[('Historical matrix profile',7)]
    x=a.iloc[:,0].values
    z=lambda w:(w-w.mean())/w.std()
    for end in [60,126,150]:
        ref=z(x[end-6:end+1])
        expected=min(np.linalg.norm(ref-z(x[j:j+7]))/np.sqrt(7) for j in range(50))
        assert np.isclose(score[end,0],expected)

def test_confusion_and_short_event_delay():
    events=pd.DataFrame([{'series':0,'type':'spike','start':126,'end':126}])
    ends,truth=report_truth(events,n=2)
    pred=truth.copy(); control=np.zeros_like(pred)
    result=measure(pred,control,events)
    assert result['tp']==1 and result['fp']==0 and result['fn']==0
    assert result['events_7']==1 and result['delay_sum']==6
    pred[:]=False; pred[-1,0]=True
    result=measure(pred,control,events)
    assert result['events_14']==0 and result['fp']==1

def test_budget_failure_stays_visible_in_selected_settings():
    from run_suite import select_settings
    rows=[dict(split='tuning',method='A',window=7,quantile=.99,persistence=1,control_weekly_flag_rate=.1,f1=.9,event_recall_7d=.9),dict(split='tuning',method='A',window=14,quantile=.99,persistence=3,control_weekly_flag_rate=.08,f1=.5,event_recall_7d=.5),dict(split='tuning',method='B',window=7,quantile=.99,persistence=1,control_weekly_flag_rate=.04,f1=.7,event_recall_7d=.8)]
    chosen=select_settings(pd.DataFrame(rows))
    assert len(chosen)==2
    a=chosen[chosen.method=='A'].iloc[0]
    assert not a.tuning_budget_met and a.window==14
    assert chosen[chosen.method=='B'].iloc[0].tuning_budget_met
