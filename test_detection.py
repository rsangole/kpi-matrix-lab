import numpy as np
from detection import pair, score_series, evaluate, reports, ONSET, CALIBRATION

def test_future_changes_do_not_change_past_scores():
    a=pair(); b=a.copy(); b.iloc[250:,1]+=1000
    for method in ['Shape AB join','Peer residual mean','Residual CUSUM']:
        x=score_series(a,14,3,method); y=score_series(b,14,3,method)
        assert np.allclose(x[:250],y[:250],equal_nan=True)

def test_controls_share_identical_pre_onset_data():
    a=pair(changed=True); b=pair(changed=False)
    assert a.iloc[:ONSET].equals(b.iloc[:ONSET])
    assert not a.equals(b)

def test_weekly_reports_use_previous_seven_days():
    scores=np.zeros(336); scores[ONSET:]=10
    days,flags=reports(scores,5,3)
    first=days[flags][0]
    assert first>=ONSET+2 and first<=ONSET+8
    assert np.all(np.diff(days)==7)

def test_threshold_does_not_know_event_amplitude():
    low=evaluate(42,7,0,'Peer residual mean',.99,1,.8,0,'Level shift')
    high=evaluate(42,7,0,'Peer residual mean',.99,1,.8,8,'Level shift')
    assert low[4]==high[4]
    assert np.isfinite(high[0]) and high[0]<=7
