import numpy as np
from analysis import generate, relationships, relationships_fast

def test_280_series_and_seed_compatibility():
    a,_=generate(days=180,countries=40,lobs=7)
    assert a.shape==(180,280) and a.columns.is_unique
    b,_=generate(days=180)
    assert np.array_equal(a['TV+ · US'],b['TV+ · US'])

def test_vectorized_matches_reference_and_constant_behavior():
    a,_=generate(days=180,countries=5,lobs=2)
    a['constant']=1
    for anchor in [0,50,166]:
        slow,g=relationships(a,14,anchor,3,.75)
        fast,h=relationships_fast(a,14,anchor,3,.75)
        assert np.allclose(slow,fast,atol=1e-7)
        assert g==h

def test_causal_relationships_ignore_future_data():
    a,_=generate(days=180,countries=5,lobs=2)
    b=a.copy(); b.iloc[100:]+=100
    x,_=relationships_fast(a,14,86,3,.75,causal=True)
    y,_=relationships_fast(b,14,86,3,.75,causal=True)
    assert np.allclose(x,y)
