import numpy as np
from analysis import generate, profile, relationships, preprocess

def test_seed_and_ground_truth():
    a,e=generate(); b,_=generate()
    assert a.equals(b) and a.shape==(540,50) and len(e)==3

def test_exact_profile_matches_brute_force():
    x=np.random.default_rng(2).normal(size=90); m=12
    p,idx=profile(x,m)
    for i in [0,20,78]:
        distances=[]
        a=x[i:i+m]; a=(a-a.mean())/a.std()
        for j in range(len(p)):
            b=x[j:j+m]; b=(b-b.mean())/b.std()
            distances.append(np.inf if abs(i-j)<=3 else np.linalg.norm(a-b)/np.sqrt(m))
        assert np.isclose(p[i],min(distances))
        assert abs(i-idx[i])>3

def test_cross_join_and_constants():
    x=np.sin(np.arange(100)/4)
    p,idx=profile(x,14,x,0)
    assert np.max(p)<1e-6 and np.array_equal(idx,np.arange(len(p)))
    p,_=profile(np.ones(100),14)
    assert np.all(p==0)

def test_group_matrix_and_preprocessing():
    a,_=generate(days=180,countries=5)
    a=preprocess(a,'Remove trend + weekly seasonality')
    s,g=relationships(a,14,50,3,.8)
    assert np.allclose(s,s.T) and np.allclose(np.diag(s),0)
    assert sorted(sum(g,[]))==sorted(a.columns)
