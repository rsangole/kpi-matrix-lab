"""Time full extension score refresh and selected-policy reporting, three repeats."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import time
import pandas as pd
from suite import generate_suite
from extensions import candidate_scores,candidate_flags

def main():
    chosen=pd.read_csv('data/extensions_selected.csv').drop_duplicates(['candidate','quantile','persistence'])
    rows=[]
    for repeat in range(3):
        start=time.perf_counter();data,_,_=generate_suite(seed=142)
        values,definitions=candidate_scores(data)
        for policy in chosen.itertuples():
            pred=candidate_flags(values[policy.candidate],definitions[policy.candidate],policy.quantile,int(policy.persistence))
            _=pred[-1].nonzero()[0]
        rows.append(dict(repeat=repeat,seconds=time.perf_counter()-start,series=280,days=180,unique_selected_policies=len(chosen)))
    result=pd.DataFrame(rows);result.to_csv('data/extensions_cadence.csv',index=False)
    print(result.to_string(index=False))
    return result

if __name__=='__main__':main()
