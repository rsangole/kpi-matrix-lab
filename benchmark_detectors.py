"""Three repeat stage timings for all 280-series detector configurations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import time
from suite import generate_suite,scores,flags
import pandas as pd

def cadence(chosen):
    rows=[]
    for repeat in range(3):
        start=time.perf_counter()
        data,_,_=generate_suite()
        outputs=scores(data)
        for setting in chosen.itertuples():
            pred=flags(outputs[(setting.method,int(setting.window))],setting.quantile,int(setting.persistence))
            latest_signal_indices=pred[-1].nonzero()[0]
        rows.append({'repeat':repeat,'seconds':time.perf_counter()-start,'series':280,'days':180,'selected_policies':len(chosen)})
    df=pd.DataFrame(rows)
    df.to_csv('data/event_suite_cadence_runtime.csv',index=False)
    return df

def main():
    x,_,_=generate_suite(); rows=[]
    for repeat in range(3):
        stage={}; scores(x,timings=stage)
        rows.extend({'workload':k,'repeat':repeat,'seconds':v} for k,v in stage.items())
    df=pd.DataFrame(rows)
    df.to_csv('data/event_suite_family_runtime.csv',index=False)
    print(df.groupby('workload').seconds.agg(['median','max']))
    from pathlib import Path
    if Path('data/event_suite_selected_settings.csv').exists():
        print('Cadence full recompute:',cadence(pd.read_csv('data/event_suite_selected_settings.csv'))) 

if __name__=='__main__': main()
