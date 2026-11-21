"""Fresh-seed evaluation of ensembles, weekly downsampling and causal LOESS."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import json,time,resource,gzip
from pathlib import Path
import numpy as np
import pandas as pd
from suite import generate_suite,measure,report_truth,SCENARIOS,TRAIN,CAL
from extensions import candidate_scores,candidate_flags,settings,WEIGHTS
from run_suite import aggregate

KEY=['candidate','quantile','persistence']

def strategy(name,group):
    if group!='Daily ensemble':return group
    if 'weighted equal' in name:return 'Equal-weight mean'
    if 'weighted ' in name:return 'Tuned weighted mean'
    if 'calibrated max' in name:return 'Calibrated max'
    if '| OR |' in name:return 'Naive OR'
    return '2-of-5 vote'

def pick(table,label):
    eligible=table[table.control_weekly_flag_rate<=.05]
    passed=not eligible.empty
    if passed:row=eligible.sort_values(['f1','event_recall_7d','control_weekly_flag_rate'],ascending=[False,False,True]).iloc[0]
    else:row=table.sort_values(['control_weekly_flag_rate','f1'],ascending=[True,False]).iloc[0]
    return dict(model=label,**{k:row[k] for k in KEY},tuning_budget_met=passed,tuning_control_flag_rate=row.control_weekly_flag_rate)

def summarize(path='data/extensions_runs.csv.gz'):
    raw=pd.read_csv(path);catalog=pd.read_csv('data/extensions_catalog.csv')
    pooled=aggregate(raw,['split']+KEY).merge(catalog,on='candidate')
    pooled.to_csv('data/extensions_pooled.csv',index=False)
    tune=pooled[pooled.split=='tuning'];selected=[]
    for group in catalog.group.unique():selected.append(pick(tune[tune.group==group],group))
    for name in ['Equal-weight mean','Tuned weighted mean','Calibrated max','Naive OR','2-of-5 vote']:
        selected.append(pick(tune[tune.strategy==name],'Fusion: '+name))
    for span in [7,14,28,42]:
        selected.append(pick(tune[tune.candidate.str.startswith(f'LOESS {span}d |')],f'LOESS span {span}d'))
    selected=pd.DataFrame(selected);selected.to_csv('data/extensions_selected.csv',index=False)
    hold=raw[raw.split=='heldout'].merge(selected,on=KEY)
    hold.to_csv('data/extensions_selected_runs.csv',index=False)
    summary=aggregate(hold,['model'])
    summary.to_csv('data/extensions_summary.csv',index=False)
    detail=aggregate(hold,['model','scenario','noise'])
    detail.to_csv('data/extensions_by_scenario.csv',index=False)
    lifts=paired_lifts(hold)
    lifts.to_csv('data/extensions_lift.csv',index=False)
    print(summary[['model','precision','recall','f1','event_recall_7d','control_weekly_flag_rate','control_family_report_rate']].to_string(index=False),flush=True)
    return raw,catalog,selected,summary,detail,lifts

def paired_lifts(hold):
    metrics=['tp','fp','fn','event_count','events_7','events_14','control_flags','control_opportunities']
    grouped=hold.groupby(['model','seed'])[metrics].sum()
    seeds=sorted(hold.seed.unique());rng=np.random.default_rng(3107)
    boot=rng.integers(0,len(seeds),size=(2000,len(seeds)))
    def values(model):
        arr=grouped.loc[model].reindex(seeds)[metrics].values
        sums=arr.sum(axis=0)
        samples=arr[boot].sum(axis=1)
        def calc(v):
            f1=2*v[...,0]/np.maximum(1,2*v[...,0]+v[...,1]+v[...,2])
            ev=v[...,4]/np.maximum(1,v[...,3])
            false=v[...,6]/np.maximum(1,v[...,7])
            return np.stack([f1,ev,false],axis=-1)
        return calc(sums),calc(samples)
    rows=[]
    comparisons=[(m,'Daily single detector') for m in hold.model.unique() if m!='Daily single detector']
    comparisons.append(('Fusion: Tuned weighted mean','Fusion: Equal-weight mean'))
    for model,reference in comparisons:
        a,ab=values(model);b,bb=values(reference)
        d=a-b;lo,hi=np.quantile(ab-bb,[.025,.975],axis=0)
        rows.append(dict(model=model,reference=reference,f1_lift=d[0],f1_low=lo[0],f1_high=hi[0],event7_lift=d[1],event7_low=lo[1],event7_high=hi[1],control_flag_lift=d[2],control_low=lo[2],control_high=hi[2]))
    return pd.DataFrame(rows)

def main():
    start=time.perf_counter();path=Path('data/extensions_runs.csv.gz');path.unlink(missing_ok=True)
    runtime=[];first=True;catalog=None
    for split,seeds in [('tuning',range(142,148)),('heldout',range(20042,20050))]:
        for noise in [.8,1.6]:
            for seed in seeds:
                base,_,_=generate_suite(seed,noise,changed=False)
                t=time.perf_counter();control,definitions=candidate_scores(base)
                runtime.append(dict(split=split,seed=seed,noise=noise,scenario='Control',seconds=time.perf_counter()-t))
                if catalog is None:
                    catalog=pd.DataFrame([dict(**d,strategy=strategy(name,d['group'])) for name,d in definitions.items()])
                    catalog.to_csv('data/extensions_catalog.csv',index=False)
                    print(len(catalog),'candidates;',sum(len(settings(v)) for v in definitions.values()),'settings per dataset',flush=True)
                cuts={};cflags={}
                for name,value in control.items():
                    definition=definitions[name]
                    for q,p in settings(definition):
                        k=(name,q)
                        if k not in cuts:
                            cuts[k]=None if definition.get('binary',False) else np.quantile(value[8:16] if definition['weekly'] else value[TRAIN:CAL],q,axis=0)
                        cflags[(name,q,p)]=candidate_flags(value,definition,q,p,cuts[k])
                rows=[]
                for scenario in SCENARIOS:
                    data,_,events=generate_suite(seed,noise,scenario)
                    t=time.perf_counter();values,_=candidate_scores(data)
                    runtime.append(dict(split=split,seed=seed,noise=noise,scenario=scenario,seconds=time.perf_counter()-t))
                    context=report_truth(events)
                    for name,value in values.items():
                        definition=definitions[name]
                        for q,p in settings(definition):
                            pred=candidate_flags(value,definition,q,p,cuts[(name,q)])
                            result=measure(pred,cflags[(name,q,p)],events,context)
                            rows.append(dict(split=split,seed=seed,noise=noise,scenario=scenario,candidate=name,quantile=q,persistence=p,**result))
                with gzip.open(path,'at') as f:pd.DataFrame(rows).to_csv(f,index=False,header=first)
                first=False
                print(f'{split} seed {seed} / noise {noise}: nine scenarios complete; elapsed {time.perf_counter()-start:.0f}s',flush=True)
    pd.DataFrame(runtime).to_csv('data/extensions_runtime.csv',index=False)
    metadata=dict(tuning_seeds=list(range(142,148)),heldout_seeds=list(range(20042,20050)),noise=[.8,1.6],scenarios=SCENARIOS,series=280,days=180,training_days=56,calibration_days=56,candidates=len(catalog),settings_per_dataset=sum(len(settings(v)) for v in definitions.values()),datasets=252,weights=WEIGHTS,members=['Historical profile 7d','Peer shape 7d','Peer residual mean 7d','Peer residual spike 1d','Peer residual CUSUM'],full_sweep_seconds=time.perf_counter()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    Path('data/extensions_metadata.json').write_text(json.dumps(metadata,indent=2))
    summarize()
    from benchmark_extension_cadence import main as benchmark_cadence
    benchmark_cadence()
    from render_event_report import main as render_report
    render_report()
    print('Extension sweep complete',flush=True)

if __name__=='__main__':main()
