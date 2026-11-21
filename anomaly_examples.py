"""Visual ground-truth examples for the event report (no detector rerun)."""
from html import escape
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from suite import generate_suite, SCENARIOS

DESCRIPTIONS={
    'Level shift':'The affected signal jumps by five units and stays there. Weekly seasonality and noise continue underneath the shift.',
    'Random spikes':'Three randomly timed, signed one-day impulses. Each shaded sliver is a separate spike, rather than one long event.',
    'Temporary seasonality':'An extra 3.5-day oscillation appears for 21 days, then disappears; the original weekly cycle remains.',
    'Gradual trend':'The signal develops an additional slope of five units per 21 days. The change builds gradually rather than jumping.',
    'Rise and recovery':'A temporary 28-day rise returns to its original baseline, illustrating an expenditure surge followed by recovery.',
    'LOB offer':'A 14-day offer adds five units only to TV+ and App Store. The Fitness+ example stays unaffected.',
    'Developing-market offer':'The offer affects only the explicitly defined developing-market test cohort. India is affected; the US comparison is not.',
    'English-market offer':'The offer affects only the explicitly defined English-market test cohort. The US is affected; Germany is not.',
    'Mixed events':'Several injections overlap in time and across signals. The heatmap shows the exact combined effect for every signal; this is ground truth, not an anomaly score.',
}

def visual_examples(chart):
    parts=['<section id="anomaly-examples"><h2>Visual examples of every anomaly</h2><p>Seed 42, noise standard deviation 0.8, the same generator and amplitude used in the benchmark. Solid teal shows an injected series; dashed gray is its exact matched no-event baseline, with identical background and noise. The lower panel subtracts that baseline to expose the injected effect. Shading follows the event ledger. These examples illustrate event truth, not detector success.</p>']
    slug=lambda s:s.lower().replace(' ','-')
    parts.append('<nav class="anomaly-nav">'+''.join(f'<a href="#example-{slug(s)}">{escape(s)}</a>' for s in SCENARIOS)+'</nav>')
    for scenario in SCENARIOS:
        data,meta,events=generate_suite(scenario=scenario)
        baseline,_,_=generate_suite(scenario=scenario,changed=False)
        delta=data-baseline
        affected=int(events.series.iloc[0])
        alltargets=set(events.series)
        unaffected=next(i for i in range(data.shape[1]) if i not in alltargets) if len(alltargets)<data.shape[1] else None
        targeted=scenario in ['LOB offer','Developing-market offer','English-market offer']
        if scenario=='LOB offer': unaffected=80
        if scenario=='Mixed events': affected=44
        rows=3 if targeted else 2
        subtitles=[f'Affected: {data.columns[affected]}']
        if targeted: subtitles.append(f'Untargeted comparison: {data.columns[unaffected]}')
        subtitles.append('Exact injected change (KPI units)' if scenario!='Mixed events' else 'Combined injected change across all 280 signals')
        fig=make_subplots(rows=rows,cols=1,shared_xaxes=True,vertical_spacing=.09,subplot_titles=subtitles,row_heights=([.34,.34,.32] if targeted else [.5,.5]))
        def signal(i,row,legend):
            fig.add_trace(go.Scatter(x=data.index,y=baseline.iloc[:,i],name='Matched baseline — no event',mode='lines',line=dict(color='#8493a8',dash='dash',width=2),legendgroup='baseline',showlegend=legend),row=row,col=1)
            fig.add_trace(go.Scatter(x=data.index,y=data.iloc[:,i],name='Observed synthetic signal',mode='lines',line=dict(color='#087f70',width=2),legendgroup='observed',showlegend=legend),row=row,col=1)
            fig.update_yaxes(title_text='KPI index',row=row,col=1)
        signal(affected,1,True)
        if targeted: signal(unaffected,2,False)
        if scenario=='Mixed events':
            fig.add_trace(go.Heatmap(x=data.index,y=data.columns,z=delta.T.values,colorscale='RdBu_r',zmid=0,colorbar=dict(title='Δ KPI',len=.38,y=.18),hovertemplate='%{y}<br>%{x|%b %d}<br>Injected change: %{z:.2f}<extra></extra>'),row=2,col=1)
            fig.update_yaxes(showticklabels=False,title_text='280 signals',row=2,col=1)
        else:
            fig.add_trace(go.Scatter(x=data.index,y=delta.iloc[:,affected],name='Injected change — affected',mode='lines+markers' if scenario=='Random spikes' else 'lines',marker=dict(size=4),line=dict(color='#c26323',width=2)),row=rows,col=1)
            if targeted:
                fig.add_trace(go.Scatter(x=data.index,y=delta.iloc[:,unaffected],name='Injected change — untargeted (zero)',line=dict(color='#526ab1',dash='dot',width=2)),row=rows,col=1)
            fig.add_hline(y=0,line_color='#ccd4df',line_width=1,row=rows,col=1)
            fig.update_yaxes(title_text='Δ KPI',row=rows,col=1)
        # Match exact labeled intervals for the displayed series, not a broad span.
        for event in events[events.series==affected].itertuples():
            x0=data.index[event.start]-np.timedelta64(12,'h')
            x1=data.index[event.end]+np.timedelta64(12,'h')
            fig.add_vrect(x0=x0,x1=x1,fillcolor='#e7ad54',opacity=.13,line_width=0,row=1,col=1)
        fig.update_xaxes(range=[data.index[100],data.index[-1]],tickformat='%b %d')
        fig.update_xaxes(title_text='Date',row=rows,col=1)
        fig.update_layout(height=790 if targeted else 660,hovermode='x unified' if scenario!='Mixed events' else 'closest',legend=dict(orientation='h',y=-.16,x=0),margin=dict(t=55,b=125,l=62,r=30))
        parts.append(f'<section id="example-{slug(scenario)}"><h3>{escape(scenario)}</h3><p>{escape(DESCRIPTIONS[scenario])}</p>')
        parts.append(chart(fig))
        if targeted:
            parts.append(f'<p>The two raw signals have different baseline levels. Compare each with its own dashed baseline; the bottom panel confirms that only the targeted series receives the offer. Shaded dates apply to the affected series.</p>')
        elif scenario=='Mixed events':
            parts.append('<p>The upper example is App Store · Germany, which has overlapping gradual-trend, rise/recovery and LOB-offer injections. Other rows in the heatmap receive different combinations. Hover to inspect a signal and date.</p>')
        parts.append('<p class="example-back"><a href="#anomaly-examples">Back to anomaly examples</a></p></section>')
    parts.append('</section>')
    return ''.join(parts)
