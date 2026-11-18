"""Standard exportable C5 figures from accepted summary tables, no data images."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def plot(summary,out):
    assert not out.exists();out.mkdir(parents=True)
    contrasts=[r for r in rows(summary/'paired_contrasts.csv') if r['metric']=='IoU' and r['threshold_kind']=='primary']
    stress=['bias_m6','bias_m2','bias_p2','bias_p6','alternating_1','alternating_3','all_missing']
    labels=['Bias -6','Bias -2','Bias +2','Bias +6','Alternate A=1','Alternate A=3','All missing']
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    for ax,(a,b) in zip(axes,[('P','V'),('GVP','GV'),('GVP','GC')]):
        r=[next(r for r in contrasts if r['scenario_a']==s and r['scenario_b']==s and r['method_a']==a and r['method_b']==b) for s in stress]
        v=np.array([float(x['point']) for x in r])*100;lo=np.array([float(x['CI95_low']) for x in r])*100;hi=np.array([float(x['CI95_high']) for x in r])*100
        for j in range(7):ax.plot([lo[j],hi[j]],[j,j],color='#17658b',linewidth=2)
        ax.plot(v,np.arange(7),'o',color='#17658b');ax.axvline(0,color='.5',linewidth=1)
        ax.set_yticks(np.arange(7));ax.set_yticklabels(labels);ax.invert_yaxis();ax.set_xlabel('Source-macro IoU difference (pp)');ax.set_title(a+' minus '+b);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Observation stress: fixed segmenter and threshold 0.5\nPaired 15-source descriptive 95% intervals; no multiplicity correction',fontsize=12);fig.tight_layout(rect=[0,0,1,.88]);save(fig,out,'observation_stress')
    ab=[r for r in contrasts if r['scenario_b']=='baseline' and r['frame_set']=='primary_2_9']
    label=[]
    for r in ab:
        label.append(r['scenario_a'].replace('_',' ')+' / '+r['method_a'])
    v=np.array([float(r['point']) for r in ab])*100;lo=np.array([float(r['CI95_low']) for r in ab])*100;hi=np.array([float(r['CI95_high']) for r in ab])*100
    fig,ax=plt.subplots(figsize=(9,6))
    for j in range(len(ab)):ax.plot([lo[j],hi[j]],[j,j],color='#9a5e16',linewidth=2)
    ax.plot(v,np.arange(len(ab)),'o',color='#9a5e16');ax.axvline(0,color='.5',linewidth=1)
    ax.set_yticks(np.arange(len(ab)));ax.set_yticklabels(label);ax.invert_yaxis();ax.set_xlabel('IoU difference from own W3 clean parent (pp)');ax.grid(axis='x',alpha=.2)
    ax.set_title('All planned window / gate / alignment / recurrence ablations\nThreshold 0.5; descriptive paired-source intervals');fig.tight_layout();save(fig,out,'component_ablation')
    pollution=[r for r in rows(summary/'pollution_by_frame.csv') if r['threshold_kind']=='primary']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for m in ['S','V','GV','GVP','R','R_unbounded']:
        r=sorted([r for r in pollution if r['method']==m],key=lambda r:int(r['frame']));frames=[int(x['frame']) for x in r]
        axes[0].plot(frames,[float(x['mean_probability_delta']) for x in r],marker='o',label=m)
        axes[1].plot(frames,[int(x['prediction_flip_pixels'])/int(x['method_frames']) for x in r],marker='o',label=m)
    axes[0].set_yscale('symlog',linthreshy=1e-10);axes[0].set_ylabel('Mean |polluted probability - own clean output|')
    axes[1].set_yscale('symlog',linthreshy=.01);axes[1].set_ylabel('Prediction flips / method-frame')
    for ax in axes:
        ax.axvline(3,color='.4',linestyle='--',linewidth=1);ax.axvline(6,color='.6',linestyle=':',linewidth=1);ax.set_xlabel('Frame (fault injected only at frame 3)');ax.set_xticks(range(1,10));ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.suptitle('Fault injection: p3 := 1 - p3; subsequent raw inputs unchanged\nControlled sequence counterfactual, not natural error frequency',fontsize=12);fig.tight_layout(rect=[0,0,1,.88]);save(fig,out,'history_contamination')
    report=dict(status='PLOTS_WRITTEN_PENDING_VISUAL_QA',source_tables={n:sha(summary/n) for n in ['paired_contrasts.csv','pollution_by_frame.csv']},
        figures={p.name:sha(p) for p in sorted(out.glob('*')) if p.suffix in ['.png','.svg']},matplotlib_version=matplotlib.__version__)
    (out/'plot_manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


def save(fig,out,name):
    fig.savefig(str(out/(name+'.png')),dpi=180,bbox_inches='tight');fig.savefig(str(out/(name+'.svg')),bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();plot(a.summary.resolve(),a.out.resolve())
