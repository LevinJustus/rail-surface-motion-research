"""Plot fixed C4 source summaries; no data selection or metric fitting."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def plot(folder,out):
    assert not out.exists(),'No overwrite';out.mkdir(parents=True)
    methods=['S','U','V','P','GV','GVP','GC','R']
    data=[r for r in rows(folder/'method_summary.csv') if r['frame_set']=='primary_2_9' and r['threshold_kind']=='primary']
    by={r['method']:r for r in data};seeds=[r for r in rows(folder/'per_seed_summary.csv') if r['frame_set']=='primary_2_9' and r['threshold_kind']=='primary']
    contrasts=[r for r in rows(folder/'paired_contrasts.csv') if r['metric']=='IoU']
    fig,axes=plt.subplots(1,2,figsize=(12.6,4.8));ax=axes[0]
    ax.bar(np.arange(8),[float(by[m]['source_macro_IoU_mean'])*100 for m in methods],width=.68,color='#3975a8',alpha=.75)
    for j,seed in enumerate([20261114,20261115,20261116]):
        values={r['method']:float(r['source_macro_IoU'])*100 for r in seeds if int(r['seed'])==seed}
        ax.plot(np.arange(8)+(j-1)*.13,[values[m] for m in methods],'o',markersize=4,color=['#101820','#b84432','#d29b21'][j],label=str(seed))
    ax.set_xticks(np.arange(8));ax.set_xticklabels(methods);ax.set_ylabel('Source-macro foreground IoU (%)')
    ax.set_title('Same-sequence validation at threshold 0.5');ax.legend(title='Initialization seed',fontsize=8,loc='best')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax=axes[1]
    for j,r in enumerate(contrasts):
        point=float(r['point'])*100;low=float(r['CI95_low'])*100;high=float(r['CI95_high'])*100
        color='#25734d' if low>0 else '#a83f35' if high<0 else '#555f68'
        ax.plot([low,high],[j,j],'-',color=color,linewidth=2);ax.plot(point,j,'o',color=color,markersize=5)
    ax.set_yticks(np.arange(len(contrasts)));ax.set_yticklabels([r['contrast'] for r in contrasts]);ax.invert_yaxis();ax.axvline(0,color='#777777',linestyle='--',linewidth=1)
    ax.set_xlabel('Difference in source-macro IoU (percentage points)');ax.set_title('Paired source bootstrap: descriptive 95% intervals')
    ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Controlled sequences: 15 source proxy groups, 3 fixed seeds, frames 2-9',fontsize=12)
    fig.text(.5,.015,'Intervals are unadjusted; simulated sequences and proxy groups do not establish real-video generalization.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(str(out/'fusion_validation_summary.png'),dpi=180);fig.savefig(str(out/'fusion_validation_summary.svg'));plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plot(a.summary_dir.resolve(),a.out.resolve())
