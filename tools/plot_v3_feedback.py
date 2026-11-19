"""Aggregate v3 outcomes; real-source derivatives and artificial cases stay separate."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main(batch,challenge,out):
    assert not out.exists();out.mkdir(parents=True)
    with (batch/'method_summary.csv').open() as f:metrics={r['method']:r for r in csv.DictReader(f)}
    with (challenge/'motion_metrics.csv').open() as f:motion=[r for r in csv.DictReader(f) if r['case']=='ambiguous_texture']
    fig,axes=plt.subplots(1,3,figsize=(15,4.8));names=['S','DECAY05_integer','V2_no_conflict','E_base_strong','D_feedback_strong'];labels=['Single frame','Fixed decay','E4 no conflict','v3 no feedback','v3 feedback']
    values=[float(metrics[m]['IoU'])*100 for m in names];axes[0].scatter(values,range(5),color=['#64748b','#059669','#d97706','#2563eb','#7c3aed'],s=55);axes[0].set_yticks(range(5));axes[0].set_yticklabels(labels);axes[0].invert_yaxis();axes[0].set_xlim(min(values)-.014,max(values)+.025)
    for y,v in enumerate(values):axes[0].text(v+.0015,y,'%.4f'%v,va='center',fontsize=9)
    axes[0].set_xlabel('Pooled IoU (%) - zoomed axis');axes[0].set_title('12 synthetic rail-source sequences')
    x=np.arange(3);width=.22
    for i,(name,color) in enumerate([('base','#2563eb'),('feedback','#d97706'),('uniform','#059669')]):axes[1].bar(x+(i-1)*width,[float(r[name+'_EPE']) for r in motion],width,label=name,color=color)
    axes[1].set_xticks(x);axes[1].set_xticklabels(['Seed 19','Seed 20','Seed 21']);axes[1].set_ylabel('Translation error (px)');axes[1].set_title('Artificial repeated-texture challenge');axes[1].legend(fontsize=8)
    axes[2].bar(x,[float(r['best_candidate_EPE']) for r in motion],color='#64748b');axes[2].set_xticks(x);axes[2].set_xticklabels(['Seed 19','Seed 20','Seed 21']);axes[2].set_ylabel('Best retained candidate error (px)');axes[2].set_title('Five candidates can miss true motion')
    for i,r in enumerate(motion):axes[2].text(i,float(r['best_candidate_EPE'])+.2,'%.3f'%float(r['best_candidate_EPE']),ha='center',fontsize=9)
    axes[2].set_ylim(0,14);fig.suptitle('v3 prototype: feedback implemented; rail-source benefit not demonstrated',fontsize=13)
    fig.text(.5,.018,'Artificial cases use manufactured probabilities. Three seeds are repeated across three cases; they are not nine independent scenes.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.92]);fig.savefig(str(out/'v3_feedback_summary.png'),dpi=165);fig.savefig(str(out/'v3_feedback_summary.svg'));plt.close(fig)
    print('V3 aggregate figure saved')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);p.add_argument('--challenge',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.batch,a.challenge,a.out)
