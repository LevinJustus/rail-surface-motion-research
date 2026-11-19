"""Aggregate-only E4 figure; no dataset images or masks are rendered."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main(folder,out):
    assert not out.exists();out.mkdir(parents=True)
    with (folder/'method_summary.csv').open(encoding='utf-8-sig',newline='') as f:rows={r['method']:r for r in csv.DictReader(f)}
    summary=json.loads((folder/'summary.json').read_text());assert summary['status']=='PASS_E4_SINGLE_FACTOR_DIAGNOSTIC'
    names=['S','V2','V2_no_conflict','DECAY05_integer','GV_subpixel'];labels=['Single frame','Original v2','No conflict (E4)','Fixed decay 0.5','Subpixel gated']
    fig,axes=plt.subplots(1,3,figsize=(15,4.8));colors=['#64748b','#2563eb','#d97706','#059669','#7c3aed']
    for ax,key,title in [(axes[0],'source_macro_IoU','Equal-source macro IoU'),(axes[1],'IoU','Pooled pixel IoU')]:
        values=[float(rows[m][key])*100 for m in names];y=np.arange(len(names));ax.scatter(values,y,c=colors,s=58)
        ax.set_yticks(y);ax.set_yticklabels(labels);ax.invert_yaxis();ax.set_xlim(min(values)-.018,max(values)+.027)
        for yi,v in zip(y,values):ax.text(v+.002,yi,'%.4f'%v,va='center',fontsize=9)
        ax.set_xlabel('IoU (%) - zoomed axis');ax.set_title(title);ax.grid(axis='x',alpha=.25);ax.set_axisbelow(True)
    pair=summary['changes_relative_to_v2'];values=[pair[k] for k in ['FN_to_TP','TP_to_FN','FP_to_TN','TN_to_FP']];labels2=['Corrected FN','New FN','Corrected FP','New FP'];axes[2].barh(np.arange(4),values,color=['#059669','#dc2626','#059669','#dc2626']);axes[2].set_yticks(np.arange(4));axes[2].set_yticklabels(labels2);axes[2].invert_yaxis();axes[2].set_xlim(0,max(values)*1.22)
    for yi,v in enumerate(values):axes[2].text(v+80,yi,str(v),va='center',fontsize=9)
    axes[2].set_xlabel('Correlated pixel observations');axes[2].set_title('E4 changes relative to original v2')
    fig.suptitle('E4 single-factor diagnostic: 3 training sources / 12 synthetic sequences',fontsize=13)
    fig.text(.5,.02,'Historical opportunity recovery: 10.08% -> 23.14%. Small mixed gains; no independent-test or real-video claim.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.07,1,.92]);fig.savefig(str(out/'v2_conflict_ablation_summary.png'),dpi=170);fig.savefig(str(out/'v2_conflict_ablation_summary.svg'));plt.close(fig)
    print('E4 aggregate PNG/SVG saved')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.evaluation,a.out)
