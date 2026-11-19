"""Aggregate-only E3 figures; no source images or masks are embedded."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main(folder,out):
    assert not out.exists();out.mkdir(parents=True);methods=rows(folder/'method_summary.csv');risk=next(r for r in rows(folder/'risk_summary.csv') if r['scope']=='opportunity');s=json.loads((folder/'summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(14,4.7));labels=['S','MAP int','Decay .5','V subpix','GV subpix','V2','V2 visual']
    axes[0].bar(np.arange(7),[100*float(r['source_macro_IoU']) for r in methods],color=['#777777']*5+['#236d98','#529c91']);axes[0].set_xticks(np.arange(7));axes[0].set_xticklabels(labels,rotation=45,ha='right',fontsize=8);axes[0].set_ylim(30.48,30.61);axes[0].set_ylabel('Equal-source foreground IoU (%)');axes[0].set_title('Three training sources (zoomed axis)');axes[0].grid(axis='y',alpha=.2)
    values=np.array([float(risk[k+'_mean']) for k in ['variance','conflict','unknown']]);values=100*values/values.sum();axes[1].bar(np.arange(3),values,color=['#5b8c5a','#c88036','#8b739e']);axes[1].set_xticks(range(3));axes[1].set_xticklabels(['Candidate\nvariance','Current\nconflict','Unknown\nmass']);axes[1].set_ylabel('Share of mean risk (%)');axes[1].set_title('At oracle-opportunity pixels');
    for j,v in enumerate(values):axes[1].text(j,v+.7,'%.1f%%'%v,ha='center',fontsize=9)
    n=s['oracle_opportunity'];recovered=s['v2_opportunity_recovered'];axes[2].bar([0,1],[100*recovered/n,100*(n-recovered)/n],color=['#397c53','#9b6670']);axes[2].set_xticks([0,1]);axes[2].set_xticklabels(['Recovered\nby V2','Still\nmissed']);axes[2].set_ylabel('Oracle-opportunity pixels (%)');axes[2].set_title('%d observed opportunities'%n)
    for j,v in enumerate([recovered,n-recovered]):axes[2].text(j,100*v/n+1,'%d'%v,ha='center',fontsize=9)
    fig.suptitle('E3 diagnostic: fixed checkpoint, 12 same-image training-source sequences\nNo final-test, real-video or causal-ablation conclusion',fontsize=12);fig.tight_layout(rect=[0,0,1,.88]);fig.savefig(str(out/'v2_screening_summary.png'),dpi=180,bbox_inches='tight');fig.savefig(str(out/'v2_screening_summary.svg'),bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.evaluation.resolve(),a.out.resolve())
