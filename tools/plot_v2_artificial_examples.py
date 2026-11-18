"""Render three artificial E2 examples, including the specified failure case."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat


def main(run,out):
    assert not out.exists();out.mkdir(parents=True);data=loadmat(str(run/'controlled_examples.mat'),squeeze_me=True,struct_as_record=False)['examples']
    examples=['recover_near_threshold','ambiguous_wrong_single_motion','confident_current_false_positive_failure']
    labels=['Near-threshold recovery','Ambiguous motion / wrong hard choice','Failure: confident current false positive']
    fields=['target','single','single_motion_fixed','v2'];titles=['Artificial ground truth','Current single frame','Hard single-motion fusion','Candidate v2']
    fig,axes=plt.subplots(3,4,figsize=(13,7.2))
    for row,(name,label) in enumerate(zip(examples,labels)):
        item=getattr(data,name)
        for col,field in enumerate(fields):
            a=getattr(item,field);view=a[74:105,89:145];axes[row,col].imshow(view,vmin=0,vmax=1,cmap='magma',interpolation='nearest',aspect='auto');axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(titles[col]+('\nA. Near-threshold recovery' if col==0 else ''),fontsize=10)
            elif col==0:axes[row,col].set_title(['','B. Ambiguous motion','C. Confident current false positive'][row],fontsize=9)
            if field!='target':
                mask=a>=.5;target=item.target.astype(bool);tp=(mask&target).sum();fp=(mask&~target).sum();fn=(~mask&target).sum();iou=tp/(tp+fp+fn) if tp+fp+fn else 1.
                axes[row,col].set_xlabel('Toy IoU %.2f | FP %d | FN %d'%(iou,fp,fn),fontsize=9)
    fig.suptitle('Artificial mechanism examples: recovery, damage control and a failure\nCropped view; full-frame toy counts; no rail dataset performance claim',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.91]);fig.savefig(str(out/'v2_artificial_examples.png'),dpi=180,bbox_inches='tight');fig.savefig(str(out/'v2_artificial_examples.svg'),bbox_inches='tight');plt.close(fig)
    (out/'plot_note.json').write_text(json.dumps(dict(scope='artificial mechanism examples, not rail accuracy',matplotlib_version=matplotlib.__version__),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.run.resolve(),a.out.resolve())
