"""Render aggregate-only figures from the oracle ablation CSVs."""
import argparse
import csv
import os
from pathlib import Path

import numpy as np


def read(path):
    with path.open(newline="",encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save(fig,target):
    for ext in (".pdf",".svg",".png"):
        fig.savefig(str(target.with_suffix(ext)),dpi=300,facecolor="white")


def plot(source,out):
    os.environ.setdefault("MPLCONFIGDIR",str(out/"mpl_config"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
        "axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none"})
    out.mkdir(exist_ok=True)
    entries=read(source/"aggregate_paired.csv")
    models=(("small_unet","Small U-Net"),("mobilenet_deeplab","MobileNetV2-DeepLabv3+"))
    tests=(("natural_test","Natural, n=28"),("artificial_test","Artificial, n=16"))
    contrasts=(("equal_est","equal_zero","Equal: estimated - zero","#0072B2","o"),
               ("equal_exact","equal_est","Equal: exact - estimated","#0072B2","s"),
               ("conservative_est","conservative_zero","Conservative: estimated - zero","#009E73","o"),
               ("conservative_exact","conservative_est","Conservative: exact - estimated","#009E73","s"))
    fig,axes=plt.subplots(2,2,figsize=(9.7,5.3),sharey=True,sharex=True)
    for i,(mod,model) in enumerate(models):
        for j,(ts,test) in enumerate(tests):
            ax=axes[i,j]
            ax.axvline(0,ls="--",color="#888888",lw=.9)
            ax.set_title(model+" | "+test,fontsize=9)
            choose=[r for r in entries if (r["model"],r["stratum"],r["profile"],r["scope"])==(mod,ts,"all","full")]
            for k,(method,ref,label,color,marker) in enumerate(contrasts):
                row=next(x for x in choose if (x["method"],x["reference"])==(method,ref))
                center,low,high=[100*float(row[c]) for c in ("mean_delta","low95","high95")]
                ax.errorbar(center,k,xerr=np.array([[center-low],[high-center]]),color=color,marker=marker,
                            markersize=4.5,capsize=3,elinewidth=1.3)
            ax.set_yticks(range(4))
            ax.set_yticklabels([r[2] for r in contrasts],fontsize=8.5)
            ax.set_ylim(3.6,-.6)
            ax.set_axisbelow(True)
            ax.grid(axis="x",alpha=.18)
    fig.text(.60,.048,"Paired source-macro IoU difference (percentage points)",ha="center")
    fig.subplots_adjust(left=.25,right=.98,bottom=.145,top=.90,hspace=.40,wspace=.15)
    save(fig,out/"oracle_alignment_effect")
    plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(9.2,5.7),sharey=True,sharex=True)
    profiles=("clean","blur_3px","combined")
    for i,(mod,model) in enumerate(models):
        for j,(ts,test) in enumerate(tests):
            ax=axes[i,j]
            ax.axhline(0,ls="--",color="#888888",lw=.8)
            subset=[r for r in entries if (r["model"],r["stratum"],r["scope"])==(mod,ts,"full")]
            for m,ref,d,c,label in (("equal_exact","equal_est",-.10,"#0072B2","Equal"),
                ("conservative_exact","conservative_est",.10,"#009E73","Conservative")):
                selected=[next(x for x in subset if (x["profile"],x["method"],x["reference"])==(p,m,ref)) for p in profiles]
                y,low,high=[100*np.array([float(x[k]) for x in selected]) for k in ("mean_delta","low95","high95")]
                ax.errorbar(np.arange(3)+d,y,yerr=np.stack((y-low,high-y)),color=c,fmt="o",markersize=4,elinewidth=1.2,
                            capsize=3,label=label)
            ax.set_title(model+" | "+test,fontsize=9)
            ax.set_xticks(range(3))
            ax.set_xticklabels(["Clean","Blur 3 px","Combined"])
            ax.grid(axis="y",alpha=.18)
            ax.set_axisbelow(True)
    axes[0,0].legend(loc="best",frameon=False,fontsize=8)
    fig.text(.032,.49,"Exact minus estimated IoU (percentage points)",rotation=90,ha="center",va="center")
    fig.subplots_adjust(left=.14,right=.98,top=.91,bottom=.12,wspace=.20,hspace=.38)
    save(fig,out/"oracle_by_degradation")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    plot(args.data,args.out or args.data/"figures")
