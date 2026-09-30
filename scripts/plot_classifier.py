"""Visualize the trained v2 miss-predictor: data boundary, learned curve, calibration, ROC."""
import glob, pickle, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patrol.models.mlp import MLP
from patrol.controllers.profile import V2_FEATURE_NAMES, V2_EST_UTIL_IDX

C = {"data":"#0072B2","model":"#D55E00","ref":"#555555","ok":"#009E73"}
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","axes.grid":True,
    "grid.color":"#e8e8e8","axes.edgecolor":"#ccc","axes.spines.top":False,
    "axes.spines.right":False,"font.size":10,"axes.titleweight":"bold","legend.frameon":False})

def load_data():
    d=[]
    for f in glob.glob("dataset/profile/*.pkl"): d+=pickle.load(open(f,"rb"))
    X=np.array([s["x_state"] for s in d],dtype=np.float32); y=np.array([s["y_miss"] for s in d],dtype=np.float32)
    return X,y

def load_model(p="patrol/models/checkpoints/model.pt"):
    ck=torch.load(p,map_location="cpu",weights_only=False)
    m=MLP(ck["input_dim"],ck["hidden"],ck.get("dropout",0.0)); m.load_state_dict(ck["model_state"]); m.eval()
    return m,ck

def predict(m,ck,X):
    Xn=(X-ck["mean"])/ck["std"]
    with torch.no_grad(): p=torch.sigmoid(m(torch.from_numpy(Xn.astype(np.float32)))/ck.get("temperature",1.0)).numpy()
    return p

def auc(y,p):
    o=np.argsort(p); y=y[o]; npos=y.sum(); nneg=len(y)-npos
    if npos==0 or nneg==0: return float("nan")
    r=np.arange(1,len(y)+1); return float((r[y==1].sum()-npos*(npos+1)/2)/(npos*nneg))

def main():
    X,y=load_data(); m,ck=load_model(); p=predict(m,ck,X)
    eu=X[:,V2_EST_UTIL_IDX]; A=auc(y,p)
    fig,ax=plt.subplots(1,3,figsize=(16,4.6))

    # (1) data miss-rate + model miss-prob vs est_util
    bins=np.linspace(0,max(1.6,eu.max()),17); ctr=(bins[:-1]+bins[1:])/2
    drate=[y[(eu>=lo)&(eu<hi)].mean() if ((eu>=lo)&(eu<hi)).sum() else np.nan for lo,hi in zip(bins[:-1],bins[1:])]
    mrate=[p[(eu>=lo)&(eu<hi)].mean() if ((eu>=lo)&(eu<hi)).sum() else np.nan for lo,hi in zip(bins[:-1],bins[1:])]
    ax[0].plot(ctr,drate,"o-",color=C["data"],lw=2,ms=5,label="empirical miss-rate")
    ax[0].plot(ctr,mrate,"s--",color=C["model"],lw=2,ms=4,label="model miss-prob")
    ax[0].axvline(1.0,color=C["ref"],ls=":",lw=1.2,label="util = 1.0")
    ax[0].set_title("Miss vs estimated utilization"); ax[0].set_xlabel("est_util  ($\\Sigma\\,C_i(q_i)/T_i$)")
    ax[0].set_ylabel("miss probability"); ax[0].set_ylim(-0.03,1.03); ax[0].legend(loc="lower right")

    # (2) calibration (reliability)
    b=np.linspace(0,1,11); cc=(b[:-1]+b[1:])/2
    obs=[y[(p>=lo)&(p<hi)].mean() if ((p>=lo)&(p<hi)).sum() else np.nan for lo,hi in zip(b[:-1],b[1:])]
    ax[1].plot([0,1],[0,1],color=C["ref"],ls="--",lw=1.2,label="perfect")
    ax[1].plot(cc,obs,"o-",color=C["ok"],lw=2,ms=5,label="v2 model")
    ax[1].set_title("Calibration"); ax[1].set_xlabel("predicted miss-prob"); ax[1].set_ylabel("observed miss-rate")
    ax[1].set_xlim(0,1); ax[1].set_ylim(0,1); ax[1].legend(loc="upper left")

    # (3) ROC
    th=np.unique(p); tpr=[];fpr=[]; P=y.sum(); N=len(y)-P
    for t in np.sort(th)[::-1]:
        pred=p>=t; tpr.append((pred&(y==1)).sum()/max(1,P)); fpr.append((pred&(y==0)).sum()/max(1,N))
    ax[2].plot([0,1],[0,1],color=C["ref"],ls="--",lw=1.2)
    ax[2].plot(fpr,tpr,color=C["model"],lw=2.2,label=f"ROC (AUC={A:.4f})")
    ax[2].set_title("ROC"); ax[2].set_xlabel("false-positive rate"); ax[2].set_ylabel("true-positive rate")
    ax[2].set_xlim(0,1); ax[2].set_ylim(0,1.02); ax[2].legend(loc="lower right")

    fig.suptitle(f"v2 miss-predictor  —  n={len(y)}  miss_rate={y.mean():.3f}  AUC={A:.4f}  "
                 f"(demand-based est_util, known period)", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out=sys.argv[1] if len(sys.argv)>1 else "outputs/sample/figures/classifier_v2.png"
    Path(out).parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=150,bbox_inches="tight")
    print("wrote",out)

if __name__=="__main__": main()
