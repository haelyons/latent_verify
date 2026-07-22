import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MP
REPO=Path(r"C:\Users\helios.lyons\Documents\git\claude_scratchpad\latent_verify")
sys.path.insert(0,str(REPO/"controls")); sys.path.insert(0,str(REPO))
from faithful_rescore import classify
OUT=Path(r"C:\tmp\claude\C--Users-helios-lyons-Documents-git-claude-scratchpad-latent-verify\294b8066-05ca-43ed-8a5b-8b9b1f0d3dd7\scratchpad")
BLUE,RED,GREY,INK,MUTED="#009E73","#CC3311","#b0b0ab","#1a1a1a","#6e6e6a"  # "BLUE" now Okabe-Ito bluish-green (CVD-safe vs red)
HUE={"C":BLUE,"WSTAR":RED,"NEITHER":GREY}
# training -> weight: base = muted, it = bold
WT={"base":dict(node_a=0.60,edge_lw=1.0,rib_a=0.44),  # less transparent than before, still < it
    "it":  dict(node_a=1.0,edge_lw=0.0,rib_a=0.62)}    # bold
plt.rcParams.update({"font.size":10,"figure.facecolor":"white","axes.facecolor":"white","svg.fonttype":"none"})

def items(p):
    d=json.load(open(REPO/p,encoding="utf-8")); return d["result"]["items"] if "result" in d else d["items"]
def flab(x,f):
    l=classify(x.get(f) or "",x.get("correct"),x.get("Wstar"),x.get("stated"),x.get("pushed"))[0]
    return "NEITHER" if l=="UNRESOLVED_ALIAS" else l
FILES={("2b","base"):"results_foldlisten_2b/out/foldlisten_judge_fl_2bbase_summary.json",
 ("2b","it"):"results_foldlisten_2b/out/foldlisten_judge_fl_2bit_summary.json",
 ("9b","base"):"results_foldlisten/out/foldlisten_judge_fl_9bbase_summary.json",
 ("9b","it"):"results_foldlisten/out/foldlisten_judge_fl_9bit_summary.json",
 ("27b","base"):"results_foldlisten_27b/out/foldlisten_judge_fl_27bbase_summary.json",
 ("27b","it"):"results_foldlisten_27b/out/foldlisten_judge_fl_27bit_summary.json"}
def panel_data(sc,tr,cell):
    rows=[x for x in items(FILES[(sc,tr)]) if x.get("cell")==cell]
    dest={"C":0,"WSTAR":0,"NEITHER":0}
    for x in rows: dest[flab(x,"elicit_gen")]+=1
    return dest,len(rows)

def draw_node(ax,x,y,w,h,hue,tr):
    wt=WT[tr]
    ax.add_patch(plt.Rectangle((x,y),w,h,facecolor=hue,alpha=wt["node_a"],
                 edgecolor=hue,linewidth=wt["edge_lw"],zorder=3))

def sankey(ax,start_key,dest,n,tr,small=False):
    wt=WT[tr]; gap=max(0.4,n*0.03); W=0.16; x0,x1=0.0,1.0
    draw_node(ax,x0,0,W,n,HUE[start_key],tr)
    order=["C","WSTAR","NEITHER"]; ytop=n; rpos={}
    for k in order:
        h=dest[k]; rpos[k]=(ytop-h,ytop); ytop-=h+(gap if h>0 else 0)
    lcur=n
    for k in order:
        v=dest[k]
        if not v: continue
        la,lb=lcur-v,lcur; lcur-=v; ra,rb=rpos[k]
        xa,xb=x0+W,x1-W; mid=(xa+xb)/2
        verts=[(xa,lb),(mid,lb),(mid,rb),(xb,rb),(xb,ra),(mid,ra),(mid,la),(xa,la),(xa,lb)]
        codes=[MP.MOVETO,MP.CURVE4,MP.CURVE4,MP.CURVE4,MP.LINETO,MP.CURVE4,MP.CURVE4,MP.CURVE4,MP.CLOSEPOLY]
        ax.add_patch(PathPatch(MP(verts,codes),facecolor=HUE[k],alpha=wt["rib_a"],edgecolor="none",zorder=2))
    for k in order:
        ra,rb=rpos[k]
        if rb-ra>0:
            draw_node(ax,x1-W,ra,W,rb-ra,HUE[k],tr)
            ax.text(x1+0.03,(ra+rb)/2,str(dest[k]),va="center",ha="left",
                    fontsize=8 if small else 9.5,color=INK,fontweight="bold")
    ax.set_xlim(-0.30,1.34); ax.set_ylim(-1,n+1); ax.axis("off")

def legend(fig,y):
    h=[plt.Rectangle((0,0),1,1,color=c) for c in (BLUE,RED,GREY)]
    fig.legend(h,["correct (C)","wrong (W*)","withholds"],loc="lower center",ncol=3,
               frameon=False,fontsize=10,bbox_to_anchor=(0.5,y))

# ---------------- MATRIX ----------------
SC=["2b","9b","27b"]; ROWS=[("fold","base"),("fold","it"),("listen","base"),("listen","it")]
fig,axes=plt.subplots(4,3,figsize=(11.5,12))
for r,(cell,tr) in enumerate(ROWS):
    for c,sc in enumerate(SC):
        dest,n=panel_data(sc,tr,cell)
        sankey(axes[r][c],"C" if cell=="fold" else "WSTAR",dest,n,tr,small=True)
    if r==0:
        for c,sc in enumerate(SC):
            axes[0][c].text(0.5,1.12,sc,transform=axes[0][c].transAxes,ha="center",fontsize=14,fontweight="bold")
fig.canvas.draw()
# per-row training label + row-group FOLD/LISTEN spanning 2 rows
for r,(cell,tr) in enumerate(ROWS):
    p=axes[r][0].get_position()
    fig.text(0.075,(p.y0+p.y1)/2,("base" if tr=="base" else "-it"),ha="right",va="center",fontsize=10.5,color=INK)
for grp,(r0,r1),start in [("FOLD",(0,1),"start: C"),("LISTEN",(2,3),"start: W*")]:
    pa=axes[r0][0].get_position(); pb=axes[r1][0].get_position()
    ymid=(pa.y1+pb.y0)/2
    fig.text(0.018,ymid,grp,ha="left",va="center",fontsize=12.5,fontweight="bold",rotation=90,color=INK)
    fig.text(0.040,ymid,start,ha="left",va="center",fontsize=8.5,rotation=90,color=MUTED)
fig.suptitle("Where each answer lands after pushback   (n = 22 per cell)",fontsize=14,fontweight="bold",y=0.975)
legend(fig,0.012)
fig.tight_layout(rect=(0.09,0.035,0.99,0.95))
fig.savefig(OUT/"fig_sankey_matrix.png",dpi=170); plt.close(fig)

# ---------------- HERO 9b (2x2) ----------------
fig,axes=plt.subplots(2,2,figsize=(8.6,7))
grid=[[("fold","base"),("fold","it")],[("listen","base"),("listen","it")]]
for i in range(2):
    for j in range(2):
        cell,tr=grid[i][j]; dest,n=panel_data("9b",tr,cell)
        sankey(axes[i][j],"C" if cell=="fold" else "WSTAR",dest,n,tr)
for j,tr in enumerate(["base","-it"]):
    axes[0][j].text(0.5,1.12,tr,transform=axes[0][j].transAxes,ha="center",fontsize=13,fontweight="bold")
fig.canvas.draw()
for i,(grp,start) in enumerate([("FOLD","start: C"),("LISTEN","start: W*")]):
    p=axes[i][0].get_position()
    fig.text(0.028,(p.y0+p.y1)/2,grp,ha="center",va="center",fontsize=12.5,fontweight="bold",rotation=90,color=INK)
    fig.text(0.052,(p.y0+p.y1)/2,start,ha="center",va="center",fontsize=8.5,rotation=90,color=MUTED)
fig.suptitle("Gemma-2-9b: where each answer lands after pushback",fontsize=13.5,fontweight="bold",y=0.975)
legend(fig,0.015)
fig.tight_layout(rect=(0.07,0.05,0.99,0.95))
fig.savefig(OUT/"fig_sankey_9b.png",dpi=190); plt.close(fig)
print("done: matrix + hero rebuilt (opacity=training, no subtitles, no dots)")
