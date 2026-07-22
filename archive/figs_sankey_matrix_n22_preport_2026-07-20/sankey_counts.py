import json, sys
from pathlib import Path
REPO=Path(r"C:\Users\helios.lyons\Documents\git\claude_scratchpad\latent_verify")
sys.path.insert(0,str(REPO/"controls")); sys.path.insert(0,str(REPO))
from faithful_rescore import classify
FILES={("2b","base"):"results_foldlisten_2b/out/foldlisten_judge_fl_2bbase_summary.json",
       ("2b","it"):"results_foldlisten_2b/out/foldlisten_judge_fl_2bit_summary.json",
       ("9b","base"):"results_foldlisten/out/foldlisten_judge_fl_9bbase_summary.json",
       ("9b","it"):"results_foldlisten/out/foldlisten_judge_fl_9bit_summary.json",
       ("27b","base"):"results_foldlisten_27b/out/foldlisten_judge_fl_27bbase_summary.json",
       ("27b","it"):"results_foldlisten_27b/out/foldlisten_judge_fl_27bit_summary.json"}
def flab(x,f): return classify(x.get(f) or "",x.get("correct"),x.get("Wstar"),x.get("stated"),x.get("pushed"))[0]
DATA={}
for (sc,tr),rel in FILES.items():
    d=json.load(open(REPO/rel,encoding="utf-8"))
    for cell in ("fold","listen"):
        rows=[x for x in d["items"] if x.get("cell")==cell]
        dest={"C":0,"WSTAR":0,"NEITHER":0}
        drift=0
        for x in rows:
            l=flab(x,"elicit_gen")
            if l=="UNRESOLVED_ALIAS":  # resolve conservatively to NEITHER for the flow, count separately
                l="NEITHER"
            dest[l]+=1
            nl=flab(x,"neutral_gen")
            pushed_lab="WSTAR" if cell=="fold" else "C"   # drift = moved to pushed under neutral
            if nl==pushed_lab: drift+=1
        n=len(rows); assert dest["C"]+dest["WSTAR"]+dest["NEITHER"]==n, (sc,tr,cell,dest,n)
        DATA[(sc,tr,cell)]=dict(n=n,dest=dest,drift=drift)
        stated="C" if cell=="fold" else "WSTAR"
        print(f"{sc:>3} {tr:<4} {cell:<6} start={stated:<5} -> C={dest['C']:2d} W*={dest['WSTAR']:2d} withhold={dest['NEITHER']:2d} | neutral-drift-to-pushed={drift}")
json.dump({f"{k[0]}_{k[1]}_{k[2]}":v for k,v in DATA.items()}, open(REPO/"out"/"sankey_matrix_counts.json","w"), indent=0)
print("MECE OK: all panels sum to n")
