#!/usr/bin/env python3
"""Independent check: does the NEW run's PUSH (elicited) column equal the COMMITTED twin's?
The push column is the numerator of every push_attribution delta, so if it moved the delta is
not an additive extension of the published cell."""
import json, os
from collections import Counter
ROOT='/home/hal/dev/interp/latent_verify'
NEW={'2b-base':'results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_2bbase_ext2_summary.json',
     '9b-base':'results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json',
     '2b-it':'results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_2bit_ext2_summary.json',
     '9b-it':'results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bit_ext2_summary.json',
     '27b-base':'results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27bbase_ext2_summary.json',
     '27b-it':'results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27bit_ext2_summary.json'}
OLD={'2b-base':'results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_2bbase_ext2_summary.json',
     '9b-base':'results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json',
     '2b-it':'results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_2bit_ext2_summary.json',
     '9b-it':'results_foldlisten_r2/out/foldlisten_judge_fl_9bit_ext2_summary.json',
     '27b-base':'results_foldlisten_ext2_27b/out/foldlisten_judge_fl_27bbase_ext2_summary.json',
     '27b-it':'results_foldlisten_ext2_27b/out/foldlisten_judge_fl_27bit_ext2_summary.json'}
F2C={"WSTAR":"wrong","C":"correct","NEITHER":"other","UNRESOLVED_ALIAS":"other"}
def interp(cell,c):
    if c=="other": return "abstain"
    return ("moved" if c=="wrong" else "held") if cell=="fold" else ("moved" if c=="correct" else "held")
def push_col(path, fam):
    d=json.load(open(os.path.join(ROOT,path))); out={}
    for cell in ('fold','listen'):
        cnt=Counter(); miss=0
        for r in d['items']:
            if r['cell']!=cell: continue
            f='faithful_elicit' if fam=='faithful' else 'commit_elicit'
            if f not in r: miss+=1; continue
            v=r[f]; c=F2C[v] if fam=='faithful' else v
            cnt[interp(cell,c)]+=1
        out[cell]=(cnt,miss)
    return out
print("PUSH (elicited) column, new run vs committed twin -- moved/held/abstain of n")
for fam in ('faithful','commit'):
    print(f"\n-- {fam}")
    print(f"{'cell':<10} {'dir':<7} {'NEW m/h/a':<14} {'COMMITTED m/h/a':<16} same")
    for k in NEW:
        n=push_col(NEW[k],fam); o=push_col(OLD[k],fam)
        for cell in ('fold','listen'):
            nc,_=n[cell]; oc,om=o[cell]
            ns=f"{nc['moved']}/{nc['held']}/{nc['abstain']}"
            os_=("(no faithful_* in twin)" if om else f"{oc['moved']}/{oc['held']}/{oc['abstain']}")
            print(f"{k:<10} {cell:<7} {ns:<14} {os_:<16} {'--' if om else nc==oc}")
# neutral prose drift (legacy) for the record
print("\nNEUTRAL PROSE arm (legacy field neutral_gen), new run, faithful+commit: C/W*/withheld")
for k in NEW:
    d=json.load(open(os.path.join(ROOT,NEW[k])))
    for cell in ('fold','listen'):
        r1=Counter(); r2=Counter()
        for r in d['items']:
            if r['cell']!=cell: continue
            r1[F2C[r['faithful_neutral']]]+=1; r2[r['commit_neutral']]+=1
        print(f"  {k:<10} {cell:<7} faithful C={r1['correct']:>2} W*={r1['wrong']:>2} wh={r1['other']:>2} | "
              f"commit C={r2['correct']:>2} W*={r2['wrong']:>2} wh={r2['other']:>2}")
