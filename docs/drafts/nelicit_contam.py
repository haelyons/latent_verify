#!/usr/bin/env python3
"""Contamination census, both arms, same rule. Reads the NEW summaries only.
Step 1 reproduces JOIN_withhold_vs_fold.md's counter-arm numbers with JOIN's own rule (validation).
Step 2 applies the identical rule to the NEUTRAL arm."""
import json, os, re
ROOT='/home/hal/dev/interp/latent_verify'
NEW={'2b-base':('results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_2bbase_ext2_summary.json','base'),
     '9b-base':('results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json','base'),
     '27b-base':('results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27bbase_ext2_summary.json','base'),
     '2b-it':('results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_2bit_ext2_summary.json','it'),
     '9b-it':('results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bit_ext2_summary.json','it'),
     '27b-it':('results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27bit_ext2_summary.json','it')}
D={k:(json.load(open(os.path.join(ROOT,p))),m) for k,(p,m) in NEW.items()}
QRE=re.compile(r"\n\s*Q:")

print("STEP 1 -- JOIN's own rule, verbatim, on the COUNTER arm (validation target 82/82 & 47/39/69, fold cell)")
print(f"{'cell':<9} {'extra-turns':>12} {'invented-Q':>11}")
for k,(d,mo) in D.items():
    its=[r for r in d['items'] if r['cell']=='fold']
    pat = r"\nQ:" if mo=='base' else r"<start_of_turn>user"
    extra=sum(1 for r in its if len(re.findall(pat,r['elicit_prompt']))>3)
    inv=0
    for r in its:
        t=r['elicit_prompt'].split("Are you sure?",1)[-1].rsplit("Q: What is your final answer?",1)[0]
        qs=[q for q in re.findall(r"\nQ: (.+)",t) if not q.startswith("Actually") and "sure" not in q]
        inv+=bool(qs)
    print(f"{k:<9} {extra:>9}/{len(its)} {inv:>8}/{len(its)}")

print()
print("STEP 2 -- SAME rule family, NEUTRAL arm. Definitions (stated so they re-run):")
print("  ctx_Q     : the inserted prior turn (neutral_gen / counter_gen) contains a self-generated '\\n Q:' turn")
print("              == the elicit context carries a Q: turn beyond the 3 designed ones")
print("  ctx_invQ  : that self-dialogue poses a NEW question (a '\\nQ: <text>' line inside the inserted gen)")
print("  ans_Q     : the elicited answer itself (neutral_elicit_gen / elicit_gen) runs away into a 'Q:' turn")
print("  ctx|ans   : union -- 'context OR answer contains an invented Q: turn' (the asked quantity)")
print()
hdr=f"{'cell':<9} {'dir':<7} {'arm':<8} {'ctx_Q':>8} {'ctx_invQ':>9} {'ans_Q':>7} {'ctx|ans':>8}"
print(hdr); print('-'*len(hdr))
TAB={}
for k,(d,mo) in D.items():
    for cell in ('fold','listen'):
        its=[r for r in d['items'] if r['cell']==cell]
        for arm,gf,af in (('neutral','neutral_gen','neutral_elicit_gen'),
                          ('push','counter_gen','elicit_gen')):
            cq=ci=aq=un=0
            for r in its:
                g=r[gf]; a=r[af]
                c1=bool(QRE.search(g))
                c2=bool(re.search(r"\n\s*Q:\s*\S", g))
                a1=bool(QRE.search(a)) or a.strip().startswith("Q:")
                cq+=c1; ci+=c2; aq+=a1; un+=bool(c1 or a1)
            TAB[(k,cell,arm)]=(cq,ci,aq,un,len(its))
            print(f"{k:<9} {cell:<7} {arm:<8} {cq:>5}/{len(its)} {ci:>6}/{len(its)} {aq:>4}/{len(its)} {un:>5}/{len(its)}")

print()
print("STEP 3 -- does contamination sit ON the withheld items? (base cells, faithful-strict labels)")
F2C={"WSTAR":"wrong","C":"correct","NEITHER":"other","UNRESOLVED_ALIAS":"other"}
print(f"{'cell':<9} {'dir':<7} {'arm':<8} {'withheld':>9} {'of which ctx_Q':>16} {'of which ans_Q':>16}")
for k,(d,mo) in D.items():
    if mo!='base': continue
    for cell in ('fold','listen'):
        for arm,lf,gf,af in (('neutral','faithful_neutral_elicit','neutral_gen','neutral_elicit_gen'),
                             ('push','faithful_elicit','counter_gen','elicit_gen')):
            wh=[r for r in d['items'] if r['cell']==cell and F2C[r[lf]]=='other']
            c=sum(1 for r in wh if QRE.search(r[gf]))
            a=sum(1 for r in wh if QRE.search(r[af]) or r[af].strip().startswith("Q:"))
            print(f"{k:<9} {cell:<7} {arm:<8} {len(wh):>9} {c:>13}/{len(wh)} {a:>13}/{len(wh)}")
