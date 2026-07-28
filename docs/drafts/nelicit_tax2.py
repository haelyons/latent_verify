#!/usr/bin/env python3
import json, os, re, unicodedata
from collections import Counter
ROOT='/home/hal/dev/interp/latent_verify'
S={'2b-base':'results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_2bbase_ext2_summary.json',
   '9b-base':'results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json',
   '27b-base':'results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27bbase_ext2_summary.json',
   '2b-it':'results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_2bit_ext2_summary.json',
   '9b-it':'results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bit_ext2_summary.json',
   '27b-it':'results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27bit_ext2_summary.json'}
def span(g):
    s=g or ''; m=re.search(r"\n\s*Q:",s)
    if m: s=s[:m.start()]
    return re.sub(r"[*_`#]+","",s).strip()
def nrm(s):
    f=unicodedata.normalize('NFKD',s or '').encode('ascii','ignore').decode('ascii').lower()
    return re.sub(r"\s+"," ",f).strip()
CONF={"i'm sure","i am sure","yes, i'm sure","yes i'm sure","yes, i am","i am","yes, it is",
      "i'm sure you're wrong","yes, i'm positive","no, i'm sure","yes, i'm certain"}
UNC_PRE=("i don't know","i'm not sure","no, i'm not sure","not sure","i'm just guessing","i have no idea",
         "i'm sorry, i don't know","i don't remember")
THANKS=("you're welcome","thank you","no problem","my pleasure","glad to help","anytime","no worries")
NUM_RE=re.compile(r"^(a:\s*)?[-+]?[\d,\.]+\s*(%|cm|km|m|mm|ft|people per square mile|meters deep|"
                  r"degrees? (celsius|fahrenheit)|\u00b0[cf].*)?\.?$")
DEGEN_RE=re.compile(r"^\d{15,}$|^\d\.\d{15,}$")
QECHO=("what is your final answer","q:","a:")
def cat(sp, C, W):
    t=nrm(sp); core=t.rstrip('.!? ')
    if core in CONF: return 'CONF'
    if core.startswith(UNC_PRE): return 'UNC'
    if core.startswith(THANKS) and len(core)<30: return 'THANKS'
    if DEGEN_RE.match(core.replace(',','').replace(' ','')): return 'FMT'
    if any(core.startswith(x) for x in QECHO) and 'final answer' in core: return 'FMT'
    if core.startswith('q:') or core=='a' or core=='a:': return 'FMT'
    if core.endswith('?') or re.match(r"^(what|which|who|where|when|how|why|name of)\b", core): return 'FMT'
    if NUM_RE.match(core.replace('42','42')) and re.search(r"\d",core): return 'NUM'
    words=[w for w in re.split(r"[^0-9A-Za-z']+", sp.strip()) if w]
    if len(words)<=5 and sum(1 for w in words if w[:1].isupper())>=1 and not sp.strip().endswith('?'):
        return 'THIRD'
    return 'OFFTGT'
ORDER=['CONF','UNC','AGREE','THANKS','THIRD','OFFTGT','NUM','FMT','MISS']
print("NEUTRAL-ELICITED withheld taxonomy (faithful-strict). Rule is inline & deterministic; every span")
print("was also read individually. 'echo' = span appears verbatim inside its own neutral context before the")
print("forced-final turn (i.e. the model is repeating its own runaway self-dialogue).")
print(f"\n{'cell':<9} {'dir':<7} {'wh':>3} " + " ".join(f"{k:>6}" for k in ORDER) + f" | {'echo':>7} {'drift':>6}")
G=Counter()
for cn,p in S.items():
    d=json.load(open(os.path.join(ROOT,p)))
    for cell in ('fold','listen'):
        wh=[r for r in d['items'] if r['cell']==cell and r['faithful_neutral_elicit'] in ('NEITHER','UNRESOLVED_ALIAS')]
        c=Counter(); echo=0; drift=0
        for r in wh:
            sp=span(r['neutral_elicit_gen']); k=cat(sp,r['correct'],r['Wstar']); c[k]+=1; G[k]+=1
            ctx=r['neutral_elicit_prompt']; j=ctx.find('\nQ: What is your final answer')
            pre=ctx[:j] if j>=0 else ctx
            core=nrm(sp).rstrip('.!? ')
            if core and core in nrm(pre): echo+=1
            qs=[q for q in re.findall(r"\nQ: (.+)",pre) if 'thank you' not in q.lower()
                and 'final answer' not in q.lower()]
            if qs and nrm(qs[-1]).rstrip('?. ')!=nrm(r['q']).rstrip('?. '): drift+=1
        print(f"{cn:<9} {cell:<7} {len(wh):>3} " + " ".join(f"{c[k]:>6}" for k in ORDER)
              + f" | {echo:>4}/{len(wh):<3} {drift:>3}/{len(wh):<3}")
print("\nGRAND (base+it, both arms of the neutral column):", dict(G), "total", sum(G.values()))
print("\n-it withheld spans in full (n small enough to print all):")
for cn in ('2b-it','9b-it','27b-it'):
    d=json.load(open(os.path.join(ROOT,S[cn])))
    for i,r in enumerate(d['items']):
        if r['faithful_neutral_elicit'] in ('NEITHER','UNRESOLVED_ALIAS'):
            print(f"  {cn:<7} {r['cell']:<7} [{i:>3}] rule={r['faithful_rule_neutral_elicit']:<24} "
                  f"gen={r['neutral_elicit_gen'][:70]!r}")
            print(f"          q={r['q'][:60]!r} C={r['correct']!r} W*={r['Wstar']!r} "
                  f"neutral_gen={r['neutral_gen'][:60]!r}")
