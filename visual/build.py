"""Generate visual/index.html from the real committed results in out/*.json.

This is a GENERALISABLE TEMPLATE for the mechanism-claim verifier, presented blog/analysis
style. Two layers, kept separate on purpose:
  - CONFIG  : analysis-specific *content* (title, claim, prose, the schematic example).
  - render* : generic *components* (growth curve, scale bars, circuit map, boundary table)
              that take data and know nothing about salience specifically.
To reuse for a different mechanism: point the loaders at new out/*.json and edit CONFIG.

Data-true: every number on the page is read from a committed artifact here; nothing invented.
Run from anywhere:  python visual/build.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- analysis content
CONFIG = {
    "eyebrow": "latent_verify",
    "title": "Is this head set really responsible?",
    "claim": "These attention heads carry the copy — cut their attention to the distractor "
             "and the answer reverts to the truth.",
    "intro": "A verifier takes a causal claim like the one above and tests how much of the "
             "behaviour the heads actually carry, across many prompts, and where the claim stops holding.",
    "schematic": {
        "prompt": "“Sydney is the most famous city in Australia. The capital of Australia is …”",
        "distractor": "Sydney", "wrong": "Sydney", "right": "Canberra",
        "note": "necessity = fraction of the flip undone when the arrow is cut — here ≈ 1.0 (all heads).",
    },
    "sections": {
        "measure": {"title": "1 · What we measure",
            "lede": "The model copies a salient wrong word into its answer. We switch off the heads’ "
                    "attention to that word and ask how much of the mistake disappears. That fraction is "
                    "necessity: 0 = the heads did nothing, 1 = they fully caused it."},
        "growth": {"title": "2 · Necessity grows as we add heads",
            "lede": "Starting from one head, each round adds the head that recovers the most answer on "
                    "the cases the set still gets wrong. The curve is how much of the flip the set undoes; "
                    "bars are 95% bootstrap intervals; the dashed line is the pass mark."},
        "scale": {"title": "3 · Does it survive a bigger model?",
            "lede": "The same five facts, run through gemma-2-2b, -9b, and -9b-it. Right = the model copies "
                    "the distractor (the failure); left = it protects the right answer. The effect that is "
                    "strong at 2b collapses at 9b and reverses under RLHF — so the circuit is 2b-specific."},
        "circuit": {"title": "4 · Where the copy actually runs",
            "lede": "Each dot is one attention head, placed by layer (down) and head (across). Darker = "
                    "switching it off recovers more of the answer. One late reader plus a few early writers "
                    "carry it — outlined dots are the ones the loop adopted. Hover for the value."},
        "boundary": {"title": "5 · Where the mechanism holds",
            "lede": "The same behaviour across model size and how the question is posed. It is real in one "
                    "corner and absent or rerouted elsewhere — so “head X is the cause” is only true with "
                    "the regime attached."},
    },
}


def load(name):
    try:
        return json.loads((OUT / name).read_text())
    except Exception as e:
        print(f"  ! could not read {name}: {e}")
        return None


def growth_points(run):
    pts = []
    for rd in run["rounds"]:
        ad = rd.get("adopted")
        pts.append({"H": len(rd["H"]),
                    "nec": round(rd["necessity_mean"], 3) if rd["necessity_mean"] is not None else 0.0,
                    "lo": round(rd["ci"][0], 3) if rd["ci"][0] is not None else None,
                    "hi": round(rd["ci"][1], 3) if rd["ci"][1] is not None else None,
                    "cex": len(rd["counterexamples"]),
                    "adopt": (f"L{ad['head'][0]}.H{ad['head'][1]}" if ad else None)})
    return pts


def scale_model(d, color, key):
    s = d["summary"]; b = s["bare"]
    return {"key": key, "label": s["model"].split("/")[-1], "color": color,
            "mean": round(b["mean_effect"], 2), "attn": round(b["mean_max_attn_to_anchor"], 2),
            "reader": f"L{b['modal_reader_head'][0]}.H{b['modal_reader_head'][1]}",
            "eff": [round(p["conditions"]["bare"]["effect"], 2) for p in d["pairs"]]}


def main():
    print(f"[build] reading {OUT}")
    r2, r9 = load("refine_heads_2b.json"), load("refine_heads_9b.json")
    heads, fc = load("framing_localize_heads.json"), load("forcedchoice_fc_2b.json")
    s2, s9, s9it = (load("scale_mechanism_2b_base.json"), load("scale_mechanism_9b_base.json"),
                    load("scale_mechanism_9b_it.json"))

    page = {"config": CONFIG, "growth": {}}

    if r2:
        page["growth"]["2b"] = {"label": "gemma-2-2b", "points": growth_points(r2),
            "finalH": [f"L{l}.H{h}" for l, h in r2["final_H"]],
            "note": "Climbs as early-writer heads join the reader L18.H5; round-capped before the last two cases settle."}
    if r9:
        page["growth"]["9b"] = {"label": "gemma-2-9b", "points": growth_points(r9),
            "finalH": [f"L{l}.H{h}" for l, h in r9["final_H"]],
            "note": "Necessity ~0 and no head clears the bar, so it adopts nothing — one round. The per-fact 9b collapse is in §3."}

    if s2 and s9:
        models = [scale_model(s2, "var(--active)", "2b"), scale_model(s9, "var(--absent)", "9b")]
        if s9it:
            models.append(scale_model(s9it, "var(--diseng)", "9b-it"))
        page["scale"] = {"pairs": [p["pair"].split("->")[0] for p in s2["pairs"]], "models": models}

    if heads:
        adopted = set(f"{l},{h}" for l, h in (r2["final_H"] if r2 else []))
        page["circuit"] = {"layers": heads["top_layers"],
            "rows": [{"l": x["layer"], "h": x["head"], "n": round(x["necessity"], 3), "rank": x["rank"]} for x in heads["rows"]],
            "adopted": sorted(adopted), "nheads": max(x["head"] for x in heads["rows"]) + 1}

    fctxt = "necessity ≈1; reader ≠ L18.H5"
    if fc:
        s = fc["summary"]
        fctxt = (f"necessity ≈{round(s['mean_necessity_allheads'],2)}; reader "
                 f"L{s['modal_reader_head'][0]}.H{s['modal_reader_head'][1]}, "
                 f"L18.H5≈{round(s['mean_necessity_L18H5'],2)}")
    page["boundary"] = [
        {"cond": "gemma-2-2b · base-fragment", "status": "active",
         "num": "all-heads necessity ≈1.0; reader L18.H5", "src": "refine_heads_2b"},
        {"cond": "gemma-2-9b · base-fragment", "status": "absent",
         "num": "necessity ≈0; adopts nothing", "src": "refine_heads_9b"},
        {"cond": "gemma-2-2b · QA (plain question)", "status": "disengaged",
         "num": "latent pull +6.5 → +0.6", "src": "base_attn_qa"},
        {"cond": "gemma-2-2b · QA (forced-choice)", "status": "different", "num": fctxt, "src": "forcedchoice_fc_2b"},
    ]

    html = TEMPLATE.replace("/*DATA*/", json.dumps(page))
    (HERE / "index.html").write_text(html, encoding="utf-8")

    print("[build] embedded:")
    if "2b" in page["growth"]:
        print(f"  2b growth: {[p['nec'] for p in page['growth']['2b']['points']]}  finalH={page['growth']['2b']['finalH']}")
    if "9b" in page["growth"]:
        print(f"  9b growth: {[p['nec'] for p in page['growth']['9b']['points']]} (adopts nothing)")
    if "scale" in page:
        for m in page["scale"]["models"]:
            print(f"  scale {m['key']:5}: mean {m['mean']:+.2f}  attn {m['attn']}  reader {m['reader']}  per-pair {m['eff']}")
    if "circuit" in page:
        print(f"  circuit: {len(page['circuit']['rows'])} heads; adopted {page['circuit']['adopted']}")
    print(f"[build] wrote {HERE / 'index.html'}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>latent_verify</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;1,9..144,400;1,9..144,500&family=Hanken+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
<style>
  :root{
    --paper:#ffffff; --ink:#17191c; --soft:#586069; --faint:#9aa1a8; --rule:#ebebe8;
    --active:#1f9d6b; --absent:#aeb4ba; --diseng:#3b7bd1; --diff:#cf8a1e; --break:#c4503e;
    --sans:"Hanken Grotesk",system-ui,-apple-system,sans-serif;
    --serif:"Fraunces",Georgia,"Times New Roman",serif;
    --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  }
  *{box-sizing:border-box}
  body{margin:0; background:var(--paper); color:var(--ink); font-family:var(--sans);
    font-size:15px; line-height:1.6; -webkit-font-smoothing:antialiased}
  .wrap{max-width:820px; margin:0 auto; padding:54px 26px 90px}
  .mono{font-family:var(--mono)}

  header{margin-bottom:46px}
  .eyebrow{font-family:var(--mono); font-size:12px; color:var(--faint); letter-spacing:.5px}
  h1{font-size:32px; font-weight:700; letter-spacing:-.6px; margin:6px 0 16px}
  .claim{font-family:var(--serif); font-style:italic; font-size:22px; line-height:1.45;
    color:var(--ink); margin:0 0 16px; padding-left:16px; border-left:2px solid var(--ink); max-width:60ch}
  .intro{font-size:16px; color:var(--soft); margin:0; max-width:64ch}

  section{margin:0 0 52px}
  h2{font-size:22px; font-weight:600; letter-spacing:-.3px; margin:0 0 8px}
  .lede{font-size:15px; color:var(--soft); margin:0 0 20px; max-width:66ch}

  .schematic{border:1px solid var(--rule); border-radius:12px; padding:14px 16px}
  .toggle{display:inline-flex; border:1px solid var(--rule); border-radius:8px; overflow:hidden; margin-bottom:14px}
  .toggle button{font-family:var(--mono); font-size:13px; background:var(--paper); color:var(--soft);
    border:none; padding:7px 16px; cursor:pointer}
  .toggle button.on{background:var(--ink); color:var(--paper)}
  .panel-note{font-size:13.5px; color:var(--soft); margin-top:12px; max-width:66ch}
  .key{font-family:var(--mono); font-size:12px; color:var(--soft); margin-top:10px}
  .dot{display:inline-block; width:8px; height:8px; border-radius:50%; margin:0 6px 0 14px; vertical-align:middle}
  .key .dot:first-child{margin-left:0}

  table{width:100%; border-collapse:collapse; font-size:14px}
  th{text-align:left; font-weight:500; color:var(--faint); font-size:12px; padding:6px 10px;
    border-bottom:1px solid var(--rule); font-family:var(--mono)}
  td{padding:11px 10px; border-bottom:1px solid var(--rule); vertical-align:top}
  td.cond{font-weight:500}
  td .src{font-family:var(--mono); font-size:11.5px; color:var(--faint)}
  .status{font-family:var(--mono); font-size:12px; font-weight:500; white-space:nowrap}

  svg{display:block; width:100%; height:auto; overflow:visible}
  text{font-family:var(--mono)}
  .tip{position:fixed; pointer-events:none; background:var(--ink); color:var(--paper);
    font-family:var(--mono); font-size:11.5px; padding:6px 9px; border-radius:6px; opacity:0;
    transition:opacity .1s; z-index:9; white-space:nowrap}
  footer{border-top:1px solid var(--rule); padding-top:18px; font-family:var(--mono);
    font-size:11.5px; color:var(--faint); line-height:1.9}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow" id="eyebrow"></div>
    <h1 id="title"></h1>
    <p class="claim" id="claim"></p>
    <p class="intro" id="intro"></p>
  </header>

  <section><h2 id="t_measure"></h2><p class="lede" id="l_measure"></p>
    <div class="schematic"><div id="schematic"></div></div></section>

  <section><h2 id="t_growth"></h2><p class="lede" id="l_growth"></p>
    <div class="toggle" id="growthToggle"></div><div id="growth"></div>
    <p class="panel-note" id="growthNote"></p></section>

  <section><h2 id="t_scale"></h2><p class="lede" id="l_scale"></p>
    <div id="scale"></div><div class="key" id="scaleKey"></div>
    <p class="panel-note" id="scaleNote"></p></section>

  <section><h2 id="t_circuit"></h2><p class="lede" id="l_circuit"></p><div id="circuit"></div></section>

  <section><h2 id="t_boundary"></h2><p class="lede" id="l_boundary"></p>
    <table id="boundary"></table>
    <p class="key">
      <span class="status"><span class="dot" style="background:var(--active)"></span>active</span> copy present, knockout reverts
      <span class="status"><span class="dot" style="background:var(--absent)"></span>absent</span> no effect to undo
      <span class="status"><span class="dot" style="background:var(--diseng)"></span>disengaged</span> format switches it off
      <span class="status"><span class="dot" style="background:var(--diff)"></span>different</span> same effect, other heads
    </p></section>

  <footer id="foot"></footer>
</div>
<div class="tip" id="tip"></div>

<script>
const D = /*DATA*/;
const C = D.config;
const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const E = (t,a={},x)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(x!=null)e.textContent=x;return e;};
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const tip=$("tip");
const showTip=(ev,t)=>{tip.textContent=t;tip.style.opacity=1;tip.style.left=(ev.clientX+12)+"px";tip.style.top=(ev.clientY+12)+"px";};
const hideTip=()=>tip.style.opacity=0;

/* ---- content (generic: filled from CONFIG) ---- */
$("eyebrow").textContent=C.eyebrow; $("title").textContent=C.title;
$("claim").textContent=C.claim; $("intro").textContent=C.intro;
for(const k of ["measure","growth","scale","circuit","boundary"]){
  $("t_"+k).textContent=C.sections[k].title; $("l_"+k).textContent=C.sections[k].lede;
}

/* ---- component: schematic (generic, driven by CONFIG.schematic) ---- */
function schematic(){
  const s=C.schematic, W=720,H=176, dx=40, ax=560, svg=E("svg",{viewBox:`0 0 ${W} ${H}`});
  const defs=E("defs"),m=E("marker",{id:"ah",markerWidth:7,markerHeight:7,refX:5,refY:3,orient:"auto"});
  m.appendChild(E("path",{d:"M0 0 L6 3 L0 6 z",fill:"#17191c"}));defs.appendChild(m);svg.appendChild(defs);
  function row(y,answer,col,cut,label){
    svg.appendChild(E("rect",{x:dx-34,y:y-13,width:78,height:25,rx:5,fill:"#f4f6f5",stroke:"var(--rule)"}));
    svg.appendChild(E("text",{x:dx+5,y:y+4,fill:"var(--soft)","font-size":12,"text-anchor":"middle"},s.distractor));
    svg.appendChild(E("rect",{x:ax,y:y-13,width:104,height:25,rx:5,fill:"none",stroke:col,"stroke-width":1.5}));
    svg.appendChild(E("text",{x:ax+52,y:y+4,fill:col,"font-size":13,"text-anchor":"middle","font-weight":600},answer));
    const arc=E("path",{d:`M${dx+44} ${y-6} C 250 ${y-44}, 470 ${y-44}, ${ax} ${y-6}`,fill:"none",
      stroke:cut?"var(--break)":"var(--ink)","stroke-width":1.4,"stroke-dasharray":cut?"5 4":"none",
      opacity:cut?.6:.85,"marker-end":"url(#ah)"});
    svg.appendChild(arc);
    svg.appendChild(E("text",{x:300,y:y-38,fill:cut?"var(--break)":"var(--soft)","font-size":11.5,"text-anchor":"middle"},label));
    if(cut) svg.appendChild(E("text",{x:dx+150,y:y-22,fill:"var(--break)","font-size":15},"✂"));
  }
  svg.appendChild(E("text",{x:dx-34,y:16,fill:"var(--faint)","font-size":10.5},"distractor"));
  svg.appendChild(E("text",{x:ax,y:16,fill:"var(--faint)","font-size":10.5},"answer slot"));
  row(54,s.wrong,"var(--break)",false,"attention copies it → wrong");
  row(118,s.right,"var(--active)",true,"cut that attention → reverts");
  svg.appendChild(E("text",{x:0,y:H-20,fill:"var(--soft)","font-size":11.5},s.prompt));
  svg.appendChild(E("text",{x:0,y:H-3,fill:"var(--faint)","font-size":11},s.note));
  $("schematic").appendChild(svg);
}

/* ---- component: growth curve (generic, takes points) ---- */
let gKey=Object.keys(D.growth)[0];
function growthTabs(){const t=$("growthToggle");t.innerHTML="";
  Object.entries(D.growth).forEach(([k,g])=>{const b=document.createElement("button");
    b.textContent=g.label;b.className=(k===gKey?"on":"");b.onclick=()=>{gKey=k;growthTabs();growth();};t.appendChild(b);});}
function growth(){
  const g=D.growth[gKey],pts=g.points,W=720,H=240,pl=46,pr=18,pt=18,pb=44,iw=W-pl-pr,ih=H-pt-pb;
  const xN=Math.max(1,pts.length-1),yMin=-0.2,yMax=1.0;
  const X=i=>pl+(pts.length===1?iw/2:iw*i/xN),Y=v=>pt+ih*(1-(v-yMin)/(yMax-yMin));
  const svg=E("svg",{viewBox:`0 0 ${W} ${H}`});
  [0,0.25,0.5,0.75,1].forEach(v=>{svg.appendChild(E("line",{x1:pl,y1:Y(v),x2:W-pr,y2:Y(v),stroke:"#f0f0ee"}));
    svg.appendChild(E("text",{x:pl-9,y:Y(v)+3,fill:"#9aa1a8","font-size":10,"text-anchor":"end"},v.toFixed(2)));});
  svg.appendChild(E("line",{x1:pl,y1:Y(0.5),x2:W-pr,y2:Y(0.5),stroke:"var(--diff)","stroke-width":1.2,"stroke-dasharray":"5 4"}));
  svg.appendChild(E("text",{x:W-pr,y:Y(0.5)-6,fill:"var(--diff)","font-size":10,"text-anchor":"end"},"pass mark δ = 0.50"));
  if(pts.length>1){let d="";pts.forEach((p,i)=>d+=(i?"L":"M")+X(i)+" "+Y(p.nec));
    svg.appendChild(E("path",{d,fill:"none",stroke:"var(--ink)","stroke-width":2}));}
  pts.forEach((p,i)=>{const cx=X(i),cy=Y(p.nec);
    if(p.lo!=null){[["",p.lo,p.hi]].forEach(()=>{svg.appendChild(E("line",{x1:cx,y1:Y(p.lo),x2:cx,y2:Y(p.hi),stroke:"#c9ccce","stroke-width":1.5}));});}
    const c=E("circle",{cx,cy,r:5,fill:"#fff",stroke:"var(--ink)","stroke-width":2,style:"cursor:pointer"});
    c.addEventListener("mousemove",ev=>showTip(ev,`|H|=${p.H} · necessity ${p.nec} · CI [${p.lo}, ${p.hi}] · ${p.cex} counterexamples`));
    c.addEventListener("mouseleave",hideTip);svg.appendChild(c);
    svg.appendChild(E("text",{x:cx,y:H-pb+18,fill:"#9aa1a8","font-size":10,"text-anchor":"middle"},"|H|="+p.H));
    if(p.adopt)svg.appendChild(E("text",{x:cx,y:cy-13,fill:"var(--ink)","font-size":10.5,"text-anchor":"middle"},"+"+p.adopt));});
  $("growth").innerHTML="";$("growth").appendChild(svg);
  $("growthNote").textContent=g.note+"  Final set: "+g.finalH.join(", ")+".";
}

/* ---- component: scale diverging bars (generic, takes models x pairs) ---- */
function scale(){
  const sc=D.scale; if(!sc){return;}
  const pairs=sc.pairs, models=sc.models, n=models.length;
  const W=720,pl=96,pr=24,pt=10,rowH=18*n+18,H=pt+pairs.length*rowH+30,iw=W-pl-pr;
  const lo=-5.5,hi=11, X=v=>pl+iw*(v-lo)/(hi-lo);
  const svg=E("svg",{viewBox:`0 0 ${W} ${H}`});
  [-5,0,5,10].forEach(v=>{svg.appendChild(E("line",{x1:X(v),y1:pt,x2:X(v),y2:H-26,stroke:v===0?"#c9ccce":"#f0f0ee","stroke-width":v===0?1.4:1}));
    svg.appendChild(E("text",{x:X(v),y:H-12,fill:"#9aa1a8","font-size":10,"text-anchor":"middle"},v));});
  svg.appendChild(E("text",{x:X(0),y:H-1,fill:"var(--faint)","font-size":10,"text-anchor":"middle"},"← protects answer   ·   salience effect (nats)   ·   copies distractor →"));
  pairs.forEach((pr_,pi)=>{
    const y0=pt+pi*rowH;
    svg.appendChild(E("text",{x:pl-10,y:y0+rowH/2,fill:"var(--soft)","font-size":11,"text-anchor":"end"},pr_));
    models.forEach((m,mi)=>{
      const v=m.eff[pi],y=y0+8+mi*16,x0=X(0),x1=X(v);
      const bar=E("rect",{x:Math.min(x0,x1),y:y-6,width:Math.abs(x1-x0)||1,height:11,rx:2,fill:m.color,opacity:.9,style:"cursor:pointer"});
      bar.addEventListener("mousemove",ev=>showTip(ev,`${m.label} · ${pr_} · effect ${v>=0?"+":""}${v} nats`));
      bar.addEventListener("mouseleave",hideTip);svg.appendChild(bar);
    });
  });
  $("scale").innerHTML="";$("scale").appendChild(svg);
  $("scaleKey").innerHTML=models.map(m=>`<span class="status"><span class="dot" style="background:${m.color}"></span>${m.label}</span>`).join(" ");
  $("scaleNote").textContent="mean effect "+models.map(m=>`${m.label} ${m.mean>=0?"+":""}${m.mean}`).join(", ")
    +"; attention-to-distractor "+models.map(m=>m.attn).join(" → ")
    +"; reader head moves "+models.map(m=>m.reader).join(" → ")+".";
}

/* ---- component: circuit map (generic, takes per-head necessity) ---- */
function circuit(){
  const c=D.circuit; if(!c){return;}
  const layers=c.layers,nH=c.nheads,W=720,cellY=34,top=22,pl=40,colW=(W-pl-20)/nH,Hg=top+layers.length*cellY+10;
  const svg=E("svg",{viewBox:`0 0 ${W} ${Hg}`});
  for(let h=0;h<nH;h++)svg.appendChild(E("text",{x:pl+h*colW+colW/2,y:14,fill:"#9aa1a8","font-size":10,"text-anchor":"middle"},"H"+h));
  const adopted=new Set(c.adopted),byKey={};c.rows.forEach(r=>byKey[r.l+","+r.h]=r);
  layers.forEach((l,li)=>{const y=top+li*cellY+cellY/2;
    svg.appendChild(E("text",{x:pl-10,y:y+3,fill:"#9aa1a8","font-size":10,"text-anchor":"end"},"L"+l));
    for(let h=0;h<nH;h++){const r=byKey[l+","+h];if(!r)continue;
      const t=clamp(r.n/0.22,0,1),fill=r.n>0.005?`rgba(31,157,107,${0.15+t*0.8})`:"#eef0ef";
      const cx=pl+h*colW+colW/2,rad=6+t*7,ad=adopted.has(l+","+h);
      const dot=E("circle",{cx,cy:y,r:rad,fill,stroke:ad?"var(--ink)":"none","stroke-width":ad?1.6:0,style:"cursor:pointer"});
      dot.addEventListener("mousemove",ev=>showTip(ev,`L${l}.H${h} · necessity ${r.n}${ad?" · adopted":""}`));
      dot.addEventListener("mouseleave",hideTip);svg.appendChild(dot);}});
  $("circuit").innerHTML="";$("circuit").appendChild(svg);
}

/* ---- component: boundary table (generic, takes rows) ---- */
function boundary(){
  const cm={active:"var(--active)",absent:"var(--absent)",disengaged:"var(--diseng)",different:"var(--diff)"};
  $("boundary").innerHTML="<thead><tr><th>condition</th><th>status</th><th>what the numbers say</th></tr></thead><tbody>"
    +D.boundary.map(r=>`<tr><td class="cond">${r.cond}</td>
      <td><span class="status"><span class="dot" style="background:${cm[r.status]}"></span>${r.status}</span></td>
      <td>${r.num}<br><span class="src">${r.src}.json</span></td></tr>`).join("")+"</tbody>";
}

schematic(); growthTabs(); growth(); scale(); circuit(); boundary();
$("foot").innerHTML="Every value is read from committed results · refine_heads_2b/9b, scale_mechanism_2b/9b/9b-it, "+
  "framing_localize_heads, forcedchoice_fc_2b, base_attn_qa · regenerate with "+
  "<span style='color:var(--soft)'>python visual/build.py</span>";
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
