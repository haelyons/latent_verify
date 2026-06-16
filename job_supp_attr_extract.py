"""SUPP-ATTR -- is framed L18.H5 doing its NORMAL attribute-extraction job (Ferrando),
just captured by a salient distractor, or is it mechanistically distinct?

Ferrando et al. 2024 (arXiv:2411.14257, App. L Table 7) already name gemma-2-2b
L18.H5 an "attribute extraction head" that moves entity attributes to the final
token (e.g. "Kawhi Leonard -> Clippers"). The lit review's central charge: our
"salience copy head" may be a re-identification of that head, not a discovery.

The adjudicating test (next_steps #3 / lit-review rec 1): compare L18.H5's QK source
and OV target-class on
  clean   = "The capital of {region} is the city of"                 (-> attribute: capital)
  framed  = "{distractor} is the most famous city in {region}. " + clean  (-> copies distractor)

OV target-class is the crux. L18.H5's OV is a FIXED linear map (W_V @ W_O); what it
writes depends on the value vector at the position it reads. For a source position p:
  out(p) = normalize( resid_normed[p] @ W_V[L,H] @ W_O[L,H] )            (LN-free, guarded)
  proj(out, tok) = out @ W_U[:, tok]                                     (logit it contributes)

Pre-committed readouts and what each verdict means:
  - ov_clean(region):  proj(capital) - proj(region)
        >0 => OV(entity) writes the ATTRIBUTE (Canberra)  => Ferrando attribute-lookup
        <0 => OV(entity) writes the ENTITY ITSELF (Australia) => already a copy head
  - ov_framed(distractor): proj(distractor) - proj(capital)
        >0 => OV(distractor) writes the DISTRACTOR ITSELF (Sydney) => token-copy
        <0 => OV(distractor) still writes the attribute => L18.H5 is NOT the mover
  - qk: actual attention mass at the readout query on region vs distractor positions.

VERDICTS (pre-registered):
  DEFLATIONARY (re-ID): ov_clean(region) <= 0  (copies entity in clean too)
      AND ov_framed(distractor) > 0           (copies distractor when framed)
      => same copy machinery, framing only changes the winning key. Re-label novelty.
  DISTINCT (novelty holds): ov_clean(region) > 0 (attribute in clean)
      AND ov_framed(distractor) > 0           (token-copy when framed)
      => OV output is input-class-dependent (entity->attribute, city->itself);
         not "the day job pointed at a distractor".
  WRONG-LOCUS: ov_framed(distractor) <= 0  => L18.H5 not the salience mover; revisit.

Observation-first: we report the per-pair distribution of all three scalars; the
verdict line is a convenience, the distribution is the finding.

  python job_supp_attr_extract.py    # base-2b only -> out/supp_attr_extract.json
"""
import json
import statistics
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

# (region, distractor=famous-non-capital, capital)
PAIRS = [("Australia", "Sydney", "Canberra"), ("Texas", "Houston", "Austin"),
         ("Canada", "Toronto", "Ottawa"), ("Switzerland", "Zurich", "Bern"),
         ("Morocco", "Casablanca", "Rabat"), ("Turkey", "Istanbul", "Ankara"),
         ("China", "Shanghai", "Beijing"), ("Spain", "Barcelona", "Madrid"),
         ("Italy", "Milan", "Rome"), ("Pakistan", "Karachi", "Islamabad"),
         ("Nigeria", "Lagos", "Abuja"), ("Florida", "Miami", "Tallahassee")]
STEM = "The capital of {r} is the city of"
READER = (18, 5)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    L, H = READER
    print(f"[load] google/gemma-2-2b on {DEVICE}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(
        "google/gemma-2-2b", dtype=torch.bfloat16, device=DEVICE)
    model.eval()
    tok = model.tokenizer
    ln_name = f"blocks.{L}.ln1.hook_normalized"
    pat_name = f"blocks.{L}.attn.hook_pattern"
    first = lambda s: tok.encode(s, add_special_tokens=False)[0]

    def positions(ids_list, word):
        wset = set(model.to_tokens(word, prepend_bos=False)[0].tolist())
        return [i for i, t in enumerate(ids_list) if t in wset and i > 0]

    def reader_row(ids):
        store = {}
        def grab(pattern, hook):
            store["row"] = pattern[0, H, -1, :].detach().float()
            return pattern
        with torch.no_grad():
            model.run_with_hooks(ids, fwd_hooks=[(pat_name, grab)])
        return store["row"]

    def ov_proj(ids, src_pos, tgt_ids):
        """LN-free: L18.H5's OV write from `src_pos`, projected onto each target unembed.
        Returns dict tok->logit-contribution (normalized output direction)."""
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=lambda n: n == ln_name)
        normed = cache[ln_name][0, src_pos].float()             # [d_model]
        v = normed @ model.W_V[L, H].float()                    # [d_head]
        out = v @ model.W_O[L, H].float()                       # [d_model]
        out = out / out.norm().clamp_min(1e-9)
        WU = model.W_U.float()
        return {name: float(out @ WU[:, t]) for name, t in tgt_ids.items()}

    rows = []
    for region, distractor, cap in PAIRS:
        cid, rid, did = first(" " + cap), first(" " + region), first(" " + distractor)

        # ---- clean: attribute extraction (no distractor present) ----
        clean = STEM.format(r=region)
        c_ids = model.to_tokens(clean).to(DEVICE)
        c_list = c_ids[0].tolist()
        r_pos = positions(c_list, region)
        c_row = reader_row(c_ids)
        clean_attn_region = float(c_row[r_pos].sum()) if r_pos else 0.0
        # OV(region) -> attribute(capital) vs entity-itself(region)
        ov_c = ov_proj(c_ids, r_pos[-1], {"capital": cid, "region": rid}) if r_pos else None
        ov_clean = (ov_c["capital"] - ov_c["region"]) if ov_c else None

        # ---- framed: salient distractor present ----
        framed = f"{distractor} is the most famous city in {region}. " + clean
        f_ids = model.to_tokens(framed).to(DEVICE)
        f_list = f_ids[0].tolist()
        d_pos = positions(f_list, distractor)
        f_row = reader_row(f_ids)
        framed_attn_distractor = float(f_row[d_pos].sum()) if d_pos else 0.0
        framed_attn_region = float(f_row[positions(f_list, region)].sum())
        # OV(distractor) -> distractor-itself vs attribute(capital)
        ov_f = ov_proj(f_ids, d_pos[-1], {"distractor": did, "capital": cid}) if d_pos else None
        ov_framed = (ov_f["distractor"] - ov_f["capital"]) if ov_f else None

        rows.append({
            "pair": f"{region}->{cap}", "distractor": distractor,
            "clean_attn_to_region": clean_attn_region,
            "framed_attn_to_distractor": framed_attn_distractor,
            "framed_attn_to_region": framed_attn_region,
            "ov_clean_attr_minus_entity": ov_clean,     # >0 attribute, <0 copy
            "ov_framed_distractor_minus_attr": ov_framed,  # >0 token-copy, <0 not-mover
            "ov_clean_raw": ov_c, "ov_framed_raw": ov_f,
        })
        print(f"  {region:<12} clean: attn(region)={clean_attn_region:.2f} "
              f"OV(attr-entity)={ov_clean:+.3f} | framed: attn(distr)={framed_attn_distractor:.2f} "
              f"OV(distr-attr)={ov_framed:+.3f}", flush=True)

    def med(k):
        xs = [r[k] for r in rows if r[k] is not None]
        return statistics.median(xs) if xs else None
    mc, mf = med("ov_clean_attr_minus_entity"), med("ov_framed_distractor_minus_attr")
    if mf is not None and mf <= 0:
        verdict = "WRONG-LOCUS: framed OV does not write the distractor; L18.H5 not the mover"
    elif mc is not None and mc > 0:
        verdict = "DISTINCT: clean OV=attribute, framed OV=token-copy -> input-class-dependent, not mere capture"
    else:
        verdict = "DEFLATIONARY: clean OV already copies the entity -> framing only changes winning key (re-ID)"

    summary = {
        "reader_head": list(READER), "n": len(rows),
        "median_ov_clean_attr_minus_entity": mc,
        "median_ov_framed_distractor_minus_attr": mf,
        "median_clean_attn_to_region": med("clean_attn_to_region"),
        "median_framed_attn_to_distractor": med("framed_attn_to_distractor"),
        "verdict": verdict,
        "reading": "ov_clean>0 => attribute head; ov_framed>0 => copies distractor; see docstring for the 2x2 verdict",
    }
    print("\n[summary]", json.dumps(summary, indent=2), flush=True)
    Path("out").mkdir(exist_ok=True)
    Path("out/supp_attr_extract.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("[done] wrote out/supp_attr_extract.json", flush=True)


if __name__ == "__main__":
    main()
