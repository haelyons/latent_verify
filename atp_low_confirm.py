"""Open control (latent_skeptic author_queue, claim r1diff-no-installed-head): activation-patch the
AtP-LOW heads to harden or overturn the R1-DIFF NULL.

The null ("no head-local RLHF-installed deference component") rested entirely on the AtP screen. The
triage crux `atp_false_negative`: R2 (instr_triangulation held-out) showed AtP can under-rank a real
mediator ~6x (L7.H5 AtP 0.15 vs activation-patch 0.86), so a genuinely installed head could sit in the
net_it~0.04 band and be missed. This re-checks the AtP-sub-threshold heads with the ARBITER
(activation-patch, no AtP) on BOTH gemma-2-9b base and -it, reusing rlhf_differential._confirm verbatim.

Per head, frac = (M_patch - M_counter)/(M_neutral - M_counter) on the 16 I1 caving items: fraction of
the cave restored by patching that head's -it counter readout output to its neutral-turn value (and the
same on base). INSTALLED iff frac_it >= INSTALL_THR (0.10) AND |frac_base| <= BASE_FLOOR (0.05) --
the activation-patch analog of the differential, with no AtP in the loop.

Decision: any installed head -> NULL OVERTURNED (AtP false-negative confirmed). none -> NULL HARDENED
(the arbiter agrees with the AtP screen; no head-local installed component).

  python atp_low_confirm.py --selftest
  python atp_low_confirm.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it
"""
import argparse
import json
from pathlib import Path

import torch

from rlhf_differential import ITEMS, _confirm, INSTALL_THR, BASE_FLOOR  # reuse the verified harness

# AtP sub-threshold heads from results_r1_diff/out/rlhf_differential_9b.json: net_it_top12 (minus the
# base-shared L38.H14) UNION differential_top12 -- exactly the heads AtP ranked just under INSTALL_THR,
# where an under-ranked installed head would hide.
HEADS = [(33, 4), (24, 2), (27, 1), (28, 8), (22, 8), (24, 11), (28, 2), (33, 9), (21, 7), (23, 14),
         (22, 0), (24, 15), (29, 4), (24, 9), (34, 12), (26, 14), (27, 13), (23, 5)]
NH_9B = 16


def decide_installed(frac_it, frac_base, heads, nH):
    """Pure decision over the per-head activation-patch fractions (exercised by --selftest)."""
    rows, installed = [], []
    for (L, H) in heads:
        f = L * nH + H
        fi, fb = frac_it.get(f), frac_base.get(f)
        is_inst = (fi is not None and fi >= INSTALL_THR and (fb is None or abs(fb) <= BASE_FLOOR))
        rows.append({"L": L, "H": H, "frac_it": fi, "frac_base": fb, "installed": is_inst})
        if is_inst:
            installed.append({"L": L, "H": H, "frac_it": round(fi, 4),
                              "frac_base": (round(fb, 4) if fb is not None else None)})
    rows.sort(key=lambda r: (r["frac_it"] if r["frac_it"] is not None else -9), reverse=True)
    verdict = ("NULL OVERTURNED: activation-patch (arbiter) finds an installed head the AtP screen "
               "missed: " + str(installed)) if installed else \
              ("NULL HARDENED: no AtP-low head restores the cave in -it (>=INSTALL_THR) while ~absent "
               "in base; the arbiter agrees with the AtP screen -- no head-local installed component")
    return {"installed": installed, "rows": rows, "verdict": verdict}


def run(name_base, name_it):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nH = NH_9B
    flats = [L * nH + H for (L, H) in HEADS]
    print(f"[sweep] activation-patch arbiter over {len(flats)} AtP-low heads on {len(ITEMS)} items", flush=True)
    frac_it = _confirm(name_it, True, device, flats, nH)
    frac_base = _confirm(name_base, False, device, flats, nH)
    dec = decide_installed(frac_it, frac_base, HEADS, nH)
    out = {"model_base": name_base, "model_it": name_it, "cue": "atp_low_activation_patch_confirm",
           "n_heads_swept": len(HEADS), "install_thr": INSTALL_THR, "base_floor": BASE_FLOOR,
           "frac_it": {f"{L}.{H}": frac_it.get(L * nH + H) for (L, H) in HEADS},
           "frac_base": {f"{L}.{H}": frac_base.get(L * nH + H) for (L, H) in HEADS},
           "decision": dec}
    Path("out").mkdir(exist_ok=True)
    Path("out/atp_low_confirm_9b.json").write_text(json.dumps(out, indent=2))
    print("\n[verdict]", dec["verdict"])
    print("[done] wrote out/atp_low_confirm_9b.json")


def selftest():
    """Model-free: the installed decision over synthetic per-head activation-patch fractions."""
    nH = 16
    fi = {24 * nH + 11: 0.30, 24 * nH + 2: 0.30, 33 * nH + 4: 0.05}
    fb = {24 * nH + 11: 0.00, 24 * nH + 2: 0.20, 33 * nH + 4: 0.00}
    d = decide_installed(fi, fb, [(24, 11), (24, 2), (33, 4)], nH)
    names = {(r["L"], r["H"]) for r in d["installed"]}
    assert names == {(24, 11)}, d                 # 24.11 installed; 24.2 base-shared; 33.4 sub-threshold
    assert d["verdict"].startswith("NULL OVERTURNED"), d
    d2 = decide_installed({1 * nH + 1: 0.04}, {1 * nH + 1: 0.0}, [(1, 1)], nH)
    assert d2["verdict"].startswith("NULL HARDENED"), d2
    # None frac (item-gated out) handled
    d3 = decide_installed({1 * nH + 1: None}, {1 * nH + 1: None}, [(1, 1)], nH)
    assert d3["verdict"].startswith("NULL HARDENED"), d3
    print(f"[selftest] decide_installed OK: installed={names}")
    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--name-base", default="google/gemma-2-9b")
    ap.add_argument("--name-it", default="google/gemma-2-9b-it")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.name_base, a.name_it)
