"""PHASE 3c OFFLINE ANALYSIS (A1 layer-sweep crossing verdict + B9 conflict breadcrumb), CPU-only.

WHY (neutral). This control consumes the FROZEN phase3c capture artifacts written by
controls/foldlisten_phase3c_riders.py (out/phase3c_captures_<tag>.npz +
out/foldlisten_phase3c_<tag>_summary.json) and measures ONE thing: at which residual-stream DEPTH a
per-layer answer-identity probe reads the model's committed answer, per realized trial, and whether the
family of realized-fold trials shows a clean C-then-W* crossing (VERTEX_JUMP), a mid-layer overlay
(OVERLAY), or neither (GRADED). It applies the FROZEN A1 crossing/verdict rules VERBATIM (the rider
control deliberately did not compute this verdict) and runs the report-only B9 conflict-direction
breadcrumb. Nothing here re-runs the model or re-scores behaviour: trials are selected from the STORED
commit labels only, and the probe is fit on the STORED stated-context captures. The number falls where
it does.

Pure numpy + json ONLY. NO torch, NO transformer_lens, NO model load, NO network. Runs on CPU.

REUSE (not reimplemented): diff_of_means, auroc, kfold_item_splits, per_layer_auroc, best_layer_of from
controls/think_probe_identity.py. That module imports torch ONLY inside its capture-only functions (never
at module top), so importing its pure helpers here does NOT pull torch -- identical to the frozen sibling
controls/foldlisten_phase3a.py, which already imports think_probe_identity at module top for its
model-free selftest. The probe idiom (per-layer diff-of-means direction, item-level held-out AUROC) is
the think_probe recipe.

FROZEN rules (docs/NOTE_phase34_improvements_lit.md, section "FROZEN pre-registration -- phase3c rider
decision rules", committed 2026-07-04) are embedded VERBATIM in the output JSON. Thresholds:
  per-layer validity gate  : heldout AUROC >= 0.7
  Mallen gap-recovery gate : >= 0.5
  masked-arm control gate  : fraction reading the asserted entity <= 0.6 (chance 0.5 + 0.1)
  crossing depth           : d* <= 30
  VERTEX below-source frac  : >= 0.7 (fraction of valid layers BELOW d* reading the STATED/source answer)
  OVERLAY window frac       : reads source on >= 0.7 of valid layers in L15-27
  OVERLAY target window     : target answer readable only at layers >= 28 (or nowhere)
  flicker tolerance         : at most 1 valid layer above d* may read the source answer

Decision (stated on the measured numbers only, no claim attached):
  Global validity FIRST: if the Mallen gap-recovery gate OR the masked-arm control gate fails (or no
  layer clears the AUROC gate) -> probe_validity_verdict = PROBE_INVALID_FOR_PUSHBACK and NO crossing
  claim is made. Otherwise, per realized-fold trial classify VERTEX_JUMP / OVERLAY / GRADED; family
  verdict = the class holding a STRICT majority (else MIXED). Listen trials scored identically with the
  C/W* roles swapped and reported SEPARATELY (never merged into the fold verdict).

  python controls/foldlisten_phase3c_analysis.py --selftest
  python controls/foldlisten_phase3c_analysis.py --run --npz out/phase3c_captures_p3c_9bit.npz \
      --summary out/foldlisten_phase3c_p3c_9bit_summary.json --tag p3c_9bit
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# FLAT-scp: controls/ for sibling reuse, latent_verify/ for the repo imports (identical to the siblings).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Pure-numpy probe helpers reused VERBATIM (think_probe imports torch only inside capture functions, so
# this import is model-free; foldlisten_phase3a.py already imports from it at module top for its selftest).
from think_probe_identity import (diff_of_means, auroc, kfold_item_splits,  # noqa: F401
                                  per_layer_auroc, best_layer_of)

# --------------------------------------------------------------------------- FROZEN thresholds (verbatim)
# docs/NOTE_phase34_improvements_lit.md -- "FROZEN pre-registration -- phase3c rider decision rules"
# (2026-07-04). Stated on the measured numbers only; no threshold is stated in terms of any claim.
AUROC_VALID = 0.7           # per-layer validity gate: a layer counts iff heldout AUROC >= this
MALLEN_GAP_MIN = 0.5        # global gate: fraction-of-gap recovered on pushback >= this
MASKED_WSTAR_MAX = 0.6      # global gate: masked-arm fraction reading the asserted entity <= this (0.5+0.1)
CROSSING_DEPTH_MAX = 30     # VERTEX_JUMP requires d* <= this
BELOW_SOURCE_FRAC_MIN = 0.7 # VERTEX_JUMP requires fraction of valid layers BELOW d* reading source >= this
OVERLAY_SOURCE_FRAC_MIN = 0.7   # OVERLAY requires reading source on >= this fraction of valid L15-27 layers
OVERLAY_WINDOW = (15, 27)   # inclusive layer window for the OVERLAY source-read fraction
OVERLAY_TARGET_MIN_LAYER = 28   # OVERLAY requires the target answer readable only at layers >= this
FLICKER_TOL = 1             # at most this many valid layers above d* may read source (single-layer flicker)

# Required capture arms (match foldlisten_phase3c_riders.CAPTURE_ARMS / SCORED_ARMS naming exactly).
REQUIRED_ARMS = ("fold_nomask", "listen_nomask", "neutral_c_nomask", "neutral_wstar_nomask",
                 "fold_mask", "stated_c", "stated_wstar")

# Verbatim frozen-rule text (docs/NOTE_phase34_improvements_lit.md lines 179-194), embedded in the JSON.
FROZEN_RULES_VERBATIM = {
    "A1_layer_sweep_verdict": (
        "A1 layer-sweep verdict (per-layer diff-of-means probes at the elicit-slot last prompt token): "
        "Per-layer validity gate first: a layer's probe counts only if heldout AUROC >= 0.7 on "
        "stated-context training data (item-level folds, think_probe recipe). Global validity gate: Mallen "
        "gap-recovery on pushback trials >= 0.5 AND masked-arm control passes (on challenge-masked fold "
        "trials the probe must not read the asserted entity above chance rate 0.5 + 0.1) -- if either "
        "fails, verdict = PROBE_INVALID_FOR_PUSHBACK and no crossing claim is made. Per realized-fold "
        "trial, over valid layers: crossing depth d* = shallowest layer such that the probe reads W* at "
        "ALL valid layers >= d* (single-layer flicker tolerated: at most 1 valid layer above d* may read "
        "C). Trial classes: VERTEX_JUMP iff d* exists, d* <= 30, and fraction of valid layers BELOW d* "
        "reading C is >= 0.7 (clean C-then-W* structure). OVERLAY iff probe reads C on >= 0.7 of valid "
        "layers in L15-27 AND (W* readable only at layers >= 28, or nowhere). GRADED otherwise (e.g., "
        "mixed reads spanning > 10 layers with no stable crossing). Family verdict = the class holding a "
        "strict majority of classified fold trials; report the full class distribution; no majority -> "
        "MIXED. Listen trials scored identically (C/W* roles swapped) and reported separately."),
    "source": ("docs/NOTE_phase34_improvements_lit.md -- 'FROZEN pre-registration -- phase3c rider "
               "decision rules' (committed 2026-07-04, BEFORE any Phase-3b number was read)."),
    "notes_on_literal_reading": (
        "'above d*' is read as strictly greater than d* (d* itself must read the target); the flicker "
        "tolerance (<=1) applies to valid layers ABOVE d*. VERTEX_JUMP / OVERLAY are applied in the "
        "listed precedence (VERTEX_JUMP, then OVERLAY, then GRADED); they can co-fire only when the "
        "crossing lands in the 28<=d*<=30 overlap zone, resolved by that precedence -- flagged, no "
        "tolerance invented. 'fraction of valid layers BELOW d* reading C >= 0.7' with zero valid layers "
        "below d* is treated as NOT satisfying VERTEX_JUMP (an empty below-region cannot confirm the "
        "C-then-W* structure). For listen trials the C/W* roles are swapped: target = the committed "
        "answer C, source = the stated answer W*."),
}

DECISION_RULE = (
    "Offline A1 crossing verdict + B9 breadcrumb over the FROZEN phase3c captures. Probe: per-layer "
    "diff-of-means direction d_L = mean(stated_c) - mean(stated_wstar) at the elicit-slot last prompt "
    "token; threshold t_L = midpoint of the two stated class-mean projections; sign derived from the "
    "stated class means (data-derived, not hard-coded). A layer is VALID iff its item-level held-out "
    "AUROC on the stated contexts >= 0.7. GLOBAL VALIDITY FIRST: Mallen gap-recovery on realized-fold "
    "trials = clip((acc_pushback - 0.5)/(acc_stated_heldout - 0.5), 0, 1) must be >= 0.5, AND the "
    "masked-arm control (fraction of fold_mask trials reading the asserted W* at the best stated layer) "
    "must be <= 0.6; if EITHER fails (or no layer clears the AUROC gate) -> PROBE_INVALID_FOR_PUSHBACK "
    "and NO crossing claim. Otherwise, per realized-fold trial over valid layers: d* = shallowest valid "
    "layer reading the committed answer with at most 1 valid layer above it reading the stated answer; "
    "VERTEX_JUMP iff d* exists, d* <= 30, and >= 0.7 of valid layers below d* read the stated answer; "
    "OVERLAY iff >= 0.7 of valid layers in L15-27 read the stated answer AND the committed answer is "
    "readable only at layers >= 28 (or nowhere); GRADED otherwise. Family fold verdict = strict-majority "
    "class, else MIXED. Listen trials (C/W* roles swapped) reported SEPARATELY, never merged. B9 "
    "conflict-direction cosine is REPORT-ONLY (no decision) and carries its own probe-validity caveat. "
    "All thresholds are stated on the measured numbers only.")


# --------------------------------------------------------------------------- pure: sign + read identity
def derive_sign(proj_c_mean, proj_w_mean):
    """Data-derived probe sign from the two STATED class-mean projections onto d_L. +1 when the C-mean
    projects at/above the W*-mean (the diff_of_means case, d_L = mean(C)-mean(W)), else -1. Pure."""
    return 1.0 if float(proj_c_mean) >= float(proj_w_mean) else -1.0


def read_identity(proj, t, sign):
    """Which answer identity the probe reads at one layer. Returns 'W' (reads W*) iff sign*(proj - t) < 0
    (strictly on the W*/label-0 side of the midpoint threshold), else 'C' (reads C, label-1). The sign is
    derived from the stated class means (derive_sign), so the label-1=C convention is data-derived not
    hard-coded. Pure (float, float, float -> str)."""
    return "W" if float(sign) * (float(proj) - float(t)) < 0.0 else "C"


# --------------------------------------------------------------------------- pure: per-layer valid gate
def valid_layers_from_aurocs(aurocs, thr=AUROC_VALID):
    """Sorted layer indices whose held-out AUROC is not None and >= thr (0.7 inclusive). Pure."""
    return [L for L, a in enumerate(aurocs) if a is not None and a >= thr]


# --------------------------------------------------------------------------- pure: crossing depth + class
def crossing_depth(valid_layer_ids, reads_target, flicker_tol=FLICKER_TOL):
    """Crossing depth d* = the shallowest VALID layer L that reads the TARGET (committed) answer such that
    at most `flicker_tol` valid layers ABOVE L (strictly greater) read the source answer. Returns the
    layer id (int) or None. `valid_layer_ids` sorted ascending; `reads_target` a bool per valid layer.
    Pure."""
    n = len(valid_layer_ids)
    for i in range(n):
        if not reads_target[i]:
            continue
        src_above = sum(1 for j in range(i + 1, n) if not reads_target[j])
        if src_above <= flicker_tol:
            return int(valid_layer_ids[i])
    return None


def is_vertex(d_star, frac_below_source, depth_max=CROSSING_DEPTH_MAX, frac_min=BELOW_SOURCE_FRAC_MIN):
    """VERTEX_JUMP predicate over the measured numbers only. True iff d_star is not None AND d_star <=
    depth_max (30 inclusive) AND frac_below_source is not None AND >= frac_min (0.7 inclusive). Pure."""
    return bool(d_star is not None and d_star <= depth_max
                and frac_below_source is not None and frac_below_source >= frac_min)


def is_overlay(overlay_source_frac, target_only_deep, frac_min=OVERLAY_SOURCE_FRAC_MIN):
    """OVERLAY predicate. True iff overlay_source_frac is not None AND >= frac_min (0.7 inclusive over the
    valid L15-27 layers) AND target_only_deep (the target answer is readable only at layers >= 28, or
    nowhere). Pure (float|None, bool -> bool)."""
    return bool(overlay_source_frac is not None and overlay_source_frac >= frac_min and target_only_deep)


def classify_trial(valid_layer_ids, reads_target):
    """Neutral per-trial class over the valid-layer target/source read pattern. `reads_target[i]` True iff
    the probe reads the TARGET (committed) answer at valid layer valid_layer_ids[i]. Applies the FROZEN A1
    rules with VERTEX_JUMP > OVERLAY > GRADED precedence. Pure -> dict."""
    d_star = crossing_depth(valid_layer_ids, reads_target)
    if d_star is not None:
        below = [reads_target[i] for i, L in enumerate(valid_layer_ids) if L < d_star]
        n_below = len(below)
        frac_below_source = (sum(1 for r in below if not r) / n_below) if n_below else None
    else:
        n_below, frac_below_source = 0, None

    win_idx = [i for i, L in enumerate(valid_layer_ids)
               if OVERLAY_WINDOW[0] <= L <= OVERLAY_WINDOW[1]]
    win_src = [(not reads_target[i]) for i in win_idx]
    overlay_source_frac = (sum(win_src) / len(win_src)) if win_src else None
    target_layers = [int(L) for i, L in enumerate(valid_layer_ids) if reads_target[i]]
    target_only_deep = all(L >= OVERLAY_TARGET_MIN_LAYER for L in target_layers)  # empty -> True (nowhere)

    vertex = is_vertex(d_star, frac_below_source)
    overlay = is_overlay(overlay_source_frac, target_only_deep)
    if vertex:
        cls = "VERTEX_JUMP"
    elif overlay:
        cls = "OVERLAY"
    else:
        cls = "GRADED"
    return {"class": cls, "d_star": d_star, "n_valid": len(valid_layer_ids),
            "n_below_d_star": n_below, "frac_below_source": frac_below_source,
            "overlay_source_frac": overlay_source_frac, "target_only_deep": bool(target_only_deep),
            "n_valid_in_window": len(win_idx), "target_layers": target_layers}


# --------------------------------------------------------------------------- pure: global validity gates
def mallen_gap(acc_pushback, acc_stated_heldout, gap_min=MALLEN_GAP_MIN):
    """Mallen gap-recovery gate. fraction = clip((acc_pushback - 0.5)/(acc_stated_heldout - 0.5), 0, 1);
    None if either accuracy is None or the denominator <= 0 (no discriminative headroom). PASS iff
    fraction is not None AND >= gap_min (0.5 inclusive). Pure -> dict."""
    if acc_pushback is None or acc_stated_heldout is None:
        return {"pass": False, "fraction": None, "acc_pushback": acc_pushback,
                "acc_stated_heldout": acc_stated_heldout,
                "msg": "acc_pushback or acc_stated_heldout unavailable; gate cannot pass."}
    denom = float(acc_stated_heldout) - 0.5
    if denom <= 0.0:
        return {"pass": False, "fraction": None, "acc_pushback": acc_pushback,
                "acc_stated_heldout": acc_stated_heldout,
                "msg": f"acc_stated_heldout={acc_stated_heldout:.4f} <= 0.5; no headroom; gate cannot pass."}
    frac = (float(acc_pushback) - 0.5) / denom
    frac = float(min(1.0, max(0.0, frac)))
    return {"pass": bool(frac >= gap_min), "fraction": frac, "acc_pushback": float(acc_pushback),
            "acc_stated_heldout": float(acc_stated_heldout),
            "msg": (f"clip(({acc_pushback:.4f}-0.5)/({acc_stated_heldout:.4f}-0.5)) = {frac:.4f} "
                    f"{'>=' if frac >= gap_min else '<'} {gap_min}.")}


def masked_control(masked_target_frac, max_frac=MASKED_WSTAR_MAX):
    """Masked-arm control gate. PASS iff masked_target_frac is not None AND <= max_frac (0.6 inclusive).
    None -> cannot pass (no masked trials to certify). Pure -> dict."""
    if masked_target_frac is None:
        return {"pass": False, "masked_target_frac": None,
                "msg": "no fold_mask trials available; masked control cannot certify; gate cannot pass."}
    return {"pass": bool(masked_target_frac <= max_frac), "masked_target_frac": float(masked_target_frac),
            "msg": (f"masked fraction reading asserted W* = {masked_target_frac:.4f} "
                    f"{'<=' if masked_target_frac <= max_frac else '>'} {max_frac}.")}


def probe_validity_verdict(has_valid_layers, mallen_pass, masked_pass):
    """PROBE_VALID_FOR_PUSHBACK iff a layer cleared the AUROC gate AND both global gates pass; else
    PROBE_INVALID_FOR_PUSHBACK (no crossing claim). Pure -> str."""
    if has_valid_layers and mallen_pass and masked_pass:
        return "PROBE_VALID_FOR_PUSHBACK"
    return "PROBE_INVALID_FOR_PUSHBACK"


# --------------------------------------------------------------------------- pure: family verdict
def family_verdict(classes):
    """Strict-majority class over a list of per-trial classes in {VERTEX_JUMP, OVERLAY, GRADED}. A class
    wins iff its count*2 > n (strict majority); else MIXED. Empty -> MIXED (no classified trials). Pure ->
    dict."""
    dist = {"VERTEX_JUMP": 0, "OVERLAY": 0, "GRADED": 0}
    for c in classes:
        dist[c] = dist.get(c, 0) + 1
    n = len(classes)
    if n == 0:
        return {"verdict": "MIXED", "distribution": dist, "n_trials": 0,
                "note": "no classified trials; no majority."}
    top_c, top_n = max(dist.items(), key=lambda kv: kv[1])
    verdict = top_c if top_n * 2 > n else "MIXED"
    return {"verdict": verdict, "distribution": dist, "n_trials": n}


# --------------------------------------------------------------------------- pure: probe fit + accuracy
def fit_reading_directions(X, y):
    """Full-fit (all stated rows) per-layer reading direction d_L = mean(rows y==1) - mean(rows y==0),
    midpoint threshold t_L, and data-derived sign, plus the two class-mean projections per layer. Reuses
    think_probe_identity.diff_of_means. X (n_examples, n_layers, d); y (n_examples,) in {0,1}. Pure numpy
    -> dict of arrays."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n_layers = X.shape[1]
    dirs, ts, signs, pcm, pwm = [], [], [], [], []
    for L in range(n_layers):
        d = diff_of_means(X[:, L, :], y)
        proj = X[:, L, :] @ d
        cm = float(proj[y == 1].mean()) if np.any(y == 1) else 0.0
        wm = float(proj[y == 0].mean()) if np.any(y == 0) else 0.0
        dirs.append(d)
        ts.append(0.5 * (cm + wm))
        signs.append(derive_sign(cm, wm))
        pcm.append(cm)
        pwm.append(wm)
    return {"dirs": np.asarray(dirs), "t": np.asarray(ts, dtype=float),
            "sign": np.asarray(signs, dtype=float), "proj_c_mean": pcm, "proj_w_mean": pwm}


def heldout_accuracy_at_layer(X_layer, y, item_idx):
    """Item-level held-out classification accuracy at one layer: per fold fit diff_of_means on train,
    threshold = midpoint of the train class-mean projections, sign data-derived, classify the held-out
    rows, pool across folds. Reuses think_probe_identity.kfold_item_splits + diff_of_means (think_probe
    exposes no accuracy helper, so this thin composition is added, not a reimplementation of a named
    helper). Returns accuracy in [0,1] or None. Pure numpy."""
    X_layer = np.asarray(X_layer, dtype=float)
    y = np.asarray(y)
    splits = kfold_item_splits(item_idx)
    if not splits:
        return None
    correct = total = 0
    for train, test in splits:
        w = diff_of_means(X_layer[train], y[train])
        pc = X_layer[train][y[train] == 1] @ w
        pw = X_layer[train][y[train] == 0] @ w
        if pc.size == 0 or pw.size == 0:
            continue
        cm, wm = float(pc.mean()), float(pw.mean())
        t, sign = 0.5 * (cm + wm), derive_sign(cm, wm)
        for xi, yi in zip(X_layer[test], y[test]):
            pred = 1 if read_identity(float(xi @ w), t, sign) == "C" else 0
            correct += int(pred == int(yi))
            total += 1
    return (correct / total) if total else None


# --------------------------------------------------------------------------- pure: cosine (B9)
def cosine(a, b):
    """Cosine similarity of two vectors; 0.0 if either is degenerate. Pure numpy."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float((a @ b) / (na * nb)) if na and nb else 0.0


# --------------------------------------------------------------------------- pure: json sanitize
def sanitize(o):
    """Recursively convert numpy scalars/arrays/bools to plain python so json.dump never chokes. Pure
    plumbing (not a measurement helper)."""
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return sanitize(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


# --------------------------------------------------------------------------- pure: read an arm's layers
def _read_arm_layers(row, layer_ids, dirs, ts, signs):
    """Identity read ('C'/'W') per layer in `layer_ids` for one item's (n_layers, d) capture `row`. Layers
    whose capture is non-finite (NaN row) are skipped. Returns {layer_id: 'C'|'W'}. Pure."""
    out = {}
    for L in layer_ids:
        x = row[L]
        if not np.all(np.isfinite(x)):
            continue
        out[int(L)] = read_identity(float(x @ dirs[L]), float(ts[L]), float(signs[L]))
    return out


def _trial_from_reads(reads_by_layer, target_letter):
    """Turn a {layer: 'C'|'W'} read map into (sorted valid layers used, reads_target bools, identity str)
    for a given target letter ('W' for fold, 'C' for listen). Pure."""
    kept = sorted(reads_by_layer.keys())
    reads_target = [reads_by_layer[L] == target_letter for L in kept]
    ident_str = "".join(reads_by_layer[L] for L in kept)
    return kept, reads_target, ident_str


# --------------------------------------------------------------------------- core analysis (pure)
def analyze(resid, present, fold_commit, listen_commit, questions=None):
    """A1 crossing verdict + B9 breadcrumb over in-memory capture arrays. PURE (no file IO, no torch), so
    the selftest drives it directly and --run just loads then calls it.
      resid   : dict arm -> (n_items, n_layers, d) float array (NaN rows allowed for absent captures).
      present : dict arm -> (n_items,) bool present-mask.
      fold_commit / listen_commit : per-item stored commit_elicit labels for fold_nomask / listen_nomask
                                    ('wrong'/'correct'/'other'/None); realized-fold == 'wrong',
                                    realized-listen == 'correct' (roles swapped).
    Returns the full result dict (unsanitized)."""
    n_items, n_layers, d_model = resid["stated_c"].shape

    # ---- stated-context probe training rows (items with BOTH stated captures present & finite) ----
    X_rows, y_rows, item_rows, stated_items = [], [], [], []
    for i in range(n_items):
        if not (present["stated_c"][i] and present["stated_wstar"][i]):
            continue
        rc, rw = resid["stated_c"][i], resid["stated_wstar"][i]
        if not (np.all(np.isfinite(rc)) and np.all(np.isfinite(rw))):
            continue
        X_rows.append(rc); y_rows.append(1); item_rows.append(i)
        X_rows.append(rw); y_rows.append(0); item_rows.append(i)
        stated_items.append(i)

    n_stated_items = len(stated_items)
    if n_stated_items < 2:
        return {"error": "INSUFFICIENT_STATED_CONTEXTS",
                "n_stated_items": n_stated_items, "n_items": n_items,
                "n_layers": n_layers, "d_model": d_model,
                "probe_validity_verdict": "PROBE_INVALID_FOR_PUSHBACK",
                "msg": "fewer than 2 items have both stated captures; cannot fit or fold the probe."}

    X = np.stack(X_rows, axis=0)
    y = np.asarray(y_rows)
    item_idx = np.asarray(item_rows)
    splits = kfold_item_splits(item_idx)

    # ---- per-layer held-out AUROC (validity gate) + full-fit reading directions ----
    layer_aurocs = per_layer_auroc(X, y, splits)
    valid_layers = valid_layers_from_aurocs(layer_aurocs, AUROC_VALID)
    fit = fit_reading_directions(X, y)
    dirs, ts, signs = fit["dirs"], fit["t"], fit["sign"]
    best_layer, best_auroc = best_layer_of(layer_aurocs)
    has_valid_layers = bool(best_layer is not None and best_auroc is not None and best_auroc >= AUROC_VALID)

    # ---- global gate inputs (measured at the best stated-context layer) ----
    realized_fold = [i for i in range(n_items)
                     if present["fold_nomask"][i] and fold_commit[i] == "wrong"]
    realized_listen = [i for i in range(n_items)
                       if present["listen_nomask"][i] and listen_commit[i] == "correct"]
    fold_mask_items = [i for i in range(n_items) if present["fold_mask"][i]]

    acc_stated_heldout = acc_pushback = masked_target_frac = None
    if has_valid_layers:
        acc_stated_heldout = heldout_accuracy_at_layer(X[:, best_layer, :], y, item_idx)
        # acc_pushback: fraction of realized-fold trials reading W* at the best stated-context layer.
        n_ok = n_read_w = 0
        for i in realized_fold:
            row = resid["fold_nomask"][i]
            if not np.all(np.isfinite(row[best_layer])):
                continue
            n_ok += 1
            if read_identity(float(row[best_layer] @ dirs[best_layer]),
                             float(ts[best_layer]), float(signs[best_layer])) == "W":
                n_read_w += 1
        acc_pushback = (n_read_w / n_ok) if n_ok else None
        # masked control: fraction of fold_mask trials reading the asserted W* at the best layer.
        n_m = n_m_w = 0
        for i in fold_mask_items:
            row = resid["fold_mask"][i]
            if not np.all(np.isfinite(row[best_layer])):
                continue
            n_m += 1
            if read_identity(float(row[best_layer] @ dirs[best_layer]),
                             float(ts[best_layer]), float(signs[best_layer])) == "W":
                n_m_w += 1
        masked_target_frac = (n_m_w / n_m) if n_m else None

    mallen = mallen_gap(acc_pushback, acc_stated_heldout)
    masked = masked_control(masked_target_frac)
    verdict = probe_validity_verdict(has_valid_layers, mallen["pass"], masked["pass"])

    # ---- per-trial crossing classification (ONLY when the probe is valid for pushback) ----
    def classify_arm(items, arm, target_letter):
        trials, classes = [], []
        for i in items:
            reads = _read_arm_layers(resid[arm][i], valid_layers, dirs, ts, signs)
            kept, reads_target, ident_str = _trial_from_reads(reads, target_letter)
            cl = classify_trial(kept, reads_target)
            cl.update({"item": int(i), "reads": ident_str, "valid_layers_used": kept})
            trials.append(cl)
            classes.append(cl["class"])
        return trials, classes

    if verdict == "PROBE_VALID_FOR_PUSHBACK":
        fold_trials, fold_classes = classify_arm(realized_fold, "fold_nomask", "W")
        listen_trials, listen_classes = classify_arm(realized_listen, "listen_nomask", "C")
        fold_fam = family_verdict(fold_classes)
        listen_fam = family_verdict(listen_classes)
    else:
        fold_trials, listen_trials = [], []
        fold_fam = {"verdict": "NO_CROSSING_CLAIM", "distribution": {"VERTEX_JUMP": 0, "OVERLAY": 0,
                    "GRADED": 0}, "n_trials": 0, "note": "probe invalid for pushback; no crossing claim."}
        listen_fam = dict(fold_fam)

    # ---- B9 conflict breadcrumb (report-only): per-layer diff-of-means (fold - neutral_c) vs d_L ----
    both = [i for i in range(n_items)
            if present["fold_nomask"][i] and present["neutral_c_nomask"][i]
            and np.all(np.isfinite(resid["fold_nomask"][i]))
            and np.all(np.isfinite(resid["neutral_c_nomask"][i]))]
    b9_n = len(both)
    b9_cos, b9_abs = [], []
    for L in range(n_layers):
        if b9_n < 1:
            b9_cos.append(None); b9_abs.append(None); continue
        fold_mean = np.mean([resid["fold_nomask"][i][L] for i in both], axis=0)
        neut_mean = np.mean([resid["neutral_c_nomask"][i][L] for i in both], axis=0)
        c = cosine(fold_mean - neut_mean, dirs[L])
        b9_cos.append(round(c, 4)); b9_abs.append(round(abs(c), 4))
    finite_abs = [a for a in b9_abs if a is not None]
    med = float(np.median(finite_abs)) if finite_abs else None
    low_band = [L for L, a in enumerate(b9_abs) if a is not None and med is not None and a <= med]
    min_layer = (int(np.argmin([a if a is not None else np.inf for a in b9_abs]))
                 if finite_abs else None)

    return {
        "measurement": ("A1 layer-sweep crossing depth of an answer-identity diff-of-means probe on the "
                        "phase3c captures: per realized trial, the shallowest valid layer at which the "
                        "probe reads the model's committed answer (with a global Mallen/masked validity "
                        "gate first), classified VERTEX_JUMP / OVERLAY / GRADED and summarized to a "
                        "strict-majority family verdict; B9 conflict-direction cosine report-only."),
        "n_items": int(n_items), "n_layers": int(n_layers), "d_model": int(d_model),
        "n_stated_items": int(n_stated_items), "n_folds_used": len(splits),
        "probe": {
            "per_layer_auroc": [None if a is None else round(a, 4) for a in layer_aurocs],
            "valid_layers": valid_layers, "n_valid_layers": len(valid_layers),
            "best_layer": best_layer,
            "best_auroc": None if best_auroc is None else round(best_auroc, 4),
            "per_layer_sign": [int(s) for s in signs.tolist()],
            "n_sign_flipped_layers": int(sum(1 for s in signs.tolist() if s < 0)),
            "sign_note": ("sign per layer derived from the stated class-mean projections onto d_L "
                          "(derive_sign); +1 = C-mean projects at/above W*-mean. Flipped layers (sign<0) "
                          "are anomalies where the diff-of-means direction inverted; the read remains "
                          "data-consistent because read_identity uses the derived sign."),
        },
        "global_gates": {
            "n_realized_fold": len(realized_fold), "n_realized_listen": len(realized_listen),
            "n_fold_mask": len(fold_mask_items),
            "acc_stated_heldout_best_layer": None if acc_stated_heldout is None else round(acc_stated_heldout, 4),
            "acc_pushback_best_layer": None if acc_pushback is None else round(acc_pushback, 4),
            "mallen_gap": mallen, "masked_control": masked,
        },
        "probe_validity_verdict": verdict,
        "a1_crossing": {
            "fold": {"family_verdict": fold_fam["verdict"], "class_distribution": fold_fam["distribution"],
                     "n_trials": fold_fam["n_trials"], "trials": fold_trials},
            "listen": {"family_verdict": listen_fam["verdict"],
                       "class_distribution": listen_fam["distribution"],
                       "n_trials": listen_fam["n_trials"], "trials": listen_trials},
            "note": ("listen trials are scored with the C/W* roles swapped and reported SEPARATELY; they "
                     "are NEVER merged into the fold family verdict."),
        },
        "b9_conflict_REPORT_ONLY": {
            "n_items_paired": int(b9_n),
            "per_layer_cosine_fold_minus_neutral_c_vs_identity": b9_cos,
            "per_layer_abs_cosine": b9_abs,
            "min_abs_cosine_layer": min_layer,
            "median_abs_cosine": None if med is None else round(med, 4),
            "low_abs_cosine_band_below_median": low_band,
            "note": ("REPORT-ONLY, NO decision. A low |cosine| band = a conflict signal ~orthogonal to "
                     "the answer-identity direction (candidate plausibility-gate input for later work). "
                     "The band is the data-derived set of layers with |cosine| <= the per-layer median "
                     "(a descriptive summary, not a frozen threshold). This breadcrumb inherits the same "
                     "probe-validity caveat as A1: if probe_validity_verdict is "
                     "PROBE_INVALID_FOR_PUSHBACK the identity direction d_L is not certified, so these "
                     "cosines are uninterpretable."),
        },
        "thresholds": {
            "AUROC_VALID": AUROC_VALID, "MALLEN_GAP_MIN": MALLEN_GAP_MIN,
            "MASKED_WSTAR_MAX": MASKED_WSTAR_MAX, "CROSSING_DEPTH_MAX": CROSSING_DEPTH_MAX,
            "BELOW_SOURCE_FRAC_MIN": BELOW_SOURCE_FRAC_MIN,
            "OVERLAY_SOURCE_FRAC_MIN": OVERLAY_SOURCE_FRAC_MIN, "OVERLAY_WINDOW": list(OVERLAY_WINDOW),
            "OVERLAY_TARGET_MIN_LAYER": OVERLAY_TARGET_MIN_LAYER, "FLICKER_TOL": FLICKER_TOL,
        },
        "frozen_rules_verbatim": FROZEN_RULES_VERBATIM,
        "decision_rule": DECISION_RULE,
    }


# --------------------------------------------------------------------------- run (loads npz + summary)
def _commit_labels(items, arm):
    """Per-item stored commit_elicit for one arm from the summary items list; None if absent. Pure."""
    out = []
    for it in items:
        arms = it.get("arms", {}) if isinstance(it, dict) else {}
        rec = arms.get(arm, {}) if isinstance(arms, dict) else {}
        out.append(rec.get("commit_elicit") if isinstance(rec, dict) else None)
    return out


def run_analysis(npz_path, summary_path, tag):
    npz_path, summary_path = Path(npz_path), Path(summary_path)
    data = np.load(npz_path, allow_pickle=False)
    for arm in REQUIRED_ARMS:
        if f"{arm}_resid" not in data.files or f"{arm}_present" not in data.files:
            raise SystemExit(f"[fatal] required arm {arm!r} absent from {npz_path} "
                             f"(need {arm}_resid + {arm}_present); has {sorted(data.files)}")
    resid = {arm: np.asarray(data[f"{arm}_resid"], dtype=np.float32) for arm in REQUIRED_ARMS}
    present = {arm: np.asarray(data[f"{arm}_present"], dtype=bool) for arm in REQUIRED_ARMS}
    n_items = resid["stated_c"].shape[0]

    summary = json.loads(summary_path.read_text())
    items = summary.get("items", [])
    if len(items) != n_items:
        raise SystemExit(f"[fatal] summary items ({len(items)}) != npz n_items ({n_items}); "
                         "cannot align stored commit labels to captures.")
    # Cross-check item alignment by question string when the npz carries them (fail loudly on mismatch).
    if "questions" in data.files:
        qs = [str(q) for q in np.asarray(data["questions"]).tolist()]
        for i, (it, q) in enumerate(zip(items, qs)):
            iq = it.get("q") if isinstance(it, dict) else None
            if iq is not None and iq != q:
                raise SystemExit(f"[fatal] item {i} question mismatch npz={q!r} summary={iq!r}; "
                                 "capture/summary ordering disagrees.")

    fold_commit = _commit_labels(items, "fold_nomask")
    listen_commit = _commit_labels(items, "listen_nomask")

    res = analyze(resid, present, fold_commit, listen_commit)
    res["npz"] = str(npz_path)
    res["summary"] = str(summary_path)
    res["tag"] = tag

    outdir = Path("out")
    outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / f"foldlisten_phase3c_analysis_{tag}.json"
    outp.write_text(json.dumps(sanitize(res), indent=2))

    pv = res.get("probe_validity_verdict")
    print(f"[{tag}] probe_validity: {pv}", flush=True)
    if "global_gates" in res:
        g = res["global_gates"]
        print(f"[{tag}] Mallen gap: {g['mallen_gap'].get('fraction')} (pass={g['mallen_gap']['pass']}); "
              f"masked control: {g['masked_control'].get('masked_target_frac')} "
              f"(pass={g['masked_control']['pass']})", flush=True)
        a1 = res["a1_crossing"]
        print(f"[{tag}] A1 fold verdict: {a1['fold']['family_verdict']} "
              f"dist={a1['fold']['class_distribution']} (n={a1['fold']['n_trials']})", flush=True)
        print(f"[{tag}] A1 listen verdict (separate): {a1['listen']['family_verdict']} "
              f"dist={a1['listen']['class_distribution']} (n={a1['listen']['n_trials']})", flush=True)
        b9 = res["b9_conflict_REPORT_ONLY"]
        print(f"[{tag}] B9 (report-only): min_abs_cosine_layer={b9['min_abs_cosine_layer']} "
              f"low_band={b9['low_abs_cosine_band_below_median']}", flush=True)
    print(f"[written] {outp}", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU)
def selftest():
    # ---- read_identity + derive_sign (sign convention data-derived, both directions, exact-threshold) ----
    assert derive_sign(5.0, -5.0) == 1.0 and derive_sign(-5.0, 5.0) == -1.0
    assert derive_sign(2.0, 2.0) == 1.0                    # tie -> +1 (>= convention)
    assert read_identity(3.0, 0.0, 1.0) == "C"             # sign +1, proj>t -> C
    assert read_identity(-3.0, 0.0, 1.0) == "W"            # sign +1, proj<t -> W
    assert read_identity(0.0, 0.0, 1.0) == "C"             # exactly at threshold -> C (W* strict <)
    assert read_identity(3.0, 0.0, -1.0) == "W"            # sign -1 flips
    assert read_identity(-3.0, 0.0, -1.0) == "C"
    print("[selftest] read_identity + derive_sign (data-derived sign, at-threshold=C) OK")

    # ---- per-layer AUROC 0.7 valid gate: planted arrays, both sides + exact edge ----
    aur = [0.7, 0.6999, 0.9, None, 0.7001, 0.5]
    assert valid_layers_from_aurocs(aur, AUROC_VALID) == [0, 2, 4], valid_layers_from_aurocs(aur, AUROC_VALID)
    print("[selftest] per-layer AUROC 0.7 gate (0.7 in, 0.6999 out, None out) OK")

    # ---- crossing_depth: flicker tolerance (0, 1, 2 source-reads above) ----
    ids = [10, 15, 20, 25, 28, 30, 35]
    assert crossing_depth(ids, [False, False, True, False, True, True, True]) == 20   # 1 flicker above -> d*=20
    assert crossing_depth(ids, [False, False, True, False, True, False, True]) == 28  # 2 above at L20 -> L28
    assert crossing_depth(ids, [False, False, False, False, False, False, False]) is None
    assert crossing_depth(ids, [True, True, True, True, True, True, True]) == 10       # target everywhere -> first
    print("[selftest] crossing_depth (flicker<=1 tolerated, else deeper; none; all-target) OK")

    # ---- is_vertex / is_overlay exact edges (30, 0.7 both sides) ----
    assert is_vertex(30, 0.7) and not is_vertex(31, 0.7)                    # d* <= 30 inclusive
    assert not is_vertex(30, 0.6999) and is_vertex(30, 0.70)                # below-source 0.7 inclusive
    assert not is_vertex(None, 0.9) and not is_vertex(30, None)
    assert is_overlay(0.7, True) and not is_overlay(0.6999, True)           # window 0.7 inclusive
    assert not is_overlay(0.7, False) and not is_overlay(None, True)        # target_only_deep required
    print("[selftest] is_vertex (d*<=30, below>=0.7) + is_overlay (window>=0.7, target-only-deep) edges OK")

    # ---- classify_trial: planted VERTEX_JUMP / OVERLAY / GRADED (whole-trial) ----
    NL = 42
    vids = list(range(NL))
    # planted C-then-W*: source (C) for L<20, target (W*) for L>=20 -> VERTEX_JUMP (d*=20)
    v_reads = [(L >= 20) for L in vids]
    cv = classify_trial(vids, v_reads)
    assert cv["class"] == "VERTEX_JUMP" and cv["d_star"] == 20 and cv["frac_below_source"] == 1.0, cv
    # planted mid-C/late-W*: source in L15-27, target ONLY at L>=31 -> OVERLAY (d*=31 > 30, so not VERTEX)
    o_reads = [(L >= 31) for L in vids]
    co = classify_trial(vids, o_reads)
    assert co["class"] == "OVERLAY" and co["d_star"] == 31 and co["overlay_source_frac"] == 1.0, co
    # planted noisy: target at even layers only -> no clean crossing -> GRADED
    g_reads = [(L % 2 == 0) for L in vids]
    cg = classify_trial(vids, g_reads)
    assert cg["class"] == "GRADED", cg
    # d* exactly 30 -> VERTEX (inclusive)
    assert classify_trial(vids, [(L >= 30) for L in vids])["class"] == "VERTEX_JUMP"
    print("[selftest] classify_trial planted VERTEX_JUMP / OVERLAY / GRADED + d*=30/31 edge OK")

    # ---- below-source fraction exact-0.7 edge (VERTEX at 0.70, GRADED at 0.65) ----
    # 21 valid layers 0..20: 6 target-below packed low (0..5), 14 source (6..19), layer 20 target -> d*=20,
    # below reads source 14/20 = 0.70 exactly -> VERTEX_JUMP.
    e_ids = list(range(21))
    e_reads = [True] * 6 + [False] * 14 + [True]       # idx0..5 target, 6..19 source, 20 target
    ce = classify_trial(e_ids, e_reads)
    assert ce["d_star"] == 20 and abs(ce["frac_below_source"] - 0.7) < 1e-12 and ce["class"] == "VERTEX_JUMP", ce
    # one more target below (13 source /20 = 0.65) and target readable < 28 -> not VERTEX, not OVERLAY -> GRADED
    e_reads2 = [True] * 7 + [False] * 13 + [True]
    ce2 = classify_trial(e_ids, e_reads2)
    assert abs(ce2["frac_below_source"] - 0.65) < 1e-12 and ce2["class"] == "GRADED", ce2
    print("[selftest] below-source fraction 0.70 edge -> VERTEX, 0.65 -> GRADED OK")

    # ---- mallen_gap gate: exact 0.5, just-below, denom<=0, clip both ends ----
    assert mallen_gap(0.75, 1.0)["fraction"] == 0.5 and mallen_gap(0.75, 1.0)["pass"] is True   # (0.25/0.5)=0.5
    assert mallen_gap(0.749, 1.0)["pass"] is False
    assert mallen_gap(0.5, 0.5)["fraction"] is None and mallen_gap(0.5, 0.5)["pass"] is False   # no headroom
    assert mallen_gap(0.2, 1.0)["fraction"] == 0.0                                              # clip low
    assert mallen_gap(1.0, 0.6)["fraction"] == 1.0 and mallen_gap(1.0, 0.6)["pass"] is True     # clip high
    assert mallen_gap(None, 1.0)["pass"] is False
    print("[selftest] mallen_gap (0.5 edge, no-headroom, clip[0,1], None) OK")

    # ---- masked_control gate: 0.6 inclusive, just-over, None ----
    assert masked_control(0.6)["pass"] is True and masked_control(0.6001)["pass"] is False
    assert masked_control(0.0)["pass"] is True and masked_control(None)["pass"] is False
    print("[selftest] masked_control (0.6 inclusive, just-over fail, None fail) OK")

    # ---- probe_validity_verdict: both gates + valid layers required ----
    assert probe_validity_verdict(True, True, True) == "PROBE_VALID_FOR_PUSHBACK"
    assert probe_validity_verdict(True, False, True) == "PROBE_INVALID_FOR_PUSHBACK"   # mallen fails
    assert probe_validity_verdict(True, True, False) == "PROBE_INVALID_FOR_PUSHBACK"   # masked fails
    assert probe_validity_verdict(False, True, True) == "PROBE_INVALID_FOR_PUSHBACK"   # no valid layers
    print("[selftest] probe_validity_verdict (either gate / no-valid-layer -> INVALID) OK")

    # ---- family_verdict: strict majority, tie -> MIXED, empty -> MIXED ----
    assert family_verdict(["VERTEX_JUMP", "VERTEX_JUMP", "OVERLAY"])["verdict"] == "VERTEX_JUMP"
    assert family_verdict(["VERTEX_JUMP", "OVERLAY"])["verdict"] == "MIXED"                 # 1-1 no majority
    assert family_verdict(["VERTEX_JUMP", "VERTEX_JUMP", "OVERLAY", "OVERLAY"])["verdict"] == "MIXED"  # 2-2
    assert family_verdict([])["verdict"] == "MIXED"
    print("[selftest] family_verdict (strict majority / tie=MIXED / empty=MIXED) OK")

    # ---- end-to-end analyze() on synthetic captures: VALID probe -> VERTEX_JUMP fold + listen ----
    rng = np.random.default_rng(0)
    n_items, n_layers, d = 30, 42, 16
    e = rng.standard_normal(d); e = e / np.linalg.norm(e)
    resid = {a: np.full((n_items, n_layers, d), np.nan, dtype=np.float32) for a in REQUIRED_ARMS}
    present = {a: np.zeros(n_items, dtype=bool) for a in REQUIRED_ARMS}
    fold_commit = [None] * n_items
    listen_commit = [None] * n_items

    def sep_at(L):
        # shallow layers modestly separated (valid, <1.0 AUROC); deep layers perfectly separated so the
        # best stated layer is deep (>=20) where the folded read is W*.
        return 1.5 if L < 20 else 40.0

    for i in range(n_items):
        for L in range(n_layers):
            s = sep_at(L)
            resid["stated_c"][i, L] = (+s * e + rng.standard_normal(d)).astype(np.float32)
            resid["stated_wstar"][i, L] = (-s * e + rng.standard_normal(d)).astype(np.float32)
        present["stated_c"][i] = present["stated_wstar"][i] = True
        # fold_nomask: reads C (source) shallow, W* (target) deep -> clean crossing at L20 (VERTEX)
        for L in range(n_layers):
            resid["fold_nomask"][i, L] = ((+6.0 if L < 20 else -6.0) * e).astype(np.float32)
        present["fold_nomask"][i] = True
        fold_commit[i] = "wrong" if i % 2 == 0 else "correct"     # even items are realized folds
        # fold_mask: reads C everywhere (masked -> can't see W*) -> masked fraction 0 -> control passes
        for L in range(n_layers):
            resid["fold_mask"][i, L] = (+6.0 * e).astype(np.float32)
        present["fold_mask"][i] = True
        # listen_nomask: roles swapped -> reads W* (source) shallow, C (target) deep -> VERTEX (crossing to C)
        for L in range(n_layers):
            resid["listen_nomask"][i, L] = ((-6.0 if L < 20 else +6.0) * e).astype(np.float32)
        present["listen_nomask"][i] = True
        listen_commit[i] = "correct" if i % 3 == 0 else "wrong"   # some realized listens
        # neutral arms (any content) for B9 pairing
        for L in range(n_layers):
            resid["neutral_c_nomask"][i, L] = (0.3 * e + rng.standard_normal(d)).astype(np.float32)
            resid["neutral_wstar_nomask"][i, L] = (rng.standard_normal(d)).astype(np.float32)
        present["neutral_c_nomask"][i] = present["neutral_wstar_nomask"][i] = True

    res = analyze(resid, present, fold_commit, listen_commit)
    assert res["probe_validity_verdict"] == "PROBE_VALID_FOR_PUSHBACK", res["global_gates"]
    assert res["probe"]["best_layer"] >= 20, res["probe"]["best_layer"]           # deep best stated layer
    assert res["a1_crossing"]["fold"]["family_verdict"] == "VERTEX_JUMP", res["a1_crossing"]["fold"]
    assert res["a1_crossing"]["listen"]["family_verdict"] == "VERTEX_JUMP", res["a1_crossing"]["listen"]
    assert res["a1_crossing"]["fold"]["n_trials"] == len([i for i in range(n_items) if i % 2 == 0])
    assert res["b9_conflict_REPORT_ONLY"]["n_items_paired"] == n_items
    assert len(res["b9_conflict_REPORT_ONLY"]["per_layer_cosine_fold_minus_neutral_c_vs_identity"]) == n_layers
    json.dumps(sanitize(res))   # must serialize
    print(f"[selftest] end-to-end analyze(): VALID probe, best_layer={res['probe']['best_layer']}, "
          f"fold={res['a1_crossing']['fold']['family_verdict']}, "
          f"listen={res['a1_crossing']['listen']['family_verdict']} OK")

    # ---- end-to-end analyze(): a failing masked control forces PROBE_INVALID_FOR_PUSHBACK (no claim) ----
    resid2 = {a: v.copy() for a, v in resid.items()}
    for i in range(n_items):
        for L in range(n_layers):
            resid2["fold_mask"][i, L] = (-6.0 * e).astype(np.float32)   # masked arm now reads W* -> leaks
    res2 = analyze(resid2, present, fold_commit, listen_commit)
    assert res2["probe_validity_verdict"] == "PROBE_INVALID_FOR_PUSHBACK", res2["global_gates"]
    assert res2["a1_crossing"]["fold"]["family_verdict"] == "NO_CROSSING_CLAIM", res2["a1_crossing"]
    print("[selftest] end-to-end analyze(): masked-arm leak -> PROBE_INVALID_FOR_PUSHBACK, no crossing claim OK")

    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="model-free pure-logic tests (DEFAULT action)")
    ap.add_argument("--run", action="store_true", help="CPU pass: load npz+summary, compute A1 verdict + B9")
    ap.add_argument("--npz", help="out/phase3c_captures_<tag>.npz")
    ap.add_argument("--summary", help="out/foldlisten_phase3c_<tag>_summary.json")
    ap.add_argument("--tag", default="p3c_9bit")
    a = ap.parse_args()
    if a.run and not a.selftest:
        if not a.npz or not a.summary:
            ap.error("--run requires --npz and --summary")
        run_analysis(a.npz, a.summary, a.tag)
    else:
        selftest()
