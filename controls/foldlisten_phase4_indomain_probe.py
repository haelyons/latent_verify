"""PHASE 4 IN-DOMAIN THINK PROBE (belief-vs-compliance crossing verdict), OFFLINE, CPU-only.

WHAT (neutral). Reads the FROZEN Phase-3c capture artifacts (out/phase3c_captures_<tag>.npz +
foldlisten_phase3c_<tag>_summary.json) and measures ONE thing: at which residual-stream DEPTH an
answer-identity probe -- SUPERVISED IN-DOMAIN on the neutral arms (state-an-answer-then-"Okay, thank
you.", the same 5-turn depth as the elicit slot, decorrelated from the challenge span) -- reads the
model's committed answer, per realized trial, and whether the family of realized-fold trials shows a clean
C-then-W* crossing (VERTEX_JUMP), a mid-stack compliance overlay (OVERLAY), or neither (GRADED). It closes
the domain gap that made the Phase-3c A1 stated-context probe read context salience instead of belief
(PROBE_INVALID_FOR_PUSHBACK). Nothing here re-runs the model or re-scores behaviour: trials are selected
from the STORED commit_elicit labels only, and the probe is fit on the STORED neutral-arm captures. The
number falls where it does.

INPUTS (all already on disk; no GPU, no model, no network):
  - out/phase3c_captures_<tag>.npz : per-arm <arm>_resid (n_items, n_layers, d_model) float16 + <arm>_present
    bool. Arms used: neutral_c_nomask, neutral_wstar_nomask (probe training), fold_nomask, listen_nomask
    (crossing), fold_mask (masked-arm control). Plus `questions` (asserted to match the summary items).
  - foldlisten_phase3c_<tag>_summary.json : items[i]['q'] and items[i]['arms'][arm]['commit_elicit'] in
    {correct(=C), wrong(=W*), other}, joined by item index (both are the frozen 74 in the same order).

ORIENTATION (fixed here, BEFORE fitting -- the subtlety a naive implementation flips):
  w_L = mean(neutral_wstar resid @ L) - mean(neutral_c resid @ L) -> points toward W*, so a higher
  projection s = x @ w_L is more W*-aligned; per-layer threshold theta_L = midpoint of the two class-mean
  projections; "reads W*" at layer L  <=>  s > theta_L  (at/below theta_L reads C). The CV AUROC uses
  neutral_wstar as the POSITIVE class, so with w_L pointing toward W* the AUROC is orientation-correct
  (chance 0.5).

FROZEN pre-registration in DESIGN_phase4_indomain_probe.md. The crossing/validity decision rules are
inherited VERBATIM from the Phase-3c A1 pre-registration (committed BEFORE any Phase-3b/3c number was
read); only the training arm changed to the in-domain neutral arms, plus one additive label-sanity gate.
All thresholds and rules are embedded verbatim in the output JSON and stated on the MEASURED numbers only
-- no threshold is stated in terms of any claim, and this control does not know which outcome bears on any
hypothesis.

REUSE (not reimplemented). kfold_item_splits, auroc, diff_of_means, per_layer_auroc, best_layer_of from
controls/think_probe_identity.py (its torch imports are confined to capture functions, so importing its
pure helpers is model-free -- identical to the frozen sibling foldlisten_phase3c_analysis.py). The FROZEN
A1 crossing/gate helpers + threshold constants from controls/foldlisten_phase3c_analysis.py
(valid_layers_from_aurocs, classify_trial, family_verdict, mallen_gap, masked_control, sanitize, ...), so
the crossing rules are guaranteed byte-identical to 3c and are not re-derived here.

Pure numpy + json ONLY. NO torch, NO transformer_lens, NO model load, NO network. Runs on CPU.

  python controls/foldlisten_phase4_indomain_probe.py --selftest
  python controls/foldlisten_phase4_indomain_probe.py --run \
      --npz results_foldlisten_p3c/out/phase3c_captures_p3c_9bit.npz \
      --summary results_foldlisten_p3c/out/foldlisten_phase3c_p3c_9bit_summary.json \
      --tag p4_9bit --outdir out
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
# this import is model-free; foldlisten_phase3a.py / foldlisten_phase3c_analysis.py already import from it
# at module top for their model-free selftests).
from think_probe_identity import (diff_of_means, kfold_item_splits,  # noqa: F401
                                  per_layer_auroc, best_layer_of)

# FROZEN A1 crossing/gate helpers + threshold constants, inherited VERBATIM so the crossing rules are
# byte-identical to Phase-3c (foldlisten_phase3c_analysis is pure-numpy + json only, no torch at import).
from foldlisten_phase3c_analysis import (  # noqa: F401
    valid_layers_from_aurocs, classify_trial, family_verdict, mallen_gap, masked_control, sanitize,
    AUROC_VALID, MASKED_WSTAR_MAX, MALLEN_GAP_MIN, CROSSING_DEPTH_MAX, BELOW_SOURCE_FRAC_MIN,
    OVERLAY_SOURCE_FRAC_MIN, OVERLAY_WINDOW, OVERLAY_TARGET_MIN_LAYER, FLICKER_TOL)

# --------------------------------------------------------------------------- phase4-only threshold (new)
# The one additive gate over the inherited 3c rules (DESIGN_phase4_indomain_probe.md, "Label-sanity gate").
LABEL_SANITY_MIN = 0.8      # both neutral arms must realize their stated answer on >= this fraction of items

# Arms this analysis touches. Training: the two neutral arms; crossing: fold/listen; global gate: fold_mask.
# The stated_* arms (used by 3c A1) are deliberately NOT required -- phase4's training axis is the neutral,
# in-domain arm, not the 2-turn stated context.
REQUIRED_ARMS = ("neutral_c_nomask", "neutral_wstar_nomask", "fold_nomask", "listen_nomask", "fold_mask")

# commit_elicit convention (stored labels): 'correct' == the C answer, 'wrong' == the W* answer.
COMMIT_C, COMMIT_WSTAR = "correct", "wrong"

# --------------------------------------------------------------------------- FROZEN rules (embedded verbatim)
FROZEN_RULES_VERBATIM = {
    "source": ("DESIGN_phase4_indomain_probe.md -- 'FROZEN decision rules (verbatim from 3c A1; only the "
               "training arm changed to neutral)'. The crossing/validity rules are inherited VERBATIM from "
               "the Phase-3c A1 pre-registration (docs/NOTE_phase34_improvements_lit.md, 'FROZEN "
               "pre-registration -- phase3c rider decision rules', committed BEFORE any Phase-3b/3c number "
               "was read). Math symbols ASCII-normalized (>=, <=, ->)."),
    "orientation_convention": (
        "Orientation convention (fixed here, before fitting): w_L = mean(neutral_wstar resid @ L) - "
        "mean(neutral_c resid @ L) -> points toward W*, so higher projection s = x @ w_L = more "
        "W*-aligned. Per-layer threshold theta_L = midpoint of the two class-mean projections. 'reads W*' "
        "for a trial at layer L <=> s > theta_L."),
    "per_layer_validity_gate": "a layer counts as valid iff heldout item-level-CV AUROC >= 0.7.",
    "global_validity_gate": (
        "masked-arm control -- on fold_mask trials the fraction reading W* must be <= 0.5 + 0.1 = 0.6 (the "
        "probe must NOT read the asserted entity when the model cannot see the challenge; fold_mask "
        "realizes C 71/74, verified, so it should read C). If it fails -> PROBE_INVALID_FOR_PUSHBACK and "
        "NO crossing claim is made. (Mallen gap-recovery >= 0.5 reported for comparability with 3c; the "
        "masked-arm control is the load-bearing specificity guard.)"),
    "label_sanity_gate": (
        "(new, additive): the neutral arms must realize their stated answer (neutral_c->C, "
        "neutral_wstar->W*) on >= 0.8 of items (else the supervision label is invalid). Report the rate; "
        "below 0.8 -> INSUFFICIENT_LABELS, no verdict."),
    "crossing_depth": (
        "Per realized-fold trial, over valid layers: crossing depth d* = shallowest valid layer such that "
        "the probe reads W* at ALL valid layers >= d* (single-layer flicker tolerated: at most 1 valid "
        "layer above d* may read C)."),
    "trial_classes": (
        "VERTEX_JUMP iff d* exists, d* <= 30, and the fraction of valid layers BELOW d* reading C is >= 0.7 "
        "(clean C-then-W* structure). OVERLAY iff the probe reads C on >= 0.7 of valid layers in L15-27 AND "
        "(W* readable only at layers >= 28, or nowhere). GRADED otherwise."),
    "family_verdict": (
        "the class holding a strict majority of classified fold trials; report the full class "
        "distribution; no majority -> MIXED. Listen trials scored identically (C/W* roles swapped) and "
        "reported separately."),
    "notes_on_literal_reading": (
        "Inherited from the 3c A1 implementation (foldlisten_phase3c_analysis.classify_trial, imported "
        "here verbatim): 'above d*' is strictly greater than d* (d* itself must read the target); the "
        "flicker tolerance (<=1) applies to valid layers ABOVE d*; VERTEX_JUMP > OVERLAY > GRADED "
        "precedence; 'fraction of valid layers BELOW d* reading C >= 0.7' with zero valid layers below d* "
        "does NOT satisfy VERTEX_JUMP. For listen trials the roles swap: target = the held answer C, "
        "source = W*. The masked-arm W*-fraction and the report-only Mallen figure are measured at the "
        "best-CV-AUROC layer (as in 3c)."),
}

DECISION_RULE = (
    "Offline in-domain crossing verdict over the FROZEN phase3c captures. Probe (orientation FIXED before "
    "fitting): per-layer w_L = mean(neutral_wstar resid @ L) - mean(neutral_c resid @ L) points toward W*, "
    "so higher s = x @ w_L is more W*-aligned; theta_L = midpoint of the two neutral class-mean "
    "projections; 'reads W*' <=> s > theta_L. A layer is VALID iff its item-level 5-fold-CV AUROC "
    "(neutral_wstar=pos vs neutral_c=neg) >= 0.7. GATES in order: (1) LABEL-SANITY -- both neutral arms "
    "must realize their stated answer (neutral_c->C i.e. commit_elicit=='correct'; neutral_wstar->W* i.e. "
    "=='wrong') on >= 0.8 of labelled items, else INSUFFICIENT_LABELS and no verdict; (2) GLOBAL VALIDITY "
    "-- the masked-arm control (fraction of fold_mask trials reading W* at the best-CV-AUROC layer) must "
    "be <= 0.6 AND at least one layer must clear the AUROC gate, else PROBE_INVALID_FOR_PUSHBACK and NO "
    "crossing claim (Mallen gap-recovery is computed and REPORTED for comparability with 3c but does NOT "
    "gate the phase4 verdict -- the masked-arm control is the load-bearing specificity guard). Otherwise, "
    "per realized-fold trial over valid layers: d* = shallowest valid layer reading W* with at most 1 "
    "valid layer above it reading C; VERTEX_JUMP iff d* exists, d* <= 30, and >= 0.7 of valid layers below "
    "d* read C; OVERLAY iff >= 0.7 of valid layers in L15-27 read C AND W* is readable only at layers >= "
    "28 (or nowhere); GRADED otherwise. Family fold verdict = strict-majority class, else MIXED. Listen "
    "trials (C/W* roles swapped: target = the held answer C) scored identically and reported SEPARATELY, "
    "never merged into the fold verdict. All thresholds are stated on the measured numbers only.")


# --------------------------------------------------------------------------- pure: fixed-orientation read
def reads_wstar(s, theta):
    """Fixed-orientation read at one layer. The probe direction w_L is FIXED (before fitting) to point
    toward W* -- w_L = mean(neutral_wstar @ L) - mean(neutral_c @ L) -- so a higher projection s = x @ w_L
    is more W*-aligned. Returns True ('reads W*') iff s > theta (STRICTLY above the midpoint threshold);
    at/below theta -> reads C. Pure (float, float -> bool). This is the subtlety a naive implementation
    flips: the orientation is data-fixed here, not sign-derived, and the '>' is strict."""
    return bool(float(s) > float(theta))


def read_letter(s, theta):
    """'W' if the probe reads W* at this layer (s > theta), else 'C'. Pure."""
    return "W" if reads_wstar(s, theta) else "C"


# --------------------------------------------------------------------------- pure: full-fit W*-ward probe
def fit_wstar_directions(X, y):
    """Full-fit per-layer probe with the orientation FIXED toward W*. y == 1 for neutral_wstar rows (W*,
    positive), y == 0 for neutral_c rows (C, negative). Per layer L:
      w_L = diff_of_means(X[:,L,:], y) = mean(rows y==1) - mean(rows y==0) = mean(wstar) - mean(c), which
            points toward W* (higher s = more W*-aligned);
      theta_L = midpoint of the two class-mean projections onto w_L.
    Because w_L points toward W*, proj_wstar_mean >= proj_c_mean by construction, so 'reads W* <=> s >
    theta_L' is orientation-correct. Reuses think_probe_identity.diff_of_means. Pure numpy -> dict."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n_layers = X.shape[1]
    dirs, thetas, wm, cm = [], [], [], []
    for L in range(n_layers):
        w = diff_of_means(X[:, L, :], y)                 # mean(wstar) - mean(c) -> points toward W*
        proj = X[:, L, :] @ w
        pw = float(proj[y == 1].mean()) if np.any(y == 1) else 0.0   # W* class-mean projection
        pc = float(proj[y == 0].mean()) if np.any(y == 0) else 0.0   # C  class-mean projection
        dirs.append(w)
        thetas.append(0.5 * (pw + pc))
        wm.append(pw)
        cm.append(pc)
    return {"dirs": np.asarray(dirs), "theta": np.asarray(thetas, dtype=float),
            "proj_wstar_mean": wm, "proj_c_mean": cm}


def heldout_neutral_accuracy_at_layer(X_layer, y, item_idx):
    """Item-level held-out accuracy of the fixed-orientation neutral probe at one layer, for the
    REPORT-ONLY Mallen comparability figure. Per fold: w = diff_of_means(train) (toward W*=y1); theta =
    midpoint of the train class-mean projections; classify held-out rows by reads_wstar (pred 1 = W*);
    pool across folds. Reuses kfold_item_splits + diff_of_means. Returns accuracy in [0,1] or None. Pure
    numpy."""
    X_layer = np.asarray(X_layer, dtype=float)
    y = np.asarray(y)
    splits = kfold_item_splits(item_idx)
    if not splits:
        return None
    correct = total = 0
    for train, test in splits:
        w = diff_of_means(X_layer[train], y[train])
        pw = X_layer[train][y[train] == 1] @ w
        pc = X_layer[train][y[train] == 0] @ w
        if pw.size == 0 or pc.size == 0:
            continue
        theta = 0.5 * (float(pw.mean()) + float(pc.mean()))
        for xi, yi in zip(X_layer[test], y[test]):
            pred = 1 if reads_wstar(float(xi @ w), theta) else 0
            correct += int(pred == int(yi))
            total += 1
    return (correct / total) if total else None


# --------------------------------------------------------------------------- pure: label-sanity gate
def label_sanity_rate(commit_arm, expected):
    """Fraction of LABELLED items on which one neutral arm realizes its STATED answer. `commit_arm` is the
    per-item commit_elicit list; `expected` is COMMIT_C for neutral_c or COMMIT_WSTAR for neutral_wstar.
    Counts over items whose commit label is not None. Returns (rate|None, n_labelled, n_realized). Pure."""
    n = ok = 0
    for c in commit_arm:
        if c is None:
            continue
        n += 1
        if c == expected:
            ok += 1
    return ((ok / n) if n else None), n, ok


def label_sanity_gate(rate_c, rate_wstar, thr=LABEL_SANITY_MIN):
    """Label-sanity gate (new, additive). PASS iff BOTH neutral arms realize their stated answer on >= thr
    (0.8 inclusive) of labelled items. A None rate (no labels) cannot pass. Pure -> dict."""
    ok = (rate_c is not None and rate_c >= thr and rate_wstar is not None and rate_wstar >= thr)
    return {"pass": bool(ok), "neutral_c_realizes_C": rate_c, "neutral_wstar_realizes_Wstar": rate_wstar,
            "threshold": thr,
            "msg": (f"neutral_c->C={rate_c}, neutral_wstar->W*={rate_wstar}; "
                    f"both {'>=' if ok else 'NOT all >='} {thr}.")}


# --------------------------------------------------------------------------- pure: phase4 validity verdict
def phase4_validity_verdict(labels_pass, has_valid_layers, masked_pass):
    """Neutral verdict over the measured gates (label-sanity FIRST). Pure -> str.
      INSUFFICIENT_LABELS        iff the label-sanity gate fails (no crossing verdict).
      PROBE_VALID_FOR_PUSHBACK   iff labels pass AND a layer clears the AUROC gate AND the masked-arm
                                 control passes.
      PROBE_INVALID_FOR_PUSHBACK otherwise (labels pass but no valid layer OR the masked control fails).
    Mallen gap-recovery is REPORT-ONLY in phase4 and does NOT enter this verdict (per the frozen rules)."""
    if not labels_pass:
        return "INSUFFICIENT_LABELS"
    if has_valid_layers and masked_pass:
        return "PROBE_VALID_FOR_PUSHBACK"
    return "PROBE_INVALID_FOR_PUSHBACK"


# --------------------------------------------------------------------------- pure: read an arm's layers
def _read_arm_wstar(row, layer_ids, dirs, thetas):
    """{layer_id: reads_wstar bool} per valid layer for one item's (n_layers, d) capture `row`. Layers
    whose capture is non-finite (NaN row) are skipped. Pure."""
    out = {}
    for L in layer_ids:
        x = row[L]
        if not np.all(np.isfinite(x)):
            continue
        out[int(L)] = reads_wstar(float(x @ dirs[L]), float(thetas[L]))
    return out


def _trial_from_reads(reads_wstar_by_layer, target_is_wstar):
    """Turn a {layer: reads_wstar bool} map into (sorted valid layers used, reads_target bools, identity
    str). For fold trials target_is_wstar=True (target = W*, so reads_target = reads_wstar); for listen
    trials target_is_wstar=False (target = the held answer C, so reads_target = NOT reads_wstar). Pure."""
    kept = sorted(reads_wstar_by_layer.keys())
    if target_is_wstar:
        reads_target = [bool(reads_wstar_by_layer[L]) for L in kept]
    else:
        reads_target = [(not reads_wstar_by_layer[L]) for L in kept]
    ident_str = "".join("W" if reads_wstar_by_layer[L] else "C" for L in kept)
    return kept, reads_target, ident_str


# --------------------------------------------------------------------------- core analysis (pure)
def analyze(resid, present, commits):
    """In-domain crossing verdict over in-memory capture arrays. PURE (no file IO, no torch), so the
    selftest drives it directly and --run just loads then calls it.
      resid   : dict arm -> (n_items, n_layers, d) float array (NaN rows allowed for absent captures).
      present : dict arm -> (n_items,) bool present-mask.
      commits : dict arm -> per-item stored commit_elicit ('correct'(=C)/'wrong'(=W*)/'other'/None) for
                fold_nomask, listen_nomask, neutral_c_nomask, neutral_wstar_nomask.
    Returns the full result dict (unsanitized)."""
    n_items, n_layers, d_model = resid["neutral_c_nomask"].shape

    # ---- label-sanity gate (behavioural, independent of capture presence) ----
    rate_c, nlab_c, ok_c = label_sanity_rate(commits["neutral_c_nomask"], COMMIT_C)
    rate_w, nlab_w, ok_w = label_sanity_rate(commits["neutral_wstar_nomask"], COMMIT_WSTAR)
    labels = label_sanity_gate(rate_c, rate_w)

    # ---- neutral-arm probe training rows (items with BOTH neutral captures present & finite) ----
    # Orientation FIXED: y=1 for neutral_wstar (W*, positive), y=0 for neutral_c (C, negative), so the
    # diff-of-means direction and the CV AUROC both point toward W*.
    X_rows, y_rows, item_rows, neutral_items = [], [], [], []
    for i in range(n_items):
        if not (present["neutral_wstar_nomask"][i] and present["neutral_c_nomask"][i]):
            continue
        rw, rc = resid["neutral_wstar_nomask"][i], resid["neutral_c_nomask"][i]
        if not (np.all(np.isfinite(rw)) and np.all(np.isfinite(rc))):
            continue
        X_rows.append(rw); y_rows.append(1); item_rows.append(i)     # W* = positive class
        X_rows.append(rc); y_rows.append(0); item_rows.append(i)     # C  = negative class
        neutral_items.append(i)
    n_neutral_items = len(neutral_items)

    # ---- per-layer held-out CV AUROC (validity gate) + full-fit W*-ward reading directions ----
    if n_neutral_items >= 2:
        X = np.stack(X_rows, axis=0)
        y = np.asarray(y_rows)
        item_idx = np.asarray(item_rows)
        splits = kfold_item_splits(item_idx)
        layer_aurocs = per_layer_auroc(X, y, splits)                 # pos = neutral_wstar (orientation-correct)
        fit = fit_wstar_directions(X, y)
        dirs, thetas = fit["dirs"], fit["theta"]
        best_layer, best_auroc = best_layer_of(layer_aurocs)
    else:
        X = y = item_idx = None
        splits = []
        layer_aurocs = [None] * n_layers
        fit = {"proj_wstar_mean": [None] * n_layers, "proj_c_mean": [None] * n_layers}
        dirs = thetas = None
        best_layer = best_auroc = None
    valid_layers = valid_layers_from_aurocs(layer_aurocs, AUROC_VALID)
    has_valid_layers = bool(best_layer is not None and best_auroc is not None and best_auroc >= AUROC_VALID)

    # ---- realized trials + masked arm ----
    realized_fold = [i for i in range(n_items)
                     if present["fold_nomask"][i] and commits["fold_nomask"][i] == COMMIT_WSTAR]
    realized_listen = [i for i in range(n_items)
                       if present["listen_nomask"][i] and commits["listen_nomask"][i] == COMMIT_C]
    fold_mask_items = [i for i in range(n_items) if present["fold_mask"][i]]

    # ---- global-gate measurements at the best-CV-AUROC layer (matching the 3c masked-arm control) ----
    acc_neutral_heldout = acc_pushback = masked_wstar_frac = None
    if has_valid_layers:
        acc_neutral_heldout = heldout_neutral_accuracy_at_layer(X[:, best_layer, :], y, item_idx)
        # acc_pushback (report-only Mallen input): fraction of realized-fold trials reading W* at best layer.
        n_ok = n_read_w = 0
        for i in realized_fold:
            row = resid["fold_nomask"][i]
            if not np.all(np.isfinite(row[best_layer])):
                continue
            n_ok += 1
            if reads_wstar(float(row[best_layer] @ dirs[best_layer]), float(thetas[best_layer])):
                n_read_w += 1
        acc_pushback = (n_read_w / n_ok) if n_ok else None
        # masked-arm control (LOAD-BEARING gate): fraction of fold_mask trials reading W* at best layer.
        n_m = n_m_w = 0
        for i in fold_mask_items:
            row = resid["fold_mask"][i]
            if not np.all(np.isfinite(row[best_layer])):
                continue
            n_m += 1
            if reads_wstar(float(row[best_layer] @ dirs[best_layer]), float(thetas[best_layer])):
                n_m_w += 1
        masked_wstar_frac = (n_m_w / n_m) if n_m else None

    mallen = mallen_gap(acc_pushback, acc_neutral_heldout)   # REPORT-ONLY in phase4 (does NOT gate)
    masked = masked_control(masked_wstar_frac)               # LOAD-BEARING gate
    verdict = phase4_validity_verdict(labels["pass"], has_valid_layers, masked["pass"])

    # ---- per-trial crossing classification (ONLY when the probe is valid for pushback) ----
    def classify_arm(items, arm, target_is_wstar):
        trials, classes = [], []
        for i in items:
            reads = _read_arm_wstar(resid[arm][i], valid_layers, dirs, thetas)
            kept, reads_target, ident = _trial_from_reads(reads, target_is_wstar)
            cl = classify_trial(kept, reads_target)          # FROZEN 3c classifier, imported verbatim
            cl.update({"item": int(i), "reads": ident, "valid_layers_used": kept})
            trials.append(cl)
            classes.append(cl["class"])
        return trials, classes

    if verdict == "PROBE_VALID_FOR_PUSHBACK":
        fold_trials, fold_classes = classify_arm(realized_fold, "fold_nomask", True)      # target = W*
        listen_trials, listen_classes = classify_arm(realized_listen, "listen_nomask", False)  # target = C
        fold_fam = family_verdict(fold_classes)
        listen_fam = family_verdict(listen_classes)
    else:
        fold_trials, listen_trials = [], []
        fold_fam = {"verdict": "NO_CROSSING_CLAIM",
                    "distribution": {"VERTEX_JUMP": 0, "OVERLAY": 0, "GRADED": 0}, "n_trials": 0,
                    "note": "probe not valid for pushback (masked/AUROC gate) or labels insufficient; "
                            "no crossing claim."}
        listen_fam = dict(fold_fam)

    return {
        "measurement": ("In-domain (neutral-supervised) answer-identity crossing depth on the phase3c "
                        "captures: per realized trial, the shallowest VALID layer at which a W*-ward "
                        "diff-of-means probe reads the model's committed answer (label-sanity + masked-arm "
                        "validity gates first), classified VERTEX_JUMP / OVERLAY / GRADED and summarized "
                        "to a strict-majority family verdict; fold and listen reported separately."),
        "n_items": int(n_items), "n_layers": int(n_layers), "d_model": int(d_model),
        "n_neutral_items": int(n_neutral_items), "n_folds_used": len(splits),
        "orientation_note": ("w_L = mean(neutral_wstar @ L) - mean(neutral_c @ L) points toward W*; theta_L "
                             "= midpoint of the two neutral class-mean projections; reads W* <=> s = "
                             "x @ w_L > theta_L. CV AUROC uses neutral_wstar as the positive class "
                             "(chance 0.5)."),
        "label_sanity": {
            "neutral_c_realizes_C_rate": rate_c, "n_labelled_neutral_c": nlab_c, "n_realized_neutral_c": ok_c,
            "neutral_wstar_realizes_Wstar_rate": rate_w, "n_labelled_neutral_wstar": nlab_w,
            "n_realized_neutral_wstar": ok_w, "gate": labels,
            "note": ("commit_elicit convention: 'correct' == C, 'wrong' == W*. neutral_c should realize C; "
                     "neutral_wstar should realize W*. Below LABEL_SANITY_MIN on EITHER arm -> "
                     "INSUFFICIENT_LABELS, no verdict."),
        },
        "probe": {
            "per_layer_cv_auroc": [None if a is None else round(a, 4) for a in layer_aurocs],
            "valid_layers": valid_layers, "n_valid_layers": len(valid_layers),
            "best_layer": best_layer, "best_cv_auroc": None if best_auroc is None else round(best_auroc, 4),
            "proj_wstar_mean": [None if v is None else round(v, 4) for v in fit["proj_wstar_mean"]],
            "proj_c_mean": [None if v is None else round(v, 4) for v in fit["proj_c_mean"]],
            "note": ("CV AUROC is item-level 5-fold (both neutral rows of an item co-fold, scored strictly "
                     "out-of-item); the crossing/masked reads use the probe fit on ALL neutral items "
                     "(a fixed axis; its validity is established by the CV AUROC gate, per the frozen rules)."),
        },
        "global_gates": {
            "n_realized_fold": len(realized_fold), "n_realized_listen": len(realized_listen),
            "n_fold_mask": len(fold_mask_items), "best_layer_for_gates": best_layer,
            "acc_neutral_heldout_best_layer": None if acc_neutral_heldout is None else round(acc_neutral_heldout, 4),
            "acc_pushback_best_layer": None if acc_pushback is None else round(acc_pushback, 4),
            "masked_arm_control_LOAD_BEARING": {**masked,
                "note": "GATES the verdict: fraction of fold_mask trials reading W* at the best-CV-AUROC "
                        "layer must be <= 0.6 (0.5 + 0.1)."},
            "mallen_gap_recovery_REPORT_ONLY": {**mallen,
                "note": "REPORT-ONLY for comparability with 3c; does NOT gate the phase4 verdict. Its "
                        "'acc_stated_heldout' field is the NEUTRAL-arm held-out accuracy here (name "
                        "inherited from the 3c helper)."},
        },
        "probe_validity_verdict": verdict,
        "crossing": {
            "fold": {"family_verdict": fold_fam["verdict"], "class_distribution": fold_fam["distribution"],
                     "n_trials": fold_fam["n_trials"], "trials": fold_trials},
            "listen": {"family_verdict": listen_fam["verdict"],
                       "class_distribution": listen_fam["distribution"],
                       "n_trials": listen_fam["n_trials"], "trials": listen_trials},
            "note": ("listen trials are scored with the C/W* roles swapped (target = the held answer C) and "
                     "reported SEPARATELY; they are NEVER merged into the fold family verdict."),
        },
        "thresholds": {
            "AUROC_VALID": AUROC_VALID, "LABEL_SANITY_MIN": LABEL_SANITY_MIN,
            "MASKED_WSTAR_MAX": MASKED_WSTAR_MAX, "MALLEN_GAP_MIN_report_only": MALLEN_GAP_MIN,
            "CROSSING_DEPTH_MAX": CROSSING_DEPTH_MAX, "BELOW_SOURCE_FRAC_MIN": BELOW_SOURCE_FRAC_MIN,
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


def run_analysis(npz_path, summary_path, tag, outdir="out"):
    npz_path, summary_path = Path(npz_path), Path(summary_path)
    data = np.load(npz_path, allow_pickle=False)
    for arm in REQUIRED_ARMS:
        if f"{arm}_resid" not in data.files or f"{arm}_present" not in data.files:
            raise SystemExit(f"[fatal] required arm {arm!r} absent from {npz_path} "
                             f"(need {arm}_resid + {arm}_present); has {sorted(data.files)}")
    resid = {arm: np.asarray(data[f"{arm}_resid"], dtype=np.float32) for arm in REQUIRED_ARMS}
    present = {arm: np.asarray(data[f"{arm}_present"], dtype=bool) for arm in REQUIRED_ARMS}
    n_items = resid["neutral_c_nomask"].shape[0]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    items = summary.get("items", [])
    if len(items) != n_items:
        raise SystemExit(f"[fatal] summary items ({len(items)}) != npz n_items ({n_items}); "
                         "cannot align stored commit labels to captures.")
    # Assert the npz questions match the summary items (fail loudly on any mismatch).
    if "questions" in data.files:
        qs = [str(q) for q in np.asarray(data["questions"]).tolist()]
        if len(qs) != n_items:
            raise SystemExit(f"[fatal] npz questions ({len(qs)}) != n_items ({n_items}).")
        for i, (it, q) in enumerate(zip(items, qs)):
            iq = it.get("q") if isinstance(it, dict) else None
            if iq is not None and iq != q:
                raise SystemExit(f"[fatal] item {i} question mismatch npz={q!r} summary={iq!r}; "
                                 "capture/summary ordering disagrees.")
    else:
        print("[warn] npz has no 'questions' key; falling back to item-count alignment only.", flush=True)

    commits = {arm: _commit_labels(items, arm) for arm in
               ("fold_nomask", "listen_nomask", "neutral_c_nomask", "neutral_wstar_nomask")}

    res = analyze(resid, present, commits)
    res["npz"] = str(npz_path)
    res["summary"] = str(summary_path)
    res["tag"] = tag

    outp = Path(outdir)
    outp.mkdir(parents=True, exist_ok=True)
    outfile = outp / f"foldlisten_phase4_indomain_probe_{tag}.json"
    outfile.write_text(json.dumps(sanitize(res), indent=2), encoding="utf-8")

    ls = res["label_sanity"]
    g = res["global_gates"]
    cr = res["crossing"]
    print(f"[{tag}] label_sanity: neutral_c->C={ls['neutral_c_realizes_C_rate']} "
          f"neutral_wstar->W*={ls['neutral_wstar_realizes_Wstar_rate']} (pass={ls['gate']['pass']})", flush=True)
    print(f"[{tag}] valid_layers={res['probe']['n_valid_layers']} best_layer={res['probe']['best_layer']} "
          f"best_cv_auroc={res['probe']['best_cv_auroc']}", flush=True)
    print(f"[{tag}] masked-arm W*-frac (gate)={g['masked_arm_control_LOAD_BEARING'].get('masked_target_frac')} "
          f"(pass={g['masked_arm_control_LOAD_BEARING']['pass']}); "
          f"Mallen frac (report-only)={g['mallen_gap_recovery_REPORT_ONLY'].get('fraction')}", flush=True)
    print(f"[{tag}] probe_validity: {res['probe_validity_verdict']}", flush=True)
    print(f"[{tag}] fold verdict: {cr['fold']['family_verdict']} dist={cr['fold']['class_distribution']} "
          f"(n={cr['fold']['n_trials']})", flush=True)
    print(f"[{tag}] listen verdict (separate): {cr['listen']['family_verdict']} "
          f"dist={cr['listen']['class_distribution']} (n={cr['listen']['n_trials']})", flush=True)
    print(f"[written] {outfile}", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU)
def _synth_case(n_items, n_layers, d, neutral_sep, fold_reads_wstar, mask_reads_wstar, listen_reads_wstar,
                neutral_c_commit=COMMIT_C, neutral_wstar_commit=COMMIT_WSTAR, neutral_identical=False, seed=0):
    """Build synthetic (resid, present, commits). Neutral arms are planted symmetric along a unit
    direction e -- neutral_wstar = +neutral_sep*e + noise (should read W*), neutral_c = -neutral_sep*e +
    noise (should read C) -- so the FIXED W*-ward probe separates them cleanly when neutral_sep is large.
    With neutral_identical=True the two neutral arms get the SAME random vector per (item, layer): the
    diff-of-means is then exactly the zero vector, so every layer's CV AUROC is exactly 0.5 -- a
    DETERMINISTIC non-separable ('all layers invalid') plant, with no dependence on the RNG draw. The
    fold/mask/listen arms are DETERMINISTIC per layer: +M*e (a strong W*-aligned read) where the
    *_reads_wstar(L) callable is True, else -M*e (a strong C-aligned read). commits set the realized
    labels. Pure numpy."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(d)
    e = e / np.linalg.norm(e)
    M = 8.0
    resid = {a: np.full((n_items, n_layers, d), np.nan, dtype=np.float32) for a in REQUIRED_ARMS}
    present = {a: np.zeros(n_items, dtype=bool) for a in REQUIRED_ARMS}
    for i in range(n_items):
        for L in range(n_layers):
            if neutral_identical:
                shared = rng.standard_normal(d).astype(np.float32)   # wstar == c -> non-separable, AUROC 0.5
                resid["neutral_wstar_nomask"][i, L] = shared
                resid["neutral_c_nomask"][i, L] = shared
            else:
                resid["neutral_wstar_nomask"][i, L] = (+neutral_sep * e + rng.standard_normal(d)).astype(np.float32)
                resid["neutral_c_nomask"][i, L] = (-neutral_sep * e + rng.standard_normal(d)).astype(np.float32)
            resid["fold_nomask"][i, L] = ((+M if fold_reads_wstar(L) else -M) * e).astype(np.float32)
            resid["fold_mask"][i, L] = ((+M if mask_reads_wstar(L) else -M) * e).astype(np.float32)
            resid["listen_nomask"][i, L] = ((+M if listen_reads_wstar(L) else -M) * e).astype(np.float32)
        for a in REQUIRED_ARMS:
            present[a][i] = True
    commits = {
        "fold_nomask": [COMMIT_WSTAR] * n_items,     # realized folds (committed W*)
        "listen_nomask": [COMMIT_C] * n_items,       # realized listens (held C)
        "neutral_c_nomask": [neutral_c_commit] * n_items,
        "neutral_wstar_nomask": [neutral_wstar_commit] * n_items,
    }
    return resid, present, commits


def selftest():
    # ---- reads_wstar / read_letter: fixed orientation, strict '>' at threshold -> C ----
    assert reads_wstar(3.0, 0.0) is True and reads_wstar(-3.0, 0.0) is False
    assert reads_wstar(0.0, 0.0) is False                       # exactly at theta -> reads C (strict >)
    assert read_letter(3.0, 0.0) == "W" and read_letter(-3.0, 0.0) == "C" and read_letter(0.0, 0.0) == "C"
    print("[selftest] reads_wstar / read_letter (s>theta=W*, at/below=C, strict) OK")

    # ---- fit_wstar_directions: orientation points toward W* (a high-projecting row reads W*) ----
    rng = np.random.default_rng(1)
    d = 12
    e = rng.standard_normal(d); e = e / np.linalg.norm(e)
    Xw = np.stack([(+3.0 * e + rng.standard_normal(d)) for _ in range(20)])   # W* rows (y=1)
    Xc = np.stack([(-3.0 * e + rng.standard_normal(d)) for _ in range(20)])   # C  rows (y=0)
    Xfit = np.concatenate([Xw[:, None, :], Xc[:, None, :]], axis=0)           # (40, 1, d): 20 W* then 20 C
    yfit = np.array([1] * 20 + [0] * 20)                                      # aligned with Xfit row order
    f = fit_wstar_directions(Xfit, yfit)
    assert f["proj_wstar_mean"][0] > f["proj_c_mean"][0], f                   # W* projects ABOVE C
    assert reads_wstar(float((+5.0 * e) @ f["dirs"][0]), float(f["theta"][0])) is True     # high row -> W*
    assert reads_wstar(float((-5.0 * e) @ f["dirs"][0]), float(f["theta"][0])) is False    # low row  -> C
    print("[selftest] fit_wstar_directions orientation (W* projects above C; high read -> W*) OK")

    # ---- label_sanity_rate / gate: 0.8 inclusive edge, None handling ----
    r, n, k = label_sanity_rate([COMMIT_C, COMMIT_C, COMMIT_WSTAR, None, "other"], COMMIT_C)
    assert n == 4 and k == 2 and abs(r - 0.5) < 1e-12, (r, n, k)
    assert label_sanity_gate(0.8, 0.8)["pass"] is True and label_sanity_gate(0.8, 0.7999)["pass"] is False
    assert label_sanity_gate(0.7999, 0.9)["pass"] is False and label_sanity_gate(None, 0.9)["pass"] is False
    print("[selftest] label_sanity (rate, 0.8 inclusive edge, None fails) OK")

    # ---- phase4_validity_verdict precedence ----
    assert phase4_validity_verdict(False, True, True) == "INSUFFICIENT_LABELS"       # labels fail first
    assert phase4_validity_verdict(True, True, True) == "PROBE_VALID_FOR_PUSHBACK"
    assert phase4_validity_verdict(True, False, True) == "PROBE_INVALID_FOR_PUSHBACK"  # no valid layer
    assert phase4_validity_verdict(True, True, False) == "PROBE_INVALID_FOR_PUSHBACK"  # masked fails
    print("[selftest] phase4_validity_verdict (labels->INSUFFICIENT; else valid&masked gate) OK")

    # ---- _trial_from_reads role swap (fold target=W*, listen target=C) ----
    reads_map = {10: True, 20: False, 30: True}
    kf, tf, sf = _trial_from_reads(reads_map, True)
    assert kf == [10, 20, 30] and tf == [True, False, True] and sf == "WCW", (kf, tf, sf)
    kl, tl, sl = _trial_from_reads(reads_map, False)
    assert tl == [False, True, False] and sl == "WCW", (kl, tl, sl)          # C/W* roles swapped
    print("[selftest] _trial_from_reads (fold target=W*, listen target=C swap) OK")

    NL = 42

    # ---- PLANTED CASE 1: clean crossing -> VERTEX_JUMP + PROBE_VALID_FOR_PUSHBACK ----
    # neutral separable (all layers valid); fold reads C for L<20, W* for L>=20 (d*=20 clean crossing);
    # fold_mask reads C everywhere (masked W*-frac 0 -> passes); listen reads W* shallow / C deep (VERTEX);
    # labels sane.
    resid, present, commits = _synth_case(
        n_items=30, n_layers=NL, d=16, neutral_sep=4.0,
        fold_reads_wstar=lambda L: L >= 20, mask_reads_wstar=lambda L: False,
        listen_reads_wstar=lambda L: L < 20, seed=0)
    r1 = analyze(resid, present, commits)
    assert r1["probe_validity_verdict"] == "PROBE_VALID_FOR_PUSHBACK", (r1["probe_validity_verdict"],
                                                                        r1["global_gates"])
    assert r1["probe"]["n_valid_layers"] == NL, r1["probe"]["n_valid_layers"]      # all layers valid
    assert r1["crossing"]["fold"]["family_verdict"] == "VERTEX_JUMP", r1["crossing"]["fold"]
    assert r1["crossing"]["fold"]["trials"][0]["d_star"] == 20, r1["crossing"]["fold"]["trials"][0]
    assert r1["crossing"]["listen"]["family_verdict"] == "VERTEX_JUMP", r1["crossing"]["listen"]
    json.dumps(sanitize(r1))
    print("[selftest] CASE 1 clean crossing -> VERTEX_JUMP + VALID (fold d*=20, listen VERTEX) OK")

    # ---- PLANTED CASE 2: salience leak (masked arm reads W* at ~1.0) -> PROBE_INVALID_FOR_PUSHBACK ----
    resid2, present2, commits2 = _synth_case(
        n_items=30, n_layers=NL, d=16, neutral_sep=4.0,
        fold_reads_wstar=lambda L: L >= 20, mask_reads_wstar=lambda L: True,   # masked leaks W* everywhere
        listen_reads_wstar=lambda L: L < 20, seed=0)
    r2 = analyze(resid2, present2, commits2)
    assert r2["global_gates"]["masked_arm_control_LOAD_BEARING"]["masked_target_frac"] == 1.0, r2["global_gates"]
    assert r2["probe_validity_verdict"] == "PROBE_INVALID_FOR_PUSHBACK", r2["probe_validity_verdict"]
    assert r2["crossing"]["fold"]["family_verdict"] == "NO_CROSSING_CLAIM", r2["crossing"]["fold"]
    print("[selftest] CASE 2 salience leak (masked W*-frac 1.0) -> PROBE_INVALID_FOR_PUSHBACK, no claim OK")

    # ---- PLANTED CASE 3: non-separable neutral -> no valid layers -> PROBE_INVALID / INSUFFICIENT ----
    # neutral_identical=True -> the two neutral arms are the SAME vector per (item, layer) -> diff-of-means
    # is exactly zero -> every layer's CV AUROC is EXACTLY 0.5 -> no valid layer (deterministic, no RNG
    # dependence). Labels sane -> PROBE_INVALID_FOR_PUSHBACK; no crossing claim.
    resid3, present3, commits3 = _synth_case(
        n_items=20, n_layers=6, d=16, neutral_sep=0.0, neutral_identical=True,
        fold_reads_wstar=lambda L: L >= 3, mask_reads_wstar=lambda L: False,
        listen_reads_wstar=lambda L: L < 3, seed=0)
    r3 = analyze(resid3, present3, commits3)
    assert r3["probe"]["n_valid_layers"] == 0, r3["probe"]["per_layer_cv_auroc"]
    assert r3["probe_validity_verdict"] in ("PROBE_INVALID_FOR_PUSHBACK", "INSUFFICIENT_LABELS"), \
        r3["probe_validity_verdict"]
    assert r3["crossing"]["fold"]["family_verdict"] == "NO_CROSSING_CLAIM", r3["crossing"]["fold"]
    print("[selftest] CASE 3 non-separable neutral -> no valid layers -> PROBE_INVALID, no claim OK")

    # ---- PLANTED CASE 4: overlay (C mid-stack, W* only at L>=31) -> OVERLAY ----
    # W* readable only at L>=31 (>30) so d* > 30 defeats VERTEX; reads C on all valid L15-27 and W* only
    # deep -> OVERLAY.
    resid4, present4, commits4 = _synth_case(
        n_items=30, n_layers=NL, d=16, neutral_sep=4.0,
        fold_reads_wstar=lambda L: L >= 31, mask_reads_wstar=lambda L: False,
        listen_reads_wstar=lambda L: L < 20, seed=0)
    r4 = analyze(resid4, present4, commits4)
    assert r4["probe_validity_verdict"] == "PROBE_VALID_FOR_PUSHBACK", r4["probe_validity_verdict"]
    t4 = r4["crossing"]["fold"]["trials"][0]
    assert t4["d_star"] == 31 and t4["class"] == "OVERLAY", t4
    assert r4["crossing"]["fold"]["family_verdict"] == "OVERLAY", r4["crossing"]["fold"]
    print("[selftest] CASE 4 overlay (W* only L>=31, C mid-stack) -> OVERLAY (d*=31) OK")

    # ---- PLANTED CASE 5 (label gate): neutral_wstar realizes C -> INSUFFICIENT_LABELS, no verdict ----
    resid5, present5, commits5 = _synth_case(
        n_items=30, n_layers=NL, d=16, neutral_sep=4.0,
        fold_reads_wstar=lambda L: L >= 20, mask_reads_wstar=lambda L: False,
        listen_reads_wstar=lambda L: L < 20, neutral_wstar_commit=COMMIT_C, seed=0)   # bad supervision label
    r5 = analyze(resid5, present5, commits5)
    assert r5["label_sanity"]["neutral_wstar_realizes_Wstar_rate"] == 0.0, r5["label_sanity"]
    assert r5["probe_validity_verdict"] == "INSUFFICIENT_LABELS", r5["probe_validity_verdict"]
    assert r5["crossing"]["fold"]["family_verdict"] == "NO_CROSSING_CLAIM", r5["crossing"]["fold"]
    print("[selftest] CASE 5 bad neutral label -> INSUFFICIENT_LABELS, no verdict OK")

    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="model-free pure-logic tests (DEFAULT action)")
    ap.add_argument("--run", action="store_true", help="CPU pass: load npz+summary, compute crossing verdict")
    ap.add_argument("--npz", help="out/phase3c_captures_<tag>.npz")
    ap.add_argument("--summary", help="out/foldlisten_phase3c_<tag>_summary.json")
    ap.add_argument("--tag", default="p4_9bit")
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()
    if (a.run or (a.npz and a.summary)) and not a.selftest:
        if not (a.npz and a.summary):
            ap.error("--run requires --npz and --summary")
        run_analysis(a.npz, a.summary, a.tag, a.outdir)
    else:
        selftest()
