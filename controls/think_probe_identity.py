"""ANSWER-IDENTITY linear probe -- an INSTRUMENT VALIDATION, not a behavioural claim.

WHY (neutral). This control measures ONE quantity: whether a model's residual stream linearly encodes
WHICH of two candidate answers an in-context stated answer names, out-of-item. For each family item
{q, correct, Wstar} it builds two contexts -- [user: q][assistant: "{correct}."] and
[user: q][assistant: "{Wstar}."] -- reads resid_post at the LAST token of the tokenized context at EVERY
layer, and asks, per layer, whether a difference-of-means direction fit on OTHER items separates the two
(y=1 when the stated answer is `correct`, y=0 when it is `Wstar`) on HELD-OUT items. The number reported is
a held-out AUROC and two chance floors (label-permutation, random-direction). Nothing here scores model
behaviour or any downstream effect; it certifies whether the residual carries a linearly-readable
answer-identity signal, and lets the AUROC fall where it does.

Item-level K-fold cross-validation: folds split by ITEM index, so BOTH contexts of an item are always in
the same fold -- the probe is scored strictly out-of-item, never on a context it saw during the fit.

DECISION (module constants, inclusive boundaries; stated on the measured numbers only, no claim attached):
  AUROC_THR = 0.8, FLOOR_MAX = 0.6, MIN_ITEMS = 20.
  INSUFFICIENT  iff n_items < MIN_ITEMS (checked first).
  PROBE_VALID   iff heldout_auroc_best >= AUROC_THR AND perm_floor <= FLOOR_MAX AND rand_floor <= FLOOR_MAX.
  PROBE_INVALID otherwise.

Modes: --selftest (model-free, numpy only); --capture (GPU pass, writes the residual npz + sidecar;
torch/transformer_lens imported ONLY here); --fit (CPU pass, pure numpy over the npz). Greedy/deterministic
throughout; no sampling; numpy RNG seeded (np.random.default_rng(SEED)) so --fit is reproducible.

  python controls/think_probe_identity.py --selftest
  python controls/think_probe_identity.py --capture --family verifier_family --name google/gemma-2-9b-it --tag tp_9bit --device cuda --chat
  python controls/think_probe_identity.py --capture --family verifier_family --name google/gemma-2-9b --tag tp_9bbase --device cuda
  python controls/think_probe_identity.py --fit out/think_probe_capture_tp_9bit.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# FLAT-scp: controls/ for sibling reuse, latent_verify/ for the repo imports (identical to foldlisten_judge.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Pre-registered constants (neutral: stated on the measured numbers only).
AUROC_THR = 0.8       # heldout answer-identity AUROC at/above which the probe reads the signal
FLOOR_MAX = 0.6       # both chance floors must sit at/below this
MIN_ITEMS = 20        # below this the split cannot certify anything -> INSUFFICIENT
N_FOLDS = 5           # item-level cross-validation folds
N_PERM = 200          # label-permutation floor repeats
N_RAND = 200          # random-direction floor repeats
SEED = 0              # numpy RNG seed for folds, permutations and random directions
RESID_HOOK = "blocks.{L}.hook_resid_post"


# --------------------------------------------------------------------------- pure fit math (numpy only)
def auroc(scores, labels):
    """Rank-based AUROC of `scores` vs binary `labels` (ties = 0.5 weight). None if a class is empty.
    Pure numpy (arrays or lists in)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    diff = pos[:, None] - neg[None, :]
    c = np.count_nonzero(diff > 0) + 0.5 * np.count_nonzero(diff == 0)
    return float(c / (pos.size * neg.size))


def diff_of_means(X_layer, y):
    """mean(rows y==1) - mean(rows y==0) at one layer. X_layer (m, d), y (m,). Returns (d,). Pure numpy."""
    X_layer = np.asarray(X_layer, dtype=float)
    y = np.asarray(y)
    return X_layer[y == 1].mean(0) - X_layer[y == 0].mean(0)


def kfold_item_splits(item_idx, k=N_FOLDS, seed=SEED):
    """K near-equal folds over the UNIQUE item indices (shuffled by np.random.default_rng(seed), striped).
    Returns a list of (train_rows, test_rows) row-index arrays such that BOTH rows of any item land in the
    SAME fold (splits are by item, not by row). k is capped at n_items. Pure."""
    item_idx = np.asarray(item_idx)
    items = np.unique(item_idx)
    n = len(items)
    if n < 2:
        return []
    k = max(2, min(k, n))
    perm = items.copy()
    np.random.default_rng(seed).shuffle(perm)
    folds = [perm[i::k] for i in range(k)]
    folds = [f for f in folds if len(f) > 0]
    out = []
    for f in folds:
        test_items = set(int(x) for x in f)
        test_rows = np.array([i for i, it in enumerate(item_idx) if int(it) in test_items], dtype=int)
        train_rows = np.array([i for i, it in enumerate(item_idx) if int(it) not in test_items], dtype=int)
        if train_rows.size and test_rows.size:
            out.append((train_rows, test_rows))
    return out


def pooled_heldout(X_layer, y, splits, direction_fn):
    """For each (train, test) fold: w = direction_fn(X_layer[train], y[train]); score test rows
    s = X_layer[test] @ w; pool scores + labels across all folds. `direction_fn(Xt, yt) -> (d,)` may
    ignore its args (e.g. a fixed random direction). Returns (pooled_scores, pooled_labels). Pure numpy."""
    X_layer = np.asarray(X_layer, dtype=float)
    y = np.asarray(y)
    ss, ll = [], []
    for train, test in splits:
        w = direction_fn(X_layer[train], y[train])
        ss.append(X_layer[test] @ w)
        ll.append(y[test])
    if not ss:
        return np.array([]), np.array([])
    return np.concatenate(ss), np.concatenate(ll)


def per_layer_auroc(X, y, splits):
    """Held-out AUROC at each layer: diff-of-means direction fit per train fold, pooled test scores, one
    rank AUROC per layer. Returns a list (len n_layers) of AUROC or None. Pure numpy. X (n, n_layers, d)."""
    X = np.asarray(X, dtype=float)
    n_layers = X.shape[1]
    out = []
    for l in range(n_layers):
        s, lab = pooled_heldout(X[:, l, :], y, splits, diff_of_means)
        out.append(auroc(s, lab))
    return out


def best_layer_of(layer_aurocs):
    """(layer, auroc) with the largest AUROC (None entries excluded), or (None, None). Pure."""
    avail = [(l, a) for l, a in enumerate(layer_aurocs) if a is not None]
    if not avail:
        return None, None
    l, a = max(avail, key=lambda t: t[1])
    return l, a


def perm_floor(X_layer, y, item_idx, splits, n_perm=N_PERM, seed=SEED):
    """Label-permutation floor at one layer: mean held-out AUROC over n_perm permutations of the labels
    WITHIN the same fold structure. Each permutation flips a random subset of ITEMS -- both rows of a
    flipped item swap labels consistently (one y=1 + one y=0 preserved per item) -- so the diff-of-means
    contrast is destroyed while the fold structure and class balance are kept. Pure numpy."""
    X_layer = np.asarray(X_layer, dtype=float)
    y = np.asarray(y)
    item_idx = np.asarray(item_idx)
    items = np.unique(item_idx)
    rng = np.random.default_rng(seed)
    aus = []
    for _ in range(n_perm):
        flip = {int(it): bool(b) for it, b in zip(items, rng.integers(0, 2, len(items)))}
        yp = np.array([(1 - y[i]) if flip[int(item_idx[i])] else y[i] for i in range(len(y))])
        s, lab = pooled_heldout(X_layer, yp, splits, diff_of_means)
        a = auroc(s, lab)
        if a is not None:
            aus.append(a)
    return float(np.mean(aus)) if aus else None


def rand_floor(X_layer, y, splits, n_rand=N_RAND, seed=SEED):
    """Random-direction floor at one layer: mean held-out AUROC over n_rand random unit directions applied
    (no training) to the SAME held-out scores protocol -- true labels, true fold structure, fixed random
    direction in place of the trained diff-of-means. Pure numpy."""
    X_layer = np.asarray(X_layer, dtype=float)
    d = X_layer.shape[1]
    rng = np.random.default_rng(seed + 1)  # distinct stream from perm_floor
    aus = []
    for _ in range(n_rand):
        r = rng.standard_normal(d)
        r = r / (np.linalg.norm(r) + 1e-12)
        s, lab = pooled_heldout(X_layer, y, splits, lambda Xt, yt, r=r: r)
        a = auroc(s, lab)
        if a is not None:
            aus.append(a)
    return float(np.mean(aus)) if aus else None


# --------------------------------------------------------------------------- pure decision
def decide(n_items, heldout_auroc_best, perm_floor_v, rand_floor_v,
          auroc_thr=AUROC_THR, floor_max=FLOOR_MAX, min_items=MIN_ITEMS):
    """Neutral category over the measured numbers only (no behavioural claim). Pure.
      INSUFFICIENT  iff n_items < min_items (checked first).
      PROBE_VALID   iff heldout_auroc_best >= auroc_thr AND perm_floor <= floor_max AND rand_floor <= floor_max.
      PROBE_INVALID otherwise. Boundaries inclusive."""
    if n_items < min_items:
        return {"category": "INSUFFICIENT",
                "msg": f"n_items={n_items} < MIN_ITEMS({min_items}); the item-level split cannot certify the "
                       f"probe.",
                "heldout_auroc_best": heldout_auroc_best, "perm_floor": perm_floor_v,
                "rand_floor": rand_floor_v}
    a_ok = heldout_auroc_best is not None and heldout_auroc_best >= auroc_thr
    p_ok = perm_floor_v is not None and perm_floor_v <= floor_max
    r_ok = rand_floor_v is not None and rand_floor_v <= floor_max
    if a_ok and p_ok and r_ok:
        cat = "PROBE_VALID"
        msg = (f"heldout_auroc_best={heldout_auroc_best:.4f} >= {auroc_thr} AND "
               f"perm_floor={perm_floor_v:.4f} <= {floor_max} AND rand_floor={rand_floor_v:.4f} <= {floor_max}.")
    else:
        cat = "PROBE_INVALID"
        why = []
        why.append(f"heldout_auroc_best={heldout_auroc_best if heldout_auroc_best is None else round(heldout_auroc_best, 4)} "
                   f"{'>=' if a_ok else '<'} {auroc_thr}")
        why.append(f"perm_floor={perm_floor_v if perm_floor_v is None else round(perm_floor_v, 4)} "
                   f"{'<=' if p_ok else '>'} {floor_max}")
        why.append(f"rand_floor={rand_floor_v if rand_floor_v is None else round(rand_floor_v, 4)} "
                   f"{'<=' if r_ok else '>'} {floor_max}")
        msg = "; ".join(why) + " -- not all of (AUROC>=THR, perm<=FLOOR, rand<=FLOOR) held."
    return {"category": cat, "heldout_auroc_best": heldout_auroc_best,
            "perm_floor": perm_floor_v, "rand_floor": rand_floor_v, "msg": msg}


def fit_arrays(X, y, item_idx):
    """Full CPU fit over in-memory arrays (shared by --fit and --selftest). Pure numpy. Returns the result
    dict: per-layer heldout AUROCs, best layer, both floors (at the best layer only), and the decision.
    X (n_examples, n_layers, d_model); y (n_examples,) in {0,1}; item_idx (n_examples,)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    item_idx = np.asarray(item_idx)
    n_items = int(len(np.unique(item_idx)))
    splits = kfold_item_splits(item_idx, N_FOLDS, SEED)
    layer_aurocs = per_layer_auroc(X, y, splits)
    best_layer, best_auroc = best_layer_of(layer_aurocs)
    pf = rf = None
    if n_items >= MIN_ITEMS and best_layer is not None:
        Xb = X[:, best_layer, :]
        pf = perm_floor(Xb, y, item_idx, splits)
        rf = rand_floor(Xb, y, splits)
    dec = decide(n_items, best_auroc, pf, rf)
    return {
        "measurement": ("answer-identity linear probe: item-level held-out AUROC of a diff-of-means "
                        "direction predicting whether an in-context stated answer names the correct (y=1) "
                        "vs the Wstar (y=0) candidate, read from resid_post at the last context token, per layer"),
        "n_items": n_items, "n_examples": int(X.shape[0]),
        "n_layers": int(X.shape[1]), "d_model": int(X.shape[2]),
        "thresholds": {"AUROC_THR": AUROC_THR, "FLOOR_MAX": FLOOR_MAX, "MIN_ITEMS": MIN_ITEMS,
                       "N_FOLDS": N_FOLDS, "N_PERM": N_PERM, "N_RAND": N_RAND, "SEED": SEED},
        "n_folds_used": len(splits),
        "per_layer_auroc": [None if a is None else round(a, 4) for a in layer_aurocs],
        "best_layer": best_layer,
        "heldout_auroc_best": None if best_auroc is None else round(best_auroc, 4),
        "perm_floor": None if pf is None else round(pf, 4),
        "rand_floor": None if rf is None else round(rf, 4),
        "decision": dec,
    }


# --------------------------------------------------------------------------- family loading
def load_family(family):
    """`--family verifier_family[_ext]` -> the module ITEMS; else treat `family` as a JSON list of
    {q, correct, Wstar, ..}. (Same idiom as foldlisten_judge.load_family.)"""
    if family == "verifier_family":
        from verifier_family import ITEMS
        return list(ITEMS)
    if family == "verifier_family_ext":
        from verifier_family_ext import ITEMS
        return list(ITEMS)
    return json.loads(Path(family).read_text())


# --------------------------------------------------------------------------- capture (torch/TL only here)
def _resid_all_layers(model, ids):
    """resid_post at the LAST token, EVERY layer, one forward. Returns (n_layers, d_model) float32 numpy.
    torch is imported inside this function so it stays confined to the capture path."""
    import torch
    store = {}

    def grab(t, hook):
        store[hook.layer()] = t[0, -1].detach().float().cpu().numpy()
        return t

    nL = model.cfg.n_layers
    with torch.no_grad():
        model.run_with_hooks(ids, fwd_hooks=[(RESID_HOOK.format(L=L), grab) for L in range(nL)])
    return np.stack([store[L] for L in range(nL)], axis=0).astype(np.float32)


def capture(family, name, tag, device, is_chat, n):
    import torch
    from transformer_lens import HookedTransformer

    items = load_family(family)
    if n:
        items = items[:n]
    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tok = model.tokenizer
    nL, d_model = model.cfg.n_layers, model.cfg.d_model

    def chat_ctx(q, A):
        """[user: q][assistant: "{A}."] tokenized via the chat template; the assistant turn is CLOSED
        (add_generation_prompt=False) so the LAST token is the end of the just-stated-answer turn."""
        ids = tok.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": f"{A}."}],
            add_generation_prompt=False, return_tensors="pt")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        return ids.to(device)

    def qa_ctx(q, A):
        return model.to_tokens(f"Q: {q}\nA: {A}.", prepend_bos=True).to(device)

    build = chat_ctx if is_chat else qa_ctx
    note = ("chat: apply_chat_template([user:q, assistant:'{A}.'], add_generation_prompt=False); resid_post "
            "at the LAST token of the assistant-final context, every layer"
            if is_chat else
            "qa: 'Q: {q}\\nA: {A}.'; resid_post at the LAST token, every layer")

    X_rows, y_rows, item_rows = [], [], []
    print(f"[family] {family} -> {len(items)} items x 2 contexts (correct/Wstar); every context captured",
          flush=True)
    for i, it in enumerate(items):
        q, C, W = it["q"], it["correct"], it["Wstar"]
        for A, lab in ((C, 1), (W, 0)):
            r = _resid_all_layers(model, build(q, A))
            X_rows.append(r)
            y_rows.append(lab)
            item_rows.append(i)
        print(f"  [{i:03d}] captured 2 ctxs (y=1:{C!r}, y=0:{W!r}) q={q[:40]!r}", flush=True)

    X = np.stack(X_rows, axis=0).astype(np.float32)   # (n_examples, n_layers, d_model)
    y = np.array(y_rows, dtype=np.int64)
    item_idx = np.array(item_rows, dtype=np.int64)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    outdir = Path("out")
    outdir.mkdir(parents=True, exist_ok=True)
    npz = outdir / f"think_probe_capture_{tag}.npz"
    np.savez(npz, X=X, y=y, item_idx=item_idx)
    side = {"model": name, "family": family, "n_items": len(items), "n_examples": int(X.shape[0]),
            "n_layers": int(nL), "d_model": int(d_model), "regime": "chat" if is_chat else "qa",
            "context_template_note": note}
    (outdir / f"think_probe_capture_{tag}.json").write_text(json.dumps(side, indent=2))
    print(f"[written] {npz}  X={X.shape} y={y.shape} item_idx={item_idx.shape}", flush=True)
    print(f"[written] {outdir / f'think_probe_capture_{tag}.json'}", flush=True)


def fit(npz_path):
    npz_path = Path(npz_path)
    data = np.load(npz_path)
    res = fit_arrays(data["X"], data["y"], data["item_idx"])
    res["npz"] = str(npz_path)
    side = npz_path.with_suffix(".json")
    if side.exists():
        res["capture_sidecar"] = json.loads(side.read_text())
    tag = npz_path.stem.replace("think_probe_capture_", "")
    outp = npz_path.parent / f"think_probe_fit_{tag}.json"
    outp.write_text(json.dumps(res, indent=2))
    d = res["decision"]
    print(f"[fit] per_layer_auroc={res['per_layer_auroc']}", flush=True)
    print(f"[fit] best_layer={res['best_layer']} heldout_auroc_best={res['heldout_auroc_best']} "
          f"perm_floor={res['perm_floor']} rand_floor={res['rand_floor']} (n_items={res['n_items']})", flush=True)
    print(f"[{tag}] {d['category']}: {d['msg']}", flush=True)
    print(f"[written] {outp}", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU, numpy only)
def _synth(n_items, d, n_layers, planted_layer, sep, noise, seed=0):
    """Synthetic residual cache: two rows per item (y=1, y=0), item_idx repeated. Every layer is pure
    N(0, noise) noise; at `planted_layer` (if not None) a signed sep*e is added along a planted unit
    direction e (+ for y=1, - for y=0). sep >> noise -> that layer separates cleanly; other layers do not.
    Returns (X (2*n_items, n_layers, d) float32, y, item_idx). Pure numpy."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(d)
    e = e / np.linalg.norm(e)
    X, y, item = [], [], []
    for i in range(n_items):
        for lab in (1, 0):
            x = rng.standard_normal((n_layers, d)) * noise
            if planted_layer is not None:
                x[planted_layer] += (1.0 if lab == 1 else -1.0) * sep * e
            X.append(x)
            y.append(lab)
            item.append(i)
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(item, dtype=np.int64))


def selftest():
    # --- AUROC unit checks: perfectly separated -> 1.0; anti-separated -> 0.0; all-equal -> 0.5; ties=0.5 ---
    assert auroc([3, 4, 5, 0, 1, 2], [1, 1, 1, 0, 0, 0]) == 1.0
    assert auroc([0, 1, 2, 3, 4, 5], [1, 1, 1, 0, 0, 0]) == 0.0
    assert auroc([2, 2, 2, 2], [1, 1, 0, 0]) == 0.5
    assert auroc([5, 5, 5], [1, 1, 1]) is None            # a class empty -> None
    print("[selftest] auroc unit: sep=1.0 / anti=0.0 / all-equal=0.5 / ties=0.5 OK")

    # --- item-level split honored: both rows of any item share a fold; test folds partition the rows ---
    _, _, item_idx = _synth(n_items=40, d=8, n_layers=1, planted_layer=None, sep=0.0, noise=1.0, seed=7)
    splits = kfold_item_splits(item_idx, N_FOLDS, SEED)
    assert len(splits) == N_FOLDS, len(splits)
    all_test = []
    for train, test in splits:
        assert set(item_idx[train].tolist()).isdisjoint(set(item_idx[test].tolist()))  # item disjoint
        all_test += test.tolist()
    assert sorted(all_test) == list(range(len(item_idx)))  # test folds partition the rows
    for it in np.unique(item_idx):
        rows_of = set(np.where(item_idx == it)[0].tolist())
        hits = [fi for fi, (_, te) in enumerate(splits) if rows_of & set(te.tolist())]
        assert len(hits) == 1, (it, hits)                  # item lands in exactly one fold
        assert rows_of <= set(splits[hits[0]][1].tolist()), it   # ...and BOTH its rows are there
    print("[selftest] item-level split: both rows co-fold, folds partition, items disjoint OK")

    # --- planted dataset: layer 1 separates (AUROC>=0.95), layers 0 & 2 noise ([0.3,0.7]); PROBE_VALID ---
    X, y, item = _synth(n_items=40, d=32, n_layers=3, planted_layer=1, sep=6.0, noise=1.0, seed=0)
    res = fit_arrays(X, y, item)
    pla = res["per_layer_auroc"]
    assert pla[1] is not None and pla[1] >= 0.95, pla
    for L in (0, 2):
        assert 0.3 <= pla[L] <= 0.7, (L, pla)
    assert res["best_layer"] == 1, res["best_layer"]
    assert 0.4 <= res["perm_floor"] <= 0.6, res["perm_floor"]
    assert 0.3 <= res["rand_floor"] <= 0.7, res["rand_floor"]
    assert res["decision"]["category"] == "PROBE_VALID", res["decision"]
    print(f"[selftest] planted: layer_auroc={pla} best={res['best_layer']} "
          f"perm={res['perm_floor']} rand={res['rand_floor']} -> PROBE_VALID")

    # --- pure-noise dataset (all layers noise) -> PROBE_INVALID (AUROC never clears the threshold) ---
    Xn, yn, itn = _synth(n_items=40, d=32, n_layers=3, planted_layer=None, sep=0.0, noise=1.0, seed=3)
    resn = fit_arrays(Xn, yn, itn)
    assert resn["heldout_auroc_best"] < AUROC_THR, resn
    assert resn["decision"]["category"] == "PROBE_INVALID", resn["decision"]
    print(f"[selftest] pure noise: best_auroc={resn['heldout_auroc_best']} -> PROBE_INVALID")

    # --- too few items -> INSUFFICIENT (checked before any AUROC threshold) ---
    Xf, yf, itf = _synth(n_items=10, d=32, n_layers=3, planted_layer=1, sep=6.0, noise=1.0, seed=1)
    resf = fit_arrays(Xf, yf, itf)
    assert resf["n_items"] == 10 and resf["decision"]["category"] == "INSUFFICIENT", resf["decision"]
    print(f"[selftest] n_items=10 -> INSUFFICIENT")

    # --- decide() boundaries (inclusive) on explicit numbers ---
    assert decide(20, 0.80, 0.60, 0.60)["category"] == "PROBE_VALID"       # all boundaries inclusive
    assert decide(20, 0.80, 0.60, 0.61)["category"] == "PROBE_INVALID"     # rand just over
    assert decide(20, 0.80, 0.61, 0.60)["category"] == "PROBE_INVALID"     # perm just over
    assert decide(20, 0.79, 0.50, 0.50)["category"] == "PROBE_INVALID"     # auroc just under
    assert decide(19, 0.99, 0.50, 0.50)["category"] == "INSUFFICIENT"      # n_items checked first
    assert decide(30, None, None, None)["category"] == "PROBE_INVALID"     # missing numbers -> invalid
    print("[selftest] decide boundaries (inclusive) PROBE_VALID/INVALID/INSUFFICIENT OK")
    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--capture", action="store_true", help="GPU pass: capture resid_post npz + sidecar")
    ap.add_argument("--fit", metavar="CAPTURE_NPZ", help="CPU pass (pure numpy) over a capture npz")
    ap.add_argument("--family", default="verifier_family")
    ap.add_argument("--name", default="google/gemma-2-9b-it")
    ap.add_argument("--tag", default="tp_9bit")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="cap items (0 = all)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.fit:
        fit(a.fit)
    elif a.capture:
        capture(a.family, a.name, a.tag, a.device, a.chat, a.n)
    else:
        ap.error("one of --selftest / --capture / --fit is required")
