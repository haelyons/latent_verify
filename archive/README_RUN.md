# Running the attribution-graph PoC on the Lambda GPU (Jupyter)

SSH from the Claude Code web session is blocked by an HTTPS-only egress proxy
(only ports 80/443, hostname-allowlisted, TLS-MITM — SSH cannot traverse it).
So the experiment is driven through GitHub instead: this branch carries the
verified code, you run it once in a Jupyter cell, results come back here.

## Pinned / verified

- circuit-tracer commit: `041a9b2cbd7f3fe7e0a625a6794e66fc4aa5f883`
- The three `# API` wrappers in `poc_minimal.py` were checked against that
  commit. Only fix applied: `get_activations` / `feature_intervention` return
  **3-D** logits `[batch, seq, vocab]`; both wrappers now take `logits[0, -1]`.
  Intervention tuple `(layer, position, feature_idx, value)` and the
  `[n_layers, seq, d]` activation shape were already correct.

## One Jupyter cell (authenticated clone — pushes results back automatically)

Replace `TOKEN` with a GitHub PAT that can read+write this repo:

```python
!cd ~ && rm -rf latent_verify && \
 git clone --branch claude/attribution-graphs-experiment-s16dze \
   https://x-access-token:TOKEN@github.com/haelyons/latent_verify && \
 cd latent_verify && bash run_poc.sh
```

The run prints both JSON files inline (between `BEGIN/END` markers) **and**
commits `out/t0.json`, `out/t1.json`, `out/run.log` back to the branch.

## No-token fallback

If you can't mint a PAT on mobile, clone read-only (works if the repo is
public) or copy the two JSON blobs from the cell output back to me:

```python
!cd ~ && rm -rf latent_verify && \
 git clone --branch claude/attribution-graphs-experiment-s16dze \
   https://github.com/haelyons/latent_verify && \
 cd latent_verify && bash run_poc.sh
```

## HF token

The run needs an HF token with the Gemma-2-2b licence accepted (for both the
model and the GemmaScope transcoders). If `run_poc.sh` warns it isn't
detected, run once in a cell, then re-run the clone cell:

```python
from huggingface_hub import login; login("hf_...")
```

## Notes

- t0 default multiplier sweep is `{0.0, -2.0}`; t1 transport multiplier
  defaults to `-2.0` and is recalibrated from t0's response curve.
- To override the t1 multiplier: prepend `TRANSPORT_M=-4.0 ` before
  `bash run_poc.sh`.
- The job is minutes, not hours; if you prefer, wrap with
  `nohup bash run_poc.sh &` and tail `out/run.log`.
