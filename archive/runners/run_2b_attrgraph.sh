#!/usr/bin/env bash
# On-box: FOUNDING-METHOD attribution graph for caving (the NODE/feature basis -- the opposite of the
# direction controls). circuit-tracer + GemmaScope-2b transcoders on gemma-2-2b BASE (Q/A). The control
# selects ONE faithful argmax-W* caving instance (layer-independent: PUSH/NEUTRAL, argmax==W*-first-tok),
# builds the feature-level attribution graph for the realized caving logit-diff (W*-C at the answer slot),
# prunes by influence, splits feature vs error/residual influence (completeness), and validates by a
# top-k-vs-matched-random feature clamp-off ablation. Decision: SPARSE_CIRCUIT / BROAD_DISTRIBUTED /
# INCOMPLETE / NO_FAITHFUL_INSTANCE / TOOLING_UNAVAILABLE. Graceful (exit 0) if no target or tracer missing.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest (model-free, no tracer) ==="
python cave_attribution_graph.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== install circuit-tracer (PyPI, then git fallback) ==="
pip install -q circuit-tracer 2>&1 | tail -4 \
  || pip install -q "git+https://github.com/safety-research/circuit-tracer.git" 2>&1 | tail -4 \
  || echo "PIP_CIRCUIT_TRACER_FAILED (control will report TOOLING_UNAVAILABLE)"
python -c "import circuit_tracer, inspect; print('[ct]', circuit_tracer.__file__)" 2>&1 | tail -2 || echo "[ct] not importable"

echo "=== 2b attribution graph (base qa) ==="
python cave_attribution_graph.py --name google/gemma-2-2b --tag 2b --device cuda > out/cave_attribution_graph_2b.log 2>&1
echo "exit=$?"
echo "--- tail cave_attribution_graph_2b.log ---"; tail -70 out/cave_attribution_graph_2b.log
echo "ALLDONE_ATTRGRAPH_2B"
