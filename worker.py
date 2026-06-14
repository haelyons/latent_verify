"""Persistent load-once worker. Loads the ReplacementModel a single time, then
runs jobs against the warm in-memory model so we stop paying the ~15-min cold
reload per experiment.

Protocol: write a python file path into the FIFO; the worker execs it with
`model`, `torch`, and the project helpers already in scope, then prints
DONE <path>. Send "QUIT" to stop.

    python worker.py            # in background; wait for '[worker] READY'
    echo /abs/path/job.py > /tmp/lv_jobs
"""
import os
import sys
import traceback

import torch

from poc_minimal import (load_model, logits_and_acts, intervened_logits,
                         rank_of, SEED_PROMPT, TEXAS)
import framing_probe
import framing_intervention

FIFO = "/tmp/lv_jobs"
if not os.path.exists(FIFO):
    os.mkfifo(FIFO)

print("[worker] loading model (one time)...", flush=True)
model = load_model()
print("[worker] READY", flush=True)

shared = {
    "model": model, "torch": torch,
    "logits_and_acts": logits_and_acts, "intervened_logits": intervened_logits,
    "rank_of": rank_of, "SEED_PROMPT": SEED_PROMPT, "TEXAS": TEXAS,
    "framing_probe": framing_probe, "framing_intervention": framing_intervention,
}

while True:
    with open(FIFO) as f:
        job = f.read().strip()
    if not job:
        continue
    if job == "QUIT":
        print("[worker] quitting", flush=True)
        break
    print(f"[worker] RUN {job}", flush=True)
    try:
        ns = dict(shared)
        exec(compile(open(job).read(), job, "exec"), ns)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
    print(f"[worker] DONE {job}", flush=True)
    sys.stdout.flush()
