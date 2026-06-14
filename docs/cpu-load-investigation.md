# CPU load investigation — gemma-2-2b + GemmaScope on the web sandbox

Purely descriptive record of attempts to load `google/gemma-2-2b` with the
GemmaScope per-layer transcoders (via `circuit-tracer`) inside the Claude Code
web execution sandbox, and the facts gathered about that environment. No
conclusions or recommendations — observations only.

## Goal of the session

Run `poc_minimal.py` (loads `ReplacementModel.from_pretrained("google/gemma-2-2b",
"gemma")`, then does forward passes and feature interventions) on CPU in the
sandbox, instead of on an external GPU.

## Environment facts observed

- **CPU / RAM:** 4 cores (Intel Xeon @ 2.10 GHz), 15 GiB RAM. No GPU
  (`nvidia-smi` absent).
- **Swap:** none at startup (`Swap: 0B`).
- **Disk:** root on `/dev/vda` (major:minor `254:0`). `df` reports a 252 G
  device but only ~37 G is usable; started at ~31 G free. The HuggingFace cache
  (~18 G) and swapfiles consumed most of it; free space fell to <1 G during the
  session.
- **Disk read speed:** cold (uncached) reads ~27 MB/s; warm reads (already in
  page cache) ~400 MB/s.
- **Python / pip:** Python 3.11.15, `pip` present, `torch` not preinstalled.
- **System setuptools is broken:** the Debian-packaged setuptools (68.1.2) has a
  patched `install_lib` that references a missing `install_layout`, so *any*
  legacy `setup.py` wheel build fails with
  `AttributeError: install_layout`. `pip` cannot replace it (Debian-managed, no
  RECORD file).
- **Network egress:** HTTPS works to `huggingface.co`, `pypi.org`,
  `github.com` (HTTP 200). `google/gemma-2-2b` returns HTTP 401 (gated) without
  a token; an HF token supplied during the session cleared the gate.
- **cgroups:** cgroup **v1**. The process sits in
  `memory:/process_api/<session-id>`. `memory.limit_in_bytes` is effectively
  unlimited by default; `memory.memsw.limit_in_bytes` is unlimited and could not
  be lowered (write denied), but `memory.limit_in_bytes` and
  `memory.swappiness` could be written. A `blkio` controller is also present.
- **Container restarts:** the sandbox restarts/reclaims the container. Each
  restart changes the `process_api` session id, wipes `swapon` state, resets the
  cgroup tweaks, clears page cache, and kills running background processes.
  Restarts were observed both during long idle gaps and ~2 min and ~28 min into
  active heavy-I/O runs.

## Weight file facts observed

- `gemma-2-2b` checkpoint is stored **float32** on disk: 3 shards totalling
  ~10.45 GB.
- GemmaScope transcoders (`mwhanna/gemma-scope-transcoders`): 26 files,
  ~7.4 GB total, dtype **F32**, `W_enc`/`W_dec` shape `[16384, 2304]`.
- Combined cache ~18 GB.

## Installation

1. Installed CPU `torch` 2.12.0+cpu from PyPI (succeeded).
2. `pip install circuit-tracer@041a9b2` failed: a transitive dependency
   (`transformers-stream-generator`, an old `setup.py` package pulled in by
   `transformer-lens`) failed to build a wheel because of the broken system
   setuptools described above.
3. A stub package for that dependency also failed to build — confirming the
   fault was the environment's setuptools, not the package.
4. **Resolved** by creating a virtualenv with `--system-site-packages` (fresh
   setuptools 82), then installing `circuit-tracer@041a9b2` into it
   successfully. `numpy` 2.4.6 present.

## Load attempts (chronological)

1. **First load:** global OOM (physical RAM) at anon-rss ~15.97 GB. The model
   checkpoint shards loaded, but loading the transcoders plus the float32
   conversion exceeded 15 GB RAM.
2. **Cause of the transient:** TransformerLens loads the HF model in float32
   (upcast from the bf16 forward path) and builds a second copy during
   conversion, producing a ~20 GB peak (total-vm 25–34 GB).
3. **Swapfile (8 GB) added; `swapon` succeeded**, but the process still OOM'd at
   ~15.5 GB with swap unused — the kernel did not swap under the fast allocation.
4. **cgroup inspection:** `memory.limit_in_bytes` was unlimited, so the kill was
   a *global* OOM, not a cgroup limit.
5. **Forced swap engagement:** set `memory.limit_in_bytes` below physical RAM
   (12 GB) with `memsw` unlimited and `swappiness=100`. A controlled test that
   touched 16 GB then survived by spilling to swap (RSS capped at 12.3 GB, swap
   used 4.6 GB).
6. `HF_HUB_OFFLINE=1` broke loading (TransformerLens still contacts the HF API
   for metadata even with cached weights); removed it.
7. **Idle restarts:** during long gaps between messages the container restarted,
   wiping swap and cgroup settings and killing the load mid-flight.
8. **Page-cache throttling:** with the 12 GB cap, cgroup v1 counts page cache
   against the limit, so reading the 18 GB of cold weights thrashed — shards took
   ~130 s each vs ~8 s warm. Raising the cap to 14 GB reduced this.
9. **Restart under active monitoring:** the container restarted ~28 min into a
   load while a heartbeat was actively running — i.e. restarts are not purely
   idle-triggered.
10. **bf16 path:** pre-loading the HF model in bfloat16 (`low_cpu_mem_usage`,
    passed via `hf_model=`) lowered the model+transcoder transient to ~14.6 GB.
    With a 14.5 GB cap and `swappiness=10`, it OOM'd (cgroup) at anon-rss
    14.59 GB — ~90 MB over the cap, because `swappiness=10` declined to swap.
11. With a 14.8 GB cap and `swappiness=100`, the container **restarted ~2 min**
    into the HF read (heavy disk I/O).
12. Reducing the swapfile (8 GB → 4 GB) to free disk did not change the
    restart-during-read behaviour.
13. **I/O-rate observation:** restarts consistently coincided with sustained
    heavy disk reads (the model/transcoder load phase).
14. **blkio read throttle:** created a child `blkio` cgroup limiting reads on
    `254:0` to 20 MB/s and ran the load inside it. The container **stayed up the
    full 44 min with no restart**, and the load read all ~18.5 GB to completion —
    the only configuration that got through the read phase without a restart.
15. With the throttle on, the run then OOM'd (cgroup) at anon-rss 14.84 GB
    against the 14.8 GB cap, during `ReplacementModel` construction. The cap was
    removed.
16. **Throttle + no cap:** the read phase completed, then the construction step
    grew to ~25 GB committed (anon-rss 15.3 GB resident + ~9.9 GB swap) and hit a
    global OOM. Swap was extended mid-run (4 → 10 GB across three swapfiles) but
    could not keep pace; disk filled to <1 G free.
17. **Construction memory measured:** building the `circuit-tracer`
    `ReplacementModel` peaks at ~25 GB committed (a float32 HookedTransformer
    intermediate on top of the bf16 model and the transcoders), while the
    steady-state model is ~9 GB.
18. **Interaction observed:** the `blkio` read throttle (needed to avoid the
    restart) also throttles swap-in from the swapfile on the same device, so
    paging a large working set during construction proceeds at ~20 MB/s.
19. **bf16 cache conversion attempt:** began converting the cached float32
    weights to bf16 in place (to free disk and reduce read volume). The atomic
    per-shard rewrite needs ~2.5 GB of temporary space, but only ~863 MB was free
    (the swapfiles occupied the disk); the conversion could not write its
    temporary files. This is where the session was interrupted.

## Key numbers gathered

| Item | Value |
| --- | --- |
| RAM | 15 GiB, no GPU |
| Cold disk read | ~27 MB/s |
| gemma-2-2b on disk | float32, ~10.45 GB (3 shards) |
| GemmaScope transcoders | F32, ~7.4 GB (26 files) |
| HF model load peak RSS | ~9.62 GB |
| float32 load transient | ~16–20 GB (global OOM ~15.5–16 GB) |
| bf16 model+transcoder transient | ~14.6 GB |
| ReplacementModel construction peak | ~25 GB committed (≈15.3 GB RSS + ~9.9 GB swap) |
| Steady-state loaded model | ~9 GB |
| Throttle that avoided restart | 20 MB/s read on `254:0`; container up 44 min |

## State at interruption

- circuit-tracer + CPU torch installed in a venv at `/home/user/.venv_lv`.
- HF weights + transcoders cached (float32) under `~/.cache/huggingface`.
- Three swapfiles present (`/swapfile`, `/swapfile2`, `/swapfile3`, ~10 GB
  total); ~863 MB disk free.
- No successful full load of `ReplacementModel`; no `out/` results produced.
