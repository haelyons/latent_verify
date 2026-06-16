#!/bin/bash
# Re-run the chat-template-dependent jobs after the jinja2>=3.1 fix.
# ARC2A is already complete (no chat template) so it is not repeated here.
cd ~/arc2 || exit 1
rm -f DONE_CHAT
echo "[chat] $(date -u) start"
for j in job_arc2c_format_weights job_arc2b_numeric_it; do
  echo "[chat] ===== $j START $(date -u) ====="
  python3 $j.py
  echo "[chat] ===== $j END rc=$? $(date -u) ====="
done
echo "[chat] ALL DONE $(date -u)"
touch DONE_CHAT
