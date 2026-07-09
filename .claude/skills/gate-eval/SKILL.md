---
name: gate-eval
description: Run the full TanitAD gate evaluation (D1/D2/D3 + spectral) on the latest training checkpoint and report vs previous
---

Run the canonical checkpoint evaluation and report the gate ladder honestly (instruments first,
BLOCKED ≠ FAIL). Steps:

1. Launch on pod1 (detached; takes ~15 min alongside training):
   `ssh tanitad-pod 'cd /workspace/TanitAD/stack && nohup python scripts/evaluate_checkpoint.py --ckpt /workspace/experiments/p0-sB01-realmix/ckpt.pt --cache-dirs /workspace/data/comma2k19/_epcache /workspace/data/physicalai/_epcache --out /workspace/experiments/p0-sB01-realmix > /workspace/experiments/eval_latest.log 2>&1 & echo LAUNCHED'`
2. Arm a single-notification waiter: Bash run_in_background with
   `until ssh tanitad-pod 'ls /workspace/experiments/p0-sB01-realmix/gates_step*.json 2>/dev/null | tail -1 | grep -qv step5000'; do sleep 60; done` — adapt the grep to exclude ALL previously-known gates files, not just step5000.
3. On completion: fetch the newest gates_step*.json summary, compare every gate + diagnostic
   (I4, imag-rel, fit R², spectral knee) against the PREVIOUS evaluation row in
   `Benchmarks & Eval/LEADERBOARD.md`, update the leaderboard gate-ladder rows, commit+push,
   and report the table with trends. Send a PushNotification with the one-line verdict.
4. If any gate REGRESSED, flag it in bold and escalate per D-018 (PROJECT_STATE §4 + push).
