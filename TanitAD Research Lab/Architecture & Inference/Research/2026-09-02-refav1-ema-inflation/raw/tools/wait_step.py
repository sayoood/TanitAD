"""Block until <train_log.jsonl> carries a row with step >= N, or the run log shows a terminal
state (run_arm summary line / Traceback / OOM), or the deadline passes. Prints the rows seen.
usage: wait_step.py <train_log.jsonl> <step> <deadline_s> <run.log>"""
import json
import re
import sys
import time

log, step, deadline, runlog = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
TERM = re.compile(r"\[run_arm\]|Traceback|out of memory|OutOfMemory|non-finite", re.I)
t0 = time.time()
state = "deadline"
while time.time() - t0 < deadline:
    rows = []
    try:
        rows = [json.loads(l) for l in open(log, encoding="utf-8") if l.strip()]
    except FileNotFoundError:
        pass
    if rows and rows[-1]["step"] >= step:
        state = f"reached step {rows[-1]['step']}"
        break
    try:
        txt = open(runlog, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        txt = ""
    m = TERM.search(txt)
    if m:
        state = "TERMINAL: " + txt[m.start():m.start() + 300].replace("\n", " | ")
        break
    time.sleep(15)
print(f"[wait] {state} after {time.time() - t0:.0f}s at {time.strftime('%H:%M:%S')}")
try:
    rows = [json.loads(l) for l in open(log, encoding="utf-8") if l.strip()]
except FileNotFoundError:
    rows = []
keys = ("step", "loss", "loss_feat_op", "loss_feat_tac", "adapter_std", "tgt_std_op",
        "tgt_std_tac", "tgt_std_str", "participation", "grad_norm", "ema_decay", "elapsed_s")
for r in rows[-6:]:
    print({k: (round(r[k], 5) if isinstance(r.get(k), float) else r.get(k)) for k in keys})
if len(rows) >= 3:
    a, b = rows[1], rows[-1]
    sps = (b["elapsed_s"] - a["elapsed_s"]) / (b["step"] - a["step"])
    print(f"s/step ({a['step']}->{b['step']}): {sps:.2f}; 250 steps ~ {sps * 250 / 60:.1f} min/arm; "
          f"remaining for this arm ~ {sps * (250 - b['step']) / 60:.1f} min")
