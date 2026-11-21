"""N for the `pred`/`random` arms, READ FROM A0's OWN LOG — the pre-registered rule.

H-EGO-LIT-4 §3.6: "warm-up on A0's bank for the first N steps (N = the step at
which the withheld-row speed MAE first drops below 2.5 m/s on A0's own log; if
never, N = 1/3 of the budget)". The per-step reading is noisy (batch 12, ~6
withheld rows), so the SPEC commits to the 5-row running mean of the logged
`withheld_speed_mae` (log-every 50 => a 250-step window). Prints N only.

usage: python warmup_from_log.py <metrics.jsonl> <budget_steps>
"""
import json
import sys

path, budget = sys.argv[1], int(sys.argv[2])
rows = []
with open(path, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "withheld_speed_mae" in r and "step" in r and "eval_loss" not in r:
            rows.append((int(r["step"]), float(r["withheld_speed_mae"])))
rows.sort()
n_default = budget // 3
chosen = n_default
for i in range(4, len(rows)):
    win = [m for _, m in rows[i - 4:i + 1]]
    if sum(win) / 5.0 < 2.5:
        chosen = rows[i][0]
        break
sys.stderr.write(f"[warmup] rows={len(rows)} first_5row_mean<2.5 at step "
                 f"{chosen if chosen != n_default else 'NEVER'} -> N={chosen}\n")
print(chosen)
