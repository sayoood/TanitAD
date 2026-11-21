"""24 navtest tokens (3 logs x 8) for the harness identity control: the control re-scores P2's BANKED seam (stage2_p2_full.npz), never a P6 checkpoint.
Tokens are the first 8 (sorted) of three logs picked deterministically (every 45th log of the sorted navtest log list). Reads the export through the same resolver the read path uses."""
import gzip
import json
import os
import sys

sys.path.insert(0, "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval")
import stage3_full as S3  # noqa: E402  (EXPORT resolved by the new root resolver)

OUT = "E:/Projects/TanitAD-artifacts/p6read_layout_fix/harness_control"
exp = json.load(gzip.open(S3.EXPORT, "rt", encoding="utf-8"))["tokens"]
by_log = {}
for t, v in exp.items():
    by_log.setdefault(v["log_name"], []).append(t)
logs = sorted(by_log)
pick = [logs[i] for i in (3, 48, 93)]
toks, tl = [], {}
for ln in pick:
    for t in sorted(by_log[ln])[:8]:
        toks.append(t)
        tl[t] = ln
json.dump({"tokens": toks, "token_log": tl}, open(f"{OUT}/tokens.json", "w", encoding="utf-8"), indent=1)
print(json.dumps({"export": S3.EXPORT, "n_logs_total": len(logs), "picked_logs": pick, "n_tokens": len(toks)}, indent=1))
