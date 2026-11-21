"""control B: 408 navtest tokens (17 logs x up to 24; every 8th log of the sorted 136) for the harness identity control (P2's banked seam re-scored on today's layout)."""
import gzip
import json
import sys

sys.path.insert(0, "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval")
import stage3_full as S3  # noqa: E402

OUT = "E:/Projects/TanitAD-artifacts/p6read_layout_fix/harness_control"
exp = json.load(gzip.open(S3.EXPORT, "rt", encoding="utf-8"))["tokens"]
by_log = {}
for t, v in exp.items():
    by_log.setdefault(v["log_name"], []).append(t)
logs = sorted(by_log)
toks, tl = [], {}
for ln in logs[::8]:
    for t in sorted(by_log[ln])[:24]:
        toks.append(t)
        tl[t] = ln
json.dump({"tokens": toks, "token_log": tl}, open(f"{OUT}/tokens_b.json", "w", encoding="utf-8"), indent=1)
print(json.dumps({"n_logs": len(set(tl.values())), "n_tokens": len(toks)}))
