"""Pre-judge the REAL map binding record with the A19 candidate's judge_map_overfit (record half only;
the closure half needs the launch tree + data on Thor). Also with the pre-A19 profile (red arm).
Usage: a19_prejudge_binding.py <A19 tree> <g_map_overfit.json>"""
import json
import sys
from pathlib import Path

tree, recp = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(tree / "stack"), str(tree / "stack" / "scripts")]
import launch_gate as LG  # noqa: E402

p = LG.PROFILES["refcv7"]
ARGV_SHA = "6402d33de75b7f1c6dbdeb9aeedd46a00a82e7325eec420fa179f366213bd5cd"
CW = "d70dec8087ed73ee6d4129b6fc6e0a97a350f826907462b6b251413ede488b67"
r, d = LG.judge_map_overfit(p, str(recp), "b711411d89e8ce62ac04a0097e415e7d06a7fba9",
                            argv_sha=ARGV_SHA, class_weights_sha256=CW)
print("A19 judge:", "PASS" if not r else r)
print("a19:", json.dumps({k: v for k, v in d["a19"].items() if k != "statement"}))
print("statement:", d["a19"].get("statement"))
print("classes:", {c: (round(x["iou"], 3), x["n"]) for c, x in d["classes"].items()})
r0, _ = LG.judge_map_overfit(dict(p, overfit_main_only=None), str(recp), "b7" * 20,
                             argv_sha=ARGV_SHA, class_weights_sha256=CW)
print("pre-A19 profile (red arm):", r0)
