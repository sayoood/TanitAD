"""The A19 map judge on a REAL harness-shaped record: the early A18 MAIN record's `results.healthy`
and `verdict` are the harness's own run_arm / controls / verdict output; reshaped as the harness's
main() writes it for the A19 spec (schema, spec sha256, no regression rows) and judged with the
candidate's refcv7 profile. Must PASS (MAIN passed); with edge forced to 0.49 it must FAIL.
Usage: a19_real_shape_check.py <A19 tree> <early A18 record>"""
import copy
import json
import sys
import tempfile
from pathlib import Path

tree, recp = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(tree / "stack"), str(tree / "stack" / "scripts")]
import launch_gate as LG  # noqa: E402

assert str(tree).replace("\\", "/").lower() in LG.__file__.replace("\\", "/").lower()
early = json.loads(recp.read_text(encoding="utf-8"))
rec = copy.deepcopy(early)
rec["schema"] = "tanitad.g_map_overfit_record/1"
rec["spec_sha256"] = LG.PROFILES["refcv7"]["overfit_main_only"]["map"]["spec_sha256"]
rec["launch_argv_sha256"] = "6402d33de75b7f1c6dbdeb9aeedd46a00a82e7325eec420fa179f366213bd5cd"
rec["verdict"] = copy.deepcopy(early["verdict_harness_as_computed"])
rec["verdict"]["regression_arms"] = {}               # the A19 spec has no must-fail rows
cw = rec["class_weights"]["sha256"]
with tempfile.TemporaryDirectory() as td:
    f = Path(td) / "g_map_overfit.json"
    f.write_text(json.dumps(rec), encoding="utf-8")
    r, d = LG.judge_map_overfit(LG.PROFILES["refcv7"], str(f), "ab" * 20,
                                argv_sha=rec["launch_argv_sha256"], class_weights_sha256=cw)
    print("real-shaped MAIN-only record:", r or "PASS", "| a19 applied:", d["a19"]["applied"],
          "| absent:", d["a19"]["absent_must_fail_arms"])
    assert r == [] and d["a19"]["applied"] is True
    rec["results"]["healthy"]["final"]["iou"]["edge"] = 0.49
    f.write_text(json.dumps(rec), encoding="utf-8")
    r2, _ = LG.judge_map_overfit(LG.PROFILES["refcv7"], str(f), "ab" * 20,
                                 argv_sha=rec["launch_argv_sha256"], class_weights_sha256=cw)
    print("edge forced to 0.49:", r2)
    assert r2 == ["G-MAP-OVERFIT: class 'edge' IoU 0.49 < the registered bar 0.5"]
print("REAL-SHAPE CHECK OK")
