"""Delete one guard at a time from ra1_full_report.py; each mutation must turn the pinned test RED.
Run from the repo root: python mutate_report.py <repo root>  (PYTHONPATH must reach taniteval). The source is restored in `finally`."""
import json
import os
import subprocess
import sys

T, P = sys.argv[1], "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/code/ra1_full_report.py"
TEST = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "stack/tests/test_ra1_full_report.py"]
src = open(os.path.join(T, P), encoding="utf-8").read()
MUT = {
    "no full1 crosscheck": ("        crosscheck_full1(f1, rows)\n", ""),
    "no window-alignment refusal": ("    if not ok:\n        raise ValueError(f\"{which} windows do not align",
                                    "    if False:\n        raise ValueError(f\"{which} windows do not align"),
    "no one-P10-540 refusal": ("            if o[\"pw\"] is None or not np.array_equal(o[\"pw\"][\"p10_ade6\"], ref[\"pw\"][\"p10_ade6\"]):",
                               "            if False:"),
    "no holdout-sha refusal": ("    if hold and next(iter(hold)) != HOLDOUT_SHA:", "    if False:"),
    "no yaw refusal": ("        if ev.get(\"kin_mode\") != \"yaw\":", "        if False:"),
    "2F ignored": ("big = bool(d141[\"separated\"] and abs(d141[\"delta\"]) > 2 * F)", "big = bool(d141[\"separated\"])"),
    "strategic dropped": ("        w(f\"| STRATEGIC | — | **{sp.get('status')}**", "        (f\"| STRATEGIC | — | **{sp.get('status')}**"),
    "interim selects": ("    star = (f1 or {}).get(\"S_star\") or {}", "    star = {\"A1\": 500, \"R0\": 500}"),
    "no gate-delta crosscheck": ("        if isinstance(gate, dict) and \"delta\" in gate and abs(", "        if False and abs("),
    "decomposition pairs the wrong run": ("(\"A1x − R0\", \"A1x\", 0, \"R0\", 0)", "(\"A1x − R0\", \"A1\", 0, \"R0\", 0)"),
}


def run():
    r = subprocess.run([sys.executable] + TEST, cwd=T, capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = (r.stdout or "").strip().splitlines()
    return r.returncode, (lines[-1] if lines else "")


res = {}
try:
    for name, (old, new) in MUT.items():
        assert src.count(old) == 1, name
        with open(os.path.join(T, P), "w", encoding="utf-8", newline="\n") as f:
            f.write(src.replace(old, new))
        rc, tail = run()
        res[name] = dict(rc=rc, RED=rc != 0, summary=tail)
finally:
    with open(os.path.join(T, P), "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
rc, tail = run()
res["_restored"] = dict(rc=rc, summary=tail)
print(json.dumps(res, indent=1))
assert all(v["RED"] for k, v in res.items() if not k.startswith("_")) and res["_restored"]["rc"] == 0
