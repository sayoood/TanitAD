#!/usr/bin/env bash
# RUNNER CHECK on the corrected build (after SPEC A2): the one-command runner, pointed at copies of
# the manual step-1,000 warmup run's banked rows and scores, must reproduce the manual statistics
# EXACTLY — the runner's bridge resumes (0 new rows), its scorer is skipped (counts PASS), and
# parse6 / decompose6 / plan_deltas / bars6 run through the runner's own wiring.
#   bash code/runner_check.sh <scratch dir>
# Output banked: raw/controls/RUNNER_CHECK_s1000.json (the comparison only; the copies stay in scratch).
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
T="$1"
rm -rf "$T"; mkdir -p "$T"
cp -r "$P/raw/bridge_warmup_s1000" "$T/bridge_warmup"
cp -r "$P/raw/scores_warmup_s1000" "$T/scores_warmup"
TW=$(cygpath -m "$T")
"$PY" "$P/code/run_navsim_refcv6.py" --ckpt D:/refcv6_eval_kit/ckpt/ckpt_step1000.pt \
   --md5 7c3ad3c1fbf3d30be5c7733e7b436c65 --splits warmup --device cpu --out "$TW" > "$T/runner.out" 2>&1
"$PY" - "$TW" "$(cygpath -m "$P")" <<'PYEOF'
import json, os, sys
t, p = sys.argv[1], sys.argv[2]
def j(x):
    try:
        return json.load(open(x, encoding="utf-8"))
    except Exception as e:
        return {"_unreadable": repr(e)}
a, b = j(os.path.join(t, "summary_warmup.json")), j(os.path.join(p, "raw", "summary_warmup_s1000.json"))
arms = sorted(set(a.get("arms", {})) | set(b.get("arms", {})))
diff = {}
for k in arms:
    ua = a.get("arms", {}).get(k, {}).get("S2_EPDMS_u", {}).get("value")
    ub = b.get("arms", {}).get(k, {}).get("S2_EPDMS_u", {}).get("value")
    diff[k] = {"runner": ua, "manual": ub, "equal": ua == ub and ua is not None}
art = {f: os.path.exists(os.path.join(t, f)) for f in
       ("MILESTONE_SUMMARY.json", "summary_warmup.json", "decomposition_warmup.json",
        "plan_deltas_warmup.json", "BARS.json")}
bars = j(os.path.join(t, "BARS.json")).get("bars")
rep = {"check": "runner reproduces the manual step-1,000 warmup statistics (corrected build)",
       "artifacts_present": art, "S2_EPDMS_u_runner_vs_manual": diff,
       "all_equal": all(v["equal"] for v in diff.values()) and bool(diff),
       "bars": bars, "bars_not_evaluated_below_5000": isinstance(bars, str) and "NOT_EVALUATED" in bars}
os.makedirs(os.path.join(p, "raw", "controls"), exist_ok=True)
json.dump(rep, open(os.path.join(p, "raw", "controls", "RUNNER_CHECK_s1000.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps({k: rep[k] for k in ("artifacts_present", "all_equal", "bars_not_evaluated_below_5000")}))
PYEOF
