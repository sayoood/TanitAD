#!/usr/bin/env bash
# ⛔ THE CHEAPEST DISCRIMINATING EXPERIMENT for the launch-nondeterminism finding.
#
# MEASURED 2026-09-17: two launches of the identical RL command diverge at step 2 by
# 3.03e-07 relative and reach O(1) by step 80. ⛔ The CAUSE is NOT measured. Two
# candidates, and they need different fixes:
#
#   (A) non-deterministic ATOMIC reductions in backward  -> use_deterministic_algorithms
#   (B) TF32 / cuDNN autotune picking different kernels  -> strict_numerics() (in-repo)
#
# ⚠️ TF32 is deterministic-but-imprecise, so (B) is the WEAKER hypothesis, and
# `strict_numerics()` must NOT be asserted as the fix before this runs.
#
# ⭐ THE READOUT IS BIT-IDENTITY, not a metric: two launches under one condition either
# agree to the last bit or they do not. Arm 0 NAMES the culprit op outright.
# ⛔ The trainer is never edited — `det_wrap.py` sets the flags in a parent process.
set -u
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
S=C:/Users/Admin/tanitad-wt-join3d/stack/scripts/ddv2_rl_refcv5.py
W=C:/Users/Admin/qland/det_wrap.py
O=C:/Users/Admin/tanitad-caches/ddv2rl-l1-20260917/determinism_probe
export PYTHONPATH=C:/Users/Admin/tanitad-wt-join3d/stack
export PYTHONIOENCODING=utf-8
mkdir -p "$O"

# --- arm 0: NAME THE OP. warn_only=False makes the first offending kernel raise.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" "$W" --mode name-it --script "$S" train --arm rl --seed 1 --steps 3 --batch 4 \
      --il-form matched --grad-clip 100 --out-dir "$O/nameit" > "$O/nameit.log" 2>&1
echo "ZZNAMEIT rc=$?ZZ"
grep -E 'does not have a deterministic|deterministic_algorithms|Error|ZZWRAP' "$O/nameit.log" | head -6

run () {   # $1 = out name, $2 = mode
  if [ "$2" = "off" ]; then unset CUBLAS_WORKSPACE_CONFIG
  else export CUBLAS_WORKSPACE_CONFIG=:4096:8; fi
  "$PY" "$W" --mode "$2" --script "$S" train --arm rl --seed 1 --steps 20 --batch 4 \
        --il-form matched --grad-clip 100 --out-dir "$O/$1" > "$O/$1.log" 2>&1
  echo "ZZRUN $1 mode=$2 rc=$? rows=$(wc -l < "$O/$1/metrics.jsonl" 2>/dev/null || echo 0)ZZ"
}

run baseA off
run baseB off
run detA  strict
run detB  strict

"$PY" - <<'PYEOF'
import json, pathlib
O = pathlib.Path("C:/Users/Admin/tanitad-caches/ddv2rl-l1-20260917/determinism_probe")
def rows(n):
    p = O / n / "metrics.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if p.exists() else []
out = {"_what": ("Does forcing torch.use_deterministic_algorithms make two launches of the RL "
                 "trainer bit-identical? The readout is bit-identity, never a metric."),
       "_evidence_class": "MEASURED (ours)", "arms": {}}
def cmp(a, b, label, key):
    A, B = rows(a), rows(b)
    if not A or not B:
        print(f"  {label:30s} INCONCLUSIVE (a={len(A)} b={len(B)} rows)")
        out["arms"][key] = {"verdict": "INCONCLUSIVE", "rows_a": len(A), "rows_b": len(B)}
        return
    n = min(len(A), len(B))
    first = next((A[i]["step"] for i in range(n) if A[i]["grad_norm"] != B[i]["grad_norm"]), None)
    worst = max(abs(A[i]["grad_norm"] - B[i]["grad_norm"]) for i in range(n))
    v = "BIT-IDENTICAL" if first is None else "DIVERGES"
    print(f"  {label:30s} n={n:3d} first_divergence_step={first} max|dg|={worst:.6g}  ** {v} **")
    out["arms"][key] = {"verdict": v, "n_steps": n, "first_divergence_step": first,
                        "max_abs_grad_norm_diff": worst}
print("⭐ readout: does a relaunch agree to the last bit?")
cmp("baseA", "baseB", "as-is (current default)", "as_is")
cmp("detA",  "detB",  "use_deterministic_algorithms", "deterministic")
pathlib.Path("C:/Users/Admin/qland/pkgrl/raw/determinism_probe.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
print("wrote raw/determinism_probe.json")
PYEOF
echo "ZZPROBE-DONEZZ"
