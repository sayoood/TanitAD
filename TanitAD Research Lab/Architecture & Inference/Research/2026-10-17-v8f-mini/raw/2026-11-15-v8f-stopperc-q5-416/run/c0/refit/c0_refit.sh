#!/usr/bin/env bash
# MM 2026-11-16: the refit part of Q5 Amendment 7 control C0. On the pulled S1 (box-2 re-extraction, shard clause PASS) and the dev-box eval npz
# (sha 0c25fe72 = banked), run S0-P's own s0p_fit.py (exported byte-exact from the tip) prep + fit E|linear and V40+E|linear at 2 threads, as S0-P did,
# then compare the eval-score .npy files byte for byte with the banked S0-P scores. Result: RESULT_C0_REFIT.json. Last log line: ZZ C0REFIT PASS/FAIL ZZ.
export PATH="/usr/bin:/mingw64/bin:/bin:$PATH"
set -u
R=/f/TanitAD-scratch/q5_416/run/c0/refit
TIP=8a716bb92df0c7cd57c72d9f4702bba5f0770616
B="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-11-11-v8f-stopperc/s0p"
G="git -c safe.directory=* -c core.autocrlf=false -C C:/Users/Admin/tanitad-push"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$R/c0_refit.log
mkdir -p $R/code $R/work $R/banked
for f in s0p_fit.py s0p_lib.py; do $G cat-file blob "$TIP:$B/$f" > $R/code/$f || { echo "ZZ C0REFIT FAIL ZZ export $f" >> $LOG; exit 2; }; done
for a in "E" "V40+E"; do
  for x in npy json; do $G cat-file blob "$TIP:$B/run/scores/fit_${a}_linear.$x" > "$R/banked/fit_${a}_linear.$x" || { echo "ZZ C0REFIT FAIL ZZ banked $a $x" >> $LOG; exit 2; }; done
done
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1
TR=F:/TanitAD-scratch/q5_416/run/c0/S1; EV=F:/TanitAD-scratch/q5_416/run/c0/s0p_eval.npz; W=F:/TanitAD-scratch/q5_416/run/c0/refit/work
cd $R/code
echo "START $(date '+%F %T')" >> $LOG
$PY s0p_fit.py prep --train $TR --eval $EV --work $W >> $LOG 2>&1 || { echo "ZZ C0REFIT FAIL ZZ prep rc=$?" >> $LOG; exit 3; }
for a in "E" "V40+E"; do
  $PY s0p_fit.py fit --train $TR --eval $EV --work $W --arm "$a" --model linear >> $LOG 2>&1 || { echo "ZZ C0REFIT FAIL ZZ fit $a rc=$?" >> $LOG; exit 4; }
done
$PY - >> $LOG 2>&1 <<'EOF'
import hashlib, json
import numpy as np
R = "F:/TanitAD-scratch/q5_416/run/c0/refit/"
out = {"tier": "T0 probe (situation-cue decodability), not driving", "arms": {}}
ok = True
for a in ("E", "V40+E"):
    n = open(R + f"work/fit_{a}_linear.npy", "rb").read(); b = open(R + f"banked/fit_{a}_linear.npy", "rb").read()
    na, ba = np.load(R + f"work/fit_{a}_linear.npy"), np.load(R + f"banked/fit_{a}_linear.npy")
    jn = json.load(open(R + f"work/fit_{a}_linear.json", encoding="utf-8")); jb = json.load(open(R + f"banked/fit_{a}_linear.json", encoding="utf-8"))
    e = {"npy_bytes_equal": n == b, "sha_new": hashlib.sha256(n).hexdigest(), "sha_banked": hashlib.sha256(b).hexdigest(),
         "shape": list(na.shape), "max_abs_diff": float(np.max(np.abs(na.astype(np.float64) - ba.astype(np.float64)))) if na.shape == ba.shape else None,
         "best_new": jn.get("best"), "best_banked": jb.get("best")}
    out["arms"][a] = e; ok = ok and e["npy_bytes_equal"]
out["C0_refit_bitwise"] = ok
json.dump(out, open(R + "RESULT_C0_REFIT.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
print("ZZ C0REFIT " + ("PASS" if ok else "FAIL") + " ZZ")
EOF
echo "END $(date '+%F %T')" >> $LOG
