#!/bin/bash
# Master Mind 2026-10-15 (PI: "run R0c"): apply the 12 R0c patches to a FRESH archive of the tip, then run every suite the change touches + the chain suites.
# The source is the IN-TREE raw/2026-10-15-r0c-build (never the stale E:/D: tree).  The superseded PATCH_10_*.on_94248ec.diff is EXCLUDED.
set -u
NSP=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/c216c849-5fe7-4d43-a3be-32429241f811/scratchpad
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TIP=${TIP:-ddc9ce9}
V=C:/Users/Admin/AppData/Local/Temp/r0f
P11=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
P27=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
OUT=$NSP/r0cf; rm -rf "$OUT"; mkdir -p "$OUT"
rm -rf "$V"; mkdir -p "$V"
git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false -c core.eol=lf archive "$TIP" "$PK" | tar --force-local -x -C "$V" --no-same-owner --exclude='*.npz' --exclude='*.pt' || { echo ARCHIVE_FAIL; exit 1; }
cp -r C:/Users/Admin/AppData/Local/Temp/tw2/FlyWheels "$V/" 2>/dev/null; cp C:/Users/Admin/AppData/Local/Temp/tw2/MODEL_REGISTRY.md "$V/" 2>/dev/null
mkdir -p "$V/Project Steering"; cp C:/Users/Admin/AppData/Local/Temp/tw2/"Project Steering"/MODEL_REGISTRY.md "$V/Project Steering/" 2>/dev/null
SRC="$V/$PK/raw/2026-10-15-r0c-build"
cd "$V/$PK" || exit 1
n=0
for p in "$SRC"/PATCH_[0-9]_*.diff "$SRC"/chain/PATCH_[0-9]*_*.diff; do
  case "$p" in *.on_94248ec.diff) continue;; esac
  git -c core.autocrlf=false apply --check "$p" > "$OUT/apply_$(basename "$p").log" 2>&1 || { echo "APPLY_CHECK_FAIL $(basename "$p")"; cat "$OUT/apply_$(basename "$p").log"; exit 1; }
  git -c core.autocrlf=false apply "$p" || { echo "APPLY_FAIL $(basename "$p")"; exit 1; }
  n=$((n+1)); echo "applied $(basename "$p")" >> "$OUT/applied.txt"
done
echo "PATCHES_APPLIED $n"
cp "$SRC/test_rl_r0c.py" refe/
"$P11" -c "
import glob,sys
bad=[f for f in glob.glob('refe/*.py')+glob.glob('refe/*.sh')+glob.glob('eval/*.md')+glob.glob('eval/*.json') if b'\r' in open(f,'rb').read()]
print('CRLF_FILES', len(bad), bad[:5])"
bash -n refe/rls1_pod.sh && echo RLS1_POD_SYNTAX_OK
cd refe
run() { local py=$1 tag=$2; shift 2; local T0=$(date +%s); local nm=$(echo "$* $tag" | tr ' /.+' '____');
  PYTHONIOENCODING=utf-8 TMP=E:/tmp_r0cf TEMP=E:/tmp_r0cf "$py" -u "$@" > "$OUT/$nm.log" 2>&1; local rc=$?
  echo "$* [$tag] rc $rc $(( $(date +%s) - T0 ))s $(grep -o -E 'ZZ[A-Z0-9_]*(PASS|SKIP)[A-Z_]* ?[0-9]*' "$OUT/$nm.log" | tail -1)"; }
mkdir -p E:/tmp_r0cf
run $P11 t2.11 test_rl_r0c.py
run $P27 t2.7.1 test_rl_r0c.py
run $P11 t2.11 test_rl_s1b.py
run $P27 t2.7.1 test_rl_s1b.py
run $P11 t2.11 test_rl_epo.py
run $P11 t2.11 test_launch_gate.py --static
run $P11 t2.11 test_aux_s1_wiring.py
run $P11 t2.11 test_aux_g12_bf16.py
CUBLAS_WORKSPACE_CONFIG=:4096:8 run $P27 t2.7.1+env test_aux_g12_bf16.py
run $P11 t2.11 test_rls1_chain.py
run $P11 t2.11 test_rls1_queue.py
run $P11 t2.11 test_flock_rw.py
run $P11 t2.11 test_s1aux_chain.py
run $P11 t2.11 test_p2rep_chain.py
run $P11 t2.11 test_p6_chain.py
run $P11 t2.11 test_p6_overlap.py
(cd ../eval && PYTHONIOENCODING=utf-8 "$P11" -u test_rl_val_eval.py > "$OUT/test_rl_val_eval.log" 2>&1; echo "eval/test_rl_val_eval.py rc $? $(grep -o -E 'ZZ[A-Z0-9_]*PASS ?[0-9]*' "$OUT/test_rl_val_eval.log" | tail -1)")
rm -rf E:/tmp_r0cf
echo ZZR0CF_DONE
