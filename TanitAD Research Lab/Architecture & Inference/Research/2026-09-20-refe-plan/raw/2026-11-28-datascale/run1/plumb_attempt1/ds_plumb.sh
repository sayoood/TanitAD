#!/bin/bash
# MM 2026-11-28: the PRE-READ PLUMBING CHECK the frozen card names (PREREG_DATASCALE.md 10): ds_read.py `read --lever ds_ep` on the PARENT EP-HEAD run1
# heads must reproduce READ_EPHEAD.json's M1 to 1e-9. No DATA-SCALE number is read. CPU only. Marker: $OUT/PLUMB.DONE | PLUMB.FAILED.
# The parent heads were selected with `cv5`, so ds_read's cv5m head-select check and CTRL-CV(e) are EXPECTED to flag them: the check is on M1 and
# on whether the path runs end to end, and every flagged control is reported, not hidden.
export PATH="/usr/bin:/bin:/mingw64/bin:/c/Windows/System32:/c/Windows/System32/WindowsPowerShell/v1.0:$PATH"
set -u; LANDED=2b4409c56c092a458130bed6f1f79c8ed2948857
export CUDA_VISIBLE_DEVICES=-1 TMP=E:/Projects/TanitAD-artifacts/datascale/plumb/tmp TEMP=E:/Projects/TanitAD-artifacts/datascale/plumb/tmp TMPDIR=E:/Projects/TanitAD-artifacts/datascale/plumb/tmp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TIP=E:/Projects/TanitAD-artifacts/datascale/plumb/tip; PKG="$TIP/$PKR"; OUT=E:/Projects/TanitAD-artifacts/datascale/run1/plumb
L2=E:/Projects/TanitAD-artifacts/l2next/run1; SH=E:/Projects/TanitAD-artifacts/safehead/run1
TAB="E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next/propscore/full_C1_s*of2.jsonl"
mkdir -p $TMP $TIP $OUT
die() { echo "$1 $(date -u +%FT%TZ)" > $OUT/PLUMB.FAILED; echo "ZZPLUMB_FAIL $1"; exit 1; }
cf() { powershell -NoProfile -Command "\$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; [math]::Floor((\$m.CommitLimit-\$m.CommittedBytes)/1MB)" < /dev/null | tr -dc 0-9; }
gate() { local ok=0; while [ $ok -lt 3 ]; do c=$(cf); [ "${c:-0}" -ge 8000 ] && ok=$((ok+1)) || ok=0; sleep 20; done; }
git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive $LANDED "$PKR/refe" "$PKR/splits" "$PKR/code" "$PKR/eval" "$PKR/raw/2026-11-27-safehead" "$PKR/raw/2026-11-27-l2next" "$PKR/raw/2026-11-28-datascale" | tar -x -C $TIP || die EXTRACT_FAIL
( cd "$PKG" && sha256sum -c --quiet raw/2026-11-28-datascale/CODE.sha256 ) > $OUT/code_sha.log 2>&1 || die "CODE.sha256 mismatch"
gate
cd "$PKG/raw/2026-11-28-datascale"
$PY ds_read.py read --lever ds_ep --table "$TAB" --proposals E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p5c_c1.proposals.npz --seam E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p5c_c1.npz \
  --navtest-index "$SH/nt_sealed.csv" --train-index "$L2/index_sealed.csv" --heldout-logs "$PKG/splits/sft_heldout_logs.txt" \
  --run-s0 "$L2/eph_s0" --run-s1 "$L2/eph_s1" --run-red "$L2/eph_red" --eval-dir "$PKG/eval" --refe "$PKG/refe" \
  --card PREREG_DATASCALE.md --card-sha256 $(cut -c1-64 PREREG_DATASCALE.sha256) --out "$OUT/read" > $OUT/read.log 2>&1
rc=$?; echo "read rc=$rc"; tail -3 $OUT/read.log | cut -c1-400
$PY - "$OUT" "$L2/read_eph/READ_EPHEAD.json" <<'PYEOF'
import json, sys, glob
out, ref = sys.argv[1], sys.argv[2]
new = sorted(glob.glob(out + "/read/READ_*.json"))
if not new:
    print("ZZPLUMB no READ_*.json written"); sys.exit(1)
a, b = json.load(open(new[0])), json.load(open(ref))
def m1(r):
    m = r.get("M1_M3", {})
    return {k: m[k] for k in m if k.lower().startswith(("delta", "m1", "d0", "d1", "pdms"))} or m
A, B = m1(a), m1(b)
diffs = {}
def cmp(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in set(x) & set(y):
            cmp(x[k], y[k], path + "/" + k)
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        diffs[path] = abs(x - y)
cmp(A, B)
worst = max(diffs.values()) if diffs else None
fails = {k: v for k, v in (a.get("controls") or {}).items() if isinstance(v, dict) and v.get("pass") is False}
res = {"new": new[0], "n_numbers_compared": len(diffs), "max_abs_diff": worst, "m1_reproduced_1e-9": worst is not None and worst <= 1e-9,
       "outcome_new": a.get("outcome"), "controls_flagged": sorted(fails)}
json.dump(res, open(out + "/PLUMB_CMP.json", "w"), indent=1)
print("ZZPLUMB_CMP", json.dumps(res)[:600])
PYEOF
date -u +%FT%TZ > $OUT/PLUMB.DONE
