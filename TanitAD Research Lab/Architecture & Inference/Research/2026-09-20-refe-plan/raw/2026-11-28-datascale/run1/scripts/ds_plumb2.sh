#!/bin/bash
# MM 2026-11-28: PREREG_DATASCALE AMENDMENT 1 plumbing check. Lever B's arms red/s0/s1 trained with --select cv5m on the PARENT's 35-log sealed index,
# each under its own launch-gate PASS token (refe_ds_ep_*), then ds_read.py read --lever ds_ep on them. PASS = rc 0, a READ file, every control PASS,
# outcome NULL-CV, Delta0 = 0 exactly. These heads are a plumbing FIXTURE, never a DATA-SCALE arm. Marker: $P2/PLUMB2.DONE | PLUMB2.FAILED.
export PATH="/usr/bin:/bin:/mingw64/bin:/c/Windows/System32:/c/Windows/System32/WindowsPowerShell/v1.0:$PATH"
set -u; LANDED=872a0dc5462ffc58974f5c677424ca97cd9f978b
RUN=E:/Projects/TanitAD-artifacts/datascale/run1; P2=$RUN/plumb35; M35=$RUN/m6_35
export CUDA_VISIBLE_DEVICES=-1 TMP=$RUN/../tmp TEMP=$RUN/../tmp TMPDIR=$RUN/../tmp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TIP=C:/Users/Admin/dsplumb_tip; PKG="$TIP/$PKR"; SH=E:/Projects/TanitAD-artifacts/safehead/run1; C1=F:/safehead_cache/ckpt/C1/model_final.pt
TAB="E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next/propscore/full_C1_s*of2.jsonl"
mkdir -p $P2 $TIP $RUN/../tmp
die() { echo "$1 $(date -u +%FT%TZ)" > $P2/PLUMB2.FAILED; echo "ZZPLUMB2_FAIL $1"; exit 1; }
cf() { powershell -NoProfile -Command "\$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; [math]::Floor((\$m.CommitLimit-\$m.CommittedBytes)/1MB)" < /dev/null | tr -dc 0-9; }
gate() { local ok=0; while [ $ok -lt 3 ]; do c=$(cf); [ "${c:-0}" -ge 8000 ] && ok=$((ok+1)) || ok=0; sleep 20; done; }
[ "$(sha256sum $C1 | cut -c1-64)" = f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4 ] || die C1_SHA_BAD
git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive $LANDED "$PKR/refe" "$PKR/splits" "$PKR/code" "$PKR/eval" "$PKR/raw/2026-11-27-safehead" "$PKR/raw/2026-11-27-l2next" "$PKR/raw/2026-11-28-datascale" | tar -x -C $TIP || die EXTRACT_FAIL
( cd "$PKG" && sha256sum -c --quiet raw/2026-11-28-datascale/CODE.sha256 ) > $P2/code_sha.log 2>&1 || die "CODE.sha256 mismatch"
[ "$(cut -c1-64 "$PKG/raw/2026-11-28-datascale/AMENDMENT_1.sha256")" = 09444131e621622c09a575b9dc34a7ef6f25b2fa165fa3c688dc3190783985ab ] || die AMEND_SHA_BAD
IDX=$M35/index_sealed.csv; [ "$(sha256sum $IDX | cut -c1-64)" = f7426318271aebc71e9e256b3dc7f2edfa96dd57d57190342a30679e87434c03 ] || die PARENT_INDEX_SHA_BAD
cd "$PKG/refe"
$PY launch_gate.py ship-manifest --pkg "$PKG" --commit $LANDED --git-dir C:/Users/Admin/tanitad-push/.git --repo-prefix "$PKR/" --out "$PKG/SHIP_MANIFEST.json" | tail -1
B="--cache-index $IDX --init-from $C1 --heldout-logs $M35/sft_heldout_logs.txt --navtest-logs $M35/navtest_logs.txt --steps 2000 --lr 0.0003 --batch 32 --gamma 2.0 --save-steps 0,250,500,1000,2000 --rows 2 --select cv5m"
for arm in red s0 s1; do
  [ -f $P2/dep35_$arm/run.json ] && { echo "skip dep35_$arm"; continue; }
  case $arm in s0) X="--seed 0";; s1) X="--seed 1";; red) X="--seed 0 --permute-labels-seed 20261127";; esac
  ARGS="$B --out $P2/dep35_$arm $X"
  gate; echo "== gate refe_ds_ep_$arm $(date -u +%H:%M:%SZ)"
  $PY launch_gate.py refusals --profile refe_ds_ep_$arm --args "$ARGS" | tail -1
  $PY launch_gate.py run --profile refe_ds_ep_$arm --pkg "$PKG" --args "$ARGS" --out-dir "$P2/gate_$arm" > $P2/gate_$arm.log 2>&1; tail -1 $P2/gate_$arm.log
  TOK=$(ls "$P2/gate_$arm"/PASS_*.json 2>/dev/null | head -1); [ -n "$TOK" ] || die "NO_PASS_TOKEN $arm"
  gate; TANITAD_SH_GATE_TOKEN="$TOK" $PY safehead_train.py $ARGS > $P2/train_$arm.log 2>&1 || die "train $arm rc=$?"
  grep -v "UserWarning\|Consider using\|rec = " $P2/train_$arm.log | tail -1 | cut -c1-250
done
gate; echo "== ds_read $(date -u +%H:%M:%SZ)"
cd "$PKG/raw/2026-11-28-datascale"
$PY ds_read.py read --lever ds_ep --table "$TAB" --proposals E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p5c_c1.proposals.npz --seam E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p5c_c1.npz \
  --navtest-index "$SH/nt_sealed.csv" --train-index "$IDX" --heldout-logs "$PKG/splits/sft_heldout_logs.txt" \
  --run-s0 "$P2/dep35_s0" --run-s1 "$P2/dep35_s1" --run-red "$P2/dep35_red" --eval-dir "$PKG/eval" --refe "$PKG/refe" \
  --card PREREG_DATASCALE.md --card-sha256 $(cut -c1-64 PREREG_DATASCALE.sha256) --out "$P2/read" > $P2/read.log 2>&1
rc=$?; echo "read rc=$rc"; tail -2 $P2/read.log | cut -c1-400
[ $rc = 0 ] || die "ds_read rc=$rc"
$PY - "$P2" <<'PYEOF'
import json, sys, glob
p2 = sys.argv[1]
f = sorted(glob.glob(p2 + "/read/READ_*.json"))
assert f, "no READ file"
r = json.load(open(f[0]))
ctrl = r.get("controls") or {}
bad = sorted(k for k, v in ctrl.items() if isinstance(v, dict) and v.get("pass") is False)
out = r.get("outcome") or r.get("outcome_provisional") or {}
oc = out.get("outcome") if isinstance(out, dict) else out
m = r.get("M1_M3") or {}
d0 = None
for k in ("delta0", "Delta0", "M1", "d0"):
    if k in m:
        d0 = m[k]; break
res = {"read": f[0], "outcome": oc, "controls_failed": bad, "n_controls": len(ctrl), "delta0": d0,
       "pass": (oc == "NULL-CV" and not bad)}
json.dump(res, open(p2 + "/PLUMB2_CHECK.json", "w"), indent=1, default=str)
print("ZZPLUMB2_CHECK", json.dumps(res, default=str)[:700])
sys.exit(0 if res["pass"] else 1)
PYEOF
[ $? = 0 ] || die "pass condition not met (see PLUMB2_CHECK.json)"
date -u +%FT%TZ > $P2/PLUMB2.DONE; echo ZZPLUMB2_DONE
