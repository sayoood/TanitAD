#!/usr/bin/env bash
# mapft_devbox_post.sh -- DEV-BOX post-run chain for the MAPFT run (PREREG_MAPFT s2): fetch from Thor, then K-PLAN-ID + K-RAST-MOVED.
#   Modelled on refcv9_mapsel/code/chain_mapsel.sh (the SAME roll, tree, lock protocol and identity checks the base trace was banked with).
#
#   mapft_devbox_post.sh selfcheck            CPU ONLY, no Thor, no GPU: identity checks + the comparator controls on the BANKED base traces
#   mapft_devbox_post.sh rehearse             ~3.5 min GPU: the BASE ckpt stands in for model_ema.pt; the smoke bit-compare must read PASS (run it while Thor trains)
#   mapft_devbox_post.sh fetch                LAN-copy the run's artifacts from Thor, verify sha256 against Thor's MANIFEST.sha256
#   mapft_devbox_post.sh kplanid [--no-full]  THE chain (one GPU job under gpu_lock):
#                                              1. MAPFT smoke roll (9 clips, 308 windows, ~3.5 min) -> bit-compare vs the banked trace_smoke.npz   [FAST FAIL]
#                                              2. only if (1) differs: BASE smoke re-roll (the rig-determinism CONTROL) -> bit-compare vs trace_smoke.npz
#                                              3. FULL MAPFT roll (4,754 windows, ~48 min) -> K-PLAN-ID (plan8, fan, sel: the sealed three) vs trace_s0.npz
#                                              4. MAPPATH twin (registered mappath_transform.py) vs trace_s0_mappath.npz; K-RAST-MOVED vs rast_s0
#   ANY: --post DIR (default F:/refcv9_mapft/post)
#
# THE VERDICTS (each read from an ARTIFACT, never from an exit code):
#   K-PLAN-ID   PASS | FAIL (planner differs AND the rig reproduces itself) | RIG-NONDETERMINISTIC (the base re-roll differs from its own banked trace: bit-equality is
#               unmeasurable on this rig -- STOP and report; do NOT call it a planner failure and do NOT relax the sealed text) | INVALID
#   Everything is under F: (NTFS).  Nothing is written to G:.  Markers ZZMK<NAME>ZZ in $POST/logs/chain.log.
set -u
MODE="${1:-}"; shift || true
POST="${MAPFT_POST:-F:/refcv9_mapft/post}"; NOFULL=0
while [ $# -gt 0 ]; do case "$1" in --post) POST="$2"; shift 2;; --no-full) NOFULL=1; shift;; *) echo "REFUSED: unknown argument $1"; exit 2;; esac; done
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; KITN="$KIT"; command -v cygpath >/dev/null 2>&1 && KITN="$(cygpath -m "$KIT")"
W=C:/Users/Admin/refcv8_pf/refcv9_mapsel; CODE=$W/code; BK="$POST/rolls"; CH="$POST/logs/chain.log"
BASE=F:/refcv9_mapsel                                        # the banked base traces / rasters (READ-ONLY here)
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58         # the roll's own tree (the one the base trace was rolled with)
LOCKPY=$T/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
BANKT=C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf
CK_BASE=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt; CFG=E:/refcv8_eval_kit/ckpt/config.json
GTROOT=E:/refcv6_eval_kit/data/sam3_gt_v3_eval
MAPPATH_PY=C:/Users/Admin/refcv8_pf/refcv9_mapedge/mappath_transform.py
JOB=refcv9-mapft-kplanid; MINCOMMIT=${MINCOMMIT:-6}; BUDGET_S=5400
ARM=refcv8-mapft-s0; RUNL="$POST/run"                         # fetched artifacts land here
THOR=tanitad-thor; TROOT=/home/nvidia/r8ns/mapft
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1 MSYS_NO_PATHCONV=1
unset CUDA_VISIBLE_DEVICES REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION PF_R8_CODE
say() { echo "[mapft-devbox-post:$MODE] $*"; }
die() { say "REFUSED: $*"; echo "ZZMAPFTDEVPOST-REFUSEDZZ"; exit 10; }
mk() { mkdir -p "$(dirname "$CH")"; echo "ZZMK$1ZZ $(date +%FT%T%z) $2" >> "$CH"; echo "ZZMK$1ZZ $2"; }
[ -n "$MODE" ] || { sed -n 2,19p "${BASH_SOURCE[0]}"; exit 2; }

identity() {   # the SAME identities chain_mapsel.sh asserts, plus: the roll script is byte-identical to the one that banked the base trace
  local seal seal2 m8 w0 rs meta
  chk() { "$PY" -I -c "import hashlib,sys; w=open(sys.argv[2]).read().split()[0]; g=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(1 if (len(w)==64 and g==w) else 0)" "$1" "$2" 2>/dev/null; }
  seal=$(cd "$W" && chk PREREG_MAPSEL.md PREREG_MAPSEL.sha256); seal2=$(cd "$W" && chk PREREG_MAPSEL_AMENDMENT1.md PREREG_MAPSEL_AMENDMENT1.sha256)
  m8=$(md5sum "$CK_BASE" | cut -c1-32); w0=$(md5sum "$BANKT/windows_s0.json" | cut -c1-32)
  rs=$("$PY" -I -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$CODE/mapsel_roll.py")
  meta=$("$PY" -I -c "import json,sys,numpy as np; print(json.loads(str(np.load(sys.argv[1])['meta_json']))['script_sha256'])" "$BASE/trace_s0.npz")
  mk IDENT "seal=$seal seal_amend1=$seal2 base_ckpt_md5=$m8 windows_md5=$w0 roll_script_sha=${rs:0:16} base_trace_script_sha=${meta:0:16}"
  [ "$seal" = 1 ] && [ "$seal2" = 1 ] && [ "$m8" = b4364029b68f29a33eeb4aebbbbd002b ] && [ "$w0" = a3a78dba889e70aa69a05271559794ad ] && [ "${#rs}" = 64 ] && [ "$rs" = "$meta" ]
}

case "$MODE" in
# =============================================================================================================== selfcheck
selfcheck)
  identity || die "identity check failed (see $CH): the roll that would run is not the roll that banked the base trace"
  ( "$PY" -I "$KITN/mapft_kplanid.py" self-test 2>&1 | tail -1 ) | grep -q "SELFTEST-OKZZ" || die "mapft_kplanid.py self-test failed"
  "$PY" "$KITN/mapft_kplanid.py" compare --base "$BASE/trace_s0.npz" --new "$BASE/trace_s0.npz" --json "$POST/selfcheck/compare_self.json" > "$POST/selfcheck.compare.out" 2>&1
  [ "$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['K_PLAN_ID_FINAL'])" "$POST/selfcheck/compare_self.json" 2>/dev/null)" = PASS ] || die "comparator does not read the base trace as equal to itself"
  "$PY" "$KITN/mapft_kplanid.py" mappath --base-derivative "$BASE/trace_s0_mappath.npz" --new "$BASE/trace_s0.npz" --base "$BASE/trace_s0.npz" --work "$POST/selfcheck/mp" --script "$MAPPATH_PY" > "$POST/selfcheck.mappath.out" 2>&1
  [ "$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['MAPPATH_LEG'])" "$POST/selfcheck/mp/mappath_leg.json" 2>/dev/null)" = PASS ] || die "the registered MAPPATH derivation does not reproduce the banked derivative"
  mk SELFCHECK "identity ok; comparator self-compare PASS with both controls; derive(base) == banked MAPPATH twin bit for bit"; echo "ZZMAPFT-DEVSELFCHECK-OKZZ";;

# =============================================================================================================== rehearse
rehearse)   # ~3.5 min of dev-box GPU (queues behind any foreign gpu_lock holder; needs >= 6 GB free commit): the BASE checkpoint stands in for model_ema.pt
  # EXPECTED: the smoke bit-compare reads PASS -- proving, BEFORE any MAPFT checkpoint exists, that (a) the roll accepts a model_ema.pt-style file, (b) this rig
  # reproduces its own banked trace bit for bit (the premise of K-PLAN-ID).  A FAIL here is the RIG, not the planner (the file IS the base model).
  RP="${MAPFT_REHEARSE:-F:/refcv9_mapft/rehearsal}"; mkdir -p "$RP/run" || die "cannot create $RP/run"
  [ -s "$RP/run/model_ema.pt" ] || cp -p "$CK_BASE" "$RP/run/model_ema.pt" || die "cannot stage the stand-in checkpoint"
  [ "$(md5sum "$RP/run/model_ema.pt" | cut -c1-32)" = b4364029b68f29a33eeb4aebbbbd002b ] || die "stand-in checkpoint md5 != the base EMA-48,852"
  exec bash "${BASH_SOURCE[0]}" kplanid --no-full --post "$RP";;

# =============================================================================================================== fetch
fetch)
  mkdir -p "$RUNL" "$POST/logs" || die "cannot create $RUNL"
  ssh -n -o BatchMode=yes -o ConnectTimeout=20 $THOR "test -s $TROOT/post/MANIFEST.sha256 && echo MANIFEST-OK" | tr -d '\r' | grep -q MANIFEST-OK || die "no $TROOT/post/MANIFEST.sha256 on Thor (run mapft_postrun.sh kfrozen, then manifest)"
  scp -q -o BatchMode=yes $THOR:$TROOT/post/MANIFEST.sha256 "$RUNL/MANIFEST.sha256" || die "could not fetch the manifest"
  NBAD=0; NOK=0
  while read -r H B REL; do
    [ -n "$REL" ] || continue
    D="$RUNL/$(basename "$REL")"
    scp -q -o BatchMode=yes "$THOR:$TROOT/$REL" "$D" || { say "scp failed: $REL"; NBAD=$((NBAD+1)); continue; }
    G="$(sha256sum "$D" | cut -c1-64)"; S="$(stat -c %s "$D")"
    if [ "${#G}" = 64 ] && [ "$G" = "$H" ] && [ "$S" = "$B" ]; then NOK=$((NOK+1)); else say "MISMATCH $REL (want $H/$B, got $G/$S)"; NBAD=$((NBAD+1)); fi
  done < "$RUNL/MANIFEST.sha256"
  [ "$NBAD" = 0 ] && [ "$NOK" -ge 6 ] || die "$NBAD file(s) failed the sha256/size check ($NOK ok): do not use what was copied"
  mk FETCH "ok=$NOK files verified against the Thor manifest -> $RUNL"; echo "ZZMAPFT-DEVFETCH-OKZZ";;

# =============================================================================================================== kplanid
kplanid)
  [ -s "$RUNL/model_ema.pt" ] || die "no $RUNL/model_ema.pt (run 'fetch' first)"
  CK_NEW="$RUNL/model_ema.pt"; mkdir -p "$BK" "$POST/result" "$POST/logs"
  identity || { mk ABORT "identity check failed"; die "identity check failed (see $CH)"; }
  # the mapsel library's own unit test, as the chain does
  unit=$( ( cd "$CODE" && CUDA_VISIBLE_DEVICES=-1 "$PY" -X utf8 test_mapsel_lib.py 2>&1 ) | grep -c UNITPASS ); mk UNIT "unitpass=$unit"
  [ "$unit" = 1 ] || { mk ABORT "unit tests of mapsel_lib failed"; die "mapsel_lib unit test failed"; }
  free_virt_gb() { powershell -NoProfile -Command "([math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,1)).ToString([cultureinfo]::InvariantCulture)" 2>/dev/null | tr -d '\r\n '; }
  t0=$(date +%s); i=0
  until awk "BEGIN{exit !($(free_virt_gb)+0 >= $MINCOMMIT)}"; do
    i=$((i+1)); [ $((i % 20)) -eq 1 ] && mk COMMITWAIT "free_commit_gb=$(free_virt_gb) need=$MINCOMMIT"
    [ $(( $(date +%s) - t0 )) -gt 10800 ] && { mk STOP "free commit stayed below ${MINCOMMIT} GB for 3 h"; die "no free commit"; }; sleep 30
  done
  HELD=0; CHILD=0
  release() { [ $HELD -eq 1 ] || return; HELD=0; "$PY" "$LOCKPY" release --job "$JOB" --pid $$ >> "$POST/logs/lock.log" 2>&1; mk GPUFREE "$1"; }
  cleanup() { rc=$?; if [ "$CHILD" != 0 ]; then kill "$CHILD" 2>/dev/null; sleep 3; kill -9 "$CHILD" 2>/dev/null; fi; release "EXIT rc=$rc"; }
  trap cleanup EXIT; trap 'exit 143' TERM INT HUP
  rm -f "$POST/logs/lock.json"
  "$PY" "$LOCKPY" acquire --job "$JOB" --pid $$ --max-wait-s 43200 --out "$POST/logs/lock.json" >> "$POST/logs/lock.log" 2>&1
  "$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$POST/logs/lock.json" || { mk NOLOCK "see logs/lock.log"; die "gpu_lock not acquired"; }
  HELD=1; mk LOCKED "free_commit_gb=$(free_virt_gb)"
  final_n() { "$PY" -I -c "import json,sys,numpy as np; z=np.load(sys.argv[1]); m=json.loads(str(z['meta_json'])); print(m['n_windows'] if m.get('final') is True else 'BAD')" "$1" 2>/dev/null || echo NOFILE; }
  roll() {  # $1 tag  $2 ckpt  $3 windows json  $4 expected windows  $5 extra roll opts
    local tag="$1" ck="$2" wj="$3" want="$4" extra="${5:-}" TS n; TS=$(date +%s)
    rm -rf "$BK/dump_${tag}" "$BK/rast_${tag}"
    "$PY" -X utf8 "$CODE/mapsel_roll.py" --trace-out "$BK/trace_${tag}.npz" --rast-dir "$BK/rast_${tag}" --map-gt-root "$GTROOT" --require-seal $extra -- \
        --ckpt "$ck" --config "$CFG" --seed 0 --dump-dir "$BK/dump_${tag}" --windows-json "$wj" --device cuda --out-json "$BK/roll_${tag}.json" --os-only \
        > "$POST/logs/roll_${tag}.log" 2>&1 &
    CHILD=$!; wait $CHILD; CHILD=0
    n=$(final_n "$BK/trace_${tag}.npz"); mk "ROLL_${tag}" "wall=$(( $(date +%s) - TS ))s windows=$n want=$want"
    [ "$n" = "$want" ] && rm -rf "$BK/dump_${tag}"
    [ "$n" = "$want" ]
  }
  cmp_json() { "$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['K_PLAN_ID_FINAL'])" "$1" 2>/dev/null || echo NOFILE; }
  SW="$BASE/windows_smoke.json"; [ -s "$SW" ] || { mk ABORT "no $SW"; die "no smoke window list"; }
  # --- 1. MAPFT smoke roll, bit-compared to the banked smoke trace (fast fail) ------------------------------------------------------------
  roll smoke_mapft "$CK_NEW" "$SW" 308 || { mk ABORT "MAPFT smoke roll produced no final 308-window trace (see logs/roll_smoke_mapft.log: does the battery child accept model_ema.pt?)"; die "smoke roll failed"; }
  "$PY" "$KITN/mapft_kplanid.py" compare --base "$BASE/trace_smoke.npz" --new "$BK/trace_smoke_mapft.npz" --json "$POST/result/kplanid_smoke.json" > "$POST/logs/kplanid_smoke.out" 2>&1
  SMOKE_V=$(cmp_json "$POST/result/kplanid_smoke.json"); mk KPLANID_SMOKE "$SMOKE_V"
  RIG=NOT-RUN
  if [ "$SMOKE_V" != PASS ]; then
    # --- 2. the rig-determinism CONTROL: the BASE checkpoint, re-rolled on the same windows, against its own banked trace ----------------
    roll smoke_base_ctl "$CK_BASE" "$SW" 308 || { mk ABORT "base control roll failed"; die "base control roll failed"; }
    "$PY" "$KITN/mapft_kplanid.py" compare --base "$BASE/trace_smoke.npz" --new "$BK/trace_smoke_base_ctl.npz" --json "$POST/result/kplanid_smoke_basectl.json" > "$POST/logs/kplanid_smoke_basectl.out" 2>&1
    CTL_V=$(cmp_json "$POST/result/kplanid_smoke_basectl.json"); mk RIGCONTROL "$CTL_V"
    if [ "$CTL_V" = PASS ]; then RIG=DETERMINISTIC; else RIG=NONDETERMINISTIC; fi
    release "stopped after the smoke comparison"
    if [ "$RIG" = DETERMINISTIC ]; then
      mk VERDICT "K-PLAN-ID=FAIL (the MAPFT smoke roll differs from the banked trace while the BASE re-roll reproduces it bit for bit) -> STOP, report to the PI, no bar is read"
      echo "ZZMAPFT-KPLANID-FAILZZ"; exit 21
    fi
    mk VERDICT "K-PLAN-ID=RIG-NONDETERMINISTIC (the BASE re-roll differs from its own banked trace: bit-equality is not measurable on this rig) -> STOP and report; do NOT relax the sealed text"
    echo "ZZMAPFT-KPLANID-RIGNONDETZZ"; exit 22
  fi
  [ "$NOFULL" = 1 ] && { release "smoke only (--no-full)"; mk VERDICT "smoke bit-compare PASS only (--no-full): K-PLAN-ID is NOT decided"; exit 0; }
  # --- 3. the FULL roll --------------------------------------------------------------------------------------------------------------------
  roll s0_mapft "$CK_NEW" "$BANKT/windows_s0.json" 4754 "--budget-probe-windows 150 --budget-max-s $BUDGET_S" || { mk ABORT "full roll artifact (exit 7 = over budget)"; die "full roll failed"; }
  release "full roll done"
  # --- 4. CPU: K-PLAN-ID on every window, the MAPPATH twin, K-RAST-MOVED ---------------------------------------------------------------------
  "$PY" "$KITN/mapft_kplanid.py" compare --base "$BASE/trace_s0.npz" --new "$BK/trace_s0_mapft.npz" --json "$POST/result/kplanid_full.json" > "$POST/logs/kplanid_full.out" 2>&1
  FULL_V=$(cmp_json "$POST/result/kplanid_full.json"); mk KPLANID_FULL "$FULL_V"
  "$PY" "$KITN/mapft_kplanid.py" mappath --base-derivative "$BASE/trace_s0_mappath.npz" --new "$BK/trace_s0_mapft.npz" --base "$BASE/trace_s0.npz" --work "$POST/result/mappath" --script "$MAPPATH_PY" > "$POST/logs/kplanid_mappath.out" 2>&1
  MP_V=$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['MAPPATH_LEG'])" "$POST/result/mappath/mappath_leg.json" 2>/dev/null || echo NOFILE); mk KPLANID_MAPPATH "$MP_V"
  "$PY" "$KITN/mapft_kplanid.py" rast --base-dir "$BASE/rast_s0" --new-dir "$BK/rast_s0_mapft" --json "$POST/result/krastmoved.json" > "$POST/logs/krastmoved.out" 2>&1
  RM_V=$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['K_RAST_MOVED'])" "$POST/result/krastmoved.json" 2>/dev/null || echo NOFILE); mk KRASTMOVED "$RM_V"
  mk VERDICT "K-PLAN-ID=$FULL_V MAPPATH-leg=$MP_V K-RAST-MOVED=$RM_V smoke=$SMOKE_V -> $POST/result/ (kplanid_full.json, mappath/mappath_leg.json, krastmoved.json)"
  if [ "$FULL_V" = PASS ] && [ "$MP_V" = PASS ] && [ "$RM_V" = PASS ]; then echo "ZZMAPFT-CONTROLS-ALL-PASSZZ"; else echo "ZZMAPFT-CONTROLS-NOT-ALL-PASSZZ"; fi
  echo ZZMKENDZZ >> "$CH";;
*) echo "usage: mapft_devbox_post.sh selfcheck|rehearse|fetch|kplanid [--no-full] [--post DIR]"; exit 2;;
esac
