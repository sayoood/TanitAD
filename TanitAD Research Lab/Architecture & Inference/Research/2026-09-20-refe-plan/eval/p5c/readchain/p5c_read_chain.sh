#!/usr/bin/env bash
# P5-C read chain: C1 vs C0 (PREREG_P5C sections 5, 6, 8, 11 + A1-A3.3) from "both arms' model_final + sha files are on HF" to the verdict line.
# ONE detached, resumable, serial chain. Marker per stage: $WORK/markers/<stage>.DONE ; failure: $WORK/markers/chain.FAILED (stage, rc, log).
# Re-run = resume (finished stages are skipped). Never trains, never commits, never pushes, never touches the pod or box 2.
# Launch: run_p5c_read.cmd (WMI, CreateFlags 0x08000200).  Docs: RUNBOOK_P5C_READ.md.
#
# CONFIG (env; defaults are the real read):
#   P5C_TIP        tip export of the PINNED commit (taniteval/, stack/, <PK>/eval, <PK>/refe, ...)  default F:/TanitAD_p5cread_tip
#   P5C_WORK       work dir (markers, logs, outputs)                                                 default E:/Projects/TanitAD-artifacts/p5c_read
#   P5C_MODELS     where the pulled arms live: <dir>/C1/model_final.pt + model_final_sha256_C1.txt (same for C0)
#                  default $P5C_WORK/arms
#   P5C_BASE_CKPT  base model_final.pt (P7)  default F:/TanitAD_p7read/run/model_final.pt   (sha256 2613fc17...)
#   P5C_BASE_SEAM / P5C_BASE_DUMP / P5C_BASE_CSV   the base's banked read (default P7: stage2_p7_full.* , score/refe_stage2_p7_full)
#   P5C_DRY=1      DRY RUN: skip HF pull and the GPU stages; arms read from P5C_DRY_SEAM_C1/_C0 (+ _DUMP_, _CSV_), tiny subset P5C_DRY_N (default 64);
#                  the read runs with --rehearsal (no P5-C claim). P5C_DRY_CKPT_C1 / _C0 are the model files that get sealed.
#   P5C_REGRESSION=seal   (dry run only) tamper one sha file: the chain MUST refuse at arms_verified and exit 4.
#   P5C_BASE_SHA   (default: computed from P5C_BASE_CKPT)
#   P5C_F_FLOOR_GB free-space floor on F: for a real run (default 20 = the pinned run's own floor). MM 2026-11-18: the read writes ~nothing to F:
#                  (work, pulled arms, seams and scores all live on E:); the floor only spares other sessions' pinned runs, which refuse below
#                  20 GB on their own, so a lower value never pushes one under its floor. E: keeps its own floor of 3 GB.
set -u
HOLD="${P5C_HOLD:-E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain}"
TIP="${P5C_TIP:-F:/TanitAD_p5cread_tip}"
WORK="${P5C_WORK:-E:/Projects/TanitAD-artifacts/p5c_read}"
DRY="${P5C_DRY:-0}"
PK="$TIP/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
EV="$PK/eval"
PY_T="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PY_N="C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
DATA="${REFE_NAVTEST_DIR:-E:/Projects/TanitAD/data/refe_navtest}"
YAW="${P5C_YAW:-E:/Projects/TanitAD-artifacts/yawrate_wp/tables/ego_yaw_navtest.json.gz}"
LOCK="C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
HF_REPO="${P5C_HF_REPO:-Sayood/tanitad-refe-p5c}"
CARD="$PK/raw/2026-11-11-p5c-readiness/PREREG_P5C.md"
MODELS="${P5C_MODELS:-$WORK/arms}"
BASE_CKPT="${P5C_BASE_CKPT:-F:/TanitAD_p7read/run/model_final.pt}"
BASE_SEAM="${P5C_BASE_SEAM:-$DATA/seams/stage2_p7_full.npz}"
BASE_DUMP="${P5C_BASE_DUMP:-$DATA/seams/stage2_p7_full.proposals.npz}"
[ "$BASE_DUMP" = "NONE" ] && BASE_DUMP=""
BASE_CSV="${P5C_BASE_CSV:-$DATA/score/refe_stage2_p7_full/refe_stage2_p7_full.csv}"
M="$WORK/markers"; L="$WORK/logs"; O="$WORK/out"; PLANS="$WORK/plans"
mkdir -p "$M" "$L" "$O" "$PLANS" "$WORK/tmp" 2>/dev/null

# arms: names, model files, seams
if [ "$DRY" = "1" ]; then
  DN="${P5C_DRY_N:-64}"
  SEAM_C1="${P5C_DRY_SEAM_C1:?}"; DUMP_C1="${P5C_DRY_DUMP_C1:-}"; CSV_C1="${P5C_DRY_CSV_C1:?}"
  SEAM_C0="${P5C_DRY_SEAM_C0:?}"; DUMP_C0="${P5C_DRY_DUMP_C0:-}"; CSV_C0="${P5C_DRY_CSV_C0:?}"
  CKPT_SRC="${P5C_DRY_CKPT_SRC:?}"; CKPT_C1="F:/tanitad_scratch/readchain/dry_models/C1.pt"; CKPT_C0="F:/tanitad_scratch/readchain/dry_models/C0.pt"
  SUBSET_ARGS="--subset $DN"; REH="--rehearsal DRY-RUN-identity-and-pipeline-exercise"
  TACT_EXTRA="--subset-ok"
else
  DN=0
  SEAM_C1="$DATA/seams/stage2_p5c_c1.npz"; DUMP_C1="$DATA/seams/stage2_p5c_c1.proposals.npz"; CSV_C1="$DATA/score/refe_stage2_p5c_c1/refe_stage2_p5c_c1.csv"
  SEAM_C0="$DATA/seams/stage2_p5c_c0.npz"; DUMP_C0="$DATA/seams/stage2_p5c_c0.proposals.npz"; CSV_C0="$DATA/score/refe_stage2_p5c_c0/refe_stage2_p5c_c0.csv"
  CKPT_C1="$MODELS/C1/model_final.pt"; CKPT_C0="$MODELS/C0/model_final.pt"
  SUBSET_ARGS=""; REH=""; TACT_EXTRA=""
fi

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" | tee -a "$WORK/CHAIN.log" >/dev/null; echo "[$(ts)] $*"; }

# ---- exit paths: the shared GPU lock is released on EVERY exit (only if it is ours: a stage-2 job named refe-stage2-p5c_*)
release_lock() {
  if [ -f "$LOCK" ] && grep -q "refe-stage2-p5c_" "$LOCK" 2>/dev/null; then
    rm -f "$LOCK" && say "released the dev-box GPU lock left by a p5c stage-2 job"
  fi
}
trap 'release_lock' EXIT
trap 'say "signal received"; release_lock; exit 143' INT TERM

fail() { # stage rc [exit_code]
  printf 'stage=%s rc=%s time=%s log=%s\n' "$1" "$2" "$(ts)" "$L/$1.log" > "$M/chain.FAILED"
  say "FAILED $1 (rc $2) log $L/$1.log"; release_lock; exit "${3:-1}"
}

if [ -f "$M/chain.FAILED" ]; then
  n=1; while [ -f "$M/chain.FAILED.attempt_$n" ]; do n=$((n+1)); done
  mv "$M/chain.FAILED" "$M/chain.FAILED.attempt_$n"; say "previous chain.FAILED kept as chain.FAILED.attempt_$n; resuming"
fi

# ---- guards
commit_ok() { local g; g=$("$PY_T" "$HOLD/tools/commit_guard.py" | tr -d '\r\n '); case "$g" in ''|*[!0-9]*) return 1;; esac; [ "$g" -ge 6 ]; }
wait_commit() { local i=0; while ! commit_ok; do i=$((i+1)); [ $i -gt "${P5C_COMMIT_WAIT_MIN:-30}" ] && { say "COMMIT-LOW (< 6 GB free commit) after ${P5C_COMMIT_WAIT_MIN:-30} min"; return 1; }; sleep 60; done; return 0; }
free_gb() { # drive letter -> integer GB free (Python, no locale)
  "$PY_T" -c "import shutil,sys;print(int(shutil.disk_usage(sys.argv[1]).free//(1<<30)))" "$1" | tr -d '\r\n '; }

# ---- environment for python stages (semicolons on Windows; module paths asserted in `preflight`)
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$TIP/stack;$TIP/taniteval;$EV;$EV/p5c"
export TANITAD_REPO="$TIP" REFE_EVAL_DIR="$EV" P5C_REFE_DIR="$PK/refe" TANITAD_NAVSIM_ROOT="${TANITAD_NAVSIM_ROOT:-F:/Archive/devbox-C/navsim}"
export TMP="$WORK/tmp" TEMP="$WORK/tmp" PYTHONIOENCODING=utf-8 PYTHONUTF8=1

# run STAGE LOG-NAME cmd... : marker-skipping, serial, commit-guarded
run() {
  local st="$1"; shift
  if [ -f "$M/$st.DONE" ]; then say "skip $st (DONE)"; return 0; fi
  wait_commit || fail "$st" 90
  say "RUN $st: $*"
  local t0; t0=$(date +%s)
  "$@" > "$L/$st.log" 2>&1 < /dev/null
  local rc=$?
  say "  rc $rc after $(( $(date +%s) - t0 )) s (log $L/$st.log)"
  [ $rc -eq 0 ] || fail "$st" $rc
  printf '%s rc=0\n' "$(ts)" > "$M/$st.DONE"
}
# same, but rc 0 or 1 accepted (the P7 chain's seam1 form is handled inside p7_read_chain.py)
mark() { printf '%s %s\n' "$(ts)" "$2" > "$M/$1.DONE"; }

# =============================================================== preflight (ALWAYS runs)
say "chain start; dry=$DRY tip=$TIP work=$WORK"
[ -d "$EV/p5c" ] || { say "REFUSING: no tip export at $TIP (need <PK>/eval/p5c)"; fail preflight 2 2; }
for d in E F; do g=$(free_gb "$d:/"); [ "${g:-0}" -ge 3 ] || { say "REFUSING: $d: has ${g:-?} GB free (< 3)"; fail preflight 2 2; }; done
FFLOOR="${P5C_F_FLOOR_GB:-20}"; [ "$FFLOOR" -ge 3 ] 2>/dev/null || { say "REFUSING: P5C_F_FLOOR_GB=$FFLOOR (integer >= 3 required)"; fail preflight 2 2; }
g=$(free_gb "F:/"); if [ "$DRY" != "1" ] && [ "${g:-0}" -lt "$FFLOOR" ]; then say "REFUSING: F: has $g GB free (< $FFLOOR; default 20 = the pinned run's floor)"; fail preflight 2 2; fi
[ "$FFLOOR" = 20 ] || say "F: floor $FFLOOR GB (P5C_F_FLOOR_GB; default 20); F: free $g GB"
wait_commit || { say "REFUSING: free commit < 6 GB for 30 min"; fail preflight 2 2; }
if [ "$DRY" != "1" ]; then
  for a in C1 C0; do [ -f "$MODELS/$a/model_final_sha256_$a.txt" ] || [ -f "$M/hf_pull_$a.DONE" ] || true; done
fi
"$PY_T" - <<'PYEOF' > "$L/preflight.log" 2>&1 || fail preflight 2 2
import os, sys
tip = os.environ["TANITAD_REPO"].replace("\\", "/").lower()
import tanitad, taniteval
import p5c_read, p5c_seal, p5c_read_prep, p5c_render, p5c_determinism
for m in (tanitad, taniteval, p5c_read, p5c_seal, p5c_read_prep, p5c_render, p5c_determinism):
    f = os.path.abspath(m.__file__).replace("\\", "/").lower()
    assert f.startswith(tip), f"module {m.__name__} imported from {f}, not from the tip export {tip}"
print("module paths ok (all under the tip export)")
PYEOF
say "preflight ok"

[ -f "$BASE_CKPT" ] || { say "REFUSING: base checkpoint $BASE_CKPT missing"; fail preflight 2 2; }
BASE_SHA="${P5C_BASE_SHA:-}"
if [ -z "$BASE_SHA" ]; then BASE_SHA=$("$PY_T" -c "import hashlib,sys;h=hashlib.sha256();f=open(sys.argv[1],'rb')
for b in iter(lambda:f.read(1<<22),b''):h.update(b)
print(h.hexdigest())" "$BASE_CKPT" | tr -d '\r\n '); fi
echo "$BASE_SHA" | grep -qE '^[0-9a-f]{64}$' || fail preflight 2 2

# =============================================================== 1 hf_pull  (CPU/network; real run only)
if [ "$DRY" != "1" ]; then
  for a in C1 C0; do
    run "hf_pull_$a" "$PY_T" "$HOLD/tools/hf_pull.py" --arm "$a" --prefix "p5c/$(echo $a | tr 'A-Z' 'a-z')" --out "$MODELS/$a" --repo "$HF_REPO" || exit 1
  done
fi

if [ "$DRY" = "1" ]; then
  run dry_standins "$PY_T" "$HOLD/tools/make_standins.py" "$CKPT_SRC" "$CKPT_C1" "$CKPT_C0" || exit 1
fi

# =============================================================== 2 arms_verified : computed sha256 == the pod's sha file, BOTH arms, before anything else
verify_arms() {
  for a in C1 C0; do
    if [ "$DRY" = "1" ]; then ck=$(eval echo "\$CKPT_$a"); sf="$WORK/dry_sha/model_final_sha256_$a.txt"; else ck="$MODELS/$a/model_final.pt"; sf="$MODELS/$a/model_final_sha256_$a.txt"; fi
    [ -f "$ck" ] && [ -f "$sf" ] || { echo "REFUSING: $a missing model ($ck) or sha file ($sf)"; return 4; }
    got=$("$PY_T" -c "import hashlib,sys;h=hashlib.sha256();f=open(sys.argv[1],'rb')
for b in iter(lambda:f.read(1<<22),b''):h.update(b)
print(h.hexdigest())" "$ck" | tr -d '\r\n ')
    want=$(grep -oE '[0-9a-f]{64}' "$sf" | head -1)
    if [ "$got" != "$want" ]; then echo "REFUSING: $a sha256 $got != pod sha file $want"; return 4; fi
    echo "$a sha256 $got == pod sha file"; eval "got_$(echo $a | tr A-Z a-z)=$got"
  done
  if [ "$DRY" != "1" ] && [ "$got_c1" = "$got_c0" ]; then echo "REFUSING: C1 and C0 have the same sha256 (the arms must differ)"; return 4; fi
  return 0
}
if [ ! -f "$M/arms_verified.DONE" ]; then
  if [ "$DRY" = "1" ]; then
    mkdir -p "$WORK/dry_sha"
    for a in C1 C0; do ck=$(eval echo "\$CKPT_$a"); "$PY_T" -c "import hashlib,sys;h=hashlib.sha256();f=open(sys.argv[1],'rb')
for b in iter(lambda:f.read(1<<22),b''):h.update(b)
print(h.hexdigest())" "$ck" | tr -d '\r\n ' > "$WORK/dry_sha/model_final_sha256_$a.txt"; done
    if [ "${P5C_REGRESSION:-}" = "seal" ]; then printf '%064d' 0 > "$WORK/dry_sha/model_final_sha256_C0.txt"; say "REGRESSION ARM: C0's sha file replaced by 64 zeros; the chain MUST refuse"; fi
  fi
  verify_arms > "$L/arms_verified.log" 2>&1; rc=$?
  cat "$L/arms_verified.log"
  [ $rc -eq 0 ] || fail arms_verified $rc 4
  mark arms_verified ok
fi
[ "${P5C_REGRESSION:-}" = "seal" ] && [ -f "$M/arms_verified.DONE" ] && { say "REGRESSION ARM NOT REFUSED: the chain is blind"; exit 5; }

# =============================================================== 3 seal : BOTH arms in ONE manifest, BEFORE any navtest pass (card section 11)
SEALED="$O/SEALED.json"
run seal "$PY_T" "$EV/p5c/p5c_seal.py" seal --arm "C1=$CKPT_C1" --arm "C0=$CKPT_C0" --card "$CARD" --base-sha "$BASE_SHA" --out "$SEALED" || exit 1
run seal_verify "$PY_T" "$EV/p5c/p5c_seal.py" verify --sealed "$SEALED" --card "$CARD" --recheck-files --out "$O/sealed_verify.json" || exit 1

# =============================================================== 4 navtest pass per arm (GPU, shared lock taken INSIDE stage2_sft.py; P7 read chain steps seam1 + score)
if [ "$DRY" != "1" ]; then
  for a in c1 c0; do
    A=$(echo "$a" | tr 'a-z' 'A-Z'); ck=$(eval echo "\$CKPT_$A")
    sha=$(grep -oE '[0-9a-f]{64}' "$MODELS/$A/model_final_sha256_$A.txt" | head -1)
    # CUDA_VISIBLE_DEVICES is for CPU stages: the GPU stage must see the GPU
    run "s2_$a" env -u CUDA_VISIBLE_DEVICES "$PY_T" "$EV/p7_read_chain.py" --ckpt "$ck" --name "p5c_$a" --expect-ckpt-sha256 "$sha" \
        --base p6 --also p2 --ego-yaw-table "$YAW" --work "$WORK/s2_$a" --only seam1,score || exit 1
    release_lock
  done
fi

# =============================================================== 5 CPU stages
# 5a convert: seams (+dumps) -> plans npz for C1, C0, base (the registered pick20 form whenever a dump exists)
arm_spec() { local s="$2" d="$3"; if [ -n "$d" ] && [ -f "$d" ]; then echo "$1=$s,$d"; else echo "$1=$s"; fi; }
run convert "$PY_T" "$EV/p5c/p5c_read_prep.py" convert --arm "$(arm_spec C1 "$SEAM_C1" "$DUMP_C1")" "$(arm_spec C0 "$SEAM_C0" "$DUMP_C0")" "$(arm_spec base "$BASE_SEAM" "$BASE_DUMP")" \
    --outdir "$PLANS" $SUBSET_ARGS || exit 1
# 5b ONE transp_score pass over {C1, C0, base, human} (navsim venv, <= 2 procs, metric cache)
run score_transp "$PY_T" "$EV/p5c/p5c_read_prep.py" score --plans-dir "$PLANS" --out-dir "$O/pass1" --procs 2 --python "$PY_N" || exit 1
# 5c identity control (B6) per model arm: tracked PDMS == the pinned harness csv, bar 1e-6
run identity_C1 "$PY_T" "$EV/p5c/p5c_read_prep.py" identity --rows "$O/pass1/rows_s*.jsonl" --arm C1 --csv "$CSV_C1" --plans "$PLANS/plans_C1.npz" --out "$O/identity_C1.json" || exit 1
run identity_C0 "$PY_T" "$EV/p5c/p5c_read_prep.py" identity --rows "$O/pass1/rows_s*.jsonl" --arm C0 --csv "$CSV_C0" --plans "$PLANS/plans_C0.npz" --out "$O/identity_C0.json" || exit 1
run identity_base "$PY_T" "$EV/p5c/p5c_read_prep.py" identity --rows "$O/pass1/rows_s*.jsonl" --arm base --csv "$BASE_CSV" --plans "$PLANS/plans_base.npz" --out "$O/identity_base.json" || exit 1
# 5d RED arm of the identity control (cache-routing line deleted must FAIL identity; exit 0 iff detected). Not a read control; evidence it can fail.
run identity_red "$PY_T" "$EV/p5c/p5c_read_prep.py" red --plans-dir "$PLANS" --arm C1 --csv "$CSV_C1" --out-dir "$O/red" --n 60 --python "$PY_N" || exit 1
# 5e determinism (B10): seeded 10 % subset, second pass through the SAME scorer, bit-identical required
run det_select "$PY_T" "$EV/p5c/p5c_determinism.py" select --plans-dir "$PLANS" --out "$O/det_subset.json" || exit 1
run det_plans "$PY_T" "$EV/p5c/p5c_determinism.py" plans --plans-dir "$PLANS" --subset "$O/det_subset.json" --out-dir "$O/det_plans" || exit 1
run det_pass2 "$PY_T" "$EV/p5c/p5c_read_prep.py" score --plans-dir "$O/det_plans" --out-dir "$O/pass2" --procs 2 --python "$PY_N" || exit 1
run det_compare "$PY_T" "$EV/p5c/p5c_determinism.py" compare --rows1 "$O/pass1/rows_s*.jsonl" --rows2 "$O/pass2/rows_s*.jsonl" --subset "$O/det_subset.json" \
    --arms C1 C0 base --human human --out "$O/determinism.json" || exit 1
# 5f four families, both frames, paired C1-C0 (+ C0-base, C1-base)
run families "$PY_T" "$EV/p5c/p5c_families.py" --rows "$O/pass1/rows_s*.jsonl" --arms C1 C0 base --pairs C1:C0 C0:base C1:base --out-dir "$O/families" \
    --python "$PY_N" --repo "$TIP" || exit 1
# 5g tactical family (command adherence; selection quality needs the 64-proposal harness tables = OPTIONAL, ~4-6.4 h CPU per arm, not in this chain)
TD=""; [ -n "$DUMP_C1" ] && [ -f "$DUMP_C1" ] && TD="--dump C1=$DUMP_C1 C0=$DUMP_C0"; [ -n "$TD" ] && [ -n "$BASE_DUMP" ] && [ -f "$BASE_DUMP" ] && TD="$TD base=$BASE_DUMP"
TF=""; [ "$DRY" = "1" ] && TF="--tokens-from $PLANS/plans_C1.npz"
if [ "$DRY" = "1" ] && [ -n "$TD" ]; then     # REHEARSAL ONLY (MM 2026-11-17): cut the dumps to the rehearsal tokens; the real read passes the full dumps
  mkdir -p "$O/dump_subset"; TD="--dump"
  for pair in "C1=$DUMP_C1" "C0=$DUMP_C0" "base=$BASE_DUMP"; do
    k="${pair%%=*}"; f="${pair#*=}"; [ -n "$f" ] && [ -f "$f" ] || continue
    "$PY_T" "$HOLD/tools/subset_dump.py" "$f" "$PLANS/plans_C1.npz" "$O/dump_subset/$k.npz" >> "$WORK/logs/dump_subset.log" 2>&1 || { say "dump subset failed for $k"; exit 1; }
    TD="$TD $k=$O/dump_subset/$k.npz"
  done
fi
run tactical "$PY_T" "$EV/p5c/p5c_tactical.py" --seam "C1=$SEAM_C1" "C0=$SEAM_C0" "base=$BASE_SEAM" $TD $TF $TACT_EXTRA --out "$O/tactical.json" || exit 1
# 5h STRATEGIC family, REPORTED beside the read (PI 2026-11-16), no bar; its own commit guard is in the CLI
run strategic "$PY_T" "$EV/instruments/strategic_family.py" read --arm "C1=$SEAM_C1" --base "C0=$SEAM_C0" --out "$O/strategic_family_C1_C0.json" || exit 1

# =============================================================== 6 the read, render, verdict line
run read "$PY_T" "$EV/p5c/p5c_read.py" --rows "$O/pass1/rows_s*.jsonl" --c1 C1 --c0 C0 --base base --human human \
    --fam-ci "$O/families/famci_C1_vs_C0.json" --controls "$O/identity_C1.json" "$O/identity_C0.json" "$O/identity_base.json" "$O/determinism.json" \
    --tactical "$O/tactical.json" --sealed "$SEALED" --card "$CARD" --recheck-files --require-kinds B10_determinism B11_sealed $REH --out "$O/read.json" || exit 1
run render "$PY_T" "$EV/p5c/p5c_render.py" --read "$O/read.json" --out "$O/report.md" || exit 1
run verdict "$PY_T" "$HOLD/tools/verdict_line.py" --read "$O/read.json" --strategic "$O/strategic_family_C1_C0.json" --out "$O/VERDICT.txt" || exit 1

cat "$O/VERDICT.txt" | tee -a "$WORK/CHAIN.log"
printf '%s\n' "$(ts)" > "$M/chain.DONE"
say "chain DONE"
exit 0
