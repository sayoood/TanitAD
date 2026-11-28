#!/usr/bin/env bash
# OPTB read chain (dev box, CPU only, tier T0). One checkpoint at a time, in the box-2 chain's RUN_ORDER. Polling is by script (sleep), never by model turns.
# Per tag: wait for the bank (optb_<tag>/BANK.done + RUN.done) -> evidence (sha256 vs RUN.done, C2, C3, C5, computed) -> planner (HWM oracle, seeds 0 1, lam 0) -> IA-3P-K read (term's 1 sd)
#          + beside read at IA-3's 1.0 m/s^2 -> BAND-MET (ia_read) -> manifest + mech refreshed. Every stage writes <tag>.<stage>.done, so a rerun resumes.
# Sentinels live in their OWN files, never in chain.log:  $OUT/ZZ_STOP.txt ("ZZ OPTB-READ-STOP"), $OUT/ZZ_DONE.txt ("ZZ OPTB-READ-DONE"), $OUT/ZZ_ENDED_BY_BOX2_STOP.txt ("ZZ OPTB-READ-ENDED-BY-BOX2-STOP").
# A box-2 STOP ($STATE/STOP.txt) ends this chain after the arms already banked. The first failure of any stage stops the chain (the stage log is named in the sentinel).
set -u
export PATH="/usr/bin:/bin:$PATH"
PY="${OPTB_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
RC="${OPTB_RC:-F:/TanitAD-scratch/actrsp/optb/readchain}"
TREE="${OPTB_TREE:-F:/TanitAD-scratch/actrsp/optb/tree}"
PULLS="${OPTB_PULLS:-C:/Users/Admin/t2v8f_pulls}"
STATE="${OPTB_STATE:-C:/Users/Admin/optb_state}"
OUT="${OPTB_RC_OUT:-F:/TanitAD-scratch/actrsp/optb/readchain/out}"
PLANS="${OPTB_PLANS:-F:/TanitAD-scratch/actrsp/optb/plans}"
READ="${OPTB_READ:-F:/TanitAD-scratch/actrsp/optb/read}"
LB="F:/TanitAD-scratch/actrsp/lonplan/build"
WC="F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py"
POLL_S="${OPTB_READ_POLL_S:-120}"; MAX_POLLS="${OPTB_READ_MAX_POLLS:-1500}"       # 120 s x 1500 = 50 h per tag
TAGS="${OPTB_READ_TAGS:-FTRSP_s0 FTNULL_s0 FTRSP_s2 FTNULL_s2 FTNULL_s4 FTRSP_s1 FTNULL_s1 FTRSP_s3 FTNULL_s3 FTNULL_s5}"
PLAN_SEEDS="${OPTB_PLAN_SEEDS:-0 1}"; PLAN_EXTRA="${OPTB_PLAN_EXTRA:-}"            # dry run only: PLAN_EXTRA="--n 8"
CHASSIS_OF_TAG() { case "$1" in *_s0|*_s2|*_s4) echo 0;; *_s1|*_s3|*_s5) echo 1;; *) echo "${OPTB_CHASSIS_OVERRIDE:-}";; esac; }
mkdir -p "$OUT"/{logs,state,evidence,reads,reads_beside,bandmet,ia_read,tmp} "$PLANS"
log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$OUT/chain.log"; }
stop() { printf 'ZZ OPTB-READ-STOP %s %s rc=%s (%s)\n' "$1" "$2" "$3" "${4:-}" > "$OUT/ZZ_STOP.txt"; log "stop at $1/$2 rc=$3 (sentinel file written)"; exit "$3"; }
[ -e "$OUT/ZZ_STOP.txt" ] && { echo "REFUSED: $OUT/ZZ_STOP.txt exists; inspect and move it aside" >&2; exit 2; }
rm -f "$OUT/ZZ_DONE.txt" "$OUT/ZZ_ENDED_BY_BOX2_STOP.txt"
[ "${OPTB_READ_DRY:-0}" = 1 ] || { RO="$("$PY" -I -c "import sys; sys.path.insert(0, sys.argv[1]); import optb_check as C; print(' '.join(C.RUN_ORDER))" "${OPTB_PKG:-C:/Users/Admin/optb_pkg}" 2>/dev/null | tr -d '\r')"; [ "$RO" = "$TAGS" ] || stop start run-order 5 "run order differs from the box-2 checker: [$RO] vs [$TAGS]"; }

# CPU environment of every LON-PLAN run (two threads: dumps at other thread counts drift)
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 TMP="$OUT/tmp" TEMP="$OUT/tmp"
export OPTB_TREE="$TREE" WMACT_CODE_ROOT="$TREE" LADDER_READ_CODE_ROOT="$TREE"
export PYTHONPATH="$(cygpath -w "$TREE/stack");$(cygpath -w "$TREE/taniteval")"

cpu_gate() { "$PY" "$WC" 5.5 120 >> "$OUT/logs/wait_commit.log" 2>&1 || stop "$1" commit-low 6 "commit stayed under 5.5 GB"; }
done_() { [ -e "$OUT/state/$1.$2.done" ]; }
mark() { date '+%F %T' > "$OUT/state/$1.$2.done"; }

# per-chassis constants (windows, windows sha256, TRAIN latents): the chain of record's score_chain.py values
win_of()  { case "$1" in 0) echo "F:/TanitAD-scratch/actsens/read/out/C7_R35_s0/win/windows.npz";; 1) echo "F:/TanitAD-scratch/actrsp/lonplan/score_windows_s1/windows.npz";; esac; }
wsha_of() { case "$1" in 0) echo 371c753bcc3f28f1052c0e5e7e4898c0bbbed6140477b3338c8133a3632bdaee;; 1) echo 8833e2923c1781d7678f8483775487db476117e3ddd563a444d2d122334061f0;; esac; }

wait_bank() {   # 0 banked, 7 ended by a box-2 STOP before this arm banked
  local tag="$1" bd="$PULLS/optb_$1" n=0
  while :; do
    [ -e "$bd/BANK.done" ] && [ -e "$bd/RUN.done" ] && return 0
    if [ -e "$STATE/STOP.txt" ] || [ -e "$STATE/ARM_REFUSED_$tag.txt" ]; then
      sleep 5; [ -e "$bd/BANK.done" ] && [ -e "$bd/RUN.done" ] && return 0
      return 7
    fi
    n=$((n + 1)); [ "$n" -le "$MAX_POLLS" ] || stop "$tag" wait-timeout 8 "no bank after $n polls"
    sleep "$POLL_S"
  done
}

for TAG in $TAGS; do
  K="$(CHASSIS_OF_TAG "$TAG")"; [ -n "$K" ] || stop "$TAG" chassis 5 "no chassis for the tag"
  BD="$PULLS/optb_$TAG"
  if ! done_ "$TAG" wait; then
    log "$TAG: waiting for $BD/BANK.done"
    wait_bank "$TAG"; rc=$?
    if [ "$rc" = 7 ]; then printf 'ZZ OPTB-READ-ENDED-BY-BOX2-STOP before %s (box-2 STOP or ARM_REFUSED present: %s)\n' "$TAG" "$(head -c 200 "$STATE/STOP.txt" 2> /dev/null | tr '\r\n' '  ')" > "$OUT/ZZ_ENDED_BY_BOX2_STOP.txt"; log "ended by box-2 stop before $TAG"; exit 0; fi
    mark "$TAG" wait
  fi
  # ---- evidence: sha256 of the banked files against RUN.done, C2 / C3 / C5 computed from the banked files
  if ! done_ "$TAG" evidence; then
    cpu_gate "$TAG"
    ECH=""; [ -n "${OPTB_EVIDENCE_ARGS:-}" ] && ECH="$OPTB_EVIDENCE_ARGS"
    # shellcheck disable=SC2086
    "$PY" -I "$RC/readchain_tools.py" evidence --tag "$TAG" --bank "$BD" --out "$OUT/evidence/$TAG.json" $ECH > "$OUT/logs/$TAG.evidence.log" 2>&1; rc=$?
    [ "$rc" = 0 ] || stop "$TAG" evidence "$rc" "$OUT/logs/$TAG.evidence.log"
    mark "$TAG" evidence
  fi
  # ---- planner: HWM oracle, inference seeds 0 and 1, lam 0, hold; the arm's own checkpoint and its chassis' windows / TRAIN latents; fresh out dir
  if ! done_ "$TAG" plan; then
    cpu_gate "$TAG"
    CKSHA="$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['evidence']['ckpt_sha256'])" "$BD/RUN.done" | tr -d '\r')"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    "$PY" "$RC/optb_route.py" "$LB/lonplan_run.py" plan --windows-src "$(win_of "$K")" --windows-sha256 "$(wsha_of "$K")" --declare-lam 0 --declare-seed-mode hold --ckpt-sha256 "$CKSHA" -- \
      --tag "$TAG" --ckpt "$BD/ckpt_last.pt" --train-latents "F:/TanitAD-scratch/hwm_plan/latents_train_R35_s$K.pt" --train-tag "R35_s$K" --seed-mode hold --lam=0 --arms hwm_oracle --seeds $PLAN_SEEDS \
      --workers 1 $PLAN_EXTRA --out "$PLANS/$TAG" > "$OUT/logs/$TAG.plan.log" 2>&1; rc=$?
    [ "$rc" = 0 ] || stop "$TAG" plan "$rc" "$OUT/logs/$TAG.plan.log"
    # MM fix 2026-11-18: lonplan_run.py writes no DONE; the chain of record (lonplan/score_chain.py:64) writes it after rc 0. Here it is written only after every seed's plan file exists.
    for s in $PLAN_SEEDS; do [ -s "$PLANS/$TAG/plans_hwm_oracle_s$s.npz" ] || stop "$TAG" plan-noseed 9 "no plans_hwm_oracle_s$s.npz in $PLANS/$TAG"; done
    printf 'ok seeds %s\n' "$PLAN_SEEDS" > "$PLANS/$TAG/DONE"
    echo "plan_wall_s=$(( $(date +%s) - t0 ))" > "$OUT/state/$TAG.plan.wall"
    mark "$TAG" plan
  fi
  # ---- IA-3P-K read (term's own 1 sd contrast) and, beside with no bar, IA-3's 1.0 m/s^2 contrast
  if ! done_ "$TAG" ia3pk; then
    cpu_gate "$TAG"; t0=$(date +%s)
    "$PY" "$RC/optb_route.py" "$READ/ia3pk_read.py" read --ckpt "$BD" --chassis "$K" --tag "$TAG" --out "$OUT/reads/ia3pk_$TAG.json" > "$OUT/logs/$TAG.ia3pk.log" 2>&1; rc=$?
    [ "$rc" = 0 ] || stop "$TAG" ia3pk "$rc" "$OUT/logs/$TAG.ia3pk.log"
    D="$("$PY" "$RC/optb_route.py" "$RC/readchain_tools.py" delta --ckpt "$BD" 2> "$OUT/logs/$TAG.delta.log" | tail -1 | "$PY" -I -c "import json,sys; print(json.loads(sys.stdin.read())['delta_wm_1ms2'])" | tr -d '\r')"
    [ -n "$D" ] || stop "$TAG" delta 10 "$OUT/logs/$TAG.delta.log"
    "$PY" "$RC/optb_route.py" "$READ/ia3pk_read.py" read --ckpt "$BD" --chassis "$K" --tag "$TAG" --out "$OUT/reads_beside/ia3pk_$TAG.json" --delta "$D" > "$OUT/logs/$TAG.ia3pk_beside.log" 2>&1; rc=$?
    [ "$rc" = 0 ] || stop "$TAG" ia3pk-beside "$rc" "$OUT/logs/$TAG.ia3pk_beside.log"
    echo "ia3pk_wall_s=$(( $(date +%s) - t0 )) delta_1ms2=$D" > "$OUT/state/$TAG.ia3pk.wall"
    mark "$TAG" ia3pk
  fi
  # ---- BAND-MET (ladder read of the arm, IA-0 at the three band ends) -> band_met_lost against R35's banked passes
  if ! done_ "$TAG" bandmet; then
    cpu_gate "$TAG"; t0=$(date +%s)
    "$PY" "$RC/optb_route.py" "F:/TanitAD-scratch/actsens/read/ia_read.py" --name "$TAG" --run-dir "$BD" --out "$OUT/ia_read/$TAG/ia_read.json" --chassis-seed "$K" ${OPTB_IA_ARGS:-} > "$OUT/logs/$TAG.ia_read.log" 2>&1; rc=$?
    [ "$rc" = 0 ] || stop "$TAG" ia_read "$rc" "$OUT/logs/$TAG.ia_read.log"
    "$PY" -I "$RC/readchain_tools.py" bandmet --ia "$OUT/ia_read/$TAG/ia_read.json" --chassis "$K" --tag "$TAG" --out "$OUT/bandmet/$TAG.json" > "$OUT/logs/$TAG.bandmet.log" 2>&1; rc=$?
    [ "$rc" = 0 ] || stop "$TAG" bandmet "$rc" "$OUT/logs/$TAG.bandmet.log"
    echo "ia_read_wall_s=$(( $(date +%s) - t0 ))" > "$OUT/state/$TAG.bandmet.wall"
    mark "$TAG" bandmet
  fi
  # ---- refresh mech.json and the decide manifest from what exists (cheap; nothing typed)
  "$PY" "$RC/optb_route.py" "$READ/ia3pk_read.py" mech --dir "$OUT/reads" --out "$OUT/mech.json.new" >> "$OUT/logs/refresh.log" 2>&1 && mv -f "$OUT/mech.json.new" "$OUT/mech.json"
  "$PY" -I "$RC/readchain_tools.py" manifest --root "$OUT" --out "$OUT/manifest_decide.json" >> "$OUT/logs/refresh.log" 2>&1 || stop "$TAG" manifest 11 "$OUT/logs/refresh.log"
  log "$TAG: read complete"
done
printf 'ZZ OPTB-READ-DONE all tags read; manifest %s/manifest_decide.json (MM runs optb_decide.py on it)\n' "$OUT" > "$OUT/ZZ_DONE.txt"
log "all tags read"
