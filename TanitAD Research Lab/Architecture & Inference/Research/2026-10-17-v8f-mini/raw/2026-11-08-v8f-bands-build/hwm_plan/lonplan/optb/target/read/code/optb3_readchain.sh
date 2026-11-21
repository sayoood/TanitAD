#!/usr/bin/env bash
# OPTB3 read chain (dev box, CPU only, tier T0). Adapted from the Amendment 2 chain (actrsp/optb/readchain/optb_readchain.sh); AMEND_OPTB3_TARGET.md (sha256 cb4f943f) changes only the arms.
# Four banked arms, in the card's order: FTT15_s0 FTT15_s1 FTT125_s0 FTT125_s1 (banks C:/Users/Admin/t2v8f_pulls/optb3_<tag>/). Per tag: evidence (sha256, C2, C3, C5 from the banked files) -> planner (HWM oracle,
# inference seeds 0 1, lam 0) -> IA-3P-K read + beside read -> BAND-MET (ia_read). Then optb3_decide.py (floors, paired deltas, attribution, MECH-RAISED, TARGET-READ, families grid, section-4 labels).
# Each stage writes <tag>.<stage>.done, so a rerun resumes. Sentinel (ONE line, in its own file $OUT/ZZ_OPTB3READ.txt and as the last line of the driver stdout):
#   ZZ OPTB3READ DONE ZZ   |   ZZ OPTB3READ FAIL ZZ <reason>
# Planner, ia3pk, ia_read and the delta step run through route/optb_route.py (tree = 0f471d57's stack + taniteval, blob-verified, manifest re-checked on every start).
set -u
export PATH="/usr/bin:/bin:$PATH"
PY="${OPTB3_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
W3="${OPTB3_W:-F:/TanitAD-scratch/optb3_read}"
RCA2="F:/TanitAD-scratch/actrsp/optb/readchain"          # Amendment 2 tools, reused unchanged (readchain_tools.py)
ROUTE="$W3/route/optb_route.py"
TREE="${OPTB3_TREE:-$W3/tree}"
PULLS="${OPTB3_PULLS:-C:/Users/Admin/t2v8f_pulls}"
OUT="${OPTB3_OUT:-$W3/out}"
PLANS="$OUT/plans"
READ="F:/TanitAD-scratch/actrsp/optb/read"
LB="F:/TanitAD-scratch/actrsp/lonplan/build"
WC="F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py"
TAGS="${OPTB3_TAGS:-FTT15_s0 FTT15_s1 FTT125_s0 FTT125_s1}"
PLAN_SEEDS="${OPTB3_PLAN_SEEDS:-0 1}"; PLAN_EXTRA="${OPTB3_PLAN_EXTRA:-}"          # smoke only: PLAN_EXTRA="--n 8"
SMOKE="${OPTB3_SMOKE:-0}"
export OPTB_PKG="$W3/shim"                                                        # readchain_tools.load_checker() -> optb3_check through the shim
mkdir -p "$OUT"/{logs,state,evidence,reads,reads_beside,bandmet,ia_read,tmp} "$PLANS"
log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$OUT/chain.log"; }
fail() { printf 'ZZ OPTB3READ FAIL ZZ %s/%s rc=%s (%s)\n' "$1" "$2" "$3" "${4:-}" > "$OUT/ZZ_OPTB3READ.txt"; cat "$OUT/ZZ_OPTB3READ.txt"; log "stop at $1/$2 rc=$3"; exit "$3"; }
[ -e "$OUT/ZZ_OPTB3READ.txt" ] && { echo "REFUSED: $OUT/ZZ_OPTB3READ.txt exists; inspect and move it aside"; exit 2; }
CHASSIS_OF_TAG() { case "$1" in *_s0) echo 0;; *_s1) echo 1;; *) echo "";; esac; }

export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 TMP="$OUT/tmp" TEMP="$OUT/tmp"
export OPTB_TREE="$TREE" WMACT_CODE_ROOT="$TREE" LADDER_READ_CODE_ROOT="$TREE"
export PYTHONPATH="$(cygpath -w "$TREE/stack");$(cygpath -w "$TREE/taniteval")"
if [ "$SMOKE" != 1 ]; then
  RO="$("$PY" -I -c "import sys; sys.path.insert(0, sys.argv[1]); import optb_check as C; print(' '.join(C.RUN_ORDER))" "$OPTB_PKG" 2>/dev/null | tr -d '\r')"
  [ "$RO" = "$TAGS" ] || fail start run-order 5 "run order differs from the OPTB3 checker: [$RO] vs [$TAGS]"
fi
GATE_GB="${OPTB3_GATE_GB:-5.5}"      # production: 5.5 GB free commit. Only the one-arm smoke (n=8 windows, small footprint) lowers it, by env, while another session holds the commit.
cpu_gate() { "$PY" "$WC" "$GATE_GB" 120>> "$OUT/logs/wait_commit.log" 2>&1 || fail "$1" commit-low 6 "commit stayed under 5.5 GB"; }
# ia_read (code of record, lr_lib.memory_guard) raises MemoryError under 6.0 GB FREE PHYSICAL memory. Wait for it (poll 60 s, up to 180 min) rather than fail the stage; the evaluator is not edited.
phys_gate() {
  local n=0
  while :; do
    f="$("$PY" -I -c "import ctypes
class M(ctypes.Structure): _fields_=[('l',ctypes.c_ulong),('ml',ctypes.c_ulong),('tp',ctypes.c_ulonglong),('ap',ctypes.c_ulonglong),('tpf',ctypes.c_ulonglong),('apf',ctypes.c_ulonglong),('tv',ctypes.c_ulonglong),('av',ctypes.c_ulonglong),('ave',ctypes.c_ulonglong)]
m=M(); m.l=ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)); print(int(m.ap/2**30*100))" | tr -d '\r')"
    [ "${f:-0}" -ge "${OPTB3_PHYS_GB100:-650}" ] && return 0           # 6.5 GB: margin over the 6.0 guard
    n=$((n + 1)); [ "$n" -le 180 ] || fail "$1" phys-low 6 "free physical memory stayed under the ia_read guard for 180 min"
    sleep 60
  done
}
done_() { [ -e "$OUT/state/$1.$2.done" ]; }
mark() { date '+%F %T' > "$OUT/state/$1.$2.done"; }
win_of()  { case "$1" in 0) echo "F:/TanitAD-scratch/actsens/read/out/C7_R35_s0/win/windows.npz";; 1) echo "F:/TanitAD-scratch/actrsp/lonplan/score_windows_s1/windows.npz";; esac; }
wsha_of() { case "$1" in 0) echo 371c753bcc3f28f1052c0e5e7e4898c0bbbed6140477b3338c8133a3632bdaee;; 1) echo 8833e2923c1781d7678f8483775487db476117e3ddd563a444d2d122334061f0;; esac; }

for TAG in $TAGS; do
  K="$(CHASSIS_OF_TAG "$TAG")"; [ -n "$K" ] || fail "$TAG" chassis 5 "no chassis for the tag"
  BD="$PULLS/optb3_$TAG"
  [ -e "$BD/BANK.done" ] && [ -e "$BD/RUN.done" ] || fail "$TAG" bank 7 "no BANK.done / RUN.done in $BD"
  if ! done_ "$TAG" evidence; then
    cpu_gate "$TAG"
    "$PY" -I "$RCA2/readchain_tools.py" evidence --tag "$TAG" --bank "$BD" --out "$OUT/evidence/$TAG.json" > "$OUT/logs/$TAG.evidence.log" 2>&1; rc=$?
    [ "$rc" = 0 ] || fail "$TAG" evidence "$rc" "$OUT/logs/$TAG.evidence.log"
    mark "$TAG" evidence
  fi
  # BAND-MET first: its model path (ia_read on the 0f471d57 tree) is the one stage the smoke could not run (6 GB free-physical guard), so a defect surfaces within minutes
  if ! done_ "$TAG" bandmet; then
    cpu_gate "$TAG"; phys_gate "$TAG"; t0=$(date +%s)
    "$PY" "$ROUTE" "F:/TanitAD-scratch/actsens/read/ia_read.py" --name "$TAG" --run-dir "$BD" --out "$OUT/ia_read/$TAG/ia_read.json" --chassis-seed "$K" ${OPTB3_IA_ARGS:-} > "$OUT/logs/$TAG.ia_read.log" 2>&1; rc=$?
    [ "$rc" = 0 ] || fail "$TAG" ia_read "$rc" "$OUT/logs/$TAG.ia_read.log"
    "$PY" -I "$RCA2/readchain_tools.py" bandmet --ia "$OUT/ia_read/$TAG/ia_read.json" --chassis "$K" --tag "$TAG" --out "$OUT/bandmet/$TAG.json" > "$OUT/logs/$TAG.bandmet.log" 2>&1; rc=$?
    [ "$rc" = 0 ] || fail "$TAG" bandmet "$rc" "$OUT/logs/$TAG.bandmet.log"
    echo "ia_read_wall_s=$(( $(date +%s) - t0 )) " > "$OUT/state/$TAG.bandmet.wall"
    mark "$TAG" bandmet
  fi
  if ! done_ "$TAG" plan; then
    cpu_gate "$TAG"
    CKSHA="$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['evidence']['ckpt_sha256'])" "$BD/RUN.done" | tr -d '\r')"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    "$PY" "$ROUTE" "$LB/lonplan_run.py" plan --windows-src "$(win_of "$K")" --windows-sha256 "$(wsha_of "$K")" --declare-lam 0 --declare-seed-mode hold --ckpt-sha256 "$CKSHA" -- \
      --tag "$TAG" --ckpt "$BD/ckpt_last.pt" --train-latents "F:/TanitAD-scratch/hwm_plan/latents_train_R35_s$K.pt" --train-tag "R35_s$K" --seed-mode hold --lam=0 --arms hwm_oracle --seeds $PLAN_SEEDS \
      --workers 1 $PLAN_EXTRA --out "$PLANS/$TAG" > "$OUT/logs/$TAG.plan.log" 2>&1; rc=$?
    [ "$rc" = 0 ] || fail "$TAG" plan "$rc" "$OUT/logs/$TAG.plan.log"
    for s in $PLAN_SEEDS; do [ -s "$PLANS/$TAG/plans_hwm_oracle_s$s.npz" ] || fail "$TAG" plan-noseed 9 "no plans_hwm_oracle_s$s.npz in $PLANS/$TAG"; done
    printf 'ok seeds %s\n' "$PLAN_SEEDS" > "$PLANS/$TAG/DONE"
    echo "plan_wall_s=$(( $(date +%s) - t0 ))" > "$OUT/state/$TAG.plan.wall"
    mark "$TAG" plan
  fi
  if ! done_ "$TAG" ia3pk; then
    cpu_gate "$TAG"; t0=$(date +%s)
    "$PY" "$ROUTE" "$READ/ia3pk_read.py" read --ckpt "$BD" --chassis "$K" --tag "$TAG" --out "$OUT/reads/ia3pk_$TAG.json" > "$OUT/logs/$TAG.ia3pk.log" 2>&1; rc=$?
    [ "$rc" = 0 ] || fail "$TAG" ia3pk "$rc" "$OUT/logs/$TAG.ia3pk.log"
    D="$("$PY" "$ROUTE" "$RCA2/readchain_tools.py" delta --ckpt "$BD" 2> "$OUT/logs/$TAG.delta.log" | tail -1 | "$PY" -I -c "import json,sys; print(json.loads(sys.stdin.read())['delta_wm_1ms2'])" | tr -d '\r')"
    [ -n "$D" ] || fail "$TAG" delta 10 "$OUT/logs/$TAG.delta.log"
    "$PY" "$ROUTE" "$READ/ia3pk_read.py" read --ckpt "$BD" --chassis "$K" --tag "$TAG" --out "$OUT/reads_beside/ia3pk_$TAG.json" --delta "$D" > "$OUT/logs/$TAG.ia3pk_beside.log" 2>&1; rc=$?
    [ "$rc" = 0 ] || fail "$TAG" ia3pk-beside "$rc" "$OUT/logs/$TAG.ia3pk_beside.log"
    echo "ia3pk_wall_s=$(( $(date +%s) - t0 )) delta_1ms2=$D" > "$OUT/state/$TAG.ia3pk.wall"
    mark "$TAG" ia3pk
  fi
  log "$TAG: read complete"
done

if [ "$SMOKE" = 1 ]; then
  printf 'ZZ OPTB3READ DONE ZZ smoke (arms read, decide not run)\n' > "$OUT/ZZ_OPTB3READ.txt"; cat "$OUT/ZZ_OPTB3READ.txt"; exit 0
fi
# ---- decide: floors, paired deltas, attribution, MECH-RAISED, TARGET-READ, families grid, labels
if ! done_ ALL decide; then
  cpu_gate decide
  "$PY" "$W3/optb3_decide.py" --root "$OUT" --out "$OUT/DECIDE_OPTB3.json" > "$OUT/logs/decide.log" 2>&1; rc=$?
  [ "$rc" = 0 ] || fail ALL decide "$rc" "$OUT/logs/decide.log"
  mark ALL decide
fi
log "all four arms read; decide written"
printf 'ZZ OPTB3READ DONE ZZ\n' > "$OUT/ZZ_OPTB3READ.txt"; cat "$OUT/ZZ_OPTB3READ.txt"
