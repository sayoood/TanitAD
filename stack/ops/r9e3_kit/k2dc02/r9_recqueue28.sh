#!/bin/bash
# r9_recqueue28.sh <REC SHA40> [K2DC02] -- MM 4a3d05, 2026-11-23 (Thor kit; derived from r9_recqueue27.sh). E3's launch-gate RECORDS for arm K2DC02 (the refcv9 full run, 73,280 steps, profile refcv9-full)
# AND for its early-D1 decay branch K2DC02_d1 (E3_STUDY_NOTE Q1/Q2). G-MAP-OVERFIT / G-BOX-OVERFIT bind the WHOLE launch-argv sha256, and E3's argv (--steps/--save-at/--out/--wsd-*) differs from
# E2's, so E3 needs its OWN records: `map-banded K2DC02` + `box K2DC02` (kinds chosen by the argv: it carries --r9-map-banded), and D1 (another argv) needs `map-banded K2DC02_d1` + `box K2DC02_d1`.
# All four use --argv-prefix r9e3 (the study r9e3: root /home/nvidia/r8ns/r9e3, records land in $R3/ovf_<ARM>_<REC12>). One REC commit serves E3 and D1 (one gate3/REC_C12 stamp per root).
# THE GPU LOCK: the queue WAITS for the Thor GPU lock (thor_gpu.lock) and never forces it: K2DC02cl's chain holds it while that arm trains. The wait is a flock -n poll, 60 s apart.
# E3_STUDY_NOTE Q2 UNVERIFIED -- can the D1 bind run before ckpt_full_30000.pt exists? The D1 `--dry-run` runs FIRST; if it REFUSES, the refusal is a marker (D1-DRY-REFUSED), the D1 GPU binds
# are SKIPPED, and the E3 binds still run (E3 needs only its own two records; D1's records can be bound later, at the SAME REC commit, by this queue's D1 half: R9Q28_D1_ONLY=1).
# Events: $R3/logs/recq28_<c12>.status. Ends ZZRECQ28-DONEZZ (E3 + D1 records) or ZZRECQ28-DONE-E3ONLYZZ (E3 records, D1 refused) or ZZRECQ28-FAIL-<step>ZZ.
#   R9Q28_DRY=1: every dry-run, NO GPU step, no lock wait, ends ZZRECQ28-DRYONLYZZ.   R9Q28_D1_ONLY=1: skip the E3 binds (their records already exist) and bind only D1's.
X=${1:?usage: r9_recqueue28.sh <REC SHA40> [K2DC02]}; K2=${2:-K2DC02}; K2D=${K2}_d1; C12=${X:0:12}
case "$K2" in K2DC02) ;; *) echo "arm must be K2DC02 (this kit binds only that arm and its _d1 branch)"; exit 2;; esac
R3=/home/nvidia/r8ns/r9e3; LT=$R3/rectree_$C12/launch_tree; BIND=$LT/stack/ops/r9e1_bind.sh; RD=$LT/stack/ops/runs.d
LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock; ST=$R3/logs/recq28_$C12.status; L=$R3/logs
mkdir -p $L || exit 2
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $ST; }
fail() { ev "$1"; echo "ZZRECQ28-FAIL-$2ZZ" >> $ST; exit "${3:-4}"; }
[ "$(cat $R3/rectree_$C12/launch_tree.commit 2>/dev/null)" = "$X" ] || fail "REFUSED $LT is not at $X" TREE 2
for A in $K2 $K2D; do
  [ -f $RD/r9e3_$A.argv.json ] || fail "REFUSED no r9e3_$A.argv.json in $LT" NOARGV 2
  [ -f $RD/r9e3_$A.pi_cost_approval.json ] || fail "REFUSED no r9e3_$A.pi_cost_approval.json in $LT" NOAPPROVAL 2
  grep -q '"max_s_per_step": 10.5,' $RD/r9e3_$A.pi_cost_approval.json || fail "REFUSED r9e3_$A.pi_cost_approval.json carries no max_s_per_step 10.5" NOCAP 2
done
# the arms' argv guards (a real python; the same checks the test runs against a fake tree): E3 = the full run, D1 = the decay branch of E3's step-30,000 checkpoint
/usr/bin/python3 -c "
import json,sys
d=json.load(open(sys.argv[1]+'/r9e3_K2DC02.argv.json'))['argv']; b=json.load(open(sys.argv[1]+'/r9e3_K2DC02_d1.argv.json'))['argv']
def one(a,k): return a.count(k)==1
def val(a,k): return a[a.index(k)+1]
ok=(one(d,'--steps') and val(d,'--steps')=='73280' and one(d,'--seed') and val(d,'--seed')=='0' and one(d,'--save-at') and val(d,'--save-at')=='30000'
    and one(d,'--r9-map-banded') and one(d,'--r9-hist-supervise-geom') and one(d,'--map-hires-share-target') and val(d,'--out').endswith('/r9e3/runs/r9e3-K2DC02')
    and '--wsd-branch-from' not in d
    and one(b,'--steps') and val(b,'--steps')=='36000' and one(b,'--seed') and val(b,'--seed')=='0' and '--save-at' not in b and one(b,'--wsd-branch-from')
    and val(b,'--wsd-branch-from').endswith('/r9e3/runs/r9e3-K2DC02/ckpt_full_30000.pt') and val(b,'--wsd-branch-decay')=='6000' and one(b,'--r9-map-banded'))
sys.exit(0 if ok else 1)" $RD 2>/dev/null || fail "REFUSED r9e3_K2DC02 / _d1 argv are not E3 (steps 73280, seed 0, save-at 30000, banded map) and its D1 branch (steps 36000, wsd-branch-from ckpt_full_30000.pt, decay 6000, no save-at)" NOARM 2
ev "QUEUE28-START $C12 code $LT k2 $K2 d1 $K2D mode ${R9Q28_DRY:+DRY}${R9Q28_D1_ONLY:+ D1-ONLY}"
dry() {   # $1 tag, rest = bind args; any ZZR9E1BIND-DRY*OKZZ marker passes; returns 1 (and writes no FAIL) when $DRY_SOFT=1 and the dry-run refuses
  local tag=$1; shift
  bash $BIND --dry-run "$@" > $L/recq28_${C12}_${tag}.dry.out 2>&1 < /dev/null
  grep -qE 'ZZR9E1BIND-DRY[A-Za-z-]*OKZZ' $L/recq28_${C12}_${tag}.dry.out && { ev "$tag dry-run OK"; return 0; }
  [ "${DRY_SOFT:-0}" = 1 ] && { ev "$tag dry-run REFUSED: $(tail -2 $L/recq28_${C12}_${tag}.dry.out | tr '\n' ' ' | cut -c1-220)"; return 1; }
  fail "$tag dry-run FAILED: $(tail -2 $L/recq28_${C12}_${tag}.dry.out | tr '\n' ' ' | cut -c1-220)" "${tag}-DRY" 3
}
gpu() {   # $1 tag, rest = bind args; waits for the lock (never forces it); the step's own ZZR9E1BIND-<kind>-<arm>-OKZZ marker passes
  local tag=$1 i w M f; shift
  # Already bound at THIS REC (an earlier try's .out carries the step's own -OKZZ marker): skip it. A D1-only re-run (R9Q28_D1_ONLY=1) at the pause then
  # binds only what is missing. The marker class includes `_` (arm K2DC02_d1): MEASURED 2026-11-24 00:22:57Z, `[A-Za-z0-9-]` missed
  # ZZR9E1BIND-map-banded-K2DC02_d1-OKZZ and failed a bind that had succeeded.
  for f in $L/recq28_${C12}_${tag}.[0-9].out; do
    [ -f "$f" ] || continue
    M=$(grep -oE 'ZZR9E1BIND-[A-Za-z0-9_-]*ZZ' "$f" | tail -1)
    case "$M" in *-OKZZ) ev "$tag OK already ($M in $(basename "$f")): skipped"; return 0;; esac
  done
  for i in 1 2 3 4 5 6; do
    for w in $(seq 1 1440); do flock -n $LOCKF true && break; sleep 60; done
    ev "$tag try $i: lock free, calling bind"
    bash $BIND "$@" > $L/recq28_${C12}_${tag}.$i.out 2>&1 < /dev/null
    M=$(grep -oE 'ZZR9E1BIND-[A-Za-z0-9_-]*ZZ' $L/recq28_${C12}_${tag}.$i.out | tail -1)
    case "$M" in *-OKZZ) ev "$tag OK ($M)"; return 0;; esac
    grep -q "stayed held" $L/recq28_${C12}_${tag}.$i.out && { ev "$tag try $i lost the lock race; retry"; continue; }
    fail "$tag FAILED (${M:-no marker}): $(tail -3 $L/recq28_${C12}_${tag}.$i.out | tr '\n' ' ' | cut -c1-240)" "$tag"
  done
  fail "$tag: 6 lost races" "${tag}-RACES" 5
}
RE3="--argv-prefix r9e3"
D1OK=1
if [ "${R9Q28_D1_ONLY:-}" != 1 ]; then
  dry mapb_$K2 map-banded $K2 $RE3 --commit $X --code $LT
  dry box_$K2 box $K2 $RE3 --commit $X --code $LT
fi
# the D1 dry-runs: E3_STUDY_NOTE Q2 UNVERIFIED (the bind before ckpt_full_30000.pt exists). A refusal is recorded, never fatal to E3's binds.
DRY_SOFT=1
dry mapb_$K2D map-banded $K2D $RE3 --commit $X --code $LT || D1OK=0
dry box_$K2D box $K2D $RE3 --commit $X --code $LT || D1OK=0
DRY_SOFT=0
if [ "$D1OK" = 1 ]; then ev "D1-DRY-OK both D1 dry-runs passed (Q2: the D1 bind can run before ckpt_full_30000.pt exists)"; else ev "D1-DRY-REFUSED a D1 dry-run refused (see recq28_${C12}_*_d1.dry.out): E3's binds continue, D1's GPU binds are SKIPPED"; fi
ev "dry-runs done (E3 dry-runs, if run, passed)"
if [ "${R9Q28_DRY:-}" = 1 ]; then
  ev "DRY ONLY (R9Q28_DRY=1): no GPU step"; echo "ZZRECQ28-DRYONLYZZ" >> $ST; exit 0
fi
if [ "${R9Q28_D1_ONLY:-}" != 1 ]; then
  gpu mapb_$K2 map-banded $K2 $RE3 --commit $X --code $LT
  gpu box_$K2 box $K2 $RE3 --commit $X --code $LT
  ev "$K2-RECORDS-READY at $C12 (map-banded + box own, E3 argv)"
fi
if [ "$D1OK" = 1 ]; then
  gpu mapb_$K2D map-banded $K2D $RE3 --commit $X --code $LT
  gpu box_$K2D box $K2D $RE3 --commit $X --code $LT
  ev "$K2D-RECORDS-READY at $C12 (map-banded + box own, D1 argv)"
  ev "QUEUE28-DONE"; echo "ZZRECQ28-DONEZZ" >> $ST
else
  ev "QUEUE28-DONE-E3ONLY (D1 records NOT bound: re-run with R9Q28_D1_ONLY=1 after ckpt_full_30000.pt exists, same REC commit, only while no closure-touching land happened)"; echo "ZZRECQ28-DONE-E3ONLYZZ" >> $ST
fi
