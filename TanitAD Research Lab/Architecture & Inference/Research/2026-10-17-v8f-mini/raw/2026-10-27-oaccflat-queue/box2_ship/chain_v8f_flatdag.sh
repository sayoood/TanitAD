#!/usr/bin/env bash
# PREREG_V8F_FLATDAG on box 2: DAgger ON THE CUT. FLAT (M2b_s<S>_flat, O's interface cut: O reads constants, e_code the declared dead weight) is the BASE and the bank GENERATOR; the two arms are fine-tuned FROM FLAT with DAG's recipe:
#   FLATDAG  = FLAT_s<S> + 3,000 steps, --o-interface cut NAMED, --dagger-p 0.1 on FLAT's OWN-STATE bank (dagger_bank_flat/s<S>), DAG's recipe otherwise
#   FLATDAG-0 = the same argv with --dagger-p 0 (no bank / hash / seed flags): the pair differs in --dagger-p, the bank flags and --out ONLY (C-F2)
# This file is GENERATED from chain_v8f_flatdag.sh.in by make_chains.py (FLAT's checkpoint / argv.json / summary.json sha256 values are computed from the banked artifacts, never retyped; the export list from export_closure_flatdag.json).
# BUILT, NOT LAUNCHED: it runs only on the PI's go. It is the DAGGER2 chain's shape with FLAT as the generator (no DAG-read predicate) and the shared functions of oaf_lib.sh.
# One chain, in PREREG_V8F_FLATDAG's order, ONE job at a time on box 2 (mk_v8f_job.py, SAFE MODE):
#   1. wait for the predecessor (AFTER_OUT, REQUIRED, no default): GO after ZZV8FMINI-DONE, or after a ZZV8FMINI-STOP of the predecessor's OWN once its BOX2_BUSY is released (a predecessor STOP must not idle box 2); HALT on a ZZGUARD-STOP
#      (temperature) or a BOX2_BUSY that stays held. Take BOX2_BUSY atomically, refresh box 2's code from the REMOTE tip with a SMALL export (repo_v8fflatdag_tip.tar, <= 300 MB asserted; no existing tar is overwritten or deleted),
#      F1: the tip's trainer must name o_interface in INIT_CKPT_MATCH (`init-match`), else ZZFLATDAG-F1-STOP and nothing is touched; sha-check the key files, md5-check the v9 releases; assert box 2's FLAT_s<S> checkpoint (sha256, the dev
#      box's copy too) and that its argv.json / summary.json equal the repo's blobs (sha256 pinned; else the parsed JSON must be equal)
#   2. BANK s0, BANK s1 (`bankflat`: four clip-sharded generators of FLAT_s<S> + merge, into $R/dagger_bank_flat/s<S>, never the M2b dagger_bank); the SMALL files are fetched first; box 2's bank.json sha256 == the pulled file's
#   3. COVERAGE + bank controls (`v8f_oaf_chain_checks.py bank-judge`, the DAG judge: K-B2 reads FLAT's checkpoint sha256): a failed CONTROL = REFUSED; with every control passing, the eligible fraction of TRAIN windows >= 0.5 at BOTH seeds,
#      else BLOCKED and the chain STOPS here: no gate is queued and nothing is trained (ZZFLATDAG-BLOCKED + ZZV8FMINI-STOP BLOCKED)
#   3b. PUBLISH THE BANKS to the dev box (`publish_bank_one`, only after the judge passed), per seed, into $ART/bank_s<S>/ (11 files + bank_stage.log), SHA256SUMS.txt written LAST; every pull is behind the 1 GiB floor on E:
#   4. ALL FOUR GATES FIRST (`gatedag flatdag|flatdag0`), each pulled and asserted HERE (gate_oaf_assert.py: the CUT (levers.o_interface cut, dead e_code, O invariant to S and T) AND the swap reached, init = FLAT, the bank = FLAT's, FLAT's every
#      other flag), then the LAUNCH-GATE TOKEN (never by hand), in the order flatdag s0, flatdag0 s0, flatdag s1, flatdag0 s1
#   5. the four runs, in that order: `traindag` (3,000 steps from FLAT_s<S>) -> `evaldag` (the standard T0 eval AND the GT-code oracle, tag <tag>-tacorc) -> pull to $ART/<run>/ -> the run record is asserted (argv.json == the pinned argv ==
#      the planned one, init = FLAT, the cut built, the probe passed) -> the REACHED control (F2: the gate does not cover the loop's dag.apply): flatdag dagger_n / dagger_eligible in 0.1 +- 0.02, flatdag0 exactly 0, else
#      ZZFLATDAG-CONTROL-FAIL and the chain stops
#   NOT HERE: the T1-kin rollouts and the reader: dev-box CPU.
#   chain_v8f_flatdag.sh --dry-run                PRINTS every stage line and the checks, touches nothing (the bank sha256 is a placeholder unless FLATDAG_DRY_SHA0 / FLATDAG_DRY_SHA1 are set)
#   chain_v8f_flatdag.sh --check-tree ROOT        OFFLINE: the tip-carries-the-lever greps AND the F1 check applied to a directory
#   chain_v8f_flatdag.sh --judge-only D0 D1       OFFLINE: the COVERAGE / controls judge on two local dirs of PULLED bank files; prints exactly the markers the real chain would
#   chain_v8f_flatdag.sh --publish-only FAKE      OFFLINE: fetch -> judge -> publish on a local stand-in for box 2 (FLATDAG_ART must be a scratch directory)
#   chain_v8f_flatdag.sh --wait-only              OFFLINE: only the predecessor wait: prints ZZOAF-WAIT-GO or the STOP marker
#   AFTER_OUT=<log> chain_v8f_flatdag.sh          the real run
# Markers in ITS OWN log (chain_v8f_flatdag.out): ZZV8FMINI-DONE / ZZV8FMINI-STOP (v8f_guard_flatdag.sh's anchored end test knows them), ZZFLATDAG-BLOCKED, ZZFLATDAG-REFUSED, ZZFLATDAG-F1-STOP, ZZFLATDAG-CONTROL-FAIL, ZZDISK-FLOOR-STOP.
# TIMEOUTS (polls of 30 s): bank 1,200 (10 h; ESTIMATED 1.1-1.5 h), gate 120, train 1,320 (11 h, one above the guard's 10 h cap), eval 600.
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
R=D:/tanitad/runs/v8f_mini_full
CH=flatdag
CKPT_SHA=(8496934a2ce262c128a3e975da156de2f40e422925893c9979169a4c114493cc 6fdfc9b02ce87248c5315dd7604d3642a3b39f7ae0f593ec6b6ef5ef8df64c95)      # FLAT_s0 / FLAT_s1 ckpt_last.pt (v8f-mini-flat/M2b_s<S>_flat.pt: the base, the bank generator and the --init-ckpt of all four arms)
ARGV_SHA=(630d16a89a626b2e17453d45dcee1253e6b492eac43b6bf2420ffe9271b5829f 8d894b6bd27cfb93061a8a81eac90e073b78a0cc12853df196b0c81365a1ba4e)      # FLAT_s0 / FLAT_s1 argv.json (the repo blobs raw/2026-10-26-flat/runs/M2b_s<S>_flat/)
SUMM_SHA=(3d35a7cca57fdfc94a609a77b059c426bc006d826fd6b292832c3b318ec8337c bf1c632a46d08e7e4ed6476ac963889d9f636d7cb8920558700e38ced3f33d3b)      # FLAT_s0 / FLAT_s1 summary.json (the repo blobs)
RUNS=("flatdag 0" "flatdag0 0" "flatdag 1" "flatdag0 1")      # PREREG_V8F_FLATDAG's order (bank s0, bank s1, then FLATDAG s0, FLATDAG-0 s0, FLATDAG s1, FLATDAG-0 s1); ARM SEED
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=${FLATDAG_ART:-/e/Projects/TanitAD-artifacts/v8f-flatdag}
FLAT_DIR=${FLATDAG_FLAT_DIR:-/e/Projects/TanitAD-artifacts/v8f-mini-flat}      # the dev box's copies of FLAT's checkpoints (M2b_s<S>_flat.pt): read-only here
FLATRUN="$PK/raw/2026-10-26-flat/runs"      # FLAT's argv.json / summary.json in the repo (their sha256 are pinned above)
PVAL=0.1      # FLATDAG's P, fixed by PREREG_V8F_FLATDAG (the queue refuses any other)
B=${OAF_SHIP_DIR:-/c/Users/Admin/box2_ship}; L=$B/logs; E=$L/v8f_flatdag
HERE_SHIP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)      # the helper scripts sit beside this file (box2_ship when installed)
CHECKS=$HERE_SHIP/v8f_oaf_chain_checks.py; ASSERT=$HERE_SHIP/gate_oaf_assert.py
DAG_STEPS=3000
GUARD_OUT=$L/v8fflatdag_guard.out
FLOOR_BYTES=1073741824      # a pull must leave >= 1 GiB free on E:
BUSY_TAKEN=0
rn() { echo "M2b_s${2}_${1}"; }       # rn <arm> <seed>: the run's directory under $R (M2b_s0_flatdag, M2b_s0_flatdag0, ...)
rt() { echo "M2b-${1}_s${2}"; }       # rt <arm> <seed>: the run's eval tag
rg() { echo "gate_M2b-${1}_s${2}"; }  # rg <arm> <seed>: the run's gate directory under $R
# the lever-in-the-tip checks: "<file>|<grep -E pattern>". `.` = a backslash in the queue patterns. (F1 -- INIT_CKPT_MATCH names o_interface -- is a SEPARATE check, `init-match`, run by lever_f1.)
LEVER=(
  'stack/scripts/v8f_mini_train.py|add_argument\("--o-interface"'
  'stack/scripts/v8f_mini_train.py|"o_interface": "lever:o_interface"'
  'stack/scripts/v8f_mini_train.py|^def o_interface_probe'
  'stack/scripts/v8f_mini_train.py|^INIT_CKPT_MATCH = '
  'stack/scripts/v8f_mini_train.py|add_argument\("--dagger-p"'
  'stack/scripts/v8f_mini_train.py|add_argument\("--dagger-bank"'
  'stack/scripts/v8f_mini_train.py|add_argument\("--dagger-bank-sha256"'
  'stack/scripts/v8f_mini_train.py|add_argument\("--dagger-seed"'
  'stack/scripts/v8f_mini_train.py|"dagger_p": "dagger:dagger_p"'
  'stack/scripts/v8f_mini_train.py|"dagger_bank": "dagger:dagger_bank"'
  'stack/scripts/v8f_mini_train.py|"dagger_bank_sha256": "dagger:dagger_bank_sha256"'
  'stack/scripts/v8f_mini_train.py|"dagger_seed": "dagger:dagger_seed"'
  'stack/scripts/v8f_mini_train.py|bad \+= bad_dagger\(a, dag, model\.cfg, data_info\)'
  'stack/scripts/v8f_mini_train.py|^ +dagp = dagger_probe\(model, probe, cfg, dag\)'
  'stack/scripts/v8f_mini_train.py|^ +b = dag\.apply\(b, step\)'
  'stack/scripts/v8f_mini_train.py|"dagger_n": dag_tot\["n"\]'
  'stack/tanitad/data/v8f_dagger.py|^class DaggerSwap'
  'stack/tanitad/data/v8f_dagger.py|NAV_REBUILD_INVALID = False'
  'stack/scripts/v8f_dagger_bank.py|^def cmd_merge'
  'stack/scripts/v8f_dagger_bank.py|window_start=cfg\.window_start'
  'stack/scripts/v8f_dagger_queue_tool.py|^def cmd_bank'
  # the FLAT-DAG queue insertions (the queue file on the tip; `.` = a backslash)
  "$PK/code/v8f_mini_box2_queue.cmd|^:bankflat"
  "$PK/code/v8f_mini_box2_queue.cmd|^:flatdagarm"
  "$PK/code/v8f_mini_box2_queue.cmd|^:evalflatdag"
  "$PK/code/v8f_mini_box2_queue.cmd|goto :bankflat"
  "$PK/code/v8f_mini_box2_queue.cmd|goto :flatdagarm"
  "$PK/code/v8f_mini_box2_queue.cmd|goto :evalflatdag"
  "$PK/code/v8f_mini_box2_queue.cmd|DAGS=flatdag0"
  "$PK/code/v8f_mini_box2_queue.cmd|dagger_bank_flat"
  "$PK/code/v8f_mini_box2_queue.cmd|DAGGEN=%RUNSM%.M2b_s%~2_flat"
  "$PK/code/v8f_mini_box2_queue.cmd|init-ckpt .*M2b_s%~2_flat.ckpt_last.pt"
  "$PK/code/v8f_mini_box2_queue.cmd|--ego-perturb-p 0 --o-interface cut"
  "$PK/code/v8f_mini_box2_queue.cmd|bankflat SEED"
)
STAGES=(bankflat gatedag traindag evaldag dagname dagarm prepdag flatdagarm evalflatdag)
lever_check() {   # lever_check <git tip | tree ROOT>: 0 iff every LEVER pattern is in its file
  local mode=$1 ref=$2 rec f pat miss=0
  for rec in "${LEVER[@]}"; do
    f=${rec%%|*}; pat=${rec#*|}
    if [ "$mode" = tree ]; then grep -qE -- "$pat" "$ref/$f" 2>/dev/null; else git -C $MIRROR show "$ref:$f" 2>/dev/null | grep -qE -- "$pat"; fi \
      || { echo "MISSING in $f: $pat"; miss=$((miss+1)); }
  done
  [ $miss = 0 ]
}
lever_f1() {   # lever_f1 <git tip | tree ROOT>: F1 -- the trainer's INIT_CKPT_MATCH names o_interface (a fine-tune would otherwise START from a checkpoint of the other interface); 0 iff it does
  local mode=$1 ref=$2 tf=${TMPDIR:-/tmp}/v8fflatdag_trainer_$$.py out rc
  if [ "$mode" = tree ]; then cp "$ref/stack/scripts/v8f_mini_train.py" "$tf" 2>/dev/null; else git -C $MIRROR show "$ref:stack/scripts/v8f_mini_train.py" > "$tf" 2>/dev/null; fi
  out=$($PY "$(cygpath -m $CHECKS)" init-match "$(cygpath -m "$tf")"); rc=$?
  rm -f "$tf"
  echo "F1 (INIT_CKPT_MATCH names o_interface): $out"
  return $rc
}
source "$HERE_SHIP/oaf_lib.sh"
plan() {   # plan <queue file> <arm> <seed> <bank sha or "">: GL / TL / EL = the lines the queue prints for the run's gate / train / eval stages (EL: the standard eval AND the oracle eval). rc 1 if a stage prints no trainer / eval line
  local parg=""; [ "$2" = flatdag ] && parg=$PVAL
  GL=$(dry_stage "$1" "$4" "" gatedag $2 $3 $parg | grep -a 'v8f_mini_train.py'); TL=$(dry_stage "$1" "$4" "" traindag $2 $3 $parg | grep -a 'v8f_mini_train.py')
  EL=$(dry_stage "$1" "" "" evaldag $2 $3 | grep -a 'v8f_mini_eval.py')
  [ -n "$GL" ] && [ -n "$TL" ] && [ -n "$EL" ]
}
judge_banks() {   # judge_banks <dir s0> <dir s1>: controls first (REFUSED), then COVERAGE (BLOCKED); prints the judge line and the markers; returns 0 / 3 / 2
  local o rc
  o=$($PY "$(cygpath -m $CHECKS)" bank-judge --banked-sha0 ${CKPT_SHA[0]} --banked-sha1 ${CKPT_SHA[1]} --seed0-dir "$(cygpath -m "$1")" --seed1-dir "$(cygpath -m "$2")"); rc=$?
  echo "$o" | cut -c1-1800
  case $rc in
    0) echo "ZZFLATDAG-JUDGE-PASS every bank control passed and the eligible fraction of TRAIN windows is >= 0.5 at both seeds: the four gates are queued next" ;;
    3) echo "ZZFLATDAG-BLOCKED outcome BLOCKED (PREREG_V8F_FLATDAG outcome table): COVERAGE < 0.5 at seed(s) $(echo "$o" | sed -n 's/.*"blocked_seeds": \(\[[^]]*\]\).*/\1/p'): no gate was queued, nothing was trained" ;;
    *) echo "ZZFLATDAG-REFUSED a bank control failed: $(echo "$o" | sed -n 's/.*"refused_because": \(\[.*\]\), "seeds".*/\1/p' | cut -c1-600) (REFUSED: fix and re-run)" ;;
  esac
  return $rc
}
bank_fetch_small() {   # bank_fetch_small <seed>: the SMALL files of the seed's bank (all the judge reads), pulled to $ART/bank_s<seed> behind the floor; box 2's bank.json sha256 must equal the pulled file's. A SHA256SUMS.txt of an earlier attempt is removed FIRST.
  local s=$1 d="$ART/bank_s$1" k f rs ls
  rm -f "$d/SHA256SUMS.txt"
  for f in bank.json PROVENANCE.json bank_stage.log; do pull "dagger_bank_flat/s$s/$f" "$d"; done
  for k in 0 1 2 3; do pull "dagger_bank_flat/s$s/shard_00${k}of004.provenance.json" "$d"; done
  rs=$(rsha "$R/dagger_bank_flat/s$s/bank.json" SHA256); ls=$(sha256sum "$d/bank.json" | cut -c1-64)
  [ -n "$rs" ] && [ "$rs" = "$ls" ] || stop "bank s$s: box 2's bank.json sha256 '$rs' != the pulled file's '$ls'"
  BSHA[$s]=$ls
  echo "$(date '+%m-%d %H:%M') bank s$s: the small files (bank.json, PROVENANCE.json, bank_stage.log, 4 shard provenance) fetched to $d (bank.json sha256 $ls); the shard npz follow only if the controls pass"
}
publish_bank_one() {   # publish_bank_one <seed>: AFTER the judge passed. The shard npz of the seed's bank (each pull behind the floor), then SHA256SUMS.txt over ALL the published files, written LAST
  local s=$1 d="$ART/bank_s$1" f rs ls names all v
  names=$($PY "$(cygpath -m $CHECKS)" bank-files "$(cygpath -m "$d")" --kind shard_npz) || stop "bank s$s: cannot name the shard npz from the pulled bank.json / PROVENANCE.json"
  names=${names//$'\r'/}                                  # belt and braces: a CR left in a file name would make every pull below fail on a name that does not exist
  for f in $names; do
    pull "dagger_bank_flat/s$s/$f" "$d"
    ls=$(sha256sum "$d/$f" | cut -c1-64); rs=$(rsha "$R/dagger_bank_flat/s$s/$f" SHA256)
    [ -n "$rs" ] && [ "$rs" = "$ls" ] || stop "bank s$s: box 2's $f sha256 '$rs' != the pulled file's '$ls'"
  done
  v=$($PY "$(cygpath -m $CHECKS)" bank-verify "$(cygpath -m "$d")") || stop "bank s$s: the pulled files do not verify, nothing was sealed: $(echo "$v" | cut -c1-600)"
  all=$($PY "$(cygpath -m $CHECKS)" bank-files "$(cygpath -m "$d")" --kind all) || stop "bank s$s: cannot name the files to seal"
  all=${all//$'\r'/}
  ( cd "$d" && sha256sum $all | tee SHA256SUMS.txt )
  v=$($PY "$(cygpath -m $CHECKS)" bank-verify "$(cygpath -m "$d")" --require-sums) || { rm -f "$d/SHA256SUMS.txt"; stop "bank s$s: SHA256SUMS.txt does not verify (removed): $(echo "$v" | cut -c1-600)"; }
  echo "$(date '+%m-%d %H:%M') bank s$s pulled to $d ($(echo "$v" | sed -n 's/.*"n_files": \([0-9]*\).*/\1/p') files: PROVENANCE.json, bank.json, shard provenance, shard npz, bank_stage.log; sha256 in SHA256SUMS.txt; bank.json sha256 ${BSHA[$s]})"
}
if [ "${1:-}" = "--check-tree" ]; then
  [ -d "${2:-}" ] || { echo "usage: --check-tree ROOT"; exit 2; }
  MIRROR=none; ok=0
  lever_check tree "$2" || ok=1
  lever_f1 tree "$2" || { echo "ZZFLATDAG-F1-STOP the trainer's INIT_CKPT_MATCH does not name o_interface: a FLATDAG fine-tune could start from a checkpoint of the other interface (land the F1 fix first)"; ok=1; }
  if [ $ok = 0 ]; then echo "lever check: every one of ${#LEVER[@]} patterns is in $2, and F1 holds"; exit 0; fi; echo "lever check: FAILED"; exit 1
fi
if [ "${1:-}" = "--judge-only" ]; then
  [ -d "${2:-}" ] && [ -d "${3:-}" ] || { echo "usage: --judge-only DIR_S0 DIR_S1"; exit 2; }
  judge_banks "$2" "$3"; jr=$?
  [ $jr = 0 ] || echo "ZZV8FMINI-STOP $([ $jr = 3 ] && echo BLOCKED || echo REFUSED): no gate was queued"
  exit $jr
fi
if [ "${1:-}" = "--publish-only" ]; then   # OFFLINE: the real bank_fetch_small -> judge -> publish_bank_one on a LOCAL stand-in for box 2 (FAKE/dagger_bank_flat/s0 and s1); pull/rsha/stop are the only things replaced. Refuses the real artifact directory.
  FAKE=${2:-}
  [ -d "$FAKE/dagger_bank_flat/s0" ] && [ -d "$FAKE/dagger_bank_flat/s1" ] || { echo "usage: FLATDAG_ART=<scratch dir> $0 --publish-only FAKE_BOX_ROOT   (FAKE_BOX_ROOT/dagger_bank_flat/s0 and s1 stand for box 2's $R/dagger_bank_flat/s<S>)"; exit 2; }
  [ -n "${FLATDAG_ART:-}" ] || { echo "ZZV8FMINI-STOP --publish-only needs FLATDAG_ART set to a scratch directory (it never writes the real $ART)"; exit 1; }
  HASHROOT=${FAKE_HASH_ROOT:-$FAKE}                   # box 2's own hash is measured here (a test points it at a tampered copy to show a transfer that corrupts is caught)
  stop() { echo "ZZV8FMINI-STOP $*"; exit 1; }
  pull() { mkdir -p "$2"; cp "$FAKE/$1" "$2/" || stop "pull $R/$1"; }
  rsha() { sha256sum "$HASHROOT/${1#$R/}" 2>/dev/null | cut -c1-64; }
  declare -a BSHA
  for s in 0 1; do bank_fetch_small $s; done
  judge_banks "$ART/bank_s0" "$ART/bank_s1"; jr=$?
  [ $jr = 0 ] || stop "$([ $jr = 3 ] && echo BLOCKED || echo REFUSED) at the bank stage (judge rc $jr): no shard npz was pulled, no gate was queued"
  for s in 0 1; do publish_bank_one $s; done
  echo "ZZFLATDAG-PUBLISH-ONLY-DONE both banks published"
  exit 0
fi
if [ "${1:-}" = "--wait-only" ]; then
  AFTER_OUT=${AFTER_OUT:-}; [ -n "$AFTER_OUT" ] || { echo "ZZV8FMINI-STOP AFTER_OUT is not set"; exit 1; }
  wait_predecessor; echo "ZZOAF-WAIT-GO"; exit 0
fi
if [ "${1:-}" = "--dry-run" ]; then
  QF=${V8F_QUEUE:-}
  MIRROR=${OAF_MIRROR:-/c/Users/Admin/tanitad-push}; BR=${OAF_BR:-agent/arch-inf-20260803}
  if [ -z "$QF" ] || [ -z "${FLATDAG_REF_DIR:-}" ]; then T=$(git -C $MIRROR ls-remote origin $BR | cut -f1); fi
  if [ -z "$QF" ]; then
    QF=${TMPDIR:-/tmp}/v8fflatdag_queue_dry_$T.cmd
    git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > "$QF" || { echo "ZZV8FMINI-STOP no queue at the tip"; exit 1; }
  fi
  echo "== predecessor"; echo "wait for ${AFTER_OUT:-<UNSET: a real run REFUSES to start without AFTER_OUT>}: GO after ^ZZV8FMINI-DONE or a predecessor ^ZZV8FMINI-STOP of its own with BOX2_BUSY released; HALT on a ZZGUARD-STOP or a held BOX2_BUSY"
  echo "== pinned (asserted on box 2 before the banks and after the last run)"
  for s in 0 1; do echo "FLAT_s$s $R/M2b_s${s}_flat/ckpt_last.pt ${CKPT_SHA[$s]} (argv.json ${ARGV_SHA[$s]}, summary.json ${SUMM_SHA[$s]})"; done
  echo "== FLAT's banked argv.json (the repo blob; the planned lines must be FLAT's every other flag)"
  for s in 0 1; do
    mkdir -p ${TMPDIR:-/tmp}/v8fflatdag_ref_$$/s$s
    if [ -n "${FLATDAG_REF_DIR:-}" ]; then cp "$FLATDAG_REF_DIR/s$s/argv.json" ${TMPDIR:-/tmp}/v8fflatdag_ref_$$/s$s/argv.json; else git -C $MIRROR show "$T:$FLATRUN/M2b_s${s}_flat/argv.json" > ${TMPDIR:-/tmp}/v8fflatdag_ref_$$/s$s/argv.json || { echo "ZZV8FMINI-STOP the tip has no FLAT_s$s argv.json"; exit 1; }; fi
    rsum=$(sha256sum ${TMPDIR:-/tmp}/v8fflatdag_ref_$$/s$s/argv.json | cut -c1-64)
    [ "$rsum" = "${ARGV_SHA[$s]}" ] || { echo "ZZV8FMINI-STOP FLAT_s$s's argv.json (the reference) sha256 $rsum != the pinned ${ARGV_SHA[$s]}"; exit 1; }
    echo "FLAT_s$s argv.json sha256 == the pinned value"
  done
  echo "== launch-gate token (written by the chain, one per run, after the gate's assertions pass; never by hand)"
  echo '{"kind": "launch-gate-token", "verdict": "PASS", "prereg": "PREREG_V8F_FLATDAG", "binding": {"commit": "<the remote tip, 40 hex>", "argv_sha256": "<launch_gate.argv_sha256 of THE TRAINING ARGV as pinned by box 2>", "data_sha256": "<the gate selfcheck'"'"'s info.train_sha256>"}, ...}'
  for s in 0 1; do
    echo "== bank s$s"; echo "-- bankflat $s"; dry_stage "$QF" "" "" bankflat $s | grep -a 'v8f_dagger_queue_tool.py' || { echo "ZZV8FMINI-STOP the queue prints no bank line for s$s"; exit 1; }
    echo "-- fetch (all the judge reads), then judge, then PUBLISH into $ART/bank_s$s/ only if the judge passes:"
    $PY "$(cygpath -m $CHECKS)" bank-plan | sed 's/^/   /' || { echo "ZZV8FMINI-STOP bank-plan failed"; exit 1; }
  done
  for run in "${RUNS[@]}"; do
    set -- $run; arm=$1; s=$2; sh=FLATDAG_DRY_SHA$s; sha=${!sh:-}
    plan "$QF" $arm $s "$sha" || { echo "ZZV8FMINI-STOP the queue prints no gate / train / eval line for $(rn $arm $s)"; exit 1; }
    echo "== $(rn $arm $s): $([ $arm = flatdag ] && echo "--dagger-p $PVAL" || echo "--dagger-p 0") --o-interface cut init $R/M2b_s${s}_flat/ckpt_last.pt (${CKPT_SHA[$s]}), tag $(rt $arm $s)${sha:+, bank sha256 $sha}"
    echo "-- gatedag $arm $s"; echo "$GL"
    echo "-- traindag $arm $s"; echo "$TL"
    echo "-- evaldag $arm $s"; echo "$EL"
    printf '%s' "$TL" > ${TMPDIR:-/tmp}/v8fflatdag_tl_$$.txt; printf '%s' "$GL" > ${TMPDIR:-/tmp}/v8fflatdag_gl_$$.txt
    echo "-- token argv_sha256 (the training argv): $(argvsha ${TMPDIR:-/tmp}/v8fflatdag_tl_$$.txt)"
    same=$($PY "$(cygpath -m $CHECKS)" argv-same "$(cygpath -m ${TMPDIR:-/tmp}/v8fflatdag_gl_$$.txt)" "$(cygpath -m ${TMPDIR:-/tmp}/v8fflatdag_tl_$$.txt)"); echo "-- gate argv = training argv with --out and --selfcheck-only changed: $same"
    # FLAT's every other flag: the planned training argv minus the fine-tune / bank flags IS FLAT's banked argv.json minus the same flags, and --o-interface cut is NAMED exactly once
    fc=$($PY - "$(cygpath -m ${TMPDIR:-/tmp}/v8fflatdag_ref_$$/s$s/argv.json)" "$(cygpath -m ${TMPDIR:-/tmp}/v8fflatdag_tl_$$.txt)" $arm $s <<'EOF'
import json, re, sys
ref = json.load(open(sys.argv[1], encoding="utf-8"))["argv"]
t = [x.strip('"') for x in re.findall(r'"[^"]*"|\S+', open(sys.argv[2], encoding="utf-8").read().strip())]
mine = t[next(k for k, x in enumerate(t) if x.lower().endswith("v8f_mini_train.py")) + 1:]
arm, seed = sys.argv[3], sys.argv[4]
FT = {"--out", "--init-ckpt", "--steps", "--lr", "--warmup", "--p-gt-start", "--p-gt-end", "--ego-perturb-p", "--dagger-p", "--dagger-bank", "--dagger-bank-sha256", "--dagger-seed"}
def flags(a):
    o, k = {}, 0
    while k < len(a):
        if a[k].startswith("--"):
            j = k + 1
            while j < len(a) and not a[j].startswith("--"): j += 1
            o.setdefault(a[k], []).append(a[k + 1:j]); k = j
        else: k += 1
    return o
def drop(a, names):
    o, k = [], 0
    while k < len(a):
        if a[k] in names:
            k += 1
            while k < len(a) and not a[k].startswith("--"): k += 1
        else: o.append(a[k]); k += 1
    return o
fm = flags(mine)
ok = flags(drop(mine, FT)) == flags(drop(ref, FT)) and fm.get("--o-interface") == [["cut"]] and fm.get("--init-ckpt", [[""]])[0][0].endswith("M2b_s%s_flat\\ckpt_last.pt" % seed)
ok = ok and (fm.get("--dagger-p") == [["0.1"]] and "--dagger-bank" in fm if arm == "flatdag" else fm.get("--dagger-p") == [["0"]] and not ({"--dagger-bank", "--dagger-bank-sha256", "--dagger-seed"} & set(fm)))
print("FLAT-CONSISTENT" if ok else "NOT-FLAT-CONSISTENT")
EOF
)
    echo "-- planned line == FLAT's banked argv.json minus the fine-tune / bank flags, --o-interface cut named, init FLAT_s$s: $fc"
    rm -f ${TMPDIR:-/tmp}/v8fflatdag_tl_$$.txt ${TMPDIR:-/tmp}/v8fflatdag_gl_$$.txt
    [ "$same" = SAME ] && [ "$fc" = FLAT-CONSISTENT ] || { echo "ZZV8FMINI-STOP the planned lines are not the FLATDAG argv ($(rn $arm $s))"; exit 1; }
  done
  rm -rf ${TMPDIR:-/tmp}/v8fflatdag_ref_$$
  echo "== not run here: the T1-kin rollouts (dev box CPU) and the reader"
  exit 0
fi
# ======================================================================== the real run
if [ -z "${AFTER_OUT:-}" ]; then
  echo "ZZV8FMINI-STOP AFTER_OUT is not set: this chain has no default predecessor log (a default can already be satisfied by an OLD log). Set AFTER_OUT to the log of the chain that holds box 2 now."
  exit 1
fi
[ -d "$(dirname "$ART")" ] || { echo "ZZV8FMINI-STOP $(dirname "$ART") does not exist: the artifacts have nowhere to go"; exit 1; }
MIRROR=${OAF_MIRROR:-/c/Users/Admin/tanitad-push}; BR=${OAF_BR:-agent/arch-inf-20260803}
HJ="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
# EXPORT: what box 2's queue READS from the repo (export_closure.py: the import closure of the entry scripts = 56 python files, plus the 7 repo data paths those programs open by default).
# Generated by make_chains.py from export_closure_flatdag.json, never typed. The tar has a NEW name; no existing tar is overwritten or deleted.
EXPORT=(
  'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/raw/p10_yaw4512/p10_eval_pred6.npy'
  'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/raw/p10_yaw4512/p10_prior.pt'
  'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/v8f_mini_box2_queue.cmd'
  'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-26-dagger-draft/probe/THRESHOLDS_FIXED.txt'
  'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval141_kin.npz'
  'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval_grid_windows.json'
  'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz'
  'stack/scripts/v8f_dagger_bank.py'
  'stack/scripts/v8f_dagger_queue_tool.py'
  'stack/scripts/v8f_mini_eval.py'
  'stack/scripts/v8f_mini_train.py'
  'stack/scripts/v8f_t1_eval.py'
  'stack/tanitad/__init__.py'
  'stack/tanitad/channel_admissibility.py'
  'stack/tanitad/config.py'
  'stack/tanitad/data/__init__.py'
  'stack/tanitad/data/_contract.py'
  'stack/tanitad/data/calib.py'
  'stack/tanitad/data/comma2k19.py'
  'stack/tanitad/data/metadrive_env.py'
  'stack/tanitad/data/stats.py'
  'stack/tanitad/data/toy_driving.py'
  'stack/tanitad/data/v8f_dagger.py'
  'stack/tanitad/data/v8f_mini_data.py'
  'stack/tanitad/data/v9_labels.py'
  'stack/tanitad/models/__init__.py'
  'stack/tanitad/models/_validate.py'
  'stack/tanitad/models/ego_history.py'
  'stack/tanitad/models/encoder.py'
  'stack/tanitad/models/fourbrain.py'
  'stack/tanitad/models/hjepa.py'
  'stack/tanitad/models/imagination.py'
  'stack/tanitad/models/inverse_dynamics.py'
  'stack/tanitad/models/kinematic.py'
  'stack/tanitad/models/kinematic_prior.py'
  'stack/tanitad/models/metric_dynamics.py'
  'stack/tanitad/models/p10_prior.py'
  'stack/tanitad/models/predictor.py'
  'stack/tanitad/models/readout.py'
  'stack/tanitad/models/sigreg.py'
  'stack/tanitad/models/v8f_beta.py'
  'stack/tanitad/models/v8f_lora_encoder.py'
  'stack/tanitad/models/v8f_mini.py'
  'stack/tanitad/refs/__init__.py'
  'stack/tanitad/refs/anchor_meta.py'
  'stack/tanitad/refs/refa.py'
  'stack/tanitad/refs/refb.py'
  'stack/tanitad/refs/refc_sampler.py'
  'stack/tanitad/refs/refc_tactical.py'
  'stack/tanitad/refs/refcv6_max_speed.py'
  'stack/tanitad/train/__init__.py'
  'stack/tanitad/train/config_hygiene.py'
  'taniteval/taniteval/__init__.py'
  'taniteval/taniteval/ci.py'
  'taniteval/taniteval/floors_kin.py'
  'taniteval/taniteval/four_families.py'
  'taniteval/taniteval/lead_metrics.py'
  'taniteval/taniteval/lead_source.py'
  'taniteval/taniteval/progress.py'
  'taniteval/taniteval/strategic_optionset.py'
  'taniteval/taniteval/v0_antiecho.py'
  'taniteval/taniteval/v8f_beta_eval.py'
  'taniteval/taniteval/v8f_consistency.py'
)
EXPORT_MAX_BYTES=314572800
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
MIN_FREE_GB=8
BANK_POLLS=1200; GATE_POLLS=120; TRAIN_POLLS=1320; EVAL_POLLS=600
TAG=$(date '+%m%d%H%M')
CUR=$L/v8fflatdag_current.txt
wait_predecessor
[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present: $(cat $B/BOX2_BUSY.txt)"; exit 1; }
# the marker is taken ATOMICALLY (noclobber): two chains that wait on the same predecessor cannot both pass the test above and both write it
( set -C; echo "v8F-mini PREREG_V8F_FLATDAG (bank s0, bank s1 of FLAT, COVERAGE gate, then FLATDAG / FLATDAG-0 x seeds 0 and 1: four gates first, 3,000-step fine-tunes from FLAT on the cut, standard T0 evals + the GT-code oracle) since $(date '+%m-%d %H:%M'). Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt ) 2>/dev/null \
  || { echo "ZZV8FMINI-STOP busy marker taken while arming: $(cat $B/BOX2_BUSY.txt 2>/dev/null)"; exit 1; }
BUSY_TAKEN=1
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
mkdir -p $E/runs $E/lines      # only now: a chain that refused or stopped while waiting has touched nothing
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
QT=$B/v8fflatdag_queue_$TAG.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" | tr -d '\r' > $QT
QC=$B/v8fflatdag_queue_$TAG.crlf.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > $QC      # the committed bytes (CRLF): what cmd.exe can run for the dry-run lines
for st in "${STAGES[@]}"; do grep -q "^:$st\$\|^if /i \"%~1\"==\"$st\" goto" $QT || stop "the tip's queue has no $st stage"; done
lever_check git "$T" || stop "the tip does not carry the DAgger / cut levers and the FLATDAG queue stages (the greps above): land them first"
lever_f1 git "$T" || { echo "ZZFLATDAG-F1-STOP the tip's INIT_CKPT_MATCH does not name o_interface: a FLATDAG fine-tune could start from a checkpoint of the other interface (the F1 fix is not landed): nothing was touched on box 2"; stop "F1 not landed"; }
for xf in "${EXPORT[@]}"; do git -C $MIRROR cat-file -e "$T:$xf" || stop "the tip has no $xf (the export list names only files box 2 reads)"; done
git -C $MIRROR cat-file -e "$T:$PK/raw/eval_grid_windows.json" || stop "the tip has no eval-141 grid (the bank's K-B3 needs it)"
for s in 0 1; do for f in argv.json summary.json; do git -C $MIRROR cat-file -e "$T:$FLATRUN/M2b_s${s}_flat/$f" || stop "the tip has no FLAT_s$s's $f (the pinned reference)"; done; done
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8fflatdag_tip.tar $T -- "${EXPORT[@]}" || stop "archive"
tsz=$(stat -c %s $B/repo_v8fflatdag_tip.tar); [ "$tsz" -gt 0 ] && [ "$tsz" -le $EXPORT_MAX_BYTES ] || stop "the code archive is $tsz bytes (must be 1..$EXPORT_MAX_BYTES): C: has little room, export only what box 2 reads"
echo "$(date '+%m-%d %H:%M') code archive $B/repo_v8fflatdag_tip.tar: $tsz bytes, ${#EXPORT[@]} paths"
$SCP $B/repo_v8fflatdag_tip.tar tanitad-box2:D:/tanitad/repo_v8fflatdag_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_v8fflatdag_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_V8FFLATDAG.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8fflatdag${TAG}_refresh.cmd
$SCP $B/v8fflatdag${TAG}_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8fflatdag${TAG}_refresh || stop "launch refresh"
waitrc v8fflatdag${TAG}_refresh 40
KEYSHA=""
for f in stack/scripts/v8f_mini_train.py stack/scripts/v8f_mini_eval.py stack/scripts/v8f_dagger_bank.py stack/scripts/v8f_dagger_queue_tool.py stack/scripts/v8f_t1_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py \
         stack/tanitad/data/v8f_dagger.py stack/tanitad/models/hjepa.py "$PK/code/v8f_mini_box2_queue.cmd" "$PK/raw/eval_grid_windows.json" "$HJ/raw/p10_yaw4512/p10_prior.pt" "$HJ/raw/p10_yaw4512/p10_eval_pred6.npy"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  [ "$l" = "$(rsha "D:/tanitad/repo/$f" SHA256)" ] || stop "box 2 $f != tip $T"
  KEYSHA="$KEYSHA\"$f\": \"$l\", "
done
for kv in v9_labels_train.npz=83e30bd3791f0767707eade5f4671759 v9_labels_eval139.npz=6b5c7f207cffc3b7eb3cd527fd433599; do
  [ "${kv#*=}" = "$(rsha "D:/tanitad/data/v9/${kv%%=*}" MD5)" ] || stop "box 2 v9 ${kv%%=*} md5 != ${kv#*=}"
done
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (13 key files sha-checked; both v9 releases md5-checked)"
CK=$E/ckpt_sha256.json
# ---- 2. the base is fixed (PREREG_V8F_FLATDAG section 1): the bank generators and all four fine-tunes start from the banked FLAT runs (box 2 is idle: a different file costs nothing)
pinned_b() {   # pinned_b <phase>: box 2's FLAT_s<S> checkpoint (sha256) IS the banked run's; its argv.json / summary.json equal the repo's blobs (the pinned sha256, else the parsed JSON)
  local s f g exp r v
  for s in 0 1; do
    g=$(rsha "$R/M2b_s${s}_flat/ckpt_last.pt" SHA256); [ -n "$g" ] || stop "$1: cannot hash box 2's M2b_s${s}_flat/ckpt_last.pt"
    [ "$g" = "${CKPT_SHA[$s]}" ] || stop "$1: box 2's M2b_s${s}_flat/ckpt_last.pt sha256 $g != the banked FLAT run's ${CKPT_SHA[$s]}"
    for f in argv.json summary.json; do
      g=$(rsha "$R/M2b_s${s}_flat/$f" SHA256); [ -n "$g" ] || stop "$1: cannot hash box 2's M2b_s${s}_flat/$f"
      case $f in argv.json) exp=${ARGV_SHA[$s]}; r="$E/ref/s$s/argv.json" ;; *) exp=${SUMM_SHA[$s]}; r="$E/runs/M2b_s$s/summary.json" ;; esac
      if [ "$g" != "$exp" ]; then      # not the repo blob's bytes: box 2 wrote its own copy; the CONTENT must be the same
        rm -rf "$E/box2chk"; pull "M2b_s${s}_flat/$f" "$E/box2chk"
        v=$($PY "$(cygpath -m $CHECKS)" json-same "$(cygpath -m "$E/box2chk/$f")" "$(cygpath -m "$r")") || stop "$1: box 2's M2b_s${s}_flat/$f (sha256 $g) is NOT the repo's blob ($exp) in content either"
        echo "$(date '+%m-%d %H:%M') $1: box 2's M2b_s${s}_flat/$f differs from the repo blob in bytes only ($v)"
      fi
    done
    ckset "M2b_s${s}_flat" "${CKPT_SHA[$s]}"
  done
  echo "$(date '+%m-%d %H:%M') $1: FLAT_s0 and FLAT_s1 (checkpoint sha256; argv.json, summary.json = the repo blobs) are the banked FLAT runs'"
}
rm -f $CK
for s in 0 1; do      # the references: FLAT's argv.json (the every-other-flag comparator) and summary.json (the gate comparator "B") are the repo's blobs at the tip, checked against the pinned sha256; the dev box's checkpoint copy too
  mkdir -p $E/runs/M2b_s$s $E/ref/s$s
  git -C $MIRROR show "$T:$FLATRUN/M2b_s${s}_flat/argv.json" > $E/ref/s$s/argv.json || stop "extract FLAT_s$s's argv.json"
  git -C $MIRROR show "$T:$FLATRUN/M2b_s${s}_flat/summary.json" > $E/runs/M2b_s$s/summary.json || stop "extract FLAT_s$s's summary.json"
  [ "$(sha256sum $E/ref/s$s/argv.json | cut -c1-64)" = "${ARGV_SHA[$s]}" ] || stop "FLAT_s$s's argv.json at the tip is not the pinned blob (${ARGV_SHA[$s]})"
  [ "$(sha256sum $E/runs/M2b_s$s/summary.json | cut -c1-64)" = "${SUMM_SHA[$s]}" ] || stop "FLAT_s$s's summary.json at the tip is not the pinned blob (${SUMM_SHA[$s]})"
  [ -f "$FLAT_DIR/M2b_s${s}_flat.pt" ] || stop "the dev box's copy of FLAT_s$s ($FLAT_DIR/M2b_s${s}_flat.pt) is missing"
  [ "$(sha256sum "$FLAT_DIR/M2b_s${s}_flat.pt" | cut -c1-64)" = "${CKPT_SHA[$s]}" ] || stop "the dev box's $FLAT_DIR/M2b_s${s}_flat.pt sha256 != the pinned FLAT_s$s checkpoint"
done
pinned_b "before the banks"
fr=$($SSH "powershell -NoProfile -Command \"[int]((Get-PSDrive D).Free / 1GB)\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$' | head -1)
[ -n "$fr" ] && [ "$fr" -ge $MIN_FREE_GB ] || stop "box 2's D: has '$fr' GB free (< $MIN_FREE_GB): two banks (~0.7 GB each, ESTIMATED), four checkpoints and the evals need room"
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep ${OAF_TELEMETRY_S:-45}; t1=$(tl); sleep ${OAF_TELEMETRY_S2:-35}; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
grep -qF "'^(ZZV8FMINI|" $B/v8f_guard_flatdag.sh || stop "v8f_guard_flatdag.sh does not know this chain's markers (anchored): it would never exit"
grep -qF 'v8f_dagger_bank|v8f_dagger_queue_tool' $B/v8f_guard_flatdag.sh || stop "v8f_guard_flatdag.sh would not stop the bank generators on a temperature stop"
echo "" > $CUR
V8F_CHAIN_OUT=$L/chain_v8f_flatdag.out V8F_CURRENT=$CUR bash $B/v8f_guard_flatdag.sh > $GUARD_OUT 2>&1 &
# ---- 3. THE BANKS of FLAT (they depend only on FLAT and the TRAIN cache), then the controls and the COVERAGE gate: BLOCKED / REFUSED stop the chain BEFORE any gate or training
declare -a BSHA
bank_one() {   # bank_one <seed>: the seed's own-state bank on box 2, then its SMALL files; the shard npz are published by publish_bank_one only after the judge passed
  GATE_LOG="$R/dagger_bank_flat/s$1/bank_stage.log"; queue "bank_s$1" $BANK_POLLS bankflat $1; GATE_LOG=
  bank_fetch_small $1
}
for s in 0 1; do bank_one $s; done
judge_banks "$ART/bank_s0" "$ART/bank_s1"; jr=$?
if [ $jr != 0 ]; then
  $PY -c "import json,sys;json.dump({'outcome':'BLOCKED' if sys.argv[1]=='3' else 'REFUSED','bank_sha256':{'s0':sys.argv[2],'s1':sys.argv[3]}},open(sys.argv[4],'w'),indent=1)" $jr "${BSHA[0]}" "${BSHA[1]}" "$(cygpath -m $ART/FLATDAG_OUTCOME.json)"
  stop "$([ $jr = 3 ] && echo BLOCKED || echo REFUSED) at the bank stage (judge rc $jr): no gate was queued, nothing was trained (see $ART/FLATDAG_OUTCOME.json)"
fi
# ---- 3b. only a bank that passed every control and COVERAGE is published to the dev box: the shard npz, then SHA256SUMS.txt over all the files, per seed
for s in 0 1; do publish_bank_one $s; done
# ---- 4. ALL FOUR GATES, each asserted and tokened, before ANY training
gate_one() {   # gate_one <arm> <seed>: the run's in-process launch gate on box 2, pulled and asserted HERE, then its launch-gate token (written by this function, never by hand)
  local arm=$1 s=$2 g gd SC pin xa
  g=$(rn $arm $s); gd=$(rg $arm $s); SC="$ART/$g/gate/selfcheck.json"; pin="$ART/$g/gate/pin.txt"
  plan "$QC" $arm $s "${BSHA[$s]:-}" || stop "the tip's queue prints no gate / train / eval line for $g"
  mkdir -p "$ART/$g/gate" "$E/lines"
  printf '%s\n' "$GL" > "$E/lines/$g.gate.txt"; printf '%s\n' "$TL" > "$E/lines/$g.train.txt"; printf '%s\n' "$EL" > "$E/lines/$g.eval.txt"
  GATE_LOG="$R/$gd.log"; queue "gate_$g" $GATE_POLLS gatedag $arm $s $([ $arm = flatdag ] && echo $PVAL); GATE_LOG=      # a refused gate says WHY in the stage's own log
  pull "$gd/selfcheck.json" "$ART/$g/gate"
  pull_opt "$gd.log" "$ART/$g/gate"
  pull "dagpin/M2b_s${s}_${arm}.train.argv.txt" "$ART/$g/gate"; mv -f "$ART/$g/gate/M2b_s${s}_${arm}.train.argv.txt" "$pin"
  xa=(--selfcheck "$(cygpath -m $SC)" --gate-line "$(cygpath -m $E/lines/$g.gate.txt)" --train-line "$(cygpath -m $E/lines/$g.train.txt)" --pin-line "$(cygpath -m $pin)" --arm $arm --seed $s --banked-init-sha ${CKPT_SHA[$s]}
      --b-summary "$(cygpath -m $E/runs/M2b_s$s/summary.json)" --ref-argv-json "$(cygpath -m $E/ref/s$s/argv.json)")
  [ $arm = flatdag ] && xa+=(--p $PVAL --bank-sha ${BSHA[$s]} --bank-json "$(cygpath -m $ART/bank_s$s/bank.json)")
  $PY "$(cygpath -m $ASSERT)" check "${xa[@]}" || stop "gate FLATDAG assertions $g"
  $PY "$(cygpath -m $ASSERT)" token "${xa[@]}" --commit "$T" --queue-sha "$(sha256sum $QT | cut -c1-64)" --key-sha-json "{${KEYSHA%, }}" --run $g --out "$(cygpath -m $ART/$g/launch_gate_token.json)" || stop "write the gate token $g"
}
ckptsha_x() {   # ckptsha_x <arm> <seed> <trained|after_evals>
  local g h h0; g=$(rn $1 $2)
  h=$(rsha "$R/$g/ckpt_last.pt" SHA256); [ -n "$h" ] || stop "cannot hash box 2's $g checkpoint ($3)"
  if [ "$3" = after_evals ]; then h0=$(ckget "$(rt $1 $2)_trained"); [ "$h" = "$h0" ] || stop "$g's checkpoint changed during the evals: $h != $h0"; fi
  ckset "$(rt $1 $2)_$3" "$h"
  echo "$(date '+%m-%d %H:%M') $g checkpoint sha256 ($3) ${h:0:16}..."
}
runone() {   # runone <arm> <seed>: train, the evals, pull, record and print the checkpoint sha256, assert the run's record and the REACHED control
  local arm=$1 s=$2 g tg f x l p bs
  g=$(rn $arm $s); tg=$(rt $arm $s); p=$([ $arm = flatdag ] && echo $PVAL || echo 0); bs=$([ $arm = flatdag ] && echo "${BSHA[$s]}" || echo none)
  echo "$g" > $CUR
  queue "t_$g" $TRAIN_POLLS traindag $arm $s $([ $arm = flatdag ] && echo $PVAL)
  echo "" > $CUR
  ckptsha_x $arm $s trained
  queue "e_$g" $EVAL_POLLS evaldag $arm $s
  ckptsha_x $arm $s after_evals
  for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull "$g/$f" "$ART/$g"; done
  for x in json npz log; do pull "eval/eval_$tg.$x" "$ART/$g"; done
  pull_opt "eval/eval_${tg}_proj.npz" "$ART/$g"
  pull "eval/eval_${tg}-tacorc.npz" "$ART/$g"; pull_opt "eval/eval_${tg}-tacorc.json" "$ART/$g"; pull_opt "eval/eval_${tg}-tacorc.log" "$ART/$g"
  l=$(sha256sum "$ART/$g/ckpt_last.pt" | cut -c1-64); [ "$l" = "$(ckget "${tg}_after_evals")" ] || stop "$g: the pulled checkpoint's sha256 $l != the value measured on box 2"
  ( cd "$ART/$g" && sha256sum ckpt_last.pt "eval_$tg.json" "eval_$tg.npz" "eval_${tg}-tacorc.npz" selfcheck.json summary.json argv.json train_log.jsonl launch_gate_token.json | tee SHA256SUMS.txt )
  echo "$(date '+%m-%d %H:%M') $g pulled to $ART/$g (checkpoint sha256 $l)"
  $PY - "$(cygpath -m $ART/$g)" "$(cygpath -m $E/lines/$g.train.txt)" "$(cygpath -m $ART/$g/gate/pin.txt)" $arm $s $p $DAG_STEPS ${CKPT_SHA[$s]} "$bs" <<'EOF' || stop "run FLATDAG record $g"
import hashlib, json, os, re, sys
d, tlf, pinf, arm, seed, p, steps, banked, bsha = sys.argv[1:10]
seed, p, steps = int(seed), float(p), int(steps)
J = lambda f: json.load(open(os.path.join(d, f), encoding="utf-8"))
def argv_of(path):
    t = [x.strip('"') for x in re.findall(r'"[^"]*"|\S+', open(path, encoding="utf-8").read().strip())]
    return t[next(k for k, x in enumerate(t) if x.lower().endswith("v8f_mini_train.py")) + 1:]
planned, pinned = argv_of(tlf), argv_of(pinf)
aj, sm, sc = J("argv.json"), J("summary.json"), J("selfcheck.json")
bad = []
if pinned != planned:
    bad.append("the pinned training argv (box 2) is not the planned one")
if aj.get("argv") != planned:
    got = aj.get("argv") or []
    k = next((k for k in range(min(len(got), len(planned))) if got[k] != planned[k]), min(len(got), len(planned)))
    bad.append(f"argv.json is not the planned training argv (first difference at token {k}: {got[k:k + 2]} vs {planned[k:k + 2]}): the token binds an argv that did not run")
sha16 = hashlib.sha256(" ".join(planned).encode()).hexdigest()[:16]
for name, v in (("argv.json sha", aj.get("sha")), ("summary.json argv_sha", sm.get("argv_sha")), ("selfcheck.json argv_sha", sc.get("argv_sha"))):
    if v != sha16: bad.append(f"{name} {v} != the planned argv's {sha16}")
if sm.get("done") is not True: bad.append("summary.json done is not true")
if sm.get("steps") != steps: bad.append(f"summary.json steps {sm.get('steps')} != {steps}")
if (sm.get("arm"), sm.get("seed")) != ("M2b", seed): bad.append(f"summary.json arm / seed {(sm.get('arm'), sm.get('seed'))} != ('M2b', {seed})")
if sc.get("declared_vs_built") != "PASS": bad.append("the run's own selfcheck.json is not a PASS")
lv = sm.get("levers") or {}
if (lv.get("init_ckpt") or {}).get("sha256") != banked: bad.append(f"summary.json levers.init_ckpt.sha256 {(lv.get('init_ckpt') or {}).get('sha256')} != FLAT_s{seed}'s checkpoint")
if lv.get("dagger_p") != p: bad.append(f"summary.json built dagger_p {lv.get('dagger_p')} != {p}")
if arm == "flatdag" and lv.get("dagger_bank_sha256") != bsha: bad.append(f"summary.json built dagger_bank_sha256 {str(lv.get('dagger_bank_sha256'))[:12]} != FLAT's bank {bsha[:12]}")
if arm == "flatdag0" and lv.get("dagger_bank_sha256") is not None: bad.append("FLATDAG-0 built a bank")
if (lv.get("o_interface") or (sc.get("levers") or {}).get("o_interface")) != "cut": bad.append("the cut is not built (levers.o_interface is not 'cut')")
if "--o-interface" not in planned or planned[planned.index("--o-interface") + 1] != "cut": bad.append("the planned argv does not name --o-interface cut")
if (sc.get("o_interface_probe") or {}).get("pass") is not True: bad.append("the run's o_interface_probe did not pass")
print("FLATDAG run record:", "PASS" if not bad else bad, "|", arm, "s%d" % seed, "| argv sha16", sha16, "| steps", sm.get("steps"), "| o_interface", lv.get("o_interface"))
sys.exit(1 if bad else 0)
EOF
  $PY - "$(cygpath -m $ART/$g/summary.json)" $arm $s $p <<'EOF' || { echo "ZZFLATDAG-CONTROL-FAIL $g: the swap did not reach training as registered (F2: the gate does not cover the loop's dag.apply); the reader would REFUSE this run"; stop "reached-control fail $g"; }
import json, sys
sm, arm, seed, p = json.load(open(sys.argv[1], encoding="utf-8")), sys.argv[2], sys.argv[3], float(sys.argv[4])
n, e = sm.get("dagger_n"), sm.get("dagger_eligible")
if arm == "flatdag0":                                          # FLATDAG-0: exactly 0 swapped
    ok = isinstance(n, int) and n == 0
    print("FLATDAG swap-reached:", "PASS" if ok else "FAIL", f"| flatdag0 s{seed} | dagger_n {n} (must be exactly 0) | eligible {e}")
else:                                                          # FLATDAG: dagger_n / dagger_eligible in p +- 0.02 (PREREG section 6)
    ok = isinstance(n, int) and isinstance(e, int) and e > 0 and 0 <= n <= e and p - 0.02 - 1e-12 <= n / e <= p + 0.02 + 1e-12
    print("FLATDAG swap-reached:", "PASS" if ok else "FAIL", f"| flatdag s{seed} | dagger_n {n} / eligible {e} = {n / e if ok else 'n/a'} (must lie in {p} +- 0.02)")
sys.exit(0 if ok else 1)
EOF
  grep ZZGUARD-STOP $GUARD_OUT 2>/dev/null && stop "guard stop"
}
for run in "${RUNS[@]}"; do set -- $run; gate_one $1 $2; done
# ---- 5. the four runs, in the PREREG's order; nothing is conditional on a read
for run in "${RUNS[@]}"; do set -- $run; runone $1 $2; done
pinned_b "after the last run"
echo "$(date '+%m-%d %H:%M') summary: $(for run in "${RUNS[@]}"; do set -- $run; echo -n "$(rn $1 $2)=$(ckget "$(rt $1 $2)_after_evals" | cut -c1-12) "; done)"
finish "PREREG_V8F_FLATDAG: banks s0/s1 of FLAT (COVERAGE >= 0.5; PROVENANCE.json, bank.json, shard provenance and shard npz published), FLATDAG (p $PVAL) / FLATDAG-0 x seeds 0 and 1 trained (3,000 steps each, from FLAT, --o-interface cut), T0-evaluated (standard + the GT-code oracle) and pulled to $ART (sha256 in each SHA256SUMS.txt); the T1-kin rollouts and the reader run on the dev box (NOT here)"
