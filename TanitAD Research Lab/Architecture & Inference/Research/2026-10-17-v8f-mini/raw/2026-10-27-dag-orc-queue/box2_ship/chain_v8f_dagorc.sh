#!/usr/bin/env bash
# PREREG_V8F_DAG_ORC on box 2: O-RECOVER's launch-recovery perturbation on DAG's states. ONE lever on DAG2-0's banked argv; its control is the banked DAG2-0 run (reused, NOT retrained). A replicate (REP) and a dose fallback (B) ride the same queue.
#   A   (dagorc)    = DAG2-0's banked argv + --ego-perturb-p 0.3 and ORC's five law flags (vscale 0.5 1.0, vmax 8.0, adrop 1.5, yawrate-std 2.0, seed = S), init DAG_s<S>, own --out
#   REP (dagorcrep) = DAG2-0's banked argv with --seed 10 (init DAG_s0) / 11 (init DAG_s1), own --out: the replicate floor RF
#   B   (dagorc01)  = A with --ego-perturb-p 0.1 (read only if A reads HARMS; trained in the same queue; DAGORC_SKIP_B=1 drops it)
# This file is GENERATED from chain_v8f_dagorc.sh.in by make_chain.py (every pinned sha256 is COPIED from a banked artifact, never retyped). BUILT, NOT LAUNCHED: it runs only on the PI's go.
# One chain, in PREREG_V8F_DAG_ORC section 10's order, ONE job at a time on box 2 (mk_v8f_job.py, SAFE MODE):
#   1. wait for the predecessor (AFTER_OUT, REQUIRED, no default: the DAG-OACC r2 chain's log): GO after ZZV8FMINI-DONE, or after a ZZV8FMINI-STOP of the predecessor's OWN once its BOX2_BUSY is released; HALT on a ZZGUARD-STOP or a BOX2_BUSY
#      that stays held. Take BOX2_BUSY atomically, refresh box 2's code from the REMOTE tip with a SMALL export (repo_v8fdagorc_tip.tar, <= 300 MB asserted) and REFUSE a tip without the DAG-ORC queue stages / the perturbation levers;
#      sha-check the key files, md5-check the v9 releases, record the K-BUILD diff (trainer / data / model blobs against 1d4f4fbb), assert box 2's DAG_s<S> checkpoint / argv.json / summary.json are the banked ones
#   2. the four gates of A and REP first (`gatedag ARM S`, the trainer's own in-process gate), each pulled and asserted HERE (gate_dagorc_assert.py: G-DVB, the perturbation reaches the loss with every control exactly 0.0, init = DAG_s, the
#      ONE-LEVER rule against DAG2-0's banked argv), then the LAUNCH-GATE TOKEN (never by hand); then, in the order A s0, REP s0, A s1, REP s1: `traindag` (3,000 steps) -> `evaldag` (the standard T0 eval) -> pull to $ART/<run>/ (every
#      pull behind the 1 GiB floor on E:) -> the run record is asserted (K-GATE / K-INIT) -> K-REACH is PRINTED and ENFORCED (A p 0.3 +- 0.02, REP exactly 0: the gate does not cover the loop's ego.apply)
#   3. B s0 and B s1 the same way (gates first, then the two runs), unless DAGORC_SKIP_B=1
#   NOT HERE: the T1-kin rollouts and the reader: dev-box CPU.
#   chain_v8f_dagorc.sh --dry-run                PRINTS every stage line and the checks, touches nothing
#   chain_v8f_dagorc.sh --check-tree ROOT        OFFLINE: the tip-carries-the-lever greps applied to a directory
#   chain_v8f_dagorc.sh --wait-only              OFFLINE: only the predecessor wait (AFTER_OUT, BOX2_BUSY.txt in $B): prints ZZOAF-WAIT-GO or the STOP marker
#   AFTER_OUT=<log> chain_v8f_dagorc.sh          the real run
# Markers in ITS OWN log (chain_v8f_dagorc.out): ZZV8FMINI-DONE / ZZV8FMINI-STOP (v8f_guard_dagorc.sh's anchored end test knows them), ZZDAGORC-CONTROL-FAIL, ZZDISK-FLOOR-STOP. Script polling, not model turns.
# TIMEOUTS (polls of 30 s): gate 120, train 1,320 (11 h, one above the guard's 10 h cap), eval 600.
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
R=D:/tanitad/runs/v8f_mini_full
CH=dagorc
CKPT_SHA=(d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1 1d12ec2eb825990d57dcc45dda50b5178d1d733961b47ab0e3d50e6155fa39ef)      # DAG_s0 / DAG_s1 ckpt_last.pt (v8f-dagger/M2b_s<S>_dag/SHA256SUMS.txt): the init of every DAG-ORC arm
ARGV_SHA=(50e9d01d3dee881367a4eb1433903b05cc093c83d7bccc32a518e2290d9f3ce3 38a591e3e5a05c6f72669a268aaf99591af167758296720d650c46ebba4541b3)      # DAG_s0 / DAG_s1 argv.json
SUMM_SHA=(06ae1555ad278f4d323fdae33987f079d5b85005f35bddfd56dbc362d6ad4b41 e77f9e0125695ff01ff8fa12898663fbb9a32672160fded821d9a32d86ed6b6f)      # DAG_s0 / DAG_s1 summary.json
D20_ARGV_SHA=(2b09baa511e25c8bfbf151027ac4f441976f67dcbf986db32e912643bae536f0 b026ec750373ce3bedd9d4601189f8d32dffecf5702c4dbd371ec1f01332bf6a)      # DAG2-0_s0 / s1 argv.json (v8f-dagger2/M2b_s<S>_dag20/SHA256SUMS.txt): the ONE-LEVER reference
D20_SUMM_SHA=(aa858211a8b159bd56df040a1723ca1a7d9ff8f89d7191c9bf9c4610f5f98f92 25ecd3897fb0bd562d951dda2f3f1496648b6cce0b5eb8c48cd3f9001b61d1f5)      # DAG2-0_s0 / s1 summary.json: the gate comparator "B" (same data, weights, parameter groups)
OAF_LIB_SHA=026e8927608b7bd4a6818be9a4e849aa7d63d22b6a171445b52c10ec1c9ad29e      # the INSTALLED oaf_lib.sh this chain sources (the FLAT-DAG / DAG-OACC chains' lib, proven there): a changed lib is a changed chain
OAF_CHECKS_SHA=d1e48176cd2fea05dc1ab0b7678370da514a0de58a1ab55d37fd14d72a20a84f      # the INSTALLED v8f_oaf_chain_checks.py (disk-check / pred-verdict / argv-sha / argv-same)
BUILD_REF=1d4f4fbb11922f4dd62b51871d898c03e0c88635      # 1d4f4fbb: the commit DAG2-0 was trained at (K-BUILD's reference)
BUILD_FILES=(stack/scripts/v8f_mini_train.py stack/tanitad/data/v8f_mini_data.py stack/tanitad/models/v8f_mini.py)
AB_RUNS=("dagorc 0" "dagorcrep 0" "dagorc 1" "dagorcrep 1")      # PREREG_V8F_DAG_ORC section 10: A s0, REP s0, A s1, REP s1 (a stopped chain still leaves complete seed pairs first); ARM SEED
B_RUNS=("dagorc01 0" "dagorc01 1")      # then B s0, B s1
if [ "${DAGORC_SKIP_B:-0}" = 1 ]; then RUNS=("${AB_RUNS[@]}"); else RUNS=("${AB_RUNS[@]}" "${B_RUNS[@]}"); fi
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=${DAGORC_ART:-/e/Projects/TanitAD-artifacts/v8f-dag-orc}
DAG1_DIR=${DAGORC_DAG1_DIR:-/e/Projects/TanitAD-artifacts/v8f-dagger}      # DAG's pulled run (iteration 1): argv.json / summary.json for the pins; read-only here
DAG2_DIR=${DAGORC_DAG2_DIR:-/e/Projects/TanitAD-artifacts/v8f-dagger2}      # DAG2-0's banked run: argv.json (the ONE-LEVER reference) and summary.json (the comparator); read-only, no eval_* file is ever opened
B=${OAF_SHIP_DIR:-/c/Users/Admin/box2_ship}; L=$B/logs; E=$L/v8f_dagorc
HERE_SHIP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)      # the helper scripts sit beside this file (box2_ship when installed)
CHECKS=$HERE_SHIP/v8f_oaf_chain_checks.py; DCHECKS=$HERE_SHIP/v8f_dagorc_chain_checks.py; ASSERT=$HERE_SHIP/gate_dagorc_assert.py
DAG_STEPS=3000
GUARD_OUT=$L/v8fdagorc_guard.out
FLOOR_BYTES=1073741824      # a pull must leave >= 1 GiB free on E:
NEED_BYTES=${DAGORC_NEED_BYTES:-805306368}      # six runs x (checkpoint 87 MB + evals 3 MB + the rest) ESTIMATED ~0.6 GB: refuse at the start if E: could not take them above the floor
BUSY_TAKEN=0
rn() { echo "M2b_s${2}_${1}"; }       # rn <arm> <seed>: the run's directory under $R (M2b_s0_dagorc, ...; REP is indexed by the INIT seed)
rt() { echo "M2b-${1}_s${2}"; }       # rt <arm> <seed>: the run's eval tag
rg() { echo "gate_M2b-${1}_s${2}"; }  # rg <arm> <seed>: the run's gate directory under $R
# the lever-in-the-tip checks: "<file>|<grep -E pattern>". `.` = a backslash in the queue patterns.
LEVER=(
  'stack/scripts/v8f_mini_train.py|add_argument\("--ego-perturb-p"'
  'stack/scripts/v8f_mini_train.py|add_argument\("--ego-perturb-vscale"'
  'stack/scripts/v8f_mini_train.py|add_argument\("--ego-perturb-vmax"'
  'stack/scripts/v8f_mini_train.py|add_argument\("--ego-perturb-adrop"'
  'stack/scripts/v8f_mini_train.py|add_argument\("--ego-perturb-yawrate-std"'
  'stack/scripts/v8f_mini_train.py|add_argument\("--ego-perturb-seed"'
  'stack/scripts/v8f_mini_train.py|add_argument\("--dagger-p"'
  'stack/scripts/v8f_mini_train.py|add_argument\("--init-ckpt"'
  'stack/scripts/v8f_mini_train.py|"ego_perturb_p": "ego:ego_perturb_p"'
  'stack/scripts/v8f_mini_train.py|"ego_perturb_seed": "ego:ego_perturb_seed"'
  'stack/scripts/v8f_mini_train.py|"dagger_p": "dagger:dagger_p"'
  'stack/scripts/v8f_mini_train.py|"init_ckpt": "data:init_ckpt"'
  'stack/scripts/v8f_mini_train.py|^def ego_perturb_probe'
  'stack/scripts/v8f_mini_train.py|egop = ego_perturb_probe\(model, probe, cfg, ego\)'
  'stack/scripts/v8f_mini_train.py|^ +b = ego\.apply\(b, step\)'
  'stack/scripts/v8f_mini_train.py|"ego_perturb_n": ego_tot\["n"\]'
  'stack/tanitad/data/v8f_mini_data.py|^class EgoPerturb'
  'stack/tanitad/data/v8f_mini_data.py|^    def apply\(self, batch'
  # the DAG-ORC queue insertions (the queue file on the tip; `.` = a backslash)
  "$PK/code/v8f_mini_box2_queue.cmd|^:dagorcarm"
  "$PK/code/v8f_mini_box2_queue.cmd|goto :dagorcarm"
  "$PK/code/v8f_mini_box2_queue.cmd|DAGS=dagorc\""
  "$PK/code/v8f_mini_box2_queue.cmd|DAGS=dagorcrep\""
  "$PK/code/v8f_mini_box2_queue.cmd|DAGS=dagorc01\""
  "$PK/code/v8f_mini_box2_queue.cmd|--ego-perturb-seed %~2\""
  "$PK/code/v8f_mini_box2_queue.cmd|DAGSEED=1%~2"
  "$PK/code/v8f_mini_box2_queue.cmd|--ego-perturb-p %DAGORCP%%DAGORCL% --dagger-p 0"
)
STAGES=(gatedag traindag evaldag dagname dagarm dagorcarm)
lever_check() {   # lever_check <git tip | tree ROOT>: 0 iff every LEVER pattern is in its file
  local mode=$1 ref=$2 rec f pat miss=0
  for rec in "${LEVER[@]}"; do
    f=${rec%%|*}; pat=${rec#*|}
    if [ "$mode" = tree ]; then grep -qE -- "$pat" "$ref/$f" 2>/dev/null; else git -C $MIRROR show "$ref:$f" 2>/dev/null | grep -qE -- "$pat"; fi \
      || { echo "MISSING in $f: $pat"; miss=$((miss+1)); }
  done
  [ $miss = 0 ]
}
lib_check() {   # the two sourced files are the proven, installed ones
  [ "$(sha256sum "$HERE_SHIP/oaf_lib.sh" | cut -c1-64)" = "$OAF_LIB_SHA" ] || { echo "ZZV8FMINI-STOP $HERE_SHIP/oaf_lib.sh is not the proven file ($OAF_LIB_SHA): nothing was touched"; return 1; }
  [ "$(sha256sum "$CHECKS" | cut -c1-64)" = "$OAF_CHECKS_SHA" ] || { echo "ZZV8FMINI-STOP $CHECKS is not the proven file ($OAF_CHECKS_SHA): nothing was touched"; return 1; }
}
if [ "${1:-}" = "--check-tree" ]; then
  [ -d "${2:-}" ] || { echo "usage: --check-tree ROOT"; exit 2; }
  MIRROR=none; if lever_check tree "$2"; then echo "lever check: every one of ${#LEVER[@]} patterns is in $2"; exit 0; fi; echo "lever check: FAILED"; exit 1
fi
lib_check || exit 1
source "$HERE_SHIP/oaf_lib.sh"
plan() {   # plan <queue file> <arm> <seed>: GL / TL / EL = the lines the queue prints for the run's gate / train / eval stages. rc 1 if a stage prints no trainer / eval line
  GL=$(dry_stage "$1" "" "" gatedag $2 $3 | grep -a 'v8f_mini_train.py'); TL=$(dry_stage "$1" "" "" traindag $2 $3 | grep -a 'v8f_mini_train.py')
  EL=$(dry_stage "$1" "" "" evaldag $2 $3 | grep -a 'v8f_mini_eval.py')
  [ -n "$GL" ] && [ -n "$TL" ] && [ -n "$EL" ]
}
if [ "${1:-}" = "--wait-only" ]; then
  AFTER_OUT=${AFTER_OUT:-}; [ -n "$AFTER_OUT" ] || { echo "ZZV8FMINI-STOP AFTER_OUT is not set"; exit 1; }
  wait_predecessor; echo "ZZOAF-WAIT-GO"; exit 0
fi
if [ "${1:-}" = "--dry-run" ]; then
  QF=${V8F_QUEUE:-}
  if [ -z "$QF" ]; then
    MIRROR=${MIRROR:-/c/Users/Admin/tanitad-push}; BR=${BR:-agent/arch-inf-20260803}
    T=$(git -C $MIRROR ls-remote origin $BR | cut -f1); QF=${TMPDIR:-/tmp}/v8fdagorc_queue_dry_$T.cmd
    git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > "$QF" || { echo "ZZV8FMINI-STOP no queue at the tip"; exit 1; }
  fi
  echo "== predecessor"; echo "wait for ${AFTER_OUT:-<UNSET: a real run REFUSES to start without AFTER_OUT>}: GO after ^ZZV8FMINI-DONE or a predecessor ^ZZV8FMINI-STOP of its own with BOX2_BUSY released; HALT on a ZZGUARD-STOP or a held BOX2_BUSY"
  echo "== pinned (asserted on box 2 before the gates and after the last run; DAG2-0's argv / summary asserted on the dev box)"
  for s in 0 1; do echo "DAG_s$s $R/M2b_s${s}_dag/ckpt_last.pt ${CKPT_SHA[$s]} (argv.json ${ARGV_SHA[$s]}, summary.json ${SUMM_SHA[$s]}); DAG2-0_s$s argv.json ${D20_ARGV_SHA[$s]} summary.json ${D20_SUMM_SHA[$s]}"; done
  echo "== runs (order): ${RUNS[*]}"
  echo "== launch-gate token (written by the chain, one per run, after the gate's assertions pass; never by hand)"
  echo '{"kind": "launch-gate-token", "verdict": "PASS", "prereg": "PREREG_V8F_DAG_ORC", "binding": {"commit": "<the remote tip, 40 hex>", "argv_sha256": "<launch_gate.argv_sha256 of THE TRAINING ARGV as pinned by box 2>", "data_sha256": "<the gate selfcheck'"'"'s info.train_sha256>"}, ...}'
  for run in "${RUNS[@]}"; do
    set -- $run; arm=$1; s=$2
    plan "$QF" $arm $s || { echo "ZZV8FMINI-STOP the queue prints no gate / train / eval line for $(rn $arm $s)"; exit 1; }
    echo "== $(rn $arm $s): init $R/M2b_s${s}_dag/ckpt_last.pt (${CKPT_SHA[$s]}), tag $(rt $arm $s), gate $(rg $arm $s)"
    echo "-- gatedag $arm $s"; echo "$GL"
    echo "-- traindag $arm $s"; echo "$TL"
    echo "-- evaldag $arm $s"; echo "$EL"
    printf '%s' "$TL" > ${TMPDIR:-/tmp}/v8fdagorc_tl_$$.txt; printf '%s' "$GL" > ${TMPDIR:-/tmp}/v8fdagorc_gl_$$.txt
    echo "-- token argv_sha256 (the training argv): $(argvsha ${TMPDIR:-/tmp}/v8fdagorc_tl_$$.txt)"
    same=$($PY "$(cygpath -m $CHECKS)" argv-same "$(cygpath -m ${TMPDIR:-/tmp}/v8fdagorc_gl_$$.txt)" "$(cygpath -m ${TMPDIR:-/tmp}/v8fdagorc_tl_$$.txt)"); echo "-- gate argv = training argv with --out and --selfcheck-only changed: $same"
    ol=$($PY "$(cygpath -m $DCHECKS)" one-lever --ref-argv-json "$(cygpath -m "$DAG2_DIR/M2b_s${s}_dag20/argv.json")" --line "$(cygpath -m ${TMPDIR:-/tmp}/v8fdagorc_tl_$$.txt)" --arm $arm --seed $s)
    echo "-- planned line == DAG2-0's banked argv.json argv + the registered change for $arm (only --out differing): $ol"
    rm -f ${TMPDIR:-/tmp}/v8fdagorc_tl_$$.txt ${TMPDIR:-/tmp}/v8fdagorc_gl_$$.txt
    [ "$same" = SAME ] && [ "$ol" = ONE-LEVER ] || { echo "ZZV8FMINI-STOP the planned lines are not the one-lever DAG-ORC argv ($(rn $arm $s))"; exit 1; }
  done
  echo "== not run here: the T1-kin rollouts (dev box CPU) and the reader"
  exit 0
fi
# ======================================================================== the real run
if [ -z "${AFTER_OUT:-}" ]; then
  echo "ZZV8FMINI-STOP AFTER_OUT is not set: this chain has no default predecessor log (a default can already be satisfied by an OLD log). Set AFTER_OUT to the log of the chain that holds box 2 now."
  exit 1
fi
[ -f "$ASSERT" ] && [ -f "$DCHECKS" ] || { echo "ZZV8FMINI-STOP the DAG-ORC files are not installed beside the chain ($ASSERT, $DCHECKS): nothing was touched"; exit 1; }
[ -d "$(dirname "$ART")" ] || { echo "ZZV8FMINI-STOP $(dirname "$ART") does not exist: the artifacts have nowhere to go"; exit 1; }
mkdir -p "$ART" || { echo "ZZV8FMINI-STOP cannot create $ART"; exit 1; }
disk_floor "$ART" "$NEED_BYTES" "the six DAG-ORC runs"      # a named STOP before anything is waited for or taken
MIRROR=${OAF_MIRROR:-/c/Users/Admin/tanitad-push}; BR=${OAF_BR:-agent/arch-inf-20260803}
# EXPORT: what box 2's queue READS from the repo (export_closure_dagorc.py: the import closure of v8f_mini_train.py and v8f_mini_eval.py, plus the repo data paths those programs open by default).
# Generated by make_chain.py from export_closure_dagorc.json, never typed. The tar has a NEW name; no existing tar is overwritten or deleted.
EXPORT=(
  'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/raw/p10_yaw4512/p10_eval_pred6.npy'
  'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/raw/p10_yaw4512/p10_prior.pt'
  'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/v8f_mini_box2_queue.cmd'
  'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-26-dagger-draft/probe/THRESHOLDS_FIXED.txt'
  'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval141_kin.npz'
  'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval_grid_windows.json'
  'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz'
  'stack/scripts/v8f_mini_eval.py'
  'stack/scripts/v8f_mini_train.py'
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
GATE_POLLS=120; TRAIN_POLLS=1320; EVAL_POLLS=600
TAG=$(date '+%m%d%H%M')
CUR=$L/v8fdagorc_current.txt
wait_predecessor
[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present: $(cat $B/BOX2_BUSY.txt)"; exit 1; }
# the marker is taken ATOMICALLY (noclobber): two chains that wait on the same predecessor cannot both pass the test above and both write it
( set -C; echo "v8F-mini PREREG_V8F_DAG_ORC (DAG2-0's argv + the launch-recovery ego perturbation: A p 0.3, replicate, B p 0.1; both gates per run, then ${#RUNS[@]} 3,000-step fine-tunes of DAG_s<S> with standard T0 evals) since $(date '+%m-%d %H:%M'). Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt ) 2>/dev/null \
  || { echo "ZZV8FMINI-STOP busy marker taken while arming: $(cat $B/BOX2_BUSY.txt 2>/dev/null)"; exit 1; }
BUSY_TAKEN=1
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
disk_floor "$ART" "$NEED_BYTES" "the six DAG-ORC runs (re-checked after the wait)"
mkdir -p $E/runs $E/lines      # only now: a chain that refused or stopped while waiting has touched nothing
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
QT=$B/v8fdagorc_queue_$TAG.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" | tr -d '\r' > $QT
QC=$B/v8fdagorc_queue_$TAG.crlf.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > $QC      # the committed bytes (CRLF): what cmd.exe can run for the dry-run lines
for st in "${STAGES[@]}"; do grep -q "^:$st\$\|^if /i \"%~1\"==\"$st\" goto" $QT || stop "the tip's queue has no $st stage"; done
lever_check git "$T" || stop "the tip does not carry the DAG-ORC queue stages / the perturbation levers (the greps above): land them first"
for xf in "${EXPORT[@]}"; do git -C $MIRROR cat-file -e "$T:$xf" || stop "the tip has no $xf (the export list names only files box 2 reads)"; done
git -C $MIRROR cat-file -e "$T:$PK/raw/eval_grid_windows.json" || stop "the tip has no eval-141 grid"
# K-BUILD: the launch commit's trainer / data / model against the commit DAG2-0 was trained at. RECORDED (the judgement -- identical, or every hunk behind a default-off flag with a passing bit-identity test -- is the reader's)
{ echo "K-BUILD (PREREG_V8F_DAG_ORC section 6): the launch commit $T against $BUILD_REF (the commit DAG2-0 was trained at)"
  for f in "${BUILD_FILES[@]}"; do
    a=$(git -C $MIRROR rev-parse "$BUILD_REF:$f"); b=$(git -C $MIRROR rev-parse "$T:$f")
    echo "$f $BUILD_REF=$a tip=$b $([ "$a" = "$b" ] && echo IDENTICAL || echo CHANGED)"
    [ "$a" = "$b" ] || git -C $MIRROR -c core.autocrlf=false diff "$BUILD_REF" "$T" -- "$f"
  done; } > "$ART/K-BUILD.txt" 2>&1
echo "$(date '+%m-%d %H:%M') K-BUILD recorded: $(grep -ac ' IDENTICAL$' "$ART/K-BUILD.txt") of ${#BUILD_FILES[@]} files identical to $BUILD_REF ($ART/K-BUILD.txt)"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8fdagorc_tip.tar $T -- "${EXPORT[@]}" || stop "archive"
tsz=$(stat -c %s $B/repo_v8fdagorc_tip.tar); [ "$tsz" -gt 0 ] && [ "$tsz" -le $EXPORT_MAX_BYTES ] || stop "the code archive is $tsz bytes (must be 1..$EXPORT_MAX_BYTES)"
echo "$(date '+%m-%d %H:%M') code archive $B/repo_v8fdagorc_tip.tar: $tsz bytes, ${#EXPORT[@]} paths"
$SCP $B/repo_v8fdagorc_tip.tar tanitad-box2:D:/tanitad/repo_v8fdagorc_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_v8fdagorc_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_V8FDAGORC.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8fdagorc${TAG}_refresh.cmd
$SCP $B/v8fdagorc${TAG}_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8fdagorc${TAG}_refresh || stop "launch refresh"
waitrc v8fdagorc${TAG}_refresh 40
KEYSHA=""
for f in stack/scripts/v8f_mini_train.py stack/scripts/v8f_mini_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py stack/tanitad/data/v8f_dagger.py "$PK/code/v8f_mini_box2_queue.cmd" "$PK/raw/eval_grid_windows.json"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  [ "$l" = "$(rsha "D:/tanitad/repo/$f" SHA256)" ] || stop "box 2 $f != tip $T"
  KEYSHA="$KEYSHA\"$f\": \"$l\", "
done
for kv in v9_labels_train.npz=83e30bd3791f0767707eade5f4671759 v9_labels_eval139.npz=6b5c7f207cffc3b7eb3cd527fd433599; do
  [ "${kv#*=}" = "$(rsha "D:/tanitad/data/v9/${kv%%=*}" MD5)" ] || stop "box 2 v9 ${kv%%=*} md5 != ${kv#*=}"
done
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (${#KEYSHA} chars of key-file sha256; both v9 releases md5-checked)"
CK=$E/ckpt_sha256.json
# ---- 2. the base is fixed (PREREG section 1): DAG_s<S> (the init of every arm) is the banked run; DAG2-0 (the control and the one-lever reference) is the banked run on the dev box
pinned_b() {   # pinned_b <phase>
  local s f g exp
  for s in 0 1; do
    for f in ckpt_last.pt argv.json summary.json; do
      case $f in ckpt_last.pt) exp=${CKPT_SHA[$s]} ;; argv.json) exp=${ARGV_SHA[$s]} ;; *) exp=${SUMM_SHA[$s]} ;; esac
      g=$(rsha "$R/M2b_s${s}_dag/$f" SHA256); [ -n "$g" ] || stop "$1: cannot hash box 2's M2b_s${s}_dag/$f"
      [ "$g" = "$exp" ] || stop "$1: box 2's M2b_s${s}_dag/$f sha256 $g != the banked DAG run's $exp"
    done
  done
  echo "$(date '+%m-%d %H:%M') $1: DAG_s0/s1 (checkpoint, argv.json, summary.json) are the banked ones (sha256 asserted)"
}
rm -f $CK
pinned_b "before the gates"
for s in 0 1; do      # the dev box's copies: DAG's argv / summary are the banked run; DAG2-0's argv is the ONE-LEVER reference and its summary the gate comparator "B"
  for f in argv.json summary.json; do
    case $f in argv.json) exp=${ARGV_SHA[$s]} ;; *) exp=${SUMM_SHA[$s]} ;; esac
    [ -f "$DAG1_DIR/M2b_s${s}_dag/$f" ] || stop "the pulled DAG run $DAG1_DIR/M2b_s${s}_dag/$f is missing"
    g=$(sha256sum "$DAG1_DIR/M2b_s${s}_dag/$f" | cut -c1-64); [ "$g" = "$exp" ] || stop "the dev box's $DAG1_DIR/M2b_s${s}_dag/$f sha256 $g != SHA256SUMS.txt's $exp"
    case $f in argv.json) exp=${D20_ARGV_SHA[$s]} ;; *) exp=${D20_SUMM_SHA[$s]} ;; esac
    [ -f "$DAG2_DIR/M2b_s${s}_dag20/$f" ] || stop "DAG2-0's banked $DAG2_DIR/M2b_s${s}_dag20/$f is missing"
    g=$(sha256sum "$DAG2_DIR/M2b_s${s}_dag20/$f" | cut -c1-64); [ "$g" = "$exp" ] || stop "DAG2-0's banked $DAG2_DIR/M2b_s${s}_dag20/$f sha256 $g != SHA256SUMS.txt's $exp"
  done
done
fr=$($SSH "powershell -NoProfile -Command \"[int]((Get-PSDrive D).Free / 1GB)\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$' | head -1)
[ -n "$fr" ] && [ "$fr" -ge $MIN_FREE_GB ] || stop "box 2's D: has '$fr' GB free (< $MIN_FREE_GB): the checkpoints and the evals need room"
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep ${OAF_TELEMETRY_S:-45}; t1=$(tl); sleep ${OAF_TELEMETRY_S2:-35}; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
grep -qF "'^(ZZV8FMINI|" $B/v8f_guard_dagorc.sh || stop "v8f_guard_dagorc.sh does not know this chain's markers (anchored): it would never exit"
echo "" > $CUR
V8F_CHAIN_OUT=$L/chain_v8f_dagorc.out V8F_CURRENT=$CUR bash $B/v8f_guard_dagorc.sh > $GUARD_OUT 2>&1 &
gate_one() {   # gate_one <arm> <seed>: the run's in-process launch gate on box 2, pulled and asserted HERE, then its launch-gate token (written by this function, never by hand)
  local arm=$1 s=$2 g gd SC pin xa
  g=$(rn $arm $s); gd=$(rg $arm $s); SC="$ART/$g/gate/selfcheck.json"; pin="$ART/$g/gate/pin.txt"
  plan "$QC" $arm $s || stop "the tip's queue prints no gate / train / eval line for $g"
  mkdir -p "$ART/$g/gate" "$E/lines"
  printf '%s\n' "$GL" > "$E/lines/$g.gate.txt"; printf '%s\n' "$TL" > "$E/lines/$g.train.txt"; printf '%s\n' "$EL" > "$E/lines/$g.eval.txt"
  GATE_LOG="$R/$gd.log"; queue "gate_$g" $GATE_POLLS gatedag $arm $s; GATE_LOG=
  pull "$gd/selfcheck.json" "$ART/$g/gate"
  pull_opt "$gd.log" "$ART/$g/gate"
  pull "dagpin/M2b_s${s}_${arm}.train.argv.txt" "$ART/$g/gate"; mv -f "$ART/$g/gate/M2b_s${s}_${arm}.train.argv.txt" "$pin"
  xa=(--selfcheck "$(cygpath -m $SC)" --gate-line "$(cygpath -m $E/lines/$g.gate.txt)" --train-line "$(cygpath -m $E/lines/$g.train.txt)" --pin-line "$(cygpath -m $pin)" --arm $arm --seed $s --banked-init-sha ${CKPT_SHA[$s]}
      --b-summary "$(cygpath -m $DAG2_DIR/M2b_s${s}_dag20/summary.json)" --ref-argv-json "$(cygpath -m $DAG2_DIR/M2b_s${s}_dag20/argv.json)")
  $PY "$(cygpath -m $ASSERT)" check "${xa[@]}" || stop "gate DAG-ORC assertions $g"
  $PY "$(cygpath -m $ASSERT)" token "${xa[@]}" --commit "$T" --queue-sha "$(sha256sum $QT | cut -c1-64)" --key-sha-json "{${KEYSHA%, }}" --run $g --out "$(cygpath -m $ART/$g/launch_gate_token.json)" || stop "write the gate token $g"
}
ckptsha_x() {   # ckptsha_x <arm> <seed> <trained|after_evals>
  local g h h0; g=$(rn $1 $2)
  h=$(rsha "$R/$g/ckpt_last.pt" SHA256); [ -n "$h" ] || stop "cannot hash box 2's $g checkpoint ($3)"
  if [ "$3" = after_evals ]; then h0=$(ckget "$(rt $1 $2)_trained"); [ "$h" = "$h0" ] || stop "$g's checkpoint changed during the evals: $h != $h0"; fi
  ckset "$(rt $1 $2)_$3" "$h"
  echo "$(date '+%m-%d %H:%M') $g checkpoint sha256 ($3) ${h:0:16}..."
}
runone() {   # runone <arm> <seed>: train, standard T0 eval, pull, record and print the checkpoint sha256, assert the run's record and the REACHED control
  local arm=$1 s=$2 g tg f x l
  g=$(rn $arm $s); tg=$(rt $arm $s)
  echo "$g" > $CUR
  queue "t_$g" $TRAIN_POLLS traindag $arm $s
  echo "" > $CUR
  ckptsha_x $arm $s trained
  queue "e_$g" $EVAL_POLLS evaldag $arm $s
  ckptsha_x $arm $s after_evals
  for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull "$g/$f" "$ART/$g"; done
  for x in json npz log; do pull "eval/eval_$tg.$x" "$ART/$g"; done
  pull_opt "eval/eval_${tg}_proj.npz" "$ART/$g"
  l=$(sha256sum "$ART/$g/ckpt_last.pt" | cut -c1-64); [ "$l" = "$(ckget "${tg}_after_evals")" ] || stop "$g: the pulled checkpoint's sha256 $l != the value measured on box 2"
  ( cd "$ART/$g" && sha256sum ckpt_last.pt "eval_$tg.json" "eval_$tg.npz" selfcheck.json summary.json argv.json train_log.jsonl launch_gate_token.json | tee SHA256SUMS.txt )
  echo "$(date '+%m-%d %H:%M') $g pulled to $ART/$g (checkpoint sha256 $l)"
  $PY "$(cygpath -m $DCHECKS)" run-record --run-dir "$(cygpath -m $ART/$g)" --train-line "$(cygpath -m $E/lines/$g.train.txt)" --pin-line "$(cygpath -m $ART/$g/gate/pin.txt)" --arm $arm --seed $s --banked-init-sha ${CKPT_SHA[$s]} --steps $DAG_STEPS || stop "run DAG-ORC record $g"
  $PY "$(cygpath -m $DCHECKS)" reached --summary "$(cygpath -m $ART/$g/summary.json)" --arm $arm || { echo "ZZDAGORC-CONTROL-FAIL $g: the perturbation did not reach training as registered (K-REACH: the gate does not cover the loop's ego.apply); the reader would REFUSE this run"; stop "reached-control fail $g"; }
  grep ZZGUARD-STOP $GUARD_OUT 2>/dev/null && stop "guard stop"
}
for run in "${AB_RUNS[@]}"; do set -- $run; gate_one $1 $2; done
for run in "${AB_RUNS[@]}"; do set -- $run; runone $1 $2; done
if [ "${DAGORC_SKIP_B:-0}" != 1 ]; then
  for run in "${B_RUNS[@]}"; do set -- $run; gate_one $1 $2; done
  for run in "${B_RUNS[@]}"; do set -- $run; runone $1 $2; done
fi
pinned_b "after the last run"
echo "$(date '+%m-%d %H:%M') summary: $(for run in "${RUNS[@]}"; do set -- $run; echo -n "$(rn $1 $2)=$(ckget "$(rt $1 $2)_after_evals" | cut -c1-12) "; done)"
finish "PREREG_V8F_DAG_ORC: ${#RUNS[@]} runs (${RUNS[*]}) each through its gate (G-DVB, ego probe, ONE-LEVER against DAG2-0's banked argv), trained 3,000 steps from DAG_s<S>, T0-evaluated (standard) and pulled to $ART (sha256 in each SHA256SUMS.txt; K-REACH enforced); the T1-kin rollouts and the reader run on the dev box (NOT here)"
