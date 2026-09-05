#!/usr/bin/env bash
# =============================================================================
# D-RL-REFCV3-MIN — launch chain for the minimal pre-registered RL-stage on/off
# experiment on the FROZEN refcv3 @ 40,284.   ⛔⛔ PREPARED, NOT LAUNCHED.
#
# SPEC:  TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-refc-rl-readiness/SPEC.md
# RESUMED 2026-09-05 (Production/Deploy FlyWheel, second agent): STAGE 0 gained the
# `humanflag` gate (H-RL-THRESH-1 class: the reward must not flag the human driver's
# own future) and every driver call carries --lead-mode (default `track`: the lead's
# own obstacle.offline track, time-aligned; `static` is the legacy comparison).
# Driver: stack/scripts/rl_refcv3_min.py   (the same file runs preflight / arm / verdict)
#
# Nothing past STAGE 0 runs unless LAUNCH_APPROVED=1 is exported by whoever owns
# the cost (Master Mind / PI).  STAGE 0 (preflight) trains nothing and writes no
# checkpoint; it is the cost measurement the approval is made against.
#
# Every stage asserts its artifact BY CONTENT (bytes + a parse), never by exit code.
# Machines: the dev-box RTX 4060 for everything (the criteria-checked eval harness
# lives here: tools/criteria_check.py + products/P7-TanitEval/CRITERIA_REGISTRY.json,
# both ABSENT on Thor's checkout, MEASURED 2026-09-05).  Thor supplies the RL-fit
# clips over the LAN (MEASURED 37.8 MB in 1.2 s per clip; 8 clips in 8 s).
# ⛔ The training pod (tanitad-refcv3, refcv4b live) is never touched.
# =============================================================================
set -euo pipefail

# ---- the box --------------------------------------------------------------------
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
REPO="${TANITAD_REPO:-/c/Users/Admin/refcv4b_repo}"        # off-Drive clone: stack/ taniteval/ tools/ products/
export TANITAD_REPO="$REPO" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
export PYTHONPATH="$(cygpath -w "$REPO/stack");$(cygpath -w "$REPO/taniteval");$(cygpath -w "$REPO")"
W="/c/Users/Admin/rl_refcv3_min"                              # work dir (local disk, never G:)
mkdir -p "$W"

# ---- the FROZEN base (md5 pinned to MODEL_REGISTRY.md §4.5) ---------------------
CKPT="$W/base/ckpt_step40284_frozen.pt"                     # copied from run_refcv3_viz/ckpt (local) or HF Sayood/tanitad-refc-v3
CKPT_MD5="b1ed7075ff730d0993d2eaa3c86f6b56"
CONFIG="$W/base/config.json"                                # the trainer's config.json beside it (required by the loader)

# ---- corpora ---------------------------------------------------------------------
EVAL_EPS="/c/Users/Admin/run_refcv3_ol/data/eval"                                  # 141 v7.2 EVAL clips (local)
EVAL_LABELS="/c/Users/Admin/run_refcv3_ol/data/s2_labels_v7.2_eval.jsonl.gz"       # md5 aa12c948f062181c3297265b51526ec5
EVAL_LEAD="/c/Users/Admin/tanitad-wt/_lead_b1/b1_eval_lead_block.npz"              # sha256 c0525943…
TRAIN_LABELS="/c/Users/Admin/tanitad-wt/_s2build/release/v72/s2_labels_v7.2_train.jsonl.gz"
BASE_DUMP="/c/Users/Admin/_wp56/dump/refcv3_40284_dump"     # the BANKED 40,284 roll: grid 2s, stride 5, 4823 win / 141 eps
FIT_N="${FIT_N:-120}"; FIT="$W/fit$FIT_N"; FIT_LIST="$W/fitlist_$FIT_N.txt"; FIT_LEAD="$W/fit${FIT_N}_lead_block.npz"
THOR="tanitad-thor-wifi"; THOR_CACHE="/home/nvidia/data/physicalai-b1-w120-256x640cyl"
REL="/c/Users/Admin/tanitad-wt/_s2build/release/tanitad-v7-training-corpus"

# ---- the committed knobs (SPEC §3) ------------------------------------------------
BATCH=2; GROUP=4; NOISE=0.1; SEED=0; STRIDE=5; NBOOT=2000
LEAD_MODE="${LEAD_MODE:-track}"; HUMAN_FLAG_MAX=0.15      # SPEC §3 / §4 (the humanflag gate)
ARMS=(ctrl0 rl reg_echo)                                    # ctrl0 (lr=0, 200 steps) · rl (2000) · reg_echo (2000)

DRV="$REPO/stack/scripts/rl_refcv3_min.py"
need () { [ -s "$1" ] || { echo "⛔ missing/empty: $1"; exit 2; }; }
jsonok () { "$PY" -c "import json,sys; json.load(open(sys.argv[1],encoding='utf-8'))" "$1" || { echo "⛔ not JSON: $1"; exit 2; }; }

stage () { echo; echo "================= STAGE $1 — $2 ($(date -u +%FT%TZ)) ================="; }

# ============================ STAGE 0 — PREFLIGHT (always runs; trains nothing) ===
stage 0 "preflight: base md5 · imports · fitlist · HUMANFLAG gate · surface · no-future-leak · audit · anchor==0 · lr=0 timing"
mkdir -p "$W/base"
[ -s "$CKPT" ] || cp /c/Users/Admin/run_refcv3_viz/ckpt/ckpt_step40284_frozen.pt "$CKPT"
[ -s "$CONFIG" ] || cp /c/Users/Admin/run_refcv3_viz/ckpt/config.json "$CONFIG"
got=$(md5sum "$CKPT" | cut -d' ' -f1); [ "$got" = "$CKPT_MD5" ] || { echo "⛔ base md5 $got != registry $CKPT_MD5"; exit 2; }
echo "base ckpt md5 OK ($CKPT_MD5)"
# the RL-fit clip list: train-split ids NOT in the eval split (refuses on any intersection)
"$PY" "$DRV" --mode fitlist --train-labels "$TRAIN_LABELS" --eval-labels "$EVAL_LABELS" --n-fit "$FIT_N" --seed "$SEED" --out "$FIT_LIST"
need "$FIT_LIST"
# preflight on whatever fit clips + lead block already exist (fit8 from the readiness audit if nothing else)
PF_EPS="${PF_EPS:-$W/fit8}"; PF_LEAD="${PF_LEAD:-$W/fit8_lead_block.npz}"
need "$PF_LEAD"
# ⛔ the H-RL-THRESH-1 gate (0 GPU): the reward's own scene context must not flag the HUMAN future
"$PY" "$DRV" --mode humanflag --lead-mode "$LEAD_MODE" --ckpt "$CKPT" --config "$CONFIG" --expect-step 40284 \
   --episodes "$PF_EPS" --labels "$TRAIN_LABELS" --lead-block "$PF_LEAD" --lru 8 \
   --human-flag-max "$HUMAN_FLAG_MAX" --out "$W/humanflag_fit8.json"
need "$W/humanflag_fit8.json"; jsonok "$W/humanflag_fit8.json"
"$PY" -c "import json,sys; r=json.load(open(sys.argv[1],encoding='utf-8')); assert r['PASS'], 'humanflag FAILED under '+r['active_lead_mode']" "$W/humanflag_fit8.json"
"$PY" "$DRV" --mode preflight --lead-mode "$LEAD_MODE" --ckpt "$CKPT" --config "$CONFIG" --expect-step 40284 \
   --episodes "$PF_EPS" --labels "$TRAIN_LABELS" --lead-block "$PF_LEAD" \
   --eval-episodes "$EVAL_EPS" --eval-labels "$EVAL_LABELS" \
   --batch "$BATCH" --group "$GROUP" --noise "$NOISE" --seed "$SEED" --preflight-steps 20 --lru 8 \
   --out "$W/preflight.json"
need "$W/preflight.json"; jsonok "$W/preflight.json"
"$PY" -c "import json,sys; r=json.load(open(sys.argv[1],encoding='utf-8')); assert r['PASS'], 'preflight FAILED'; t=r['timing']; print('preflight PASS ·', round(t['total_s_per_step'],3),'s/step · est train min: rl',t['est_train_min_rl'],'reg_echo',t['est_train_min_reg_echo'],'ctrl0',t['est_train_min_ctrl0'])" "$W/preflight.json"

if [ "${LAUNCH_APPROVED:-0}" != "1" ]; then
  echo; echo "⛔ NOT LAUNCHED. STAGE 0 complete; export LAUNCH_APPROVED=1 to run STAGES 1-5 (approval of the SPEC's cost)."; exit 0
fi

# ============================ STAGE 1 — the RL-fit view (Thor -> dev box) ===========
stage 1 "RL-fit clips: $FIT_N train-split v2ep files from Thor over the LAN"
mkdir -p "$FIT"
while read -r c; do [ -s "$FIT/$c.v2ep.pt" ] || scp -q -o BatchMode=yes "$THOR:$THOR_CACHE/$c.v2ep.pt" "$FIT/" || echo "FAIL $c"; done < "$FIT_LIST"
n=$(ls "$FIT" | grep -c 'v2ep.pt'); [ "$n" -ge $((FIT_N * 9 / 10)) ] || { echo "⛔ only $n/$FIT_N fit clips landed"; exit 2; }
[ "$(comm -12 <(sort "$FIT_LIST") <(ls "$EVAL_EPS" | sed 's/\.v2ep\.pt//' | sort) | wc -l)" = "0" ] || { echo "⛔ GATE 3: fit ∩ eval != 0"; exit 2; }
echo "$n fit clips · fit ∩ eval = 0"

# ============================ STAGE 2 — the fit lead block (scene facts, t0 only) ==
stage 2 "lead block for the fit clips (build_lead_block_b1.py --pull; ~1 min / 150 clips MEASURED)"
[ -s "$FIT_LEAD" ] || "$PY" "$REPO/taniteval/tools/build_lead_block_b1.py" --clips "$FIT_LIST" \
   --ego-tar "$REL/egomotion/egomotion_alpamayo.tar" --ts-tar "$REL/timestamps/timestamps.tar" \
   --obs-dir "$W/obs_fit$FIT_N" --pull --chunk-map "$REL/index/clip_to_chunk.parquet" \
   --keys "$REPO/Keys.txt" --v2ep-dir "$FIT" --k 10 --dt 0.2 --out "$FIT_LEAD"
need "$FIT_LEAD"; jsonok "$FIT_LEAD.report.json"
"$PY" -c "import json,sys; r=json.load(open(sys.argv[1],encoding='utf-8')); c=r['counts']; print('lead rows', c, '· clips with any LEAD', r['n_clips_with_any_lead'], '/', r['n_clips'])" "$FIT_LEAD.report.json"

# ============================ STAGE 3 — the arms (sequential, one GPU) =============
stage 3 "arms ${ARMS[*]}: before-readout -> run_posttrain -> ckpt_after.pt -> after-readout"
export LAUNCH_APPROVED=1
for arm in "${ARMS[@]}"; do
  echo "--- arm $arm ---"
  "$PY" "$DRV" --mode arm --lead-mode "$LEAD_MODE" --arm "$arm" --ckpt "$CKPT" --config "$CONFIG" --expect-step 40284 \
     --episodes "$FIT" --labels "$TRAIN_LABELS" --lead-block "$FIT_LEAD" \
     --eval-episodes "$EVAL_EPS" --eval-labels "$EVAL_LABELS" --eval-lead-block "$EVAL_LEAD" \
     --batch "$BATCH" --group "$GROUP" --noise "$NOISE" --seed "$SEED" --lru 16 --readout-windows 120 \
     --out-dir "$W/arms" 2>&1 | tee "$W/arms_$arm.log"
  need "$W/arms/$arm/ckpt/ckpt_after.pt"; need "$W/arms/$arm/ckpt/config.json"
  jsonok "$W/arms/$arm/arm_summary.json"; jsonok "$W/arms/$arm/rl/summary.json"
  "$PY" -c "import json,sys; s=json.load(open(sys.argv[1],encoding='utf-8')); assert s['done'] is True; print('done-marker OK · steps', s['steps'], '· counters', {k:v for k,v in s['counters'].items() if k!='components_fired'})" "$W/arms/$arm/rl/summary.json"
done

# ============================ STAGE 4 — the BINDING eval (T1, OPEN LOOP, 4 families) =
stage 4 "openloop_suite.py per arm on the 141 eval clips, grid 2s, stride $STRIDE (same grid as the banked base)"
for arm in "${ARMS[@]}"; do
  "$PY" "$REPO/taniteval/tools/openloop_suite.py" --ckpt "$W/arms/$arm/ckpt/ckpt_after.pt" --config "$W/arms/$arm/ckpt/config.json" \
     --episodes "$EVAL_EPS" --labels "$EVAL_LABELS" --lead-block "$EVAL_LEAD" \
     --nav-source v72 --grid 2s --action-units steer --with-oracle-sel \
     --window-stride "$STRIDE" --n-boot "$NBOOT" --seed "$SEED" --expect-step 40284 \
     --dump-dir "$W/eval/${arm}_dump" --out-dir "$W/eval" --tag "refcv3-40284-rlmin-$arm" \
     --tiers os=T1,os_navshuf=T1,os_navzero=T1,oracle_sel=T0 \
     --corpus "physicalai B1 v7.2 EVAL split, 141 clips" --parity-status "NON-PARITY (B1, v2_parity false)" \
     --train-eval-disjoint "ASSERTED by rl_refcv3_min.py GATE 3 (fit ∩ eval = 0)" --strict 2>&1 | tee "$W/eval_$arm.log"
  need "$W/eval/refcv3-40284-rlmin-$arm.json"; jsonok "$W/eval/refcv3-40284-rlmin-$arm.json"
done

# ============================ STAGE 5 — pairing + the committed exit =================
stage 5 "paired_openloop.py (arm vs the BANKED base, shared floor ha0) + rl_refcv3_min.py --mode verdict"
need "$BASE_DUMP/manifest.json"
for arm in "${ARMS[@]}"; do
  "$PY" "$REPO/taniteval/tools/paired_openloop.py" --a-dump "$BASE_DUMP" --a-name base --a-arm os \
     --b-dump "$W/eval/${arm}_dump" --b-name "$arm" --b-arm os --floor ha0 --n-boot "$NBOOT" --seed "$SEED" \
     --out "$W/arms/paired_${arm}_vs_base.json" --md "$W/arms/paired_${arm}_vs_base.md"
  need "$W/arms/paired_${arm}_vs_base.json"; jsonok "$W/arms/paired_${arm}_vs_base.json"
done
"$PY" "$DRV" --mode verdict --out-dir "$W/arms"
need "$W/arms/verdict.json"
echo; echo "RLMIN_CHAIN_DONE  ->  $W/arms/verdict.json  (bank: verdict.json, paired_*.json/.md, eval/*.json, arms/*/arm_summary.json, preflight.json, humanflag_fit8.json)"
