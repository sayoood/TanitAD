#!/bin/sh
# ============================================================================
# THE ONE COMMAND that scores refcv5-v2 against refcv4b when the checkpoint lands.
#
#   bash C:/Users/Admin/refcv5cmp/compare.sh \
#        --ckpt C:/Users/Admin/refcv5v2_final/ckpt_40284_FINAL.pt \
#        --config C:/Users/Admin/refcv5v2_final/config.json \
#        --tag refcv5-v2
#
# It (1) re-syncs the repo code off the G: Drive and md5-manifests it, (2) rolls
# the checkpoint on the DEV-BOX RTX 4060 over the 141-episode / 4,823-window B1
# v7.2 EVAL grid, (3) runs refcv5_compare.py against the banked refcv4b baseline
# dump and the banked refcv3 @40,284 prior, and (4) prints the four-family panel
# and the VERDICT against the bar registered on 2026-09-07.
#
# ⛔ DEV-BOX RTX 4060 ONLY. Never the A40 (it is training), never Thor.
# ⛔ G: cannot RUN the stack (Errno 22 mid-import) — everything runs from
#    C:\Users\Admin\refcv5cmp\repo, which step (1) refreshes.
# ============================================================================
set -e
R=/c/Users/Admin/refcv5cmp
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6

CKPT=""; CONFIG=""; TAG="refcv5-v2"; SKIP_ROLL=0
# STOP: refcv5-v2 runs `--sampler ddim`, which DRAWS NOISE AT EVAL by design
#    (refc.py:1926). The same checkpoint rolled twice does NOT give the same
#    answer, and the episode-cluster bootstrap is structurally blind to that.
#    The replicate is therefore one flag:
#        bash compare.sh --ckpt ... --tag refcv5-v2-seed1 --infer-seed 1
#    MEASURED 2026-09-07: on refcv4b (no sampler) two seeds are BIT-IDENTICAL,
#    4 of 4 episode dumps, so this cannot move a deterministic arm.
INFER_SEED=0
BASE_DUMP="$R/out/refcv4b_devbox_dump"
BASE_JSON="$R/out/refcv4b_devbox.json"
PRIOR_DUMP="$R/dumps/refcv3_40284/refcv3_40284_dump"
BASE_CFG="/c/Users/Admin/refcv4b_final/config.json"
BASE_CKPT="/c/Users/Admin/refcv4b_final/ckpt_40284.pt"
while [ $# -gt 0 ]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --base-dump) BASE_DUMP="$2"; shift 2;;
    --base-config) BASE_CFG="$2"; shift 2;;
    --base-ckpt) BASE_CKPT="$2"; shift 2;;
    --base-json) BASE_JSON="$2"; shift 2;;
    --prior-dump) PRIOR_DUMP="$2"; shift 2;;
    --infer-seed) INFER_SEED="$2"; shift 2;;
    --skip-roll) SKIP_ROLL=1; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
[ -n "$CKPT" ] || { echo "⛔ --ckpt is required" >&2; exit 2; }
[ -n "$CONFIG" ] || CONFIG="$(dirname "$CKPT")/config.json"

# ⛔ The refcv4b BASELINE dump is what makes the comparison paired. If this
#    off-Drive working copy has been wiped, restore it from the repo — it is
#    banked, it is not only here:
#      tar -xzf ".../2026-09-07-refcv5-v2-comparison/raw/refcv4b-40284-devbox4060-dump.tar.gz" -C "$R/out"
#      md5 c638603ead015d5547b8e88f0148e159
#    and the refcv3 prior from taniteval/results/refcv3-40284-openloop-dump.tar.gz.
#    ⛔ A dump that reads ZERO episodes means the mount flapped — RETRY, do not
#    believe it; refcv5_compare.py refuses rather than scoring an empty panel.
if [ ! -d "$BASE_DUMP" ]; then
  echo "⛔ baseline dump missing: $BASE_DUMP" >&2
  echo "   restore it from raw/refcv4b-40284-devbox4060-dump.tar.gz (see comment above)" >&2
  exit 3
fi
if [ ! -d "$PRIOR_DUMP" ]; then
  echo "⚠️  refcv3 prior dump missing: $PRIOR_DUMP — the refcv4b-vs-refcv3 control" >&2
  echo "   will be skipped. Restore from taniteval/results/refcv3-40284-openloop-dump.tar.gz" >&2
  PRIOR_ARG=""
else
  PRIOR_ARG="--prior-dump $PRIOR_DUMP --prior-label refcv3"
fi

# STOP: The BASE arm's own config + checkpoint, when they are on this box. They are
#    WITNESSES to the bank size (`core.decoder.anchors.shape[0]`), which the
#    comparison now RESOLVES instead of remembering: a hardcoded 1/128 was once
#    banked into sidecars while the live bank was 117. Absent is fine - the dump
#    manifest is a witness too - but a bank size is never defaulted.
# NOTE: `set -e` is on: a bare `[ -f x ] && y` EXITS THE SCRIPT when x is absent,
#    which would turn "the reference checkpoint is not on this box" into a
#    silent abort. if/then/fi, always.
BASE_REF_ARG=""
if [ -f "$BASE_CFG" ]; then BASE_REF_ARG="$BASE_REF_ARG --base-config $BASE_CFG"; fi
if [ -f "$BASE_CKPT" ]; then BASE_REF_ARG="$BASE_REF_ARG --base-ckpt $BASE_CKPT"; fi

OUT="$R/out/${TAG}"
DUMP="$R/out/${TAG}_dump"

echo "== [1/3] sync the code off G: and manifest it =="
"$PY" "$R/sync_repo.py"

export PYTHONPATH="C:\\Users\\Admin\\refcv5cmp\\repo\\stack;C:\\Users\\Admin\\refcv5cmp\\repo\\taniteval;C:\\Users\\Admin\\refcv5cmp\\repo"

if [ "$SKIP_ROLL" = "0" ]; then
  echo "== [2/3] roll $TAG on the dev-box RTX 4060 (141 eps / 4,823 windows, ~30 min) =="
  # ⛔ --analyze-only <dump> re-runs the ANALYSIS with zero GPU. The rollout is
  #    the only expensive part; an analysis-time failure must never cost a re-roll.
  ( cd "$R/repo" && "$PY" taniteval/tools/refcv3_arm.py \
      --ckpt "$CKPT" --config "$CONFIG" \
      --episodes "$R/data/eval" \
      --labels "$R/data/s2_labels_v7.2_eval.jsonl.gz" \
      --nav-source v72 --grid 2s --action-units steer --device cuda \
      --window-stride 5 --with-oracle-sel \
      --lead-block "$R/data/b1_eval_lead_block.npz" \
      --n-boot 2000 --seed 0 --infer-seed "$INFER_SEED" \
      --dump-dir "$DUMP" --out "${OUT}.json" \
      --tiers "os=T1,os_navshuf=T1,os_navzero=T1,oracle_sel=T0" ) \
    2>&1 | tee "${OUT}_roll.log"
else
  echo "== [2/3] SKIPPED the roll (--skip-roll); using ${OUT}.json + $DUMP =="
fi

echo "== [3/3] four families + paired margins + VERDICT =="
( cd "$R/repo" && "$PY" "$R/refcv5_compare.py" \
    --new-json "${OUT}.json" --new-dump "$DUMP" --new-label "$TAG" \
    --new-config "$CONFIG" --new-ckpt "$CKPT" \
    $BASE_REF_ARG \
    --base-json "$BASE_JSON" --base-dump "$BASE_DUMP" --base-label refcv4b \
    $PRIOR_ARG \
    --n-boot 2000 --seed 0 \
    --out-prefix "$R/out/${TAG}_vs_refcv4b" ) 2>&1 | tee "${OUT}_compare.log"

echo
echo "artifacts:"
echo "  $R/out/${TAG}.json                 (refcv3_arm result, all 7 arms)"
echo "  $DUMP/                             (per-window dump; re-analysable with --analyze-only, 0 GPU)"
echo "  $R/out/${TAG}_vs_refcv4b.json      (comparison record)"
echo "  $R/out/${TAG}_vs_refcv4b.txt       (the panel + verdict, as printed)"
