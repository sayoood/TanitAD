#!/bin/bash
# Build the DriveRL TEACHER environment on a Linux pod, from public sources, and PROVE it works.
#
# ⛔ WHY THIS EXISTS. The pod must generate ~82 % of navtrain's targets (we hold DBs for only 214 of
# 1,192 logs), and target generation runs the frozen teacher. Until 2026-09-23 neither POD_HANDOFF
# nor pod_bootstrap.sh mentioned the teacher environment at all -- the only working copy was a
# Windows venv in a session scratchpad. A pod launched on that plan would have stalled at
# "build the teacher", with no recipe.
#
# Everything here is reproduced from what VALIDATED every bank on the dev box:
#   DriveZero  https://github.com/XiaomiAutoL3/DriveZero.git @ 2495954  (checkpoint is git-tracked)
#   Python 3.11 · torch 2.7.1+cu128 · 113 pinned deps in requirements_teacher_lock.txt
#   nuPlan maps v1.0 from the public S3 bucket (0.90 GiB zip)
# Opaque ZZ markers only (CLAUDE.md: never grep for words the command itself contains).
set -u
WORK="${WORK:-/workspace}"
PKG="${PKG:-$WORK/refe-plan}"            # the shipped refe-plan directory
DZ_PARENT="$WORK/dz"; DZ="$DZ_PARENT/DriveZero/DriveRL"
VENV="${VENV:-$WORK/venv-teacher}"
MAPS="${MAPS:-$WORK/data/nuplan-maps}"
DZ_COMMIT="2495954"
CKPT_SHA="d7b1fdd06e1aadf4dc429b014807534f3b2028a4137f47f8fdb2a0620e3cb27b"
# ⛔ The config pin is the CANONICAL git blob (LF). The first pin, d318ab6a..., was hashed on the
# Windows dev box whose checkout runs core.autocrlf=true -- 118 CRLF lines -- so the first pod
# REFUSED a byte-correct file (MEASURED 2026-09-23: CR-stripped dev-box file == pod file ==
# `git show 2495954:DriveRL/release/configs/driverl_teacher.yaml` == 0521575f...).
CFG_SHA="0521575f25e0c24327646a07f07f815c80dafead2dcfc98b486480eae1d15e2a"
S3="https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1"
fail () { echo "ZZFAIL $1ZZ"; exit 1; }

echo "=== 1. DriveZero @ $DZ_COMMIT ==="
mkdir -p "$DZ_PARENT"
if [ ! -d "$DZ_PARENT/DriveZero/.git" ]; then
  git clone -q https://github.com/XiaomiAutoL3/DriveZero.git "$DZ_PARENT/DriveZero" || fail clone
fi
git -C "$DZ_PARENT/DriveZero" checkout -q "$DZ_COMMIT" || fail checkout
# ⛔ assert on the ARTIFACT, not the checkout's exit code: the teacher must be byte-identical
got=$(sha256sum "$DZ/release/checkpoints/checkpoint_2400.pt" | cut -d' ' -f1)
[ "$got" = "$CKPT_SHA" ] || fail "checkpoint sha256 $got"
got=$(sha256sum "$DZ/release/configs/driverl_teacher.yaml" | cut -d' ' -f1)
[ "$got" = "$CFG_SHA" ] || fail "config sha256 $got"
echo "ZZOK teacher checkpoint + config byte-identical to the dev box ZZ"

echo "=== 2. venv: Python 3.11, deps --no-deps, torch LAST from the pinned index ==="
command -v uv >/dev/null || pip install -q uv || fail uv
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.11 "$VENV" || fail venv
PY="$VENV/bin/python"
uv pip install -q --python "$PY" --no-deps -r "$PKG/code/requirements_teacher_lock.txt" || fail deps
uv pip install -q --python "$PY" --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.7.1" || fail torch
uv pip install -q --python "$PY" --no-deps -e "$DZ/nuplan-devkit" -e "$DZ" || fail editable

echo "=== 3. prove CUDA with a real conv2d, not an import ==="
# CLAUDE.md: cuBLAS/matmul can succeed while cuDNN/conv is broken, and `import torch` proves nothing.
"$PY" - <<'PYEOF' || fail cuda
import torch
assert torch.cuda.is_available(), "cuda not available"
x = torch.randn(1, 3, 64, 64, device="cuda")
w = torch.randn(8, 3, 3, 3, device="cuda")
y = torch.nn.functional.conv2d(x, w)
torch.cuda.synchronize()
print("ZZOK conv2d", tuple(y.shape), torch.__version__, torch.version.cuda, "ZZ")
PYEOF

echo "=== 4. prove the imports come from THIS checkout, not a stray install ==="
"$PY" - "$DZ" <<'PYEOF' || fail imports
import sys, nuplan, driverl
dz = sys.argv[1]
for m in (nuplan, driverl):
    assert m.__file__.startswith(dz), f"{m.__name__} imported from {m.__file__}, not {dz}"
print("ZZOK imports from the shipped checkout ZZ")
PYEOF

echo "=== 5. nuPlan maps v1.0 ==="
mkdir -p "$MAPS"
if [ ! -d "$MAPS/nuplan-maps-v1.0/us-nv-las-vegas-strip" ]; then
  curl -sSL -C - --retry 10 -o "$MAPS/maps.zip" "$S3/nuplan-maps-v1.0.zip" || fail maps-download
  [ "$(stat -c %s "$MAPS/maps.zip")" = "971557640" ] || fail "maps size $(stat -c %s "$MAPS/maps.zip")"
  "$PY" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
      "$MAPS/maps.zip" "$MAPS" || fail maps-extract
fi
# the PUBLIC zip unpacks to maps/ (MEASURED on the first pod, 2026-09-23); NUPLAN_MAPS_ROOT and the
# dev box use nuplan-maps-v1.0/ -- same contents (4 cities + nuplan-maps-v1.0.json), so rename
[ -d "$MAPS/maps" ] && [ ! -d "$MAPS/nuplan-maps-v1.0" ] && mv "$MAPS/maps" "$MAPS/nuplan-maps-v1.0"
n=$(ls -d "$MAPS"/nuplan-maps-v1.0/*/ 2>/dev/null | wc -l)
[ "$n" -ge 4 ] || fail "maps: $n city dirs"
echo "ZZOK maps $n cities ZZ"

cat > "$WORK/teacher_env.sh" <<EOF
# source this before any teacher rollout / scorer / gate
export DZ_ROOT="$DZ" PYTHONPATH="$DZ/nuplan-devkit"
export NUPLAN_MAPS_ROOT="$MAPS/nuplan-maps-v1.0"
export DRIVERL_EVAL_ROOT="$DZ" DRIVERL_EVAL_NUPLAN_ROOT="$DZ/nuplan-devkit"
export DRIVERL_EVAL_RELEASE_ID=driverl-teacher-u2400
export DRIVERL_EVAL_CONFIG_PATH="$DZ/release/configs/driverl_teacher.yaml"
export DRIVERL_EVAL_CHECKPOINT_PATH="$DZ/release/checkpoints/checkpoint_2400.pt"
export DRIVERL_EVAL_PYTHON="$PY" DRIVERL_EVAL_DEVICE=cuda
unset DRIVERL_EVAL_ROUTE_LANE_RANK
EOF
echo "ZZOK teacher env ready -> source $WORK/teacher_env.sh ZZ"
echo "NEXT: set NUPLAN_DATA_ROOT to where fetch_navtrain_dbs.py wrote the DBs, then run the"
echo "      navtrain smoke (1 log, 2 frames) before any full generation."
