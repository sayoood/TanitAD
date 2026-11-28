#!/usr/bin/env bash
# TRK-SCORE smoke (PREREG_TRKSCORE.md section 8.1): 50 navtest tokens on C1, RED on every token, tracked states kept for CTRL-PARITY.
# Reads controls only (trk_ctrl.py): NO PDMS. GPU under the shared dev-box lock (taken inside trk_run_seam.py).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
TIP=F:/TanitAD_p5cread_tip
PK="$TIP/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"; EV="$PK/eval"
PY_T=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PY_N=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
NAV=F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896
DATA=E:/Projects/TanitAD/data/refe_navtest; YAW=E:/Projects/TanitAD-artifacts/yawrate_wp/tables/ego_yaw_navtest.json.gz
W=E:/Projects/TanitAD-artifacts/trkscore/work/smoke; mkdir -p "$W" E:/Projects/TanitAD-artifacts/trkscore/work/tmp
TOK=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next/propscore/smoke_tokens50.json
export TANITAD_NAVSIM_ROOT=F:/Archive/devbox-C/navsim REFE_NAVTEST_DIR="$DATA" TMP=E:/Projects/TanitAD-artifacts/trkscore/work/tmp TEMP=E:/Projects/TanitAD-artifacts/trkscore/work/tmp
export PYTHONPATH="$TIP/stack;$TIP/taniteval;$EV;$EV/p5c" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
env -u CUDA_VISIBLE_DEVICES "$PY_T" "$HERE/trk_run_seam.py" --ckpt E:/Projects/TanitAD-artifacts/trkscore/arms/C1/model_final.pt \
  --expect-sha256 f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4 --name trk_smoke_c1 --tokens "$TOK" --yaw-table "$YAW" \
  --seam-out "$W/stage2_trk_smoke_c1.npz" --props-out "$W/stage2_trk_smoke_c1.proposals.npz" --dump "$W/trk_smoke_c1.npz" --red-all --keep-sim < /dev/null
echo "seam rc $?"
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="$NAV;$PK/refe;$EV/instruments" "$PY_N" "$HERE/trk_ctrl.py" --trk-dump "$W/trk_smoke_c1.npz" --trk-seam "$W/stage2_trk_smoke_c1.npz" \
  --trk-props "$W/stage2_trk_smoke_c1.proposals.npz" --banked-seam "$DATA/seams/stage2_p5c_c1.npz" --banked-props "$DATA/seams/stage2_p5c_c1.proposals.npz" \
  --red-all --parity --out "$W/ctrl_smoke_c1.json" < /dev/null
echo "ctrl rc $?"
