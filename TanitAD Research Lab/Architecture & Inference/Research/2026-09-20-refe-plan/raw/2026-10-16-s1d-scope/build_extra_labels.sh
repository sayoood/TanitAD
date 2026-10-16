#!/usr/bin/env bash
# S1d-X2: build aux drivable-area labels for the 1,167 nuPlan trainval logs that are NOT in navtrain (DB-origin path of refe/aux_label_store.py).
# CPU only, dev box, no GPU, no pod.  Resumable by key (re-running skips what is stored).  DONE marker = the LAST line of $OUT/build.log:
#       ZZAUXSTORE_DONE <stored_keys> <failed> new=<n> skipped=<n>        (a trailing " ABORTED" = free RAM fell under 2.5 GB: just re-run)
# ⛔ RUN THIS ONLY AFTER the S1d dose arm reads row L1 (PREREG_S1D_DRAFT.md): labels without pixels train nothing (the pixels for these logs are on no disk we hold).
# Env (all optional):  OUTROOT (default D:/Projects/TanitAD-artifacts/aux_drivable_extra)  WORKERS (default 1; >1 needs >= 6 GB free RAM, the builder refuses below)
#                      STRIDE (default 60 = navtrain's temporal density)  LIMIT_LOGS (default 0 = all 1,167; use 3 for a smoke test)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PK="${PK:-D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan}"   # a tree holding refe/aux_label_store.py at the S1d-scope commit
OUTROOT="${OUTROOT:-D:/Projects/TanitAD-artifacts/aux_drivable_extra}"
WORKERS="${WORKERS:-1}"; STRIDE="${STRIDE:-60}"; LIMIT_LOGS="${LIMIT_LOGS:-0}"
PY_TAN="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"          # make_extra_yaml.py (sqlite only)
PY_EVAL="C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"    # the builder (nuplan-devkit for the map API)
DBROOT="D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"; MAPS="D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
NAVTEST="D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json"
INV="${INV:-$HERE/inventory_db.json}"
mkdir -p "$OUTROOT/tmp"; export PYTHONIOENCODING=utf-8 TMP="$OUTROOT/tmp" TEMP="$OUTROOT/tmp"
LOG="$OUTROOT/build.log"; : > "$OUTROOT/empty_bank.jsonl"
{
  echo "S1D_EXTRA_BUILD start $(date '+%F %T') workers=$WORKERS stride=$STRIDE limit_logs=$LIMIT_LOGS out=$OUTROOT"
  LIM=(); [ "$LIMIT_LOGS" != "0" ] && LIM=(--limit "$LIMIT_LOGS")
  "$PY_TAN" -I "$HERE/make_extra_yaml.py" --inventory "$INV" --out "$OUTROOT/extra.yaml" --stride "$STRIDE" "${LIM[@]}" || { echo "S1D_EXTRA_BUILD_FAILED yaml"; exit 2; }
  cd "$PK" || { echo "S1D_EXTRA_BUILD_FAILED no PK"; exit 2; }
  "$PY_EVAL" -u refe/aux_label_store.py logmap --bank "$OUTROOT/empty_bank.jsonl" --db-root "$DBROOT" --maps-root "$MAPS" --yaml "$OUTROOT/extra.yaml" --out "$OUTROOT/log_map.json" \
     || { echo "S1D_EXTRA_BUILD_FAILED logmap"; exit 2; }
  "$PY_EVAL" -u refe/aux_label_store.py build --bank "$OUTROOT/empty_bank.jsonl" --db-root "$DBROOT" --maps-root "$MAPS" --yaml "$OUTROOT/extra.yaml" \
     --navtest "$NAVTEST" --logmap "$OUTROOT/log_map.json" --out "$OUTROOT/label_store" --workers "$WORKERS"
} >> "$LOG" 2>&1
tail -1 "$LOG"
