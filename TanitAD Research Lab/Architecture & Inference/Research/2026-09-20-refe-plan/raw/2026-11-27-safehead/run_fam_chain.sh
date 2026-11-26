#!/bin/bash
# L2 SAFE-HEAD: the four families on the SH picks (PREREG_SAFEHEAD.md 5 M4).  8-pose like-for-like form, the pattern of
# PK/raw/2026-11-20-p5c-next/run_srule_families.sh, on ALL navtest tokens (SMOKE=<n>: a subset of n tokens, plumbing only, no number is read).
# CPU only (navsim-crun venv, <= 2 processes).  Writes markers  <read dir>/fam/fam.DONE | fam.FAILED  -- never a sentinel inside a log.
# usage: SMOKE=50 bash run_fam_chain.sh <read out dir> <exported landed tip root>     then      bash run_fam_chain.sh <read out dir> <tip root>
RD="$1"; TIP="$2"
[ -d "$RD/seams" ] || { echo "no $RD/seams (run safehead_read.py read first)"; exit 2; }
EV="$TIP/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval"
PYT=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PYN=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
SD="$RD/seams"
F="$RD/fam${SMOKE:+_smoke$SMOKE}"; mkdir -p "$F/strategic"
SUB=(); [ -n "$SMOKE" ] && SUB=(--subset "$SMOKE")
step() { echo "ZZFAM_STEP $1 start $(date -u +%H:%MZ)"; local nm=$1; shift; "$@" > "$F/log_$nm.txt" 2>&1; local rc=$?; echo "ZZFAM_STEP $nm rc=$rc $(date -u +%H:%MZ)"; return $rc; }
fail() { echo "$1" > "$F/fam.FAILED"; echo "ZZFAM_FAIL $1"; exit 1; }
step convert "$PYT" "$EV/p5c/p5c_read_prep.py" convert --arm "SH0=$SD/seam_SH0.npz" "SH1=$SD/seam_SH1.npz" "C1pick=$SD/seam_C1pick.npz" "${SUB[@]}" --outdir "$F/plans" || fail convert
step score "$PYT" "$EV/p5c/p5c_read_prep.py" score --plans-dir "$F/plans" --out-dir "$F/pass" --procs 2 --python "$PYN" --no-human || fail score
step families "$PYT" "$EV/p5c/p5c_families.py" --rows "$F/pass/rows_s*.jsonl" --arms SH0 SH1 C1pick --pairs SH0:C1pick SH1:C1pick --out-dir "$F/families" --python "$PYN" --repo "$TIP" || fail families
step tactical "$PYT" "$EV/p5c/p5c_tactical.py" --seam "SH0=$SD/seam_SH0.npz" "SH1=$SD/seam_SH1.npz" "C1pick=$SD/seam_C1pick.npz" ${SMOKE:+--subset-ok} --out "$F/tactical.json" || fail tactical
step strategic_s0 "$PYT" "$EV/instruments/strategic_family.py" read --arm "SH0=$SD/seam_SH0.npz" --base "C1pick=$SD/seam_C1pick.npz" --out "$F/strategic/strategic_SH0.json" || fail strategic_s0
step strategic_s1 "$PYT" "$EV/instruments/strategic_family.py" read --arm "SH1=$SD/seam_SH1.npz" --base "C1pick=$SD/seam_C1pick.npz" --out "$F/strategic/strategic_SH1.json" || fail strategic_s1
date -u +%FT%H:%MZ > "$F/fam.DONE"
echo "ZZFAM_DONE $F"
