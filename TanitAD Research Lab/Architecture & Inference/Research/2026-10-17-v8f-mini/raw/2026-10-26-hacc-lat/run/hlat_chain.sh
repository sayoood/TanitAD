#!/bin/bash
# PREREG_V8F_HACC_LAT: the whole pipeline as ONE detached chain on the dev-box CPU (no GPU, ZERO training).
#   0  wait for ZZT1K-DONE (another chain finishes there); freeze the code (copy of hlat/code -> hlat/frozen), assert its sha256 pins; verify the inputs
#   1  the suite of the frozen copy; C-T / C-T2 smoke against the ORIGINAL tool (both seeds)
#   2  two lanes (s0, s1), at most 2 python processes at a time, OMP/MKL 8 threads each:
#        bake CA1 / CB1 / CB2 -> C-B2 (train subset) -> C-A1, C-B1 (eval-141, m 10, vs the T1KIN HACC dumps) -> train tick dump (HACC model, m 10, 141 train clips) -> C-T2 -> fit (A: lambda rule + C-A2; B: v*)
#        -> bake A, B -> eval-141 runs: A m10, B m10, A m30, B m30
#   3  RUN_SUMMARY.json (controls + fit + bakes ONLY: the arm runs' outcome rows are NOT opened here)
# Markers in logs/chain.out: ZZHLAT-DONE <n runs>  |  ZZHLAT-STOP <reason>.  Every run asserts the sha256 of v8f_t1_eval.py, v8f_mini.py (and the other pinned files) of the FROZEN tree before and after
# (a mutant or an edit under hlat/code can never reach this chain: it runs from hlat/frozen only).
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
H=/c/Users/Admin/hlat; LOG=$H/logs; FR=$H/frozen; OUT=$H/run; FIT=$H/fit; CK=$H/ckpt; DT=$H/dumps_train; CTL=$H/controls
SC=/c/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad
SF=$SC/sfresh; HB=$SC/hblend
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
T1RUN=/c/Users/Admin/t1k_run
HACCCK=E:/Projects/TanitAD-artifacts/v8f-mini-hacc
mkdir -p "$LOG" "$OUT" "$FIT" "$CK" "$DT" "$CTL" "$H/pt"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG/chain.out"; }
stop() { say "STOP: $*"; echo "ZZHLAT-STOP $*" >> "$LOG/chain.out"; touch "$LOG/STOP"; exit 1; }
m() { cygpath -m "$1"; }
Hm=$(m "$H"); FRm=$(m "$FR"); OUTm=$(m "$OUT"); FITm=$(m "$FIT"); CKm=$(m "$CK"); DTm=$(m "$DT"); CTLm=$(m "$CTL"); SFm=$(m "$SF"); HBm=$(m "$HB")
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONPATH="$FRm/stack;$FRm/taniteval;$FRm/stack/scripts"
GRID=$Hm/inputs/eval_grid_windows.json; JOIN=$Hm/inputs/join_eval141.npz
TJOIN=$HBm/box2/join_train.npz; FCLIPS=$HBm/fit_clips.json; TCACHE=$HBm/cache; TRACKS=$SFm/inputs/native_tracks.pt

pin_check() {   # pin_check <label>: the frozen tree's pinned files and the helper files are byte-identical to the pins taken at launch
  ( cd "$FR" && sha256sum -c "$H/PINS.sha256" --status ) || stop "pin mismatch in the frozen tree ($1)"
  ( cd "$H" && sha256sum -c "$H/PINS_H.sha256" --status ) || stop "pin mismatch in the helper files ($1)"
}
nostop() { [ -e "$LOG/STOP" ] && exit 1; return 0; }

# ------------------------------------------------------------------------------------------------ 0
say "== wait for ZZT1K-DONE"
until grep -q "^ZZT1K-DONE" "$T1RUN/logs/chain.out" 2>/dev/null; do sleep 20; done
say "ZZT1K-DONE seen"
rm -f "$LOG/STOP"
if [ ! -e "$FR" ]; then cp -r "$H/code" "$FR" && find "$FR" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; fi
pin_check "freeze"
say "frozen tree pinned: $(cut -c1-12 "$H/PINS.sha256" | tr '\n' ' ')"
md5chk() { [ "$(md5sum < "$1" | cut -c1-32)" = "$2" ] || stop "md5 $1"; }
md5chk "$GRID" 34ed8ac56365eeff72e42581bde2f5c4
md5chk "$JOIN" 532fcd644c0e0835e847b2f1cee011ad
md5chk "$TJOIN" 9ebc7a48cbca45cc634eab2aab5a3960
md5chk "$FCLIPS" e419b10e3b1234986eb6b0d0a3372b04
md5chk "$TRACKS" 189f4e68913c0163912a509203963f8d
declare -A HSHA=([0]=71de3c6195ce651ee656d26579bd68a0cc5d7370090044140ed634fe10b627c0 [1]=32e4809dc4204e70544bceb5c3c57d54310c387decd3b95ae0a830177616523a)
for s in 0 1; do [ "$(sha256sum < "$HACCCK/M2b_s${s}_hacc.pt" | cut -c1-64)" = "${HSHA[$s]}" ] || stop "HACC checkpoint sha256 s$s"; done
[ "$(ls "$SF/cache/eval/clips" | wc -l)" = 141 ] || stop "eval cache is not 141 clips"
[ "$(ls "$HB/cache/train/clips" | wc -l)" = 141 ] || stop "train cache is not 141 clips"
for s in 0 1; do [ -d "$T1RUN/dump_HACC_s${s}_m10" ] || stop "T1KIN HACC m10 dump s$s missing"; done
say "inputs OK (eval grid/join, train join/fit_clips/tracks, 2 HACC checkpoints, 141+141 clips, the T1KIN dumps)"

# ------------------------------------------------------------------------------------------------ 1
say "== suite of the frozen copy"
( cd "$FR/stack" && "$PY" -m pytest -q -p no:cacheprovider --basetemp "$Hm/pt/suite" tests/test_v8f_t1.py tests/test_v8f_hacc_infer.py tests/test_v8f_hlat_gate.py tests/test_v8f_hlat.py tests/test_v8f_mini.py tests/test_v8f_oacc.py ) > "$LOG/suite.out" 2>&1
tail -1 "$LOG/suite.out" | grep -q " passed" && ! tail -1 "$LOG/suite.out" | grep -q "failed\|error" || stop "suite not green: $(tail -1 "$LOG/suite.out")"
say "suite: $(tail -1 "$LOG/suite.out")"
pin_check "after suite"
for s in 0 1; do
  say "== C-T / C-T2 smoke s$s (original tool vs frozen copy, 48 windows)"
  CODE=$FRm bash "$H/ct_smoke.sh" $s 48 8 ct_s$s > "$LOG/ct_smoke_s$s.out" 2>&1
  tail -1 "$LOG/ct_smoke_s$s.out" | grep -q "^ZZCT-DONE" || stop "C-T smoke s$s: $(tail -2 "$LOG/ct_smoke_s$s.out" | tr '\n' ' ')"
  say "C-T s$s PASS ($(grep -o '"tick0_plan_vs_dump_ol_max_abs_m": [^,]*' "$LOG/ct_smoke_s$s.out"))"
  pin_check "after C-T smoke s$s"
done

# ------------------------------------------------------------------------------------------------ 2
t1run() {   # t1run <ckpt> <tag> <m> <outdir> [extra args]   (eval-141 windows unless the extras say --windows train)
  local ck=$1 tag=$2 mm=$3 od=$4; shift 4
  [ -f "$od/t1v8f_$tag.json" ] && { say "skip $tag (exists)"; return 0; }
  pin_check "before $tag"
  say "start $tag"
  "$PY" "$FRm/stack/scripts/v8f_t1_eval.py" run --ckpt "$ck" --tag "$tag" --tracks "$TRACKS" --out-dir "$od" --replan-steps "$mm" --device cpu "$@" > "$LOG/$tag.out" 2>&1 < /dev/null
  local rc=$?
  say "end $tag rc=$rc"
  pin_check "after $tag"
  { [ $rc -eq 0 ] && [ -f "$od/t1v8f_$tag.json" ]; } || stop "run $tag failed (rc $rc)"
}
EV="--eval-cache $SFm/cache --eval-grid $GRID --eval-join $JOIN"
TR="--windows train --train-cache $TCACHE --train-join $TJOIN --fit-clips $FCLIPS"
hl() { "$PY" "$FRm/stack/scripts/v8f_hlat.py" "$@"; }
bake() {   # bake <seed> <arm>
  local f="$CK/hlat$2_s$1.pt"
  if [ -f "$f" ]; then say "skip bake $2 s$1 (exists)"; return 0; fi
  pin_check "before bake $2 s$1"
  hl bake --seed $1 --arm $2 --out-dir "$CKm" --fit-dir "$FITm" > "$LOG/bake_$2_s$1.out" 2>&1 || stop "bake $2 s$1 failed: $(tail -2 "$LOG/bake_$2_s$1.out" | tr '\n' ' ')"
  say "bake $2 s$1: $(tail -1 "$LOG/bake_$2_s$1.out" | cut -c1-200)"
}
lane() {   # lane <seed>
  local s=$1
  nostop; bake $s CA1; bake $s CB1; bake $s CB2
  # C-B2: v* = +inf on a train subset: every tick's cross plan equals P's cross plan
  nostop; t1run "$CKm/hlatCB2_s$s.pt" CB2_s${s}_train 10 "$CTLm/cb2_s$s" $TR --dump-ticks "$CTLm/cb2_s$s" --max-windows 192
  hl check-b2 --ticks "$CTLm/cb2_s$s/ticks_CB2_s${s}_train.npz" --out "$CTLm/CB2_s$s.json" > "$LOG/check_b2_s$s.out" 2>&1 || stop "C-B2 s$s failed: $(tail -1 "$LOG/check_b2_s$s.out")"
  say "C-B2 s$s PASS: $(tail -1 "$LOG/check_b2_s$s.out")"
  # C-A1 / C-B1: the baked HACC table (gate absent / v* = 0) reproduces the T1KIN HACC m10 dumps to <= 1e-9 m
  nostop; t1run "$CKm/hlatCA1_s$s.pt" CA1_s${s}_m10 10 "$OUTm" $EV --no-analyze
  hl compare-dumps --a "$OUTm/dump_CA1_s${s}_m10" --b "$(m "$T1RUN")/dump_HACC_s${s}_m10" --label C-A1 --tol 1e-9 --out "$CTLm/CA1_s$s.json" > "$LOG/cmp_ca1_s$s.out" 2>&1 || stop "C-A1 s$s failed: $(tail -1 "$LOG/cmp_ca1_s$s.out")"
  say "C-A1 s$s PASS: $(tail -1 "$LOG/cmp_ca1_s$s.out")"
  nostop; t1run "$CKm/hlatCB1_s$s.pt" CB1_s${s}_m10 10 "$OUTm" $EV --no-analyze
  hl compare-dumps --a "$OUTm/dump_CB1_s${s}_m10" --b "$(m "$T1RUN")/dump_HACC_s${s}_m10" --label C-B1 --tol 1e-9 --out "$CTLm/CB1_s$s.json" > "$LOG/cmp_cb1_s$s.out" 2>&1 || stop "C-B1 s$s failed: $(tail -1 "$LOG/cmp_cb1_s$s.out")"
  say "C-B1 s$s PASS: $(tail -1 "$LOG/cmp_cb1_s$s.out")"
  # the train tick samples of the HACC model's own T1-kin loop (m 10) on HBLEND's 141 train clips
  nostop; t1run "$(m "$HACCCK")/M2b_s${s}_hacc.pt" HACC_s${s}_m10_train 10 "$DTm" $TR --dump-ticks "$DTm"
  hl check-t2 --seed $s --ticks "$DTm/ticks_HACC_s${s}_m10_train.npz" --n-chunks 8 --out "$CTLm/CT2_s$s.json" > "$LOG/check_t2_s$s.out" 2>&1 || stop "C-T2 s$s failed: $(tail -1 "$LOG/check_t2_s$s.out")"
  say "C-T2 s$s PASS: $(tail -1 "$LOG/check_t2_s$s.out")"
  # the fits (train tick samples only; C-A2 and the deliberate-regression control inside)
  nostop; pin_check "before fit s$s"
  if [ ! -f "$FIT/hlat_fit_M2b_s$s.json" ]; then
    hl fit --seed $s --ticks "$DTm/ticks_HACC_s${s}_m10_train.npz" --out-dir "$FITm" > "$LOG/fit_s$s.out" 2>&1 || stop "fit s$s failed (C-A2 / regression / KKT / input guard): $(tail -2 "$LOG/fit_s$s.out" | tr '\n' ' ')"
  fi
  say "fit s$s: $(tail -1 "$LOG/fit_s$s.out" | cut -c1-260)"
  nostop; bake $s A; bake $s B
  # the arms on eval-141: m 10 first (the decision rows), then m 30 (beside)
  t1run "$CKm/hlatA_s$s.pt" A_s${s}_m10 10 "$OUTm" $EV
  t1run "$CKm/hlatB_s$s.pt" B_s${s}_m10 10 "$OUTm" $EV
  t1run "$CKm/hlatA_s$s.pt" A_s${s}_m30 30 "$OUTm" $EV
  t1run "$CKm/hlatB_s$s.pt" B_s${s}_m30 30 "$OUTm" $EV
  say "lane s$s done"
}
lane 0 & P0=$!
lane 1 & P1=$!
wait $P0 $P1
[ -e "$LOG/STOP" ] && { say "a lane stopped: no DONE"; exit 1; }
pin_check "end"
"$PY" "$Hm/hlat_summary.py" > "$LOG/summary.out" 2>&1 || stop "summary: controls not all PASS or runs incomplete: $(tail -1 "$LOG/summary.out")"
n=$(ls "$OUT"/t1v8f_*.json 2>/dev/null | wc -l)
say "runs with a json: $n / 12 (4 arm m10, 4 arm m30 and 4 controls)"
echo "ZZHLAT-DONE $n" >> "$LOG/chain.out"
