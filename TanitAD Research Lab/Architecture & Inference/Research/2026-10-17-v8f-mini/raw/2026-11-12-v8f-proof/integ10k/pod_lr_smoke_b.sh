#!/bin/bash
# pod_lr_smoke_b.sh -- smoke of the study "longroll", second build (v8f_t1_eval_reader_rule_rzgfhq.py: R1 first-tick adjustment, R2 segments, R3 tick-recorder frames; make_longroll_grid.py;
# check_preroll_b.py). Run by the lead on the pod; nothing here is run on the dev box.
# THE LEAD COPIES INTO THE BUILD FOLDER $B (= $W/build, where the queue of card CYCLEBOUND10K keeps its code) BEFORE RUNNING:
#   v8f_t1_eval_reader_rule_rzgfh.py (parent), v8f_t1_eval_reader_rule_rzgfhq.py (the final wrapper), check_preroll_b.py, make_longroll_grid.py, cmp_dumps.py,
#   dist_rule.py (the real progress geometry that check_preroll_b.py loads from its own folder), and the distrule files the parent already needs.
#   Also in place: the overlay tree $B/tree_ov; the checkpoint $W/ckpt/UE_s0/A/ckpt_last.pt; the reader $W/lr16/TK16_s0/ckpt_last.pt and the map QMAP_FILE (for form RR).
# Prints no outcome number: return codes, file counts, array shapes and keys, row counts, the preroll_log keys, the compare lines of cmp_dumps.py. s3 additionally prints the largest |a_m0| and the
# largest position offset of the carry-identity run (expected 0 or numeric noise; diagnostics of the instrument, not a result).
#   s1 check_preroll_b.py   s2 parent and new wrapper with P = 0, R0, readout pinned, kappa 0.2: dumps equal
#   s3 carry identity: P = 30 executor logged nav logged against P = 0 nav logged (compare line printed) + the a_m0 and offset lines of its preroll log
#   s4 P = 30 and P = 60 (executor plan: two segments), forms R0 RLc RR, readout K, rule-every 5 (not for R0), kappa 0.2: shapes and row counts against P = 0 on the same restricted mini grid;
#      keys and shapes of preroll_offsets_<run>.npz (with a_m0_first_tick and seg_*) and of the tick file against the P = 0 run
# Folder: $W/lr_smoke_b (refused if it exists). Markers LRSMOKEB_DONE.txt / LRSMOKEB_STOP.txt. Uses no rm, no pkill, no pgrep; writes only inside the folder.
# Usage: QMAP_FILE=<qmap json> bash pod_lr_smoke_b.sh      (DRY=1 prints the commands and runs nothing)
set -u
W=${I10_W:-/workspace/v8proof/integ10k}
B=$W/build
PY=${I10_PY:-/workspace/venv/bin/python}
TREE=$B/tree_ov
V=/workspace/v8proof
TRACKS=$V/data/inputs/native_tracks.pt
GRID=$V/hierrep/pins/eval_grid_windows.json
JOINE=$V/hierrep/pins/join_eval141.npz
EVALC=$W/eval_cache
CKPT=$W/ckpt/UE_s0/A/ckpt_last.pt
READER=$W/lr16/TK16_s0/ckpt_last.pt
QMAP=${QMAP_FILE:-}
DEV=${LR_DEVICE:-cuda}
TMO=${LR_TIMEOUT:-1800}
DRY=${DRY:-0}
S=$W/lr_smoke_b
PAR=$B/v8f_t1_eval_reader_rule_rzgfh.py
NEW=$B/v8f_t1_eval_reader_rule_rzgfhq.py

say() { echo "$*"; [ "$DRY" = 1 ] || [ ! -d "$S" ] || echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$S/lr_smoke_b.log"; }
stop() { say "ZZ LRSMOKEB STOP $*"; [ "$DRY" = 1 ] || { [ -d "$S" ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" > "$S/LRSMOKEB_STOP.txt"; }; exit 3; }

if [ "$DRY" != 1 ]; then
  [ -e "$S" ] && { echo "ZZ LRSMOKEB STOP OUTPUT-EXISTS $S"; exit 3; }
  mkdir -p "$S" || { echo "ZZ LRSMOKEB STOP CANNOT-MKDIR $S"; exit 3; }
fi
for f in "$PAR" "$NEW" "$B/check_preroll_b.py" "$B/make_longroll_grid.py" "$B/cmp_dumps.py" "$B/dist_rule.py" "$TRACKS" "$GRID" "$JOINE" "$CKPT" "$READER"; do
  [ "$DRY" = 1 ] || [ -f "$f" ] || stop "MISSING-INPUT $f"
done
if [ "$DRY" != 1 ]; then { [ -n "$QMAP" ] && [ -f "$QMAP" ]; } || stop "QMAP_FILE-NOT-SET-OR-MISSING"; fi

# ---------------------------------------------------------------- s1 the check
say "s1 check_preroll_b.py"
if [ "$DRY" = 1 ]; then echo "CMD: CUDA_VISIBLE_DEVICES=-1 $PY $B/check_preroll_b.py $NEW > $S/CHECK_PREROLL_B.txt"; else
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 "$PY" "$B/check_preroll_b.py" "$NEW" > "$S/CHECK_PREROLL_B.txt" 2>&1
  tail -n 1 "$S/CHECK_PREROLL_B.txt"
  tail -n 1 "$S/CHECK_PREROLL_B.txt" | grep -q "ZZ CHECK-PREROLL-B PASS" || stop "S1-CHECK-FAILED"
fi

# ---------------------------------------------------------------- mini grids: the first two clips of the pinned grid, then the restrictions
mk_mini() {
  "$PY" - "$GRID" "$S/grid_mini.json" <<'PYEOF'
import json, sys
g = json.load(open(sys.argv[1], encoding="utf-8"))
o = g["order"][:2]
out = dict(g)
out["order"] = o
out["windows"] = {c: g["windows"][c] for c in o}
out["n_windows"] = sum(len(out["windows"][c]) for c in o)
with open(sys.argv[2], "x", encoding="utf-8") as fh:
    json.dump(out, fh)
print("mini grid", sys.argv[2], "clips", len(o), "windows", out["n_windows"])
PYEOF
}
if [ "$DRY" = 1 ]; then echo "CMD: mk_mini ($GRID -> $S/grid_mini.json)"; else
  mk_mini || stop "MINI-GRID-FAILED"
  for P in 30 60; do
    "$PY" "$B/make_longroll_grid.py" --grid "$S/grid_mini.json" --preroll-steps $P --out "$S/grid_mini_p$P.json" || stop "RESTRICT-FAILED-P$P"
  done
fi

# lr_run NAME WRAPPER FORM NAV GRIDFILE [wrapper args ...]   -> rc in $LR_RC, output in $S/NAME
lr_run() {
  local name=$1 wrap=$2 form=$3 nav=$4 grid=$5; shift 5
  local out=$S/$name tag=${name}_m10_causal_heads
  local extra=("$@")
  case "$form" in RR) extra+=(--reader "$READER" --qmap "$QMAP") ;; esac
  [ "$form" = R0 ] || extra+=(--rule-log "$out/ticks/rule_log_$name.npz")
  if [ "$DRY" = 1 ]; then
    echo "CMD: DISTRULE_DIR=$B PYTHONPATH=$TREE/stack timeout $TMO $PY $wrap --tree $TREE --form $form ${extra[*]} run --ckpt $CKPT --tag $tag --tracks $TRACKS --eval-cache $EVALC --eval-grid $grid --eval-join $JOINE --out-dir $out --replan-steps 10 --nav $nav --scene logged --schedule periodic --limit causal --dump-ticks $out/ticks --dump-heads --no-analyze --device $DEV >> $out/t1.log"
    LR_RC=0; return 0
  fi
  [ -e "$out" ] && stop "OUTPUT-EXISTS $out"
  mkdir -p "$out/ticks"
  DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" timeout "$TMO" "$PY" "$wrap" --tree "$TREE" --form "$form" "${extra[@]}" run --ckpt "$CKPT" --tag "$tag" --tracks "$TRACKS" --eval-cache "$EVALC" --eval-grid "$grid" --eval-join "$JOINE" --out-dir "$out" --replan-steps 10 --nav "$nav" --scene logged --schedule periodic --limit causal --dump-ticks "$out/ticks" --dump-heads --no-analyze --device "$DEV" >> "$out/t1.log" 2>&1
  LR_RC=$?
  say "RUN $name wrapper=$(basename "$wrap") form=$form nav=$nav grid=$(basename "$grid") rc=$LR_RC"
  [ "$LR_RC" = 0 ] || stop "RUN-FAILED $name rc=$LR_RC (see $out/t1.log)"
}
dumpdir() { echo "$S/$1/dump_$1_m10_causal_heads"; }
cmp_line() { # DIR_A DIR_B -> the compare line of cmp_dumps.py
  [ "$DRY" = 1 ] && { echo "CMD: $PY $B/cmp_dumps.py $1 $2"; return 0; }
  "$PY" "$B/cmp_dumps.py" "$1" "$2" 2>&1 | grep "^CMP" | tail -n 1
}

# ---------------------------------------------------------------- s2 P = 0: parent and new wrapper, R0, readout pinned, kappa 0.2
say "s2 P = 0 identity"
RO=(--readout pinned --kappa-max 0.2)
lr_run S2_parent "$PAR" R0 deadreckon "$S/grid_mini.json" "${RO[@]}" --readout-log "$S/S2_parent/ticks/readout_log_S2_parent.json"
lr_run S2_new0 "$NEW" R0 deadreckon "$S/grid_mini.json" "${RO[@]}" --readout-log "$S/S2_new0/ticks/readout_log_S2_new0.json" --preroll-steps 0
L=$(cmp_line "$(dumpdir S2_parent)" "$(dumpdir S2_new0)"); say "S2 $L"
if [ "$DRY" != 1 ]; then
  echo "$L" | grep -q "differing=0 " || echo "$L" | grep -q "differing=0$" || stop "S2-DUMPS-DIFFER $L"
  say "S2 files in ticks: parent $(ls "$S/S2_parent/ticks" | wc -l) new0 $(ls "$S/S2_new0/ticks" | wc -l); pre-roll files in new0: $(ls "$S/S2_new0/ticks" | grep -c 'preroll\|_pre_')"
fi

# ---------------------------------------------------------------- s3 carry identity: P = 30, executor logged, nav logged, against P = 0 nav logged on the P = 30 restricted grid
say "s3 carry identity"
lr_run S3_p0 "$NEW" R0 logged "$S/grid_mini_p30.json" "${RO[@]}" --readout-log "$S/S3_p0/ticks/readout_log_S3_p0.json"
lr_run S3_p30l "$NEW" R0 logged "$S/grid_mini_p30.json" "${RO[@]}" --readout-log "$S/S3_p30l/ticks/readout_log_S3_p30l.json" --preroll-steps 30 --preroll-exec logged
L=$(cmp_line "$(dumpdir S3_p0)" "$(dumpdir S3_p30l)"); say "S3 $L (numeric noise of recomputed kinematics expected; not judged here)"
if [ "$DRY" = 1 ]; then echo "CMD: s3 report from $S/S3_p30l/ticks/preroll_log_S3_p30l.json (largest |a_m0|, largest position offset)"; else
  "$PY" - "$S/S3_p30l/ticks/preroll_log_S3_p30l.json" <<'PYEOF'
import json, sys
j = json.load(open(sys.argv[1]))
a, o = j.get("a_m0", {}), j.get("offsets_at_t0", {})
for kind in sorted(a):
    print("S3 a_m0 kind=%s main_max_abs=%s segments_max_abs=%s main_exact_zero=%s" % (kind, a[kind].get("main_max_abs"), a[kind].get("segments_max_abs"), a[kind].get("main_exact_zero")))
for kind in sorted(o):
    print("S3 position offset at t0 kind=%s dpos_max_m=%s dpsi_max_deg=%s" % (kind, o[kind]["dpos"]["max"], o[kind]["dpsi_deg"]["max"]))
print("S3 a_m0_used_in_rule:", j.get("a_m0_used_in_rule"))
PYEOF
fi

# ---------------------------------------------------------------- s4 shapes and counts, forms R0 RLc RR, P = 30 and 60 (executor plan), readout K
summ() { # DIR_REF DIR_PRE : file counts, shapes and row counts of the logs, nothing else
  [ "$DRY" = 1 ] && { echo "CMD: summ $1 $2"; return 0; }
  "$PY" - "$1" "$2" <<'PYEOF'
import glob, json, os, re, sys
import numpy as np
ref, pre = sys.argv[1], sys.argv[2]
nr, npre = os.path.basename(ref), os.path.basename(pre)
def gen(f, n):
    return f.replace(n, "<RUN>")
def shapes(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: tuple(z[k].shape) for k in z.files}
def listing(d, n):
    t = os.path.join(d, "ticks")
    return {gen(f, n): os.path.join(t, f) for f in sorted(os.listdir(t))} if os.path.isdir(t) else {}
a, b = listing(ref, nr), listing(pre, npre)
main_b = {k: v for k, v in b.items() if not (k.startswith("rule_log_pre_") or k.startswith("preroll_"))}
print("  ticks files: ref %d, pre-roll run %d (of which pre-roll files %d)" % (len(a), len(b), len(b) - len(main_b)))
print("  same file names as the reference (main-phase files):", sorted(a) == sorted(main_b))
for k in sorted(a):
    if k not in main_b:
        print("  MISSING in pre-roll run:", k); continue
    if k.endswith(".npz"):
        sa, sb = shapes(a[k]), shapes(main_b[k])
        diff = sorted(x for x in set(sa) | set(sb) if sa.get(x) != sb.get(x))
        n = {x: sa[x][0] for x in ("ci", "plan", "tau") if x in sa and sa[x]}
        print("  %-34s shapes equal: %-5s rows(ref)=%s%s" % (k, not diff, n, ("  DIFF " + str(diff[:4])) if diff else ""))
    elif k.endswith(".json"):
        ja, jb = json.load(open(a[k])), json.load(open(main_b[k]))
        print("  %-34s json keys equal: %s" % (k, sorted(ja) == sorted(jb)))
for k in sorted(a):
    if k.endswith(".npz") and k in main_b:
        sa, sb = shapes(a[k]), shapes(main_b[k])
        if "pose_w0" in sa:
            print("  tick file %s: keys equal to the P = 0 run: %s; keys/shapes (P = 0 run) %s" % (k, sorted(sa) == sorted(sb), {x: sa[x] for x in sorted(sa)}))
            print("  tick file %s: shapes equal: %s; keys/shapes (pre-roll run) %s" % (k, sa == sb, {x: sb[x] for x in sorted(sb)}))
da, db = [len(glob.glob(os.path.join(d, "dump_*", "ep*.npz"))) for d in (ref, pre)]
print("  dump files: ref %d, pre-roll run %d" % (da, db))
for k, p in sorted(b.items()):
    if k.startswith("rule_log_pre_") and k.endswith(".npz"):
        s = shapes(p); print("  %-34s rows=%d fields=%d" % (k, s["ci"][0] if "ci" in s else -1, len(s)))
    if k.startswith("preroll_offsets_"):
        s = shapes(p); print("  %-34s keys/shapes=%s" % (k, {x: s[x] for x in sorted(s)}))
    if k.startswith("preroll_log_"):
        j = json.load(open(p)); print("  %-34s keys=%s" % (k, sorted(j)))
        print("    windows=%s ticks_pre_rolled=%s calls=%s" % (j.get("windows"), j.get("ticks_pre_rolled"), j.get("rollout_calls")))
PYEOF
}
for P in 30 60; do
  for form in R0 RLc RR; do
    K=(--readout causal_g --acc-gain 0 --path-exec --kappa-max 0.2)
    [ "$form" = R0 ] || K+=(--rule-every 5)
    r0=S4_${form}_P0_g$P; rp=S4_${form}_P$P
    lr_run "$r0" "$NEW" "$form" deadreckon "$S/grid_mini_p$P.json" "${K[@]}" --readout-log "$S/$r0/ticks/readout_log_$r0.json"
    lr_run "$rp" "$NEW" "$form" deadreckon "$S/grid_mini_p$P.json" "${K[@]}" --readout-log "$S/$rp/ticks/readout_log_$rp.json" --preroll-steps $P
    say "S4 form $form P $P: rc ref/pre-roll = ok/ok"
    [ "$DRY" = 1 ] || summ "$S/$r0" "$S/$rp"
  done
done

if [ "$DRY" = 1 ]; then echo "ZZ LRSMOKEB DONE dry run"; exit 0; fi
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) all steps ran" > "$S/LRSMOKEB_DONE.txt"
say "ZZ LRSMOKEB DONE s1 s2 s3 s4 ran, nothing deleted"
