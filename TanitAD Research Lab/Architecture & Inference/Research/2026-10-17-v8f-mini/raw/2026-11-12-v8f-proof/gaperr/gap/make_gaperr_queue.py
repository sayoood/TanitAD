"""make_gaperr_queue.py -- writes pod_gaperr10k_queue.sh from ../../fallattr/gap/pod_fallattr10k_queue.sh (card ADD110, sha256 prefix below) by asserted, counted text edits (card ADD111, GAPERR10K).
Kept: the rollout function (asserted equal to the model's), the slot rule, the start marker with set -C and its first line `<epoch seconds> <ISO time> pid <pid>`, the DRY=1 mode, the timeout, the checks of the build files,
of the overlay tree, of the inputs, the wrapper admission (hash 28bc31f21ecb4ccb and the pod check record), the check of the banked runs, the time rule, RESUME=1.
Changed: pod root $GE /workspace/v8proof/gaperr (kits kit_gap and kit_read, gap files in gapfiles), markers GAPERR10K_START.txt, GAPERR10K_STOP.txt, GAPERR10K_DONE.txt, log gaperr_queue.log, time variable GE_STOP_S
(default 10800), the gap-file step (make_gaperr_gapfiles.py with the LA file of card ADD110; the queue stops unless GAPERR_GAPFILES_RECORD.json has pass_b0 true and the hashes of the join, the base, the LA file and the
four gap files), eight rollouts for the arms NR, BR, SM, SL and model seeds 0 and 1, run folders I10RQ50KF5P30Hb20b<ARM>_s<k>. One added precondition: the banked run LA of card ADD110 (both seeds) must have its T1_DONE.
Usage: python make_gaperr_queue.py   (run in gaperr/gap/). Refuses to overwrite (VQ_OVERWRITE_OWN=1 rewrites it). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "fallattr", "gap", "pod_fallattr10k_queue.sh"))
OUT = os.path.join(HERE, "pod_gaperr10k_queue.sh")
SRC_SHA_PREFIX = "dc6f6c6405092fe3"
WRAP_SHA = "28bc31f21ecb4ccb"
ARMS = ("NR", "BR", "SM", "SL")
if hashlib.sha256(io.open(SRC, "rb").read()).hexdigest()[:16] != SRC_SHA_PREFIX:
    sys.exit("REFUSED: the source queue (card ADD110) on disk is not %s" % SRC_SHA_PREFIX)
raw = io.open(SRC, "rb").read()
if os.path.exists(OUT) and os.environ.get("VQ_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists (VQ_OVERWRITE_OWN=1 rewrites it)" % OUT)
text = raw.decode("utf-8")
assert "\r" not in text
N = 0


def once(a, b, n=1):
    global text, N
    assert text.count(a) == n, (a[:90], text.count(a), n)
    text = text.replace(a, b)
    N += 1


def between(a, b, new):
    """replace the text from the start of a up to the start of b (both once)"""
    global text, N
    assert text.count(a) == 1 and text.count(b) == 1, (a[:60], b[:60])
    i, j = text.index(a), text.index(b)
    assert i < j
    text = text[:i] + new + text[j:]
    N += 1


def fn_of(t):
    i0 = t.index("rollout() {\n")
    return t[i0:t.index("\n}\n", i0) + 3]


ROLL_BEFORE = fn_of(text)

# ---- head
j = text.index("set -u\n")
head = ["#!/usr/bin/env bash",
        "# pod_gaperr10k_queue.sh -- card ADD111 (GAPERR10K): 8 rollouts of the reader-fed form RR with the wrapper variant rzgfhqetjv of card ADD102 (hash %s, unchanged), the arguments of the VD5c call of the ADD102 queue" % WRAP_SHA,
        "# (pre-roll 30 steps = 6 s, readout K, --rule-every 5, --kappa-max 0.2, heading bound 0.2 1/m, --vis-closing), model seeds 0 and 1, with one gap file per arm: NR, BR, SM, SL (label gap plus the slow part /",
        "# label gap plus the fast part / median of the base gap over the trailing run / end value of the line over the trailing run). Run names I10RQ50KF5P30Hb20b<ARM>_s<k>. NR and BR carry label values and are attribution only.",
        "# Step 0 gate: exit 4 unless $GE/kit_gap/lead_pytest.out and $GE/kit_read/lead_pytest.out exist (control b4: the tests of every kit passed before the first rollout).",
        "# Step 1: when $GE/gapfiles is absent, make_gaperr_gapfiles.py runs on the CPU; the queue stops unless GAPERR_GAPFILES_RECORD.json has pass_b0 true and the hashes of the four files, of the LA file and of the evaluation join.",
        "# Step 2: the eight rollouts, one after the other, each started only with fewer than 3 GPU compute processes and more than 28 GB free (memory.max minus anon); a run with its T1_DONE is not run again.",
        "# Time rule: before each start the seconds since the first line of $GE/GAPERR10K_START.txt must be at most GE_STOP_S (default 10800), else $GE/GAPERR10K_STOP.txt. A failed rollout writes the stop marker.",
        "# Done marker $GE/GAPERR10K_DONE.txt. Log $GE/gaperr_queue.log. DRY=1 prints every command and creates nothing. RESUME=1 continues after a stop marker. Nothing is removed or moved.",
        "# Banked runs needed: RR (I10RQ50KF5P30Hb20), RLc (I10RLcKF5P30Hb20), V0 of card ADD102, GEc (I10RQ50KF5P30Hb20gGEc) of card ADD106 and LA (I10RQ50KF5P30Hb20aLA) of card ADD110, both seeds.",
        "# Written by make_gaperr_queue.py from the queue of card ADD110 (sha256 prefix %s). No training." % SRC_SHA_PREFIX]
text = "\n".join(head) + "\n" + text[j:]
N += 1

# ---- variables
once("FA=${FA_ROOT:-/workspace/v8proof/fallattr}\nGP=$FA\n", "GE=${GE_ROOT:-/workspace/v8proof/gaperr}\nGP=$GE\n")
once("GF=$FA/gapfiles\nKG=$FA/kit_gap\nKR=$FA/kit_read\n", "GF=$GE/gapfiles\nKG=$GE/kit_gap\nKR=$GE/kit_read\n")
once("BASER=/workspace/v8proof/cropscale/gapfiles/CROPSCALE_GAPFILES_RECORD.json\n",
     "BASER=/workspace/v8proof/cropscale/gapfiles/CROPSCALE_GAPFILES_RECORD.json\nLAF=/workspace/v8proof/fallattr/gapfiles/VGAPA_LA.npz\n")
once("SM=$GP/small_fa", "SM=$GP/small_ge")
once("STOPS=${FA_STOP_S:-10800}\n", "STOPS=${GE_STOP_S:-10800}\n")
once("LOG=$GP/fallattr_queue.log\n", "LOG=$GP/gaperr_queue.log\n")
once('log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD110 (FALLATTR10K): reader-fed rule RR with the closing input of the run rule and one gap file per arm (LA, LN, LM, LF: the label gap on its range, the base file VGAPLE elsewhere), at age"',
     'log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD111 (GAPERR10K): reader-fed rule RR with the closing input of the run rule and one gap file per arm (NR, BR, SM, SL: label gap plus the slow part, label gap plus the fast part, two smoothing forms of the base gap), at age"')
once("FALLATTR10K", "GAPERR10K", 10)
once("FALLATTR_GAPFILES_RECORD.json", "GAPERR_GAPFILES_RECORD.json", 3)

# ---- the rollout function is unchanged
assert fn_of(text) == ROLL_BEFORE

# ---- admission and the gap-file step: own functions in place of the ones of card ADD110
ADMIT = r'''admit() { # RECORD GAPDIR EVALJOIN BASEFILE LAFILE -> prints why on failure
  "$PY" - "$@" <<'PYEOF'
import hashlib, json, os, sys
rec_path, gdir, ejoin, basef, laf = sys.argv[1:6]
rec = json.load(open(rec_path, encoding="utf-8"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if rec.get("pass_b0") is not True:
    print("B0-NOT-PASS"); sys.exit(1)
if rec.get("eval_join_sha256") != sha(ejoin):
    print("RECORD-IS-FOR-ANOTHER-EVAL-JOIN"); sys.exit(1)
b = rec.get("base") or {}
if b.get("sha256") != sha(basef) or b.get("named_in_record_add106") is not True:
    print("BASE-IS-NOT-THE-RECORDED-ONE"); sys.exit(1)
if not os.path.isfile(laf) or (rec.get("la_file") or {}).get("sha256") != sha(laf):
    print("LA-FILE-IS-NOT-THE-RECORDED-ONE"); sys.exit(1)
for a in ("NR", "BR", "SM", "SL"):
    fn = os.path.join(gdir, "VGAPA_%s.npz" % a)
    if not os.path.isfile(fn):
        print("NO-GAPFILE " + a); sys.exit(1)
    if sha(fn) != (rec.get("files", {}).get(a) or {}).get("sha256"):
        print("GAPFILE-HASH-DIFFERS " + a); sys.exit(1)
print("ADMIT-OK files 4")
PYEOF
}
step_gapfiles() { # step 1: make the gap files when $GF is absent (CPU), then admit them
  if [ "$DRY" = 1 ]; then
    echo "CMD: [only when $GF is absent] CUDA_VISIBLE_DEVICES=-1 $PY $KG/make_gaperr_gapfiles.py --join $JOINE --base $BASEF --base-record $BASER --la-file $LAF --out-dir $GF"
    echo "CHECK: $GF/GAPERR_GAPFILES_RECORD.json has pass_b0 true, the sha256 of the evaluation join, of the base file, of the LA file and of each of the four gap files"
    return 0
  fi
  if [ ! -d "$GF" ]; then
    check_time "gap files"
    [ -f "$KG/make_gaperr_gapfiles.py" ] || stop "NO-GAPFILE-TOOL $KG/make_gaperr_gapfiles.py"
    [ -f "$BASEF" ] && [ -f "$BASER" ] && [ -f "$LAF" ] || stop "NO-BASE-FILE-RECORD-OR-LA-FILE"
    CUDA_VISIBLE_DEVICES=-1 "$PY" "$KG/make_gaperr_gapfiles.py" --join "$JOINE" --base "$BASEF" --base-record "$BASER" --la-file "$LAF" --out-dir "$GF" >> "$LOG" 2>&1 || log "gap-file build rc $?"
  fi
  [ -f "$GF/GAPERR_GAPFILES_RECORD.json" ] || stop "NO-GAPFILES-RECORD"
  local adm
  adm=$(admit "$GF/GAPERR_GAPFILES_RECORD.json" "$GF" "$JOINE" "$BASEF" "$LAF") || stop "GAPFILE-ADMISSION $adm"
  log "$adm"
}
'''
between("admit() {", '[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"', ADMIT)

# ---- banked runs: the LA run of card ADD110 joins the list
once('  for nm in I10RQ50KF5P30Hb20vV0 I10RQ50KF5P30Hb20gGEc; do\n', '  for nm in I10RQ50KF5P30Hb20vV0 I10RQ50KF5P30Hb20gGEc I10RQ50KF5P30Hb20aLA; do\n')
once('stop "NO-BANKED-RUN-OF-ADD102-OR-ADD106 ${nm}_s$k"', 'stop "NO-BANKED-RUN-OF-ADD102-ADD106-OR-ADD110 ${nm}_s$k"')

# ---- the rollouts and the end
r0 = text.index('for k in "${ARMS[@]}"; do\n  RGRID=')
r1 = text.index('if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi')
lines = [x for x in text[r0:r1].splitlines() if "rollout I10RQ50KF5P30Hb20aLA RR" in x]
assert len(lines) == 1
env0 = lines[0].strip().split(" VGAP=")[0]
assert env0.startswith("RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE=")
blocks = ['for k in "${ARMS[@]}"; do\n  %s VGAP="$GF/VGAPA_%s.npz" VCLOSE=1 rollout I10RQ50KF5P30Hb20b%s RR "$k" "$k"\ndone\n' % (env0, a, a) for a in ARMS]
text = text[:r0] + "".join(blocks) + text[r1:]
N += 1
once('NAMES="I10RQ50KF5P30Hb20aLA I10RQ50KF5P30Hb20aLN I10RQ50KF5P30Hb20aLM I10RQ50KF5P30Hb20aLF"', 'NAMES="%s"' % " ".join("I10RQ50KF5P30Hb20b%s" % a for a in ARMS))

# ---- guards
assert fn_of(text) == ROLL_BEFORE
chk = text.replace("/workspace/v8proof/fallattr/gapfiles/VGAPA_LA.npz", "").replace("/workspace/v8proof/cropscale/gapfiles/CROPSCALE_GAPFILES_RECORD.json", "").replace("/workspace/v8proof/cropscale/gapfiles/", "")
chk = chk.replace("I10RQ50KF5P30Hb20aLA", "").replace("card ADD110", "")
for bad in ("FALLATTR", "fallattr", "FA_ROOT", "FA_STOP_S", "$FA", "make_attr_gapfiles", "pass_a0", "Hb20a", "small_fa", "VGAPA_LA", "VGAPA_LN", "VGAPA_LM", "VGAPA_LF", "--eval-join \"$JOINE\" --base"):
    assert bad not in chk, bad
assert text.count("rollout I10RQ50KF5P30Hb20b") == 4 and text.count("VCLOSE=1") == 4 and text.count("VGAP=") == 4
assert text.count(WRAP_SHA) >= 3 and "VGAPA_NR.npz" in text and "VGAPA_SL.npz" in text
assert text.count('echo "$(date +%s) $(ts) pid $$" > "$GP/GAPERR10K_START.txt"') == 1
assert "ZZ GAPERR10K DONE" in text and "GE_STOP_S" in text and "GAPERR10K_STOP.txt" in text and "gaperr_queue.log" in text
for w in ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop"):
    assert w not in text, w
assert "\r" not in text
text = text if text.endswith("\n") else text + "\n"
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d rollouts 8 edits %d (source %s, wrapper hash %s)" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n"), N, SRC_SHA_PREFIX, WRAP_SHA))
