"""make_fallattr_queue.py -- writes pod_fallattr10k_queue.sh from ../../cropscale/fit/pod_cropscale10k_queue.sh (card ADD106, sha256 prefix below) by asserted, counted text edits (card ADD110, FALLATTR10K).
Kept: the rollout function (asserted equal to the model's except two counted insertions: a finished run with its T1_DONE is not run again, and the time rule is checked before each start), the slot rule, the start marker
with set -C, the DRY=1 mode, the timeout, the checks of the build files, of the overlay tree, of the inputs, the wrapper admission of card ADD102 (hash 28bc31f21ecb4ccb and the pod check record), the check of the banked runs.
Changed: root $FA /workspace/v8proof/fallattr; step 0 gate (exit 4 without the test outputs of the two kits); step 1 the gap files (make_attr_gapfiles.py on the CPU when $FA/gapfiles is absent, stop unless pass_a0 is true);
eight rollouts = the VD5c call of the ADD102 queue (same environment, --vis-closing) with one gap file each, names I10RQ50KF5P30Hb20a<ARM>_s<k> for the arms LA, LN, LM, LF and model seeds 0 and 1; one process (no GR_SEED);
time rule FA_STOP_S (default 10800 s from the first line of FALLATTR10K_START.txt); RESUME=1 continues after a stop marker; markers FALLATTR10K_START.txt, FALLATTR10K_STOP.txt, FALLATTR10K_DONE.txt; log fallattr_queue.log.
Usage: python make_fallattr_queue.py   (run in fallattr/gap/). Refuses to overwrite (VQ_OVERWRITE_OWN=1 rewrites it). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "cropscale", "fit", "pod_cropscale10k_queue.sh"))
OUT = os.path.join(HERE, "pod_fallattr10k_queue.sh")
SRC_SHA_PREFIX = "5c079e6b279e350b"
WRAP_SHA = "28bc31f21ecb4ccb"
ARMS = ("LA", "LN", "LM", "LF")
if hashlib.sha256(io.open(SRC, "rb").read()).hexdigest()[:16] != SRC_SHA_PREFIX:
    sys.exit("REFUSED: the source queue (card ADD106) on disk is not %s" % SRC_SHA_PREFIX)
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
        "# pod_fallattr10k_queue.sh -- card ADD110 (FALLATTR10K): 8 rollouts of the reader-fed form RR with the wrapper variant rzgfhqetjv of card ADD102 (hash %s, unchanged), the arguments of the VD5c call of the ADD102 queue" % WRAP_SHA,
        "# (pre-roll 30 steps = 6 s, readout K, --rule-every 5, --kappa-max 0.2, heading bound 0.2 1/m, --vis-closing), model seeds 0 and 1, with one gap file per arm: LA, LN, LM, LF (the label gap on all label rows / below 20 m /",
        "# from 20 m to below 40 m / 40 m and above, the base file VGAPLE of card ADD106 elsewhere). Run names I10RQ50KF5P30Hb20a<ARM>_s<k>. The gap files are for attribution only; no arm is a candidate.",
        "# Step 0 gate: exit 4 unless $FA/kit_gap/lead_pytest.out and $FA/kit_read/lead_pytest.out exist (control a4: the tests of every kit passed before the first rollout).",
        "# Step 1: when $FA/gapfiles is absent, make_attr_gapfiles.py runs on the CPU; the queue stops unless FALLATTR_GAPFILES_RECORD.json has pass_a0 true and the hashes of the four files and of the evaluation join.",
        "# Step 2: the eight rollouts, one after the other, each started only with fewer than 3 GPU compute processes and more than 28 GB free (memory.max minus anon); a run with its T1_DONE is not run again.",
        "# Time rule: before each start the seconds since the first line of $FA/FALLATTR10K_START.txt must be at most FA_STOP_S (default 10800), else $FA/FALLATTR10K_STOP.txt. A failed rollout writes the stop marker.",
        "# Done marker $FA/FALLATTR10K_DONE.txt. Log $FA/fallattr_queue.log. DRY=1 prints every command and creates nothing. RESUME=1 continues after a stop marker. Nothing is removed or moved.",
        "# Banked runs needed: RR (I10RQ50KF5P30Hb20), RLc (I10RLcKF5P30Hb20), V0 of card ADD102 and GEc (I10RQ50KF5P30Hb20gGEc) of card ADD106, both seeds.",
        "# Written by make_fallattr_queue.py from the queue of card ADD106 (sha256 prefix %s). No training." % SRC_SHA_PREFIX]
text = "\n".join(head) + "\n" + text[j:]
N += 1

# ---- variables
once("GP=${GR_ROOT:-/workspace/v8proof/cropscale}\n", "FA=${FA_ROOT:-/workspace/v8proof/fallattr}\nGP=$FA\n")
once("GF=${GR_GAPFILES:-$GP/gapfiles}\n", "GF=$FA/gapfiles\nKG=$FA/kit_gap\nKR=$FA/kit_read\nBASEF=/workspace/v8proof/cropscale/gapfiles/VGAPLE.npz\nBASER=/workspace/v8proof/cropscale/gapfiles/CROPSCALE_GAPFILES_RECORD.json\n")
once("SM=$GP/small_cs", "SM=$GP/small_fa")
between("SEEDSW=${GR_SEED:-ALL}\n", "MM=${I10_MEMMAX_FILE",
        'STOPS=${FA_STOP_S:-10800}\nRESUME=${RESUME:-0}\nSEEDSW=ALL\nARMS=(0 1); PS=""\nLOG=$GP/fallattr_queue.log\n')
assert "CROPSCALE10K_STOP" in text
once('echo "$(ts) $*" > "$GP/CROPSCALE10K_STOP$PS.txt"', 'echo "$(ts) $*" >> "$GP/FALLATTR10K_STOP.txt"')

# ---- step 0 (gate) and the time rule, before the first check of the pod
GATE = r'''# step 0 gate: the tests of both kits passed in the lead's run (control a4); nothing is created before this point
gate_ok() { [ -f "$KG/lead_pytest.out" ] && [ -f "$KR/lead_pytest.out" ]; }
if [ "$DRY" = 1 ]; then
  echo "GATE: needs $KG/lead_pytest.out and $KR/lead_pytest.out (exit 4 without them)"
else
  gate_ok || { echo "REFUSED: step 0 gate: $KG/lead_pytest.out or $KR/lead_pytest.out is missing"; exit 4; }
fi
check_time() { # step name; seconds since the first line of the start file at most STOPS
  if [ "$DRY" = 1 ]; then echo "STOPRULE before $1: seconds since the first line of $GP/FALLATTR10K_START.txt at most $STOPS"; return 0; fi
  local t0 now
  t0=$(head -n 1 "$GP/FALLATTR10K_START.txt" 2>/dev/null | awk '{print $1}')
  [[ "$t0" =~ ^[0-9]+$ ]] || stop "START-LINE-UNREADABLE before $1"
  now=$(date +%s)
  [ $(( now - t0 )) -le "$STOPS" ] || stop "STOP-RULE $(( now - t0 )) s since the first start line, limit $STOPS s, before $1"
}
'''
once('if [ "$DRY" != 1 ]; then\n  [ -d "$W" ] || { echo "REFUSED: no $W"; exit 2; }\n', GATE + 'if [ "$DRY" != 1 ]; then\n  [ -d "$W" ] || { echo "REFUSED: no $W"; exit 2; }\n')
once('  ( set -C; echo "$(ts) pid $$" > "$GP/START_CROPSCALE10K$PS.txt" ) 2>/dev/null || { echo "REFUSED: $GP/START_CROPSCALE10K$PS.txt exists (a queue was started)"; exit 2; }\n',
     '  if [ "$RESUME" = 1 ] && [ -f "$GP/FALLATTR10K_START.txt" ]; then\n'
     '    echo "$(date +%s) resume $(ts) pid $$" >> "$GP/FALLATTR10K_START.txt"\n'
     '  else\n'
     '    ( set -C; echo "$(date +%s) $(ts) pid $$" > "$GP/FALLATTR10K_START.txt" ) 2>/dev/null || { echo "REFUSED: $GP/FALLATTR10K_START.txt exists (a queue was started; RESUME=1 continues)"; exit 2; }\n'
     '  fi\n')
once('log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD106 (CROPSCALE10K): reader-fed rule RR with the closing input of the run rule and the gap of the head on five frames fit on the large fit set (G2, G3, G4, GEc) and the mean of the 500-clip fits of seeds 2, 3, 4 (E3c), at age"',
     'log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD110 (FALLATTR10K): reader-fed rule RR with the closing input of the run rule and one gap file per arm (LA, LN, LM, LF: the label gap on its range, the base file VGAPLE elsewhere), at age"')

# ---- the rollout function: two counted insertions
once('  done_step "ro_$arm" && return 0\n', '  done_step "ro_$arm" && return 0\n  [ -f "$out/T1_DONE" ] && return 0\n')
once('  wait_slot "rollout $arm"\n', '  check_time "rollout $arm"\n  wait_slot "rollout $arm"\n')

# ---- admission: own function in place of the one of the model
a0 = text.index("admit() {")
a1 = text.index('PYEOF\n}\n', a0) + len('PYEOF\n}\n')
ADMIT = r'''admit() { # RECORD GAPDIR EVALJOIN BASEFILE -> prints why on failure
  "$PY" - "$@" <<'PYEOF'
import hashlib, json, os, sys
rec_path, gdir, ejoin, basef = sys.argv[1:5]
rec = json.load(open(rec_path, encoding="utf-8"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if rec.get("pass_a0") is not True:
    print("A0-NOT-PASS"); sys.exit(1)
if rec.get("eval_join_sha256") != sha(ejoin):
    print("RECORD-IS-FOR-ANOTHER-EVAL-JOIN"); sys.exit(1)
b = rec.get("base") or {}
if b.get("sha256") != sha(basef) or b.get("named_in_record_add106") is not True:
    print("BASE-IS-NOT-THE-RECORDED-ONE"); sys.exit(1)
for a in ("LA", "LN", "LM", "LF"):
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
    echo "CMD: [only when $GF is absent] CUDA_VISIBLE_DEVICES=-1 $PY $KG/make_attr_gapfiles.py --eval-join $JOINE --base $BASEF --base-record $BASER --out-dir $GF"
    echo "CHECK: $GF/FALLATTR_GAPFILES_RECORD.json has pass_a0 true, the sha256 of the evaluation join, of the base file and of each of the four gap files"
    return 0
  fi
  if [ ! -d "$GF" ]; then
    check_time "gap files"
    [ -f "$KG/make_attr_gapfiles.py" ] || stop "NO-GAPFILE-TOOL $KG/make_attr_gapfiles.py"
    [ -f "$BASEF" ] && [ -f "$BASER" ] || stop "NO-BASE-FILE-OR-RECORD"
    CUDA_VISIBLE_DEVICES=-1 "$PY" "$KG/make_attr_gapfiles.py" --eval-join "$JOINE" --base "$BASEF" --base-record "$BASER" --out-dir "$GF" >> "$LOG" 2>&1 || log "gap-file build rc $?"
  fi
  [ -f "$GF/FALLATTR_GAPFILES_RECORD.json" ] || stop "NO-GAPFILES-RECORD"
  local adm
  adm=$(admit "$GF/FALLATTR_GAPFILES_RECORD.json" "$GF" "$JOINE" "$BASEF") || stop "GAPFILE-ADMISSION $adm"
  log "$adm"
}
'''
text = text[:a0] + ADMIT + text[a1:]
N += 1
once('  [ -f "$GF/CROPSCALE_GAPFILES_RECORD.json" ] || stop "NO-GAPFILES-RECORD"\n'
     '  adm=$(admit "$GF/CROPSCALE_GAPFILES_RECORD.json" "$GF" "$JOINE" "VGAPL_s2,VGAPL_s3,VGAPL_s4,VGAPLE,VGAPSE3" "g2,g3") || stop "GAPFILE-ADMISSION $adm"\n'
     '  log "$adm"\n', "")
once('  for nm in I10RQ50KF5P30Hb20vV0 I10RQ50KF5P30Hb20vVD5c I10RQ50KF5P30Hb20fF2 I10RQ50KF5P30Hb20fF3 I10RQ50KF5P30Hb20fF4 I10RQ50KF5P30Hb20eE5c; do\n',
     '  for nm in I10RQ50KF5P30Hb20vV0 I10RQ50KF5P30Hb20gGEc; do\n')
once('stop "NO-BANKED-RUN-OF-ADD102-ADD104-OR-ADD105 ${nm}_s$k"', 'stop "NO-BANKED-RUN-OF-ADD102-OR-ADD106 ${nm}_s$k"')

# ---- the rollouts and the end
r0 = text.index('for k in "${ARMS[@]}"; do\n  RGRID=')
r1 = text.index('if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi')
lines = [x for x in text[r0:r1].splitlines() if "rollout I10RQ50KF5P30Hb20gG2 RR" in x]
assert len(lines) == 1
env0 = lines[0].strip().split(" VGAP=")[0]
assert env0.startswith("RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE=")
blocks = ['for k in "${ARMS[@]}"; do\n  %s VGAP="$GF/VGAPA_%s.npz" VCLOSE=1 rollout I10RQ50KF5P30Hb20a%s RR "$k" "$k"\ndone\n' % (env0, a, a) for a in ARMS]
text = text[:r0] + "step_gapfiles\n" + "".join(blocks) + text[r1:]
N += 1
once('NAMES="I10RQ50KF5P30Hb20gG2 I10RQ50KF5P30Hb20gG3 I10RQ50KF5P30Hb20gG4 I10RQ50KF5P30Hb20gGEc I10RQ50KF5P30Hb20gE3c"', 'NAMES="%s"' % " ".join("I10RQ50KF5P30Hb20a%s" % a for a in ARMS))
once('echo "$(ts) all steps done (seeds $SEEDSW)" > "$GP/CROPSCALE10K_DONE$PS.txt"', 'echo "ZZ FALLATTR10K DONE $(ts) all steps done" > "$GP/FALLATTR10K_DONE.txt"')
once('log "CROPSCALE10K_DONE seeds $SEEDSW"', 'log "FALLATTR10K_DONE"')

# ---- guards
RB = fn_of(text)
assert RB == ROLL_BEFORE.replace('  done_step "ro_$arm" && return 0\n', '  done_step "ro_$arm" && return 0\n  [ -f "$out/T1_DONE" ] && return 0\n').replace('  wait_slot "rollout $arm"\n', '  check_time "rollout $arm"\n  wait_slot "rollout $arm"\n')
chk = text.replace("/workspace/v8proof/cropscale/gapfiles/CROPSCALE_GAPFILES_RECORD.json", "").replace("/workspace/v8proof/cropscale/gapfiles/", "")
for bad in ("CROPSCALE", "ropscale", "GR_SEED", "GR_ROOT", "GR_GAPFILES", "VGAPL_", "VGAPSE3", "VGAPLE.npz\" VCLOSE", "gG2", "gG3", "gG4", "gE3c", "gGEc ", "$PS.txt", "fF2", "eE5c"):
    assert bad not in chk, bad
assert text.count("rollout I10RQ50KF5P30Hb20a") == 4 and text.count("VCLOSE=1") == 4 and text.count("VGAP=") == 4
assert text.count(WRAP_SHA) >= 3 and "VGAPA_LA.npz" in text
for w in ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop"):
    assert w not in text, w
assert "\r" not in text
text = text if text.endswith("\n") else text + "\n"
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d rollouts 8 edits %d (source %s, wrapper hash %s)" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n"), N, SRC_SHA_PREFIX, WRAP_SHA))
