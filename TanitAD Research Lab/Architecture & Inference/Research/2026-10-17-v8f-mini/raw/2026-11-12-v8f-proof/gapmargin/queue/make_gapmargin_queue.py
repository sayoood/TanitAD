#!/usr/bin/env python
"""make_gapmargin_queue.py -- card PREREG_V8F_PROOF_ADD117 (GAPMARGIN10K). Writes pod_gapmargin10k_queue.sh from the queue of card ADD115
(presattrc/queue/pod_presattrc10k_queue.sh, sha256 prefix 033b776d6338ee3e) by asserted substitutions. The rollout function, the slot wait, the stop rule and
the marker line formats are kept as they are. Step 1 checks the base gap file and makes the two scaled gap files (CPU) when their folder is absent; step 2 runs
eight rollouts of the candidate form with a scaled gap file: A90 and A80 at readout K with no pre-roll, then at the convention.

  python make_gapmargin_queue.py [--src FILE] [--out FILE] [--print]
"""
import argparse
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "presattrc", "queue", "pod_presattrc10k_queue.sh"))
SRC_SHA16 = "033b776d6338ee3e"
QM = '"$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"'
K = "READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5"
C = "READOUT=pinned KAPPA=0.2"
RUNS = [("%s QMAP_FILE=%s VGAP=\"$GF/VGAPLE_A90.npz\" VCLOSE=1" % (K, QM), "I10RQ50KF5Hb20gA90"),
        ("%s QMAP_FILE=%s VGAP=\"$GF/VGAPLE_A80.npz\" VCLOSE=1" % (K, QM), "I10RQ50KF5Hb20gA80"),
        ("%s QMAP_FILE=%s VGAP=\"$GF/VGAPLE_A90.npz\" VCLOSE=1" % (C, QM), "I10RQ50Hb20gA90"),
        ("%s QMAP_FILE=%s VGAP=\"$GF/VGAPLE_A80.npz\" VCLOSE=1" % (C, QM), "I10RQ50Hb20gA80")]
REFS = ["I10RLcKF5Hb20", "I10RQ50KF5Hb20", "I10RQ50KF5Hb20gGEc", "I10RLcHb20", "I10RQ50Hb20", "I10RQ50Hb20gGEc", "I10R0Hb20"]
TAG = "ZZ MAKE-GAPMARGIN-QUEUE STOP: "

HEADER = [
    "# pod_gapmargin10k_queue.sh -- card ADD117 (GAPMARGIN10K): 8 rollouts, model seeds 0 and 1, through the wrapper variant rzgfhqetjv of card ADD102 (hash 28bc31f21ecb4ccb, unchanged):",
    "# the candidate form (reader-fed form RR with --vis-gap-file and --vis-closing, no label at inference) with the camera gap file of card ADD106 scaled by 0.90 (A90) and by 0.80 (A80),",
    "# at readout K with no pre-roll (I10RQ50KF5Hb20gA90, ...gA80; --rule-every 5) and at the convention (I10RQ50Hb20gA90, ...gA80; pinned readout, the rule every 10 steps), pinned grid.",
    "# A margin on the gap is a parameter of the rule's input: a study arm, measured here; its adoption is a design decision.",
    "# Step 0 gate: exit 4 unless $GE/lead_tests/lead_pytest_queue.out exists (the test of this kit passed in the lead's run before the queue start).",
    "# Step 1: the base gap file (sha256 prefix 55de2f7c) is checked; when $GE/gapfiles is absent make_margin_gapfiles.py runs on the CPU;",
    "# the queue stops unless GAPMARGIN_GAPFILES_RECORD.json has pass_k0 true and the hashes of the base file and of both scaled files.",
    "# Step 2: the eight rollouts, one after the other, each started only with fewer than 3 GPU compute processes and more than 28 GB free (memory.max minus anon); a run with its T1_DONE is not run again.",
    "# Time rule: before each start the seconds since the first line of $GE/GAPMARGIN10K_START.txt must be at most GM_STOP_S (default 10800), else $GE/GAPMARGIN10K_STOP.txt. A failed rollout writes the stop marker.",
    "# Done marker $GE/GAPMARGIN10K_DONE.txt. Log $GE/gapmargin_queue.log. DRY=1 prints every command and creates nothing. RESUME=1 continues after a stop marker. Nothing is removed or moved.",
    "# Banked runs needed (both seeds, T1_DONE): " + " ".join(REFS) + ". Written by make_gapmargin_queue.py from the queue of card ADD115 (sha256 prefix %s). No training." % SRC_SHA16,
]

STEP = r'''step_inputs() { # step 1: the base gap file as it is, and the two scaled gap files (made on the CPU when $GF is absent)
  if [ "$DRY" = 1 ]; then
    echo "CHECK: sha256 of $BASEF begins 55de2f7c"
    echo "CMD: [only when $GF is absent] CUDA_VISIBLE_DEVICES=-1 $PY $KQ/make_margin_gapfiles.py --base $BASEF --out-dir $GF --scales 0.90,0.80"
    echo "CHECK: $GMREC has pass_k0 true and the sha256 of the base file and of both scaled files"
    return 0
  fi
  [ -f "$BASEF" ] || stop "NO-BASE-GAP-FILE $BASEF"
  local got; got=$(sha256sum "$BASEF" | cut -c1-8); [ "$got" = "55de2f7c" ] || stop "BASE-GAP-FILE-SHA $got"
  if [ ! -d "$GF" ]; then
    check_time "scaled gap files"
    [ -f "$KQ/make_margin_gapfiles.py" ] || stop "NO-GAP-TOOL $KQ/make_margin_gapfiles.py"
    CUDA_VISIBLE_DEVICES=-1 "$PY" "$KQ/make_margin_gapfiles.py" --base "$BASEF" --out-dir "$GF" --scales 0.90,0.80 >> "$LOG" 2>&1 || log "gap file build rc $?"
  fi
  [ -f "$GMREC" ] || stop "NO-GAP-RECORD"
  local adm
  adm=$("$PY" - "$GMREC" "$BASEF" "$GF" <<'PYEOF'
import hashlib, json, os, sys
rec_path, basef, gf = sys.argv[1:4]
rec = json.load(open(rec_path, encoding="utf-8"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if rec.get("pass_k0") is not True:
    print("K0-NOT-PASS"); sys.exit(1)
if (rec.get("base") or {}).get("sha256") != sha(basef):
    print("BASE-IS-NOT-THE-RECORDED-ONE"); sys.exit(1)
for nm in ("VGAPLE_A90", "VGAPLE_A80"):
    p = os.path.join(gf, nm + ".npz")
    if not os.path.isfile(p) or ((rec.get("files") or {}).get(nm) or {}).get("sha256") != sha(p):
        print("SCALED-FILE-IS-NOT-THE-RECORDED-ONE " + nm); sys.exit(1)
print("ADMIT-OK gap files")
PYEOF
) || stop "INPUT-ADMISSION $adm"
  log "$adm"
}
'''


def build(src_text):
    state = {"t": src_text.replace("\r\n", "\n")}

    def once(a, b):
        if state["t"].count(a) != 1:
            raise SystemExit(TAG + "expected once, found %d: %r" % (state["t"].count(a), a[:90]))
        state["t"] = state["t"].replace(a, b)

    def span(a, b, new):
        s = state["t"]
        if s.count(a) != 1 or s.count(b) != 1 or not s.index(a) < s.index(b):
            raise SystemExit(TAG + "anchors: %r (%d) / %r (%d)" % (a[:60], s.count(a), b[:60], s.count(b)))
        state["t"] = s[:s.index(a)] + new + s[s.index(b):]

    def line_from(a, new):
        s = state["t"]
        if s.count(a) != 1:
            raise SystemExit(TAG + "line anchor found %d times: %r" % (s.count(a), a[:60]))
        i = s.index(a)
        state["t"] = s[:i] + new + s[s.index("\n", i):]

    lines = state["t"].split("\n")
    if not (lines[0] == "#!/usr/bin/env bash" and lines[12] == "set -u" and all(x.startswith("#") for x in lines[1:12])):
        raise SystemExit(TAG + "the head of the source is not the expected one")
    state["t"] = "\n".join([lines[0]] + HEADER + lines[12:])
    once("GE=${PC_ROOT:-/workspace/v8proof/presattrc}", "GE=${GM_ROOT:-/workspace/v8proof/gapmargin}")
    once("GF=$GE/gapfiles\n", "GF=$GE/gapfiles\nGMREC=$GF/GAPMARGIN_GAPFILES_RECORD.json\n")
    once("KQ=${PC_KIT:-$GE/kit_queue}", "KQ=${GM_KIT:-$GE/kit_queue}")
    once("PJ=${PC_JOINS:-/workspace/v8proof/presattr/joins}\n", "")
    once("SM=$GP/small_pc", "SM=$GP/small_gm")
    once("STOPS=${PC_STOP_S:-7200}", "STOPS=${GM_STOP_S:-10800}")
    once("LOG=$GP/presattrc_queue.log", "LOG=$GP/gapmargin_queue.log")
    line_from('log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD115',
              'log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD117 (GAPMARGIN10K): the candidate form with the camera gap scaled by 0.90 and by 0.80, '
              'at readout K with no pre-roll and at the convention, model seeds ${ARMS[*]}"')
    span("step_inputs() {", '[ "$DRY" = 1 ] || [ -f "$GRID" ] || stop "NO-PINNED-GRID"', STEP)
    line_from("  for nm in I10RLcHb20 ", "  for nm in %s; do" % " ".join(REFS))
    blocks = ['for k in "${ARMS[@]}"; do\n  %s rollout %s RR "$k" "$k"\ndone\n' % (env, name) for env, name in RUNS]
    span('for k in "${ARMS[@]}"; do\n  READOUT=pinned KAPPA=0.2 QMAP_FILE=', 'if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi', "".join(blocks))
    once('NAMES="I10RQ50Hb20aLA I10RLcHb20pLE"', 'NAMES="%s"' % " ".join(n for _, n in RUNS))
    t = state["t"].replace("PRESATTRC10K", "GAPMARGIN10K")
    for bad in ("PC_ROOT", "PC_KIT", "PC_STOP_S", "PC_JOINS", "small_pc", "presattrc", "PRESATTR", "JLE", "$PJ", "RJOIN=\"", "aLA RR", "pLE RLc", "ADD114"):
        if bad in t:
            raise SystemExit(TAG + "left over: " + bad)
    if t.count(" rollout I10") != 4 or t.count("VCLOSE=1 rollout") != 4 or t.count("EVERY=5 QMAP_FILE") != 2 or t.count("GAPMARGIN10K_START.txt") < 4:
        raise SystemExit(TAG + "counts")
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=os.path.join(HERE, "pod_gapmargin10k_queue.sh"))
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    raw = io.open(a.src, "rb").read().replace(b"\r\n", b"\n")
    got = hashlib.sha256(raw).hexdigest()[:16]
    if got != SRC_SHA16:
        raise SystemExit(TAG + "source sha256 prefix %s, expected %s" % (got, SRC_SHA16))
    t = build(raw.decode("utf-8"))
    if a.print:
        sys.stdout.write(t)
        return 0
    with io.open(a.out, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(t)
    print("ZZ MAKE-GAPMARGIN-QUEUE DONE %s sha256 prefix %s" % (os.path.basename(a.out), hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
