#!/usr/bin/env python
"""make_boxfuse_queue.py -- card PREREG_V8F_PROOF_ADD131 (BOXFUSE10K). Writes pod_boxfuse10k_queue.sh from the queue of card ADD115
(presattrc/queue/pod_presattrc10k_queue.sh, sha256 prefix 033b776d6338ee3e) by asserted substitutions, the technique of make_gapmargin_queue.py (card ADD117).
The rollout function, the slot wait, the memory checks, the DRY mode and the marker line formats are kept as they are. Step 1 builds no gap file: it checks that the
joined gap files and their record are in $GF. Step 2 runs six rollouts of the candidate form with a joined gap file. The queue deletes, moves and stops nothing.

  python make_boxfuse_queue.py [--src FILE] [--out FILE] [--print]
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
RUNS = [("%s QMAP_FILE=%s VGAP=\"$GF/VGAPJE.npz\" VCLOSE=1" % (K, QM), "I10RQ50KF5Hb20gJEc"),
        ("%s QMAP_FILE=%s VGAP=\"$GF/VGAPJE90.npz\" VCLOSE=1" % (K, QM), "I10RQ50KF5Hb20gJA90"),
        ("%s QMAP_FILE=%s VGAP=\"$GF/VGAPJE.npz\" VCLOSE=1" % (C, QM), "I10RQ50Hb20gJEc")]
REFS = ["I10RLcKF5Hb20", "I10RQ50KF5Hb20", "I10RQ50KF5Hb20gGEc", "I10RQ50KF5Hb20gA90", "I10RLcHb20", "I10RQ50Hb20", "I10RQ50Hb20gGEc"]
TAG = "ZZ MAKE-BOXFUSE-QUEUE STOP: "

HEADER = [
    "# pod_boxfuse10k_queue.sh -- card ADD131 (BOXFUSE10K): 6 rollouts, model seeds 0 and 1, through the wrapper variant rzgfhqetjv of card ADD102 (hash 28bc31f21ecb4ccb, unchanged):",
    "# the candidate form (reader-fed form RR with --vis-gap-file and --vis-closing, no label at inference) with the joined gap file VGAPJE.npz (box read joined with the camera gap of card ADD106)",
    "# and with VGAPJE90.npz (0.90 times it), at readout K with no pre-roll (I10RQ50KF5Hb20gJEc, ...gJA90; --rule-every 5) and, for VGAPJE.npz, at the convention (I10RQ50Hb20gJEc; pinned readout, the rule every 10 steps), pinned grid.",
    "# A changed gap input of the rule is a study arm, measured here; its adoption is a design decision.",
    "# Step 0 gate: exit 4 unless $GE/lead_tests/lead_pytest_queue.out exists (the test of this kit passed in the lead's run before the queue start).",
    "# Step 1 builds no gap file: the queue stops (no rollout) unless $GF holds BOXFUSE10K_GAPFILES_RECORD.json, VGAPJE.npz and VGAPJE90.npz, the record has pass_j0 to pass_j4 all true,",
    "# and the sha256 of both files equals the one in the record.",
    "# Step 2: the six rollouts, one after the other, each started only with fewer than 3 GPU compute processes and more than 28 GB free (memory.max minus anon); a run with its T1_DONE is not run again.",
    "# Time rule: before each start the seconds since the first line of $GE/BOXFUSE10K_START.txt must be at most BF_STOP_S (default 4320, 1.2 pod-hours), else $GE/BOXFUSE10K_STOP.txt. A failed rollout writes the stop marker.",
    "# Shared pod state file FORK_STATE_FILE (default /workspace/FORK_STATE.txt; empty = no write): a BUSY line before the first rollout and an IDLE line on every exit path after it.",
    "# Done marker $GE/BOXFUSE10K_DONE.txt. Log $GE/boxfuse10k_queue.log. DRY=1 prints every command and creates nothing. RESUME=1 continues after a stop marker. Nothing is removed or moved.",
    "# Banked runs needed (both seeds, T1_DONE): " + " ".join(REFS) + ". Written by make_boxfuse_queue.py from the queue of card ADD115 (sha256 prefix %s). No training." % SRC_SHA16,
]

STEP = r'''step_inputs() { # step 1: the joined gap files and their record are checked; nothing is built
  if [ "$DRY" = 1 ]; then
    echo "CHECK: $GF holds BOXFUSE10K_GAPFILES_RECORD.json, VGAPJE.npz and VGAPJE90.npz"
    echo "CHECK: $BFREC has pass_j0 to pass_j4 all true and the sha256 of both gap files"
    return 0
  fi
  [ -f "$BFREC" ] || stop "NO-GAP-RECORD $BFREC"
  [ -f "$GF/VGAPJE.npz" ] || stop "NO-GAP-FILE $GF/VGAPJE.npz"
  [ -f "$GF/VGAPJE90.npz" ] || stop "NO-GAP-FILE $GF/VGAPJE90.npz"
  local adm
  adm=$("$PY" - "$BFREC" "$GF" <<'PYEOF'
import hashlib, json, os, sys
rec_path, gf = sys.argv[1:3]
rec = json.load(open(rec_path, encoding="utf-8"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for k in ("pass_j0", "pass_j1", "pass_j2", "pass_j3", "pass_j4"):
    if rec.get(k) is not True:
        print(k.upper() + "-NOT-PASS"); sys.exit(1)
for nm in ("VGAPJE", "VGAPJE90"):
    p = os.path.join(gf, nm + ".npz")
    if not os.path.isfile(p) or ((rec.get("files") or {}).get(nm) or {}).get("sha256") != sha(p):
        print("GAP-FILE-IS-NOT-THE-RECORDED-ONE " + nm); sys.exit(1)
print("ADMIT-OK gap files")
PYEOF
) || stop "INPUT-ADMISSION $adm"
  log "$adm"
}
'''

FORKSTATE = r'''# fork-state begin
FS=${FORK_STATE_FILE-/workspace/FORK_STATE.txt}
BUSYSET=0
fork_idle() { [ "$BUSYSET" = 1 ] || return 0; BUSYSET=0; [ -n "$FS" ] && echo "IDLE $(ts)" >> "$FS"; return 0; }
trap fork_idle EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [ "$DRY" = 1 ]; then
  echo "FORKSTATE: append 'BUSY card ADD131 rollouts of the v8proof fork <UTC>' to ${FS:-(no write)} before the first rollout and 'IDLE <UTC>' on every exit path after it"
elif [ -n "$FS" ]; then
  echo "BUSY card ADD131 rollouts of the v8proof fork $(ts)" >> "$FS" && BUSYSET=1
fi
# fork-state end
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
    once("GE=${PC_ROOT:-/workspace/v8proof/presattrc}", "GE=${BF_ROOT:-/workspace/v8proof/boxfuse10k}")
    once("GF=$GE/gapfiles\n", "GF=${GF:-$GE/gapfiles}\nBFREC=$GF/BOXFUSE10K_GAPFILES_RECORD.json\n")
    once("KQ=${PC_KIT:-$GE/kit_queue}", "KQ=${BF_KIT:-$GE/kit_queue}")
    once("PJ=${PC_JOINS:-/workspace/v8proof/presattr/joins}\n", "")
    once("SM=$GP/small_pc", "SM=$GP/small_boxfuse10k")
    once("STOPS=${PC_STOP_S:-7200}", "STOPS=${BF_STOP_S:-4320}")
    once("LOG=$GP/presattrc_queue.log", "LOG=$GP/boxfuse10k_queue.log")
    line_from('log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD115',
              'log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD131 (BOXFUSE10K): the candidate form with the joined gap file, '
              'at readout K with no pre-roll and at the convention, model seeds ${ARMS[*]}"')
    once('if [ "$DRY" != 1 ]; then\n  for k in "${ARMS[@]}"; do\n    [ -f "$W/t1/I10R0Hb20_s$k/T1_DONE" ] || stop "NO-BANKED-I10R0Hb20 s$k"\n  done\nfi\n', "")
    span("step_inputs() {", '[ "$DRY" = 1 ] || [ -f "$GRID" ] || stop "NO-PINNED-GRID"', STEP)
    line_from("  for nm in I10RLcHb20 ", "  for nm in %s; do" % " ".join(REFS))
    blocks = ['for k in "${ARMS[@]}"; do\n  %s rollout %s RR "$k" "$k"\ndone\n' % (env, name) for env, name in RUNS]
    once("step_inputs\nfor k in", "step_inputs\n" + FORKSTATE + "for k in")
    span('for k in "${ARMS[@]}"; do\n  READOUT=pinned KAPPA=0.2 QMAP_FILE=', 'if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi', "".join(blocks))
    once('NAMES="I10RQ50Hb20aLA I10RLcHb20pLE"', 'NAMES="%s"' % " ".join(n for _, n in RUNS))
    t = state["t"].replace("PRESATTRC10K", "BOXFUSE10K")
    for bad in ("PC_ROOT", "PC_KIT", "PC_STOP_S", "PC_JOINS", "small_pc", "presattrc", "PRESATTR", "JLE", "$PJ", "RJOIN=\"", "aLA RR", "pLE RLc", "ADD114", "I10R0Hb20", "make_margin"):
        if bad in t:
            raise SystemExit(TAG + "left over: " + bad)
    if t.count(" rollout I10") != 3 or t.count("VCLOSE=1 rollout") != 3 or t.count("EVERY=5 QMAP_FILE") != 2 or t.count("BOXFUSE10K_START.txt") < 4:
        raise SystemExit(TAG + "counts")
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=os.path.join(HERE, "pod_boxfuse10k_queue.sh"))
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
    print("ZZ MAKE-BOXFUSE-QUEUE DONE %s sha256 prefix %s" % (os.path.basename(a.out), hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
