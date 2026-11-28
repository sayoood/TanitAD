#!/usr/bin/env python
"""make_presattr_queue.py -- card PREREG_V8F_PROOF_ADD114 (PRESATTR10K). Writes pod_presattr10k_queue.sh from the queue of card ADD113
(candset/queue/pod_candset10k_queue.sh, sha256 prefix 0b531cd3deda5ecc) by asserted substitutions. The rollout function, the slot wait, the stop rule and
the marker line formats are kept as they are. Step 1 checks the two gap files and makes the planted join (CPU) when it is absent; step 2 runs six rollouts.

  python make_presattr_queue.py [--src FILE] [--out FILE] [--print]
"""
import argparse
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "candset", "queue", "pod_candset10k_queue.sh"))
SRC_SHA16 = "0b531cd3deda5ecc"
QM = '"$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"'
K = "READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5"
RUNS = [("%s QMAP_FILE=%s VGAP=\"$LAF\" VCLOSE=1" % (K, QM), "I10RQ50KF5Hb20aLA", "RR"),
        ("%s RJOIN=\"$PJ/JLE.npz\"" % K, "I10RLcKF5Hb20pLE", "RLc"),
        ("RGRID=$G30 PREROLL=30 %s RJOIN=\"$PJ/JLE.npz\"" % K, "I10RLcKF5P30Hb20pLE", "RLc")]
REFS = ["I10RLcKF5Hb20", "I10RQ50KF5Hb20", "I10RQ50KF5Hb20gGEc", "I10RLcKF5P30Hb20", "I10RQ50KF5P30Hb20", "I10RQ50KF5P30Hb20gGEc", "I10RQ50KF5P30Hb20aLA", "I10R0Hb20"]

HEADER = [
    "# pod_presattr10k_queue.sh -- card ADD114 (PRESATTR10K): 6 rollouts, model seeds 0 and 1, through the wrapper variant rzgfhqetjv of card ADD102 (hash 28bc31f21ecb4ccb, unchanged):",
    "# LA at K0 (I10RQ50KF5Hb20aLA: the reader-fed form RR at readout K with no pre-roll, --rule-every 5, with --vis-gap-file <VGAPA_LA.npz of card ADD110> and --vis-closing),",
    "# PL at K0 (I10RLcKF5Hb20pLE) and PL at K30 (I10RLcKF5P30Hb20pLE: pre-roll 30 steps = 6 s, grid_p30): the label form RLc with --rule-join <JLE.npz>, the planted join with the",
    "# camera gap of card ADD106 on the label rows; the scoring keeps the evaluation join. Cells with a label input are attribution arms and never candidates.",
    "# Step 0 gate: exit 4 unless $GE/lead_tests/lead_pytest_queue.out exists (the test of this kit passed in the lead's run before the queue start).",
    "# Step 1: the base gap file (sha256 prefix 55de2f7c) and the LA gap file (the hash in the record of card ADD111) are checked; when $GE/joins is absent make_planted_join_le.py runs on the CPU;",
    "# the queue stops unless JLE_RECORD.json has pass_e0 true and the hashes of the join, of the base file and of the evaluation join.",
    "# Step 2: the six rollouts, one after the other, each started only with fewer than 3 GPU compute processes and more than 28 GB free (memory.max minus anon); a run with its T1_DONE is not run again.",
    "# Time rule: before each start the seconds since the first line of $GE/PRESATTR10K_START.txt must be at most PA_STOP_S (default 10800), else $GE/PRESATTR10K_STOP.txt. A failed rollout writes the stop marker.",
    "# Done marker $GE/PRESATTR10K_DONE.txt. Log $GE/presattr_queue.log. DRY=1 prints every command and creates nothing. RESUME=1 continues after a stop marker. Nothing is removed or moved.",
    "# Banked runs needed (both seeds, T1_DONE): " + " ".join(REFS) + ". Written by make_presattr_queue.py from the queue of card ADD113 (sha256 prefix %s). No training." % SRC_SHA16,
]

STEP = r'''step_inputs() { # step 1: the two gap files as they are, and the planted join (made on the CPU when $PJ is absent)
  if [ "$DRY" = 1 ]; then
    echo "CHECK: sha256 of $BASEF begins 55de2f7c; sha256 of $LAF equals la_file.sha256 of $GREC"
    echo "CMD: [only when $PJ is absent] CUDA_VISIBLE_DEVICES=-1 $PY $KQ/make_planted_join_le.py --kit-gap $KITG --join $JOINE --base $BASEF --out-dir $PJ"
    echo "CHECK: $PJ/JLE_RECORD.json has pass_e0 true and the sha256 of the join, of the base file and of the evaluation join"
    return 0
  fi
  [ -f "$BASEF" ] || stop "NO-BASE-GAP-FILE $BASEF"
  local got; got=$(sha256sum "$BASEF" | cut -c1-8); [ "$got" = "55de2f7c" ] || stop "BASE-GAP-FILE-SHA $got"
  [ -f "$LAF" ] && [ -f "$GREC" ] || stop "NO-LA-FILE-OR-RECORD"
  if [ ! -d "$PJ" ]; then
    check_time "planted join"
    [ -f "$KQ/make_planted_join_le.py" ] || stop "NO-JOIN-TOOL $KQ/make_planted_join_le.py"
    CUDA_VISIBLE_DEVICES=-1 "$PY" "$KQ/make_planted_join_le.py" --kit-gap "$KITG" --join "$JOINE" --base "$BASEF" --out-dir "$PJ" >> "$LOG" 2>&1 || log "join build rc $?"
  fi
  [ -f "$PJ/JLE_RECORD.json" ] || stop "NO-JOIN-RECORD"
  local adm
  adm=$("$PY" - "$PJ/JLE_RECORD.json" "$PJ/JLE.npz" "$JOINE" "$BASEF" "$LAF" "$GREC" <<'PYEOF'
import hashlib, json, os, sys
rec_path, jle, ejoin, basef, laf, grec = sys.argv[1:7]
rec = json.load(open(rec_path, encoding="utf-8"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if rec.get("pass_e0") is not True:
    print("E0-NOT-PASS"); sys.exit(1)
if not os.path.isfile(jle) or (rec.get("join") or {}).get("sha256") != sha(jle):
    print("JOIN-IS-NOT-THE-RECORDED-ONE"); sys.exit(1)
if rec.get("eval_join_sha256") != sha(ejoin):
    print("RECORD-IS-FOR-ANOTHER-EVAL-JOIN"); sys.exit(1)
if (rec.get("base") or {}).get("sha256") != sha(basef):
    print("BASE-IS-NOT-THE-RECORDED-ONE"); sys.exit(1)
g = json.load(open(grec, encoding="utf-8"))
if (g.get("la_file") or {}).get("sha256") != sha(laf):
    print("LA-FILE-IS-NOT-THE-ONE-OF-THE-ADD111-RECORD"); sys.exit(1)
print("ADMIT-OK join and gap files")
PYEOF
) || stop "INPUT-ADMISSION $adm"
  log "$adm"
}
'''


def build(src_text):
    t = src_text.replace("\r\n", "\n")
    state = {"t": t}

    def once(a, b):
        if state["t"].count(a) != 1:
            raise SystemExit("ZZ MAKE-PRESATTR-QUEUE STOP: expected once, found %d: %r" % (state["t"].count(a), a[:90]))
        state["t"] = state["t"].replace(a, b)

    def between(a, b, new, na=1):
        s = state["t"]
        if s.count(a) != na or s.count(b) != 1 or not s.index(a) < s.index(b):
            raise SystemExit("ZZ MAKE-PRESATTR-QUEUE STOP: anchors: %r (%d) / %r (%d)" % (a[:60], s.count(a), b[:60], s.count(b)))
        state["t"] = s[:s.index(a)] + new + s[s.index(b):]

    lines = state["t"].split("\n")
    if not (lines[0] == "#!/usr/bin/env bash" and lines[12] == "set -u" and all(x.startswith("#") for x in lines[1:12])):
        raise SystemExit("ZZ MAKE-PRESATTR-QUEUE STOP: the head of the source is not the expected one")
    state["t"] = "\n".join([lines[0]] + HEADER + lines[12:])
    once("GE=${CS_ROOT:-/workspace/v8proof/candset}", "GE=${PA_ROOT:-/workspace/v8proof/presattr}")
    once("SM=$GP/small_cs", "SM=$GP/small_pa")
    once("STOPS=${CS_STOP_S:-10800}", "STOPS=${PA_STOP_S:-10800}")
    once("LOG=$GP/candset_queue.log", "LOG=$GP/presattr_queue.log")
    once("KG=$GE/kit_gap\n", "KITG=/workspace/v8proof/gaperr/kit_gap\nKQ=${PA_KIT:-$GE/kit_queue}\nPJ=$GE/joins\nGREC=/workspace/v8proof/gaperr/gapfiles/GAPERR_GAPFILES_RECORD.json\n")
    once("KR=$GE/kit_read\n", "")
    s = state["t"]
    a = 'log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD113'
    if s.count(a) != 1:
        raise SystemExit("ZZ MAKE-PRESATTR-QUEUE STOP: log start line")
    i = s.index(a)
    j = s.index("\n", i)
    state["t"] = (s[:i] + 'log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD114 (PRESATTR10K): LA at K0 (reader presence, label gap), PL at K0 and at K30 '
                  '(label presence, camera gap through the planted join), model seeds ${ARMS[*]}"' + s[j:])
    between("step_base() {", '[ "$DRY" = 1 ] || [ -f "$G60" ] || stop "NO-GRID-P60"', STEP)
    once('[ "$DRY" = 1 ] || [ -f "$G60" ] || stop "NO-GRID-P60"', '[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"')
    s = state["t"]
    a2 = "  for nm in I10RQ50Hb20 "
    if s.count(a2) != 1:
        raise SystemExit("ZZ MAKE-PRESATTR-QUEUE STOP: the list of banked runs")
    i = s.index(a2)
    j = s.index("\n", i)
    state["t"] = s[:i] + "  for nm in %s; do" % " ".join(REFS) + s[j:]
    once("step_base\n", "step_inputs\n")
    blocks = ['for k in "${ARMS[@]}"; do\n  %s rollout %s %s "$k" "$k"\ndone\n' % (env, name, form) for env, name, form in RUNS]
    s = state["t"]
    a4, b4 = 'for k in "${ARMS[@]}"; do\n  READOUT=pinned KAPPA=0.2 QMAP_FILE=', 'if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi'
    if s.count(a4) != 1 or s.count(b4) != 1 or not s.index(a4) < s.index(b4):
        raise SystemExit("ZZ MAKE-PRESATTR-QUEUE STOP: the rollout loops of the source were not found")
    state["t"] = s[:s.index(a4)] + "".join(blocks) + s[s.index(b4):]
    once('NAMES="I10RQ50Hb20gGEc I10RQ50KF5Hb20gGEc I10RQ50KF5P60Hb20gGEc"', 'NAMES="%s"' % " ".join(n for _, n, _ in RUNS))
    t = state["t"].replace("CANDSET10K", "PRESATTR10K")
    for bad in ("CANDSET", "candset", "step_base", "$KG", "$KR", "gGEc RR", "P60Hb20gGEc"):
        if bad in t:
            raise SystemExit("ZZ MAKE-PRESATTR-QUEUE STOP: left over: " + bad)
    if t.count(" rollout I10") != 3 or t.count('RJOIN="$PJ/JLE.npz"') != 2 or t.count('VGAP="$LAF" VCLOSE=1') != 1 or t.count("PRESATTR10K_START.txt") < 4:
        raise SystemExit("ZZ MAKE-PRESATTR-QUEUE STOP: counts")
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=os.path.join(HERE, "pod_presattr10k_queue.sh"))
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    raw = io.open(a.src, "rb").read().replace(b"\r\n", b"\n")
    got = hashlib.sha256(raw).hexdigest()[:16]
    if got != SRC_SHA16:
        raise SystemExit("ZZ MAKE-PRESATTR-QUEUE STOP: source sha256 prefix %s, expected %s" % (got, SRC_SHA16))
    t = build(raw.decode("utf-8"))
    if a.print:
        sys.stdout.write(t)
        return 0
    with io.open(a.out, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(t)
    print("ZZ MAKE-PRESATTR-QUEUE DONE %s sha256 prefix %s" % (os.path.basename(a.out), hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
