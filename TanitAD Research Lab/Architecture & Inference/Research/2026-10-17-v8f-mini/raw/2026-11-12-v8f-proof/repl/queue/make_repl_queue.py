#!/usr/bin/env python
"""make_repl_queue.py -- card PREREG_V8F_PROOF_ADD121 (REPL10K). Writes the rollout queue of this card from the queue of card ADD117 by asserted substitutions.
Source (sha256 prefix of its LF text): ../../gapmargin/queue/pod_gapmargin10k_queue.sh e5811bf79826a718. What changes against the source: the names of the root, markers and log;
the model seeds (2 and 3, each with the reader seed and map of seed m - 2); a stage switch (RP_STAGE=identity runs three banked reference runs of model seed 0 again through the
wrapper of this card and compares each with its banked run, with markers of its own; the main stage needs that stage's done marker); a gate on the lead's record of the two new
models; the list of rollouts (13 per new model seed). The slot rule, the time rule, the gap file admission, the wrapper gate and the rollout call are the source's.

  python make_repl_queue.py [--out-dir DIR]      writes pod_repl10k_queue.sh under DIR (default: this folder). An output that exists is refused.
"""
import argparse
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = (os.path.normpath(os.path.join(HERE, "..", "..", "gapmargin", "queue", "pod_gapmargin10k_queue.sh")), "e5811bf79826a718")
OUT_NAME = "pod_repl10k_queue.sh"
TAG = "ZZ MAKE-REPL-QUEUE STOP: "
CONV = "READOUT=pinned KAPPA=0.2"
K0 = "READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5"
K30 = "RGRID=$G30 PREROLL=30 " + K0
# (setting part, run name, form, needs the reader map, gap file or None) in the order of the queue, per new model seed
RUNS = [(CONV, "I10R0Hb20", "R0", False, None), (CONV, "I10RLcHb20", "RLc", False, None), (CONV, "I10RQ50Hb20", "RR", True, None),
        (CONV, "I10RQ50Hb20gGEc", "RR", True, "$BASEF"), (CONV, "I10RQ50Hb20gA90", "RR", True, "$GF/VGAPLE_A90.npz"),
        (K0, "I10RLcKF5Hb20", "RLc", False, None), (K0, "I10RQ50KF5Hb20", "RR", True, None),
        (K0, "I10RQ50KF5Hb20gGEc", "RR", True, "$BASEF"), (K0, "I10RQ50KF5Hb20gA90", "RR", True, "$GF/VGAPLE_A90.npz"),
        (K30, "I10RLcKF5P30Hb20", "RLc", False, None), (K30, "I10RQ50KF5P30Hb20", "RR", True, None),
        (K30, "I10RQ50KF5P30Hb20gGEc", "RR", True, "$BASEF"), (K30, "I10RQ50KF5P30Hb20gA90", "RR", True, "$GF/VGAPLE_A90.npz")]
# identity stage: (setting part, new run name, form, needs the reader map, banked run name), model seed 0
IDRUNS = [(CONV, "I10R0Hb20IDR", "R0", False, "I10R0Hb20"), (K0, "I10RLcKF5Hb20IDR", "RLc", False, "I10RLcKF5Hb20"), (CONV, "I10RQ50Hb20IDR", "RR", True, "I10RQ50Hb20")]
NAMES = [r[1] for r in RUNS]

Q_HEADER = [
    "# pod_repl10k_queue.sh -- card ADD121 (REPL10K): the lead-rule candidate set on the model seeds 2 and 3 of the 10k line, through the wrapper variant rzgfhqetjv (hash 28bc31f21ecb4ccb,",
    "# unchanged). Per new model seed m, with the reader seed and map of seed m - 2: at the convention the spread run R0, the label form RLc, the reader-fed form RR, the candidate GEc and the",
    "# candidate with the gap file scaled by 0.90 (A90); at readout K with no pre-roll and with a pre-roll of 30 steps (grid_p30) RLc, RR, GEc and A90. 26 rollouts, names as banked.",
    "# RP_STAGE=identity (run before the main stage, needs model seed 0 only): three banked reference runs are made again through this wrapper (I10R0Hb20IDR, I10RLcKF5Hb20IDR,",
    "# I10RQ50Hb20IDR, model seed 0) and each is compared with its banked run (dump files with cmp_dumps.py, rule log arrays); a difference stops the queue. Markers REPL10K_ID_*.",
    "# Step 0 gate: exit 4 unless $GE/lead_tests/lead_pytest_queue.out exists (the test of this kit passed in the lead's run before the queue start).",
    "# Main stage: needs $GE/REPL10K_ID_DONE.txt and the lead's record of the two new models ($GE/models/REPL_MODELS_RECORD.json: for UE_s2 and UE_s3 the checkpoint hash, gate_pass true,",
    "# recipe_same_but_seed true); the checkpoints on disk must have the recorded hashes. Gap files: the base file (sha256 prefix 55de2f7c) and the record of card ADD117.",
    "# Each rollout starts only with fewer than 3 GPU compute processes and more than 28 GB free (memory.max minus anon); a run with its T1_DONE is not run again.",
    "# Time rule: before each start the seconds since the first line of the stage's start file must be at most RP_STOP_S (default 25200), else the stage's stop file. A failed rollout writes it.",
    "# Done marker $GE/REPL10K_DONE.txt (identity stage: REPL10K_ID_DONE.txt). Log $GE/repl_queue.log. DRY=1 prints every command and creates nothing. RESUME=1 continues after a stop marker.",
    "# Nothing is removed or moved. Written by make_repl_queue.py from the queue of card ADD117 (sha256 prefix e5811bf79826a718). No training.",
]

FUNCS = '''step_inputs
step_models() { # the lead's record of the two new models against the checkpoints on disk
  if [ "$DRY" = 1 ]; then
    echo "CHECK: $MREC names for UE_s2 and UE_s3 the sha256 of $W/ckpt/UE_s<m>/A/ckpt_last.pt, gate_pass true and recipe_same_but_seed true"
    return 0
  fi
  [ -f "$MREC" ] || stop "NO-MODELS-RECORD"
  local adm
  adm=$("$PY" - "$MREC" "$W/ckpt" "${ARMS[@]}" <<'PYEOF'
import hashlib, json, os, re, sys
rec_path, ck, seeds = sys.argv[1], sys.argv[2], sys.argv[3:]
rec = json.load(open(rec_path, encoding="utf-8"))
if not re.fullmatch(r"[0-9a-f]{64}", str(rec.get("note_sha256") or "")):
    print("NO-NOTE-HASH"); sys.exit(1)
for k in seeds:
    m = (rec.get("models") or {}).get("UE_s" + k) or {}
    p = os.path.join(ck, "UE_s" + k, "A", "ckpt_last.pt")
    if m.get("gate_pass") is not True or m.get("recipe_same_but_seed") is not True:
        print("MODEL-NOT-ADMITTED UE_s" + k); sys.exit(1)
    if not os.path.isfile(p) or hashlib.sha256(open(p, "rb").read()).hexdigest() != m.get("ckpt_sha256"):
        print("CKPT-IS-NOT-THE-RECORDED-ONE UE_s" + k); sys.exit(1)
print("ADMIT-OK models")
PYEOF
) || stop "MODELS-ADMISSION $adm"
  log "$adm"
}
step_identity() { # NEW BANKED (run folder names): dump files and rule log arrays; a difference stops the queue
  local a=$1 b=$2
  if [ "$DRY" = 1 ]; then
    echo "CMD: CUDA_VISIBLE_DEVICES=-1 $PY $B/cmp_dumps.py $W/t1/$a/dump_${a}_m10_causal_heads $W/t1/$b/dump_${b}_m10_causal_heads"
    echo "CHECK: the arrays of the rule logs of $a (where the banked run has them) equal those of $b; a difference stops the queue"
    return 0
  fi
  done_step "idr_$a" && return 0
  [ -f "$W/t1/$a/T1_DONE" ] || stop "NO-IDENTITY-RUN $a"
  mkdir -p "$GP/idr"
  CUDA_VISIBLE_DEVICES=-1 "$PY" "$B/cmp_dumps.py" "$W/t1/$a/dump_${a}_m10_causal_heads" "$W/t1/$b/dump_${b}_m10_causal_heads" > "$GP/idr/CMP_DUMPS_$a.out" 2>&1 || stop "IDENTITY-RUN-DUMPS-DIFFER $a"
  local adm
  adm=$(CUDA_VISIBLE_DEVICES=-1 "$PY" - "$W/t1/$a/ticks" "$a" "$W/t1/$b/ticks" "$b" <<'PYEOF'
import os, sys
import numpy as np
da, a, db, b = sys.argv[1:5]
bad, n, skipped, files, extra = [], 0, 0, 0, 0
for pre in ("rule_log_", "rule_log_mid_"):
    for suf in (".npz", "_rd.npz"):
        pa, pb = os.path.join(da, pre + a + suf), os.path.join(db, pre + b + suf)
        ea, eb = os.path.isfile(pa), os.path.isfile(pb)
        if not ea and not eb:
            continue
        if eb and not ea:
            bad.append("lost-file:" + pre + suf); continue
        if ea and not eb:
            extra += 1; continue                                                    # a file of a later wrapper variant, counted
        files += 1
        za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
        lost = [k for k in zb.files if k not in za.files and zb[k].ndim >= 1]      # a row array of the banked log that the new log lacks
        if lost:
            bad.append("lost:" + ",".join(lost[:4]) + ":" + pre + suf)
        extra += len(set(za.files) - set(zb.files))                                 # keys of a later wrapper variant, counted
        for k in sorted(set(za.files) & set(zb.files)):
            x, y = za[k], zb[k]
            if x.dtype.kind == "O" or y.dtype.kind == "O" or (x.dtype.kind in "US" and x.ndim == 0):
                skipped += 1; continue
            same = x.shape == y.shape and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f" else np.array_equal(x, y))
            n += 1
            if not same:
                bad.append(k + ":" + pre + suf)
if bad:
    print("RULE-LOG-DIFFERS files %d arrays %d: %s" % (files, n, " ".join(bad[:12]))); sys.exit(1)
print("ADMIT-OK identity of the rule logs: files %d, arrays compared %d, not compared (objects, text of one value) %d, keys only in the new run %d" % (files, n, skipped, extra))
PYEOF
) || { echo "$adm" > "$GP/idr/CMP_RULELOG_$a.out"; stop "IDENTITY-RUN-RULE-LOG-DIFFERS $a"; }
  echo "$adm" > "$GP/idr/CMP_RULELOG_$a.out"
  mark "idr_$a"
  log "identity run $a equals the banked run $b: $(tail -n 1 "$GP/idr/CMP_DUMPS_$a.out" | cut -c1-80); $adm"
}
'''


class Text:
    def __init__(self, t):
        self.t = t.replace("\r\n", "\n")

    def once(self, a, b):
        if self.t.count(a) != 1:
            raise SystemExit(TAG + "expected once, found %d: %r" % (self.t.count(a), a[:90]))
        self.t = self.t.replace(a, b)

    def span(self, a, b, new):
        s = self.t
        if s.count(a) != 1 or s.count(b) != 1 or not s.index(a) < s.index(b):
            raise SystemExit(TAG + "anchors: %r (%d) / %r (%d)" % (a[:60], s.count(a), b[:60], s.count(b)))
        self.t = s[:s.index(a)] + new + s[s.index(b):]

    def line_from(self, a, new):
        s = self.t
        if s.count(a) != 1:
            raise SystemExit(TAG + "line anchor found %d times: %r" % (s.count(a), a[:60]))
        i = s.index(a)
        self.t = s[:i] + new + s[s.index("\n", i):]


def read_source():
    path, want = SOURCE
    t = io.open(path, "rb").read().replace(b"\r\n", b"\n")
    got = hashlib.sha256(t).hexdigest()[:16]
    if got != want:
        raise SystemExit(TAG + "the source %s has the hash %s, expected %s" % (os.path.basename(path), got, want))
    return t.decode("utf-8")


def run_line(setting, name, form, qmap, gap, m, r, indent):
    parts = [setting]
    if qmap:
        parts.append('QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s%s_q50.json"' % r)
    if gap:
        parts.append('VGAP="%s" VCLOSE=1' % gap)
    return "%s%s rollout %s %s %s %s\n" % (indent, " ".join(parts), name, form, m, r)


def steps():
    out = [FUNCS, 'if [ "$STAGE" = identity ]; then\n']
    for setting, name, form, qmap, banked in IDRUNS:
        out.append(run_line(setting, name, form, qmap, None, "0", "0", "  "))
        out.append("  step_identity %s_s0 %s_s0\n" % (name, banked))
    out.append('else\n  [ "$DRY" = 1 ] || [ -f "$GP/REPL10K_ID_DONE.txt" ] || stop "NO-IDENTITY-STAGE"\n  step_models\n  for k in "${ARMS[@]}"; do\n    r=$((k - 2))\n')
    for setting, name, form, qmap, gap in RUNS:
        out.append(run_line(setting, name, form, qmap, gap, '"$k"', '"$r"', "    ").replace("QMAP_LR16_s\"$r\"_q50", "QMAP_LR16_s${r}_q50"))
    out.append("  done\nfi\n")
    return "".join(out)


def build_queue(src):
    x = Text(src)
    lines = x.t.split("\n")
    if not (lines[0] == "#!/usr/bin/env bash" and lines[12] == "set -u" and all(v.startswith("#") for v in lines[1:12])):
        raise SystemExit(TAG + "the head of the queue source is not the expected one")
    x.t = "\n".join([lines[0]] + Q_HEADER + lines[12:])
    x.once("GE=${GM_ROOT:-/workspace/v8proof/gapmargin}", "GE=${RP_ROOT:-/workspace/v8proof/repl}")
    x.once("GF=$GE/gapfiles\n", "GF=${RP_GAPFILES:-/workspace/v8proof/gapmargin/gapfiles}\nMREC=${RP_MODELS_RECORD:-$GE/models/REPL_MODELS_RECORD.json}\nSTAGE=${RP_STAGE:-main}\n")
    x.once("KQ=${GM_KIT:-$GE/kit_queue}", "KQ=${RP_KIT:-$GE/kit_queue}")
    x.once("SM=$GP/small_gm", "SM=$GP/small_rp")
    x.once("STOPS=${GM_STOP_S:-10800}", "STOPS=${RP_STOP_S:-25200}")
    x.once("LOG=$GP/gapmargin_queue.log", "LOG=$GP/repl_queue.log")
    x.once('ARMS=(0 1); PS=""\n',
           'ARMS=(2 3); FIRST=(0 1); PS=""\nMK=REPL10K; CKS="${ARMS[*]}"\ncase "$STAGE" in main) ;; identity) MK=REPL10K_ID; CKS=0 ;; *) echo "REFUSED: RP_STAGE must be main or identity"; exit 2 ;; esac\n')
    x.line_from('log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD117',
                'log "queue start (stage $STAGE, DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD121 (REPL10K): the lead-rule candidate set on the model seeds ${ARMS[*]} '
                '(reader seeds of the first pair), identity stage on model seed 0"')
    x.once('  for k in "${ARMS[@]}"; do\n    [ -f "$W/t1/I10R0Hb20_s$k/T1_DONE" ] || stop "NO-BANKED-I10R0Hb20 s$k"\n',
           '  for k in "${FIRST[@]}"; do\n    [ -f "$W/t1/I10R0Hb20_s$k/T1_DONE" ] || stop "NO-BANKED-I10R0Hb20 s$k"\n')
    x.once('  for k in "${ARMS[@]}"; do\n    [ -f "$W/ckpt/UE_s$k/A/ckpt_last.pt" ] || stop "NO-CKPT UE_s$k"\n',
           '  for k in $CKS; do\n    [ -f "$W/ckpt/UE_s$k/A/ckpt_last.pt" ] || stop "NO-CKPT UE_s$k"\n')
    x.line_from("  for nm in I10RLcKF5Hb20 ", "  for nm in %s; do" % " ".join(NAMES))
    x.once('    for k in "${ARMS[@]}"; do [ -f "$W/t1/${nm}_s$k/T1_DONE" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done\n',
           '    for k in "${FIRST[@]}"; do [ -f "$W/t1/${nm}_s$k/T1_DONE" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done\n')
    x.span("step_inputs\nfor k in", 'if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi', steps())
    x.once('NAMES="I10RQ50KF5Hb20gA90 I10RQ50KF5Hb20gA80 I10RQ50Hb20gA90 I10RQ50Hb20gA80"', 'NAMES="%s"\ncp -f "$GP/idr/"*.out "$SM/" 2>/dev/null' % " ".join(NAMES))
    t = x.t.replace("GAPMARGIN10K", "${MK}")
    for bad in ("GM_ROOT", "GM_KIT", "GM_STOP_S", "small_gm", "gapmargin_queue.log", "ADD117 (GAPMARGIN", "VGAPLE_A80.npz", "ARMS=(0 1)", "GAPMARGIN10K"):
        if bad in t:
            raise SystemExit(TAG + "left over in the queue: " + bad)
    if (t.count(" rollout I10") != 16 or t.count('"$k" "$r"\n') != 13 or t.count("IDR ") != 3 or t.count("  step_identity I10") != 3 or t.count("PREROLL=30 ") != 4
            or t.count("${MK}_START.txt") < 4 or t.count('echo "ZZ ${MK} DONE $(ts) all steps done" > "$GP/${MK}_DONE.txt"') != 1):
        raise SystemExit(TAG + "counts of the queue")
    return t


def make():
    return build_queue(read_source())


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--out-dir", default=HERE)
    A = ap.parse_args(argv)
    t = make()
    os.makedirs(A.out_dir, exist_ok=True)
    p = os.path.join(A.out_dir, OUT_NAME)
    with io.open(p, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(t)
    print("wrote %s %s" % (OUT_NAME, hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
