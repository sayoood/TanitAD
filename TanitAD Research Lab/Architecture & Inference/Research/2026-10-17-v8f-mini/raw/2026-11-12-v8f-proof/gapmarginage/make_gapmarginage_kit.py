#!/usr/bin/env python
"""make_gapmarginage_kit.py -- card PREREG_V8F_PROOF_ADD119 (GAPMARGINAGE10K). Writes the queue, the reader and the reader's pod driver of this card from the kit of card ADD117
by asserted substitutions (tables, run lines, names of markers and files); no logic is changed. Sources (sha256 prefixes of their LF text):
  ../gapmargin/queue/pod_gapmargin10k_queue.sh e5811bf79826a718     ../gapmargin/read/read_gapmargin10k.py 0aab17c67eb60756     ../gapmargin/read/podread_gapmargin10k.sh b44d8181d3894ae9

  python make_gapmarginage_kit.py [--out-dir DIR]      writes queue/pod_gapmarginage10k_queue.sh, read/read_gapmarginage10k.py, read/podread_gapmarginage10k.sh under DIR (default: this folder)
An output that exists is refused.
"""
import argparse
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRCD = os.path.normpath(os.path.join(HERE, "..", "gapmargin"))
SOURCES = {"queue": (os.path.join(SRCD, "queue", "pod_gapmargin10k_queue.sh"), "e5811bf79826a718"),
           "reader": (os.path.join(SRCD, "read", "read_gapmargin10k.py"), "0aab17c67eb60756"),
           "driver": (os.path.join(SRCD, "read", "podread_gapmargin10k.sh"), "b44d8181d3894ae9")}
TAG = "ZZ MAKE-GAPMARGINAGE-KIT STOP: "
QM = '"$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"'
K = "READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5"
RUNS = [("RGRID=$G30 PREROLL=30", "A90", "I10RQ50KF5P30Hb20gA90"), ("RGRID=$G30 PREROLL=30", "A80", "I10RQ50KF5P30Hb20gA80"),
        ("RGRID=$G60 PREROLL=60", "A90", "I10RQ50KF5P60Hb20gA90"), ("RGRID=$G60 PREROLL=60", "A80", "I10RQ50KF5P60Hb20gA80")]
REFS = ["I10RLcKF5P30Hb20", "I10RQ50KF5P30Hb20", "I10RQ50KF5P30Hb20gGEc", "I10RLcKF5P60Hb20", "I10RQ50KF5P60Hb20", "I10RQ50KF5P60Hb20gGEc", "I10R0Hb20"]
READ_RUNS = ["I10RQ50KF5P30Hb20gA90", "I10RQ50KF5P30Hb20gA80", "I10RQ50KF5P30Hb20gGEc", "I10RLcKF5P30Hb20", "I10RQ50KF5P30Hb20",
             "I10RQ50KF5P60Hb20gA90", "I10RQ50KF5P60Hb20gA80", "I10RQ50KF5P60Hb20gGEc", "I10RLcKF5P60Hb20", "I10RQ50KF5P60Hb20", "I10R0Hb20"]

Q_HEADER = [
    "# pod_gapmarginage10k_queue.sh -- card ADD119 (GAPMARGINAGE10K): 8 rollouts, model seeds 0 and 1, through the wrapper variant rzgfhqetjv of card ADD102 (hash 28bc31f21ecb4ccb, unchanged):",
    "# the candidate form (reader-fed form RR with --vis-gap-file and --vis-closing, no label at inference) with the scaled camera gap files of card ADD117 (A90: 0.90, A80: 0.80; not rebuilt),",
    "# at readout K with a pre-roll of 30 steps (I10RQ50KF5P30Hb20gA90, ...gA80; grid_p30) and of 60 steps (I10RQ50KF5P60Hb20gA90, ...gA80; grid_p60), --rule-every 5.",
    "# A margin on the gap is a parameter of the rule's input: a study arm, measured here; its adoption is a design decision.",
    "# Step 0 gate: exit 4 unless $GE/lead_tests/lead_pytest_queue.out exists (the test of this kit passed in the lead's run before the queue start).",
    "# Step 1: the base gap file (sha256 prefix 55de2f7c) is checked; the folder of the scaled files is that of card ADD117 ($GF; when it is absent the queue stops, the gap tool is not in this kit);",
    "# the queue stops unless GAPMARGIN_GAPFILES_RECORD.json has pass_k0 true and the hashes of the base file and of both scaled files.",
    "# Step 2: the eight rollouts, one after the other, each started only with fewer than 3 GPU compute processes and more than 28 GB free (memory.max minus anon); a run with its T1_DONE is not run again.",
    "# Time rule: before each start the seconds since the first line of $GE/GAPMARGINAGE10K_START.txt must be at most GA_STOP_S (default 10800), else $GE/GAPMARGINAGE10K_STOP.txt. A failed rollout writes the stop marker.",
    "# Done marker $GE/GAPMARGINAGE10K_DONE.txt. Log $GE/gapmarginage_queue.log. DRY=1 prints every command and creates nothing. RESUME=1 continues after a stop marker. Nothing is removed or moved.",
    "# Banked runs needed (both seeds, T1_DONE): " + " ".join(REFS) + ". Written by make_gapmarginage_kit.py from the queue of card ADD117 (sha256 prefix e5811bf79826a718). No training.",
]
SPEC_NEW = '''SPEC = {
    "K30": {"P": 30, "readout": "K", "every": 5, "grid": "30", "grid_file": "grid_p30.json",
            "runs": {"A90": "RQ50KF5P30Hb20gA90", "A80": "RQ50KF5P30Hb20gA80", "GEc": "RQ50KF5P30Hb20gGEc", "RLc": "RLcKF5P30Hb20", "RR": "RQ50KF5P30Hb20"}},
    "K60": {"P": 60, "readout": "K", "every": 5, "grid": "60", "grid_file": "grid_p60.json",
            "runs": {"A90": "RQ50KF5P60Hb20gA90", "A80": "RQ50KF5P60Hb20gA80", "GEc": "RQ50KF5P60Hb20gGEc", "RLc": "RLcKF5P60Hb20", "RR": "RQ50KF5P60Hb20"}},
}
NEW_ARMS = {"K30": ("A90", "A80"), "K60": ("A90", "A80")}           # the runs of this card; the other cells are banked
'''
EXPECTED_NEW = '''EXPECTED = {                                                        # banked levels, (C, U, F), card section 5 control m3 (rows 179 and 181)
    "K30": {"GEc": (0.0062, 0.0839, 0.5154), "RR": (0.0954, 0.1339, 0.4954), "RLc": (0.0000, 0.0714, 0.4908)},
    "K60": {"GEc": (0.0137, 0.0738, 0.6438), "RR": (0.1096, 0.1230, 0.6575), "RLc": (0.0137, 0.0410, 0.6301)},
}
'''


class Text:
    def __init__(self, t):
        self.t = t.replace("\r\n", "\n")

    def once(self, a, b):
        if self.t.count(a) != 1:
            raise SystemExit(TAG + "expected once, found %d: %r" % (self.t.count(a), a[:90]))
        self.t = self.t.replace(a, b)

    def span(self, a, b, new, first=False):
        s = self.t
        if (s.count(a) != 1 and not first) or s.count(a) < 1 or s.count(b) != 1 or not s.index(a) < s.index(b):
            raise SystemExit(TAG + "anchors: %r (%d) / %r (%d)" % (a[:60], s.count(a), b[:60], s.count(b)))
        self.t = s[:s.index(a)] + new + s[s.index(b):]

    def line_from(self, a, new):
        s = self.t
        if s.count(a) != 1:
            raise SystemExit(TAG + "line anchor found %d times: %r" % (s.count(a), a[:60]))
        i = s.index(a)
        self.t = s[:i] + new + s[s.index("\n", i):]

    def every(self, a, b, n):
        if self.t.count(a) != n:
            raise SystemExit(TAG + "expected %d times, found %d: %r" % (n, self.t.count(a), a[:60]))
        self.t = self.t.replace(a, b)


def build_queue(src):
    x = Text(src)
    lines = x.t.split("\n")
    if not (lines[0] == "#!/usr/bin/env bash" and lines[12] == "set -u" and all(v.startswith("#") for v in lines[1:12])):
        raise SystemExit(TAG + "the head of the queue source is not the expected one")
    x.t = "\n".join([lines[0]] + Q_HEADER + lines[12:])
    x.once("GE=${GM_ROOT:-/workspace/v8proof/gapmargin}", "GE=${GA_ROOT:-/workspace/v8proof/gapmarginage}")
    x.once("GF=$GE/gapfiles\n", "GF=${GA_GAPFILES:-/workspace/v8proof/gapmargin/gapfiles}\n")
    x.once("KQ=${GM_KIT:-$GE/kit_queue}", "KQ=${GA_KIT:-$GE/kit_queue}")
    x.once("SM=$GP/small_gm", "SM=$GP/small_ga")
    x.once("STOPS=${GM_STOP_S:-10800}", "STOPS=${GA_STOP_S:-10800}")
    x.once("LOG=$GP/gapmargin_queue.log", "LOG=$GP/gapmarginage_queue.log")
    x.line_from('log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD117',
                'log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD119 (GAPMARGINAGE10K): the candidate form with the camera gap scaled by 0.90 and by 0.80, '
                'at readout K with a pre-roll of 30 and of 60 steps, model seeds ${ARMS[*]}"')
    x.once('[ "$DRY" = 1 ] || [ -f "$GRID" ] || stop "NO-PINNED-GRID"\n',
           '[ "$DRY" = 1 ] || [ -f "$GRID" ] || stop "NO-PINNED-GRID"\n[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"\n[ "$DRY" = 1 ] || [ -f "$G60" ] || stop "NO-GRID-P60"\n')
    x.line_from("  for nm in I10RLcKF5Hb20 ", "  for nm in %s; do" % " ".join(REFS))
    blocks = ['for k in "${ARMS[@]}"; do\n  %s %s QMAP_FILE=%s VGAP="$GF/VGAPLE_%s.npz" VCLOSE=1 rollout %s RR "$k" "$k"\ndone\n' % (pre, K, QM, arm, name) for pre, arm, name in RUNS]
    x.span('for k in "${ARMS[@]}"; do\n  READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE=', 'if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi', "".join(blocks), first=True)
    x.once('NAMES="I10RQ50KF5Hb20gA90 I10RQ50KF5Hb20gA80 I10RQ50Hb20gA90 I10RQ50Hb20gA80"', 'NAMES="%s"' % " ".join(n for _, _, n in RUNS))
    t = x.t.replace("GAPMARGIN10K", "GAPMARGINAGE10K")
    for bad in ("GM_ROOT", "GM_KIT", "GM_STOP_S", "small_gm", "gapmargin_queue.log", "READOUT=pinned", "ADD117 (GAPMARGIN", "Hb20gA90 RR\" ", "KF5Hb20gA"):
        if bad in t:
            raise SystemExit(TAG + "left over in the queue: " + bad)
    if t.count(" rollout I10") != 4 or t.count("VCLOSE=1 rollout") != 4 or t.count("PREROLL=30 ") != 2 or t.count("PREROLL=60 ") != 2 or t.count("GAPMARGINAGE10K_START.txt") < 4:
        raise SystemExit(TAG + "counts of the queue")
    return t


def build_reader(src):
    x = Text(src)
    x.t = "# GENERATED by make_gapmarginage_kit.py from read_gapmargin10k.py (sha256 prefix 0aab17c67eb60756) for card PREREG_V8F_PROOF_ADD119 (GAPMARGINAGE10K): tables and names only.\n" + x.t
    x.once('CARD, PRE = "PREREG_V8F_PROOF_ADD117", "GAPMARGIN10K"', 'CARD, PRE = "PREREG_V8F_PROOF_ADD119", "GAPMARGINAGE10K"')
    x.once('SETTINGS = ("K0", "CONV")', 'SETTINGS = ("K30", "K60")')
    x.span("SPEC = {\n", "# pairs, first minus second", SPEC_NEW)
    x.span("EXPECTED = {", "LEVEL_TOL = 1e-4", EXPECTED_NEW)
    x.once("K0: readout K, no pre-roll; CONV: pinned readout, no pre-roll\")", "K30: readout K, pre-roll of 30 steps (6 s); K60: readout K, pre-roll of 60 steps (12 s)\")")
    x.once('    for S in ("K0", "CONV"):\n        d = payload["settings"][S]\n        pre = "GAPMARGIN10K-" + S + "-"',
           '    for S in ("K30", "K60"):\n        d = payload["settings"][S]\n        pre = "GAPMARGINAGE10K-" + S + "-"')
    n = x.t.count("GAPMARGIN10K")
    if n < 5:
        raise SystemExit(TAG + "the reader names its card %d times" % n)
    t = x.t.replace("GAPMARGIN10K", "GAPMARGINAGE10K")
    if 'NEW_ARMS = {"K0"' in t or '"CONV": {"P": 0' in t or t.count('"K60": {"P": 60') != 1 or t.count("ZZ READ-GAPMARGINAGE10K") < 4:
        raise SystemExit(TAG + "left over or counts in the reader")
    return t


def build_driver(src):
    x = Text(src)
    x.once("GE=$V/gapmargin; K=$GE/kit_read; OUT=$GE/read; LT=$GE/lead_tests; GFM=$GE/gapfiles; K114=$V/presattr/kit_read",
           "GE=$V/gapmarginage; K=$GE/kit_read; OUT=$GE/read; LT=$GE/lead_tests; GFM=$V/gapmargin/gapfiles; K114=$V/presattr/kit_read")
    x.line_from("  for x in I10RQ50KF5Hb20gA90 ", "  for x in %s; do" % " ".join(READ_RUNS))
    if x.t.count("read_gapmargin10k.py") < 3:
        raise SystemExit(TAG + "the driver names the reader fewer than 3 times")
    t = x.t.replace("read_gapmargin10k.py", "read_gapmarginage10k.py").replace("GAPMARGIN10K", "GAPMARGINAGE10K").replace("podread_gapmargin10k.sh", "podread_gapmarginage10k.sh")
    t = t.replace("PREREG_V8F_PROOF_ADD117", "PREREG_V8F_PROOF_ADD119")
    for bad in ("GE=$V/gapmargin;", "read_gapmargin10k", "KF5Hb20gA90", "PROOF_ADD117"):
        if bad in t:
            raise SystemExit(TAG + "left over in the driver: " + bad)
    return t


BUILD = {"queue": (build_queue, os.path.join("queue", "pod_gapmarginage10k_queue.sh")), "reader": (build_reader, os.path.join("read", "read_gapmarginage10k.py")),
         "driver": (build_driver, os.path.join("read", "podread_gapmarginage10k.sh"))}


def make_all():
    """-> {key: text}; refuses a source whose hash is not the named one"""
    out = {}
    for key, (path, sha16) in SOURCES.items():
        raw = io.open(path, "rb").read().replace(b"\r\n", b"\n")
        got = hashlib.sha256(raw).hexdigest()[:16]
        if got != sha16:
            raise SystemExit(TAG + "%s source sha256 prefix %s, expected %s" % (key, got, sha16))
        out[key] = BUILD[key][0](raw.decode("utf-8"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args(argv)
    texts = make_all()
    for key, t in texts.items():
        p = os.path.join(a.out_dir, BUILD[key][1])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with io.open(p, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(t)
        print("ZZ MAKE-GAPMARGINAGE-KIT wrote %s sha256 prefix %s" % (BUILD[key][1].replace(os.sep, "/"), hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
