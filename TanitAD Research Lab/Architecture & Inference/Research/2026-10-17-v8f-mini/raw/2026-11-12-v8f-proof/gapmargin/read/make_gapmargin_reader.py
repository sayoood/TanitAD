#!/usr/bin/env python
"""make_gapmargin_reader.py -- card PREREG_V8F_PROOF_ADD117 (GAPMARGIN10K). Writes the reader and the pod driver of this card from those of card ADD114
(presattr/read/read_presattr10k.py and podread_presattr10k.sh) by asserted substitutions. The engine part (grid read, strategic grid, levels, pairs, families,
the argument rule, the identity control, the tests control, the counts control, the loader of rule logs) is kept as it is. Changed: the tables (settings K0 and
CONV, arms A90, A80, GEc, RLc, RR, the pairs), the reference run of a new run (the banked candidate run of its setting), the first control (the record of the
scaled gap files in place of the planted join), the rule-log control (each new run against its own scaled file, as control b2 of card ADD111), and the names.
The selftest of the source reader is tied to its own world: the generated reader refuses --selftest and the generated driver runs the selftest of the source
reader from its own kit folder before the controls stage.

  python make_gapmargin_reader.py [--print reader|driver]
"""
import argparse
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "presattr", "read"))
SRC = {"reader": ("read_presattr10k.py", "26a9097f22577750"), "driver": ("podread_presattr10k.sh", "04eb1dc2b7cbc889")}
OUT = {"reader": "read_gapmargin10k.py", "driver": "podread_gapmargin10k.sh"}
TAG = "ZZ MAKE-GAPMARGIN-READER STOP: "

TABLES = '''CARD, PRE = "PREREG_V8F_PROOF_ADD117", "GAPMARGIN10K"
SEEDS = (0, 1)
ARMS = ("A90", "A80", "GEc", "RLc", "RR")
MARGIN_ARMS = ("A90", "A80")                                        # larger a first
MARGIN_FILES = {"A90": "VGAPLE_A90", "A80": "VGAPLE_A80"}
CLOSE_TOL = 0.0045                                                  # card section 4: pooled unsafe rate at most that of the label form plus 0.0045
SETTINGS = ("K0", "CONV")
SPREAD = "R0Hb20"
# run names without the prefix I10; P = pre-roll steps; grid = the key of the engine's grid read
SPEC = {
    "K0": {"P": 0, "readout": "K", "every": 5, "grid": "full", "grid_file": "pinned",
           "runs": {"A90": "RQ50KF5Hb20gA90", "A80": "RQ50KF5Hb20gA80", "GEc": "RQ50KF5Hb20gGEc", "RLc": "RLcKF5Hb20", "RR": "RQ50KF5Hb20"}},
    "CONV": {"P": 0, "readout": "pinned", "every": None, "grid": "full", "grid_file": "pinned",
             "runs": {"A90": "RQ50Hb20gA90", "A80": "RQ50Hb20gA80", "GEc": "RQ50Hb20gGEc", "RLc": "RLcHb20", "RR": "RQ50Hb20"}},
}
NEW_ARMS = {"K0": ("A90", "A80"), "CONV": ("A90", "A80")}           # the runs of this card; the other cells are banked
# pairs, first minus second
PAIRS = {"A90-VS-GEc": ("A90", "GEc"), "A80-VS-GEc": ("A80", "GEc"), "A90-VS-RLc": ("A90", "RLc"), "A80-VS-RLc": ("A80", "RLc"), "GEc-VS-RLc": ("GEc", "RLc")}
STRAT_PAIRS = PAIRS
MEASURES = (("U", "UNSAFE"), ("C", "CONTACT"), ("F", "FALLING-BEHIND"))
EXPECTED = {                                                        # banked levels, (C, U, F), card section 5 control k3 (rows 179 and 181)
    "K0": {"GEc": (0.0000, 0.0498, 0.1651), "RR": (0.0296, 0.0738, 0.1682), "RLc": (0.0000, 0.0365, 0.1550)},
    "CONV": {"GEc": (0.0016, 0.0569, 0.2329), "RR": (0.0280, 0.0783, 0.2593), "RLc": (0.0016, 0.0320, 0.2243)},
}
'''

RULE_CHECKS = '''def margin_file(a, arm):
    return os.path.join(a.margin_dir, MARGIN_FILES[arm] + ".npz")


def rule_checks(a):
    """-> per-rollout dict (key S_arm_s<m>) of the rule-log checks of the eight new rollouts: the model reader's w2_one against the scaled gap file of the arm (as control b2 of card ADD111); the gap the
    reader itself logs, for the lead steps where the file has no output, comes from the rule log of the reference run RR of the setting at the same model seed; the rule log meta names the hash of the
    scaled file and the closing switch."""
    per = {}
    for S, arm in new_pairs():
        gf = margin_file(a, arm)
        for m in SEEDS:
            key = "%s_%s_s%d" % (S, arm, m)
            try:
                if not os.path.isfile(gf):
                    per[key] = {"PASS": False, "failures": ["no scaled gap file %s" % gf]}
                    continue
                gref = MD.gapfile_ref(gf)
                fsha = sha_file(gf)
                logs = load_logs_setting(a.pull, run_name(S, arm), m, S)
                ref_logs = load_logs_setting(a.pull, run_name(S, "RR"), m, S)
                v0 = MD.lead_multi(MD.rows_of(ref_logs))
                meta = MD.meta_of(logs[0])
                r = MD.w2_one(MD.rows_of(logs), gref, v0, meta_expect={"vis_closing": 1}, meta=meta)
                ms = str(meta.get("vis_gap_file_sha256") or "").lower()
                if not (len(ms) >= 16 and (fsha.startswith(ms) or ms.startswith(fsha))):
                    r["failures"] = list(r.get("failures", [])) + ["rule log meta names the gap file hash %s, the scaled file on disk has %s" % (ms[:16] or None, fsha[:16])]
                    r["PASS"] = False
                per[key] = r
            except (SystemExit, OSError, ValueError, KeyError) as ex:
                per[key] = {"PASS": False, "failures": ["cannot read the rollout, its reference or its gap file: %s: %s" % (type(ex).__name__, ex)]}
    return per


'''

CTL_E0 = '''def ctl_e0(rec, sha_base, shas):
    """e0 (control k0 of card ADD117) scaled gap files: the record GAPMARGIN_GAPFILES_RECORD.json has pass_k0 true, the scale of one gave the base arrays, every clause of both files is true, and the base
    file and both scaled files on disk have the hashes the record holds; the base file is the one of card ADD106 (sha256 begins 55de2f7c). shas {arm: sha256 of its scaled file on disk}."""
    bad = []
    rec = rec if isinstance(rec, dict) else {}
    if rec.get("pass_k0") is not True:
        bad.append("the record of the scaled gap files has pass_k0 %s" % rec.get("pass_k0"))
    if rec.get("scale_one_gives_the_base_arrays") is not True:
        bad.append("the record does not say that the scale of one gives the base arrays")
    want_b = (rec.get("base") or {}).get("sha256")
    if not sha_base or not want_b or str(sha_base).lower() != str(want_b).lower():
        bad.append("the base gap file on disk has sha256 %s, the record holds %s" % ((sha_base or "none")[:16], str(want_b or "none")[:16]))
    if not str(sha_base or "").startswith("55de2f7c"):
        bad.append("the base gap file is not the one of card ADD106 (sha256 %s)" % (sha_base or "none")[:16])
    newly = {}
    for arm in MARGIN_ARMS:
        nm = MARGIN_FILES[arm]
        cl = (rec.get("clauses") or {}).get(nm)
        if not isinstance(cl, dict) or not cl or not all(v is True for v in cl.values()):
            bad.append("%s: the clauses of the record are not all true" % nm)
        f = (rec.get("files") or {}).get(nm) or {}
        got = (shas or {}).get(arm)
        if not got or not f.get("sha256") or str(got).lower() != str(f.get("sha256")).lower():
            bad.append("%s on disk has sha256 %s, the record holds %s" % (nm, (got or "none")[:16], str(f.get("sha256") or "none")[:16]))
        want_a = {"A90": 0.90, "A80": 0.80}[arm]
        if f.get("a") is None or abs(float(f.get("a")) - want_a) > 1e-12:
            bad.append("%s: the record holds the scale %s, the card has %.2f" % (nm, f.get("a"), want_a))
        newly[arm] = f.get("n_rows_base_gap_above_60_and_scaled_at_most_60")
    return {"pass": not bad, "failures": bad, "values": {"n_rows_newly_in_range": newly, "counts": rec.get("counts")}}


'''

REF_ARM = '''def ref_arm(S, arm):
    """the reference run of a new run: the banked candidate run GEc of its setting (the same call with the base gap file)"""
    return "GEc"


'''

CTL_E7 = '''def ctl_e7(per):
    """e7 (control k7 of card ADD117) rule logs: each new run against its own scaled gap file (rule_checks)"""
    bad = []
    if sorted(per) != sorted(new_keys()):
        bad.append("rollouts read: %s, expected %s" % (sorted(per), sorted(new_keys())))
    for k, v in sorted(per.items()):
        bad += ["%s: %s" % (k, x) for x in v.get("failures", [])]
        if v.get("PASS") is not True and not v.get("failures"):
            bad.append("%s: not pass" % k)
    shares = {k: {"share_of_rule_steps_with_a_lead": v.get("share_of_rule_steps_with_a_lead"), "n_rows": v.get("n_rows"), "n_lead_rows": v.get("n_lead_rows"),
                  "share_of_lead_steps_with_nonzero_closing": v.get("share_of_lead_steps_with_nonzero_closing")} for k, v in per.items()}
    return {"pass": not bad and bool(per), "failures": bad, "values": {"rollout_shares": shares}}


'''

NAMES = '''def closes(payload, S, arm):
    """card section 4: the arm lowers the unsafe rate against the candidate (interval below 0) and its pooled unsafe rate is at most that of the label form plus CLOSE_TOL"""
    d = payload["settings"][S]
    return bool(d["pairs"][arm + "-VS-GEc"]["U"]["interval"][1] < 0.0 and d["levels"][arm]["U"] <= d["levels"]["RLc"]["U"] + CLOSE_TOL)


def closing_arm(payload, S):
    """the arm the name of the setting speaks of: of the arms that close without raising falling behind the one with the larger a; if none, of the arms that close the one with the larger a; else none"""
    d = payload["settings"][S]
    cl = [x for x in MARGIN_ARMS if closes(payload, S, x)]
    free = [x for x in cl if not d["pairs"][x + "-VS-GEc"]["F"]["interval"][0] > 0.0]
    return (free or cl or ["none"])[0]


def names_of(payload):
    """card section 4, written from the interval ends"""
    out = {}
    for S in SETTINGS:
        d = payload["settings"][S]
        pr = d["pairs"]
        p = "%s-%s-" % (PRE, S)
        for arm in MARGIN_ARMS:
            for mk, mw in MEASURES:
                iv = pr[arm + "-VS-GEc"][mk]["interval"]
                if iv[1] < 0.0:
                    out["%s_%s_%s" % (S, arm, mw)] = p + arm + "-LOWERS-" + mw
                elif iv[0] > 0.0:
                    out["%s_%s_%s" % (S, arm, mw)] = p + arm + "-RAISES-" + mw
                else:
                    out["%s_%s_%s" % (S, arm, mw)] = p + arm + "-" + mw + "-NOT-SEPARATED"
            iv = pr[arm + "-VS-RLc"]["U"]["interval"]
            out["%s_%s_LABEL" % (S, arm)] = p + arm + ("-UNSAFE-ABOVE-LABEL-FORM" if iv[0] > 0.0 else ("-UNSAFE-BELOW-LABEL-FORM" if iv[1] < 0.0 else "-UNSAFE-AT-LABEL-FORM"))
        cl = [x for x in MARGIN_ARMS if closes(payload, S, x)]
        if not cl:
            out[S + "_OVERALL"] = p + "MARGIN-DOES-NOT-CLOSE-UNSAFE"
        elif any(not pr[x + "-VS-GEc"]["F"]["interval"][0] > 0.0 for x in cl):
            out[S + "_OVERALL"] = p + "MARGIN-CLOSES-UNSAFE-AT-NO-SEPARATED-PRICE"
        else:
            out[S + "_OVERALL"] = p + "MARGIN-CLOSES-UNSAFE-AT-A-PRICE"
    return out


def derive_all(payload):
    """independent second derivation of every name from the signs of the intervals (written apart from names_of)"""
    out = {}
    word = {1: "RAISES", -1: "LOWERS"}
    lab = {1: "UNSAFE-ABOVE-LABEL-FORM", -1: "UNSAFE-BELOW-LABEL-FORM", 0: "UNSAFE-AT-LABEL-FORM"}
    for S in ("K0", "CONV"):
        d = payload["settings"][S]
        pre = "GAPMARGIN10K-" + S + "-"
        state = []                                                                         # per arm: (closes, raises falling behind)
        for arm in ("A90", "A80"):
            sg = {mk: _sg(d["pairs"][arm + "-VS-GEc"][mk]["interval"]) for mk in "UCF"}
            for mk, mw in (("U", "UNSAFE"), ("C", "CONTACT"), ("F", "FALLING-BEHIND")):
                out["%s_%s_%s" % (S, arm, mw)] = pre + arm + ("-" + word[sg[mk]] + "-" + mw if sg[mk] != 0 else "-" + mw + "-NOT-SEPARATED")
            out["%s_%s_LABEL" % (S, arm)] = pre + arm + "-" + lab[_sg(d["pairs"][arm + "-VS-RLc"]["U"]["interval"])]
            near = d["levels"][arm]["U"] <= d["levels"]["RLc"]["U"] + 0.0045               # the same arithmetic as closes(), so that the two routines cannot part at the bound
            state.append((sg["U"] == -1 and near, sg["F"] == 1))
        n_close = sum(1 for c, _ in state if c)
        n_free = sum(1 for c, r in state if c and not r)
        out[S + "_OVERALL"] = pre + ("MARGIN-DOES-NOT-CLOSE-UNSAFE" if n_close == 0 else ("MARGIN-CLOSES-UNSAFE-AT-NO-SEPARATED-PRICE" if n_free > 0 else "MARGIN-CLOSES-UNSAFE-AT-A-PRICE"))
    return out


'''

DRIVER_HEAD = [
    "# podread_gapmargin10k.sh -- pod-side read of card PREREG_V8F_PROOF_ADD117 (GAPMARGIN10K). CPU only (CUDA_VISIBLE_DEVICES=-1 is set here), no rollout.",
    "# Generated by make_gapmargin_reader.py from the driver of card ADD114. Stage controls: needs the done marker of the rollout queue of this card and no stop marker, links the four new run names (both seeds)",
    "#   and the banked runs of the two settings (and the spread run) into the pull folder of the readers, runs the selftest of the source reader of card ADD114 from its own kit folder first, then",
    "#   read_gapmargin10k.py --stage controls (writes $OUT/GAPMARGIN10K_CONTROLS.json). The time arguments of the tests control are filled here with stat -c %Y (the two lead test outputs in $GE/lead_tests),",
    "#   the first token of the first line of GAPMARGIN10K_START.txt (`<epoch seconds> <ISO time>Z pid <pid>`) and date +%s at the start of the stage.",
    "# Stage names: read_gapmargin10k.py --stage names on that controls file (the reader refuses unless all eight controls pass), writes $OUT/GAPMARGIN10K_RESULT.json, $OUT/read_names.log and $OUT/read_names.out.",
    "# The eight controls print as e0 to e7; they are the controls k0 to k7 of card ADD117 in that order. Never reads twice into one file. It deletes, moves and stops nothing.",
    "# Usage: bash podread_gapmargin10k.sh controls|names",
    "# Prints only the lines e<i> PASS|FAIL, the reader's last line and: ZZ PODREAD-GAPMARGIN10K <stage> DONE rc <n> <time>   (or ZZ PODREAD-GAPMARGIN10K STOP <why>).",
]


class Sub(object):
    def __init__(self, text):
        self.t = text.replace("\r\n", "\n")

    def once(self, a, b):
        if self.t.count(a) != 1:
            raise SystemExit(TAG + "expected once, found %d: %r" % (self.t.count(a), a[:90]))
        self.t = self.t.replace(a, b)

    def times(self, a, b, n):
        if self.t.count(a) != n:
            raise SystemExit(TAG + "expected %d times, found %d: %r" % (n, self.t.count(a), a[:90]))
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


def build_reader(src_text):
    S = Sub(src_text)
    S.span('CARD, PRE = "PREREG_V8F_PROOF_ADD114", "PRESATTR10K"', "LEVEL_TOL = 1e-4", TABLES)
    S.line_from("TIER = (", 'TIER = ("T1-kin/hb0.2 (kinematic rollout on the model\'s own actions, executor of record v8f_t1_eval_hb.py with the heading bound 0.2 1/m); '
                            'K0: readout K, no pre-roll; CONV: pinned readout, no pre-roll")')
    S.once('DONE_RE = re.compile(r"^ZZ PRESATTR10K DONE \\S+ all steps done$")', 'DONE_RE = re.compile(r"^ZZ GAPMARGIN10K DONE \\S+ all steps done$")')
    S.span("def rule_checks(a):", "def _hash_match(x, y):", RULE_CHECKS)
    S.span("def ctl_e0(rec, sha_jle, sha_base, sha_eval):", "def ctl_e1(", CTL_E0)
    S.span("def ref_arm(S, arm):", "def ctl_e2(", REF_ARM)
    S.span("def ctl_e7(gap_rec, la_sha_disk, per, jle_counts):", "def compute_controls(a, m, wsha):", CTL_E7)
    S.span("    jrec = read_json(a.jle_record)\n", "    shas = {", "    mrec = read_json(a.margin_record)\n    sb = sha_file(a.gap_file) if a.gap_file and os.path.isfile(a.gap_file) else None\n"
           "    c[\"e0\"] = ctl_e0(mrec, sb, {arm: (sha_file(margin_file(a, arm)) if os.path.isfile(margin_file(a, arm)) else None) for arm in MARGIN_ARMS})\n")
    S.line_from('    c["e7"] = ctl_e7(read_json(a.gap_record)', '    c["e7"] = ctl_e7(rc)')
    S.span("def names_of(payload):", "def all_names(payload):", NAMES)
    S.once('    e7 = cv.get("e7") or {}\n', '    e0, e7 = cv.get("e0") or {}, cv.get("e7") or {}\n    for S in SETTINGS:\n        L.append("closing arm %s: %s" % (S, closing_arm(payload, S)))\n')
    S.line_from('    L.append("count n_replaced_rows_above_gap_max', '    L.append("count rows with a base gap above 60 m and a scaled gap of at most 60 m, per arm: %s" % json.dumps(e0.get("n_rows_newly_in_range"), sort_keys=True))')
    S.span('              "gap_file_sha256": sha_file(a.gap_file)', "    return assemble_controls(controls, payload, inputs)",
           '              "gap_file_sha256": sha_file(a.gap_file) if os.path.isfile(a.gap_file) else None,\n'
           '              "margin_record_sha256": sha_file(a.margin_record) if os.path.isfile(a.margin_record) else None}\n')
    S.once('"la-file", "jle-file", "jle-record",', '"margin-dir", "margin-record",')
    S.span("        a.la_file = a.la_file or", "        a.grid_pinned = a.grid_pinned or",
           '        a.margin_dir = a.margin_dir or WS + "/gapmargin/gapfiles"\n        a.margin_record = a.margin_record or a.margin_dir + "/GAPMARGIN_GAPFILES_RECORD.json"\n')
    S.once('        a.tests_queue = a.tests_queue or WS + "/presattr/lead_tests/lead_pytest_queue.out"', '        a.tests_queue = a.tests_queue or WS + "/gapmargin/lead_tests/lead_pytest_queue.out"')
    S.once('        a.tests_read = a.tests_read or WS + "/presattr/lead_tests/lead_pytest_read.out"', '        a.tests_read = a.tests_read or WS + "/gapmargin/lead_tests/lead_pytest_read.out"')
    S.once("    if a.selftest:\n        return selftest()\n",
           '    if a.selftest:\n        print("REFUSED: the selftest is that of the source reader of card ADD114; run it from that file")\n        return 2\n')
    n = S.t.count("READ-PRESATTR10K")
    if n < 4:
        raise SystemExit(TAG + "the reader's own last-line tag was found %d times" % n)
    S.t = S.t.replace("READ-PRESATTR10K", "READ-GAPMARGIN10K")
    first = S.t.split("\n", 1)
    S.t = ("# GENERATED by make_gapmargin_reader.py from read_presattr10k.py (sha256 prefix %s) for card PREREG_V8F_PROOF_ADD117 (GAPMARGIN10K). The docstring and the comments of the kept\n"
           "# functions are those of the source and speak of its card; the tables, ref_arm, rule_checks, ctl_e0, ctl_e7, the names and the two changed calls in compute_controls are those of card ADD117.\n"
           "# The functions of the source's own world (make_world, mk_payload, selftest, pl_check, load_join_arrays) are kept unused.\n" % SRC["reader"][1]) + first[0] + "\n" + first[1]
    t = S.t
    for must in ('CARD, PRE = "PREREG_V8F_PROOF_ADD117", "GAPMARGIN10K"', "def closing_arm(payload, S):", "def margin_file(a, arm):", '    c["e7"] = ctl_e7(rc)\n', "LEVEL_TOL = 1e-4"):
        if t.count(must) != 1:
            raise SystemExit(TAG + "not once after the substitutions: " + must[:50])
    for bad in ('"PRESATTR10K"', '"PRESATTR10K-"', "ZZ PRESATTR10K DONE", 'SETTINGS = ("K0", "K30")'):
        if bad in t:
            raise SystemExit(TAG + "left over: " + bad)
    for word, n in (("a.jle_record", 1), ("a.la_file", 1), ("a.jle_file", 1), ("jrec", 2)):          # only inside the unused synthetic world of the source (make_world)
        if t.count(word) != n:
            raise SystemExit(TAG + "%s found %d times, expected %d (the unused world of the source only)" % (word, t.count(word), n))
    return t


def build_driver(src_text):
    S = Sub(src_text)
    lines = S.t.split("\n")
    if lines[0] != "#!/usr/bin/env bash" or "set -u" not in lines[:20]:
        raise SystemExit(TAG + "the head of the driver is not the expected one")
    k = lines.index("set -u")
    if not all(x.startswith("#") for x in lines[1:k]):
        raise SystemExit(TAG + "the head of the driver has a line that is no comment")
    S.t = "\n".join([lines[0]] + DRIVER_HEAD + lines[k:])
    S.once("GE=$V/presattr; K=$GE/kit_read; OUT=$GE/read; LT=$GE/lead_tests; PJ=$GE/joins",
           "GE=$V/gapmargin; K=$GE/kit_read; OUT=$GE/read; LT=$GE/lead_tests; GFM=$GE/gapfiles; K114=$V/presattr/kit_read")
    S.once('for f in "$K/read_presattr10k.py" "$MODEL"; do', 'for f in "$K/read_presattr10k.py" "$K114/read_presattr10k.py" "$MODEL"; do')
    S.once('"$PY" "$K/read_presattr10k.py" --selftest', '"$PY" "$K114/read_presattr10k.py" --selftest')
    S.once('"$LT/lead_pytest_read.out" "$W/longroll/grid_p30.json" "$PINNED"', '"$LT/lead_pytest_read.out" "$PINNED"')
    S.times('"$LAF" "$GREC" "$PJ/JLE.npz" "$PJ/JLE_RECORD.json"', '"$GFM/VGAPLE_A90.npz" "$GFM/VGAPLE_A80.npz" "$GFM/GAPMARGIN_GAPFILES_RECORD.json"', 2)
    S.once('--gap-file "$GAPF" --gap-record "$GREC" --la-file "$LAF" --jle-file "$PJ/JLE.npz" --jle-record "$PJ/JLE_RECORD.json" --eval-join "$JOIN"',
           '--gap-file "$GAPF" --margin-dir "$GFM" --margin-record "$GFM/GAPMARGIN_GAPFILES_RECORD.json"')
    S.line_from("  for x in I10RLcKF5Hb20 ", "  for x in I10RQ50KF5Hb20gA90 I10RQ50KF5Hb20gA80 I10RQ50KF5Hb20gGEc I10RLcKF5Hb20 I10RQ50KF5Hb20 "
                                               "I10RQ50Hb20gA90 I10RQ50Hb20gA80 I10RQ50Hb20gGEc I10RLcHb20 I10RQ50Hb20 I10R0Hb20; do")
    t = S.t.replace('"$K/read_presattr10k.py"', '"$K/read_gapmargin10k.py"').replace("PRESATTR10K", "GAPMARGIN10K")
    if t.count('"$K114/read_presattr10k.py"') != 2 or t.count('"$K/read_gapmargin10k.py"') < 4:
        raise SystemExit(TAG + "driver counts: source reader %d, generated reader %d" % (t.count('"$K114/read_presattr10k.py"'), t.count('"$K/read_gapmargin10k.py"')))
    for bad in ("grid_p30", "PRESATTR10K", "GE=$V/presattr;", "$PJ", "JLE", "aLA", "pLE", "--la-file"):
        if bad in t:
            raise SystemExit(TAG + "driver left over: " + bad)
    return t


BUILD = {"reader": build_reader, "driver": build_driver}


def make(kind):
    name, sha16 = SRC[kind]
    raw = io.open(os.path.join(SRC_DIR, name), "rb").read().replace(b"\r\n", b"\n")
    got = hashlib.sha256(raw).hexdigest()[:16]
    if got != sha16:
        raise SystemExit(TAG + "%s: source sha256 prefix %s, expected %s" % (name, got, sha16))
    return BUILD[kind](raw.decode("utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", choices=("reader", "driver"), default="")
    a = ap.parse_args(argv)
    if a.print:
        sys.stdout.write(make(a.print))
        return 0
    for kind in ("reader", "driver"):
        t = make(kind)
        with io.open(os.path.join(HERE, OUT[kind]), "x", encoding="utf-8", newline="\n") as fh:
            fh.write(t)
        print("ZZ MAKE-GAPMARGIN-READER DONE %s sha256 prefix %s" % (OUT[kind], hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
