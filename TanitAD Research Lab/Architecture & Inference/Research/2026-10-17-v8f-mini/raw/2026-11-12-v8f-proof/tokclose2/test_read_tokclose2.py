"""test_read_tokclose2.py -- test of read_tokclose2.py on a synthetic pod folder built from the files of card ADD81 (whose values are on record in row 143), with deliberate regressions:
every regression has to turn the control stage red or change the name. Plain script: prints one line per case and ends with ZZ TEST-READ-TOKCLOSE2 <passed>/<total>."""
import copy, io, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "read_tokclose2.py")
ADD81 = os.path.join(HERE, "..", "tokclose", "pod")
T_TAGS = ["T10_s0", "T10_s1", "T25_s0", "T25_s1"]
COPIED = ["E10_s0", "E10_s1", "E25_s0", "E25_s1", "T10shuf_s0"]
HP_LONG = {"batch": 8, "lr": 1e-3, "clip_norm": 1.0, "max_evals": 24, "patience": 4}
HP_E = {"batch": 8, "lr": 1e-3, "clip_norm": 1.0, "max_evals": 100, "patience": 10}


def dump(path, obj):
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f)


def build(root, mut=None):
    pod = os.path.join(root, "pod")
    os.makedirs(os.path.join(pod, "fit_logs"))
    C = json.load(io.open(os.path.join(ADD81, "TOKCLOSE_CONTROLS.json"), encoding="utf-8"))
    R = json.load(io.open(os.path.join(ADD81, "TOKCLOSE_RESULT.json"), encoding="utf-8"))
    logs = {}
    for t in T_TAGS:
        logs[t] = {"hp_override": True, "hp": dict(HP_LONG), "shuffled": False, "loss_before": 1.5, "loss_at_stop": 1.2, "best_step": 900, "steps": 1500, "steps_per_eval": 100,
                   "stopped_by": "patience", "curve": [[100, 1.4]] * 15, "seconds": 10.0}
    for t in COPIED:
        logs[t] = {"hp_override": False, "hp": dict(HP_E), "shuffled": t.startswith("T10shuf")}
    names = sorted("tokclose_%s%s" % (t, s) for t in COPIED for s in (".pt", "_log.json", "_table.npz"))
    copies = {n: ("%064x" % (i + 1)) for i, n in enumerate(names)}
    sources = dict(copies)
    st = {"C": C, "R": R, "logs": logs, "copies": copies, "sources": sources}
    if mut:
        mut(st)
    dump(os.path.join(pod, "TOKCLOSE_CONTROLS.json"), st["C"])
    dump(os.path.join(pod, "TOKCLOSE_RESULT.json"), st["R"])
    for t, L in st["logs"].items():
        dump(os.path.join(pod, "fit_logs", "tokclose_%s_log.json" % t), L)
    for fn, d in (("c0_copies.sha256", st["copies"]), ("c0_sources.sha256", st["sources"])):
        with io.open(os.path.join(pod, fn), "w", encoding="utf-8", newline="\n") as f:
            f.write("".join("%s  %s\n" % (h, n) for n, h in sorted(d.items())))
    return pod


def run(args):
    p = subprocess.run([sys.executable, TOOL] + args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def stages(root, pod, after_controls=None):
    cf, nf = os.path.join(root, "controls.json"), os.path.join(root, "names.json")
    rc1, o1 = run(["controls", "--pod-dir", pod, "--out", cf])
    if after_controls:
        after_controls(pod)
    rc2, o2 = run(["names", "--pod-dir", pod, "--controls", cf, "--add81-result", os.path.join(ADD81, "TOKCLOSE_RESULT.json"),
                   "--add81-controls", os.path.join(ADD81, "TOKCLOSE_CONTROLS.json"), "--out", nf])
    names = json.load(io.open(nf, encoding="utf-8"))["names"] if os.path.exists(nf) else None
    return rc1, o1, rc2, o2, names


def set_ci(arm, lo, hi):
    def m(st):
        st["R"]["arms"][arm]["r_ci"] = [lo, hi]
    return m


def both(*ms):
    def m(st):
        for f in ms:
            f(st)
    return m


def better(st):
    st["R"]["names"]["T25"] = ["TOKCLOSE-T25-READABLE", "TOKCLOSE-T25-BETTER-THAN-OUTPUTS"]


def tamper_result(pod):
    p = os.path.join(pod, "TOKCLOSE_RESULT.json")
    R = json.load(io.open(p, encoding="utf-8"))
    R["arms"]["T25"]["r_pooled"] = 0.99
    dump(p, R)


CASES = [
    # name, mutation, hook after the control stage, expected rc of controls, expected: names refused?, a name that has to be present
    ("base", None, None, 0, False, ["TOKCLOSE2-T25L-NOT-BETTER", "TOKCLOSE2-FULL-FIT-NO-GAIN", "TOKCLOSE2-POOLED-TOKENS-BELOW-THE-REQUIREMENT", "TOKCLOSE2-T10L-READABLE"]),
    ("c0 one hash differs", lambda st: st["copies"].__setitem__("tokclose_E25_s1.pt", "f" * 64), None, 1, True, None),
    ("c0 one file missing", lambda st: st["copies"].pop("tokclose_T10shuf_s0_table.npz"), None, 1, True, None),
    ("c1 a T fit without the override", lambda st: st["logs"]["T25_s1"].__setitem__("hp_override", False), None, 1, True, None),
    ("c1 a T fit with the cap of card ADD81", lambda st: st["logs"]["T10_s0"]["hp"].__setitem__("max_evals", 8), None, 1, True, None),
    ("c1 a T fit with another patience", lambda st: st["logs"]["T10_s1"]["hp"].__setitem__("patience", 2), None, 1, True, None),
    ("c1 an E fit with an override", lambda st: st["logs"]["E10_s0"].__setitem__("hp_override", True), None, 1, True, None),
    ("c1 a T fit on shuffled labels", lambda st: st["logs"]["T25_s0"].__setitem__("shuffled", True), None, 1, True, None),
    ("t3 failed in the scorer", lambda st: st["C"]["controls"]["t3"].__setitem__("pass", False), None, 1, True, None),
    ("t0 failed in the scorer", lambda st: st["C"]["controls"]["t0"].__setitem__("pass", False), None, 1, True, None),
    ("scorer flag controls_all_pass false", lambda st: st["C"].__setitem__("controls_all_pass", False), None, 1, True, None),
    ("t2 planted loss judged (not the form of note N1)", lambda st: st["C"]["controls"]["t2"]["fits"]["T10shuf_s0"].__setitem__("loss_judged", True), None, 1, True, None),
    ("t2 failed in the scorer", lambda st: st["C"]["controls"]["t2"].__setitem__("pass", False), None, 1, True, None),
    ("result file changed after the controls", None, tamper_result, 0, True, None),
    ("T25 lower end at 0.9", set_ci("T25", 0.90, 0.97), None, 0, False, ["TOKCLOSE2-POOLED-TOKENS-REACH-THE-REQUIREMENT"]),
    ("T10 upper end at 0.9, T25 below", set_ci("T10", 0.5, 0.90), None, 0, False, ["TOKCLOSE2-REQUIREMENT-NOT-DECIDED"]),
    ("T25 upper end above 0.9, lower below", set_ci("T25", 0.6, 0.95), None, 0, False, ["TOKCLOSE2-REQUIREMENT-NOT-DECIDED"]),
    ("T25 better than outputs", better, None, 0, False, ["TOKCLOSE2-T25L-BETTER-THAN-OUTPUTS", "TOKCLOSE2-FULL-FIT-READS-MORE"]),
]


def main():
    base = os.environ.get("TC2_TEST_TMP") or tempfile.gettempdir()
    n_ok = 0
    for name, mut, hook, want_rc1, want_refused, want_names in CASES:
        with tempfile.TemporaryDirectory(dir=base, prefix="tc2t_") as root:
            pod = build(root, mut)
            rc1, o1, rc2, o2, names = stages(root, pod, hook)
            ok = rc1 == want_rc1
            ok = ok and ((rc2 != 0 and names is None and "REFUSED" in o2) if want_refused else (rc2 == 0 and names is not None))
            if want_names and names is not None:
                ok = ok and all(n in names for n in want_names)
            leaked = any(tok in o1 for tok in ("r_pooled", "mae", "0.468", "0.257"))
            ok = ok and not leaked
            n_ok += bool(ok)
            print("%-4s %-52s controls rc %d, names rc %d%s" % ("ok" if ok else "BAD", name, rc1, rc2, "" if ok else " :: " + (o1 + o2)[-300:].replace("\n", " | ")))
    print("ZZ TEST-READ-TOKCLOSE2 %d/%d" % (n_ok, len(CASES)))
    sys.exit(0 if n_ok == len(CASES) else 1)


if __name__ == "__main__":
    main()
