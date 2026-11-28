"""test_tokgrid.py -- tests of the kit of card PREREG_V8F_PROOF_ADD87 (TOKGRID): the generator and the generated tool, the chain script, the pair tool and the reader.
Plain script: one line per check, last line ZZ TEST-TOKGRID <passed>/<total>. Synthetic data only, CPU only.
Run: PYCAP_MB=3000 bash ops/pycap.sh tokgrid/test_tokgrid.py   (from the folder of the fork; the child jobs it starts run plain inside the one lock).
Temporary folders: tokgrid/tmp_work only."""
import copy, importlib.util, io, json, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, "tmp_work")
os.makedirs(TMP, exist_ok=True)
os.environ.setdefault("TOKCLOSE_BUILD_DIR", os.path.join(HERE, "..", "integ10k"))
for _p in (HERE, os.path.join(HERE, "..", "closeread"), os.path.join(HERE, "..", "integ10k")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True
PY = sys.executable
BASH = "C:/Program Files/Git/usr/bin/bash.exe"
ADD81 = os.path.join(HERE, "..", "tokclose", "pod")
READER = os.path.join(HERE, "read_tokgrid.py")
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print("%-4s %s%s" % ("ok" if cond else "BAD", name, "" if cond else " :: " + str(detail)[-300:].replace("\n", " | ")), flush=True)


def loadmod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def run(args, env=None, cwd=None):
    e = dict(os.environ)
    e.update(env or {})
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    e["TMP"] = e["TEMP"] = TMP
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", env=e, cwd=cwd)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# ============================================================================================ generators and the chain script
def part_generators():
    rc, out = run([PY, os.path.join(HERE, "make_tokclose_fit_g8.py")])
    check("generator of the tool runs (12 edits)", rc == 0 and "12 edits" in out, out)
    src = io.open(os.path.join(HERE, "tokclose_fit_g8.py"), encoding="utf-8").read()
    try:
        compile(src, "tokclose_fit_g8.py", "exec")
        ok = True
    except SyntaxError as e:
        ok = False
    check("the generated tool compiles", ok)
    # the generator refuses when the parent differs
    g = loadmod("mk_g8", os.path.join(HERE, "make_tokclose_fit_g8.py"))
    with tempfile.TemporaryDirectory(dir=TMP, prefix="gen_") as d:
        bad = os.path.join(d, "parent.py")
        raw = io.open(g.PARENT, "rb").read()
        io.open(bad, "wb").write(raw + b"\n#")
        g.PARENT = bad
        try:
            g.main()
            refused = False
        except SystemExit as e:
            refused = "REFUSED" in str(e)
    check("the tool generator refuses a parent with another hash", refused)
    # an old text that occurs twice, or not at all, is refused (and nothing is written)
    for label, extra in (("twice", ("import ", "import ")), ("not at all", ("THIS TEXT IS NOT IN THE PARENT", "x"))):
        g2 = loadmod("mk_g8b", os.path.join(HERE, "make_tokclose_fit_g8.py"))
        g2.EDITS = list(g2.EDITS) + [extra]
        try:
            g2.main()
            refused = False
        except SystemExit as e:
            refused = "REFUSED" in str(e) and "matches" in str(e)
        check("the tool generator refuses an edit whose old text occurs %s" % label, refused)
    rc, out = run([PY, os.path.join(HERE, "make_run_tokgrid.py")])
    check("generator of the chain script runs (16 edits)", rc == 0 and "16 edits" in out, out)
    sh = os.path.join(HERE, "run_tokgrid.sh")
    rc, out = run([BASH, "-n", sh])
    check("bash -n run_tokgrid.sh", rc == 0, out)
    rc, out = run([BASH, sh], env={"DRY": "1"})
    n_slot = out.count("SLOTWAIT before fit ")
    n_cp = sum(1 for l in out.split("\n") if l.startswith("CMD: cp "))
    fit_lines = [l for l in out.split("\n") if l.startswith("CMD: timeout") and " fit --arm " in l]
    check("DRY: the whole command list (self-test, g0, 12 copies, 5 fits each after a slot wait, score, pair)",
          rc == 0 and "DRY run complete" in out and n_slot == 5 and n_cp == 12 and len(fit_lines) == 5 and "--selftest" in out and "g0_score" in out and "pair_tokgrid.py" in out
          and "TOKCLOSE_T_GRID=pool8x20" in out and "--allow-hp-override" in out and "TOKCLOSE2_DONE.txt" in out and sum(1 for l in fit_lines if "--shuffle-closing-labels" in l) == 1
          and "tokclose_fit_g8.py" in out and "tokclose_fit.py" not in out, out[-400:])
    text = io.open(sh, encoding="utf-8").read()
    code = "\n".join(l for l in text.split("\n") if not l.lstrip().startswith("#"))
    bad_words = [w for w in ("rm", "mv", "kill", "pkill", "pgrep", "ps") if re.search(r"(^|[;&|(\s])" + w + r"\s", code)]
    check("the chain script removes, moves, stops and lists nothing", not bad_words, bad_words)
    with tempfile.TemporaryDirectory(dir=TMP, prefix="gen_") as d:
        gs = loadmod("mk_run", os.path.join(HERE, "make_run_tokgrid.py"))
        bad = os.path.join(d, "run.sh")
        io.open(bad, "wb").write(io.open(gs.PARENT, "rb").read() + b"\n#")
        gs.PARENT = bad
        try:
            gs.main()
            refused = False
        except SystemExit as e:
            refused = "REFUSED" in str(e)
    check("the chain generator refuses a parent with another hash", refused)
    if os.path.exists(os.path.join(HERE, "run_tokgrid.sh")):
        pass


# ============================================================================================ the tool: self-test, g0 basis, the grid in the fit log, the pair tool
def part_tool():
    tool = os.path.join(HERE, "tokclose_fit_g8.py")
    rc, out = run([PY, tool, "--selftest"])
    last = [l for l in out.split("\n") if l.strip()][-1:] or [""]
    nfail = sum(1 for l in out.split("\n") if l.startswith("FAIL"))
    check("the self-test of the generated tool is green (last line %s)" % last[0][:60], rc == 0 and last[0].startswith("ZZ TOKCLOSE-SELFTEST rc 0") and nfail == 0, out)
    G = loadmod("tgg8", tool)
    P = loadmod("tgparent", os.path.join(HERE, "..", "tokclose", "tokclose_fit.py"))
    with tempfile.TemporaryDirectory(dir=TMP, prefix="w_") as root:
        world = G.make_world(root)
        # coarse fits with the PARENT tool (the card-ADD86 style fit folder), the same folder scored by both
        os.environ.pop("TOKCLOSE_T_GRID", None)
        rc_f = P.run_world_fits(world)
        check("parent fits on the synthetic world (9 fits)", rc_f == 0)
        outs = {}
        for nm, mod in (("parent", P), ("gen", G)):
            args = G.score_args(world, out="score_" + nm)
            rc_s = mod.run_score(args, log=lambda *z: None)
            outs[nm] = (rc_s, json.load(io.open(os.path.join(root, "score_" + nm, "TOKCLOSE_RESULT.json"), encoding="utf-8")) if rc_s == 0 else None)
        ctl = {nm: json.load(io.open(os.path.join(root, "score_" + nm, "TOKCLOSE_CONTROLS.json"), encoding="utf-8")) for nm in ("parent", "gen")}
        check("g0 basis: parent and generated tool give the same score result on one fit folder (arms equal, exactly)",
              outs["parent"][0] == 0 and outs["gen"][0] == 0 and outs["parent"][1]["arms"] == outs["gen"][1]["arms"] and outs["parent"][1]["names"] == outs["gen"][1]["names"]
              and ctl["parent"]["controls_all_pass"] and ctl["gen"]["controls_all_pass"])
        # fine-grid fits: the log names the grid and the token count; E arms read the coarse grid whatever the variable says
        prev = {k: os.environ.get(k) for k in ("TOKCLOSE_HP_OVERRIDE", "TOKCLOSE_T_GRID")}
        os.environ["TOKCLOSE_HP_OVERRIDE"] = json.dumps(G.TEST_HP)
        os.environ["TOKCLOSE_T_GRID"] = "pool8x20"
        try:
            rcs = [G.run_fit(G.fit_args(world, arm, k, out="fine"), lambda *z: None) for k in (0, 1) for arm in ("T10", "T25")]
            rce = G.run_fit(G.fit_args(world, "E10", 0, out="fine_e"), lambda *z: None)
            os.environ["TOKCLOSE_T_GRID"] = "pool9x9"
            rcb = G.run_fit(G.fit_args(world, "T10", 0, out="fine_bad"), lambda *z: None)
            rcb_e = G.run_fit(G.fit_args(world, "E10", 0, out="fine_bad_e"), lambda *z: None)
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        L = json.load(io.open(os.path.join(root, "fine", "tokclose_T25_s1_log.json"), encoding="utf-8"))
        Le = json.load(io.open(os.path.join(root, "fine_e", "tokclose_E10_s0_log.json"), encoding="utf-8"))
        Lc = json.load(io.open(os.path.join(root, "fits", "tokclose_T25_s1_log.json"), encoding="utf-8"))
        n0 = world["dims"]["n_tok"]
        check("a fine-grid fit logs grid pool8x20 and %d tokens (a fit of the parent tool has no grid key and %d tokens)" % (4 * n0, n0),
              rcs == [0, 0, 0, 0] and L["grid"] == "pool8x20" and L["dims"]["n_tok"] == 4 * n0 and Lc.get("grid", "pool4x10") == "pool4x10" and Lc["dims"]["n_tok"] == n0, [L.get("grid"), L["dims"], Lc.get("grid")])
        check("an E fit reads the coarse grid when the variable names the fine grid", rce == 0 and Le["grid"] == "pool4x10" and Le["dims"]["n_tok"] == n0, Le.get("grid"))
        check("a grid other than the two is REFUSED by the fit (T arm), and not by an E arm", rcb == 2 and rcb_e == 0 and not os.path.exists(os.path.join(root, "fine_bad")))
        # the no-leak control reads the grid of each head: the fine heads score with the fine tokens
        heads = {"T10_s0": os.path.join(root, "fine", "tokclose_T10_s0.pt"), "T25_s1": os.path.join(root, "fine", "tokclose_T25_s1.pt")}
        t3 = G.control_t3(heads, os.path.join(root, "evalcache"), world["eval"], G.default_build_dir(), 3, "cpu")
        check("control t3 on fine-grid heads loads the fine tokens and passes", t3["pass"] and len(t3["heads"]) == 2, t3)
        # the pair tool: fine folder of four T fits against the coarse folder of the same arms
        pod_fine = os.path.join(root, "fine")
        out_dir = os.path.join(root, "pair")
        args = [PY, os.path.join(HERE, "pair_tokgrid.py"), "--fit-dir-fine", pod_fine, "--fit-dir-coarse", os.path.join(root, "fits"), "--dump-dir", os.path.join(root, "dump"),
                "--episode-map", os.path.join(root, "episode_map.json"), "--eval-cache", os.path.join(root, "evalcache"), "--join-eval", os.path.join(root, "join_eval.npz"), "--out-dir", out_dir, "--boot", "200"]
        rc, out = run(args)
        pj = json.load(io.open(os.path.join(out_dir, "PAIR_TOKGRID.json"), encoding="utf-8")) if os.path.exists(os.path.join(out_dir, "PAIR_TOKGRID.json")) else {}
        ok = rc == 0 and out.strip() == "ZZ PAIR-TOKGRID DONE" and all(k in pj for k in ("H25", "H10", "T25_fine_error_per_seed"))
        ok = ok and all(len(pj[h]["ci"]) == 2 and pj[h]["ci"][0] <= pj[h]["paired_diff_abs_err_fine_minus_coarse"] <= pj[h]["ci"][1] for h in ("H25", "H10"))
        ok = ok and sorted(pj["T25_fine_error_per_seed"]) == ["0", "1"] and all(v["sd_error"] > 0 and v["n_pairs"] > 10 for v in pj["T25_fine_error_per_seed"].values())
        ok = ok and pj["grids"]["T25_s0_fine"]["grid"] == "pool8x20" and pj["grids"]["T25_s0_coarse"]["grid"] == "pool4x10"
        check("pair tool: writes the file, prints only the marker, the difference lies in its interval, sd and lag-1 autocorrelation per seed", ok, out + str(pj)[:200])
        rc2, out2 = run(args)
        check("pair tool refuses when its output folder exists", rc2 == 2 and "REFUSED" in out2, out2)
        # identical folders: the difference is exactly zero
        out_dir2 = os.path.join(root, "pair_same")
        a2 = list(args)
        a2[a2.index("--fit-dir-fine") + 1] = os.path.join(root, "fits")
        a2[a2.index("--out-dir") + 1] = out_dir2
        # the coarse folder holds T fits of both seeds
        rc3, out3 = run(a2)
        pj2 = json.load(io.open(os.path.join(out_dir2, "PAIR_TOKGRID.json"), encoding="utf-8")) if rc3 == 0 else {}
        check("pair tool: the same folder against itself gives a difference of exactly 0 with interval [0, 0]",
              rc3 == 0 and all(pj2[h]["paired_diff_abs_err_fine_minus_coarse"] == 0.0 and pj2[h]["ci"] == [0.0, 0.0] for h in ("H25", "H10")), out3)


# ============================================================================================ the reader
T_TAGS = ["T10_s0", "T10_s1", "T25_s0", "T25_s1"]
E_TAGS = ["E10_s0", "E10_s1", "E25_s0", "E25_s1"]
PL = "T10shuf_s0"
HP_T = {"batch": 8, "lr": 1e-3, "clip_norm": 1.0, "max_evals": 24, "patience": 4}
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
    for t in T_TAGS + [PL]:
        logs[t] = {"hp_override": True, "hp": dict(HP_T), "shuffled": t == PL, "grid": "pool8x20", "dims": {"n_tok": 160}, "loss_before": 1.5, "loss_at_stop": 1.2, "best_step": 900,
                   "steps": 1500, "steps_per_eval": 100, "stopped_by": "patience", "curve": [[100, 1.4]] * 15, "seconds": 10.0}
    for t in E_TAGS:
        logs[t] = {"hp_override": False, "hp": dict(HP_E), "shuffled": False}
    names = sorted("tokclose_%s%s" % (t, s) for t in E_TAGS for s in (".pt", "_log.json", "_table.npz"))
    copies = {n: ("%064x" % (i + 1)) for i, n in enumerate(names)}
    pair = {"H25": {"paired_diff_abs_err_fine_minus_coarse": -0.02, "ci": [-0.05, 0.01], "n_frames": 1000}, "H10": {"paired_diff_abs_err_fine_minus_coarse": 0.01, "ci": [-0.02, 0.04]},
            "T25_fine_error_per_seed": {"0": {"sd_error": 1.1, "lag1_autocorr": 0.9, "n_pairs": 500}, "1": {"sd_error": 1.2, "lag1_autocorr": 0.8, "n_pairs": 500}}}
    st = {"C": C, "R": R, "G0": copy.deepcopy(R), "R86": copy.deepcopy(R), "C86": {"all_pass": True}, "pair": pair, "logs": logs, "copies": copies, "sources": dict(copies)}
    if mut:
        mut(st)
    dump(os.path.join(pod, "TOKCLOSE_CONTROLS.json"), st["C"])
    dump(os.path.join(pod, "TOKCLOSE_RESULT.json"), st["R"])
    dump(os.path.join(pod, "g0_TOKCLOSE_RESULT.json"), st["G0"])
    dump(os.path.join(pod, "PAIR_TOKGRID.json"), st["pair"])
    dump(os.path.join(root, "add86_result.json"), st["R86"])
    dump(os.path.join(root, "add86_controls.json"), st["C86"])
    for t, L in st["logs"].items():
        dump(os.path.join(pod, "fit_logs", "tokclose_%s_log.json" % t), L)
    for fn, d in (("c0_copies.sha256", st["copies"]), ("c0_sources.sha256", st["sources"])):
        with io.open(os.path.join(pod, fn), "w", encoding="utf-8", newline="\n") as f:
            f.write("".join("%s  %s\n" % (h, n) for n, h in sorted(d.items())))
    return pod


def stages(root, pod, after_controls=None):
    cf, nf = os.path.join(root, "controls.json"), os.path.join(root, "names.json")
    common = ["--add86-result", os.path.join(root, "add86_result.json"), "--add86-controls", os.path.join(root, "add86_controls.json")]
    rc1, o1 = run([PY, READER, "controls", "--pod-dir", pod, "--out", cf] + common)
    if after_controls:
        after_controls(root, pod)
    rc2, o2 = run([PY, READER, "names", "--pod-dir", pod, "--controls", cf, "--out", nf] + common)
    names = json.load(io.open(nf, encoding="utf-8"))["names"] if os.path.exists(nf) else None
    return rc1, o1, rc2, o2, names


def set_ci(arm, lo, hi):
    return lambda st: st["R"]["arms"][arm].__setitem__("r_ci", [lo, hi])


def set_pair(lo, hi):
    return lambda st: st["pair"]["H25"].__setitem__("ci", [lo, hi])


def better(st):
    st["R"]["names"]["T25"] = ["TOKCLOSE-T25-READABLE", "TOKCLOSE-T25-BETTER-THAN-OUTPUTS"]


def tamper_result(root, pod):
    p = os.path.join(pod, "TOKCLOSE_RESULT.json")
    R = json.load(io.open(p, encoding="utf-8"))
    R["arms"]["T25"]["r_pooled"] = 0.99
    dump(p, R)


def tamper_pair(root, pod):
    p = os.path.join(pod, "PAIR_TOKGRID.json")
    R = json.load(io.open(p, encoding="utf-8"))
    R["H25"]["ci"] = [-0.5, -0.4]
    dump(p, R)


def tamper_add86(root, pod):
    p = os.path.join(root, "add86_result.json")
    R = json.load(io.open(p, encoding="utf-8"))
    R["arms"]["T25"]["r_pooled"] = 0.5
    dump(p, R)


def g0_value(st):
    st["G0"]["arms"]["T25"]["r_ci"][1] += 1e-12


def g0_name(st):
    st["G0"]["arms"]["T25"]["names"] = ["x", "y"]


def g0_arm_missing(st):
    st["G0"]["arms"].pop("E25")


def g0_nested_diff(st):
    st["G0"]["arms"]["T10"]["r_per_seed"][1] += 1e-9


BASE_NAMES = ["TOKGRID-T25G-NOT-BETTER", "TOKGRID-T10G-READABLE", "TOKGRID-BELOW-THE-REQUIREMENT", "TOKGRID-FINER-GRID-NOT-SEPARATED"]
# name, mutation, hook after the control stage, expected rc of controls, names refused?, names that have to be present, names that must be absent
CASES = [
    ("base", None, None, 0, False, BASE_NAMES, []),
    ("g0 one value differs by 1e-12", g0_value, None, 1, True, None, []),
    ("g0 one value in a per-seed list differs", g0_nested_diff, None, 1, True, None, []),
    ("g0 a name differs", g0_name, None, 1, True, None, []),
    ("g0 an arm is missing", g0_arm_missing, None, 1, True, None, []),
    ("g0 the file of card ADD86 differs from the g0 score", lambda st: st["R86"]["arms"]["T10"].__setitem__("mae", 9.0), None, 1, True, None, []),
    ("c0 one hash differs", lambda st: st["copies"].__setitem__("tokclose_E25_s1.pt", "f" * 64), None, 1, True, None, []),
    ("c0 one file missing", lambda st: st["copies"].pop("tokclose_E10_s0_table.npz"), None, 1, True, None, []),
    ("c1 a T fit on the coarse grid", lambda st: st["logs"]["T25_s1"].__setitem__("grid", "pool4x10"), None, 1, True, None, []),
    ("c1 a T fit with 40 tokens", lambda st: st["logs"]["T10_s0"]["dims"].__setitem__("n_tok", 40), None, 1, True, None, []),
    ("c1 a T fit without the grid key", lambda st: st["logs"]["T10_s1"].pop("grid"), None, 1, True, None, []),
    ("c1 a T fit without the override", lambda st: st["logs"]["T25_s0"].__setitem__("hp_override", False), None, 1, True, None, []),
    ("c1 a T fit with the cap of card ADD81", lambda st: st["logs"]["T10_s0"]["hp"].__setitem__("max_evals", 8), None, 1, True, None, []),
    ("c1 a T fit with another patience", lambda st: st["logs"]["T10_s1"]["hp"].__setitem__("patience", 2), None, 1, True, None, []),
    ("c1 the planted fit not on the fine grid", lambda st: st["logs"][PL].__setitem__("grid", "pool4x10"), None, 1, True, None, []),
    ("c1 the planted fit not shuffled", lambda st: st["logs"][PL].__setitem__("shuffled", False), None, 1, True, None, []),
    ("c1 a real T fit on shuffled labels", lambda st: st["logs"]["T25_s0"].__setitem__("shuffled", True), None, 1, True, None, []),
    ("c1 an E fit with an override", lambda st: st["logs"]["E10_s0"].__setitem__("hp_override", True), None, 1, True, None, []),
    ("t3 failed in the scorer", lambda st: st["C"]["controls"]["t3"].__setitem__("pass", False), None, 1, True, None, []),
    ("t0 failed in the scorer", lambda st: st["C"]["controls"]["t0"].__setitem__("pass", False), None, 1, True, None, []),
    ("t1 failed in the scorer", lambda st: st["C"]["controls"]["t1"].__setitem__("pass", False), None, 1, True, None, []),
    ("t4 failed in the scorer", lambda st: st["C"]["controls"]["t4"].__setitem__("pass", False), None, 1, True, None, []),
    ("scorer flag controls_all_pass false", lambda st: st["C"].__setitem__("controls_all_pass", False), None, 1, True, None, []),
    ("t2 planted loss judged (not the form of note N1)", lambda st: st["C"]["controls"]["t2"]["fits"][PL].__setitem__("loss_judged", True), None, 1, True, None, []),
    ("t2 failed in the scorer", lambda st: st["C"]["controls"]["t2"].__setitem__("pass", False), None, 1, True, None, []),
    ("result file changed after the controls", None, tamper_result, 0, True, None, []),
    ("pair file changed after the controls", None, tamper_pair, 0, True, None, []),
    ("file of card ADD86 changed after the controls", None, tamper_add86, 0, True, None, []),
    ("FINER-GRID reads more (upper end below 0)", set_pair(-0.2, -0.05), None, 0, False, ["TOKGRID-FINER-GRID-READS-MORE"], ["TOKGRID-FINER-GRID-NOT-SEPARATED", "TOKGRID-FINER-GRID-READS-LESS"]),
    ("FINER-GRID reads less (lower end above 0)", set_pair(0.05, 0.2), None, 0, False, ["TOKGRID-FINER-GRID-READS-LESS"], ["TOKGRID-FINER-GRID-READS-MORE"]),
    ("FINER-GRID not separated", set_pair(-0.05, 0.05), None, 0, False, ["TOKGRID-FINER-GRID-NOT-SEPARATED"], ["TOKGRID-FINER-GRID-READS-MORE", "TOKGRID-FINER-GRID-READS-LESS"]),
    ("FINER-GRID upper end at 0 is not separated", set_pair(-0.2, 0.0), None, 0, False, ["TOKGRID-FINER-GRID-NOT-SEPARATED"], ["TOKGRID-FINER-GRID-READS-MORE"]),
    ("FINER-GRID lower end at 0 is not separated", set_pair(0.0, 0.2), None, 0, False, ["TOKGRID-FINER-GRID-NOT-SEPARATED"], ["TOKGRID-FINER-GRID-READS-LESS"]),
    ("controls of card ADD86 not all passed: the pair has no name", lambda st: st["C86"].__setitem__("all_pass", False), None, 0, False, ["TOKGRID-T10G-READABLE", "TOKGRID-BELOW-THE-REQUIREMENT"],
     ["TOKGRID-FINER-GRID-NOT-SEPARATED", "TOKGRID-FINER-GRID-READS-MORE", "TOKGRID-FINER-GRID-READS-LESS"]),
    ("requirement: T25G lower end at 0.9", set_ci("T25", 0.90, 0.97), None, 0, False, ["TOKGRID-REACHES-THE-REQUIREMENT"], ["TOKGRID-BELOW-THE-REQUIREMENT", "TOKGRID-REQUIREMENT-NOT-DECIDED"]),
    ("requirement: T10G upper end at 0.9, T25G below", set_ci("T10", 0.5, 0.90), None, 0, False, ["TOKGRID-REQUIREMENT-NOT-DECIDED"], ["TOKGRID-BELOW-THE-REQUIREMENT"]),
    ("requirement: T25G upper end above 0.9, lower below", set_ci("T25", 0.6, 0.95), None, 0, False, ["TOKGRID-REQUIREMENT-NOT-DECIDED"], ["TOKGRID-REACHES-THE-REQUIREMENT"]),
    ("requirement: both upper ends below 0.9", lambda st: (set_ci("T25", 0.6, 0.85)(st), set_ci("T10", 0.4, 0.89)(st)), None, 0, False, ["TOKGRID-BELOW-THE-REQUIREMENT"], ["TOKGRID-REACHES-THE-REQUIREMENT"]),
    ("T25G better than outputs", better, None, 0, False, ["TOKGRID-T25G-BETTER-THAN-OUTPUTS", "TOKGRID-T25G-READABLE"], ["TOKGRID-T25G-NOT-BETTER"]),
]


def part_reader():
    for name, mut, hook, want_rc1, want_refused, want_names, want_absent in CASES:
        with tempfile.TemporaryDirectory(dir=TMP, prefix="tgt_") as root:
            pod = build(root, mut)
            rc1, o1, rc2, o2, names = stages(root, pod, hook)
            ok = rc1 == want_rc1
            ok = ok and ((rc2 != 0 and names is None and "REFUSED" in o2) if want_refused else (rc2 == 0 and names is not None))
            if want_names and names is not None:
                ok = ok and all(n in names for n in want_names)
            if want_absent and names is not None:
                ok = ok and not any(n in names for n in want_absent)
            leaked = any(tok in o1 for tok in ("r_pooled", "mae", "0.468", "0.257", "lag1"))
            ok = ok and not leaked
            check("reader: %s (controls rc %d, names rc %d)" % (name, rc1, rc2), ok, (o1 + o2)[-300:])
    # the outputs refuse to be overwritten
    with tempfile.TemporaryDirectory(dir=TMP, prefix="tgt_") as root:
        pod = build(root)
        stages(root, pod)
        rc, out = run([PY, READER, "controls", "--pod-dir", pod, "--out", os.path.join(root, "controls.json"), "--add86-result", os.path.join(root, "add86_result.json"),
                       "--add86-controls", os.path.join(root, "add86_controls.json")])
        check("reader: the controls stage refuses an existing output", rc != 0 and "REFUSED" in out, out)


def main():
    part_generators()
    part_tool()
    part_reader()
    n_ok = sum(RESULTS)
    print("ZZ TEST-TOKGRID %d/%d" % (n_ok, len(RESULTS)))
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
