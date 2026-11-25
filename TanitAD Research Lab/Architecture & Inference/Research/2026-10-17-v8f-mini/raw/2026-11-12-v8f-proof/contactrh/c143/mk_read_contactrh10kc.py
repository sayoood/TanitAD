"""mk_read_contactrh10kc.py -- writes read_contactrh10kc.py (card PREREG_V8F_PROOF_ADD143) from the landed read_contactrh10kb.py (card ADD142) by asserted text replacement.

Changes, all of card ADD143 sections 3 to 5: card and name prefix; run tags H_RQ50K_P30 / U_RQ50K_P30; bootstrap seed 143; the expected file count is the clip count of the grid of card ADD70
(grid_p30.json), and the dump windows must be exactly the grid windows; the pre-roll handling of record (read_longroll10k_n4.py, imported, never edited): its engine patch is installed, the runs
are registered as aged runs under readout K, every dump carries its own start point p0 (tick file), and the contact windows come from the patched headway engine; the tick of a window is the last
rule tick of the MAIN phase (the rule log of a pre-rolled run holds main-phase rows only; the pre-roll rows live in rule_log_pre_<run>.npz) at or before the first contact step;
control c0 (queue of card ADD143 = box2_crhc_queue.build_runs; done files with rc 0; dump files = grid clip count; windows = grid windows; finite plans; tick-file and rule-log window counts;
rule, tick, pre-roll logs; RH and UE dumps differ; same (clip, t0) list), c1 (engine patch installed, start points finite, cells equal the independent recomputation), c2 (main-phase rows only;
g0 identity on 20 planted ticks; logged v0 equals the own speed at the aged tick), c3 (planted +2.0 and the n3 planted start offset of read_longroll10k_n4), c4 and c5 as card ADD142.
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "read_contactrh10kb.py")
s = io.open(SRC, encoding="utf-8").read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:90])
    s = s.replace(a, b)


def between(start, end):
    i = s.index(start)
    j = s.index(end, i)
    return i, j


# ---------------------------------------------------------------------------------------------------------------- header and constants
rep('"""read_contactrh10kb.py -- read tool of card PREREG_V8F_PROOF_ADD142 (CONTACTRH10KB), written by mk_read_contactrh10kb.py from read_contactrh10k.py (card ADD141); the text below is the parent\'s:',
    '"""read_contactrh10kc.py -- read tool of card PREREG_V8F_PROOF_ADD143 (CONTACTRH10KC), written by mk_read_contactrh10kc.py from read_contactrh10kb.py (card ADD142), in AGED states '
    '(pre-roll of 30 steps on the grid of card ADD70); the text below is the parent\'s:')
rep("  python read_contactrh10k.py read --root PULL_DIR --join JOIN_NPZ --out-dir DIR --tests-out DEV_PYTEST_OUT [--seeds 0,1,2]",
    "  python read_contactrh10kc.py read --root PULL_DIR --join JOIN_NPZ --grid GRID_P30_JSON --out-dir DIR --tests-out DEV_PYTEST_OUT [--seeds 0,1,2] [--kit-dir DIR] [--integ-dir DIR]\n"
    "Pre-roll handling (card ADD143 section 3): read_longroll10k_n4.py is imported from --integ-dir (default F:/TanitAD-scratch/v8proof/integ10k) and its engine patch is installed on the headway\n"
    "engine; each run is registered as an aged run under readout K. Step 1 of a pre-rolled track comes from the track's own start point p0 (tick file), not from (0, 0). The contact cells and the\n"
    "first contact step are position quantities and are read on the track as written. The rule log of a pre-rolled run holds main-phase rows only; the tick of a window is its last row at or before\n"
    "the first contact step. The logged v0 of a tick is the own rolled speed at the aged tick: for k above 0 the chord of own step k (step 1 from p0), for k = 0 the speed recorded in the tick file.")
rep('CARD = "PREREG_V8F_PROOF_ADD142"', 'CARD = "PREREG_V8F_PROOF_ADD143"')
rep("import hashlib\nimport io\nimport json\n", "import glob\nimport hashlib\nimport io\nimport json\n")
rep('PRE = "CONTACTRH10KB"', 'PRE = "CONTACTRH10KC"')
rep('TAG_H, TAG_U = "H_RQ50Kx", "U_RQ50Kx"', 'TAG_H, TAG_U = "H_RQ50K_P30", "U_RQ50K_P30"')
rep('N_BOOT, BOOT_SEED = 2000, 142', 'N_BOOT, BOOT_SEED = 2000, 143')
rep('TOL_G = 1e-3\n', 'TOL_G = 1e-3\nTOL_V = 1e-6\nPREROLL_STEPS = 30\n'
    'INTEG_DIR = os.environ.get("CONTACTRH_INTEG", "F:/TanitAD-scratch/v8proof/integ10k")\n'
    'GRID_DEFAULT = "F:/TanitAD-scratch/v8proof/rhacc10k/pins/grid_p30.json"\n'
    '# registration of the aged runs in read_longroll10k.ALL_RUNS, as card ADD140 read R3 registered them (form, readout K, pre-roll 30, route, ..., plan executor, rule every 5, kappa)\n'
    'R3_SPEC = ("RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2)\n')
rep('HW = None\n', 'HW = None\nN4 = None\n')
rep('LIMITS = ("T1-kin only:', 'LIMITS = ("AGED states: every window starts after a pre-roll of 30 steps (6 s of the model\'s own driving); a result here does not transfer by itself to the fresh reader-form cost. "\n          "T1-kin only:')
rep('TIER = "T1-kin/hb0.2 (kinematic rollout on the model\'s own actions, logged camera frames; the lead does not react; not driving performance)"',
    'TIER = "T1-kin/hb0.2 (kinematic rollout on the model\'s own actions after a pre-roll of 30 steps, logged camera frames; the lead does not react; not driving performance)"')

# ---------------------------------------------------------------------------------------------------------------- pre-roll binding and loading
i, j = between("def load_run(", "# ============================================================================================ per-window tables")
s = s[:i] + '''def bind_longroll(path=None):
    """imports read_longroll10k_n4.py (the pre-roll handling of record) unedited and installs its engine patch on the headway engine (once); returns the module"""
    global N4, INTEG_DIR
    H = bind_headway()
    if N4 is None:
        if path:
            INTEG_DIR = path
        if not os.path.isfile(os.path.join(INTEG_DIR, "read_longroll10k_n4.py")):
            raise LoadError("the pre-roll reader of record cannot be found")
        if INTEG_DIR not in sys.path:
            sys.path.insert(0, INTEG_DIR)
        import read_longroll10k_n4 as mod
        N4 = mod
    N4.patch_engines(H, H.RH)
    return N4


def register_runs(seeds):
    """the aged runs as card ADD140 registered its R3 arms (readout K, pre-roll 30, plan executor)"""
    for m in seeds:
        for tag in (TAG_H, TAG_U):
            N4.R.ALL_RUNS["%s_s%d" % (tag, m)] = R3_SPEC


def engines_patched():
    """True when the headway window engine and the family engines are the patched ones of read_longroll10k_n4 (step 1 from the track's own start point)"""
    H = bind_headway()
    return bool(getattr(H.RH, "_note_n4", False) and H.window_cells.__qualname__.startswith("patch_engines") and H.RH.comps.__qualname__.startswith("patch_engines"))


def tick_path(root, tag, m):
    key = "%s_s%d" % (tag, m)
    return os.path.join(run_folder(root, tag, m), "ticks", "ticks_%s_m10_causal_heads.npz" % key)


def read_tick_file(path):
    """{(ci, t0): own speed recorded at tick 0 (pose[3])}, number of (ci, t0) windows with a tick-0 row; the first tick-0 row of a window is taken (as p0_rows of read_longroll10k_n4)"""
    if not os.path.isfile(path):
        raise LoadError("a tick file is missing")
    with np.load(path, allow_pickle=True) as z:
        for k in ("ci", "t0", "tick", "pose"):
            if k not in z.files:
                raise LoadError("a tick file lacks a field")
        ci, t0, tk, pose = (np.asarray(z[k]) for k in ("ci", "t0", "tick", "pose"))
    out = {}
    for r in np.flatnonzero(np.asarray(tk, np.int64) == 0):
        out.setdefault((int(ci[r]), int(t0[r])), float(np.asarray(pose[r], np.float64)[3]))
    return out


def load_run(root, tag, m, J, expect_files, keys):
    """dump (the windows of the grid, in the grid order, with their own start points p0), labels, contact cells and the main-phase rule log of one aged run"""
    H = bind_headway()
    bind_longroll()
    dd = dump_folder(root, tag, m)
    if not os.path.isdir(dd):
        raise LoadError("a dump folder is missing")
    nf = len(glob.glob(os.path.join(dd, "ep*.npz")))
    if nf != expect_files:
        raise LoadError("a dump folder holds a file count different from the clip count of the grid")
    raw = N4.R.load_dump_dir(dd)
    key = "%s_s%d" % (tag, m)
    rep_ = N4.REPORT.get(key) or {}
    if raw is None or not (rep_.get("tick_file") and rep_.get("missing_rows") == 0) or not np.isfinite(np.asarray(raw["p0"], np.float64)).all():
        raise LoadError("an aged run lacks its tick file rows, so its own start points are unknown")
    idx, rr = N4.R.restrict_idx(raw["clip_index"], raw["ws"], keys)
    if rr["missing"] or rr["duplicated_windows"] or rr["extra_rows"]:
        raise LoadError("the dump windows are not exactly the windows of the grid")
    D = N4.R.cut(raw, idx)
    L = H.label_windows(J, D["clip_index"], D["ws"])
    C = H.window_cells(D["g"], D["cl"], L)
    gm = C["_gm"]
    with np.errstate(invalid="ignore"):
        ct = np.isfinite(gm) & (gm <= 0.0)
    contact = ct.any(1) & L["lead"]
    kc = np.where(contact, np.argmax(ct, axis=1) + 1, 0)
    log = read_log(log_path(root, tag, m))
    tickv = read_tick_file(tick_path(root, tag, m))
    return {"D": D, "L": L, "C": C, "contact": contact, "kc": kc, "log": log, "idx": index_log(log), "tag": tag, "m": m, "key": key,
            "tickv": tickv, "finite": bool(np.isfinite(np.asarray(D["cl"], np.float64)).all()), "n_files": nf, "n_windows": int(len(D["ws"]))}


''' + s[j:]

# ---------------------------------------------------------------------------------------------------------------- c0
i, j = between("def ctl_c0(", "# ============================================================================================ controls")
s = s[:i] + '''def ctl_c0(root, seeds, runs, grid, kit_dir=None):
    """the runs: argv of every run equals the ADD143 queue's table (box2_crhc_queue.build_runs, box-2 paths); done files with rc 0; dump files = clip count of the grid; windows = windows of the grid;
    finite plans; tick-file and rule-log window counts = windows of the grid; rule log, tick file, pre-roll logs present (pre-roll steps 30); RH and UE dumps differ; same (clip, t0) list"""
    kit = kit_dir or os.path.join(os.path.dirname(os.path.abspath(root)), "kit")
    if kit not in sys.path:
        sys.path.insert(0, kit)
    import box2_crhc_queue as Qc
    want = {r["arm"]: r["argv"] for r in Qc.build_runs()}
    n_win, n_clip = int(grid["n_windows"]), len(grid["order"])
    bad = []
    for arm in ["%s_s%d" % (t, m) for m in seeds for t in (TAG_H, TAG_U)]:
        if arm not in want:
            bad.append("queue lacks " + arm)
    for arm, argv in sorted(want.items()):
        p = os.path.join(root, arm, "argv.json")
        if not os.path.isfile(p) or json.load(io.open(p, encoding="utf-8")).get("argv") != argv:
            bad.append("argv " + arm)
        dp = os.path.join(root, arm, "RUN_DONE.json")
        if not os.path.isfile(dp):
            bad.append("done " + arm)
        elif json.load(io.open(dp, encoding="utf-8")).get("rc") != 0:
            bad.append("done rc " + arm)
    for (arm_, m), r in sorted(runs.items()):
        nm = "%s_s%d" % (r["tag"], m)
        if r["n_files"] != n_clip:
            bad.append("dump files " + nm)
        if r["n_windows"] != n_win:
            bad.append("dump windows " + nm)
        if not r["finite"]:
            bad.append("plans not finite " + nm)
        if len(r["tickv"]) != n_win:
            bad.append("tick-file windows " + nm)
        lg = r["log"]
        first = {(int(c), int(t)) for c, t, k in zip(lg["ci"], lg["t0"], lg["k"]) if int(k) == 0}
        if len(first) != n_win:
            bad.append("rule-log windows " + nm)
        tdir = os.path.join(run_folder(root, r["tag"], m), "ticks")
        for f in ("preroll_log_%s.json" % nm, "preroll_offsets_%s.npz" % nm, "rule_log_pre_%s.npz" % nm):
            if not os.path.isfile(os.path.join(tdir, f)):
                bad.append("missing " + f)
        pj = os.path.join(tdir, "preroll_log_%s.json" % nm)
        if os.path.isfile(pj) and json.load(io.open(pj, encoding="utf-8")).get("preroll_steps") != PREROLL_STEPS:
            bad.append("pre-roll steps " + nm)
    for m in seeds:
        a, b = runs[("H", m)]["D"], runs[("U", m)]["D"]
        if not (np.array_equal(a["clip_index"], b["clip_index"]) and np.array_equal(a["ws"], b["ws"])):
            bad.append("the two runs of seed %d have different (clip, t0) lists" % m)
        hs = []
        for tag in (TAG_H, TAG_U):
            d = dump_folder(root, tag, m)
            h = hashlib.sha256()
            for f in sorted(os.listdir(d)):
                h.update(io.open(os.path.join(d, f), "rb").read())
            hs.append(h.hexdigest())
        if hs[0] == hs[1]:
            bad.append("dumps identical seed %d" % m)
    return {"problems": bad, "n_runs": len(want), "n_windows_grid": n_win, "n_clips_grid": n_clip, "PASS": not bad}


''' + s[j:]

# ---------------------------------------------------------------------------------------------------------------- c1
rep("    H = bind_headway()\n    bad, n_cells = [], 0\n",
    "    H = bind_headway()\n    bad, n_cells = [], 0\n    if not engines_patched():\n        bad.append(\"the pre-roll engine patch of read_longroll10k_n4 is not installed\")\n")
rep("        D, L = r[\"D\"], r[\"L\"]\n        G, ns, lead = L[\"G\"], L[\"ns\"], L[\"lead\"]\n",
    "        D, L = r[\"D\"], r[\"L\"]\n        if not np.isfinite(np.asarray(D[\"p0\"], np.float64)).all():\n            bad.append(\"%s_s%d own start point\" % key)\n        G, ns, lead = L[\"G\"], L[\"ns\"], L[\"lead\"]\n")
rep('    """contact windows recomputed independently equal the contact cells of the landed headway reader, per run; same windows in the two runs of a seed"""',
    '    """contact windows recomputed independently equal the contact cells of the landed headway reader under the pre-roll handling of record (patched engine installed, own start points finite), per run; same windows in the two runs of a seed"""')

# ---------------------------------------------------------------------------------------------------------------- c2
i, j = between("def ctl_c2(", "def plant_logs(")
s = s[:i] + '''def own_speed_at(cl_row, p0, k, tick_v0):
    """own rolled speed at the tick step k of an aged window: k above 0 the chord of own step k (cl_row[k-1] minus the previous own point; step 1 from the own start point p0, not from (0, 0));
    k = 0 the speed recorded in the tick file at tick 0 (the dump holds no step before step 1)"""
    if k == 0:
        return float("nan") if tick_v0 is None else float(tick_v0)
    cl_row = np.asarray(cl_row, np.float64)
    prev = np.asarray(p0, np.float64) if k == 1 else cl_row[k - 2]
    d = cl_row[k - 1] - prev
    return float(math.hypot(d[0], d[1])) / bind_headway().DT


def ctl_c2(runs, seeds, tabs, J):
    """main-phase rows only; every X and S window has a tick at or before its first contact step; the logged g_src of a planted sample of 20 ticks equals g0 - a_l + a_m with a_l, a_m recomputed
    from the dump, within 1e-3 m; and for those ticks the logged v0 equals the own rolled speed at the aged tick (own_speed_at, from the dump and the tick file) within 1e-6, not the logged speed at t0"""
    H = bind_headway()
    probs = []
    n_win = 0
    tk, arm_of = {}, {"X": "H", "S": "H", "S_U": "U"}
    for m in seeds:
        for s_ in ("X", "S", "S_U"):
            T = tabs[m][s_]
            probs += T["problems"]
            if len(T["i"]) != int(tabs[m]["sets"]["S" if s_ == "S_U" else s_].sum()):
                probs.append("a window of %s has no usable tick" % s_)
            n_win += len(T["i"])
            tk[(m, s_)] = T
        for a_ in ("H", "U"):
            r = runs[(a_, m)]
            lg, D = r["log"], r["D"]
            if not {(int(c), int(t)) for c, t in zip(lg["ci"], lg["t0"])} <= {(int(c), int(t)) for c, t in zip(D["clip_index"], D["ws"])}:
                probs.append("a rule log holds rows of windows outside the grid (pre-roll rows)")
            k_ = np.asarray(lg["k"], np.int64)
            if len(k_) and (k_.min() < 0 or k_.max() >= H.K or (np.asarray(lg["j"], np.int64) - np.asarray(lg["t0"], np.int64) != k_).any()):
                probs.append("a rule log row is not a main-phase row (k outside 0..29 or j different from t0 + k)")
    rows = sample_rows(tk)
    worst, worst_v = 0.0, 0.0
    n_ok, n_v = 0, 0
    for (m, s_, q) in rows:
        T, r = tk[(m, s_)], runs[(arm_of[s_], m)]
        i, row, k = int(T["i"][q]), int(T["row"][q]), int(T["k"][q])
        lg, D = r["log"], r["D"]
        sp = own_speed_at(D["cl"][i], D["p0"][i], k, r["tickv"].get((int(D["clip_index"][i]), int(D["ws"][i]))))
        v0 = float(lg["v0"][row])
        if math.isfinite(sp) and math.isfinite(v0):
            worst_v = max(worst_v, abs(v0 - sp))
            n_v += 1
        else:
            probs.append("a planted tick has no own speed to compare with its logged v0")
        g1 = D["g"][i:i + 1]
        a_l = 0.0 if k == 0 else float(H.logged_arc(g1)[0, min(max(k - 1, 0), 29)])
        a_m = 0.0 if k == 0 else float(H.progress(g1, D["cl"][i:i + 1])[0, k - 1])
        g0, gs = float(lg["g0"][row]), float(lg["g_src"][row])
        if not (math.isfinite(g0) and math.isfinite(gs)):
            continue
        dev = max(abs((g0 - a_l + a_m) - gs), abs(float(lg["a_m"][row]) - a_m), abs(float(lg["a_l"][row]) - a_l))
        worst = max(worst, dev)
        n_ok += 1
    if n_ok == 0:
        probs.append("no tick of the planted sample could be recomputed")
    if worst > TOL_G:
        probs.append("a planted tick differs from the recomputed gap by more than %g m" % TOL_G)
    if n_v == 0:
        probs.append("no planted tick could be compared on v0")
    if worst_v > TOL_V:
        probs.append("the logged v0 of a planted tick differs from the own speed at the aged tick by more than %g m/s" % TOL_V)
    return {"PASS": not probs, "n_windows_X_S": n_win, "n_sample": len(rows), "n_sample_checked": n_ok, "max_abs_dev_m": worst, "tol_m": TOL_G,
            "n_sample_v0_checked": n_v, "max_abs_v0_dev_mps": worst_v, "tol_v0_mps": TOL_V, "problems": sorted(set(probs))}


''' + s[j:]

# ---------------------------------------------------------------------------------------------------------------- c3 (planted gap error, and the planted start offset n3)
i = s.index("def ctl_c4(")
s = s[:i] + '''def n3_judge(plant):
    """the planted start-offset check of read_longroll10k_n4 (note N3 form: speed, heading, yaw-rate error 0 and along-track error 5 m on a track started 5 m behind the logged origin, same gaps, zero offset equals the engines)"""
    p = dict(plant or {})
    return {"PASS": bool(p.get("PASS") is True), "planted": p}


def ctl_c3_full(runs, seeds, tabs, J, inv, n_ep, q1_main):
    """c3 of card ADD143: a planted +2.0 m gap error reads +2.0 (as card ADD142) and the n3 planted start offset of read_longroll10k_n4 passes; n2 of that reader is reported beside"""
    c = ctl_c3(runs, seeds, tabs, J, inv, n_ep, q1_main)
    n3 = n3_judge(N4.PLANT if N4 is not None else None)
    c["n3_planted_start_offset"] = n3
    c["PASS"] = bool(c["PASS"] and n3["PASS"])
    try:
        c["n2_beside_not_gating"] = N4.controls_n4()["n2_prerolled_runs_start_at_p0"]
    except Exception as e:  # noqa: BLE001
        c["n2_beside_not_gating"] = {"PASS": None, "why": type(e).__name__}
    return c


''' + s[i:]

# ---------------------------------------------------------------------------------------------------------------- stage_read and parser
rep("    for p in (a.join, a.root):\n", "    for p in (a.join, a.root, a.grid):\n")
rep('''        J = HW.load_join(a.join)
        runs = {}
        for m in seeds:
            runs[("H", m)] = load_run(a.root, TAG_H, m, J, a.expect_files)
            runs[("U", m)] = load_run(a.root, TAG_U, m, J, a.expect_files)
''', '''        bind_longroll(a.integ_dir)
        register_runs(seeds)
        J = HW.load_join(a.join)
        grid = json.load(io.open(a.grid, encoding="utf-8"))
        keys = N4.R.grid_keys(grid, [str(c) for c in J["clip"]])
        ef = a.expect_files if a.expect_files else len(grid["order"])
        runs = {}
        for m in seeds:
            runs[("H", m)] = load_run(a.root, TAG_H, m, J, ef, keys)
            runs[("U", m)] = load_run(a.root, TAG_U, m, J, ef, keys)
''')
rep('"c0": ctl_c0(a.root, seeds)', '"c0": ctl_c0(a.root, seeds, runs, grid, a.kit_dir)')
rep('ctl["c3"] = ctl_c3(runs, seeds, tabs, J, inv, n_ep, q1_main)', 'ctl["c3"] = ctl_c3_full(runs, seeds, tabs, J, inv, n_ep, q1_main)')
rep('    r.add_argument("--expect-files", type=int, default=141)\n',
    '    r.add_argument("--expect-files", type=int, default=None, help="default: the clip count of the grid")\n'
    '    r.add_argument("--grid", default=GRID_DEFAULT)\n    r.add_argument("--kit-dir", default=None)\n    r.add_argument("--integ-dir", default=None)\n')
rep('print("ZZ READ-CONTACTRH10KB all_pass', 'print("ZZ READ-CONTACTRH10KC all_pass')
rep('"* Q3 of record, own rolled speed minus the rule lead speed (v0 - v_L), S, RH minus UE,', '"* Q3 of record, own rolled speed at the aged tick minus the rule lead speed (v0 - v_L), S, RH minus UE,')
rep('"Q3 (v0 - v_L) above zero with an interval excluding zero; Q1 and Q2 not separated"', '"Q3 (v0 - v_L, aged states) above zero with an interval excluding zero; Q1 and Q2 not separated"')

io.open(os.path.join(HERE, "read_contactrh10kc.py"), "w", encoding="utf-8", newline="\n").write(s)
print("wrote read_contactrh10kc.py")
