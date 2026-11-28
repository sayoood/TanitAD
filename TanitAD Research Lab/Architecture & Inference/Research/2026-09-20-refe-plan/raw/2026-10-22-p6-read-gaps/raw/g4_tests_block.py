# ============================================================ G4: P6's commands.json as the command source of the S0 dump and the candidate-bank builder
PKG_ROOT = HERE.parent
DUMP_PY = PKG_ROOT / "raw" / "2026-10-18-sel-rank" / "code" / "srttc_s0_dump.py"
BCB_PY = PKG_ROOT / "raw" / "2026-10-19-p6-read-prep" / "code" / "build_candidate_bank.py"
P6_COMMANDS = Path(os.environ.get("P6_COMMANDS_JSON", "D:/Projects/TanitAD-artifacts/p6dry/hf/logs/commands.json"))
S0_PLAN = Path("C:/Users/Admin/srttc_s0/run/plan/master_plan.json")
P6_EPOCH6 = Path("D:/Projects/TanitAD-artifacts/p6dry/hf/run/ckpt_last.pt")


def _load_mod(path: Path, name: str, src: str | None = None):
    if src is not None:
        td = Path(tempfile.mkdtemp())
        p = td / path.name
        p.write_bytes(src.encode("utf-8"))
        path = p
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(m)
    finally:
        sys.path.remove(str(path.parent))
    return m


def _small_table(td: Path, rows=None):
    rows = rows or {"LOG_A/tokA": [1, 0, 0, 0], "LOG_A/tokB": [0, 1, 0, 0], "LOG_B/tokC": [0, 0, 1, 0]}
    p = td / "commands.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return p


def g4_dump_helpers(DM):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = _small_table(td)
        tab = DM.load_command_table(str(p))
        assert DM.command_vec(tab, "LOG_A", "tokA") == [1, 0, 0, 0] and DM.command_vec(tab, "LOG_B", "tokC") == [0, 0, 1, 0]
        try:
            DM.command_vec(tab, "LOG_A", "tokMISSING")
        except RuntimeError as e:
            assert "no command" in str(e)
        else:
            raise AssertionError("a token with NO command was answered (a default command): the refusal is gone")
        for bad in ({"LOG_A/t": [1, 1, 0, 0]}, {"LOG_A/t": [0, 0, 0, 0]}, {"t": [1, 0, 0, 0]}, {}):
            q = td / "bad.json"
            q.write_text(json.dumps(bad), encoding="utf-8")
            try:
                DM.load_command_table(str(q))
            except SystemExit:
                continue
            raise AssertionError(f"a malformed table was accepted: {bad}")
        try:
            DM.load_command_table(str(p), sha256="0" * 64)
        except SystemExit as e:
            assert "sha256" in str(e)
        else:
            raise AssertionError("a wrong sha256 pin was accepted")


@check("G4 srttc_s0_dump: command-table helpers load, look up, and REFUSE a missing token / malformed row / wrong sha256 pin")
def t_g4_dump_helpers():
    g4_dump_helpers(_load_mod(DUMP_PY, "srttc_s0_dump_g4"))


@check("G4 RED: a mutant dump that answers a missing token with a default command fails the helper check")
def t_g4_dump_helpers_red():
    src = DUMP_PY.read_text(encoding="utf-8")
    old = '        raise RuntimeError(f"token {tok} (log {log}): no command in {table[\'path\']}: a command-mode checkpoint is never run on a default command")'
    assert src.count(old) == 1
    mut = _load_mod(DUMP_PY, "srttc_s0_dump_g4_mut", src.replace(old, "        return [0, 1, 0, 0]"))
    must_fail(g4_dump_helpers, mut)


def _fake_dump_env(nav_mode):
    """stub every heavy import of srttc_s0_dump.main so it can run to its command-source decision; returns the sentinel raised right after it"""
    class Stop(Exception):
        pass

    class Cfg:
        pass

    Cfg.nav_mode = nav_mode

    class Model:
        cfg = Cfg()

        class score_head:
            @staticmethod
            def register_forward_hook(f):
                raise Stop("passed the command-source decision")

    class FakePlanner:
        def __init__(self, **kw):
            self.model, self.trained, self.per_sample_calib, self.device, self.select, self.rule = Model(), True, True, "cpu", "best", "navsim_v1"
            self.sanitize_goal, self.goal_fix, self.repair_last_heading = False, "pdm_route", True
    mods = {}
    for n in ("torch", "navtrain_scenarios", "planner", "refe_navtest_seam", "nuplan", "nuplan.planning", "nuplan.planning.simulation", "nuplan.planning.simulation.planner",
              "nuplan.planning.simulation.planner.abstract_planner"):
        mods[n] = types.ModuleType(n)
    mods["planner"].REFePlanner = FakePlanner
    mods["planner"].FrameResolver = lambda **kw: types.SimpleNamespace(**kw)
    mods["refe_navtest_seam"].to_navsim = lambda x: x
    mods["refe_navtest_seam"].set_driving_command = lambda *a, **k: True
    mods["nuplan.planning.simulation.planner.abstract_planner"].PlannerInitialization = object
    return mods, Stop


def run_dump_main(DM, commands, nav_mode="command"):
    mods, Stop = _fake_dump_env(nav_mode)
    saved = {n: sys.modules.get(n) for n in mods}
    sys.modules.update(mods)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        plan = {"tokens": ["tokA"], "token_log": {"tokA": "LOG_A"}, "token_root": {"tokA": "R"}, "db": {"LOG_A": str(td / "LOG_A.db")}}
        (td / "plan.json").write_text(json.dumps(plan))
        argv = ["srttc_s0_dump.py", "--ckpt", "x.pt", "--plan", str(td / "plan.json"), "--out", str(td / "o.npz"), "--device", "cpu", "--lean"]
        if commands:
            argv += ["--commands", str(commands)]
        old_argv, old_env = sys.argv, {k: os.environ.pop(k, None) for k in ("SRTTC_COMMANDS_JSON", "SRTTC_COMMANDS_SHA256")}
        sys.argv = argv
        try:
            try:
                rc = DM.main()
                return {"rc": rc, "passed_decision": False}
            except Stop:
                return {"rc": None, "passed_decision": True}
        finally:
            sys.argv = old_argv
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
            for n, m in saved.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m


@check("G4 srttc_s0_dump.main: a command-mode checkpoint with NO source is still REFUSED (rc 2); with a source it passes the decision; a goal-mode checkpoint is untouched")
def t_g4_dump_main_decision():
    DM = _load_mod(DUMP_PY, "srttc_s0_dump_g4_main")
    with tempfile.TemporaryDirectory() as td:
        src = _small_table(Path(td))
        r0 = run_dump_main(DM, None)
        assert r0 == {"rc": 2, "passed_decision": False}, r0                   # the refusal kept
        r1 = run_dump_main(DM, src)
        assert r1["passed_decision"] is True, r1                                 # a source: past the refusal
        r2 = run_dump_main(DM, None, nav_mode="goal")
        assert r2["passed_decision"] is True, r2                                 # goal mode: the landed behaviour, no source needed


@check("G4 RED: a mutant dump that drops the refusal (accepts command-mode with no source) fails the decision check")
def t_g4_dump_main_decision_red():
    src = DUMP_PY.read_text(encoding="utf-8")
    old = "        if not a.commands:"
    assert src.count(old) == 1
    mut = _load_mod(DUMP_PY, "srttc_s0_dump_g4_main_mut", src.replace(old, "        if False:"))

    def go():
        r0 = run_dump_main(mut, None)
        assert r0 == {"rc": 2, "passed_decision": False}, r0
    must_fail(go)


@check("G4 srttc_s0_dump: the per-token hop sits BEFORE planner.infer in the loop and calls the seam's own set_driving_command with the table's one-hot (static; the hop itself is MEASURED in the dry run and in the e2e CPU run)")
def t_g4_dump_hop_static():
    src = DUMP_PY.read_text(encoding="utf-8")
    i_hop = src.index("set_driving_command(planner, tok, {\"ego_statuses\": [{\"driving_command\": command_vec(cmd_table, log, tok)}]})")
    i_inf = src.index("traj, _score, k = planner.infer(ego, img)")
    assert i_hop < i_inf


def _bcb():
    return _load_mod(BCB_PY, "build_candidate_bank_g4")


@check("G4 build_candidate_bank: S0_PINS accepts the landed AND the G4 srttc_s0_dump.py, refuses any third form")
def t_g4_bcb_pins():
    BB = _bcb()
    if not Path(BB.S0_CODE).exists():
        RESULTS.append(("G4 build_candidate_bank pins: SKIPPED (no dev-box S0 code dir)", True, ""))
        return
    got = BB.check_s0_pins()                                                      # the landed dev-box S0 code
    assert got["srttc_s0_dump.py"] == BB.S0_DUMP_LANDED_SHA256
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "s0code"
        shutil.copytree(BB.S0_CODE, d, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(DUMP_PY, d / "srttc_s0_dump.py")
        assert BB.check_s0_pins(str(d))["srttc_s0_dump.py"] == BB.S0_DUMP_G4_SHA256
        (d / "srttc_s0_dump.py").write_bytes(DUMP_PY.read_bytes() + b"# a third form\r\n")
        try:
            BB.check_s0_pins(str(d))
        except BB.BuildError as e:
            assert "srttc_s0_dump.py" in str(e)
        else:
            raise AssertionError("a third form of the dump was accepted")


@check("G4 build_candidate_bank.check_command_source: P6's real commands.json covers ALL 8,922 S0 plan tokens; a table missing one / a non-one-hot row / a wrong sha256 is refused")
def t_g4_bcb_command_source():
    BB = _bcb()
    if not (P6_COMMANDS.exists() and S0_PLAN.exists()):
        raise AssertionError(f"{P6_COMMANDS} or {S0_PLAN} absent: the coverage control cannot run")
    plan_tokens = json.load(open(S0_PLAN, encoding="utf-8"))["tokens"]
    rec = BB.check_command_source(str(P6_COMMANDS), None, plan_tokens)
    assert rec["plan_tokens_covered"] == len(plan_tokens) == 8922 and rec["rows"] == 103037, rec
    tab = json.load(open(P6_COMMANDS, encoding="utf-8"))
    tk, lg = next(iter(plan_tokens.items()))
    with tempfile.TemporaryDirectory() as td:
        miss = dict(tab)
        del miss[f"{lg}/{tk}"]
        p = Path(td) / "miss.json"
        p.write_text(json.dumps(miss))
        try:
            BB.check_command_source(str(p), None, plan_tokens)
        except BB.BuildError as e:
            assert "lacks 1 of the 8922" in str(e)
        else:
            raise AssertionError("a table missing a plan token was accepted")
        bad = dict(tab)
        bad[f"{lg}/{tk}"] = [1, 1, 0, 0]
        p.write_text(json.dumps(bad))
        try:
            BB.check_command_source(str(p), None, plan_tokens)
        except BB.BuildError as e:
            assert "non-one-hot" in str(e)
        else:
            raise AssertionError("a non-one-hot row was accepted")
    try:
        BB.check_command_source(str(P6_COMMANDS), "0" * 64, plan_tokens)
    except BB.BuildError as e:
        assert "sha256" in str(e)
    else:
        raise AssertionError("a wrong sha256 was accepted")


def g4_apply_cases(BB):
    cand_src = {"commands": {"path": str(P6_COMMANDS), "sha256": BB.sha256_file(str(P6_COMMANDS))} if P6_COMMANDS.exists() else None,
                "s0_code_pins": {"srttc_s0_dump.py": BB.S0_DUMP_G4_SHA256}}
    # command-mode + source + the G4 dump: exported
    env = {}
    r = BB.apply_command_source(cand_src, "command", env)
    assert r["commands_exported"] and env["SRTTC_COMMANDS_JSON"] == cand_src["commands"]["path"]
    # command-mode, NO source: REFUSED (the kept refusal)
    for c in ({**cand_src, "commands": None},):
        try:
            BB.apply_command_source(c, "command", {})
        except BB.BuildError as e:
            assert "needs a command source" in str(e)
        else:
            raise AssertionError("a command-mode checkpoint with NO source was accepted")
    # command-mode + source + the LANDED dump: refused (it would refuse at the GPU, after the lock)
    try:
        BB.apply_command_source({**cand_src, "s0_code_pins": {"srttc_s0_dump.py": BB.S0_DUMP_LANDED_SHA256}}, "command", {})
    except BB.BuildError as e:
        assert "G4 srttc_s0_dump.py" in str(e)
    else:
        raise AssertionError("command-mode with the landed dump was accepted")
    # goal-mode: the source is ignored and nothing is exported, even if a stale export is present
    env = {"SRTTC_COMMANDS_JSON": "stale", "SRTTC_COMMANDS_SHA256": "stale"}
    r = BB.apply_command_source({**cand_src, "s0_code_pins": {"srttc_s0_dump.py": BB.S0_DUMP_LANDED_SHA256}}, "goal", env)
    assert r["commands_exported"] is False and not env


@check("G4 build_candidate_bank.apply_command_source: command-mode + source exports it; NO source is REFUSED; the landed dump is refused for command-mode; goal-mode exports nothing")
def t_g4_bcb_apply():
    if not P6_COMMANDS.exists():
        raise AssertionError(f"{P6_COMMANDS} absent")
    g4_apply_cases(_bcb())


@check("G4 RED: a mutant that exports a default / drops the no-source refusal fails the apply cases")
def t_g4_bcb_apply_red():
    src = BCB_PY.read_text(encoding="utf-8")
    old = '        if not src:\n            raise BuildError("a command-mode checkpoint needs a command source'
    assert src.count(old) == 1
    mut = _load_mod(BCB_PY, "build_candidate_bank_g4_mut", src.replace(old, '        if False:\n            raise BuildError("a command-mode checkpoint needs a command source'))
    must_fail(g4_apply_cases, mut)


@check("G4 build_candidate_bank.ckpt_nav_mode reads meta.nav_mode: the epoch-6 P6 file reads 'command'; a goal-mode file reads 'goal' (skipped, stated, if the files are absent)")
def t_g4_bcb_nav_mode():
    BB = _bcb()
    if not P6_EPOCH6.exists():
        RESULTS.append(("G4 ckpt_nav_mode: SKIPPED (epoch-6 checkpoint absent)", True, ""))
        return
    assert BB.ckpt_nav_mode(str(P6_EPOCH6)) == "command"
    if Path(BB.P2_CKPT).exists():
        assert BB.ckpt_nav_mode(BB.P2_CKPT) == "goal"


