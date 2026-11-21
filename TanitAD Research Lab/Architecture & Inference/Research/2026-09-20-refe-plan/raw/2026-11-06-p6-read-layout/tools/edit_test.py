PK = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/"
p = PK + "test_p6_read_gaps.py"
s = open(p, "rb").read().decode("utf-8")
assert "\r\n" not in s


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


# 1. D:-defaulted artifact inputs -> the artifacts root of today's layout (a TEST default: absent files still FAIL / SKIP as before)
rep('''P6_COMMANDS = Path(os.environ.get("P6_COMMANDS_JSON", "D:/Projects/TanitAD-artifacts/p6dry/hf/logs/commands.json"))
S0_PLAN = Path("C:/Users/Admin/srttc_s0/run/plan/master_plan.json")
P6_EPOCH6 = Path("D:/Projects/TanitAD-artifacts/p6dry/hf/run/ckpt_last.pt")''',
    '''def _artifacts_root_default() -> str:
    """P6 layout fix 2026-11-06: $TANITAD_ARTIFACTS_ROOT, else the first existing of E:/ D:/ Projects/TanitAD-artifacts (a TEST default: if none exists the first candidate is returned and the check FAILs / SKIPs as before)"""
    r = os.environ.get("TANITAD_ARTIFACTS_ROOT")
    cands = [r.replace("\\\\", "/").rstrip("/")] if r else [f"{d}:/Projects/TanitAD-artifacts" for d in "ED"]
    return next((c for c in cands if os.path.isdir(c)), cands[0])


P6_COMMANDS = Path(os.environ.get("P6_COMMANDS_JSON") or (_artifacts_root_default() + "/p6dry/hf/logs/commands.json"))
S0_PLAN = Path("C:/Users/Admin/srttc_s0/run/plan/master_plan.json")
P6_EPOCH6 = Path(_artifacts_root_default() + "/p6dry/hf/run/ckpt_last.pt")''')

# 2. G3 RED arm that can fail WITHOUT the D: W3 original on disk (a temp decoy stands in for it)
rep('''    cands = tuple(c for c in S3.FAMILIES6_CANDIDATES if c != os.environ.get("REFE_FAMILIES6", "") or not c)
    f6_first(cands)
    must_fail(f6_first, tuple(reversed(cands)))                                          # the old order (W3 original first, banked copy last) must fail the check''',
    '''    cands = tuple(c for c in S3.FAMILIES6_CANDIDATES if c != os.environ.get("REFE_FAMILIES6", "") or not c)
    f6_first(cands)
    banked = (HP.HARNESS / "ev6e" / "families6.py").resolve()
    if any(c and Path(c).exists() and Path(c).resolve() != banked for c in cands):          # the W3 original is on this box: the real old order must fail
        must_fail(f6_first, tuple(reversed(cands)))
    with tempfile.TemporaryDirectory() as td:                                            # P6 layout fix 2026-11-06: without D: nothing else exists, so a DECOY stands in for the W3 original
        decoy = Path(td) / "families6.py"
        decoy.write_text("# decoy standing in for the W3 original\\n", encoding="utf-8")
        must_fail(f6_first, (str(decoy),) + cands)                                       # an existing non-banked file FIRST must fail the check, on any box''')

# 3. the layout checks (resolver semantics + wiring) with deliberate-regression arms
NEW = r'''
# ============================================================ P6 layout fix 2026-11-06: ONE root resolver, same semantics, path roots only
import ast  # noqa: E402
import re as _re  # noqa: E402

ROOT_USERS = ["stage2_sft.py", "stage3_full.py", "selection_quality.py", "command_adherence.py", "harness/score_e.py",
              "harness/w3e/code/run_v1.py", "harness/w3e/code/navsim_v1_win.py", "harness/w3e/code/write_cache_metadata.py"]


def _resolver_src(rel: str):
    src = (HERE / rel).read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in ("_navsim_root", "_root"))
    return ast.get_source_segment(src, fn), fn.name


def _call_resolver(fn_src: str, name: str, env: dict, isdir):
    ns = {"os": types.SimpleNamespace(environ=dict(env), path=types.SimpleNamespace(isdir=isdir))}
    exec(compile(fn_src, "<resolver>", "exec"), ns)
    if name == "_root":
        return ns[name]("TANITAD_NAVSIM_ROOT", [f"{d}:/Archive/devbox-C/navsim" for d in "EFD"], "navsim archive")
    return ns[name]()


def resolver_semantics(fn_src: str, name: str):
    """env override wins; else the FIRST EXISTING of E:/ F:/ D:/; an env path that does not exist REFUSES; none existing REFUSES (never a silent default)"""
    with tempfile.TemporaryDirectory() as td:
        got = _call_resolver(fn_src, name, {"TANITAD_NAVSIM_ROOT": td + "\\"}, os.path.isdir)         # a trailing backslash is normalised
        assert got == td.replace("\\", "/").rstrip("/"), (got, td)
    fake = lambda have: (lambda p: p in have)                                                          # noqa: E731
    E, F, D = ("E:/Archive/devbox-C/navsim", "F:/Archive/devbox-C/navsim", "D:/Archive/devbox-C/navsim")
    assert _call_resolver(fn_src, name, {}, fake({F, D})) == F                                       # today's layout: E: absent, F: holds it
    assert _call_resolver(fn_src, name, {}, fake({E, F, D})) == E                                    # E: first
    assert _call_resolver(fn_src, name, {}, fake({D})) == D                                          # the old root is the last resort
    for env, isd in (({}, fake(set())), ({"TANITAD_NAVSIM_ROOT": "Z:/does/not/exist"}, fake({E, F, D}))):
        try:
            _call_resolver(fn_src, name, env, isd)
        except SystemExit as e:
            assert "REFUSING" in str(e), str(e)
            continue
        raise AssertionError(f"no refusal for env={env}")


@check("P6 layout: every read-path module carries ONE root resolver with the same semantics (env override > first existing of E:/F:/D:; a non-existent root or none REFUSES), and no code line hard-codes an Archive drive")
def t_layout_resolver_semantics():
    for rel in ROOT_USERS:
        fn_src, name = _resolver_src(rel)
        resolver_semantics(fn_src, name)
        code = [ln for ln in (HERE / rel).read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("#")]
        hard = [ln for ln in code if _re.search(r"[A-Za-z]:[/\\]+Archive", ln)]
        assert not hard, (rel, hard[:2])


def _env_swap(**kv):
    """set/unset env vars; returns the restore function"""
    was = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def restore():
        for k, v in was.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return restore


@check("P6 layout RED: a resolver that never refuses / has the wrong order / ignores the env override each FAIL the semantics check, and a nonexistent root REFUSES at import in every eval module")
def t_layout_resolver_red():
    fn_src, name = _resolver_src("stage2_sft.py")
    resolver_semantics(fn_src, name)
    must_fail(resolver_semantics, fn_src.replace("raise SystemExit(", "return cands[0]; _ = (", 1), name)       # silently returns a (wrong) default instead of refusing
    must_fail(resolver_semantics, fn_src.replace('"EFD"', '"DFE"'), name)                                       # the old root first
    must_fail(resolver_semantics, fn_src.replace('r = os.environ.get("TANITAD_NAVSIM_ROOT")', "r = None"), name)  # the override is ignored
    for rel in ("stage2_sft.py", "stage3_full.py", "selection_quality.py", "command_adherence.py"):          # the real modules, imported with a bad override: they REFUSE
        restore = _env_swap(TANITAD_NAVSIM_ROOT="Z:/does/not/exist", REFE_NAVTEST_EXPORT=None)
        try:
            try:
                _load_stage2(HERE / rel, "layout_red_" + rel[:-3])
            except SystemExit as e:
                assert "REFUSING" in str(e) and "Z:/does/not/exist" in str(e), str(e)
            else:
                raise AssertionError(f"{rel} imported with a non-existent navsim root: no refusal")
        finally:
            restore()


@check("P6 layout: with TANITAD_NAVSIM_ROOT set, EXPORT of every eval module is <root>/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz (the same file name as before); a mutant with the old literal FAILs")
def t_layout_export_wiring():
    with tempfile.TemporaryDirectory() as td:
        root = td.replace("\\", "/")
        want = root + "/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
        restore = _env_swap(TANITAD_NAVSIM_ROOT=root, REFE_NAVTEST_EXPORT=None)
        try:
            for rel in ("stage2_sft.py", "stage3_full.py", "selection_quality.py", "command_adherence.py"):
                m = _load_stage2(HERE / rel, "layout_wire_" + rel[:-3])
                assert m.EXPORT == want, (rel, m.EXPORT, want)
            src = (HERE / "stage2_sft.py").read_text(encoding="utf-8")
            old = 'EXPORT = _navsim_root() + "/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"'
            assert src.count(old) == 1
            bad = Path(tempfile.mkdtemp()) / "stage2_sft.py"
            bad.write_text(src.replace(old, 'EXPORT = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"'), encoding="utf-8")

            def mut():
                assert _load_stage2(bad, "layout_wire_mut").EXPORT == want
            must_fail(mut)
        finally:
            restore()

'''
rep('''def collect():
    return [v for k, v in sorted(globals().items()) if k.startswith("t_") and hasattr(v, "check_name")]''', NEW + '''
def collect():
    return [v for k, v in sorted(globals().items()) if k.startswith("t_") and hasattr(v, "check_name")]''')
open(p, "wb").write(s.encode("utf-8"))
print("test edited")
