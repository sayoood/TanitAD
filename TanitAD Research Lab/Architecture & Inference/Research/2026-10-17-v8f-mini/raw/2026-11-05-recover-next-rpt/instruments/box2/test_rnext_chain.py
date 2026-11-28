"""Dev-box tests of the RECOVER-NEXT box-2 package (CPU only, no network: tests/netguard.py raises before anything leaves the process; shell children get BASH_ENV functions that block ssh/scp/...).
Run:  CUDA_VISIBLE_DEVICES=-1 python -I tests/test_rnext_chain.py            (work dir: RNEXT_TEST_WORK, else a temp dir). Exit 0 iff every test passed. Log: logs/test_rnext_chain.log.
The checker is driven through its CLI in child processes with the RNEXT_HOME / RNEXT_RUNSM / RNEXT_REPO / RNEXT_DISK / RNEXT_FLOOR_BYTES hooks; the shipped trainer is imported from the shipped tar (sha256-checked) to read its parser and its real RecoverPerturb."""
import copy, hashlib, importlib.util, json, os, re, shutil, subprocess, sys, tarfile, tempfile, time, ast

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PKG = os.path.dirname(HERE); B2 = PKG + "/box2_files"; CHECK = os.environ.get("RNEXT_TEST_CHECK", B2 + "/rnext_check.py").replace("\\", "/")      # RNEXT_TEST_CHECK: a mutated copy of the checker, for the tests' own mutation runs
TAR = "E:/Projects/TanitAD-artifacts/dagland/recover_ft_launch/repo_recft_tip.tar"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, HERE)
import netguard  # noqa: E402
netguard.install()
WORK = (os.environ.get("RNEXT_TEST_WORK") or tempfile.mkdtemp(prefix="rnext_test_")).replace("\\", "/")
os.makedirs(WORK, exist_ok=True)
spec = importlib.util.spec_from_file_location("rnext_check_mod", CHECK)
os.environ.update({"RNEXT_HOME": B2, "RNEXT_REPO": WORK + "/tip", "RNEXT_RUNSM": WORK + "/modrun", "RNEXT_DISK": WORK})
RC = importlib.util.module_from_spec(spec); spec.loader.exec_module(RC)          # module for constants / pure functions only (the CLI is what the tests drive)
TAGS = [t for t, _, _ in RC.RUNS]
XT = [t for t, _, _ in RC.EXTRA]                       # RAPR-P01 (addendum 2 row E): in scope only where its argv is shipped
XT_R = [t for t in XT if RC.INFO[t][0] == "RAPRP01"]; XT_T = [t for t in XT if RC.INFO[t][0] == "RPT"]       # RAPR-P01's two tags / RPT's two tags
XTAGS = TAGS + XT
TESTS, FAILS = [], []


def test(fn):
    TESTS.append(fn); return fn


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def write(p, s, mode="w"):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode, **({} if "b" in mode else {"encoding": "utf-8", "newline": ""})) as f:
        f.write(s)


# ---- the shipped tree (extracted once from the shipped tar; the two bound files are sha256-checked) ---------------------------------------------------------------------------------------------------------------------------
TIPDIR = WORK + "/tip"
if not os.path.exists(TIPDIR + "/stack/scripts/v8f_mini_train.py"):
    with tarfile.open(TAR) as tf:
        tf.extractall(TIPDIR, members=[m for m in tf.getmembers() if m.name.startswith(("stack/", "taniteval/"))], filter="data")
_sp = importlib.util.spec_from_file_location("mk_rpt_pins_mod", PKG + "/mk_rpt_pins.py"); MP = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(MP)
for rel, src in MP.SOURCES.items():      # RPT: the shipped tree on box 2 is that tip with exactly these three files replaced (launch_rpt.sh ships them); SHIPPED binds their sha256
    os.makedirs(os.path.dirname(f"{TIPDIR}/{rel}"), exist_ok=True); shutil.copy(src, f"{TIPDIR}/{rel}")
for rel, want in RC.SHIPPED.items():
    got = hashlib.sha256(open(os.path.join(TIPDIR, *rel.split("/")), "rb").read()).hexdigest()
    check(got == want, f"shipped tar member {rel} sha256 {got} != {want}")


class SB:
    def __init__(self, name, repo=None, floor="0", extra=False):
        self.extra = extra; self.root = f"{WORK}/{name}"; shutil.rmtree(self.root, ignore_errors=True)
        self.home, self.runsm, self.repo, self.disk, self.floor = self.root + "/home", self.root + "/runs", repo or TIPDIR, self.root, floor
        os.makedirs(self.home + "/argv", exist_ok=True)
        for t in self.tags():
            shutil.copy(f"{B2}/argv/{t}.train.txt", f"{self.home}/argv/{t}.train.txt")

    def tags(self): return TAGS + XT_R if self.extra is True else (XTAGS if self.extra == "rpt" else TAGS)      # extra=True: the six + RAPR-P01 (eight argv); extra="rpt": all ten
    def argv(self, tag): return open(f"{self.home}/argv/{tag}.train.txt", encoding="utf-8").read().strip()
    def set_argv(self, tag, text): write(f"{self.home}/argv/{tag}.train.txt", text + "\n")
    def rc(self): return self.runsm + "/rnext"
    def gdir(self, tag): k, s = RC.INFO[tag]; return f"{self.runsm}/gate_rnext/M2b_s{s}_recft_{k}"
    def odir(self, tag): k, s = RC.INFO[tag]; return f"{self.runsm}/M2b_s{s}_recft_{k}"
    def mk_inputs(self):
        for t in self.tags():
            toks = self.argv(t).split(" ")
            write(self.runsm + toks[toks.index("--init-ckpt") + 1][len(RC.DECL):].replace("\\", "/"), "x", "w")


BOOT = "import sys,runpy; sys.path.insert(0,%r); import netguard; netguard.install(); sys.argv=%r; runpy.run_path(%r, run_name='__main__')"


def run_check(args, sb):
    env = {**os.environ, "RNEXT_HOME": sb.home, "RNEXT_RUNSM": sb.runsm, "RNEXT_REPO": sb.repo, "RNEXT_DISK": sb.disk, "RNEXT_FLOOR_BYTES": sb.floor, "CUDA_VISIBLE_DEVICES": "-1"}
    r = subprocess.run([sys.executable, "-I", "-c", BOOT % (HERE, [CHECK] + list(args), CHECK)], env=env, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)


_TR = {}


def trainer():
    """The shipped trainer module (imported from the sha256-checked tar extract) - its parser and its real RecoverPerturb."""
    if not _TR:
        sys.path.insert(0, TIPDIR + "/stack"); sys.path.insert(0, TIPDIR + "/taniteval")
        sp = importlib.util.spec_from_file_location("v8f_mini_train", TIPDIR + "/stack/scripts/v8f_mini_train.py")
        m = importlib.util.module_from_spec(sp); sys.modules["v8f_mini_train"] = m; sp.loader.exec_module(m)
        import tanitad
        check(os.path.abspath(tanitad.__file__).replace("\\", "/").startswith(TIPDIR), f"tanitad imported from {tanitad.__file__}, not the shipped tree")
        _TR["m"] = m
    return _TR["m"]


def real_perturb(tag, line=None):
    """RecoverPerturb built the way the trainer's make_recover builds it, from the parsed shipped argv (a 1-row stub table stands in for the bank-derived one)."""
    m = trainer(); R = m.R; a = m.build_parser().parse_args((line or open(f"{B2}/argv/{tag}.train.txt").read().strip()).split(" "))
    s = RC.INFO[tag][1]; tb = None
    if a.recover_p > 0 and a.recover_dev_law == "empirical" and a.recover_dev_bank:
        cols = {k: [0.0] for k in R.TABLE_FIELDS}
        if a.recover_dev_shape == "pertick" and not getattr(real_perturb, "flat_table", False): cols.update({k: [[0.1 * i for i in range(10)]] for k in R.PER_TICK_FIELDS})      # the per-tick columns [N, F] (a 1-record stand-in)
        tb = R.RecoverDevTable(cols, source_sha256=RC.BANK_SHA[s], path=a.recover_dev_bank + "\\bank.json")
    return a, R.RecoverPerturb(p=a.recover_p, dev_law=a.recover_dev_law, table=tb, target=a.recover_target, rejoin_steps=a.recover_rejoin_steps, seed=a.recover_seed, kin_mode=a.kin_mode,
                               bank_path=(tb.path if tb else (a.recover_dev_bank or None)), bank_sha256=(a.recover_dev_bank_sha256 or None), dev_shape=getattr(real_perturb, "force_shape", None) or a.recover_dev_shape)


BASE = json.load(open(HERE + "/fixtures/R_0.selfcheck.json", encoding="utf-8"))          # the REAL banked R_0 gate selfcheck (recover_ft_launch/evidence/R_0), patched per kind below


def stub_sc(tag):
    kind, s = RC.INFO[tag]; sc = copy.deepcopy(BASE); _, rec = real_perturb(tag)
    sc["levers"]["recover"] = rec.built_levers()                               # the REAL library object's built levers
    sc["info"]["init_ckpt"]["sha256"] = RC.INIT_SHA[s]
    if RC.KINDS[kind]["table"]:
        sc["info"]["recover"]["bank_sha256"] = RC.BANK_SHA[s]; sc["levers"]["recover_table"]["source_sha256"] = RC.BANK_SHA[s]
    else:
        sc["info"].pop("recover", None); sc["levers"]["recover_table"] = None
    if RC.KINDS[kind].get("shape"):                                            # RPT: what the trainer's selfcheck adds for the per-tick shape (info.recover.dev_shape, the table's per_tick block and drop counts, the probe's read-back)
        sc["info"]["recover"]["dev_shape"] = "pertick"
        tb = {"ticks": 10, **{f"mean_abs_{k}": [0.1] * 10 for k in ("dv_t", "da_t", "dom_t")}}
        for t_ in (sc["levers"]["recover_table"], sc["info"]["recover"]["table"]):
            t_["per_tick"] = copy.deepcopy(tb); t_["counts"] = {"records_ok": 1000, "kept": 990, "dropped_logged_row_unreal": 4, "dropped_per_tick_row_unreal": 6}
        sc["recover_probe"]["pertick_v_max_abs_err"] = 1.2e-6
    return sc


def stage_gate(sb, tag, sc, log="SELFCHECK PASS {}"):
    write(f"{sb.gdir(tag)}/selfcheck.json", json.dumps(sc, indent=1))
    if log is not None:
        write(f"{sb.rc()}/logs/{tag}.gate.log", log + "\n")


def pinned_sb(name, tags=None, **kw):
    sb = SB(name, **kw)
    for t in (tags or TAGS):
        rc, out = run_check(["pin", t], sb); check(rc == 0, f"pin {t}: rc {rc}: {out}")
    return sb


# ================================================================================ the tests ===================================================================================================================================
@test
def t_netguard_canaries():
    check(netguard.installed(), "netguard is not installed in the test process")
    for fn, args in ((lambda: __import__("socket").create_connection(("127.0.0.1", 1)), ()), (lambda: __import__("socket").getaddrinfo("tanitad-box2", 22), ())):
        try: fn(); raise AssertionError("a connect / lookup did not raise")
        except RuntimeError as e: check("NETWORK-BLOCKED" in str(e), str(e))
    canary = WORK + "/canary/ssh.cmd"; write(canary, "@echo off\r\necho harmless stand-in\r\n")        # a harmless file NAMED ssh: the guard must refuse it by name, before any exec
    try: subprocess.run([canary]); raise AssertionError("a process named ssh was spawned")
    except RuntimeError as e: check("NETWORK-BLOCKED" in str(e), str(e))
    check(netguard.blocked_cmd(["scp", "x"]) and netguard.blocked_cmd(["bash", "-c", "echo tanitad-box2"]) and not netguard.blocked_cmd(["bash", "pull_rnext.sh"]), "blocked_cmd decisions")
    r = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0,%r); import netguard; netguard.install(); print(netguard.installed())" % HERE], capture_output=True, text=True)
    check(r.stdout.strip() == "True", "the child bootstrap does not install the guard: " + r.stdout + r.stderr)
    nb = bash(["-c", "type -t ssh; type -t scp; type -t rsync"], {}); check(nb[1].split() == ["function"] * 3, f"BASH_ENV does not define blocking functions: {nb}")
    del netguard.TRIPPED[:]                                  # the deliberate canary trips above are not counted against the run


@test
def t_checker_has_no_network_code():
    tree = ast.parse(open(CHECK, encoding="utf-8").read()); mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import): mods |= {a.name.split(".")[0] for a in n.names}
        if isinstance(n, ast.ImportFrom): mods.add((n.module or "").split(".")[0])
    check(mods <= {"ast", "hashlib", "json", "os", "shutil", "sys", "time"}, f"unexpected imports in rnext_check.py: {mods}")


@test
def t_argv_files_and_card():
    for t in TAGS:
        a = open(f"{B2}/argv/{t}.train.txt", "rb").read(); b = open(f"{PKG}/argv/{t}.train.txt", "rb").read()
        check(a == b, f"{t}: box2_files/argv differs from recover_next/argv"); check(b"\r" not in a and a.count(b"\n") == 1, f"{t}: not one LF line")
        check(" ".join(RC.expected_tokens(t)) == a.decode().strip(), f"{t}: the checker's frozen spec does not reproduce the argv file")
    sha = hashlib.sha256(open(f"{PKG}/PREREG_RECOVER_NEXT_FROZEN.md", "rb").read()).hexdigest()
    check(sha == RC.PREREG_SHA and sha in open(f"{PKG}/FROZEN_SHA.txt").read(), f"frozen card sha256 {sha} != the checker's / FROZEN_SHA.txt")
    check(open(f"{PKG}/PREREG_RECOVER_NEXT.md", "rb").read() == open(f"{PKG}/PREREG_RECOVER_NEXT_FROZEN.md", "rb").read(), "the working copy of the card differs from the frozen one")


@test
def t_recover_defaults_bound_to_shipped_bytes():
    d, ln = RC.shipped_defaults()
    check((d, ln) == (RC.DEFAULTS, 28) and d["dev_shape"] == "constant" and RC.DEFAULTS_LINE == ln, f"shipped RECOVER_DEFAULTS {d} at line {ln} (the checker expects line {RC.DEFAULTS_LINE})")
    check(RC.KINDS["RTL"]["rejoin"] == d["rejoin_steps"] == 10, "RTL's built rejoin_steps is not the shipped default")
    sb = SB("defaults", repo=WORK + "/repo_edit"); shutil.rmtree(sb.repo, ignore_errors=True)
    for rel in RC.SHIPPED:
        os.makedirs(os.path.dirname(f"{sb.repo}/{rel}"), exist_ok=True); shutil.copy(f"{TIPDIR}/{rel}", f"{sb.repo}/{rel}")
    p = f"{sb.repo}/stack/tanitad/data/v8f_recover.py"; txt = open(p, encoding="utf-8", newline="").read()
    write(p, txt.replace('"rejoin_steps": 10', '"rejoin_steps": 12')); sb.mk_inputs()
    rc, out = run_check(["preflight"], sb); check(rc == 3 and "RECOVER_DEFAULTS" in out and "v8f_recover.py: sha256" in out, f"edited default not refused: {rc} {out[:300]}")


@test
def t_pin_passes_for_all_six_and_is_idempotent():
    sb = SB("pin")
    for t in TAGS:
        rc, out = run_check(["pin", t], sb); check(rc == 0 and f"pinned {t}" in out, f"{t}: {rc} {out}")
        tr = open(f"{sb.rc()}/pin/{t}.train.txt").read().strip(); ga = open(f"{sb.rc()}/pin/{t}.gate.txt").read().strip()
        k, s = RC.INFO[t]; check(tr == sb.argv(t) and ga.endswith(" --selfcheck-only") and f"gate_rnext\\M2b_s{s}_recft_{k}" in ga.replace("/", "\\"), f"{t}: pin files")
        check(run_check(["pin", t], sb)[0] == 0, f"{t}: second pin refused")
    open(f"{sb.rc()}/pin/RTL_0.train.txt", "w").write("--different line\r\n")
    rc, out = run_check(["pin", "RTL_0"], sb); check(rc == 2 and "REFUSED" in out and "pins another train argv" in out, f"a changed pin was not refused: {rc} {out}")


MUT = [("RAPR_0", lambda l: l.replace("--recover-p 0.3", "--recover-p 0.2"), "--recover-p = '0.2' but the frozen RAPR_0 argv has '0.3'"),
       ("RAPR_1", lambda l: l + " --recover-dev-bank D:\\tanitad\\runs\\v8f_mini_full\\dagger_bank_dag2\\s1", "unexpected flag --recover-dev-bank"),
       ("RAPR_0", lambda l: l + " --recover-dev-bank-sha256 " + RC.BANK_SHA[0], "a bank flag with the a-priori law"),
       ("RAPR_0", lambda l: l.replace("--recover-dev-law apriori", "--recover-dev-law empirical"), "--recover-dev-law = 'empirical' but the frozen RAPR_0 argv has 'apriori'"),
       ("RAPR_1", lambda l: l.replace("M2b_s1_dagoacc\\", "M2b_s1_dagoacc_old\\"), "--init-ckpt ="),
       ("RTL_0", lambda l: l.replace("--recover-target logged", "--recover-target rejoin"), "--recover-target = 'rejoin' but the frozen RTL_0 argv has 'logged'"),
       ("RTL_1", lambda l: l + " --recover-rejoin-steps 10", "does nothing under --recover-target logged"),
       ("RTL_0", lambda l: l.replace("M2b_s0_dagoacc", "M2b_s1_dagoacc"), "--init-ckpt ="),
       ("RTL_1", lambda l: l.replace("--recover-seed 1", "--recover-seed 0"), "--recover-seed = '0' but the frozen RTL_1 argv has '1'"),
       ("RTL_0", lambda l: l.replace("dagger_bank_dag2\\s0", "dagger_bank_dag2\\s1"), "--recover-dev-bank ="),
       ("RP01_0", lambda l: l.replace("--recover-p 0.1", "--recover-p 0.3"), "--recover-p = '0.3' but the frozen RP01_0 argv has '0.1'"),
       ("RP01_1", lambda l: l + " --recover-p 0.3", "--recover-p is given 2 times"),
       ("RP01_0", lambda l: l.replace("--steps 3000", "--steps 2000"), "--steps = '2000'"),
       ("RP01_1", lambda l: l.replace(RC.BANK_SHA[1], RC.BANK_SHA[1][:40]), "--recover-dev-bank-sha256 ="),
       ("RAPR_0", lambda l: l + "\n--arm M2b", "exactly one line")]


@test
def t_mutated_argv_refused_with_named_reason():
    for tag, mut, reason in MUT:
        sb = SB("mut"); line = sb.argv(tag); new = mut(line); check(new != line, f"{tag}: the mutation changed nothing")
        sb.set_argv(tag, new)
        rc, out = run_check(["pin", tag], sb)
        check(rc == 2 and "REFUSED" in out and reason in out, f"{tag} [{reason}]: rc {rc}: {out[:400]}")
        check(not os.path.exists(f"{sb.rc()}/pin/{tag}.train.txt"), f"{tag}: a pin was written for a refused argv")
        sib = [t for t in TAGS if t != tag][0]; check(run_check(["pin", sib], sb)[0] == 0, f"control: unmutated sibling {sib} was refused")
        sb.mk_inputs(); rc2, out2 = run_check(["preflight"], sb); check(rc2 == 3 and f"{tag}: argv audit" in out2 or (rc2 == 3 and f"{tag}:" in out2), f"{tag}: preflight did not refuse: {rc2} {out2[:300]}")


@test
def t_preflight_positive_and_negatives():
    sb = SB("pre"); sb.mk_inputs()
    rc, out = run_check(["preflight"], sb); check(rc == 3 and "bank.json missing or its sha256" in out and "init ckpt" not in out, f"banks absent on the dev box must be named (inits are staged): {rc} {out[:300]}")
    for t in TAGS:
        k, s_ = RC.INFO[t]
        if RC.KINDS[k]["table"]: write(f"{sb.runsm}/dagger_bank_dag2/s{s_}/bank.json", "{}")
    env = {**os.environ, "RNEXT_HOME": sb.home, "RNEXT_RUNSM": sb.runsm, "RNEXT_REPO": sb.repo, "RNEXT_DISK": sb.disk, "RNEXT_FLOOR_BYTES": "0"}
    code = ("import sys,os; sys.path.insert(0,%r); sys.path.insert(0,%r); import netguard; netguard.install(); import rnext_check as rc; o=rc.sha_file; "
            "rc.sha_file=lambda f: ((rc.BANK_SHA[0] if os.sep.join(['s0','bank.json']) in f else rc.BANK_SHA[1]) if f.endswith('bank.json') else o(f)); sys.exit(rc.main(['preflight']))") % (HERE, B2)
    r = subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True)
    check(r.returncode == 0 and "preflight OK" in r.stdout, f"positive preflight (bank.json hash stubbed: the real bank is not on the dev box): {r.returncode} {r.stdout} {r.stderr[-300:]}")
    os.remove(sb.runsm + "/M2b_s1_dagoacc/ckpt_last.pt"); rc, out = run_check(["preflight"], sb); check(rc == 3 and "init ckpt" in out and "RAPR_1" in out, f"missing init not named: {out[:300]}")
    sb2 = SB("pre_floor", floor="1e18"); rc, out = run_check(["preflight", "run", "RAPR_0"], sb2); check(rc == 9 and "disk floor" in out, f"floor stop: {rc} {out}")
    rc, out = run_check(["preflight"], sb2); check(rc == 3 and "disk floor" in out, f"floor in preflight: {rc} {out[:200]}")
    sb3 = SB("pre_sha", repo=WORK + "/repo_edit2"); shutil.rmtree(sb3.repo, ignore_errors=True)
    for rel in RC.SHIPPED: os.makedirs(os.path.dirname(f"{sb3.repo}/{rel}"), exist_ok=True); shutil.copy(f"{TIPDIR}/{rel}", f"{sb3.repo}/{rel}")
    write(f"{sb3.repo}/stack/scripts/v8f_mini_train.py", "# edited\n" + open(f"{TIPDIR}/stack/scripts/v8f_mini_train.py", encoding="utf-8", newline="").read())
    rc, out = run_check(["preflight"], sb3); check(rc == 3 and "v8f_mini_train.py: sha256" in out, f"edited trainer not refused: {out[:300]}")


@test
def t_trainer_accepts_the_six_argv_and_builds_the_registered_levers():
    m = trainer(); sha = hashlib.sha256(open(TIPDIR + "/stack/scripts/v8f_mini_train.py", "rb").read()).hexdigest(); check(sha == RC.SHIPPED["stack/scripts/v8f_mini_train.py"], "trainer sha")
    for t in TAGS:
        k, s = RC.INFO[t]; K = RC.KINDS[k]; a, rec = real_perturb(t)
        check((a.recover_p, a.recover_dev_law, a.recover_target, a.recover_rejoin_steps, a.recover_seed) == (K["p"], K["law"], K["target"], K["rejoin"], s), f"{t}: parsed {a.recover_p, a.recover_dev_law, a.recover_target, a.recover_rejoin_steps, a.recover_seed}")
        check(rec.problems() == [], f"{t}: perturber problems {rec.problems()}")
        lv = rec.built_levers(); check((lv["recover_p"], lv["recover_dev_law"], lv["recover_target"], lv["recover_rejoin_steps"], lv["recover_seed"]) == (K["p"], K["law"], K["target"], K["rejoin"], s), f"{t}: built {lv}")
        check(m.build_parser().parse_args(open(f"{B2}/argv/{t}.train.txt").read().split() + ["--selfcheck-only"]).selfcheck_only, f"{t}: --selfcheck-only")
        if k == "RAPR":
            check(m.bad_recover(a, rec, None) == [] and lv["recover_dev_bank"] is None and lv["recover_dev_bank_sha256"] is None, f"{t}: bad_recover {m.bad_recover(a, rec, None)}")
            a2, rec2 = real_perturb(t, open(f"{B2}/argv/{t}.train.txt").read().strip() + " --recover-dev-bank X --recover-dev-bank-sha256 " + RC.BANK_SHA[s])
            br = m.bad_recover(a2, rec2, None); check(any("apriori" in x for x in br), f"{t}: the trainer's own refusal of a bank with the a-priori law did not fire: {br}")


@test
def t_gate_stubs_pass_and_token_binds():
    for t in TAGS:
        sb = pinned_sb("gate", [t]); stage_gate(sb, t, stub_sc(t))
        rc, out = run_check(["gate", t, "0"], sb); check(rc == 0 and f"ZZRNEXT-GATE-PASS {t}" in out, f"{t}: {rc} {out}")
        tok = json.load(open(f"{sb.rc()}/gate_token_{t}.json")); b = tok["binding"]
        check(tok["verdict"] == "PASS" and tok["prereg_sha256"] == RC.PREREG_SHA and b["commit"] == RC.TIP and b["argv_sha256"] == RC.tokens_sha(sb.argv(t).split(" ")), f"{t}: token")
        check(run_check(["gate-cached", t], sb)[0] == 0, f"{t}: gate-cached no after PASS")
        sc = stub_sc(t); sc["argv_sha"] = "changed"; stage_gate(sb, t, sc, None); check(run_check(["gate-cached", t], sb)[0] == 1, f"{t}: cached after a selfcheck change")
        stage_gate(sb, t, stub_sc(t), None); check(run_check(["gate-cached", t], sb)[0] == 0, f"{t}: not cached again after restoring the selfcheck")
        os.remove(f"{sb.rc()}/gate_token_{t}.json"); check(run_check(["gate-cached", t], sb)[0] == 1, f"{t}: cached without a token")


def _set(path, val):
    def f(sc):
        d = sc
        for k in path[:-1]: d = d[k]
        d[path[-1]] = val
    return f


def _pop(path):
    def f(sc):
        d = sc
        for k in path[:-1]: d = d[k]
        d.pop(path[-1])
    return f


WRONG = {"RAPR": [("built lever recover_p = 0.2", _set(["levers", "recover", "recover_p"], 0.2)), ("built lever recover_dev_law = 'empirical'", _set(["levers", "recover", "recover_dev_law"], "empirical")),
                  ("built lever recover_target = 'logged'", _set(["levers", "recover", "recover_target"], "logged")), ("built lever recover_seed", _set(["levers", "recover", "recover_seed"], 9)),
                  ("a bank is built under the a-priori law", _set(["levers", "recover", "recover_dev_bank"], "D:\\x\\bank.json")),
                  ("a deviation table was built", lambda sc: sc["info"].__setitem__("recover", copy.deepcopy(BASE["info"]["recover"]))), ("levers.recover_table is not None", _set(["levers", "recover_table"], {"n": 5}))],
         "RTL": [("built lever recover_target = 'rejoin'", _set(["levers", "recover", "recover_target"], "rejoin")), ("built lever recover_rejoin_steps = 5", _set(["levers", "recover", "recover_rejoin_steps"], 5)),
                 ("built lever recover_p = 0.1", _set(["levers", "recover", "recover_p"], 0.1)), ("built lever recover_dev_law = 'apriori'", _set(["levers", "recover", "recover_dev_law"], "apriori")),
                 ("info.recover (the table build record) is absent", _pop(["info", "recover"])), ("built bank sha256", _set(["info", "recover", "bank_sha256"], "0" * 64)),
                 ("levers.recover_table is empty", _set(["levers", "recover_table"], None)), ("K-B3 window-list", _set(["info", "recover", "kb3", "window_list_sha256"], "0" * 64))],
         "RP01": [("built lever recover_p = 0.3", _set(["levers", "recover", "recover_p"], 0.3)), ("built lever recover_target = 'logged'", _set(["levers", "recover", "recover_target"], "logged")),
                  ("built lever recover_rejoin_steps = 12", _set(["levers", "recover", "recover_rejoin_steps"], 12)), ("info.recover (the table build record) is absent", _pop(["info", "recover"])),
                  ("built lever recover_dev_bank_sha256", _set(["levers", "recover", "recover_dev_bank_sha256"], None))]}
GENERIC = [("init ckpt sha256", _set(["info", "init_ckpt", "sha256"], "0" * 64)), ("gate train_sha256", _set(["info", "train_sha256"], "0" * 64)), ("steps_total 2000", _set(["steps_total"], 2000)),
           ("recover_probe pass = False", _set(["recover_probe", "pass"], False)), ("declared_vs_built = FAIL", _set(["declared_vs_built"], "FAIL"))]


@test
def t_gate_wrong_built_lever_fails():
    for t in TAGS:
        kind = RC.INFO[t][0]
        for reason, mut in WRONG[kind] + GENERIC:
            sb = pinned_sb("gatebad", [t]); sc = stub_sc(t); mut(sc); stage_gate(sb, t, sc)
            rc, out = run_check(["gate", t, "0"], sb); mk = f"{sb.rc()}/RUN_{t}.FAILED"
            check(rc == 1 and os.path.exists(mk) and reason in open(mk).read() and not os.path.exists(f"{sb.rc()}/gate_token_{t}.json"), f"{t} [{reason}]: rc {rc} {out[:300]}")
    for t, other in (("RAPR_0", "RTL_0"), ("RTL_0", "RAPR_0"), ("RP01_0", "RTL_0"), ("RTL_1", "RP01_1")):       # a stub of another kind is refused
        sb = pinned_sb("gatecross", [t]); sc = stub_sc(other); sc["info"]["init_ckpt"]["sha256"] = RC.INIT_SHA[RC.INFO[t][1]]; stage_gate(sb, t, sc)
        rc, out = run_check(["gate", t, "0"], sb); check(rc == 1, f"{other}'s selfcheck passed the {t} gate")
    sb = pinned_sb("gatelog", ["RTL_0"]); stage_gate(sb, "RTL_0", stub_sc("RTL_0"), "no pass line here"); check(run_check(["gate", "RTL_0", "0"], sb)[0] == 1, "a log without SELFCHECK PASS passed")
    sb = pinned_sb("gaterc", ["RAPR_1"]); stage_gate(sb, "RAPR_1", stub_sc("RAPR_1")); check(run_check(["gate", "RAPR_1", "1"], sb)[0] == 1, "a gate process rc 1 passed")
    sb = pinned_sb("gatemissing", ["RP01_0"]); check(run_check(["gate", "RP01_0", "0"], sb)[0] == 1, "a missing selfcheck.json passed")


def stage_run(sb, t, **over):
    k, s = RC.INFO[t]; out = sb.odir(t); sm = {"steps": 3000, "recover_n": 900, "recover_share": 0.3, **RC.KP1}; sm.update(over.get("summary", {}))
    write(f"{out}/summary.json", json.dumps(sm)); write(f"{out}/argv.json", json.dumps({"argv": over.get("argv", sb.argv(t).split(" "))}))
    if not over.get("no_ckpt"): write(f"{out}/ckpt_last.pt", b"ckpt-" + t.encode(), "wb")


@test
def t_post_markers():
    for t in TAGS:
        sb = pinned_sb("post", [t]); stage_run(sb, t); rc, out = run_check(["post", t, "0"], sb)
        mk = f"{sb.rc()}/RUN_{t}.done"; check(rc == 0 and os.path.exists(mk), f"{t}: {rc} {out}")
        ev = json.load(open(mk))["evidence"]; check(ev["ckpt_sha256"] == hashlib.sha256(b"ckpt-" + t.encode()).hexdigest() and ev["argv_match_pin"] is True, f"{t}: evidence {ev}")
        run_check(["post", t, "0"], sb); check(any(n.startswith(f"RUN_{t}.done.prev_") for n in os.listdir(sb.rc())), f"{t}: the earlier marker was not moved aside")
    for reason, kw, rcode in (("KR1", {"summary": {"recover_n": 0}}, "0"), ("summary steps 2000", {"summary": {"steps": 2000}}, "0"), ("KP1: summary train_sha256", {"summary": {"train_sha256": "0" * 64}}, "0"),
                              ("argv.json != the pinned", {"argv": ["x"]}, "0"), ("the trainer exited rc 1", {}, "1"), ("ckpt_last.pt missing", {"no_ckpt": True}, "0")):
        sb = pinned_sb("postbad", ["RTL_1"]); stage_run(sb, "RTL_1", **kw); rc, out = run_check(["post", "RTL_1", rcode], sb); mk = f"{sb.rc()}/RUN_RTL_1.FAILED"
        check(rc == 0 and os.path.exists(mk) and reason in open(mk).read() and not os.path.exists(f"{sb.rc()}/RUN_RTL_1.done"), f"[{reason}] {rc} {out[:200]}")


@test
def t_finish_and_stopped_markers():
    sb = pinned_sb("fin", ["RAPR_0"]); stage_run(sb, "RAPR_0"); run_check(["post", "RAPR_0", "0"], sb); run_check(["fail", "RAPR_1", "pin", "refused"], sb)
    rc, out = run_check(["finish"], sb); st = json.load(open(f"{sb.rc()}/CHAIN_RNEXT.done"))["runs"]
    check(rc == 0 and st == {"RAPR_0": "done", "RAPR_1": "FAILED", "RTL_0": "none", "RTL_1": "none", "RP01_0": "none", "RP01_1": "none"}, f"{rc} {st}")
    run_check(["stopped", "safety", "stop"], sb); check(os.path.exists(f"{sb.rc()}/CHAIN_RNEXT.STOPPED") and not os.path.exists(f"{sb.rc()}/CHAIN_RECOVER.done"), "stopped marker")


def bash(args, extra, timeout=120):
    exe = "C:/Program Files/Git/bin/bash.exe"; exe = exe if os.path.exists(exe) else "bash"
    env = {**os.environ, "BASH_ENV": HERE + "/blockenv.sh", "RNEXT_TRIPWIRE": WORK + "/tripwire.txt", **extra}
    r = subprocess.run([exe] + args, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.replace("\r", ""), r.stderr


@test
def t_shell_scripts_static_and_waker_scenarios():
    for f in ("pull_rnext.sh", "wake_rnext.sh", "tests/fake_box.sh", "tests/fake_scp.sh"):
        rc, o, e = bash(["-n", f"{PKG}/{f}"], {}); check(rc == 0, f"bash -n {f}: {e}")
    w = open(f"{PKG}/wake_rnext.sh").read(); code = "\n".join(l for l in w.splitlines() if not l.lstrip().startswith("#"))
    check('[ -e "$MIR/CHAIN_RNEXT.done" ] || [ -e "$MIR/CHAIN_RNEXT.STOPPED" ]' in code, "the waker's end test is not `[ -e a ] || [ -e b ]`")
    check(not re.search(r"(^|[;&|(]\s*|\s)ls\s+[^\s|;&]+\s+[^\s|;&]+", code.replace("ls=$(", "")), "the waker runs `ls a b`")
    fb = f"bash {HERE}/fake_box.sh"; lst = WORK + "/listing.txt"
    cases = [("", {}, 5, "WAKER TIMEOUT"), ("CHAIN_RNEXT.done\nRUN_RAPR_0.done", {}, 0, "CHAIN DONE"), ("CHAIN_RNEXT.STOPPED", {}, 4, "CHAIN STOPPED"), ("GUARD_STOP.txt", {}, 4, "GUARD STOP"),
             ("RUN_RTL_0.FAILED", {}, 3, "RUN FAILED"), ("RUN_RTL_0.FAILED", {"RNEXT_SEEN_FAILED": "1"}, 5, "WAKER TIMEOUT"), ("RUN_RTL_0.FAILED.prev_20261103_010101", {}, 5, "WAKER TIMEOUT"),
             ("RUN_RAPR_0.done\nRUN_RAPR_1.done", {}, 0, "NEW DONE (2 of 6)"), ("RUN_RAPR_0.done\nRUN_RAPR_1.done", {"RNEXT_SEEN": "2"}, 5, "WAKER TIMEOUT"), ("CHAIN_RECOVER.done", {}, 5, "WAKER TIMEOUT"),
             ("CHAIN_RNEXT.done\nCHAIN_RNEXT.STOPPED", {}, 0, "CHAIN DONE")]
    for listing, extra, ec, msg in cases:
        write(lst, listing + "\n"); rc, o, e = bash([f"{PKG}/wake_rnext.sh"], {"RNEXT_SSH": fb, "RNEXT_FAKE_LISTING": lst, "RNEXT_WAKE_N": "2", "RNEXT_SLEEP": "0", **extra})
        check(rc == ec and msg in o, f"waker listing {listing!r} {extra}: rc {rc} (want {ec}) out {o!r} err {e[:200]}")


@test
def t_pull_script_with_local_stand_in():
    root = WORK + "/fakebox"; shutil.rmtree(root, ignore_errors=True)
    for t in XTAGS:
        k, s = RC.INFO[t]; run = f"M2b_s{s}_recft_{k}"; ck = b"ckpt-pull-" + t.encode()
        for f, c in (("ckpt_last.pt", ck), ("summary.json", b"{}"), ("argv.json", b"{}"), ("selfcheck.json", b"{}"), ("train_log.jsonl", b""), ("train.log", b"")): write(f"{root}/{run}/{f}", c, "wb")
        write(f"{root}/gate_rnext/{run}/selfcheck.json", "{}"); write(f"{root}/rnext/RUN_{t}.done", json.dumps({"evidence": {"ckpt_sha256": hashlib.sha256(ck).hexdigest()}})); write(f"{root}/rnext/logs/{t}.gate.log", "x")
    env = {"RNEXT_SCP": f"bash {HERE}/fake_scp.sh", "RNEXT_FAKE_ROOT": root, "RNEXT_PY": sys.executable.replace("\\", "/")}
    dest = WORK + "/pulldest"; shutil.rmtree(dest, ignore_errors=True)
    rc, o, e = bash([f"{PKG}/pull_rnext.sh"], {**env, "RNEXT_DEST": dest}); check(rc == 0 and o.count("\nOK ") + (o.startswith("OK ")) == len(XTAGS) and "bad=0" in o, f"happy path: rc {rc} {o} {e[:300]}")
    check(os.path.exists(f"{dest}/M2b_s1_recft_RP01/SHA256SUMS.txt") and os.path.exists(f"{dest}/rnext_markers/RUN_RTL_0.done"), "pulled layout")
    write(f"{root}/M2b_s0_recft_RTL/ckpt_last.pt", b"corrupt", "wb"); os.remove(f"{root}/rnext/RUN_RP01_1.done")
    rc, o, e = bash([f"{PKG}/pull_rnext.sh"], {**env, "RNEXT_DEST": WORK + "/pulldest2"}); check(rc == 1 and "MISMATCH RTL_0" in o and "SKIP RP01_1" in o and "bad=1" in o, f"bad path: rc {rc} {o}")
    if os.path.exists(WORK + "/tripwire.txt"):
        raise AssertionError("a network tool was invoked by a shell script: " + open(WORK + "/tripwire.txt").read())


PS_PARSE = "param($f) $t=$null;$e=$null;$null = [System.Management.Automation.Language.Parser]::ParseFile($f,[ref]$t,[ref]$e); if ($e.Count) { $e | ForEach-Object { $_.Message }; exit 1 }; Write-Output 'PARSE-OK'"


def ps(args, timeout=90):
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass"] + args, capture_output=True, text=True, timeout=timeout); return r.returncode, r.stdout + r.stderr


@test
def t_guard_script_parses_and_stops_on_heat():
    write(WORK + "/parse.ps1", PS_PARSE); rc, o = ps(["-File", WORK + "/parse.ps1", "-f", B2 + "/rnext_guard.ps1"]); check(rc == 0 and "PARSE-OK" in o, f"rnext_guard.ps1 does not parse: {o}")
    src = open(B2 + "/rnext_guard.ps1", encoding="utf-8").read()
    for a, b in ((r"D:\tanitad\runs\v8f_mini_full\rnext", "RCDIR"), (r"D:\tanitad\logs\telemetry_full.csv", "CSV"), (r"D:\tanitad\logs\rnext_guard.log", "LOG")): check(a in src, f"{a} not in the guard")

    def run_guard(name, temps, marker_after=None, stale=False, marker_first=False):
        d = f"{WORK}/guard_{name}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d + "/rc")
        rows = "".join(f"2026-11-03 10:00:{i:02d},a,b,c,d,e,f,g,h,i,{t}\r\n" for i, t in enumerate(temps)); write(d + "/tel.csv", "ts,x\r\n" + rows, "wb" if False else "w")
        if stale: os.utime(d + "/tel.csv", (time.time() - 7200, time.time() - 7200))
        if marker_first: write(d + "/rc/CHAIN_RNEXT.done", "{}")
        s = src.replace(r"D:\tanitad\runs\v8f_mini_full\rnext", (d + "/rc").replace("/", "\\")).replace(r"D:\tanitad\logs\telemetry_full.csv", (d + "/tel.csv").replace("/", "\\")).replace(r"D:\tanitad\logs\rnext_guard.log", (d + "/guard.log").replace("/", "\\"))
        s = s.replace("v8f_mini_train", "zz_no_such_trainer_zz").replace("Start-Sleep -Seconds 60", "Start-Sleep -Seconds 1")       # the kill pattern is neutralised: no process on this machine can match
        write(d + "/g.ps1", s)
        p = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", d + "/g.ps1"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if marker_after: time.sleep(marker_after); write(d + "/rc/CHAIN_RNEXT.done", "{}")
        try: out, _ = p.communicate(timeout=60)
        except subprocess.TimeoutExpired: p.kill(); raise AssertionError(f"guard {name} did not exit")
        return os.path.exists(d + "/rc/GUARD_STOP.txt"), (open(d + "/guard.log").read() if os.path.exists(d + "/guard.log") else ""), out

    st, lg, o = run_guard("hot4", [55] * 16 + [62] * 4); check(st and "ZZGUARD-STOP TEMP: 4 of 20 >= 61 C, max 62 C" in lg, f"4 of 20 >= 61 C must stop: {lg} {o}")
    st, lg, o = run_guard("hot63", [50] * 19 + [63]); check(st and "max 63 C" in lg, f"one >= 63 C must stop: {lg} {o}")
    st, lg, o = run_guard("warm3", [50] * 17 + [61] * 3, marker_after=4, stale=True); check(not st and "chain ended; guard off (max USB 61 C)" in lg and "WARNING telemetry csv" in lg, f"3 of 20 must not stop, a stale csv must be logged: {lg} {o}")
    st, lg, o = run_guard("done", [50] * 20, marker_first=True); check(not st and "chain ended; guard off" in lg, f"the chain-end marker must end the guard: {lg} {o}")


@test
def t_package_consistency_and_hygiene():
    cmd = open(B2 + "/rnext_chain.cmd", encoding="utf-8", newline="").read()
    check("for %%T in (RAPR_0 RAPR_1 RTL_0 RTL_1 RP01_0 RP01_1)" in cmd and TAGS == ["RAPR_0", "RAPR_1", "RTL_0", "RTL_1", "RP01_0", "RP01_1"], "chain order")
    pull = open(f"{PKG}/pull_rnext.sh").read(); order = re.findall(r"(\w+_\d):M2b_s\d_recft_\w+", pull); check(order == XTAGS, f"pull order {order}")
    for f, needles in (("rnext_chain.cmd", ["CHAIN_RNEXT", "GUARD_STOP.txt", r"D:\tanitad\rnext\rnext_check.py", r"\rnext"]), ("rnext_guard.ps1", ["CHAIN_RNEXT.done", "CHAIN_RNEXT.STOPPED", "GUARD_STOP.txt"]),
                       ("rnext_run.cmd", [r"\rnext\pin", r"\rnext\logs"]), ("rnext_gate_only.cmd", [r"D:\tanitad\rnext\rnext_run.cmd"])):
        t = open(f"{B2}/{f}", encoding="utf-8", newline="").read(); check(all(n in t for n in needles), f"{f} lacks {[n for n in needles if n not in t]}")
    files = [os.path.join(d, f).replace("\\", "/") for d, _, fs in os.walk(PKG) for f in fs if "__pycache__" not in d and "/fixtures" not in d.replace("\\", "/") and not d.replace("\\", "/").endswith("/argv") and "/logs" not in d.replace("\\", "/")]
    mine = [f for f in files if os.path.basename(f) not in ("README.md", "FROZEN_SHA.txt", "MANIFEST.txt") and "PREREG" not in f and "/argv/" not in f and "/rpt_stack/" not in f.replace("\\", "/")]
    words = [r"\bvo" + "id\\b", "super" + "seded", "retr" + "acted", "for" + "merly", "no lon" + "ger", "corr" + "ection", "with" + "drawn"]
    for f in mine:
        raw = open(f, "rb").read(); txt = raw.decode("utf-8", "replace")
        for w in words: check(not re.search(w, txt, re.I), f"{f}: contains a banned word ({w})")
        ext = os.path.splitext(f)[1]
        if ext in (".py", ".sh", ".md"): check(b"\r" not in raw, f"{f}: CR in an LF file")
        if ext in (".cmd", ".ps1"): check(raw.count(b"\r\n") == raw.count(b"\n") and raw.count(b"\n") > 0, f"{f}: not CRLF")
    for f in [x for x in files if "/box2_files/" in x and not x.endswith(".txt")]:
        t = open(f, encoding="utf-8", errors="replace").read(); body = re.sub(r"_recft_(RAPR|RTL|RP01|\{kind\})", "", t)
        check(not re.search(r"recft|CHAIN_RECOVER|tanitad-box2|\bssh\b|\bscp\b", body, re.I), f"{f}: names the previous chain or a remote: {re.findall(r'recft|CHAIN_RECOVER|tanitad-box2|ssh|scp', body, re.I)[:3]}")


# ================================================================================ RAPR-P01 (addendum 2 row E): RAPR's frozen argv with --recover-p 0.1 =============================================================================================
WRONG["RAPRP01"] = WRONG["RAPR"] + [("built lever recover_p = 0.3", _set(["levers", "recover", "recover_p"], 0.3))]       # RAPR's dose under the RAPRP01 tag is refused


@test
def t_raprp01_argv_spec_and_trainer():
    m = trainer()
    for t in XT_R:
        a = open(f"{B2}/argv/{t}.train.txt", "rb").read(); b = open(f"{PKG}/argv/{t}.train.txt", "rb").read()
        check(a == b and b"\r" not in a and a.count(b"\n") == 1, f"{t}: box2_files/argv differs from recover_next/argv, or not one LF line")
        check(" ".join(RC.expected_tokens(t)) == a.decode().strip(), f"{t}: the checker's frozen spec does not reproduce the argv file")
        k, s = RC.INFO[t]; K = RC.KINDS[k]; pa, rec = real_perturb(t)
        check((pa.recover_p, pa.recover_dev_law, pa.recover_target, pa.recover_rejoin_steps, pa.recover_seed) == (0.1, "apriori", "rejoin", 10, s), f"{t}: parsed levers")
        check(rec.problems() == [] and m.bad_recover(pa, rec, None) == [], f"{t}: perturber problems {rec.problems()} / {m.bad_recover(pa, rec, None)}")
        lv = rec.built_levers(); check((lv["recover_p"], lv["recover_dev_law"], lv["recover_target"], lv["recover_rejoin_steps"], lv["recover_seed"], lv["recover_dev_bank"], lv["recover_dev_bank_sha256"]) == (K["p"], K["law"], K["target"], K["rejoin"], s, None, None), f"{t}: built {lv}")
        check(m.build_parser().parse_args(a.decode().split() + ["--selfcheck-only"]).selfcheck_only, f"{t}: --selfcheck-only")
        a2, rec2 = real_perturb(t, a.decode().strip() + " --recover-dev-bank X --recover-dev-bank-sha256 " + RC.BANK_SHA[s])
        check(any("apriori" in x for x in m.bad_recover(a2, rec2, None)), f"{t}: the trainer's own refusal of a bank with the a-priori law did not fire")
    for s in (0, 1):                                        # RAPRP01_s is RAPR_s with exactly two tokens changed: the out dir suffix (token 7) and the dose (token 51)
        r = open(f"{B2}/argv/RAPR_{s}.train.txt").read().split(); p = open(f"{B2}/argv/RAPRP01_{s}.train.txt").read().split()
        d = [(i, x, y) for i, (x, y) in enumerate(zip(r, p)) if x != y]
        check(len(r) == len(p) == 60 and [i for i, _, _ in d] == [7, 51] and d[0][2] == d[0][1] + "P01" and d[1][1:] == ("0.3", "0.1"), f"RAPRP01_{s} vs RAPR_{s}: {d}")


@test
def t_raprp01_pin_refusals_preflight_finish():
    sb = SB("rp01_pin", extra=True)
    for t in XT_R:
        rc, out = run_check(["pin", t], sb); check(rc == 0 and f"pinned {t}" in out, f"{t}: {rc} {out}")
        tr = open(f"{sb.rc()}/pin/{t}.train.txt").read().strip(); ga = open(f"{sb.rc()}/pin/{t}.gate.txt").read().strip(); k, s = RC.INFO[t]
        check(tr == sb.argv(t) and ga.endswith(" --selfcheck-only") and f"gate_rnext\\M2b_s{s}_recft_{k}" in ga.replace("/", "\\"), f"{t}: pin files")
        check(run_check(["pin", t], sb)[0] == 0, f"{t}: second pin refused")
    MUTX = [("RAPRP01_0", lambda l: l.replace("--recover-p 0.1", "--recover-p 0.3"), "--recover-p = '0.3' but the frozen RAPRP01_0 argv has '0.1'"),
            ("RAPRP01_1", lambda l: l + " --recover-dev-bank D:\\tanitad\\runs\\v8f_mini_full\\dagger_bank_dag2\\s1", "unexpected flag --recover-dev-bank"),
            ("RAPRP01_0", lambda l: l + " --recover-dev-bank-sha256 " + RC.BANK_SHA[0], "a bank flag with the a-priori law"),
            ("RAPRP01_0", lambda l: l.replace("--recover-dev-law apriori", "--recover-dev-law empirical"), "--recover-dev-law = 'empirical' but the frozen RAPRP01_0 argv has 'apriori'"),
            ("RAPRP01_1", lambda l: l.replace(" --recover-dev-law apriori", ""), "missing --recover-dev-law"),
            ("RAPRP01_1", lambda l: l.replace("recft_RAPRP01", "recft_RAPR"), "--out ="),
            ("RAPRP01_0", lambda l: l.replace("--steps 3000", "--steps 2000"), "--steps = '2000'"),
            ("RAPRP01_1", lambda l: l.replace("--recover-seed 1", "--recover-seed 0"), "--recover-seed = '0' but the frozen RAPRP01_1 argv has '1'"),
            ("RAPRP01_0", lambda l: l + " --recover-p 0.3", "--recover-p is given 2 times"),
            ("RAPRP01_1", lambda l: l.replace("M2b_s1_dagoacc\\", "M2b_s1_dagoacc_old\\"), "--init-ckpt ="),
            ("RAPRP01_0", lambda l: l + "\n--arm M2b", "exactly one line")]
    for tag, mut, reason in MUTX:
        sb = SB("rp01_mut", extra=True); line = sb.argv(tag); new = mut(line); check(new != line, f"{tag}: the mutation changed nothing"); sb.set_argv(tag, new)
        rc, out = run_check(["pin", tag], sb); check(rc == 2 and "REFUSED" in out and reason in out, f"{tag} [{reason}]: rc {rc}: {out[:400]}")
        check(not os.path.exists(f"{sb.rc()}/pin/{tag}.train.txt"), f"{tag}: a pin was written for a refused argv")
        sib = [t for t in XT_R if t != tag][0]; check(run_check(["pin", sib], sb)[0] == 0, f"control: unmutated sibling {sib} was refused")
        sb.mk_inputs(); rc2, out2 = run_check(["preflight"], sb); check(rc2 == 3 and f"{tag}:" in out2, f"{tag}: preflight did not refuse: {rc2} {out2[:300]}")
    # preflight with the extra argv shipped: all eight audited (banks are stubbed: the real banks are not on the dev box); the six-run package is untouched when they are not shipped
    sb = SB("rp01_pre", extra=True); sb.mk_inputs()
    for t in TAGS:
        k, s_ = RC.INFO[t]
        if RC.KINDS[k]["table"]: write(f"{sb.runsm}/dagger_bank_dag2/s{s_}/bank.json", "{}")
    env = {**os.environ, "RNEXT_HOME": sb.home, "RNEXT_RUNSM": sb.runsm, "RNEXT_REPO": sb.repo, "RNEXT_DISK": sb.disk, "RNEXT_FLOOR_BYTES": "0"}
    code = ("import sys,os; sys.path.insert(0,%r); sys.path.insert(0,%r); import netguard; netguard.install(); import rnext_check as rc; o=rc.sha_file; "
            "rc.sha_file=lambda f: ((rc.BANK_SHA[0] if os.sep.join(['s0','bank.json']) in f else rc.BANK_SHA[1]) if f.endswith('bank.json') else o(f)); sys.exit(rc.main(['preflight']))") % (HERE, B2)
    r = subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True)
    check(r.returncode == 0 and "preflight OK" in r.stdout and "8 argv files" in r.stdout, f"positive preflight with the RAPRP01 argv shipped: {r.returncode} {r.stdout} {r.stderr[-300:]}")
    os.remove(sb.runsm + "/M2b_s1_dagoacc/ckpt_last.pt"); rc, out = run_check(["preflight"], sb); check(rc == 3 and "RAPRP01_1: init ckpt" in out, f"missing init not named for RAPRP01_1: {out[:300]}")
    # finish records the eight when the extra argv is shipped, and exactly the six when it is not
    sb = pinned_sb("rp01_fin", XT_R, extra=True); stage_run(sb, "RAPRP01_0"); run_check(["post", "RAPRP01_0", "0"], sb); run_check(["fail", "RAPRP01_1", "pin", "refused"], sb)
    rc, out = run_check(["finish"], sb); st = json.load(open(f"{sb.rc()}/CHAIN_RNEXT.done"))["runs"]
    check(rc == 0 and st == {"RAPR_0": "none", "RAPR_1": "none", "RTL_0": "none", "RTL_1": "none", "RP01_0": "none", "RP01_1": "none", "RAPRP01_0": "done", "RAPRP01_1": "FAILED"}, f"{rc} {st}")
    sb = SB("rp01_fin6"); rc, out = run_check(["finish"], sb); st = json.load(open(f"{sb.rc()}/CHAIN_RNEXT.done"))["runs"]; check(rc == 0 and list(st) == TAGS, f"six-run finish changed: {st}")


@test
def t_raprp01_gate_and_post():
    for t in XT_R:
        sb = pinned_sb("rp01_gate", [t], extra=True); stage_gate(sb, t, stub_sc(t))
        rc, out = run_check(["gate", t, "0"], sb); check(rc == 0 and f"ZZRNEXT-GATE-PASS {t}" in out, f"{t}: {rc} {out}")
        tok = json.load(open(f"{sb.rc()}/gate_token_{t}.json")); b = tok["binding"]
        check(tok["verdict"] == "PASS" and b["commit"] == RC.TIP and b["argv_sha256"] == RC.tokens_sha(sb.argv(t).split(" ")), f"{t}: token")
        check(run_check(["gate-cached", t], sb)[0] == 0, f"{t}: gate-cached no after PASS")
        for reason, mut in WRONG["RAPRP01"] + GENERIC:
            sb = pinned_sb("rp01_gatebad", [t], extra=True); sc = stub_sc(t); mut(sc); stage_gate(sb, t, sc)
            rc, out = run_check(["gate", t, "0"], sb); mk = f"{sb.rc()}/RUN_{t}.FAILED"
            check(rc == 1 and os.path.exists(mk) and reason in open(mk).read() and not os.path.exists(f"{sb.rc()}/gate_token_{t}.json"), f"{t} [{reason}]: rc {rc} {out[:300]}")
    for t, other in (("RAPRP01_0", "RAPR_0"), ("RAPRP01_0", "RP01_0"), ("RAPRP01_1", "RTL_1"), ("RAPR_0", "RAPRP01_0")):          # a stub of another kind is refused (the dose, the law and the target each differ)
        sb = pinned_sb("rp01_gatecross", [t], extra=True); sc = stub_sc(other); sc["info"]["init_ckpt"]["sha256"] = RC.INIT_SHA[RC.INFO[t][1]]; stage_gate(sb, t, sc)
        rc, out = run_check(["gate", t, "0"], sb); check(rc == 1, f"{other}'s selfcheck passed the {t} gate")
    for t in XT_R:
        sb = pinned_sb("rp01_post", [t], extra=True); stage_run(sb, t); rc, out = run_check(["post", t, "0"], sb)
        mk = f"{sb.rc()}/RUN_{t}.done"; check(rc == 0 and os.path.exists(mk), f"{t}: {rc} {out}")
        ev = json.load(open(mk))["evidence"]; check(ev["ckpt_sha256"] == hashlib.sha256(b"ckpt-" + t.encode()).hexdigest() and ev["argv_match_pin"] is True, f"{t}: evidence {ev}")
        sb = pinned_sb("rp01_postbad", [t], extra=True); stage_run(sb, t, summary={"recover_n": 0}); run_check(["post", t, "0"], sb)
        check(os.path.exists(f"{sb.rc()}/RUN_{t}.FAILED") and not os.path.exists(f"{sb.rc()}/RUN_{t}.done"), f"{t}: KR1 failure not marked")


@test
def t_raprp01_chain_launcher_and_package():
    ra = open(B2 + "/rnext_chain_rapr.cmd", "rb").read(); rb = open(B2 + "/rnext_chain_raprp01.cmd", "rb").read()
    check(rb.count(b"\r\n") == rb.count(b"\n") == ra.count(b"\n") and rb.endswith(b"\r\n"), "raprp01 chain is not CRLF like the RAPR one")
    la, lb = ra.decode().split("\r\n"), rb.decode().split("\r\n"); dl = [i + 1 for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    check(len(la) == len(lb) and dl == [2, 3, 6, 23], f"the RAPRP01 chain differs from the RAPR chain at lines {dl} (only the two comment lines, the launch line and the loop may)")
    check(lb[22] == "for %%T in (RAPRP01_0 RAPRP01_1) do (" and "rnext_chain_raprp01.cmd" in lb[5], "loop / launch line")
    for i in (1, 2, 5, 22): la[i] = lb[i] = ""
    check(la == lb, "other lines differ")
    sh = open(f"{PKG}/launch_raprp01.sh", "rb").read(); check(b"\r" not in sh, "CR in launch_raprp01.sh"); st = sh.decode()
    rc, o, e = bash(["-n", f"{PKG}/launch_raprp01.sh"], {}); check(rc == 0, f"bash -n launch_raprp01.sh: {e}")
    ship = re.search(r"for f in ([^;]+); do \$SCP", st).group(1).split(); tags = re.search(r"for t in ([^;]+); do \$SCP", st).group(1).split()
    on_disk = sorted(f for f in os.listdir(B2) if f.startswith("rnext_") and os.path.isfile(f"{B2}/{f}")); check(sorted(ship) == [f for f in on_disk if f != "rnext_chain_rpt.cmd"], f"shipped file list {sorted(ship)} != box2_files/rnext_* {on_disk} minus the RPT chain (a later arm's launcher ships that one)")
    check(sorted(tags) == sorted(f[:-len(".train.txt")] for f in os.listdir(B2 + "/argv") if not f.startswith("RPT_")) and tags == TAGS + XT_R, f"shipped argv tags {tags}")
    for needle in ("RAPR-P01", "RAPRP01_PI_GO:?", "rnext_chain_raprp01.cmd", "rnext_chain_raprp01.log", "ZZRAPRP01-LAUNCHED", "BOX2_BUSY.raprp01_launch_failed_", "BOX2_BUSY.txt", "CHAIN_RNEXT.rapr_"):
        check(needle in st, f"launcher lacks {needle}")
    check("RAPR2" not in st and "rapr2" not in st and "ssh" in st, "launcher keeps RAPR2 names")
    mk = open(f"{PKG}/mk_launch_raprp01.py", encoding="utf-8").read(); check("launch_rapr2.sh" in mk and "assert s.count(a) == 1" in mk, "generator shape")


# ================================================================================ RPT (addendum 4): R's frozen argv with --recover-dev-shape pertick ====================================================================================
import importlib.machinery  # noqa: E402
OLD = importlib.util.module_from_spec(importlib.util.spec_from_loader("rnext_check_old", importlib.machinery.SourceFileLoader("rnext_check_old", PKG + "/_pre_rpt/rnext_check.py.pre_rpt"))); OLD.__spec__.loader.exec_module(OLD)       # the checker as it stood before RPT
WRONG["RPT"] = [w for w in WRONG["RP01"] if not w[0].startswith("built lever recover_p")] + [w for w in WRONG["RTL"] if w[0] in ("built bank sha256", "levers.recover_table is empty", "K-B3 window-list")] + [
    ("built lever recover_p = 0.1", _set(["levers", "recover", "recover_p"], 0.1)),
    ("built lever recover_dev_shape = 'constant'", _set(["levers", "recover", "recover_dev_shape"], "constant")),
    ("built lever recover_dev_shape = None", _set(["levers", "recover", "recover_dev_shape"], None)),
    ("built lever recover_dev_shape", _pop(["levers", "recover", "recover_dev_shape"])),
    ("selfcheck info.recover.dev_shape = 'constant'", _set(["info", "recover", "dev_shape"], "constant")),
    ("selfcheck info.recover.dev_shape = None", _pop(["info", "recover", "dev_shape"])),
    ("levers.recover_table.per_tick is absent", _pop(["levers", "recover_table", "per_tick"])),
    ("levers.recover_table.per_tick is absent", _set(["levers", "recover_table", "per_tick", "ticks"], 0)),
    ("levers.recover_table.per_tick is absent", lambda sc: sc["levers"]["recover_table"]["per_tick"].__setitem__("mean_abs_dom_t", [0.1] * 9)),
    ("carries no per-tick drop count", _pop(["levers", "recover_table", "counts", "dropped_per_tick_row_unreal"])),
    ("carries no per-tick drop count", _set(["levers", "recover_table", "counts", "kept"], 0)),
    ("recover_probe.pertick_v_max_abs_err = None", _set(["recover_probe", "pertick_v_max_abs_err"], None)),
    ("recover_probe.pertick_v_max_abs_err = 0.001", _set(["recover_probe", "pertick_v_max_abs_err"], 1e-3)),
    ("recover_probe.pertick_v_max_abs_err = True", _set(["recover_probe", "pertick_v_max_abs_err"], True))]


@test
def t_rpt_argv_spec_and_trainer():
    m = trainer()
    for s in (0, 1):
        a = open(f"{B2}/argv/RPT_{s}.train.txt", "rb").read(); b = open(f"{PKG}/argv/RPT_{s}.train.txt", "rb").read(); t = f"RPT_{s}"
        check(a == b and b"\r" not in a and a.count(b"\n") == 1, f"{t}: box2_files/argv differs from recover_next/argv, or not one LF line")
        check(" ".join(RC.expected_tokens(t)) == a.decode().strip(), f"{t}: the checker's frozen spec does not reproduce the argv file")
        x = a.decode().split(); r = open(f"E:/Projects/TanitAD-artifacts/dagland/recover_ft_launch/box2_files/argv/R_{s}.train.txt", "rb").read().decode().replace("\r", "").split()
        d = [(i, p_, q) for i, (p_, q) in enumerate(zip(r, x)) if p_ != q]
        check(len(x) == len(r) + 2 and x[:len(r)] != r and [i for i, _, _ in d] == [7] and d[0][1].endswith(f"M2b_s{s}_recft_R") and d[0][2].endswith(f"M2b_s{s}_recft_RPT") and x[len(r):] == ["--recover-dev-shape", "pertick"] and d[0][2] == d[0][1] + "PT", f"{t} vs R_{s}: {d} {x[len(r):]}")
        rp = open(f"{PKG}/argv/RP01_{s}.train.txt").read().split(); y = [("0.3" if (i > 0 and rp[i - 1] == "--recover-p") else q.replace("recft_RP01", "recft_RPT")) for i, q in enumerate(rp)]
        check(y + ["--recover-dev-shape", "pertick"] == x, f"{t} is not RP01_{s} with p 0.3, the run name RPT and the two tokens appended")
        check([z.replace("recft_RP01T", "recft_RPT") for z in y] + ["--recover-dev-shape", "pertick"] == x, f"{t} is not RP01_{s} with p 0.3, the run name RPT and the two tokens appended")
        k, s_ = RC.INFO[t]; K = RC.KINDS[k]; pa, rec = real_perturb(t)
        check((pa.recover_p, pa.recover_dev_law, pa.recover_target, pa.recover_rejoin_steps, pa.recover_seed, pa.recover_dev_shape) == (0.3, "empirical", "rejoin", 10, s, "pertick"), f"{t}: parsed levers")
        check(rec.problems() == [], f"{t}: perturber problems {rec.problems()}")
        lv = rec.built_levers(); check((lv["recover_p"], lv["recover_dev_law"], lv["recover_target"], lv["recover_rejoin_steps"], lv["recover_seed"], lv["recover_dev_shape"], lv["recover_dev_bank_sha256"]) == (K["p"], K["law"], K["target"], K["rejoin"], s, K["shape"], RC.BANK_SHA[s]), f"{t}: built {lv}")
        check(m.build_parser().parse_args(a.decode().split() + ["--selfcheck-only"]).selfcheck_only, f"{t}: --selfcheck-only")
        shape_bad = lambda br: any("recover_dev_shape" in z for z in br)      # noqa: E731
        real_perturb.force_shape = "constant"; pa2, rec2 = real_perturb(t); real_perturb.force_shape = None           # a perturber BUILT with the constant shape under the pertick argv: G-DVB's read-back must name it
        check(shape_bad(m.bad_recover(pa2, rec2, None)), f"{t}: the trainer's own declared-vs-built did not flag a constant perturber under a pertick argv: {m.bad_recover(pa2, rec2, None)}")
        real_perturb.flat_table = True; pa3, rec3 = real_perturb(t); real_perturb.flat_table = False                  # pertick with a table that has no per-tick columns
        check(any("per-tick columns" in z for z in rec3.problems()), f"{t}: a pertick perturber over a constant table was not refused: {rec3.problems()}")
        pa4, rec4 = real_perturb(t, a.decode().strip() + " --recover-dev-law apriori")      # the same argv given the a-priori law
        check(any("apriori" in z and "pertick" in z for z in m.bad_recover(pa4, rec4, None)) or any("pertick" in z for z in rec4.problems()), f"{t}: pertick with the a-priori law was not refused")
        check(m.build_parser().parse_args(r).recover_dev_shape == "constant", f"R_{s}: the trainer's default shape is not constant")      # R's argv does not name the flag: R runs the constant shape


def pin_all(sb, tags):
    for t in tags:
        rc, out = run_check(["pin", t], sb); check(rc == 0 and f"pinned {t}" in out, f"{t}: {rc} {out}")
        tr = open(f"{sb.rc()}/pin/{t}.train.txt").read().strip(); ga = open(f"{sb.rc()}/pin/{t}.gate.txt").read().strip(); k, s = RC.INFO[t]
        check(tr == sb.argv(t) and ga.endswith(" --selfcheck-only") and f"gate_rnext\\M2b_s{s}_recft_{k}" in ga.replace("/", "\\"), f"{t}: pin files")
        check(run_check(["pin", t], sb)[0] == 0, f"{t}: second pin refused")


@test
def t_rpt_pin_refusals_preflight_finish():
    sb = SB("rpt_pin", extra="rpt"); pin_all(sb, XT_T)
    MUTT = [("RPT_0", lambda l: l.replace("--recover-p 0.3", "--recover-p 0.1"), "--recover-p = '0.1' but the frozen RPT_0 argv has '0.3'"),
            ("RPT_1", lambda l: l.replace(" --recover-dev-shape pertick", ""), "missing --recover-dev-shape"),
            ("RPT_0", lambda l: l.replace("--recover-dev-shape pertick", "--recover-dev-shape constant"), "--recover-dev-shape = 'constant' but the frozen RPT_0 argv has 'pertick'"),
            ("RPT_0", lambda l: l.replace(" --recover-dev-shape pertick", "").replace("--recover-seed 0", "--recover-dev-shape pertick --recover-seed 0"), "the flags and values are the frozen ones but the token order differs"),
            ("RPT_1", lambda l: l + " --recover-dev-shape pertick", "--recover-dev-shape is given 2 times"),
            ("RPT_0", lambda l: l + " --recover-dev-law apriori", "unexpected flag --recover-dev-law"),
            ("RPT_1", lambda l: l.replace("--recover-dev-bank-sha256 " + RC.BANK_SHA[1], "--recover-dev-bank-sha256 " + RC.BANK_SHA[0]), "--recover-dev-bank-sha256 ="),
            ("RPT_1", lambda l: l.replace("recft_RPT", "recft_R"), "--out ="),
            ("RPT_0", lambda l: l.replace("--steps 3000", "--steps 2000"), "--steps = '2000'"),
            ("RPT_1", lambda l: l.replace("--recover-seed 1", "--recover-seed 0"), "--recover-seed = '0' but the frozen RPT_1 argv has '1'"),
            ("RPT_0", lambda l: l.replace("--recover-target rejoin", "--recover-target logged"), "--recover-target = 'logged' but the frozen RPT_0 argv has 'rejoin'"),
            ("RPT_1", lambda l: l.replace("M2b_s1_dagoacc\\", "M2b_s1_dagoacc_old\\"), "--init-ckpt ="),
            ("RPT_0", lambda l: l + "\n--arm M2b", "exactly one line")]
    for tag, mut, reason in MUTT:
        sb = SB("rpt_mut", extra="rpt"); line = sb.argv(tag); new = mut(line); check(new != line, f"{tag}: the mutation changed nothing"); sb.set_argv(tag, new)
        rc, out = run_check(["pin", tag], sb); check(rc == 2 and "REFUSED" in out and reason in out, f"{tag} [{reason}]: rc {rc}: {out[:400]}")
        check(not os.path.exists(f"{sb.rc()}/pin/{tag}.train.txt"), f"{tag}: a pin was written for a refused argv")
        sib = [t for t in XT_T if t != tag][0]; check(run_check(["pin", sib], sb)[0] == 0, f"control: unmutated sibling {sib} was refused")
        sb.mk_inputs(); rc2, out2 = run_check(["preflight"], sb); check(rc2 == 3 and f"{tag}:" in out2, f"{tag}: preflight did not refuse: {rc2} {out2[:300]}")
    for tag in TAGS + XT_R:                                  # no other arm names the shape flag: each is refused with the named reason
        sb = SB("rpt_other", extra=True); sb.set_argv(tag, sb.argv(tag) + " --recover-dev-shape pertick")
        rc, out = run_check(["pin", tag], sb); check(rc == 2 and "unexpected flag --recover-dev-shape" in out and "belongs to the RPT arm only" in out, f"{tag} + the shape flag: rc {rc} {out[:300]}")
    sb = SB("rpt_asR", extra="rpt"); sb.set_argv("RPT_0", open("E:/Projects/TanitAD-artifacts/dagland/recover_ft_launch/box2_files/argv/R_0.train.txt").read().replace("\r", "").strip().replace("recft_R", "recft_RPT"))
    rc, out = run_check(["pin", "RPT_0"], sb); check(rc == 2 and "missing --recover-dev-shape" in out, f"R's argv (no shape flag) under the RPT tag was not refused: {rc} {out[:300]}")
    # preflight with all ten shipped (the banks' hash is stubbed: the real banks are not on the dev box); the shipped bytes are the pinned ones, and each of the three files, edited or at its box-2 state before RPT, is refused by name
    def preflight_with(sb, stub=True):
        env = {**os.environ, "RNEXT_HOME": sb.home, "RNEXT_RUNSM": sb.runsm, "RNEXT_REPO": sb.repo, "RNEXT_DISK": sb.disk, "RNEXT_FLOOR_BYTES": "0"}
        code = ("import sys,os; sys.path.insert(0,%r); sys.path.insert(0,%r); import netguard; netguard.install(); import rnext_check as rc; o=rc.sha_file; "
                "rc.sha_file=lambda f: ((rc.BANK_SHA[0] if os.sep.join(['s0','bank.json']) in f else rc.BANK_SHA[1]) if f.endswith('bank.json') else o(f)); sys.exit(rc.main(['preflight']))") % (HERE, B2)
        r = subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True); return r.returncode, r.stdout + r.stderr[-300:]
    sb = SB("rpt_pre", extra="rpt"); sb.mk_inputs()
    for t in TAGS:
        k, s_ = RC.INFO[t]
        if RC.KINDS[k]["table"]: write(f"{sb.runsm}/dagger_bank_dag2/s{s_}/bank.json", "{}")
    rc, out = preflight_with(sb); check(rc == 0 and "preflight OK" in out and "10 argv files" in out, f"positive preflight with the RPT argv shipped: {rc} {out}")
    for rel in RC.SHIPPED:                                   # an edited file of each of the three
        sb3 = SB("rpt_pre_edit", extra="rpt", repo=WORK + "/repo_edit3"); shutil.rmtree(sb3.repo, ignore_errors=True); sb3.mk_inputs()
        for r_ in RC.SHIPPED: os.makedirs(os.path.dirname(f"{sb3.repo}/{r_}"), exist_ok=True); shutil.copy(f"{TIPDIR}/{r_}", f"{sb3.repo}/{r_}")
        write(f"{sb3.repo}/{rel}", "# edited\n" + open(f"{TIPDIR}/{rel}", encoding="utf-8", newline="").read())
        rc, out = run_check(["preflight"], sb3); check(rc == 3 and f"{os.path.basename(rel)}: sha256" in out, f"edited {rel} not refused: {rc} {out[:300]}")
    old_pins = dict(re.findall(r'"(stack/[^"]+\.py)": "([0-9a-f]{64})"', open(PKG + "/_pre_rpt/rnext_check.py.pre_rpt", encoding="utf-8").read().split("INIT_SHA")[0]))
    check(set(old_pins) == {"stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_recover.py"} and all(old_pins[r_] != RC.SHIPPED[r_] for r_ in old_pins), f"old pins {old_pins}")
    sb4 = SB("rpt_pre_old", extra="rpt", repo=WORK + "/repo_old"); shutil.rmtree(sb4.repo, ignore_errors=True); sb4.mk_inputs()
    for r_ in RC.SHIPPED: os.makedirs(os.path.dirname(f"{sb4.repo}/{r_}"), exist_ok=True); shutil.copy(f"{TIPDIR}/{r_}", f"{sb4.repo}/{r_}")
    with tarfile.open(TAR) as tf:                            # the box-2 trainer and library as they stood before RPT (the tar's bytes, equal to the OLD pins)
        for r_ in old_pins:
            data = tf.extractfile(r_).read(); check(hashlib.sha256(data).hexdigest() == old_pins[r_], f"tar member {r_} is not the old pin"); write(f"{sb4.repo}/{r_}", data, "wb")
    rc, out = run_check(["preflight"], sb4); check(rc == 3 and "v8f_mini_train.py: sha256" in out and "v8f_recover.py: sha256" in out and "RECOVER_DEFAULTS" in out, f"the pre-RPT trainer and library were not refused: {rc} {out[:400]}")
    # finish records all ten when the RPT argv is shipped, and the eight / six when it is not
    sb = pinned_sb("rpt_fin", XT_T, extra="rpt"); stage_run(sb, "RPT_0"); run_check(["post", "RPT_0", "0"], sb); run_check(["fail", "RPT_1", "pin", "refused"], sb)
    rc, out = run_check(["finish"], sb); st = json.load(open(f"{sb.rc()}/CHAIN_RNEXT.done"))["runs"]
    check(rc == 0 and list(st) == XTAGS and st["RPT_0"] == "done" and st["RPT_1"] == "FAILED" and all(v == "none" for k_, v in st.items() if not k_.startswith("RPT_")), f"{rc} {st}")
    sb = SB("rpt_fin8", extra=True); rc, out = run_check(["finish"], sb); st = json.load(open(f"{sb.rc()}/CHAIN_RNEXT.done"))["runs"]; check(rc == 0 and list(st) == TAGS + XT_R, f"eight-run finish changed: {st}")


@test
def t_rpt_gate_and_post():
    for t in XT_T:
        sb = pinned_sb("rpt_gate", [t], extra="rpt"); stage_gate(sb, t, stub_sc(t))
        rc, out = run_check(["gate", t, "0"], sb); check(rc == 0 and f"ZZRNEXT-GATE-PASS {t}" in out, f"{t}: {rc} {out}")
        tok = json.load(open(f"{sb.rc()}/gate_token_{t}.json")); b = tok["binding"]
        check(tok["verdict"] == "PASS" and b["commit"] == RC.TIP and b["argv_sha256"] == RC.tokens_sha(sb.argv(t).split(" ")), f"{t}: token")
        check(b["shipped_trainer_sha256"] == RC.SHIPPED["stack/scripts/v8f_mini_train.py"] and b["shipped_recover_lib_sha256"] == RC.SHIPPED["stack/tanitad/data/v8f_recover.py"], f"{t}: token binds the old shipped bytes")
        check(run_check(["gate-cached", t], sb)[0] == 0, f"{t}: gate-cached no after PASS")
        for reason, mut in WRONG["RPT"] + GENERIC:
            sb = pinned_sb("rpt_gatebad", [t], extra="rpt"); sc = stub_sc(t); mut(sc); stage_gate(sb, t, sc)
            rc, out = run_check(["gate", t, "0"], sb); mk = f"{sb.rc()}/RUN_{t}.FAILED"
            check(rc == 1 and os.path.exists(mk) and reason in open(mk).read() and not os.path.exists(f"{sb.rc()}/gate_token_{t}.json"), f"{t} [{reason}]: rc {rc} {out[:300]}")
    # the discriminating stub: R's levers with the CONSTANT shape (a trainer that ignored the flag, or R's own gate) must not pass the RPT gate
    sb = pinned_sb("rpt_asR", ["RPT_0"], extra="rpt"); sc = stub_sc("RPT_0"); real_perturb.force_shape = "constant"; sc["levers"]["recover"] = real_perturb("RPT_0")[1].built_levers(); real_perturb.force_shape = None
    sc["info"]["recover"].pop("dev_shape"); sc["levers"]["recover_table"].pop("per_tick"); sc["info"]["recover"]["table"].pop("per_tick"); sc["recover_probe"].pop("pertick_v_max_abs_err"); stage_gate(sb, "RPT_0", sc)
    rc, out = run_check(["gate", "RPT_0", "0"], sb); mk = f"{sb.rc()}/RUN_RPT_0.FAILED"
    check(rc == 1 and os.path.exists(mk) and "recover_dev_shape" in open(mk).read(), f"R-shaped (constant) selfcheck passed the RPT gate: {rc} {out[:300]}")
    for t, other in (("RPT_0", "RP01_0"), ("RP01_0", "RPT_0"), ("RPT_1", "RTL_1"), ("RPT_0", "RAPR_0"), ("RAPR_0", "RPT_0"), ("RPT_1", "RAPRP01_1"), ("RAPRP01_1", "RPT_1")):      # a stub of another kind is refused both ways
        sb = pinned_sb("rpt_gatecross", [t], extra="rpt"); sc = stub_sc(other); sc["info"]["init_ckpt"]["sha256"] = RC.INIT_SHA[RC.INFO[t][1]]; stage_gate(sb, t, sc)
        rc, out = run_check(["gate", t, "0"], sb); check(rc == 1, f"{other}'s selfcheck passed the {t} gate")
    for t in XT_T:
        sb = pinned_sb("rpt_post", [t], extra="rpt"); stage_run(sb, t); rc, out = run_check(["post", t, "0"], sb)
        mk = f"{sb.rc()}/RUN_{t}.done"; check(rc == 0 and os.path.exists(mk), f"{t}: {rc} {out}")
        ev = json.load(open(mk))["evidence"]; check(ev["ckpt_sha256"] == hashlib.sha256(b"ckpt-" + t.encode()).hexdigest() and ev["argv_match_pin"] is True, f"{t}: evidence {ev}")
        sb = pinned_sb("rpt_postbad", [t], extra="rpt"); stage_run(sb, t, summary={"recover_n": 0}); run_check(["post", t, "0"], sb)
        check(os.path.exists(f"{sb.rc()}/RUN_{t}.FAILED") and not os.path.exists(f"{sb.rc()}/RUN_{t}.done"), f"{t}: KR1 failure not marked")


@test
def t_rpt_leaves_the_existing_tags_unchanged():
    """The pre-RPT checker (_pre_rpt/rnext_check.py.pre_rpt) against the new one, on the eight existing tags: the frozen argv, the registered levers and the audit's verdict on every mutated argv are identical; what differs is only what RPT changes."""
    old_tags = TAGS + XT_R
    check([t for t, _, _ in OLD.RUNS + OLD.EXTRA] == old_tags and set(OLD.KINDS) == set(RC.KINDS) - {"RPT"}, "tag / kind sets")
    for t in old_tags:
        check(OLD.expected_tokens(t) == RC.expected_tokens(t) and OLD.INFO[t] == RC.INFO[t], f"{t}: frozen argv or info changed")
        check(OLD.KINDS[RC.INFO[t][0]] == RC.KINDS[RC.INFO[t][0]], f"{t}: registered levers changed")
        toks = open(f"{B2}/argv/{t}.train.txt").read().strip().split(" ")
        check(OLD.audit_argv(t, toks) == [] == RC.audit_argv(t, toks), f"{t}: audit of the shipped argv")
    muts = [(t, mut(open(f"{B2}/argv/{t}.train.txt").read().strip())) for t, mut, _ in MUT if t in old_tags]
    muts += [(t, f(open(f"{B2}/argv/{t}.train.txt").read().strip())) for t, f in (("RAPRP01_0", lambda l: l.replace("--recover-p 0.1", "--recover-p 0.3")), ("RAPRP01_1", lambda l: l + " --recover-dev-law apriori"), ("RP01_1", lambda l: l.replace("--recover-seed 1", "--recover-seed 0")))]
    n_changed = 0
    for t, new in muts:
        a, b = OLD.audit_argv(t, new.split(" ")), RC.audit_argv(t, new.split(" ")); n_changed += (a != b); check(a == b and a != [], f"{t}: the audit's verdict on a mutated argv changed: {a} / {b}")
    check(len(muts) >= 15 and n_changed == 0, f"{len(muts)} mutated argv, {n_changed} verdicts changed")
    check(all("shape" not in RC.KINDS[k] for k in OLD.KINDS), "an old kind gained a shape key")
    check(set(OLD.SHIPPED) < set(RC.SHIPPED) and set(RC.SHIPPED) - set(OLD.SHIPPED) == {"stack/tanitad/data/v8f_mini_data.py"}, "SHIPPED: only the data module is added")
    check({k: v for k, v in RC.DEFAULTS.items() if k != "dev_shape"} == OLD.DEFAULTS and RC.DEFAULTS["dev_shape"] == "constant" and (OLD.DEFAULTS_LINE, RC.DEFAULTS_LINE) == (23, 28), "RECOVER_DEFAULTS: only dev_shape is added")
    check(all(OLD.SHIPPED[k] != RC.SHIPPED[k] for k in OLD.SHIPPED), "the two re-pinned files carry their old pin")


@test
def t_rpt_chain_launcher_and_package():
    ra = open(B2 + "/rnext_chain_raprp01.cmd", "rb").read(); rb = open(B2 + "/rnext_chain_rpt.cmd", "rb").read()
    check(rb.count(b"\r\n") == rb.count(b"\n") == ra.count(b"\n") and rb.endswith(b"\r\n"), "rpt chain is not CRLF like the RAPR-P01 one")
    la, lb = ra.decode().split("\r\n"), rb.decode().split("\r\n"); dl = [i + 1 for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    check(len(la) == len(lb) and dl == [2, 3, 6, 23], f"the RPT chain differs from the RAPR-P01 chain at lines {dl} (only the two comment lines, the launch line and the loop may)")
    check(lb[22] == "for %%T in (RPT_0 RPT_1) do (" and "rnext_chain_rpt.cmd" in lb[5] and "rnext_chain_rpt.log" in lb[5], "loop / launch line")
    for i in (1, 2, 5, 22): la[i] = lb[i] = ""
    check(la == lb, "other lines differ")
    sh = open(f"{PKG}/launch_rpt.sh", "rb").read(); check(b"\r" not in sh, "CR in launch_rpt.sh"); st = sh.decode()
    rc, o, e = bash(["-n", f"{PKG}/launch_rpt.sh"], {}); check(rc == 0, f"bash -n launch_rpt.sh: {e}")
    ship = re.search(r"for f in ([^;]+); do \$SCP", st).group(1).split(); tags = re.search(r"for t in ([^;]+); do \$SCP", st).group(1).split()
    on_disk = sorted(f for f in os.listdir(B2) if f.startswith("rnext_") and os.path.isfile(f"{B2}/{f}")); check(sorted(ship) == on_disk, f"shipped file list {sorted(ship)} != box2_files/rnext_* {on_disk}")
    check(sorted(tags) == sorted(f[:-len(".train.txt")] for f in os.listdir(B2 + "/argv")) and tags == XTAGS, f"shipped argv tags {tags}")
    for needle in ("RPT two-run chain", "RPT_PI_GO:?", "rnext_chain_rpt.cmd", "rnext_chain_rpt.log", "ZZRPT-LAUNCHED", "ZZRPT-ABORT", "BOX2_BUSY.rpt_launch_failed_", "BOX2_BUSY.txt", "CHAIN_RNEXT.raprp01_", "Move-Item", "pre_rpt_", "rpt_failed_", "Get-FileHash"):
        check(needle in st, f"launcher lacks {needle}")
    check("RAPR2" not in st and "rapr2" not in st and "RAPR-P01" not in st and "ZZRAPR" not in st and "ssh" in st, "launcher keeps an older arm's names")
    i_lock, i_pre, i_ship, i_beg, i_end, i_tele = (st.index(x) for x in ('echo "[$(ts)] lock taken"', 'pre=$($SSH', "diff $W/dev.txt", "# ---- STACK-SHIP-BEGIN", "# ---- STACK-SHIP-END", "# 4. telemetry first"))
    check(i_lock < i_pre < i_ship < i_beg < i_end < i_tele, "the stack ship is not after the lock, the preconditions and the rnext ship, and before the telemetry")
    blk = st[i_beg:i_end]; rows = re.findall(r'^  "([^|"]+)\|([^|"]+)\|([0-9a-f]{64})"$', blk, re.M)
    check({r_: h for r_, _, h in rows} == RC.SHIPPED and len(rows) == 3, f"the launcher's pinned rows {rows} != the checker's SHIPPED")
    for r_, src, h in rows:
        src_l = src.replace("$SRC", B2).replace("/e/", "E:/", 1); check(os.path.exists(src_l) and hashlib.sha256(open(src_l, "rb").read()).hexdigest() == h, f"{r_}: source {src_l} missing or not the pinned bytes")
    mk = open(f"{PKG}/mk_launch_rpt.py", encoding="utf-8").read(); check("launch_raprp01.sh" in mk and "assert s.count(a) == n" in mk and "STACK-SHIP-BEGIN" in mk, "generator shape")
    r = subprocess.run([sys.executable, "-I", PKG + "/mk_rpt_pins.py", "--check"], capture_output=True, text=True); check(r.returncode == 0 and r.stdout.count("same") == 3, f"mk_rpt_pins.py --check: {r.stdout} {r.stderr}")
    # the launcher refuses without the PI go (before the lock, before any remote call; the shell tests' BASH_ENV blocks every network tool anyway)
    saved = os.environ.pop("RPT_PI_GO", None)
    try: rc, o, e = bash([f"{PKG}/launch_rpt.sh"], {"RNEXT_TRIPWIRE": WORK + "/tripwire_go.txt"}, timeout=25)
    finally:
        if saved is not None: os.environ["RPT_PI_GO"] = saved
    check(rc != 0 and "RPT_PI_GO" in e and not os.path.exists(WORK + "/tripwire_go.txt") and "lock taken" not in o, f"the launcher ran without RPT_PI_GO: rc {rc} out {o[:200]} err {e[:300]}")


FAKE_SHIP = r'''import hashlib, os, re, shutil, sys
ROOT = os.environ["FAKE_BOX_ROOT"]
def loc(p): return p.replace("D:/tanitad/repo", ROOT + "/repo", 1)
def main(a):
    if a[0] == "scp":
        src, dst = a[-2], a[-1]; dst = loc(dst.split(":", 1)[1])
        if os.environ.get("FAKE_SCP_FAIL") and os.environ["FAKE_SCP_FAIL"] in dst: return 1
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(src, dst)
        if os.environ.get("FAKE_SCP_CORRUPT") and os.environ["FAKE_SCP_CORRUPT"] in dst: open(dst, "ab").write(b"#")
        return 0
    cmd = a[-1]
    m = re.fullmatch(r'powershell -NoProfile -Command "if \(Test-Path (\S+)\) \{ Move-Item (\S+) (\S+) \}"', cmd)
    if m:
        if os.environ.get("FAKE_MOVE_FAIL") and os.environ["FAKE_MOVE_FAIL"] in m.group(1): return 1
        if os.path.exists(loc(m.group(1))): os.rename(loc(m.group(2)), loc(m.group(3)))
        return 0
    m = re.fullmatch(r'powershell -NoProfile -Command "if \(Test-Path (\S+)\) \{ Move-Item (\S+) (\S+) \}; if \(Test-Path (\S+)\) \{ Move-Item (\S+) (\S+) \}"', cmd)
    if m:
        if os.path.exists(loc(m.group(1))): os.rename(loc(m.group(2)), loc(m.group(3)))
        if os.path.exists(loc(m.group(4))): os.rename(loc(m.group(5)), loc(m.group(6)))
        return 0
    m = re.fullmatch(r'powershell -NoProfile -Command "\(Get-FileHash (\S+) -Algorithm SHA256\)\.Hash\.ToLower\(\)"', cmd)
    if m:
        print(hashlib.sha256(open(loc(m.group(1)), "rb").read()).hexdigest()); return 0
    print("UNKNOWN REMOTE COMMAND: " + cmd, file=sys.stderr); return 97
sys.exit(main(sys.argv[1:]))
'''


@test
def t_rpt_launcher_stack_ship_on_a_fake_box():
    """The STACK-SHIP block of launch_rpt.sh, run on its own against a local fake box (a script that understands exactly the three remote commands the block uses): ship + re-hash, the aside copies, and every failure path."""
    st = open(f"{PKG}/launch_rpt.sh", encoding="utf-8").read(); blk = st[st.index("# ---- STACK-SHIP-BEGIN"):st.index("# ---- STACK-SHIP-END")]
    fz = WORK + "/fakeship"; shutil.rmtree(fz, ignore_errors=True); write(fz + "/fake_ship.py", FAKE_SHIP)
    for rel, src in MP.SOURCES.items(): os.makedirs(os.path.dirname(f"{fz}/src/{rel}"), exist_ok=True); shutil.copy(src, f"{fz}/src/{rel}")
    os.makedirs(fz + "/box2_files", exist_ok=True); os.makedirs(fz + "/rpt_stack", exist_ok=True); shutil.copytree(fz + "/src/stack", fz + "/rpt_stack/stack")
    blk = blk.replace("/e/Projects/TanitAD-artifacts/dagland/pertick/stack_patch", fz + "/src")
    py = sys.executable.replace("\\", "/")
    harness = f'set -u\nSSH="{py} -I {fz}/fake_ship.py ssh"; SCP="{py} -I {fz}/fake_ship.py scp"\nSRC={fz}/box2_files\nts() {{ date +%H:%M:%S; }}\ndie() {{ echo "DIE $*"; exit 2; }}\n{blk}\necho SHIPOK\n'
    write(fz + "/harness.sh", harness)

    def box(old="OLD"):
        d = f"{fz}/box"; shutil.rmtree(d, ignore_errors=True)
        for rel in RC.SHIPPED: write(f"{d}/repo/{rel}", f"{old}:{rel}\n")
        return d

    def run(env):
        d = box(); rc, o, e = bash([fz + "/harness.sh"], {"FAKE_BOX_ROOT": d, "RNEXT_TRIPWIRE": WORK + "/tripwire_ship.txt", **env}); return rc, o, e, d
    rc, o, e, d = run({}); check(rc == 0 and "SHIPOK" in o, f"happy path: rc {rc} {o} {e[:300]}")
    for rel, want in RC.SHIPPED.items():
        check(hashlib.sha256(open(f"{d}/repo/{rel}", "rb").read()).hexdigest() == want, f"{rel}: box file is not the pinned bytes")
        aside = [f for f in os.listdir(os.path.dirname(f"{d}/repo/{rel}")) if f.startswith(os.path.basename(rel) + ".pre_rpt_")]
        check(len(aside) == 1 and open(f"{os.path.dirname(f'{d}/repo/{rel}')}/{aside[0]}").read() == f"OLD:{rel}\n", f"{rel}: the old file was not moved aside intact ({aside})")
    for label, env, msg in (("scp fails on the recover library", {"FAKE_SCP_FAIL": "v8f_recover.py"}, "scp stack/tanitad/data/v8f_recover.py"),
                            ("box 2 holds other bytes after the scp", {"FAKE_SCP_CORRUPT": "v8f_mini_data.py"}, "on box 2 hashes"),
                            ("the aside move fails", {"FAKE_MOVE_FAIL": "v8f_recover.py"}, "moving box 2's stack/tanitad/data/v8f_recover.py aside failed")):
        rc, o, e, d = run(env); check(rc == 2 and "DIE" in o and msg in o and "SHIPOK" not in o, f"{label}: rc {rc} {o} {e[:200]}")
        for rel in RC.SHIPPED:                              # every file back to its box-2 content, any new bytes kept aside (never deleted)
            check(open(f"{d}/repo/{rel}").read() == f"OLD:{rel}\n", f"{label}: {rel} was not restored")
        left = sorted(f for rel in RC.SHIPPED for f in os.listdir(os.path.dirname(f"{d}/repo/{rel}")) if ".rpt_failed_" in f)
        check(len(left) >= (1 if "scp fails" in label or "other bytes" in label else 0), f"{label}: the failed new bytes were not kept aside: {left}")
    bad = fz + "/src/stack/scripts/v8f_mini_train.py"; write(bad, "# edited\n" + open(bad, encoding="utf-8", newline="").read())      # a local source that differs from its pin is refused BEFORE box 2 is touched
    rc, o, e, d = run({}); check(rc == 2 and "sha256" in o and "!= pinned" in o and all(open(f"{d}/repo/{rel}").read() == f"OLD:{rel}\n" for rel in RC.SHIPPED) and not any(".pre_rpt_" in f for rel in RC.SHIPPED for f in os.listdir(os.path.dirname(f"{d}/repo/{rel}"))), f"local pin mismatch: rc {rc} {o}")
    check(not os.path.exists(WORK + "/tripwire_ship.txt"), "a network tool was invoked")


@test
def t_rpt_package_hygiene():
    """Banned words and line endings for exactly the files written for RPT (the package-wide check above stops at its first pre-existing offender)."""
    mine = [f"{B2}/rnext_chain_rpt.cmd", f"{B2}/rnext_check.py", f"{PKG}/launch_rpt.sh", f"{PKG}/mk_launch_rpt.py", f"{PKG}/mk_rpt_pins.py", f"{PKG}/pull_rnext.sh", f"{PKG}/tests/test_rnext_chain.py", f"{PKG}/read/code/recover_read.py",
            f"{PKG}/read/tests/test_recover_read_arms.py", f"{PKG}/read/rnext_read_chain.sh", f"{PKG}/read/rnext_read_chain_RPT.cmd", f"{PKG}/read/PINS.txt", f"{PKG}/HANDOFF_RPT.md", f"{PKG}/_pre_rpt/SHA_CHANGES.txt"]
    words = [r"\bvo" + "id\\b", "super" + "seded", "retr" + "acted", "for" + "merly", "no lon" + "ger", "corr" + "ection", "with" + "drawn"]
    for f in mine:
        if not os.path.exists(f): raise AssertionError(f"{f}: missing")
        raw = open(f, "rb").read(); txt = raw.decode("utf-8")
        for w in words:
            ln = [l for l in txt.splitlines() if re.search(w, l, re.I)]; ref = os.path.join(PKG, "_pre_rpt", os.path.basename(f) + ".pre_rpt")
            old = open(ref, encoding="utf-8").read().splitlines() if os.path.exists(ref) else []
            check(not [l for l in ln if l not in old], f"{f}: a new line contains a banned word ({w}): {[l for l in ln if l not in old][:1]}")
        ext = os.path.splitext(f)[1]
        if ext in (".py", ".sh", ".md", ".txt"): check(b"\r" not in raw, f"{f}: CR in an LF file")
        if ext == ".cmd" and "/read/" not in f.replace("\\", "/"): check(raw.count(b"\r\n") == raw.count(b"\n") > 0, f"{f}: not CRLF")
        if ext == ".cmd" and "/read/" in f.replace("\\", "/"): check(b"\r" not in raw, f"{f}: CR in an LF read-chain file (the RAPR one is LF)")
    for f in [f"{B2}/rnext_chain_rpt.cmd", f"{B2}/rnext_check.py"]:                  # the box-2 files name no remote and no earlier chain (the package check's rule)
        t = open(f, encoding="utf-8").read(); body = re.sub(r"_recft_(RAPR|RTL|RP01|\{kind\})", "", t)
        check(not re.search(r"recft|CHAIN_RECOVER|tanitad-box2|\bssh\b|\bscp\b", body, re.I), f"{f}: names the previous chain or a remote")
    for t in XT_T:
        a = open(f"{B2}/argv/{t}.train.txt", "rb").read(); check(b"\r" not in a and a.count(b"\n") == 1 and a.endswith(b"\n"), f"{t}: argv is not one LF line")


if __name__ == "__main__":
    only = sys.argv[1:]
    t0 = time.time()
    for fn in TESTS:
        if only and fn.__name__ not in only: continue
        t1 = time.time()
        try: fn(); print(f"PASS {fn.__name__} ({time.time() - t1:.1f}s)")
        except Exception as e:                              # noqa: BLE001
            FAILS.append(fn.__name__); print(f"FAIL {fn.__name__}: {type(e).__name__}: {str(e)[:900]}")
    print(f"TESTS: {len(TESTS) - len(FAILS) if not only else len(only) - len(FAILS)} passed, {len(FAILS)} failed ({time.time() - t0:.0f}s); netguard tripped: {netguard.TRIPPED}; work dir {WORK}")
    sys.exit(1 if FAILS or netguard.TRIPPED else 0)
