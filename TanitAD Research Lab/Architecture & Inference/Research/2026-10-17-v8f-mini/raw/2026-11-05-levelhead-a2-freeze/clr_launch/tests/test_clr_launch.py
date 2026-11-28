"""Dev-box tests of the arm-CLR box-2 package (CPU only, NO network: tests/netguard.py raises before anything leaves this process; shell children get BASH_ENV functions that block ssh/scp/...; the launcher dry runs use a local fake box).
Run:  CUDA_VISIBLE_DEVICES=-1 python -I tests/test_clr_launch.py [test names]       (work dir: CLR_TEST_WORK, else a temp dir). Exit 0 iff every test passed. Log: logs/test_clr_launch.log.
Two packages are exercised: the REAL one on disk (final state: W = CL's frozen 0.2027, there is no placeholder state) and a copy built in the work dir (own a2_prereg / cl_prereg folders, never written to the real ones) for the cases that change a card, a W or a probe."""
import copy, hashlib, importlib.util, json, os, re, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PKG = os.path.dirname(HERE)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, HERE)
import netguard  # noqa: E402
netguard.install()
WORK = (os.environ.get("CLR_TEST_WORK") or tempfile.mkdtemp(prefix="cl_test_")).replace("\\", "/")
os.makedirs(WORK, exist_ok=True)
PYW = sys.executable.replace("\\", "/")
C0 = "E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s{s}_recft_C0"
RAPR = "E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s{s}_recft_RAPRP01"
CLREAL = "E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s{s}_recft_CL"        # CL's real run: its gate selfcheck.json carries the real level_head_probe at W 0.2027
RPT_CHAIN = "E:/Projects/TanitAD-artifacts/dagland/recover_next/box2_files"
A2CARD = f"{PKG}/../a2_prereg"; CLW = f"{PKG}/../cl_prereg"
TESTS, FAILS = [], []
REAL_W = "0.2027"


def test(fn):
    TESTS.append(fn); return fn


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def write(p, s, mode="w"):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode, **({} if "b" in mode else {"encoding": "utf-8", "newline": ""})) as f:
        f.write(s)


def rd(p, mode="r"):
    with open(p, mode, **({} if "b" in mode else {"encoding": "utf-8", "newline": ""})) as f:
        return f.read()


def sh256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def loadmod(name, path, **env):
    os.environ.update(env)
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


# ---- the work-dir copy (own a2_prereg / cl_prereg; the default copy is byte-identical to the real package) -----------------------------------------------------------------------------------------------------------------------
def build_frozen(name="pk", w=REAL_W, card_extra=None, probe=None):
    root = f"{WORK}/{name}"; shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(PKG, f"{root}/clr_launch", ignore=shutil.ignore_patterns("__pycache__", "logs", "tests"))
    os.makedirs(f"{root}/a2_prereg"); os.makedirs(f"{root}/cl_prereg")
    extra = card_extra if card_extra is not None else ("" if w == REAL_W else f"\n--w-level {w} (test stand-in)\n")
    card = rd(f"{A2CARD}/PREREG_LEVELHEAD_A2.md", "rb") + extra.encode("utf-8")
    write(f"{root}/a2_prereg/PREREG_LEVELHEAD_A2.md", card, "wb")
    write(f"{root}/a2_prereg/FROZEN_SHA.txt", hashlib.sha256(card).hexdigest() + " *PREREG_LEVELHEAD_A2.md\n")
    write(f"{root}/cl_prereg/FROZEN_W.txt", rd(f"{CLW}/FROZEN_W.txt") if w == REAL_W else f"W={w}\n# test stand-in\n")
    write(f"{root}/cl_prereg/W_PROBE.json", json.dumps(probe) if probe is not None else (rd(f"{CLW}/W_PROBE.json") if w == REAL_W else json.dumps({"w_level_for_10pct_share": float(w) * 1.0003})))
    r = subprocess.run([sys.executable, "-I", f"{root}/clr_launch/mk_clr_package.py"], capture_output=True, text=True)
    check(r.returncode == 0, f"generator in the copy: {r.stdout} {r.stderr}")
    return root


def bash(args, extra, timeout=120, drop=()):
    exe = "C:/Program Files/Git/bin/bash.exe"; exe = exe if os.path.exists(exe) else "bash"
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update({"BASH_ENV": HERE + "/blockenv.sh", "CLR_TRIPWIRE": WORK + "/tripwire.txt", **extra})
    r = subprocess.run([exe] + args, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.replace("\r", ""), r.stderr


FROZEN = build_frozen()
REAL_CK = loadmod("clr_check_real", f"{PKG}/box2_files/clr_check.py", CLR_HOME=f"{PKG}/box2_files")
FRZ_CK_PATH = f"{FROZEN}/clr_launch/box2_files/clr_check.py"
FRZ = loadmod("clr_check_frz", FRZ_CK_PATH, CLR_HOME=f"{FROZEN}/clr_launch/box2_files")
BOOT = "import sys,runpy; sys.path.insert(0,%r); import netguard; netguard.install(); sys.argv=%r; runpy.run_path(%r, run_name='__main__')"


class SB:
    """A box-2 sandbox for the checker: HOME (shipped argv + FROZEN_W), RUNSM, a repo holding the four shipped stack files, init ckpts, a disk floor."""
    def __init__(self, name, pkg=None, floor="0", mutate_stack=None):
        pkg = pkg or f"{FROZEN}/clr_launch"; self.pkg = pkg; self.root = f"{WORK}/{name}"; shutil.rmtree(self.root, ignore_errors=True)
        self.home, self.runsm, self.repo, self.disk, self.floor = self.root + "/home", self.root + "/runs", self.root + "/repo", self.root, floor
        shutil.copytree(f"{pkg}/box2_files", self.home, ignore=shutil.ignore_patterns("__pycache__"))
        self.check = f"{pkg}/box2_files/clr_check.py"
        for rel in FRZ.SHIPPED:
            write(f"{self.repo}/{rel}", rd(f"{pkg}/stack_ship/{rel}", "rb"), "wb")
        for s in (0, 1):
            write(f"{self.runsm}/M2b_s{s}_dagoacc/ckpt_last.pt", "x")

    def run(self, args):
        env = {**os.environ, "CLR_HOME": self.home, "CLR_RUNSM": self.runsm, "CLR_REPO": self.repo, "CLR_DISK": self.disk, "CLR_FLOOR_BYTES": self.floor, "CUDA_VISIBLE_DEVICES": "-1"}
        r = subprocess.run([sys.executable, "-I", "-c", BOOT % (HERE, [self.check] + list(args), self.check)], env=env, capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    def rc(self): return self.runsm + "/clr"
    def gdir(self, tag): return f"{self.runsm}/gate_clr/M2b_s{FRZ.INFO[tag][1]}_recft_CLR"
    def odir(self, tag): return f"{self.runsm}/M2b_s{FRZ.INFO[tag][1]}_recft_CLR"
    def argv(self, tag): return rd(f"{self.home}/argv/{tag}.train.txt").strip()
    def set_argv(self, tag, text): write(f"{self.home}/argv/{tag}.train.txt", text + "\n")


LVL6 = ["lvl_head.0.weight", "lvl_head.0.bias", "lvl_head.2.weight", "lvl_head.2.bias", "e_level.weight", "e_level.bias"]


def stub_sc(tag):
    s = FRZ.INFO[tag][1]
    sc = json.load(open(CLREAL.format(s=s) + "/gate_selfcheck.json", encoding="utf-8"))      # CL's REAL gate selfcheck.json (the shipped trainer, level head on at W 0.2027): every level key is real
    rp = json.load(open(RAPR.format(s=s) + "/gate_selfcheck.json", encoding="utf-8"))        # RAPR-P01's REAL gate selfcheck.json: the recovery blocks (levers.recover, levers.recover_table, recover_probe, info.recover)
    sc["levers"]["recover"] = rp["levers"]["recover"]; sc["levers"]["recover_table"] = rp["levers"]["recover_table"]; sc["recover_probe"] = rp["recover_probe"]
    sc["info"].pop("recover", None)
    if "recover" in rp["info"]: sc["info"]["recover"] = rp["info"]["recover"]
    sc["argv_sha"] = "stub0000"
    return sc


def stage_gate(sb, tag, sc, log="SELFCHECK PASS {}"):
    write(f"{sb.gdir(tag)}/selfcheck.json", json.dumps(sc, indent=1)); write(f"{sb.rc()}/logs/{tag}.gate.log", log + "\n")


def pinned_sb(name, tags=("CLR_0", "CLR_1"), **kw):
    sb = SB(name, **kw)
    for t in tags:
        rc, o = sb.run(["pin", t]); check(rc == 0, f"pin {t}: {o}")
    return sb


def setp(d, path, val):
    for k in path[:-1]: d = d[k]
    d[path[-1]] = val


def popp(d, path):
    for k in path[:-1]: d = d[k]
    d.pop(path[-1])


# =====================================================================================================================================================================================================================================
@test
def t_netguard_canaries():
    check(netguard.installed(), "netguard not installed")
    for fn, what in ((lambda: subprocess.run(["ssh", "tanitad-box2", "true"]), "ssh"), (lambda: subprocess.run("scp x tanitad-box2:y", shell=True), "scp via shell"), (lambda: os.system("echo hi"), "os.system"),
                     (lambda: __import__("socket").create_connection(("192.168.178.171", 22), 1), "socket")):
        try: fn()
        except RuntimeError as e: check("NETWORK-BLOCKED" in str(e), f"{what}: {e}")
        else: raise AssertionError(f"{what} was not blocked")
    check(len(netguard.TRIPPED) >= 4, "tripwire list")
    del netguard.TRIPPED[:]                                                               # the canaries tripped it on purpose
    rc, o, e = bash(["-c", "ssh somewhere true; scp a b"], {}); check(rc != 0 and os.path.exists(WORK + "/tripwire.txt"), "BASH_ENV functions did not block ssh/scp in a shell child"); os.remove(WORK + "/tripwire.txt")
    import ast
    tree = ast.parse(open(f"{PKG}/box2_files/clr_check.py", encoding="utf-8").read())
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    bad_attrs = attrs & {"system", "popen", "Popen", "run", "connect", "create_connection", "urlopen"}
    check(mods == {"hashlib", "json", "os", "shutil", "sys", "time"} and not bad_attrs, f"the checker imports {sorted(mods)} / calls {sorted(bad_attrs)}: stdlib file code only")
    check("tanitad-box2" not in open(f"{PKG}/box2_files/clr_check.py", encoding="utf-8").read(), "the checker names the remote")


@test
def t_generator_is_reproducible_and_pins_are_the_source_bytes():
    r = subprocess.run([sys.executable, "-I", f"{PKG}/mk_clr_package.py", "check"], capture_output=True, text=True); check(r.returncode == 0 and "OK" in r.stdout, f"generator check on the real package: {r.stdout} {r.stderr}")
    r = subprocess.run([sys.executable, "-I", f"{FROZEN}/clr_launch/mk_clr_package.py", "check"], capture_output=True, text=True); check(r.returncode == 0, f"work-dir copy: {r.stdout}")
    pins = json.load(open(f"{PKG}/PINS.json")); check(pins == REAL_CK.PINS, "PINS.json != the pins embedded in the shipped checker")
    check(pins["w"] == REAL_W and rd(f"{CLW}/FROZEN_W.txt").startswith("W=" + REAL_W) and pins["frozen_w_sha256"] == sh256(f"{CLW}/FROZEN_W.txt"), "the pinned W is not CL's frozen W")
    tip = open(f"{PKG}/tip_commit.txt").read().strip()                                    # all four shipped files equal the blobs at the pinned tip (local mirror, offline)
    check(len(pins["stack"]) == 4 and all(x["origin"] == "tip" for x in pins["stack"]), "four stack files, every one from the tip")
    for x in pins["stack"]:
        got = sh256(f"{PKG}/{x['src']}"); check(got == x["sha256"], f"{x['rel']}: staged bytes {got} != pin")
        blob = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "cat-file", "blob", f"{tip}:{x['rel']}"], capture_output=True)
        check(blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == x["sha256"], f"{x['rel']}: the blob at tip {tip[:8]} is not the pin")
    check([x["sha256"][:8] for x in pins["stack"]] == ["9ffdf1b4", "a3634276", "0257ccb9", "209da7a6"], "trainer / model / recover lib / data lib are not 9ffdf1b4 / a3634276 / 0257ccb9 / 209da7a6")
    for s in (0, 1):
        check(sh256(f"{PKG}/rapr_argv/RAPRP01_s{s}.argv.json") == sh256(RAPR.format(s=s) + "/argv.json") == pins["rapr_argv_sha256"][str(s)], f"rapr_argv copy s{s} != RAPR-P01's argv.json")
    card = f"{A2CARD}/PREREG_LEVELHEAD_A2.md"
    check(pins["card_sha256"] == sh256(card) and sh256(card) in rd(f"{A2CARD}/FROZEN_SHA.txt"), "the pinned card sha256 is not the frozen A2 card")
    for rel in REAL_CK_ship():
        check(sh256(f"{PKG}/box2_files/{rel}") == sh256(f"{FROZEN}/clr_launch/box2_files/{rel}"), f"{rel}: the default work-dir copy differs from the real package")
    check(REAL_CK_ship() == ["FROZEN_W.txt", "argv/CLR_0.train.txt", "argv/CLR_1.train.txt", "clr_chain.cmd", "clr_check.py", "clr_guard_chain.ps1", "clr_run.cmd", "clr_run_guard_chain.cmd", "clr_run_telemetry.cmd"], f"the shipped file set: {REAL_CK_ship()}")


def REAL_CK_ship():
    r = subprocess.run([sys.executable, "-I", f"{PKG}/clr_local_check.py", "ship-files"], capture_output=True, text=True); return [x.strip() for x in r.stdout.splitlines() if x.strip()]


@test
def t_argv_is_raprp01_plus_exactly_out_and_the_two_level_tokens():
    for s in (0, 1):
        rp = json.load(open(RAPR.format(s=s) + "/argv.json", encoding="utf-8"))["argv"]; c0 = json.load(open(C0.format(s=s) + "/argv.json", encoding="utf-8"))["argv"]
        i = rp.index("--out") + 1; exp = list(rp); exp[i] = exp[i].replace("_recft_RAPRP01", "_recft_CLR")
        for pkg in (PKG, f"{FROZEN}/clr_launch"):
            cl = rd(f"{pkg}/box2_files/argv/CLR_{s}.train.txt").strip().split(" ")
            check(cl == exp + ["--level-head", "on", "--w-level", REAL_W], f"CLR_{s} != RAPR-P01's argv with --out replaced + the two level tokens")
            check(sorted(t for t in cl if t not in rp) == sorted(["--level-head", "on", "--w-level", REAL_W, cl[i]]), "the tokens CLR has and RAPR-P01 lacks are the two levers and --out")
            check(cl[:i] == rp[:i] and cl[i + 1:len(rp)] == rp[i + 1:] and len(cl) == len(rp) + 4 and cl[i].endswith(f"M2b_s{s}_recft_CLR") and rp[i].endswith(f"M2b_s{s}_recft_RAPRP01"), "the tokens outside --out and the tail differ from RAPR-P01's")
            check(FRZ.expected_tokens(f"CLR_{s}") == cl and FRZ.audit_argv(f"CLR_{s}", cl) == [], f"CLR_{s}: clr_check.expected_tokens != the generated argv: {FRZ.audit_argv(f'CLR_{s}', cl)}")   # two independent encodings
        j = rp.index("--recover-p")                                                       # RAPR-P01's recovery tail: the registered lever
        check(rp[j:] == ["--recover-p", "0.1", "--recover-target", "rejoin", "--recover-rejoin-steps", "10", "--recover-seed", str(s), "--recover-dev-law", "apriori"], f"RAPR-P01 s{s}'s recovery tokens: {rp[j:]}")
        k, k0 = c0.index("--out") + 1, c0.index("--recover-p")                            # every OTHER lever is C0's: same tokens up to --recover-p (only --out differs), and C0's tail is `--recover-p 0`
        check(c0[k0:] == ["--recover-p", "0"] and rp[:i] == c0[:k] and rp[i + 1:j] == c0[k + 1:k0], f"RAPR-P01's non-recovery tokens differ from C0's beyond --out: {[(a, b) for a, b in zip(rp[:j], c0[:k0]) if a != b][:3]}")


@test
def t_real_package_local_check_passes_and_has_no_probe_mode():
    r = subprocess.run([sys.executable, "-I", f"{PKG}/clr_local_check.py", "chain"], capture_output=True, text=True); check(r.returncode == 0 and "local check OK (chain)" in r.stdout, f"local check chain on the real package: rc {r.returncode} {r.stdout[:300]}")
    r = subprocess.run([sys.executable, "-I", f"{PKG}/clr_local_check.py", "wprobe"], capture_output=True, text=True); check(r.returncode == 2, f"there is no probe mode: rc {r.returncode}")
    check(not [f for f in os.listdir(PKG) if "wprobe" in f.lower() or f in ("freeze_w.py", "w_probe_read.py")] and not [f for f in os.listdir(f"{PKG}/box2_files") if "wprobe" in f.lower()], "no probe launcher / pull / guard / freeze helper is shipped")
    sb = SB("real_pf", pkg=PKG); rc, o = sb.run(["preflight", "chain"]); check(rc == 0 and "preflight OK (chain)" in o, f"preflight chain on the real package: rc {rc} {o[:300]}")


@test
def t_local_check_chain_refusals():
    def lc(root): return subprocess.run([sys.executable, "-I", f"{root}/clr_launch/clr_local_check.py", "chain"], capture_output=True, text=True)
    r = lc(FROZEN); check(r.returncode == 0, f"work-dir copy chain: {r.stdout}")
    cases = []
    r0 = build_frozen("lc1"); os.remove(f"{r0}/cl_prereg/W_PROBE.json"); cases.append((r0, "W_PROBE.json is absent"))
    r0 = build_frozen("lc2", probe={"w_level_for_10pct_share": 0.9}); cases.append((r0, "not within 1%"))
    r0 = build_frozen("lc3"); write(f"{r0}/a2_prereg/PREREG_LEVELHEAD_A2.md", rd(f"{r0}/a2_prereg/PREREG_LEVELHEAD_A2.md") + "\nedited after the freeze\n"); cases.append((r0, "card on disk is not the card the pins were made from"))
    r0 = build_frozen("lc4"); write(f"{r0}/cl_prereg/FROZEN_W.txt", "W=0.8\n"); cases.append((r0, "FROZEN_W.txt says '0.8' but the package pins"))
    r0 = build_frozen("lc5"); write(f"{r0}/clr_launch/box2_files/FROZEN_W.txt", "W=0.2027\n# other bytes\n"); cases.append((r0, "box2_files/FROZEN_W.txt is not the cl_prereg/FROZEN_W.txt"))
    r0 = build_frozen("lc6", w="0.5", card_extra="\nno W named here\n"); cases.append((r0, "the card does not name `--w-level 0.5`"))
    r0 = build_frozen("lc7"); a = rd(f"{r0}/clr_launch/box2_files/argv/CLR_1.train.txt").replace("--w-level 0.2027", "--w-level 0.9"); write(f"{r0}/clr_launch/box2_files/argv/CLR_1.train.txt", a); cases.append((r0, "CLR_1: argv audit"))
    r0 = build_frozen("lc8"); write(f"{r0}/clr_launch/stack_ship/stack/scripts/v8f_mini_train.py", "# edited\n" + rd(f"{r0}/clr_launch/stack_ship/stack/scripts/v8f_mini_train.py")); cases.append((r0, "stack source"))
    r0 = build_frozen("lc9"); os.remove(f"{r0}/a2_prereg/FROZEN_SHA.txt"); cases.append((r0, "FROZEN_SHA.txt is absent"))
    r0 = build_frozen("lc10"); write(f"{r0}/a2_prereg/FROZEN_SHA.txt", "0" * 64 + " *PREREG_LEVELHEAD_A2.md\n"); cases.append((r0, "is not the one FROZEN_SHA.txt records"))
    r0 = build_frozen("lc11"); write(f"{r0}/clr_launch/rapr_argv/RAPRP01_s0.argv.json", rd(f"{r0}/clr_launch/rapr_argv/RAPRP01_s0.argv.json") + " "); cases.append((r0, "is not the pinned RAPR-P01 argv.json"))
    r0 = build_frozen("lc12"); a = rd(f"{r0}/clr_launch/box2_files/argv/CLR_0.train.txt").replace("--recover-p 0.1", "--recover-p 0.3"); write(f"{r0}/clr_launch/box2_files/argv/CLR_0.train.txt", a); cases.append((r0, "--recover-p = '0.3'"))
    r0 = build_frozen("lc13"); write(f"{r0}/clr_launch/PINS.json", rd(f"{r0}/clr_launch/PINS.json").replace('"w": "0.2027"', '"w": "0.2028"')); cases.append((r0, "PINS.json differs from the pins embedded"))
    for root, msg in cases:
        r = lc(root); check(r.returncode == 6 and msg in r.stdout, f"{root}: rc {r.returncode}, want a refusal naming {msg!r}: {r.stdout[:400]}")


@test
def t_checker_preflight_pin_and_mutated_argv():
    sb = SB("pf", pkg=PKG)
    rc, o = sb.run(["preflight", "chain"]); check(rc == 0 and "preflight OK (chain)" in o and f"W {REAL_W}" in o, f"preflight chain: rc {rc} {o[:300]}")
    for t in ("CLR_0", "CLR_1"):
        rc, o = sb.run(["pin", t]); check(rc == 0 and "pinned" in o, f"pin {t}: {o}")
        check(rd(f"{sb.rc()}/pin/{t}.train.txt").strip() == sb.argv(t) and rd(f"{sb.rc()}/pin/{t}.gate.txt").strip().endswith(" --selfcheck-only") and "/gate_clr/" in rd(f"{sb.rc()}/pin/{t}.gate.txt").replace("\\", "/"), f"pin files {t}")
    check(sb.run(["pin", "CLR_0"])[0] == 0, "a second identical pin is accepted")
    p = f"{sb.rc()}/pin/CLR_0.train.txt"; write(p, rd(p).strip() + " --extra 1\r\n"); rc, o = sb.run(["pin", "CLR_0"]); check(rc == 2 and "pins another train argv" in o, f"a pin that differs is refused: {o}")
    base = sb.argv("CLR_0")
    muts = [("other W", base.replace("--w-level 0.2027", "--w-level 0.5"), "--w-level = '0.5'"), ("steps", base.replace("--steps 3000", "--steps 2000"), "--steps = '2000'"),
            ("order", base.replace("--level-head on --w-level 0.2027", "--w-level 0.2027 --level-head on"), "token order differs"),
            ("recover p 0.3", base.replace("--recover-p 0.1", "--recover-p 0.3"), "--recover-p = '0.3'"), ("recover off", base.replace("--recover-p 0.1", "--recover-p 0"), "--recover-p = '0'"),
            ("recover law", base.replace("--recover-dev-law apriori", "--recover-dev-law empirical"), "--recover-dev-law = 'empirical'"), ("recover target", base.replace("--recover-target rejoin", "--recover-target logged"), "--recover-target = 'logged'"),
            ("rejoin steps", base.replace("--recover-rejoin-steps 10", "--recover-rejoin-steps 5"), "--recover-rejoin-steps = '5'"), ("recover seed", base.replace("--recover-seed 0", "--recover-seed 1"), "--recover-seed = '1'"),
            ("recover tokens dropped", base.replace(" --recover-p 0.1", ""), "missing --recover-p"), ("a bank flag", base + " --recover-dev-bank /x", "unexpected flag --recover-dev-bank"),
            ("level head dropped", base.replace(" --level-head on", ""), "missing --level-head"), ("a dagger flag", base + " --dagger-p 0.1", "unexpected flag --dagger-p"),
            ("selfcheck-only in the train argv", base + " --selfcheck-only", "unexpected flag --selfcheck-only"), ("W given twice", base + " --w-level 0.3", "is given 2 times"),
            ("ego perturb", base.replace("--ego-perturb-p 0", "--ego-perturb-p 0.1"), "--ego-perturb-p = '0.1'"), ("w-acc", base.replace("--w-acc 49.6969", "--w-acc 1"), "--w-acc = '1'"),
            ("init ckpt", base.replace("M2b_s0_dagoacc", "M2b_s0_other"), "--init-ckpt"), ("stray token", base + " stray", "stray token"),
            ("out is RAPR-P01's folder", base.replace("_recft_CLR", "_recft_RAPRP01"), "--out ="), ("out is CL's folder", base.replace("_recft_CLR", "_recft_CL"), "--out =")]
    for label, text, msg in muts:
        s_ = SB("pfm"); s_.set_argv("CLR_0", text); rc, o = s_.run(["pin", "CLR_0"]); check(rc == 2 and "REFUSED" in o and msg in o, f"{label}: pin rc {rc}, want {msg!r}: {o[:300]}")
        rc, o = s_.run(["preflight", "chain"]); check(rc == 3 and "argv audit" in o, f"{label}: preflight rc {rc}: {o[:200]}")
    s1 = SB("pf1", pkg=PKG); rel = "stack/tanitad/models/v8f_mini.py"; write(f"{s1.repo}/{rel}", "# x\n" + rd(f"{s1.repo}/{rel}")); rc, o = s1.run(["preflight", "chain"]); check(rc == 3 and f"{rel}: sha256" in o, f"a changed shipped file: {o[:300]}")
    s2 = SB("pf2", pkg=PKG); os.remove(f"{s2.runsm}/M2b_s1_dagoacc/ckpt_last.pt"); rc, o = s2.run(["preflight", "chain"]); check(rc == 3 and "init ckpt" in o, f"a missing init ckpt: {o[:300]}")
    s3 = SB("pf3", pkg=PKG); write(f"{s3.home}/FROZEN_W.txt", "W=0.3\n"); rc, o = s3.run(["preflight", "chain"]); check(rc == 3 and "FROZEN_W.txt differs from the pinned bytes" in o, f"a changed shipped W file: {o[:300]}")
    s4 = SB("pf4", pkg=PKG, floor="1e18"); rc, o = s4.run(["preflight", "chain"]); check(rc == 3 and "disk floor" in o, f"disk floor chain: {o[:200]}")
    rc, o = s4.run(["preflight", "run", "CLR_0"]); check(rc == 9 and "ZZCLR-STOP disk floor" in o, f"disk floor run: rc {rc} {o[:200]}")
    check(sb.run(["preflight", "wprobe"])[0] == 2 and sb.run(["preflight"])[0] == 2, "preflight takes chain | run <tag> only")


@test
def t_gate_pass_token_binding_and_cache():
    sb = pinned_sb("gp")
    for t in ("CLR_0", "CLR_1"):
        stage_gate(sb, t, stub_sc(t)); rc, o = sb.run(["gate", t, "0"]); check(rc == 0 and "ZZCLR-GATE-PASS" in o, f"gate {t}: {o}")
        tok = json.load(open(f"{sb.rc()}/gate_token_{t}.json")); b = tok["binding"]
        check(tok["verdict"] == "PASS" and b["w_level"] == REAL_W and b["argv_sha256"] == FRZ.tokens_sha(sb.argv(t).split(" ")) and b["shipped_sha256"] == FRZ.SHIPPED and tok["prereg_sha256"] == FRZ.CARD_SHA
              and b["init_ckpt_sha256"] == FRZ.INIT_SHA[FRZ.INFO[t][1]] and (abs(b["level_share_at_step0"] - 0.1) < 1e-3 if t == "CLR_0" else 0.0 < b["level_share_at_step0"] < 0.5) and "never by hand" in tok["written_by"], f"token {t}: {tok}")
        rc, o = sb.run(["gate-cached", t]); check(rc == 0 and "YES" in o, f"cached {t}: {o}")
    p = f"{sb.gdir('CLR_0')}/selfcheck.json"; txt = rd(p); write(p, txt.replace('"stub0000"', '"stub0001"')); check(sb.run(["gate-cached", "CLR_0"])[0] == 1, "a changed selfcheck.json must not be cached"); write(p, txt)
    check(sb.run(["gate-cached", "CLR_0"])[0] == 0, "restored selfcheck is cached again")
    rel = "stack/tanitad/models/v8f_mini.py"; write(f"{sb.repo}/{rel}", "# x\n" + rd(f"{sb.repo}/{rel}")); check(sb.run(["gate-cached", "CLR_0"])[0] == 1, "a changed shipped model must not be cached")
    tk = f"{sb.rc()}/gate_token_CLR_1.json"; j = json.load(open(tk)); j["binding"]["w_level"] = "0.5"; write(tk, json.dumps(j)); check(sb.run(["gate-cached", "CLR_1"])[0] == 1, "a token for another W must not be cached")
    os.remove(f"{sb.rc()}/gate_token_CLR_0.json"); check(sb.run(["gate-cached", "CLR_0"])[0] == 1, "no token = not cached")


@test
def t_gate_negatives_each_named():
    cases = [("declared_vs_built FAIL", lambda d: setp(d, ["declared_vs_built"], "FAIL"), "declared_vs_built = FAIL"),
             ("init sha", lambda d: setp(d, ["info", "init_ckpt", "sha256"], "0" * 64), "init ckpt sha256"),
             ("init not applied", lambda d: setp(d, ["info", "init_ckpt", "applied"], False), "init ckpt: applied"),
             ("a key fresh that is not declared", lambda d: setp(d, ["info", "init_ckpt", "fresh_keys"], LVL6 + ["S.w"]), "init ckpt: applied"),
             ("five fresh keys", lambda d: (setp(d, ["info", "init_ckpt", "fresh_keys"], LVL6[:5]), setp(d, ["info", "init_ckpt", "declared_new_keys"], LVL6[:5])), "exactly the six"),
             ("mismatch after load", lambda d: setp(d, ["info", "init_ckpt", "n_mismatch_after_load"], 3), "init ckpt: applied"),
             ("train sha", lambda d: setp(d, ["info", "train_sha256"], "1" * 64), "train_sha256"),
             ("steps", lambda d: setp(d, ["steps_total"], 2000), "steps_total"),
             ("lever off", lambda d: setp(d, ["levers", "level_head"], "off"), "built level_head"),
             ("W built differs", lambda d: setp(d, ["levers", "w_level"], 0.5), "built level_head"),
             ("token not zero at init", lambda d: setp(d, ["levers", "level_token_zero_init"], False), "level_token_zero_init"),
             ("non-vision source", lambda d: setp(d, ["levers", "level_vision_only_source"], False), "level_vision_only_source"),
             ("recover p 0.3", lambda d: setp(d, ["levers", "recover", "recover_p"], 0.3), "built recover lever recover_p = 0.3"),
             ("recover off (C0's recipe)", lambda d: setp(d, ["levers", "recover", "recover_p"], 0.0), "built recover lever recover_p = 0.0"),
             ("recover law", lambda d: setp(d, ["levers", "recover", "recover_dev_law"], "empirical"), "built recover lever recover_dev_law"),
             ("recover target", lambda d: setp(d, ["levers", "recover", "recover_target"], "logged"), "built recover lever recover_target"),
             ("rejoin steps", lambda d: setp(d, ["levers", "recover", "recover_rejoin_steps"], 5), "built recover lever recover_rejoin_steps"),
             ("recover seed", lambda d: setp(d, ["levers", "recover", "recover_seed"], 1), "built recover lever recover_seed"),
             ("recover probe fails", lambda d: setp(d, ["recover_probe", "pass"], False), "recover_probe pass"),
             ("recover probe absent", lambda d: popp(d, ["recover_probe"]), "recover_probe pass"),
             ("a table built", lambda d: setp(d, ["levers", "recover_table"], {"n": 5}), "levers.recover_table is not None"),
             ("a table record", lambda d: setp(d, ["info", "recover"], {"bank_sha256": "x"}), "a deviation table was built"),
             ("a bank built", lambda d: setp(d, ["levers", "recover", "recover_dev_bank_sha256"], "ab" * 32), "a bank is built"),
             ("dagger on", lambda d: setp(d, ["levers", "dagger_p"], 0.1), "another lever is on"),
             ("launch intent on", lambda d: setp(d, ["levers", "launch_intent_w"], 0.5), "another lever is on"),
             ("ego perturb on", lambda d: setp(d, ["levers", "ego_perturb", "ego_perturb_p"], 0.2), "another lever is on"),
             ("w_acc differs", lambda d: setp(d, ["levers", "w_acc"], 1.0), "built lever w_acc"),
             ("p_gt_start differs", lambda d: setp(d, ["levers", "p_gt_start"], 0.5), "built lever p_gt_start"),
             ("kin_mode differs", lambda d: setp(d, ["levers", "kin_mode"], "xy"), "built lever kin_mode"),
             ("no probe", lambda d: popp(d, ["level_head_probe"]), "level_head_probe is absent"),
             ("probe fail", lambda d: setp(d, ["level_head_probe", "pass"], False), "level_head_probe pass"),
             ("probe W", lambda d: setp(d, ["level_head_probe", "w_level_declared"], 0.5), "w_level_declared"),
             ("detach control non-zero", lambda d: setp(d, ["level_head_probe", "head_grad_at_w0_detach_control"], 1e-9), "head_grad_at_w0_detach_control"),
             ("prior control non-zero", lambda d: setp(d, ["level_head_probe", "control_prior_grad_diff"], 0.1), "control_prior_grad_diff"),
             ("allfalse term non-zero", lambda d: setp(d, ["level_head_probe", "control_allfalse_term"], 0.1), "control_allfalse_term"),
             ("head gets no gradient", lambda d: setp(d, ["level_head_probe", "head_grad_diff"], 0.0), "head_grad_diff"),
             ("token does not reach O", lambda d: setp(d, ["level_head_probe", "plan_moves_with_token"], 0.0), "plan_moves_with_token"),
             ("term does not reach the loss", lambda d: setp(d, ["level_head_probe", "contribution"], 0.0), "does not reach the loss"),
             ("W10 inconsistent", lambda d: setp(d, ["level_head_probe", "w_level_for_10pct_share"], 2.0), "w_level_for_10pct_share"),
             ("share off the tenth (seed 0)", lambda d: setp(d, ["level_head_probe", "step0_level_term"], 0.5), "level share at step 0")]
    for label, mut, msg in cases:
        sb = pinned_sb("gn", tags=("CLR_0",)); sc = stub_sc("CLR_0"); mut(sc); stage_gate(sb, "CLR_0", sc)
        rc, o = sb.run(["gate", "CLR_0", "0"]); check(rc == 1 and msg in o, f"{label}: rc {rc}, want {msg!r}: {o[:300]}")
        check(os.path.exists(f"{sb.rc()}/RUN_CLR_0.FAILED") and not os.path.exists(f"{sb.rc()}/gate_token_CLR_0.json"), f"{label}: FAILED marker / token state")
    def share_sc(tag, share):                                    # a step-0 record whose level share is `share` (W10 kept consistent with the record)
        sc = stub_sc(tag); lp = sc["level_head_probe"]; t = lp["step0_level_term"]; w = float(REAL_W); base = w * t * (1.0 / share - 1.0)
        lp["step0_total_without_level"] = base; lp["step0_total"] = base + w * t; lp["w_level_for_10pct_share"] = (0.1 / 0.9) * base / t
        return sc
    for share, ok in ((0.079, False), (0.081, True), (0.119, True), (0.121, False)):             # the card's window [0.08, 0.12] at seed 0
        sb = pinned_sb("gsh", tags=("CLR_0",)); stage_gate(sb, "CLR_0", share_sc("CLR_0", share)); rc, o = sb.run(["gate", "CLR_0", "0"])
        check((rc == 0) == ok and (ok or ("level share at step 0" in o and "[0.08, 0.12]" in o)), f"seed-0 share {share}: rc {rc} {o[:300]}")
    sb = pinned_sb("gsh1", tags=("CLR_1",)); stage_gate(sb, "CLR_1", share_sc("CLR_1", 0.05)); rc, o = sb.run(["gate", "CLR_1", "0"])
    check(rc == 0 and abs(json.load(open(f"{sb.rc()}/gate_token_CLR_1.json"))["binding"]["level_share_at_step0"] - 0.05) < 1e-6, f"seed 1: the share is recorded, not gated: {o[:200]}")
    sb = pinned_sb("gn2", tags=("CLR_0",)); stage_gate(sb, "CLR_0", stub_sc("CLR_0")); rc, o = sb.run(["gate", "CLR_0", "1"]); check(rc == 1 and "exited rc 1" in o, f"gate rc 1: {o}")
    sb = pinned_sb("gn3", tags=("CLR_0",)); stage_gate(sb, "CLR_0", stub_sc("CLR_0"), log="x"); rc, o = sb.run(["gate", "CLR_0", "0"]); check(rc == 1 and "'SELFCHECK PASS' not in the gate log" in o, f"no PASS line: {o}")
    sb = pinned_sb("gn4", tags=("CLR_0",)); rc, o = sb.run(["gate", "CLR_0", "0"]); check(rc == 1 and "no readable selfcheck.json" in o, f"no selfcheck: {o}")
    sb = pinned_sb("gn5", tags=("CLR_1",)); sc = stub_sc("CLR_1"); setp(sc, ["level_head_probe", "step0_level_term"], 0.5); setp(sc, ["level_head_probe", "w_level_for_10pct_share"], (0.1 / 0.9) * sc["level_head_probe"]["step0_total_without_level"] / 0.5)
    stage_gate(sb, "CLR_1", sc); rc, o = sb.run(["gate", "CLR_1", "0"]); check(rc == 0, f"seed 1 reads another batch: its share is recorded, not gated to a tenth: {o}")


@test
def t_post_finish_stopped_and_fail_markers():
    def stage_run(sb, t, **over):
        s = FRZ.INFO[t][1]; o = sb.odir(t); sm = {"steps": 3000, "train_sha256": FRZ.KP1["train_sha256"], "train_window_list_sha256": FRZ.KP1["train_window_list_sha256"], "level_head": "on", "w_level": float(REAL_W), "recover_n": 19208, "recover_share": 0.1}
        sm.update({k: v for k, v in over.items() if k != "argv"})
        write(f"{o}/summary.json", json.dumps(sm)); write(f"{o}/argv.json", json.dumps({"argv": over.get("argv", sb.argv(t).split(" "))})); write(f"{o}/ckpt_last.pt", b"ckpt" + bytes([s]), "wb")
    sb = pinned_sb("po")
    stage_run(sb, "CLR_0"); rc, o = sb.run(["post", "CLR_0", "0"]); check(rc == 0 and os.path.exists(f"{sb.rc()}/RUN_CLR_0.done"), f"post: {o}")
    ev = json.load(open(f"{sb.rc()}/RUN_CLR_0.done"))["evidence"]; check(ev["ckpt_sha256"] == hashlib.sha256(b"ckpt\x00").hexdigest() and ev["argv_match_pin"] is True and ev["level_head"] == "on" and ev["recover_n"] == 19208, f"evidence {ev}")
    for label, over, rc_, msg in (("trainer rc", {}, "3", "trainer exited rc 3"), ("steps", {"steps": 2999}, "0", "summary steps 2999"), ("level head off in the summary", {"level_head": "off"}, "0", "summary level_head 'off'"),
                                  ("w differs", {"w_level": 0.5}, "0", "w_level 0.5"), ("no sample perturbed", {"recover_n": 0}, "0", "KR1: arm CLR run perturbed no sample"), ("recover_n missing", {"recover_n": None}, "0", "KR1"), ("train sha", {"train_sha256": "2" * 64}, "0", "KP1"),
                                  ("argv.json differs", {"argv": ["--x", "1"]}, "0", "argv.json != the pinned")):
        s2 = pinned_sb("po2", tags=("CLR_1",)); stage_run(s2, "CLR_1", **over); rc, o = s2.run(["post", "CLR_1", rc_]); check(rc == 0 and msg in o and os.path.exists(f"{s2.rc()}/RUN_CLR_1.FAILED") and not os.path.exists(f"{s2.rc()}/RUN_CLR_1.done"), f"post {label}: rc {rc} {o[:300]}")
    s3 = pinned_sb("po3", tags=("CLR_1",)); rc, o = s3.run(["post", "CLR_1", "0"]); check(rc == 0 and "summary.json missing" in o and "ckpt_last.pt missing" in o, f"post with nothing: {o[:200]}")
    rc, o = sb.run(["finish"]); st = json.load(open(f"{sb.rc()}/CHAIN_CLR.done")); check(rc == 0 and st["runs"] == {"CLR_0": "done", "CLR_1": "none"}, f"finish: {st}")
    rc, o = sb.run(["fail", "CLR_1", "pin", "refused"]); rc, o = sb.run(["finish"]); st = json.load(open(f"{sb.rc()}/CHAIN_CLR.done")); check(st["runs"]["CLR_1"] == "FAILED" and any(f.startswith("CHAIN_CLR.done.prev_") for f in os.listdir(sb.rc())), "a second finish moves the first marker aside")
    rc, o = sb.run(["stopped", "safety", "stop"]); check(os.path.exists(f"{sb.rc()}/CHAIN_CLR.STOPPED"), f"stopped: {o}")
    check(sb.run(["nonsense"])[0] == 2, "unknown command must exit 2")


@test
def t_shell_static_waker_and_pull_scripts():
    sh = ["clr_lib.sh", "launch_clr.sh", "wake_clr.sh", "pull_clr.sh", "wake_pull_clr.sh"]
    for f in sh:
        rc, o, e = bash(["-n", f"{PKG}/{f}"], {}); check(rc == 0, f"bash -n {f}: {e}")
        raw = rd(f"{PKG}/{f}", "rb"); check(b"\r" not in raw and raw.startswith(b"#!/usr/bin/env bash\n"), f"{f}: CR or shebang")
    for f in [x for x in sh if x != "clr_lib.sh"]:
        code = "\n".join(l for l in rd(f"{PKG}/{f}").splitlines() if not l.lstrip().startswith("#"))
        check(re.search(r'^: "\$\{CLR_PI_GO:\?[^}]+\}"$', code, re.M) is not None, f"{f}: no CLR_PI_GO refusal")
        first_remote = min([code.find(x) for x in ("ssh ", "$SSH", "$SCP", "scp ", "clr_launch", ". \"$(dirname", "bash wake_clr", "bash pull_") if code.find(x) >= 0] or [10 ** 9])
        check(code.find("CLR_PI_GO") < first_remote, f"{f}: the go check is not before the first remote or sourcing step")
    w = rd(f"{PKG}/wake_clr.sh"); code = "\n".join(l for l in w.splitlines() if not l.lstrip().startswith("#"))
    check('[ -e "$MIR/CHAIN_CLR.done" ] || [ -e "$MIR/CHAIN_CLR.STOPPED" ]' in code and not re.search(r"(^|[;&|(]\s*|\s)ls\s+[^\s|;&]+\s+[^\s|;&]+", code.replace("ls=$(", "")), "the waker's end test")
    fb = f"{PYW} -I {WORK}/fake_listing.py"; write(f"{WORK}/fake_listing.py", "import os,sys\nsys.stdout.write(open(os.environ['CL_FAKE_LISTING']).read() if os.path.exists(os.environ['CL_FAKE_LISTING']) else '')\n")
    lst = WORK + "/listing.txt"
    cases = [("chain", "", {}, 5, "WAKER TIMEOUT"), ("chain", "WPROBE.done", {}, 5, "WAKER TIMEOUT"), ("chain", "CHAIN_CLR.done\nRUN_CLR_0.done", {}, 0, "CHAIN DONE"), ("chain", "CHAIN_CLR.STOPPED", {}, 4, "CHAIN STOPPED"),
             ("chain", "GUARD_STOP.txt", {}, 4, "GUARD STOP"), ("chain", "RUN_CLR_0.FAILED", {}, 3, "RUN FAILED"), ("chain", "RUN_CLR_0.FAILED", {"CLR_SEEN_FAILED": "1"}, 5, "WAKER TIMEOUT"),
             ("chain", "RUN_CLR_0.done", {}, 0, "NEW DONE (1 of 2)"), ("chain", "RUN_CLR_0.done", {"CLR_SEEN": "1"}, 5, "WAKER TIMEOUT"), ("chain", "RUN_CLR_0.done\nRUN_CLR_1.done", {"CLR_SEEN": "1"}, 0, "NEW DONE (2 of 2)"),
             ("chain", "RUN_CLR_0.FAILED.prev_20261105_010101", {}, 5, "WAKER TIMEOUT"), ("chain", "RUN_RPT_0.done", {}, 5, "WAKER TIMEOUT")]
    for mode, listing, extra, ec, msg in cases:
        write(lst, listing + "\n"); rc, o, e = bash([f"{PKG}/wake_clr.sh", mode], {"CLR_PI_GO": "10:00", "CLR_SSH": fb, "CL_FAKE_LISTING": lst, "CLR_WAKE_N": "2", "CLR_SLEEP": "0", **extra})
        check(rc == ec and msg in o, f"waker {mode} listing {listing!r} {extra}: rc {rc} (want {ec}) out {o!r} err {e[:200]}")
    rc, o, e = bash([f"{PKG}/wake_clr.sh", "wprobe"], {"CLR_PI_GO": "10:00", "CLR_SSH": fb, "CL_FAKE_LISTING": lst}); check(rc == 64 and "usage" in o, f"wake_clr.sh takes chain only: rc {rc} {o}")
    # pulls against a local stand-in for scp (copies from a fake box root)
    write(f"{WORK}/fake_scp.py", "import os,shutil,sys,glob\nroot=os.environ['CL_FAKE_ROOT']; a=[x for x in sys.argv[1:] if x not in ('-q','-r') and not x.startswith(('BatchMode','ConnectTimeout')) and x!='-o']; rec='-r' in sys.argv\n"
          "src=a[0].split(':',1)[1].replace('D:/tanitad/runs/v8f_mini_full',root); dst=a[1]\nfs=glob.glob(src) if '*' in src else [src]\nif not fs or not all(os.path.exists(f) for f in fs): sys.exit(1)\n"
          "for f in fs:\n    (shutil.copytree(f, os.path.join(dst, os.path.basename(f)), dirs_exist_ok=True) if os.path.isdir(f) else shutil.copy(f, dst))\n")
    scp = f"{PYW} -I {WORK}/fake_scp.py"; root = WORK + "/fakebox"; shutil.rmtree(root, ignore_errors=True)
    for t in ("CLR_0", "CLR_1"):
        s = FRZ.INFO[t][1]; run = f"M2b_s{s}_recft_CLR"; ck = b"ckpt-pull-" + t.encode()
        for f, c in (("ckpt_last.pt", ck), ("summary.json", b"{}"), ("argv.json", b"{}"), ("selfcheck.json", b"{}"), ("train_log.jsonl", b""), ("train.log", b"")): write(f"{root}/{run}/{f}", c, "wb")
        write(f"{root}/gate_clr/{run}/selfcheck.json", "{}"); write(f"{root}/clr/RUN_{t}.done", json.dumps({"evidence": {"ckpt_sha256": hashlib.sha256(ck).hexdigest()}})); write(f"{root}/clr/logs/{t}.gate.log", "x")
    lock = WORK + "/lock/BOX2_BUSY.txt"; env = {"CLR_PI_GO": "10:00", "CLR_SCP": scp, "CL_FAKE_ROOT": root, "CLR_PY": PYW, "CLR_LOCK": lock}
    dest = WORK + "/pulldest"; shutil.rmtree(dest, ignore_errors=True); write(lock, "LEVELHEAD CLR two-run chain (D:\\tanitad\\cl\\clr_chain.cmd), owner: x")
    rc, o, e = bash([f"{PKG}/pull_clr.sh"], {**env, "CLR_DEST": dest}); check(rc == 0 and o.count("OK CLR_") == 2 and "bad=0" in o and "lock released" in o and not os.path.exists(lock), f"pull_clr happy: rc {rc} {o} {e[:300]}")
    check(os.path.exists(f"{dest}/M2b_s1_recft_CLR/SHA256SUMS.txt") and os.path.exists(f"{dest}/clr_markers/RUN_CLR_0.done") and any(f.startswith("BOX2_BUSY.clr_chain_done_") for f in os.listdir(os.path.dirname(lock))), "pulled layout / lock moved aside, not deleted")
    write(lock, "LEVELHEAD CLR two-run chain (...)"); write(f"{root}/M2b_s0_recft_CLR/ckpt_last.pt", b"corrupt", "wb"); os.remove(f"{root}/clr/RUN_CLR_1.done")
    rc, o, e = bash([f"{PKG}/pull_clr.sh"], {**env, "CLR_DEST": WORK + "/pulldest2"}); check(rc == 1 and "MISMATCH CLR_0" in o and "SKIP CLR_1" in o and "bad=1" in o and os.path.exists(lock) and "lock released" not in o, f"pull_clr bad path keeps the lock: rc {rc} {o}")
    write(lock, "somebody else's job"); shutil.rmtree(root, ignore_errors=True)
    for t in ("CLR_0", "CLR_1"):
        s = FRZ.INFO[t][1]; run = f"M2b_s{s}_recft_CLR"; ck = b"c" + t.encode()
        write(f"{root}/{run}/ckpt_last.pt", ck, "wb"); write(f"{root}/clr/RUN_{t}.done", json.dumps({"evidence": {"ckpt_sha256": hashlib.sha256(ck).hexdigest()}}))
    rc, o, e = bash([f"{PKG}/pull_clr.sh"], {**env, "CLR_DEST": WORK + "/pulldest3"}); check(rc == 0 and os.path.exists(lock) and "lock not released" in o, f"a lock that is not the chain's must be left alone: rc {rc} {o[-200:]}")
    check(not os.path.exists(WORK + "/tripwire.txt"), "a network tool was invoked by a shell script: " + (rd(WORK + "/tripwire.txt") if os.path.exists(WORK + "/tripwire.txt") else ""))


@test
def t_wake_pull_chain_on_stand_ins():
    pkg = f"{FROZEN}/clr_launch"; write(f"{WORK}/wp_listing.py", "import os,sys\nsys.stdout.write(open(os.environ['CL_FAKE_LISTING']).read() if os.path.exists(os.environ['CL_FAKE_LISTING']) else '')\n")
    write(f"{WORK}/wp_scp.py", "import os,shutil,sys,glob\nroot=os.environ['CL_FAKE_ROOT']; a=[x for x in sys.argv[1:] if x not in ('-q','-r') and not x.startswith(('BatchMode','ConnectTimeout')) and x!='-o']\n"
          "src=a[0].split(':',1)[1].replace('D:/tanitad/runs/v8f_mini_full',root); dst=a[1]\nfs=glob.glob(src) if '*' in src else [src]\nif not fs or not all(os.path.exists(f) for f in fs): sys.exit(1)\n"
          "for f in fs:\n    (shutil.copytree(f, os.path.join(dst, os.path.basename(f)), dirs_exist_ok=True) if os.path.isdir(f) else shutil.copy(f, dst))\n")
    root = WORK + "/wpbox"; shutil.rmtree(root, ignore_errors=True)
    for t in ("CLR_0", "CLR_1"):
        s = FRZ.INFO[t][1]; run = f"M2b_s{s}_recft_CLR"; ck = b"ckpt-wp-" + t.encode()
        for f, c in (("ckpt_last.pt", ck), ("summary.json", b"{}"), ("argv.json", b"{}"), ("selfcheck.json", b"{}"), ("train_log.jsonl", b""), ("train.log", b"")): write(f"{root}/{run}/{f}", c, "wb")
        write(f"{root}/clr/RUN_{t}.done", json.dumps({"evidence": {"ckpt_sha256": hashlib.sha256(ck).hexdigest()}}))
    lock = WORK + "/wp_lock/BOX2_BUSY.txt"; lst = WORK + "/wp_listing.txt"
    env = {"CLR_PI_GO": "10:00", "CLR_SSH": f"{PYW} -I {WORK}/wp_listing.py", "CLR_SCP": f"{PYW} -I {WORK}/wp_scp.py", "CL_FAKE_LISTING": lst, "CL_FAKE_ROOT": root, "CLR_PY": PYW, "CLR_LOCK": lock, "CLR_WAKE_N": "3", "CLR_SLEEP": "0"}
    write(lst, "RUN_CLR_0.done\nRUN_CLR_1.done\nCHAIN_CLR.done\n"); write(lock, "LEVELHEAD CLR two-run chain (D:\\tanitad\\clr\\clr_chain.cmd), owner: x"); dest = WORK + "/wp_dest"; shutil.rmtree(dest, ignore_errors=True)
    rc, o, e = bash([f"{pkg}/wake_pull_clr.sh", "chain"], {**env, "CLR_DEST": dest}); check(rc == 0 and "CHAIN DONE" in o and "pull rc=0" in o and not os.path.exists(lock), f"wake_pull chain end to end: rc {rc} {o} {e[:300]}")
    check(os.path.exists(f"{dest}/M2b_s0_recft_CLR/ckpt_last.pt") and os.path.exists(f"{dest}/clr_markers/RUN_CLR_1.done"), "the pull landed in M2b_s{0,1}_recft_CLR and clr_markers")
    shutil.rmtree(dest, ignore_errors=True); write(lock, "x"); write(lst, "RUN_CLR_0.done\nRUN_CLR_1.FAILED\n")
    rc, o, e = bash([f"{pkg}/wake_pull_clr.sh", "chain"], {**env, "CLR_DEST": dest}); check(rc == 3 and "RUN FAILED" in o and not os.path.exists(dest) and os.path.exists(lock), f"a failed run exits 3 and pulls nothing: rc {rc} {o}")
    write(lst, "CHAIN_CLR.STOPPED\n"); rc, o, e = bash([f"{pkg}/wake_pull_clr.sh", "chain"], {**env, "CLR_DEST": dest}); check(rc == 4 and not os.path.exists(dest), f"a safety stop exits 4: rc {rc} {o}")
    write(lst, ""); rc, o, e = bash([f"{pkg}/wake_pull_clr.sh", "chain"], {**env, "CLR_DEST": dest}); check(rc == 5, f"nothing on box 2: waker timeout rc 5, got {rc} {o}")
    rc, o, e = bash([f"{pkg}/wake_pull_clr.sh", "wprobe"], env); check(rc == 64, f"wake_pull_clr.sh takes chain only: rc {rc}")
    check(not os.path.exists(WORK + "/tripwire.txt"), "tripwire")


@test
def t_nothing_contacts_box2_without_the_pi_go():
    calls = WORK + "/nogo_calls.log"; shutil.rmtree(calls, ignore_errors=True)
    if os.path.exists(calls): os.remove(calls)
    stub = f"{PYW} -I {WORK}/stub_ssh.py"; write(f"{WORK}/stub_ssh.py", "import os,sys\nopen(os.environ['CL_STUB_LOG'],'a').write(' '.join(sys.argv[1:])+'\\n')\nsys.exit(0)\n")
    lock = WORK + "/nogo_lock/BOX2_BUSY.txt"
    for f, args in (("launch_clr.sh", []), ("wake_clr.sh", ["chain"]), ("pull_clr.sh", []), ("wake_pull_clr.sh", ["chain"])):
        for pkg in (PKG, f"{FROZEN}/clr_launch"):                                          # the real package and the frozen copy (whose chain would otherwise be allowed to launch)
            rc, o, e = bash([f"{pkg}/{f}"] + args, {"CLR_SSH": stub, "CLR_SCP": stub, "CL_STUB_LOG": calls, "CLR_LOCK": lock, "CLR_PY": PYW, "CLR_WAKE_N": "1", "CLR_SLEEP": "0", "CLR_DEST": WORK + "/nogo_dest"}, timeout=40, drop=("CLR_PI_GO",))
            check(rc != 0 and "CLR_PI_GO" in e and not os.path.exists(calls) and not os.path.exists(WORK + "/tripwire.txt") and not os.path.exists(lock), f"{f} {args} ran without CLR_PI_GO: rc {rc} out {o[:200]} err {e[:300]}")
    # and WITH the go but with the box-2 lock held / W not frozen: still nothing remote
    write(lock, "somebody's job\n")
    rc, o, e = bash([f"{FROZEN}/clr_launch/launch_clr.sh"], {"CLR_PI_GO": "10:00", "CLR_SSH": stub, "CLR_SCP": stub, "CL_STUB_LOG": calls, "CLR_LOCK": lock, "CLR_PY": PYW}, timeout=40)
    check(rc == 5 and "ZZCLR-LOCK-HELD" in o and not os.path.exists(calls) and rd(lock) == "somebody's job\n", f"a held lock must refuse without any remote call: rc {rc} {o[:300]}")
    os.remove(lock)
    r0 = build_frozen("nogo_card"); write(f"{r0}/a2_prereg/PREREG_LEVELHEAD_A2.md", rd(f"{r0}/a2_prereg/PREREG_LEVELHEAD_A2.md") + "\nedited\n")
    rc, o, e = bash([f"{r0}/clr_launch/launch_clr.sh"], {"CLR_PI_GO": "10:00", "CLR_SSH": stub, "CLR_SCP": stub, "CL_STUB_LOG": calls, "CLR_LOCK": lock, "CLR_PY": PYW}, timeout=40)
    check(rc == 6 and "ZZCLR-REFUSED" in o and "card on disk" in o and not os.path.exists(calls) and not os.path.exists(lock), f"an edited card must be refused locally: rc {rc} {o[:300]}")
    r0 = build_frozen("nogo_w"); os.remove(f"{r0}/cl_prereg/FROZEN_W.txt")
    rc, o, e = bash([f"{r0}/clr_launch/launch_clr.sh"], {"CLR_PI_GO": "10:00", "CLR_SSH": stub, "CLR_SCP": stub, "CL_STUB_LOG": calls, "CLR_LOCK": lock, "CLR_PY": PYW}, timeout=40)
    check(rc == 6 and "ZZCLR-REFUSED" in o and "FROZEN_W.txt" in o and not os.path.exists(calls) and not os.path.exists(lock), f"a missing frozen W must be refused locally: rc {rc} {o[:300]}")
    check(not os.path.exists(WORK + "/tripwire.txt"), "tripwire")


FAKE_BOX = r'''import hashlib, os, re, shutil, sys
ROOT = os.environ["FAKE_BOX_ROOT"]
def loc(p):
    p = p.replace("D:\\", "D:/")
    for a, b in (("D:/tanitad/runs/v8f_mini_full/clr", ROOT + "/markers"), ("D:/tanitad/clr", ROOT + "/cl"), ("D:/tanitad/repo", ROOT + "/repo"), ("D:/tanitad/logs", ROOT + "/logs")):
        if p.startswith(a): return b + p[len(a):]
    raise SystemExit("UNMAPPED " + p)
def log(s): open(ROOT + "/calls.log", "a").write(s + "\n")
def main(a):
    env = os.environ
    if a[0] == "scp":
        src, dst = a[-2], loc(a[-1].split(":", 1)[1])
        if env.get("FAKE_SCP_FAIL") and env["FAKE_SCP_FAIL"] in dst: return 1
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(src, dst); log("scp " + dst[len(ROOT):])
        if env.get("FAKE_SCP_CORRUPT") and env["FAKE_SCP_CORRUPT"] in dst: open(dst, "ab").write(b"#")
        return 0
    cmd = a[-1]
    if "Get-Process python" in cmd:
        mk = os.listdir(ROOT + "/markers") if os.path.isdir(ROOT + "/markers") else []
        print(env.get("FAKE_PY", "0")); print(env.get("FAKE_DFREE", "500"))
        for m in mk: print(m)
        return 0
    m = re.fullmatch(r'powershell -NoProfile -Command "Move-Item (\S+) (\S+)"', cmd)
    if m: os.rename(loc(m.group(1)), loc(m.group(2))); log("move " + m.group(1)); return 0
    if "New-Item -ItemType Directory" in cmd:
        os.makedirs(ROOT + "/cl/argv", exist_ok=True); return 0
    if cmd == "powershell -NoProfile -Command -":
        script = sys.stdin.read(); assert "Get-ChildItem D:\\tanitad\\clr -Recurse" in script, script
        for dp, _, fs in os.walk(ROOT + "/cl"):
            for f in fs: print(hashlib.sha256(open(os.path.join(dp, f), "rb").read()).hexdigest() + "  " + f)
        return 0
    m = re.fullmatch(r'powershell -NoProfile -Command "if \(Test-Path (\S+)\) \{ Move-Item (\S+) (\S+) \}"', cmd)
    if m:
        if env.get("FAKE_MOVE_FAIL") and env["FAKE_MOVE_FAIL"] in m.group(1): return 1
        if os.path.exists(loc(m.group(1))): os.rename(loc(m.group(2)), loc(m.group(3)))
        return 0
    m = re.fullmatch(r'powershell -NoProfile -Command "if \(Test-Path (\S+)\) \{ Move-Item (\S+) (\S+) \}; if \(Test-Path (\S+)\) \{ Move-Item (\S+) (\S+) \}"', cmd)
    if m:
        if os.path.exists(loc(m.group(1))): os.rename(loc(m.group(2)), loc(m.group(3)))
        if os.path.exists(loc(m.group(4))): os.rename(loc(m.group(5)), loc(m.group(6)))
        log("restore"); return 0
    m = re.fullmatch(r'powershell -NoProfile -Command "\(Get-FileHash (\S+) -Algorithm SHA256\)\.Hash\.ToLower\(\)"', cmd)
    if m: print(hashlib.sha256(open(loc(m.group(1)), "rb").read()).hexdigest()); return 0
    m = re.search(r"launch\.ps1 -Job (\S+) -Log (\S+)", cmd)
    if m:
        job = m.group(1).replace("\\", "/").split("/")[-1]; log("job " + job); os.makedirs(ROOT + "/logs", exist_ok=True)
        if job.endswith("clr_run_telemetry.cmd"): open(ROOT + "/logs/telemetry_full.csv", "w").write("")
        elif job.endswith("clr_chain.cmd"):
            open(loc(m.group(2).replace("\\", "/")), "w").write(env.get("FAKE_JOB_LOG", "===== CLR CHAIN START x") + chr(10))
        return 0
    if "telemetry_full.csv -Tail 1" in cmd:
        n = int(open(ROOT + "/tele.n").read()) if os.path.exists(ROOT + "/tele.n") else 0; open(ROOT + "/tele.n", "w").write(str(n + (0 if env.get("FAKE_TELE_FROZEN") else 1)))
        print("2026-11-05 12:00:%02d,x" % n); return 0
    if "Set-Content -Path D:/tanitad/logs/telemetry.stop" in cmd: log("telemetry.stop"); return 0
    m = re.search(r"Get-Content (\S+) -Tail 12", cmd)
    if m:
        pth = loc(m.group(1).replace("\\", "/")); print(open(pth).read() if os.path.exists(pth) else ""); return 0
    print("UNKNOWN REMOTE COMMAND: " + cmd, file=sys.stderr); return 97
sys.exit(main(sys.argv[1:]))
'''


@test
def t_launchers_dry_run_on_a_fake_box():
    write(f"{WORK}/fake_box_cl.py", FAKE_BOX)
    fbx = f"{PYW} -I {WORK}/fake_box_cl.py"; lock = WORK + "/dr_lock/BOX2_BUSY.txt"

    def run(script, name, env=None, pre=None, pkg=None, stack_old=True, ssh=None):
        root = f"{WORK}/dr_{name}"; shutil.rmtree(root, ignore_errors=True); os.makedirs(root + "/markers"); os.makedirs(os.path.dirname(lock), exist_ok=True)
        if os.path.exists(lock): os.remove(lock)
        if stack_old:
            for rel in FRZ.SHIPPED: write(f"{root}/repo/{rel}", f"OLD:{rel}\n")
        for nm in (pre or []): write(f"{root}/markers/{nm}", "{}")
        e = {"CLR_PI_GO": "10:00", "CLR_SSH": ssh or f"{fbx} ssh", "CLR_SCP": f"{fbx} scp", "CLR_LOCK": lock, "CLR_PY": PYW, "CLR_NAP": "0", "FAKE_BOX_ROOT": root, **(env or {})}
        rc, o, er = bash([f"{pkg or FROZEN + '/clr_launch'}/{script}"], e, timeout=300)
        return rc, o, er, root

    calls = lambda root: [l for l in rd(root + "/calls.log").splitlines()] if os.path.exists(root + "/calls.log") else []
    # happy chain: old CHAIN_CLR.done moved aside, files shipped and re-hashed, four stack files shipped with the old ones aside, telemetry -> guard -> chain, lock kept
    rc, o, er, root = run("launch_clr.sh", "chain", pre=["CHAIN_CLR.done", "RUN_CLR_0.done"]); c = calls(root)
    check(rc == 0 and "ZZCLR-LAUNCHED chain" in o and "lock taken" in o and "shipped bytes identical" in o and "old CHAIN_CLR.done moved aside" in o, f"happy chain: rc {rc} {o[-600:]} {er[:300]}")
    check(os.path.exists(lock) and rd(lock).startswith("LEVELHEAD CLR two-run chain") and "PI go 10:00" in rd(lock), "the lock was not left in place with the chain's text")
    check(os.path.exists(root + "/markers/RUN_CLR_0.done") and any(f.startswith("CHAIN_CLR.done.prev_") for f in os.listdir(root + "/markers")) and not os.path.exists(root + "/markers/CHAIN_CLR.done"), "only CHAIN_CLR.done is moved aside; a run marker stays")
    jobs = [x for x in c if x.startswith("job ")]; check(jobs == ["job clr_run_telemetry.cmd", "job clr_run_guard_chain.cmd", "job clr_chain.cmd"], f"order telemetry -> guard -> chain: {jobs}")
    for rel, want in FRZ.SHIPPED.items():
        check(sh256(f"{root}/repo/{rel}") == want and any(f.startswith(os.path.basename(rel) + ".pre_clr_") and rd(f"{root}/repo/{os.path.dirname(rel)}/{f}") == f"OLD:{rel}\n" for f in os.listdir(f"{root}/repo/{os.path.dirname(rel)}")), f"{rel}: not shipped as pinned with the old file aside")
    shipped = sorted(os.path.relpath(os.path.join(d, f), root + "/cl").replace("\\", "/") for d, _, fs in os.walk(root + "/cl") for f in fs)
    check(shipped == json.loads(json.dumps(sorted(FRZ_ship()))), f"shipped file set {shipped}")
    check(not os.path.exists(WORK + "/tripwire.txt"), "tripwire")
    rc, o, er, root = run("launch_clr.sh", "chain_real", pkg=PKG); check(rc == 0 and "ZZCLR-LAUNCHED chain" in o, f"the REAL package's chain launches on the fake box: rc {rc} {o[-400:]} {er[:200]}")
    # refusals, each leaves box 2 as it was and releases (moves aside) our lock
    def refused(label, name, msg, env=None, pre=None, rc_want=2, lock_gone=True, script="launch_clr.sh"):
        rc, o, er, root = run(script, name, env=env, pre=pre)
        check(rc == rc_want and msg in o, f"{label}: rc {rc} (want {rc_want}) want {msg!r}: {o[-500:]} {er[:200]}")
        check((not os.path.exists(lock)) == lock_gone, f"{label}: lock state (gone={not os.path.exists(lock)})")
        if lock_gone: check(any(f.startswith("BOX2_BUSY.clr_launch_failed_") for f in os.listdir(os.path.dirname(lock))), f"{label}: lock was not moved aside")
        for f in os.listdir(os.path.dirname(lock)):
            if f.startswith("BOX2_BUSY.clr_launch_failed_"): os.remove(os.path.dirname(lock) + "/" + f)
        return root
    root = refused("python running", "py", "a python process is running on box 2", env={"FAKE_PY": "1"}); check(not os.path.exists(root + "/cl") and not any("job" in x for x in calls(root)), "nothing shipped when python runs")
    root = refused("disk low", "disk", "box 2 D: below 8 GB", env={"FAKE_DFREE": "3"})
    root = refused("a heat stop marker", "gs", "GUARD_STOP.txt", pre=["GUARD_STOP.txt"]); check(not os.path.exists(root + "/cl"), "nothing shipped over a heat stop")
    root = refused("bytes differ on box 2", "corrupt", "shipped bytes differ", env={"FAKE_SCP_CORRUPT": "clr_chain.cmd"}); check(not any(x.startswith("job") for x in calls(root)), "no job after a byte mismatch")
    root = refused("stack scp corrupt", "stk", "on box 2 hashes", env={"FAKE_SCP_CORRUPT": "v8f_mini.py"})
    for rel in FRZ.SHIPPED: check(rd(f"{root}/repo/{rel}") == f"OLD:{rel}\n", f"{rel}: not restored after a failed stack ship")
    check(any(".clr_failed_" in f for rel in FRZ.SHIPPED for f in os.listdir(f"{root}/repo/{os.path.dirname(rel)}")), "the failed new bytes are kept aside")
    root = refused("stack aside move fails", "mv", "moving box 2's stack/tanitad/data/v8f_recover.py aside failed", env={"FAKE_MOVE_FAIL": "v8f_recover.py"})
    for rel in FRZ.SHIPPED: check(rd(f"{root}/repo/{rel}") == f"OLD:{rel}\n", f"{rel}: not restored")
    root = refused("stale telemetry", "tele", "no fresh telemetry", env={"FAKE_TELE_FROZEN": "1"}); check("telemetry.stop" in calls(root) and not any(x == "job clr_chain.cmd" for x in calls(root)), "a stale telemetry must stop the telemetry and not start the chain")
    refused("job stopped at start", "stop", "ZZCLR-JOB-STOPPED-AT-START", env={"FAKE_JOB_LOG": "ZZCLR-STOP preflight rc 3"}, rc_want=3, lock_gone=False); os.remove(lock)
    refused("no start line", "nostart", "ZZCLR-NO-START-LINE", env={"FAKE_JOB_LOG": "nothing"}, rc_want=4, lock_gone=False); os.remove(lock)
    # a held lock: refuse without any remote call (the lock is somebody else's and stays)
    os.makedirs(os.path.dirname(lock), exist_ok=True); write(lock, "someone else's job\n"); root = f"{WORK}/dr_held"; shutil.rmtree(root, ignore_errors=True); os.makedirs(root)
    rc, o, er = bash([f"{FROZEN}/clr_launch/launch_clr.sh"], {"CLR_PI_GO": "10:00", "CLR_SSH": f"{fbx} ssh", "CLR_SCP": f"{fbx} scp", "CLR_LOCK": lock, "CLR_PY": PYW, "FAKE_BOX_ROOT": root}, timeout=60)
    check(rc == 5 and not calls(root) and rd(lock) == "someone else's job\n", f"held lock: rc {rc} {o[:200]}")
    os.remove(lock)
    # fail closed: an unknown remote command is refused by the fake box (the launcher issues only the commands this test models)
    rc, o, er, root = run("launch_clr.sh", "unk", ssh=f"{PYW} -I -c \"import sys;sys.exit(97)\""); check(rc == 2 and "ZZCLR-ABORT" in o, f"an ssh that fails must abort: rc {rc} {o[-200:]}")
    check(not os.path.exists(WORK + "/tripwire.txt"), "tripwire")


def FRZ_ship():
    r = subprocess.run([sys.executable, "-I", f"{FROZEN}/clr_launch/clr_local_check.py", "ship-files"], capture_output=True, text=True); return [x.strip() for x in r.stdout.splitlines() if x.strip()]


PS_PARSE = "param($f) $t=$null;$e=$null;$null = [System.Management.Automation.Language.Parser]::ParseFile($f,[ref]$t,[ref]$e); if ($e.Count) { $e | ForEach-Object { $_.Message }; exit 1 }; Write-Output 'PARSE-OK'"


@test
def t_guard_scripts_parse_stop_on_heat_and_ignore_stale_markers():
    write(WORK + "/parse.ps1", PS_PARSE)
    for g in ("clr_guard_chain.ps1",):
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", WORK + "/parse.ps1", "-f", f"{PKG}/box2_files/{g}"], capture_output=True, text=True, timeout=90); check(r.returncode == 0 and "PARSE-OK" in r.stdout, f"{g} does not parse: {r.stdout}{r.stderr}")
    chain = rd(f"{PKG}/box2_files/clr_guard_chain.ps1")
    check("$ends = @('CHAIN_CLR.done', 'CHAIN_CLR.STOPPED')" in chain and "WPROBE" not in chain.split("$ends")[1].split("\n")[0], "the chain guard must end on the chain's markers only")

    def run_guard(src, name, temps, marker=None, marker_after=None, first=None):
        d = f"{WORK}/guard_{name}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d + "/rc")
        write(d + "/tel.csv", "ts,x\r\n" + "".join(f"2026-11-05 10:00:{i:02d},a,b,c,d,e,f,g,h,i,{t}\r\n" for i, t in enumerate(temps)))
        for f in (first or []): write(f"{d}/rc/{f}", "{}")
        s = src.replace(r"D:\tanitad\runs\v8f_mini_full\clr", (d + "/rc").replace("/", "\\")).replace(r"D:\tanitad\logs\telemetry_full.csv", (d + "/tel.csv").replace("/", "\\")).replace(r"D:\tanitad\logs\clr_guard.log", (d + "/guard.log").replace("/", "\\"))
        s = s.replace("v8f_mini_train", "zz_no_such_trainer_zz").replace("Start-Sleep -Seconds 60", "Start-Sleep -Seconds 1"); write(d + "/g.ps1", s)
        t0 = time.time(); p = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", d + "/g.ps1"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if marker_after: time.sleep(marker_after); write(f"{d}/rc/{marker}", "{}")
        try: out, _ = p.communicate(timeout=60)
        except subprocess.TimeoutExpired: p.kill(); raise AssertionError(f"guard {name} did not exit")
        return os.path.exists(d + "/rc/GUARD_STOP.txt"), (rd(d + "/guard.log") if os.path.exists(d + "/guard.log") else ""), out, time.time() - t0
    st, lg, o, _ = run_guard(chain, "hot4", [55] * 16 + [62] * 4); check(st and "ZZGUARD-STOP TEMP: 4 of 20 >= 61 C, max 62 C" in lg, f"4 of 20 >= 61 C must stop: {lg} {o}")
    st, lg, o, _ = run_guard(chain, "hot63", [50] * 19 + [63]); check(st and "max 63 C" in lg, f"one >= 63 C must stop: {lg} {o}")
    st, lg, o, _ = run_guard(chain, "warm3", [50] * 17 + [61] * 3, marker="CHAIN_CLR.done", marker_after=4); check(not st and "job ended; guard off (max USB 61 C)" in lg, f"3 of 20 must not stop; the chain marker ends the chain guard: {lg} {o}")
    st, lg, o, el = run_guard(chain, "stale", [50] * 20, marker="CHAIN_CLR.STOPPED", marker_after=5, first=["WPROBE.done", "WPROBE.FAILED"]); check(not st and el >= 4.5 and "job ended" in lg, f"a stale WPROBE marker must NOT end the chain guard (ended after {el:.1f}s): {lg}")


@test
def t_cmd_files_are_clones_with_the_rpt_safety():
    for f in sorted(os.listdir(f"{PKG}/box2_files")):
        p = f"{PKG}/box2_files/{f}"
        if not os.path.isfile(p): continue
        raw = rd(p, "rb")
        if f.endswith(".cmd"): check(raw.count(b"\r\n") == raw.count(b"\n") > 0 and raw.endswith(b"\r\n"), f"{f}: not CRLF")
        if f.endswith((".py", ".txt")): check(b"\r" not in raw, f"{f}: CR in an LF file")
        if f.endswith(".cmd"):
            t = raw.decode()
            check(not re.search("[\x09\x0b\x08\x0c]", t) and "D:/tanitad" not in t, f"{f}: a mangled backslash (a TAB / VT / BS / FF byte) or a forward-slash box-2 path")
            for l in t.split("\r\n"):
                if re.match(r"\s*set [A-Za-z]", l) and "/a" not in l and "/p" not in l: check(re.match(r'\s*set "[A-Za-z]+=[^"]*"\s*$', l), f"{f}: `set` without the quoted form: {l}")
    def norm(t):
        t = t.replace("ZZRNEXT-", "ZZCLR-").replace("RNEXT CHAIN", "CLR CHAIN").replace("CHAIN_RNEXT", "CHAIN_CLR").replace(r"D:\tanitad\rnext\rnext_check.py", r"D:\tanitad\clr\clr_check.py").replace(r"D:\tanitad\rnext\rnext_run.cmd", r"D:\tanitad\clr\clr_run.cmd")
        return t.replace(r"%RUNSM%\rnext", r"%RUNSM%\clr").replace("RPT_0 RPT_1", "CLR_0 CLR_1").replace("rnext_chain_rpt", "clr_chain").replace(r"D:\tanitad\rnext", r"D:\tanitad\clr")
    a = rd(f"{RPT_CHAIN}/rnext_chain_rpt.cmd", "rb").decode().split("\r\n"); b = rd(f"{PKG}/box2_files/clr_chain.cmd", "rb").decode().split("\r\n")
    code = lambda ls: [norm(l) for l in ls if not l.startswith("rem ")]
    ca, cb = code(a), code(b); ca = [l.replace('"%PY%" "%CHK%" preflight', '"%PY%" "%CHK%" preflight chain') if l == '"%PY%" "%CHK%" preflight' else l for l in ca]
    diff = [(x, y) for x, y in zip(ca, cb) if x != y]; check(len(ca) == len(cb) and not diff, f"clr_chain.cmd differs from the RPT chain (names normalised) beyond the preflight argument: {diff[:3]} {len(ca)} vs {len(cb)}")
    ra = [norm(l) for l in rd(f"{RPT_CHAIN}/rnext_run.cmd", "rb").decode().split("\r\n") if not l.startswith("rem ")]; rb = [l for l in rd(f"{PKG}/box2_files/clr_run.cmd", "rb").decode().split("\r\n") if not l.startswith("rem ")]
    ra = [l.replace("RUN-START", "RUN-START") for l in ra]; check(ra == rb, f"clr_run.cmd differs from rnext_run.cmd: {[(x, y) for x, y in zip(ra, rb) if x != y][:3]}")
    ch = rd(f"{PKG}/box2_files/clr_chain.cmd")
    check(ch.count("%RUNSAFE% cmd /c %RUNCMD%") == 2 and "call D:\\tanitad\\env.cmd" in rd(f"{PKG}/box2_files/clr_run.cmd"), "SAFE MODE: every python call goes through %RUNSAFE% (env.cmd sourced by the run file)")
    check("find /c \"python.exe\"" in ch and "ZZCLR-STOP a python process is already running" in ch and "telemetry.stop" in ch and "wprobe" not in ch.lower(), "one job at a time / telemetry stop")
    check('if exist "%RCD%\\GUARD_STOP.txt"' in ch and "exit /b 9" in ch and "preflight run %TAG%" in ch and "gate-cached" in ch and "gate %TAG% !RC!" in ch and "post %TAG% !RC!" in ch, "chain keeps the marker / guard / gate / post steps")
    tel = rd(f"{PKG}/box2_files/clr_run_telemetry.cmd", "rb").decode().split("\r\n"); tel0 = rd(f"{RPT_CHAIN}/rnext_run_telemetry.cmd", "rb").decode().split("\r\n")
    check([l for l in tel if not l.startswith("rem ")] == [l for l in tel0 if not l.startswith("rem ")], "the telemetry job differs from the RPT one")


@test
def t_package_hygiene():
    words = [r"\bvo" + "id\\b", "super" + "seded", "retr" + "acted", "for" + "merly", "no lon" + "ger", "corr" + "ection", "with" + "drawn", "closed" + r"[ -]loop"]
    mine = [f for f in os.listdir(PKG) if os.path.isfile(f"{PKG}/{f}") and f.endswith((".py", ".sh", ".md", ".json", ".txt"))] + [f"box2_files/{f}" for f in os.listdir(f"{PKG}/box2_files") if os.path.isfile(f"{PKG}/box2_files/{f}")] + ["tests/test_clr_launch.py"]
    for f in mine:
        raw = rd(f"{PKG}/{f}", "rb"); txt = raw.decode("utf-8")
        for w in words:
            ln = [l for l in txt.splitlines() if re.search(w, l, re.I)]; check(not ln, f"{f}: banned word ({w}): {ln[:1]}")
        if not f.endswith(".cmd"): check(b"\r" not in raw, f"{f}: CR in an LF file")
    for t in ("CLR_0", "CLR_1"):
        a = rd(f"{PKG}/box2_files/argv/{t}.train.txt", "rb"); check(a.count(b"\n") == 1 and a.endswith(b"\n") and b"\r" not in a, f"{t}: argv is not one LF line")
    for f in ("PINS.json",): json.loads(rd(f"{PKG}/{f}"))
    check(os.path.exists(f"{PKG}/HANDOFF_CLR.md"), "HANDOFF_CLR.md is missing")


@test
def t_no_cl_folder_paths_in_clr_scripts():
    """CLR runs in D:\tanitad\clr only. A CL folder path left by the clone (launch 2026-11-05 16:19: the chain was started as
    D:\tanitad\cl\clr_chain.cmd and cmd could not find it) must be caught here, because the fake box never checks paths."""
    import re
    bs = chr(92)
    pat = re.compile("tanitad(?:" + re.escape(bs) + "+|/)(?:runs(?:" + re.escape(bs) + "+|/)v8f_mini_full(?:" + re.escape(bs) + "+|/))?cl(?:" + re.escape(bs) + "+|/)")
    hits = []
    for root, _, files in os.walk(PKG):
        if "tests" in root.replace(bs, "/").split("/") or "logs" in root.replace(bs, "/").split("/"):
            continue
        for f in files:
            if not f.endswith((".sh", ".cmd", ".ps1", ".py")) or ".pre_" in f:
                continue
            for i, line in enumerate(open(os.path.join(root, f), encoding="utf-8", errors="replace"), 1):
                s = line.strip()
                if s.startswith(("#", "rem ", "REM ", "::")):
                    continue
                if pat.search(line):
                    hits.append(f"{f}:{i}")
    check(not hits, f"CL folder paths in CLR scripts: {hits}")


if __name__ == "__main__":
    only = sys.argv[1:]
    t0 = time.time()
    for fn in TESTS:
        if only and fn.__name__ not in only: continue
        t1 = time.time()
        try: fn(); print(f"PASS {fn.__name__} ({time.time() - t1:.1f}s)")
        except Exception as e:                              # noqa: BLE001
            FAILS.append(fn.__name__); print(f"FAIL {fn.__name__}: {type(e).__name__}: {str(e)[:1200]}")
    ran = len([f for f in TESTS if not only or f.__name__ in only])
    print(f"TESTS: {ran - len(FAILS)} passed, {len(FAILS)} failed ({time.time() - t0:.0f}s); netguard tripped: {netguard.TRIPPED}; work dir {WORK}")
    sys.exit(1 if FAILS or netguard.TRIPPED else 0)
