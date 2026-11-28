"""Dev-box tests of the arm-CL box-2 package (CPU only, NO network: tests/netguard.py raises before anything leaves this process; shell children get BASH_ENV functions that block ssh/scp/...; the launcher dry runs use a local fake box).
Run:  CUDA_VISIBLE_DEVICES=-1 python -I tests/test_cl_launch.py [test names]       (work dir: CL_TEST_WORK, else a temp dir). Exit 0 iff every test passed. Log: logs/test_cl_launch.log.
Two packages are exercised: the REAL one on disk (W is the placeholder: the state the MM receives) and a FROZEN copy built in the work dir with a stand-in W (never written to the real cl_prereg)."""
import copy, hashlib, importlib.util, json, os, re, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PKG = os.path.dirname(HERE)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, HERE)
import netguard  # noqa: E402
netguard.install()
WORK = (os.environ.get("CL_TEST_WORK") or tempfile.mkdtemp(prefix="cl_test_")).replace("\\", "/")
os.makedirs(WORK, exist_ok=True)
PYW = sys.executable.replace("\\", "/")
C0 = "E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s{s}_recft_C0"
RPT_CHAIN = "E:/Projects/TanitAD-artifacts/dagland/recover_next/box2_files"
TESTS, FAILS = [], []
STAND_W = "0.7312"


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


# ---- the frozen stand-in package (a copy; W = STAND_W; its own cl_prereg) ---------------------------------------------------------------------------------------------------------------------------------------------------
def build_frozen(name="pk", w=STAND_W, card_extra=None, probe=None):
    root = f"{WORK}/{name}"; shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(PKG, f"{root}/cl_launch", ignore=shutil.ignore_patterns("__pycache__", "logs", "tests"))
    os.makedirs(f"{root}/cl_prereg")
    card = rd(f"{PKG}/../cl_prereg/PREREG_LEVELHEAD_CL.md") + (card_extra if card_extra is not None else f"\n**FROZEN W = {w}** (test stand-in)\n")
    write(f"{root}/cl_prereg/PREREG_LEVELHEAD_CL.md", card)
    write(f"{root}/cl_prereg/FROZEN_W.txt", f"W={w}\n# test stand-in\n")
    write(f"{root}/cl_prereg/W_PROBE.json", json.dumps(probe if probe is not None else {"w_level_for_10pct_share": float(w) * 1.0003}))
    r = subprocess.run([sys.executable, "-I", f"{root}/cl_launch/mk_cl_package.py"], capture_output=True, text=True, env={**os.environ, "CL_PREREG_DIR": f"{root}/cl_prereg"})
    check(r.returncode == 0, f"generator in the frozen copy: {r.stdout} {r.stderr}")
    return root


def bash(args, extra, timeout=120, drop=()):
    exe = "C:/Program Files/Git/bin/bash.exe"; exe = exe if os.path.exists(exe) else "bash"
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update({"BASH_ENV": HERE + "/blockenv.sh", "CL_TRIPWIRE": WORK + "/tripwire.txt", **extra})
    r = subprocess.run([exe] + args, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.replace("\r", ""), r.stderr


FROZEN = build_frozen()
REAL_CK = loadmod("cl_check_real", f"{PKG}/box2_files/cl_check.py", CL_HOME=f"{PKG}/box2_files")
FRZ_CK_PATH = f"{FROZEN}/cl_launch/box2_files/cl_check.py"
FRZ = loadmod("cl_check_frz", FRZ_CK_PATH, CL_HOME=f"{FROZEN}/cl_launch/box2_files")
BOOT = "import sys,runpy; sys.path.insert(0,%r); import netguard; netguard.install(); sys.argv=%r; runpy.run_path(%r, run_name='__main__')"


class SB:
    """A box-2 sandbox for the checker: HOME (shipped argv + FROZEN_W), RUNSM, a repo holding the four shipped stack files, init ckpts, a disk floor."""
    def __init__(self, name, pkg=None, floor="0", mutate_stack=None):
        pkg = pkg or f"{FROZEN}/cl_launch"; self.pkg = pkg; self.root = f"{WORK}/{name}"; shutil.rmtree(self.root, ignore_errors=True)
        self.home, self.runsm, self.repo, self.disk, self.floor = self.root + "/home", self.root + "/runs", self.root + "/repo", self.root, floor
        shutil.copytree(f"{pkg}/box2_files", self.home, ignore=shutil.ignore_patterns("__pycache__"))
        self.check = f"{pkg}/box2_files/cl_check.py"
        for rel in FRZ.SHIPPED:
            write(f"{self.repo}/{rel}", rd(f"{pkg}/stack_ship/{rel}", "rb"), "wb")
        for s in (0, 1):
            write(f"{self.runsm}/M2b_s{s}_dagoacc/ckpt_last.pt", "x")

    def run(self, args):
        env = {**os.environ, "CL_HOME": self.home, "CL_RUNSM": self.runsm, "CL_REPO": self.repo, "CL_DISK": self.disk, "CL_FLOOR_BYTES": self.floor, "CUDA_VISIBLE_DEVICES": "-1"}
        r = subprocess.run([sys.executable, "-I", "-c", BOOT % (HERE, [self.check] + list(args), self.check)], env=env, capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    def rc(self): return self.runsm + "/cl"
    def gdir(self, tag): return f"{self.runsm}/M2b_s0_wprobe_CL" if tag == "WPROBE_0" else f"{self.runsm}/gate_cl/M2b_s{FRZ.INFO[tag][1]}_recft_CL"
    def odir(self, tag): return f"{self.runsm}/M2b_s{FRZ.INFO[tag][1]}_recft_CL"
    def argv(self, tag): return rd(f"{self.home}/argv/{tag}" + (".gate.txt" if tag == "WPROBE_0" else ".train.txt")).strip()
    def set_argv(self, tag, text): write(f"{self.home}/argv/{tag}" + (".gate.txt" if tag == "WPROBE_0" else ".train.txt"), text + "\n")


LVL6 = ["lvl_head.0.weight", "lvl_head.0.bias", "lvl_head.2.weight", "lvl_head.2.bias", "e_level.weight", "e_level.bias"]


def stub_sc(tag, w=STAND_W):
    s = FRZ.INFO[tag][1]; w = "1" if tag == "WPROBE_0" else w
    sc = json.load(open(C0.format(s=s) + "/selfcheck.json", encoding="utf-8"))          # the REAL C0 selfcheck.json is the base: every key the gate reads that C0 has is real
    ic = sc["info"]["init_ckpt"]; ic["fresh_keys"] = list(LVL6); ic["declared_new_keys"] = list(LVL6)
    sc["levers"].update({"level_head": "on", "w_level": float(w), "level_token_zero_init": True, "level_vision_only_source": True, "n_params_trainable": sc["levers"]["n_params_trainable"] + 66561})
    term = 0.3; base = 9.0 * float(STAND_W) * term                                         # W10 = (0.1/0.9) x base / term = STAND_W
    sc["level_head_probe"] = {"pass": True, "w_level_declared": float(w), "vision_only": {}, "step0_total": base + float(w) * term, "step0_level_term": term, "step0_total_without_level": base,
                              "w_level_for_10pct_share": (0.1 / 0.9) * base / term, "term": term, "total_w": base + float(w) * term + 0.2, "total_0": base + 0.2, "contribution": float(w) * term, "w_level_x_term": float(w) * term,
                              "head_grad_diff": 0.02, "head_grad_at_w0_detach_control": 0.0, "e_level_grad_at_w0": 0.01, "plan_moves_with_token": 0.1, "control_repeat_total_diff": 0.0, "control_repeat_grad_diff": 0.0,
                              "control_allfalse_term": 0.0, "control_allfalse_total_diff": 0.0, "control_allfalse_head_grad_diff": 0.0, "control_prior_grad_diff": 0.0}
    sc["argv_sha"] = "stub0000"
    return sc


def stage_gate(sb, tag, sc, log="SELFCHECK PASS {}"):
    write(f"{sb.gdir(tag)}/selfcheck.json", json.dumps(sc, indent=1)); write(f"{sb.rc()}/logs/{tag}.gate.log", log + "\n")


def pinned_sb(name, tags=("CL_0", "CL_1", "WPROBE_0"), **kw):
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
    tree = ast.parse(open(f"{PKG}/box2_files/cl_check.py", encoding="utf-8").read())
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    bad_attrs = attrs & {"system", "popen", "Popen", "run", "connect", "create_connection", "urlopen"}
    check(mods == {"hashlib", "json", "os", "shutil", "sys", "time"} and not bad_attrs, f"the checker imports {sorted(mods)} / calls {sorted(bad_attrs)}: stdlib file code only")
    check("tanitad-box2" not in open(f"{PKG}/box2_files/cl_check.py", encoding="utf-8").read(), "the checker names the remote")


@test
def t_generator_is_reproducible_and_pins_are_the_source_bytes():
    r = subprocess.run([sys.executable, "-I", f"{PKG}/mk_cl_package.py", "check"], capture_output=True, text=True); check(r.returncode == 0 and "OK" in r.stdout, f"generator check on the real package: {r.stdout} {r.stderr}")
    r = subprocess.run([sys.executable, "-I", f"{FROZEN}/cl_launch/mk_cl_package.py", "check"], capture_output=True, text=True, env={**os.environ, "CL_PREREG_DIR": f"{FROZEN}/cl_prereg"}); check(r.returncode == 0, f"frozen copy: {r.stdout}")
    pins = json.load(open(f"{PKG}/PINS.json")); check(pins == REAL_CK.PINS, "PINS.json != the pins embedded in the shipped checker")
    ch = {}; cur = None                                                                   # SHA_CHANGES.txt `new` for the two new files
    for l in open(f"{PKG}/../build/SHA_CHANGES.txt", encoding="utf-8"):
        m = re.match(r"^(stack/\S+)\s+\[", l); cur = m.group(1) if m else cur
        m = re.match(r"^\s+new ([0-9a-f]{64})$", l.rstrip())
        if m and cur: ch[cur] = m.group(1)
    rpt = {"stack/tanitad/data/v8f_recover.py": "0257ccb93251827fe5f1b3d7195999b5836145e8615cfd0d0a599262b8834b4c", "stack/tanitad/data/v8f_mini_data.py": "209da7a60765ba81268ee794aba187e21b0a6c73a49b9e98c86e2d876581ece6"}
    for x in pins["stack"]:
        got = sh256(f"{PKG}/{x['src']}"); check(got == x["sha256"], f"{x['rel']}: staged bytes {got} != pin")
        want = ch.get(x["rel"]) if x["origin"] == "new" else rpt[x["rel"]]; check(want == x["sha256"], f"{x['rel']}: pin {x['sha256'][:12]} != {'SHA_CHANGES.txt' if x['origin'] == 'new' else 'the RPT launcher pin'} {str(want)[:12]}")
    check(pins["stack"][0]["sha256"].startswith("9ffdf1b4") and pins["stack"][1]["sha256"].startswith("a3634276"), "trainer / model are not 9ffdf1b4 / a3634276")
    tip = open(f"{PKG}/tip_commit.txt").read().strip()                                    # the unchanged files equal the tip's blobs (local mirror, offline)
    for x in pins["stack"]:
        if x["origin"] == "tip":
            blob = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "cat-file", "blob", f"{tip}:{x['rel']}"], capture_output=True)
            check(blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == x["sha256"], f"{x['rel']}: the blob at tip {tip[:8]} is not the pin")
    for s in (0, 1):
        check(sh256(f"{PKG}/c0_argv/C0_s{s}.argv.json") == sh256(C0.format(s=s) + "/argv.json") == pins["c0_argv_sha256"][str(s)], f"c0_argv copy s{s} != C0's argv.json")


@test
def t_argv_differs_from_c0_only_by_out_and_the_two_lever_tokens():
    for s in (0, 1):
        c0 = json.load(open(C0.format(s=s) + "/argv.json", encoding="utf-8"))["argv"]
        cl = rd(f"{FROZEN}/cl_launch/box2_files/argv/CL_{s}.train.txt").strip().split(" "); pr = rd(f"{PKG}/box2_files/argv/CL_{s}.train.txt").strip().split(" ")
        i = c0.index("--out") + 1; exp = list(c0); exp[i] = exp[i].replace("_recft_C0", "_recft_CL")
        check(cl == exp + ["--level-head", "on", "--w-level", STAND_W], f"CL_{s} != C0 with --out replaced + the two lever tokens")
        check(sorted(t for t in cl if t not in c0) == sorted(["--level-head", "on", "--w-level", STAND_W, cl[i]]), "the tokens CL has and C0 lacks are the two levers and --out")
        check(cl[:i] == c0[:i] and cl[i + 1:len(c0)] == c0[i + 1:] and len(cl) == len(c0) + 4 and cl[i].endswith(f"M2b_s{s}_recft_CL"), "the tokens outside --out and the tail differ from C0's")
        check(pr[:-1] == cl[:-1] and pr[-1] == "__W__", "the placeholder-state argv is the frozen one with __W__ as the last token")
        wp = rd(f"{FROZEN}/cl_launch/box2_files/argv/WPROBE_0.gate.txt").strip().split(" ")
        if s == 0:
            j = wp.index("--out") + 1; e2 = list(cl[:len(c0)]) + ["--level-head", "on", "--w-level", "1", "--selfcheck-only"]; e2[j] = e2[j].replace("_recft_CL", "_wprobe_CL")
            check(wp == e2 and wp[j].endswith("M2b_s0_wprobe_CL"), "WPROBE_0 != CL_0 with --out wprobe, --w-level 1 and --selfcheck-only appended")
        for src, tag in ((cl, f"CL_{s}"),):                                               # the checker's own literal encoding agrees with the generator's tokens (two independent encodings)
            check(FRZ.expected_tokens(tag) == src and FRZ.audit_argv(tag, src) == [], f"{tag}: cl_check.expected_tokens != the generated argv: {FRZ.audit_argv(tag, src)}")
    check(FRZ.expected_tokens("WPROBE_0") == rd(f"{FROZEN}/cl_launch/box2_files/argv/WPROBE_0.gate.txt").strip().split(" "), "WPROBE_0 literal encoding")


@test
def t_placeholder_state_refuses_the_chain_and_allows_the_probe():
    check("__W__" in rd(f"{PKG}/box2_files/argv/CL_0.train.txt") and REAL_CK.W == "__W__", "the on-disk package is not in the placeholder state")
    sb = SB("ph", pkg=PKG)
    rc, o = sb.run(["preflight", "chain"]); check(rc == 3 and "W is '__W__'" in o and "placeholder" in o, f"preflight chain with __W__: rc {rc} {o[:300]}")
    rc, o = sb.run(["pin", "CL_0"]); check(rc == 2 and "REFUSED" in o and "__W__" in o, f"pin CL_0 with __W__: rc {rc} {o[:200]}")
    rc, o = sb.run(["preflight", "wprobe"]); check(rc == 0 and "preflight OK (wprobe)" in o, f"preflight wprobe in the placeholder state: rc {rc} {o[:300]}")
    r = subprocess.run([sys.executable, "-I", f"{PKG}/cl_local_check.py", "chain"], capture_output=True, text=True)
    check(r.returncode == 6 and "ZZCL-REFUSED" in r.stdout and "FROZEN_W.txt" in r.stdout, f"local check chain on the real package: rc {r.returncode} {r.stdout[:300]}")
    r = subprocess.run([sys.executable, "-I", f"{PKG}/cl_local_check.py", "wprobe"], capture_output=True, text=True); check(r.returncode == 0, f"local check wprobe: {r.stdout}")


@test
def t_local_check_chain_frozen_positive_and_refusals():
    def lc(root, mode="chain"): return subprocess.run([sys.executable, "-I", f"{root}/cl_launch/cl_local_check.py", mode], capture_output=True, text=True)
    r = lc(FROZEN); check(r.returncode == 0, f"frozen copy chain: {r.stdout}")
    cases = []
    r0 = build_frozen("lc1"); os.remove(f"{r0}/cl_prereg/W_PROBE.json"); cases.append((r0, "W_PROBE.json is absent"))
    r0 = build_frozen("lc2", probe={"w_level_for_10pct_share": 0.9}); cases.append((r0, "not within 1%"))
    r0 = build_frozen("lc3"); write(f"{r0}/cl_prereg/PREREG_LEVELHEAD_CL.md", rd(f"{r0}/cl_prereg/PREREG_LEVELHEAD_CL.md") + "\nedited after the freeze\n"); cases.append((r0, "card on disk is not the card the pins were made from"))
    r0 = build_frozen("lc4"); write(f"{r0}/cl_prereg/FROZEN_W.txt", "W=0.8\n"); cases.append((r0, "FROZEN_W.txt says '0.8' but the package pins"))
    r0 = build_frozen("lc5"); write(f"{r0}/cl_launch/box2_files/FROZEN_W.txt", "W=0.7312\n# other bytes\n"); cases.append((r0, "box2_files/FROZEN_W.txt is not the cl_prereg/FROZEN_W.txt"))
    r0 = build_frozen("lc6", card_extra="\nno freeze line here\n"); cases.append((r0, "does not name the frozen W"))
    r0 = build_frozen("lc7"); a = rd(f"{r0}/cl_launch/box2_files/argv/CL_1.train.txt").replace("--w-level 0.7312", "--w-level 0.9"); write(f"{r0}/cl_launch/box2_files/argv/CL_1.train.txt", a); cases.append((r0, "CL_1: argv audit"))
    r0 = build_frozen("lc8"); write(f"{r0}/cl_launch/stack_ship/stack/scripts/v8f_mini_train.py", "# edited\n" + rd(f"{r0}/cl_launch/stack_ship/stack/scripts/v8f_mini_train.py")); cases.append((r0, "stack source"))
    for root, msg in cases:
        r = lc(root); check(r.returncode == 6 and msg in r.stdout, f"{root}: rc {r.returncode}, want a refusal naming {msg!r}: {r.stdout[:400]}")


@test
def t_checker_preflight_pin_and_mutated_argv():
    sb = SB("pf")
    for mode in ("wprobe", "chain"):
        rc, o = sb.run(["preflight", mode]); check(rc == 0 and f"preflight OK ({mode})" in o, f"preflight {mode}: {o}")
    check(sb.run(["preflight", "run", "CL_0"])[0] == 0, "preflight run with the floor off")
    sbh = SB("pfh", floor="9e18")
    check(sbh.run(["preflight", "run", "CL_0"])[0] == 9 and sbh.run(["preflight", "chain"])[0] == 3, "a huge disk floor must stop (9) / fail preflight (3)")
    for rel in FRZ.SHIPPED:                                                               # every shipped stack file is bound: one mutated byte refuses the chain
        s2 = SB("pf_mut"); write(f"{s2.repo}/{rel}", "# x\n" + rd(f"{s2.repo}/{rel}")); rc, o = s2.run(["preflight", "chain"]); check(rc == 3 and rel in o and "sha256" in o, f"mutated {rel}: rc {rc} {o[:200]}")
    s3 = SB("pf_nofw"); os.remove(s3.home + "/FROZEN_W.txt"); rc, o = s3.run(["preflight", "chain"]); check(rc == 3 and "FROZEN_W.txt is not shipped" in o, f"no FROZEN_W: {o}")
    s4 = SB("pf_noic"); os.remove(s4.runsm + "/M2b_s1_dagoacc/ckpt_last.pt"); rc, o = s4.run(["preflight", "chain"]); check(rc == 3 and "init ckpt" in o and "missing" in o, f"no init ckpt: {o}")
    for t in ("CL_0", "CL_1", "WPROBE_0"):
        rc, o = sb.run(["pin", t]); check(rc == 0 and "pinned" in o, f"pin {t}: {o}")
        rc, o = sb.run(["pin", t]); check(rc == 0, f"pin {t} is not idempotent: {o}")
    check(rd(f"{sb.rc()}/pin/CL_0.gate.txt").strip().endswith("--selfcheck-only") and "gate_cl" in rd(f"{sb.rc()}/pin/CL_0.gate.txt"), "the CL gate argv is not the training argv with --out gate_cl and --selfcheck-only")
    check(rd(f"{sb.rc()}/pin/WPROBE_0.gate.txt").strip().endswith("--w-level 1 --selfcheck-only"), "the probe pin")
    base = sb.argv("CL_0")
    muts = [("drop the head", base.replace(" --level-head on", ""), "missing --level-head"), ("head off", base.replace("--level-head on", "--level-head off"), "--level-head = 'off'"),
            ("other W", base.replace("--w-level " + STAND_W, "--w-level 0.5"), "--w-level = '0.5'"), ("steps", base.replace("--steps 3000", "--steps 2000"), "--steps = '2000'"),
            ("a recovery dose", base + " --recover-p 0.3", "given 2 times"), ("an extra lever", base + " --launch-intent-w 1.0", "unexpected flag --launch-intent-w"),
            ("lr", base.replace("--lr 1e-4", "--lr 3e-4"), "--lr = '3e-4'"), ("wrong init", base.replace("M2b_s0_dagoacc", "M2b_s1_dagoacc"), "--init-ckpt"),
            ("order", base.replace("--level-head on --w-level " + STAND_W, "--w-level " + STAND_W + " --level-head on"), "token order differs"), ("placeholder", base.replace(STAND_W, "__W__"), "placeholder")]
    for label, text, msg in muts:
        s5 = SB("pf_argv"); s5.set_argv("CL_0", text); rc, o = s5.run(["pin", "CL_0"]); check(rc == 2 and "REFUSED" in o and msg in o, f"mutated argv ({label}): rc {rc}, want {msg!r}: {o[:300]}")
        rc, o = s5.run(["preflight", "chain"]); check(rc == 3 and "CL_0: argv audit" in o, f"preflight must refuse the {label} argv: {o[:200]}")
    s6 = pinned_sb("pf_repin", tags=("CL_0",)); s6.set_argv("CL_0", base.replace("--steps 3000", "--steps 3000").replace("--lr 1e-4", "--lr 1e-4")); write(f"{s6.rc()}/pin/CL_0.train.txt", "other\r\n")
    rc, o = s6.run(["pin", "CL_0"]); check(rc == 2 and "pins another train argv" in o, f"a differing pin file must be refused: {o}")
    wp = sb.argv("WPROBE_0"); s7 = SB("pf_wp"); s7.set_argv("WPROBE_0", wp.replace("--w-level 1 ", "--w-level 2 ")); rc, o = s7.run(["pin", "WPROBE_0"]); check(rc == 2 and "--w-level = '2'" in o, f"mutated probe argv: {o}")
    s8 = SB("pf_wp2"); s8.set_argv("WPROBE_0", wp.replace(" --selfcheck-only", "")); rc, o = s8.run(["pin", "WPROBE_0"]); check(rc == 2 and "missing --selfcheck-only" in o, f"probe argv without --selfcheck-only: {o}")


@test
def t_gate_pass_token_binding_and_cache():
    sb = pinned_sb("gp")
    for t in ("CL_0", "CL_1"):
        stage_gate(sb, t, stub_sc(t)); rc, o = sb.run(["gate", t, "0"]); check(rc == 0 and "ZZCL-GATE-PASS" in o, f"gate {t}: {o}")
        tok = json.load(open(f"{sb.rc()}/gate_token_{t}.json")); b = tok["binding"]
        check(tok["verdict"] == "PASS" and b["w_level"] == STAND_W and b["argv_sha256"] == FRZ.tokens_sha(sb.argv(t).split(" ")) and b["shipped_sha256"] == FRZ.SHIPPED and tok["prereg_sha256"] == FRZ.CARD_SHA
              and b["init_ckpt_sha256"] == FRZ.INIT_SHA[FRZ.INFO[t][1]] and abs(b["level_share_at_step0"] - 0.1) < 1e-3 and "never by hand" in tok["written_by"], f"token {t}: {tok}")
        rc, o = sb.run(["gate-cached", t]); check(rc == 0 and "YES" in o, f"cached {t}: {o}")
    p = f"{sb.gdir('CL_0')}/selfcheck.json"; txt = rd(p); write(p, txt.replace('"stub0000"', '"stub0001"')); check(sb.run(["gate-cached", "CL_0"])[0] == 1, "a changed selfcheck.json must not be cached"); write(p, txt)
    check(sb.run(["gate-cached", "CL_0"])[0] == 0, "restored selfcheck is cached again")
    rel = "stack/tanitad/models/v8f_mini.py"; write(f"{sb.repo}/{rel}", "# x\n" + rd(f"{sb.repo}/{rel}")); check(sb.run(["gate-cached", "CL_0"])[0] == 1, "a changed shipped model must not be cached")
    tk = f"{sb.rc()}/gate_token_CL_1.json"; j = json.load(open(tk)); j["binding"]["w_level"] = "0.5"; write(tk, json.dumps(j)); check(sb.run(["gate-cached", "CL_1"])[0] == 1, "a token for another W must not be cached")
    os.remove(f"{sb.rc()}/gate_token_CL_0.json"); check(sb.run(["gate-cached", "CL_0"])[0] == 1, "no token = not cached")


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
             ("recover on", lambda d: setp(d, ["levers", "recover", "recover_p"], 0.3), "recover must be off"),
             ("dagger on", lambda d: setp(d, ["levers", "dagger_p"], 0.1), "another lever is on"),
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
        sb = pinned_sb("gn", tags=("CL_0",)); sc = stub_sc("CL_0"); mut(sc); stage_gate(sb, "CL_0", sc)
        rc, o = sb.run(["gate", "CL_0", "0"]); check(rc == 1 and msg in o, f"{label}: rc {rc}, want {msg!r}: {o[:300]}")
        check(os.path.exists(f"{sb.rc()}/RUN_CL_0.FAILED") and not os.path.exists(f"{sb.rc()}/gate_token_CL_0.json"), f"{label}: FAILED marker / token state")
    sb = pinned_sb("gn2", tags=("CL_0",)); stage_gate(sb, "CL_0", stub_sc("CL_0")); rc, o = sb.run(["gate", "CL_0", "1"]); check(rc == 1 and "exited rc 1" in o, f"gate rc 1: {o}")
    sb = pinned_sb("gn3", tags=("CL_0",)); stage_gate(sb, "CL_0", stub_sc("CL_0"), log="x"); rc, o = sb.run(["gate", "CL_0", "0"]); check(rc == 1 and "'SELFCHECK PASS' not in the gate log" in o, f"no PASS line: {o}")
    sb = pinned_sb("gn4", tags=("CL_0",)); rc, o = sb.run(["gate", "CL_0", "0"]); check(rc == 1 and "no readable selfcheck.json" in o, f"no selfcheck: {o}")
    sb = pinned_sb("gn5", tags=("CL_1",)); sc = stub_sc("CL_1"); setp(sc, ["level_head_probe", "step0_level_term"], 0.5); setp(sc, ["level_head_probe", "w_level_for_10pct_share"], (0.1 / 0.9) * sc["level_head_probe"]["step0_total_without_level"] / 0.5)
    stage_gate(sb, "CL_1", sc); rc, o = sb.run(["gate", "CL_1", "0"]); check(rc == 0, f"seed 1 reads another batch: its share is recorded, not gated to a tenth: {o}")


@test
def t_wprobe_gate_writes_the_number():
    sb = pinned_sb("wp"); sc = stub_sc("WPROBE_0"); stage_gate(sb, "WPROBE_0", sc)
    rc, o = sb.run(["gate", "WPROBE_0", "0"]); check(rc == 0 and os.path.exists(f"{sb.rc()}/WPROBE.done"), f"wprobe gate: {o}")
    mk = json.load(open(f"{sb.rc()}/WPROBE.done")); check(abs(mk["w_level_for_10pct_share"] - float(STAND_W)) < 1e-9 and mk["evidence"]["selfcheck_sha256"] == sh256(f"{sb.gdir('WPROBE_0')}/selfcheck.json"), f"WPROBE.done: {mk}")
    check(not os.path.exists(f"{sb.rc()}/gate_token_WPROBE_0.json"), "the probe writes no launch-gate token")
    for label, mut, msg in (("key missing", lambda d: popp(d, ["level_head_probe", "w_level_for_10pct_share"]), "w_level_for_10pct_share"), ("negative", lambda d: setp(d, ["level_head_probe", "w_level_for_10pct_share"], -1.0), "w_level_for_10pct_share"),
                            ("W declared 2", lambda d: setp(d, ["level_head_probe", "w_level_declared"], 2.0), "w_level_declared"), ("not applied", lambda d: setp(d, ["info", "init_ckpt", "applied"], False), "init ckpt: applied")):
        s2 = pinned_sb("wp2", tags=("WPROBE_0",)); sc = stub_sc("WPROBE_0"); mut(sc); stage_gate(s2, "WPROBE_0", sc)
        rc, o = s2.run(["gate", "WPROBE_0", "0"]); check(rc == 1 and msg in o and os.path.exists(f"{s2.rc()}/WPROBE.FAILED") and not os.path.exists(f"{s2.rc()}/WPROBE.done"), f"{label}: rc {rc} {o[:300]}")
    s3 = pinned_sb("wp3", tags=("WPROBE_0",)); rc, o = s3.run(["wprobe-stopped", "python process present"]); check(rc == 0 and os.path.exists(f"{s3.rc()}/WPROBE.STOPPED"), f"wprobe-stopped: {o}")


@test
def t_post_finish_stopped_and_fail_markers():
    def stage_run(sb, t, **over):
        s = FRZ.INFO[t][1]; o = sb.odir(t); sm = {"steps": 3000, "train_sha256": FRZ.KP1["train_sha256"], "train_window_list_sha256": FRZ.KP1["train_window_list_sha256"], "level_head": "on", "w_level": float(STAND_W), "recover_n": 0}
        sm.update({k: v for k, v in over.items() if k != "argv"})
        write(f"{o}/summary.json", json.dumps(sm)); write(f"{o}/argv.json", json.dumps({"argv": over.get("argv", sb.argv(t).split(" "))})); write(f"{o}/ckpt_last.pt", b"ckpt" + bytes([s]), "wb")
    sb = pinned_sb("po")
    stage_run(sb, "CL_0"); rc, o = sb.run(["post", "CL_0", "0"]); check(rc == 0 and os.path.exists(f"{sb.rc()}/RUN_CL_0.done"), f"post: {o}")
    ev = json.load(open(f"{sb.rc()}/RUN_CL_0.done"))["evidence"]; check(ev["ckpt_sha256"] == hashlib.sha256(b"ckpt\x00").hexdigest() and ev["argv_match_pin"] is True and ev["level_head"] == "on", f"evidence {ev}")
    for label, over, rc_, msg in (("trainer rc", {}, "3", "trainer exited rc 3"), ("steps", {"steps": 2999}, "0", "summary steps 2999"), ("level head off in the summary", {"level_head": "off"}, "0", "summary level_head 'off'"),
                                  ("w differs", {"w_level": 0.5}, "0", "w_level 0.5"), ("a perturbed sample", {"recover_n": 4}, "0", "recover_n 4"), ("train sha", {"train_sha256": "2" * 64}, "0", "KP1"),
                                  ("argv.json differs", {"argv": ["--x", "1"]}, "0", "argv.json != the pinned")):
        s2 = pinned_sb("po2", tags=("CL_1",)); stage_run(s2, "CL_1", **over); rc, o = s2.run(["post", "CL_1", rc_]); check(rc == 0 and msg in o and os.path.exists(f"{s2.rc()}/RUN_CL_1.FAILED") and not os.path.exists(f"{s2.rc()}/RUN_CL_1.done"), f"post {label}: rc {rc} {o[:300]}")
    s3 = pinned_sb("po3", tags=("CL_1",)); rc, o = s3.run(["post", "CL_1", "0"]); check(rc == 0 and "summary.json missing" in o and "ckpt_last.pt missing" in o, f"post with nothing: {o[:200]}")
    rc, o = sb.run(["finish"]); st = json.load(open(f"{sb.rc()}/CHAIN_CL.done")); check(rc == 0 and st["runs"] == {"CL_0": "done", "CL_1": "none"}, f"finish: {st}")
    rc, o = sb.run(["fail", "CL_1", "pin", "refused"]); rc, o = sb.run(["finish"]); st = json.load(open(f"{sb.rc()}/CHAIN_CL.done")); check(st["runs"]["CL_1"] == "FAILED" and any(f.startswith("CHAIN_CL.done.prev_") for f in os.listdir(sb.rc())), "a second finish moves the first marker aside")
    rc, o = sb.run(["stopped", "safety", "stop"]); check(os.path.exists(f"{sb.rc()}/CHAIN_CL.STOPPED"), f"stopped: {o}")
    check(sb.run(["nonsense"])[0] == 2, "unknown command must exit 2")


@test
def t_w_probe_reader_and_freeze_helper():
    sb = pinned_sb("wr"); stage_gate(sb, "WPROBE_0", stub_sc("WPROBE_0")); check(sb.run(["gate", "WPROBE_0", "0"])[0] == 0, "stage")
    pulled = f"{WORK}/wr_pulled"; shutil.rmtree(pulled, ignore_errors=True); os.makedirs(pulled)
    shutil.copy(f"{sb.gdir('WPROBE_0')}/selfcheck.json", pulled); shutil.copy(f"{sb.rc()}/WPROBE.done", pulled)
    out = f"{WORK}/wr_out/W_PROBE.json"; r = lambda extra=(): subprocess.run([sys.executable, "-I", f"{PKG}/w_probe_read.py", pulled, "--out", out, *extra], capture_output=True, text=True)
    p = r(); check(p.returncode == 0 and abs(json.load(open(out))["w_level_for_10pct_share"] - float(STAND_W)) < 1e-9, f"reader: {p.stdout} {p.stderr}")
    j = json.load(open(out)); check(j["selfcheck_sha256"] == sh256(pulled + "/selfcheck.json") and "0.1/0.9" in j["formula"], f"W_PROBE.json fields {list(j)}")
    for label, mut, msg in (("tampered file", lambda d: d.update({"argv_sha": "tampered"}), "is not the one WPROBE.done recorded"), ("key missing", lambda d: d["level_head_probe"].pop("w_level_for_10pct_share"), "no level_head_probe.w_level_for_10pct_share"),
                            ("value inconsistent", lambda d: d["level_head_probe"].update({"w_level_for_10pct_share": 3.0}), "!= (0.1/0.9)"), ("probe failed", lambda d: d["level_head_probe"].update({"pass": False}), "level_head_probe.pass")):
        d = json.load(open(pulled + "/selfcheck.json")); mut(d); w2 = f"{WORK}/wr_bad"; shutil.rmtree(w2, ignore_errors=True); os.makedirs(w2); shutil.copy(pulled + "/WPROBE.done", w2); write(w2 + "/selfcheck.json", json.dumps(d))
        if label != "tampered file":                                                   # keep the marker's sha consistent so the check under test is the one that fires
            mk = json.load(open(w2 + "/WPROBE.done")); mk["evidence"]["selfcheck_sha256"] = sh256(w2 + "/selfcheck.json"); write(w2 + "/WPROBE.done", json.dumps(mk))
        o2 = f"{WORK}/wr_bad_out/W_PROBE.json"; shutil.rmtree(os.path.dirname(o2), ignore_errors=True)
        q = subprocess.run([sys.executable, "-I", f"{PKG}/w_probe_read.py", w2, "--out", o2], capture_output=True, text=True); check(q.returncode == 1 and msg in q.stdout and not os.path.exists(o2), f"{label}: rc {q.returncode} {q.stdout[:300]}")
    fz = f"{WORK}/fz"; shutil.rmtree(fz, ignore_errors=True); os.makedirs(fz + "/cl_launch"); os.makedirs(fz + "/cl_prereg"); shutil.copy(f"{PKG}/freeze_w.py", fz + "/cl_launch"); write(fz + "/cl_prereg/W_PROBE.json", json.dumps({"w_level_for_10pct_share": 0.731234567, "selfcheck_sha256": "ab"}))
    q = subprocess.run([sys.executable, "-I", fz + "/cl_launch/freeze_w.py"], capture_output=True, text=True); check(q.returncode == 0 and "W = 0.7312" in q.stdout and "FROZEN W = 0.7312" in q.stdout and not os.path.exists(fz + "/cl_prereg/FROZEN_W.txt"), f"proposal: {q.stdout}")
    q = subprocess.run([sys.executable, "-I", fz + "/cl_launch/freeze_w.py", "--write"], capture_output=True, text=True); check(q.returncode == 0 and rd(fz + "/cl_prereg/FROZEN_W.txt").startswith("W=0.7312\n"), f"--write: {q.stdout}")
    write(fz + "/cl_prereg/W_PROBE.json", json.dumps({"w_level_for_10pct_share": 0.9, "selfcheck_sha256": "ab"})); q = subprocess.run([sys.executable, "-I", fz + "/cl_launch/freeze_w.py", "--write"], capture_output=True, text=True); check(q.returncode == 1 and "not overwritten" in q.stdout, f"re-freeze must be refused: {q.stdout}")


@test
def t_shell_static_waker_and_pull_scripts():
    sh = ["cl_lib.sh", "launch_wprobe.sh", "launch_cl.sh", "wake_cl.sh", "pull_wprobe.sh", "pull_cl.sh", "wake_pull_cl.sh"]
    for f in sh:
        rc, o, e = bash(["-n", f"{PKG}/{f}"], {}); check(rc == 0, f"bash -n {f}: {e}")
        raw = rd(f"{PKG}/{f}", "rb"); check(b"\r" not in raw and raw.startswith(b"#!/usr/bin/env bash\n"), f"{f}: CR or shebang")
    for f in [x for x in sh if x != "cl_lib.sh"]:
        code = "\n".join(l for l in rd(f"{PKG}/{f}").splitlines() if not l.lstrip().startswith("#"))
        check(re.search(r'^: "\$\{CL_PI_GO:\?[^}]+\}"$', code, re.M) is not None, f"{f}: no CL_PI_GO refusal")
        first_remote = min([code.find(x) for x in ("ssh ", "$SSH", "$SCP", "scp ", "cl_launch", ". \"$(dirname", "bash wake_cl", "bash pull_") if code.find(x) >= 0] or [10 ** 9])
        check(code.find("CL_PI_GO") < first_remote, f"{f}: the go check is not before the first remote or sourcing step")
    w = rd(f"{PKG}/wake_cl.sh"); code = "\n".join(l for l in w.splitlines() if not l.lstrip().startswith("#"))
    check('[ -e "$MIR/CHAIN_CL.done" ] || [ -e "$MIR/CHAIN_CL.STOPPED" ]' in code and not re.search(r"(^|[;&|(]\s*|\s)ls\s+[^\s|;&]+\s+[^\s|;&]+", code.replace("ls=$(", "")), "the waker's end test")
    fb = f"{PYW} -I {WORK}/fake_listing.py"; write(f"{WORK}/fake_listing.py", "import os,sys\nsys.stdout.write(open(os.environ['CL_FAKE_LISTING']).read() if os.path.exists(os.environ['CL_FAKE_LISTING']) else '')\n")
    lst = WORK + "/listing.txt"
    cases = [("wprobe", "", {}, 5, "WAKER TIMEOUT"), ("wprobe", "WPROBE.done", {}, 0, "WPROBE DONE"), ("wprobe", "WPROBE.FAILED", {}, 3, "WPROBE FAILED"), ("wprobe", "WPROBE.STOPPED", {}, 4, "WPROBE STOPPED"),
             ("wprobe", "GUARD_STOP.txt", {}, 4, "GUARD STOP"), ("wprobe", "WPROBE.done.prev_20261105_010101", {}, 5, "WAKER TIMEOUT"), ("wprobe", "CHAIN_CL.done", {}, 5, "WAKER TIMEOUT"),
             ("chain", "", {}, 5, "WAKER TIMEOUT"), ("chain", "WPROBE.done", {}, 5, "WAKER TIMEOUT"), ("chain", "CHAIN_CL.done\nRUN_CL_0.done", {}, 0, "CHAIN DONE"), ("chain", "CHAIN_CL.STOPPED", {}, 4, "CHAIN STOPPED"),
             ("chain", "GUARD_STOP.txt", {}, 4, "GUARD STOP"), ("chain", "RUN_CL_0.FAILED", {}, 3, "RUN FAILED"), ("chain", "RUN_CL_0.FAILED", {"CL_SEEN_FAILED": "1"}, 5, "WAKER TIMEOUT"),
             ("chain", "RUN_CL_0.done", {}, 0, "NEW DONE (1 of 2)"), ("chain", "RUN_CL_0.done", {"CL_SEEN": "1"}, 5, "WAKER TIMEOUT"), ("chain", "RUN_CL_0.done\nRUN_CL_1.done", {"CL_SEEN": "1"}, 0, "NEW DONE (2 of 2)"),
             ("chain", "RUN_CL_0.FAILED.prev_20261105_010101", {}, 5, "WAKER TIMEOUT"), ("chain", "RUN_RPT_0.done", {}, 5, "WAKER TIMEOUT")]
    for mode, listing, extra, ec, msg in cases:
        write(lst, listing + "\n"); rc, o, e = bash([f"{PKG}/wake_cl.sh", mode], {"CL_PI_GO": "10:00", "CL_SSH": fb, "CL_FAKE_LISTING": lst, "CL_WAKE_N": "2", "CL_SLEEP": "0", **extra})
        check(rc == ec and msg in o, f"waker {mode} listing {listing!r} {extra}: rc {rc} (want {ec}) out {o!r} err {e[:200]}")
    # pulls against a local stand-in for scp (copies from a fake box root)
    write(f"{WORK}/fake_scp.py", "import os,shutil,sys,glob\nroot=os.environ['CL_FAKE_ROOT']; a=[x for x in sys.argv[1:] if x not in ('-q','-r') and not x.startswith(('BatchMode','ConnectTimeout')) and x!='-o']; rec='-r' in sys.argv\n"
          "src=a[0].split(':',1)[1].replace('D:/tanitad/runs/v8f_mini_full',root); dst=a[1]\nfs=glob.glob(src) if '*' in src else [src]\nif not fs or not all(os.path.exists(f) for f in fs): sys.exit(1)\n"
          "for f in fs:\n    (shutil.copytree(f, os.path.join(dst, os.path.basename(f)), dirs_exist_ok=True) if os.path.isdir(f) else shutil.copy(f, dst))\n")
    scp = f"{PYW} -I {WORK}/fake_scp.py"; root = WORK + "/fakebox"; shutil.rmtree(root, ignore_errors=True)
    for t in ("CL_0", "CL_1"):
        s = FRZ.INFO[t][1]; run = f"M2b_s{s}_recft_CL"; ck = b"ckpt-pull-" + t.encode()
        for f, c in (("ckpt_last.pt", ck), ("summary.json", b"{}"), ("argv.json", b"{}"), ("selfcheck.json", b"{}"), ("train_log.jsonl", b""), ("train.log", b"")): write(f"{root}/{run}/{f}", c, "wb")
        write(f"{root}/gate_cl/{run}/selfcheck.json", "{}"); write(f"{root}/cl/RUN_{t}.done", json.dumps({"evidence": {"ckpt_sha256": hashlib.sha256(ck).hexdigest()}})); write(f"{root}/cl/logs/{t}.gate.log", "x")
    lock = WORK + "/lock/BOX2_BUSY.txt"; env = {"CL_PI_GO": "10:00", "CL_SCP": scp, "CL_FAKE_ROOT": root, "CL_PY": PYW, "CL_LOCK": lock}
    dest = WORK + "/pulldest"; shutil.rmtree(dest, ignore_errors=True); write(lock, "LEVELHEAD CL two-run chain (D:\\tanitad\\cl\\cl_chain.cmd), owner: x")
    rc, o, e = bash([f"{PKG}/pull_cl.sh"], {**env, "CL_DEST": dest}); check(rc == 0 and o.count("OK CL_") == 2 and "bad=0" in o and "lock released" in o and not os.path.exists(lock), f"pull_cl happy: rc {rc} {o} {e[:300]}")
    check(os.path.exists(f"{dest}/M2b_s1_recft_CL/SHA256SUMS.txt") and os.path.exists(f"{dest}/cl_markers/RUN_CL_0.done") and any(f.startswith("BOX2_BUSY.cl_chain_done_") for f in os.listdir(os.path.dirname(lock))), "pulled layout / lock moved aside, not deleted")
    write(lock, "LEVELHEAD CL two-run chain (...)"); write(f"{root}/M2b_s0_recft_CL/ckpt_last.pt", b"corrupt", "wb"); os.remove(f"{root}/cl/RUN_CL_1.done")
    rc, o, e = bash([f"{PKG}/pull_cl.sh"], {**env, "CL_DEST": WORK + "/pulldest2"}); check(rc == 1 and "MISMATCH CL_0" in o and "SKIP CL_1" in o and "bad=1" in o and os.path.exists(lock) and "lock released" not in o, f"pull_cl bad path keeps the lock: rc {rc} {o}")
    write(lock, "somebody else's job"); shutil.rmtree(root, ignore_errors=True)
    for t in ("CL_0", "CL_1"):
        s = FRZ.INFO[t][1]; run = f"M2b_s{s}_recft_CL"; ck = b"c" + t.encode()
        write(f"{root}/{run}/ckpt_last.pt", ck, "wb"); write(f"{root}/cl/RUN_{t}.done", json.dumps({"evidence": {"ckpt_sha256": hashlib.sha256(ck).hexdigest()}}))
    rc, o, e = bash([f"{PKG}/pull_cl.sh"], {**env, "CL_DEST": WORK + "/pulldest3"}); check(rc == 0 and os.path.exists(lock) and "lock not released" in o, f"a lock that is not the chain's must be left alone: rc {rc} {o[-200:]}")
    # the probe pull: a staged probe result (from the checker run above) -> W_PROBE.json in a temp prereg dir
    sb = pinned_sb("wpull", tags=("WPROBE_0",)); stage_gate(sb, "WPROBE_0", stub_sc("WPROBE_0")); check(sb.run(["gate", "WPROBE_0", "0"])[0] == 0, "stage probe")
    shutil.rmtree(root, ignore_errors=True); shutil.copytree(sb.rc(), f"{root}/cl"); shutil.copytree(sb.gdir("WPROBE_0"), f"{root}/M2b_s0_wprobe_CL")
    pre = WORK + "/wpull_prereg"; shutil.rmtree(pre, ignore_errors=True); write(lock, "LEVELHEAD CL W probe (D:\\tanitad\\cl\\cl_wprobe.cmd), owner: x")
    rc, o, e = bash([f"{PKG}/pull_wprobe.sh"], {**env, "CL_DEST": WORK + "/wpdest", "CL_PREREG_DIR": pre}); check(rc == 0 and "W_PROBE.json written" in o and "lock released" in o, f"pull_wprobe: rc {rc} {o} {e[:300]}")
    check(abs(json.load(open(pre + "/W_PROBE.json"))["w_level_for_10pct_share"] - float(STAND_W)) < 1e-9, "W_PROBE.json value")
    os.remove(f"{root}/M2b_s0_wprobe_CL/selfcheck.json"); shutil.rmtree(pre); write(lock, "LEVELHEAD CL W probe (x)")
    rc, o, e = bash([f"{PKG}/pull_wprobe.sh"], {**env, "CL_DEST": WORK + "/wpdest2", "CL_PREREG_DIR": pre}); check(rc == 1 and not os.path.exists(pre + "/W_PROBE.json") and os.path.exists(lock), f"a probe pull without selfcheck.json: rc {rc} {o}")
    check(not os.path.exists(WORK + "/tripwire.txt"), "a network tool was invoked by a shell script: " + (rd(WORK + "/tripwire.txt") if os.path.exists(WORK + "/tripwire.txt") else ""))


@test
def t_nothing_contacts_box2_without_the_pi_go():
    calls = WORK + "/nogo_calls.log"; shutil.rmtree(calls, ignore_errors=True)
    if os.path.exists(calls): os.remove(calls)
    stub = f"{PYW} -I {WORK}/stub_ssh.py"; write(f"{WORK}/stub_ssh.py", "import os,sys\nopen(os.environ['CL_STUB_LOG'],'a').write(' '.join(sys.argv[1:])+'\\n')\nsys.exit(0)\n")
    lock = WORK + "/nogo_lock/BOX2_BUSY.txt"
    for f, args in (("launch_wprobe.sh", []), ("launch_cl.sh", []), ("wake_cl.sh", ["wprobe"]), ("wake_cl.sh", ["chain"]), ("pull_wprobe.sh", []), ("pull_cl.sh", []), ("wake_pull_cl.sh", ["wprobe"]), ("wake_pull_cl.sh", ["chain"])):
        for pkg in (PKG, f"{FROZEN}/cl_launch"):                                          # the real package and the frozen copy (whose chain would otherwise be allowed to launch)
            rc, o, e = bash([f"{pkg}/{f}"] + args, {"CL_SSH": stub, "CL_SCP": stub, "CL_STUB_LOG": calls, "CL_LOCK": lock, "CL_PY": PYW, "CL_WAKE_N": "1", "CL_SLEEP": "0", "CL_DEST": WORK + "/nogo_dest"}, timeout=40, drop=("CL_PI_GO",))
            check(rc != 0 and "CL_PI_GO" in e and not os.path.exists(calls) and not os.path.exists(WORK + "/tripwire.txt") and not os.path.exists(lock), f"{f} {args} ran without CL_PI_GO: rc {rc} out {o[:200]} err {e[:300]}")
    # and WITH the go but with the box-2 lock held / W not frozen: still nothing remote
    write(lock, "somebody's job\n")
    rc, o, e = bash([f"{FROZEN}/cl_launch/launch_cl.sh"], {"CL_PI_GO": "10:00", "CL_SSH": stub, "CL_SCP": stub, "CL_STUB_LOG": calls, "CL_LOCK": lock, "CL_PY": PYW}, timeout=40)
    check(rc == 5 and "ZZCL-LOCK-HELD" in o and not os.path.exists(calls) and rd(lock) == "somebody's job\n", f"a held lock must refuse without any remote call: rc {rc} {o[:300]}")
    os.remove(lock)
    rc, o, e = bash([f"{PKG}/launch_cl.sh"], {"CL_PI_GO": "10:00", "CL_SSH": stub, "CL_SCP": stub, "CL_STUB_LOG": calls, "CL_LOCK": lock, "CL_PY": PYW}, timeout=40)
    check(rc == 6 and "ZZCL-REFUSED" in o and "FROZEN_W.txt" in o and not os.path.exists(calls) and not os.path.exists(lock), f"the placeholder chain must be refused locally: rc {rc} {o[:300]}")
    check(not os.path.exists(WORK + "/tripwire.txt"), "tripwire")


FAKE_BOX = r'''import hashlib, os, re, shutil, sys
ROOT = os.environ["FAKE_BOX_ROOT"]
def loc(p):
    p = p.replace("D:\\", "D:/")
    for a, b in (("D:/tanitad/runs/v8f_mini_full/cl", ROOT + "/markers"), ("D:/tanitad/cl", ROOT + "/cl"), ("D:/tanitad/repo", ROOT + "/repo"), ("D:/tanitad/logs", ROOT + "/logs")):
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
        script = sys.stdin.read(); assert "Get-ChildItem D:\\tanitad\\cl -Recurse" in script, script
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
        if job.endswith("cl_run_telemetry.cmd"): open(ROOT + "/logs/telemetry_full.csv", "w").write("")
        elif job.endswith(("cl_chain.cmd", "cl_wprobe.cmd")):
            open(loc(m.group(2).replace("\\", "/")), "w").write(env.get("FAKE_JOB_LOG", "===== CL CHAIN START x" if "chain" in job else "===== CL WPROBE START x") + chr(10))
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
        e = {"CL_PI_GO": "10:00", "CL_SSH": ssh or f"{fbx} ssh", "CL_SCP": f"{fbx} scp", "CL_LOCK": lock, "CL_PY": PYW, "CL_NAP": "0", "FAKE_BOX_ROOT": root, **(env or {})}
        rc, o, er = bash([f"{pkg or FROZEN + '/cl_launch'}/{script}"], e, timeout=300)
        return rc, o, er, root

    calls = lambda root: [l for l in rd(root + "/calls.log").splitlines()] if os.path.exists(root + "/calls.log") else []
    # happy chain: old CHAIN_CL.done moved aside, files shipped and re-hashed, four stack files shipped with the old ones aside, telemetry -> guard -> chain, lock kept
    rc, o, er, root = run("launch_cl.sh", "chain", pre=["CHAIN_CL.done", "RUN_CL_0.done"]); c = calls(root)
    check(rc == 0 and "ZZCL-LAUNCHED chain" in o and "lock taken" in o and "shipped bytes identical" in o and "old CHAIN_CL.done moved aside" in o, f"happy chain: rc {rc} {o[-600:]} {er[:300]}")
    check(os.path.exists(lock) and rd(lock).startswith("LEVELHEAD CL two-run chain") and "PI go 10:00" in rd(lock), "the lock was not left in place with the chain's text")
    check(os.path.exists(root + "/markers/RUN_CL_0.done") and any(f.startswith("CHAIN_CL.done.prev_") for f in os.listdir(root + "/markers")) and not os.path.exists(root + "/markers/CHAIN_CL.done"), "only CHAIN_CL.done is moved aside; a run marker stays")
    jobs = [x for x in c if x.startswith("job ")]; check(jobs == ["job cl_run_telemetry.cmd", "job cl_run_guard_chain.cmd", "job cl_chain.cmd"], f"order telemetry -> guard -> chain: {jobs}")
    for rel, want in FRZ.SHIPPED.items():
        check(sh256(f"{root}/repo/{rel}") == want and any(f.startswith(os.path.basename(rel) + ".pre_cl_") and rd(f"{root}/repo/{os.path.dirname(rel)}/{f}") == f"OLD:{rel}\n" for f in os.listdir(f"{root}/repo/{os.path.dirname(rel)}")), f"{rel}: not shipped as pinned with the old file aside")
    shipped = sorted(os.path.relpath(os.path.join(d, f), root + "/cl").replace("\\", "/") for d, _, fs in os.walk(root + "/cl") for f in fs)
    check(shipped == json.loads(json.dumps(sorted(FRZ_ship()))), f"shipped file set {shipped}")
    check(not os.path.exists(WORK + "/tripwire.txt"), "tripwire")
    # happy probe
    rc, o, er, root = run("launch_wprobe.sh", "wprobe", pre=["WPROBE.FAILED", "CHAIN_CL.done"]); jobs = [x for x in calls(root) if x.startswith("job ")]
    check(rc == 0 and "ZZCL-LAUNCHED wprobe" in o and "old WPROBE.FAILED moved aside" in o and jobs == ["job cl_run_telemetry.cmd", "job cl_run_guard_wprobe.cmd", "job cl_wprobe.cmd"] and os.path.exists(root + "/markers/CHAIN_CL.done"), f"happy probe: rc {rc} {o[-500:]} {jobs}")
    check(rd(lock).startswith("LEVELHEAD CL W probe"), "probe lock text")
    rc, o, er, root = run("launch_wprobe.sh", "wprobe_real", pkg=PKG); check(rc == 0 and "ZZCL-LAUNCHED wprobe" in o, f"the REAL package's probe launches on the fake box in the placeholder state: rc {rc} {o[-400:]} {er[:200]}")
    # refusals, each leaves box 2 as it was and releases (moves aside) our lock
    def refused(label, name, msg, env=None, pre=None, rc_want=2, lock_gone=True, script="launch_cl.sh"):
        rc, o, er, root = run(script, name, env=env, pre=pre)
        check(rc == rc_want and msg in o, f"{label}: rc {rc} (want {rc_want}) want {msg!r}: {o[-500:]} {er[:200]}")
        check((not os.path.exists(lock)) == lock_gone, f"{label}: lock state (gone={not os.path.exists(lock)})")
        if lock_gone: check(any(f.startswith("BOX2_BUSY.cl_launch_failed_") for f in os.listdir(os.path.dirname(lock))), f"{label}: lock was not moved aside")
        for f in os.listdir(os.path.dirname(lock)):
            if f.startswith("BOX2_BUSY.cl_launch_failed_"): os.remove(os.path.dirname(lock) + "/" + f)
        return root
    root = refused("python running", "py", "a python process is running on box 2", env={"FAKE_PY": "1"}); check(not os.path.exists(root + "/cl") and not any("job" in x for x in calls(root)), "nothing shipped when python runs")
    root = refused("disk low", "disk", "box 2 D: below 8 GB", env={"FAKE_DFREE": "3"})
    root = refused("a heat stop marker", "gs", "GUARD_STOP.txt", pre=["GUARD_STOP.txt"]); check(not os.path.exists(root + "/cl"), "nothing shipped over a heat stop")
    root = refused("bytes differ on box 2", "corrupt", "shipped bytes differ", env={"FAKE_SCP_CORRUPT": "cl_chain.cmd"}); check(not any(x.startswith("job") for x in calls(root)), "no job after a byte mismatch")
    root = refused("stack scp corrupt", "stk", "on box 2 hashes", env={"FAKE_SCP_CORRUPT": "v8f_mini.py"})
    for rel in FRZ.SHIPPED: check(rd(f"{root}/repo/{rel}") == f"OLD:{rel}\n", f"{rel}: not restored after a failed stack ship")
    check(any(".cl_failed_" in f for rel in FRZ.SHIPPED for f in os.listdir(f"{root}/repo/{os.path.dirname(rel)}")), "the failed new bytes are kept aside")
    root = refused("stack aside move fails", "mv", "moving box 2's stack/tanitad/data/v8f_recover.py aside failed", env={"FAKE_MOVE_FAIL": "v8f_recover.py"})
    for rel in FRZ.SHIPPED: check(rd(f"{root}/repo/{rel}") == f"OLD:{rel}\n", f"{rel}: not restored")
    root = refused("stale telemetry", "tele", "no fresh telemetry", env={"FAKE_TELE_FROZEN": "1"}); check("telemetry.stop" in calls(root) and not any(x == "job cl_chain.cmd" for x in calls(root)), "a stale telemetry must stop the telemetry and not start the chain")
    refused("job stopped at start", "stop", "ZZCL-JOB-STOPPED-AT-START", env={"FAKE_JOB_LOG": "ZZCL-STOP preflight rc 3"}, rc_want=3, lock_gone=False); os.remove(lock)
    refused("no start line", "nostart", "ZZCL-NO-START-LINE", env={"FAKE_JOB_LOG": "nothing"}, rc_want=4, lock_gone=False); os.remove(lock)
    # a held lock: refuse without any remote call (the lock is somebody else's and stays)
    os.makedirs(os.path.dirname(lock), exist_ok=True); write(lock, "someone else's job\n"); root = f"{WORK}/dr_held"; shutil.rmtree(root, ignore_errors=True); os.makedirs(root)
    rc, o, er = bash([f"{FROZEN}/cl_launch/launch_cl.sh"], {"CL_PI_GO": "10:00", "CL_SSH": f"{fbx} ssh", "CL_SCP": f"{fbx} scp", "CL_LOCK": lock, "CL_PY": PYW, "FAKE_BOX_ROOT": root}, timeout=60)
    check(rc == 5 and not calls(root) and rd(lock) == "someone else's job\n", f"held lock: rc {rc} {o[:200]}")
    os.remove(lock)
    # fail closed: an unknown remote command is refused by the fake box (the launcher issues only the commands this test models)
    rc, o, er, root = run("launch_cl.sh", "unk", ssh=f"{PYW} -I -c \"import sys;sys.exit(97)\""); check(rc == 2 and "ZZCL-ABORT" in o, f"an ssh that fails must abort: rc {rc} {o[-200:]}")
    check(not os.path.exists(WORK + "/tripwire.txt"), "tripwire")


def FRZ_ship():
    r = subprocess.run([sys.executable, "-I", f"{FROZEN}/cl_launch/cl_local_check.py", "ship-files"], capture_output=True, text=True); return [x.strip() for x in r.stdout.splitlines() if x.strip()]


PS_PARSE = "param($f) $t=$null;$e=$null;$null = [System.Management.Automation.Language.Parser]::ParseFile($f,[ref]$t,[ref]$e); if ($e.Count) { $e | ForEach-Object { $_.Message }; exit 1 }; Write-Output 'PARSE-OK'"


@test
def t_guard_scripts_parse_stop_on_heat_and_ignore_stale_markers():
    write(WORK + "/parse.ps1", PS_PARSE)
    for g in ("cl_guard_wprobe.ps1", "cl_guard_chain.ps1"):
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", WORK + "/parse.ps1", "-f", f"{PKG}/box2_files/{g}"], capture_output=True, text=True, timeout=90); check(r.returncode == 0 and "PARSE-OK" in r.stdout, f"{g} does not parse: {r.stdout}{r.stderr}")
    chain = rd(f"{PKG}/box2_files/cl_guard_chain.ps1"); wp = rd(f"{PKG}/box2_files/cl_guard_wprobe.ps1")
    check("$ends = @('CHAIN_CL.done', 'CHAIN_CL.STOPPED')" in chain and "WPROBE" not in chain.split("$ends")[1].split("\n")[0], "the chain guard must end on the chain's markers only")
    check("$ends = @('WPROBE.done', 'WPROBE.FAILED', 'WPROBE.STOPPED')" in wp, "the probe guard's end markers")

    def run_guard(src, name, temps, marker=None, marker_after=None, first=None):
        d = f"{WORK}/guard_{name}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d + "/rc")
        write(d + "/tel.csv", "ts,x\r\n" + "".join(f"2026-11-05 10:00:{i:02d},a,b,c,d,e,f,g,h,i,{t}\r\n" for i, t in enumerate(temps)))
        for f in (first or []): write(f"{d}/rc/{f}", "{}")
        s = src.replace(r"D:\tanitad\runs\v8f_mini_full\cl", (d + "/rc").replace("/", "\\")).replace(r"D:\tanitad\logs\telemetry_full.csv", (d + "/tel.csv").replace("/", "\\")).replace(r"D:\tanitad\logs\cl_guard.log", (d + "/guard.log").replace("/", "\\"))
        s = s.replace("v8f_mini_train", "zz_no_such_trainer_zz").replace("Start-Sleep -Seconds 60", "Start-Sleep -Seconds 1"); write(d + "/g.ps1", s)
        t0 = time.time(); p = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", d + "/g.ps1"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if marker_after: time.sleep(marker_after); write(f"{d}/rc/{marker}", "{}")
        try: out, _ = p.communicate(timeout=60)
        except subprocess.TimeoutExpired: p.kill(); raise AssertionError(f"guard {name} did not exit")
        return os.path.exists(d + "/rc/GUARD_STOP.txt"), (rd(d + "/guard.log") if os.path.exists(d + "/guard.log") else ""), out, time.time() - t0
    st, lg, o, _ = run_guard(chain, "hot4", [55] * 16 + [62] * 4); check(st and "ZZGUARD-STOP TEMP: 4 of 20 >= 61 C, max 62 C" in lg, f"4 of 20 >= 61 C must stop: {lg} {o}")
    st, lg, o, _ = run_guard(wp, "hot63", [50] * 19 + [63]); check(st and "max 63 C" in lg, f"one >= 63 C must stop: {lg} {o}")
    st, lg, o, _ = run_guard(chain, "warm3", [50] * 17 + [61] * 3, marker="CHAIN_CL.done", marker_after=4); check(not st and "job ended; guard off (max USB 61 C)" in lg, f"3 of 20 must not stop; the chain marker ends the chain guard: {lg} {o}")
    st, lg, o, el = run_guard(chain, "stale", [50] * 20, marker="CHAIN_CL.STOPPED", marker_after=5, first=["WPROBE.done", "WPROBE.FAILED"]); check(not st and el >= 4.5 and "job ended" in lg, f"a stale WPROBE marker must NOT end the chain guard (ended after {el:.1f}s): {lg}")
    st, lg, o, el = run_guard(wp, "wp_end", [50] * 20, first=["WPROBE.done"]); check(not st and "job ended" in lg and el < 30, f"the probe guard ends on WPROBE.done: {lg}")


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
        t = t.replace("ZZRNEXT-", "ZZCL-").replace("RNEXT CHAIN", "CL CHAIN").replace("CHAIN_RNEXT", "CHAIN_CL").replace(r"D:\tanitad\rnext\rnext_check.py", r"D:\tanitad\cl\cl_check.py").replace(r"D:\tanitad\rnext\rnext_run.cmd", r"D:\tanitad\cl\cl_run.cmd")
        return t.replace(r"%RUNSM%\rnext", r"%RUNSM%\cl").replace("RPT_0 RPT_1", "CL_0 CL_1").replace("rnext_chain_rpt", "cl_chain").replace(r"D:\tanitad\rnext", r"D:\tanitad\cl")
    a = rd(f"{RPT_CHAIN}/rnext_chain_rpt.cmd", "rb").decode().split("\r\n"); b = rd(f"{PKG}/box2_files/cl_chain.cmd", "rb").decode().split("\r\n")
    code = lambda ls: [norm(l) for l in ls if not l.startswith("rem ")]
    ca, cb = code(a), code(b); ca = [l.replace('"%PY%" "%CHK%" preflight', '"%PY%" "%CHK%" preflight chain') if l == '"%PY%" "%CHK%" preflight' else l for l in ca]
    diff = [(x, y) for x, y in zip(ca, cb) if x != y]; check(len(ca) == len(cb) and not diff, f"cl_chain.cmd differs from the RPT chain (names normalised) beyond the preflight argument: {diff[:3]} {len(ca)} vs {len(cb)}")
    ra = [norm(l) for l in rd(f"{RPT_CHAIN}/rnext_run.cmd", "rb").decode().split("\r\n") if not l.startswith("rem ")]; rb = [l for l in rd(f"{PKG}/box2_files/cl_run.cmd", "rb").decode().split("\r\n") if not l.startswith("rem ")]
    ra = [l.replace("RUN-START", "RUN-START") for l in ra]; check(ra == rb, f"cl_run.cmd differs from rnext_run.cmd: {[(x, y) for x, y in zip(ra, rb) if x != y][:3]}")
    w = rd(f"{PKG}/box2_files/cl_wprobe.cmd"); ch = rd(f"{PKG}/box2_files/cl_chain.cmd")
    check(ch.count("%RUNSAFE% cmd /c %RUNCMD%") == 2 and w.count("%RUNSAFE% cmd /c %RUNCMD%") == 1 and "call D:\\tanitad\\env.cmd" in rd(f"{PKG}/box2_files/cl_run.cmd"), "SAFE MODE: every python call goes through %RUNSAFE% (env.cmd sourced by the run file)")
    check("find /c \"python.exe\"" in ch and "find /c \"python.exe\"" in w and "ZZCL-STOP a python process is already running" in ch and "telemetry.stop" in w and "telemetry.stop" in ch, "one job at a time / telemetry stop")
    check('if exist "%RCD%\\GUARD_STOP.txt"' in ch and "exit /b 9" in ch and "preflight run %TAG%" in ch and "gate-cached" in ch and "gate %TAG% !RC!" in ch and "post %TAG% !RC!" in ch, "chain keeps the marker / guard / gate / post steps")
    tel = rd(f"{PKG}/box2_files/cl_run_telemetry.cmd", "rb").decode().split("\r\n"); tel0 = rd(f"{RPT_CHAIN}/rnext_run_telemetry.cmd", "rb").decode().split("\r\n")
    check([l for l in tel if not l.startswith("rem ")] == [l for l in tel0 if not l.startswith("rem ")], "the telemetry job differs from the RPT one")


@test
def t_package_hygiene():
    words = [r"\bvo" + "id\\b", "super" + "seded", "retr" + "acted", "for" + "merly", "no lon" + "ger", "corr" + "ection", "with" + "drawn", "closed" + r"[ -]loop"]
    mine = [f for f in os.listdir(PKG) if os.path.isfile(f"{PKG}/{f}") and f.endswith((".py", ".sh", ".md", ".json", ".txt"))] + [f"box2_files/{f}" for f in os.listdir(f"{PKG}/box2_files") if os.path.isfile(f"{PKG}/box2_files/{f}")] + ["tests/test_cl_launch.py"]
    for f in mine:
        raw = rd(f"{PKG}/{f}", "rb"); txt = raw.decode("utf-8")
        for w in words:
            ln = [l for l in txt.splitlines() if re.search(w, l, re.I)]; check(not ln, f"{f}: banned word ({w}): {ln[:1]}")
        if not f.endswith(".cmd"): check(b"\r" not in raw, f"{f}: CR in an LF file")
    for t in ("CL_0", "CL_1", "WPROBE_0"):
        a = rd(f"{PKG}/box2_files/argv/{t}" + (".gate.txt" if t == "WPROBE_0" else ".train.txt"), "rb"); check(a.count(b"\n") == 1 and a.endswith(b"\n") and b"\r" not in a, f"{t}: argv is not one LF line")
    for f in ("PINS.json",): json.loads(rd(f"{PKG}/{f}"))
    check(os.path.exists(f"{PKG}/HANDOFF_CL.md"), "HANDOFF_CL.md is missing")


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
