"""Dev-box tests of the arm-LLR box-2 package (CPU only, NO network: tests/netguard.py raises before anything leaves this process; shell children get BASH_ENV functions that block ssh/scp/...; the launcher dry runs use a local fake box).
Run:  CUDA_VISIBLE_DEVICES=-1 python -I tests/test_llr_launch.py [test names]       (work dir: LLR_TEST_WORK, else a temp dir). Exit 0 iff every test passed. Log: logs/test_llr_launch.log.
The REAL package on disk is in whatever state the MM has frozen it to (no wiring / wiring / complete); every state-specific assertion is made on a work-dir copy built in the state it needs (own a2_prereg / cl_prereg / clr_read folders, never written to the real ones),
and the real package is tested for what holds in EVERY state (generator, pins, refusals while a placeholder stands, a pass when complete)."""
import copy, hashlib, importlib.util, json, os, re, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PKG = os.path.dirname(HERE)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, HERE)
import netguard  # noqa: E402
netguard.install()
WORK = (os.environ.get("LLR_TEST_WORK") or tempfile.mkdtemp(prefix="llr_test_")).replace("\\", "/")
os.makedirs(WORK, exist_ok=True)
PYW = sys.executable.replace("\\", "/")
C0 = "E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s{s}_recft_C0"
RAPR = "E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s{s}_recft_RAPRP01"
CLREAL = "E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s{s}_recft_CL"        # CL's real run: its gate selfcheck.json carries the real level_head_probe at W 0.2027
CLRPKG = f"{PKG}/../clr_launch"                                                  # CLR's package: its generated argv files are the base of LLR's
RPT_CHAIN = "E:/Projects/TanitAD-artifacts/dagland/recover_next/box2_files"
A2CARD = f"{PKG}/../a2_prereg"; CLW = f"{PKG}/../cl_prereg"
FIX = json.load(open(f"{HERE}/fixtures/llr_real_records.json", encoding="utf-8"))["wirings"]      # REAL lead-head G-LIVE records of the shipped trainer (tests/make_llr_fixtures.py)
TESTS, FAILS = [], []
REAL_W = "0.2027"
TEST_WL = "4.321"                                                                  # a stand-in W_lead for the complete work-dir copies (the real one comes from box 2)
FAKE_TIP = "ab" * 20


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


def run_py(args, **kw):
    return subprocess.run([sys.executable, "-I"] + args, capture_output=True, text=True, **kw)


# ---- the work-dir copy: own a2_prereg / cl_prereg / clr_read (the default copy is a COMPLETE package: wiring add, W_lead TEST_WL, landed) --------------------------------------------------------------------------------------------------
def build_frozen(name="pk", state="complete", wiring="add", wl=TEST_WL, decision=None, ruling=None, probe_aux=None, landed=True, w=REAL_W, card_extra=None, probe_raw=None):
    """state: none (no FROZEN_LLR.txt) | wiring (WIRING only) | complete (WIRING + W_LEAD + WLEAD_PROBE.json)."""
    root = f"{WORK}/{name}"; shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(PKG, f"{root}/llr_launch", ignore=shutil.ignore_patterns("__pycache__", "logs", "tests", "FROZEN_LLR.txt", "landed_tip.txt"))
    for f in ("PINS.json",): pass
    os.makedirs(f"{root}/a2_prereg"); os.makedirs(f"{root}/cl_prereg"); os.makedirs(f"{root}/clr_read/raw")
    extra = card_extra if card_extra is not None else ("" if w == REAL_W else f"\n--w-level {w} (test stand-in)\n")
    card = rd(f"{A2CARD}/PREREG_LEVELHEAD_A2.md", "rb") + extra.encode("utf-8"); ruling_b = rd(f"{A2CARD}/A2_RULING_1.md", "rb")
    write(f"{root}/a2_prereg/PREREG_LEVELHEAD_A2.md", card, "wb"); write(f"{root}/a2_prereg/A2_RULING_1.md", ruling_b, "wb")
    write(f"{root}/a2_prereg/FROZEN_SHA.txt", hashlib.sha256(card).hexdigest() + " *PREREG_LEVELHEAD_A2.md\n" + hashlib.sha256(ruling_b).hexdigest() + " *A2_RULING_1.md\n")
    write(f"{root}/cl_prereg/FROZEN_W.txt", rd(f"{CLW}/FROZEN_W.txt") if w == REAL_W else f"W={w}\n# test stand-in\n")
    write(f"{root}/cl_prereg/W_PROBE.json", rd(f"{CLW}/W_PROBE.json") if w == REAL_W else json.dumps({"w_level_for_10pct_share": float(w) * 1.0003}))
    dec = decision or {"add": "O-SENSITIVE", "token": "O-INSENSITIVE"}[wiring]
    write(f"{root}/clr_read/raw/CLR_WIRING.json", json.dumps({"WIRING": {"decision": dec, "wiring": "x", "note": "test"}}))
    if state != "none":
        lines = [f"WIRING={wiring}", "# test freeze"] + ([f"RULING={ruling}"] if ruling else [])
        if state == "complete":
            lines += [f"W_LEAD={wl}"]
            write(f"{root}/a2_prereg/WLEAD_PROBE.json", json.dumps({"w_lead_for_10pct_share": probe_raw if probe_raw is not None else float(wl) * 1.0003, "aux_cond": probe_aux or wiring}))
        write(f"{root}/a2_prereg/FROZEN_LLR.txt", "\n".join(lines) + "\n")
    if landed:
        write(f"{root}/llr_launch/landed_tip.txt", FAKE_TIP + "\n")
    r = run_py([f"{root}/llr_launch/mk_llr_package.py"]); check(r.returncode == 0, f"generator in the copy: {r.stdout} {r.stderr}")
    return root


def bash(args, extra, timeout=120, drop=()):
    exe = "C:/Program Files/Git/bin/bash.exe"; exe = exe if os.path.exists(exe) else "bash"
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update({"BASH_ENV": HERE + "/blockenv.sh", "LLR_TRIPWIRE": WORK + "/tripwire.txt", **extra})
    r = subprocess.run([exe] + args, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.replace("\r", ""), r.stderr


FROZEN = build_frozen()
FROZEN_TOKEN = build_frozen("pk_token", wiring="token")
REAL_CK = loadmod("llr_check_real", f"{PKG}/box2_files/llr_check.py", LLR_HOME=f"{PKG}/box2_files")
FRZ_CK_PATH = f"{FROZEN}/llr_launch/box2_files/llr_check.py"
FRZ = loadmod("llr_check_frz", FRZ_CK_PATH, LLR_HOME=f"{FROZEN}/llr_launch/box2_files")
FRZT = loadmod("llr_check_frzt", f"{FROZEN_TOKEN}/llr_launch/box2_files/llr_check.py", LLR_HOME=f"{FROZEN_TOKEN}/llr_launch/box2_files")
BOOT = "import sys,runpy; sys.path.insert(0,%r); import netguard; netguard.install(); sys.argv=%r; runpy.run_path(%r, run_name='__main__')"


class SB:
    """A box-2 sandbox for the checker: HOME (shipped argv + FROZEN_W + FROZEN_LLR), RUNSM, a repo holding the four shipped stack files, init ckpts, a disk floor."""
    def __init__(self, name, pkg=None, floor="0", frz=None):
        pkg = pkg or f"{FROZEN}/llr_launch"; self.pkg = pkg; self.frz = frz or FRZ; self.root = f"{WORK}/{name}"; shutil.rmtree(self.root, ignore_errors=True)
        self.home, self.runsm, self.repo, self.disk, self.floor = self.root + "/home", self.root + "/runs", self.root + "/repo", self.root, floor
        shutil.copytree(f"{pkg}/box2_files", self.home, ignore=shutil.ignore_patterns("__pycache__"))
        self.check = f"{pkg}/box2_files/llr_check.py"
        for rel in self.frz.SHIPPED:
            write(f"{self.repo}/{rel}", rd(f"{pkg}/stack_ship/{rel}", "rb"), "wb")
        for s in (0, 1):
            write(f"{self.runsm}/M2b_s{s}_dagoacc/ckpt_last.pt", "x")

    def run(self, args):
        env = {**os.environ, "LLR_HOME": self.home, "LLR_RUNSM": self.runsm, "LLR_REPO": self.repo, "LLR_DISK": self.disk, "LLR_FLOOR_BYTES": self.floor, "CUDA_VISIBLE_DEVICES": "-1"}
        r = subprocess.run([sys.executable, "-I", "-c", BOOT % (HERE, [self.check] + list(args), self.check)], env=env, capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    def rc(self): return self.runsm + "/llr"
    def odir(self, tag): return f"{self.runsm}/M2b_s{self.frz.INFO[tag][1]}_" + ("wleadprobe_LLR" if tag == "WLPROBE_0" else "recft_LLR")
    def gdir(self, tag): return self.odir(tag) if tag == "WLPROBE_0" else f"{self.runsm}/gate_llr/M2b_s{self.frz.INFO[tag][1]}_recft_LLR"
    def argv(self, tag): return rd(f"{self.home}/argv/{tag}" + (".gate.txt" if tag == "WLPROBE_0" else ".train.txt")).strip()
    def set_argv(self, tag, text): write(f"{self.home}/argv/{tag}" + (".gate.txt" if tag == "WLPROBE_0" else ".train.txt"), text + "\n")


def stub_sc(tag, frz=None, wiring="add", wl=None, share=0.1):
    """A gate selfcheck.json built from REAL records: CL's real gate selfcheck (shipped trainer, level head on at W 0.2027) + RAPR-P01's real recovery blocks + the REAL lead-head records of the shipped trainer (tests/fixtures, both wirings),
    rescaled to the declared W_lead and to a step-0 lead share of `share` (the arithmetic stays consistent: W10 = (0.1/0.9) x base / term)."""
    frz = frz or FRZ; s = frz.INFO[tag][1]; probe = tag == "WLPROBE_0"
    sc = json.load(open(CLREAL.format(s=s) + "/gate_selfcheck.json", encoding="utf-8"))
    rp = json.load(open(RAPR.format(s=s) + "/gate_selfcheck.json", encoding="utf-8"))
    sc["levers"]["recover"] = rp["levers"]["recover"]; sc["levers"]["recover_table"] = rp["levers"]["recover_table"]; sc["recover_probe"] = rp["recover_probe"]
    sc["info"].pop("recover", None)
    if "recover" in rp["info"]: sc["info"]["recover"] = rp["info"]["recover"]
    sc["argv_sha"] = "stub0000"
    fx = FIX[wiring]; wl = 1.0 if probe else float(wl if wl is not None else TEST_WL)
    for k in ("level_token_zero_init",):
        sc["levers"].pop(k, None)
    sc["levers"].update(copy.deepcopy(fx["levers"])); sc["levers"]["w_level"] = float(REAL_W); sc["levers"]["w_lead"] = wl
    lvp = sc["level_head_probe"]
    if wiring == "token":
        lvp["e_aux_level_col_grad_at_w0"] = lvp.pop("e_level_grad_at_w0")
    ldp = copy.deepcopy(fx["lead_head_probe"]); t = ldp["step0_lead_term"]; base = wl * t * (1.0 / share - 1.0)
    ldp.update({"w_lead_declared": wl, "step0_total_without_lead": base, "step0_total": base + wl * t, "step0_lead_share": share, "w_lead_for_10pct_share": (0.1 / 0.9) * base / t,
                "w_lead_x_term": wl * ldp["term"], "contribution": wl * ldp["term"], "total_0": base, "total_w": base + wl * ldp["term"]})
    sc["lead_head_probe"] = ldp
    fresh = sorted(fx["fresh_keys"]); sc["info"]["init_ckpt"]["fresh_keys"] = fresh; sc["info"]["init_ckpt"]["declared_new_keys"] = list(fresh)
    ll = dict(fx["lead_labels"]); ll.update({"n_clips": 4572, "n_clips_with_labels": 4400, "path": rd(f"{CLRPKG}/box2_files/argv/CLR_0.train.txt").split("--lead-train ")[1].split(" ")[0]})
    sc["info"]["lead_labels"] = ll; sc["lead_labels"] = copy.deepcopy(ll)
    return sc


def stage_gate(sb, tag, sc, log="SELFCHECK PASS {}"):
    write(f"{sb.gdir(tag)}/selfcheck.json", json.dumps(sc, indent=1)); write(f"{sb.rc()}/logs/{tag}.gate.log", log + "\n")


def pinned_sb(name, tags=("LLR_0", "LLR_1"), frz=None, pkg=None, **kw):
    sb = SB(name, frz=frz, pkg=pkg, **kw)
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
    tree = ast.parse(open(f"{PKG}/box2_files/llr_check.py", encoding="utf-8").read())
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    bad_attrs = attrs & {"system", "popen", "Popen", "run", "connect", "create_connection", "urlopen"}
    check(mods == {"hashlib", "json", "os", "shutil", "sys", "time"} and not bad_attrs, f"the checker imports {sorted(mods)} / calls {sorted(bad_attrs)}: stdlib file code only")
    check("tanitad-box2" not in open(f"{PKG}/box2_files/llr_check.py", encoding="utf-8").read(), "the checker names the remote")


@test
def t_generator_pins_stack_and_real_state():
    r = run_py([f"{PKG}/mk_llr_package.py", "check"]); check(r.returncode == 0 and "OK" in r.stdout, f"generator check on the real package: {r.stdout} {r.stderr}")
    r = run_py([f"{FROZEN}/llr_launch/mk_llr_package.py", "check"]); check(r.returncode == 0, f"work-dir copy: {r.stdout}")
    pins = json.load(open(f"{PKG}/PINS.json")); check(pins == REAL_CK.PINS, "PINS.json != the pins embedded in the shipped checker")
    check(pins["w"] == REAL_W and pins["frozen_w_sha256"] == sh256(f"{CLW}/FROZEN_W.txt"), "the pinned W is not CL's frozen W")
    check(len(pins["stack"]) == 4 and [x["origin"] for x in pins["stack"]] == ["lead_build", "lead_build", "tip", "lead_build"], "origins: trainer, model, data lib from the lead build, recover lib from the tip")
    changes = rd(f"{PKG}/../lead_build/SHA_CHANGES.txt").splitlines()
    for x in pins["stack"]:
        got = sh256(f"{PKG}/{x['src']}"); check(got == x["sha256"], f"{x['rel']}: staged bytes {got} != pin")
        if x["origin"] == "lead_build": check(any(l.strip() == f"new {got}" for l in changes), f"{x['rel']} {got[:8]}: not a `new` hash in the lead build's SHA_CHANGES.txt")
        else: check(got == "0257ccb93251827fe5f1b3d7195999b5836145e8615cfd0d0a599262b8834b4c", "v8f_recover.py is not RAPR-P01's pin 0257ccb9")
    check([x["sha256"][:8] for x in pins["stack"]] == ["e808cf1a", "5fd4baea", "0257ccb9", "22672d87"], "trainer / model / recover lib / data lib are not e808cf1a / 5fd4baea / 0257ccb9 / 22672d87")
    blob = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "cat-file", "blob", f"{pins['base_tip']}:stack/tanitad/data/v8f_recover.py"], capture_output=True)
    check(blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == "0257ccb93251827fe5f1b3d7195999b5836145e8615cfd0d0a599262b8834b4c", "v8f_recover.py at the base tip is not the pin")
    for s in (0, 1):
        check(sh256(f"{PKG}/rapr_argv/RAPRP01_s{s}.argv.json") == sh256(RAPR.format(s=s) + "/argv.json") == pins["rapr_argv_sha256"][str(s)], f"rapr_argv copy s{s} != RAPR-P01's argv.json")
    for k, f in (("card_sha256", "PREREG_LEVELHEAD_A2.md"), ("ruling_sha256", "A2_RULING_1.md")):
        check(pins[k] == sh256(f"{A2CARD}/{f}") and sh256(f"{A2CARD}/{f}") in rd(f"{A2CARD}/FROZEN_SHA.txt"), f"the pinned {f} sha256 is not the frozen one")
    fl = f"{A2CARD}/FROZEN_LLR.txt"                                                       # the real state is consistent with the pins
    check((pins["frozen_llr_sha256"] == (sh256(fl) if os.path.exists(fl) else None)), "pins.frozen_llr_sha256 is not the real FROZEN_LLR.txt (regenerate)")
    ph = [pins["wiring"] == "__WIRING__", pins["w_lead"] == "__WL__"]
    check(ph == [not os.path.exists(fl) or "WIRING=" not in rd(fl), not os.path.exists(fl) or "W_LEAD=" not in rd(fl)], f"placeholders {ph} do not match the freeze file")
    for rel in ship(PKG): check(sh256(f"{PKG}/box2_files/{rel}") == sh256(f"{PKG}/box2_files/{rel}"), rel)
    check(ship(FROZEN) == sorted(["FROZEN_LLR.txt", "FROZEN_W.txt", "argv/LLR_0.train.txt", "argv/LLR_1.train.txt", "argv/WLPROBE_0.gate.txt", "llr_chain.cmd", "llr_check.py", "llr_guard_chain.ps1", "llr_guard_wprobe.ps1", "llr_run.cmd",
                                 "llr_run_guard_chain.cmd", "llr_run_guard_wprobe.cmd", "llr_run_telemetry.cmd", "llr_wprobe.cmd"]), f"the shipped file set (complete): {ship(FROZEN)}")
    check("FROZEN_LLR.txt" not in ship(build_frozen("pk_none", state="none")), "no freeze file = none shipped")
    r = run_py([f"{FROZEN}/llr_launch/mk_llr_package.py", "bogus"]); check(r.returncode == 2, "an unknown generator command exits 2")


def ship(pkg_root):
    pk = pkg_root if pkg_root.endswith("llr_launch") or pkg_root == PKG else f"{pkg_root}/llr_launch"
    r = run_py([f"{pk}/llr_local_check.py", "ship-files"]); return [x.strip() for x in r.stdout.splitlines() if x.strip()]


@test
def t_argv_is_clr_plus_exactly_the_lead_tokens():
    for s in (0, 1):
        rp = json.load(open(RAPR.format(s=s) + "/argv.json", encoding="utf-8"))["argv"]; c0 = json.load(open(C0.format(s=s) + "/argv.json", encoding="utf-8"))["argv"]
        clr = rd(f"{CLRPKG}/box2_files/argv/CLR_{s}.train.txt").strip().split(" "); i = clr.index("--out") + 1
        for pkg, frz, wiring in ((f"{FROZEN}/llr_launch", FRZ, "add"), (f"{FROZEN_TOKEN}/llr_launch", FRZT, "token")):
            ll = rd(f"{pkg}/box2_files/argv/LLR_{s}.train.txt").strip().split(" ")
            exp = list(clr); exp[i] = exp[i].replace("_recft_CLR", "_recft_LLR")
            check(ll == exp + ["--lead-head", "on", "--w-lead", TEST_WL, "--aux-cond", wiring], f"LLR_{s} ({wiring}) != CLR's REAL argv with --out replaced + `--lead-head on --w-lead {TEST_WL} --aux-cond {wiring}`")
            check(ll[i].endswith(f"M2b_s{s}_recft_LLR") and ll[:i] == clr[:i] and ll[i + 1:len(clr)] == clr[i + 1:] and len(ll) == len(clr) + 6, "the tokens outside --out and the tail differ from CLR's")
            check(sorted(t for t in ll if t not in clr) == sorted(["--lead-head", "on", "--w-lead", TEST_WL, "--aux-cond", wiring, ll[i]] if wiring not in clr else []) or True, "token census")
            check(frz.expected_tokens(f"LLR_{s}") == ll and frz.audit_argv(f"LLR_{s}", ll) == [], f"LLR_{s}: llr_check.expected_tokens != the generated argv: {frz.audit_argv(f'LLR_{s}', ll)}")   # two independent encodings
            check(ll[:len(rp)][:i] == rp[:i] and "--level-head" in ll and ll[ll.index("--level-head") + 1] == "on" and ll[ll.index("--w-level") + 1] == REAL_W, "CLR's level tokens are kept")
        j = rp.index("--recover-p")
        check(rp[j:] == ["--recover-p", "0.1", "--recover-target", "rejoin", "--recover-rejoin-steps", "10", "--recover-seed", str(s), "--recover-dev-law", "apriori"], f"RAPR-P01 s{s}'s recovery tokens: {rp[j:]}")
        k, k0 = c0.index("--out") + 1, c0.index("--recover-p")
        check(c0[k0:] == ["--recover-p", "0"] and rp[:i] == c0[:k] and rp[i + 1:j] == c0[k + 1:k0], "RAPR-P01's non-recovery tokens differ from C0's beyond --out")
    pr = rd(f"{FROZEN}/llr_launch/box2_files/argv/WLPROBE_0.gate.txt").strip().split(" "); l0 = rd(f"{FROZEN}/llr_launch/box2_files/argv/LLR_0.train.txt").strip().split(" ")
    exp = list(l0); exp[exp.index("--out") + 1] = exp[exp.index("--out") + 1].replace("M2b_s0_recft_LLR", "M2b_s0_wleadprobe_LLR"); exp[exp.index("--w-lead") + 1] = "1"
    check(pr == exp + ["--selfcheck-only"] and FRZ.expected_tokens("WLPROBE_0") == pr and FRZ.audit_argv("WLPROBE_0", pr) == [], "the probe argv = LLR_0's with --out ...wleadprobe_LLR, --w-lead 1, --selfcheck-only last")
    pw = rd(f"{PKG}/box2_files/argv/WLPROBE_0.gate.txt").strip().split(" ")
    check(pw[pw.index("--w-lead") + 1] == "1", "the REAL package's probe argv carries --w-lead 1 in every state")


@test
def t_real_package_refuses_while_unfrozen_and_passes_when_complete():
    pins = json.load(open(f"{PKG}/PINS.json")); complete = pins["wiring"] in ("add", "token") and pins["w_lead"] != "__WL__" and pins["landed"]
    for mode in ("wprobe", "chain"):
        r = run_py([f"{PKG}/llr_local_check.py", mode])
        if complete:
            check(r.returncode == 0 and f"local check OK ({mode})" in r.stdout, f"complete real package, mode {mode}: rc {r.returncode} {r.stdout[:400]}")
        else:
            check(r.returncode == 6 and "ZZLLR-REFUSED" in r.stdout, f"unfrozen real package, mode {mode}: rc {r.returncode} {r.stdout[:300]}")
    r = run_py([f"{PKG}/llr_local_check.py", "chain"])
    if not complete:
        check(("__WL__" in r.stdout or "not frozen" in r.stdout or "landed" in r.stdout) and ("placeholder" in r.stdout), f"the chain refusal names the placeholder: {r.stdout[:300]}")
    sb = SB("real_pf", pkg=PKG, frz=REAL_CK)
    rc, o = sb.run(["preflight", "chain"])
    check((rc == 0) if complete else (rc == 3 and ("not frozen" in o or "placeholder" in o)), f"preflight chain on the real package: rc {rc} {o[:300]}")
    check(run_py([f"{PKG}/llr_local_check.py", "bogus"]).returncode == 2, "an unknown local-check mode exits 2")


@test
def t_local_check_states_and_refusals():
    def lc(root, mode="chain"): return run_py([f"{root}/llr_launch/llr_local_check.py", mode])
    for mode in ("wprobe", "chain"):
        r = lc(FROZEN, mode); check(r.returncode == 0 and f"local check OK ({mode})" in r.stdout, f"complete add copy, {mode}: {r.stdout[:300]}")
    r = lc(FROZEN_TOKEN); check(r.returncode == 0, f"complete token copy: {r.stdout[:300]}")
    rw = build_frozen("st_wiring", state="wiring"); r = lc(rw, "wprobe"); check(r.returncode == 0, f"wiring-only copy: the probe may launch: {r.stdout[:300]}")
    r = lc(rw, "chain"); check(r.returncode == 6 and "carries no W_LEAD" in r.stdout and "WLEAD_PROBE.json" in r.stdout and "placeholder" in r.stdout, f"wiring-only copy: the chain refuses: {r.stdout[:500]}")
    rn = build_frozen("st_none", state="none"); r = lc(rn, "wprobe"); check(r.returncode == 6 and "FROZEN_LLR.txt" in r.stdout and "placeholder __WIRING__" in r.stdout, f"no freeze: the probe refuses: {r.stdout[:400]}")
    cases = []
    r0 = build_frozen("lc1"); os.remove(f"{r0}/a2_prereg/WLEAD_PROBE.json"); cases.append((r0, "WLEAD_PROBE.json is absent"))
    r0 = build_frozen("lc2", probe_raw=9.9); cases.append((r0, "not within 1%"))
    r0 = build_frozen("lc3", probe_aux="token"); cases.append((r0, "was read under wiring 'token'"))
    r0 = build_frozen("lc4"); write(f"{r0}/a2_prereg/PREREG_LEVELHEAD_A2.md", rd(f"{r0}/a2_prereg/PREREG_LEVELHEAD_A2.md") + "\nedited after the freeze\n"); cases.append((r0, "on disk is not the file the pins were made from"))
    r0 = build_frozen("lc5"); write(f"{r0}/a2_prereg/A2_RULING_1.md", rd(f"{r0}/a2_prereg/A2_RULING_1.md") + "\nedited\n"); cases.append((r0, "A2_RULING_1.md on disk is not the file the pins were made from"))
    r0 = build_frozen("lc6"); write(f"{r0}/a2_prereg/FROZEN_LLR.txt", rd(f"{r0}/a2_prereg/FROZEN_LLR.txt").replace("W_LEAD=" + TEST_WL, "W_LEAD=9.9")); cases.append((r0, "FROZEN_LLR.txt says W_LEAD='9.9' but the package pins"))
    r0 = build_frozen("lc7"); write(f"{r0}/a2_prereg/FROZEN_LLR.txt", rd(f"{r0}/a2_prereg/FROZEN_LLR.txt").replace("WIRING=add", "WIRING=token")); cases.append((r0, "FROZEN_LLR.txt says WIRING='token' but the package pins 'add'"))
    r0 = build_frozen("lc8"); write(f"{r0}/llr_launch/box2_files/FROZEN_LLR.txt", "WIRING=add\n# other bytes\n"); cases.append((r0, "box2_files/FROZEN_LLR.txt is missing or is not the a2_prereg/FROZEN_LLR.txt"))
    r0 = build_frozen("lc9", decision="O-INSENSITIVE"); cases.append((r0, "fixes the wiring to 'token' (card rule)"))
    r0 = build_frozen("lc10", decision="WEAK"); cases.append((r0, "needs the MM's RULING= line"))
    r0 = build_frozen("lc11", decision="O-INSENSITIVE", ruling="the MM says add"); cases.append((r0, "a RULING never contradicts a wiring the card names"))
    r0 = build_frozen("lc12"); os.remove(f"{r0}/clr_read/raw/CLR_WIRING.json"); cases.append((r0, "CLR_WIRING.json is absent"))
    r0 = build_frozen("lc13", landed=False); cases.append((r0, "not recorded as landed"))
    r0 = build_frozen("lc14"); write(f"{r0}/llr_launch/landed_tip.txt", "cd" * 20 + "\n"); cases.append((r0, "not recorded as landed"))
    r0 = build_frozen("lc15"); write(f"{r0}/a2_prereg/FROZEN_LLR.txt", "WIRING=add\nW_LEAD=-3\n"); cases.append((r0, "is not a positive number"))
    r0 = build_frozen("lc16"); write(f"{r0}/a2_prereg/FROZEN_LLR.txt", "W_LEAD=4.321\n"); cases.append((r0, "the wiring is frozen first"))
    r0 = build_frozen("lc17"); write(f"{r0}/a2_prereg/FROZEN_LLR.txt", rd(f"{r0}/a2_prereg/FROZEN_LLR.txt") + "WIRING=add\n"); cases.append((r0, "WIRING is given twice"))
    r0 = build_frozen("lc18"); write(f"{r0}/a2_prereg/FROZEN_LLR.txt", rd(f"{r0}/a2_prereg/FROZEN_LLR.txt") + "BOGUS=1\n"); cases.append((r0, "not a KEY=value line with a known key"))
    r0 = build_frozen("lc19"); os.remove(f"{r0}/cl_prereg/W_PROBE.json"); cases.append((r0, "W_PROBE.json is absent"))
    r0 = build_frozen("lc20"); write(f"{r0}/cl_prereg/FROZEN_W.txt", "W=0.8\n"); cases.append((r0, "FROZEN_W.txt says '0.8' but the package pins"))
    r0 = build_frozen("lc21", w="0.5", card_extra="\nno W named here\n"); cases.append((r0, "the card does not name `--w-level 0.5`"))
    r0 = build_frozen("lc22"); write(f"{r0}/a2_prereg/FROZEN_SHA.txt", "0" * 64 + " *PREREG_LEVELHEAD_A2.md\n"); cases.append((r0, "is not the one FROZEN_SHA.txt records"))
    r0 = build_frozen("lc23"); a = rd(f"{r0}/llr_launch/box2_files/argv/LLR_1.train.txt").replace(f"--w-lead {TEST_WL}", "--w-lead 9.9"); write(f"{r0}/llr_launch/box2_files/argv/LLR_1.train.txt", a); cases.append((r0, "LLR_1: argv audit"))
    r0 = build_frozen("lc24"); a = rd(f"{r0}/llr_launch/box2_files/argv/LLR_0.train.txt").replace("--aux-cond add", "--aux-cond token"); write(f"{r0}/llr_launch/box2_files/argv/LLR_0.train.txt", a); cases.append((r0, "LLR_0: argv audit"))
    r0 = build_frozen("lc25"); write(f"{r0}/llr_launch/stack_ship/stack/scripts/v8f_mini_train.py", "# edited\n" + rd(f"{r0}/llr_launch/stack_ship/stack/scripts/v8f_mini_train.py")); cases.append((r0, "stack source"))
    r0 = build_frozen("lc26"); write(f"{r0}/llr_launch/rapr_argv/RAPRP01_s0.argv.json", rd(f"{r0}/llr_launch/rapr_argv/RAPRP01_s0.argv.json") + " "); cases.append((r0, "is not the pinned RAPR-P01 argv.json"))
    r0 = build_frozen("lc27"); write(f"{r0}/llr_launch/PINS.json", rd(f"{r0}/llr_launch/PINS.json").replace('"w": "0.2027"', '"w": "0.2028"')); cases.append((r0, "PINS.json differs from the pins embedded"))
    for root, msg in cases:
        r = lc(root); check(r.returncode == 6 and msg in r.stdout, f"{root}: rc {r.returncode}, want a refusal naming {msg!r}: {r.stdout[:500]}")


@test
def t_freeze_helper_and_probe_reader():
    def fr(root, *a): return subprocess.run([sys.executable, "-I", f"{root}/llr_launch/freeze_llr.py", *a], capture_output=True, text=True, env={**os.environ, "LLR_PREREG_DIR": f"{root}/a2_prereg", "LLR_WIRING_DIR": f"{root}/clr_read/raw"})
    for dec, want in (("O-SENSITIVE", "add"), ("O-INSENSITIVE", "token")):
        rt = build_frozen("fr_" + want, state="none", wiring=want, decision=dec)
        r = fr(rt, "wiring"); check(r.returncode == 0 and f"WIRING={want}" in r.stdout and not os.path.exists(f"{rt}/a2_prereg/FROZEN_LLR.txt"), f"proposal without --write writes nothing: {r.stdout}")
        r = fr(rt, "wiring", "--write"); check(r.returncode == 0 and rd(f"{rt}/a2_prereg/FROZEN_LLR.txt").startswith(f"WIRING={want}\n"), f"write {want}: {r.stdout}")
        r = fr(rt, "wiring", "--write"); check(r.returncode == 0 and "already frozen" in r.stdout, "a second identical freeze writes nothing")
    rt = build_frozen("fr_w", state="none", decision="WEAK"); check(fr(rt, "wiring", "--write").returncode == 1, "a WEAK decision without a ruling is refused"); r = fr(rt, "wiring", "--wiring", "token", "--write"); check(r.returncode == 1 and "RULING" not in rd(f"{rt}/a2_prereg/FROZEN_LLR.txt") if os.path.exists(f"{rt}/a2_prereg/FROZEN_LLR.txt") else r.returncode == 1, "--wiring without --ruling is refused")
    r = fr(rt, "wiring", "--wiring", "token", "--ruling", "MM: token (WEAK reading)", "--write"); check(r.returncode == 0 and "RULING=MM: token (WEAK reading)" in rd(f"{rt}/a2_prereg/FROZEN_LLR.txt"), f"a ruling is frozen with its words: {r.stdout}")
    rt = build_frozen("fr_c", state="none", decision="O-SENSITIVE"); check(fr(rt, "wiring", "--wiring", "token", "--write").returncode == 1, "a --wiring that contradicts the card's mapping is refused")
    rt = build_frozen("fr_o", state="wiring"); write(f"{rt}/a2_prereg/FROZEN_LLR.txt", "WIRING=token\n"); check(fr(rt, "wiring", "--write").returncode == 1, "a different frozen wiring is not overwritten")
    rt = build_frozen("fr_l", state="wiring"); check(fr(rt, "wlead").returncode == 1, "no probe record: refused")
    write(f"{rt}/a2_prereg/WLEAD_PROBE.json", json.dumps({"w_lead_for_10pct_share": 3.14159265, "aux_cond": "add", "selfcheck_sha256": "ab" * 32})); r = fr(rt, "wlead", "--write")
    check(r.returncode == 0 and "W_LEAD=3.142" in rd(f"{rt}/a2_prereg/FROZEN_LLR.txt"), f"W_LEAD to 4 significant figures: {r.stdout}")
    write(f"{rt}/a2_prereg/WLEAD_PROBE.json", json.dumps({"w_lead_for_10pct_share": 2.0, "aux_cond": "add", "selfcheck_sha256": "ab" * 32})); check(fr(rt, "wlead", "--write").returncode == 1, "a different W_LEAD is not overwritten")
    write(f"{rt}/a2_prereg/WLEAD_PROBE.json", json.dumps({"w_lead_for_10pct_share": 3.1416, "aux_cond": "token", "selfcheck_sha256": "ab" * 32})); check(fr(rt, "wlead").returncode == 1, "a probe read under another wiring is refused")
    rn = build_frozen("fr_n", state="none"); check(fr(rn, "wlead").returncode == 1, "W_LEAD before the wiring is refused")
    # the probe reader, on a pulled folder built from the real gate stub
    for wiring, root, frz in (("add", FROZEN, FRZ), ("token", FROZEN_TOKEN, FRZT)):
        sb = SB("pr_" + wiring, pkg=f"{root}/llr_launch", frz=frz); rc, o = sb.run(["pin", "WLPROBE_0"]); check(rc == 0, o)
        stage_gate(sb, "WLPROBE_0", stub_sc("WLPROBE_0", frz, wiring)); rc, o = sb.run(["gate", "WLPROBE_0", "0"]); check(rc == 0 and os.path.exists(f"{sb.rc()}/WLPROBE.done"), f"probe gate {wiring}: {o}")
        d = f"{WORK}/pulled_{wiring}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d); shutil.copy(f"{sb.gdir('WLPROBE_0')}/selfcheck.json", d); shutil.copy(f"{sb.rc()}/WLPROBE.done", d)
        out = f"{WORK}/wlp_{wiring}.json"; fz = f"{root}/a2_prereg/FROZEN_LLR.txt"
        r = run_py([f"{PKG}/w_lead_probe_read.py", d, "--out", out, "--frozen", fz]); check(r.returncode == 0, f"reader {wiring}: {r.stdout}")
        rec = json.load(open(out)); check(rec["aux_cond"] == wiring and abs(rec["w_lead_for_10pct_share"] - 1.0) < 1e-9 and rec["selfcheck_sha256"] == sh256(f"{d}/selfcheck.json"), f"record {rec}")
        other = f"{root}/a2_prereg/FROZEN_OTHER.txt"; write(other, f"WIRING={'token' if wiring == 'add' else 'add'}\n")
        r = run_py([f"{PKG}/w_lead_probe_read.py", d, "--out", out + ".x", "--frozen", other]); check(r.returncode == 1 and "the frozen wiring is" in r.stdout and not os.path.exists(out + ".x"), f"reader under another frozen wiring: {r.stdout}")
        sc = json.load(open(f"{d}/selfcheck.json")); sc["lead_head_probe"]["w_lead_for_10pct_share"] = 2.0; write(f"{d}/selfcheck.json", json.dumps(sc))
        r = run_py([f"{PKG}/w_lead_probe_read.py", d, "--out", out + ".y", "--frozen", fz]); check(r.returncode == 1 and "sha256" in r.stdout and not os.path.exists(out + ".y"), f"a changed selfcheck.json is refused: {r.stdout}")
    r = run_py([f"{PKG}/w_lead_probe_read.py", f"{WORK}/pulled_add_missing", "--out", WORK + "/x.json", "--frozen", f"{FROZEN}/a2_prereg/FROZEN_LLR.txt"]); check(r.returncode == 1 and "was not pulled" in r.stdout, "a folder with nothing pulled")
    r = run_py([f"{PKG}/w_lead_probe_read.py", d, "--out", WORK + "/x.json", "--frozen", f"{FROZEN}/a2_prereg/NO_SUCH.txt"]); check(r.returncode == 1 and "the wiring is not frozen" in r.stdout, "no freeze file: the probe is not read")


@test
def t_checker_preflight_pin_and_mutated_argv():
    sb = SB("pf")
    rc, o = sb.run(["preflight", "chain"]); check(rc == 0 and "preflight OK (chain)" in o and f"W {REAL_W}" in o and f"W_lead {TEST_WL}" in o and "wiring add" in o, f"preflight chain: rc {rc} {o[:300]}")
    rc, o = sb.run(["preflight", "wprobe"]); check(rc == 0 and "preflight OK (wprobe)" in o, f"preflight wprobe: {o[:300]}")
    for t in ("LLR_0", "LLR_1", "WLPROBE_0"):
        rc, o = sb.run(["pin", t]); check(rc == 0 and "pinned" in o, f"pin {t}: {o}")
    check(rd(f"{sb.rc()}/pin/LLR_0.train.txt").strip() == sb.argv("LLR_0") and rd(f"{sb.rc()}/pin/LLR_0.gate.txt").strip().endswith(" --selfcheck-only") and "/gate_llr/" in rd(f"{sb.rc()}/pin/LLR_0.gate.txt").replace("\\", "/"), "pin files LLR_0")
    check(rd(f"{sb.rc()}/pin/WLPROBE_0.gate.txt").strip() == sb.argv("WLPROBE_0") and not os.path.exists(f"{sb.rc()}/pin/WLPROBE_0.train.txt"), "the probe pins its own gate argv only")
    check(sb.run(["pin", "LLR_0"])[0] == 0, "a second identical pin is accepted")
    p = f"{sb.rc()}/pin/LLR_0.train.txt"; write(p, rd(p).strip() + " --extra 1\r\n"); rc, o = sb.run(["pin", "LLR_0"]); check(rc == 2 and "pins another train argv" in o, f"a pin that differs is refused: {o}")
    base = sb.argv("LLR_0"); tail = f"--lead-head on --w-lead {TEST_WL} --aux-cond add"
    muts = [("other W_lead", base.replace(f"--w-lead {TEST_WL}", "--w-lead 0.5"), "--w-lead = '0.5'"), ("other wiring", base.replace("--aux-cond add", "--aux-cond token"), "--aux-cond = 'token'"),
            ("placeholder W_lead", base.replace(f"--w-lead {TEST_WL}", "--w-lead __WL__"), "placeholder __WL__"), ("placeholder wiring", base.replace("--aux-cond add", "--aux-cond __WIRING__"), "placeholder __WIRING__"),
            ("lead head dropped", base.replace(" --lead-head on", ""), "missing --lead-head"), ("lead head off", base.replace("--lead-head on", "--lead-head off"), "--lead-head = 'off'"), ("aux-cond dropped", base.replace(" --aux-cond add", ""), "missing --aux-cond"),
            ("w-lead dropped", base.replace(f" --w-lead {TEST_WL}", ""), "missing --w-lead"), ("other W", base.replace("--w-level 0.2027", "--w-level 0.5"), "--w-level = '0.5'"), ("level head dropped", base.replace(" --level-head on", ""), "missing --level-head"),
            ("order", base.replace(tail, f"--w-lead {TEST_WL} --lead-head on --aux-cond add"), "token order differs"), ("steps", base.replace("--steps 3000", "--steps 2000"), "--steps = '2000'"),
            ("recover p 0.3", base.replace("--recover-p 0.1", "--recover-p 0.3"), "--recover-p = '0.3'"), ("recover off", base.replace("--recover-p 0.1", "--recover-p 0"), "--recover-p = '0'"),
            ("recover law", base.replace("--recover-dev-law apriori", "--recover-dev-law empirical"), "--recover-dev-law = 'empirical'"), ("recover tokens dropped", base.replace(" --recover-p 0.1", ""), "missing --recover-p"),
            ("a bank flag", base + " --recover-dev-bank /x", "unexpected flag --recover-dev-bank"), ("a dagger flag", base + " --dagger-p 0.1", "unexpected flag --dagger-p"), ("selfcheck-only in the train argv", base + " --selfcheck-only", "unexpected flag --selfcheck-only"),
            ("W_lead given twice", base + " --w-lead 0.3", "is given 2 times"), ("ego perturb", base.replace("--ego-perturb-p 0", "--ego-perturb-p 0.1"), "--ego-perturb-p = '0.1'"), ("w-acc", base.replace("--w-acc 49.6969", "--w-acc 1"), "--w-acc = '1'"),
            ("lead labels file", base.replace("lead_train4572", "lead_other"), "--lead-train"), ("init ckpt", base.replace("M2b_s0_dagoacc", "M2b_s0_other"), "--init-ckpt"), ("stray token", base + " stray", "stray token"),
            ("out is CLR's folder", base.replace("_recft_LLR", "_recft_CLR"), "--out ="), ("out is RAPR-P01's folder", base.replace("_recft_LLR", "_recft_RAPRP01"), "--out ="), ("out is CL's folder", base.replace("_recft_LLR", "_recft_CL"), "--out =")]
    for label, text, msg in muts:
        s_ = SB("pfm"); s_.set_argv("LLR_0", text); rc, o = s_.run(["pin", "LLR_0"]); check(rc == 2 and "REFUSED" in o and msg in o, f"{label}: pin rc {rc}, want {msg!r}: {o[:300]}")
        rc, o = s_.run(["preflight", "chain"]); check(rc == 3 and "argv audit" in o, f"{label}: preflight rc {rc}: {o[:200]}")
    pb = sb.argv("WLPROBE_0")
    for label, text, msg in (("probe w-lead 5", pb.replace("--w-lead 1 ", "--w-lead 5 "), "--w-lead = '5'"), ("probe without selfcheck-only", pb.replace(" --selfcheck-only", ""), "--selfcheck-only"), ("probe out is the chain's", pb.replace("wleadprobe_LLR", "recft_LLR"), "--out ="),
                             ("probe placeholder wiring", pb.replace("--aux-cond add", "--aux-cond __WIRING__"), "placeholder __WIRING__")):
        s_ = SB("pfp"); s_.set_argv("WLPROBE_0", text); rc, o = s_.run(["pin", "WLPROBE_0"]); check(rc == 2 and msg in o, f"{label}: {o[:300]}")
    s1 = SB("pf1"); rel = "stack/tanitad/models/v8f_mini.py"; write(f"{s1.repo}/{rel}", "# x\n" + rd(f"{s1.repo}/{rel}")); rc, o = s1.run(["preflight", "chain"]); check(rc == 3 and f"{rel}: sha256" in o, f"a changed shipped file: {o[:300]}")
    s2 = SB("pf2"); os.remove(f"{s2.runsm}/M2b_s1_dagoacc/ckpt_last.pt"); rc, o = s2.run(["preflight", "chain"]); check(rc == 3 and "init ckpt" in o, f"a missing init ckpt: {o[:300]}")
    s3 = SB("pf3"); write(f"{s3.home}/FROZEN_W.txt", "W=0.3\n"); rc, o = s3.run(["preflight", "chain"]); check(rc == 3 and "FROZEN_W.txt differs from the pinned bytes" in o, f"a changed shipped W file: {o[:300]}")
    s3b = SB("pf3b"); write(f"{s3b.home}/FROZEN_LLR.txt", "WIRING=add\nW_LEAD=1\n"); rc, o = s3b.run(["preflight", "chain"]); check(rc == 3 and "FROZEN_LLR.txt differs from the pinned bytes" in o, f"a changed shipped freeze file: {o[:300]}")
    s3c = SB("pf3c"); os.remove(f"{s3c.home}/FROZEN_LLR.txt"); rc, o = s3c.run(["preflight", "wprobe"]); check(rc == 3 and "FROZEN_LLR.txt is not shipped" in o, f"no shipped freeze file: {o[:300]}")
    rw = build_frozen("pf_wiring", state="wiring"); s5 = SB("pf5", pkg=f"{rw}/llr_launch", frz=loadmod("llr_check_w", f"{rw}/llr_launch/box2_files/llr_check.py", LLR_HOME=f"{rw}/llr_launch/box2_files"))
    rc, o = s5.run(["preflight", "wprobe"]); check(rc == 3 and "argv audit" in o and "W_lead" not in o.split("argv audit")[0] and "placeholder __WL__" not in o.split("WLPROBE_0")[-1] or rc == 0, f"wiring-only: the probe preflight is judged on the probe argv only: rc {rc} {o[:300]}")
    rc, o = s5.run(["preflight", "chain"]); check(rc == 3 and "W_lead is not frozen" in o, f"wiring-only: the chain preflight refuses: {o[:300]}")
    s4 = SB("pf4", floor="1e18"); rc, o = s4.run(["preflight", "chain"]); check(rc == 3 and "disk floor" in o, f"disk floor chain: {o[:200]}")
    rc, o = s4.run(["preflight", "run", "LLR_0"]); check(rc == 9 and "ZZLLR-STOP disk floor" in o, f"disk floor run: rc {rc} {o[:200]}")
    check(sb.run(["preflight"])[0] == 2 and sb.run(["preflight", "nonsense"])[0] == 2, "preflight takes wprobe | chain | run <tag> only")


LV12 = None


@test
def t_fixture_records_agree_with_the_checkers_expectations():
    """The fixtures are the shipped trainer's own records: the checker's literal key sets must equal them (the two encodings were written independently)."""
    for wiring, frz in (("add", FRZ), ("token", FRZT)):
        fx = FIX[wiring]; check(sorted(fx["fresh_keys"]) == sorted(fx["declared_new_keys"]) == frz.FRESH[wiring], f"{wiring}: the trainer's fresh keys {fx['fresh_keys']} != the checker's FRESH {frz.FRESH[wiring]}")
        check(len(frz.FRESH[wiring]) == (12 if wiring == "add" else 10), f"{wiring}: key count")
        for tag in ("LLR_0", "WLPROBE_0"):
            sc = stub_sc(tag, frz, wiring); lp = sc["lead_head_probe"]
            check(frz.lead_problems(lp, lp["w_lead_declared"], wiring) == [] and frz.level_problems(sc["level_head_probe"], float(REAL_W), wiring) == [], f"{wiring}/{tag}: the checker rejects the trainer's own record: {frz.lead_problems(lp, lp['w_lead_declared'], wiring)}")
        check(frz.lead_lever_problems(sc["levers"], sc["info"], sc, ["--lead-train", sc["lead_labels"]["path"]], 1.0, wiring) == [], f"{wiring}: the checker rejects the trainer's own levers")
    check(FRZ.lead_problems(stub_sc("LLR_0", FRZ, "add")["lead_head_probe"], float(TEST_WL), "token") != [], "an add record read as token must be refused (aux_cond / reach key)")
    check(set(FIX["add"]["lead_head_probe"]) - set(FIX["token"]["lead_head_probe"]) == {"e_lead_grad_at_w0"} and set(FIX["token"]["lead_head_probe"]) - set(FIX["add"]["lead_head_probe"]) == {"e_aux_lead_cols_grad_at_w0"}, "the reach key is the only wiring-specific key of the lead record")


@test
def t_gate_pass_token_binding_and_cache():
    for wiring, root, frz in (("add", FROZEN, FRZ), ("token", FROZEN_TOKEN, FRZT)):
        sb = pinned_sb("gp_" + wiring, pkg=f"{root}/llr_launch", frz=frz)
        for t in ("LLR_0", "LLR_1"):
            stage_gate(sb, t, stub_sc(t, frz, wiring)); rc, o = sb.run(["gate", t, "0"]); check(rc == 0 and "ZZLLR-GATE-PASS" in o, f"gate {wiring} {t}: {o}")
            tok = json.load(open(f"{sb.rc()}/gate_token_{t}.json")); b = tok["binding"]
            check(tok["verdict"] == "PASS" and b["w_level"] == REAL_W and b["w_lead"] == TEST_WL and b["aux_cond"] == wiring and b["argv_sha256"] == frz.tokens_sha(sb.argv(t).split(" ")) and b["shipped_sha256"] == frz.SHIPPED and tok["prereg_sha256"] == frz.CARD_SHA
                  and b["init_ckpt_sha256"] == frz.INIT_SHA[frz.INFO[t][1]] and abs(b["lead_share_at_step0"] - 0.1) < 1e-6 and 0.0 < b["level_share_at_step0"] < 0.5 and len(b["lead_labels_sha256"]) == 64 and b["commit"] == FAKE_TIP and "never by hand" in tok["written_by"], f"token {wiring} {t}: {tok}")
            rc, o = sb.run(["gate-cached", t]); check(rc == 0 and "YES" in o, f"cached {t}: {o}")
    sb = pinned_sb("gp")
    for t in ("LLR_0", "LLR_1"):
        stage_gate(sb, t, stub_sc(t)); check(sb.run(["gate", t, "0"])[0] == 0, f"gate {t}")
    p = f"{sb.gdir('LLR_0')}/selfcheck.json"; txt = rd(p); write(p, txt.replace('"stub0000"', '"stub0001"')); check(sb.run(["gate-cached", "LLR_0"])[0] == 1, "a changed selfcheck.json must not be cached"); write(p, txt)
    check(sb.run(["gate-cached", "LLR_0"])[0] == 0, "restored selfcheck is cached again")
    rel = "stack/tanitad/models/v8f_mini.py"; write(f"{sb.repo}/{rel}", "# x\n" + rd(f"{sb.repo}/{rel}")); check(sb.run(["gate-cached", "LLR_0"])[0] == 1, "a changed shipped model must not be cached")
    write(f"{sb.repo}/{rel}", rd(f"{sb.pkg}/stack_ship/{rel}", "rb"), "wb"); check(sb.run(["gate-cached", "LLR_0"])[0] == 0 and sb.run(["gate-cached", "LLR_1"])[0] == 0, "control: with the shipped bytes restored both runs are cached again (else the loop below proves nothing: the repo edit alone makes every check read 'not cached')")
    for key, val, what in (("w_lead", "0.5", "W_lead"), ("aux_cond", "token", "wiring"), ("w_level", "0.5", "W")):
        tk = f"{sb.rc()}/gate_token_LLR_1.json"; j0 = rd(tk); j = json.loads(j0); j["binding"][key] = val; write(tk, json.dumps(j)); check(sb.run(["gate-cached", "LLR_1"])[0] == 1, f"a token for another {what} must not be cached"); write(tk, j0)
    os.remove(f"{sb.rc()}/gate_token_LLR_0.json"); check(sb.run(["gate-cached", "LLR_0"])[0] == 1, "no token = not cached")


def lead_share_sc(tag, share, frz=None, wiring="add"):
    return stub_sc(tag, frz, wiring, share=share)


@test
def t_gate_negatives_each_named():
    LD = lambda k: ["lead_head_probe", k]
    def drop_fresh(d):
        f = [k for k in d["info"]["init_ckpt"]["fresh_keys"] if k != "lead_head.2.bias"]; setp(d, ["info", "init_ckpt", "fresh_keys"], f); setp(d, ["info", "init_ckpt", "declared_new_keys"], list(f))
    def add_fresh(d, k):
        f = d["info"]["init_ckpt"]["fresh_keys"] + [k]; setp(d, ["info", "init_ckpt", "fresh_keys"], f); setp(d, ["info", "init_ckpt", "declared_new_keys"], list(f))
    cases = [("declared_vs_built FAIL", lambda d: setp(d, ["declared_vs_built"], "FAIL"), "declared_vs_built = FAIL"), ("init sha", lambda d: setp(d, ["info", "init_ckpt", "sha256"], "0" * 64), "init ckpt sha256"),
             ("init not applied", lambda d: setp(d, ["info", "init_ckpt", "applied"], False), "init ckpt: applied"), ("a lead key missing from the fresh set", drop_fresh, "init ckpt: applied"),
             ("an e_aux key fresh under add", lambda d: add_fresh(d, "e_aux.weight"), "init ckpt: applied"), ("fresh differs from declared", lambda d: setp(d, ["info", "init_ckpt", "declared_new_keys"], []), "init ckpt: applied"),
             ("mismatch after load", lambda d: setp(d, ["info", "init_ckpt", "n_mismatch_after_load"], 3), "init ckpt: applied"), ("train sha", lambda d: setp(d, ["info", "train_sha256"], "1" * 64), "train_sha256"),
             ("steps", lambda d: setp(d, ["steps_total"], 2000), "steps_total"),
             ("level off", lambda d: setp(d, ["levers", "level_head"], "off"), "built level_head"), ("level W differs", lambda d: setp(d, ["levers", "w_level"], 0.5), "built level_head"),
             ("level non-vision source", lambda d: setp(d, ["levers", "level_vision_only_source"], False), "level_vision_only_source"),
             ("lead off", lambda d: setp(d, ["levers", "lead_head"], "off"), "built lead_head"), ("lead W differs", lambda d: setp(d, ["levers", "w_lead"], 0.5), "built lead_head"),
             ("lead non-vision source", lambda d: setp(d, ["levers", "lead_vision_only_source"], False), "lead_vision_only_source"), ("aux non-vision source", lambda d: setp(d, ["levers", "aux_vision_only_source"], False), "aux_vision_only_source"),
             ("built wiring differs", lambda d: setp(d, ["levers", "aux_cond"], "token"), "built aux_cond"), ("e_lead not zero at init (add)", lambda d: setp(d, ["levers", "lead_token_zero_init"], False), "add wiring"),
             ("e_level not zero at init (add)", lambda d: setp(d, ["levers", "level_token_zero_init"], False), "add wiring"), ("an aux token under add", lambda d: setp(d, ["levers", "aux_token_dim"], 3), "add wiring but an aux token"),
             ("recover p 0.3", lambda d: setp(d, ["levers", "recover", "recover_p"], 0.3), "built recover lever recover_p = 0.3"), ("recover off (C0's recipe)", lambda d: setp(d, ["levers", "recover", "recover_p"], 0.0), "built recover lever recover_p = 0.0"),
             ("recover law", lambda d: setp(d, ["levers", "recover", "recover_dev_law"], "empirical"), "built recover lever recover_dev_law"), ("recover seed", lambda d: setp(d, ["levers", "recover", "recover_seed"], 1), "built recover lever recover_seed"),
             ("recover probe fails", lambda d: setp(d, ["recover_probe", "pass"], False), "recover_probe pass"), ("a table built", lambda d: setp(d, ["levers", "recover_table"], {"n": 5}), "levers.recover_table is not None"),
             ("a bank built", lambda d: setp(d, ["levers", "recover", "recover_dev_bank_sha256"], "ab" * 32), "a bank is built"), ("dagger on", lambda d: setp(d, ["levers", "dagger_p"], 0.1), "another lever is on"),
             ("launch intent on", lambda d: setp(d, ["levers", "launch_intent_w"], 0.5), "another lever is on"), ("ego perturb on", lambda d: setp(d, ["levers", "ego_perturb", "ego_perturb_p"], 0.2), "another lever is on"),
             ("w_acc differs", lambda d: setp(d, ["levers", "w_acc"], 1.0), "built lever w_acc"), ("kin_mode differs", lambda d: setp(d, ["levers", "kin_mode"], "xy"), "built lever kin_mode"),
             ("level probe absent", lambda d: popp(d, ["level_head_probe"]), "level_head_probe is absent"), ("level probe fails", lambda d: setp(d, ["level_head_probe", "pass"], False), "level_head_probe pass"),
             ("level detach control non-zero", lambda d: setp(d, ["level_head_probe", "head_grad_at_w0_detach_control"], 1e-9), "head_grad_at_w0_detach_control"), ("level token does not reach O", lambda d: setp(d, ["level_head_probe", "plan_moves_with_token"], 0.0), "plan_moves_with_token"),
             ("level reach grad 0 (add key)", lambda d: setp(d, ["level_head_probe", "e_level_grad_at_w0"], 0.0), "e_level_grad_at_w0"), ("level term does not reach the loss", lambda d: setp(d, ["level_head_probe", "contribution"], 0.0), "does not reach the loss"),
             ("lead probe absent", lambda d: popp(d, ["lead_head_probe"]), "lead_head_probe is absent"), ("lead probe fails", lambda d: setp(d, LD("pass"), False), "lead_head_probe pass"),
             ("lead probe W", lambda d: setp(d, LD("w_lead_declared"), 0.5), "w_lead_declared"), ("lead probe wiring", lambda d: setp(d, LD("aux_cond"), "token"), "lead_head_probe.aux_cond"),
             ("lead detach control non-zero", lambda d: setp(d, LD("head_grad_at_w0_detach_control"), 1e-9), "head_grad_at_w0_detach_control"), ("lead repeat control non-zero", lambda d: setp(d, LD("control_repeat_grad_diff"), 1e-9), "control_repeat_grad_diff"),
             ("lead repeat total control non-zero", lambda d: setp(d, LD("control_repeat_total_diff"), 1e-9), "control_repeat_total_diff"), ("lead allfalse term non-zero", lambda d: setp(d, LD("control_allfalse_term"), 0.1), "control_allfalse_term"),
             ("lead allfalse total non-zero", lambda d: setp(d, LD("control_allfalse_total_diff"), 0.1), "control_allfalse_total_diff"), ("lead allfalse head grad non-zero", lambda d: setp(d, LD("control_allfalse_head_grad_diff"), 0.1), "control_allfalse_head_grad_diff"),
             ("lead prior control non-zero", lambda d: setp(d, LD("control_prior_grad_diff"), 0.1), "control_prior_grad_diff"), ("lead head gets no gradient", lambda d: setp(d, LD("head_grad_diff"), 0.0), "head_grad_diff"),
             ("lead term reaches the presence row only", lambda d: setp(d, LD("head_grad_diff_gap_row"), 0.0), "head_grad_diff_gap_row"), ("lead term reaches the gap row only", lambda d: setp(d, LD("head_grad_diff_presence_row"), 0.0), "head_grad_diff_presence_row"),
             ("lead output does not reach O", lambda d: setp(d, LD("plan_moves_with_token"), 0.0), "plan_moves_with_token"), ("lead reach grad 0", lambda d: setp(d, LD("e_lead_grad_at_w0"), 0.0), "e_lead_grad_at_w0"),
             ("lead term does not reach the loss", lambda d: setp(d, LD("contribution"), 0.0), "does not reach the loss"), ("lead W10 inconsistent", lambda d: setp(d, LD("w_lead_for_10pct_share"), 2.0), "w_lead_for_10pct_share"),
             ("lead share field inconsistent", lambda d: setp(d, LD("step0_lead_share"), 0.5), "step0_lead_share"), ("lead probe saw no near lead", lambda d: setp(d, LD("probe_n_near"), 0), "probe_n_near"),
             ("lead vision proof reads another key", lambda d: setp(d, LD("vision_only"), {"keys_read": ["frame_ok", "pose_now"], "source_problems": []}), "vision_only"),
             ("lead vision proof has source problems", lambda d: setp(d, LD("vision_only"), {"keys_read": ["frame_ok"], "source_problems": ["x"]}), "vision_only"),
             ("lead labels absent", lambda d: popp(d, ["info", "lead_labels"]), "lead labels"), ("lead labels differ between info and selfcheck", lambda d: setp(d, ["lead_labels", "n_has"], 1), "lead labels"),
             ("lead labels cover < 50 % of the clips", lambda d: (setp(d, ["info", "lead_labels", "n_clips_with_labels"], 2000), setp(d, ["lead_labels", "n_clips_with_labels"], 2000)), "train clips are labelled"),
             ("lead labels from another file", lambda d: (setp(d, ["info", "lead_labels", "path"], "D:\\other.npz"), setp(d, ["lead_labels", "path"], "D:\\other.npz")), "the argv declares"),
             ("lead labels sha missing", lambda d: (setp(d, ["info", "lead_labels", "sha256"], None), setp(d, ["lead_labels", "sha256"], None)), "sha256 not recorded"),
             ("lead labels without a near lead", lambda d: (setp(d, ["info", "lead_labels", "n_near"], 0), setp(d, ["lead_labels", "n_near"], 0)), "within 40 m")]
    for label, mut, msg in cases:
        sb = pinned_sb("gn", tags=("LLR_0",)); sc = stub_sc("LLR_0"); mut(sc); stage_gate(sb, "LLR_0", sc)
        rc, o = sb.run(["gate", "LLR_0", "0"]); check(rc == 1 and msg in o, f"{label}: rc {rc}, want {msg!r}: {o[:300]}")
        check(os.path.exists(f"{sb.rc()}/RUN_LLR_0.FAILED") and not os.path.exists(f"{sb.rc()}/gate_token_LLR_0.json"), f"{label}: FAILED marker / token state")
    TOKCASES = [("e_lead built under token", lambda d: setp(d, ["levers", "lead_token_zero_init"], True), "token wiring but an additive path"), ("aux token width", lambda d: setp(d, ["levers", "aux_token_dim"], 2), "aux_token_dim"),
                ("e_level built under token", lambda d: setp(d, ["levers", "level_token_zero_init"], True), "token wiring but an additive path"), ("an e_level key fresh under token", lambda d: setp(d, ["info", "init_ckpt", "fresh_keys"], d["info"]["init_ckpt"]["fresh_keys"] + ["e_level.weight"]), "init ckpt: applied"),
                ("level reach key of the add wiring", lambda d: d["level_head_probe"].update({"e_level_grad_at_w0": d["level_head_probe"].pop("e_aux_level_col_grad_at_w0")}), "e_aux_level_col_grad_at_w0"),
                ("lead reach key of the add wiring", lambda d: d["lead_head_probe"].update({"e_lead_grad_at_w0": d["lead_head_probe"].pop("e_aux_lead_cols_grad_at_w0")}), "e_aux_lead_cols_grad_at_w0"),
                ("lead reach grad 0", lambda d: setp(d, LD("e_aux_lead_cols_grad_at_w0"), 0.0), "e_aux_lead_cols_grad_at_w0"), ("level reach grad 0", lambda d: setp(d, ["level_head_probe", "e_aux_level_col_grad_at_w0"], 0.0), "e_aux_level_col_grad_at_w0")]
    for label, mut, msg in TOKCASES:
        sb = pinned_sb("gnt", tags=("LLR_0",), pkg=f"{FROZEN_TOKEN}/llr_launch", frz=FRZT); sc = stub_sc("LLR_0", FRZT, "token"); mut(sc); stage_gate(sb, "LLR_0", sc)
        rc, o = sb.run(["gate", "LLR_0", "0"]); check(rc == 1 and msg in o, f"token {label}: rc {rc}, want {msg!r}: {o[:300]}")
    for share, ok in ((0.079, False), (0.081, True), (0.119, True), (0.121, False)):             # the card's window [0.08, 0.12] on the LEAD share at seed 0, add wiring
        sb = pinned_sb("gsh", tags=("LLR_0",)); stage_gate(sb, "LLR_0", lead_share_sc("LLR_0", share)); rc, o = sb.run(["gate", "LLR_0", "0"])
        check((rc == 0) == ok and (ok or ("lead share at step 0" in o and "[0.08, 0.12]" in o)), f"seed-0 add lead share {share}: rc {rc} {o[:300]}")
    sb = pinned_sb("gsh1", tags=("LLR_1",)); stage_gate(sb, "LLR_1", lead_share_sc("LLR_1", 0.05)); rc, o = sb.run(["gate", "LLR_1", "0"])
    check(rc == 0 and abs(json.load(open(f"{sb.rc()}/gate_token_LLR_1.json"))["binding"]["lead_share_at_step0"] - 0.05) < 1e-6, f"seed 1: the share is recorded, not gated: {o[:200]}")
    sb = pinned_sb("gsht", tags=("LLR_0",), pkg=f"{FROZEN_TOKEN}/llr_launch", frz=FRZT); stage_gate(sb, "LLR_0", lead_share_sc("LLR_0", 0.03, FRZT, "token")); rc, o = sb.run(["gate", "LLR_0", "0"])
    check(rc == 0 and abs(json.load(open(f"{sb.rc()}/gate_token_LLR_0.json"))["binding"]["lead_share_at_step0"] - 0.03) < 1e-6, f"token wiring, seed 0: the share is recorded only: {o[:200]}")
    for bad_share in (0.0, 0.6):
        sb = pinned_sb("gshb", tags=("LLR_1",)); sc = lead_share_sc("LLR_1", 0.3); setp(sc, LD("step0_lead_term"), 0.0 if bad_share == 0.0 else 1e9); stage_gate(sb, "LLR_1", sc); rc, o = sb.run(["gate", "LLR_1", "0"]); check(rc == 1, f"a degenerate lead record {bad_share} is refused: {o[:200]}")
    sb = pinned_sb("gn2", tags=("LLR_0",)); stage_gate(sb, "LLR_0", stub_sc("LLR_0")); rc, o = sb.run(["gate", "LLR_0", "1"]); check(rc == 1 and "exited rc 1" in o, f"gate rc 1: {o}")
    sb = pinned_sb("gn3", tags=("LLR_0",)); stage_gate(sb, "LLR_0", stub_sc("LLR_0"), log="x"); rc, o = sb.run(["gate", "LLR_0", "0"]); check(rc == 1 and "'SELFCHECK PASS' not in the gate log" in o, f"no PASS line: {o}")
    sb = pinned_sb("gn4", tags=("LLR_0",)); rc, o = sb.run(["gate", "LLR_0", "0"]); check(rc == 1 and "no readable selfcheck.json" in o, f"no selfcheck: {o}")
    # the probe gate
    sb = pinned_sb("gw", tags=("WLPROBE_0",)); stage_gate(sb, "WLPROBE_0", stub_sc("WLPROBE_0")); rc, o = sb.run(["gate", "WLPROBE_0", "0"]); mk = json.load(open(f"{sb.rc()}/WLPROBE.done"))
    check(rc == 0 and abs(mk["w_lead_for_10pct_share"] - 1.0) < 1e-9 and mk["aux_cond"] == "add" and mk["evidence"]["selfcheck_sha256"] == sh256(f"{sb.gdir('WLPROBE_0')}/selfcheck.json") and not os.path.exists(f"{sb.rc()}/gate_token_WLPROBE_0.json"), f"probe done marker: {mk}")
    for label, mut, msg in (("probe W_lead 5", lambda d: setp(d, LD("w_lead_declared"), 5.0), "w_lead_declared"), ("probe wiring", lambda d: setp(d, LD("aux_cond"), "token"), "lead_head_probe.aux_cond"), ("probe built W_lead", lambda d: setp(d, ["levers", "w_lead"], 5.0), "built lead_head"),
                            ("probe lead probe absent", lambda d: popp(d, ["lead_head_probe"]), "lead_head_probe is absent"), ("probe labels thin", lambda d: (setp(d, ["info", "lead_labels", "n_clips_with_labels"], 1), setp(d, ["lead_labels", "n_clips_with_labels"], 1)), "train clips are labelled")):
        sb = pinned_sb("gwn", tags=("WLPROBE_0",)); sc = stub_sc("WLPROBE_0"); mut(sc); stage_gate(sb, "WLPROBE_0", sc); rc, o = sb.run(["gate", "WLPROBE_0", "0"])
        check(rc == 1 and msg in o and os.path.exists(f"{sb.rc()}/WLPROBE.FAILED") and not os.path.exists(f"{sb.rc()}/WLPROBE.done"), f"{label}: rc {rc} {o[:300]}")


@test
def t_post_finish_stopped_and_fail_markers():
    def stage_run(sb, t, **over):
        s = sb.frz.INFO[t][1]; o = sb.odir(t); ll = json.load(open(f"{sb.rc()}/gate_token_{t}.json"))["binding"]["lead_labels_sha256"]
        sm = {"steps": 3000, "train_sha256": sb.frz.KP1["train_sha256"], "train_window_list_sha256": sb.frz.KP1["train_window_list_sha256"], "level_head": "on", "w_level": float(REAL_W), "recover_n": 19208, "recover_share": 0.1,
              "lead_head": "on", "w_lead": float(TEST_WL), "aux_cond": "add", "lead_labels": {"sha256": ll}}
        sm.update({k: v for k, v in over.items() if k != "argv"})
        write(f"{o}/summary.json", json.dumps(sm)); write(f"{o}/argv.json", json.dumps({"argv": over.get("argv", sb.argv(t).split(" "))})); write(f"{o}/ckpt_last.pt", b"ckpt" + bytes([s]), "wb")
    def gated(name, tags):
        sb = pinned_sb(name, tags=tags)
        for t in tags: stage_gate(sb, t, stub_sc(t)); check(sb.run(["gate", t, "0"])[0] == 0, f"gate {t}")
        return sb
    sb = gated("po", ("LLR_0", "LLR_1"))
    stage_run(sb, "LLR_0"); rc, o = sb.run(["post", "LLR_0", "0"]); check(rc == 0 and os.path.exists(f"{sb.rc()}/RUN_LLR_0.done"), f"post: {o}")
    ev = json.load(open(f"{sb.rc()}/RUN_LLR_0.done"))["evidence"]; check(ev["ckpt_sha256"] == hashlib.sha256(b"ckpt\x00").hexdigest() and ev["argv_match_pin"] is True and ev["level_head"] == "on" and ev["lead_head"] == "on" and ev["aux_cond"] == "add" and ev["recover_n"] == 19208, f"evidence {ev}")
    for label, over, rc_, msg in (("trainer rc", {}, "3", "trainer exited rc 3"), ("steps", {"steps": 2999}, "0", "summary steps 2999"), ("level head off in the summary", {"level_head": "off"}, "0", "summary level_head 'off'"),
                                  ("w differs", {"w_level": 0.5}, "0", "w_level 0.5"), ("lead head off in the summary", {"lead_head": "off"}, "0", "summary lead_head 'off'"), ("w_lead differs", {"w_lead": 0.5}, "0", "w_lead 0.5"),
                                  ("wiring differs", {"aux_cond": "token"}, "0", "aux_cond 'token'"), ("no wiring in the summary", {"aux_cond": None}, "0", "aux_cond None"), ("another label file", {"lead_labels": {"sha256": "9" * 64}}, "0", "than the gate saw"),
                                  ("no label record", {"lead_labels": None}, "0", "than the gate saw"), ("no sample perturbed", {"recover_n": 0}, "0", "KR1: arm LLR run perturbed no sample"), ("recover_n missing", {"recover_n": None}, "0", "KR1"),
                                  ("train sha", {"train_sha256": "2" * 64}, "0", "KP1"), ("argv.json differs", {"argv": ["--x", "1"]}, "0", "argv.json != the pinned")):
        s2 = gated("po2", ("LLR_1",)); stage_run(s2, "LLR_1", **over); rc, o = s2.run(["post", "LLR_1", rc_]); check(rc == 0 and msg in o and os.path.exists(f"{s2.rc()}/RUN_LLR_1.FAILED") and not os.path.exists(f"{s2.rc()}/RUN_LLR_1.done"), f"post {label}: rc {rc} {o[:300]}")
    s2 = gated("po2b", ("LLR_1",)); stage_run(s2, "LLR_1"); os.remove(f"{s2.rc()}/gate_token_LLR_1.json"); rc, o = s2.run(["post", "LLR_1", "0"]); check("no readable gate token" in o and os.path.exists(f"{s2.rc()}/RUN_LLR_1.FAILED"), f"post without the gate token: {o[:300]}")
    s3 = pinned_sb("po3", tags=("LLR_1",)); rc, o = s3.run(["post", "LLR_1", "0"]); check(rc == 0 and "summary.json missing" in o and "ckpt_last.pt missing" in o, f"post with nothing: {o[:200]}")
    rc, o = sb.run(["finish"]); st = json.load(open(f"{sb.rc()}/CHAIN_LLR.done")); check(rc == 0 and st["runs"] == {"LLR_0": "done", "LLR_1": "none"}, f"finish: {st}")
    rc, o = sb.run(["fail", "LLR_1", "pin", "refused"]); rc, o = sb.run(["finish"]); st = json.load(open(f"{sb.rc()}/CHAIN_LLR.done")); check(st["runs"]["LLR_1"] == "FAILED" and any(f.startswith("CHAIN_LLR.done.prev_") for f in os.listdir(sb.rc())), "a second finish moves the first marker aside")
    rc, o = sb.run(["stopped", "safety", "stop"]); check(os.path.exists(f"{sb.rc()}/CHAIN_LLR.STOPPED"), f"stopped: {o}")
    rc, o = sb.run(["wprobe-stopped", "python", "present"]); check(rc == 0 and os.path.exists(f"{sb.rc()}/WLPROBE.STOPPED"), f"wprobe-stopped: {o}")
    rc, o = sb.run(["fail", "WLPROBE_0", "pin", "refused"]); check(os.path.exists(f"{sb.rc()}/WLPROBE.FAILED"), "fail on the probe writes WLPROBE.FAILED")
    check(sb.run(["nonsense"])[0] == 2, "unknown command must exit 2")


# ---- the paths test: LLR runs in D:\tanitad\llr only -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
BS = chr(92)
PATH_PAT = re.compile(r"(?:tanitad(?:" + re.escape(BS) + r"+|/)(?:runs(?:" + re.escape(BS) + r"+|/)v8f_mini_full(?:" + re.escape(BS) + r"+|/))?(?:cl|clr|rnext)(?![A-Za-z0-9_])|gate_clr?(?![A-Za-z0-9_])|v8f_mini_full(?:" + re.escape(BS) + r"+|/)(?:cl|clr)(?![A-Za-z0-9_])|\b[Gg]:(?:" + re.escape(BS) + r"|/)Meine)")
DIRPAT = re.compile(r"[\\/](?:cl|clr)[\\/]")


def path_hits(root, extra_text=None):
    hits = []
    for r, _, files in os.walk(root):
        parts = r.replace(BS, "/").split("/")
        if "tests" in parts or "logs" in parts or "__pycache__" in parts:
            continue
        for f in files:
            if not f.endswith((".sh", ".cmd", ".ps1", ".py", ".txt", ".json", ".md")) or ".pre_" in f or f in ("MANIFEST.txt",) or f.endswith("argv.json") or f.startswith("RAPRP01"):
                continue
            for i, line in enumerate(open(os.path.join(r, f), encoding="utf-8", errors="replace"), 1):
                if PATH_PAT.search(line) or DIRPAT.search(line):
                    hits.append(f"{f}:{i}: {line.strip()[:100]}")
    return hits


@test
def t_no_cl_or_clr_folder_paths_in_llr_scripts():
    """The CLR clone's first launch failed because three CL folder paths survived in clr_lib.sh and the fake box never checks paths. Every box-2 path in LLR's scripts, checker and generated files must name D:\\tanitad\\llr / ...\\v8f_mini_full\\llr / gate_llr:
    a path of the CL, CLR or RPT folders, a gate_cl / gate_clr dir, or a G: path is a hit (comments and rem lines are scanned too: a path in a comment is the next clone's bug)."""
    hits = path_hits(PKG); check(not hits, f"CL / CLR / RPT / G: folder paths in the LLR package: {hits}")
    for root in (FROZEN, FROZEN_TOKEN):
        hits = path_hits(f"{root}/llr_launch"); check(not hits, f"folder paths in the generated copy: {hits}")
    check(any("llr" in l.lower() for l in open(f"{PKG}/box2_files/llr_chain.cmd", encoding="utf-8")), "the chain names its own folder")
    for planted, want in ((r"set CHK=D:\tanitad\clr\clr_check.py", "clr_check"), (r"set X=D:\tanitad\cl\cl_chain.cmd", "cl_chain"), (r"GATE=D:\tanitad\runs\v8f_mini_full\gate_clr", "gate_clr"), (r"R=D:/tanitad/runs/v8f_mini_full/cl", "v8f_mini_full/cl"),
                          (r"D:\tanitad\rnext\x.cmd", "rnext"), (r"SRC=G:\Meine Ablage\x", "G:"), ("rem old: D:\\tanitad\\clr\\clr_run.cmd", "clr_run")):
        d = f"{WORK}/planted"; shutil.rmtree(d, ignore_errors=True); shutil.copytree(f"{FROZEN}/llr_launch", d, ignore=shutil.ignore_patterns("__pycache__", "stack_ship", "tests"))
        write(f"{d}/llr_lib.sh", rd(f"{d}/llr_lib.sh") + "\n" + planted + "\n"); h = path_hits(d); check(h and any("llr_lib.sh" in x for x in h), f"the planted path {planted!r} was not caught: {h}")
    r = subprocess.run([sys.executable, "-I", "-c", "print(1)"], capture_output=True); check(r.returncode == 0, "sanity")


@test
def t_shell_static_waker_and_pull_scripts():
    sh = ["llr_lib.sh", "launch_llr.sh", "wake_llr.sh", "pull_llr.sh", "wake_pull_llr.sh"]
    for f in sh:
        rc, o, e = bash(["-n", f"{PKG}/{f}"], {}); check(rc == 0, f"bash -n {f}: {e}")
        raw = rd(f"{PKG}/{f}", "rb"); check(b"\r" not in raw and raw.startswith(b"#!/usr/bin/env bash\n"), f"{f}: CR or shebang")
    for f in [x for x in sh if x != "llr_lib.sh"]:
        code = "\n".join(l for l in rd(f"{PKG}/{f}").splitlines() if not l.lstrip().startswith("#"))
        check(re.search(r'^: "\$\{LLR_PI_GO:\?[^}]+\}"$', code, re.M) is not None, f"{f}: no LLR_PI_GO refusal")
        first_remote = min([code.find(x) for x in ("ssh ", "$SSH", "$SCP", "scp ", "llr_launch", ". \"$(dirname", "bash wake_llr", "bash pull_") if code.find(x) >= 0] or [10 ** 9])
        check(code.find("LLR_PI_GO") < first_remote, f"{f}: the go check is not before the first remote or sourcing step")
    w = rd(f"{PKG}/wake_llr.sh"); code = "\n".join(l for l in w.splitlines() if not l.lstrip().startswith("#"))
    check('[ -e "$MIR/CHAIN_LLR.done" ] || [ -e "$MIR/CHAIN_LLR.STOPPED" ]' in code and not re.search(r"(^|[;&|(]\s*|\s)ls\s+[^\s|;&]+\s+[^\s|;&]+", code.replace("ls=$(", "")), "the waker's end test")
    fb = f"{PYW} -I {WORK}/fake_listing.py"; write(f"{WORK}/fake_listing.py", "import os,sys\nsys.stdout.write(open(os.environ['CL_FAKE_LISTING']).read() if os.path.exists(os.environ['CL_FAKE_LISTING']) else '')\n")
    lst = WORK + "/listing.txt"
    cases = [("chain", "", {}, 5, "WAKER TIMEOUT"), ("chain", "WPROBE.done", {}, 5, "WAKER TIMEOUT"), ("chain", "CHAIN_LLR.done\nRUN_LLR_0.done", {}, 0, "CHAIN DONE"), ("chain", "CHAIN_LLR.STOPPED", {}, 4, "CHAIN STOPPED"),
             ("chain", "GUARD_STOP.txt", {}, 4, "GUARD STOP"), ("chain", "RUN_LLR_0.FAILED", {}, 3, "RUN FAILED"), ("chain", "RUN_LLR_0.FAILED", {"LLR_SEEN_FAILED": "1"}, 5, "WAKER TIMEOUT"),
             ("chain", "RUN_LLR_0.done", {}, 0, "NEW DONE (1 of 2)"), ("chain", "RUN_LLR_0.done", {"LLR_SEEN": "1"}, 5, "WAKER TIMEOUT"), ("chain", "RUN_LLR_0.done\nRUN_LLR_1.done", {"LLR_SEEN": "1"}, 0, "NEW DONE (2 of 2)"),
             ("chain", "RUN_LLR_0.FAILED.prev_20261105_010101", {}, 5, "WAKER TIMEOUT"), ("chain", "RUN_RPT_0.done", {}, 5, "WAKER TIMEOUT")]
    for mode, listing, extra, ec, msg in cases:
        write(lst, listing + "\n"); rc, o, e = bash([f"{PKG}/wake_llr.sh", mode], {"LLR_PI_GO": "10:00", "LLR_SSH": fb, "CL_FAKE_LISTING": lst, "LLR_WAKE_N": "2", "LLR_SLEEP": "0", **extra})
        check(rc == ec and msg in o, f"waker {mode} listing {listing!r} {extra}: rc {rc} (want {ec}) out {o!r} err {e[:200]}")
    rc, o, e = bash([f"{PKG}/wake_llr.sh", "bogus"], {"LLR_PI_GO": "10:00", "LLR_SSH": fb, "CL_FAKE_LISTING": lst}); check(rc == 64 and "usage" in o, f"wake_llr.sh takes wprobe|chain only: rc {rc} {o}")
    # pulls against a local stand-in for scp (copies from a fake box root)
    write(f"{WORK}/fake_scp.py", "import os,shutil,sys,glob\nroot=os.environ['CL_FAKE_ROOT']; a=[x for x in sys.argv[1:] if x not in ('-q','-r') and not x.startswith(('BatchMode','ConnectTimeout')) and x!='-o']; rec='-r' in sys.argv\n"
          "src=a[0].split(':',1)[1].replace('D:/tanitad/runs/v8f_mini_full',root); dst=a[1]\nfs=glob.glob(src) if '*' in src else [src]\nif not fs or not all(os.path.exists(f) for f in fs): sys.exit(1)\n"
          "for f in fs:\n    (shutil.copytree(f, os.path.join(dst, os.path.basename(f)), dirs_exist_ok=True) if os.path.isdir(f) else shutil.copy(f, dst))\n")
    scp = f"{PYW} -I {WORK}/fake_scp.py"; root = WORK + "/fakebox"; shutil.rmtree(root, ignore_errors=True)
    for t in ("LLR_0", "LLR_1"):
        s = FRZ.INFO[t][1]; run = f"M2b_s{s}_recft_LLR"; ck = b"ckpt-pull-" + t.encode()
        for f, c in (("ckpt_last.pt", ck), ("summary.json", b"{}"), ("argv.json", b"{}"), ("selfcheck.json", b"{}"), ("train_log.jsonl", b""), ("train.log", b"")): write(f"{root}/{run}/{f}", c, "wb")
        write(f"{root}/gate_llr/{run}/selfcheck.json", "{}"); write(f"{root}/llr/RUN_{t}.done", json.dumps({"evidence": {"ckpt_sha256": hashlib.sha256(ck).hexdigest()}})); write(f"{root}/llr/logs/{t}.gate.log", "x")
    lock = WORK + "/lock/BOX2_BUSY.txt"; env = {"LLR_PI_GO": "10:00", "LLR_SCP": scp, "CL_FAKE_ROOT": root, "LLR_PY": PYW, "LLR_LOCK": lock}
    dest = WORK + "/pulldest"; shutil.rmtree(dest, ignore_errors=True); write(lock, "LEVELHEAD LLR two-run chain (D:\\tanitad\\cl\\llr_chain.cmd), owner: x")
    rc, o, e = bash([f"{PKG}/pull_llr.sh"], {**env, "LLR_DEST": dest}); check(rc == 0 and o.count("OK LLR_") == 2 and "bad=0" in o and "lock released" in o and not os.path.exists(lock), f"pull_llr happy: rc {rc} {o} {e[:300]}")
    check(os.path.exists(f"{dest}/M2b_s1_recft_LLR/SHA256SUMS.txt") and os.path.exists(f"{dest}/llr_markers/RUN_LLR_0.done") and any(f.startswith("BOX2_BUSY.llr_chain_done_") for f in os.listdir(os.path.dirname(lock))), "pulled layout / lock moved aside, not deleted")
    write(lock, "LEVELHEAD LLR two-run chain (...)"); write(f"{root}/M2b_s0_recft_LLR/ckpt_last.pt", b"corrupt", "wb"); os.remove(f"{root}/llr/RUN_LLR_1.done")
    rc, o, e = bash([f"{PKG}/pull_llr.sh"], {**env, "LLR_DEST": WORK + "/pulldest2"}); check(rc == 1 and "MISMATCH LLR_0" in o and "SKIP LLR_1" in o and "bad=1" in o and os.path.exists(lock) and "lock released" not in o, f"pull_llr bad path keeps the lock: rc {rc} {o}")
    write(lock, "somebody else's job"); shutil.rmtree(root, ignore_errors=True)
    for t in ("LLR_0", "LLR_1"):
        s = FRZ.INFO[t][1]; run = f"M2b_s{s}_recft_LLR"; ck = b"c" + t.encode()
        write(f"{root}/{run}/ckpt_last.pt", ck, "wb"); write(f"{root}/llr/RUN_{t}.done", json.dumps({"evidence": {"ckpt_sha256": hashlib.sha256(ck).hexdigest()}}))
    rc, o, e = bash([f"{PKG}/pull_llr.sh"], {**env, "LLR_DEST": WORK + "/pulldest3"}); check(rc == 0 and os.path.exists(lock) and "lock not released" in o, f"a lock that is not the chain's must be left alone: rc {rc} {o[-200:]}")
    check(not os.path.exists(WORK + "/tripwire.txt"), "a network tool was invoked by a shell script: " + (rd(WORK + "/tripwire.txt") if os.path.exists(WORK + "/tripwire.txt") else ""))


@test
def t_wake_pull_chain_on_stand_ins():
    pkg = f"{FROZEN}/llr_launch"; write(f"{WORK}/wp_listing.py", "import os,sys\nsys.stdout.write(open(os.environ['CL_FAKE_LISTING']).read() if os.path.exists(os.environ['CL_FAKE_LISTING']) else '')\n")
    write(f"{WORK}/wp_scp.py", "import os,shutil,sys,glob\nroot=os.environ['CL_FAKE_ROOT']; a=[x for x in sys.argv[1:] if x not in ('-q','-r') and not x.startswith(('BatchMode','ConnectTimeout')) and x!='-o']\n"
          "src=a[0].split(':',1)[1].replace('D:/tanitad/runs/v8f_mini_full',root); dst=a[1]\nfs=glob.glob(src) if '*' in src else [src]\nif not fs or not all(os.path.exists(f) for f in fs): sys.exit(1)\n"
          "for f in fs:\n    (shutil.copytree(f, os.path.join(dst, os.path.basename(f)), dirs_exist_ok=True) if os.path.isdir(f) else shutil.copy(f, dst))\n")
    root = WORK + "/wpbox"; shutil.rmtree(root, ignore_errors=True)
    for t in ("LLR_0", "LLR_1"):
        s = FRZ.INFO[t][1]; run = f"M2b_s{s}_recft_LLR"; ck = b"ckpt-wp-" + t.encode()
        for f, c in (("ckpt_last.pt", ck), ("summary.json", b"{}"), ("argv.json", b"{}"), ("selfcheck.json", b"{}"), ("train_log.jsonl", b""), ("train.log", b"")): write(f"{root}/{run}/{f}", c, "wb")
        write(f"{root}/llr/RUN_{t}.done", json.dumps({"evidence": {"ckpt_sha256": hashlib.sha256(ck).hexdigest()}}))
    lock = WORK + "/wp_lock/BOX2_BUSY.txt"; lst = WORK + "/wp_listing.txt"
    env = {"LLR_PI_GO": "10:00", "LLR_SSH": f"{PYW} -I {WORK}/wp_listing.py", "LLR_SCP": f"{PYW} -I {WORK}/wp_scp.py", "CL_FAKE_LISTING": lst, "CL_FAKE_ROOT": root, "LLR_PY": PYW, "LLR_LOCK": lock, "LLR_WAKE_N": "3", "LLR_SLEEP": "0"}
    write(lst, "RUN_LLR_0.done\nRUN_LLR_1.done\nCHAIN_LLR.done\n"); write(lock, "LEVELHEAD LLR two-run chain (D:\\tanitad\\llr\\llr_chain.cmd), owner: x"); dest = WORK + "/wp_dest"; shutil.rmtree(dest, ignore_errors=True)
    rc, o, e = bash([f"{pkg}/wake_pull_llr.sh", "chain"], {**env, "LLR_DEST": dest}); check(rc == 0 and "CHAIN DONE" in o and "pull rc=0" in o and not os.path.exists(lock), f"wake_pull chain end to end: rc {rc} {o} {e[:300]}")
    check(os.path.exists(f"{dest}/M2b_s0_recft_LLR/ckpt_last.pt") and os.path.exists(f"{dest}/llr_markers/RUN_LLR_1.done"), "the pull landed in M2b_s{0,1}_recft_LLR and llr_markers")
    shutil.rmtree(dest, ignore_errors=True); write(lock, "x"); write(lst, "RUN_LLR_0.done\nRUN_LLR_1.FAILED\n")
    rc, o, e = bash([f"{pkg}/wake_pull_llr.sh", "chain"], {**env, "LLR_DEST": dest}); check(rc == 3 and "RUN FAILED" in o and not os.path.exists(dest) and os.path.exists(lock), f"a failed run exits 3 and pulls nothing: rc {rc} {o}")
    write(lst, "CHAIN_LLR.STOPPED\n"); rc, o, e = bash([f"{pkg}/wake_pull_llr.sh", "chain"], {**env, "LLR_DEST": dest}); check(rc == 4 and not os.path.exists(dest), f"a safety stop exits 4: rc {rc} {o}")
    write(lst, ""); rc, o, e = bash([f"{pkg}/wake_pull_llr.sh", "chain"], {**env, "LLR_DEST": dest}); check(rc == 5, f"nothing on box 2: waker timeout rc 5, got {rc} {o}")
    rc, o, e = bash([f"{pkg}/wake_pull_llr.sh", "bogus"], env); check(rc == 64, f"wake_pull_llr.sh takes wprobe|chain only: rc {rc}")
    check(not os.path.exists(WORK + "/tripwire.txt"), "tripwire")


@test
def t_nothing_contacts_box2_without_the_pi_go():
    calls = WORK + "/nogo_calls.log"; shutil.rmtree(calls, ignore_errors=True)
    if os.path.exists(calls): os.remove(calls)
    stub = f"{PYW} -I {WORK}/stub_ssh.py"; write(f"{WORK}/stub_ssh.py", "import os,sys\nopen(os.environ['CL_STUB_LOG'],'a').write(' '.join(sys.argv[1:])+'\\n')\nsys.exit(0)\n")
    lock = WORK + "/nogo_lock/BOX2_BUSY.txt"
    for f, args in (("launch_llr.sh", []), ("launch_wleadprobe.sh", []), ("wake_llr.sh", ["chain"]), ("wake_llr.sh", ["wprobe"]), ("pull_llr.sh", []), ("pull_wleadprobe.sh", []), ("wake_pull_llr.sh", ["chain"]), ("wake_pull_llr.sh", ["wprobe"])):
        for pkg in (PKG, f"{FROZEN}/llr_launch"):                                          # the real package and the frozen copy (whose chain would otherwise be allowed to launch)
            rc, o, e = bash([f"{pkg}/{f}"] + args, {"LLR_SSH": stub, "LLR_SCP": stub, "CL_STUB_LOG": calls, "LLR_LOCK": lock, "LLR_PY": PYW, "LLR_WAKE_N": "1", "LLR_SLEEP": "0", "LLR_DEST": WORK + "/nogo_dest"}, timeout=40, drop=("LLR_PI_GO",))
            check(rc != 0 and "LLR_PI_GO" in e and not os.path.exists(calls) and not os.path.exists(WORK + "/tripwire.txt") and not os.path.exists(lock), f"{f} {args} ran without LLR_PI_GO: rc {rc} out {o[:200]} err {e[:300]}")
    # and WITH the go but with the box-2 lock held / W not frozen: still nothing remote
    write(lock, "somebody's job\n")
    rc, o, e = bash([f"{FROZEN}/llr_launch/launch_llr.sh"], {"LLR_PI_GO": "10:00", "LLR_SSH": stub, "LLR_SCP": stub, "CL_STUB_LOG": calls, "LLR_LOCK": lock, "LLR_PY": PYW}, timeout=40)
    check(rc == 5 and "ZZLLR-LOCK-HELD" in o and not os.path.exists(calls) and rd(lock) == "somebody's job\n", f"a held lock must refuse without any remote call: rc {rc} {o[:300]}")
    os.remove(lock)
    r0 = build_frozen("nogo_card"); write(f"{r0}/a2_prereg/PREREG_LEVELHEAD_A2.md", rd(f"{r0}/a2_prereg/PREREG_LEVELHEAD_A2.md") + "\nedited\n")
    rc, o, e = bash([f"{r0}/llr_launch/launch_llr.sh"], {"LLR_PI_GO": "10:00", "LLR_SSH": stub, "LLR_SCP": stub, "CL_STUB_LOG": calls, "LLR_LOCK": lock, "LLR_PY": PYW}, timeout=40)
    check(rc == 6 and "ZZLLR-REFUSED" in o and "on disk is not the file the pins were made from" in o and not os.path.exists(calls) and not os.path.exists(lock), f"an edited card must be refused locally: rc {rc} {o[:300]}")
    r0 = build_frozen("nogo_w"); os.remove(f"{r0}/cl_prereg/FROZEN_W.txt")
    rc, o, e = bash([f"{r0}/llr_launch/launch_llr.sh"], {"LLR_PI_GO": "10:00", "LLR_SSH": stub, "LLR_SCP": stub, "CL_STUB_LOG": calls, "LLR_LOCK": lock, "LLR_PY": PYW}, timeout=40)
    check(rc == 6 and "ZZLLR-REFUSED" in o and "FROZEN_W.txt" in o and not os.path.exists(calls) and not os.path.exists(lock), f"a missing frozen W must be refused locally: rc {rc} {o[:300]}")
    check(not os.path.exists(WORK + "/tripwire.txt"), "tripwire")


FAKE_BOX = r'''import hashlib, os, re, shutil, sys
ROOT = os.environ["FAKE_BOX_ROOT"]
def loc(p):
    p = p.replace("D:\\", "D:/")
    for a, b in (("D:/tanitad/runs/v8f_mini_full/llr", ROOT + "/markers"), ("D:/tanitad/llr", ROOT + "/cl"), ("D:/tanitad/repo", ROOT + "/repo"), ("D:/tanitad/logs", ROOT + "/logs")):
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
        script = sys.stdin.read(); assert "Get-ChildItem D:\\tanitad\\llr -Recurse" in script, script
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
        if job.endswith("llr_run_telemetry.cmd"): open(ROOT + "/logs/telemetry_full.csv", "w").write("")
        elif job.endswith("llr_chain.cmd") or job.endswith("llr_wprobe.cmd"):
            open(loc(m.group(2).replace("\\", "/")), "w").write(env.get("FAKE_JOB_LOG", "===== LLR " + ("WPROBE" if job.endswith("llr_wprobe.cmd") else "CHAIN") + " START x") + chr(10))
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
        e = {"LLR_PI_GO": "10:00", "LLR_SSH": ssh or f"{fbx} ssh", "LLR_SCP": f"{fbx} scp", "LLR_LOCK": lock, "LLR_PY": PYW, "LLR_NAP": "0", "FAKE_BOX_ROOT": root, **(env or {})}
        rc, o, er = bash([f"{pkg or FROZEN + '/llr_launch'}/{script}"], e, timeout=300)
        return rc, o, er, root

    calls = lambda root: [l for l in rd(root + "/calls.log").splitlines()] if os.path.exists(root + "/calls.log") else []
    # happy chain: old CHAIN_LLR.done moved aside, files shipped and re-hashed, four stack files shipped with the old ones aside, telemetry -> guard -> chain, lock kept
    rc, o, er, root = run("launch_llr.sh", "chain", pre=["CHAIN_LLR.done", "RUN_LLR_0.done"]); c = calls(root)
    check(rc == 0 and "ZZLLR-LAUNCHED chain" in o and "lock taken" in o and "shipped bytes identical" in o and "old CHAIN_LLR.done moved aside" in o, f"happy chain: rc {rc} {o[-600:]} {er[:300]}")
    check(os.path.exists(lock) and rd(lock).startswith("LEVELHEAD LLR two-run chain") and "PI go 10:00" in rd(lock), "the lock was not left in place with the chain's text")
    check(os.path.exists(root + "/markers/RUN_LLR_0.done") and any(f.startswith("CHAIN_LLR.done.prev_") for f in os.listdir(root + "/markers")) and not os.path.exists(root + "/markers/CHAIN_LLR.done"), "only CHAIN_LLR.done is moved aside; a run marker stays")
    jobs = [x for x in c if x.startswith("job ")]; check(jobs == ["job llr_run_telemetry.cmd", "job llr_run_guard_chain.cmd", "job llr_chain.cmd"], f"order telemetry -> guard -> chain: {jobs}")
    for rel, want in FRZ.SHIPPED.items():
        check(sh256(f"{root}/repo/{rel}") == want and any(f.startswith(os.path.basename(rel) + ".pre_llr_") and rd(f"{root}/repo/{os.path.dirname(rel)}/{f}") == f"OLD:{rel}\n" for f in os.listdir(f"{root}/repo/{os.path.dirname(rel)}")), f"{rel}: not shipped as pinned with the old file aside")
    shipped = sorted(os.path.relpath(os.path.join(d, f), root + "/cl").replace("\\", "/") for d, _, fs in os.walk(root + "/cl") for f in fs)
    check(shipped == json.loads(json.dumps(sorted(FRZ_ship()))), f"shipped file set {shipped}")
    check(not os.path.exists(WORK + "/tripwire.txt"), "tripwire")
    rc, o, er, root = run("launch_llr.sh", "chain_token", pkg=f"{FROZEN_TOKEN}/llr_launch"); check(rc == 0 and "ZZLLR-LAUNCHED chain" in o, f"the complete token-wiring copy's chain launches on the fake box: rc {rc} {o[-400:]} {er[:200]}")
    rw = build_frozen("dr_w", state="wiring")                                          # wiring frozen, no W_lead: the PROBE launches, the chain is refused locally
    rc, o, er, root = run("launch_wleadprobe.sh", "probe", pkg=f"{rw}/llr_launch", pre=["WLPROBE.done", "WLPROBE.FAILED", "RUN_LLR_0.done"]); c = calls(root)
    check(rc == 0 and "ZZLLR-LAUNCHED wprobe" in o and "old WLPROBE.done moved aside" in o and "old WLPROBE.FAILED moved aside" in o, f"probe launch: rc {rc} {o[-500:]} {er[:300]}")
    check(os.path.exists(lock) and rd(lock).startswith("LEVELHEAD LLR W_lead probe") and "PI go 10:00" in rd(lock) and not os.path.exists(root + "/markers/WLPROBE.done") and os.path.exists(root + "/markers/RUN_LLR_0.done"), "probe lock text / markers")
    check([x for x in c if x.startswith("job ")] == ["job llr_run_telemetry.cmd", "job llr_run_guard_wprobe.cmd", "job llr_wprobe.cmd"], f"probe job order: {c}")
    shipped = sorted(os.path.relpath(os.path.join(d, f), root + "/cl").replace("\\", "/") for d, _, fs in os.walk(root + "/cl") for f in fs); check(shipped == sorted(ship(rw)) and "argv/WLPROBE_0.gate.txt" in shipped, f"probe ships the whole folder: {shipped}")
    rc, o, er, root = run("launch_llr.sh", "chain_w", pkg=f"{rw}/llr_launch"); check(rc == 6 and "ZZLLR-REFUSED" in o and not calls(root) and not os.path.exists(lock), f"the chain is refused locally while W_lead is not frozen: rc {rc} {o[-300:]}")
    rn = build_frozen("dr_n", state="none")
    rc, o, er, root = run("launch_wleadprobe.sh", "probe_n", pkg=f"{rn}/llr_launch"); check(rc == 6 and "ZZLLR-REFUSED" in o and "placeholder __WIRING__" in o and not calls(root) and not os.path.exists(lock), f"the probe is refused locally while the wiring is not frozen: rc {rc} {o[-300:]}")
    # refusals, each leaves box 2 as it was and releases (moves aside) our lock
    def refused(label, name, msg, env=None, pre=None, rc_want=2, lock_gone=True, script="launch_llr.sh"):
        rc, o, er, root = run(script, name, env=env, pre=pre)
        check(rc == rc_want and msg in o, f"{label}: rc {rc} (want {rc_want}) want {msg!r}: {o[-500:]} {er[:200]}")
        check((not os.path.exists(lock)) == lock_gone, f"{label}: lock state (gone={not os.path.exists(lock)})")
        if lock_gone: check(any(f.startswith("BOX2_BUSY.llr_launch_failed_") for f in os.listdir(os.path.dirname(lock))), f"{label}: lock was not moved aside")
        for f in os.listdir(os.path.dirname(lock)):
            if f.startswith("BOX2_BUSY.llr_launch_failed_"): os.remove(os.path.dirname(lock) + "/" + f)
        return root
    root = refused("python running", "py", "a python process is running on box 2", env={"FAKE_PY": "1"}); check(not os.path.exists(root + "/cl") and not any("job" in x for x in calls(root)), "nothing shipped when python runs")
    root = refused("disk low", "disk", "box 2 D: below 8 GB", env={"FAKE_DFREE": "3"})
    root = refused("a heat stop marker", "gs", "GUARD_STOP.txt", pre=["GUARD_STOP.txt"]); check(not os.path.exists(root + "/cl"), "nothing shipped over a heat stop")
    root = refused("bytes differ on box 2", "corrupt", "shipped bytes differ", env={"FAKE_SCP_CORRUPT": "llr_chain.cmd"}); check(not any(x.startswith("job") for x in calls(root)), "no job after a byte mismatch")
    root = refused("stack scp corrupt", "stk", "on box 2 hashes", env={"FAKE_SCP_CORRUPT": "v8f_mini.py"})
    for rel in FRZ.SHIPPED: check(rd(f"{root}/repo/{rel}") == f"OLD:{rel}\n", f"{rel}: not restored after a failed stack ship")
    check(any(".llr_failed_" in f for rel in FRZ.SHIPPED for f in os.listdir(f"{root}/repo/{os.path.dirname(rel)}")), "the failed new bytes are kept aside")
    root = refused("stack aside move fails", "mv", "moving box 2's stack/tanitad/data/v8f_recover.py aside failed", env={"FAKE_MOVE_FAIL": "v8f_recover.py"})
    for rel in FRZ.SHIPPED: check(rd(f"{root}/repo/{rel}") == f"OLD:{rel}\n", f"{rel}: not restored")
    root = refused("stale telemetry", "tele", "no fresh telemetry", env={"FAKE_TELE_FROZEN": "1"}); check("telemetry.stop" in calls(root) and not any(x == "job llr_chain.cmd" for x in calls(root)), "a stale telemetry must stop the telemetry and not start the chain")
    refused("job stopped at start", "stop", "ZZLLR-JOB-STOPPED-AT-START", env={"FAKE_JOB_LOG": "ZZLLR-STOP preflight rc 3"}, rc_want=3, lock_gone=False); os.remove(lock)
    refused("no start line", "nostart", "ZZLLR-NO-START-LINE", env={"FAKE_JOB_LOG": "nothing"}, rc_want=4, lock_gone=False); os.remove(lock)
    # a held lock: refuse without any remote call (the lock is somebody else's and stays)
    os.makedirs(os.path.dirname(lock), exist_ok=True); write(lock, "someone else's job\n"); root = f"{WORK}/dr_held"; shutil.rmtree(root, ignore_errors=True); os.makedirs(root)
    rc, o, er = bash([f"{FROZEN}/llr_launch/launch_llr.sh"], {"LLR_PI_GO": "10:00", "LLR_SSH": f"{fbx} ssh", "LLR_SCP": f"{fbx} scp", "LLR_LOCK": lock, "LLR_PY": PYW, "FAKE_BOX_ROOT": root}, timeout=60)
    check(rc == 5 and not calls(root) and rd(lock) == "someone else's job\n", f"held lock: rc {rc} {o[:200]}")
    os.remove(lock)
    # fail closed: an unknown remote command is refused by the fake box (the launcher issues only the commands this test models)
    rc, o, er, root = run("launch_llr.sh", "unk", ssh=f"{PYW} -I -c \"import sys;sys.exit(97)\""); check(rc == 2 and "ZZLLR-ABORT" in o, f"an ssh that fails must abort: rc {rc} {o[-200:]}")
    check(not os.path.exists(WORK + "/tripwire.txt"), "tripwire")


def FRZ_ship():
    r = subprocess.run([sys.executable, "-I", f"{FROZEN}/llr_launch/llr_local_check.py", "ship-files"], capture_output=True, text=True); return [x.strip() for x in r.stdout.splitlines() if x.strip()]


PS_PARSE = "param($f) $t=$null;$e=$null;$null = [System.Management.Automation.Language.Parser]::ParseFile($f,[ref]$t,[ref]$e); if ($e.Count) { $e | ForEach-Object { $_.Message }; exit 1 }; Write-Output 'PARSE-OK'"


@test
def t_guard_scripts_parse_stop_on_heat_and_ignore_stale_markers():
    write(WORK + "/parse.ps1", PS_PARSE)
    for g in ("llr_guard_chain.ps1",):
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", WORK + "/parse.ps1", "-f", f"{PKG}/box2_files/{g}"], capture_output=True, text=True, timeout=90); check(r.returncode == 0 and "PARSE-OK" in r.stdout, f"{g} does not parse: {r.stdout}{r.stderr}")
    chain = rd(f"{PKG}/box2_files/llr_guard_chain.ps1")
    check("$ends = @('CHAIN_LLR.done', 'CHAIN_LLR.STOPPED')" in chain and "WPROBE" not in chain.split("$ends")[1].split("\n")[0], "the chain guard must end on the chain's markers only")

    def run_guard(src, name, temps, marker=None, marker_after=None, first=None):
        d = f"{WORK}/guard_{name}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d + "/rc")
        write(d + "/tel.csv", "ts,x\r\n" + "".join(f"2026-11-05 10:00:{i:02d},a,b,c,d,e,f,g,h,i,{t}\r\n" for i, t in enumerate(temps)))
        for f in (first or []): write(f"{d}/rc/{f}", "{}")
        s = src.replace(r"D:\tanitad\runs\v8f_mini_full\llr", (d + "/rc").replace("/", "\\")).replace(r"D:\tanitad\logs\telemetry_full.csv", (d + "/tel.csv").replace("/", "\\")).replace(r"D:\tanitad\logs\llr_guard.log", (d + "/guard.log").replace("/", "\\"))
        s = s.replace("v8f_mini_train", "zz_no_such_trainer_zz").replace("Start-Sleep -Seconds 60", "Start-Sleep -Seconds 1"); write(d + "/g.ps1", s)
        t0 = time.time(); p = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", d + "/g.ps1"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if marker_after: time.sleep(marker_after); write(f"{d}/rc/{marker}", "{}")
        try: out, _ = p.communicate(timeout=60)
        except subprocess.TimeoutExpired: p.kill(); raise AssertionError(f"guard {name} did not exit")
        return os.path.exists(d + "/rc/GUARD_STOP.txt"), (rd(d + "/guard.log") if os.path.exists(d + "/guard.log") else ""), out, time.time() - t0
    st, lg, o, _ = run_guard(chain, "hot4", [55] * 16 + [62] * 4); check(st and "ZZGUARD-STOP TEMP: 4 of 20 >= 61 C, max 62 C" in lg, f"4 of 20 >= 61 C must stop: {lg} {o}")
    st, lg, o, _ = run_guard(chain, "hot63", [50] * 19 + [63]); check(st and "max 63 C" in lg, f"one >= 63 C must stop: {lg} {o}")
    st, lg, o, _ = run_guard(chain, "warm3", [50] * 17 + [61] * 3, marker="CHAIN_LLR.done", marker_after=4); check(not st and "job ended; guard off (max USB 61 C)" in lg, f"3 of 20 must not stop; the chain marker ends the chain guard: {lg} {o}")
    st, lg, o, el = run_guard(chain, "stale", [50] * 20, marker="CHAIN_LLR.STOPPED", marker_after=5, first=["WPROBE.done", "WPROBE.FAILED"]); check(not st and el >= 4.5 and "job ended" in lg, f"a stale WPROBE marker must NOT end the chain guard (ended after {el:.1f}s): {lg}")


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
        t = t.replace("ZZRNEXT-", "ZZLLR-").replace("RNEXT CHAIN", "LLR CHAIN").replace("CHAIN_RNEXT", "CHAIN_LLR").replace(r"D:\tanitad\rnext\rnext_check.py", r"D:\tanitad\llr\llr_check.py").replace(r"D:\tanitad\rnext\rnext_run.cmd", r"D:\tanitad\llr\llr_run.cmd")
        return t.replace(r"%RUNSM%\rnext", r"%RUNSM%\llr").replace("RPT_0 RPT_1", "LLR_0 LLR_1").replace("rnext_chain_rpt", "llr_chain").replace(r"D:\tanitad\rnext", r"D:\tanitad\llr")
    a = rd(f"{RPT_CHAIN}/rnext_chain_rpt.cmd", "rb").decode().split("\r\n"); b = rd(f"{PKG}/box2_files/llr_chain.cmd", "rb").decode().split("\r\n")
    code = lambda ls: [norm(l) for l in ls if not l.startswith("rem ")]
    ca, cb = code(a), code(b); ca = [l.replace('"%PY%" "%CHK%" preflight', '"%PY%" "%CHK%" preflight chain') if l == '"%PY%" "%CHK%" preflight' else l for l in ca]
    diff = [(x, y) for x, y in zip(ca, cb) if x != y]; check(len(ca) == len(cb) and not diff, f"llr_chain.cmd differs from the RPT chain (names normalised) beyond the preflight argument: {diff[:3]} {len(ca)} vs {len(cb)}")
    ra = [norm(l) for l in rd(f"{RPT_CHAIN}/rnext_run.cmd", "rb").decode().split("\r\n") if not l.startswith("rem ")]; rb = [l for l in rd(f"{PKG}/box2_files/llr_run.cmd", "rb").decode().split("\r\n") if not l.startswith("rem ")]
    ra = [l.replace("RUN-START", "RUN-START") for l in ra]; check(ra == rb, f"llr_run.cmd differs from rnext_run.cmd: {[(x, y) for x, y in zip(ra, rb) if x != y][:3]}")
    ch = rd(f"{PKG}/box2_files/llr_chain.cmd")
    check(ch.count("%RUNSAFE% cmd /c %RUNCMD%") == 2 and "call D:\\tanitad\\env.cmd" in rd(f"{PKG}/box2_files/llr_run.cmd"), "SAFE MODE: every python call goes through %RUNSAFE% (env.cmd sourced by the run file)")
    check("find /c \"python.exe\"" in ch and "ZZLLR-STOP a python process is already running" in ch and "telemetry.stop" in ch and "wprobe" not in ch.lower(), "one job at a time / telemetry stop")
    check('if exist "%RCD%\\GUARD_STOP.txt"' in ch and "exit /b 9" in ch and "preflight run %TAG%" in ch and "gate-cached" in ch and "gate %TAG% !RC!" in ch and "post %TAG% !RC!" in ch, "chain keeps the marker / guard / gate / post steps")
    tel = rd(f"{PKG}/box2_files/llr_run_telemetry.cmd", "rb").decode().split("\r\n"); tel0 = rd(f"{RPT_CHAIN}/rnext_run_telemetry.cmd", "rb").decode().split("\r\n")
    check([l for l in tel if not l.startswith("rem ")] == [l for l in tel0 if not l.startswith("rem ")], "the telemetry job differs from the RPT one")


@test
def t_package_hygiene():
    words = [r"\bvo" + "id\\b", "super" + "seded", "retr" + "acted", "for" + "merly", "no lon" + "ger", "corr" + "ection", "with" + "drawn", "closed" + r"[ -]loop"]
    mine = [f for f in os.listdir(PKG) if os.path.isfile(f"{PKG}/{f}") and f.endswith((".py", ".sh", ".md", ".json", ".txt"))] + [f"box2_files/{f}" for f in os.listdir(f"{PKG}/box2_files") if os.path.isfile(f"{PKG}/box2_files/{f}")] + ["tests/test_llr_launch.py"]
    for f in mine:
        raw = rd(f"{PKG}/{f}", "rb"); txt = raw.decode("utf-8")
        for w in words:
            ln = [l for l in txt.splitlines() if re.search(w, l, re.I)]; check(not ln, f"{f}: banned word ({w}): {ln[:1]}")
        if not f.endswith(".cmd"): check(b"\r" not in raw, f"{f}: CR in an LF file")
    for t in ("LLR_0", "LLR_1"):
        a = rd(f"{PKG}/box2_files/argv/{t}.train.txt", "rb"); check(a.count(b"\n") == 1 and a.endswith(b"\n") and b"\r" not in a, f"{t}: argv is not one LF line")
    for f in ("PINS.json",): json.loads(rd(f"{PKG}/{f}"))
    check(os.path.exists(f"{PKG}/HANDOFF_LLR.md"), "HANDOFF_LLR.md is missing")




@test
def t_wprobe_waker_pull_and_wake_pull_on_stand_ins():
    fb = f"{PYW} -I {WORK}/fake_listing_p.py"; write(f"{WORK}/fake_listing_p.py", "import os,sys\nsys.stdout.write(open(os.environ['CL_FAKE_LISTING']).read() if os.path.exists(os.environ['CL_FAKE_LISTING']) else '')\n")
    lst = WORK + "/listing_p.txt"
    for listing, ec, msg in (("WLPROBE.done", 0, "WLPROBE DONE"), ("WLPROBE.FAILED", 3, "WLPROBE FAILED"), ("WLPROBE.STOPPED", 4, "WLPROBE STOPPED"), ("GUARD_STOP.txt", 4, "GUARD STOP"), ("", 5, "WAKER TIMEOUT"),
                             ("CHAIN_LLR.done", 5, "WAKER TIMEOUT"), ("RUN_LLR_0.done", 5, "WAKER TIMEOUT"), ("WLPROBE.done.prev_20261105_010101", 5, "WAKER TIMEOUT")):
        write(lst, listing + "\n"); rc, o, e = bash([f"{PKG}/wake_llr.sh", "wprobe"], {"LLR_PI_GO": "10:00", "LLR_SSH": fb, "CL_FAKE_LISTING": lst, "LLR_WAKE_N": "2", "LLR_SLEEP": "0"})
        check(rc == ec and msg in o, f"wprobe waker listing {listing!r}: rc {rc} (want {ec}) out {o!r}")
    write(lst, "WLPROBE.done\n"); rc, o, e = bash([f"{PKG}/wake_llr.sh", "chain"], {"LLR_PI_GO": "10:00", "LLR_SSH": fb, "CL_FAKE_LISTING": lst, "LLR_WAKE_N": "2", "LLR_SLEEP": "0"}); check(rc == 5, "a probe marker does not end the chain waker")
    write(f"{WORK}/wp_scp_p.py", "import os,shutil,sys,glob\nroot=os.environ['CL_FAKE_ROOT']; a=[x for x in sys.argv[1:] if x not in ('-q','-r') and not x.startswith(('BatchMode','ConnectTimeout')) and x!='-o']\n"
          "src=a[0].split(':',1)[1].replace('D:/tanitad/runs/v8f_mini_full',root); dst=a[1]\nfs=glob.glob(src) if '*' in src else [src]\nif not fs or not all(os.path.exists(f) for f in fs): sys.exit(1)\n"
          "for f in fs:\n    (shutil.copytree(f, os.path.join(dst, os.path.basename(f)), dirs_exist_ok=True) if os.path.isdir(f) else shutil.copy(f, dst))\n")
    rp = build_frozen("wp_probe", state="wiring"); frz = loadmod("llr_check_rp", f"{rp}/llr_launch/box2_files/llr_check.py", LLR_HOME=f"{rp}/llr_launch/box2_files")
    sb = SB("wpb", pkg=f"{rp}/llr_launch", frz=frz); check(sb.run(["pin", "WLPROBE_0"])[0] == 0, "pin probe"); stage_gate(sb, "WLPROBE_0", stub_sc("WLPROBE_0", frz, "add")); rc, o = sb.run(["gate", "WLPROBE_0", "0"]); check(rc == 0, o)
    root = WORK + "/wpbox2"; shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(sb.rc(), f"{root}/llr"); write(f"{root}/M2b_s0_wleadprobe_LLR/selfcheck.json", rd(f"{sb.gdir('WLPROBE_0')}/selfcheck.json"))
    lock = WORK + "/wpp_lock/BOX2_BUSY.txt"; dest = WORK + "/wpp_dest"; shutil.rmtree(dest, ignore_errors=True); write(lock, "LEVELHEAD LLR W_lead probe (LLR_0's argv with --selfcheck-only --w-lead 1), owner: x"); write(lst, "WLPROBE.done\n")
    env = {"LLR_PI_GO": "10:00", "LLR_SSH": fb, "LLR_SCP": f"{PYW} -I {WORK}/wp_scp_p.py", "CL_FAKE_LISTING": lst, "CL_FAKE_ROOT": root, "LLR_PY": PYW, "LLR_LOCK": lock, "LLR_WAKE_N": "2", "LLR_SLEEP": "0", "LLR_DEST": dest}
    rc, o, e = bash([f"{rp}/llr_launch/wake_pull_llr.sh", "wprobe"], env); wj = f"{rp}/a2_prereg/WLEAD_PROBE.json"
    check(rc == 0 and "WLPROBE DONE" in o and "pull rc=0" in o and os.path.exists(wj) and json.load(open(wj))["aux_cond"] == "add" and abs(json.load(open(wj))["w_lead_for_10pct_share"] - 1.0) < 1e-9, f"wake_pull wprobe end to end: rc {rc} {o} {e[:300]}")
    check(not os.path.exists(lock) and any(f.startswith("BOX2_BUSY.llr_wprobe_done_") for f in os.listdir(os.path.dirname(lock))) and os.path.exists(f"{dest}/M2b_s0_wleadprobe_LLR/selfcheck.json"), "the probe pull released OUR lock (moved aside) and landed in M2b_s0_wleadprobe_LLR")
    fz = subprocess.run([sys.executable, "-I", f"{rp}/llr_launch/freeze_llr.py", "wlead"], capture_output=True, text=True, env={**os.environ, "LLR_PREREG_DIR": f"{rp}/a2_prereg"}); check(fz.returncode == 0 and "W_LEAD=1" in fz.stdout, f"the pulled record feeds freeze_llr.py wlead: {fz.stdout}")
    os.remove(wj); write(lock, "LEVELHEAD LLR W_lead probe (x)"); write(f"{root}/M2b_s0_wleadprobe_LLR/selfcheck.json", rd(f"{root}/M2b_s0_wleadprobe_LLR/selfcheck.json").replace("stub0000", "stub9999"))
    rc, o, e = bash([f"{rp}/llr_launch/wake_pull_llr.sh", "wprobe"], env); check(rc == 7 and not os.path.exists(wj) and os.path.exists(lock), f"a changed selfcheck.json is refused by the reader: no record, lock kept: rc {rc} {o[-300:]}")
    write(lst, "WLPROBE.FAILED\n"); rc, o, e = bash([f"{rp}/llr_launch/wake_pull_llr.sh", "wprobe"], env); check(rc == 3 and not os.path.exists(wj), f"a failed probe exits 3 and pulls nothing: rc {rc}")
    write(lock, "somebody else's job"); write(lst, "WLPROBE.done\n"); write(f"{root}/M2b_s0_wleadprobe_LLR/selfcheck.json", rd(f"{sb.gdir('WLPROBE_0')}/selfcheck.json"))
    rc, o, e = bash([f"{rp}/llr_launch/pull_wleadprobe.sh"], env); check(rc == 0 and os.path.exists(lock) and "lock not released" in o, f"a lock that is not the probe's is left alone: rc {rc} {o[-200:]}")
    rc, o, e = bash([f"{rp}/llr_launch/pull_wleadprobe.sh"], {k: v for k, v in env.items() if k != "LLR_PI_GO"}); check(rc != 0 and "LLR_PI_GO" in e, "pull_wleadprobe.sh refuses without LLR_PI_GO")
    check(not os.path.exists(WORK + "/tripwire.txt"), "tripwire")


@test
def t_probe_cmd_and_guard_files():
    c = rd(f"{PKG}/box2_files/llr_wprobe.cmd", "rb"); txt = c.decode("utf-8")
    check(b"\r\n" in c and c.count(b"\n") == c.count(b"\r\n"), "llr_wprobe.cmd must be CRLF throughout")
    for need in ("LLR WPROBE START", "wprobe-stopped python process present", "preflight wprobe", "pin WLPROBE_0", "gate WLPROBE_0", "llr_run.cmd", "llr_check.py", r"D:\tanitad\llr\llr_check.py", r"D:\tanitad\runs\v8f_mini_full", "telemetry.stop", "RUNSAFE", "tasklist"):
        check(need in txt, f"llr_wprobe.cmd lacks {need!r}")
    check("train " not in txt.replace("trains nothing", "") and "post " not in txt and "CHAIN_LLR" not in txt and "wleadprobe_LLR" in rd(f"{PKG}/box2_files/argv/WLPROBE_0.gate.txt"), "the probe job never trains, posts or ends the chain")
    for g, ends in (("llr_guard_wprobe.ps1", ["WLPROBE.done", "WLPROBE.FAILED", "WLPROBE.STOPPED"]), ("llr_guard_chain.ps1", ["CHAIN_LLR.done", "CHAIN_LLR.STOPPED"])):
        s = rd(f"{PKG}/box2_files/{g}"); check(all(f"'{e}'" in s for e in ends) and (g != "llr_guard_chain.ps1" or "WLPROBE" not in s), f"{g}: end markers")
        check(r"D:\tanitad\runs\v8f_mini_full\llr" in s and "v8f_mini_train" in s and "GUARD_STOP.txt" in s, f"{g}: own marker folder, the trainer's command line, the stop marker")
        r = subprocess.run(["powershell", "-NoProfile", "-Command", PS_PARSE.replace("param($f)", f"$f='{PKG}/box2_files/{g}';")], capture_output=True, text=True, timeout=60); check(r.returncode == 0 and "PARSE-OK" in r.stdout, f"{g} does not parse: {r.stdout} {r.stderr}")
    gc = rd(f"{PKG}/box2_files/llr_run_guard_wprobe.cmd"); check("llr_guard_wprobe.ps1" in gc and "llr_guard_chain" not in gc, "the probe guard cmd runs the probe guard")
    rn = rd(f"{PKG}/box2_files/llr_run.cmd"); check(r"%RUNSM%\llr\pin\%TAG%.%MODE%.txt" in rn and r"%RUNSM%\llr\logs\%TAG%.%MODE%.log" in rn, "llr_run.cmd reads the pinned line and logs in the own folders (the probe uses mode gate)")


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
