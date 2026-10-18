#!/usr/bin/env python3
"""Tests for the REFe launch gate (gate_token.py, gate_checks.py, launch_gate.py, gate_fixture.py).  Contract: eval/REFE_LAUNCH_GATE.md.

  --static   (seconds, no trainer run)  the token, the profiles, the refusals, the registry, the cheap checks and their regression arms.
             Prints `ZZLG_STATIC_PASS <n>`.  This is what the gate's own G-SUITE runs (with --no-banked: the pod has no banked copy).
  --fp-identity-only  (seconds) just the block that proves the faster fingerprint is the SAME fingerprint (the frozen pre-2026-10-15 code is the oracle).  Prints `ZZLG_FP_IDENTITY_PASS <n>`.
  default    static + FULL (minutes, CPU): the real trainer run in-process on a tiny rehearsal bank -- every check PASSes clean and every
             regression arm makes ITS check FAIL; an end-to-end run -> evidence -> token -> verify; the P6 refusal.  Prints `ZZLG_FULL_PASS <n>`.
A check that cannot go RED is not a check: every positive assertion below has its negative twin.  Any skip prints ZZTEST_SKIP and is NOT counted.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))
import gate_checks as C  # noqa: E402
import gate_fixture as F  # noqa: E402
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402

BANKED = Path("D:/Projects/TanitAD/products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_ckpt/code/fix/stack/scripts/launch_gate.py")
N = 0
POD = "--no-banked" in sys.argv          # the gate's own G-SUITE run on the pod: no banked generic gate, no git


def ok(cond, msg=""):
    global N
    assert cond, msg
    N += 1


def real_args() -> str:
    m = re.search(r'^ARGS="(.*)"$', (HERE / "p2r_pod.sh").read_text(encoding="utf-8"), re.M)
    assert m
    return m.group(1)


def posix_bash():
    """Absolute path of a REAL POSIX bash, or None (SO-A16).  A bare `bash` given to CreateProcess is not safe on a Windows box: CreateProcess searches System32 BEFORE PATH, and
    C:\\Windows\\System32\\bash.exe (and the WindowsApps alias) is the WSL launcher, which drops the argument after `-c '<script>'`, so the word-splitting cross-check read an EMPTY stdout and
    failed -- while `shutil.which("bash")` (the old guard) answered Git's bash.  So resolve once, to an absolute path, and skip the WSL launcher wherever it sits on PATH.
    POSIX: the first `bash` on PATH."""
    if os.name != "nt":
        return shutil.which("bash")

    def wsl_launcher(p):
        q = os.path.normcase(os.path.abspath(p))
        sysroot = os.path.normcase(os.environ.get("SystemRoot", "C:\\Windows"))
        return q.startswith(sysroot + os.sep) or (os.sep + "windowsapps" + os.sep) in q

    cands = [os.path.join(d, "bash.exe") for d in os.environ.get("PATH", "").split(os.pathsep) if d]
    git = shutil.which("git")
    roots = [os.path.dirname(os.path.dirname(git))] if git else []
    roots += [os.path.join(os.environ[v], "Git") for v in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)") if os.environ.get(v)]
    for r in roots:
        cands += [os.path.join(r, "usr", "bin", "bash.exe"), os.path.join(r, "bin", "bash.exe")]
    for c in cands:
        if os.path.isfile(c) and not wsl_launcher(c):
            return os.path.abspath(c)
    return None


P6_ARGS = ("--backbone vitl16 --targets /w/train_grow --images /w/pix --calib /w/calib.json --metric-cache /w/mc --scorer-sees-trunk --yaw-wrap off --nav-mode command "
           "--labeller online --label-chunks 2 --epochs 25 --batch 4 --accum 64 --lr 2e-4 --weight-decay 0.01 --amp bf16 --score-w 0.1 --tf32 --compile --workers 2 "
           "--log-every 1 --ckpt-every-min 20 --out /w/run --resume")


def mini_pkg(root: Path) -> Path:
    for rel, txt in (("refe/a.py", "print(1)\n"), ("refe/b.py", "print(2)\n"), ("code/c.sh", "echo 3\n"), ("splits/d.yaml", "x: 1\n")):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8", newline="\n")
    return root


TEST_PROFILE = {**L.PROFILES["refe_p2"], "name": "refe_p2", "required_flags": {}, "provisional_flags": (), "require_commit": False,
                "checks": ("G-HYG", "G-DVB", "G-SUITE"), "data_flags": ("--targets",), "forbidden_flags": ()}


def evid(check, status="PASS", **kw):
    return C.evidence(check, status, kw.pop("reasons", [] if status == "PASS" else ["x"]), {}, **kw)


def static() -> None:
    # ------------------------------------------------------------------ S1 primitives == the generic gate's, byte for byte
    if BANKED.is_file() and "--no-banked" not in sys.argv:
        spec = importlib.util.spec_from_file_location("banked_launch_gate", BANKED)
        BG = importlib.util.module_from_spec(spec)
        sys.modules["banked_launch_gate"] = BG
        spec.loader.exec_module(BG)
        ok(BG.GATE_SCHEMA == GT.GATE_SCHEMA)
        for argv in (["--a", "1", "--b"], ["--lr", "5e-5", "--out", "/x/y"], []):
            ok(BG.argv_sha256(argv) == GT.argv_sha256(argv))
        obj = {"b": [1, 2, {"z": "ü"}], "a": 1.5}
        ok(BG.canonical_json(obj) == GT.canonical_json(obj))
        key = bytes(range(32))
        ok(BG.sign(obj, key) == GT.sign(obj, key) and BG.key_id(key) == GT.key_id(key))
        tok = {**obj, "hmac_sha256": GT.sign(obj, key)}
        ok(BG.signature_ok(tok, key) and GT.signature_ok(tok, key))
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "d").mkdir()
            (td / "d" / "f.txt").write_text("hi")
            (td / "g.bin").write_bytes(b"\x00\x01")
            for p in (td / "g.bin", td / "d", td / "missing"):
                fa, fb = BG.fingerprint(p), GT.fingerprint(p)
                ok(fa == fb and BG.fp_key(fa) == GT.fp_key(fb), f"fingerprint {p.name}")
    elif "--no-banked" not in sys.argv:             # an EXPLICIT --no-banked (the pod has no banked copy) is not a skip; an absent file otherwise is
        print("ZZTEST_SKIP banked generic gate not present: primitives not cross-checked here")
    # ------------------------------------------------------------------ S2 mint / verify and every way a token must be refused
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pkg = mini_pkg(td / "pkg")
        data_file = td / "data.bin"
        data_file.write_bytes(b"bank")
        keyf = td / "k" / "gate.key"
        key = GT.load_key(keyf, create=True, forbid_under=[pkg])
        ok(re.fullmatch(r"[0-9a-f]{64}", keyf.read_text().strip()) is not None)
        try:
            GT.load_key(pkg / "refe" / "gate.key", create=True, forbid_under=[pkg])
            ok(False, "a key inside the code tree was accepted")
        except GT.GateError:
            ok(True)
        argv = ["--targets", str(data_file), "--lr", "5e-5", "--epochs", "3"]
        tree = GT.tree_manifest(pkg)
        ok(set(tree) == {"refe/a.py", "refe/b.py", "code/c.sh", "splits/d.yaml"})
        data = GT.data_manifest(argv, ("--targets",))
        verdict, path, tok = GT.mint(profile="refe_p2", commit="c" * 40, tree_man=tree, argv=argv, trainer="refe/train.py", data=data,
                                      checks={"G-HYG": {"status": "PASS"}}, reasons=[], key=key, out_dir=td / "out")
        ok(verdict == "PASS" and path.name == f"PASS_{'c' * 12}.json")

        def ver(argv2=None, **kw):
            return GT.verify_token(kw.pop("token", path), argv2 if argv2 is not None else argv, kw.pop("pkg", pkg), key=kw.pop("key", key),
                                   data_flags=("--targets",), **kw)
        good, why, _ = ver()
        ok(good and not why, why)
        ok(not ver(argv[:-1] + ["4"])[0] and "argv differs" in " ".join(ver(argv[:-1] + ["4"])[1]))          # a changed value
        ok(not ver(["--lr", "5e-5", "--targets", str(data_file), "--epochs", "3"])[0])                        # the same flags, reordered
        ok(not ver(argv + ["--resume"])[0])                                                                   # an extra flag
        ok(ver(profile="refe_p6")[1][0].startswith("token profile"))
        ok(not ver(commit="d" * 40)[0])
        (pkg / "refe" / "b.py").write_text("print(99)\n", encoding="utf-8", newline="\n")                                   # a shipped file edited
        ok(not ver()[0] and any("code tree differs" in r for r in ver()[1]))
        (pkg / "refe" / "b.py").write_text("print(2)\n", encoding="utf-8", newline="\n")
        ok(ver()[0])
        (pkg / "refe" / "new.py").write_text("x\n", encoding="utf-8", newline="\n")                                         # an extra file
        ok(not ver()[0])
        (pkg / "refe" / "new.py").unlink()
        data_file.write_bytes(b"BANK")                                                                        # the data edited after the gate
        ok(not ver()[0] and any("data differs" in r for r in ver()[1]))
        data_file.write_bytes(b"bank")
        ok(ver()[0])
        ok(not ver(key=bytes(32))[0])                                                                         # another host's key
        raw = json.loads(path.read_text(encoding="utf-8"))
        for mut in (lambda t: t.update(verdict="FAIL"), lambda t: t["binding"].update(argv_sha256="0" * 64), lambda t: t["checks"].update(X={})):
            t2 = copy.deepcopy(raw)
            mut(t2)
            (td / "edited.json").write_text(json.dumps(t2), encoding="utf-8")
            ok(not ver(token=td / "edited.json")[0])
        t3 = copy.deepcopy(raw)                                                                               # an edit RE-SIGNED with the wrong key
        t3["verdict"] = "PASS"
        t3["hmac_sha256"] = GT.sign({k: v for k, v in t3.items() if k != "hmac_sha256"}, bytes(32))
        (td / "forged.json").write_text(json.dumps(t3), encoding="utf-8")
        ok(not ver(token=td / "forged.json")[0])
        ok(not ver(token=td / "nope.json")[0])
        (td / "garbage.json").write_text("not json")
        ok(not ver(token=td / "garbage.json")[0])
        v2, p2, _ = GT.mint(profile="refe_p2", commit="c" * 40, tree_man=tree, argv=argv, trainer="t", data=data,
                            checks={"G-HYG": {"status": "FAIL"}}, reasons=["x"], key=key, out_dir=td / "out")
        ok(v2 == "FAIL" and not ver(token=p2)[0] and not (td / "out" / f"PASS_{'c' * 12}.json").exists()                # a FAIL demotes the stale PASS
           and list((td / "out" / "superseded").glob("PASS_*")))
        # ---------------------------------------------------------------- S3 finalize: the verdict comes from the evidence ON DISK
        P = copy.deepcopy(TEST_PROFILE)
        out = td / "fin"

        def write_all(prof, argv_, skip=(), mutate=None, commit=None):
            shutil.rmtree(out, ignore_errors=True)
            tr = GT.tree_manifest(pkg)
            binding = GT.binding_of(prof["name"], commit, GT.tree_digest(tr), argv_)
            dat = GT.data_manifest(argv_, prof["data_flags"])
            for c in prof["checks"]:
                if c in skip:
                    continue
                ev = evid(c)
                if mutate:
                    mutate(c, ev)
                L.write_evidence(out, ev, binding, dat)
            for c in prof.get("na_ok", ()):
                if c not in skip:
                    L.write_evidence(out, C.evidence(c, "NA", ["na"], {}), binding, dat)
        P["na_ok"] = ()
        write_all(P, argv)
        v, pth, t = L.finalize(P, argv, pkg, out, key)
        ok(v == "PASS" and GT.verify_token(pth, argv, pkg, key=key, data_flags=("--targets",))[0], t["reasons"])
        write_all(P, argv, skip=("G-SUITE",))
        v, _, t = L.finalize(P, argv, pkg, out, key)
        ok(v == "FAIL" and any("no evidence file" in r for r in t["reasons"]))
        write_all(P, argv, mutate=lambda c, ev: ev.update(status="FAIL", reasons=["boom"]) if c == "G-DVB" else None)
        ok(L.finalize(P, argv, pkg, out, key)[0] == "FAIL")
        write_all(P, argv, mutate=lambda c, ev: ev.update(arm="lever_ignored") if c == "G-HYG" else None)
        v, _, t = L.finalize(P, argv, pkg, out, key)
        ok(v == "FAIL" and any("regression arm" in r for r in t["reasons"]))
        write_all(P, argv, mutate=lambda c, ev: ev.update(rehearsal=True))
        v, _, t = L.finalize(P, argv, pkg, out, key)
        ok(v == "FAIL" and any("REHEARSAL" in r for r in t["reasons"]))
        write_all(P, argv)                                                                                    # evidence for another argv
        v, _, t = L.finalize(P, ["--targets", str(data_file), "--lr", "1e-4", "--epochs", "3"], pkg, out, key)
        ok(v == "FAIL" and any("binding differs" in r for r in t["reasons"]))
        write_all(P, argv)
        data_file.write_bytes(b"changed after the checks")                                                    # the data moved between check and mint
        v, _, t = L.finalize(P, argv, pkg, out, key)
        ok(v == "FAIL" and any("different fingerprint" in r for r in t["reasons"]))
        data_file.write_bytes(b"bank")
        write_all(P, argv)
        os.environ["TANITAD_GATE_ARM"] = "zero_score"
        try:
            v, _, t = L.finalize(P, argv, pkg, out, key)
        finally:
            del os.environ["TANITAD_GATE_ARM"]
        ok(v == "FAIL" and any("an arm run can never PASS" in r for r in t["reasons"]))
        write_all(P, argv)
        ok(L.finalize({**P, "mintable": False}, argv, pkg, out, key)[0] == "FAIL")
        ok(L.finalize({**P, "require_commit": True}, argv, pkg, out, key)[0] == "FAIL")                      # no commit bound
        # the commit comes from the SHIP_MANIFEST, which the pod's tree must equal
        GT.write_json(pkg / GT.SHIP_MANIFEST, GT.ship_manifest(pkg, commit=None) | {"commit": "e" * 40})
        write_all(P, argv, commit="e" * 40)
        v, pth, t = L.finalize({**P, "require_commit": True}, argv, pkg, out, key)
        ok(v == "PASS" and t["binding"]["commit"] == "e" * 40, t["reasons"])
        (pkg / "refe" / "a.py").write_text("print('edited on the pod')\n", encoding="utf-8", newline="\n")
        v, _, t = L.finalize({**P, "require_commit": True}, argv, pkg, out, key)
        ok(v == "FAIL" and any("not the shipped tree" in r for r in t["reasons"]))
        (pkg / "refe" / "a.py").write_text("print(1)\n", encoding="utf-8", newline="\n")
        (pkg / GT.SHIP_MANIFEST).unlink()
        # G-CLOCK: NA is admitted for P2, never for P6
        write_all({**P, "na_ok": ("G-CLOCK",)}, argv)
        ok(L.finalize({**P, "na_ok": ("G-CLOCK",)}, argv, pkg, out, key)[0] == "PASS")
        write_all({**P, "na_ok": ("G-CLOCK",)}, argv, skip=("G-CLOCK",))
        ok(L.finalize({**P, "na_ok": ("G-CLOCK",)}, argv, pkg, out, key)[0] == "FAIL")
        # ---------------------------------------------------------------- S6 ship-manifest: the commit names bytes that exist (DEV BOX only: the pod has no git)
        if not POD:
            repo = td / "repo"
            shutil.copytree(pkg, repo / "pkgdir")
            g = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *a], capture_output=True, text=True)  # noqa: E731
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            g("add", "-A")
            g("commit", "-q", "-m", "x")
            sha = g("rev-parse", "HEAD").stdout.strip()
            man = GT.ship_manifest(repo / "pkgdir", commit=sha, git_dir=str(repo / ".git"), repo_prefix="pkgdir/")
            ok(man["commit"] == sha and len(man["files"]) == 4)
            (repo / "pkgdir" / "refe" / "b.py").write_text("print('dirty')\n", encoding="utf-8", newline="\n")
            for why_, fn in (("a file edited since the commit", None),):
                try:
                    GT.ship_manifest(repo / "pkgdir", commit=sha, git_dir=str(repo / ".git"), repo_prefix="pkgdir/")
                    ok(False, "a dirty file shipped under a commit")
                except GT.GateError as e:
                    ok("differs from its blob" in str(e))
            (repo / "pkgdir" / "refe" / "b.py").write_text("print(2)\n", encoding="utf-8", newline="\n")
            (repo / "pkgdir" / "refe" / "unlanded.py").write_text("x\n", encoding="utf-8", newline="\n")
            try:
                GT.ship_manifest(repo / "pkgdir", commit=sha, git_dir=str(repo / ".git"), repo_prefix="pkgdir/")
                ok(False, "an unlanded file shipped under a commit")
            except GT.GateError as e:
                ok("not in" in str(e))
            try:
                GT.ship_manifest(repo / "pkgdir", commit="abc", git_dir=str(repo / ".git"))
                ok(False)
            except GT.GateError:
                ok(True)
    # ------------------------------------------------------------------ S4 profiles: P2 and P6 are clean on their real ARGS; an INCOMPLETE P6 argv is refused
    ra = L.parse_args_string(real_args())
    ok(L.profile_refusals(L.PROFILES["refe_p2"], ra) == [], L.profile_refusals(L.PROFILES["refe_p2"], ra))
    for bad_args, frag in ((GT.argv_from_string(real_args().replace("--lr 5e-5", "--lr 1e-4")), "--lr"),
                           (ra + ["--synthetic"], "FORBIDS"), (ra + ["--n-cameras", "1"], "FORBIDS"),
                           ([t for t in ra if t != "--scorer-sees-trunk"], "--scorer-sees-trunk"),
                           ([t for t in ra if t != "--pdm-only"], "--pdm-only")):
        rs = L.profile_refusals(L.PROFILES["refe_p2"], bad_args)
        ok(rs and any(frag in r for r in rs), (frag, rs))
    p6 = L.parse_args_string(P6_ARGS)
    ok(L.profile_refusals(L.PROFILES["refe_p6"], p6) == [], L.profile_refusals(L.PROFILES["refe_p6"], p6))             # P6 no longer refuses: D1/D5/D6 are built
    rc = subprocess.run([sys.executable, str(HERE / "launch_gate.py"), "refusals", "--profile", "refe_p6", "--args", P6_ARGS], capture_output=True, text=True,
                        encoding="utf-8")
    ok(rc.returncode == 0 and "ZZGATE-CLEANZZ" in rc.stdout, rc.stdout)
    rc2 = subprocess.run([sys.executable, str(HERE / "launch_gate.py"), "refusals", "--profile", "refe_p2", "--args", real_args()], capture_output=True, text=True,
                         encoding="utf-8")
    ok(rc2.returncode == 0 and "ZZGATE-CLEANZZ" in rc2.stdout, rc2.stdout)
    # an argv that does NOT name the open ruling (--yaw-wrap) or the labeller is refused, exits 5, and no PASS token can exist
    p6_bad = [t for t in p6 if t not in ("--yaw-wrap", "off")]
    rs = L.profile_refusals(L.PROFILES["refe_p6"], p6_bad)
    ok(any("--yaw-wrap" in r for r in rs), rs)
    rc3 = subprocess.run([sys.executable, str(HERE / "launch_gate.py"), "refusals", "--profile", "refe_p6", "--args", " ".join(p6_bad)], capture_output=True, text=True,
                         encoding="utf-8")
    ok(rc3.returncode == 5 and "ZZGATE-REFUSEDZZ" in rc3.stdout)
    # the loss terms the profile requires are emitted by the trainer (a declared term the trainer never emits is a launch blocker)
    src_train = (HERE / "train.py").read_text(encoding="utf-8")
    ok(all(('"' + t + '"') in src_train for t in L.P6_TERMS), [t for t in L.P6_TERMS if ('"' + t + '"') not in src_train])
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        key = GT.load_key(td / "k" / "gate.key", forbid_under=[PKG])
        out = td / "p6"
        st = L.run_checks(L.PROFILES["refe_p6"], p6_bad, PKG, out)
        ok(set(st.values()) == {"FAIL"} and set(st) == set(L.PROFILES["refe_p6"]["checks"]), st)
        v, pth, t = L.finalize(L.PROFILES["refe_p6"], p6_bad, PKG, out, key)
        ok(v == "FAIL" and not list(out.glob("PASS_*")) and any("--yaw-wrap" in r for r in t["reasons"]), t["reasons"][:3])
        good, why, _ = GT.verify_token(pth, p6_bad, PKG, key=key, profile="refe_p6")
        ok(not good and any("not PASS" in r for r in why))
    # ------------------------------------------------------------------ S5 the registry and the cheap checks, each with its red arm
    import train as T
    dests = {x.dest for x in T.build_parser()._actions if x.dest != "help"}
    ok(set(C.DVB_KINDS) == dests, (sorted(dests - set(C.DVB_KINDS)), sorted(set(C.DVB_KINDS) - dests)))
    ok(len(dests) == 61)                                                                                      # 55 + the five L_field levers (--w-field, --field-margin, --field-interp, --field-labels, --field-shuffle-seed) + --label-chunks (2026-10-18)
    ok(C.namespace_reads(C._train_run_source()) <= dests)
    ok(C.check_hyg()["status"] == "PASS")
    for arm, frag in (("no_cfg_guard", "ACCEPTED an undeclared attribute"), ("undeclared_namespace_read", "undeclared_lever_g_hyg")):
        ev = C.check_hyg(arm=arm)
        ok(ev["status"] == "FAIL" and frag in " ".join(ev["reasons"]), (arm, ev["reasons"]))
    ok(C.check_hyg()["status"] == "PASS")                                                                      # the arm left no residue
    dummy = SimpleNamespace(parser_unknown=[], rc="X", error=None, log="-")
    prof = L.PROFILES["refe_p2"]
    for arm, frag in (("unknown_flag", "NO declared-vs-built entry"), ("stale_registry", "no longer has")):
        ev = C.check_dvb(ra, dummy, None, prof, arm=arm)
        ok(ev["status"] == "FAIL" and frag in " ".join(ev["reasons"]), (arm, ev["reasons"]))
    # a REAL new flag on the parser, no arm: the gate itself refuses (CLAUDE.md: a new trainer flag without a G-DVB entry is refused)
    orig_bp = T.build_parser

    def bp_new():
        ap = orig_bp()
        ap.add_argument("--brand-new-lever", type=float, default=0.0)
        return ap
    T.build_parser = bp_new
    try:
        ev = C.check_dvb(ra, dummy, None, prof)
        ok(ev["status"] == "FAIL" and "brand_new_lever" in " ".join(ev["reasons"]))
    finally:
        T.build_parser = orig_bp
    # P6's flags are rejected by the parser itself -> G-DVB FAIL (not a pass on a partial parse)
    pb = SimpleNamespace(parser_unknown=["--yaw-wrap", "off"], rc="UNRECOGNISED_ARGUMENTS", error="x", log="-")
    ev = C.check_dvb(p6, pb, None, L.PROFILES["refe_p6"])
    ok(ev["status"] == "FAIL" and "does not build" in " ".join(ev["reasons"]))
    ok(C.instrumented_run(p6 + ["--not-a-flag", "1"], mode="build", work=tempfile.mkdtemp()).rc == "UNRECOGNISED_ARGUMENTS")
    # D0: the config refuses an undeclared attribute and still copies / pickles
    import pickle
    from model import REFeConfig
    cfg = REFeConfig.for_backbone("vits16")
    try:
        cfg.undistrot = False
        ok(False)
    except AttributeError:
        ok(True)
    ok(pickle.loads(pickle.dumps(cfg)) == cfg and copy.deepcopy(cfg) == cfg)
    # the SHELL's word splitting of $ARGS is the gate's
    bash = posix_bash()
    if bash:
        r = subprocess.run([bash, "-c", 'ARGS="$1"; printf "%s\\n" $ARGS', "_", real_args()], capture_output=True, text=True, encoding="utf-8")
        ok(r.stdout.split("\n")[:-1] == GT.argv_from_string(real_args()), (bash, r.returncode, r.stdout, r.stderr))
    else:
        print("ZZTEST_SKIP no POSIX bash (none, or only the WSL launcher): $ARGS word splitting not cross-checked")
    # derived argvs: BUILD keeps every launch lever, SMOKE cuts only the schedule and points at the gate's small data
    w = Path(tempfile.mkdtemp())
    bld, smk = L.derived_argvs(L.PROFILES["refe_p2"], ra, w, "/gs", "/gp", "/gb")
    ok(GT.flag_values(bld, "--epochs") == ["3"] and GT.flag_values(bld, "--accum") == ["64"] and GT.flag_values(bld, "--onpolicy-targets") == ["/gs"]
       and GT.flag_values(bld, "--targets") == [ra[ra.index("--targets") + 1]] and "--resume" not in bld and "--compile" in bld)
    ok("--epochs" not in smk and GT.flag_values(smk, "--steps") == ["2"] and GT.flag_values(smk, "--accum") == ["2"] and GT.flag_values(smk, "--workers") == ["0"]
       and GT.flag_values(smk, "--targets") == ["/gb"] and GT.flag_values(smk, "--lr") == ["5e-5"] and "--scorer-sees-trunk" in smk and "--amp" in smk)
    # p2_gate.py's own output is consumed as a sub-check: parsed, all four present and ok
    gp = Path(tempfile.mkdtemp()) / "gate.json"
    full = {"G1_a": {"ok": True}, "G2_b": {"ok": True}, "G3_pdm_only": {"ok": True}, "G4_real": {"ok": True}}
    for gates, verdict_, want in ((full, "PASS", "PASS"), (full, "FAIL", "FAIL"), ({**full, "G3_pdm_only": {"ok": False}}, "PASS", "FAIL"),
                                  ({k: v for k, v in full.items() if not k.startswith("G4")}, "PASS", "FAIL"), ({**full, "G2_b": {}}, "PASS", "FAIL")):
        gp.write_text(json.dumps({"verdict": verdict_, "gates": gates}), encoding="utf-8")
        ok(L.consume_p2_gate(gp)["status"] == want, (gates, verdict_))
    ok(L.consume_p2_gate(gp.parent / "absent.json")["status"] == "FAIL")
    # the chain's ARGS are the profile's design, byte for byte (the token binds this string)
    ok(real_args().count('"') == 0 and "  " not in real_args())
    rls1_static()
    aux_s1_static()
    field_static()
    fingerprint_identity_static()


# ====================================================================================================== the FASTER FINGERPRINT is the SAME fingerprint (2026-10-15)
# gate_token.fingerprint / sha256_file / data_manifest were made faster for the GPU-idle gap before every S1 arm (raw/2026-10-15-s1-gate-gap/REPORT.md): the 413,184-file image listing is
# stat'ed on worker threads, the independent inputs are fingerprinted concurrently, and the checkpoint is hashed through a reusable buffer.  A token binds these digests, so the change is only
# admissible if EVERY digest is bit-identical to the old serial one.  The `_old_*` functions below are the pre-change implementations, VERBATIM (with their constants frozen here too, so a
# change to gate_token's constants cannot move the oracle with it), kept as the oracle.  The comparison runs over fixtures that exercise every branch (nesting, empty dirs, skip names, odd
# names, a directory spanning many stat tasks, symlinks where this box has them, a missing path, a flag named twice, path_map, gate inputs) at several thread counts, and every
# digest-altering mutation of the new code must make it RED.
_OLD_DIR_SKIP_NAMES = frozenset({"_v2manifest.pt", "_v2manifest.pt.tmp", "__pycache__"})
_OLD_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache"})
_OLD_TREE_SUFFIX = (".py", ".sh", ".yaml", ".yml", ".json", ".txt", ".patch")
_OLD_TREE_ROOTS = ("refe", "code", "splits")


def _old_sha256_file(p, chunk: int = 1 << 22) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def _old_fingerprint(path) -> dict:
    import hashlib
    p = Path(path)
    if p.is_file():
        return {"kind": "file", "bytes": p.stat().st_size, "sha256": _old_sha256_file(p), "method": "sha256-v1"}
    if p.is_dir():
        lines, n, total = [], 0, 0
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = sorted(d for d in dirnames if d not in _OLD_DIR_SKIP_NAMES)
            for fn in filenames:
                if fn in _OLD_DIR_SKIP_NAMES:
                    continue
                f = Path(dirpath) / fn
                st = f.stat()
                lines.append(f"{f.relative_to(p).as_posix()}\t{st.st_size}\n")
                n += 1
                total += st.st_size
        lines.sort()
        return {"kind": "dir", "n_files": n, "bytes": total, "listing_sha256": hashlib.sha256("".join(lines).encode("utf-8")).hexdigest(),
                "method": "dir-listing-v1 (relpath + size of every file; contents NOT hashed; _v2manifest.pt excluded -- a cache "
                          "rewrites it on first use)"}
    return {"kind": "missing"}


def _old_data_manifest(argv, data_flags, gate_inputs=None, path_map=None) -> dict:
    man: dict = {}
    want = set(data_flags)
    for flag, vals in GT.flag_pairs(argv):
        if flag in want and vals:
            local = vals[0]
            for a, b in (path_map or []):
                if local.startswith(a):
                    local = b + local[len(a):]
            man[flag] = {"path": vals[0], **({"local": local} if local != vals[0] else {}), **_old_fingerprint(local)}
    for key, path in sorted((gate_inputs or {}).items()):
        man[key] = {"path": str(path), **_old_fingerprint(path)}
    return man


def _old_tree_manifest(pkg) -> dict:
    pkg = Path(pkg)
    out: dict = {}
    for root in _OLD_TREE_ROOTS:
        base = pkg / root
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in _OLD_SKIP_DIRS)
            for fn in sorted(filenames):
                if fn.endswith(_OLD_TREE_SUFFIX):
                    f = Path(dirpath) / fn
                    out[f.relative_to(pkg).as_posix()] = _old_sha256_file(f)
    return out


def fp_fixtures(td: Path) -> dict:
    """build the directories / files the identity oracle runs over under `td`; returns {name: path} plus the flags `symlinks` / `posix_names`"""
    fx: dict = {}

    def put(rel, data=b"x"):
        p = td / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p
    for i, n in enumerate((0, 1, 7, 4095, 4096, 4097, 100000)):
        put(f"flat/f{i}.bin", bytes(range(256)) * (n // 256) + bytes(range(n % 256)))
    for rel in ("a/1.txt", "a/b/2.txt", "a/b/c/3.txt", "a/b/c/d/4.txt", "a/z.txt", "m/n/o/5.txt", "m/6.txt", "top.txt", "same/x.txt", "other/same/x.txt", "a/same/x.txt"):
        put("nested/" + rel, rel.encode() * 3)
    for i in range(41):                                                  # many stat tasks for ONE directory once the test shrinks the chunk
        put(f"big/f{i:03d}.dat", b"y" * i)
    for rel in ("_v2manifest.pt", "_v2manifest.pt.tmp", "keep.txt", "__pycache__/x.pyc", "sub/keep2.txt", "sub/__pycache__/y.pyc", "sub/_v2manifest.pt", "sub/deeper/__pycache__/z.pyc"):
        put("skips/" + rel, rel.encode())
    for rel in ("with space.txt", "ünï.txt", "a#b.txt", "x'y.txt", "dir with space/é.txt", "100%.txt", "a,b.txt", "UPPER.TXT", "lower.txt"):
        put("names/" + rel, rel.encode("utf-8"))
    fx["posix_names"] = False
    if os.name != "nt":
        for rel in ("tab\tname.txt", "back\\slash.txt", "new\nline.txt", "-dash.txt"):
            try:
                put("names/" + rel, b"p")
                fx["posix_names"] = True
            except OSError:
                pass
    (td / "empty").mkdir()
    (td / "emptynest" / "a" / "b").mkdir(parents=True)
    put("emptynest/a/only.txt", b"o")
    fx["symlinks"] = False
    try:
        (td / "links").mkdir()
        put("links/real.txt", b"real-file")
        put("links/realdir/inner.txt", b"inner")
        os.symlink(str(td / "links" / "real.txt"), str(td / "links" / "to_file"))
        os.symlink(str(td / "links" / "realdir"), str(td / "links" / "to_dir"), target_is_directory=True)
        fx["symlinks"] = True
    except (OSError, NotImplementedError):
        (td / "links").mkdir(exist_ok=True)
    for k in ("flat", "nested", "big", "skips", "names", "empty", "emptynest", "links"):
        fx[k] = td / k
    fx["whole"] = td
    return fx


def fp_violations(fp_fn, sha_fn, dm_fn, tm_fn, fx: dict, pkg: Path, threads=("1", "2", "16"), thorough: bool = True) -> list:
    """every way the candidate functions differ from the frozen pre-change ones, over the fixtures `fx`.  [] = bit-identical.  `thorough` = every path form (str / Path / trailing
    separator / relative) and every thread count; the mutation runs use the cheap form."""
    v: list = []
    td = fx["whole"]
    saved_env, saved_chunk = os.environ.get(GT.FP_THREADS_ENV), GT.FP_STAT_CHUNK
    GT.FP_STAT_CHUNK = 4
    try:
        for nthr in threads:
            os.environ[GT.FP_THREADS_ENV] = nthr
            for name in ("flat", "nested", "big", "skips", "names", "empty", "emptynest", "links", "whole"):
                p = fx[name]
                for form in ([str(p)] if not thorough else [str(p), p, str(p) + os.sep]):
                    if _old_fingerprint(form) != fp_fn(form):
                        v.append(f"fingerprint[{name}] threads={nthr} form={type(form).__name__}")
            if _old_fingerprint(td / "absent") != fp_fn(td / "absent"):
                v.append("fingerprint[missing]")
            for fl in ("flat/f0.bin", "flat/f6.bin", "nested/top.txt"):
                if _old_fingerprint(td / fl) != fp_fn(td / fl):
                    v.append(f"fingerprint[file {fl}]")
            if thorough:
                cwd = os.getcwd()
                os.chdir(str(td))
                try:
                    for rel in ("nested", "./nested", "nested/a"):
                        if _old_fingerprint(rel) != fp_fn(rel):
                            v.append(f"fingerprint[relative {rel}] threads={nthr}")
                finally:
                    os.chdir(cwd)
            argv = ["--targets", str(fx["flat"]), "--images", str(fx["nested"]), "--calib", str(td / "flat" / "f4.bin"), "--targets", str(fx["skips"]), "--gone", str(td / "absent"),
                    "--onpolicy", str(fx["big"]), "--noval", "--names", str(fx["names"]), "--ck", str(td / "flat" / "f6.bin"), "--lane", str(fx["links"])]
            flags = ("--targets", "--images", "--calib", "--gone", "--onpolicy", "--noval", "--names", "--ck", "--lane")
            gi = {"gate:b": str(td / "flat" / "f5.bin"), "gate:a": str(fx["empty"]), "gate:c": str(td / "absent")}
            cases = [(None, argv)]
            if thorough:                                                 # path_map: the argv names a path the box does not have, mapped onto one it does
                cases.append(([(str(td) + os.sep + "nested", str(fx["flat"]))], ["--targets", str(td) + os.sep + "nested"] + argv[2:]))
            for pmap, ag in cases:
                o, n = _old_data_manifest(ag, flags, gi, pmap), dm_fn(ag, flags, gi, pmap)
                if o != n or list(o) != list(n):
                    v.append(f"data_manifest threads={nthr} path_map={bool(pmap)}")
            if _old_data_manifest([], flags) != dm_fn([], flags) or _old_data_manifest(argv, ()) != dm_fn(argv, ()):
                v.append(f"data_manifest[empty] threads={nthr}")
            if tm_fn(pkg) != _old_tree_manifest(pkg):
                v.append(f"tree_manifest threads={nthr}")
        for size in (0, 1, 4095, 4096, 4097, 3 * 4096 + 5, (1 << 22) + 3):
            f = td / f"hash_{size}.bin"
            f.write_bytes((bytes(range(251)) * (size // 251 + 1))[:size])
            want = _old_sha256_file(f)
            for chunk in ((4096, 1 << 22) if size <= 3 * 4096 + 5 else (1 << 22,)):
                if sha_fn(f, chunk) != want or _old_sha256_file(f, chunk) != want:
                    v.append(f"sha256_file size={size} chunk={chunk}")
            if sha_fn(f) != want:
                v.append(f"sha256_file size={size} default chunk")
            f.unlink()
        if fx.get("symlinks"):                                           # a dangling symlink: both raise, the same error class
            dang = td / "links" / "dangling"
            os.symlink(str(td / "links" / "nowhere"), str(dang))
            try:
                errs = []
                for fn in (_old_fingerprint, fp_fn):
                    try:
                        fn(td / "links")
                        errs.append(None)
                    except OSError as e:
                        errs.append(type(e).__name__)
                if errs[0] != errs[1]:
                    v.append(f"dangling symlink: old {errs[0]} vs new {errs[1]}")
            finally:
                dang.unlink()
    finally:
        GT.FP_STAT_CHUNK = saved_chunk
        if saved_env is None:
            os.environ.pop(GT.FP_THREADS_ENV, None)
        else:
            os.environ[GT.FP_THREADS_ENV] = saved_env
    return v


def fingerprint_identity_static() -> None:
    import contextlib
    import hashlib
    with tempfile.TemporaryDirectory() as td_:
        td = Path(td_)
        fx = fp_fixtures(td / "fx")
        pkg = td / "pkg"
        for rel, txt in (("refe/a.py", "print(1)\n"), ("refe/sub/b.sh", "echo\n"), ("refe/__pycache__/c.pyc", "x"), ("refe/notes.md", "n"), ("code/c.json", "{}"), ("code/d.patch", "p"),
                         ("splits/e.yaml", "x: 1\n"), ("splits/f.yml", "y: 2\n"), ("refe/.pytest_cache/g.txt", "g"), ("other/h.py", "h"), ("refe/big.txt", "z" * 300000)):
            (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
            (pkg / rel).write_text(txt, encoding="utf-8", newline="\n")
        # the oracle is itself stable and sees what the fixtures were built to show
        ok(_old_fingerprint(fx["nested"]) == _old_fingerprint(fx["nested"]) and _old_fingerprint(fx["nested"])["n_files"] == 11)
        ok(_old_fingerprint(fx["skips"])["n_files"] == 2 and _old_fingerprint(fx["empty"])["n_files"] == 0 and _old_fingerprint(fx["empty"])["kind"] == "dir")
        ok(_old_tree_manifest(pkg).keys() == {"refe/a.py", "refe/sub/b.sh", "code/c.json", "code/d.patch", "splits/e.yaml", "splits/f.yml", "refe/big.txt"})
        # 1. the new code == the frozen old code, everywhere, at every thread count
        v = fp_violations(GT.fingerprint, GT.sha256_file, GT.data_manifest, GT.tree_manifest, fx, pkg)
        ok(v == [], v)
        ok(GT.fp_threads() >= 1)
        # 2. ... and a digest-altering change to the new code is RED: each mutation below makes the same oracle report a violation
        orig = {k: getattr(GT, k) for k in ("_stat_lines", "sha256_file", "_DIR_SKIP_NAMES", "data_manifest")}

        @contextlib.contextmanager
        def mutated(**kw):
            for k, f in kw.items():
                setattr(GT, k, f)
            try:
                yield
            finally:
                for k, f in orig.items():
                    setattr(GT, k, f)

        def cheap():
            return fp_violations(GT.fingerprint, GT.sha256_file, GT.data_manifest, GT.tree_manifest, fx, pkg, threads=("16",), thorough=False)

        def no_prefix(dirpath, rel_prefix, names):                      # relative path dropped: two files with one name look like one
            return orig["_stat_lines"](dirpath, "", names)

        def short_tail(dirpath, rel_prefix, names):                     # the last file of every stat task is lost
            return orig["_stat_lines"](dirpath, rel_prefix, names[:-1] if len(names) > 1 else names)

        def size_plus_one(dirpath, rel_prefix, names):                  # a size off by one on every file
            ls, t = orig["_stat_lines"](dirpath, rel_prefix, names)
            return [f"{ln.rsplit(chr(9), 1)[0]}\t{int(ln.rsplit(chr(9), 1)[1]) + 1}\n" for ln in ls], t + len(ls)

        def lose_last_byte(p, chunk: int = 1 << 22):                    # a digest that drops one byte of the last read
            h = hashlib.sha256()
            with open(p, "rb", buffering=0) as fh:
                buf = bytearray(chunk)
                mv = memoryview(buf)
                while True:
                    n = fh.readinto(mv)
                    if not n:
                        break
                    h.update(mv[:n - 1] if 1 < n < chunk else mv[:n])
            return h.hexdigest()

        def dm_no_gate_inputs(argv, data_flags, gate_inputs=None, path_map=None):
            return orig["data_manifest"](argv, data_flags, None, path_map)
        for name, kw in (("relative path dropped from the listing", dict(_stat_lines=no_prefix)),
                         ("last file of each stat task lost", dict(_stat_lines=short_tail)),
                         ("every size off by one", dict(_stat_lines=size_plus_one)),
                         ("sha256 loses a byte of the last read", dict(sha256_file=lose_last_byte)),
                         ("skip names no longer pruned", dict(_DIR_SKIP_NAMES=frozenset())),
                         ("gate inputs dropped from the data manifest", dict(data_manifest=dm_no_gate_inputs))):
            with mutated(**kw):
                red = cheap()
            ok(len(red) > 0, f"mutation not caught: {name}")
        if fx["symlinks"]:
            def lstat_lines(dirpath, rel_prefix, names):                 # symlinks NOT followed: a link's own size, not its target's
                lines, total = [], 0
                for fn in names:
                    size = os.lstat(os.path.join(dirpath, fn)).st_size
                    lines.append(f"{rel_prefix}{fn}\t{size}\n")
                    total += size
                return lines, total
            with mutated(_stat_lines=lstat_lines):
                ok(len(cheap()) > 0, "mutation not caught: symlinks no longer followed")
        # 3. the mutated state was restored: the clean run is clean again
        ok(cheap() == [])
        # 4. serial == threaded, on a directory that spans many stat tasks, and the thread-count switch is sane
        os.environ[GT.FP_THREADS_ENV] = "1"
        s1 = GT.fingerprint(fx["big"])
        os.environ[GT.FP_THREADS_ENV] = "8"
        GT.FP_STAT_CHUNK = 3
        try:
            s8 = GT.fingerprint(fx["big"])
        finally:
            GT.FP_STAT_CHUNK = 256
            os.environ.pop(GT.FP_THREADS_ENV, None)
        ok(s1 == s8 == _old_fingerprint(fx["big"]) and s1["n_files"] == 41)
        for junk in ("junk", "0", "-3", ""):
            os.environ[GT.FP_THREADS_ENV] = junk
            ok(GT.fp_threads() == GT.FP_THREADS_DEFAULT, junk)
        os.environ[GT.FP_THREADS_ENV] = "1"
        ok(GT.fp_threads() == 1)
        os.environ.pop(GT.FP_THREADS_ENV, None)
        # 5. the token path end to end: a token minted from the OLD manifests verifies against the NEW ones, and a file added afterwards still refuses
        key = bytes(range(32))
        argv = ["--targets", str(fx["nested"]), "--calib", str(td / "fx" / "flat" / "f6.bin")]
        _, path, tok = GT.mint(profile="refe_x", commit="a" * 40, tree_man=_old_tree_manifest(pkg), argv=argv, trainer="refe/train.py",
                               data=_old_data_manifest(argv, ("--targets", "--calib")), checks={"G": {"status": "PASS"}}, reasons=[], key=key, out_dir=td / "out")
        good, why, _ = GT.verify_token(path, argv, pkg, key=key, data_flags=("--targets", "--calib"))
        ok(good and not why, why)
        late = td / "fx" / "nested" / "a" / "new_file.txt"
        late.write_text("late", encoding="utf-8")
        ok(not GT.verify_token(path, argv, pkg, key=key, data_flags=("--targets", "--calib"))[0])
        late.unlink()
        ok(GT.verify_token(path, argv, pkg, key=key, data_flags=("--targets", "--calib"))[0])


# ====================================================================================================== RL-S1: the refe_rls1 profile
# RL-S1 trains with scorer_finetune.py (the EPO loss), not train.py: the profile names that parser and source, classifies every one of ITS flags (DVB_KINDS_SCORER, a table
# SEPARATE from DVB_KINDS -- S5 pins that one equal to train.py's 60), states the arm rules, and consumes the trainer's own `--preflight` (G1-G11) as parsed JSON.
# Every refusal below has its positive twin (the clean argv) and every check its regression arm.  Nothing here runs the suite (that would recurse into this file).
RLS1_BASE = ("--ckpt /w/model_sft4_B_u400.pt --backbone vitl16 --targets /w/train --images /w/pix --calib /w/calib.json --onpolicy /w/onp --heldout /w/onh "
             "--heldout-logs /w/hl.txt --out /w/rl --lane-head --teacher-lane-train /w/tl --teacher-lane-heldout /w/tlh --pdm-labels-train /w/pdm "
             "--pdm-labels-heldout /w/pdmh --require-pdm --b-mode listnet --lam 0 --lr 1e-4 --max-updates 1000 --epochs 4 --workers 3 --seed 0")


def rls1_gate_keys() -> list:
    """The preflight's gate keys as scorer_finetune.py's SOURCE spells them (the fixtures are built from the trainer's own names, not retyped)."""
    return re.findall(r'gates\["(G\d+_[A-Za-z0-9_]+)"\]', (HERE / "scorer_finetune.py").read_text(encoding="utf-8"))


def rls1_pf_rows(argv, *, verdict="PASS", drop=(), notok=(), with_data=True, g11=None, add=None, data_mut=None) -> list:
    """The two rows `scorer_finetune --preflight` appends to preflight.jsonl (REAL shape: verified against main() end to end, see HANDOFF_RL_S1) for this argv."""
    import scorer_finetune as S
    ns, err = C.parse_scorer_argv(S.build_parser(), argv)
    assert err is None, err
    req = C.scorer_required_gates(ns)
    gates = {}
    for k in rls1_gate_keys():
        p = k.split("_", 1)[0]
        if (p == "G4" and "compw" in k) or p not in req or p in drop:
            continue
        gates[k] = {"ok": k not in notok}
    if "G11" in req and "G11" not in drop:
        mods = C.SCORER_GROUPS
        k11 = next(k for k in rls1_gate_keys() if k.startswith("G11_"))
        gates[k11] = g11 or {
            "ok": True, "per_module_on": {m: 1.5 for m in mods}, "per_module_lam0": {m: 0.0 for m in mods}, "per_module_off": {m: 0.0 for m in mods},
            "live": {m: True for m in mods}, "controls_exactly_zero": True, "value_on": -0.3, "J_on": 0.3, "kl_step0": 0.0, "kl_tol": 0.0, "kl_ok": True,
            "mutate_detach": False, "rows_have_terms": True, "heldout_terms": None}
        if k11 in notok:
            gates[k11] = {**gates[k11], "ok": False}
    gates.update(add or {})
    rows = []
    if with_data:
        d = {"event": "data", "b_mode": ns.b_mode, "b_compw": ns.b_compw, "train_sets": 100, "heldout_sets": 12}
        if ns.epo:
            d["epo"] = {k: getattr(ns, "epo_" + k) for k in C.SCORER_EPO_KEYS}
        d.update(data_mut or {})
        rows.append(d)
    rows.append({"event": "preflight", "verdict": verdict, "gates": gates, "s_per_microbatch": 1.2, "gpu_mem_gb": 20.0})
    return rows


def rls1_write(path, rows, jsonl=True):
    path = Path(path)
    path.write_text(("\n".join(json.dumps(r) for r in rows) + "\n") if jsonl else json.dumps(rows[-1], indent=1), encoding="utf-8")
    return path


def rls1_static() -> None:
    import scorer_finetune as S
    P = L.PROFILES["refe_rls1"]
    base = L.parse_args_string(RLS1_BASE)
    r0 = base + ["--epo", "--epo-terms-train", "/w/terms"]
    r2 = r0 + ["--epo-k", "--epo-l"]
    c0 = list(base)

    def refs(argv, arm=None, prof=None):
        return L.profile_refusals(prof or ({**P, "declared_arm": arm} if arm else P), argv)

    def clean(argv, arm=None):
        rs = refs(argv, arm)
        ok(rs == [], (arm, rs))

    def red(argv, frag, arm=None, prof=None):
        rs = refs(argv, arm, prof)
        ok(any(frag in r for r in rs), (frag, arm, rs))

    def without(argv, flag, nvalues=1):
        return C._drop(argv, flag, nvalues)
    # ------------------------------------------------------------------ R1 the profile reads ITS trainer; every other profile still reads train.py
    for nm in ("refe_p2", "refe_p6", "refe_rehearsal"):
        ok(L.PROFILES[nm]["parser_module"] == "train" and L.PROFILES[nm]["trainer_src"] == "train.py" and not L.PROFILES[nm].get("arm_rules"), nm)
    ok(P["parser_module"] == "scorer_finetune" and P["trainer_src"] == "scorer_finetune.py" and P["trainer"] == "refe/scorer_finetune.py" and P["runner"] == "scorer")
    ok(P["namespace_funcs"] == C.SCORER_NAMESPACE_FUNCS and "G-PREFLIGHT" in L.CHECKS)
    ok(all(c in L.CHECKS for p in L.PROFILES.values() for c in tuple(p["checks"]) + tuple(p.get("na_ok", ()))), "a profile names a check the gate does not know")
    ok(P["mintable"] and P["require_commit"] and not P["rehearsal"])
    for e in L.RLS1_SUITE:                                                # a renamed PASS token would make G-SUITE fail on the pod, not here
        ok((HERE / e["script"]).is_file() and e["token"] in (HERE / e["script"]).read_text(encoding="utf-8"), e)
    # ------------------------------------------------------------------ R2 the scorer registry: every parser dest classified, separate from train.py's
    ap = S.build_parser()
    sdests = {x.dest for x in ap._actions if x.dest != "help"}
    ok(set(C.DVB_KINDS_SCORER) == sdests, (sorted(sdests - set(C.DVB_KINDS_SCORER)), sorted(set(C.DVB_KINDS_SCORER) - sdests)))
    ok(set(C.DVB_KINDS_SCORER.values()) <= {"data", "built", "consumed", "run", "control"})
    ok(not any(d == "epo" or d.startswith("epo_") or d in ("lane_head", "max_updates", "b_mode") for d in C.DVB_KINDS), "scorer flags leaked into the train.py registry")
    opt_of = {x.dest: x.option_strings[0] for x in ap._actions if x.option_strings}
    ok({opt_of[d] for d, k in C.DVB_KINDS_SCORER.items() if k == "data"} <= set(P["data_flags"]))              # every input path the token fingerprints
    ok(set(P["data_flags"]) <= {opt_of[d] for d in sdests if opt_of.get(d)})                                 # and nothing the parser lacks
    reads = C.namespace_reads((HERE / "scorer_finetune.py").read_text(encoding="utf-8"), C.SCORER_NAMESPACE_FUNCS)
    ok(sdests <= reads, sorted(sdests - reads))                                                              # no dead lever today
    ok(reads <= sdests, sorted(reads - sdests))                                                              # no undeclared namespace read today
    pref = {k.split("_", 1)[0] for k in rls1_gate_keys()}
    ok(pref == set(C.SCORER_GATES) and "G6" not in pref, (sorted(pref), sorted(C.SCORER_GATES)))              # the consumer knows exactly the gates the trainer writes
    for dest, want in {**L.RLS1_EPO_REGISTERED, **L.RLS1_R2_REGISTERED}.items():                           # the registered values ARE the parser defaults (an R0 argv need not name them)
        ok(C._close(ap.get_default(dest), want), (dest, ap.get_default(dest), want))
    ok(all(ap.get_default(d) == v for d, v in (("warmup", 50), ("batch", 8), ("accum", 8))) and ap.get_default("lr") != L.RLS1_RECIPE["lr"])
    # ------------------------------------------------------------------ R3 G-HYG on the scorer, with its two arms
    ev = C.check_hyg(profile=P)
    ok(ev["status"] == "PASS" and ev["details"]["namespace_reads"] == len(reads) >= 40, ev["details"])           # it READ the scorer's namespace (not train.py's functions, which would read nothing here)
    for arm, frag in (("no_cfg_guard", "ACCEPTED an undeclared attribute"), ("undeclared_namespace_read", "undeclared_lever_g_hyg")):
        ev = C.check_hyg(arm=arm, profile=P)
        ok(ev["status"] == "FAIL" and frag in " ".join(ev["reasons"]), (arm, ev["reasons"]))
    ok(C.check_hyg(profile=P)["status"] == "PASS" and C.check_hyg()["status"] == "PASS")                     # no residue; the default (train.py) check is unchanged
    # ------------------------------------------------------------------ R4 the arm rules: the clean argvs (the exact shapes the handoff records)
    for nm, argv, arm in (("R0", r0, None), ("R0", r0, "R0"), ("R1", C._set(r0, "--seed", "1"), "R1"), ("R2", r2, None), ("R2", r2, "R2"),
                          ("C0", c0, None), ("C0", c0, "C0"), ("C1", C._set(c0, "--seed", "1"), "C1")):
        clean(argv, arm)
    clean(without(without(without(r0, "--lane-head", 0), "--teacher-lane-train"), "--teacher-lane-heldout"), "R0")        # a 6-output base (P2's model_final): no lane head
    ok([L.rls1_arm_class(a) for a in (c0, r0, r2, r0 + ["--epo-k"], r0 + ["--epo-l"])] == ["C", "R", "R2", "R?", "R?"])
    ok(L.rls1_control_argv(r0) == c0 and L.rls1_control_argv(r2) == c0 and L.rls1_control_argv(c0) == c0)               # R vs C differ by the EPO flags and nothing else
    clean(L.rls1_control_argv(r2), "C0")
    # ------------------------------------------------------------------ R5 missing required flag => refused (one test each)
    for flag, nv, frag in (("--epo", 0, "requires --epo for R0"), ("--epo-terms-train", 1, "requires --epo-terms-train for R0"), ("--pdm-labels-train", 1, "requires --pdm-labels-train"),
                           ("--pdm-labels-heldout", 1, "requires --pdm-labels-heldout"), ("--require-pdm", 0, "requires --require-pdm"),
                           ("--b-mode", 1, "requires --b-mode listnet"), ("--lam", 1, "requires --lam 0")):
        red(without(r0, flag, nv), frag, "R0")
        red(without(r2, flag, nv), frag.replace("R0", "R2"), "R2")
    red(without(r0, "--epo-terms-train"), "scorer_finetune refuses", "R0")                                  # the trainer's OWN refusal is surfaced, not re-derived
    red(without(r0, "--epo-terms-train"), "needs --epo-terms-train", "R0")
    red(without(r0, "--lam") + ["--lam", "0.3"], "requires --lam 0", "R0")                                    # a changed value, not only an absent flag
    red(C._set(r0, "--b-mode", "compw"), "requires --b-mode listnet", "R0")
    for flag, nv in (("--pdm-labels-train", 1), ("--pdm-labels-heldout", 1), ("--require-pdm", 0), ("--b-mode", 1), ("--lam", 1)):                # the control needs the SAME label source and loss
        red(without(c0, flag, nv), "requires " + flag, "C0")
    red(without(r2, "--epo-k", 0), "requires --epo-k for R2", "R2")
    red(without(r2, "--epo-l", 0), "requires --epo-l for R2", "R2")
    red(r0, "requires --epo-k for R2", "R2")
    red(r0 + ["--epo-k"], "exactly one of --epo-k / --epo-l")                                                # undeclared: neither R0/R1 nor R2
    red(r0 + ["--epo-l"], "exactly one of --epo-k / --epo-l")
    red(r2, "belongs to R2 only", "R0")                                                                     # R0 is the CORE reward
    red(r2, "belongs to R2 only", "R1")
    red(r0 + ["--epo-lam-k", "0.05"], "belongs to R2 only", "R0")
    red(without(r2, "--epo-k", 0) + ["--epo-lam-k", "0.05"], "named without --epo-k", "R2")                  # a weight for a term that is OFF
    red(without(r2, "--epo-l", 0) + ["--epo-lam-l", "0.05"], "named without --epo-l", "R2")
    # ------------------------------------------------------------------ R6 a control must NOT carry --epo (nor any EPO lever)
    red(r0, "must NOT carry", "C0")
    red(r2, "must NOT carry", "C1")
    red(c0 + ["--epo-terms-train", "/w/terms"], "must NOT carry", "C0")
    red(c0 + ["--epo-lam-d", "0.15"], "must NOT carry", "C0")
    red(c0 + ["--epo-terms-train", "/w/terms"], "must NOT carry")                                            # undeclared: an EPO lever without --epo is a control carrying EPO
    red(c0 + ["--epo-temp", "0.2"], "scorer_finetune refuses")
    red(c0 + ["--epo-temp", "0.1"], "must NOT carry")                                                          # named AT its default: scorer_finetune's own test (value != default) cannot see it; the gate's token test does
    ok(S.epo_refusals(S.build_parser().parse_args(c0 + ["--epo-temp", "0.1"]), S.build_parser()) == [])
    # ------------------------------------------------------------------ R7 registered lever values, recipe, seeds (effective value: named or the parser DEFAULT)
    for flag, bad, frag in (("--epo-temp", "0.2", "requires --epo-temp 0.1"), ("--epo-beta", "0.5", "requires --epo-beta 0.1"), ("--epo-ramp", "100", "requires --epo-ramp 200"),
                            ("--epo-lam", "0.5", "requires --epo-lam 1.0"), ("--epo-lam-d", "0", "requires --epo-lam-d 0.15"),
                            ("--epo-w-prog", "0.3", "requires --epo-w-prog 0.5")):
        red(C._set(r0, flag, bad), frag, "R0")
    red(C._set(r2, "--epo-lam-k", "0.1"), "requires --epo-lam-k 0.05", "R2")
    red(C._set(r2, "--epo-lam-l", "0.1"), "requires --epo-lam-l 0.05", "R2")
    clean(C._set(r0, "--epo-temp", "0.1"), "R0")                                                           # naming a registered value is fine
    for flag, bad in (("--lr", "3e-5"), ("--max-updates", "500"), ("--batch", "4"), ("--accum", "4"), ("--warmup", "10"), ("--epochs", "2")):
        red(C._set(r0, flag, bad), "requires " + flag, "R0")
        red(C._set(c0, flag, bad), "requires " + flag, "C0")
    red(without(r0, "--max-updates"), "requires --max-updates", "R0")                                        # absent = the parser default 0, which is not 1000
    red(without(r0, "--lr"), "requires --lr", "R0")
    red(without(r0, "--epochs"), "requires --epochs", "R0")                                                  # absent = the parser default 1.0: 250 updates
    red(C._set(r0, "--seed", "0"), "registered with --seed 1", "R1")
    red(C._set(r0, "--seed", "1"), "registered with --seed 0", "R0")
    red(c0, "registered with --seed 1", "C1")
    red(C._set(r2, "--seed", "1"), "registered with --seed 0", "R2")
    ok(any("unknown arm" in r for r in L.profile_refusals({**P, "declared_arm": "R9"}, r0)))
    # ------------------------------------------------------------------ R8 forbidden / inert flags, the lane head's pairing, the parser's verdict, exact spelling
    for extra, frag in ((["--preflight"], "FORBIDS"), (["--lane-labels-train", "/w/ll"], "FORBIDS"), (["--lane-labels-heldout", "/w/ll"], "FORBIDS"),
                        (["--b-compw", "1,1,1,1,1,1"], "FORBIDS"), (["--tau-s", "0.1"], "FORBIDS"), (["--tau-t", "0.2"], "FORBIDS")):
        red(r0 + extra, frag, "R0")
        red(c0 + extra, frag, "C0")
    red(without(r0, "--teacher-lane-heldout"), "--lane-head needs", "R0")
    red(without(r0, "--lane-head", 0), "named without --lane-head", "R0")
    red(r0 + ["--epo-bogus", "1"], "parser rejects", "R0")
    red(without(r0, "--ckpt"), "parser rejects", "R0")
    red(["--require-p" if t == "--require-pdm" else t for t in r0], "not an exact option", "R0")             # argparse accepts the prefix; the gate's token tests do not
    red(["--require-p" if t == "--require-pdm" else t for t in r0], "requires --require-pdm", "R0")
    red(r0, "DECLARED, NOT BUILT", prof={**P, "declared_arm": "R0", "required_flags": {**P["required_flags"], "--g-probe-lever": None}})
    red(r0, "scorer_finetune.py never emits it", prof={**P, "declared_arm": "R0", "live_terms": ("g_probe_term",)})
    L_terms = L.RLS1_EPO_TERMS
    L.RLS1_EPO_TERMS = ("g_probe_epo_term",)
    try:
        red(r0, "scorer_finetune.py never emits it", "R0")                                                  # an EPO arm requires its metric rows ...
        clean(c0, "C0")                                                                                    # ... a control does not
    finally:
        L.RLS1_EPO_TERMS = L_terms
    clean(r0, "R0")
    # a lever the profile declares that the trainer does not BUILD (the flag is gone from the parser) is a launch blocker
    orig_bp = S.build_parser

    def bp_old():
        a_ = orig_bp()
        a_._actions[:] = [x for x in a_._actions if x.dest != "epo_k"]
        a_._option_string_actions.pop("--epo-k", None)
        return a_
    S.build_parser = bp_old
    try:
        red(r2, "parser rejects", "R2")
    finally:
        S.build_parser = orig_bp
    # ------------------------------------------------------------------ R9 the CLI: `refusals` with and without --arm-name; `run` will not mint an unnamed arm
    def cli(*args):
        return subprocess.run([sys.executable, str(HERE / "launch_gate.py"), *args], capture_output=True, text=True, encoding="utf-8")
    rc = cli("refusals", "--profile", "refe_rls1", "--arm-name", "R0", "--args", " ".join(r0))
    ok(rc.returncode == 0 and "ZZGATE-CLEANZZ" in rc.stdout, rc.stdout[-400:] + rc.stderr[-400:])
    rc = cli("refusals", "--profile", "refe_rls1", "--arm-name", "R2", "--args", " ".join(r0))
    ok(rc.returncode == 5 and "ZZGATE-REFUSEDZZ" in rc.stdout and "--epo-k" in rc.stdout, rc.stdout[-400:])
    with tempfile.TemporaryDirectory() as td_:
        rc = cli("run", "--profile", "refe_rls1", "--args", " ".join(r0), "--pkg", td_, "--out-dir", str(Path(td_) / "o"))
        ok(rc.returncode != 0 and "mints for ONE named arm" in (rc.stdout + rc.stderr), rc.stderr[-300:])
        rc = cli("refusals", "--profile", "refe_p2", "--arm-name", "R0", "--args", real_args())
        ok(rc.returncode != 0 and "only meaningful for a profile with arm rules" in (rc.stdout + rc.stderr))
    # ------------------------------------------------------------------ R10 G-DVB for the scorer: the registry, its two regression arms (+3), the observed side
    P0 = {**P, "declared_arm": "R0"}
    pf_ok = C.load_scorer_preflight(rls1_write(Path(tempfile.mkdtemp()) / "preflight.jsonl", rls1_pf_rows(r0)))
    ev = C.check_dvb_scorer(r0, pf_ok, P0)
    ok(ev["status"] == "PASS" and ev["details"]["observed_rows"] == 27 and ev["details"]["unknown_flags"] == [] and ev["details"]["stale_registry"] == [], ev["reasons"])      # 14 -> 15: R0c's epo.ref row; 15 -> 23: SAFEOBJ's eight epo_con rows; 23 -> 24: PICKCON's epo_con_temp row; 24 -> 27: B8's epo_con_feed, epo_carve_sha256 and epo_carve_micro rows (the carve-only epo_carve_logs row appears only for --epo-con-feed carve)
    ok(sum(ev["details"]["kinds"].values()) == len(C.DVB_KINDS_SCORER) == len(sdests))
    ok(C.check_dvb_scorer(c0, C.load_scorer_preflight(rls1_write(Path(tempfile.mkdtemp()) / "p.jsonl", rls1_pf_rows(c0))), P)["status"] == "PASS")
    for arm, frag in (("unknown_flag", "NO declared-vs-built entry"), ("stale_registry", "no longer has"), ("dead_lever", "never read by the trainer"),
                      ("unbound_data", "would NOT fingerprint"), ("lever_ignored", "declared 'listnet' != built 'compw'")):
        ev = C.check_dvb_scorer(r0, pf_ok, P0, arm=arm)
        ok(ev["status"] == "FAIL" and frag in " ".join(ev["reasons"]) and ev["arm"] == arm, (arm, ev["reasons"]))
    ok(C.check_dvb_scorer(r0, pf_ok, P0)["status"] == "PASS")                                                # the arms left no residue
    # ------------------------------------------------------------------ R11 SAFEOBJ (refe_rls1d, raw/2026-10-17-safety-objective): the arms R0d / R1d / R1o (the full matrix and its mutations: refe/test_rl_safeobj.py)
    Pd = L.PROFILES["refe_rls1d"]
    CON = ["--epo-con", "g5", "--epo-con-eps", "0", "--epo-con-lr", "4", "--epo-con-max", "10", "--epo-con-lam0", "0"]
    r1_ = C._set(r0, "--seed", "1")
    r0d, r1d, r1o = r0 + CON, r1_ + CON, r1_ + ["--epo-con", "off"]
    redd = lambda argv, frag, arm: red(argv, frag, arm, {**Pd, "declared_arm": arm})          # noqa: E731
    for argv_, arm_ in ((r0d, "R0d"), (r1d, "R1d"), (r1o, "R1o")):
        ok(refs(argv_, arm_, {**Pd, "declared_arm": arm_}) == [], arm_)
    for fl in ("--epo-con", "--epo-con-eps", "--epo-con-lr", "--epo-con-max", "--epo-con-lam0"):                 # a parser DEFAULT never carries a lever: each flag must be NAMED
        redd(without(r0d, fl), f"requires {fl} NAMED", "R0d")
        redd(without(r1d, fl), f"requires {fl} NAMED", "R1d")
    redd(r0, "requires --epo-con g5", "R0d")                                                                     # R0's own argv (the default `off`) is not R0d
    redd(C._set(r0d, "--seed", "1"), "registered with --seed 0", "R0d")
    redd(C._set(r1d, "--seed", "0"), "registered with --seed 1", "R1d")
    redd(C._set(r1o, "--seed", "0"), "registered with --seed 1", "R1o")
    redd(C._set(r1o, "--epo-con", "g5"), "requires --epo-con off", "R1o")                                        # the lever-off control must not carry the lever
    redd(without(r1o, "--epo-con"), "requires --epo-con NAMED", "R1o")
    redd(C._set(r0d, "--epo-con-lr", "2"), "requires --epo-con-lr 4.0", "R0d")
    for nm_, arm_, argv_ in (("refe_rls1", "R0", r0), ("refe_rls1b", "R0b", r0 + ["--epo-lam", "0.25"]), ("refe_rls1c", "R0c", r0 + ["--epo-ref", "armA"])):          # every older profile refuses the lever ...
        red(argv_ + CON, "requires --epo-con off", arm_, {**L.PROFILES[nm_], "declared_arm": arm_})
        ok(refs(argv_ + ["--epo-con", "off"], arm_, {**L.PROFILES[nm_], "declared_arm": arm_}) == [], nm_)         # ... and still accepts the switch NAMED off
    ok([e["script"] for e in Pd["suite"]][-1] == "test_rl_safeobj.py" and Pd["suite"][:-1] == L.PROFILES["refe_rls1c"]["suite"])                  # M8: the suite carries the new test
    ok(all(C.DVB_KINDS_SCORER.get(k) == "built" for k in ("epo_con", "epo_con_eps", "epo_con_lr", "epo_con_max", "epo_con_lam0", "epo_con_temp", "epo_con_feed", "epo_carve_sha256", "epo_carve_micro")) and C.DVB_KINDS_SCORER.get("epo_carve_logs") == "data")               # M7: the five flags (+ PICKCON's epo_con_temp) are classified

    def bp_new():                                                                                           # a REAL new flag on the parser, no arm: the gate itself refuses
        a_ = orig_bp()
        a_.add_argument("--brand-new-lever", type=float, default=0.0)
        return a_
    S.build_parser = bp_new
    try:
        ev = C.check_dvb_scorer(r0, pf_ok, P0)
        ok(ev["status"] == "FAIL" and "brand_new_lever" in " ".join(ev["reasons"]) and "NO declared-vs-built entry" in " ".join(ev["reasons"]))
    finally:
        S.build_parser = orig_bp
    # observed side: the trainer's own data event is the BUILT value
    ns0 = S.build_parser().parse_args(r0)
    eblk = {k: getattr(ns0, "epo_" + k) for k in C.SCORER_EPO_KEYS}
    for mut, frag in (({"epo": {**eblk, "k": True}}, "--epo-k: declared False != built True"), ({"epo": {**eblk, "lam_d": 0.0}}, "--epo-lam-d: declared 0.15 != built 0.0"),
                      ({"b_mode": "compw"}, "--b-mode: declared 'listnet' != built 'compw'")):
        pf_bad = C.load_scorer_preflight(rls1_write(Path(tempfile.mkdtemp()) / "p.jsonl", rls1_pf_rows(r0, data_mut=mut)))
        ev = C.check_dvb_scorer(r0, pf_bad, P0)
        ok(ev["status"] == "FAIL" and frag in " ".join(ev["reasons"]), (frag, ev["reasons"]))
    pf_c_epo = C.load_scorer_preflight(rls1_write(Path(tempfile.mkdtemp()) / "p.jsonl", rls1_pf_rows(c0, data_mut={"epo": {"k": False}})))
    ev = C.check_dvb_scorer(c0, pf_c_epo, P)
    ok(ev["status"] == "FAIL" and "--epo: declared False != built True" in " ".join(ev["reasons"]))         # a control whose preflight carries an EPO block ran EPO
    pf_nodata = C.load_scorer_preflight(rls1_write(Path(tempfile.mkdtemp()) / "PREFLIGHT_PASS.json", rls1_pf_rows(r0, with_data=False), jsonl=False))
    for pf_ in (None, pf_nodata, C.load_scorer_preflight(Path(tempfile.mkdtemp()) / "absent.jsonl")):
        ev = C.check_dvb_scorer(r0, pf_, P0)
        ok(ev["status"] == "FAIL" and "no preflight `data` event" in " ".join(ev["reasons"]))
    ev = C.check_dvb_scorer(r0 + ["--preflight"], pf_ok, P0)
    ok(ev["status"] == "FAIL" and "FORBIDDEN" in " ".join(ev["reasons"]) and "test/ablation control" in " ".join(ev["reasons"]))
    ev = C.check_dvb_scorer(r0 + ["--epo-bogus", "1"], pf_ok, P0)
    ok(ev["status"] == "FAIL" and "does not build" in " ".join(ev["reasons"]))
    kept = C.DVB_KINDS_SCORER["lr"]
    C.DVB_KINDS_SCORER["lr"] = "built"                                                                       # `built` with nothing observing it is a registry lie
    try:
        ev = C.check_dvb_scorer(r0, pf_ok, P0)
        ok(ev["status"] == "FAIL" and "nothing observes it" in " ".join(ev["reasons"]) and ev["details"]["built_without_observer"] == ["lr"])
    finally:
        C.DVB_KINDS_SCORER["lr"] = kept
    # ------------------------------------------------------------------ R11 the preflight consumer: parsed, every obliged gate present AND ok, the file's sha256 bound
    td = Path(tempfile.mkdtemp())

    def cons(argv, rows=None, name="preflight.jsonl", jsonl=True, **kw):
        p = td / name
        rls1_write(p, rows if rows is not None else rls1_pf_rows(argv, **kw), jsonl)
        return L.consume_scorer_preflight(p, argv, P)
    for nm, argv in (("R0", r0), ("R2", r2), ("C0", c0)):
        ev = cons(argv)
        ok(ev["status"] == "PASS" and ev["details"]["file_sha256"] == GT.sha256_file(td / "preflight.jsonl") and "G6" not in ev["details"]["required"], (nm, ev["reasons"]))
    ev = cons(r0)
    ok("G11" in ev["details"]["required"] and "G10" in ev["details"]["required"] and len(ev["details"]["gates"]) == 9 and "G11" not in cons(c0)["details"]["required"])
    ev = cons(r0, drop=("G11",))
    ok(ev["status"] == "FAIL" and "ABSENT: ['G11']" in " ".join(ev["reasons"]), ev["reasons"])               # the headline: an --epo arm without G11 is not proved live
    ok(cons(r2, drop=("G11",))["status"] == "FAIL")
    for g in ("G1", "G2", "G3", "G4", "G5", "G7", "G9", "G10"):
        ev = cons(r0, drop=(g,))
        ok(ev["status"] == "FAIL" and f"ABSENT: ['{g}']" in " ".join(ev["reasons"]), (g, ev["reasons"]))
    ok(cons(without(without(without(r0, "--lane-head", 0), "--teacher-lane-train"), "--teacher-lane-heldout"))["details"]["required"] == ["G1", "G2", "G3", "G4", "G5", "G7", "G9", "G11"])
    for k in rls1_gate_keys():
        if k.startswith("G4_compw"):
            continue
        ev = cons(r0, notok=(k,))
        if k.startswith(("G8_", "G12_", "G13_")):                  # r0 carries no lane labels / no --aux-drv / no --epo-con: G8 / G12 / G13 are not obliged, so such a flag in the fixture is simply not written
            ok(ev["status"] == "PASS", (k, ev["reasons"]))
        else:
            ok(ev["status"] == "FAIL" and "not ok" in " ".join(ev["reasons"]) and k in " ".join(ev["reasons"]), (k, ev["reasons"]))
    g11_key = next(k for k in rls1_gate_keys() if k.startswith("G11_"))
    ok(cons(r0, add={"G2_live": {"ok": 1}})["status"] == "FAIL" and cons(r0, add={"G2_live": {"ok": "yes"}})["status"] == "FAIL")           # ok must be True, not truthy
    ok(cons(r0, add={"G2_live": "ok"})["status"] == "FAIL")
    ev = cons(r0, verdict="FAIL")
    ok(ev["status"] == "FAIL" and "not PASS" in " ".join(ev["reasons"]))
    ev = cons(c0, add={g11_key: {"ok": True}})
    ok(ev["status"] == "FAIL" and "DIFFERENT configuration" in " ".join(ev["reasons"]))                      # a control offered an EPO run's preflight
    ev = cons(r0, add={"G14_something_new": {"ok": True}})                                                   # (G13 is a KNOWN gate since SAFEOBJ: the unknown-gate fixture moves to G14)
    ok(ev["status"] == "FAIL" and "not known to the gate" in " ".join(ev["reasons"]))
    ok(cons(r0, add={"G8_lane_labels_in_bank": {"ok": True}})["status"] == "FAIL")                           # a gate the argv does not oblige
    good = rls1_pf_rows(r0)[-1]["gates"][g11_key]
    for mut, frag in ((lambda g: g["per_module_on"].update(score_head=0.0), "non-zero on every scorer module"),                      # G11's own `ok` is not trusted: it is re-judged
                      (lambda g: g["per_module_on"].pop("score_dec"), "non-zero on every scorer module"),
                      (lambda g: g["per_module_on"].update(score_q_mlp=float("nan")), "non-zero on every scorer module"),
                      (lambda g: g["per_module_lam0"].update(score_head=1e-9), "EXACTLY 0.0"),
                      (lambda g: g["per_module_off"].update(score_dec=0.5), "EXACTLY 0.0"),
                      (lambda g: g.update(controls_exactly_zero=False), "EXACTLY 0.0"),
                      (lambda g: g.update(per_module_lam0={}, per_module_off={}), "EXACTLY 0.0"),
                      (lambda g: g.update(kl_ok=False, kl_step0=0.3), "did not start AT the base"),
                      (lambda g: g.update(mutate_detach=True), "REFE_RL_MUTATE_EPO_DETACH")):
        g = copy.deepcopy(good)
        mut(g)
        ev = cons(r0, g11=g)
        ok(ev["status"] == "FAIL" and frag in " ".join(ev["reasons"]), (frag, ev["reasons"]))
    # the file itself
    ok(L.consume_scorer_preflight(td / "absent.jsonl", r0, P)["status"] == "FAIL")
    (td / "garbage.jsonl").write_text("not json\n", encoding="utf-8")
    ok(L.consume_scorer_preflight(td / "garbage.jsonl", r0, P)["status"] == "FAIL")
    rows_ = rls1_pf_rows(r0)
    (td / "trunc.jsonl").write_text(json.dumps(rows_[0]) + "\n" + json.dumps(rows_[1])[:-9] + "\n", encoding="utf-8")                      # killed mid-write
    ev = L.consume_scorer_preflight(td / "trunc.jsonl", r0, P)
    ok(ev["status"] == "FAIL" and "not JSON" in " ".join(ev["reasons"]))
    (td / "empty.jsonl").write_text("", encoding="utf-8")
    ok(L.consume_scorer_preflight(td / "empty.jsonl", r0, P)["status"] == "FAIL")
    (td / "other.jsonl").write_text(json.dumps({"event": "train", "update": 25}) + "\n", encoding="utf-8")
    ev = L.consume_scorer_preflight(td / "other.jsonl", r0, P)
    ok(ev["status"] == "FAIL" and "no event=preflight row" in " ".join(ev["reasons"]))
    ok(L.consume_scorer_preflight(td / "preflight.jsonl", r0 + ["--epo-bogus", "1"], P)["status"] == "FAIL")
    # PREFLIGHT_PASS.json (the preflight event alone) is consumable for its GATES, but cannot be bound to the argv: G-DVB is what refuses it
    ev = cons(r0, rows=rls1_pf_rows(r0, with_data=False), name="PREFLIGHT_PASS.json", jsonl=False)
    ok(ev["status"] == "PASS" and ev["details"]["has_data_event"] is False)
    # appended runs: the LAST preflight is the verdict, and its data event is the last one BEFORE it
    a_, b_ = rls1_pf_rows(r0, verdict="FAIL", notok=("G2_live",)), rls1_pf_rows(r0)
    ok(cons(r0, rows=a_ + b_)["status"] == "PASS" and cons(r0, rows=a_ + b_)["details"]["n_preflight_rows"] == 2)
    ok(cons(r0, rows=b_ + a_)["status"] == "FAIL")
    d_r0, p_r0, d_c0 = rls1_pf_rows(r0)[0], rls1_pf_rows(r0)[1], rls1_pf_rows(c0)[0]
    ok("epo" not in C.load_scorer_preflight(rls1_write(td / "mix1.jsonl", [d_r0, d_c0, p_r0]))["data"])         # the data event is the last one BEFORE the verdict ...
    ok("epo" in C.load_scorer_preflight(rls1_write(td / "mix2.jsonl", [d_r0, p_r0, d_c0]))["data"])             # ... a later one (another run's) is not the verdict's
    # ------------------------------------------------------------------ R12 END TO END on the real run_checks/finalize (G-SUITE excluded: it would recurse into this file)
    with tempfile.TemporaryDirectory() as td2:
        td2 = Path(td2)
        pkg = mini_pkg(td2 / "pkg")
        key = GT.load_key(td2 / "k" / "gate.key", create=True, forbid_under=[pkg])
        real = [t.replace("/w", td2.as_posix() + "/w") for t in r0]
        (td2 / "w").mkdir()
        for f_ in ("--ckpt", "--heldout-logs", "--calib"):
            Path(GT.flag_values(real, f_)[0]).write_text("x", encoding="utf-8")
        for f_ in P["data_flags"]:
            if f_ in real and not Path(GT.flag_values(real, f_)[0]).exists():
                Path(GT.flag_values(real, f_)[0]).mkdir(parents=True)
        Pe = {**P, "declared_arm": "R0", "checks": ("G-HYG", "G-DVB", "G-PREFLIGHT"), "require_commit": False}
        ok(L.profile_refusals(Pe, real) == [], L.profile_refusals(Pe, real))
        pfp = rls1_write(td2 / "preflight.jsonl", rls1_pf_rows(real))
        out = td2 / "gate"
        st = L.run_checks(Pe, real, pkg, out, preflight_json=pfp)
        ok(st == {"G-HYG": "PASS", "G-DVB": "PASS", "G-PREFLIGHT": "PASS", "G-CLOCK": "NA"}, st)
        v, pth, tok = L.finalize(Pe, real, pkg, out, key, preflight_json=pfp)
        ok(v == "PASS" and tok["binding"]["profile"] == "refe_rls1" and tok["binding"]["trainer"] == "refe/scorer_finetune.py", tok["reasons"])
        ok(tok["binding"]["data"]["gate:preflight_json"]["sha256"] == GT.sha256_file(pfp))                   # the preflight file's sha256 IS in the token
        ok(all(f_ in tok["binding"]["data"] for f_ in ("--ckpt", "--onpolicy", "--epo-terms-train", "--pdm-labels-train")))
        good_, why, _ = GT.verify_token(pth, real, pkg, key=key, profile="refe_rls1", data_flags=P["data_flags"])
        ok(good_, why)
        ok(not GT.verify_token(pth, L.rls1_control_argv(real), pkg, key=key, profile="refe_rls1", data_flags=P["data_flags"])[0])      # the control's argv is not R0's
        pfp.write_text(pfp.read_text(encoding="utf-8") + "\n", encoding="utf-8")                              # the preflight edited between the checks and the mint
        v, _, t = L.finalize(Pe, real, pkg, out, key, preflight_json=pfp)
        ok(v == "FAIL" and any("different fingerprint" in r for r in t["reasons"]) and any("gate:preflight_json" in r for r in t["reasons"]), t["reasons"])
        rls1_write(pfp, rls1_pf_rows(real))
        ev0 = L.run_checks(Pe, real, pkg, td2 / "gate_noflight")                                              # no preflight at all: G-DVB and G-PREFLIGHT both FAIL
        ok(ev0["G-DVB"] == "FAIL" and ev0["G-PREFLIGHT"] == "FAIL" and L.finalize(Pe, real, pkg, td2 / "gate_noflight", key)[0] == "FAIL")
        # the CLI wiring: `run` hands --gate-preflight-json to run_checks AND to finalize.  --only leaves G-SUITE out (it would recurse into this file), so the verdict is
        # FAIL by design (no G-SUITE evidence, no commit); what is asserted is the three checks' status IN THE TOKEN and the preflight's sha256 in its data manifest.
        rls1_write(pfp, rls1_pf_rows(real))
        rc = cli("run", "--profile", "refe_rls1", "--arm-name", "R0", "--pkg", str(pkg), "--out-dir", str(td2 / "gate_cli"), "--key-file", str(td2 / "k" / "gate.key"),
                 "--args", " ".join(real), "--gate-preflight-json", str(pfp), "--only", "G-HYG,G-DVB,G-PREFLIGHT")
        tokp = next((td2 / "gate_cli").glob("FAIL_*.json"), None)
        ok(rc.returncode == 1 and tokp is not None, rc.stdout[-400:] + rc.stderr[-400:])
        tk = GT.read_json(tokp)
        ok({c: tk["checks"][c]["status"] for c in ("G-HYG", "G-DVB", "G-PREFLIGHT")} == dict.fromkeys(("G-HYG", "G-DVB", "G-PREFLIGHT"), "PASS")
           and tk["checks"]["G-SUITE"]["status"] == "MISSING", tk["checks"])
        ok(tk["binding"]["data"]["gate:preflight_json"]["sha256"] == GT.sha256_file(pfp))
        rls1_write(pfp, rls1_pf_rows(real, drop=("G11",)))
        out_b = td2 / "gate_nog11"
        st = L.run_checks(Pe, real, pkg, out_b, preflight_json=pfp)
        v, _, t = L.finalize(Pe, real, pkg, out_b, key, preflight_json=pfp)
        ok(st["G-PREFLIGHT"] == "FAIL" and st["G-DVB"] == "PASS" and v == "FAIL" and any("G-PREFLIGHT" in r for r in t["reasons"]))
        r2_real = real + ["--epo-k", "--epo-l"]
        rls1_write(pfp, rls1_pf_rows(real))                                                                  # R0's preflight offered for R2's argv: refused, twice over
        ev_ = C.check_dvb_scorer(r2_real, C.load_scorer_preflight(pfp), {**P, "declared_arm": "R2"})
        ok(ev_["status"] == "FAIL" and "--epo-k: declared True != built False" in " ".join(ev_["reasons"]), ev_["reasons"])
        # a REFUSED argv: every check FAILs before one runs, no PASS token can exist
        bad = without(real, "--epo-terms-train")
        out_c = td2 / "gate_refused"
        st = L.run_checks(Pe, bad, pkg, out_c, preflight_json=pfp)
        ok(set(st.values()) == {"FAIL"} and set(st) == {"G-HYG", "G-DVB", "G-PREFLIGHT", "G-CLOCK"}, st)
        v, _, t = L.finalize(Pe, bad, pkg, out_c, key, preflight_json=pfp)
        ok(v == "FAIL" and not list(out_c.glob("PASS_*")) and any("--epo-terms-train" in r for r in t["reasons"]))
    # the real profile: a refused argv fails every check including the suite and the NA clock, and starts nothing
    with tempfile.TemporaryDirectory() as td3:
        st = L.run_checks({**P, "declared_arm": "R2"}, r0, PKG, Path(td3) / "o")
        ok(set(st.values()) == {"FAIL"} and set(st) == set(P["checks"]) | set(P["na_ok"]), st)


# ====================================================================================================== L_field: the refe_field / refe_field_ctrl profiles (eval/PREREG_AUX_FIELD_LOSS_DRAFT.md)
FIELD_TAIL = " --w-field 8 --field-margin 0.25 --field-interp 5hz --field-labels /w/label_store"


def field_static() -> None:
    """The L_field arm's gate entries, each with its negative twin: the registry, the profile refusals (F needs --w-field > 0 NAMED; the control C names EXACTLY 0.0; the registered margin /
    interp / labels are named), G-DVB's observed rows (the margin and interp the term RECEIVED, the multiplier measured at the graph, the store's pinned sha256 / key count / coverage /
    raster constants) judged on synthetic captures, G-LIVE's reach judge, and the SUITE.  The instrumented end-to-end arms (the real trainer's smoke run) are
    raw/2026-10-13-field-loss-wiring/test_field_wiring.py part (d)."""
    import aux_drivable_head as AUX
    import aux_label_store as ALS
    import field_loss as FL
    import train as T
    PF, PC = L.PROFILES["refe_field"], L.PROFILES["refe_field_ctrl"]
    # ---- the registry: the five flags are in DVB_KINDS (S5 pins the table equal to the parser's dests)
    for d in ("w_field", "field_margin", "field_interp", "field_labels", "field_shuffle_seed"):
        ok(C.DVB_KINDS.get(d) == "built", d)
    # ---- the profiles: F and C are clean on their argv and differ by --w-field alone
    fa = L.parse_args_string(real_args() + FIELD_TAIL)
    fc = [("0.0" if (i > 0 and fa[i - 1] == "--w-field") else t) for i, t in enumerate(fa)]
    ok(fc != fa and sum(1 for x, y in zip(fa, fc) if x != y) == 1)
    ok(L.profile_refusals(PF, fa) == [], L.profile_refusals(PF, fa))
    ok(L.profile_refusals(PC, fc) == [], L.profile_refusals(PC, fc))

    def refs(prof, argv):
        return L.profile_refusals(prof, argv)

    def red(prof, argv, frag):
        rs = refs(prof, argv)
        ok(rs and any(frag in r for r in rs), (frag, rs))

    def swap(argv, flag, value):
        return [(value if (i > 0 and argv[i - 1] == flag) else t) for i, t in enumerate(argv)]

    def without(argv, flag, nvalues=1):
        i = argv.index(flag)
        return argv[:i] + argv[i + 1 + nvalues:]
    red(PF, without(fa, "--w-field"), "requires --w-field")                          # --w-field ABSENT from the argv of an arm that means it (its default disables the term)
    red(PC, without(fc, "--w-field"), "requires --w-field")                          # the control NAMES 0.0
    red(PF, fc, "arm F) requires --w-field > 0")                                     # the control's argv is not arm F
    red(PC, fa, "EXACTLY 0.0")                                                       # arm F's argv is not the control
    red(PF, swap(fa, "--w-field", "-1"), "requires --w-field > 0")
    red(PF, swap(fa, "--w-field", "nan"), "not a finite number")
    red(PC, swap(fc, "--w-field", "0.0001"), "EXACTLY 0.0")
    for pr, base in ((PF, fa), (PC, fc)):
        red(pr, without(base, "--field-labels"), "--field-labels")
        red(pr, without(base, "--field-margin"), "--field-margin")
        red(pr, swap(base, "--field-margin", "0.5"), "--field-margin")
        red(pr, swap(base, "--field-interp", "navsim2hz"), "--field-interp")
        red(pr, without(base, "--field-interp"), "--field-interp")
        red(pr, base + ["--field-shuffle-seed", "20261011"], "FORBIDS")             # the label-shuffled control arm is not admitted until a dated amendment adds it
        red(pr, base + ["--synthetic"], "FORBIDS")
        red(pr, swap(base, "--lr", "1e-4"), "--lr")                                  # P2's levers are asserted too
        red(pr, without(base, "--pdm-only", 0), "--pdm-only")
    # a profile that DECLARES the levers against a trainer that does not build them is a launch blocker (train.py without PATCH_train): simulated with a parser that lacks one
    orig_bp = T.build_parser

    def bp_old():
        ap = orig_bp()
        ap._actions[:] = [x for x in ap._actions if x.dest != "w_field"]
        for o in [o for o in list(ap._option_string_actions) if o == "--w-field"]:
            del ap._option_string_actions[o]
        return ap
    T.build_parser = bp_old
    try:
        red(PF, fa, "DECLARED, NOT BUILT")
    finally:
        T.build_parser = orig_bp
    # the CLI: exit 0 / ZZGATE-CLEANZZ on F's and C's argv, exit 5 / ZZGATE-REFUSEDZZ on an argv with --w-field absent
    for prof_name, argv_ in (("refe_field", fa), ("refe_field_ctrl", fc)):
        rc = subprocess.run([sys.executable, str(HERE / "launch_gate.py"), "refusals", "--profile", prof_name, "--args", " ".join(argv_)], capture_output=True, text=True, encoding="utf-8")
        ok(rc.returncode == 0 and "ZZGATE-CLEANZZ" in rc.stdout, rc.stdout[-300:])
    rc = subprocess.run([sys.executable, str(HERE / "launch_gate.py"), "refusals", "--profile", "refe_field", "--args", " ".join(without(fa, "--w-field"))], capture_output=True, text=True,
                        encoding="utf-8")
    ok(rc.returncode == 5 and "ZZGATE-REFUSEDZZ" in rc.stdout)
    # P2's own profile is unchanged by all of this (its argv is clean, and it does not carry the field suite)
    ok(L.profile_refusals(L.PROFILES["refe_p2"], L.parse_args_string(real_args())) == [])
    ok({"script": "test_field_loss.py", "token": "ZZTEST_PASS"} in PF["suite"] and {"script": "test_field_loss.py", "token": "ZZTEST_PASS"} in PC["suite"])
    ok({"script": "test_field_loss.py", "token": "ZZTEST_PASS"} not in L.SUITE and (HERE / "test_field_loss.py").is_file())
    ok(PF["live_terms"] == ("traj_L1", "score", "field_L") and "field_L" not in PC["live_terms"] and "--field-labels" in PF["data_flags"] and "--field-labels" in PC["data_flags"])
    ok('"field_L"' in (HERE / "train.py").read_text(encoding="utf-8"))
    # ---- admissibility (prereg "Admissibility"; G-EVAL is unchanged by construction): the planner / model import no label path.  `grep -c "field_loss\|sdf" refe/model.py refe/planner.py` must stay 0
    def label_path_free(src: str) -> bool:
        return "field_loss" not in src and "sdf" not in src
    for f_ in ("model.py", "planner.py"):
        ok(label_path_free((HERE / f_).read_text(encoding="utf-8")), f_)
    ok(not label_path_free("from field_loss import field_loss") and not label_path_free("x = sdf_int8"))                  # the check can go red
    # ---- G-DVB's observed rows on synthetic captures
    td = Path(tempfile.mkdtemp())
    (td / "manifest.json").write_text(json.dumps({"raster": ALS.raster_spec()}), encoding="utf-8")
    ev = {"event": "field", "store_sha256": AUX.PINNED_STORE_SHA256, "store_keys": AUX.PINNED_STORE_KEYS, "rows": 4000, "rows_labelled": 4000, "shuffle_seed": None}

    def caps(w=8.0, **mut):
        a = SimpleNamespace(w_field=w, field_margin=0.25, field_interp="5hz", field_labels=str(td), field_shuffle_seed=mut.pop("declared_seed", None))
        b = SimpleNamespace(banks={"TargetBank": [{"args": [], "kwargs": {"field_labels": mut.pop("built_labels", str(td)), "field_shuffle_seed": mut.pop("built_seed", None)}}]})
        calls = mut.pop("calls", [{"margin": 0.25, "interp": "5hz", "valid": True}] * 4)
        grads = mut.pop("grads", [w / 2.0] * 4)
        events = mut.pop("events", [dict(ev), {"field_w": w}, {"field_w": w}])
        s_ = SimpleNamespace(args=SimpleNamespace(steps=2, accum=2), field_calls=calls, field_grads=grads, metrics=events)
        assert not mut, mut
        return a, b, s_

    def rows_of(**mut):
        a, b, s_ = caps(**mut)
        return C.field_dvb_rows(a, b, s_)
    ok(len(rows_of()) >= 12 and all(r["ok"] for r in rows_of()), [r for r in rows_of() if not r["ok"]])
    # the default arm (no field flags) adds only the two TargetBank rows, both equal
    a0, b0, _ = caps()
    a0.w_field, a0.field_labels = 0.0, None
    b0.banks["TargetBank"][0]["kwargs"]["field_labels"] = None
    r0 = C.field_dvb_rows(a0, b0, None)
    ok(len(r0) == 2 and all(r["ok"] for r in r0))
    b0.banks["TargetBank"][0]["kwargs"]["field_labels"] = "/somewhere"                  # a loader that was handed a store the argv did not declare
    ok(not all(r["ok"] for r in C.field_dvb_rows(a0, b0, None)))

    def bad(frag, **mut):
        rs = [r for r in rows_of(**mut) if not r["ok"]]
        ok(rs and any(frag in (r["flag"] + " " + r["how"]) for r in rs), (frag, rs))
    bad("margin", calls=[{"margin": 0.5, "interp": "5hz", "valid": True}] * 4)                                          # the margin the term RECEIVED
    bad("interp", calls=[{"margin": 0.25, "interp": "navsim2hz", "valid": True}] * 4)
    bad("valid", calls=[{"margin": 0.25, "interp": "5hz", "valid": False}] * 4)
    bad("multiplier", grads=[2.0] * 4)                                                                                  # wired off by a factor (0.5 w): the gradient arriving is w/accum / 2
    bad("multiplier", grads=[8.0] * 4)                                                                                  # ... or by 2
    bad("multiplier", grads=[])                                                                                         # computed but never added (the hook never fires)
    bad("multiplier", grads=[4.0] * 3)                                                                                  # added on 3 of 4 micro-batches
    bad("once per micro-batch", calls=[{"margin": 0.25, "interp": "5hz", "valid": True}] * 3)
    bad("logged field_w", events=[dict(ev), {"field_w": 4.0}, {"field_w": 4.0}])
    bad("`field` event", events=[{"field_w": 8.0}])                                                                      # the trainer wrote no store record
    bad("pinned store", events=[dict(ev, store_sha256="0" * 64), {"field_w": 8.0}])
    bad("key count", events=[dict(ev, store_keys=AUX.PINNED_STORE_KEYS - 1), {"field_w": 8.0}])
    bad("coverage", events=[dict(ev, rows_labelled=3990), {"field_w": 8.0}])                                             # 99.75 % < 99.9 %
    bad("derangement seed", declared_seed=20261011)
    bad("TargetBank(field_shuffle_seed=)", declared_seed=20261011, built_seed=1)
    bad("TargetBank(field_labels=)", built_labels="/elsewhere")
    ok(all(r["ok"] for r in rows_of(events=[dict(ev, rows_labelled=3997), {"field_w": 8.0}])))                          # 99.925 % >= 99.9 %: the threshold is a threshold
    man = ALS.raster_spec()
    for mut_ in ({"x0": man["x0"] + 1.0}, {"nx": man["nx"] + 1}, {"sdf_unit_m": 0.2}):
        (td / "manifest.json").write_text(json.dumps({"raster": {**man, **mut_}}), encoding="utf-8")
        bad("raster constants")
    (td / "manifest.json").write_text(json.dumps({"raster": man}), encoding="utf-8")
    ok(all(r["ok"] for r in rows_of()))
    ok(dict(zip(("x0", "x1", "y0", "y1", "res_m", "nx", "ny", "sdf_unit_m"), (FL.X0, FL.X1, FL.Y0, FL.Y1, FL.RES, FL.NX, FL.NY, FL.SDF_UNIT_M)))
       == {k: man[k] for k in ("x0", "x1", "y0", "y1", "res_m", "nx", "ny", "sdf_unit_m")})
    shutil.rmtree(td, ignore_errors=True)

    # ---- G-LIVE's reach judge, from raw fields: clean, then each fault
    def grp(abs_sum, none=0, n=3):
        return {"abs_sum": abs_sum, "n": n, "n_none": none}

    def fr(**over):
        g = {k: grp(1.0) for k in C.FIELD_REACH_GROUPS}
        g.update({k: grp(0.0, none=3) for k in C.SCORER_GROUPS})
        g0 = {k: grp(0.0) for k in list(C.FIELD_REACH_GROUPS) + list(C.SCORER_GROUPS)}
        d = {"ran": True, "labelled": 2, "declared": {"loss": 1.0, "groups": g}, "zero": {"loss": 0.0, "groups": g0}}
        for k, v in over.items():
            d[k] = v
        return d
    ok(C.judge_field_reach(fr(), 8.0) == [])
    gd = {k: grp(1.0) for k in C.FIELD_REACH_GROUPS}
    gd.update({k: grp(0.0, none=3) for k in C.SCORER_GROUPS})
    for g in C.FIELD_REACH_GROUPS:                                   # a term that does not reach one of its ancestors
        gx = dict(gd)
        gx[g] = grp(0.0)
        rs = C.judge_field_reach(fr(declared={"loss": 1.0, "groups": gx}), 8.0)
        ok(rs and g in rs[0], (g, rs))
    for g in C.SCORER_GROUPS:                                        # ... or that reaches the scorer, with a value or merely with a path
        for bad_ in (grp(0.5, none=0), grp(0.0, none=0)):
            gx = dict(gd)
            gx[g] = bad_
            rs = C.judge_field_reach(fr(declared={"loss": 1.0, "groups": gx}), 8.0)
            ok(rs and "scorer group" in rs[0], (g, rs))
    g0x = {k: grp(0.0) for k in list(C.FIELD_REACH_GROUPS) + list(C.SCORER_GROUPS)}
    g0x["traj_head"] = grp(1e-9)
    ok("weight-0 control" in " ".join(C.judge_field_reach(fr(zero={"loss": 0.0, "groups": g0x}), 8.0)))
    ok(C.judge_field_reach(fr(labelled=0), 8.0) and C.judge_field_reach({"ran": False}, 8.0))


# ====================================================================================================== aux drivable-area head, S1: the refe_aux_s1 profile
# scorer_finetune.py with --aux-drv (WIRING_PLAN.md).  The profile is the RL-S1 runner (static checks + the trainer's own --preflight, now with G12) with its own arm rules: arms are
# N (no head) / H1 (head, feed off) / H2 (head + feed) x seeds 0..2.  Every refusal has its clean twin; every G-DVB row its mutation; every G12 field its mutation.
AUXS1_BASE = ("--ckpt /w/model_sft4_B_u400.pt --backbone vitl16 --targets /w/train --images /w/pix --calib /w/calib.json --onpolicy /w/onp --heldout /w/onh "
              "--heldout-logs /w/hl.txt --out /w/aux --lane-head --teacher-lane-train /w/tl --teacher-lane-heldout /w/tlh --pdm-labels-train /w/pdm "
              "--pdm-labels-heldout /w/pdmh --require-pdm --b-mode expreward --lam 1.0 --tau-s 0.1 --lr 1e-4 --max-updates 400 --eval-every 400 --workers 3 --seed 0")
AUXS1_HEAD = "--aux-drv --w-aux-drv 0.1 --aux-drv-lr 2e-4 --aux-drv-labels /w/store"
AUX_GROUPS = ("q", "pos_mlp", "blocks.0", "blocks.1", "norm", "out")
_FACTS: dict = {}


def aux_block(ns, **mut) -> dict:
    """the `aux_drv` block the trainer's data event carries (BUILT facts), for this namespace: AuxDrivable.built_facts() of a REAL head + the trainer-side keys"""
    import aux_drivable_head as AUX
    cfg = AUX.AuxDrivableConfig(tap=ns.aux_drv_tap, depth=ns.aux_drv_depth, feed=ns.aux_drv_feed, w_aux=ns.w_aux_drv, seed=ns.seed)
    k = (cfg.depth, cfg.feed)
    if k not in _FACTS:
        _FACTS[k] = AUX.AuxDrivable(cfg).built_facts()
    f = dict(_FACTS[k], w_aux=ns.w_aux_drv)
    b = {**f, "tap": cfg.tap, "feed": cfg.feed, "lr": ns.aux_drv_lr, "store_sha256": AUX.PINNED_STORE_SHA256, "frames": AUX.PINNED_STORE_KEYS,
         "shuffle_seed": ns.aux_drv_shuffle_seed, "head_params_in_optimiser": f["n_params"]}
    if ns.aux_drv_shuffle_seed is not None:
        b.update(shuffle_keys=AUX.PINNED_STORE_KEYS, derangement_fixed_points=0)
    b.update(mut)
    return b


def aux_g12(ns, **mut) -> dict:
    """a G12 exactly as scorer_finetune's g12_probe writes it (the same raw-field names), all proved"""
    g = {"ok": True, "w_aux": ns.w_aux_drv, "feed": ns.aux_drv_feed, "labelled_frames_in_batch": 8, "batch_frames": 8, "mutate": [], "scene_proj_calls": 2,
         "tap_is_visual_ctx": True, "tap_is_scene_ctx": False, "score_to_head_grads_all_none": True, "bce_declared": 0.70, "huber_declared": 0.4, "bce_w0": 0.70, "huber_w0": 0.4,
         "per_group_declared": {k: (1.5 if ns.w_aux_drv > 0 else 0.0) for k in AUX_GROUPS}, "per_group_w0": {k: 0.0 for k in AUX_GROUPS}, "bce_near_ln2": True,
         "adam_head_params": 43, "adam_head_params_with_state": 43, "head_retained_gb": None}
    if ns.aux_drv_feed == "off":
        g.update(reference_replay_identical=True, scorer_params_bit_identical_2steps=True, identity_probe_reads_one_ulp=True)
    else:
        g.update(feed_zero_init=True, feed_step0_equal_base=True, feed_probe_reads_difference=True, feed_proj_grad_abs=0.3)
    g.update(mut)
    return g


def aux_pf_rows(argv, *, g12=None, data_aux=None, drop=(), notok=(), **kw) -> list:
    """rls1_pf_rows + the head's G12 (full raw fields) and the data event's `aux_drv` block, for an argv that carries --aux-drv"""
    import scorer_finetune as S
    ns, err = C.parse_scorer_argv(S.build_parser(), argv)
    assert err is None, err
    rows = rls1_pf_rows(argv, drop=drop, notok=notok, **kw)
    k12 = next(k for k in rls1_gate_keys() if k.startswith("G12_"))
    if ns.aux_drv:
        if "G12" not in drop:
            rows[-1]["gates"][k12] = dict(g12 if g12 is not None else aux_g12(ns), ok=(k12 not in notok) and (g12 or {}).get("ok", True))
        if rows[0].get("event") == "data":
            rows[0]["aux_drv"] = data_aux if data_aux is not None else aux_block(ns)
    return rows


def aux_s1_static() -> None:
    import scorer_finetune as S
    import aux_drivable_head as AUX
    P = L.PROFILES["refe_aux_s1"]
    PR = L.PROFILES["refe_rls1"]
    base = L.parse_args_string(AUXS1_BASE)
    h1 = base + L.parse_args_string(AUXS1_HEAD)
    h2 = h1 + ["--aux-drv-feed", "on"]
    seeded = lambda argv, k: C._set(argv, "--seed", str(k))                      # noqa: E731

    def refs(argv, arm=None, prof=None):
        return L.profile_refusals(prof or ({**P, "declared_arm": arm} if arm else P), argv)

    def clean(argv, arm=None):
        rs = refs(argv, arm)
        ok(rs == [], (arm, rs))

    def red(argv, frag, arm=None, prof=None):
        rs = refs(argv, arm, prof)
        ok(any(frag in r for r in rs), (frag, arm, rs))

    def without(argv, flag, nvalues=1):
        return C._drop(argv, flag, nvalues)
    # ------------------------------------------------------------------ A1 the profile's shape; RL-S1 stays head-free; the registry classifies every new flag
    ok(P["parser_module"] == "scorer_finetune" and P["trainer_src"] == "scorer_finetune.py" and P["runner"] == "scorer" and P["arm_rules"] == "auxs1" and P["mintable"] and P["require_commit"])
    ok(P["namespace_funcs"] == C.SCORER_NAMESPACE_FUNCS and PR["namespace_funcs"] == C.SCORER_NAMESPACE_FUNCS and "aux_s1_refusals" in C.SCORER_NAMESPACE_FUNCS)
    ok(set(L.AUXS1_ARM_CLASS) == {f"{c}-s{k}" for c in ("N", "H1", "H2") for k in (0, 1, 2)} and len(L.AUXS1_SEEDS) >= 3 and all(L.AUXS1_ARM_SEED[a_] in L.AUXS1_SEEDS for a_ in L.AUXS1_ARM_CLASS))
    for e in L.AUXS1_SUITE:
        ok((HERE / e["script"]).is_file() and e["token"] in (HERE / e["script"]).read_text(encoding="utf-8"), e)
    ok({"test_aux_s1_wiring.py", "test_aux_drivable_head.py", "test_rl_epo.py"} <= {e["script"] for e in L.AUXS1_SUITE})
    ap = S.build_parser()
    opt_of = {x.dest: x.option_strings[0] for x in ap._actions if x.option_strings}
    aux_dests = ("aux_drv", "w_aux_drv", "aux_drv_tap", "aux_drv_depth", "aux_drv_feed", "aux_drv_labels", "aux_drv_shuffle_seed", "aux_drv_lr")
    ok(set(L.AUX_HEAD_FLAGS) == {opt_of[d] for d in aux_dests})
    ok(all(C.DVB_KINDS_SCORER.get(d) == ("data" if d == "aux_drv_labels" else "built") for d in aux_dests), {d: C.DVB_KINDS_SCORER.get(d) for d in aux_dests})
    ok(set(C.DVB_KINDS_SCORER) == {x.dest for x in ap._actions if x.dest != "help"})                         # a new flag without an entry is refused by the gate itself (tested in R10 + A6)
    ok({"--aux-drv-labels"} <= set(L.RLS1_DATA) and set(L.RLS1_DATA) <= set(opt_of.values()) and all(f in L.RLS1_FORBIDDEN for f in L.AUX_HEAD_FLAGS))
    ok({d: ap.get_default(d) for d in aux_dests} == {"aux_drv": False, "w_aux_drv": 0.0, "aux_drv_tap": "visual", "aux_drv_depth": 2, "aux_drv_feed": "off", "aux_drv_labels": "",
                                                       "aux_drv_shuffle_seed": None, "aux_drv_lr": None})                                    # every default DISABLES the head
    ok("G12" in C.SCORER_GATES and "aux_drv" in " ".join(rls1_gate_keys()))
    for argv, req in ((base, False), (h1, True), (h2, True)):
        ns_, _ = C.parse_scorer_argv(S.build_parser(), argv)
        ok(("G12" in C.scorer_required_gates(ns_)) is req)                                                    # G12 is obliged by --aux-drv and by nothing else
    # RL-S1: each head flag is FORBIDDEN in its argv (any RL-S1 re-ship passes the gate only with the head flags OFF)
    r0 = L.parse_args_string(RLS1_BASE) + ["--epo", "--epo-terms-train", "/w/terms"]
    ok(L.profile_refusals({**PR, "declared_arm": "R0"}, r0) == [])
    for f_ in L.AUX_HEAD_FLAGS:
        ok(any("FORBIDS" in r and f_ in r for r in L.profile_refusals({**PR, "declared_arm": "R0"}, r0 + [f_] + ([] if f_ == "--aux-drv" else ["1"]))), f_)
    # ------------------------------------------------------------------ A2 the clean argvs: the 9 arms (declared) and the 3 classes (undeclared)
    for k in (0, 1, 2):
        clean(seeded(base, k), f"N-s{k}")
        clean(seeded(h1, k), f"H1-s{k}")
        clean(seeded(h2, k), f"H2-s{k}")
    clean(base)
    clean(h1)
    clean(h2)
    ok([L.auxs1_arm_class(a_) for a_ in (base, h1, h2, h1 + ["--aux-drv-feed", "off"])] == ["N", "H1", "H2", "H1"])
    ok(L.auxs1_control_argv(h2) == base and L.auxs1_control_argv(h1) == base and L.auxs1_control_argv(base) == base)       # a head arm and N differ by the head flags and nothing else
    clean(L.auxs1_control_argv(h2), "N-s0")
    ok(L.profile_refusals({**P, "declared_arm": "N-s0"}, base) == [] and "aux_bce" not in " ".join(L.profile_refusals({**P, "declared_arm": "N-s0"}, base)))
    # ------------------------------------------------------------------ A3 the head arms NAME every flag whose default disables the head; N carries none of them
    for flag, nv in (("--aux-drv", 0), ("--w-aux-drv", 1), ("--aux-drv-lr", 1), ("--aux-drv-labels", 1)):
        for arm, argv in (("H1-s0", h1), ("H2-s0", h2)):
            red(without(argv, flag, nv), f"requires {flag} for {arm}", arm)
    red(without(h2, "--aux-drv-feed"), "requires --aux-drv-feed on for H2-s0", "H2-s0")                       # H2 without its feed named is an H1 arm
    red(without(h2, "--aux-drv-feed"), "arm H2-s0 is the H2 class; the argv resolves to the H1 class", "H2-s0")
    red(h2, "arm H1-s0 is the H1 class; the argv resolves to the H2 class", "H1-s0")
    red(h1, "must NOT carry", "N-s0")
    red(h2, "must NOT carry", "N-s1")
    red(base + ["--aux-drv-lr", "2e-4"], "must NOT carry", "N-s0")
    red(base + ["--aux-drv-lr", "2e-4"], "scorer_finetune refuses")                                           # undeclared: the trainer's own refusal surfaces
    red(base + ["--w-aux-drv", "0.1"], "w-aux-drv is set but --aux-drv is not")
    red(without(h1, "--aux-drv", 0), "w-aux-drv is set but --aux-drv is not", "H1-s0")
    red(without(h1, "--aux-drv-labels"), "needs --aux-drv-labels", "H1-s0")
    red(without(h1, "--aux-drv-lr"), "needs --aux-drv-lr", "H1-s0")
    # ------------------------------------------------------------------ A4 registered values, recipe, seeds; forbidden flags; exact spelling; lane pairing
    for flag, bad, frag in (("--w-aux-drv", "0.2", "requires --w-aux-drv 0.1"), ("--w-aux-drv", "0", "requires --w-aux-drv 0.1"), ("--aux-drv-lr", "1e-4", "requires --aux-drv-lr 0.0002"),
                            ("--aux-drv-depth", "3", "requires --aux-drv-depth 2"), ("--aux-drv-depth", "1", "requires --aux-drv-depth 2")):
        red(C._set(h1, flag, bad) if flag in h1 else h1 + [flag, bad], frag, "H1-s0")
        red(C._set(h2, flag, bad) if flag in h2 else h2 + [flag, bad], frag, "H2-s0")
    red(h1 + ["--aux-drv-tap", "scene"], "requires --aux-drv-tap visual", "H1-s0")
    red(h1 + ["--aux-drv-tap", "scene"], "scene is not built")                                                # and the trainer refuses it on its own
    clean(h1 + ["--aux-drv-depth", "2", "--aux-drv-tap", "visual"], "H1-s0")                                  # naming a registered value is fine
    for flag, bad in (("--lr", "3e-5"), ("--max-updates", "500"), ("--b-mode", "listnet"), ("--lam", "0.3"), ("--tau-s", "0.2"), ("--eval-every", "100")):
        for arm, argv in (("N-s0", base), ("H1-s0", h1), ("H2-s0", h2)):
            red(C._set(argv, flag, bad), "requires " + flag, arm)
    for flag, bad in (("--warmup", "10"), ("--batch", "4"), ("--accum", "4"), ("--lane-weight", "0.5")):
        for arm, argv in (("N-s0", base), ("H1-s0", h1)):
            red(argv + [flag, bad], "requires " + flag, arm)
    for flag, nv in (("--lr", 1), ("--max-updates", 1), ("--b-mode", 1), ("--lam", 1), ("--tau-s", 1), ("--lane-head", 0), ("--pdm-labels-train", 1), ("--require-pdm", 0)):
        red(without(h1, flag, nv), "requires " + flag, "H1-s0")
    red(seeded(h2, 1), "registered with --seed 0", "H2-s0")
    red(seeded(base, 3), "registered with --seed 0", "N-s0")
    ok(any("unknown arm" in r for r in L.profile_refusals({**P, "declared_arm": "H3-s0"}, h1)))
    ok(any("unknown arm" in r for r in L.profile_refusals({**P, "declared_arm": "R0"}, h1)))                 # an RL-S1 arm name is not an aux arm
    for extra, frag in ((["--preflight"], "FORBIDS"), (["--lane-labels-train", "/w/ll"], "FORBIDS"), (["--b-compw", "1,1,1,1,1,1"], "FORBIDS"), (["--tau-t", "0.2"], "FORBIDS"),
                        (["--aux-drv-shuffle-seed", "7"], "FORBIDS"), (["--epo"], "FORBIDS"), (["--epo-terms-train", "/w/t"], "FORBIDS"), (["--epo-lam", "1.0"], "FORBIDS"),
                        (["--epo-k"], "FORBIDS")):
        red(h1 + extra, frag, "H1-s0")
        red(base + extra, frag, "N-s0")
    red(without(h1, "--teacher-lane-heldout"), "--lane-head needs", "H1-s0")
    red(h1 + ["--epo-bogus", "1"], "parser rejects", "H1-s0")
    red(without(h1, "--ckpt"), "parser rejects", "H1-s0")
    red(["--require-p" if t == "--require-pdm" else t for t in h1], "not an exact option", "H1-s0")
    red(h1, "DECLARED, NOT BUILT", prof={**P, "declared_arm": "H1-s0", "required_flags": {**P["required_flags"], "--g-probe-lever": None}})
    red(h1, "scorer_finetune.py never emits it", prof={**P, "declared_arm": "H1-s0", "live_terms": ("g_probe_term",)})
    terms_ = L.AUXS1_HEAD_TERMS
    L.AUXS1_HEAD_TERMS = ("g_probe_aux_term",)
    try:
        red(h1, "scorer_finetune.py never emits it", "H1-s0")                                              # a head arm requires its aux_bce / aux_huber rows ...
        clean(base, "N-s0")                                                                               # ... N does not
    finally:
        L.AUXS1_HEAD_TERMS = terms_
    clean(h1, "H1-s0")
    # ------------------------------------------------------------------ A5 the CLI: the arm name belongs to the profile it is passed with
    def cli(*args):
        return subprocess.run([sys.executable, str(HERE / "launch_gate.py"), *args], capture_output=True, text=True, encoding="utf-8")
    rc = cli("refusals", "--profile", "refe_aux_s1", "--arm-name", "H2-s1", "--args", " ".join(seeded(h2, 1)))
    ok(rc.returncode == 0 and "ZZGATE-CLEANZZ" in rc.stdout, rc.stdout[-400:] + rc.stderr[-400:])
    rc = cli("refusals", "--profile", "refe_aux_s1", "--arm-name", "H1-s1", "--args", " ".join(seeded(h2, 1)))
    ok(rc.returncode == 5 and "ZZGATE-REFUSEDZZ" in rc.stdout and "H2 class" in rc.stdout, rc.stdout[-400:])
    rc = cli("refusals", "--profile", "refe_aux_s1", "--arm-name", "R0", "--args", " ".join(h2))
    ok(rc.returncode != 0 and "is not an arm of refe_aux_s1" in (rc.stdout + rc.stderr), rc.stderr[-300:])
    rc = cli("refusals", "--profile", "refe_rls1", "--arm-name", "H1-s0", "--args", " ".join(h2))
    ok(rc.returncode != 0 and "is not an arm of refe_rls1" in (rc.stdout + rc.stderr), rc.stderr[-300:])
    with tempfile.TemporaryDirectory() as td_:
        rc = cli("run", "--profile", "refe_aux_s1", "--args", " ".join(h2), "--pkg", td_, "--out-dir", str(Path(td_) / "o"))
        ok(rc.returncode != 0 and "mints for ONE named arm" in (rc.stdout + rc.stderr), rc.stderr[-300:])
    # ------------------------------------------------------------------ A6 G-HYG on the scorer reads the new namespace; G-DVB: the observed rows, one mutation each, the registry arms
    ev = C.check_hyg(profile=P)
    reads = C.namespace_reads((HERE / "scorer_finetune.py").read_text(encoding="utf-8"), C.SCORER_NAMESPACE_FUNCS)
    ok(ev["status"] == "PASS" and {d for d in aux_dests} <= reads, ev["details"])
    for arm, frag in (("no_cfg_guard", "ACCEPTED an undeclared attribute"), ("undeclared_namespace_read", "undeclared_lever_g_hyg")):
        ev = C.check_hyg(arm=arm, profile=P)
        ok(ev["status"] == "FAIL" and frag in " ".join(ev["reasons"]), (arm, ev["reasons"]))
    ns_h1, _ = C.parse_scorer_argv(S.build_parser(), h1)
    ns_h2, _ = C.parse_scorer_argv(S.build_parser(), h2)
    sh = h1 + ["--aux-drv-shuffle-seed", "7"]                                                              # a CONTROL argv (the profile forbids it; the check itself is profile-agnostic)
    Pn, P1, P2 = {**P, "declared_arm": "N-s0"}, {**P, "declared_arm": "H1-s0"}, {**P, "declared_arm": "H2-s0"}
    Pc = {**P1, "forbidden_flags": tuple(f for f in P1["forbidden_flags"] if f != "--aux-drv-shuffle-seed")}

    def pf_of(argv, name="p.jsonl", **kw):
        return C.load_scorer_preflight(rls1_write(Path(tempfile.mkdtemp()) / name, aux_pf_rows(argv, **kw)))
    ev = C.check_dvb_scorer(h1, pf_of(h1), P1)
    ok(ev["status"] == "PASS" and ev["details"]["observed_rows"] == 19 and ev["details"]["unknown_flags"] == [] and ev["details"]["stale_registry"] == [], ev["reasons"])
    ev = C.check_dvb_scorer(h2, pf_of(h2), P2)
    ok(ev["status"] == "PASS" and ev["details"]["observed_rows"] == 19, ev["reasons"])
    ev = C.check_dvb_scorer(sh, pf_of(sh), Pc)
    ok(ev["status"] == "PASS" and ev["details"]["observed_rows"] == 21, ev["reasons"])
    ev = C.check_dvb_scorer(base, pf_of(base), Pn)
    ok(ev["status"] == "PASS" and ev["details"]["observed_rows"] == 4, ev["reasons"])                      # N: b_mode, b_compw, the epo presence row + the `aux_drv` presence row (no head: nothing else is observed)
    ok(sum(ev["details"]["kinds"].values()) == len(C.DVB_KINDS_SCORER))
    for arm, frag in (("unknown_flag", "NO declared-vs-built entry"), ("stale_registry", "no longer has"), ("dead_lever", "never read by the trainer"),
                      ("unbound_data", "would NOT fingerprint"), ("aux_lever_ignored", "--w-aux-drv: declared 0.1 != built 0.15")):
        ev = C.check_dvb_scorer(h1, pf_of(h1), P1, arm=arm)
        ok(ev["status"] == "FAIL" and frag in " ".join(ev["reasons"]) and ev["arm"] == arm, (arm, ev["reasons"]))
    ev = C.check_dvb_scorer(h1, pf_of(h1), {**P1, "data_flags": tuple(f for f in P1["data_flags"] if f != "--aux-drv-labels")})
    ok(ev["status"] == "FAIL" and "would NOT fingerprint (not in refe_aux_s1's data_flags): ['--aux-drv-labels']" in " ".join(ev["reasons"]), ev["reasons"])       # the label store must be bound by the token
    ok(C.check_dvb_scorer(h1, pf_of(h1), P1)["status"] == "PASS")                                           # the arms left no residue
    kept = C.DVB_KINDS_SCORER.pop("aux_drv_lr")
    try:
        ev = C.check_dvb_scorer(h1, pf_of(h1), P1)
        ok(ev["status"] == "FAIL" and "aux_drv_lr" in " ".join(ev["reasons"]) and "NO declared-vs-built entry" in " ".join(ev["reasons"]), ev["reasons"])      # a new flag without an entry
    finally:
        C.DVB_KINDS_SCORER["aux_drv_lr"] = kept
    kept = C.DVB_KINDS_SCORER["aux_drv_lr"]
    C.DVB_KINDS_SCORER["aux_drv_lr"] = "consumed"
    try:
        ok(C.check_dvb_scorer(h1, pf_of(h1), P1)["status"] == "PASS")                                       # `consumed` is static; the head optimiser's lr row is still observed
    finally:
        C.DVB_KINDS_SCORER["aux_drv_lr"] = kept
    for nm, argv, prof_, mut, frag in (
            ("weight", h1, P1, {"w_aux": 0.15}, "--w-aux-drv: declared 0.1 != built 0.15"), ("weight zero", h1, P1, {"w_aux": 0.0}, "--w-aux-drv: declared 0.1 != built 0.0"),
            ("tap", h1, P1, {"tap": "scene"}, "--aux-drv-tap: declared 'visual' != built 'scene'"), ("depth", h1, P1, {"depth": 1}, "--aux-drv-depth: declared 2 != built 1"),
            ("feed built on", h1, P1, {"feed_built": True}, "--aux-drv-feed: declared 'off' != built 'on'"), ("feed not built", h2, P2, {"feed_built": False}, "--aux-drv-feed: declared 'on' != built 'off'"),
            ("width 60 for H2", h2, P2, {"scorer_in_width": 60}, "declared 63 != built 60"), ("width 63 for H1", h1, P1, {"scorer_in_width": 63}, "declared 60 != built 63"),
            ("lr", h1, P1, {"lr": 1e-4}, "--aux-drv-lr: declared 0.0002 != built 0.0001"), ("params not all in the optimiser", h1, P1, {"head_params_in_optimiser": 100},
                                                                                              "every head parameter is in the head optimiser"),
            ("store sha", h1, P1, {"store_sha256": "0" * 64}, "aux_drv.store_sha256"), ("frames", h1, P1, {"frames": 103138}, "aux_drv.frames"),
            ("cell", h1, P1, {"cell": 4.0}, "aux_drv.cell"), ("x0", h1, P1, {"x0": -9.0}, "aux_drv.x0"), ("y0", h1, P1, {"y0": -30.0}, "aux_drv.y0"),
            ("nx", h1, P1, {"nx": 20}, "aux_drv.nx"), ("ny", h1, P1, {"ny": 16}, "aux_drv.ny"),
            ("shuffle seed ignored", sh, Pc, {"shuffle_seed": None}, "--aux-drv-shuffle-seed: declared 7 != built None"),
            ("shuffle has fixed points", sh, Pc, {"derangement_fixed_points": 3}, "the shuffle map has no fixed point"),
            ("shuffle covers too few keys", sh, Pc, {"shuffle_keys": 1000}, "covers every key of the store")):
        ev = C.check_dvb_scorer(argv, pf_of(argv, data_aux=aux_block(C.parse_scorer_argv(S.build_parser(), argv)[0], **mut)), prof_)
        ok(ev["status"] == "FAIL" and frag in " ".join(ev["reasons"]), (nm, ev["reasons"]))
    ev = C.check_dvb_scorer(h1, pf_of(h1, data_aux={}), P1)                                               # a head arm whose data event carries no aux block: nothing was built
    ok(ev["status"] == "FAIL" and any("declared" in r for r in ev["reasons"]))
    rows_noaux = aux_pf_rows(h1)
    del rows_noaux[0]["aux_drv"]
    ev = C.check_dvb_scorer(h1, C.load_scorer_preflight(rls1_write(Path(tempfile.mkdtemp()) / "p.jsonl", rows_noaux)), P1)
    ok(ev["status"] == "FAIL" and "--aux-drv: declared True != built False" in " ".join(ev["reasons"]), ev["reasons"])
    ev = C.check_dvb_scorer(base, pf_of(base, data_mut={"aux_drv": aux_block(ns_h1)}), Pn)               # N whose preflight built a head
    ok(ev["status"] == "FAIL" and "--aux-drv: declared False != built True" in " ".join(ev["reasons"]), ev["reasons"])
    # ------------------------------------------------------------------ A7 the preflight consumer: G12 obliged by --aux-drv, re-judged from its raw fields (its own ok is not trusted)
    td = Path(tempfile.mkdtemp())

    def cons(argv, prof_, name="aux_pf.jsonl", **kw):
        p_ = td / name
        rls1_write(p_, aux_pf_rows(argv, **kw))
        return L.consume_scorer_preflight(p_, argv, prof_)
    for nm, argv, prof_ in (("N", base, Pn), ("H1", h1, P1), ("H2", h2, P2)):
        ev = cons(argv, prof_)
        ok(ev["status"] == "PASS" and ("G12" in ev["details"]["required"]) == (nm != "N"), (nm, ev["reasons"]))
    ev = cons(h1, P1, drop=("G12",))
    ok(ev["status"] == "FAIL" and "ABSENT: ['G12']" in " ".join(ev["reasons"]), ev["reasons"])               # the headline: a head arm without G12 has not been proved live
    ev = cons(h2, P2, drop=("G12",))
    ok(ev["status"] == "FAIL" and "ABSENT: ['G12']" in " ".join(ev["reasons"]))
    ev = cons(h1, P1, notok=("G12_aux_drv_live_and_identity",))
    ok(ev["status"] == "FAIL" and "not ok" in " ".join(ev["reasons"]))
    k12 = next(k for k in rls1_gate_keys() if k.startswith("G12_"))
    ev = cons(base, Pn, add={k12: {"ok": True}})
    ok(ev["status"] == "FAIL" and "DIFFERENT configuration" in " ".join(ev["reasons"]))                      # N offered a head run's preflight
    ev = cons(h1, P1, g12=aux_g12(ns_h1, ok=False))
    ok(ev["status"] == "FAIL" and "not ok" in " ".join(ev["reasons"]))
    ns_w0, _ = C.parse_scorer_argv(S.build_parser(), C._set(h1, "--w-aux-drv", "0"))
    ev = cons(C._set(h1, "--w-aux-drv", "0"), {**P1}, g12=aux_g12(ns_w0))
    ok(ev["status"] == "PASS", ev["reasons"])                                                              # a declared weight of 0 reads all-zero and is fine as such
    for nm, mut, frag in (
            ("regression switch", {"mutate": ["REFE_AUX_MUTATE_TAP"]}, "regression switch"), ("no labelled frame", {"labelled_frames_in_batch": 0}, "no labelled frame"),
            ("a group is dead", {"per_group_declared": {**aux_g12(ns_h1)["per_group_declared"], "q": 0.0}}, "non-zero on every head sub-module"),
            ("a group is nan", {"per_group_declared": {**aux_g12(ns_h1)["per_group_declared"], "out": float("nan")}}, "non-zero on every head sub-module"),
            ("no groups", {"per_group_declared": {}}, "no per-sub-module gradient read"),
            ("control reads non-zero", {"per_group_w0": {**aux_g12(ns_h1)["per_group_w0"], "norm": 1e-9}}, "EXACTLY 0.0"),
            ("control missing a group", {"per_group_w0": {k: 0.0 for k in AUX_GROUPS[:-1]}}, "EXACTLY 0.0"), ("control empty", {"per_group_w0": {}}, "EXACTLY 0.0"),
            ("probed weight != argv", {"w_aux": 0.2}, "probed weight"), ("probed feed != argv", {"feed": "on"}, "probed feed"),
            ("bce far from ln2", {"bce_declared": 1.2}, "within 0.15 of ln 2"), ("bce flag lies", {"bce_near_ln2": False}, "within 0.15 of ln 2"),
            ("tap is a copy", {"tap_is_visual_ctx": False}, "storage of visual_ctx"), ("tap is scene_ctx", {"tap_is_scene_ctx": True}, "storage of visual_ctx"),
            ("path from score to head", {"score_to_head_grads_all_none": False}, "autograd path"),
            ("adam state missing", {"adam_head_params_with_state": 42}, "hold Adam state"), ("no head params", {"adam_head_params": 0, "adam_head_params_with_state": 0}, "hold Adam state"),
            ("memory", {"head_retained_gb": 1.5}, "> 1 GB"),
            ("scorer not bit identical", {"scorer_params_bit_identical_2steps": False}, "scorer_params_bit_identical_2steps"),
            ("reference not deterministic", {"reference_replay_identical": False}, "reference_replay_identical"),
            ("identity probe cannot read a difference", {"identity_probe_reads_one_ulp": False}, "identity_probe_reads_one_ulp")):
        ev = cons(h1, P1, g12=aux_g12(ns_h1, **mut))
        ok(ev["status"] == "FAIL" and frag in " ".join(ev["reasons"]), (nm, ev["reasons"]))
    for nm, mut, frag in (("feed not zero-initialised", {"feed_zero_init": False}, "feed_zero_init"), ("step 0 differs from the base", {"feed_step0_equal_base": False}, "feed_step0_equal_base"),
                          ("probe cannot see a feed", {"feed_probe_reads_difference": False}, "feed_probe_reads_difference"),
                          ("feed unreachable", {"feed_proj_grad_abs": 0.0}, "received no gradient"), ("feed grad missing", {"feed_proj_grad_abs": None}, "received no gradient")):
        ev = cons(h2, P2, g12=aux_g12(ns_h2, **mut))
        ok(ev["status"] == "FAIL" and frag in " ".join(ev["reasons"]), (nm, ev["reasons"]))
    g_dead = aux_g12(ns_h1, per_group_declared={k: 0.0 for k in AUX_GROUPS})
    ev = cons(h1, P1, g12=g_dead)
    ok(ev["status"] == "FAIL" and "non-zero on every head sub-module" in " ".join(ev["reasons"]))            # the loss never reached the head
    ns_w0b = ns_w0
    ev = cons(C._set(h1, "--w-aux-drv", "0"), P1, g12=aux_g12(ns_w0b, per_group_declared={k: 0.5 for k in AUX_GROUPS}))
    ok(ev["status"] == "FAIL" and "received gradient" in " ".join(ev["reasons"]))                           # w = 0 that trained is a leak
    # ------------------------------------------------------------------ A8 END TO END on the real run_checks / finalize for an H2 arm (G-SUITE excluded: it would recurse into this file)
    with tempfile.TemporaryDirectory() as td2:
        td2 = Path(td2)
        pkg = mini_pkg(td2 / "pkg")
        key = GT.load_key(td2 / "k" / "gate.key", create=True, forbid_under=[pkg])
        real = [t.replace("/w", td2.as_posix() + "/w") for t in seeded(h2, 1)]
        (td2 / "w").mkdir()
        for f_ in ("--ckpt", "--heldout-logs", "--calib"):
            Path(GT.flag_values(real, f_)[0]).write_text("x", encoding="utf-8")
        for f_ in P["data_flags"]:
            if f_ in real and not Path(GT.flag_values(real, f_)[0]).exists():
                Path(GT.flag_values(real, f_)[0]).mkdir(parents=True)
        Pe = {**P, "declared_arm": "H2-s1", "checks": ("G-HYG", "G-DVB", "G-PREFLIGHT"), "require_commit": False}
        ok(L.profile_refusals(Pe, real) == [], L.profile_refusals(Pe, real))
        pfp = rls1_write(td2 / "preflight.jsonl", aux_pf_rows(real))
        out = td2 / "gate"
        st = L.run_checks(Pe, real, pkg, out, preflight_json=pfp)
        ok(st == {"G-HYG": "PASS", "G-DVB": "PASS", "G-PREFLIGHT": "PASS", "G-CLOCK": "NA"}, st)
        v, pth, tok = L.finalize(Pe, real, pkg, out, key, preflight_json=pfp)
        ok(v == "PASS" and tok["binding"]["profile"] == "refe_aux_s1" and tok["binding"]["trainer"] == "refe/scorer_finetune.py", tok["reasons"])
        ok("--aux-drv-labels" in tok["binding"]["data"] and tok["binding"]["data"]["gate:preflight_json"]["sha256"] == GT.sha256_file(pfp))      # the label store IS bound by the token
        good_, why, _ = GT.verify_token(pth, real, pkg, key=key, profile="refe_aux_s1", data_flags=P["data_flags"])
        ok(good_, why)
        ok(not GT.verify_token(pth, L.auxs1_control_argv(real), pkg, key=key, profile="refe_aux_s1", data_flags=P["data_flags"])[0])        # the no-head argv is not H2's
        ok(not GT.verify_token(pth, C._set(real, "--w-aux-drv", "0.2"), pkg, key=key, profile="refe_aux_s1", data_flags=P["data_flags"])[0])  # a changed weight is another launch
        rls1_write(pfp, aux_pf_rows(real, drop=("G12",)))
        out_b = td2 / "gate_nog12"
        st = L.run_checks(Pe, real, pkg, out_b, preflight_json=pfp)
        v, _, t = L.finalize(Pe, real, pkg, out_b, key, preflight_json=pfp)
        ok(st["G-PREFLIGHT"] == "FAIL" and v == "FAIL" and any("G-PREFLIGHT" in r for r in t["reasons"]))
        rls1_write(pfp, aux_pf_rows(real, data_aux=aux_block(C.parse_scorer_argv(S.build_parser(), real)[0], depth=1)))
        out_c = td2 / "gate_depth"
        st = L.run_checks(Pe, real, pkg, out_c, preflight_json=pfp)
        ok(st["G-DVB"] == "FAIL" and st["G-PREFLIGHT"] == "PASS", st)                                          # built depth 1 for an argv that says 2: G-DVB alone refuses it
    with tempfile.TemporaryDirectory() as td3:
        st = L.run_checks({**P, "declared_arm": "H1-s0"}, h2, PKG, Path(td3) / "o")
        ok(set(st.values()) == {"FAIL"} and set(st) == set(P["checks"]) | set(P["na_ok"]), st)             # a refused argv fails every check including the suite and starts nothing


def full() -> None:
    import torch
    root = Path(tempfile.mkdtemp(prefix="lg_full_"))
    fx = F.tiny_p2_fixture(str(root))
    argv = F.rehearsal_argv(fx, str(root / "launch_out"))
    prof = L.PROFILES["refe_rehearsal"]
    ok(L.profile_refusals(prof, argv) == [], L.profile_refusals(prof, argv))
    # ---------------------------------------------------------------- F1 every check PASSes clean on the real trainer (rehearsal scale)
    out = root / "clean"
    st = L.run_checks(prof, argv, PKG, out)
    ok(all(st[c] == "PASS" for c in prof["checks"]) and st["G-CLOCK"] == "NA", st)
    live = GT.read_json(out / "evidence_G-LIVE.json")["details"]
    ok(all(live["groups"][g]["max_abs_sum"] > 0 for g in C.TRAINABLE_GROUPS) and live["frozen_control_exact_zero"] is True)
    ok(live["zero_loss_control"]["zero"] == 0.0 and live["zero_loss_control"]["real"] > 0.0)
    dvb = GT.read_json(out / "evidence_G-DVB.json")["details"]
    ok(dvb["observed_rows"] >= 30 and all(r["ok"] for r in dvb["rows"]) and dvb["unknown_flags"] == [] and dvb["stale_registry"] == [])
    ok(GT.read_json(out / "evidence_G-CKPT.json")["details"]["model_tensors_differing"] == 0)
    # the evidence is REHEARSAL: it can never mint
    key = GT.load_key(root / "k" / "gate.key", forbid_under=[PKG])
    v, _, t = L.finalize(prof, argv, PKG, out, key)
    ok(v == "FAIL" and any("REHEARSAL" in r for r in t["reasons"]))
    # ---------------------------------------------------------------- F2 one deliberate-regression arm per check: each makes ITS check FAIL
    arms = [("G-DVB", "lever_ignored", "declared False != built True"), ("G-LIVE", "zero_score", "EXACTLY zero"),
            ("G-LIVE", "frozen_group", "is FROZEN"), ("G-EVAL", "planner_cfg_differs", "planner builds a different config"),
            ("G-EVAL", "loader_drops_tensor", "differ between the trained model and the planner"), ("G-CKPT", "resume_drops_opt", "optimiser state"),
            ("G-SUITE", "suite_failing_test", "ZZTEST_PASS")]
    for chk, arm, frag in arms:
        o2 = root / f"arm_{arm}"
        s2 = L.run_checks(prof, argv, PKG, o2, only={chk}, arm=arm)
        ev = GT.read_json(o2 / f"evidence_{chk}.json")
        ok(s2[chk] == "FAIL" and frag in " ".join(ev["reasons"]) and ev["arm"] == arm, (arm, ev["reasons"][:2]))
        v, _, t = L.finalize({**prof, "mintable": True, "rehearsal": False}, argv, PKG, o2, key)
        ok(v == "FAIL")
    # ---------------------------------------------------------------- F3 G-CLOCK against a registered schedule (--epochs form), with its arm
    cl_argv = [t for t in argv]
    cl_argv = C._set(C._drop(cl_argv, "--steps", 1), "--epochs", "1")
    cl_prof = {**prof, "expect_schedule": {"epochs": 1, "steps": 2, "epoch_len": 4}}
    o3 = root / "clock"
    ok(L.run_checks(cl_prof, cl_argv, PKG, o3, only={"G-CLOCK"})["G-CLOCK"] == "PASS")
    ok(L.run_checks(cl_prof, cl_argv, PKG, root / "clock_arm", only={"G-CLOCK"}, arm="clock_wrong_epochs")["G-CLOCK"] == "FAIL")
    ok(L.run_checks({**cl_prof, "expect_schedule": {"epochs": 1, "steps": 3, "epoch_len": 4}}, cl_argv, PKG, root / "clock2", only={"G-CLOCK"})["G-CLOCK"] == "FAIL")
    # ---------------------------------------------------------------- F4 the smoke-bank carve reaches the scorer on every micro-batch
    cb = F.carve_gate_bank(fx["bank"], fx["sets"], str(root / "carved"))
    ok(cb["keys"] == 6 and cb["rows"]["targets_rank0.jsonl"] + cb["rows"].get("targets_rank1.jsonl", 0) == 6, cb)
    # ---------------------------------------------------------------- F5 END TO END on a mintable copy of the rehearsal: run -> evidence -> token -> verify -> refuse
    mp = {**prof, "name": "refe_p2", "mintable": True, "rehearsal": False, "require_commit": False}
    pkg2 = root / "pkg"
    for sub in ("refe", "code", "splits"):
        (pkg2 / sub).mkdir(parents=True)
    for f in HERE.glob("*.py"):
        shutil.copyfile(f, pkg2 / "refe" / f.name)
    shutil.copyfile(HERE / "p2r_pod.sh", pkg2 / "refe" / "p2r_pod.sh")
    o5 = root / "e2e"
    L.run_checks(mp, argv, pkg2, o5)
    v, pth, t = L.finalize(mp, argv, pkg2, o5, key)
    ok(v == "PASS" and pth.name.startswith("PASS_"), t["reasons"])
    good, why, _ = GT.verify_token(pth, argv, pkg2, key=key, data_flags=mp["data_flags"])
    ok(good, why)
    good, why, _ = GT.verify_token(pth, C._set(argv, "--lr", "1e-4"), pkg2, key=key, data_flags=mp["data_flags"])
    ok(not good and any("argv differs" in r for r in why))
    good, why, _ = GT.verify_token(pth, argv, pkg2, key=key, data_flags=mp["data_flags"][:0] or mp["data_flags"])
    ok(good)
    shutil.copyfile(pkg2 / "refe" / "gate_checks.py", root / "gc.bak")
    (pkg2 / "refe" / "gate_checks.py").write_text((pkg2 / "refe" / "gate_checks.py").read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    ok(not GT.verify_token(pth, argv, pkg2, key=key, data_flags=mp["data_flags"])[0])            # the gate's own code is in the bound tree
    (pkg2 / "refe" / "gate_checks.py").write_bytes((root / "gc.bak").read_bytes())
    Path(fx["pdm"], "pdm_synth.jsonl").write_text(Path(fx["pdm"], "pdm_synth.jsonl").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    ok(not GT.verify_token(pth, argv, pkg2, key=key, data_flags=mp["data_flags"])[0])            # a data directory changed after the gate
    shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    if "--fp-identity-only" in sys.argv:               # seconds: only the faster-fingerprint identity block -- run it after ANY change to gate_token's fingerprint / sha256_file / data_manifest
        fingerprint_identity_static()
        print(f"ZZLG_FP_IDENTITY_PASS {N}")
        return 0
    static()
    if "--static" in sys.argv:
        print(f"ZZLG_STATIC_PASS {N}")
        return 0
    full()
    print(f"ZZLG_FULL_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
