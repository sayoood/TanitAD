#!/usr/bin/env python3
"""Tests for the REFe launch gate (gate_token.py, gate_checks.py, launch_gate.py, gate_fixture.py).  Contract: eval/REFE_LAUNCH_GATE.md.

  --static   (seconds, no trainer run)  the token, the profiles, the refusals, the registry, the cheap checks and their regression arms.
             Prints `ZZLG_STATIC_PASS <n>`.  This is what the gate's own G-SUITE runs (with --no-banked: the pod has no banked copy).
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


P6_ARGS = ("--backbone vitl16 --targets /w/train_grow --images /w/pix --calib /w/calib.json --metric-cache /w/mc --scorer-sees-trunk --yaw-wrap off --nav-mode command "
           "--labeller online --epochs 25 --batch 4 --accum 64 --lr 2e-4 --weight-decay 0.01 --amp bf16 --score-w 0.1 --tf32 --compile --workers 2 "
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
    ok(len(dests) == 55)
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
    if shutil.which("bash"):
        r = subprocess.run(["bash", "-c", 'ARGS="$1"; printf "%s\\n" $ARGS', "_", real_args()], capture_output=True, text=True, encoding="utf-8")
        ok(r.stdout.split("\n")[:-1] == GT.argv_from_string(real_args()))
    else:
        print("ZZTEST_SKIP bash not found: $ARGS word splitting not cross-checked")
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
    static()
    if "--static" in sys.argv:
        print(f"ZZLG_STATIC_PASS {N}")
        return 0
    full()
    print(f"ZZLG_FULL_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
