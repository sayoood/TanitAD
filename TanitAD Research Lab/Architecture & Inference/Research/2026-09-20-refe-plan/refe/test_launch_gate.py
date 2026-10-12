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
    rls1_static()


# ====================================================================================================== RL-S1: the refe_rls1 profile
# RL-S1 trains with scorer_finetune.py (the EPO loss), not train.py: the profile names that parser and source, classifies every one of ITS flags (DVB_KINDS_SCORER, a table
# SEPARATE from DVB_KINDS -- S5 pins that one equal to train.py's 55), states the arm rules, and consumes the trainer's own `--preflight` (G1-G11) as parsed JSON.
# Every refusal below has its positive twin (the clean argv) and every check its regression arm.  Nothing here runs the suite (that would recurse into this file).
RLS1_BASE = ("--ckpt /w/model_sft4_B_u400.pt --backbone vitl16 --targets /w/train --images /w/pix --calib /w/calib.json --onpolicy /w/onp --heldout /w/onh "
             "--heldout-logs /w/hl.txt --out /w/rl --lane-head --teacher-lane-train /w/tl --teacher-lane-heldout /w/tlh --pdm-labels-train /w/pdm "
             "--pdm-labels-heldout /w/pdmh --require-pdm --b-mode listnet --lam 0 --lr 1e-4 --max-updates 1000 --epochs 2 --workers 3 --seed 0")


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
    for flag, bad in (("--lr", "3e-5"), ("--max-updates", "500"), ("--batch", "4"), ("--accum", "4"), ("--warmup", "10")):
        red(C._set(r0, flag, bad), "requires " + flag, "R0")
        red(C._set(c0, flag, bad), "requires " + flag, "C0")
    red(without(r0, "--max-updates"), "requires --max-updates", "R0")                                        # absent = the parser default 0, which is not 1000
    red(without(r0, "--lr"), "requires --lr", "R0")
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
    ok(ev["status"] == "PASS" and ev["details"]["observed_rows"] == 13 and ev["details"]["unknown_flags"] == [] and ev["details"]["stale_registry"] == [], ev["reasons"])
    ok(sum(ev["details"]["kinds"].values()) == len(C.DVB_KINDS_SCORER) == len(sdests))
    ok(C.check_dvb_scorer(c0, C.load_scorer_preflight(rls1_write(Path(tempfile.mkdtemp()) / "p.jsonl", rls1_pf_rows(c0))), P)["status"] == "PASS")
    for arm, frag in (("unknown_flag", "NO declared-vs-built entry"), ("stale_registry", "no longer has"), ("dead_lever", "never read by the trainer"),
                      ("unbound_data", "would NOT fingerprint"), ("lever_ignored", "declared 'listnet' != built 'compw'")):
        ev = C.check_dvb_scorer(r0, pf_ok, P0, arm=arm)
        ok(ev["status"] == "FAIL" and frag in " ".join(ev["reasons"]) and ev["arm"] == arm, (arm, ev["reasons"]))
    ok(C.check_dvb_scorer(r0, pf_ok, P0)["status"] == "PASS")                                                # the arms left no residue

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
        if k.startswith("G8_"):                                    # r0 carries no lane labels: G8 is not obliged, so a G8 flag in the fixture is simply not written
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
    ev = cons(r0, add={"G12_something_new": {"ok": True}})
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
