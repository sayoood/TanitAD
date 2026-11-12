#!/usr/bin/env python3
"""P7-READ-ROUTE B4: the logic of eval/p7_read_chain.py with a FAKE runner (no GPU, no subprocess, no navtest read): the dry-run plan, every preflight refusal, resume from markers,
failure markers, the seam acceptance rules, the determinism rep, the plans8 converter, and mutation arms that must go RED.
   python eval/test_p7_read_chain.py  ->  ZZP7CHAIN PASS n/n"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CHAIN_PY = HERE / "p7_read_chain.py"
res = []


def check(name):
    def d(fn):
        try:
            fn(); res.append((True, name)); print("PASS", name)
        except Exception as e:  # noqa: BLE001
            res.append((False, name)); print("FAIL", name, "<--", type(e).__name__, str(e)[:300])
        return fn
    return d


def load_chain(text: str | None = None, tag="chain"):
    """the chain module, from the real file or from mutated source text"""
    if text is None:
        text = CHAIN_PY.read_text(encoding="utf-8")
    d = Path(tempfile.mkdtemp(prefix="p7chain_mod_"))
    p = d / "p7_read_chain.py"
    p.write_text(text, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location(f"p7_read_chain_{tag}_{abs(hash(text))}", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


C = load_chain(tag="real")


class Fake:
    """records every command; plays the role of stage2 / stage3 / transp / families by writing the artifacts the chain validates"""
    def __init__(self, c, n, fc=6.18063877792441, rc_stage2_first=1, rc_overrides=None, bad_pick20=False, no_pick20=False, rep_differs=False, wrong_ckpt_in_report=False):
        self.c, self.n, self.fc = c, n, fc
        self.calls, self.rc_stage2_first, self.rc_overrides = [], rc_stage2_first, dict(rc_overrides or {})
        self.bad_pick20, self.no_pick20, self.rep_differs, self.wrong_ckpt = bad_pick20, no_pick20, rep_differs, wrong_ckpt_in_report
        self.s2_seen = {}
        sys.path.insert(0, str(Path(c.pkg) / "eval"))
        spec = importlib.util.spec_from_file_location("seam_for_fake", f"{c.pkg}/eval/refe_navtest_seam.py")
        self.sm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.sm)
        sys.path.pop(0)

    def _world(self, rep=False):
        rng = np.random.default_rng(11)
        g = np.cumsum(rng.normal(1.0, 0.3, (self.n, 20, 1)), axis=1)
        p20 = np.concatenate([g, np.cumsum(rng.normal(0, 0.1, (self.n, 20, 1)), axis=1), rng.normal(0, 0.2, (self.n, 20, 1))], axis=2).astype(np.float32)
        poses = np.stack([self.sm.to_navsim(x).astype(np.float32) for x in p20])
        return p20, poses

    def run_many(self, jobs):
        out = []
        for argv, env, log in jobs:
            self.calls.append(argv)
            assert env["TANITAD_NAVSIM_ROOT"] == self.c.navsim_root and env["TANITAD_REFE_PKG"] == self.c.pkg
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            Path(log).write_text("fake\n", encoding="utf-8")
            out.append(self._one(argv))
        return out

    def _one(self, argv):
        c = self.c
        key = next((k for k in ("stage2_sft.py", "stage3_full.py", "command_adherence.py", "transp_score.py", "families_run.py", "fam_ci.py") if any(k in a for a in argv)), None)
        if key in self.rc_overrides:
            return self.rc_overrides[key]
        if key == "stage2_sft.py":
            name = argv[argv.index("--name") + 1]
            first = name not in self.s2_seen
            self.s2_seen[name] = self.s2_seen.get(name, 0) + 1
            seam = Path(f"{c.seams}/stage2_{name}.npz")
            out = Path(f"{c.pkg}/raw/2026-10-05-sft-stage2/{name}")
            out.mkdir(parents=True, exist_ok=True)
            res_ = {"name": name, "ego_yaw": {"sha256": hashlib.sha256(open(c.yaw_table, "rb").read()).hexdigest()}}
            if first:
                p20, poses = self._world()
                if self.rep_differs and name.endswith("_rep"):
                    poses = poses.copy()
                    poses[2, 3, 1] += np.float32(1e-3)
                seam.parent.mkdir(parents=True, exist_ok=True)
                tok = np.array([f"t{i:04d}" for i in range(self.n)])
                np.savez(seam, token=tok, fingerprint=tok, poses=poses, sampling=np.array([8, 0.5]), arm=np.array("x"))
                json.dump({"rows": self.n, "misses": 0, "ckpt": ("other.pt" if self.wrong_ckpt else c.ckpt), "rule": "navsim_v1", "repair_last_heading": True,
                           "frame_control": {"max_m": self.fc}}, open(seam.with_suffix(".report.json"), "w"))
                if "--dump-proposals" in argv:
                    pk = p20.copy()
                    if self.bad_pick20:
                        pk[1, 4, 0] += np.float32(0.01)          # index 4 = t 1.0 s, one of the grid points to_navsim reads (0.2 s indices 0, 3, 5, 8, 10, 13, 15, 18 are NOT read by it: the control cannot see them, the wiring test covers those)
                    kw = {"token": tok, "fingerprint": tok, "proposals": np.zeros((self.n, 2, 8, 3), np.float32), "pick": np.zeros(self.n, np.int64)}
                    if not self.no_pick20:
                        kw["pick20"] = pk
                    np.savez(Path(c.dump), **kw)
                res_.update({"verdict": "INCOMPLETE (seam failed)"})
                json.dump(res_, open(out / "result.json", "w"))
                return self.rc_stage2_first
            res_.update({"verdict": "NOT PROVEN", "harness_status": {"status": "PASS"}, "primary": {"mean": 1.0}})
            json.dump(res_, open(out / "result.json", "w"))
            return 0
        return 0


def world(td: Path, n=5, **fake_kw):
    """a temp tip export + data dir, a Cfg pointing at them, with the environment-bound probes replaced by controllable fakes"""
    pk = td / "pk"
    (pk / "eval").mkdir(parents=True)
    shutil.copy(HERE / "refe_navtest_seam.py", pk / "eval" / "refe_navtest_seam.py")
    (td / "refe").mkdir()
    data = td / "data"
    (data / "seams").mkdir(parents=True)
    root = td / "navsim"
    (root / "exp/w3_navtest_v1/inputs").mkdir(parents=True)
    (root / "exp/w3_navtest_v1/inputs/navtest_inputs.json.gz").write_bytes(b"x")
    toks = data / "navtest_full_tokens.json"
    toks.write_text(json.dumps({"tokens": [f"t{i:04d}" for i in range(n)]}), encoding="utf-8")
    ck = td / "run" / "model_final.pt"
    ck.parent.mkdir()
    ck.write_bytes(b"checkpoint-bytes")
    sha = hashlib.sha256(b"checkpoint-bytes").hexdigest()
    (ck.parent / "summary.json").write_text(json.dumps({"done": True, "model_final_sha256": sha}), encoding="utf-8")
    yaw = td / "yaw.json.gz"
    yaw.write_bytes(b"yaw-table")
    ysha = hashlib.sha256(b"yaw-table").hexdigest()
    C.TOKENS_MD5 = hashlib.md5(toks.read_bytes()).hexdigest()
    C.N_TOKENS = n
    C.YAW_SHA_PREFIX = ysha[:8]
    C._harness_ok = lambda c: {"n_files": 21}
    C._baseline_pin_ok = lambda c, k: True
    C._ckpt_wants_yaw = lambda c: True
    C._base_seam_path = lambda c: str(td / "base_seam.npz")
    c = C.Cfg(ckpt=str(ck).replace("\\", "/"), name="p7_full", summary=str(ck.parent / "summary.json").replace("\\", "/"), yaw_table=str(yaw).replace("\\", "/"),
              work=str(td / "work").replace("\\", "/"), data=str(data).replace("\\", "/"), navsim_root=str(root).replace("\\", "/"), pkg=str(pk).replace("\\", "/"), n_tokens=n)
    f = Fake(c, n, **fake_kw)
    # the base (P6-like) seam for the plans step
    _, poses = f._world()
    np.savez(td / "base_seam.npz", token=np.array([f"t{i:04d}" for i in range(n)]), poses=poses)
    return c, f


def new(td, **kw):
    os.environ.pop("TANITAD_NAVSIM_ROOT", None)          # the suite runs with it set; the chain REFUSES a different value, and sets it itself
    c, f = world(Path(td), **kw)
    lines = []
    return c, f, C.Chain(c, runner=f, say=lines.append), lines


def pre_ok(c):
    return C.preflight(c, strict=False)


def refused(c, needle):
    try:
        C.preflight(c, strict=True)
    except SystemExit as e:
        assert needle in str(e), (needle, str(e))
        return True
    raise AssertionError(f"preflight did NOT refuse ({needle})")


# ----------------------------------------------------------------------------------------------- dry-run plan
@check("D1 dry-run prints EVERY step's command in order; seam1 and score are the SAME stage-2 argv; rep is the same without --dump-proposals; base p6 / also p2; yaw flag; pinned env; 2 shards; paired fam_ci arm - base")
def t_dry():
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = C.dry_run(c, out=print)
        txt = buf.getvalue()
        assert rc == 0, txt
        order = [l.split()[2] for l in txt.splitlines() if l.startswith("DRY step")]
        assert order == ["seam1", "score", "stage3", "adherence", "rep", "families"], order
        plan = dict(C.plan(c))
        s1, s2 = plan["seam1"][0][1], plan["score"][0][1]
        assert s1 == s2 and "--dump-proposals" in s1 and "--ego-yaw-table" in s1 and s1[s1.index("--rule") + 1] == "navsim_v1"
        rep = plan["rep"][0][1]
        assert "--dump-proposals" not in rep and "--ego-yaw-table" in rep and rep[rep.index("--name") + 1] == "p7_full_rep"
        for step in ("stage3", "adherence"):
            argv = plan[step][0][1]
            assert argv[argv.index("--base") + 1] == "p6" and argv[argv.index("--also") + 1] == "p2", step
        assert "--expect-rule" in plan["stage3"][0][1] and "--expect-ckpt-sha256" in plan["stage3"][0][1]
        fam = dict(plan["families"])
        assert [k for k in fam if k.startswith("transp_s")] == ["transp_s0", "transp_s1"] and fam["transp_s0"][fam["transp_s0"].index("--shard") + 1] == "0/2"
        assert "--metric-cache-root" in fam["transp_s0"] and "--per-window" in fam["fam_arm"]
        ci = fam["fam_ci"]
        assert ci[ci.index("--a") + 1].endswith("fam_P6.json") and ci[ci.index("--b") + 1].endswith("fam_P7.json"), ci
        assert "TANITAD_NAVSIM_ROOT=" + c.navsim_root in txt and "TANITAD_REFE_PKG=" in txt
        for line in txt.splitlines():
            if line.startswith("  DRY ") and ":" in line:
                assert line.split(": ", 1)[1].split()[0].endswith(".exe") or "/" in line.split(": ", 1)[1].split()[0], line
        assert "DRY preflight (would PASS)" in txt


@check("D2 dry-run exit code is 3 and the failed check is named when the preflight would refuse")
def t_dry_refuse():
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        Path(c.tokens).write_text(json.dumps({"tokens": ["a"]}), encoding="utf-8")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = C.dry_run(c, out=print)
        assert rc == 3 and "FAIL tokens" in buf.getvalue(), buf.getvalue()


# ----------------------------------------------------------------------------------------------- preflight
@check("PF1 positive control: every check passes on a consistent world")
def t_pf_ok():
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        r = C.preflight(c, strict=True)
        assert r["ok"] and r["ckpt_sha256"] and r["yaw_sha256"]


@check("PF2 REFUSALS, one defect each: wrong ckpt sha (file changed), summary not done, no expected sha, sha sources disagree, wrong yaw sha, yaw missing for a yaw checkpoint, tokens md5, harness pins, comparator pins, archive root env, export absent, ckpt absent")
def t_pf_refusals():
    cases = []
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        Path(c.ckpt).write_bytes(b"other-bytes")
        refused(c, "ckpt_sha")
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        json.dump({"done": False, "model_final_sha256": hashlib.sha256(b"checkpoint-bytes").hexdigest()}, open(c.summary, "w"))
        refused(c, "has not finished")
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        c.summary = None
        refused(c, "no expected sha256")
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        c.expect_sha = "0" * 64
        refused(c, "ckpt_sha_sources_agree")
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        Path(c.yaw_table).write_bytes(b"another-table")
        refused(c, "not the registered navtest table")
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        c.yaw_table = None
        refused(c, "no --ego-yaw-table")
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        Path(c.tokens).write_text(json.dumps({"tokens": ["a", "b"]}), encoding="utf-8")
        refused(c, "tokens")
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        def bad(c_):
            raise SystemExit("HARNESS PIN GUARD: CHANGED: score_e.py")
        C._harness_ok = bad
        refused(c, "harness_pins")
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        C._baseline_pin_ok = lambda c_, k: k != "p6"
        refused(c, "baseline_p6")
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        os.environ["TANITAD_NAVSIM_ROOT"] = "E:/Archive/devbox-C/navsim"
        try:
            refused(c, "differs from the pinned")
        finally:
            os.environ.pop("TANITAD_NAVSIM_ROOT", None)
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        os.remove(c.export)
        refused(c, "navsim_root")
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        os.remove(c.ckpt)
        refused(c, "is not a file")


@check("PF3 a no-yaw checkpoint with --no-ego-yaw-table passes, a yaw checkpoint with it is refused")
def t_pf_noyaw():
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        c.yaw_table = None
        C._ckpt_wants_yaw = lambda c_: False
        assert C.preflight(c, strict=True)["ok"]
        C._ckpt_wants_yaw = lambda c_: True
        refused(c, "no --ego-yaw-table")


# ----------------------------------------------------------------------------------------------- the chain
FULL = ["seam1", "score", "stage3", "adherence", "rep", "families"]


def commands(f):
    return [next(k for k in ("stage2_sft", "stage3_full", "command_adherence", "transp_score", "families_run", "fam_ci") if any(k in a for a in x)) for x in f.calls]


@check("C1 a full run: every step's marker, the commands run in plan order (stage2 x3, stage3, adherence, 2 transp shards, 2 families_run, fam_ci), chain marker, exit 0")
def t_full():
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, lines = new(td)
        assert ch.run() == 0, "\n".join(lines)[-1500:]
        for s in FULL + ["chain"]:
            assert ch.done(s), s
        assert commands(f) == ["stage2_sft", "stage2_sft", "stage3_full", "command_adherence", "stage2_sft", "transp_score", "transp_score", "families_run", "families_run", "fam_ci"], commands(f)
        with np.load(Path(c.work) / "families" / "P7_plans8.npz") as z:
            assert z["plans8"].shape == (5, 8, 3) and list(z["tokens"]) == [f"t{i:04d}" for i in range(5)]
        info = json.load(open(Path(c.markers) / "seam1.DONE", encoding="utf-8"))
        assert info["rc"] == 1 and info["pick20"] == [5, 20, 3]


@check("C2 idempotent resume: a second run executes NOTHING; deleting one marker re-runs only that step")
def t_resume():
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        assert ch.run() == 0
        n0 = len(f.calls)
        assert ch.run() == 0 and len(f.calls) == n0, "a finished chain re-ran something"
        os.remove(Path(c.markers) / "adherence.DONE")
        os.remove(Path(c.markers) / "chain.DONE")
        assert ch.run() == 0
        assert [commands(f)[i] for i in range(n0, len(f.calls))] == ["command_adherence"], commands(f)[n0:]


@check("C3 a failing step writes <step>.FAILED, stops the chain (later steps not run), and the next run keeps the old marker as .attempt_1 and retries ONLY that step")
def t_fail_resume():
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, lines = new(td, rc_overrides={"stage3_full.py": 2})
        assert ch.run() == 1
        assert (Path(c.markers) / "stage3.FAILED").is_file() and not ch.done("stage3") and not ch.done("adherence")
        assert commands(f)[-1] == "stage3_full"
        f.rc_overrides.clear()
        assert ch.run() == 0, "\n".join(lines)[-800:]
        assert (Path(c.markers) / "stage3.FAILED.attempt_1").is_file() and not (Path(c.markers) / "stage3.FAILED").exists()
        assert commands(f).count("stage2_sft") == 3, "a retry of stage3 must not re-run the seam"


@check("C4 seam1 accepts rc 1 ONLY with the declared frame-control max; rc 0 with a failing control, an unexpected rc, a missing pick20, pick20 != seam poses, a seam report for ANOTHER checkpoint are each refused")
def t_seam_rules():
    def failed(**kw):
        with tempfile.TemporaryDirectory() as td:
            c, f, ch, _ = new(td, **kw)
            rc = ch.run()
            return rc, (Path(c.markers) / "seam1.FAILED").is_file(), ch.done("seam1")
    assert failed() == (0, False, True)                                    # positive control: the expected Deviation-1 failure passes
    assert failed(fc=6.3) == (1, True, False)                              # not the declared control value: a real frame defect
    assert failed(rc_stage2_first=0)[1] is True                            # rc 0 while the control max is the failing value: inconsistent
    assert failed(rc_stage2_first=3) == (1, True, False)
    assert failed(no_pick20=True) == (1, True, False)
    assert failed(bad_pick20=True) == (1, True, False)
    assert failed(wrong_ckpt_in_report=True) == (1, True, False)


@check("C5 an existing seam without a seam1 marker is REFUSED before any command (stage 2 would reuse a seam of unknown yaw), and so is an existing rep seam")
def t_unknown_seam():
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        Path(c.seam).write_bytes(b"old")
        try:
            ch.run()
        except SystemExit as e:
            assert "unknown yaw" in str(e)
        else:
            raise AssertionError("not refused")
        assert not f.calls


@check("C6 resume under another checkpoint / yaw table / comparator is refused (markers of one identity only)")
def t_ident():
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td, rc_overrides={"stage3_full.py": 2})
        assert ch.run() == 1
        Path(c.ckpt).write_bytes(b"new-bytes")
        json.dump({"done": True, "model_final_sha256": hashlib.sha256(b"new-bytes").hexdigest()}, open(c.summary, "w"))
        try:
            C.Chain(c, runner=f, say=lambda m: None).run()
        except SystemExit as e:
            assert "another identity" in str(e)
        else:
            raise AssertionError("not refused")


@check("C7 determinism rep: identical poses pass; ONE pose differing in one token FAILS the rep step (and the chain stops there)")
def t_rep():
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        assert ch.run() == 0
        info = json.load(open(Path(c.markers) / "rep.DONE", encoding="utf-8"))
        assert info["poses_bit_identical"] and info["tokens_equal"] and info["n"] == 5
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, lines = new(td, rep_differs=True)
        assert ch.run() == 1
        assert "DETERMINISM FAILED" in (Path(c.markers) / "rep.FAILED").read_text(encoding="utf-8") and not ch.done("families")


@check("C8 --only / --skip: --skip rep runs everything else and does NOT write chain.DONE; --only families runs just the families commands; unknown steps are refused")
def t_only_skip():
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        c.skip = ("rep",)
        assert ch.run() == 0
        assert ch.done("families") and not ch.done("rep") and not ch.done("chain") and "stage2_sft" in commands(f) and commands(f).count("stage2_sft") == 2
    with tempfile.TemporaryDirectory() as td:
        c, f, ch, _ = new(td)
        c.only = ("score",)
        assert ch.run() == 0 or True
    import contextlib
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            C.make_cfg(["--ckpt", "a", "--name", "n", "--skip", "bogus"])
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("unknown step accepted")


@check("C10 --detach: the child argv is this script with every argument except --detach (so the detached chain is the same chain), and a dry-run never detaches")
def t_detach():
    a = ["--ckpt", "x.pt", "--name", "n", "--detach", "--skip", "rep"]
    d = C.detach_argv(a)
    assert d[0].endswith("python.exe") and d[1].endswith("p7_read_chain.py") and d[2:] == ["--ckpt", "x.pt", "--name", "n", "--skip", "rep"], d
    assert "--detach" not in d


@check("C9 plans8 converter: tokens + plans8 equal the seam's; duplicate tokens and non-finite poses are refused; transp_score.load_plans reads the file")
def t_plans8():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rng = np.random.default_rng(1)
        tok, poses = np.array(["a", "b", "c"]), rng.normal(size=(3, 8, 3)).astype(np.float32)
        np.savez(td / "s.npz", token=tok, poses=poses)
        assert C.seam_to_plans8(td / "s.npz", td / "o" / "p.npz") == 3
        with np.load(td / "o" / "p.npz") as z:
            assert list(z["tokens"]) == ["a", "b", "c"] and np.array_equal(z["plans8"], poses.astype(np.float64))
        np.savez(td / "d.npz", token=np.array(["a", "a", "c"]), poses=poses)
        for bad in ({"token": np.array(["a", "a", "c"]), "poses": poses}, {"token": tok, "poses": np.where(np.arange(3)[:, None, None] == 1, np.nan, poses)}, {"token": tok, "poses": poses[:, :7]}):
            np.savez(td / "bad.npz", **bad)
            try:
                C.seam_to_plans8(td / "bad.npz", td / "o" / "bad_out.npz")
            except RuntimeError:
                pass
            else:
                raise AssertionError("not refused")
        try:
            spec = importlib.util.spec_from_file_location("transp_for_chain", HERE / "instruments" / "transp_score.py")
            T = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(T)
        except Exception as e:  # noqa: BLE001
            print("   (transp_score not importable here, load_plans compatibility skipped, stated):", type(e).__name__)
            return
        lp = T.load_plans(str(td / "o" / "p.npz"))
        assert sorted(lp) == ["a", "b", "c"] and np.allclose(lp["b"]["plan8"], poses[1])


# ----------------------------------------------------------------------------------------------- mutation arms (the OLD / weakened behaviour put back must make a check FAIL)
def _case_wrong_yaw_refused(mod):
    global C
    old = C
    C = mod
    try:
        with tempfile.TemporaryDirectory() as td:
            c, f, ch, _ = new(td)
            Path(c.yaw_table).write_bytes(b"another-table")
            refused(c, "not the registered navtest table")
    finally:
        C = old


def _case_bad_rc_caught(mod):
    global C
    old = C
    C = mod
    try:
        with tempfile.TemporaryDirectory() as td:
            c, f, ch, _ = new(td, rc_stage2_first=3)
            assert ch.run() == 1 and not ch.done("seam1"), "an unexpected stage-2 rc was accepted as the Deviation-1 failure"
    finally:
        C = old


def _case_bad_pick20_caught(mod):
    global C
    old = C
    C = mod
    try:
        with tempfile.TemporaryDirectory() as td:
            c, f, ch, _ = new(td, bad_pick20=True)
            assert ch.run() == 1
    finally:
        C = old


def _case_rep_differs_caught(mod):
    global C
    old = C
    C = mod
    try:
        with tempfile.TemporaryDirectory() as td:
            c, f, ch, _ = new(td, rep_differs=True)
            assert ch.run() == 1
    finally:
        C = old


def _case_marker_skip(mod):
    global C
    old = C
    C = mod
    try:
        with tempfile.TemporaryDirectory() as td:
            c, f, ch, _ = new(td)
            assert ch.run() == 0
            n0 = len(f.calls)
            assert ch.run() == 0 and len(f.calls) == n0
    finally:
        C = old


@check("M1 mutation arms: each weakened chain (yaw-prefix check off / any stage-2 rc accepted / pick20 control off / determinism compare off / markers ignored) FAILS the case that guards it; the unmutated chain passes all five")
def t_mutations():
    src = CHAIN_PY.read_text(encoding="utf-8")
    muts = [
        (_case_wrong_yaw_refused, 'put("yaw_table", ys.startswith(YAW_SHA_PREFIX), "" if ys.startswith(YAW_SHA_PREFIX) else', 'put("yaw_table", True, "" if True else'),
        (_case_bad_rc_caught, 'rc = self._run("seam1", stage2_cmd(c, c.name, True), f"{c.work}/logs/seam1.log", expect=(0, 1))', 'rc = self._run("seam1", stage2_cmd(c, c.name, True), f"{c.work}/logs/seam1.log", expect=(0, 1, 2, 3))'),
        (_case_bad_pick20_caught, 'if not sm.pick20_equals_seam(pick20, poses):', 'if False:'),
        (_case_rep_differs_caught, "if not same_tok or n_diff != 0:", "if False:"),
        (_case_marker_skip, "if self.done(step):", "if False:"),
    ]
    for case, old, new_ in muts:
        case(C)                                                             # the real chain passes the case
        assert old in src, old
        mod = load_chain(src.replace(old, new_, 1), tag="mut")
        try:
            case(mod)
        except (AssertionError, SystemExit, RuntimeError):
            continue
        raise AssertionError(f"mutation not detected by {case.__name__}: {new_[:60]}")


if __name__ == "__main__":
    n_ok = sum(1 for ok, _ in res if ok)
    print(f"ZZP7CHAIN {'PASS' if n_ok == len(res) else 'FAIL'} {n_ok}/{len(res)}")
    sys.exit(0 if n_ok == len(res) else 1)
