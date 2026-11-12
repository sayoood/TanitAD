#!/usr/bin/env python3
"""P7 navtest read chain (P7-READ-ROUTE B4, 2026-11-12): ONE detached, resumable chain from a finished checkpoint to the P7 read, run from a tip export.

Generalised from P6's single-copy chain (F:/TanitAD_p6read_work/chain_p6.py, which was bound to P6's sha and paths and lived outside the repo).
`--ckpt / --name / --base / --also` make it reusable for the P5-C arms. Everything below the preflight is a command run through ONE runner, so
`--dry-run` prints exactly what a real run would execute.

STEPS (each has a marker <work>/markers/<step>.DONE; a re-run skips finished steps, so the chain is idempotent):
  preflight  ckpt sha256 == summary.json's model_final_sha256 (or --expect-ckpt-sha256), summary done, yaw table sha256 prefix f5ebe24f, tokens md5 3a8b3653 / n 12146,
             eval/harness_pins.py --verify (HP.require), the comparators' pins (stage3_full BASELINES), $TANITAD_NAVSIM_ROOT pinned and present.  ALWAYS runs.
  seam1      eval/stage2_sft.py --dump-proposals --ego-yaw-table: ATTEMPT 1 of RESULT_SFT4.md Deviation 1. It ends `INCOMPLETE (seam failed)` (rc 1) at the 36-token frame
             control, EXPECTED (the control's max is the declared C1 set, model independent); the chain accepts rc 1 ONLY if the seam report reads rows 12146, misses 0 and the
             frame-control max equals the declared 6.18063877792441 m, and the dump carries pick20 equal to the seam's poses bit for bit.
  score      the SAME stage-2 command again: the seam exists, so it only scores (harness) and writes result.json + the stage-2 families. rc 0 required.
  stage3     eval/stage3_full.py --base <base> --also <also> --expect-rule navsim_v1 (paired log-cluster bootstrap, adverse families). rc 0 (gates ok) required.
  adherence  eval/command_adherence.py read --base <base> --also <also> (paired tactical rows vs the pinned comparators).
  rep        determinism: the same seam again as <name>_rep WITHOUT the dump; rc 1 (frame control) is expected again; tokens equal and poses bit-identical is required.
  families   Q3 route: seam npz -> plans8 (arm and base) -> transp_score.py (2 shards, ONE pass for both arms, harness MetricCache) -> families_run.py --per-window
             (both arms) -> fam_ci.py paired (arm - base): headway / time-gap / TTC to the lead and the paired family CIs.
  chain.DONE written when every selected step is done.

  python eval/p7_read_chain.py --ckpt <model_final.pt> --name p7_full [--summary <summary.json>] [--base p6] [--also p2] [--work <dir>] [--dry-run] [--only a,b] [--skip rep]
A step that fails writes <step>.FAILED (the traceback) and stops the chain; the next run renames it to <step>.FAILED.attempt_N and retries that step only.
Exit codes: 0 done, 1 a step failed, 2 a preflight refusal, 3 dry-run whose preflight would refuse.
CPU parts never touch the GPU; the GPU lock is taken INSIDE stage2_sft.py (take_lock/release), never here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = Path(os.environ.get("TANITAD_REFE_PKG") or HERE.parent)       # a tip export runs its OWN code

TOKENS_MD5 = "3a8b3653fd70891e50a230a37eb76ae8"                     # stage3_full.FULL_PINS: the 12,146 navtest tokens
N_TOKENS = 12146
YAW_SHA_PREFIX = "f5ebe24f"                                         # SPEC_P7 Amendment YAW; stage2_sft.NAVTEST_YAW_SHA_PREFIX
FRAME_CONTROL_DECLARED_M = 6.18063877792441                         # the C1 36-token frame-control max (MEASURED in P6's refe_p6final.report.json; identical on the deployed seam)
FRAME_CONTROL_BAR_M = 0.001
NAVSIM_ROOT_DEFAULT = "F:/Archive/devbox-C/navsim"
DATA_DEFAULT = "E:/Projects/TanitAD/data/refe_navtest"
YAW_DEFAULT = "E:/Projects/TanitAD-artifacts/yawrate_wp/tables/ego_yaw_navtest.json.gz"
WORK_ROOT_DEFAULT = "E:/Projects/TanitAD-artifacts"
PY_T = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PY_NAV = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"       # transp_score / families_run need navsim + nuplan
STEPS = ("preflight", "seam1", "score", "stage3", "adherence", "rep", "families")
FAM_SUBSTEPS = ("plans", "transp_s0", "transp_s1", "fam_arm", "fam_base", "fam_ci")


class Refusal(SystemExit):
    """a preflight refusal: nothing was started"""


@dataclass
class Cfg:
    ckpt: str
    name: str
    base: str = "p6"
    also: str = "p2"
    summary: str | None = None
    expect_sha: str | None = None
    yaw_table: str | None = YAW_DEFAULT
    work: str = ""
    data: str = DATA_DEFAULT
    navsim_root: str = NAVSIM_ROOT_DEFAULT
    pkg: str = str(PKG).replace("\\", "/")
    py_t: str = PY_T
    py_nav: str = PY_NAV
    n_tokens: int = N_TOKENS
    n_shards: int = 2
    only: tuple = ()
    skip: tuple = ()
    steps_env: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.work:
            self.work = f"{WORK_ROOT_DEFAULT}/{self.name}_read"

    # ---- derived paths (forward slashes)
    @property
    def tokens(self): return f"{self.data}/navtest_full_tokens.json"
    @property
    def seams(self): return f"{self.data}/seams"
    @property
    def seam(self): return f"{self.seams}/stage2_{self.name}.npz"
    @property
    def dump(self): return f"{self.seams}/stage2_{self.name}.proposals.npz"
    @property
    def rep_name(self): return f"{self.name}_rep"
    @property
    def rep_seam(self): return f"{self.seams}/stage2_{self.rep_name}.npz"
    @property
    def s2_out(self): return f"{self.pkg}/raw/2026-10-05-sft-stage2/{self.name}"
    @property
    def export(self): return f"{self.navsim_root}/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
    @property
    def markers(self): return f"{self.work}/markers"
    @property
    def base_label(self): return self.base.upper()
    @property
    def arm_label(self): return self.name.upper().removesuffix("_FULL")        # p7_full -> P7


# ------------------------------------------------------------------ commands (pure functions of the config: this is what --dry-run prints)
def stage2_cmd(c: Cfg, name: str, dump: bool) -> list:
    cmd = [c.py_t, f"{c.pkg}/eval/stage2_sft.py", "--ckpt", c.ckpt, "--name", name, "--rule", "navsim_v1", "--tokens", c.tokens]
    if dump:
        cmd.append("--dump-proposals")
    if c.yaw_table:
        cmd += ["--ego-yaw-table", c.yaw_table]
    return cmd


def plan(c: Cfg) -> list:
    """[(step, [(label, argv)])] in order. Pure: touches nothing."""
    p = f"{c.pkg}/eval"
    wk = c.work
    fam = f"{wk}/families"
    rows = f"{fam}/rows_s*.jsonl"
    out = []
    out.append(("seam1", [("stage2 attempt 1 (seam + dump; rc 1 at the frame control is EXPECTED)", stage2_cmd(c, c.name, True))]))
    out.append(("score", [("stage2 attempt 2 (scores the existing seam)", stage2_cmd(c, c.name, True))]))
    out.append(("stage3", [("stage 3", [c.py_t, f"{p}/stage3_full.py", "--name", c.name, "--ckpt", c.ckpt, "--expect-ckpt-sha256", "<ckpt sha256>",
                                         "--base", c.base, "--also", c.also, "--expect-rule", "navsim_v1", "--out-dir", f"{wk}/stage3"])]))
    out.append(("adherence", [("command adherence", [c.py_t, f"{p}/command_adherence.py", "read", "--name", c.name, "--dump", c.dump,
                                                      "--base", c.base, "--also", c.also, "--out", f"{wk}/command_adherence_{c.name}.json"])]))
    out.append(("rep", [("determinism rep (seam only, no dump; rc 1 expected)", stage2_cmd(c, c.rep_name, False))]))
    plans_arm, plans_base = f"{fam}/{c.arm_label}_plans8.npz", f"{fam}/{c.base_label}_plans8.npz"
    tr = [c.py_nav, f"{p}/instruments/transp_score.py", "--refe", f"{c.pkg}/refe", "--plans", f"{c.arm_label}={plans_arm}", f"{c.base_label}={plans_base}",
          "--split", "navtest", "--token-log", c.export, "--metric-cache-root", f"{c.navsim_root}/exp/w3_navtest_v1/metric_cache_navtest"]
    repo = Path(c.pkg).parents[3].as_posix()
    fr = [c.py_nav, f"{p}/instruments/families_run.py", "--rows", rows, "--per-window", "--tanitad-repo", repo]
    out.append(("families", [
        ("plans", [c.py_t, Path(__file__).resolve().as_posix(), "--internal-plans8", c.seam, plans_arm]),       # run IN-PROCESS by step_families (same function)
        ("plans_base", [c.py_t, Path(__file__).resolve().as_posix(), "--internal-plans8", f"<stage3_full BASELINES[{c.base!r}] seam>", plans_base]),
        *[(f"transp_s{i}", [*tr, "--out", f"{fam}/rows_s{i}.jsonl", "--shard", f"{i}/{c.n_shards}"]) for i in range(c.n_shards)],
        ("fam_arm", [*fr, "--arm", c.arm_label, "--label", c.arm_label, "--out", f"{fam}/fam_{c.arm_label}.json"]),
        ("fam_base", [*fr, "--arm", c.base_label, "--label", c.base_label, "--out", f"{fam}/fam_{c.base_label}.json"]),
        ("fam_ci", [c.py_t, f"{c.pkg}/raw/2026-11-12-instruments/fam_ci/fam_ci.py", "--a", f"{fam}/fam_{c.base_label}.json", "--b", f"{fam}/fam_{c.arm_label}.json",
                    "--out", f"{wk}/fam_ci_{c.base_label}_vs_{c.arm_label}.json"]),
    ]))
    return out


def child_env(c: Cfg) -> dict:
    e = dict(os.environ)
    e.update({"TANITAD_NAVSIM_ROOT": c.navsim_root, "TANITAD_REFE_PKG": c.pkg, "PYTHONIOENCODING": "utf-8", "HF_HUB_DISABLE_XET": "1"})
    e["PYTHONPATH"] = os.pathsep.join([f"{c.navsim_root}/navsim-3e8291bfa89ff247231e0227778840cd0a036896", f"{c.pkg}/refe"])
    return e


# ------------------------------------------------------------------ helpers
def sha256_file(p, mb=16) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(mb << 20), b""):
            h.update(b)
    return h.hexdigest()


def md5_file(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load_mod(name: str, path: str):
    sys.path.insert(0, str(Path(path).parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m
    finally:
        sys.path.pop(0)


def _harness_ok(c: Cfg) -> dict:
    sys.path.insert(0, f"{c.pkg}/eval")
    try:
        import harness_pins as HP
        return HP.require(Path(c.pkg) / "eval" / "harness", caller=Path(c.pkg) / "eval" / "score_navtest_refe.py")
    finally:
        sys.path.pop(0)


def _baseline_pin_ok(c: Cfg, key: str) -> bool:
    sys.path.insert(0, f"{c.pkg}/eval")
    try:
        os.environ["TANITAD_NAVSIM_ROOT"] = c.navsim_root
        import stage3_full as S3
        return bool(S3.pick_baseline(key)["pin_ok"])
    finally:
        sys.path.pop(0)


def _base_seam_path(c: Cfg) -> str:
    sys.path.insert(0, f"{c.pkg}/eval")
    try:
        os.environ["TANITAD_NAVSIM_ROOT"] = c.navsim_root
        import stage3_full as S3
        return str(S3.pick_baseline(c.base)["seam"])
    finally:
        sys.path.pop(0)


def _ckpt_wants_yaw(c: Cfg) -> bool:
    sys.path.insert(0, f"{c.pkg}/refe")
    try:
        import ego_yaw as EY
        return bool(EY.wants_yaw(EY.checkpoint_meta(c.ckpt)))
    finally:
        sys.path.pop(0)


def preflight(c: Cfg, strict: bool = True) -> dict:
    """returns {"ok", "checks": {name: {"ok", "why"}}, "ckpt_sha256", "yaw_sha256"}; strict=True raises Refusal naming every failed check"""
    chk, info = {}, {}

    def put(name, ok, why=""):
        chk[name] = {"ok": bool(ok), "why": why}

    # 1. checkpoint identity
    want = c.expect_sha
    summ = None
    if c.summary:
        try:
            summ = json.load(open(c.summary, encoding="utf-8"))
        except (OSError, ValueError) as e:
            put("summary", False, f"{c.summary}: {type(e).__name__}")
        else:
            put("summary", summ.get("done") is True, "" if summ.get("done") is True else f"summary.json done={summ.get('done')!r}: the run has not finished")
            s_sha = summ.get("model_final_sha256")
            if want and s_sha and want != s_sha:
                put("ckpt_sha_sources_agree", False, f"--expect-ckpt-sha256 {want[:12]} != summary.json {s_sha[:12]}")
            want = want or s_sha
    if not want:
        put("ckpt_sha", False, "no expected sha256: give --summary (model_final_sha256) or --expect-ckpt-sha256")
    elif not os.path.isfile(c.ckpt):
        put("ckpt_sha", False, f"{c.ckpt} is not a file")
    else:
        got = sha256_file(c.ckpt)
        info["ckpt_sha256"] = got
        put("ckpt_sha", got == want, "" if got == want else f"sha256 {got[:12]} != expected {want[:12]}")
    # 2. yaw table
    wants_yaw = None
    if os.path.isfile(c.ckpt):
        try:
            wants_yaw = _ckpt_wants_yaw(c)
        except Exception as e:  # noqa: BLE001
            put("ckpt_meta", False, f"cannot read the checkpoint meta: {type(e).__name__}: {str(e)[:120]}")
    if c.yaw_table:
        if not os.path.isfile(c.yaw_table):
            put("yaw_table", False, f"{c.yaw_table} is not a file")
        else:
            ys = sha256_file(c.yaw_table)
            info["yaw_sha256"] = ys
            put("yaw_table", ys.startswith(YAW_SHA_PREFIX), "" if ys.startswith(YAW_SHA_PREFIX) else f"sha256 {ys[:12]} is not the registered navtest table {YAW_SHA_PREFIX}...")
    elif wants_yaw:
        put("yaw_table", False, "the checkpoint was trained with ego_yaw but no --ego-yaw-table is set (a yaw of 0.0 is a train/eval mismatch)")
    # 3. tokens
    if not os.path.isfile(c.tokens):
        put("tokens", False, f"{c.tokens} is not a file")
    else:
        m = md5_file(c.tokens)
        n = len(json.load(open(c.tokens, encoding="utf-8"))["tokens"]) if m == TOKENS_MD5 else -1
        put("tokens", m == TOKENS_MD5 and n == c.n_tokens, "" if (m == TOKENS_MD5 and n == c.n_tokens) else f"md5 {m[:8]} / n {n} != {TOKENS_MD5[:8]} / {c.n_tokens}")
    # 4. harness pins
    try:
        hp = _harness_ok(c)
        put("harness_pins", True, f"({hp['n_files']} files)")
    except SystemExit as e:
        put("harness_pins", False, str(e)[:300])
    # 5. comparators
    for key in dict.fromkeys([c.base, c.also]):
        try:
            okb = _baseline_pin_ok(c, key)
            put(f"baseline_{key}", okb, "" if okb else f"stage3_full BASELINES[{key!r}] pins do not hold on the files present")
        except SystemExit as e:
            put(f"baseline_{key}", False, str(e)[:200])
    # 6. archive root
    prev = os.environ.get("TANITAD_NAVSIM_ROOT")
    if prev and prev.replace("\\", "/").rstrip("/") != c.navsim_root.rstrip("/"):
        put("navsim_root", False, f"$TANITAD_NAVSIM_ROOT={prev} differs from the pinned {c.navsim_root}")
    else:
        put("navsim_root", os.path.isdir(c.navsim_root) and os.path.isfile(c.export), f"{c.export} not found" if not os.path.isfile(c.export) else "")
    ok = all(v["ok"] for v in chk.values())
    res = {"ok": ok, "checks": chk, **info, "ckpt": c.ckpt, "name": c.name, "ego_yaw_table": c.yaw_table}
    if strict and not ok:
        raise Refusal("REFUSING (nothing was started):\n" + "\n".join(f"  {k}: {v['why']}" for k, v in chk.items() if not v["ok"]))
    return res


def validate_seam(c: Cfg, name: str, expect_rc: int, need_dump: bool) -> dict:
    """the seam artifacts of stage2 run `name` are the ones this chain asked for; raises RuntimeError naming the failure"""
    import numpy as np
    seam = Path(f"{c.seams}/stage2_{name}.npz")
    rep_p = seam.with_suffix(".report.json")
    if not seam.is_file() or not rep_p.is_file():
        raise RuntimeError(f"seam artifacts absent: {seam} / {rep_p}")
    rep = json.load(open(rep_p, encoding="utf-8"))
    if rep.get("rows") != c.n_tokens or rep.get("misses") != 0:
        raise RuntimeError(f"seam report: rows {rep.get('rows')} (want {c.n_tokens}) misses {rep.get('misses')} (want 0)")
    if os.path.normcase(os.path.abspath(str(rep.get("ckpt")))) != os.path.normcase(os.path.abspath(c.ckpt)):
        raise RuntimeError(f"seam report names checkpoint {rep.get('ckpt')!r}, this chain is reading {c.ckpt!r}: a stale seam")
    if rep.get("rule") != "navsim_v1" or rep.get("repair_last_heading") is not True:
        raise RuntimeError(f"seam report: rule {rep.get('rule')!r} repair_last_heading {rep.get('repair_last_heading')!r} (want navsim_v1 / True)")
    fc = float(rep["frame_control"]["max_m"])
    if expect_rc == 1 and abs(fc - FRAME_CONTROL_DECLARED_M) > 1e-6:
        raise RuntimeError(f"frame control max {fc!r} m is not the declared C1 set's {FRAME_CONTROL_DECLARED_M!r}: this is NOT the expected Deviation-1 failure")
    if expect_rc == 0 and fc > FRAME_CONTROL_BAR_M:
        raise RuntimeError(f"stage 2 returned 0 but the frame control max {fc} m exceeds the bar")
    with np.load(seam) as z:
        tok, poses = z["token"], z["poses"]
    if len(tok) != c.n_tokens or poses.shape != (c.n_tokens, 8, 3) or not np.isfinite(poses).all():
        raise RuntimeError(f"seam npz: {len(tok)} tokens, poses {poses.shape}")
    out = {"seam": str(seam), "frame_control_max_m": fc}
    if need_dump:
        dp = Path(c.dump)
        if not dp.is_file():
            raise RuntimeError(f"proposal dump absent: {dp}")
        with np.load(dp) as d:
            if "pick20" not in d.files:
                raise RuntimeError("the dump has no `pick20` (an old refe_navtest_seam.py ran)")
            if not np.array_equal(d["token"], tok):
                raise RuntimeError("dump tokens differ from the seam's")
            pick20 = d["pick20"]
        sm = _load_mod("refe_navtest_seam_chk", f"{c.pkg}/eval/refe_navtest_seam.py")
        if not sm.pick20_equals_seam(pick20, poses):
            raise RuntimeError("pick20 != seam poses: to_navsim(pick20) must equal seam.poses bit for bit")
        out["pick20"] = list(pick20.shape)
    return out


def validate_result(c: Cfg, need_pass: bool) -> dict:
    rp = Path(c.s2_out) / "result.json"
    if not rp.is_file():
        raise RuntimeError(f"stage 2 wrote no {rp}")
    res = json.load(open(rp, encoding="utf-8"))
    if c.yaw_table:
        ys = (res.get("ego_yaw") or {}).get("sha256", "")
        if not ys.startswith(YAW_SHA_PREFIX):
            raise RuntimeError(f"result.json records yaw sha256 {ys[:12]!r}: the seam did not run under the registered yaw table")
    if need_pass:
        v = str(res.get("verdict"))
        if v.startswith("INCOMPLETE") or (res.get("harness_status") or {}).get("status") != "PASS":
            raise RuntimeError(f"stage 2 did not score: verdict {v!r}, harness {(res.get('harness_status') or {}).get('status')!r}")
    return {"verdict": res.get("verdict"), "primary": res.get("primary")}


def seam_to_plans8(seam_npz, out_npz) -> int:
    """the executed 8-pose plan of a seam as a transp_score plan file (tokens + plans8, ego frame, 0.5 s); refuses duplicate tokens / non-finite poses"""
    import numpy as np
    with np.load(seam_npz) as z:
        tok, poses = np.asarray(z["token"]).astype(str), np.asarray(z["poses"])
    if len(set(tok.tolist())) != len(tok):
        raise RuntimeError(f"{seam_npz}: duplicate tokens")
    if poses.ndim != 3 or poses.shape[1:] != (8, 3) or not np.isfinite(poses).all():
        raise RuntimeError(f"{seam_npz}: poses {poses.shape} (want [N,8,3], finite)")
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, tokens=tok, plans8=poses.astype(np.float64))
    return len(tok)


# ------------------------------------------------------------------ runner
class Runner:
    """runs argv lists; run_many starts all jobs then waits (the two transp_score shards run concurrently, never more than 2 processes)"""
    def run_many(self, jobs):
        procs = []
        for argv, env, log in jobs:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            procs.append((subprocess.Popen(argv, env=env, stdout=open(log, "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL), log))
        return [p.wait() for p, _ in procs]


class Chain:
    def __init__(self, c: Cfg, runner=None, say=print):
        self.c, self.runner, self._say = c, runner or Runner(), say
        self.mk = Path(c.markers)
        self.ident_p = self.mk / "ident.json"

    def say(self, m):
        line = f"[{time.strftime('%F %T')}] {m}"
        self._say(line)
        Path(self.c.work).mkdir(parents=True, exist_ok=True)
        with open(f"{self.c.work}/CHAIN.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def selected(self, step):
        c = self.c
        return (not c.only or step in c.only) and step not in c.skip

    def done(self, step) -> bool:
        return (self.mk / f"{step}.DONE").is_file()

    def mark(self, step, payload):
        self.mk.mkdir(parents=True, exist_ok=True)
        (self.mk / f"{step}.DONE").write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")

    def _retire_failed(self, step):
        f = self.mk / f"{step}.FAILED"
        if f.is_file():
            n = 1
            while (self.mk / f"{step}.FAILED.attempt_{n}").exists():
                n += 1
            f.rename(self.mk / f"{step}.FAILED.attempt_{n}")
            self.say(f"previous {step}.FAILED kept as {step}.FAILED.attempt_{n}; retrying {step}")

    def _ident_check(self, pf):
        """the first run records (ckpt sha, name, yaw sha); a resume with anything else is REFUSED (it would mix two checkpoints' artifacts under one marker set)"""
        ident = {"name": self.c.name, "ckpt_sha256": pf.get("ckpt_sha256"), "yaw_sha256": pf.get("yaw_sha256"), "base": self.c.base, "also": self.c.also}
        if self.ident_p.is_file():
            old = json.load(open(self.ident_p, encoding="utf-8"))
            if old != ident:
                raise Refusal(f"REFUSING: {self.mk} holds markers of another identity {old} != {ident}; use a fresh --work")
        else:
            self.mk.mkdir(parents=True, exist_ok=True)
            self.ident_p.write_text(json.dumps(ident, indent=1) + "\n", encoding="utf-8")

    def _run(self, label, argv, log, expect=(0,)):
        c = self.c
        argv = [a.replace("<ckpt sha256>", self._sha) if isinstance(a, str) else a for a in argv]
        self.say(f"RUN {label}: {subprocess.list2cmdline([str(a) for a in argv])}")
        t0 = time.time()
        rc = self.runner.run_many([(argv, child_env(c), log)])[0]
        self.say(f"  rc {rc} after {round(time.time() - t0)} s (log {log})")
        if rc not in expect:
            raise RuntimeError(f"{label}: rc {rc}, expected {expect}; see {log}")
        return rc

    # ---- steps
    def step_seam1(self):
        c = self.c
        for p in (c.seam, c.dump):
            if Path(p).exists():
                raise Refusal(f"REFUSING: {p} exists but seam1 has no DONE marker: stage 2 would reuse a seam of unknown yaw / checkpoint. Move it aside (the PI deletes) or point --work at the run that made it.")
        rc = self._run("seam1", stage2_cmd(c, c.name, True), f"{c.work}/logs/seam1.log", expect=(0, 1))
        info = validate_seam(c, c.name, rc, need_dump=True)
        info.update(validate_result(c, need_pass=False))
        return {"rc": rc, **info}

    def step_score(self):
        c = self.c
        rc = self._run("score", stage2_cmd(c, c.name, True), f"{c.work}/logs/score.log", expect=(0,))
        return {"rc": rc, **validate_result(c, need_pass=True)}

    def step_stage3(self):
        c = self.c
        argv = dict(plan(c))["stage3"][0][1]
        rc = self._run("stage3", argv, f"{c.work}/logs/stage3.log", expect=(0,))
        return {"rc": rc, "out": f"{c.work}/stage3"}

    def step_adherence(self):
        c = self.c
        rc = self._run("adherence", dict(plan(c))["adherence"][0][1], f"{c.work}/logs/adherence.log", expect=(0,))
        return {"rc": rc, "out": f"{c.work}/command_adherence_{c.name}.json"}

    def step_rep(self):
        import numpy as np
        c = self.c
        if Path(c.rep_seam).exists():
            raise Refusal(f"REFUSING: {c.rep_seam} exists but the rep has no DONE marker (stage 2 would reuse it)")
        rc = self._run("rep", stage2_cmd(c, c.rep_name, False), f"{c.work}/logs/rep.log", expect=(0, 1))
        with np.load(c.seam) as a, np.load(c.rep_seam) as b:
            ta, tb, pa, pb = a["token"], b["token"], a["poses"], b["poses"]
        same_tok = bool(np.array_equal(ta, tb))
        n_diff = int((pa != pb).reshape(len(pa), -1).any(1).sum()) if pa.shape == pb.shape else -1
        if not same_tok or n_diff != 0:
            raise RuntimeError(f"DETERMINISM FAILED: tokens equal {same_tok}, tokens with different poses {n_diff}")
        return {"rc": rc, "tokens_equal": True, "poses_bit_identical": True, "n": int(len(ta))}

    def step_families(self):
        c = self.c
        fam = Path(c.work) / "families"
        fam.mkdir(parents=True, exist_ok=True)
        subs = dict(dict(plan(c))["families"])         # ({label: argv}) of the families step
        done = fam / "substeps_done.json"
        state = json.load(open(done, encoding="utf-8")) if done.is_file() else {}

        def fin(k, v=True):
            state[k] = v
            done.write_text(json.dumps(state, indent=1), encoding="utf-8")

        if "plans" not in state:
            base_seam = _base_seam_path(c)
            n1 = seam_to_plans8(c.seam, f"{fam}/{c.arm_label}_plans8.npz")
            n2 = seam_to_plans8(base_seam, f"{fam}/{c.base_label}_plans8.npz")
            if n1 != n2:
                raise RuntimeError(f"plans8: arm {n1} tokens vs base {n2}")
            self.say(f"plans8 written: {n1} tokens each (base seam {base_seam})")
            fin("plans", {"n": n1, "base_seam": base_seam})
        shard_keys = [k for k in subs if k.startswith("transp_s")]
        todo = [k for k in shard_keys if k not in state]
        if todo:       # the shards run CONCURRENTLY (2 processes), each resumable on its own rows file
            rcs = self.runner.run_many([(subs[k], child_env(c), f"{c.work}/logs/{k}.log") for k in todo])
            for k, rc in zip(todo, rcs):
                self.say(f"RUN {k}: rc {rc}")
                if rc != 0:
                    raise RuntimeError(f"{k}: rc {rc}; see {c.work}/logs/{k}.log (re-run resumes the rows file)")
                fin(k, {"rc": rc})
        for k in ("fam_arm", "fam_base", "fam_ci"):
            if k not in state:
                self._run(k, subs[k], f"{c.work}/logs/{k}.log")
                fin(k, True)
        return {"substeps": state, "fam_ci": f"{c.work}/fam_ci_{c.base_label}_vs_{c.arm_label}.json"}

    # ---- driver
    def run(self) -> int:
        c = self.c
        os.environ["TANITAD_NAVSIM_ROOT"] = c.navsim_root          # pinned: the first-existing-root trap cannot arm
        pf = preflight(c, strict=True)
        self._sha = pf.get("ckpt_sha256", "")
        self.say(f"preflight ok: ckpt sha256 {self._sha[:12]}, yaw {str(pf.get('yaw_sha256'))[:8]}, tokens md5 {TOKENS_MD5[:8]}, comparators {c.base}/{c.also}, navsim root {c.navsim_root}")
        self._ident_check(pf)
        for step in STEPS[1:]:
            if not self.selected(step):
                continue
            if self.done(step):
                self.say(f"skip {step}: marker present")
                continue
            self._retire_failed(step)
            try:
                payload = getattr(self, f"step_{step}")()
            except Refusal:
                raise
            except BaseException:
                tb = traceback.format_exc()
                self.say(f"FAILED {step}\n{tb}")
                self.mk.mkdir(parents=True, exist_ok=True)
                (self.mk / f"{step}.FAILED").write_text(tb, encoding="utf-8")
                return 1
            self.mark(step, {"at": time.strftime("%F %T"), **payload})
            self.say(f"DONE {step}")
        if all(self.done(s) for s in STEPS[1:] if self.selected(s)) and not c.only and not c.skip:
            self.mark("chain", {"at": time.strftime("%F %T"), "name": c.name})
            self.say("chain DONE")
        return 0


def dry_run(c: Cfg, out=print) -> int:
    pf = preflight(c, strict=False)
    out(f"DRY preflight (would {'PASS' if pf['ok'] else 'REFUSE'}):")
    for k, v in pf["checks"].items():
        out(f"  {'ok  ' if v['ok'] else 'FAIL'} {k} {v['why']}")
    env = child_env(c)
    out(f"DRY env for every child: TANITAD_NAVSIM_ROOT={env['TANITAD_NAVSIM_ROOT']} TANITAD_REFE_PKG={env['TANITAD_REFE_PKG']} PYTHONPATH={env['PYTHONPATH']}")
    for step, cmds in plan(c):
        out(f"DRY step {step}{'  [selected]' if Chain(c).selected(step) else '  [skipped]'}")
        for label, argv in cmds:
            out(f"  DRY {label}: {subprocess.list2cmdline([str(a) for a in argv])}")
    return 0 if pf["ok"] else 3


def make_cfg(argv=None) -> Cfg:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True, help="the stage-2 run name (seams/stage2_<name>.npz), e.g. p7_full")
    ap.add_argument("--summary", default=None, help="the run's summary.json (done + model_final_sha256)")
    ap.add_argument("--expect-ckpt-sha256", default=None)
    ap.add_argument("--base", default="p6", help="the stage-3 / adherence / families comparator (a stage3_full BASELINES key)")
    ap.add_argument("--also", default="p2")
    ap.add_argument("--ego-yaw-table", default=YAW_DEFAULT)
    ap.add_argument("--no-ego-yaw-table", action="store_true", help="a checkpoint trained WITHOUT ego_yaw (the preflight still refuses a yaw checkpoint)")
    ap.add_argument("--work", default="")
    ap.add_argument("--data", default=os.environ.get("REFE_NAVTEST_DIR", DATA_DEFAULT))
    ap.add_argument("--navsim-root", default=NAVSIM_ROOT_DEFAULT)
    ap.add_argument("--only", default="", help="comma list of steps")
    ap.add_argument("--skip", default="", help="comma list of steps, e.g. rep")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--detach", action="store_true", help="re-launch this chain as a detached process (own process group, no window, stdin NUL, stdout -> <work>/chain_stdout.log) and exit; P6's launch_detached.py form")
    ap.add_argument("--internal-plans8", nargs=2, metavar=("SEAM", "OUT"), help=argparse.SUPPRESS)
    a, _ = ap.parse_known_args(argv)
    c = Cfg(ckpt=a.ckpt, name=a.name, base=a.base, also=a.also, summary=a.summary, expect_sha=a.expect_ckpt_sha256,
            yaw_table=None if a.no_ego_yaw_table else a.ego_yaw_table, work=a.work, data=a.data.replace("\\", "/"),
            navsim_root=a.navsim_root.replace("\\", "/").rstrip("/"), only=tuple(x for x in a.only.split(",") if x), skip=tuple(x for x in a.skip.split(",") if x))
    c.dry_run = a.dry_run
    for s in (*c.only, *c.skip):
        if s not in STEPS[1:]:
            ap.error(f"unknown step {s!r}; steps: {', '.join(STEPS[1:])}")
    return c


def detach_argv(argv: list) -> list:
    """the child's argv for --detach: this script with every original argument except --detach"""
    return [PY_T, Path(__file__).resolve().as_posix(), *[a for a in argv if a != "--detach"]]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--internal-plans8" in argv:        # plans helper: python p7_read_chain.py --internal-plans8 <seam npz> <out npz>
        i = argv.index("--internal-plans8")
        return 0 if seam_to_plans8(argv[i + 1], argv[i + 2]) else 1
    c = make_cfg(argv)
    if "--detach" in argv and not c.dry_run:
        Path(c.work).mkdir(parents=True, exist_ok=True)
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW | getattr(subprocess, "DETACHED_PROCESS", 0)
        p = subprocess.Popen(detach_argv(argv), cwd=c.work, stdin=subprocess.DEVNULL, stdout=open(f"{c.work}/chain_stdout.log", "a", encoding="utf-8"), stderr=subprocess.STDOUT, creationflags=flags)
        print(f"detached pid {p.pid}; progress: {c.work}/CHAIN.log, markers: {c.markers}")
        return 0
    if c.dry_run:
        return dry_run(c)
    try:
        return Chain(c).run()
    except Refusal as e:
        print(e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
