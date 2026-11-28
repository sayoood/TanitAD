#!/usr/bin/env python3
"""Self-test of the I-rep package (PREREG_SAFEOBJ section 7 "inference" floor): `fp_forward_irep.py` (the navtest forward with --eval-batch / --token-order-seed), `irep_tool.py`
(drive / seams / make) and the hand-off to the consumer `navtestE/navtest_E.py irep`.   CPU only, no pod, no GPU, no network, ONE child process at a time.

    "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe" -I -B -X utf8 test_irep.py [--no-mutations] [--only g1,g2] [--keep]       -> ZZTEST_PASS <n>

WHAT IS REAL AND WHAT IS A STAND-IN  (the "built, tested, unreachable from its caller" rule: say it, do not blur it)
  REAL   * the ORIGINAL navtestE/fp/fp_forward.py, run unchanged as the reference (md5-pinned; it is also the free-point worker's own file)
         * the real variant file `fp_forward_irep.py`, run as a child process with the real argparse, the real ZipCache / cv2 JPEG decode, the real Taps hooks, the real IC2 check
         * `navtest_E.py irep` (the consumer), run in its own process, and `navtest_E.read_irep` (the reader `build` uses on the file we write)
         * `navtest_E.build_seams` on the free-point dry-run chunk (a REAL CPU forward of 2 tokens): the I-rep seam step reproduces the live picker's seam bit for bit
         * the REAL harness artifact formats (seam npz, counts.json, status.json, CSV) of the free-point dry run, read by the guard's own parser
  STAND-IN (labelled SYNTHETIC) * the model: a ~10 k-parameter torch module with the interface the worker uses (`score_q_mlp` / `score_dec` / `score_head`, `model(img, ego, goal, calib=)`),
           built in a temp dir as `model.py` / `ckpt_io.py`.  Its scorer head can add a deterministic function of the BATCH SIZE (env IREP_FIXTURE_BATCH_EPS), standing in for a GPU kernel that is
           not batch-invariant: it exists so the pipeline has a signal to carry, and says NOTHING about the real model's I-rep.  With the variable unset the model is batch-invariant mathematically.
         * the harness: `fake_harness` picks by the trainer's sel_score formula and scores the pick from a fixed per-token table.  The real harness (score_e.py, nuPlan) is NOT run here.
  NO REAL I-REP NUMBER EXISTS.  Every irep_abs_delta below is a fixture value.

Mutations rewrite the SOURCE of the variant or of irep_tool.py (old -> new, `old` occurring exactly once), re-run the groups they exist for, and each must go RED AT the named checks and stay
GREEN at the named controls.  "RED for its reason" is that: it fails where it should and nowhere it should not.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PKG = HERE.parent
PY = sys.executable
ORIG = PKG / "navtestE" / "fp" / "fp_forward.py"
VARIANT = HERE / "fp_forward_irep.py"
TOOL = HERE / "irep_tool.py"
NAVTEST_E = PKG / "navtestE" / "navtest_E.py"
DRY = "C:/Users/Admin/freept/work/dry"
DRY_SCORE = "E:/Projects/TanitAD/data/refe_navtest/score"
FP_ORIGINAL_COPY = "C:/Users/Admin/freept/raw/2026-10-17-free-point-navtest/code/fp_forward.py"
ORIG_MD5 = "82bf21514f919ee4eafab72690004e34"
STATE = "R0d_B"
N_LOGS, PER_LOG = 4, 10
N = N_LOGS * PER_LOG
M = 6

# ------------------------------------------------------------------------------------------------ the stand-in model code (SYNTHETIC)
MODEL_PY = r'''
import json, os
import torch, torch.nn as nn, torch.nn.functional as F


class REFeConfig:
    def __init__(self):
        self.img_h, self.img_w, self.n_cameras = 16, 24, 2
        self.n_proposals, self.n_score_components, self.nav_mode = 6, 7, "goal"

    @classmethod
    def for_backbone(cls, name):
        return cls()


class _Blk(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.q, self.k, self.v, self.o = nn.Linear(d, d), nn.Linear(c, d), nn.Linear(c, d), nn.Linear(d, d)

    def forward(self, s, ctx):
        return s + self.o(F.scaled_dot_product_attention(self.q(self.n(s)), self.k(ctx), self.v(ctx)))


class _Head(nn.Module):
    """a linear head + (fixture only) a deterministic function of the BATCH SIZE: the stand-in for a kernel that is not batch-invariant"""
    def __init__(self, d, c):
        super().__init__()
        self.lin = nn.Linear(d, c)

    def forward(self, s):
        out = self.lin(s)
        eps = float(os.environ.get("IREP_FIXTURE_BATCH_EPS", "0"))
        if eps:
            out = out + eps * torch.sin(torch.arange(s.shape[1], dtype=out.dtype).view(1, -1, 1) * 0.9 + 0.31 * s.shape[0])
        return out


class REFe(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        torch.manual_seed(1234)
        self.cfg = cfg
        D, d, Mq, T = 16, 16, cfg.n_proposals, 20
        self.stem = nn.Linear(3, D)
        self.queries = nn.Parameter(torch.randn(Mq, D))
        self.ego_enc = nn.Linear(7 + 4, D)
        self.traj_head = nn.Linear(D, T * 3)
        self.scene_proj = nn.Linear(D, d)
        self.score_q_mlp = nn.Sequential(nn.Linear(T * 3, d), nn.GELU(), nn.Linear(d, d))
        self.score_dec = nn.ModuleList([_Blk(d, d) for _ in range(2)])
        self.score_head = _Head(d, cfg.n_score_components)

    def forward(self, img, ego, goal, detach_scorer=True, calib=None, score_extra=None):
        path = os.environ.get("IREP_FIXTURE_FWDLOG")
        if path:                                                   # the INDEPENDENT record of what the worker really did (batch size, token ids in call order)
            with open(path, "a") as f:
                f.write(json.dumps({"b": int(img.shape[0]), "ids": [int(round(float(x) * 1000)) for x in ego[:, 0]]}) + "\n")
        B, N = img.shape[:2]
        x = F.avg_pool2d(img.flatten(0, 1), 4).flatten(2).transpose(1, 2).reshape(B, -1, 3)
        vis = self.stem(x)
        if calib is not None:
            vis = vis + 0.001 * calib.float().mean(dim=(1, 2)).view(B, 1, 1)
        q = self.queries.unsqueeze(0) + (self.ego_enc(torch.cat([ego, goal], -1)) + vis.mean(1)).unsqueeze(1)
        traj = self.traj_head(q).view(B, self.cfg.n_proposals, 20, 3)
        s = self.score_q_mlp(traj.detach().flatten(2))
        sctx = self.scene_proj(vis)
        for blk in self.score_dec:
            s = blk(s, sctx)
        return traj, self.score_head(s)
'''
CKPT_IO_PY = r'''
def config_for_checkpoint(cfg, ckpt):
    return cfg


def load_for_inference(model, ckpt, map_location="cpu", backbone="vitl16", meta_out=None):
    open(ckpt, "rb").close()
    if meta_out is not None:
        meta_out["per_sample_calib"] = True
    return "fixture"
'''
RUNNER_PY = r'''
import sys, time, runpy
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
time.time = lambda: 1.7e9                       # a frozen clock: the npz zip headers and meta `seconds` become reproducible, so two runs can be compared BYTE for byte
script = sys.argv[2]
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
'''

SCORER = ("score_q_mlp", "score_dec", "score_head")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_path(p) -> str:
    return sha256_bytes(Path(p).read_bytes())


# ------------------------------------------------------------------------------------------------ the world
class World:
    """the fixture: tokens, JPEG frames in a zip, per-log rigs, two scorer-state files, the stand-in model code, and a cache of the child-process runs"""

    def __init__(self, td: Path, variant: Path = VARIANT, tool: Path = TOOL, shared: "World | None" = None, keep_all: bool = False):
        self.td, self.variant, self.tool = Path(td), Path(variant), Path(tool)
        self.cache: dict = {}
        if shared is not None:                                           # a mutated world re-uses the fixture files and the ORIGINAL's runs
            self.__dict__.update({k: v for k, v in shared.__dict__.items() if k not in ("variant", "tool", "cache")})
            self.cache = dict(shared.cache) if keep_all else {k: v for k, v in shared.cache.items() if isinstance(k, tuple) and k[0] == "orig"}
            return
        self.model_dir = self.td / "modelcode"
        self.model_dir.mkdir()
        (self.model_dir / "model.py").write_text(MODEL_PY, encoding="utf-8")
        (self.model_dir / "ckpt_io.py").write_text(CKPT_IO_PY, encoding="utf-8")
        (self.td / "runner.py").write_text(RUNNER_PY, encoding="utf-8")
        self.logs = [f"2021.00.00.00.00.0{i}_veh-0_{i:05d}_{i + 10:05d}" for i in range(N_LOGS)]
        self.tokens = [f"tok{i:03d}" for i in range(N)]                  # sorted by (log, token) == by id: position p IS token id p
        self.log_of = {t: self.logs[i // PER_LOG] for i, t in enumerate(self.tokens)}
        self.build_inputs()
        self.build_states()

    # ---- inputs
    def build_inputs(self):
        import cv2
        fr = self.td / "frames"
        fr.mkdir()
        inputs = {}
        with zipfile.ZipFile(fr / "fx.zip", "w", zipfile.ZIP_STORED) as z:
            for i, t in enumerate(self.tokens):
                refs = []
                for c in range(2):
                    rng = np.random.RandomState(1000 * i + c)
                    img = (rng.randint(0, 255, (16, 24, 3)) * 0.5 + np.linspace(0, 120, 24)[None, :, None]).clip(0, 255).astype(np.uint8)
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    assert ok
                    z.writestr(f"{t}_c{c}.jpg", buf.tobytes())
                    refs.append(f"fx.zip::{t}_c{c}.jpg")
                rng = np.random.RandomState(7 * i + 3)
                inputs[t] = {"frames": refs, "ego": [i / 1000.0] + [float(v) for v in rng.uniform(-1, 1, 6)], "goal": [float(v) for v in rng.uniform(-1, 1, 4)], "log": self.log_of[t]}
        self.frames_dir = str(fr)
        self.inputs = str(self.td / "inputs.json")
        json.dump(inputs, open(self.inputs, "w"))
        self.calib = str(self.td / "calib.npz")
        np.savez(self.calib, **{lg: np.random.RandomState(50 + li).uniform(-1.5, 1.5, (1, 2, 16)) for li, lg in enumerate(self.logs)})
        self.ckpt = str(self.td / "ckpt.pt")
        Path(self.ckpt).write_bytes(b"fixture checkpoint")
        plan = {"chunks": [{"name": f"chunk_{k:03d}", "logs": self.logs[2 * k:2 * k + 2], "bytes": 1, "tokens": 2 * PER_LOG} for k in range(2)]}
        json.dump(plan, open(self.td / "ship_plan.json", "w"))

    def build_states(self):
        import torch
        sys.path.insert(0, str(self.model_dir))
        spec = importlib.util.spec_from_file_location("fixture_model", self.model_dir / "model.py")
        FM = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(FM)

        def one(seed):
            m = FM.REFe(FM.REFeConfig())
            g = torch.Generator().manual_seed(seed)
            return {n: {k: v + 0.3 * torch.randn(v.shape, generator=g) for k, v in getattr(m, n).state_dict().items()} for n in SCORER}
        self.states = {}
        for nm, (sa, sb) in {"X": (11, 12), "Y": (21, 22)}.items():          # X = "R0d", Y = "R1d" (a DIFFERENT trained scorer)
            p = self.td / f"scorer_states_{nm}.pt"
            torch.save({"A": one(sa), "B": one(sb), "update": 1000}, p)
            self.states[nm] = str(p)

    # ---- running a forward
    def fwd(self, script, tag, args=(), eps=0.0, state="X", name=STATE, key="B", extra_state_args=None):
        ck = ("orig" if Path(script) == ORIG else sha256_path(script), tag, tuple(args), eps, state)
        if ck in self.cache:
            return self.cache[ck]
        out = self.td / f"{tag}.npz"
        flog = self.td / f"{tag}.fwdlog"
        flog.unlink(missing_ok=True)
        cmd = [PY, "-I", "-B", "-X", "utf8", str(self.td / "runner.py"), str(self.model_dir), str(script), "run", "--inputs", self.inputs, "--calib", self.calib, "--ckpt", self.ckpt,
               "--ckpt-md5-prefix", "", "--state", f"{name}={self.states[state]}:{key}", "--frames-dir", self.frames_dir, "--device", "cpu", "--out", str(out), *args]
        env = dict(os.environ, OMP_NUM_THREADS="1", IREP_FIXTURE_FWDLOG=str(flog), IREP_FIXTURE_BATCH_EPS=str(eps))
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        res = {"rc": r.returncode, "stdout": r.stdout.replace(str(out), "<OUT>"), "stderr": r.stderr, "out": out,
               "fwdlog": [json.loads(l) for l in flog.read_text().splitlines()] if flog.exists() else [],
               "z": ({k: v for k, v in np.load(out, allow_pickle=False).items()} if out.exists() else {}), "sha256": (sha256_path(out) if out.exists() else None)}
        if res["z"]:
            res["meta"] = json.loads(str(res["z"]["meta_json"]))
        self.cache[ck] = res
        return res

    def orig(self, extra=()):
        return self.fwd(ORIG, "orig" + ("_" + "_".join(extra) if extra else ""), ("--repeat-first", "4") if not extra else tuple(extra))

    # ---- the I-rep reads through the driver
    def drive(self, tag, batch, seed, state="X", eps=0.3, python=PY, extra=()):
        rd = self.td / f"read_{tag}"
        cmd = [PY, "-I", "-B", "-X", "utf8", str(self.tool), "drive", "--work", str(self.td), "--read-dir", str(rd), "--python", python, "--refe-code", str(self.model_dir),
               "--backbone-root", "unused", "--forward", str(self.variant), "--inputs", self.inputs, "--calib", self.calib, "--ckpt", self.ckpt, "--ckpt-md5-prefix", "",
               "--frames-dir", self.frames_dir, "--state", f"{STATE}={self.states[state]}:B", "--batch", str(batch), "--seed", str(seed), "--device", "cpu", "--threads", "1", *extra]
        r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, IREP_FIXTURE_BATCH_EPS=str(eps)))
        return r.returncode, r.stdout + r.stderr, rd

    def read(self, tag, batch, seed, state="X", eps=0.3):
        ck = ("read", self.variant.name, tag, batch, seed, state, eps)
        if ck not in self.cache:
            rc, txt, rd = self.drive(tag, batch, seed, state, eps)
            assert rc == 0 and "ZZIREP_DRIVE_DONE 2 chunk(s)" in txt, txt[-600:]
            self.cache[ck] = rd
        return self.cache[ck]


# ------------------------------------------------------------------------------------------------ the fake harness (SYNTHETIC): picks by sel_score, PDMS from a fixed table
def table_of(token: str) -> np.ndarray:
    return np.random.RandomState(int(token[3:])).uniform(0.3, 1.0, M)


def picks_of(logits: np.ndarray) -> np.ndarray:
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64)))
    return (p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12 * p[..., 6]).argmax(-1)


def merge(read_dir: Path, state=STATE):
    toks, lg = [], []
    for f in sorted((read_dir / "out").glob("chunk_*.npz")):
        z = np.load(f, allow_pickle=False)
        toks += [str(t) for t in z["token"]]
        lg.append(z[f"logits_{state}"])
    return toks, np.concatenate(lg)


def fake_harness(read_dir: Path, label: str, state=STATE) -> dict:
    """the files the production chain leaves behind for one read: seam npz + points npz (build_seams), score_logs status (fp_score), counts.json + CSV (score_e.py)"""
    toks, lg = merge(read_dir, state)
    pk = picks_of(lg)
    seam = read_dir / f"seam_{state}.npz"
    np.savez(seam, token=np.array(toks), fingerprint=np.array(["f"] * len(toks)), poses=np.zeros((len(toks), 8, 3), np.float32), sampling=np.array([8, 0.5]), arm=np.array(f"SOBJE_{state}"))
    np.savez(read_dir / "points_E.npz", token=np.array(toks), **{f"pick_{state}": pk, "pick_BASE": pk})
    (read_dir / "score_logs").mkdir(exist_ok=True)
    json.dump({"label": label, "status": "PASS", "csv_valid_rows": len(toks)}, open(read_dir / "score_logs" / f"{label}.status.json", "w"))
    cd = read_dir / "csv" / label
    cd.mkdir(parents=True, exist_ok=True)
    pdms = {t: float(table_of(t)[k]) for t, k in zip(toks, pk)}
    rows = [",token,valid,no_at_fault_collisions,drivable_area_compliance,ego_progress,time_to_collision_within_bound,comfort,driving_direction_compliance,score"]
    for i, t in enumerate(toks):
        rows.append(f"{i},{t},True,1.0,1.0,1.0,1.0,1.0,1.0,{pdms[t]!r}")
    rows.append(f"{len(toks)},average,True,1.0,1.0,1.0,1.0,1.0,1.0,{float(np.mean(list(pdms.values())))!r}")
    (cd / f"{label}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    json.dump({"label": label, "arm": "SEAM:" + str(seam).replace("/", "\\")}, open(cd / f"{label}.counts.json", "w"))
    return {"csv": str(cd / f"{label}.csv"), "work": str(read_dir), "chunks": str(read_dir / "out"), "pdms": pdms, "mean_x100": 100.0 * float(np.mean(list(pdms.values()))), "picks": dict(zip(toks, pk.tolist()))}


def make_cmd(w: World, A: dict, B: dict, out: Path, tool=None, **over):
    kw = dict(state=STATE, a_chunks=A["chunks"], b_chunks=B["chunks"], a_work=A["work"], b_work=B["work"], a_csv=A["csv"], b_csv=B["csv"], out=str(out), n_tokens=N, batch_a=8, batch_b=32,
              expected_states=str(w.td / "EXPECTED.json"))
    kw.update(over)
    cmd = [PY, "-I", "-B", "-X", "utf8", str(tool or w.tool), "make"]
    for k, v in kw.items():
        if v is not None:
            cmd += [f"--{k.replace('_', '-')}", str(v)]
    return cmd


def run_make(w: World, A, B, name: str, tool=None, **over):
    key = ("make", name, str(tool or w.tool))                      # keyed by the tool file: a mutated tool must not read the unmutated tool's answer
    if name == "legit" and key in w.cache:
        return w.cache[key]
    res = _run_make(w, A, B, name, tool, **over)
    if name == "legit":
        w.cache[key] = res
    return res


def _run_make(w: World, A, B, name: str, tool=None, **over):
    out = w.td / f"irep_{name}.json"
    out.unlink(missing_ok=True)
    r = subprocess.run(make_cmd(w, A, B, out, tool, **over), capture_output=True, text=True, cwd=str(HERE))
    side = Path(str(out)[:-5] + ".check.json")
    return r.returncode, r.stdout + r.stderr, out, (json.load(open(side)) if side.exists() else None)


def harness_world(w: World):
    """reads A (X, batch 8, seed 1), B (X, batch 32, seed 2), Bd (Y under the SAME state name, batch 32, seed 2), Bs (X, batch 32, seed 1): driver -> fake harness"""
    if "HW" in w.cache:
        return w.cache["HW"]
    H = {}
    for tag, (b, s, st) in {"A": (8, 1, "X"), "B": (32, 2, "X"), "Bd": (32, 2, "Y"), "Bs": (32, 1, "X")}.items():
        rd = w.read(tag, b, s, st)
        H[tag] = fake_harness(rd, f"refe_sobj_irep{tag}_{STATE}")
    # EXPECTED_STATES: the independently computed hash of state X (the main read's EXPECTED_STATES_E.json plays this role)
    sys.path.insert(0, str(w.model_dir))
    import torch
    sys.path.insert(0, str(ORIG.parent))
    import fp_forward as FF
    ss = torch.load(w.states["X"], map_location="cpu", weights_only=False)
    json.dump({STATE: {"sha256_tensors": FF.sha256_tensors(ss["B"]), "update": 1000, "key": "B"}}, open(w.td / "EXPECTED.json", "w"))
    sy = torch.load(w.states["Y"], map_location="cpu", weights_only=False)
    w.cache["HW"] = H
    w.cache["sha_x"], w.cache["sha_y"] = FF.sha256_tensors(ss["B"]), FF.sha256_tensors(sy["B"])
    return H


# ------------------------------------------------------------------------------------------------ the checks
CHECKS: dict = {}


def check(name, group):
    def deco(fn):
        CHECKS[name] = (group, fn)
        return fn
    return deco


def aligned_margin(V: np.ndarray, O: np.ndarray):
    """-> (max |V[i]-O[i]|, min over i of the distance from V[i] to its NEAREST OTHER original row): per-token alignment with a margin"""
    n = len(V)
    Vf, Of = V.reshape(n, -1).astype(np.float64), O.reshape(n, -1).astype(np.float64)
    D = np.abs(Vf[:, None, :] - Of[None, :, :]).max(-1)
    diag = np.diag(D).max()
    off = D + np.eye(n) * 1e9
    return float(diag), float(off.min())


def perm_ids(seed: int, n: int = N):
    return [int(i) for i in np.random.RandomState(seed).permutation(n)]


# ---- variant provenance
@check("variant:original_is_the_pinned_free_point_file", "variant")
def _(w):
    md5 = hashlib.md5(ORIG.read_bytes()).hexdigest()
    line = [l for l in (PKG / "navtestE" / "fp_COPIES.md5").read_text().splitlines() if l.endswith("fp_forward.py")][0].split()[0]
    assert md5 == ORIG_MD5 == line == hashlib.md5(Path(FP_ORIGINAL_COPY).read_bytes()).hexdigest(), (md5, line)


@check("variant:shipped_file_is_regenerated_from_the_original_by_make_variant", "variant")
def _(w):
    sys.path.insert(0, str(HERE))
    import make_variant as MV
    out, diff = MV.build()
    assert out.encode("utf-8") == VARIANT.read_bytes(), "the shipped fp_forward_irep.py is not make_variant.py's output"
    assert (HERE / "fp_forward_irep.diff").read_text(encoding="utf-8") == diff


@check("variant:diff_is_a_pure_addition_and_the_default_function_body_is_verbatim", "variant")
def _(w):
    import ast
    diff = (HERE / "fp_forward_irep.diff").read_text(encoding="utf-8").splitlines()
    removed = [l for l in diff if l.startswith("-") and not l.startswith("---")]
    assert not removed, f"the variant REMOVES {len(removed)} original line(s): {removed[:3]}"
    o_src, v_src = ORIG.read_text(encoding="utf-8"), VARIANT.read_text(encoding="utf-8")

    def fn(src, name):
        t = ast.parse(src)
        node = [n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name][0]
        return ast.get_source_segment(src, node)
    ob, vb = fn(o_src, "run_tokens"), fn(v_src, "run_tokens")
    dispatch = "    if irep_active(a):\n        return run_tokens_irep(a, model, cfg, states, tokens, inputs, calib, zc, log=log)\n"
    assert vb.replace("\r\n", "\n").replace(dispatch, "") == ob.replace("\r\n", "\n"), "run_tokens (the default path) differs from the original by more than the one dispatch"
    for other in ("build_img", "score_with", "load_model", "load_states", "setup", "cmd_run", "cmd_serve"):
        assert fn(o_src, other) == fn(v_src, other), f"{other} changed"


# ---- the defaults are byte-identical
def defaults_pair(w):
    return w.orig(), w.fwd(w.variant, "v_defaults", ("--repeat-first", "4"))


@check("identity:defaults_return_code_and_stdout_identical", "identity")
def _(w):
    o, v = defaults_pair(w)
    assert o["rc"] == 0 and v["rc"] == 0, (o["rc"], v["rc"], v["stderr"][-400:])
    assert o["stdout"] == v["stdout"], "\n--- original\n" + o["stdout"] + "\n--- variant\n" + v["stdout"]


@check("identity:defaults_file_byte_identical", "identity")
def _(w):
    o, v = defaults_pair(w)
    assert o["sha256"] and o["sha256"] == v["sha256"], f"original npz {str(o['sha256'])[:16]} != variant npz {str(v['sha256'])[:16]}"
    print(f"      (original and variant npz: sha256 {o['sha256'][:16]}..., {o['out'].stat().st_size} bytes, {len(o['z']['token'])} tokens, {len(o['z'])} arrays)", flush=True)
    proof = {"original_npz_sha256": o["sha256"], "variant_npz_sha256": v["sha256"], "bytes": o["out"].stat().st_size, "tokens": int(len(o["z"]["token"])), "arrays": sorted(o["z"]),
             "original": str(ORIG), "original_md5": hashlib.md5(ORIG.read_bytes()).hexdigest(), "variant_sha256": sha256_path(w.variant), "args": "run ... --repeat-first 4 (both)",
             "clock": "time.time frozen by runner.py, so the zip headers and meta `seconds` are reproducible", "byte_identical": o["sha256"] == v["sha256"]}
    if w.variant == VARIANT:
        json.dump(proof, open(HERE / "logs" / "identity_proof.json", "w"), indent=1)


@check("identity:defaults_arrays_and_meta_equal", "identity")
def _(w):
    o, v = defaults_pair(w)
    assert set(o["z"]) == set(v["z"]), (sorted(o["z"]), sorted(v["z"]))
    for k in o["z"]:
        a, b = o["z"][k], v["z"][k]
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), f"array {k} differs"
    assert o["meta"] == v["meta"] and "irep" not in v["meta"], (o["meta"], v["meta"])


@check("identity:default_forward_calls_are_the_planners_batch_1_in_sorted_order", "identity")
def _(w):
    o, v = defaults_pair(w)
    ids = [i for c in v["fwdlog"] for i in c["ids"]]
    # 40 calls of batch 1 for the main pass + 4 for the IC5 repeat of the first 4 tokens
    assert [c["b"] for c in v["fwdlog"]] == [1] * (N + 4) and ids == list(range(N)) + list(range(4)), ([c["b"] for c in v["fwdlog"]][:6], ids[:6])
    assert [c["b"] for c in o["fwdlog"]] == [c["b"] for c in v["fwdlog"]]


@check("identity:batch_1_with_a_shuffled_order_reproduces_the_original_bit_for_bit", "identity")
def _(w):
    o = w.orig()
    v = w.fwd(w.variant, "v_b1_s7", ("--eval-batch", "1", "--token-order-seed", "7"))
    assert v["rc"] == 0, v["stderr"][-400:]
    for k in o["z"]:
        if k != "meta_json":
            a, b = o["z"][k], v["z"][k]
            assert a.tobytes() == b.tobytes(), f"array {k} differs after shuffle + restore at batch 1"
    assert [i for c in v["fwdlog"] for i in c["ids"]] == perm_ids(7)


# ---- batch and order
@check("batch_order:output_token_array_is_the_original_order", "batch_order")
def _(w):
    o = w.orig()
    for tag, args in (("v_b8_s3", ("--eval-batch", "8", "--token-order-seed", "3")), ("v_b32_s5", ("--eval-batch", "32", "--token-order-seed", "5"))):
        v = w.fwd(w.variant, tag, args)
        assert v["rc"] == 0, v["stderr"][-500:]
        assert np.array_equal(v["z"]["token"], o["z"]["token"]) and np.array_equal(v["z"]["log"], o["z"]["log"]), tag


@check("batch_order:forward_ran_at_the_requested_batch_sizes", "batch_order")
def _(w):
    for tag, args, sizes in (("v_b8_s3", ("--eval-batch", "8", "--token-order-seed", "3"), [8] * 5), ("v_b32_s5", ("--eval-batch", "32", "--token-order-seed", "5"), [32, 8])):
        v = w.fwd(w.variant, tag, args)
        got = [c["b"] for c in v["fwdlog"]]
        assert got == sizes, f"{tag}: the model was called with batch sizes {got}, expected {sizes}"


@check("batch_order:processing_order_is_the_seeded_permutation", "batch_order")
def _(w):
    for tag, args, seed in (("v_b8_s3", ("--eval-batch", "8", "--token-order-seed", "3"), 3), ("v_b32_s5", ("--eval-batch", "32", "--token-order-seed", "5"), 5)):
        v = w.fwd(w.variant, tag, args)
        got = [i for c in v["fwdlog"] for i in c["ids"]]
        assert got == perm_ids(seed), f"{tag}: tokens were visited in order {got[:8]}..., the seeded permutation starts {perm_ids(seed)[:8]}"
        assert got != list(range(N)), "the order was not shuffled"


@check("batch_order:rows_are_per_token_aligned_to_the_same_token_ids", "batch_order")
def _(w):
    o = w.orig()
    for tag, args in (("v_b8_s3", ("--eval-batch", "8", "--token-order-seed", "3")), ("v_b32_s5", ("--eval-batch", "32", "--token-order-seed", "5"))):
        v = w.fwd(w.variant, tag, args)
        for k in ("traj", "logits_BASE", f"logits_{STATE}"):
            diag, off = aligned_margin(v["z"][k], o["z"][k])
            assert diag <= 1e-5, f"{tag}/{k}: row i of the batched read is {diag:.3g} away from the batch-1 row of the SAME token (not float noise: misaligned or mis-batched)"
            assert off >= 1000 * max(diag, 1e-9) and off > 1e-3, f"{tag}/{k}: the fixture cannot tell tokens apart (nearest other row {off:.3g} vs aligned {diag:.3g})"
            print(f"      {tag}/{k}: aligned max |diff| {diag:.3g}; nearest OTHER token's row {off:.3g} away (margin x{off / max(diag, 1e-12):.3g})", flush=True)


@check("batch_order:meta_records_the_run_and_ic2_is_zero_per_token", "batch_order")
def _(w):
    for tag, args, bs, seed, counts in (("v_b8_s3", ("--eval-batch", "8", "--token-order-seed", "3"), 8, 3, {"8": 5}), ("v_b32_s5", ("--eval-batch", "32", "--token-order-seed", "5"), 32, 5, {"32": 1, "8": 1})):
        v = w.fwd(w.variant, tag, args)
        r = v["meta"]["irep"]
        want = hashlib.sha256(np.array(perm_ids(seed), np.int64).tobytes()).hexdigest()
        assert r["eval_batch"] == bs and r["token_order_seed"] == seed and r["n"] == N and r["batch_size_counts"] == counts, r
        assert r["order_sha256"] == want, "the recorded order hash is not the hash of the seeded permutation"
        assert float(v["z"]["ic2_maxabs"].max()) == 0.0 and v["z"]["ic2_maxabs"].shape == (N,)


@check("batch_order:a_bad_seed_is_refused", "batch_order")
def _(w):
    v = w.fwd(w.variant, "v_badseed", ("--eval-batch", "8", "--token-order-seed", "4294967296"))
    assert v["rc"] != 0 and "REFUSED" in (v["stderr"] + v["stdout"]), (v["rc"], v["stderr"][-300:])


# ---- the chain: driver -> seams -> harness -> make
@check("chain:driver_writes_one_chunk_per_plan_entry_and_passes_the_options", "chain")
def _(w):
    H = harness_world(w)
    rd = w.read("A", 8, 1)
    fs = sorted(p.name for p in (rd / "out").glob("*.npz"))
    assert fs == ["chunk_000.npz", "chunk_001.npz"], fs
    z0 = np.load(rd / "out" / "chunk_000.npz", allow_pickle=False)
    m = json.loads(str(z0["meta_json"]))["irep"]
    assert m["eval_batch"] == 8 and m["token_order_seed"] == 1 and m["n"] == 2 * PER_LOG and m["batch_size_counts"] == {"8": 2, "4": 1}, m
    # the same logs, directly, with the same options, bitwise
    lf = w.td / "chunk0.logs"
    lf.write_text("\n".join(w.logs[:2]) + "\n")
    d = w.fwd(w.variant, "direct_c0", ("--eval-batch", "8", "--token-order-seed", "1", "--logs-file", str(lf)), eps=0.3)
    for k in ("token", "log", "traj", "logits_BASE", f"logits_{STATE}", "ic2_maxabs"):
        assert d["z"][k].tobytes() == z0[k].tobytes(), k


@check("chain:driver_resumes_and_refuses_mixed_settings", "chain")
def _(w):
    harness_world(w)
    rc, txt, rd = w.drive("A", 8, 1)
    assert rc == 0 and txt.count("skip chunk_") == 2, txt[-400:]
    rc, txt, _ = w.drive("A", 32, 1)
    assert rc == 2 and "REFUSED" in txt and "settings" in txt, txt[-400:]


@check("chain:serve_mode_the_pods_entry_point_matches_run_mode", "chain")
def _(w):
    """the pod runs `serve`, not `run`: the same chunk through `serve --eval-batch 8 --token-order-seed 1` must equal the `run` output of the same logs bitwise, frames deleted after use, IC5 skipped"""
    harness_world(w)
    sw = w.td / "serve_w"
    shutil.rmtree(sw, ignore_errors=True)
    (sw / "in" / "chunk_000").mkdir(parents=True)
    shutil.copy(Path(w.frames_dir) / "fx.zip", sw / "in" / "chunk_000" / "fx.zip")
    (sw / "in" / "chunk_000.logs").write_text("\n".join(w.logs[:2]) + "\n")
    (sw / "in" / "chunk_000.ready").write_text("")
    (sw / "in" / "ALL_DONE").write_text("")
    cmd = [PY, "-I", "-B", "-X", "utf8", str(w.td / "runner.py"), str(w.model_dir), str(w.variant), "serve", "--work-dir", str(sw), "--inputs", w.inputs, "--calib", w.calib, "--ckpt", w.ckpt,
           "--ckpt-md5-prefix", "", "--state", f"{STATE}={w.states['X']}:B", "--device", "cpu", "--no-gate", "--eval-batch", "8", "--token-order-seed", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1", IREP_FIXTURE_BATCH_EPS="0.3"))
    assert r.returncode == 0 and "ZZFWD_SERVE_DONE" in r.stdout and "ZZFWD_CHUNK_DONE chunk_000 20 tokens ic2_max 0" in r.stdout, (r.stdout + r.stderr)[-600:]
    npz = sw / "out" / "chunk_000.npz"
    assert npz.exists() and (sw / "out" / "chunk_000.done").exists() and (sw / "out" / "chunk_000.sha256").read_text() == sha256_path(npz)
    assert not (sw / "in" / "chunk_000").exists(), "the frames of a processed chunk must be deleted"
    z = np.load(npz, allow_pickle=False)
    lf = w.td / "chunk0.logs"
    lf.write_text("\n".join(w.logs[:2]) + "\n")
    d = w.fwd(w.variant, "direct_c0", ("--eval-batch", "8", "--token-order-seed", "1", "--logs-file", str(lf)), eps=0.3)
    for k in ("token", "log", "traj", "logits_BASE", f"logits_{STATE}", "ic2_maxabs"):
        assert d["z"][k].tobytes() == z[k].tobytes(), f"serve and run differ at {k}"
    m = json.loads(str(z["meta_json"]))
    assert m["irep"]["eval_batch"] == 8 and m["irep"]["token_order_seed"] == 1 and m["ic5_forward"].get("skipped"), m.get("ic5_forward")
    assert json.load(open(sw / "out" / "ic5_forward.json"))["bit_identical"] is None


@check("chain:expected_subcommand_is_the_workers_hash_and_refuses_a_wrong_key_or_update", "chain")
def _(w):
    harness_world(w)
    out = w.td / "EXP_SUB.json"
    base = [PY, "-I", "-B", "-X", "utf8", str(w.tool), "expected"]
    r = subprocess.run(base + ["--state", f"{STATE}={w.states['X']}:B", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0 and "ZZIREP_EXPECTED_OK" in r.stdout, (r.stdout + r.stderr)[-400:]
    d = json.load(open(out))[STATE]
    assert d["sha256_tensors"] == w.cache["sha_x"] and d["update"] == 1000 and d["key"] == "B"
    assert json.load(open(out))[STATE]["sha256_tensors"] == json.load(open(w.td / "EXPECTED.json"))[STATE]["sha256_tensors"], "not the value the guard compares to"
    r = subprocess.run(base + ["--state", f"{STATE}={w.states['X']}:A", "--out", str(w.td / "x1.json")], capture_output=True, text=True)
    assert r.returncode == 2 and "REFUSED" in r.stdout and not (w.td / "x1.json").exists(), r.stdout[-300:]
    import torch
    ss = torch.load(w.states["X"], map_location="cpu", weights_only=False)
    ss["update"] = 999
    bad = w.td / "scorer_states_update999.pt"
    torch.save(ss, bad)
    r = subprocess.run(base + ["--state", f"{STATE}={bad}:B", "--out", str(w.td / "x2.json")], capture_output=True, text=True)
    assert r.returncode == 2 and "update" in r.stdout and not (w.td / "x2.json").exists(), r.stdout[-300:]


@check("chain:fixture_has_signal_so_the_pipeline_has_something_to_carry", "chain")
def _(w):
    H = harness_world(w)
    flips = sum(H["A"]["picks"][t] != H["B"]["picks"][t] for t in H["A"]["picks"])
    assert flips >= 1 and abs(H["A"]["mean_x100"] - H["B"]["mean_x100"]) > 0, (flips, H["A"]["mean_x100"], H["B"]["mean_x100"])


@check("guard:legit_pair_passes_and_the_value_is_independently_recomputed", "guard")
def _(w):
    H = harness_world(w)
    rc, txt, out, side = run_make(w, H["A"], H["B"], "legit")
    assert rc == 0 and out.exists(), txt[-800:]
    d = json.load(open(out))
    expect = abs(H["A"]["mean_x100"] - H["B"]["mean_x100"])
    assert d["unit"] == "PDMS x100" and d["n_tokens"] == N and abs(d["irep_abs_delta"] - expect) <= 1e-9 and expect > 0, (d["irep_abs_delta"], expect)
    assert side["failed"] == [] and all(v["ok"] for v in side["gates"].values()) and d["irep_guard"]["gates_all_pass"] is True
    assert side["facts"]["pick_changes_a_vs_b"] >= 1 and side["facts"]["logit_maxabs_a_vs_b"] > 0


@check("chain:the_file_passes_the_reader_build_uses", "chain")
def _(w):
    H = harness_world(w)
    rc, txt, out, side = run_make(w, H["A"], H["B"], "legit")
    assert rc == 0, txt[-500:]
    sys.path.insert(0, str(NAVTEST_E.parent))
    sys.path.insert(0, str(NAVTEST_E.parent / "fp"))
    import navtest_E as NE
    r = NE.read_irep(str(out))
    assert r["value"] == json.load(open(out))["irep_abs_delta"] and NE.floors_object(r) == {"irep_abs_delta": r["value"]}


@check("guard:two_different_scorer_states_are_refused", "guard")
def _(w):
    H = harness_world(w)
    rc, txt, out, side = run_make(w, H["A"], H["Bd"], "diffstate")
    assert w.cache["sha_x"] != w.cache["sha_y"]
    assert rc == 2 and not out.exists() and "both_reads_are_one_scorer_state" in side["failed"], (rc, txt[-500:])
    # and what the consumer ALONE would have said about those two tables: a number that looks like an I-rep and is the training-seed contrast
    sys.path.insert(0, str(NAVTEST_E.parent))
    sys.path.insert(0, str(NAVTEST_E.parent / "fp"))
    import navtest_E as NE
    r = NE.irep_from_tables(H["A"]["csv"], H["Bd"]["csv"])
    assert r["unit"] == "PDMS x100" and r["irep_abs_delta"] >= 0, "the consumer alone accepts two different states without complaint"


@check("guard:identical_token_order_seeds_are_refused", "guard")
def _(w):
    H = harness_world(w)
    rc, txt, out, side = run_make(w, H["A"], H["Bs"], "sameseed")
    assert rc == 2 and not out.exists() and "settings_are_the_registered_ones_and_differ" in side["failed"], (rc, txt[-500:])


@check("guard:unregistered_batch_sizes_are_refused", "guard")
def _(w):
    H = harness_world(w)
    rc, txt, out, side = run_make(w, H["A"], H["B"], "badbatch", batch_b=16)
    assert rc == 2 and not out.exists() and "settings_are_the_registered_ones_and_differ" in side["failed"], (rc, txt[-500:])


@check("guard:a_hash_other_than_the_independently_computed_one_is_refused", "guard")
def _(w):
    H = harness_world(w)
    bad = w.td / "EXPECTED_BAD.json"
    json.dump({STATE: {"sha256_tensors": w.cache["sha_y"], "update": 1000, "key": "B"}}, open(bad, "w"))
    rc, txt, out, side = run_make(w, H["A"], H["B"], "badhash", expected_states=str(bad))
    assert rc == 2 and not out.exists() and "state_equals_the_independently_computed_hash" in side["failed"], (rc, txt[-500:])
    rc, txt, out, side = run_make(w, H["A"], H["B"], "nohash", expected_states=None)
    assert rc == 2 and not out.exists() and "state_equals_the_independently_computed_hash" in side["failed"], "a read not bound to an independent hash must be refused"


@check("guard:a_csv_scored_from_another_reads_seam_is_refused", "guard")
def _(w):
    H = harness_world(w)
    cj = Path(H["B"]["csv"]).parent / f"{Path(H['B']['csv']).stem}.counts.json"
    orig = cj.read_text()
    try:
        d = json.loads(orig)
        d["arm"] = "SEAM:" + str(Path(H["A"]["work"]) / f"seam_{STATE}.npz").replace("/", "\\")
        cj.write_text(json.dumps(d))
        rc, txt, out, side = run_make(w, H["A"], H["B"], "wrongseam")
    finally:
        cj.write_text(orig)
    assert rc == 2 and not out.exists() and "read_b_csv_was_scored_from_its_own_seam" in side["failed"], (rc, txt[-500:])


@check("guard:a_failed_harness_status_or_a_wrong_token_count_is_refused", "guard")
def _(w):
    H = harness_world(w)
    rc, txt, out, side = run_make(w, H["A"], H["B"], "ntok", n_tokens=N - 1)
    assert rc == 2 and not out.exists() and "same_tokens_per_chunk_and_the_expected_count" in side["failed"], (rc, txt[-500:])
    sp = Path(H["A"]["work"]) / "score_logs" / f"refe_sobj_irepA_{STATE}.status.json"
    orig = sp.read_text()
    try:
        d = json.loads(orig)
        d["status"] = "FAIL"
        sp.write_text(json.dumps(d))
        rc, txt, out, side = run_make(w, H["A"], H["B"], "badstatus")
    finally:
        sp.write_text(orig)
    assert rc == 2 and not out.exists() and "read_a_csv_was_scored_from_its_own_seam" in side["failed"], (rc, txt[-500:])


@check("guard:parses_the_real_harness_artifacts_of_the_free_point_dry_run", "real")
def _(w):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("irep_tool_inproc", TOOL)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    toks = {str(t) for t in np.load(f"{DRY}/seam_C0A.npz", allow_pickle=False)["token"]}
    csvp = f"{DRY_SCORE}/refe_fpdry_C0A/refe_fpdry_C0A.csv"
    ok, det = T.harness_side(DRY, csvp, f"{DRY}/seam_C0A.npz", toks, "FREEPOINT_C0A")
    assert ok and det["harness_status"] == "PASS" and det["csv_rows"] == 2, det
    assert not T.harness_side(DRY, csvp, f"{DRY}/seam_C0A.npz", toks, "SOBJE_C0A")[0], "a wrong arm string must fail"
    assert not T.harness_side(DRY, f"{DRY_SCORE}/refe_fpdry_BASE/refe_fpdry_BASE.csv", f"{DRY}/seam_C0A.npz", toks, "FREEPOINT_C0A")[0], "BASE's counts.json names BASE's seam, not C0A's"


@check("real:i_rep_seam_step_reproduces_the_live_pickers_seam_bit_for_bit", "real")
def _(w):
    d = w.td / "dry_chunks"
    d.mkdir(exist_ok=True)
    shutil.copy(f"{DRY}/out/chunk_000.npz", d / "chunk_000.npz")
    work = w.td / "dry_work"
    cmd = [PY, "-I", "-B", "-X", "utf8", str(w.tool), "seams", "--chunks", str(d), "--work", str(work), "--state", "R0B", "--present-only", "--allow-plain-chunks"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
    assert r.returncode == 0 and "ZZIREP_SEAMS_OK R0B 2 tokens" in r.stdout, (r.stdout + r.stderr)[-800:]
    for k in ("BASE", "R0B"):
        mine, live = np.load(work / f"seam_{k}.npz", allow_pickle=False), np.load(f"{DRY}/seam_{k}.npz", allow_pickle=False)
        assert [str(t) for t in mine["token"]] == [str(t) for t in live["token"]] and np.array_equal(mine["poses"], live["poses"]), k
    pts = np.load(work / "points_E.npz", allow_pickle=False)
    assert "pick_R0B" in pts.files
    # and the same call without the test-only flag refuses a chunk that is not an I-rep read
    r2 = subprocess.run([c for c in cmd if c != "--allow-plain-chunks"], capture_output=True, text=True, cwd=str(HERE))
    assert r2.returncode == 2 and "not an I-rep read" in r2.stdout, r2.stdout[-400:]


# ------------------------------------------------------------------------------------------------ mutations
def vmut(old, new):
    return ("variant", old, new)


def tmut(old, new):
    return ("tool", old, new)


ALIGN = "batch_order:rows_are_per_token_aligned_to_the_same_token_ids"
SIZES = "batch_order:forward_ran_at_the_requested_batch_sizes"
PERM = "batch_order:processing_order_is_the_seeded_permutation"
META = "batch_order:meta_records_the_run_and_ic2_is_zero_per_token"
B1 = "identity:batch_1_with_a_shuffled_order_reproduces_the_original_bit_for_bit"
DEF_FILE = "identity:defaults_file_byte_identical"
DEF_ARR = "identity:defaults_arrays_and_meta_equal"
DEF_OUT = "identity:defaults_return_code_and_stdout_identical"
DEF_CALLS = "identity:default_forward_calls_are_the_planners_batch_1_in_sorted_order"
DIFFSTATE = "guard:two_different_scorer_states_are_refused"
SAMESEED = "guard:identical_token_order_seeds_are_refused"
BADHASH = "guard:a_hash_other_than_the_independently_computed_one_is_refused"
LEGIT = "guard:legit_pair_passes_and_the_value_is_independently_recomputed"

# (name, kind, old, new, groups to run, checks that MUST go red, checks that MUST stay green)
MUTATIONS = [
    ("order_not_restored", *vmut("        ix = np.asarray(idx, dtype=np.int64)\n", "        ix = np.arange(done, done + len(idx), dtype=np.int64)\n"),
     ("identity", "batch_order"), {ALIGN, B1}, {SIZES, PERM, DEF_FILE, DEF_ARR}),
    ("batch_option_ignored_but_still_reported", *vmut("    for s in range(0, n, bs):\n        idx = [int(j) for j in order[s:s + bs]]", "    for s in range(0, n, 1):\n        idx = [int(j) for j in order[s:s + 1]]"),
     ("identity", "batch_order"), {SIZES}, {ALIGN, PERM, DEF_FILE, DEF_ARR, B1}),
    ("shuffle_ignored", *vmut("    order = np.arange(n, dtype=np.int64) if seed is None else np.random.RandomState(int(seed)).permutation(n).astype(np.int64)    # RandomState: the frozen legacy stream\n",
                              "    order = np.arange(n, dtype=np.int64)\n"), ("identity", "batch_order"), {PERM, META}, {ALIGN, SIZES, DEF_FILE, DEF_ARR}),
    ("one_rig_for_the_whole_batch", *vmut("cals.append(torch.tensor(calib[r[\"log\"]], dtype=torch.float64))", "cals.append(torch.tensor(calib[inputs[tokens[idx[0]]][\"log\"]], dtype=torch.float64))"),
     ("identity", "batch_order"), {ALIGN}, {SIZES, PERM, DEF_FILE, DEF_ARR, B1}),
    ("default_path_changed_output_perturbed", *vmut("            logits_o[nm][i] = outs[nm][0].float().cpu().numpy()\n", "            logits_o[nm][i] = outs[nm][0].float().cpu().numpy() + np.float32(1e-7)\n"),
     ("identity", "batch_order"), {DEF_FILE, DEF_ARR}, {ALIGN, SIZES, PERM, B1}),
    ("default_path_changed_default_seed_is_0", *vmut('ap.add_argument("--token-order-seed", type=int, default=None,', 'ap.add_argument("--token-order-seed", type=int, default=0,'),
     ("identity", "batch_order"), {DEF_FILE, DEF_ARR, DEF_CALLS}, {ALIGN, SIZES, PERM}),
    ("default_path_changed_default_batch_is_2", *vmut('ap.add_argument("--eval-batch", type=int, default=1,', 'ap.add_argument("--eval-batch", type=int, default=2,'),
     ("identity", "batch_order"), {DEF_FILE, DEF_ARR, DEF_CALLS}, {ALIGN, SIZES, PERM}),
    ("irep_from_two_different_states", *tmut("    same_state = (len(sa) == 1 and sa == sb and sa[0] is not None)\n", "    same_state = (len(sa) == 1 and len(sb) == 1 and sa[0] is not None)\n"),
     ("guard",), {DIFFSTATE}, {LEGIT, SAMESEED}),
    ("identical_seeds_accepted_both_layers_removed", *tmut('and seeds_a != seeds_b and all((x.get("order_sha256") != y.get("order_sha256")) or x.get("n", 0) < 2 for x, y in zip(ra, rb))', ""),
     ("guard",), {SAMESEED}, {LEGIT, DIFFSTATE}),
    ("expected_hash_not_compared", *tmut('gate("state_equals_the_independently_computed_hash", bool(sa) and sa == [ex.get("sha256_tensors")],', 'gate("state_equals_the_independently_computed_hash", True,'),
     ("guard",), {BADHASH}, {LEGIT, DIFFSTATE}),
]


def checks_in(groups):
    return [n for n, (g, _) in CHECKS.items() if g in groups]


def run_checks(w: World, names) -> dict:
    res = {}
    for n in names:
        t0 = time.time()
        try:
            CHECKS[n][1](w)
            res[n] = None
        except AssertionError as e:
            res[n] = f"{type(e).__name__}: {str(e)[:300]}"
        except Exception as e:
            res[n] = f"{type(e).__name__} (check crashed): {str(e)[:300]}"
        print(f"  {'PASS' if res[n] is None else 'FAIL'}  {n}  [{time.time() - t0:.0f}s]" + ("" if res[n] is None else f"   <- {res[n]}"), flush=True)
    return res


def mutated_world(base: World, kind: str, old: str, new: str, name: str) -> World:
    src_path = VARIANT if kind == "variant" else TOOL
    txt = src_path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert txt.count(old) == 1, f"mutation {name}: `old` occurs {txt.count(old)} times in {src_path.name}"
    mut = txt.replace(old, new)
    assert mut != txt
    if kind == "variant":
        p = base.td / f"mut_{name}_fp_forward_irep.py"
        p.write_text(mut, encoding="utf-8", newline="\n")
        return World(base.td, variant=p, tool=TOOL, shared=base)
    p = HERE / f"_mut_{os.getpid()}_{name}_irep_tool.py"                   # next to the real tool, so its HERE (and the path to navtest_E.py) is the same
    p.write_text(mut, encoding="utf-8", newline="\n")
    return World(base.td, variant=VARIANT, tool=p, shared=base)


def main() -> int:
    t_all = time.time()
    keep = "--keep" in sys.argv
    only = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), None)
    groups_all = sorted({g for g, _ in CHECKS.values()})
    sel = set(only.split(",")) if only else set(groups_all)
    names = checks_in(sel)
    # 0. pin the files that must not change
    ro = {str(p): sha256_path(p) for p in (ORIG, NAVTEST_E, PKG / "outcome" / "safeobj_outcome.py")}
    print(f"original fp_forward.py md5 {hashlib.md5(ORIG.read_bytes()).hexdigest()}; variant sha256 {sha256_path(VARIANT)[:16]}; irep_tool sha256 {sha256_path(TOOL)[:16]}; "
          f"{len(names)} checks, {len(MUTATIONS) if '--no-mutations' not in sys.argv else 0} RED mutations", flush=True)
    td = Path(tempfile.mkdtemp(prefix="_tmp_", dir=str(HERE)))
    n_pass = 0
    try:
        w = World(td)
        print("fixture built:", td.name, flush=True)
        # 1. the unmutated suite
        res = run_checks(w, names)
        bad = {k: v for k, v in res.items() if v is not None}
        print(f"unmutated suite: {len(res) - len(bad)}/{len(res)} checks green", flush=True)
        assert not bad, f"the UNMUTATED package fails {len(bad)} check(s): {sorted(bad)}"
        n_pass += 1
        # 2. the mutations
        if "--no-mutations" not in sys.argv:
            for name, kind, old, new, groups, must_red, must_green in MUTATIONS:
                t0 = time.time()
                mw = mutated_world(w, kind, old, new, name)
                try:
                    r = run_checks(mw, checks_in(set(groups)))
                finally:
                    if kind == "tool":
                        mw.tool.unlink(missing_ok=True)
                red = {k for k, v in r.items() if v is not None}
                missing = must_red - red
                wrongly = must_green & red
                assert red, f"mutation {name!r} stayed GREEN"
                assert not missing, f"mutation {name!r} went RED but NOT at {sorted(missing)}; got {sorted(red)}"
                assert not wrongly, f"mutation {name!r} went RED at controls that must stay green: {sorted(wrongly)}"
                print(f"  RED  {name}  ->  {len(red)} check(s): {', '.join(sorted(red))}  [{time.time() - t0:.0f}s]", flush=True)
                n_pass += 1
        # 3. nothing read-only changed
        for p, h in ro.items():
            assert sha256_path(p) == h, f"READ-ONLY FILE CHANGED: {p}"
        print(f"ZZTEST_PASS {n_pass}  (1 unmutated suite of {len(names)} checks, {0 if '--no-mutations' in sys.argv else len(MUTATIONS)} RED mutations, 3 read-only files unchanged)  total {time.time() - t_all:.0f} s")
        return 0
    finally:
        for p in HERE.glob(f"_mut_{os.getpid()}_*"):
            p.unlink(missing_ok=True)
        if not keep:
            shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
