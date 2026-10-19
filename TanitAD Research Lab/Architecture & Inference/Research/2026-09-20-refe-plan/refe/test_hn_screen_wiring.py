#!/usr/bin/env python3
"""Tests for the WIRING of --hn-bank / --ctx-cache into refe/scorer_finetune.py (P6-HN-S checklist item 3; raw/2026-10-18-p6hn-prereg/PREREG_P6HN_SCREEN.md sections 2.4, 3, 4.3, 4.4, 5).
Run:  PYTHONIOENCODING=utf-8 python refe/test_hn_screen_wiring.py   ->   ZZTEST_PASS <n>   (prints the mutation table).   CPU only, no pod, no checkpoint, no nuPlan, no real data.

WHY THIS FILE EXISTS (CLAUDE.md "built, tested, and unreachable from its caller"): test_hn_bank.py proves the MODULE (the masked two-call term) and test_ctx_cache.py proves the cache; neither proves the
trainer CALLS them.  Here the REAL main() of scorer_finetune.py runs on CPU, preflight and training, against stubbed heavy parts (the stub REFe is built from the REAL CrossBlock and the REAL
REFe.score_trajectories; train.py's banks are replaced by seeded fixtures; the cache and the hard-negative bank are minted by the REAL ctx_cache.py / hn_bank.py).  So a typo in glue that no module test
executes shows here, not on the pod.
  PIN   the P6-HN-S layer reverses to the bcb9a1f tip blob exactly (and, through carvefeed_layer, to B8's pre-B8 blob); the pre-edit trainer is stored here (sha256-pinned) and equals that reversal; the only
        new parser dests are hn_bank and ctx_cache, default ""; every nested function of the pre-edit main() is AST-identical in the new one (the loop, evaluate(), fwd(), train_losses() are untouched).
  I3    both flags absent = the pre-edit trainer.  The REAL main() of the pre-edit source and of the new source, run twice, on the plain model and on the lane-head (7-output) model: every logged row
        (timestamps stripped) and every saved tensor of arms A and B is equal bit for bit.  Controls: the pre-edit source twice (the harness is deterministic), and a run with --lr moved (the comparison sees a change).
  ON    the cache alone is bit-identical to the live trunk (A, B, every eval row); the bank moves arm B ONLY (arm A's tensors and A-side eval numbers equal the cache-only run; B differs); the live-trunk
        bank run equals the cached bank run; a bank whose mask is forced empty leaves B == A exactly.  The section 4.3 gradient table is read from the RAW G14 fields of the preflight (never from its `ok`):
        non-zero on score_q_mlp / score_dec / score_head rows 0-5, EXACTLY 0.0 on the lane row (the control that proves the probe can read zero), on every arm-A parameter and on the base; the bank-off and
        bank-empty controls read exactly 0.0; two optimiser steps with the bank off / empty leave B bit-identical to A and with it on do not; the fan rows of the masked call equal the one-call fan logits bit
        for bit; pad slots cannot reach the term; the rows that entered the term equal the rows the bank holds (D1); held-out items carry no bank (I4).  G15's raw fields: ctx / base / A / B logits equal
        live vs cache bit for bit, a row-shuffled cache is far from it, the manifest binds the checkpoint's trunk fingerprint and the calib file.
  REF   REFUSED (rc 4, before the first step): --hn-bank with --epo / --lam != 0 / --b-mode != listnet / --aux-drv, --ctx-cache with --aux-drv, a bank of another partition / label_version / fan fingerprint /
        bytes, a bank of 0 hard negatives, a training set labelled at another version, a cache of another trunk, a cache that lacks a scene.
Each of the 7 mutations of section 4.3 edits SOURCE TEXT (the snippet must occur exactly once) and must turn >= 1 EXPECTED check RED.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import types

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
import carvefeed_layer as CL  # noqa: E402
import ckpt_io as REAL_CKPT_IO  # noqa: E402
import ctx_cache as CC0  # noqa: E402
import hn_bank as HN0  # noqa: E402
import hnscreen_layer as HL  # noqa: E402
import model as M  # noqa: E402
import test_hn_bank as TH  # noqa: E402

SRC_PATH = os.path.join(HERE, "scorer_finetune.py")
HN_PATH = os.path.join(HERE, "hn_bank.py")
CC_PATH = os.path.join(HERE, "ctx_cache.py")
PRE_PATH = os.path.join(HERE, "fixtures", "scorer_finetune_pre_hn.py.txt")
PRE_SHA256 = "1f7d23740b6be0f2239686416c015038cd5e328c8078442bccd1fbfebbf610f7"      # the bcb9a1f tip blob of scorer_finetune.py (e357ede0...) as text
TIP_BLOB = "e357ede0f538caf7f7bf68278c7382127a8eb96f"
PRE_B8_BLOB = "869b186293150dc6c07253391aeed9f84738225f"

NT, NH, K_PROP, T_STEPS, P_TOK, D_W, CKPT = 24, 12, 8, 4, 8, 16, 5                    # train sets, held-out sets, proposals, horizon, ctx tokens, width, checkpoint step of the fan
TOP = "T"                                                                              # train log prefix; held logs are H0..H2


# ------------------------------------------------------------------------------------------------ helpers
NCHECK = [0]


def check(bad, name, fn):
    NCHECK[0] += 1
    try:
        ok = bool(fn())
    except (Exception, SystemExit) as e:                             # noqa: BLE001 -- an exception IS a failed check
        ok = False
        if os.environ.get("HNW_DEBUG"):
            print("  [check raised]", name, type(e).__name__, str(e)[:300])
    if not ok:
        bad.append(name)


def blob_sha1(text: str) -> str:
    b = text.encode("utf-8")
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def read_text(p: str) -> str:
    with open(p, "rb") as f:
        return f.read().decode("utf-8")


def load_mod(src: str, name: str, path: str):
    m = types.ModuleType(name)
    m.__file__ = path
    sys.modules[name] = m
    exec(compile(src, path, "exec"), m.__dict__)                       # noqa: S102 -- the module under test, possibly mutated
    return m


def once(src: str, snippet: str, repl: str) -> str:
    assert src.count(snippet) == 1, f"mutation snippet must occur exactly once ({src.count(snippet)}): {snippet[:80]!r}"
    return src.replace(snippet, repl)


def same_sd(x, y) -> bool:
    return x is not None and y is not None and list(x) == list(y) and all(torch.equal(x[k], y[k]) for k in x)


TIMING = ("at", "hours", "model_A", "model_B")                   # wall-clock fields and the run-directory paths of the saved checkpoints


def norm(obj):
    """a logged row with the wall-clock fields removed (everything else must be equal)"""
    if isinstance(obj, dict):
        return {k: norm(v) for k, v in obj.items() if k not in TIMING and not k.startswith("s_per") and not k.startswith("est_hours") and not k.startswith("gpu_mem")}
    if isinstance(obj, (list, tuple)):
        return [norm(v) for v in obj]
    return obj


def rows_equal(a, b) -> bool:
    return json.dumps(norm(a), sort_keys=True, default=str) == json.dumps(norm(b), sort_keys=True, default=str)


# ------------------------------------------------------------------------------------------------ the stubs (the heavy parts only)
class StubScene(nn.Module):
    """scene_proj: a Linear whose OUTPUT is bf16 (the dtype the live path hands the scoring decoder under the trainer's autocast, and the dtype the cache stores)"""

    def __init__(self, d_in, d):
        super().__init__()
        self.lin = nn.Linear(d_in, d)

    def forward(self, x):
        return self.lin(x).to(torch.bfloat16)


class ManualMHA(nn.MultiheadAttention):
    """nn.MultiheadAttention's parameters and call signature, attention written out with plain matmuls: torch's fused attention rounds differently for weights with requires_grad False (the trainer's frozen
    `base` copy) than for requires_grad True (arm A) on CPU, which would make G3 (base == A bit for bit at step 0) fail on THIS harness for the pre-edit trainer too (measured: 1.8e-7).  `attn_mask` is the
    bool convention (True = not allowed) that hn_bank.hn_attn_mask uses.  test_hn_bank.py runs the term against the REAL nn.MultiheadAttention; this file proves the WIRING."""

    def forward(self, q, k, v, attn_mask=None, need_weights=False, **kw):
        E, Hh = self.embed_dim, self.num_heads
        if self._qkv_same_embed_dim:
            wq, wk, wv = self.in_proj_weight.chunk(3)
        else:
            wq, wk, wv = self.q_proj_weight, self.k_proj_weight, self.v_proj_weight
        bq, bk, bv = self.in_proj_bias.chunk(3)
        Q, Kx, V = torch.nn.functional.linear(q, wq, bq), torch.nn.functional.linear(k, wk, bk), torch.nn.functional.linear(v, wv, bv)
        B, Lq, Lk = Q.shape[0], Q.shape[1], Kx.shape[1]
        Q, Kx, V = (t.view(B, -1, Hh, E // Hh).transpose(1, 2) for t in (Q, Kx, V))
        att = (Q @ Kx.transpose(-1, -2)) / ((E // Hh) ** 0.5)
        if attn_mask is not None:
            att = att.masked_fill(attn_mask, float("-inf"))
        out = (att.softmax(-1) @ V).transpose(1, 2).reshape(B, Lq, E)
        return self.out_proj(out), None


class StubCross(M.CrossBlock):
    """the REAL CrossBlock structure (n1 / self_attn / n2 / cross / n3 / mlp, forward inherited); its attention modules are ManualMHA; the bf16 context is cast to the query dtype (what autocast does on the pod)"""

    def __init__(self, dim, heads, ctx):
        super().__init__(dim, heads, ctx)
        self.self_attn = ManualMHA(dim, heads, batch_first=True)
        self.cross = ManualMHA(dim, heads, kdim=ctx, vdim=ctx, batch_first=True)

    def forward(self, q, ctx):
        return super().forward(q, ctx.to(q.dtype))


class WStubREFe(nn.Module):
    score_trajectories = M.REFe.score_trajectories                       # the REAL function, an unbound callable (ScorerCopy calls it with itself)

    def __init__(self, cfg):
        super().__init__()
        self.trunk = nn.Linear(3, D_W)
        self.scene_proj = StubScene(D_W, D_W)
        self.traj_head = nn.Linear(D_W, K_PROP * T_STEPS * 3)
        self.score_q_mlp = nn.Linear(T_STEPS * 3, D_W)
        self.score_dec = nn.ModuleList([StubCross(D_W, 2, D_W) for _ in range(2)])
        self.score_head = nn.Linear(D_W, cfg.n_score_components)

    def forward(self, img, ego, goal, detach_scorer=True, calib=None, score_extra=None):
        feat = self.trunk(img)
        _scene = self.scene_proj(feat)                                    # call 1 of scene_proj (scene_ctx), as REFe.forward
        visual = self.scene_proj(feat * 0.5 + 1.0)                        # call 2 (visual_ctx): the tensor the scoring decoder reads
        traj = self.traj_head(feat.mean(1)).view(-1, K_PROP, T_STEPS, 3)
        sctx = visual.detach()
        return traj, None, self.score_trajectories(score_extra if score_extra is not None else traj.detach(), sctx)


class StubCfgFactory:
    @staticmethod
    def for_backbone(_b):
        return types.SimpleNamespace(n_score_components=6, nav_mode="goal", n_proposals=K_PROP, horizon_steps=T_STEPS)


def keys_of(train: bool):
    n = NT if train else NH
    return [((f"{TOP}{i // 4}" if train else f"H{i // 4}"), f"tok{i}", i, 0) for i in range(n)]


def scene_tensors(i: int):
    """the (img [P, 3], cxy [K, T, 3]) of row i, drawn in order from one seeded generator (the dataset and the cache builder both call this)"""
    g = torch.Generator().manual_seed(1000 + i)
    return torch.randn(P_TOK, 3, generator=g), torch.randn(K_PROP, T_STEPS, 3, generator=g)


class WStubBank:
    LV = 5                                                               # the label_version of every set (a test sets 4 to prove the screen refuses it)

    def __init__(self, path, n_prop, horizon, lane_labels=None, pdm_labels=None, teacher_lane_labels=None):
        train = "train" in path
        g = np.random.default_rng(1 if train else 2)
        self.by = {}
        for k in keys_of(train):
            tg = g.random((n_prop, 6)).astype(np.float32)
            tg[:, :2] = (tg[:, :2] > 0.3)
            self.by[k] = (CKPT, None, tg, (CKPT, self.LV), False)
        self.pdm_keys = set(self.by) if pdm_labels else set()
        self.n_pdm, self.n_pdm_stale, self.n_pdm_superseded = len(self.pdm_keys), 0, 0
        self.n_navsim_dac = self.n_lane = self.n_lane_stale = self.n_lane_superseded = 0
        self.lane_keep: dict = {}
        self.teacher_lane = {k: g.random(n_prop).astype(np.float32) for k in list(self.by)[::2]} if teacher_lane_labels else {}
        self.n_tlane, self.n_cols = len(self.teacher_lane), (7 if teacher_lane_labels else 6)


class WStubTargets(torch.utils.data.Dataset):
    def __init__(self, targets, images, cfg, calib=None, onpolicy=None):
        self.onpolicy = onpolicy
        self.rows = [{"log_name": k[0], "token": k[1], "step": k[2], "rank": k[3]} for k in keys_of(True) + keys_of(False)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        k = (r["log_name"], r["token"], int(r["step"]), int(r["rank"]))
        e = self.onpolicy.by[k]
        img, cxy = scene_tensors(i)
        tg = torch.as_tensor(e[2])
        if self.onpolicy.n_cols == 7:
            lane = self.onpolicy.teacher_lane.get(k)
            tg = torch.cat([tg, torch.as_tensor(lane if lane is not None else np.full(K_PROP, -1.0, np.float32))[:, None]], -1)
        return (img, torch.zeros(4), torch.zeros(2), torch.zeros(1), cxy, tg, torch.ones(K_PROP), torch.full((K_PROP,), CKPT), torch.zeros(0))


class StubCkptIO:
    FORMAT_FULL = "full"
    config_for_checkpoint = staticmethod(REAL_CKPT_IO.config_for_checkpoint)

    @staticmethod
    def load_for_inference(model, path, map_location="cpu", backbone=None, meta_out=None):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"], strict=True)
        if meta_out is not None:
            meta_out.update(sd.get("meta", {}) or {})
        return "full"

    @staticmethod
    def trainable_names(model):
        return {n for n, p in model.named_parameters() if p.requires_grad}

    @staticmethod
    def atomic_save(obj, path):
        torch.save(obj, path)


# ------------------------------------------------------------------------------------------------ the rig: checkpoint, cache, bank
def anchors_for(train_keys):
    """one anchor per training scene: a straight 6 m path at lateral offset y0 (hard negatives under the fake labeller: 2, 2, 0 (STATIONARY), 3, 4 per scene, cycling)"""
    pats = [0.0, 0.2, None, -0.3, 0.4]
    out = []
    for i, k in enumerate(train_keys):
        y0 = pats[i % 5]
        n = np.arange(1, T_STEPS + 1)
        x = (1.5 if y0 is not None else 0.4) * n                         # 6.0 m, or 1.6 m = STATIONARY (< 2.0 m)
        y = np.full(T_STEPS, 0.0 if y0 is None else y0)
        out.append({"log_name": k[0], "token": k[1], "step": k[2], "rank": k[3], "ckpt_step": CKPT, "traj": np.stack([x, y], -1).tolist(), "yaw": [0.0] * T_STEPS})
    return out


def expected_hn_counts(n=NT):
    return [[2, 2, 0, 3, 4][i % 5] for i in range(n)]


class Rig:
    """one checkpoint (plain 6-output head, or the lane-head 7-output model), its calib file, a GOOD cache and a GOOD bank; further caches / banks are minted on demand"""

    def __init__(self, td: str, n7: bool):
        self.td, self.n7 = td, n7
        self.name = "n7" if n7 else "plain"
        self.train_keys, self.held_keys = keys_of(True), keys_of(False)
        torch.manual_seed(0)
        self.cfg = types.SimpleNamespace(n_score_components=7 if n7 else 6)
        self.model = WStubREFe(self.cfg)
        self.ck = os.path.join(td, f"ck_{self.name}.pt")
        torch.save({"format": "full", "model": self.model.state_dict(), "meta": {"n_score_components": 7} if n7 else {}}, self.ck)
        self.ho_logs = os.path.join(td, "ho_logs.txt")
        with open(self.ho_logs, "w") as fh:
            fh.write(" ".join(f"H{i}" for i in range(4)))
        self.calib = os.path.join(td, "calib.bin")
        with open(self.calib, "wb") as fh:
            fh.write(b"calib-fixture-bytes")
        self._cache, self._bank = None, None

    @property
    def trunk_fp(self):
        return CC0.trunk_fingerprint(self.model.state_dict())

    def build_cache(self, name, *, drop_log=None, perturb_trunk=False, units_logs=None):
        out = os.path.join(self.td, f"cache_{self.name}_{name}")
        model = WStubREFe(self.cfg)
        model.load_state_dict(torch.load(self.ck, weights_only=False)["model"])
        model.eval()
        if perturb_trunk:
            with torch.no_grad():
                model.trunk.weight.add_(0.01)                              # a cache of ANOTHER trunk
        allk = self.train_keys + self.held_keys
        idx = {k: i for i, k in enumerate(allk)}
        logs = sorted({k[0] for k in allk})
        units = [(f"u_{lg}", [k for k in allk if k[0] == lg]) for lg in logs if lg != drop_log]

        def load_batch(keys):
            items = [scene_tensors(idx[tuple(k)]) for k in keys]
            return (torch.stack([x[0] for x in items]), torch.zeros(len(keys), 4), torch.zeros(len(keys), 2), None, torch.stack([x[1] for x in items]))

        def forward(b):
            return model(b[0], b[1], b[2], score_extra=b[4])[0]
        fp = CC0.trunk_fingerprint(model.state_dict())
        CC0.build_units(model, units, load_batch, forward, out, trunk_fp=fp, device="cpu", batch=4, log=lambda *a, **k: None)
        CC0.finalize(out, units, trunk_fp=fp, calib_sha256=CC0.sha256_file(self.calib))
        return out

    def mint(self, name, *, partition_id=None, label_version=5, fan_fp=None, anchors=None):
        path = os.path.join(self.td, f"bank_{name}.hnb")
        anchors = anchors if anchors is not None else anchors_for(self.train_keys)
        fan_fp = fan_fp or HN0.keys_fingerprint([k + (CKPT,) for k in self.train_keys])
        HN0.mint_bank(anchors, path, partition_id=partition_id or HNW["partition"], fan_fingerprint=fan_fp, label_fn=TH.fake_label_fn, label_version=label_version)
        return path

    @property
    def cache(self):
        if self._cache is None:
            self._cache = self.build_cache("good")
        return self._cache

    @property
    def bank(self):
        if self._bank is None:
            self._bank = self.mint("good")
        return self._bank


HNW: dict = {}                                                         # filled by main(): {"partition": SF.HN_PARTITION_ID}


# ------------------------------------------------------------------------------------------------ the REAL main(), on CPU
RUNS: dict = {}


def patch_src(src: str) -> str:
    c = src.replace('dev = "cuda"', 'dev = "cpu"').replace("pin_memory=True", "pin_memory=False").replace("num_workers=2", "num_workers=0").replace("n=10000, seed=20260927", "n=40, seed=20260927")
    return c.replace('torch.autocast("cuda", dtype=torch.bfloat16)', 'torch.autocast("cpu", enabled=False)')


@contextlib.contextmanager
def swapped(hn_src=None, cc_src=None, empty_term=False):
    """the modules main() imports at call time (`import hn_bank as HN`, `import ctx_cache as CC`) are replaced for one run -- by a MUTATED source, or (empty_term) by a copy whose hn_term sees an all-False mask"""
    saved = {n: sys.modules.get(n) for n in ("hn_bank", "ctx_cache")}
    try:
        if hn_src is not None or empty_term:
            m = load_mod(hn_src if hn_src is not None else read_text(HN_PATH), "hn_bank", HN_PATH)
            if empty_term:
                orig = m.hn_term
                m.hn_term = lambda scorer, fan_xy, hn_xy, hn_tg, hn_m, sctx: orig(scorer, fan_xy, hn_xy, hn_tg, torch.zeros_like(hn_m), sctx)
        else:
            sys.modules["hn_bank"] = HN0
        if cc_src is not None:
            load_mod(cc_src, "ctx_cache", CC_PATH)
        else:
            sys.modules["ctx_cache"] = CC0
        yield
    finally:
        for n, v in saved.items():
            if v is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v


def wrun(sf_src: str, rig: Rig, tag: str, *, extra=(), phases=("pre", "run"), hn_src=None, cc_src=None, empty_term=False, lr="1e-3", nonce="", pass_json=None) -> dict:
    """run the real main() (preflight, then train) on CPU; -> {rc_pre, rc_run, out_pre, out_run, rows (both logs), A, B (the saved scorer state dicts), pass_json}"""
    sig = (hashlib.sha1(sf_src.encode()).hexdigest()[:12], rig.name, tuple(extra), tuple(phases), hashlib.sha1((hn_src or "").encode()).hexdigest()[:8],
           hashlib.sha1((cc_src or "").encode()).hexdigest()[:8], empty_term, lr, nonce, pass_json is not None)
    if sig in RUNS:
        return RUNS[sig]
    sm = load_mod(patch_src(sf_src), f"sft_wire_{tag}", SRC_PATH)
    sm.REFe, sm.REFeConfig, sm.ckpt_io, sm.T = WStubREFe, StubCfgFactory, StubCkptIO, types.SimpleNamespace(OnPolicyBank=WStubBank, TargetBank=WStubTargets)
    td = os.path.join(rig.td, "run_" + tag + (("_" + nonce) if nonce else ""))
    os.makedirs(td, exist_ok=True)
    out = os.path.join(td, "out")
    if pass_json is not None:                       # a run phase whose preflight is someone else's (the empty-term control kills G14 on purpose)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "PREFLIGHT_PASS.json"), "w") as fh:
            json.dump(pass_json, fh)
    args = ["--ckpt", rig.ck, "--targets", "t", "--images", "i", "--calib", rig.calib, "--onpolicy", "train", "--heldout", "held", "--heldout-logs", rig.ho_logs, "--out", out, "--batch", "2",
            "--accum", "2", "--workers", "0", "--epochs", "5", "--max-updates", "25", "--eval-every", "12", "--warmup", "2", "--lr", lr, "--b-mode", "listnet", "--lam", "0", "--pdm-labels-train", "d",
            "--pdm-labels-heldout", "d", "--require-pdm"]
    if rig.n7:
        args += ["--lane-head", "--teacher-lane-train", "x", "--teacher-lane-heldout", "y"]
    args += list(extra)
    saved_argv, saved_cuda = sys.argv, (torch.cuda.synchronize, torch.cuda.max_memory_allocated)
    torch.cuda.synchronize, torch.cuda.max_memory_allocated = (lambda *a, **k: None), (lambda *a, **k: 0)
    res: dict = {"rc_pre": None, "rc_run": None, "out_pre": "", "out_run": ""}
    try:
        with swapped(hn_src, cc_src, empty_term):
            for ph in phases:
                sys.argv = ["scorer_finetune.py"] + args + (["--preflight"] if ph == "pre" else [])
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = sm.main()
                res["rc_" + ph], res["out_" + ph] = rc, buf.getvalue()
                if rc != 0:
                    break
    finally:
        sys.argv = saved_argv
        torch.cuda.synchronize, torch.cuda.max_memory_allocated = saved_cuda
    rows = {}
    for nm in ("preflight.jsonl", "sft.jsonl"):
        p = os.path.join(out, nm)
        rows[nm] = [json.loads(x) for x in open(p, encoding="utf-8")] if os.path.exists(p) else []
    res["rows"] = rows
    pf = os.path.join(out, "PREFLIGHT_PASS.json")
    res["pass_json"] = json.load(open(pf)) if os.path.exists(pf) else None
    pre_rows = [r for r in rows["preflight.jsonl"] if r.get("event") == "preflight"]
    res["gates"] = pre_rows[-1]["gates"] if pre_rows else None
    res["data"] = next((r for r in rows["sft.jsonl"] + rows["preflight.jsonl"] if r.get("event") == "data"), None)
    for arm in ("A", "B"):
        p = os.path.join(out, f"model_sft_{arm}.pt")
        res[arm] = torch.load(p, weights_only=False)["model"] if os.path.exists(p) else None
    RUNS[sig] = res
    return res


def refused(res, text=None) -> bool:
    out = res["out_pre"] + res["out_run"]
    return res["rc_pre"] == 4 and "REFUSED" in out and (text is None or text in out)




# ------------------------------------------------------------------------------------------------ the judges of the RAW G14 / G15 fields (never the trainer's own `ok`)
def g14_bad(g) -> list:
    b: list = []

    def c(n, f):
        check(b, "G14_" + n, f)
    gs = g["grad_abs_sum"]
    c("n_hn_positive", lambda: g["n_hn"] > 0)
    c("grads_finite", lambda: g["grads_finite"] is True)
    c("value_on_positive", lambda: g["value_on"] > 0.0)
    c("grad_q_mlp_nonzero", lambda: gs["score_q_mlp"] > 0.0)
    c("grad_dec_nonzero", lambda: gs["score_dec"] > 0.0)
    c("grad_head_rows_0_5_nonzero", lambda: gs["score_head_rows_0_5"] > 0.0)
    c("lane_head_present", lambda: g["lane_head"] is True)                                    # the exact-zero control reads zero for a reason only if there IS a lane row
    c("lane_row_exactly_zero", lambda: gs["score_head_row_6_lane"] == 0.0)
    c("armA_exactly_zero", lambda: gs["armA_all"] == 0.0)
    c("base_and_frozen_no_grad", lambda: gs["base_params_with_grad"] == 0 and gs["frozen_params_with_grad"] == 0 and g["trunk_requires_grad"] is False)
    c("armA_bce_control_nonzero", lambda: len(g["A_bce_control"]) == 3 and all(v > 0.0 for v in g["A_bce_control"].values()))
    c("controls_exactly_zero", lambda: set(g["controls"]) == {"off", "empty"} and all(v["value"] == 0.0 and v["B_all"] == 0.0 and v["A_all"] == 0.0 for v in g["controls"].values()))
    c("two_step_identities", lambda: g["two_step"]["off_equal_A"] and g["two_step"]["empty_equal_A"] and g["two_step"]["on_differs_from_A"] and g["two_step"]["probe_reads_one_ulp"])
    c("dup_identity", lambda: g["dup_identity"]["share"] > 0.0 and g["dup_identity"]["abs_diff"] <= 1e-5 * max(1.0, g["dup_identity"]["share"]))
    c("pad_mask_equals_bank", lambda: g["pad"]["served_mask_equals_bank"] and g["pad"]["pad_slots"] > 0)
    c("pad_perturbation_bit_identical", lambda: g["pad"]["perturbed_loss_equal"])
    c("fan_call_bit_identical", lambda: g["fan_call_max_abs_diff"] == 0.0)
    c("fan_rows_isolated_from_hn", lambda: g["fan_rows_hn_isolation_max_abs"] == 0.0)
    c("hn_row_independent_of_other_hn_rows", lambda: g["hn_row_independence_max_abs"] == 0.0)
    c("d1_rows", lambda: g["d1"]["rows_in_term"] == g["d1"]["rows_expected"] and g["d1"]["rows_expected"] > 0)
    c("eval_reads_no_bank", lambda: g["eval_reads_bank"]["held_items_checked"] > 0 and g["eval_reads_bank"]["held_items_with_hn_tail"] == 0)
    c("mutate_list_empty", lambda: g["mutate"] == [])
    return b


def g15_bad(g) -> list:
    b: list = []

    def c(n, f):
        check(b, "G15_" + n, f)
    c("ctx_bit_identical", lambda: g["ctx_max_abs_diff"] == 0.0)
    c("logits_bit_identical", lambda: set(g["logit_max_abs_diff"]) == {"base", "A", "B"} and all(v == 0.0 for v in g["logit_max_abs_diff"].values()))
    c("shuffled_cache_far", lambda: g["shuffled"]["ctx_max_abs_diff"] > 0.0 and all(g["shuffled"]["logit_max_abs_diff"][k] > 10.0 * g["logit_max_abs_diff"][k] and g["shuffled"]["logit_max_abs_diff"][k] > 0.0
                                                                                      for k in ("base", "A", "B")))
    c("trunk_fingerprint_bound", lambda: g["trunk_fingerprint_manifest"] == g["trunk_fingerprint_checkpoint"] and len(g["trunk_fingerprint_checkpoint"]) == 64)
    c("covers_run", lambda: g["covers_run"] == {"missing_train": 0, "missing_val": 0})
    c("dtype_bf16", lambda: g["dtype"] == "bf16")
    c("calib_bound", lambda: g["calib_sha256_run"] is not None and g["calib_sha256_manifest"] == g["calib_sha256_run"])
    c("scenes_compared", lambda: g["n_train"] == min(64, NT) and g["n_val"] == min(64, NH) and g["micro_batch"] == 2)
    c("mutate_list_empty", lambda: g["mutate"] == [])
    return b


# ------------------------------------------------------------------------------------------------ the suite
EVID: dict = {}                                                        # tag -> the RAW G14 / G15 fields of that trainer (HNW_DUMP=<path> writes them: fixtures/hn_wiring_evidence.json)


def evals(res):
    return [r for r in res["rows"]["sft.jsonl"] if r.get("event") == "eval"]


def trains(res):
    return [r for r in res["rows"]["sft.jsonl"] if r.get("event") == "train"]


def expected_micro_hn_total(n_micro: int, counts) -> int:
    """the hard-negative rows the loop MUST have fed arm B over its first `n_micro` micro-batches: replay the loader's shuffle (a Generator seeded 0, one base-seed draw + one randperm per epoch, batches of 2 consecutive, drop_last) and sum
    the bank's per-scene counts -- computed here from the fixture's construction, not from the trainer's own mask"""
    g = torch.Generator().manual_seed(0)
    tot, done = 0, 0
    while done < n_micro:
        torch.empty((), dtype=torch.int64).random_(generator=g)                    # _BaseDataLoaderIter draws its worker base seed from the loader's generator when the epoch's iterator is created
        perm = torch.randperm(NT, generator=g).tolist()
        for j in range(NT // 2):
            if done >= n_micro:
                break
            tot += counts[perm[2 * j]] + counts[perm[2 * j + 1]]
            done += 1
    return tot


def suite(rigs, srcs, groups, tag="x") -> list:
    """-> the list of FAILING check names.  srcs = {"sf": trainer source, "hn": hn_bank source | None, "cc": ctx_cache source | None}"""
    bad: list = []
    SF, HNS, CCS = srcs["sf"], srcs.get("hn"), srcs.get("cc")
    PRE = read_text(PRE_PATH)
    r7, rp = rigs["n7"], rigs["plain"]

    def run(src, rig, name, **kw):
        return wrun(src, rig, f"{tag}_{name}_{rig.name}", **kw)

    def run_through(src, rig, name, **kw):
        """preflight then train.  When a (mutated) trainer FAILS its own preflight -- the gate doing its job -- the training phase is run ANYWAY with a forced pass token, so the end-to-end comparisons below are a
        SECOND, independent line of defence: they must see the defect without G14 / G15.  rc_pre / gates / pass_json stay the real preflight's, so the preflight check itself stays red."""
        r = run(src, rig, name, **kw)
        if r["rc_pre"] == 0 or r["rc_pre"] == 4:
            return r
        r2 = dict(run(src, rig, name, **{**kw, "phases": ("run",), "nonce": "forced", "pass_json": {"verdict": "FORCED_BY_TEST"}}))
        r2["rc_pre"], r2["gates"], r2["pass_json"] = r["rc_pre"], r["gates"], r["pass_json"]
        return r2

    if "PIN" in groups:
        mod = load_mod(SF, "sft_wire_pin_new", SRC_PATH)
        pre_mod = load_mod(PRE, "sft_wire_pin_pre", SRC_PATH)
        import rls1_checks as RC
        check(bad, "PIN_peel_equals_tip_blob", lambda: blob_sha1(HL.peel(SF)) == TIP_BLOB == HL.TIP_TRAINER_BLOB)
        check(bad, "PIN_carvefeed_chain_equals_pre_B8_blob", lambda: blob_sha1(CL.peel(SF)) == PRE_B8_BLOB)
        check(bad, "PIN_fixture_sha256", lambda: hashlib.sha256(PRE.encode("utf-8")).hexdigest() == PRE_SHA256 and blob_sha1(PRE) == TIP_BLOB)
        check(bad, "PIN_fixture_is_the_peeled_trainer", lambda: HL.peel(SF) == PRE)
        check(bad, "PIN_layer_marker", lambda: HL.has_layer(SF) and not HL.has_layer(PRE) and HL.peel(PRE) == PRE)
        check(bad, "PIN_partition_and_label_version", lambda: mod.HN_PARTITION_ID == RC.PARTITION["partition_id"] and mod.HN_LABEL_VERSION == 5 and mod.HN_ARM == "B")
        check(bad, "PIN_new_dests_are_exactly_the_two_flags_default_empty", lambda: (
            {a_.dest for a_ in mod.build_parser()._actions} - {a_.dest for a_ in pre_mod.build_parser()._actions} == {"hn_bank", "ctx_cache"}
            and {a_.dest for a_ in pre_mod.build_parser()._actions} <= {a_.dest for a_ in mod.build_parser()._actions}
            and {a_.dest: a_.default for a_ in mod.build_parser()._actions}["hn_bank"] == "" and {a_.dest: a_.default for a_ in mod.build_parser()._actions}["ctx_cache"] == ""))

        def nested_identical():
            def nest(src):
                m = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main")
                return {n.name: ast.dump(n) for n in m.body if isinstance(n, ast.FunctionDef)}
            a_, b_ = nest(PRE), nest(SF)
            return set(a_) <= set(b_) and all(a_[k] == b_[k] for k in a_) and len(a_) >= 5
        check(bad, "PIN_every_nested_function_of_main_is_AST_identical", nested_identical)
        check(bad, "PIN_loop_rebinding_only_inside_if", lambda: SF.count("\n    def fwd(batch):") == 1 and SF.count("\n        def fwd(batch):") == 1)

    if "I3" in groups:
        for rg in (rp, r7):
            a_, b_ = run(PRE, rg, "pre"), run(SF, rg, "new")
            a2 = run(PRE, rg, "pre", nonce="again")
            n = "I3_" + rg.name
            check(bad, n + "_runs_complete", lambda: a_["rc_pre"] == 0 and a_["rc_run"] == 0 and b_["rc_pre"] == 0 and b_["rc_run"] == 0 and a_["pass_json"] is not None)
            check(bad, n + "_preflight_gates_equal", lambda: a_["gates"] is not None and rows_equal(a_["gates"], b_["gates"]))
            check(bad, n + "_every_logged_row_equal", lambda: len(a_["rows"]["sft.jsonl"]) >= 6 and rows_equal(a_["rows"]["sft.jsonl"], b_["rows"]["sft.jsonl"]))
            check(bad, n + "_has_a_train_row_and_evals", lambda: len(trains(b_)) == 1 and len(evals(b_)) >= 3 and "hn_rows" not in trains(b_)[0])
            check(bad, n + "_saved_tensors_A_B_equal", lambda: same_sd(a_["A"], b_["A"]) and same_sd(a_["B"], b_["B"]))
            check(bad, n + "_data_event_has_no_hn_or_cache_block", lambda: "hn" not in b_["data"] and "ctx_cache" not in b_["data"])
            check(bad, n + "_control_pre_edit_twice_is_deterministic", lambda: rows_equal(a_["rows"]["sft.jsonl"], a2["rows"]["sft.jsonl"]) and same_sd(a_["A"], a2["A"]) and same_sd(a_["B"], a2["B"]))
            c_ = run(SF, rg, "newlr", lr="1.1e-3")
            check(bad, n + "_control_comparison_sees_a_changed_lr", lambda: not same_sd(b_["A"], c_["A"]) and not rows_equal(b_["rows"]["sft.jsonl"], c_["rows"]["sft.jsonl"]))
        check(bad, "I3_B_equals_A_without_the_bank_lam0", lambda: same_sd(run(SF, r7, "new")["A"], run(SF, r7, "new")["B"]))

    if "ON" in groups:
        counts = expected_hn_counts()
        RL = run(SF, r7, "new")                                                                         # flags absent
        RC_ = run_through(SF, r7, "cache", extra=("--ctx-cache", r7.cache), hn_src=HNS, cc_src=CCS)
        RCB = run_through(SF, r7, "cb", extra=("--ctx-cache", r7.cache, "--hn-bank", r7.bank), hn_src=HNS, cc_src=CCS)
        RB = run_through(SF, r7, "bank", extra=("--hn-bank", r7.bank), hn_src=HNS, cc_src=CCS)
        RE = run(SF, r7, "cbempty", extra=("--ctx-cache", r7.cache, "--hn-bank", r7.bank), hn_src=HNS, cc_src=CCS, empty_term=True, phases=("run",), pass_json={"verdict": "FORCED_BY_TEST"})
        for nm, rr in (("cache", RC_), ("cb", RCB), ("bank", RB)):
            check(bad, f"ON_{nm}_preflight_pass_and_run_complete", lambda rr=rr: rr["rc_pre"] == 0 and rr["rc_run"] == 0 and rr["pass_json"] is not None and rr["A"] is not None and rr["B"] is not None)
        check(bad, "ON_cbempty_run_complete", lambda: RE["rc_run"] == 0 and RE["A"] is not None and RE["B"] is not None)
        # the cache alone is the live trunk, bit for bit
        check(bad, "ON_cache_only_equals_live_tensors", lambda: same_sd(RL["A"], RC_["A"]) and same_sd(RL["B"], RC_["B"]))
        check(bad, "ON_cache_only_equals_live_eval_and_train_rows", lambda: len(evals(RL)) >= 3 and rows_equal(evals(RL), evals(RC_)) and rows_equal(trains(RL), trains(RC_)))
        check(bad, "ON_cache_only_B_equals_A", lambda: same_sd(RC_["A"], RC_["B"]))
        # the bank moves arm B only
        check(bad, "ON_bank_leaves_arm_A_bit_identical", lambda: same_sd(RC_["A"], RCB["A"]))
        check(bad, "ON_bank_moves_arm_B", lambda: not same_sd(RC_["B"], RCB["B"]) and not same_sd(RCB["A"], RCB["B"]))
        check(bad, "ON_bank_arm_A_and_base_eval_numbers_equal_the_cache_only_run", lambda: all(x["A"] == y["A"] and x["base"] == y["base"] for x, y in zip(evals(RC_), evals(RCB))) and len(evals(RCB)) >= 3)
        check(bad, "ON_bank_step0_eval_row_fully_equal_I4", lambda: rows_equal(evals(RC_)[0], evals(RCB)[0]) and evals(RCB)[0]["tag"] == "step0")
        check(bad, "ON_bank_arm_B_eval_moves_after_training", lambda: any(x["B"] != y["B"] for x, y in zip(evals(RC_)[1:], evals(RCB)[1:])))
        check(bad, "ON_live_trunk_bank_run_equals_cached_bank_run", lambda: same_sd(RB["A"], RCB["A"]) and same_sd(RB["B"], RCB["B"]) and rows_equal(evals(RB), evals(RCB)))
        check(bad, "ON_empty_term_B_equals_A_equals_cache_only", lambda: same_sd(RE["A"], RE["B"]) and same_sd(RE["A"], RC_["A"]))
        # D1: the rows that entered L_B equal the rows the bank holds for the scenes the loop served (independent replay of the loader)
        check(bad, "ON_D1_train_row_counts", lambda: len(trains(RCB)) == 1 and trains(RCB)[0]["hn_rows"] == trains(RCB)[0]["hn_rows_expected"] > 0 and trains(RCB)[0]["hn_micro"] == 50
              and trains(RCB)[0]["hn_micro_nonempty"] > 0 and trains(RCB)[0]["hn_loss"] > 0.0)
        check(bad, "ON_D1_rows_equal_an_independent_replay_of_the_loader", lambda: trains(RCB)[0]["hn_rows"] == expected_micro_hn_total(50, counts))
        check(bad, "ON_D1_no_bank_no_hn_fields", lambda: all("hn_rows" not in t for t in trains(RC_) + trains(RL)))
        # the data event
        bk = HN0.load_bank(r7.bank)
        check(bad, "ON_data_event_hn_block_is_the_banks", lambda: RCB["data"]["hn"] == bk.hn_block("B") and RCB["data"]["hn"]["rows"] == sum(counts) and RCB["data"]["hn"]["arm"] == "B" and "ctx_cache" not in RB["data"])
        check(bad, "ON_data_event_cache_block", lambda: RCB["data"]["ctx_cache"] == {"manifest_sha256": CC0.sha256_file(os.path.join(r7.cache, "manifest.json")), "trunk_fingerprint": r7.trunk_fp,
                                                                                       "n_scenes": NT + NH, "dtype": "bf16"} and "hn" not in RC_["data"])
        # the raw gate fields
        g14 = RCB["gates"]["G14_hn_live_and_identity"] if RCB["gates"] else None
        g15 = RCB["gates"]["G15_ctx_cache_identity"] if RCB["gates"] else None
        EVID[tag] = {"g14": g14, "g15": g15}
        for nm in (g14_bad(g14) if g14 else ["G14_missing"]):
            bad.append(nm)
        for nm in (g15_bad(g15) if g15 else ["G15_missing"]):
            bad.append(nm)
        check(bad, "ON_G14_absent_without_bank_G15_absent_without_cache", lambda: "G14_hn_live_and_identity" not in RC_["gates"] and "G15_ctx_cache_identity" not in RB["gates"] and "G14_hn_live_and_identity" in RB["gates"]
              and "G14_hn_live_and_identity" not in RL["gates"] and "G15_ctx_cache_identity" not in RL["gates"])
        check(bad, "ON_bank_only_G14_lane_row_zero_and_pass", lambda: g14_bad(RB["gates"]["G14_hn_live_and_identity"]) == [])
        # the gate rows the launch gate reads: ok is the trainer's, the judges above are ours -- both must agree
        check(bad, "ON_trainer_ok_agrees_with_the_judges", lambda: g14["ok"] is True and g15["ok"] is True and RCB["pass_json"]["verdict"] == "PASS")

    if "REF" in groups:
        good_cache, good_bank = r7.cache, r7.bank

        mod_u = load_mod(SF, "sft_wire_unit", SRC_PATH)
        REQ = ["--ckpt", "k", "--targets", "t", "--images", "i", "--calib", "c", "--onpolicy", "p", "--heldout", "h", "--heldout-logs", "l", "--out", "o"]

        def unit(name, argv, text):
            def f():
                ap_ = mod_u.build_parser()
                return any(text in m for m in mod_u.hn_refusals(ap_.parse_args(REQ + argv), ap_))
            check(bad, "REFU_" + name, f)
        unit("hn_with_epo", ["--hn-bank", "x", "--epo"], "--hn-bank with --epo")
        unit("hn_with_lam", ["--hn-bank", "x", "--lam", "0.5"], "--hn-bank needs --lam 0")
        unit("hn_with_bmode", ["--hn-bank", "x", "--b-mode", "compw"], "--hn-bank needs --b-mode listnet")
        unit("hn_with_aux", ["--hn-bank", "x", "--aux-drv"], "--hn-bank with --aux-drv")
        unit("cache_with_aux", ["--ctx-cache", "x", "--aux-drv"], "--ctx-cache with --aux-drv")
        check(bad, "REFU_nothing_named_nothing_refused", lambda: (lambda ap_: mod_u.hn_refusals(ap_.parse_args(REQ), ap_) == [] and mod_u.hn_refusals(ap_.parse_args(REQ + ["--hn-bank", "x", "--lam", "0"]), ap_) == [] and mod_u.hn_refusals(ap_.parse_args(REQ + ["--ctx-cache", "x"]), ap_) == [])(mod_u.build_parser()))

        def ref(name, text, extra, **kw):
            res = run(SF, r7, "ref_" + name, extra=extra, phases=("pre",), hn_src=HNS, cc_src=CCS, **kw)
            check(bad, "REF_" + name, lambda: refused(res, text) and not os.path.exists(os.path.join(r7.td, f"run_{tag}_ref_{name}_{r7.name}", "out", "PREFLIGHT_PASS.json")))
        ref("hn_with_epo", None, ("--hn-bank", good_bank, "--epo"))                                    # the stock epo refusals fire first: main() must return 4 (the message is checked by REFU_)
        ref("hn_with_lam", "--hn-bank needs --lam 0", ("--hn-bank", good_bank, "--lam", "0.5"))
        ref("hn_with_bmode", "--hn-bank needs --b-mode listnet", ("--hn-bank", good_bank, "--b-mode", "compw"))
        ref("hn_with_aux", None, ("--hn-bank", good_bank, "--aux-drv"))
        ref("cache_with_aux", None, ("--ctx-cache", good_cache, "--aux-drv"))
        ref("bank_other_partition", "bank partition_id", ("--hn-bank", r7.mint("partition", partition_id="0" * 64)))
        ref("bank_other_label_version", "bank label_version", ("--hn-bank", r7.mint("lv4", label_version=4)))
        ref("bank_other_fan_fingerprint", "bank fan_fingerprint", ("--hn-bank", r7.mint("fanfp", fan_fp="1" * 64)))
        flipped = os.path.join(r7.td, "bank_flipped.hnb")
        with open(good_bank, "rb") as f_:
            data_ = bytearray(f_.read())
        data_[-1] ^= 0xFF
        with open(flipped, "wb") as f_:
            f_.write(bytes(data_))
        with open(good_bank + ".manifest.json", "rb") as f_, open(flipped + ".manifest.json", "wb") as g_:
            g_.write(f_.read())
        ref("bank_bytes_changed", "changed after minting", ("--hn-bank", flipped))
        stat = anchors_for(r7.train_keys)
        for a_ in stat:
            a_["traj"] = (np.array(a_["traj"]) * 0.2).tolist()                                           # every path < 2.0 m: STATIONARY, no row is labelled, 0 hard negatives
        ref("bank_zero_hard_negatives", "0 hard-negative rows", ("--hn-bank", r7.mint("empty", anchors=stat)))
        WStubBank.LV = 4
        try:
            ref("training_sets_other_label_version", "label versions", ("--hn-bank", good_bank))
        finally:
            WStubBank.LV = 5
        ref("cache_other_trunk", "cache trunk", ("--ctx-cache", r7.build_cache("trunk2", perturb_trunk=True)))
        ref("cache_lacks_a_scene", "holds no context", ("--ctx-cache", r7.build_cache("drop", drop_log="T3")))
    return bad


# ------------------------------------------------------------------------------------------------ the 7 mutations of PREREG section 4.3 (each edits SOURCE and must turn the NAMED checks RED)
def mutations(SF: str, HNS: str, CCS: str):
    """-> [(id, what, srcs, groups, expected-RED check names)]"""
    return [
        ("M1", "the hn term is computed but never added to arm B's loss (`lB = lB + hn_out.loss` removed)",
         {"sf": once(SF, "            lB = lB + hn_out.loss\n", "            pass\n"), "hn": None, "cc": None}, ("ON",),
         ["ON_bank_moves_arm_B", "G14_value_on_positive", "G14_grad_q_mlp_nonzero", "G14_grad_dec_nonzero", "G14_grad_head_rows_0_5_nonzero"]),
        ("M2", "the term is applied to arm A's scorer (`HN.hn_term(armB,` -> `HN.hn_term(model,`)",
         {"sf": once(SF, "hn_out = HN.hn_term(armB,", "hn_out = HN.hn_term(model,"), "hn": None, "cc": None}, ("ON",),
         ["ON_bank_leaves_arm_A_bit_identical", "G14_armA_exactly_zero"]),
        ("M3", "pad slots are served as valid (the bank's mask replaced by all-True)",
         {"sf": once(SF, 'b["hn_m"][0]', 'torch.ones_like(b["hn_m"][0])'), "hn": None, "cc": None}, ("ON",),
         ["G14_pad_mask_equals_bank", "G14_d1_rows", "ON_D1_train_row_counts"]),
        ("M4", "the fan rows can see the hn slots (hn_attn_mask no longer blocks fan -> hn)",
         {"sf": SF, "hn": once(HNS, "blocked = (j >= K) & (j != i)\n", "blocked = (j >= K) & (j != i) & (i >= K)\n"), "cc": None}, ("ON",),
         ["G14_fan_rows_isolated_from_hn"]),
        ("M5", "the hn rows train the LANE output (a loss term on logits[..., 6:])",
         {"sf": SF, "hn": once(HNS, 'logits[..., :N_COMP], tg[..., :N_COMP], reduction="none")', 'logits[..., :N_COMP], tg[..., :N_COMP], reduction="none") + 1e-3 * logits[..., N_COMP:].sum(-1, keepdim=True)'),
          "cc": None}, ("ON",),
         ["G14_lane_row_exactly_zero"]),
        ("M6", "the bank's partition / label_version / fan-fingerprint identity is not checked (`expect=hn_expect` removed)",
         {"sf": once(SF, "expect=hn_expect", "expect=None"), "hn": None, "cc": None}, ("REF",),
         ["REF_bank_other_partition", "REF_bank_other_label_version", "REF_bank_other_fan_fingerprint"]),
        ("M7", "the cache's trunk fingerprint is not checked against the checkpoint's (`expect_trunk_fingerprint=trunk_fp` removed)",
         {"sf": once(SF, "expect_trunk_fingerprint=trunk_fp", "expect_trunk_fingerprint=None"), "hn": None, "cc": None}, ("REF",),
         ["REF_cache_other_trunk"]),
    ]


def main() -> int:
    SF, HNS, CCS = read_text(SRC_PATH), read_text(HN_PATH), read_text(CC_PATH)
    HNW["partition"] = load_mod(patch_src(SF), "sft_wire_probe", SRC_PATH).HN_PARTITION_ID
    n_fail = 0
    with tempfile.TemporaryDirectory(prefix="hnwire_") as td:
        rigs = {"plain": Rig(td, False), "n7": Rig(td, True)}
        bad = suite(rigs, {"sf": SF, "hn": None, "cc": None}, ("PIN", "I3", "ON", "REF"), tag="base")
        n_checks = NCHECK[0]
        print("-- checks:", "ALL GREEN" if not bad else f"{len(bad)} RED: {bad}")
        n_fail += len(bad)
        print("-- mutation table (each must be RED on the named checks)")
        for mid, what, srcs, groups, expect in mutations(SF, HNS, CCS):
            red = suite(rigs, srcs, groups, tag=mid)
            hit = [e for e in expect if e in red]
            ok = bool(red) and len(hit) == len(expect)
            print(f"   {mid} {'RED as required' if ok else 'NOT RED'}  {what}\n      named: {expect}\n      red:   {red}")
            if not ok:
                n_fail += 1
    if os.environ.get("HNW_DUMP"):
        with open(os.environ["HNW_DUMP"], "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"_what": "RAW preflight G14 / G15 fields of scorer_finetune.py's trainer on the CPU rig of refe/test_hn_screen_wiring.py (24 train / 12 val scenes, micro-batch 2): base = the clean trainer, "
                                "M1..M5 = the five trainer / module mutations of PREREG_P6HN_SCREEN 4.3 (their G14 is the evidence a launch gate must reject). CPU stub evidence: not a pod measurement.",
                       **{k: EVID[k] for k in sorted(EVID)}}, fh, indent=1, sort_keys=True)
            fh.write("\n")
    if n_fail:
        print(f"ZZTEST_FAIL {n_fail}")
        return 1
    print("ZZTEST_PASS", n_checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
