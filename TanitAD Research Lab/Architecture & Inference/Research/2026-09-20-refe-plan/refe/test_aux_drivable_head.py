#!/usr/bin/env python3
"""Proof of `aux_drivable_head.py` (SPEC_AUX_DRIVABLE_HEAD.md sections 2-4, 6-7) to the programme's standard.  CPU only.

Run:  PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/tanitad/Scripts/python.exe refe/test_aux_drivable_head.py   ->   ZZTEST_PASS <n>
(needs the label store at $AUX_LABEL_STORE, default D:/Projects/TanitAD-artifacts/aux_drivable/label_store; a missing store FAILS the real-key
checks unless AUX_ALLOW_NO_STORE=1 names the skip, and the final line then says so.)

WHAT IS PROVED, and how (each group has a control that must read a known value and a mutation that must turn it RED):
  A  shape / parameter count (= the prototype's 2,440,450, parsed from its selftest output) / one-tensor signature / own RNG stream /
     import hygiene (the planner path never reaches the label path) / config + CLI hygiene
  B  the target builder EXACT on hand-built SDFs (orientation, sign, 0.1 m unit, 4 x 4 block statistics, clip-before-mean, extent mask),
     loss identities, the H2 sampler and the three feed scalars against an INDEPENDENT numpy chain (navsim_dac + a numpy bilinear)
  C  gradient controls on the REAL REFe (tiny backbone): w = 0 => head grads exactly 0.0 and the trunk's grads bit-identical to the no-head
     run; w > 0 => every head tensor non-zero; aux-only reaches LoRA / scene_proj / pos3d_mlp and EXACTLY 0.0 on traj_head / reg_compress
  D  H1 vs H2: H1's scorer output is bit-identical with and without (and with a DIFFERENT) head; H2's is not; no autograd path head -> score
  E  the label source on the REAL store: pin, internal consistency, shard bytes, one read per log, >= 32 real keys round-tripped, the
     block-statistic mask vs sign(SDF) down-sampled (the agreement is reported), whole-store content hash, navtest disjointness
  F  RED mutations of the shipped source text: each must fail at least one check
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import types
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import aux_label_store as ALS  # noqa: E402
import field_loss as FL  # noqa: E402
import model as RM  # noqa: E402
import navsim_dac as ND  # noqa: E402  (numpy-only at import)

PKG = os.path.dirname(HERE)
SRC = os.path.join(HERE, "aux_drivable_head.py")
STORE = os.environ.get("AUX_LABEL_STORE", "D:/Projects/TanitAD-artifacts/aux_drivable/label_store")
NO_STORE_OK = os.environ.get("AUX_ALLOW_NO_STORE") == "1"
PROTO_OUT = os.path.join(PKG, "raw", "2026-10-11-aux-drivable", "head_proto_selftest.out")
REPORT: dict = {}


# ----------------------------------------------------------------------------------------------------------------------
# harness
# ----------------------------------------------------------------------------------------------------------------------
def load(src: str, name: str):
    """Exec `src` as a fresh module (the mutants); registered in sys.modules first because dataclasses resolve annotations through it."""
    mod = types.ModuleType(name)
    mod.__file__ = SRC
    sys.modules[name] = mod
    exec(compile(src, SRC, "exec"), mod.__dict__)
    return mod


class Rec:
    def __init__(self, verbose: bool):
        self.rows, self.verbose = [], verbose

    def check(self, name, fn):
        try:
            r = fn()
            ok, detail = r if isinstance(r, tuple) else (bool(r), "")
        except Exception as e:  # noqa: BLE001 -- a mutant that breaks the suite's own setup is RED, and says so
            ok, detail = False, f"raised {type(e).__name__}: {str(e)[:100]}"
        self.rows.append((name, bool(ok), detail))
        if self.verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        return ok

    def failing(self):
        return [n for n, ok, _ in self.rows if not ok]


def equal(a, b):
    return bool(torch.equal(a, b))


def bits_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and bool((a.contiguous().view(torch.int32) == b.contiguous().view(torch.int32)).all())


def i8(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int8))[None]


def np_stats(lab):
    s = lab.astype(np.float64) * 0.1
    blk = s.reshape(40, 4, 32, 4)
    return (blk > 0).mean((1, 3)), np.clip(blk, -6.0, 6.0).mean((1, 3))


# ----------------------------------------------------------------------------------------------------------------------
# fixtures (independent of the module under test, so the mutants see the same world)
# ----------------------------------------------------------------------------------------------------------------------
@lru_cache(None)
def tiny():
    torch.manual_seed(0)
    cfg = RM.REFeConfig.for_backbone("vits16", img_h=64, img_w=128, detach_scorer_context=False)
    net = RM.REFe(cfg)
    g = torch.Generator().manual_seed(1)
    img = torch.rand(2, 4, 3, 64, 128, generator=g)
    ego, goal = torch.zeros(2, cfg.ego_dim), torch.zeros(2, 2 * cfg.n_goal_points)
    ys = -32.0 + (np.arange(128) + 0.5) * 0.5
    road = np.clip(np.round((3.5 - np.abs(ys)) / 0.1), -127, 127).astype(np.int8)               # a 7 m road along x
    sdf = np.broadcast_to(road[None, :], (160, 128)).copy()
    lab = torch.from_numpy(np.stack([sdf, np.roll(sdf, 24, axis=1)]))
    return net, cfg, img, ego, goal, lab


def fwd(net, img, ego, goal):
    cap = []
    h = net.scene_proj.register_forward_hook(lambda m, i, o: cap.append(o))
    try:
        traj, score = net(img, ego, goal)
    finally:
        h.remove()
    return traj, score, cap                                   # cap[0] = scene_ctx [B, 64, 256], cap[1] = visual_ctx [B, P, 256]


def group_of(name):
    for g in ("scene_proj", "pos3d_mlp", "reg_compress", "traj_head"):
        if name.startswith(g + "."):
            return g
    if name.startswith("backbone.") and (".A" in name or ".B" in name):
        return "lora"
    return None


def collect(net):
    grads = {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in net.named_parameters()}
    groups = {}
    for n, g in grads.items():
        k = group_of(n)
        if k is not None:
            groups[k] = groups.get(k, 0.0) + (0.0 if g is None else float(g.abs().sum()))
    return grads, groups


def run(M, w, mode, tap="visual", detach=False, seed=3):
    """One forward/backward on the real (tiny) REFe. mode: task | task+aux | aux."""
    net, cfg, img, ego, goal, lab = tiny()
    net.zero_grad(set_to_none=True)
    traj, score, cap = fwd(net, img, ego, goal)
    ctx = cap[1] if tap == "visual" else cap[0]
    ctx.retain_grad()
    l_task = traj.pow(2).mean() + score.pow(2).mean()
    aux = None
    if mode == "task":
        loss = l_task
    else:
        aux = M.AuxDrivable(M.AuxDrivableConfig(w_aux=w, tap=tap, seed=seed))
        logit, sdf = aux.predict(ctx.detach() if detach else ctx)
        l_aux, _, _ = aux.loss(logit, sdf, lab)
        loss = aux.combine(l_task, l_aux) if mode == "task+aux" else aux.combine(l_task * 0.0, l_aux)
    loss.backward()
    grads, groups = collect(net)
    hg = None if aux is None else {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in aux.head.named_parameters()}
    return {"grads": grads, "groups": groups, "ctx_grad": None if ctx.grad is None else ctx.grad.clone(), "head": hg}


@lru_cache(None)
def baseline():
    return run(None, 0.0, "task"), run(None, 0.0, "task")


@lru_cache(None)
def scorer_fixture():
    net, cfg, img, ego, goal, lab = tiny()
    with torch.no_grad():
        traj, _, cap = fwd(net, img, ego, goal)
        trajs, sctx = traj.detach().clone(), cap[1].detach().clone()
        base = net.score_trajectories(trajs, sctx)
    return net, trajs, sctx, base


@lru_cache(None)
def real_keys():
    """48 real keys from 12 logs over the 4 maps, chosen from the store's own manifest and shard headers (independent of the module under test)."""
    man = json.load(open(os.path.join(STORE, "manifest.json"), encoding="utf-8"))
    by_map = {}
    for rel, v in sorted(man["files"].items()):
        by_map.setdefault(v["map"], []).append(rel)
    keys = []
    for m in sorted(by_map):
        rels = by_map[m]
        for r in [rels[int(i)] for i in np.linspace(0, len(rels) - 1, 3)]:
            head, _ = ALS.read_shard(os.path.join(STORE, r), with_blobs=False)
            ents = head["entries"]
            for j in np.unique(np.linspace(0, len(ents) - 1, 4).astype(int)):
                keys.append((head["log"], ents[int(j)]["t"]))
    return keys


@lru_cache(None)
def real_labels():
    return {k: ALS.read_label(STORE, *k) for k in real_keys()}


# ----------------------------------------------------------------------------------------------------------------------
# the checks
# ----------------------------------------------------------------------------------------------------------------------
def suite(M, heavy: bool, verbose: bool = False) -> Rec:
    R = Rec(verbose)
    C = R.check
    cfg_ = M.AuxDrivableConfig

    # ======================================================= A. structure =================================================
    def a_params():
        proto = int(re.search(r"head params ([\d,]+)", open(PROTO_OUT, encoding="utf-8").read()).group(1).replace(",", ""))
        n = sum(p.numel() for p in M.build_head(cfg_()).parameters())
        REPORT["params"], REPORT["proto_params"] = n, proto
        return n == proto == 2440450, f"{n:,} vs prototype {proto:,}"
    C("A1 parameter count == the prototype's 2,440,450 (parsed from head_proto_selftest.out)", a_params)

    def a_shapes():
        h = M.build_head(cfg_())
        with torch.no_grad():
            lg, sd = h(torch.randn(2, 7680, 256))
            lg2, sd2 = M.build_head(cfg_(tap="scene"))(torch.randn(2, 64, 256))
        return (tuple(lg.shape) == tuple(sd.shape) == (2, 40, 32) and tuple(lg2.shape) == (2, 40, 32)
                and float(sd.abs().max()) <= 6.0 and float(sd2.abs().max()) <= 6.0), f"{tuple(lg.shape)} |sdf|<={float(sd.abs().max()):.2f}"
    C("A2 output [B, 40, 32] for visual_ctx [B, 7680, 256] and scene_ctx [B, 64, 256]; SDF bounded by 6 m", a_shapes)

    C("A3 the head's forward takes exactly ONE tensor (admissibility: no ego / goal / nav / map input)",
      lambda: list(inspect.signature(M.DrivableAuxHead.forward).parameters) == ["self", "ctx"])

    def a_geom():
        sp = ALS.raster_spec()
        ok = (M.X0, M.X1, M.Y0, M.Y1, M.FINE_RES, M.NX_F, M.NY_F, M.SDF_UNIT_M) == (sp["x0"], sp["x1"], sp["y0"], sp["y1"], sp["res_m"], sp["nx"], sp["ny"], sp["sdf_unit_m"])
        ok &= (M.X0, M.X1, M.Y0, M.Y1) == (FL.X0, FL.X1, FL.Y0, FL.Y1)
        h = M.build_head(cfg_())
        ok &= (h.nx, h.ny) == (40, 32) and abs(float(h.pos_feat[0, 0]) - (0.5 * 2.0) / 80.0) < 1e-6
        return ok, ""
    C("A4 the head's / label's geometry constants equal aux_label_store.raster_spec() and field_loss's", a_geom)

    def a_rng():
        torch.manual_seed(123)
        s0 = torch.get_rng_state().clone()
        h1 = M.build_head(cfg_(seed=7))
        s1 = torch.get_rng_state().clone()
        torch.manual_seed(999)
        h2 = M.build_head(cfg_(seed=7))
        h3 = M.build_head(cfg_(seed=8))
        same = all(equal(a, b) for a, b in zip(h1.state_dict().values(), h2.state_dict().values()))
        diff = any(not equal(a, b) for a, b in zip(h1.state_dict().values(), h3.state_dict().values()))
        return equal(s0, s1) and same and diff, f"global RNG untouched={equal(s0, s1)} same-seed bit-equal={same} other-seed differs={diff}"
    C("A5 the head builds from its OWN stream: global RNG state unchanged, bit-equal for the same seed whatever the global seed", a_rng)

    def a_cfg():
        c = cfg_()
        errs = []
        for fn in (lambda: setattr(c, "feed", "on"), lambda: setattr(c, "aux_drv_feeed", 1), lambda: cfg_(aux_drv_feeed="on"),
                   lambda: cfg_(tap="nope"), lambda: cfg_(feed="maybe"), lambda: cfg_(cell=1.3), lambda: cfg_(w_aux=-1.0)):
            try:
                fn()
                errs.append(False)
            except (AttributeError, TypeError, ValueError):
                errs.append(True)
        return all(errs) and c.w_aux == 0.0 and c.feed == "off", f"refusals {errs} defaults w_aux={c.w_aux} feed={c.feed}"
    C("A6 config hygiene (G-HYG): frozen, typos and invalid values RAISE; w_aux defaults to 0.0 and feed to off", a_cfg)

    def a_cli():
        import argparse
        ap = argparse.ArgumentParser()
        M.add_aux_args(ap)
        d = ap.parse_args([])
        dests = {a.dest for a in ap._actions}
        off = (d.aux_drv is False and d.w_aux_drv == 0.0 and d.aux_drv_feed == "off" and d.aux_drv_labels == "" and d.aux_drv_shuffle_seed is None)
        ids = set(M.AUX_IDENTITY_ARGS) <= dests
        r1 = M.aux_refusals(d)
        r2 = M.aux_refusals(ap.parse_args(["--w-aux-drv", "0.1"]))                       # a weight with no head: silently nothing
        r3 = M.aux_refusals(ap.parse_args(["--aux-drv"]))                                 # a head with no labels
        r4 = M.aux_refusals(ap.parse_args(["--aux-drv", "--aux-drv-labels", "x", "--w-aux-drv", "0.1"]))
        r5 = M.aux_refusals(ap.parse_args(["--aux-drv", "--aux-drv-labels", "x", "--aux-drv-feed", "on", "--aux-drv-shuffle-seed", "1"]))
        c = M.AuxDrivableConfig.from_args(ap.parse_args(["--aux-drv", "--aux-drv-labels", "x", "--w-aux-drv", "0.1", "--aux-drv-feed", "on", "--aux-drv-tap", "scene"]))
        ok = off and ids and not r1 and r2 and r3 and not r4 and r5 and (c.w_aux, c.feed, c.tap) == (0.1, "on", "scene")
        return ok, f"defaults all off={off} identity dests declared={ids} refusals(silent weight, no labels, shuffle+feed)={bool(r2)},{bool(r3)},{bool(r5)}"
    C("A7 the six flags default OFF; inconsistent command lines are refused; IDENTITY dests exist", a_cli)

    def a_facts():
        a0 = M.AuxDrivable(cfg_(depth=3, w_aux=0.1))
        a1 = M.AuxDrivable(cfg_(feed="on"))
        f0, f1 = a0.built_facts(), a1.built_facts()
        ok = (f0["depth"], f0["feed_built"], f0["scorer_in_width"], f0["w_aux"], f0["cell"], f0["x0"], f0["y0"]) == (3, False, 60, 0.1, 2.0, -8.0, -32.0)
        ok &= (f1["feed_built"], f1["scorer_in_width"]) == (True, 63) and f1["n_params"] == 2440450
        return ok, f"H1 width {f0['scorer_in_width']}, H2 width {f1['scorer_in_width']}"
    C("A8 built_facts() (what G-DVB compares to the declared config): depth, cell, x0/y0, w_aux, scorer width 60 vs 63", a_facts)

    def a_amp():
        aux = M.AuxDrivable(cfg_(feed="on", feed_init="normal", w_aux=0.1))
        ctx = torch.randn(2, 7680, 256)
        lab = tiny()[5]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            lg, sd = aux.predict(ctx)
            feats, _ = M.feed_scalars(sd, torch.randn(2, 64, 20, 3) * 3)
        l, bce, hub = aux.loss(lg, sd, lab)
        l.backward()
        gs = [p.grad for p in aux.head.parameters()]
        ok = (bool(torch.isfinite(l)) and l.dtype == torch.float32 and bool(torch.isfinite(feats).all()) and feats.dtype == torch.float32
              and all(g is not None and bool(torch.isfinite(g).all()) for g in gs))
        return ok, f"under bf16 autocast: head outputs {lg.dtype}, loss {l.dtype} finite, feed {feats.dtype} finite, {len(gs)} finite head grads"
    C("A10 bf16 autocast (the trainers' mode): head forward, fp32 loss, feed scalars and head gradients are finite", a_amp)

    def a_tap():
        x = torch.randn(2, 5, 256)
        return M.tap_is(x, x) and not M.tap_is(x.clone(), x) and not M.tap_is(x[:, :4], x) and not M.tap_is(x[:, 1:], x), "same storage and shape only"
    C("A11 tap_is (G-DVB): True only for the very tensor (same storage and shape), False for a clone or a slice", a_tap)

    if heavy:
        def a_hygiene():
            code = ("import sys; sys.path.insert(0, %r); import aux_drivable_head; "
                    "print('MODS=' + ','.join(m for m in ('aux_label_store','shapely','nuplan','navsim_dac') if m in sys.modules))" % HERE)
            proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
            mods = [ln for ln in proc.stdout.splitlines() if ln.startswith("MODS=")]
            src = open(SRC, encoding="utf-8").read()
            hits = sum(src.count(t) for t in ("map_api", "nuplan", "origin_world"))
            return mods == ["MODS="] and hits == 0, f"modules pulled in by `import aux_drivable_head`: {mods}; forbidden strings in the file: {hits}"
        C("A9 import hygiene (SPEC 6.1): importing the head pulls in no label-store / map module; grep 'map_api|nuplan|origin_world' == 0", a_hygiene)

    # ======================================================= B. targets, loss, sampler, feed ==============================
    def mk(fn):
        a = np.zeros((160, 128), np.int8)
        fn(a)
        return i8(a)

    def b_const():
        fr, sd = M.block_stats(mk(lambda a: a.fill(40)))
        return float(fr.min()) == 1.0 and float(fr.max()) == 1.0 and float((sd - 4.0).abs().max()) < 1e-6 and tuple(fr.shape) == (1, 40, 32), f"sdf {float(sd[0, 0, 0]):.6f}"
    C("B1 all +40 (4.0 m): mask fraction 1 everywhere, block SDF 4.0 m exactly (0.1 m unit)", b_const)

    def b_x():
        fr, sd = M.block_stats(mk(lambda a: (a.__setitem__((slice(None), slice(None)), -30), a.__setitem__((slice(0, 80), slice(None)), 30))))
        ok = bool((fr[0, :20] == 1).all() and (fr[0, 20:] == 0).all() and (sd[0, :20] == 3.0).all() and (sd[0, 20:] == -3.0).all())
        return ok, "x < 0 m drivable: coarse rows 0..19 inside, 20..39 outside, constant along y"
    C("B2 x half-plane (ix < 80 inside): rows 0..19 -> fraction 1 / +3.0 m, rows 20..39 -> 0 / -3.0 m (x forward is axis 0)", b_x)

    def b_y():
        fr, sd = M.block_stats(mk(lambda a: (a.__setitem__((slice(None), slice(None)), -30), a.__setitem__((slice(None), slice(64, 128)), 30))))
        ok = bool((fr[0, :, 16:] == 1).all() and (fr[0, :, :16] == 0).all() and (sd[0, :, 16:] == 3.0).all() and (sd[0, :, :16] == -3.0).all())
        return ok, "y > 0 (left) drivable: coarse columns 16..31 inside"
    C("B3 y half-plane (iy >= 64 inside, y left): columns 16..31 -> inside, 0..15 -> outside, constant along x (y left is axis 1)", b_y)

    def b_block():
        def f(a):
            a[:, :] = -10
            blk = a[12:16, 20:24]
            blk[:, :] = -10
            blk[0, :] = 10                                                          # one fine row of the block inside: 4 of 16 cells
        fr, sd = M.block_stats(mk(f))
        return abs(float(fr[0, 3, 5]) - 0.25) < 1e-7 and abs(float(sd[0, 3, 5]) - (-0.5)) < 1e-6, f"fraction {float(fr[0, 3, 5])}, block SDF {float(sd[0, 3, 5]):.6f} (want 0.25, -0.5)"
    C("B4 a partial block: 4 of 16 fine cells at +1.0 m, 12 at -1.0 m -> fraction 0.25, block mean -0.5 m", b_block)

    def b_clip():
        def f(a):
            a[:, :] = -10
            a[12:16, 20:24] = -10
            a[12, 20:24] = 127                                                       # 4 cells at +12.7 m (clipped to 6) and 12 at -1.0 m
            a[40:44, 40:44] = 127
        fr, sd = M.block_stats(mk(f))
        return abs(float(sd[0, 3, 5]) - 0.75) < 1e-6 and float(sd[0, 10, 10]) == 6.0, f"clip-then-mean {float(sd[0, 3, 5]):.4f} (want 0.75; mean-then-clip would give 2.425), saturated block {float(sd[0, 10, 10])}"
    C("B5 the SDF is clipped to +-6 m BEFORE the block mean (4 x +12.7 m, 12 x -1.0 m -> 0.75, not 2.425); a saturated block reads 6.0", b_clip)

    def b_signunit():
        fr, sd = M.block_stats(mk(lambda a: a.fill(-1)))
        fr2, sd2 = M.block_stats(mk(lambda a: a.fill(25)))
        return float(fr.max()) == 0.0 and abs(float(sd.mean()) + 0.1) < 1e-6 and float(fr2.min()) == 1.0 and abs(float(sd2.mean()) - 2.5) < 1e-6, "-1 LSB -> -0.1 m / fraction 0; +25 LSB -> +2.5 m"
    C("B6 sign and unit at the LSB: -1 -> -0.1 m (outside), +25 -> +2.5 m (inside)", b_signunit)

    def b_refuse():
        bad = 0
        for t in (torch.zeros(1, 160, 128, dtype=torch.float32), torch.zeros(1, 128, 160, dtype=torch.int8), torch.zeros(160, 128, dtype=torch.int8)):
            try:
                M.block_stats(t)
            except ValueError:
                bad += 1
        return bad == 3, f"{bad}/3 malformed labels refused"
    C("B7 a float label, a transposed [128,160] label and an unbatched label are REFUSED", b_refuse)

    def b_known():
        lab = torch.cat([mk(lambda a: a.fill(40)), mk(lambda a: a.fill(-120))], 0)
        garbage = lab.clone()
        garbage[1] = 127 - lab[1]
        valid = torch.tensor([True, False])
        t = M.build_targets(lab, valid)
        z = torch.zeros(2, 40, 32, requires_grad=True)
        s = torch.full((2, 40, 32), 1.0, requires_grad=True)
        l_a, _, _ = M.aux_loss(z, s, lab, valid)
        l_b, _, _ = M.aux_loss(z, s, garbage, valid)
        l_a.backward()
        one, _, _ = M.aux_loss(torch.zeros(1, 40, 32), torch.full((1, 40, 32), 1.0), lab[:1])
        ok = (bool(t.known[0].all()) and not bool(t.known[1].any()) and float(l_a) == float(l_b)
              and float(z.grad[1].abs().max()) == 0.0 and float(s.grad[1].abs().max()) == 0.0 and abs(float(l_a) - float(one)) < 1e-7
              and float(z.grad[0].abs().max()) > 0.0)
        return ok, f"known: labelled {bool(t.known[0].all())} / unlabelled {bool(t.known[1].any())}; unlabelled-frame grad max {float(z.grad[1].abs().max())}"
    C("B8 extent/validity mask: an unlabelled frame has no known cell, contributes EXACTLY 0 loss and 0.0 gradient whatever its label bytes, and does not dilute the mean", b_known)

    def b_loss():
        full = torch.cat([mk(lambda a: a.fill(127))] * 2, 0)
        z = torch.zeros(2, 40, 32)
        l, bce, hub = M.aux_loss(z, torch.full((2, 40, 32), 6.0), full)
        l0, _, _ = M.aux_loss(z, z, full, torch.tensor([False, False]))
        zz = torch.zeros(2, 40, 32, requires_grad=True)
        yy = torch.zeros(2, 40, 32, requires_grad=True)
        lz, _, _ = M.aux_loss(zz, yy, full, torch.tensor([False, False]))
        lz.backward()
        gz = torch.zeros(2, 40, 32, requires_grad=True)
        lg, _, _ = M.aux_loss(gz, torch.zeros(2, 40, 32), full)
        lg.backward()
        ok = (abs(float(bce) - math.log(2)) < 1e-6 and float(hub) == 0.0 and float(l0) == 0.0 and float(zz.grad.abs().max()) == 0.0
              and float(yy.grad.abs().max()) == 0.0 and float(gz.grad.max()) < 0.0)
        return ok, f"BCE {float(bce):.7f} (ln2 {math.log(2):.7f}); Huber at the clip {float(hub)!r}; no-label loss {float(l0)!r}"
    C("B9 loss identities: all-inside at logit 0 -> BCE == ln 2; SDF at +clip -> Huber == 0.0; no label -> loss and grads exactly 0.0; BCE pushes the logit UP", b_loss)

    def b_metrics():
        lab = mk(lambda a: (a.__setitem__((slice(None), slice(None)), -30), a.__setitem__((slice(0, 80), slice(None)), 30)))
        t = M.build_targets(lab)
        perfect = M.head_metrics((t.mask_frac - 0.5) * 20.0, t.sdf_m, lab)
        inverted = M.head_metrics(-(t.mask_frac - 0.5) * 20.0, -t.sdf_m, lab)
        return perfect["iou"] == 1.0 and perfect["acc"] == 1.0 and perfect["sdf_mae_m"] < 1e-6 and inverted["iou"] == 0.0 and inverted["acc"] == 0.0, f"{perfect} / {inverted['iou']}"
    C("B10 head_metrics (the S1 perception-proof read): a perfect prediction reads IoU 1.0 / MAE 0, an inverted one IoU 0.0", b_metrics)

    def b_sampler():
        sdf = torch.zeros(1, 40, 32)
        sdf[0] = torch.arange(40, dtype=torch.float32)[:, None] * 0.1 + torch.arange(32, dtype=torch.float32)[None, :] * 0.001     # x-dominant ramp
        cx = lambda i: M.X0 + (i + 0.5) * 2.0
        cy = lambda j: M.Y0 + (j + 0.5) * 2.0
        pts = torch.tensor([[[cx(7), cy(9)], [0.5 * (cx(7) + cx(8)), cy(9)], [cx(7), 0.5 * (cy(9) + cy(10))],
                             [M.X0 + 0.1, M.Y0 + 0.1], [M.X1, 0.0], [M.X1 - 1e-4, 0.0], [M.X0, 0.0], [0.0, M.Y1], [0.0, M.Y0 - 0.01], [100.0, 0.0]]])
        s, inb = M.sample_pred_sdf(sdf, pts)
        want = [sdf[0, 7, 9], 0.5 * (sdf[0, 7, 9] + sdf[0, 8, 9]), 0.5 * (sdf[0, 7, 9] + sdf[0, 7, 10]), sdf[0, 0, 0]]
        ok = all(abs(float(s[0, i]) - float(want[i])) < 1e-6 for i in range(4))
        ok &= [bool(v) for v in inb[0]] == [True, True, True, True, False, True, True, False, False, False]
        ok &= all(float(s[0, i]) == 6.0 for i in (4, 7, 8, 9)) and float(s[0, 5]) != 6.0
        return ok, f"cell-centre {float(s[0, 0]):.4f}, x-midpoint {float(s[0, 1]):.4f}, y-midpoint {float(s[0, 2]):.4f}, extent flags {[int(v) for v in inb[0]]}"
    C("B11 the H2 sampler: exact at cell centres, bilinear between them, x is axis 0 / y is axis 1, the outer half cell replicates, outside [X0,X1)x[Y0,Y1) -> +6 m and flag False", b_sampler)

    def ref_feed(sdf_np, traj_np):
        p = ND.at_10hz(ND.to_navsim(traj_np))[1:]
        cor = ND.corners_global(p, 0.0, 0.0, 0.0)
        cen = p[:, :2] + ND.REAR_AXLE_TO_CENTER * np.stack([np.cos(p[:, 2]), np.sin(p[:, 2])], -1)

        def samp(x, y):
            if not (-8.0 <= x < 72.0 and -32.0 <= y < 32.0):
                return 6.0
            u, v = np.clip((x + 8.0) / 2.0 - 0.5, 0, 39), np.clip((y + 32.0) / 2.0 - 0.5, 0, 31)
            i0, j0 = int(np.floor(u)), int(np.floor(v))
            i1, j1 = min(i0 + 1, 39), min(j0 + 1, 31)
            fu, fv = u - i0, v - j0
            return (1 - fu) * (1 - fv) * sdf_np[i0, j0] + fu * (1 - fv) * sdf_np[i1, j0] + (1 - fu) * fv * sdf_np[i0, j1] + fu * fv * sdf_np[i1, j1]
        sc = np.array([[samp(*c) for c in row] for row in cor.reshape(40, 4, 2)])
        sm = np.array([samp(*c) for c in cen])
        return np.array([sc.min(), sm.mean(), (sc < 0).mean()])

    def b_feed():
        rng = np.random.default_rng(5)
        xs, ys = np.meshgrid(np.arange(40), np.arange(32), indexing="ij")
        sdf = (4.5 * np.sin(xs / 5.0) * np.cos(ys / 4.0) + 0.7).astype(np.float32)
        t = 0.2 * np.arange(1, 21)
        trajs = []
        for v, kappa in ((9.0, 0.0), (12.0, 0.03), (6.0, -0.06), (25.0, 0.0), (15.0, 0.1), (1.0, 0.0)):
            th = np.cumsum(np.full(20, kappa * v * 0.2))
            x = np.cumsum(v * 0.2 * np.cos(th))
            y = np.cumsum(v * 0.2 * np.sin(th))
            trajs.append(np.stack([x, y, th], -1) + rng.normal(0, 0.02, (20, 3)))
        trajs = np.stack(trajs).astype(np.float32)
        feats, oob = M.feed_scalars(torch.from_numpy(sdf)[None], torch.from_numpy(trajs)[None])
        err = max(float(np.abs(feats[0, k].numpy() - ref_feed(sdf.astype(np.float64), trajs[k].astype(np.float64))).max()) for k in range(len(trajs)))
        REPORT["feed_ref_err"] = err
        return err < 5e-4 and float(oob[0, 3]) > 0.0 and float(oob[0, 5]) == 0.0, f"max |torch - independent numpy chain| = {err:.2e} over 6 trajectories incl. one leaving the extent (oob share {float(oob[0, 3]):.2f})"
    C("B12 the three H2 scalars (min corner SDF, mean centre SDF, share < 0) equal an INDEPENDENT numpy chain (navsim_dac poses + corners + numpy bilinear)", b_feed)

    def b_feed_sem():
        trajs = torch.tensor([[[[1.0 * (i + 1) * 0.2 * 8.0, 0.0, 0.0] for i in range(20)]]])                  # straight, 8 m/s
        inside = torch.full((1, 40, 32), 3.0)
        outside = torch.full((1, 40, 32), -3.0)
        f_in, _ = M.feed_scalars(inside, trajs)
        f_out, _ = M.feed_scalars(outside, trajs)
        half = torch.full((1, 40, 32), 3.0)
        half[:, :, :16] = -3.0                                                                                  # y < 0 outside: the footprint straddles y = 0
        f_h, _ = M.feed_scalars(half, trajs)
        ok = (abs(float(f_in[0, 0, 0]) - 3.0) < 1e-5 and abs(float(f_in[0, 0, 1]) - 3.0) < 1e-5 and float(f_in[0, 0, 2]) == 0.0
              and float(f_out[0, 0, 2]) == 1.0 and abs(float(f_out[0, 0, 0]) + 3.0) < 1e-5 and float(f_h[0, 0, 0]) < 0.0 and 0.2 < float(f_h[0, 0, 2]) < 0.8)
        return ok, f"inside (min, centre, share) = {f_in[0, 0].tolist()}, outside = {f_out[0, 0].tolist()}, straddling share {float(f_h[0, 0, 2]):.2f}"
    C("B13 H2 semantics: all-inside -> (+3, +3, 0); all-outside -> (-3, -3, 1); a footprint straddling the boundary -> min < 0 and a share strictly between", b_feed_sem)

    # ======================================================= C. gradient controls on the real REFe ========================
    def c_det():
        a, b = baseline()
        return all(bits_equal(a["grads"][n], b["grads"][n]) for n in a["grads"]) and bits_equal(a["ctx_grad"], b["ctx_grad"]), "no-head run twice"
    C("C1 baseline determinism: the no-head run twice gives bit-identical trunk gradients (otherwise the next control proves nothing)", c_det)

    def c_zero():
        a, _ = baseline()
        r = run(M, 0.0, "task+aux")
        head_zero = all(g is not None and float(g.abs().max()) == 0.0 for g in r["head"].values())
        eq_val = all((a["grads"][n] is None) == (r["grads"][n] is None) and (a["grads"][n] is None or equal(a["grads"][n], r["grads"][n])) for n in a["grads"])
        eq_bit = sum(1 for n in a["grads"] if not bits_equal(a["grads"][n], r["grads"][n]))
        ctx_eq = bits_equal(a["ctx_grad"], r["ctx_grad"])
        nz = sum(a["groups"].values()) > 0 and a["groups"]["lora"] > 0 and a["groups"]["scene_proj"] > 0
        REPORT["w0_bit_mismatch_tensors"] = eq_bit
        return head_zero and eq_val and eq_bit == 0 and ctx_eq and nz, (f"head grads exactly 0.0 on {len(r['head'])}/{len(r['head'])} tensors={head_zero}; trunk grads equal to the no-head run on "
                                                        f"{len(a['grads'])} tensors={eq_val} (bit-level differences: {eq_bit}); tapped-feature grad BIT-equal={ctx_eq}; control non-vacuous={nz}")
    C("C2 w_aux = 0: EVERY head gradient is exactly 0.0, and the trunk's gradients (all REFe tensors + the tapped features) equal the no-head run's", c_zero)

    def c_on():
        a, _ = baseline()
        r = run(M, 0.1, "task+aux")
        nzh = {n: float(g.abs().sum()) for n, g in r["head"].items()}
        all_nz = all(v > 0 for v in nzh.values())
        moved = {k: r["groups"][k] != a["groups"][k] for k in ("lora", "scene_proj", "pos3d_mlp")}
        return all_nz and all(moved.values()), f"head tensors with non-zero grad {sum(v > 0 for v in nzh.values())}/{len(nzh)} (min |g|_1 = {min(nzh.values()):.2e}); trunk groups changed vs no-head: {moved}"
    C("C3 w_aux = 0.1: EVERY head parameter tensor gets a NON-ZERO gradient and the trunk's LoRA / scene_proj / pos3d_mlp gradients move (the C2 equality is not vacuous)", c_on)

    def c_paths():
        r = run(M, 0.1, "aux", tap="visual")
        g = r["groups"]
        ok = g["lora"] > 0 and g["scene_proj"] > 0 and g["pos3d_mlp"] > 0 and g["reg_compress"] == 0.0 and g["traj_head"] == 0.0
        return ok, f"aux-only |g|_1: lora {g['lora']:.2e} scene_proj {g['scene_proj']:.2e} pos3d_mlp {g['pos3d_mlp']:.2e}; CONTROLS reg_compress {g['reg_compress']!r} traj_head {g['traj_head']!r}"
    C("C4 aux-only gradient (tap visual_ctx): NON-ZERO on LoRA / scene_proj / pos3d_mlp and EXACTLY 0.0 on reg_compress and traj_head", c_paths)

    def c_scene():
        r = run(M, 0.1, "aux", tap="scene")
        g = r["groups"]
        return g["reg_compress"] > 0 and g["scene_proj"] > 0 and g["lora"] > 0 and g["traj_head"] == 0.0, f"aux-only (tap scene_ctx): reg_compress {g['reg_compress']:.2e} (non-zero), traj_head {g['traj_head']!r}"
    C("C5 the registered scene tap: aux-only gradient reaches reg_compress and the LoRA, and is still EXACTLY 0.0 on traj_head", c_scene)

    def c_detach():
        r = run(M, 0.1, "aux", detach=True)
        g = r["groups"]
        head_learns = all(float(x.abs().sum()) > 0 for x in r["head"].values())
        return head_learns and g["lora"] == 0.0 and g["scene_proj"] == 0.0 and g["pos3d_mlp"] == 0.0, "head fed ctx.detach(): head still learns; trunk groups read exactly 0.0"
    C("C6 the probe works: a head fed ctx.detach() learns but the trunk groups read EXACTLY 0.0 (this is the arm C4's non-zero reading rules out)", c_detach)

    # ======================================================= D. H1 vs H2 ==================================================
    def d_h1():
        net, trajs, sctx, base = scorer_fixture()
        a1 = M.AuxDrivable(cfg_(feed="off", feed_init="normal", seed=5))
        a2 = M.AuxDrivable(cfg_(feed="off", feed_init="normal", seed=6))
        with torch.no_grad():
            _, s1 = a1.predict(sctx)
            _, s2 = a2.predict(sctx)
            o1, o2, o3 = a1.score(net, trajs, sctx, s1), a2.score(net, trajs, sctx, s2), a1.score(net, trajs, sctx, None)
        no_feed = a1.feed is None and all(n.startswith("head.") for n, _ in a1.named_parameters())
        return equal(o1, base) and equal(o2, base) and equal(o3, base) and not equal(s1, s2) and no_feed, f"H1 == net.score_trajectories bit for bit with head A, with a DIFFERENT head B, and with no prediction; heads differ={not equal(s1, s2)}; no feed module={no_feed}"
    C("D1 H1: the scorer's output is BIT-IDENTICAL to the model's own score_trajectories with the head, with a different head, and without one; H1 builds no feed", d_h1)

    def d_h1_grad():
        net, trajs, sctx, _ = scorer_fixture()
        a1 = M.AuxDrivable(cfg_(feed="off", feed_init="normal", seed=5))
        _, s1 = a1.predict(sctx)
        out = a1.score(net, trajs, sctx, s1)
        g = torch.autograd.grad(out.sum(), list(a1.head.parameters()), allow_unused=True)
        return all(x is None for x in g), f"{sum(x is None for x in g)}/{len(g)} head tensors have no autograd path from the score"
    C("D2 H1: NO autograd path from the score to any head parameter (G-DVB: torch.autograd.grad(..., allow_unused=True) all None)", d_h1_grad)

    def d_h2():
        net, trajs, sctx, base = scorer_fixture()
        a = M.AuxDrivable(cfg_(feed="on", feed_init="normal", seed=5))
        b = M.AuxDrivable(cfg_(feed="on", feed_init="normal", seed=5))
        with torch.no_grad():
            _, s1 = a.predict(sctx)
            o = a.score(net, trajs, sctx, s1)
            o_const = a.score(net, trajs, sctx, torch.full_like(s1, 6.0))
            o_other = a.score(net, trajs, sctx, torch.roll(s1, 3, dims=1) * 1.7)
            o_b = b.score(net, trajs, sctx, s1)
        d_base, d_const, d_other = float((o - base).abs().max()), float((o - o_const).abs().max()), float((o - o_other).abs().max())
        return d_base > 1e-5 and d_const > 1e-5 and d_other > 1e-5 and equal(o, o_b), f"H2 vs base max|d|={d_base:.3e}; vs a constant field {d_const:.3e}; vs a different prediction {d_other:.3e}; same seed reproducible={equal(o, o_b)}"
    C("D3 H2: the scorer's output is NOT the base scorer's, and it DEPENDS on the predicted SDF (a different prediction gives a different score)", d_h2)

    def d_h2_grad():
        net, trajs, sctx, _ = scorer_fixture()
        a = M.AuxDrivable(cfg_(feed="on", feed_init="zero", seed=5))
        _, s1 = a.predict(sctx)
        out = a.score(net, trajs, sctx, s1)
        gh = torch.autograd.grad(out.sum(), list(a.head.parameters()), allow_unused=True, retain_graph=True)
        gp = torch.autograd.grad(out.sum(), a.feed.proj.weight)[0]
        return all(x is None for x in gh) and float(gp.abs().sum()) > 0, f"head tensors with a path from the score: {sum(x is not None for x in gh)}/{len(gh)} (must be 0); feed.proj grad |g|_1 {float(gp.abs().sum()):.2e}"
    C("D4 H2: the feed is DETACHED (no autograd path score -> head) while the scorer's own feed projection does receive gradient", d_h2_grad)

    def d_h2_zero():
        net, trajs, sctx, base = scorer_fixture()
        a = M.AuxDrivable(cfg_(feed="on", feed_init="zero", seed=5))
        with torch.no_grad():
            _, s1 = a.predict(sctx)
            o0 = a.score(net, trajs, sctx, s1)
            a.feed.proj.weight.add_(0.05)
            o1 = a.score(net, trajs, sctx, s1)
        return equal(o0, base) and not equal(o1, base), f"zero-init H2 == base scorer bit for bit ({equal(o0, base)}); after perturbing the projection max|d|={float((o1 - base).abs().max()):.2e}"
    C("D5 H2 with feed_init='zero' starts as the base scorer's EXACT function (so the u400 fine-tune begins unchanged) and moves once the projection moves", d_h2_zero)

    def d_h2_wide():
        net, trajs, sctx, _ = scorer_fixture()
        a = M.AuxDrivable(cfg_(feed="on", feed_init="normal", seed=5))
        with torch.no_grad():
            _, s1 = a.predict(sctx)
            o = a.score(net, trajs, sctx, s1)
            feats, _ = M.feed_scalars(s1, trajs)
            l0 = net.score_q_mlp[0]
            W63 = torch.cat([l0.weight, a.feed.proj.weight], 1)
            h = F.linear(torch.cat([trajs.flatten(2), feats], -1), W63, l0.bias)
            for m in list(net.score_q_mlp)[1:]:
                h = m(h)
            for blk in net.score_dec:
                h = blk(h, sctx)
            ref = net.score_head(h)
        err = float((o - ref).abs().max())
        return err < 1e-5 and tuple(W63.shape) == (256, 63), f"additive projection vs an explicit 63-wide first layer [W60 | proj]: max|d| = {err:.2e}"
    C("D6 the additive feed projection IS a 63-wide first layer whose last three columns are proj.weight (SPEC 2.4's 60 -> 63), to 1e-5", d_h2_wide)

    def d_compat():
        net, _, _, _ = scorer_fixture()
        a = M.AuxDrivable(cfg_(feed="on"))
        mine, theirs = set(a.state_dict()), set(net.state_dict())
        ok = not (mine & theirs) and all(k.startswith(("head.", "feed.")) for k in mine) and tuple(net.score_q_mlp[0].weight.shape) == (256, 60)
        return ok, f"{len(mine)} new keys, none colliding with the base checkpoint's {len(theirs)}; score_q_mlp.0.weight stays [256, 60]"
    C("D7 checkpoint compatibility: the head/feed state lives under its own keys and the base score_q_mlp.0.weight stays [256, 60] (u400 loads strictly)", d_compat)

    # ======================================================= E. the label source ==========================================
    def tmp_store(manifest_edit=None, shard_log=None, flip_last_byte=False):
        d = tempfile.mkdtemp(prefix="auxstore_")
        man = json.load(open(os.path.join(STORE, "manifest.json"), encoding="utf-8"))
        if manifest_edit:
            manifest_edit(man)
        json.dump(man, open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"))
        if shard_log:
            os.makedirs(os.path.join(d, "shards"))
            data = bytearray(open(os.path.join(STORE, "shards", f"{shard_log}.aux1"), "rb").read())
            if flip_last_byte:
                data[-1] ^= 0xFF
            open(os.path.join(d, "shards", f"{shard_log}.aux1"), "wb").write(bytes(data))
        return d

    def refused(d, **kw):
        try:
            M.AuxLabelSource(d, **kw)
        except M.StoreRefused:
            return True
        finally:
            shutil.rmtree(d, ignore_errors=True)
        return False

    def resha(man):
        man["store_sha256"] = hashlib.sha256("\n".join(f"{k}:{v['sha256']}" for k, v in sorted(man["files"].items())).encode()).hexdigest()

    store_ok = os.path.isfile(os.path.join(STORE, "manifest.json"))
    if not store_ok and NO_STORE_OK:
        REPORT["store_skipped"] = True
        return R

    def e_pin():
        src = M.AuxLabelSource(STORE)
        wrong_pin = refused_dir_free = False
        try:
            M.AuxLabelSource(STORE, pinned_sha256="0" * 64)
        except M.StoreRefused:
            wrong_pin = True

        def drop_one(man):
            man["files"].pop(sorted(man["files"])[0])
            resha(man)                                                          # SELF-CONSISTENT: only the pin can refuse this store
        alt = refused(tmp_store(drop_one))
        return src.manifest["store_sha256"] == M.PINNED_STORE_SHA256 and wrong_pin and alt, f"pinned store accepted; wrong pin refused={wrong_pin}; a self-consistent store that is NOT the pinned one refused={alt}"
    C("E1 the label source accepts the pinned store and REFUSES one whose store_sha256 is not the pinned value (even a self-consistent one)", e_pin)

    def e_consist():
        def keep_sha_change_file(man):
            man["files"][sorted(man["files"])[0]]["sha256"] = "f" * 64          # the pinned sha is kept, the files map is altered
        def keys(man):
            man["counts"]["stored_keys"] += 1
        def leak(man):
            man["leakage"]["token_intersection"] = 1
        r = [refused(tmp_store(keep_sha_change_file)), refused(tmp_store(keys)), refused(tmp_store(leak)), refused(tmp_store(lambda m: m.pop("raster")))]
        return all(r), f"refused: files-map altered under the pinned sha={r[0]}, stored_keys changed={r[1]}, navtest leak reported={r[2]}, raster spec missing={r[3]}"
    C("E2 the manifest must be internally consistent: files map hashes to store_sha256, stored_keys == 103,139, no navtest leak, raster spec intact", e_consist)

    def e_shard_bytes():
        log, tok = real_keys()[0]
        d = tmp_store(shard_log=log, flip_last_byte=True)
        try:
            src = M.AuxLabelSource(d)
            try:
                src.label(log, tok)
                got = False
            except M.StoreRefused:
                got = True
        finally:
            shutil.rmtree(d, ignore_errors=True)
        d2 = tmp_store(shard_log=log)
        try:
            lab = M.AuxLabelSource(d2).label(log, tok)
            clean = lab is not None and np.array_equal(lab, ALS.read_label(STORE, log, tok))
        finally:
            shutil.rmtree(d2, ignore_errors=True)
        return got and clean, f"a flipped byte in a shard is refused={got}; the untouched copy reads the same label={clean}"
    C("E3 shard bytes are verified against the manifest on load (content, not presence): one flipped byte is REFUSED", e_shard_bytes)

    def e_once():
        src = M.AuxLabelSource(STORE)
        keys = real_keys()
        by_log = {}
        for k in keys:
            by_log.setdefault(k[0], []).append(k)
        lg = next(iter(by_log))
        for k in by_log[lg] * 3:
            src.label(*k)
        one = src.stats["shard_reads"]
        for k in keys:
            src.label(*k)
        for k in keys:
            src.label(*k)
        reads = src.stats["shard_reads"]
        bounded = M.AuxLabelSource(STORE, max_logs=2)
        for k in keys:
            bounded.label(*k)
        return one == 1 and reads == len(by_log) and len(bounded._cache) <= 2, f"{len(by_log[lg]) * 3} reads from one log -> {one} shard read; {2 * len(keys)} more reads over {len(by_log)} logs -> {reads} shard reads in total; bounded cache holds {len(bounded._cache)} logs"
    C("E4 one shard read per log (cached), not per token; max_logs bounds the cache", e_once)

    def e_missing():
        src = M.AuxLabelSource(STORE)
        log = real_keys()[0][0]
        lab, valid = src.labels([real_keys()[0], (log, "0" * 16), ("no_such_log", "ff" * 8)])
        return valid.tolist() == [True, False, False] and int(lab[1].abs().sum()) == 0 and int(lab[2].abs().sum()) == 0 and src.stats["missing_key"] == 1 and src.stats["missing_log"] == 1, f"valid={valid.tolist()} stats={src.stats}"
    C("E5 a key with no label (unknown token, unknown log) reads zeros with valid False and is counted, never raised", e_missing)

    def e_real():
        src = M.AuxLabelSource(STORE)
        keys = real_keys()
        lab, valid = src.labels(keys)
        rt = all(np.array_equal(lab[i].numpy(), real_labels()[k]) for i, k in enumerate(keys)) and bool(valid.all()) and lab.dtype == torch.int8
        t = M.build_targets(lab, valid)
        ref = [np_stats(real_labels()[k]) for k in keys]
        err = max(max(float(np.abs(t.mask_frac[i].numpy() - r[0]).max()), float(np.abs(t.sdf_m[i].numpy() - r[1]).max())) for i, r in enumerate(ref))
        REPORT["n_real_keys"], REPORT["n_real_logs"], REPORT["real_target_err"] = len(keys), len({k[0] for k in keys}), err
        return len(keys) >= 32 and rt and err < 2e-6, f"{len(keys)} real keys / {REPORT['n_real_logs']} logs / 4 maps: labels equal aux_label_store.read_label byte for byte={rt}; targets equal an independent numpy block-statistic, max |d| = {err:.1e}"
    C("E6 >= 32 REAL stored keys: the source's labels round-trip the store exactly and the targets equal an independent numpy block statistic", e_real)

    def e_agree():
        src = M.AuxLabelSource(STORE)
        keys = real_keys()
        lab, valid = src.labels(keys)
        t = M.build_targets(lab, valid)
        mask = (t.mask_frac >= 0.5)
        fine = lab.float()
        a_blockmean = float((mask == (t.sdf_m > 0)).float().mean())
        b_stride = float((mask == (fine[:, 2::4, 2::4] > 0)).float().mean())
        c_pool = float((mask == (F.avg_pool2d(fine[:, None] * 0.1, 4)[:, 0] > 0)).float().mean())
        pure = float(((t.mask_frac == 0) | (t.mask_frac == 1)).float().mean())
        inside = float(t.mask_frac.mean())
        REPORT.update(agree_blockmean=a_blockmean, agree_stride=b_stride, agree_pool=c_pool, pure_share=pure, inside_share=inside)
        return a_blockmean >= 0.97 and b_stride >= 0.95 and c_pool >= 0.97, (f"mask(fraction >= 0.5) vs sign(block-mean clipped SDF) {a_blockmean:.4f} | vs sign(SDF) at the block's stride-4 fine cell {b_stride:.4f} | "
                                                                           f"vs sign(avg-pooled SDF) {c_pool:.4f} | pure blocks (fraction 0 or 1) {pure:.4f} | inside share {inside:.3f}")
    C("E7 the block-statistic mask agrees with sign(SDF) down-sampled three ways on the real keys (bars fixed before the run: 0.97 / 0.95 / 0.97; measured values reported)", e_agree)

    def e_parse():
        log = real_keys()[0][0]
        p = os.path.join(STORE, "shards", f"{log}.aux1")
        h0, b0 = ALS.read_shard(p)
        h1, b1 = M._parse_shard_bytes(open(p, "rb").read(), ALS)
        return h0 == h1 and b0 == b1, f"{len(b0)} blobs identical to aux_label_store.read_shard"
    C("E8 the shard parser (parses the very bytes that were hashed) equals aux_label_store.read_shard", e_parse)

    def e_derange():
        keys = [f"k{i}" for i in range(500)]
        sh = M.derangement(keys, 7)
        again = M.derangement(keys, 7)
        return all(a != b for a, b in zip(keys, sh)) and sorted(sh) == sorted(keys) and sh == again and sh != M.derangement(keys, 8), "0 fixed points, same multiset, deterministic per seed"
    C("E9 the label-shuffled control's key map is a DERANGEMENT: 0 fixed points, identical multiset, reproducible per seed", e_derange)

    if heavy:
        def e_whole():
            src = M.AuxLabelSource(STORE)
            v = src.verify_all()
            ks = src.all_keys()
            # the navtest list the store was checked against, found by CONTENT: the manifest records the build machine's path (absent on a pod),
            # so AUX_NAVTEST may name another location, but the file must carry the manifest's sha256 (2026-10-14: G-SUITE failed on the pod here)
            nt_rec = src.manifest["runs"][0]["inputs"]["navtest"]
            nt_path = os.environ.get("AUX_NAVTEST") or nt_rec["path"]
            if not os.path.isfile(nt_path):
                return False, f"navtest list not found at {nt_path} (set AUX_NAVTEST to a copy with sha256 {nt_rec['sha256'][:12]}...)"
            nt_raw = open(nt_path, "rb").read()
            nt_sha = hashlib.sha256(nt_raw).hexdigest()
            nt_ok, nt_how = navtest_identity(nt_raw, nt_rec["sha256"])
            if not nt_ok:
                return False, f"{nt_path} has sha256 {nt_sha[:12]}..., not the manifest's {nt_rec['sha256'][:12]}...: not the navtest the store was checked against ({nt_how})"
            nt_tokens, nt_logs = ALS.load_navtest(nt_path)
            inter = len({t for _, t in ks} & nt_tokens)
            REPORT.update(store_files=v["files"], store_mismatch=v["mismatch"], store_keys=len(ks), navtest_tokens=len(nt_tokens), navtest_intersection=inter)
            return v["mismatch"] == 0 and len(ks) == len(set(ks)) == M.PINNED_STORE_KEYS and inter == 0, f"{v['files']} shard files re-hashed, {v['mismatch']} mismatch; {len(ks):,} unique keys; keys ∩ {len(nt_tokens):,} navtest tokens = {inter}"
        C("E10 WHOLE STORE by content: every shard file re-hashes to the manifest, 103,139 unique keys, key set ∩ navtest tokens = ∅ (G-LIVE no-leakage)", e_whole)

        def e_shuffle():
            plain = M.AuxLabelSource(STORE)
            shuf = M.AuxLabelSource(STORE, shuffle_seed=11)
            km = shuf.key_map()
            ks = list(km)
            fixed = sum(1 for k in ks if km[k] == k)
            ok = fixed == 0 and sorted(km.values()) == ks
            for k in real_keys()[:6]:
                got = shuf.label(*k)
                want = plain.label(*km[k])
                ok &= got is not None and np.array_equal(got, want)
            return ok, f"{len(ks):,} keys, {fixed} fixed points, values are a permutation of the keys; the shuffled source returns the label of the mapped key"
        C("E11 the shuffle arm on the REAL key set: 0 fixed points over 103,139 keys, a permutation of the same keys, and labels follow the map", e_shuffle)

        def e_pickle():
            src = M.AuxLabelSource(STORE)
            src.label(*real_keys()[0])
            s2 = pickle.loads(pickle.dumps(src))
            return np.array_equal(s2.label(*real_keys()[0]), src.label(*real_keys()[0])), "picklable (DataLoader workers)"
        C("E12 the source is picklable and works after a round trip (DataLoader workers)", e_pickle)

    # ======================================================= G. facts used by the report ==================================
    return R


# ----------------------------------------------------------------------------------------------------------------------
# F. the mutations of the shipped source text
# ----------------------------------------------------------------------------------------------------------------------
MUTATIONS = [
    # (name, class it models, exact old text, new text)
    ("x/y swapped in the target builder", "REQUIRED", "s = s.view(B, NX_F // b, b, NY_F // b, b)", "s = s.transpose(1, 2).reshape(B, NX_F // b, b, NY_F // b, b)"),
    ("sign flipped (mask)", "REQUIRED", "frac = (s > 0).float().mean((2, 4))", "frac = (s < 0).float().mean((2, 4))"),
    ("sign flipped (SDF)", "REQUIRED", "sdf = s.clamp(-sdf_scale, sdf_scale).mean((2, 4))", "sdf = -s.clamp(-sdf_scale, sdf_scale).mean((2, 4))"),
    ("int8 unit forgotten", "REQUIRED", "s = label_i8.float() * SDF_UNIT_M", "s = label_i8.float()"),
    ("block size 2 instead of 4", "REQUIRED", "b = int(round(cell / FINE_RES))", "b = 2"),
    ("store sha pin ignored", "REQUIRED", "if got != pinned_sha256:", "if False:"),
    ("H1 leaking the prediction into the scorer", "REQUIRED", 'self.feed = DrivableFeed(cfg.width, cfg.feed_init, cfg.seed) if cfg.feed == "on" else None',
     'self.feed = DrivableFeed(cfg.width, "normal", cfg.seed)'),
    ("mean-then-clip instead of clip-then-mean", "extra", "sdf = s.clamp(-sdf_scale, sdf_scale).mean((2, 4))", "sdf = s.mean((2, 4)).clamp(-sdf_scale, sdf_scale)"),
    ("manifest self-consistency not checked", "extra", "if again != got:", "if False:"),
    ("shard bytes not hashed", "extra", 'if self.verify_shards and hashlib.sha256(data).hexdigest() != ent["sha256"]:', "if False:"),
    ("H2 feed not detached from the head", "extra", "sdf = sdf_pred.detach().float()", "sdf = sdf_pred.float()"),
    ("out-of-extent corner keeps its raw value", "extra", "s = torch.where(in_ext, s, torch.full_like(s, sdf_scale))", "s = s"),
    ("sampler reads the SDF with x/y swapped", "extra", "img = sdf_pred.transpose(1, 2).unsqueeze(1)", "img = sdf_pred.unsqueeze(1)"),
    ("unlabelled frames counted (valid ignored)", "extra", "known = valid.to(frac.device).bool()[:, None, None].expand_as(frac)", "known = torch.ones_like(frac, dtype=torch.bool)"),
    ("loss weight ignored", "extra", "return l_task + self.w_aux * l_aux", "return l_task + l_aux"),
    ("head fed ctx.detach() (trunk gets nothing)", "extra", "return self.head(ctx)", "return self.head(ctx.detach())"),
    ("head built from the global RNG stream", "extra", "with torch.random.fork_rng(devices=[]):\n        torch.manual_seed(int(cfg.seed))\n        return DrivableAuxHead(",
     "if True:\n        return DrivableAuxHead("),
    ("feed share counts the wrong sign", "extra", "(s_corner < 0).float().mean(-1)", "(s_corner > 0).float().mean(-1)"),
    ("derangement with fixed points", "extra", "out[order[pos]] = keys[order[(pos + 1) % n]]", "out[order[pos]] = keys[order[pos]]"),
]


# The navtest list the store was checked against is recorded in the store's manifest by the sha256 of the BUILD MACHINE's file: a Windows `autocrlf` checkout (CRLF, 765,337 bytes), not the git blob
# (LF, 753,185 bytes, our ship standard since 2026-10-15).  The two files are the same list (12,152 lines; stripping every CR from either gives the same bytes), so the identity is decided on the
# CR-stripped bytes: NAVTEST_LF_SHA maps the manifest's recorded (CRLF) sha256 to the sha256 of the list with CR removed.  A genuinely different list matches neither (test_s1c.py).
# raw/2026-10-16-s1c-build/: found because G-SUITE's E10 refused an LF ship exactly as the chain's `data` stage did (s1aux_pod.sh), 25 minutes into the first gate instead of at the start.
NAVTEST_LF_SHA = {"0b9c63895d4e8da4205e194c90379cbf986badd1baf8beab948adb3743a33700": "3ba1c96488f157fd2ca7c57c83abe439b4c9680ebdb1eadf65c34ac323d42d44"}


def navtest_identity(raw: bytes, manifest_sha256: str) -> tuple:
    """(ok, how): `raw` is the manifest's navtest list iff its sha256 is the recorded one (the build machine's bytes) or its CR-stripped sha256 is the registered LF form of that record."""
    if hashlib.sha256(raw).hexdigest() == manifest_sha256:
        return True, "raw sha256 equals the manifest's"
    want = NAVTEST_LF_SHA.get(manifest_sha256)
    if want is not None and hashlib.sha256(raw.replace(b"\r", b"")).hexdigest() == want:
        return True, f"CR-stripped sha256 {want[:12]} (the manifest's {manifest_sha256[:12]} is the CRLF checkout of the same list)"
    return False, f"neither the raw sha256 nor the CR-stripped one matches the manifest's {manifest_sha256[:12]}"


def main() -> int:
    src0 = open(SRC, encoding="utf-8").read()
    import aux_drivable_head as base
    print("== unmutated module ==")
    R = suite(base, heavy=True, verbose=True)
    bad = R.failing()
    if bad:
        print(f"\nZZTEST_FAIL  {len(bad)} failing: {bad}")
        return 1
    n_checks = len(R.rows)
    rep = dict(REPORT)                                        # snapshot: the mutants' suites overwrite REPORT
    print(f"\n== {len(MUTATIONS)} RED mutations of the shipped source (each must fail at least one check) ==")
    n = n_checks
    required_seen = set()
    for i, (name, cls, old, new) in enumerate(MUTATIONS):
        assert src0.count(old) == 1, f"mutation '{name}': old text occurs {src0.count(old)} times (need exactly 1)"
        try:
            mod = load(src0.replace(old, new), f"aux_drivable_head_mut{i}")
            fails = suite(mod, heavy=False).failing()
        except Exception as e:  # noqa: BLE001
            fails = [f"module/suite setup raised {type(e).__name__}: {str(e)[:60]}"]
        if not fails:
            print(f"ZZTEST_FAIL  mutation '{name}' stayed GREEN: the suite cannot see it")
            return 1
        if cls == "REQUIRED":
            required_seen.add(name)
        print(f"  RED  {name:46s} {len(fails):2d} failing: {', '.join(f.split(' ')[0] for f in fails[:6])}")
        n += 1
    assert len(required_seen) == 7, required_seen
    print("\n== measured ==")
    for k in ("params", "proto_params", "n_real_keys", "n_real_logs", "real_target_err", "agree_blockmean", "agree_stride", "agree_pool", "pure_share", "inside_share",
              "feed_ref_err", "w0_bit_mismatch_tensors", "store_files", "store_mismatch", "store_keys", "navtest_tokens", "navtest_intersection"):
        if k in rep:
            print(f"  {k:26s} {rep[k]}")
    print(f"\nZZTEST_PASS {n}  ({n_checks} checks green on the unmutated module + {len(MUTATIONS)} RED mutations, the 7 required classes among them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
