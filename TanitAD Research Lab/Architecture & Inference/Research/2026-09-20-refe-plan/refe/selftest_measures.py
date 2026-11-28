#!/usr/bin/env python3
"""VALIDITY PROOF for measures.py (M3 separate clipping, M4a slowed-target twins, M4b anchored slow slots).

Everything runs on TEMPORARY COPIES of refe/*.py: `orig/` (today's code, untouched) and `patched/` (the same
files with measures_hooks.patch + measures_consumers.patch applied by `git apply`). The live files are never
imported from their own directory and never modified; the pod's copy is never touched. CPU only.

The REAL trainer (`train.main`) runs end-to-end in child processes with a tiny backbone shim (width 64, depth
2, 32x64 images, 4 cameras, 64 proposals, 20 steps) -- the loop, the losses, the clip, the optimiser, the
checkpoint and the resume identity are the shipped code, only the tensor sizes are small. Gradients are read
AFTER the clip, at AdamW.step.

Every check carries a DELIBERATE-REGRESSION arm that must go RED, and controls that must read a known value
exactly (the rule: a check that shares the defect it checks is green forever):

  T1  patch        applies to the recorded base blobs; the patched files compile
  T2  OFF == today real trainer, on-policy scorer, accum 2: every post-clip gradient, the final weights and
                   the AdamW state, orig vs patched-all-OFF: max|diff| EXACTLY 0.0. Controls: orig vs orig
                   (determinism) 0.0; orig seed 0 vs seed 1 > 0 (the comparison can see a difference);
                   `--log-grad-norms` also 0.0 (read-only diagnostic)
  T3  M3           --clip-split, score loss x100 (score_w 0.1 -> 10): trajectory-group post-clip gradients AND
                   weights bit-identical over every step; scorer group differs (positive control). MUTATIONS:
                   flag OFF -> trajectory gradients change; a wrong partition (score_head in the trajectory
                   group) -> they change
  T4  M3 groups    name partition on the REAL ViT-L 4-camera config (every trainable tensor in exactly one
                   group); the same partition DERIVED FROM GRADIENT FLOW (score loss alone / trajectory loss
                   alone) on the tiny model. MUTATION: detach_scorer_context=False -> the score loss reaches the
                   trajectory group, and --clip-split refuses
  T5  M4b units    anchor profiles vs hand-computed literals; re-timing on analytic paths (accelerating line,
                   U-turn arc, extrapolation, identity); the loss's closest-anchor choice vs hand-computed
                   literals. MUTATIONS: three broken re-timings (no origin, time-index, no heading wrap) each fail
  T6  M4b hook     un-installed: forward + loss bit-identical to today's model; installed: free slots
                   bit-identical, reserved slots never travel further than their anchor, state dict / trainable
                   params / frozen fingerprint unchanged, the scorer sees the anchored fan
  T7  M4b drift    N real-trainer steps on FAST-ONLY targets: ON keeps every reserved slot within its anchor
                   (exceedance <= 1 mm); OFF lets the same slots drift (>= 10 m). MUTATION: the loss kept but the
                   forward re-timing removed -> drift
  T8  M4a          factor 1.0 twins reproduce the target BIT-exactly (synthetic + 2,000 real navtrain rows);
                   twins == slow_copy bitwise; every twin passes grow_assemble._target_ok; twins load through the
                   REAL TargetBank and carry their source's on-policy set. MUTATION/guard: without --slow-twins
                   the patched loader refuses; the UNPATCHED loader absorbs them silently (the hazard)
  T9  mid-run      resume today's checkpoint with the patched trainer, all OFF: bit-identical to today's resume;
                   switching a measure without --declare-change is refused, with it accepted + logged; M4b's
                   checkpoint meta makes ckpt_io.load_for_inference install the hook; under --grow the unpatched
                   trainer absorbs a twin file at the next epoch boundary (MEASURED), the patched one refuses it
                   unless --slow-twins, and with it the twins join exactly at the boundary

    python refe/selftest_measures.py [--keep] [--results <json>]      # prints ZZMEASURES_OK / ZZMEASURES_FAIL
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent                     # refe/
PKG = HERE.parent
PATCHES = ("measures_hooks.patch", "measures_consumers.patch")
REAL_BANK = Path("D:/Projects/TanitAD/data/refe_navtrain/targets_rank0.jsonl")
DEFAULT_RESULTS = PKG / "raw" / "2026-09-27-training-measures" / "selftest_measures.json"
TINY = dict(width=64, depth=2, heads=4, img_h=32, img_w=64, lora_rank=4, n_registers=2,
            dec_width=32, dec_depth=2, dec_heads=4, reg_heads=4)
CAMS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")


# ================================================================================================ child side
def _setup(root):
    sys.path.insert(0, root)
    import torch
    torch.set_num_threads(2)
    import model as M
    M.BACKBONES["tiny"] = dict(width=64, depth=2, heads=4, weights="none", published_pdms=None)
    orig = M.REFeConfig.for_backbone

    def fb(name, **kw):
        if name == "tiny":
            return M.REFeConfig(backbone="tiny", **TINY, **kw)
        return orig(name, **kw)
    M.REFeConfig.for_backbone = staticmethod(fb)
    return M, torch


def _mutate(names):
    import measures as MS
    for m in names or []:
        if m == "m3_bad_partition":            # score_head's tensors land in the TRAJECTORY group
            MS.SCORER_PREFIXES = ("score_q_mlp.", "score_dec.")
        elif m == "m4b_no_retime":             # the forward re-timing deleted; the anchored LOSS kept
            import torch

            def no_retime(self, traj, ego):
                t32 = traj.float()
                self.last_raw = t32.index_select(1, torch.tensor(self.cfg.slots, device=traj.device))
                return t32                     # everything as built EXCEPT the re-timing line
            MS.AnchoredSlowSlots.forward = no_retime
        elif m in ("m4b_loss_through_hook", "m4b_loss_through_hook_fullgrad"):
            # the two REJECTED designs: regress the ANCHORED output (gradient through the re-timing)
            orig_loss = MS.anchored_wta_loss

            def through_hook(*a, **k):
                k["raw_reserved"] = None
                return orig_loss(*a, **k)
            MS.anchored_wta_loss = through_hook
            if m.endswith("fullgrad"):
                orig_rt = MS.retime_to_profile
                MS.retime_to_profile = lambda traj, sigma, eps=1e-6, detach_weights=False: orig_rt(traj, sigma, eps)
        else:
            raise ValueError(m)


def child_trainer(spec):
    M, torch = _setup(spec["root"])
    cap = []
    if spec.get("capture"):
        orig_step = torch.optim.AdamW.step

        def step(self, *a, **k):
            cap.append([None if p.grad is None else p.grad.detach().clone()
                        for g in self.param_groups for p in g["params"]])
            return orig_step(self, *a, **k)
        torch.optim.AdamW.step = step
    if spec.get("mutate"):
        _mutate(spec["mutate"])
    import train as T
    names = [n for n, p in M.REFe(M.REFeConfig.for_backbone("tiny")).named_parameters() if p.requires_grad]
    sys.argv = ["train.py"] + spec["argv"]
    rc, msg = 0, None
    try:
        T.main()
    except SystemExit as e:
        if isinstance(e.code, int) or e.code is None:
            rc = int(e.code or 0)
        else:
            rc, msg = 1, str(e.code)
    torch.save({"rc": rc, "msg": msg, "grads": cap, "names": names}, spec["out"])


def child_eval_slots(spec):
    """Load a finished tiny run's model_final.pt through ckpt_io.load_for_inference (the CONSUMER path, which
    installs M4b from the checkpoint's meta), run the real forward on the bank's samples, and measure the
    reserved slots against their anchor profiles."""
    M, torch = _setup(spec["root"])
    if spec.get("mutate"):
        _mutate(spec["mutate"])
    import ckpt_io
    import measures as MS
    import train as T
    cfg = M.REFeConfig.for_backbone("tiny")
    model = M.REFe(cfg).eval()
    meta = {}
    fmt = ckpt_io.load_for_inference(model, spec["ckpt"], meta_out=meta)
    installed = getattr(model, "slow_slots", None) is not None
    import inspect
    kw = {"slow_twins": True} if "slow_twins" in inspect.signature(T.TargetBank).parameters else {}
    ds = T.TargetBank(spec["bank"], None, cfg, synthetic=True, **kw)
    n = min(len(ds), 16)
    items = [ds[i] for i in range(n)]
    img = torch.stack([x[0] for x in items])
    ego = torch.stack([x[1] for x in items])
    goal = torch.stack([x[2] for x in items])
    tgt = torch.stack([x[3] for x in items])
    K = 8
    ss = ((meta.get("measures") or {}).get("slow_slots") or {})
    # the reserved slots the CHECKPOINT declares (a switch may name any 8); an OFF run: the default last 8
    slots = list(ss["slots"]) if ss else list(range(cfg.n_proposals - K, cfg.n_proposals))
    prof = MS.parse_profiles(ss["profiles"]) if ss else MS.parse_profiles(MS.DEFAULT_SLOW_PROFILES)
    with torch.no_grad():
        traj, _ = model(img, ego, goal)
        raw = None
        if installed:
            hook = model.slow_slots
            del model.slow_slots
            raw, _ = model(img, ego, goal)
            model.slow_slots = hook
    S = MS.anchor_profiles(ego[:, 6].float(), prof, cfg.horizon_steps).double()          # [B, K, T]
    out = traj.double()[:, slots]                                                        # [B, K, T, 3]
    L = MS.progress_profile(out)
    anc = torch.stack([S, torch.zeros_like(S)], -1)
    free = [i for i in range(cfg.n_proposals) if i not in slots]
    fd = (traj.double()[:, free, :, :2] - tgt.double()[:, None, :, :2]).abs().sum(-1).mean(-1)   # [B, F]
    res = {"fmt": fmt, "installed": installed, "meta_measures": meta.get("measures"), "n": n, "slots": slots,
           "exceed_max_m": float((L - S).max()), "dev_mean_m": float((L - S).abs().mean()),
           "euclid_to_straight_anchor_max_m": float((out[..., :2] - anc).norm(dim=-1).max()),
           "reserved_len_4s_mean_m": float(L[..., -1].mean()), "anchor_len_4s_mean_m": float(S[..., -1].mean()),
           "target_len_4s_mean_m": float(MS.progress_profile(tgt.double())[:, -1].mean()),
           "best_free_wta_m": float(fd.min(dim=1).values.mean())}
    if raw is not None:
        Lr = MS.progress_profile(raw.double()[:, slots])
        res["raw_reserved_exceed_max_m"] = float((Lr - S).max())
        res["raw_reserved_len_4s_max_m"] = float(Lr[..., -1].max())
    json.dump(res, open(spec["out"], "w"), indent=1)


def _polyline_distance(traj, q):
    """[T, 3] polyline with the origin prepended; [K, 2] points -> [K] distances (independent numpy code)."""
    import numpy as np
    p = np.concatenate([np.zeros((1, 2)), np.asarray(traj, np.float64)[:, :2]], 0)
    a, b = p[:-1], p[1:]
    out = []
    for pt in np.asarray(q, np.float64)[:, :2]:
        ab = b - a
        s = np.clip(((pt - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-300), 0, 1)
        out.append(float(np.sqrt((((a + s[:, None] * ab) - pt) ** 2).sum(-1)).min()))
    return np.asarray(out)


def child_unit_m4b(spec):
    """Analytic checks of the M4b primitives and loss, each with broken variants that must fail."""
    M, torch = _setup(spec["root"])
    import numpy as np
    import measures as MS
    R = {}
    wrap = MS.wrap_angle
    dt = 0.2
    t = dt * torch.arange(1, 21, dtype=torch.float64)

    # ---- anchor profiles vs HAND-COMPUTED literals (v0 = 10)
    S = MS.anchor_profiles(torch.tensor([10.0], dtype=torch.float64),
                           [(5.0, 0.0), (2.0, 0.5), (math.inf, 0.0), (math.inf, 0.5)], 20, dt)[0]
    lit = {"a5f0_t0.2": (S[0, 0], 1.9), "a5f0_t1.0": (S[0, 4], 7.5), "a5f0_t4.0": (S[0, 19], 10.0),
           "a2f.5_t2.4": (S[1, 11], 18.24), "a2f.5_t2.6": (S[1, 12], 19.25), "a2f.5_t4.0": (S[1, 19], 26.25),
           "inf_f0_all": (S[2].abs().max(), 0.0), "inf_f.5_t4.0": (S[3, 19], 20.0)}
    worst = max(abs(float(v) - w) for v, w in lit.values())
    R["profiles_literals"] = {"ok": worst <= 1e-9, "max_err": worst,
                              "values": {k: float(v) for k, (v, _) in lit.items()}}
    S0 = MS.anchor_profiles(torch.tensor([0.0, -3.0], dtype=torch.float64),
                            MS.parse_profiles(MS.DEFAULT_SLOW_PROFILES), 20, dt)
    R["profiles_standstill"] = {"ok": float(S0.abs().max()) == 0.0, "max": float(S0.abs().max())}

    # ---- re-timing: the real function and three deliberately broken ones
    def retime_no_origin(traj, sigma):
        P = traj
        d = P[..., 1:, :2] - P[..., :-1, :2]
        seg = torch.sqrt((d * d).sum(-1) + 1e-12)
        c = torch.cat([torch.zeros_like(seg[..., :1]), seg.cumsum(-1)], -1)
        out = []
        for i, s in enumerate(sigma.tolist()):
            j = min(int(torch.searchsorted(c, torch.tensor(s, dtype=c.dtype))), len(c) - 1)
            j = max(j, 1)
            w = (s - float(c[j - 1])) / max(float(c[j] - c[j - 1]), 1e-12)
            w = min(max(w, 0.0), 1.0)
            out.append(P[j - 1] + w * (P[j] - P[j - 1]))
        return torch.stack(out)

    def retime_time_index(traj, sigma):
        """interpolates as if every segment had the same length (index, not arc length)"""
        P = torch.cat([torch.zeros(1, 3, dtype=traj.dtype), traj], 0)
        total = float(torch.sqrt(((P[1:, :2] - P[:-1, :2]) ** 2).sum(-1)).sum())
        out = []
        for s in sigma.tolist():
            u = min(s / max(total, 1e-12), 1.0) * (len(P) - 1)
            j = min(int(math.floor(u)), len(P) - 2)
            w = u - j
            out.append(P[j] + w * (P[j + 1] - P[j]))
        return torch.stack(out)

    def retime_no_wrap(traj, sigma):
        out = MS.retime_to_profile(traj, sigma)
        P = torch.cat([torch.zeros(1, 3, dtype=traj.dtype), traj], 0)
        seg = torch.sqrt(((P[1:, :2] - P[:-1, :2]) ** 2).sum(-1) + 1e-12)
        c = torch.cat([torch.zeros(1, dtype=traj.dtype), seg.cumsum(0)])
        hs = []
        for s in sigma.tolist():
            j = min(max(int(torch.searchsorted(c[1:], torch.tensor(s, dtype=c.dtype))), 0), len(traj) - 1)
            w = min(max((s - float(c[j])) / float(c[j + 1] - c[j]), 0.0), 1.0)
            hs.append(float(P[j, 2]) + w * (float(P[j + 1, 2]) - float(P[j, 2])))    # NO wrap
        out = out.clone()
        out[:, 2] = torch.tensor(hs, dtype=traj.dtype)
        return out

    variants = {"REAL": MS.retime_to_profile, "NO_ORIGIN": retime_no_origin,
                "TIME_INDEX": retime_time_index, "NO_WRAP": retime_no_wrap}
    must_fail = {"NO_ORIGIN": "line_accel", "TIME_INDEX": "line_accel", "NO_WRAP": "arc_heading"}

    # accelerating straight line x = 0.5 * 3 * t^2 and a profile sigma: the answer is x == sigma EXACTLY
    line = torch.stack([0.5 * 3.0 * t * t, torch.zeros_like(t), torch.zeros_like(t)], -1)
    sig_line = torch.tensor([0.3 * k for k in range(1, 21)], dtype=torch.float64)          # 0.3 .. 6.0 m
    # a U-turn arc (R 10 m at 8 m/s: 3.2 rad in 4 s, crossing +-pi); headings stored WRAPPED
    Rr, v = 10.0, 8.0
    th = v * t / Rr
    arc = torch.stack([Rr * torch.sin(th), Rr * (1 - torch.cos(th)), wrap(th)], -1)
    arc_c = MS.progress_profile(arc)
    sig_arc = torch.linspace(0.5, float(arc_c[-1]) - 0.1, 20, dtype=torch.float64)
    per_variant = {}
    for name, fn in variants.items():
        r = {}
        o = fn(line, sig_line)
        e = max(float((o[:, 0] - sig_line).abs().max()), float(o[:, 1].abs().max()))
        r["line_accel"] = (e <= 1e-9, e)
        o = fn(arc, sig_arc)
        dpoly = float(_polyline_distance(arc.numpy(), o.numpy()).max())
        cum = np.concatenate([[0.0], arc_c.numpy()])
        th_samples = np.concatenate([[0.0], th.numpy()])                # UNWRAPPED analytic sample angles
        want_h = []
        for s in sig_arc.tolist():
            j = min(max(int(np.searchsorted(cum, s)), 1), len(cum) - 1)
            w = (s - cum[j - 1]) / (cum[j] - cum[j - 1])
            want_h.append(th_samples[j - 1] + w * (th_samples[j] - th_samples[j - 1]))
        eh = float(np.abs(wrap(o[:, 2].numpy() - np.asarray(want_h))).max())
        r["arc_on_polyline"] = (dpoly <= 1e-9, dpoly)
        r["arc_heading"] = (eh <= 1e-9, eh)
        per_variant[name] = {k: {"ok": bool(ok), "err": float(err)} for k, (ok, err) in r.items()}
    R["retime_real"] = {"ok": all(v["ok"] for v in per_variant["REAL"].values()), "checks": per_variant["REAL"]}
    for name, check in must_fail.items():
        R[f"retime_mutation_{name}"] = {"ok": not per_variant[name][check]["ok"],
                                        "must_fail": check, "err": per_variant[name][check]["err"]}
    # identity: sigma = the path's own chord lengths reproduces it
    o = MS.retime_to_profile(arc, arc_c)
    e = float((o - arc).abs().max())
    R["retime_identity"] = {"ok": e <= 1e-9, "err": e}
    # extrapolation past the end, along the last heading (analytic)
    sig_ex = arc_c[-1] + torch.arange(1, 21, dtype=torch.float64)
    o = MS.retime_to_profile(arc, sig_ex)
    hT = float(arc[-1, 2])
    want = arc[-1, :2][None] + (sig_ex - arc_c[-1])[:, None] * torch.tensor([math.cos(hT), math.sin(hT)],
                                                                          dtype=torch.float64)
    e = max(float((o[:, :2] - want).abs().max()), float((o[:, 2] - hT).abs().max()))
    R["retime_extrapolation"] = {"ok": e <= 1e-9, "err": e}
    # a stationary (all-zero) path: defined output and FINITE gradients
    z = torch.zeros(3, 20, 3, requires_grad=True)
    o = MS.retime_to_profile(z, torch.linspace(0, 12, 20).expand(3, 20).contiguous())
    o.sum().backward()
    R["retime_zero_path_grad"] = {"ok": bool(torch.isfinite(z.grad).all()) and bool(torch.isfinite(o).all())}

    # ---- explicit reserved slots (the mid-run switch's least-used slots) are honoured; bad lists refused
    least = "8,16,19,21,22,34,38,39"
    ok_idx = MS.SlowSlotConfig.from_args(8, MS.DEFAULT_SLOW_PROFILES, 64, indices=least).slots == \
        (8, 16, 19, 21, 22, 34, 38, 39) and MS.SlowSlotConfig.from_args(8, MS.DEFAULT_SLOW_PROFILES, 64).slots == \
        tuple(range(56, 64))
    refused = 0
    for bad in ("8,16,19", "8,8,19,21,22,34,38,39", "8,16,19,21,22,34,38,64"):
        try:
            c_ = MS.SlowSlotConfig.from_args(8, MS.DEFAULT_SLOW_PROFILES, 64, indices=bad)
            c_.validate(64, 20, 3, 7)
        except ValueError:
            refused += 1
    R["slot_indices_honoured_and_checked"] = {"ok": bool(ok_idx) and refused == 3, "explicit": least,
                                              "bad_lists_refused": f"{refused}/3"}
    # ---- the loss: closest-anchor choice vs HAND-COMPUTED literals; value literals
    cfg = MS.SlowSlotConfig.from_args(8, MS.DEFAULT_SLOW_PROFILES, 64)
    tf = t.float()

    def straight(vv):
        return torch.stack([vv * tf, torch.zeros_like(tf), torch.zeros_like(tf)], -1)
    brake = torch.stack([torch.where(tf <= 2.0, 10 * tf - 2.5 * tf * tf, torch.full_like(tf, 10.0)),
                         torch.zeros_like(tf), torch.zeros_like(tf)], -1)
    tgt = torch.stack([straight(15.0), torch.zeros(20, 3), brake])
    ego = torch.zeros(3, 7)
    ego[:, 6] = torch.tensor([15.0, 0.0, 10.0])
    ego[:, 0] = ego[:, 6]
    from model import wta_loss
    traj = torch.randn(3, 64, 20, 3, generator=torch.Generator().manual_seed(3)) * 5
    info = {}
    loss, idx = MS.anchored_wta_loss(traj, tgt, ego, cfg, wta_fn=wta_loss, info=info)
    free = torch.tensor([i for i in range(64) if i not in cfg.slots])
    lf, _ = wta_loss(traj.index_select(1, free), tgt)
    ks = info["k_star"].tolist()
    R["loss_closest_anchor_literals"] = {"ok": ks == [7, 0, 1], "k_star": ks, "want": [7, 0, 1],
                                         "why": "15 m/s cruise -> (1,0.875); standstill -> first; "
                                                "10 m/s braking at 5 m/s2 -> (6,0) (hand-computed means "
                                                "2.41 / 0 / 1.199 m)"}
    R["loss_free_term_is_wta_loss"] = {"ok": bool(torch.equal(info["l_free"], lf.detach())),
                                       "free_winner_in_free": bool(all(int(i) not in cfg.slots for i in idx))}
    # value literals: free slots ON the target and the positive reserved slot ON its anchored target -> 0;
    # +1 m lateral on the positive -> exactly w_res * 1.0; +1 m on a NON-positive reserved slot -> 0
    tg1 = straight(15.0)[None]
    e1 = torch.zeros(1, 7)
    e1[0, 6] = 15.0
    tr = tg1[:, None].expand(1, 64, 20, 3).clone()
    S7 = MS.anchor_profiles(e1[:, 6], cfg.profiles, 20)[0]                                  # [K, T]
    for k, s in enumerate(cfg.slots):
        tr[0, s, :, 0] = S7[k]
    l0, _ = MS.anchored_wta_loss(tr, tg1, e1, cfg, wta_fn=wta_loss)
    tr2 = tr.clone()
    tr2[0, cfg.slots[7], :, 1] += 1.0
    l1, _ = MS.anchored_wta_loss(tr2, tg1, e1, cfg, wta_fn=wta_loss)
    tr3 = tr.clone()
    tr3[0, cfg.slots[2], :, 1] += 1.0
    l2, _ = MS.anchored_wta_loss(tr3, tg1, e1, cfg, wta_fn=wta_loss)
    R["loss_value_literals"] = {"ok": float(l0) < 1e-4 and abs(float(l1) - 1.0) < 1e-4 and float(l2) < 1e-4,
                                "l_on_anchor": float(l0), "l_positive_plus_1m": float(l1),
                                "l_nonpositive_plus_1m": float(l2)}
    json.dump(R, open(spec["out"], "w"), indent=1, default=float)


def child_hook(spec):
    """Tiny REFe from ONE root: forward + wta loss (and, patched, the installed-hook forward) on fixed inputs."""
    M, torch = _setup(spec["root"])
    import ckpt_io
    cfg = M.REFeConfig.for_backbone("tiny")
    torch.manual_seed(0)
    model = M.REFe(cfg).eval()
    g = torch.Generator().manual_seed(1)
    B = 3
    img = torch.randn(B, 4, 3, cfg.img_h, cfg.img_w, generator=g)
    ego = torch.rand(B, 7, generator=g) * 10
    goal = torch.randn(B, 4, generator=g) * 20
    tgt = torch.randn(B, 20, 3, generator=g) * 10
    out = {}
    with torch.no_grad():
        traj, score = model(img, ego, goal)
        l, idx = M.wta_loss(traj.float(), tgt)
    out.update(traj=traj, score=score, loss=l, idx=idx)
    if spec.get("patched"):
        import measures as MS
        keys0 = list(model.state_dict().keys())
        ntr0 = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fp0 = ckpt_io.frozen_fingerprint(model)
        cfg_s = MS.SlowSlotConfig.from_args(8, MS.DEFAULT_SLOW_PROFILES, cfg.n_proposals)
        MS.install_slow_slots(model, cfg_s)
        with torch.no_grad():
            traj_on, score_on = model(img, ego, goal)
        S = MS.anchor_profiles(ego[:, 6], cfg_s.profiles, cfg.horizon_steps)
        L = MS.progress_profile(traj_on[:, list(cfg_s.slots)].double())

        def extended(p):                          # the raw path, continued 10 km along its last heading
            h = float(p[-1, 2])
            far = p[-1:].clone()
            far[0, 0] += 1e4 * math.cos(h)
            far[0, 1] += 1e4 * math.sin(h)
            return torch.cat([p, far]).double().numpy()
        dpoly = max(float(_polyline_distance(extended(traj[b, s]), traj_on[b, s].numpy()).max())
                    for b in range(B) for s in cfg_s.slots)
        free = [i for i in range(cfg.n_proposals) if i not in cfg_s.slots]
        out.update(traj_on=traj_on, score_on=score_on,
                   keys_same=keys0 == list(model.state_dict().keys()),
                   trainable_same=ntr0 == sum(p.numel() for p in model.parameters() if p.requires_grad),
                   fingerprint_same=fp0 == ckpt_io.frozen_fingerprint(model),
                   free_equal=bool(torch.equal(traj_on[:, free], traj[:, free])),
                   exceed_max=float((L - S.double()).max()), on_raw_polyline_max=dpoly,
                   score_changed=not bool(torch.equal(score_on, score)))
    torch.save(out, spec["out"])


def child_partition(spec):
    M, torch = _setup(spec["root"])
    import measures as MS
    import torch.nn.functional as F
    res = {}
    if spec.get("real_vitl"):
        cfg = M.REFeConfig.for_backbone("vitl16")
        model = M.REFe(cfg)
        grp = MS.clip_groups(model)
        tr = [n for n, p in model.named_parameters() if p.requires_grad]
        gt, gs = [n for n, _ in grp["traj"]], [n for n, _ in grp["score"]]
        res = {"n_trainable": len(tr), "n_traj": len(gt), "n_score": len(gs),
               "numel_traj": sum(p.numel() for _, p in grp["traj"]),
               "numel_score": sum(p.numel() for _, p in grp["score"]),
               "exactly_once": sorted(gt + gs) == sorted(tr) and not set(gt) & set(gs),
               "score_modules": sorted({n.split(".")[0] for n in gs}),
               "cameras": list(cfg.cameras), "backbone": cfg.backbone}
        json.dump(res, open(spec["out"], "w"), indent=1)
        return
    for arm, detach in (("detached", True), ("MUTATION_context_attached", False)):
        cfg = M.REFeConfig.for_backbone("tiny")
        cfg.detach_scorer_context = detach
        torch.manual_seed(0)
        model = M.REFe(cfg)
        with torch.no_grad():                    # LoRA B is zero at init: give it a value so A can get gradient
            for n, p in model.named_parameters():
                if n.endswith(".B") and p.requires_grad:
                    p.add_(0.01 * torch.randn_like(p))
        g = torch.Generator().manual_seed(2)
        B = 2
        img = torch.randn(B, 4, 3, cfg.img_h, cfg.img_w, generator=g)
        ego, goal = torch.rand(B, 7, generator=g), torch.randn(B, 4, generator=g)
        tgt = torch.randn(B, 20, 3, generator=g) * 5
        extra = torch.randn(B, 64, 20, 3, generator=g) * 5
        names = [n for n, p in model.named_parameters() if p.requires_grad]

        def grads(which):
            model.zero_grad(set_to_none=True)
            traj, score, sx = model(img, ego, goal, score_extra=extra)
            if which == "score":
                loss = F.binary_cross_entropy_with_logits(score, torch.rand_like(score)) + \
                    F.binary_cross_entropy_with_logits(sx, torch.rand_like(sx))
            else:
                loss = M.wta_loss(traj, tgt)[0]
            loss.backward()
            return {n for n, p in model.named_parameters()
                    if p.requires_grad and p.grad is not None and bool((p.grad != 0).any())}
        gs_flow, gt_flow = grads("score"), grads("traj")
        grp_names = {k: {n for n, _ in v} for k, v in MS.clip_groups(model).items()}
        premise = "accepted"
        try:
            MS.check_clip_premise(model)
        except ValueError:
            premise = "REFUSED"
        res[arm] = {"score_loss_reaches": sorted({n.split(".")[0] for n in gs_flow}),
                    "score_loss_hits_traj_group": sorted(gs_flow & grp_names["traj"])[:6],
                    "n_score_loss_hits_traj_group": len(gs_flow & grp_names["traj"]),
                    "n_traj_loss_hits_score_group": len(gt_flow & grp_names["score"]),
                    "flow_score_equals_name_group": gs_flow == grp_names["score"],
                    "never_touched": sorted(set(names) - gs_flow - gt_flow),
                    "clip_premise": premise}
    json.dump(res, open(spec["out"], "w"), indent=1)


def child_loader(spec):
    """The REAL TargetBank (orig or patched) on a bank holding twin files."""
    M, torch = _setup(spec["root"])
    import train as T
    cfg = M.REFeConfig.for_backbone("tiny")
    res = {}
    try:
        op = T.OnPolicyBank(spec["onpolicy"], cfg.n_proposals, cfg.horizon_steps) if spec.get("onpolicy") else None
        kw = {"slow_twins": spec["slow_twins"]} if spec.get("patched") else {}
        ds = T.TargetBank(spec["bank"], None, cfg, synthetic=True, onpolicy=op, **kw)
    except SystemExit as e:
        json.dump({"refused": True, "msg": str(e.code)}, open(spec["out"], "w"), indent=1)
        return
    res["refused"] = False
    res["n_rows"] = len(ds)
    per_rank = {}
    for r in ds.rows:
        per_rank[str(r["rank"])] = per_rank.get(str(r["rank"]), 0) + 1
    res["per_rank"] = per_rank
    res["group_sizes"] = sorted({len(g) for g in ds.scene_groups})
    if op is not None:
        res["cov_norm"] = T.onpolicy_cov_norm(ds, op, cfg.n_proposals)
    key = {(r["log_name"], r["token"], r["step"], r["rank"]): i for i, r in enumerate(ds.rows)}
    checks = {"traj_exact": 0, "ego_goal_exact": 0, "images_same": 0, "set_same_as_source": 0,
              "set_empty": 0, "n": 0}
    for i, r in enumerate(ds.rows):
        if not isinstance(r.get("slow"), dict):
            continue
        j = key[(r["log_name"], r["token"], r["step"], int(r["slow"]["src_rank"]))]
        a, b = ds[i], ds[j]
        checks["n"] += 1
        checks["traj_exact"] += int(torch.equal(a[3], torch.tensor(r["traj"], dtype=torch.float32)))
        checks["ego_goal_exact"] += int(torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]))
        checks["images_same"] += int(r["image"] == ds.rows[j]["image"])
        checks["set_same_as_source"] += int(all(torch.equal(a[k], b[k]) for k in (4, 5, 6)) and float(a[6].sum()) > 0)
        checks["set_empty"] += int(float(a[6].sum()) == 0)
    res["twin_checks"] = checks
    json.dump(res, open(spec["out"], "w"), indent=1)


def child_unit_m6(spec):
    """M6 units: 'wrapped' IS model.wta_loss (bits); 'plain' equals it where no winner error crosses pi; a
    branch-shifted winner heading scores ~0 wrapped and exactly yaw_w * 2*pi * share plain (hand-computed)."""
    M, torch = _setup(spec["root"])
    import measures as MS
    R = {}
    g = torch.Generator().manual_seed(5)
    traj = torch.randn(4, 64, 20, 3, generator=g) * 3
    tgt = torch.randn(4, 20, 3, generator=g)
    tgt[..., 2] = tgt[..., 2].clamp(-2.5, 2.5)
    l0, i0 = M.wta_loss(traj, tgt)
    l1, i1 = MS.wta_loss_yaw(traj, tgt, mode="wrapped", wta_fn=M.wta_loss)
    R["wrapped_is_wta_loss_bitwise"] = {"ok": bool(torch.equal(l0, l1) and torch.equal(i0, i1))}
    # in-branch: every slot's heading within 1 rad of its target -> plain == wrapped to float rounding
    tr = traj.clone()
    tr[..., 2] = tgt[:, None, :, 2] + (torch.rand(4, 64, 20, generator=g) - 0.5)
    lw, _ = MS.wta_loss_yaw(tr, tgt, mode="wrapped", wta_fn=M.wta_loss)
    lp, _ = MS.wta_loss_yaw(tr, tgt, mode="plain", wta_fn=M.wta_loss)
    R["plain_equals_wrapped_in_branch"] = {"ok": abs(float(lw) - float(lp)) <= 1e-6, "diff": abs(float(lw) - float(lp))}
    # branch shift: every slot ON the target except slot 0's heading at steps 15..19 = target - 2*pi (slot 0 wins
    # the tie) -> wrapped ~0, plain = 0.1 * 2*pi * 5/20 = 0.15707963 (literal)
    tb = tgt[:, None].expand(4, 64, 20, 3).clone()
    tb[:, 0, 15:, 2] -= 2 * math.pi
    tb.requires_grad_(True)
    lw, _ = MS.wta_loss_yaw(tb, tgt, mode="wrapped", wta_fn=M.wta_loss)
    tp = tb.detach().clone().requires_grad_(True)
    lp, _ = MS.wta_loss_yaw(tp, tgt, mode="plain", wta_fn=M.wta_loss)
    lp.backward()
    up = bool((tp.grad[:, 0, 15:, 2] < 0).all())          # gradient DESCENT raises the shifted headings
    R["branch_shift_literals"] = {"ok": float(lw) < 1e-5 and abs(float(lp) - 0.15707963) < 1e-6 and up,
                                  "wrapped": float(lw), "plain": float(lp), "want_plain": 0.15707963,
                                  "plain_pushes_toward_principal": up}
    rows = [{"traj": [[1.0, 0.0, 0.1]] * 20}, {"traj": [[1.0, 0.0, 2.46]] * 20}]
    ok_guard = MS.check_plain_yaw_targets(rows) == 2.46
    try:
        MS.check_plain_yaw_targets(rows + [{"traj": [[1.0, 0.0, 3.1]] * 20}])
        refused = False
    except ValueError:
        refused = True
    R["seam_guard"] = {"ok": ok_guard and refused, "accepts_2.46": ok_guard, "refuses_3.1": refused}
    json.dump(R, open(spec["out"], "w"), indent=1)


def child_unit_m6b(spec):
    """M6b units, every expectation a LITERAL: the tangent of a straight path is exactly 0; a circle's central difference
    is exact and its backward difference lags by exactly v*dt/(2R); a heading 2*pi off receives a pull back; masked steps
    contribute exactly 0; positions receive EXACTLY zero gradient; plain_tangent == plain + tan_w * tangent_loss; and --
    the mechanism -- a NON-winning slot gets a heading gradient under plain_tangent and none under plain."""
    M, torch = _setup(spec["root"])
    import measures as MS
    R = {}
    t = 0.2 * torch.arange(1, 21, dtype=torch.float64)
    st = torch.zeros(1, 1, 20, 3, dtype=torch.float64)
    st[..., 0] = 10 * t
    th, mk = MS.tangent_targets(st)
    R["straight_tangent_exactly_0"] = {"ok": float(th.abs().max()) == 0.0 and bool(mk.all())}
    Rr, v = 20.0, 10.0
    ang = v * t / Rr
    ci = torch.zeros(1, 1, 20, 3, dtype=torch.float64)
    ci[..., 0], ci[..., 1], ci[..., 2] = Rr * torch.sin(ang), Rr * (1 - torch.cos(ang)), ang
    th2, _ = MS.tangent_targets(ci)
    e = (th2 - ang).abs()[0, 0]
    R["circle_central_exact_backward_lag"] = {"ok": float(e[:19].max()) < 1e-12 and abs(float(e[19]) - 0.05) < 1e-12,
                                              "central_max": float(e[:19].max()), "backward_19": float(e[19])}
    tb = st.clone()
    tb[..., 19, 2] = 2 * math.pi
    tb.requires_grad_(True)
    MS.tangent_loss(tb).backward()
    R["2pi_heading_pulled_back"] = {"ok": float(tb.grad[0, 0, 19, 2]) == 1.0 / 20.0, "grad": float(tb.grad[0, 0, 19, 2])}
    R["positions_get_exactly_zero_gradient"] = {"ok": float(tb.grad[..., :2].abs().max()) == 0.0}
    sl = st.clone()
    sl[..., :10, 0] = 0.01 * torch.arange(1, 11, dtype=torch.float64)          # 5 cm/step: below the 0.2 m mask
    sl[..., :10, 2] = 3.0                                                       # wildly wrong, but masked
    sl.requires_grad_(True)
    MS.tangent_loss(sl).backward()
    R["masked_steps_contribute_exactly_0"] = {"ok": float(sl.grad[0, 0, :9, 2].abs().max()) == 0.0}
    # MUTATION 1: no stop-gradient -> positions DO get gradient (the check above must be able to go RED)
    xy = st[..., :2].clone().requires_grad_(True)
    d = torch.cat([xy[..., :1, :], xy[..., 1:, :] - xy[..., :-1, :]], dim=-2)
    thm = torch.atan2(d[..., 1], d[..., 0])
    ((torch.full_like(thm, 0.3) - thm).abs().mean()).backward()
    R["MUTATION_no_stopgrad_positions_get_gradient"] = {"ok": float(xy.grad.abs().max()) > 0.0}
    # MUTATION 2: a WRAPPED difference gives the 2*pi heading no pull
    wr = ((((tb.detach()[..., 2] - MS.tangent_targets(tb.detach())[0]) + math.pi) % (2 * math.pi)) - math.pi).abs()
    R["MUTATION_wrapped_difference_no_pull"] = {"ok": float(wr[0, 0, 19]) < 1e-9}
    g = torch.Generator().manual_seed(11)
    tgt = torch.randn(2, 20, 3, generator=g, dtype=torch.float64)
    tgt[..., 0] = 10 * t
    tgt[..., 2] = 0.0
    tr = tgt[:, None].expand(2, 64, 20, 3).clone()
    tr[:, 1:, :, 1] += 5.0                                                      # slot 0 wins, 1..63 are far
    tr[:, 7, 19, 2] = 2 * math.pi                                               # a NON-winning slot trapped at step 19
    lp, _ = MS.wta_loss_yaw(tr, tgt, mode="plain", wta_fn=M.wta_loss)
    lt, _ = MS.wta_loss_yaw(tr, tgt, mode="plain_tangent", wta_fn=M.wta_loss, tan_w=0.1)
    R["plain_tangent_is_plain_plus_tanw_times_tangent"] = {
        "ok": abs(float(lt) - (float(lp) + 0.1 * float(MS.tangent_loss(tr)))) < 1e-12}
    a1 = tr.clone().requires_grad_(True)
    MS.wta_loss_yaw(a1, tgt, mode="plain", wta_fn=M.wta_loss)[0].backward()
    a2 = tr.clone().requires_grad_(True)
    MS.wta_loss_yaw(a2, tgt, mode="plain_tangent", wta_fn=M.wta_loss, tan_w=0.1)[0].backward()
    R["mechanism_nonwinner_slot_gradient"] = {"ok": float(a1.grad[:, 7, 19, 2].abs().max()) == 0.0 and
                                              float(a2.grad[:, 7, 19, 2].min()) > 0.0,
                                              "plain": float(a1.grad[:, 7, 19, 2].abs().max()),
                                              "plain_tangent": float(a2.grad[:, 7, 19, 2].min())}
    json.dump(R, open(spec["out"], "w"), indent=1)


def child_eval_heading(spec):
    """Heading health of a finished tiny run on its bank: which native steps are TRAPPED (> 10 % of raw headings
    beyond +-pi AND the WTA winner's median wrapped error > 0.5 rad), per-step winner errors, position error."""
    M, torch = _setup(spec["root"])
    import numpy as np
    import ckpt_io
    import train as T
    cfg = M.REFeConfig.for_backbone("tiny")
    m = M.REFe(cfg).eval()
    ckpt_io.load_for_inference(m, spec["ckpt"])
    ds = T.TargetBank(spec["bank"], None, cfg, synthetic=True)
    idx = list(range(0, len(ds), max(1, len(ds) // 120)))[:120]
    it = [ds[i] for i in idx]
    with torch.no_grad():
        tr = m(torch.stack([x[0] for x in it]), torch.stack([x[1] for x in it]), torch.stack([x[2] for x in it]))[0]
    tr = tr.double().numpy()
    tg = torch.stack([x[3] for x in it]).double().numpy()
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi                                   # noqa: E731
    d = np.abs(tr[..., :2] - tg[:, None, :, :2]).sum(-1).mean(-1)
    w = d.argmin(1)
    ar = np.arange(len(idx))
    frac = [float(np.mean(np.abs(tr[:, :, k, 2]) > np.pi)) for k in range(20)]
    werr = [float(np.median(np.abs(wrap(tr[ar, w, k, 2] - tg[:, k, 2])))) for k in range(20)]
    json.dump({"trapped_steps": [k for k in range(20) if frac[k] > 0.10 and werr[k] > 0.5],
               "max_frac_beyond_pi": max(frac), "winner_wrap_err_by_step": werr,
               "winner_pos_L1_m": float(d[ar, w].mean()), "n": len(idx)}, open(spec["out"], "w"), indent=1)


CHILDREN = {"trainer": child_trainer, "eval_slots": child_eval_slots, "unit_m4b": child_unit_m4b,
            "hook": child_hook, "partition": child_partition, "loader": child_loader,
            "unit_m6": child_unit_m6, "eval_heading": child_eval_heading, "unit_m6b": child_unit_m6b}


# ================================================================================================ parent side
class Suite:
    def __init__(self, tmp: Path, py: str, log):
        self.tmp, self.py, self.log = tmp, py, log
        self.results: dict = {}
        self.n_child = 0
        self.tested_sha256: dict = {}

    def say(self, s=""):
        print(s, flush=True)
        self.log.write(s + "\n")
        self.log.flush()

    def record(self, tid, name, ok, **values):
        self.results.setdefault(tid, {})[name] = {"ok": bool(ok), **values}
        vs = "  ".join(f"{k}={_fmt(v)}" for k, v in values.items())
        self.say(f"  [{tid}] {'PASS' if ok else 'FAIL'}  {name:40s} {vs}")

    def child(self, kind, root, **spec):
        self.n_child += 1
        out = self.tmp / f"child_{self.n_child:03d}_{kind}.out"
        spec.update(root=str(root), out=str(out))
        sp = self.tmp / f"child_{self.n_child:03d}.json"
        sp.write_text(json.dumps(spec), encoding="utf-8")
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONHASHSEED="0",
                   PYTHONIOENCODING="utf-8")
        env.pop("PYTHONPATH", None)
        lf = self.tmp / f"child_{self.n_child:03d}_{kind}.log"
        with open(lf, "w", encoding="utf-8") as fh:
            p = subprocess.run([self.py, str(Path(__file__).resolve()), "--child", kind, "--spec", str(sp)],
                               stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(root))
        if p.returncode != 0 or not out.exists():
            tail = lf.read_text(encoding="utf-8", errors="replace")[-3000:]
            raise RuntimeError(f"child {kind} failed rc={p.returncode}; log {lf}:\n{tail}")
        return out, lf


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)) and len(v) > 8:
        return f"[{len(v)} items]"
    return str(v)


def _blob(path) -> str:
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _maxdiff(a, b) -> float:
    """max |a - b| over matching tensors (lists / dicts / tensors); +inf on a structural mismatch."""
    import torch
    if isinstance(a, dict):
        if set(a) != set(b):
            return math.inf
        return max([_maxdiff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return math.inf
        return max([_maxdiff(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
            return math.inf
        if a.numel() == 0:
            return 0.0
        if a.is_floating_point():
            return float((a.double() - b.double()).abs().max())
        return float((a != b).any())
    if a is None and b is None:
        return 0.0
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    return 0.0 if a == b else math.inf


# --------------------------------------------------------------------------------------- synthetic banks
def _line(v, curv=0.0, T=20, dt=0.2):
    out = []
    for k in range(1, T + 1):
        s = v * dt * k
        if abs(curv) < 1e-12:
            x, y, h = s, 0.0, 0.0
        else:
            h = curv * s
            x, y = math.sin(h) / curv, (1 - math.cos(h)) / curv
        out.append([x, y, (h + math.pi) % (2 * math.pi) - math.pi])
    return out


def make_bank(d: Path, n_scenes: int, speeds, curvs, rank1_every: int = 0, onpolicy: Path | None = None,
              world_fields: bool = True, seed: int = 0):
    d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    r0, r1 = [], []
    for i in range(n_scenes):
        v, c = speeds[i % len(speeds)], curvs[i % len(curvs)]
        base = {"image": [f"log{i % 4:02d}/{cam}/{i:06d}{j}.jpg" for j, cam in enumerate(CAMS)],
                "cameras": list(CAMS), "ego": [v, 0.0, 0.0, 0.0, c * v, 0.0, v],
                "goal": [2.0 * v, 0.0, 4.0 * v, 0.0], "log_name": f"log{i % 4:02d}", "token": f"tok{i:04d}",
                "step": 0, "dt_ms": 20.0, "scenario_type": "synthetic", "source": "per_frame_teacher_rollout"}
        row = dict(base, traj=_line(v, c), rank=0)
        if world_fields:
            row["origin_world"] = [100.0 + i, 50.0, 0.3]
            row["traj_world"] = [[p[0] + 100.0, p[1] + 50.0, p[2] + 0.3] for p in row["traj"]]
        r0.append(row)
        if rank1_every and i % rank1_every == 0:
            r1.append(dict(base, traj=_line(0.9 * v, -c), rank=1, goal=[2.0 * v, 3.0, 4.0 * v, 3.5],
                           aug={"kind": "lane_rank", "value": 1, "divergence_m": 1.0, "lanes": 3}))
    for rank, rows in ((0, r0), (1, r1)):
        if rows:
            with open(d / f"targets_rank{rank}.jsonl", "w", encoding="utf-8", newline="\n") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
    if onpolicy is not None:
        onpolicy.mkdir(parents=True, exist_ok=True)
        with open(onpolicy / "onpolicy_r0_w0.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in r0 + r1:
                f.write(json.dumps({
                    "kind": "onpolicy_set", "log_name": r["log_name"], "token": r["token"], "step": r["step"],
                    "rank": r["rank"], "ckpt_step": 0, "label_version": 1,
                    "traj": [[[round(rng.uniform(0, 60), 3), round(rng.uniform(-3, 3), 3)] for _ in range(20)]
                             for _ in range(64)],
                    "yaw": [[round(rng.uniform(-0.2, 0.2), 4) for _ in range(20)] for _ in range(64)],
                    "targets": [{"collision.NuPlanCollision.info": float(rng.random() < 0.1),
                                 "dac.violation": float(rng.random() < 0.2), "progress.ep": rng.random(),
                                 "ttc.NuPlanTTC.ttc_reward": rng.random(),
                                 "comfort.Comfort.reward": rng.random(),
                                 "ddc.violation": float(rng.random() < 0.05)} for _ in range(64)]}) + "\n")
    return r0, r1


def targs(bank, out, empty, *extra, steps=4, batch=2, accum=2, seed=0, onpolicy=None, log_every=1):
    a = ["--backbone", "tiny", "--weights", "none", "--synthetic", "--cpu", "--targets", str(bank),
         "--scorer-targets", str(empty), "--steps", str(steps), "--batch", str(batch), "--accum", str(accum),
         "--seed", str(seed), "--log-every", str(log_every), "--out", str(out), "--ckpt-every-min", "1e9"]
    if onpolicy is not None:
        a += ["--scorer-mode", "onpolicy", "--onpolicy-targets", str(onpolicy)]
    return a + [str(x) for x in extra]


def load_ckpt(run: Path):
    import torch
    st = torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=False)
    return {"model": st["model_partial"], "opt": st["opt"]["state"]}


def metrics(run: Path) -> list:
    p = run / "metrics.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


# --------------------------------------------------------------------------------------- the tests
def main_parent(a) -> int:
    import torch
    t_start = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="measures_selftest_"))
    res_path = Path(a.results)
    res_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(res_path.with_suffix(".log"), "w", encoding="utf-8")
    S = Suite(tmp, sys.executable, log)
    S.say(f"selftest_measures: temp {tmp}")
    S.say(f"  python {sys.version.split()[0]}  torch {torch.__version__}  exe {sys.executable}")

    # ---------------------------------------------------------------- T1 patch
    orig, pat = tmp / "orig", tmp / "patched"
    for d in (orig, pat):
        d.mkdir()
        for f in HERE.glob("*.py"):
            if f.name != Path(__file__).name:
                shutil.copy2(f, d / f.name)
    S.tested_sha256 = {f: hashlib.sha256((orig / f).read_bytes()).hexdigest()
                       for f in ("measures.py", "slow_twins.py", "slow_copies.py", "model.py", "train.py", "ckpt_io.py")
                       if (orig / f).exists()}
    S.tested_sha256.update({p: hashlib.sha256((HERE / p).read_bytes()).hexdigest() for p in PATCHES})
    S.tested_sha256["selftest_measures.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bases = {}
    for pname in PATCHES:
        for line in open(HERE / pname, encoding="utf-8"):
            if line.startswith("#   ") and "git-blob" in line:
                parts = line.split()
                bases[parts[1]] = parts[3]
    base_ok = {f: _blob(orig / f) == b for f, b in bases.items()}
    S.record("T1", "patch base blobs == today's files", all(base_ok.values()), bases=base_ok)
    applied = True
    for pname in PATCHES:
        p = subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "-p1", str(HERE / pname)],
                           cwd=str(pat), capture_output=True, text=True)
        applied &= p.returncode == 0
        if p.returncode:
            S.say(p.stderr)
    import py_compile
    comp = True
    for f in ("model.py", "train.py", "ckpt_io.py", "measures.py", "slow_twins.py"):
        try:
            py_compile.compile(str(pat / f), doraise=True)
        except py_compile.PyCompileError as e:
            comp = False
            S.say(str(e))
    changed = sorted(f.name for f in pat.glob("*.py") if _blob(f) != _blob(orig / f.name))
    S.record("T1", "patches apply + patched files compile", applied and comp, changed=changed)
    if not applied:
        return _finish(S, res_path, t_start, a)

    empty = tmp / "empty_scorer"
    empty.mkdir()
    # ---------------------------------------------------------------- T2 OFF == today
    bankA, opA = tmp / "bankA", tmp / "bankA_op"
    make_bank(bankA, 8, [8.0, 12.0, 15.0], [0.0, 0.02, -0.03], rank1_every=2, onpolicy=opA)
    runs = {}
    for tag, root, seed, extra in (("orig", orig, 0, []), ("orig_again", orig, 0, []),
                                   ("orig_seed1", orig, 1, []), ("patched_off", pat, 0, []),
                                   ("patched_lognorms", pat, 0, ["--log-grad-norms"])):
        out = tmp / f"runA_{tag}"
        o, _ = S.child("trainer", root, argv=targs(bankA, out, empty, *extra, seed=seed, onpolicy=opA),
                       capture=True)
        runs[tag] = {"cap": torch.load(o, weights_only=False), "ck": load_ckpt(out), "m": metrics(out)}
    ref = runs["orig"]
    sc = [r.get("score") for r in ref["m"] if "traj_L1" in r]
    S.record("T2", "control: the on-policy score loss is live", all(abs(x) > 0 for x in sc) and len(sc) == 4,
             score_per_step=[round(x, 5) for x in sc])
    for tag, want_zero in (("orig_again", True), ("patched_off", True), ("patched_lognorms", True),
                           ("orig_seed1", False)):
        dg = _maxdiff(ref["cap"]["grads"], runs[tag]["cap"]["grads"])
        dw = _maxdiff(ref["ck"], runs[tag]["ck"])
        ok = (dg == 0.0 and dw == 0.0) if want_zero else (dg > 0.0 and dw > 0.0)
        S.record("T2", f"orig vs {tag}: grads/weights/AdamW", ok and runs[tag]["cap"]["rc"] == 0,
                 max_grad_diff=dg, max_state_diff=dw, n_steps=len(runs[tag]["cap"]["grads"]),
                 expect="exactly 0.0" if want_zero else "> 0")
    mrows = [r for r in runs["patched_lognorms"]["m"] if r.get("event") == "measures"]
    S.record("T2", "--log-grad-norms writes norm rows", len(mrows) == 4 and all("gn_all" in r for r in mrows),
             rows=len(mrows), gn_all=[round(r["gn_all"], 4) for r in mrows],
             clip_coef_traj=[round(r["clip_coef_traj"], 4) for r in mrows])

    # ---------------------------------------------------------------- T3 M3 invariance
    names = ref["cap"]["names"]
    is_score = [n.startswith(("score_q_mlp.", "score_dec.", "score_head.")) for n in names]

    def split_diff(c1, c2):
        dt_, ds_ = 0.0, 0.0
        for s1, s2 in zip(c1["grads"], c2["grads"]):
            for g1, g2, sc_ in zip(s1, s2, is_score):
                d = _maxdiff(g1, g2)
                if sc_:
                    ds_ = max(ds_, d)
                else:
                    dt_ = max(dt_, d)
        return dt_, ds_

    def weights_diff(k1, k2, score_side):
        m = 0.0
        for n in k1["model"]:
            if n.startswith(("score_q_mlp.", "score_dec.", "score_head.")) == score_side:
                m = max(m, _maxdiff(k1["model"][n], k2["model"][n]))
        return m
    m3 = {}
    for tag, root, extra, mut in (("on_w0.1", pat, ["--clip-split", "--score-w", "0.1"], None),
                                  ("on_w10", pat, ["--clip-split", "--score-w", "10"], None),
                                  ("off_w0.1", pat, ["--log-grad-norms", "--score-w", "0.1"], None),
                                  ("off_w10", pat, ["--log-grad-norms", "--score-w", "10"], None),
                                  ("badpart_w0.1", pat, ["--clip-split", "--score-w", "0.1"], ["m3_bad_partition"]),
                                  ("badpart_w10", pat, ["--clip-split", "--score-w", "10"], ["m3_bad_partition"])):
        out = tmp / f"runM3_{tag}"
        o, _ = S.child("trainer", root, argv=targs(bankA, out, empty, *extra, onpolicy=opA), capture=True,
                       mutate=mut)
        m3[tag] = {"cap": torch.load(o, weights_only=False), "ck": load_ckpt(out), "m": metrics(out)}
    dt_on, ds_on = split_diff(m3["on_w0.1"]["cap"], m3["on_w10"]["cap"])
    wt_on = weights_diff(m3["on_w0.1"]["ck"], m3["on_w10"]["ck"], False)
    S.record("T3", "ON: traj grads+weights invariant to score x100", dt_on == 0.0 and wt_on == 0.0 and ds_on > 0,
             traj_grad_diff=dt_on, traj_weight_diff=wt_on, score_grad_diff=ds_on,
             expect="traj 0.0 exactly, score > 0")
    dt_off, ds_off = split_diff(m3["off_w0.1"]["cap"], m3["off_w10"]["cap"])
    nr = {k: [(round(r["gn_all"], 3), round(r["clip_coef_traj"], 4)) for r in m3[k]["m"] if r.get("event") == "measures"]
          for k in ("off_w0.1", "off_w10")}
    S.record("T3", "MUTATION flag OFF: traj grads change", dt_off > 0, traj_grad_diff=dt_off,
             norms_and_traj_clip_factor=nr)
    dt_bad, _ = split_diff(m3["badpart_w0.1"]["cap"], m3["badpart_w10"]["cap"])
    S.record("T3", "MUTATION score_head in traj group: change", dt_bad > 0, traj_grad_diff=dt_bad)
    on_rows = [(round(r["gn_traj"], 3), round(r["gn_score"], 3), round(r["clip_coef_traj"], 4))
               for r in m3["on_w10"]["m"] if r.get("event") == "measures"]
    S.record("T3", "ON logs both group norms + traj clip factor", len(on_rows) == 4, rows_w10=on_rows)

    # ---------------------------------------------------------------- T4 M3 groups
    o, _ = S.child("partition", pat, real_vitl=True)
    pv = json.load(open(o))
    S.record("T4", "ViT-L 4-cam: every trainable tensor in 1 group", pv["exactly_once"] and
             pv["score_modules"] == ["score_dec", "score_head", "score_q_mlp"],
             n_trainable=pv["n_trainable"], n_traj=pv["n_traj"], n_score=pv["n_score"],
             numel_traj=pv["numel_traj"], numel_score=pv["numel_score"], score_modules=pv["score_modules"])
    o, _ = S.child("partition", pat)
    pf = json.load(open(o))
    d, m = pf["detached"], pf["MUTATION_context_attached"]
    S.record("T4", "gradient flow == name partition (detached)",
             d["n_score_loss_hits_traj_group"] == 0 and d["n_traj_loss_hits_score_group"] == 0
             and d["flow_score_equals_name_group"] and d["clip_premise"] == "accepted",
             score_loss_reaches=d["score_loss_reaches"], never_touched=d["never_touched"])
    S.record("T4", "MUTATION context attached: leak + refusal",
             m["n_score_loss_hits_traj_group"] > 0 and m["clip_premise"] == "REFUSED",
             score_loss_hits_traj_group=m["n_score_loss_hits_traj_group"], e_g=m["score_loss_hits_traj_group"],
             premise=m["clip_premise"])

    # ---------------------------------------------------------------- T5 M4b units
    o, _ = S.child("unit_m4b", pat)
    u = json.load(open(o))
    for k, v in u.items():
        vals = {kk: vv for kk, vv in v.items() if kk != "ok"}
        S.record("T5", k, v["ok"], **vals)

    # ---------------------------------------------------------------- T6 M4b hook
    o1, _ = S.child("hook", orig)
    o2, _ = S.child("hook", pat, patched=True)
    h1, h2 = torch.load(o1, weights_only=False), torch.load(o2, weights_only=False)
    dd = max(_maxdiff(h1[k], h2[k]) for k in ("traj", "score", "loss", "idx"))
    S.record("T6", "not installed: forward+loss == today", dd == 0.0, max_diff=dd, expect="exactly 0.0")
    S.record("T6", "installed: free slots bit-identical", h2["free_equal"], free_equal=h2["free_equal"])
    S.record("T6", "installed: reserved never ahead of anchor", h2["exceed_max"] <= 1e-3 and
             h2["on_raw_polyline_max"] <= 1e-4, exceed_max_m=h2["exceed_max"],
             on_raw_polyline_max_m=h2["on_raw_polyline_max"])
    S.record("T6", "installed: no new state", h2["keys_same"] and h2["trainable_same"] and h2["fingerprint_same"],
             keys=h2["keys_same"], trainable=h2["trainable_same"], frozen_fingerprint=h2["fingerprint_same"])
    S.record("T6", "installed: the scorer sees the anchored fan", h2["score_changed"],
             score_changed=h2["score_changed"])

    # ---------------------------------------------------------------- T7 M4b drift on fast-only targets
    bankF = tmp / "bankFast"
    make_bank(bankF, 32, [12.0, 15.0, 18.0], [0.0], world_fields=False)
    drift = {}
    for tag, extra, mut in (("ON", ["--slow-slots", "8"], None), ("OFF", [], None),
                            ("MUT_no_retime", ["--slow-slots", "8"], ["m4b_no_retime"]),
                            ("MUT_loss_through_hook", ["--slow-slots", "8"], ["m4b_loss_through_hook"]),
                            ("MUT_loss_through_hook_fullgrad", ["--slow-slots", "8"],
                             ["m4b_loss_through_hook_fullgrad"])):
        out = tmp / f"runF_{tag}"
        S.child("trainer", pat, argv=targs(bankF, out, empty, *extra, "--lr", "5e-3", "--no-cosine",
                                           steps=a.drift_steps, batch=8, accum=1, log_every=50),
                mutate=mut)
        o, _ = S.child("eval_slots", pat, ckpt=str(out / "model_final.pt"), bank=str(bankF), mutate=mut)
        drift[tag] = json.load(open(o))
        drift[tag]["traj_L1"] = [round(r["traj_L1"], 3) for r in metrics(out) if "traj_L1" in r]
    o, _ = S.child("trainer", pat, argv=targs(bankF, tmp / "runF_init", empty, steps=1, batch=8, accum=1))
    o, _ = S.child("eval_slots", pat, ckpt=str(tmp / "runF_init" / "model_final.pt"), bank=str(bankF))
    init = json.load(open(o))
    on, off, mu = drift["ON"], drift["OFF"], drift["MUT_no_retime"]
    S.record("T7", "ON: reserved slots within their anchor", on["installed"] and on["exceed_max_m"] <= 1e-3,
             exceed_max_m=on["exceed_max_m"], dev_mean_m=on["dev_mean_m"],
             euclid_to_straight_anchor_max_m=on["euclid_to_straight_anchor_max_m"],
             reserved_len_4s=on["reserved_len_4s_mean_m"], anchor_len_4s=on["anchor_len_4s_mean_m"],
             raw_reserved_exceed_max_m=on.get("raw_reserved_exceed_max_m"))
    S.record("T7", "OFF (contrast): the same slots drift", off["exceed_max_m"] >= 10.0 and not off["installed"],
             exceed_max_m=off["exceed_max_m"], reserved_len_4s=off["reserved_len_4s_mean_m"],
             at_init_exceed_max_m=init["exceed_max_m"], target_len_4s=off["target_len_4s_mean_m"])
    S.record("T7", "MUTATION loss kept, re-timing removed: drift", mu["exceed_max_m"] >= 10.0,
             exceed_max_m=mu["exceed_max_m"], reserved_len_4s=mu["reserved_len_4s_mean_m"])
    bar = 0.1 * init["best_free_wta_m"]
    tmax = 2.0 * off["target_len_4s_mean_m"]
    S.record("T7", "free slots learn with M4b ON (<= 10 % of init)", on["best_free_wta_m"] <= bar
             and off["best_free_wta_m"] <= bar, best_free_wta_init_m=init["best_free_wta_m"],
             on_m=on["best_free_wta_m"], off_m=off["best_free_wta_m"], bar_m=bar,
             traj_L1_on=on["traj_L1"], traj_L1_off=off["traj_L1"])
    S.record("T7", "raw reserved proposals stay bounded (ON)", on.get("raw_reserved_len_4s_max_m", 1e9) <= tmax,
             raw_len_4s_max_m=on.get("raw_reserved_len_4s_max_m"), bar_m=tmax)
    for tag in ("MUT_loss_through_hook", "MUT_loss_through_hook_fullgrad"):
        d = drift[tag]
        broke = d["best_free_wta_m"] > bar or d.get("raw_reserved_len_4s_max_m", 0.0) > tmax
        S.record("T7", f"MUTATION {tag[4:]}: free stall or raw blow-up", broke,
                 best_free_wta_m=d["best_free_wta_m"], raw_len_4s_max_m=d.get("raw_reserved_len_4s_max_m"),
                 exceed_max_m=d["exceed_max_m"])

    # ---------------------------------------------------------------- T8 M4a twins
    t8(S, tmp, orig, pat, empty)
    # ---------------------------------------------------------------- T9 mid-run switches
    t9(S, tmp, orig, pat, empty, bankA, opA)
    # ---------------------------------------------------------------- T10 M6 heading-branch trap
    t10(S, tmp, orig, pat, empty)
    # ---------------------------------------------------------------- T11 M6b tangent-consistency term (units)
    o, _ = S.child("unit_m6b", pat)
    for k, v in json.load(open(o)).items():        # the child writes JSON (as T10's unit_m6); S.child returns its PATH
        S.record("T11", k, v["ok"], **{kk: vv for kk, vv in v.items() if kk != "ok"})
    return _finish(S, res_path, t_start, a, tmp)


def _twin_tool(root, *args):
    p = subprocess.run([sys.executable, str(Path(root) / "slow_twins.py"), *map(str, args)], capture_output=True,
                       text=True, cwd=str(root), env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    return p.returncode, p.stdout + p.stderr


def _f64_bits(xs):
    return [struct.pack("<d", float(v)) for row in xs for v in row]


def t8(S, tmp, orig, pat, empty):
    import importlib.util
    import numpy as np
    sys.path.insert(0, str(pat))
    import slow_copies as SC
    spec = importlib.util.spec_from_file_location("grow_assemble", PKG / "code" / "grow_assemble.py")
    GA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(GA)
    bankW, opW = tmp / "bankW", tmp / "bankW_op"
    r0, r1 = make_bank(bankW, 12, [6.0, 11.0, 16.0], [0.0, 0.015, -0.02], rank1_every=2, onpolicy=opW)
    # ---- factor 1.0 identity: synthetic and 2,000 REAL navtrain rows
    real = tmp / "bankReal"
    real.mkdir()
    n_real = 0
    if REAL_BANK.exists():
        with open(REAL_BANK, encoding="utf-8") as src, open(real / "targets_rank0.jsonl", "w", encoding="utf-8",
                                                          newline="\n") as dst:
            for line in src:
                dst.write(line)
                n_real += 1
                if n_real >= 2000:
                    break
    for tag, bank, n_src in (("synthetic", bankW, len(r0) + len(r1)), ("real_navtrain", real, n_real)):
        if not n_src:
            S.record("T8", f"factor 1.0 bit-exact ({tag})", False, note="source bank missing -- NOT RUN")
            continue
        out = tmp / f"tw1_{tag}"
        ranks = "0,1" if tag == "synthetic" else "0"
        rc, txt = _twin_tool(pat, "--bank", bank, "--out", out, "--factors", "1.0", "--src-ranks", ranks,
                             "--min-divergence-m", "0")
        src = {}
        for rk in ranks.split(","):
            for line in open(bank / f"targets_rank{rk}.jsonl", encoding="utf-8"):
                r = json.loads(line)
                src[(r["log_name"], r["token"], r["step"], r["rank"])] = r
        n, exact = 0, 0
        for f in sorted(out.glob("targets_rank*.jsonl")):
            for line in open(f, encoding="utf-8"):
                tw = json.loads(line)
                s = src[(tw["log_name"], tw["token"], tw["step"], tw["slow"]["src_rank"])]
                n += 1
                exact += int(_f64_bits(tw["traj"]) == _f64_bits(s["traj"]))
        S.record("T8", f"factor 1.0 twin == target BIT-exact ({tag})", rc == 0 and n == n_src and exact == n,
                 twins=n, bit_exact=exact, sources=n_src)
    # ---- the default factors: the tool's traj IS slow_copy's (float64 bits), ranks mapped, format valid
    outW = tmp / "twW"
    rc, txt = _twin_tool(pat, "--bank", bankW, "--out", outW, "--factors", "0.75,0.5", "--src-ranks", "0,1")
    man = json.load(open(outW / "slow_twins_manifest.json"))
    src = {(r["log_name"], r["token"], r["step"], r["rank"]): r for r in r0 + r1}
    n, same, ok_fmt, keys_ok, dropped = 0, 0, 0, 0, 0
    for f in sorted(outW.glob("targets_rank*.jsonl")):
        R = int(f.stem[len("targets_rank"):])
        for line in open(f, encoding="utf-8"):
            tw = json.loads(line)
            s = src[(tw["log_name"], tw["token"], tw["step"], tw["slow"]["src_rank"])]
            want = SC.slow_copy(np.asarray(s["traj"], np.float64), tw["slow"]["factor"]).tolist()
            n += 1
            same += int(_f64_bits(tw["traj"]) == _f64_bits(want))
            ok_fmt += int(bool(GA._target_ok(tw)))
            mp = man["rank_map"][str(R)]
            keys_ok += int(set(tw) == (set(s) - {"traj_world"}) | {"slow"} and tw["rank"] == R
                           and mp["src_rank"] == tw["slow"]["src_rank"] and mp["factor"] == tw["slow"]["factor"])
            dropped += int("traj_world" in s and "traj_world" not in tw and tw["slow"]["traj_world_dropped"])
    S.record("T8", "twins == slow_copy (bits), valid rows, rank map", rc == 0 and n > 0 and same == n
             and ok_fmt == n and keys_ok == n, twins=n, equal_to_slow_copy=same, grow_assemble_target_ok=ok_fmt,
             fields_and_rank_ok=keys_ok, traj_world_dropped=dropped, rank_map=man["rank_map"])
    # ---- the OPTIONAL decel-limited variant: bounded start, on the path, never ahead; the pure copy is not
    sys.path.insert(0, str(pat))
    import slow_twins as STW

    def decel_stats(twin_dir, src_rows, limit=None):
        """worst_a: the largest implied first-interval deceleration; with `limit`, the largest EXCESS over
        max(limit, the source's own) -- the variant never runs ahead of its source, so a source that is
        itself slower than braking allows bounds the twin, not the limit."""
        worst_a, worst_poly, worst_ahead, n_ = -1e9, 0.0, -1e9, 0
        for f in sorted(twin_dir.glob("targets_rank*.jsonl")):
            for line in open(f, encoding="utf-8"):
                tw = json.loads(line)
                s = src_rows[(tw["log_name"], tw["token"], tw["step"], tw["slow"]["src_rank"])]
                n_ += 1
                a_tw = STW.first_interval_decel(s["ego"], tw["traj"])
                if limit is not None:
                    a_tw -= max(limit, STW.first_interval_decel(s["ego"], s["traj"]))
                worst_a = max(worst_a, a_tw)
                worst_poly = max(worst_poly, float(_polyline_distance(s["traj"], tw["traj"]).max()))
                cs = np.cumsum(np.linalg.norm(np.diff(np.vstack([[0, 0], np.asarray(s["traj"])[:, :2]]), axis=0), axis=1))
                ct = np.cumsum(np.linalg.norm(np.diff(np.vstack([[0, 0], np.asarray(tw["traj"])[:, :2]]), axis=0), axis=1))
                worst_ahead = max(worst_ahead, float((ct - cs).max()))
        return n_, worst_a, worst_poly, worst_ahead
    real_src = {}
    if n_real:
        for line in open(real / "targets_rank0.jsonl", encoding="utf-8"):
            r = json.loads(line)
            real_src[(r["log_name"], r["token"], r["step"], r["rank"])] = r
    for tag, bank, srcs, ranks in (("synthetic", bankW, src, "0,1"), ("real_navtrain", real, real_src, "0")):
        if not srcs:
            continue
        od = tmp / f"twDL_{tag}"
        rc, _ = _twin_tool(pat, "--bank", bank, "--out", od, "--factors", "0.75,0.5", "--src-ranks", ranks,
                           "--decel-limit", "4")
        n_, wa, wp, wh = decel_stats(od, srcs, limit=4.0)
        S.record("T8", f"decel-limited 4 m/s2: bounded/on path/not ahead ({tag})",
                 rc == 0 and n_ > 0 and wa <= 0.05 and wp <= 1e-6 and wh <= 1e-6, twins=n_,
                 max_excess_over_bound=wa, max_off_path_m=wp, max_ahead_m=wh)
        if tag == "real_navtrain":
            od2 = tmp / "twPure_real"
            _twin_tool(pat, "--bank", bank, "--out", od2, "--factors", "0.75,0.5", "--src-ranks", "0")
            n2, wa2, _, _ = decel_stats(od2, srcs)
            S.record("T8", "CONTRAST the pure slow copy breaks that bound", wa2 > 8.0, twins=n2,
                     max_implied_decel=wa2)
            od3 = tmp / "twDL1_real"
            _twin_tool(pat, "--bank", bank, "--out", od3, "--factors", "1.0", "--src-ranks", "0",
                       "--decel-limit", "4", "--min-divergence-m", "0")
            e1 = 0.0
            for line in open(od3 / "targets_rank2.jsonl", encoding="utf-8"):
                tw = json.loads(line)
                s = srcs[(tw["log_name"], tw["token"], tw["step"], 0)]
                e1 = max(e1, float(np.abs(np.asarray(tw["traj"]) - np.asarray(s["traj"])).max()))
            S.record("T8", "decel-limited factor 1.0 == source (rounding)", e1 <= 1e-9, max_abs_diff=e1)
    # ---- append-only + refusals
    before = {f.name: f.read_bytes() for f in outW.glob("targets_rank*.jsonl")}
    rc2, txt2 = _twin_tool(pat, "--bank", bankW, "--out", outW, "--factors", "0.75,0.5", "--src-ranks", "0,1")
    after = {f.name: f.read_bytes() for f in outW.glob("targets_rank*.jsonl")}
    S.record("T8", "second pass appends nothing (append-only)", rc2 == 0 and before == after,
             files=len(after), unchanged=before == after)
    refusals = {
        "out == bank": _twin_tool(pat, "--bank", bankW, "--out", bankW)[0],
        "factor 1.5": _twin_tool(pat, "--bank", bankW, "--out", tmp / "tw_bad1", "--factors", "1.5")[0],
        "twin rank collides w/ genuine rank 1": _twin_tool(pat, "--bank", bankW, "--out", tmp / "tw_bad2",
                                                           "--rank-base", "1")[0],
        "twin ranks overlap source ranks": _twin_tool(pat, "--bank", bankW, "--out", tmp / "tw_bad4",
                                                      "--src-ranks", "0,1", "--rank-base", "1")[0],
    }
    tw_bank = tmp / "bank_twin_src"
    tw_bank.mkdir()
    shutil.copy2(outW / "targets_rank2.jsonl", tw_bank / "targets_rank0.jsonl")
    refusals["twin of a twin"] = _twin_tool(pat, "--bank", tw_bank, "--out", tmp / "tw_bad3")[0]
    S.record("T8", "tool refusals (rc 4 each)", all(v == 4 for v in refusals.values()), **refusals)
    # ---- the REAL loader: bank + twins in one directory
    mixed = tmp / "bankMixed"
    shutil.copytree(bankW, mixed)
    for f in outW.glob("targets_rank*.jsonl"):
        shutil.copy2(f, mixed / f.name)
    o, _ = S.child("loader", pat, patched=True, slow_twins=True, bank=str(mixed), onpolicy=str(opW))
    L = json.load(open(o))
    c = L.get("twin_checks", {})
    S.record("T8", "patched loader + --slow-twins: twins load right", not L["refused"] and c.get("n", 0) == n and
             c["traj_exact"] == n and c["ego_goal_exact"] == n and c["images_same"] == n
             and c["set_same_as_source"] == n, per_rank=L.get("per_rank"), group_sizes=L.get("group_sizes"),
             **{k: v for k, v in c.items()}, cov_norm=L.get("cov_norm"))
    o, _ = S.child("loader", pat, patched=True, slow_twins=False, bank=str(mixed), onpolicy=str(opW))
    L2 = json.load(open(o))
    S.record("T8", "GUARD patched loader w/o --slow-twins refuses", L2["refused"] and "TWIN" in L2.get("msg", ""),
             refused=L2["refused"], msg=L2.get("msg", "")[:90])
    o, _ = S.child("loader", orig, bank=str(mixed), onpolicy=str(opW))
    L3 = json.load(open(o))
    c3 = L3.get("twin_checks", {})
    S.record("T8", "HAZARD today's loader absorbs twins silently", not L3["refused"] and c3.get("n", 0) == n,
             refused=L3["refused"], twins_loaded=c3.get("n"), twins_with_NO_scorer_set=c3.get("set_empty"),
             note="today's code would train on them with no flag, and without their scorer labels")


def t9(S, tmp, orig, pat, empty, bankA, opA):
    import torch
    # ---- a checkpoint written by TODAY's trainer, halted mid-run
    base = tmp / "runR_base"
    args = targs(bankA, base, empty, "--ckpt-every-steps", "2", steps=6, onpolicy=opA)
    o, _ = S.child("trainer", orig, argv=args + ["--halt-after-steps", "2"])
    rc0 = torch.load(o, weights_only=False)["rc"]
    runs = {}
    least = "8,16,19,21,22,34,38,39"                 # the realistic switch: the least-used slots, named
    cases = (("orig_resume", orig, []), ("patched_resume_off", pat, []),
             ("m4b_undeclared", pat, ["--slow-slots", "8"]),
             ("m4b_indices_undeclared", pat, ["--slow-slots", "8", "--slow-slot-indices", least,
                                              "--declare-change", "slow_slots"]),
             ("m4b_declared", pat, ["--slow-slots", "8", "--slow-slot-indices", least,
                                    "--declare-change", "slow_slots", "--declare-change", "slow_slot_indices"]),
             ("m3_undeclared", pat, ["--clip-split"]),
             ("m3_declared", pat, ["--clip-split", "--declare-change", "clip_split"]))
    for tag, root, extra in cases:
        d = tmp / f"runR_{tag}"
        shutil.copytree(base, d)
        o, lf = S.child("trainer", root, argv=targs(bankA, d, empty, "--ckpt-every-steps", "2", "--resume",
                                                    *extra, steps=6, onpolicy=opA))
        runs[tag] = {"rc": torch.load(o, weights_only=False)["rc"], "log": lf.read_text(encoding="utf-8"),
                     "dir": d, "m": metrics(d)}
    S.record("T9", "today's trainer halts at step 2 (rc 7)", rc0 == 7, rc=rc0)
    dw = _maxdiff(load_ckpt(runs["orig_resume"]["dir"]), load_ckpt(runs["patched_resume_off"]["dir"]))
    S.record("T9", "patched resume, all OFF == today's resume", runs["patched_resume_off"]["rc"] == 0
             and runs["orig_resume"]["rc"] == 0 and dw == 0.0, max_state_diff=dw, expect="exactly 0.0")
    for tag, key in (("m4b_undeclared", "--slow-slots: 0 -> 8"), ("m3_undeclared", "--clip-split: False -> True"),
                     ("m4b_indices_undeclared", "--slow-slot-indices: '' -> '8,16,19,21,22,34,38,39'")):
        r = runs[tag]
        S.record("T9", f"{tag}: refused", r["rc"] == 4 and "REFUSING TO RESUME" in r["log"] and key in r["log"],
                 rc=r["rc"], names=key)
    for tag, key in (("m4b_declared", "slow-slots"), ("m3_declared", "clip-split")):
        r = runs[tag]
        ev = [e for e in r["m"] if e.get("event") == "declared_change"]
        mrows = [e for e in r["m"] if e.get("event") == "measures"]
        S.record("T9", f"{tag}: resumes + logs the change", r["rc"] == 0 and ev and key in json.dumps(ev)
                 and len(mrows) > 0, rc=r["rc"], declared=ev[0]["changes"] if ev else None,
                 measures_rows=len(mrows))
    # the switched checkpoint carries the measure, and the CONSUMER path installs it
    o, _ = S.child("eval_slots", pat, ckpt=str(runs["m4b_declared"]["dir"] / "model_final.pt"), bank=str(bankA))
    ev = json.load(open(o))
    S.record("T9", "switched ckpt: load_for_inference installs M4b", ev["installed"] and ev["exceed_max_m"] <= 1e-3
             and (ev["meta_measures"] or {}).get("slow_slots") is not None
             and ev["slots"] == [int(x) for x in least.split(",")], installed=ev["installed"],
             exceed_max_m=ev["exceed_max_m"], fmt=ev["fmt"], reserved=ev["slots"])
    o, _ = S.child("eval_slots", orig, ckpt=str(runs["m4b_declared"]["dir"] / "model_final.pt"), bank=str(bankA))
    ev0 = json.load(open(o))
    S.record("T9", "HAZARD today's consumer ignores M4b", not ev0["installed"], installed=ev0["installed"],
             exceed_max_m=ev0["exceed_max_m"], note="an eval/dump without measures_consumers.patch runs the RAW slots")

    # ---- M4a under --grow (the live run's mode): the twin file lands mid-epoch
    bankG = tmp / "bankG"
    make_bank(bankG, 8, [9.0, 14.0], [0.0, 0.02], rank1_every=0)
    gbase = tmp / "runG_base"
    gargs = ["--grow", "--epochs", "3", "--grow-scenes", "8", "--ckpt-every-steps", "1"]
    o, _ = S.child("trainer", orig, argv=targs(bankG, gbase, empty, *gargs, "--halt-after-steps", "1",
                                               batch=2, accum=1))
    rcg = torch.load(o, weights_only=False)["rc"]
    rc_tw, _ = _twin_tool(pat, "--bank", bankG, "--out", bankG, "--factors", "0.5", "--into-live-bank")
    graw = {}
    for tag, root, extra in (("orig", orig, []), ("patched_no_flag", pat, []),
                             ("patched_declared", pat, ["--slow-twins", "--declare-change", "slow_twins"])):
        d = tmp / f"runG_{tag}"
        shutil.copytree(gbase, d)
        o, lf = S.child("trainer", root, argv=targs(bankG, d, empty, *gargs, "--resume", *extra,
                                                    batch=2, accum=1))
        c = torch.load(o, weights_only=False)
        banks = [(e["epoch"], e["per_rank"]) for e in metrics(d) if e.get("event") == "bank"]
        graw[tag] = {"rc": c["rc"], "msg": c["msg"], "banks": banks,
                     "steps": max([e["step"] for e in metrics(d) if "traj_L1" in e] or [-1])}
    S.record("T9", "grow: halted run + twin file written into its bank", rcg == 7 and rc_tw == 0, rc=rcg,
             tool_rc=rc_tw)
    g = graw["orig"]
    S.record("T9", "HAZARD today's grow trainer absorbs twins", g["rc"] == 0 and
             any("2" in pr for ep, pr in g["banks"] if ep >= 1), bank_events=g["banks"],
             note="no flag, no identity change -- at the next epoch boundary")
    g = graw["patched_no_flag"]
    S.record("T9", "GUARD patched grow w/o --slow-twins refuses", g["rc"] != 0 and "TWIN" in (g["msg"] or ""),
             rc=g["rc"], last_step=g["steps"], msg=(g["msg"] or "")[:80])
    g = graw["patched_declared"]
    first = [pr for ep, pr in g["banks"]][:1]
    S.record("T9", "declared: twins join AT the next epoch boundary", g["rc"] == 0 and first and "2" not in first[0]
             and any("2" in pr for ep, pr in g["banks"] if ep >= 1), bank_events=g["banks"])


LIVE_BANK_R0 = Path("D:/Projects/TanitAD/data/refe_navtrain10/r0/targets_rank0.jsonl")   # the live run's DEV10 rows


def t10(S, tmp, orig, pat, empty, steps=1500):
    """M6: the wrapped heading L1 traps heading outputs on the wrong 2*pi branch; --yaw-loss plain prevents it and a
    declared mid-run switch frees them. End to end on REAL live-bank targets (the trap needs their distribution)."""
    import torch
    o, _ = S.child("unit_m6", pat)
    for k, v in json.load(open(o)).items():
        S.record("T10", k, v["ok"], **{kk: vv for kk, vv in v.items() if kk != "ok"})
    bankS = tmp / "bankSeam"
    make_bank(bankS, 4, [8.0], [0.0], world_fields=False)
    rows = [json.loads(l) for l in open(bankS / "targets_rank0.jsonl", encoding="utf-8")]
    rows[1]["traj"][19][2] = 3.1                                      # one target at the +-pi seam
    with open(bankS / "targets_rank0.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    o, lf = S.child("trainer", pat, argv=targs(bankS, tmp / "runSeam", empty, "--yaw-loss", "plain", steps=1,
                                               batch=2, accum=1))
    rc = torch.load(o, weights_only=False)["rc"]
    S.record("T10", "seam guard refuses --yaw-loss plain end to end", rc == 4 and "seam" in lf.read_text(encoding="utf-8"),
             rc=rc)
    if not LIVE_BANK_R0.exists():
        S.record("T10", "trap / fix / recovery on live-bank rows", False, note=f"{LIVE_BANK_R0} missing -- NOT RUN")
        return
    bankH = tmp / "bankHeading"
    bankH.mkdir()
    lines = open(LIVE_BANK_R0, encoding="utf-8").readlines()
    import numpy as np_
    pick = sorted(np_.random.default_rng(0).choice(len(lines), 3000, replace=False))
    with open(bankH / "targets_rank0.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i in pick:
            f.write(lines[i])
    base = ["--lr", "1e-3", "--seed", "0"]
    runs = {}
    for tag, root, extra, n, halt in (("off", orig, [], steps, 0), ("plain", pat, ["--yaw-loss", "plain"], steps, 0),
                                      ("sw_base", pat, [], 2 * steps, steps)):
        d = tmp / f"runH_{tag}"
        argv = targs(bankH, d, empty, *base, *extra, steps=n, batch=16, accum=1, log_every=500)
        if halt:
            argv += ["--ckpt-every-steps", str(halt), "--halt-after-steps", str(halt)]
        o, _ = S.child("trainer", root, argv=argv)
        runs[tag] = {"rc": torch.load(o, weights_only=False)["rc"], "dir": d}
    for tag, extra in (("sw_undeclared", ["--yaw-loss", "plain"]), ("sw_wrapped", []),
                       ("sw_plain", ["--yaw-loss", "plain", "--declare-change", "yaw_loss"])):
        d = tmp / f"runH_{tag}"
        shutil.copytree(runs["sw_base"]["dir"], d)
        o, lf = S.child("trainer", pat, argv=targs(bankH, d, empty, *base, *extra, "--resume", steps=2 * steps,
                                                   batch=16, accum=1, log_every=500))
        runs[tag] = {"rc": torch.load(o, weights_only=False)["rc"], "dir": d, "log": lf.read_text(encoding="utf-8")}
    ev = {}
    for tag in ("off", "plain", "sw_wrapped", "sw_plain"):
        o, _ = S.child("eval_heading", pat, ckpt=str(runs[tag]["dir"] / "model_final.pt"), bank=str(bankH))
        ev[tag] = json.load(open(o))
    S.record("T10", "DEFECT reproduces under today's wrapped loss", runs["off"]["rc"] == 0 and len(ev["off"]["trapped_steps"]) > 0,
             trapped_steps=ev["off"]["trapped_steps"], max_frac_beyond_pi=ev["off"]["max_frac_beyond_pi"])
    S.record("T10", "plain: no step trapped, every winner step <= 0.3 rad",
             runs["plain"]["rc"] == 0 and not ev["plain"]["trapped_steps"] and max(ev["plain"]["winner_wrap_err_by_step"]) <= 0.3,
             trapped_steps=ev["plain"]["trapped_steps"], max_winner_err=max(ev["plain"]["winner_wrap_err_by_step"]),
             max_frac_beyond_pi=ev["plain"]["max_frac_beyond_pi"])
    S.record("T10", "plain costs no position accuracy (<= +0.2 m)",
             ev["plain"]["winner_pos_L1_m"] <= ev["off"]["winner_pos_L1_m"] + 0.2,
             pos_off_m=ev["off"]["winner_pos_L1_m"], pos_plain_m=ev["plain"]["winner_pos_L1_m"])
    S.record("T10", "switch without --declare-change refused", runs["sw_undeclared"]["rc"] == 4 and
             "--yaw-loss: 'wrapped' -> 'plain'" in runs["sw_undeclared"]["log"], rc=runs["sw_undeclared"]["rc"])
    S.record("T10", "declared mid-run switch frees the trapped steps",
             runs["sw_plain"]["rc"] == 0 and not ev["sw_plain"]["trapped_steps"] and len(ev["sw_wrapped"]["trapped_steps"]) > 0,
             trapped_if_kept_wrapped=ev["sw_wrapped"]["trapped_steps"], trapped_after_switch=ev["sw_plain"]["trapped_steps"],
             max_winner_err_after_switch=max(ev["sw_plain"]["winner_wrap_err_by_step"]),
             pos_kept_m=ev["sw_wrapped"]["winner_pos_L1_m"], pos_switched_m=ev["sw_plain"]["winner_pos_L1_m"])


def _finish(S, res_path, t_start, a, tmp=None) -> int:
    bad = [f"{tid}:{name}" for tid, d in S.results.items() for name, v in d.items() if not v["ok"]]
    out = {"results": S.results, "failed": bad, "n_checks": sum(len(d) for d in S.results.values()),
           "seconds": round(time.time() - t_start, 1), "python": sys.version.split()[0],
           # the bytes that were TESTED (the copies taken at the start), not whatever the repo holds at the end
           "tested_sha256": dict(S.tested_sha256),
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(out, open(res_path, "w", encoding="utf-8"), indent=1, default=str)
    S.say(f"\n  {out['n_checks'] - len(bad)}/{out['n_checks']} checks PASS in {out['seconds']} s -> {res_path}")
    if tmp is not None and not a.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    elif tmp is not None:
        S.say(f"  kept {tmp}")
    S.say("ZZMEASURES_OK" if not bad else "ZZMEASURES_FAIL " + " ".join(bad))
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default=None)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the temporary directory")
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    ap.add_argument("--drift-steps", type=int, default=200)
    a = ap.parse_args()
    if a.child:
        CHILDREN[a.child](json.load(open(a.spec, encoding="utf-8")))
        return 0
    return main_parent(a)


if __name__ == "__main__":
    sys.exit(main())
