"""WP-D P-MAP arms (PREREG_WPD_PROBES.md, registered sha256 c054190b…): the refcv7 10 cm map branch retrained on a
FROZEN refcv7-50,400 trunk, supervised on x in [0, 40) m only, features recomputed on the fly.

Per step: the batch's LAST window stack [B, 3K, H, W] (the frame instant the map label is read at) -> the trunk's own
`normalise` -> no_grad: stem..layer2 on the NEWEST frame (the branch's stride-8 input, == `s8_from_normalised`, control
C-s8), layer1 of the same pass (arm PM1) and, for PM2, `forward_features` (the temporally fused stride-16 map). The
per-clip 0.25 m lift geometry comes from the model's own `_lift_bank_hires.for_episodes(map_ep)`. The branch
(`model._map_hires`: lift, shared 0.25 m encoder, refine, near lift, near refine) trains from its refcv7 weights on
the class-weighted 10 cm CE (`map_head_hires.hires_map_ce`) over supervised, lift-valid cells of rows [0, 400).
3,000 steps, batch 8, AdamW lr 1e-4 constant then cosine to 0 over the final 300, wd 1e-4, clip 10, seed.
Resumable: state every 250 steps; a job stops at its wall budget and the next job continues (the step count, not the
wall clock, defines the arm).

Arms: PM0 control · PM0r seed 1 · PM1 stride-4 near lift over 0-40 m (layer1 -> d_up, zero-init, ADDED after the
existing near block) · PM2 deeper-stage fusion (s8 input += up x2(1x1(s16 fused)), 1x1 zero-init) · PM3 soft lateral
line targets for lane / edge (mass 1 on the GT cell, exp(-k^2/2) at +-k = 1, 2 columns, renormalised) · PM4 =
PM1+PM2+PM3 · must-fail R-s4zero (PM1 fed zeros), R-s4shuf (PM1 with layer1 ROWS permuted per image), R-gtshift (PM0
with lane / edge GT moved +5 columns = +0.5 m, vacated cells IGNORED).
Evaluation (end of the arm): TRAIN-DIAG -> per-class thresholds on p_hat = softmax(z - ln w) (the diagnostics' F1 fit,
`diag_metrics.best_threshold` on exact histograms, bands 0-40 m pooled); EVAL-DIAG -> per-episode tolerant counts
(k = 0, 1, 2, 5, 10) per class x band and the drivable-boundary registration histogram, written to
<out>/<arm>/eval_acc.pt for `pmap_analyze.py`. Env: REFCV6_REPO, REFCV6_KIT, PYTHONPATH=<launch tree>/stack;
DIAG_CODE=<diagnostics code dir> (diag_metrics). ⛔ sha12 only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARMS = {"PM0": {}, "PM0r": {"seed": 1}, "PM1": {"s4": True}, "PM2": {"fpn": True}, "PM3": {"line": True},
        "PM4": {"s4": True, "fpn": True, "line": True},
        "R-s4zero": {"s4": True, "s4zero": True}, "R-s4shuf": {"s4": True, "s4shuf": True},
        "R-gtshift": {"gtshift": True}}
ROWS = 400                    # x in [0, 40) m at 0.1 m
LANE, EDGE = 2, 5             # diag_metrics.CLASS_KEYS: nocls, drivable, lane, crosswalk, arrow, edge, hatched, sidewalk
# Portable (bundle): every Thor path below can be overridden by an env var; unset == the Thor paths the arms ran with.
_D = os.environ.get("WPD_DATA", "/home/nvidia/data")
TRAIN_FULL = {"--eval-cache": os.environ.get("WPD_TRAIN_CACHE", f"{_D}/refcv6-b1-416x1024-train"),
              "--eval-labels": f"{_D}/v8labels/labels/s2_labels_v8_train.jsonl.gz",
              "--speed-max-sidecar-v6-eval": f"{_D}/refcv6_speed_max_v8_train.jsonl"}
TRAIN_DIAG = {"--eval-cache": os.environ.get("WPD_TRAIN_DIAG_CACHE", "/home/nvidia/refcv7_post/diag/cache/train_diag139"),
              "--eval-labels": f"{_D}/v8labels/labels/s2_labels_v8_train.jsonl.gz",
              "--speed-max-sidecar-v6-eval": f"{_D}/refcv6_speed_max_v8_train.jsonl"}


class Feats:
    """The frozen trunk's map inputs for the batch's last window stack."""

    def __init__(self, model, tr):
        self.enc = model.core.encoder
        self.tr = tr

    @torch.no_grad()
    def __call__(self, b, dev, want_s4=False, want_s16=False):
        fr = self.tr.frames_to_device(b["frames"], dev)                 # [B, W, 3K, H, W] float [0, 1]
        x = self.enc.normalise(fr[:, -1])
        newest = x[:, -3:] if (str(self.enc.cfg.mode) == "shared" and self.enc.k > 1) else x
        lv = self.enc.memory_levers
        out = {}
        h = newest.contiguous(memory_format=torch.channels_last) if lv.get("channels_last") else newest
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(lv.get("bf16"))):
            for name, mod in self.enc.net.items():
                h = mod(h)
                if name == "layer1":
                    out["s4"] = h.float().contiguous()
                if name == self.enc.s8_module:
                    out["s8"] = h.float().contiguous()
                    break
        if want_s16:
            out["s16"] = self.enc.forward_features(x, already_normalised=True)[0].float()
        return out


class MapArm(nn.Module):
    def __init__(self, branch, cfg, s4_dim, s16_dim, dev):
        super().__init__()
        import copy
        self.br = copy.deepcopy(branch).to(dev)
        self.cfg = cfg
        mc = self.br.cfg
        if cfg.get("s4"):
            from tanitad.models.map_head_hires import BEVLiftProjectFirst
            self.s4lift = BEVLiftProjectFirst(d_in=int(s4_dim), d_out=int(mc.d_up), n_heights=len(mc.heights_m),
                                              feat_hw=None).to(dev)
            with torch.no_grad():
                self.s4lift.proj.weight.zero_()
                self.s4lift.proj.bias.zero_()
                self.s4lift.unobserved.zero_()
        if cfg.get("fpn"):
            self.fpn = nn.Conv2d(int(s16_dim), int(self.br.d_image), 1).to(dev)
            nn.init.zeros_(self.fpn.weight)
            nn.init.zeros_(self.fpn.bias)

    def forward(self, f, grid, valid, H, W):
        from tanitad.models.map_head_hires import derive_near_geometry
        br = self.br
        s8 = f["s8"]
        if self.cfg.get("fpn"):
            s8 = s8 + F.interpolate(self.fpn(f["s16"]), size=s8.shape[-2:], mode="bilinear", align_corners=False)
        bev = br.lift(s8, grid, valid)
        feats = br.encoder(bev)
        rf = br.refine
        x = rf.inp(feats)
        x = F.interpolate(x, size=rf.out_hw, mode="bilinear", align_corners=False)
        # ⭐ only rows [0, ROWS + 4) are decoded: the two 3x3 convs below reach 2 rows, so rows < ROWS are EXACT
        # (the crop changes nothing the loss or the metric reads; it saves ~60 % of the 10 cm decoder)
        x = x[:, :, :ROWS + 4]
        if br.near is not None:
            n = br.near.rows
            top = x[:, :, :n] + br.near(s8, grid, valid)
            for blk in (br.near_refine or ()):
                top = blk(top)
            x = torch.cat([top, x[:, :, n:]], dim=2)
        if self.cfg.get("s4"):
            s4 = f["s4"]
            if self.cfg.get("s4zero"):
                s4 = torch.zeros_like(s4)
            if self.cfg.get("s4shuf"):
                perm = torch.stack([torch.randperm(s4.shape[2], device=s4.device) for _ in range(s4.shape[0])])
                s4 = torch.gather(s4, 2, perm[:, None, :, None].expand(-1, s4.shape[1], -1, s4.shape[3]))
            g, v = derive_near_geometry(grid, valid, near_rows=ROWS, out_w=int(rf.out_hw[1]))
            # the 0.25 m geometry is normalised for the STRIDE-8 map: (2 col + 8) / W - 1. Stride 4 needs
            # (2 col + 4) / W - 1, i.e. a shift of -4 / W (x) and -4 / H (y) in normalised units.
            g = torch.stack([g[..., 0] - 4.0 / W, g[..., 1] - 4.0 / H], -1)
            g = torch.where(v[..., None], g, torch.zeros_like(g))
            add = self.s4lift(s4, g, v)                                   # [B, d_up, ROWS, W10]
            x = torch.cat([x[:, :, :ROWS] + add, x[:, :, ROWS:]], dim=2)
        x = rf.act(rf.norm1(rf.conv1(x)))
        x = rf.act(rf.norm2(rf.conv2(x)))
        return rf.cls(x)[:, :, :ROWS], valid.any(dim=1)


def soft_line_targets(codes, sup):
    """[B, H, W] codes -> [B, 8, H, W] soft targets: one-hot + lateral (column) Gaussian spread of lane / edge."""
    oh = F.one_hot(codes.clamp(0, 7).long(), 8).permute(0, 3, 1, 2).float() * sup[:, None]
    k = torch.tensor([math.exp(-2.0), math.exp(-0.5), 0.0, math.exp(-0.5), math.exp(-2.0)], device=codes.device)
    t = oh.clone()
    for c in (LANE, EDGE):
        m = oh[:, c:c + 1]
        spread = F.conv2d(m, k.view(1, 1, 1, 5), padding=(0, 2))
        t[:, c:c + 1] = t[:, c:c + 1] + spread * sup[:, None]
    t = t * sup[:, None]
    return t / t.sum(1, keepdim=True).clamp_min(1e-12)


def shift_lines(codes):
    """R-gtshift: lane / edge labels moved +5 columns (+0.5 m, toward +y); vacated cells -> 255 (not seen)."""
    out = codes.clone()
    for c in (LANE, EDGE):
        m = codes == c
        out[m] = 255
    for c in (LANE, EDGE):
        m = codes == c
        ms = torch.zeros_like(m)
        ms[..., 5:] = m[..., :-5]
        out[ms & (codes != 255)] = c
    return out


def map_loss(logits, codes, lv_fine, cw, cfg):
    sup = (codes != 255) & lv_fine
    if not bool(sup.any()):
        return logits.sum() * 0.0
    if cfg.get("line"):
        t = soft_line_targets(codes, sup)
        lp = F.log_softmax(logits.float(), dim=1)
        w = cw.view(1, -1, 1, 1)
        num = -(w * t * lp).sum(1)[sup].sum()
        den = (w * t).sum(1)[sup].sum()
        return num / den
    from tanitad.models.map_head_hires import hires_map_ce
    cd = torch.where(sup, codes, torch.full_like(codes, 255)).to(torch.uint8)
    return hires_map_ce(logits, cd, class_weight=cw)["loss"]


def build_ds(L, model, cfg, config, remap):
    args, _, _ = L.parse_args(config, remap=dict(L.PATH_REMAP, **remap))
    ds, eps, rec = L.build_eval_dataset(model, cfg, args, config)
    return ds, args


def grid_windows(ds, sha_of_eid, K):
    by_ep = {}
    for i, (e_i, t) in enumerate(ds.index):
        by_ep.setdefault(int(e_i), []).append((int(t), i))
    idx = []
    for e_i in sorted(by_ep, key=lambda e: sha_of_eid[int(ds.episodes[e].episode_id)]):
        ts = sorted(by_ep[e_i])
        for j in range(K):
            idx.append(ts[int((j + 0.5) * len(ts) / K)][1])
    return idx


@torch.no_grad()
def evaluate(arm, feats, model, ds, idx, sha_of_eid, dev, cw, L, dm, taus=None, workers=6):
    """taus None -> TRAIN histograms (for the fit); else -> per-episode tolerant counts with the fitted taus."""
    arm.eval()
    lw = dm.log_w(cw, dev)
    nb = (ROWS + dm.BAND_ROWS - 1) // dm.BAND_ROWS
    M = dm.band_matrix(ROWS, dev)
    win_sha = [sha_of_eid[int(ds.episodes[ds.index[i][0]].episode_id)] for i in idx]
    eps = sorted(set(win_sha))
    ep_pos = {s: k for k, s in enumerate(eps)}
    stats = ["pred", "gt", "inter"] + [f"tp{s}{k}" for k in dm.TOL_K for s in ("p", "g")]
    acc = {s: np.zeros((len(eps), 8, nb)) for s in stats}
    reg = torch.zeros(nb, len(dm.REG_BUCKETS), dtype=torch.int64, device=dev)
    hist = torch.zeros(8, nb, 2, dm.HIST_NB, dtype=torch.int64, device=dev)
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=8, shuffle=False,
                                     num_workers=workers)
    pos = 0
    H, W = model.core.encoder.cfg.image_hw
    for b in dl:
        B = int(b["map_fine_label"].shape[0])
        wins = list(range(pos, pos + B))
        pos += B
        lab = b["map_fine_label"].bool()
        if not bool(lab.any()):
            continue
        sel = lab.nonzero(as_tuple=False).flatten()
        f = feats(b, dev, want_s4=bool(arm.cfg.get("s4")), want_s16=bool(arm.cfg.get("fpn")))
        grid, valid = model._lift_bank_hires.for_episodes(b["map_ep"], device=dev)
        logits, lv = arm(f, grid, valid, H, W)
        from tanitad.models.map_head_hires import lift_valid_to_fine
        lvf = lift_valid_to_fine(lv, tuple(model._map_hires.cfg.out_hw))
        z = logits[sel.to(dev), :, :ROWS].float()
        cd = b["map_fine"].to(dev)[sel.to(dev), :ROWS].long()
        sup = (cd != 255) & lvf[sel.to(dev), :ROWS]
        G = dm.onehot_masks(cd, sup)
        if taus is None:
            hist += dm.score_histograms(dm.class_logits(z - lw.view(1, -1, 1, 1)), G, sup, nb)
            continue
        P = dm.decision_masks(z, sup, lw, {"thr": ("thr_phat", taus)})["thr"]
        tc = dm.tolerant_counts(P, G, M)
        tcn = {s: tc[s].cpu().numpy() for s in stats}
        for j, s_i in enumerate(sel.tolist()):
            e = ep_pos[win_sha[wins[s_i]]]
            for s in stats:
                acc[s][e] += tcn[s][j]
        edge = (cd == EDGE) & sup
        reg += dm.registration_hist(edge, dm.boundary(P[:, 1], sup), nb)
    arm.train()
    if taus is None:
        return hist.cpu()
    return {"acc": {s: torch.from_numpy(v) for s, v in acc.items()}, "episodes": eps,
            "reg_pred_drivable_vs_gt_edge": reg.cpu(), "rows": ROWS, "band_rows": dm.BAND_ROWS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", default=os.environ.get("WPD_RUN_DIR", "/home/nvidia/refcv7_run/runs/refcv7-r101-s0"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--budget-s", type=float, default=2100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-limit", type=int, default=0, help="SMOKE ONLY: first N eval windows")
    ap.add_argument("--timing-steps", type=int, default=0,
                    help="TIMING ONLY (not an arm): run N steps into --out/<arm>_timing, write timing.json "
                         "(s/step excluding the first 3, cuda max memory), no state, no eval")
    a = ap.parse_args()
    if a.timing_steps:
        a.arm_dir_suffix = "_timing"
    t0 = time.time()
    sys.path.insert(0, os.environ.get("DIAG_CODE", "/home/nvidia/refcv7_post/diag/code"))
    import diag_metrics as dm
    from tanitad.eval import refcv7_loader as L
    from tanitad.models.map_head_hires import lift_valid_to_fine
    dev = "cuda"
    tr = L.trainer()
    cfg_arm = dict(ARMS[a.arm])
    seed = int(cfg_arm.get("seed", 0))
    config = L.load_config(Path(a.run_dir) / "config.json")
    model, cfg, args, mrec = L.build_model(config, a.ckpt, device=dev, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    od = Path(a.out) / (a.arm + getattr(a, "arm_dir_suffix", ""))
    od.mkdir(parents=True, exist_ok=True)
    feats = Feats(model, tr)
    enc = model.core.encoder
    s4_dim = int(dict(enc.net.named_children())["layer1"][-1].conv3.out_channels) if hasattr(
        dict(enc.net.named_children())["layer1"][-1], "conv3") else 256
    arm = MapArm(model._map_hires, cfg_arm, s4_dim, 1024, dev)
    arm.train()
    for p in arm.parameters():
        p.requires_grad_(True)
    cw = model._map_hires_class_weight.detach().float().to(dev)
    H, W = enc.cfg.image_hw
    params = [p for p in arm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    state_p = od / "state.pt"
    step, log = 0, []
    gen = torch.Generator().manual_seed(seed)
    if a.timing_steps:
        torch.cuda.reset_peak_memory_stats()
        tds, _ = build_ds(L, model, cfg, config, TRAIN_FULL)
        sampler = torch.utils.data.RandomSampler(tds, replacement=True, num_samples=(a.timing_steps + 5) * a.batch,
                                                 generator=gen)
        it = iter(torch.utils.data.DataLoader(tds, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                                              prefetch_factor=2))
        ts, tw = [], []
        while len(ts) < a.timing_steps:
            t_a = time.time()
            b = next(it)
            t_b = time.time()
            lab = b["map_fine_label"].bool()
            if not bool(lab.any()):
                continue
            f_ = feats(b, dev, want_s4=bool(cfg_arm.get("s4")), want_s16=bool(cfg_arm.get("fpn")))
            grid, valid = model._lift_bank_hires.for_episodes(b["map_ep"], device=dev)
            logits, lv = arm(f_, grid, valid, H, W)
            sel = lab.nonzero(as_tuple=False).flatten().to(dev)
            lvf = lift_valid_to_fine(lv, tuple(model._map_hires.cfg.out_hw))[sel, :ROWS]
            codes = b["map_fine"].to(dev)[sel, :ROWS].long()
            loss = map_loss(logits[sel, :, :ROWS], codes, lvf, cw, cfg_arm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            torch.cuda.synchronize()
            ts.append(time.time() - t_a)
            tw.append(t_b - t_a)
        rec = {"arm": a.arm, "batch": a.batch, "workers": a.workers, "n_steps": len(ts),
               "s_per_step_median_excl_first3": float(np.median(ts[3:])) if len(ts) > 3 else None,
               "data_wait_s_median_excl_first3": float(np.median(tw[3:])) if len(tw) > 3 else None,
               "s_per_step_all": ts, "cuda_max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
               "device": torch.cuda.get_device_name(0), "NOT_AN_ARM": True}
        (od / "timing.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"[pmap] TIMING {a.arm} {rec['s_per_step_median_excl_first3']:.3f} s/step "
              f"(data wait {rec['data_wait_s_median_excl_first3']:.3f}) max mem {rec['cuda_max_mem_gb']:.2f} GB",
              flush=True)
        return 0
    if state_p.exists():
        st = torch.load(state_p, map_location="cpu", weights_only=False)
        arm.load_state_dict(st["arm"])
        opt.load_state_dict(st["opt"])
        step, log = int(st["step"]), list(st["log"])
        gen.set_state(st["gen"])
        print(f"[pmap] {a.arm} resumed at step {step}", flush=True)

    # ---- control C-s8 (once per arm, at its first job): my stride-8 pass == the trunk's s8_from_normalised ----
    if step < a.steps:
        tds, _ = build_ds(L, model, cfg, config, TRAIN_FULL)
        n = len(tds)
        if step == 0 and not (od / "control_s8.json").exists():
            b = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(tds, list(range(4))), batch_size=4)))
            f = feats(b, dev, want_s4=True)
            with torch.no_grad():
                x = enc.normalise(tr.frames_to_device(b["frames"], dev)[:, -1])
                ref = enc.s8_from_normalised(x, torch.arange(4, device=dev))
            d = float((f["s8"] - ref.float()).abs().max())
            (od / "control_s8.json").write_text(json.dumps({"max_abs_diff": d, "pass": d <= 1e-3,
                                                            "s4_shape": list(f["s4"].shape)}), encoding="utf-8")
            print(f"[pmap] C-s8 max_abs_diff {d:.2e}", flush=True)
            if d > 1e-3:
                raise SystemExit("[pmap] C-s8 FAILED: the stride-8 input is not the trunk's")
        tl = time.time()
        sampler = torch.utils.data.RandomSampler(tds, replacement=True, num_samples=(a.steps - step + 50) * a.batch,
                                                 generator=gen)
        it = iter(torch.utils.data.DataLoader(tds, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                                              prefetch_factor=2))
        while step < a.steps:
            b = next(it)
            lab = b["map_fine_label"].bool()
            if not bool(lab.any()):
                continue
            f_ = feats(b, dev, want_s4=bool(cfg_arm.get("s4")), want_s16=bool(cfg_arm.get("fpn")))
            grid, valid = model._lift_bank_hires.for_episodes(b["map_ep"], device=dev)
            logits, lv = arm(f_, grid, valid, H, W)
            sel = lab.nonzero(as_tuple=False).flatten().to(dev)
            lvf = lift_valid_to_fine(lv, tuple(model._map_hires.cfg.out_hw))[sel, :ROWS]
            codes = b["map_fine"].to(dev)[sel, :ROWS].long()
            if cfg_arm.get("gtshift"):
                codes = shift_lines(codes)
            loss = map_loss(logits[sel, :, :ROWS], codes, lvf, cw, cfg_arm)
            fct = 1.0 if step < a.steps - 300 else 0.5 * (1 + math.cos(math.pi * (step - (a.steps - 300)) / 300))
            for g in opt.param_groups:
                g["lr"] = a.lr * fct
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(params, 10.0))
            opt.step()      # ⛔ was MISSING (found in P-BOX's first PB0 job, same pattern) -- nothing trained
            step += 1
            if step % 50 == 0 or step == 1:
                log.append({"step": step, "loss": float(loss.detach()), "grad_norm": gn, "lr": a.lr * fct,
                            "s_per_step": (time.time() - tl) / max(1, step - (log[-1]["step"] if log else 0)),
                            "cuda_max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30})
                tl = time.time()
                print(f"[pmap] {a.arm} step {step} loss {float(loss.detach()):.4f} gn {gn:.2f} "
                      f"{time.time() - t0:.0f}s", flush=True)
            if step % 250 == 0 or step == a.steps or time.time() - t0 > a.budget_s:
                torch.save({"arm": arm.state_dict(), "opt": opt.state_dict(), "step": step, "log": log,
                            "gen": gen.get_state()}, str(state_p) + ".tmp")
                os.replace(str(state_p) + ".tmp", state_p)
            if time.time() - t0 > a.budget_s and step < a.steps:
                print(f"[pmap] {a.arm} PARTIAL at step {step} (budget) -- resumable", flush=True)
                return 2
    # ---- evaluation: TRAIN-DIAG fit, then EVAL-DIAG counts ----
    if (od / "eval_acc.pt").exists():
        print(f"[pmap] {a.arm} already evaluated", flush=True)
        return 0
    ftp = od / "fit.json"
    if not ftp.exists() and time.time() - t0 > a.budget_s - 600:
        print(f"[pmap] {a.arm} trained; fit + eval next job (budget)", flush=True)
        return 2
    if not ftp.exists():
        dds, _ = build_ds(L, model, cfg, config, TRAIN_DIAG)
        sha = dict(dds._vis1_sha12)
        hist = evaluate(arm, feats, model, dds, grid_windows(dds, sha, 8)[:a.eval_limit or None], sha, dev, cw, L, dm, taus=None,
                        workers=a.workers)
        h = hist.sum(dim=1).numpy()
        edges = dm.hist_edges()
        taus = [float(edges[dm.best_threshold(h[c, 0], h[c, 1])[0]]) for c in range(8)]
        ftp.write_text(json.dumps({"tau_phat_logit": taus, "split": "TRAIN-DIAG", "rows": ROWS}), encoding="utf-8")
        if time.time() - t0 > a.budget_s:
            print(f"[pmap] {a.arm} fit done, EVAL next job (budget)", flush=True)
            return 2
    taus = json.loads(ftp.read_text(encoding="utf-8"))["tau_phat_logit"]
    if time.time() - t0 > a.budget_s - 600:
        print(f"[pmap] {a.arm} fit done; eval next job (budget)", flush=True)
        return 2
    from tanitad.eval import refcv7_loader as L2  # noqa: F401  (same module; eval uses the run's own eval split)
    eds, _ = build_ds(L, model, cfg, config, {})
    sha = dict(eds._vis1_sha12)
    res = evaluate(arm, feats, model, eds, grid_windows(eds, sha, 8)[:a.eval_limit or None], sha, dev, cw, L, dm, taus=taus,
                   workers=a.workers)
    th0 = torch.cat([p_.detach().reshape(-1).float() for p_ in model._map_hires.parameters()])
    th1 = torch.cat([p_.detach().reshape(-1).float() for p_ in arm.br.parameters()])
    dtheta = float((th1 - th0).norm() / th0.norm().clamp_min(1e-12))
    if dtheta == 0.0 and step > 0:
        raise SystemExit(f"[pmap] {a.arm}: CONTROL C-step FAILED -- the branch did not change in {step} steps")
    res.update({"arm": a.arm, "cfg": cfg_arm, "taus": taus, "steps": step, "log": log, "eval_limit": a.eval_limit,
                "C_step_rel_param_change": dtheta})
    torch.save(res, str(od / "eval_acc.pt") + ".tmp")
    os.replace(str(od / "eval_acc.pt") + ".tmp", od / "eval_acc.pt")
    (od / "record.json").write_text(json.dumps({"arm": a.arm, "cfg": cfg_arm, "steps": step, "batch": a.batch,
                                                "C_step_rel_param_change": dtheta,
                                                "cuda_max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                                                "lr": a.lr, "seed": seed, "taus": taus, "log": log},
                                               indent=1), encoding="utf-8")
    print(f"[pmap] DONE {a.arm} wall {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
