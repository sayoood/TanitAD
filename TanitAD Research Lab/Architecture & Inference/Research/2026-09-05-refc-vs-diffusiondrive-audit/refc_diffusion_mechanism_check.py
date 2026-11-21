"""REF-C vs DiffusionDrive — numerical check of OUR "truncated diffusion" mechanism.

Runs the SHIPPED refcv3 weights (step 40,284, the frozen model-only copy the
five-panel reel was rendered from) on real B1-v7.2 EVAL windows and answers, with
numbers rather than prose, the seven questions the PI's audit asks of the
decoder loop in ``tanitad/refs/refc.py`` (``AnchoredDiffusionDecoder._decode`` and
the ``for i in range(steps)`` loop in ``forward``):

  Q1  Is ``x_est`` updated between passes?            -> per-pass fan displacement (m)
  Q2  Does the timestep embedding vary across passes? -> ``time_embed`` row geometry
                                                         + a t_idx=0-FORCED ablation
  Q3  Does ``_decode`` at t=0 differ from t=N?         -> paired offset/conf deltas
  Q4  Same noise schedule train vs inference?          -> train-mode noise measured
                                                         against the eval pass (std)
  Q5  How many passes run at inference?                -> t_idx trace for steps=8
  Q6  Does iterating move the fan TOWARD the target?   -> ADE / oracle-in-fan vs steps
  Q7  Is the loop a contraction (DDIM-like)?           -> displacement_k vs k

Two parts:
  A. trained refcv3 @ 40,284 on the three local EVAL clips (GPU, minutes);
  B. structural checks on a random-init ``refc_smoke_config`` model (CPU, seconds):
     eval determinism, steps=0 == classifier pass, t=0 vs t=2 differ, x updated,
     train-noise std, t_idx clamp for steps > diffusion_steps.

Every number is printed AND written to ``--out`` (JSON). Estimator disclosure: the
Part-A intervals are WINDOW-LEVEL paired SEs over overlapping windows from
n_clips = 3 — a MECHANISM probe, NOT the decision-grade episode-cluster
bootstrap; they must not be quoted as benchmark numbers.

The model code executed is the pod-identical tree under ``run_refcv3_viz/repo``
(the harness pins it to sys.path); ``_decode``, the denoise loop and
``CrossAttnLayer`` were diff-verified identical to the worktree before running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

VIZ = r"C:\Users\Admin\run_refcv3_viz"
TOOLS = os.path.join(VIZ, "repo", "taniteval", "tools")
sys.path.insert(0, TOOLS)
import refcv3_arm as arm  # noqa: E402  (bootstraps <repo>/stack, taniteval, scripts)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import tanitad  # noqa: E402
from tanitad.refs import refc  # noqa: E402
import refb_labels  # noqa: E402
from tanitad.data.v2_dataset import build_v2_providers  # noqa: E402


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class MemoEncoder(nn.Module):
    """Encode each window ONCE: the ten decoder configurations per window then
    share bit-identical ``kv``/``cond``, so every comparison is paired on the
    same conv map (cuDNN autotune can otherwise pick different kernels per call)."""

    def __init__(self, enc: nn.Module):
        super().__init__()
        self.enc = enc
        self._key = None
        self._out = None
        self.n_encodes = 0

    def forward(self, x):
        key = (x.data_ptr(), tuple(x.shape), x.dtype, x.device)
        if self._key == key:
            return self._out
        out = self.enc(x)
        self._key, self._out = key, out
        self.n_encodes += 1
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.enc, name)


class DecodeTap:
    """Wrap ``decoder._decode`` to record every (t_idx, x_est) it is called with,
    and optionally FORCE t_idx to a constant (the timestep ablation)."""

    def __init__(self, dec):
        self.dec = dec
        self.orig = dec._decode
        self.calls: list = []
        self.force_t = None

    def __enter__(self):
        def patched(kv, cond, x_est, t_idx):
            t_use = self.force_t if self.force_t is not None else t_idx
            self.calls.append((int(t_idx), int(t_use), x_est.detach()))
            return self.orig(kv, cond, x_est, t_use)
        self.dec._decode = patched
        return self

    def __exit__(self, *a):
        self.dec._decode = self.orig


def ade(t: torch.Tensor, gt: torch.Tensor) -> float:
    return float((t - gt).norm(dim=-1).mean())


def mean_se(v: list) -> dict:
    a = np.asarray(v, dtype=np.float64)
    n = int(a.size)
    return {"mean": float(a.mean()) if n else None,
            "se_window_level": float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
            "n": n}


def part_a(a, res: dict) -> None:
    dev = a.device
    torch.backends.cudnn.benchmark = False
    ck = os.path.join(VIZ, "ckpt", "ckpt_step40284_frozen.pt")
    cf = os.path.join(VIZ, "ckpt", "config.json")
    res["part_a"] = {"ckpt": ck, "ckpt_md5": md5(ck), "config": cf}
    t0 = time.time()
    model, cfg, targs, prov = arm.load_model(ck, cf, device=dev)
    res["part_a"]["load_s"] = round(time.time() - t0, 1)
    res["part_a"]["state_dict_load"] = prov["state_dict_load"]
    res["part_a"]["decoder_steps_in_harness"] = prov["decoder_steps"]
    res["part_a"]["decoder_mode_in_harness"] = prov["decoder_mode"]
    res["part_a"]["ckpt_step"] = prov["step"]
    dec = model.core.decoder
    res["part_a"]["cfg_decoder"] = {"diffusion_steps": dec.cfg.diffusion_steps,
                                    "noise_std": dec.cfg.noise_std,
                                    "d": dec.cfg.d, "layers": dec.cfg.layers,
                                    "n_heads": dec.cfg.n_heads}
    res["part_a"]["n_anchors"] = int(dec.anchors.shape[0])
    res["part_a"]["anchor_v0_conditioned"] = bool(getattr(dec, "anchor_v0_cond", False))
    res["part_a"]["decoder_has_dropout_modules"] = any(
        isinstance(m, nn.Dropout) for m in dec.modules())
    res["part_a"]["mha_dropout_p"] = sorted({float(m.dropout) for m in dec.modules()
                                             if isinstance(m, nn.MultiheadAttention)})

    # ---- Q2 static: the timestep embedding table ------------------------------
    te = dec.time_embed.weight.detach().float().cpu()
    rows = te.shape[0]
    d = torch.cdist(te, te)
    cos = torch.nn.functional.normalize(te, dim=-1) @ torch.nn.functional.normalize(te, dim=-1).T
    res["part_a"]["time_embed"] = {
        "rows": int(rows), "norms": [round(float(x), 4) for x in te.norm(dim=-1)],
        "pairwise_l2": [[round(float(x), 4) for x in r] for r in d],
        "pairwise_cos": [[round(float(x), 4) for x in r] for r in cos],
        "q_proj_bias_norm_for_scale": round(float(dec.traj_proj.bias.detach().norm()), 4),
        "traj_proj_weight_fro": round(float(dec.traj_proj.weight.detach().norm()), 4),
    }
    print("[A] time_embed rows", rows, "norms", res["part_a"]["time_embed"]["norms"],
          "pairwise L2", res["part_a"]["time_embed"]["pairwise_l2"])

    # ---- corpus ------------------------------------------------------------------
    tr = arm.trainer()
    Ds = arm.make_eval_dataset_class()
    eval_dir = os.path.join(VIZ, "data", "eval")
    eps = build_v2_providers([eval_dir], lru_size=4, verbose=False)
    ds = Ds(eps, window=cfg.core.window, max_horizon=20,
            channels=cfg.core.encoder.in_channels)
    ds.u8_frames = True
    horizons = list(cfg.core.trajectory.horizons)
    need = [h - 1 for h in horizons]
    n2 = sum(1 for h in horizons if h <= 20)
    res["part_a"]["horizons"] = horizons
    res["part_a"]["n_clips"] = len(eps)
    res["part_a"]["clips"] = [os.path.basename(p) for p in sorted(
        os.listdir(eval_dir)) if p.endswith(".v2ep.pt")]

    memo = MemoEncoder(model.core.encoder)
    model.core.encoder = memo

    KS = [0, 1, 2, 3, 4, 8]
    per: dict = {f"ade_k{k}": [] for k in KS}
    per.update({f"ade2s_k{k}": [] for k in KS})
    per.update({f"oracle_k{k}": [] for k in KS})
    per.update({f"spread_k{k}": [] for k in KS})
    per.update({f"selchange_vs0_k{k}": [] for k in KS})
    per.update({f"selbase_change_vs0_k{k}": [] for k in KS})
    per.update({f"disp_k{k}": [] for k in KS if k > 0})
    per.update({"offset0_mean_norm": [], "ade_t0forced_k2": [], "fandiff_t0forced_vs_k2": [],
                "sel_agree_t0forced_vs_k2": [], "conf_delta_t0_vs_t2_same_x": [],
                "offset_delta_t0_vs_t2_same_x": [], "offset_norm_t0_same_x": [],
                "offset_norm_t2_same_x": [],
                "noise_std_pass1": [], "ade_noise_k2": [], "fandiff_noise_vs_k2": [],
                "sel_agree_noise_vs_k2": [], "ade_ha0": [], "v0": []})
    t_trace_steps8 = None
    n_win, n_skip = 0, 0
    idx_all = list(range(len(ds)))[::a.stride]
    if a.max_windows:
        idx_all = idx_all[:a.max_windows]
    t_start = time.time()
    for wi in idx_all:
        item = ds[wi]
        fv = item["future_valid_ext"]
        if not bool(fv[need].all()):
            n_skip += 1
            continue
        pose_last = item["pose_last"].float()
        v0 = float(pose_last[3])
        gt = refb_labels.waypoint_targets(pose_last[None],
                                          item["future_poses_ext"].float()[None],
                                          horizons)[0].float()                # [S,2]
        fr = tr.frames_to_device(item["frames"][None], dev)
        v0_t = torch.tensor([v0], dtype=torch.float32, device=dev)
        # constant-velocity straight line from v0 (ha0): the trivial floor
        ha0 = torch.stack([torch.tensor([v0 * 0.1 * h, 0.0]) for h in horizons])
        per["ade_ha0"].append(ade(ha0, gt))
        per["v0"].append(v0)
        with torch.no_grad():
            fans, trajs, trajs_b, sels, sels_b = {}, {}, {}, {}, {}
            for k in KS:
                with DecodeTap(dec) as tap:
                    out = model(fr, nav_cmd=None, v0=v0_t, steps=k)
                    if k == 8 and t_trace_steps8 is None:
                        t_trace_steps8 = [c[0] for c in tap.calls]
                fans[k] = out["anchor_traj"][0].float().cpu()
                trajs[k] = out["traj"][0].float().cpu()
                trajs_b[k] = out["traj_base"][0].float().cpu()
                sels[k] = int(out["sel_idx"][0])
                sels_b[k] = int(out["sel_idx_base"][0])
                if k == 0:
                    per["offset0_mean_norm"].append(
                        float(out["offset"][0].float().norm(dim=-1).mean()))
            for k in KS:
                per[f"ade_k{k}"].append(ade(trajs[k], gt))
                per[f"ade2s_k{k}"].append(ade(trajs[k][:n2], gt[:n2]))
                per[f"oracle_k{k}"].append(
                    float((fans[k] - gt[None]).norm(dim=-1).mean(-1).min()))
                end = fans[k][:, -1]
                per[f"spread_k{k}"].append(float(torch.cdist(end, end).mean()))
                per[f"selchange_vs0_k{k}"].append(float(sels[k] != sels[0]))
                per[f"selbase_change_vs0_k{k}"].append(float(sels_b[k] != sels_b[0]))
            for k in KS:
                if k == 0:
                    continue
                prev = KS[KS.index(k) - 1]
                dpp = float((fans[k] - fans[prev]).norm(dim=-1).mean()) / (k - prev)
                per[f"disp_k{k}"].append(dpp)          # per-pass displacement (m)
            # Q3: _decode at t=0 vs t=2 on the SAME x (the classifier-pass fan)
            kv = dec.feat_proj(memo._out[0].reshape(fr.shape[0], fr.shape[1],
                                                    *memo._out[0].shape[1:])[:, -1]
                               .flatten(2).transpose(1, 2))
            # (cond is recomputed inside forward; rebuild it the same way)
            m_vec = None
            with DecodeTap(dec) as tap:
                model(fr, nav_cmd=None, v0=v0_t, steps=0)
                x0 = tap.calls[-1][2]
            # capture cond by tapping a layer input
            cond_box = {}
            def _hook(mod, args, kwargs=None):
                cond_box["c"] = args[2]
            h = dec.layers[0].register_forward_pre_hook(_hook)
            try:
                model(fr, nav_cmd=None, v0=v0_t, steps=0)
            finally:
                h.remove()
            cond = cond_box["c"]
            c0, o0 = dec._decode(kv, cond, x0, 0)
            c2, o2 = dec._decode(kv, cond, x0, dec.cfg.diffusion_steps)
            per["conf_delta_t0_vs_t2_same_x"].append(float((c0 - c2).abs().mean()))
            per["offset_delta_t0_vs_t2_same_x"].append(
                float((o0 - o2).norm(dim=-1).mean()))
            per["offset_norm_t0_same_x"].append(float(o0.norm(dim=-1).mean()))
            per["offset_norm_t2_same_x"].append(float(o2.norm(dim=-1).mean()))
            # Q2 ablation: force t_idx = 0 on every pass at steps=2
            with DecodeTap(dec) as tap:
                tap.force_t = 0
                out_t0 = model(fr, nav_cmd=None, v0=v0_t, steps=2)
            fan_t0 = out_t0["anchor_traj"][0].float().cpu()
            per["ade_t0forced_k2"].append(ade(out_t0["traj"][0].float().cpu(), gt))
            per["fandiff_t0forced_vs_k2"].append(float((fan_t0 - fans[2]).norm(dim=-1).mean()))
            per["sel_agree_t0forced_vs_k2"].append(float(int(out_t0["sel_idx"][0]) == sels[2]))
            # Q4: train-time noise, measured. decoder.train() flips ONLY the
            # `if self.training` noise branch (no Dropout modules inside; MHA p=0).
            dec.train()
            try:
                stds, ades, fds, agr = [], [], [], []
                for seed in range(a.noise_seeds):
                    torch.manual_seed(1000 + seed)
                    with DecodeTap(dec) as tap:
                        out_n = model(fr, nav_cmd=None, v0=v0_t, steps=2)
                    x_in1 = tap.calls[1][2].float().cpu()      # pass-1 input (noised)
                    stds.append(float((x_in1 - fans[0]).std()))
                    ades.append(ade(out_n["traj"][0].float().cpu(), gt))
                    fds.append(float((out_n["anchor_traj"][0].float().cpu() - fans[2])
                                     .norm(dim=-1).mean()))
                    agr.append(float(int(out_n["sel_idx"][0]) == sels[2]))
            finally:
                dec.eval()
            per["noise_std_pass1"].append(float(np.mean(stds)))
            per["ade_noise_k2"].append(float(np.mean(ades)))
            per["fandiff_noise_vs_k2"].append(float(np.mean(fds)))
            per["sel_agree_noise_vs_k2"].append(float(np.mean(agr)))
        n_win += 1
        if n_win % 20 == 0:
            el = time.time() - t_start
            print(f"[A] {n_win} windows  {el:.0f}s  ({el / n_win:.2f} s/window)  "
                  f"encodes={memo.n_encodes}", flush=True)
    res["part_a"]["n_windows"] = n_win
    res["part_a"]["n_skipped_horizon"] = n_skip
    res["part_a"]["stride"] = a.stride
    res["part_a"]["encodes"] = memo.n_encodes
    res["part_a"]["t_idx_trace_steps8"] = t_trace_steps8
    res["part_a"]["wall_s"] = round(time.time() - t_start, 1)
    summ = {k: mean_se(v) for k, v in per.items()}
    # paired deltas vs the shipped setting (steps = 2)
    def paired(a_key, b_key):
        x = np.asarray(per[a_key]) - np.asarray(per[b_key])
        return {"mean": float(x.mean()), "se_window_level": float(x.std(ddof=1) / np.sqrt(x.size)),
                "n": int(x.size)}
    summ["paired"] = {
        "ade_k0_minus_k2": paired("ade_k0", "ade_k2"),
        "ade_k1_minus_k2": paired("ade_k1", "ade_k2"),
        "ade_k4_minus_k2": paired("ade_k4", "ade_k2"),
        "ade_k8_minus_k2": paired("ade_k8", "ade_k2"),
        "ade_t0forced_minus_k2": paired("ade_t0forced_k2", "ade_k2"),
        "ade_noise_minus_k2": paired("ade_noise_k2", "ade_k2"),
        "ade_k2_minus_ha0": paired("ade_k2", "ade_ha0"),
        "oracle_k0_minus_k2": paired("oracle_k0", "oracle_k2"),
        "oracle_k8_minus_k2": paired("oracle_k8", "oracle_k2"),
    }
    res["part_a"]["summary"] = summ
    res["part_a"]["per_window"] = per
    print("\n[A] ===== SUMMARY (trained refcv3 @ 40,284; nav_cmd=None; n_windows=%d, n_clips=%d) =====" % (n_win, len(eps)))
    for k in KS:
        print(f"  steps={k}: ADE0-6s {summ[f'ade_k{k}']['mean']:.4f}  ADE0-2s {summ[f'ade2s_k{k}']['mean']:.4f}  "
              f"oracle-in-fan {summ[f'oracle_k{k}']['mean']:.4f}  endpoint-spread {summ[f'spread_k{k}']['mean']:.3f} m  "
              f"sel!=k0 {summ[f'selchange_vs0_k{k}']['mean']:.3f}"
              + (f"  per-pass disp {summ[f'disp_k{k}']['mean']:.4f} m" if k > 0 else ""))
    print(f"  ha0 (constant-velocity straight line) ADE0-6s {summ['ade_ha0']['mean']:.4f}")
    print(f"  classifier-pass |offset| mean {summ['offset0_mean_norm']['mean']:.4f} m")
    print(f"  _decode same x: |conf(t0)-conf(t2)| {summ['conf_delta_t0_vs_t2_same_x']['mean']:.4f}  "
          f"|off(t0)-off(t2)| {summ['offset_delta_t0_vs_t2_same_x']['mean']:.4f} m  "
          f"(|off t0| {summ['offset_norm_t0_same_x']['mean']:.4f}, |off t2| {summ['offset_norm_t2_same_x']['mean']:.4f})")
    print(f"  t_idx FORCED 0 at steps=2: ADE {summ['ade_t0forced_k2']['mean']:.4f}  fan diff vs k2 {summ['fandiff_t0forced_vs_k2']['mean']:.4f} m  sel agree {summ['sel_agree_t0forced_vs_k2']['mean']:.3f}")
    print(f"  train-noise at eval (std measured {summ['noise_std_pass1']['mean']:.4f} vs cfg {dec.cfg.noise_std}): ADE {summ['ade_noise_k2']['mean']:.4f}  fan diff vs k2 {summ['fandiff_noise_vs_k2']['mean']:.4f} m  sel agree {summ['sel_agree_noise_vs_k2']['mean']:.3f}")
    print(f"  t_idx trace for steps=8: {t_trace_steps8}")
    for k, v in summ["paired"].items():
        print(f"  paired {k}: {v['mean']:+.4f} (window-level SE {v['se_window_level']:.4f}, n={v['n']})")


def part_b(res: dict) -> None:
    torch.manual_seed(0)
    cfg = refc.refc_smoke_config()
    model = refc.RefCModel(cfg).eval()
    dec = model.decoder
    b, w = 3, cfg.window
    hh, ww = cfg.encoder.image_hw()
    frames = torch.rand(b, w, cfg.encoder.in_channels, hh, ww)
    v0 = torch.tensor([3.0, 8.0, 12.0])
    out: dict = {}
    with torch.no_grad():
        o0a = model(frames, v0=v0, steps=0)
        o0b = model(frames, v0=v0, steps=0)
        o1 = model(frames, v0=v0, steps=1)
        o2a = model(frames, v0=v0, steps=2)
        o2b = model(frames, v0=v0, steps=2)
        with DecodeTap(dec) as tap:
            model(frames, v0=v0, steps=4)
        out["t_idx_trace_steps4"] = [c[0] for c in tap.calls]
        out["eval_deterministic_steps0"] = bool(torch.equal(o0a["anchor_traj"], o0b["anchor_traj"]))
        out["eval_deterministic_steps2"] = bool(torch.equal(o2a["anchor_traj"], o2b["anchor_traj"]))
        bank = o0a["anchor_bank"] if "anchor_bank" in o0a else \
            dec.anchors[None].expand(b, *dec.anchors.shape)   # pre-v4 tree: fixed bank
        out["steps0_fan_equals_bank_plus_offset"] = bool(torch.allclose(
            o0a["anchor_traj"], bank + o0a["offset"], atol=1e-6))
        out["has_anchor_bank_key(v4_tree)"] = bool("anchor_bank" in o0a)
        out["x_updated_pass1_mean_disp_m"] = float((o1["anchor_traj"] - o0a["anchor_traj"]).norm(dim=-1).mean())
        out["x_updated_pass2_mean_disp_m"] = float((o2a["anchor_traj"] - o1["anchor_traj"]).norm(dim=-1).mean())
        out["anchor_logits_identical_across_steps"] = bool(
            torch.equal(o0a["anchor_logits"], o2a["anchor_logits"]))
        out["refined_logits_differ_from_anchor_logits_at_steps2"] = bool(
            not torch.equal(o2a["refined_logits"], o2a["anchor_logits"]))
        # t=0 vs t=2 on the same x
        with DecodeTap(dec) as tap:
            model(frames, v0=v0, steps=0)
            x0 = tap.calls[-1][2]
        cond_box = {}
        h = dec.layers[0].register_forward_pre_hook(lambda m, args: cond_box.__setitem__("c", args[2]))
        try:
            model(frames, v0=v0, steps=0)
        finally:
            h.remove()
        fmap, _ = model.encoder(frames.reshape(b * w, *frames.shape[2:]))
        fmap = fmap.reshape(b, w, *fmap.shape[1:])[:, -1]
        kv = dec.feat_proj(fmap.flatten(2).transpose(1, 2))
        c0, of0 = dec._decode(kv, cond_box["c"], x0, 0)
        c2, of2 = dec._decode(kv, cond_box["c"], x0, cfg.decoder.diffusion_steps)
        out["decode_t0_vs_t2_conf_absdiff"] = float((c0 - c2).abs().mean())
        out["decode_t0_vs_t2_offset_diff_m"] = float((of0 - of2).norm(dim=-1).mean())
        te = dec.time_embed.weight
        out["time_embed_rows"] = int(te.shape[0])
        out["time_embed_pairwise_l2"] = [[round(float(x), 3) for x in r] for r in torch.cdist(te, te)]
        # train-mode noise: decoder only
        dec.train()
        try:
            torch.manual_seed(1)
            with DecodeTap(dec) as tap:
                n1 = model(frames, v0=v0, steps=2)
            x_in1 = tap.calls[1][2]
            out["train_noise_std_pass1"] = float((x_in1 - o0a["anchor_traj"]).std())
            out["train_noise_mean_pass1"] = float((x_in1 - o0a["anchor_traj"]).mean())
            torch.manual_seed(2)
            n2 = model(frames, v0=v0, steps=2)
            out["train_mode_nondeterministic"] = bool(not torch.equal(n1["anchor_traj"], n2["anchor_traj"]))
        finally:
            dec.eval()
        out["cfg_noise_std"] = cfg.decoder.noise_std
        out["cfg_diffusion_steps"] = cfg.decoder.diffusion_steps
    res["part_b"] = out
    print("\n[B] ===== smoke-config structural checks (random init, CPU) =====")
    for k, v in out.items():
        print(f"  {k}: {v}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--noise-seeds", type=int, default=3)
    ap.add_argument("--skip-a", action="store_true")
    a = ap.parse_args(argv)
    res = {"tanitad_file": tanitad.__file__,
           "refc_py": refc.__file__, "refc_py_md5": md5(refc.__file__),
           "torch": torch.__version__,
           "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "omp_threads": os.environ.get("OMP_NUM_THREADS")}
    print("[env]", json.dumps({k: v for k, v in res.items()}, indent=None))
    part_b(res)
    if not a.skip_a:
        part_a(a, res)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[done] wrote", a.out)


if __name__ == "__main__":
    main()
