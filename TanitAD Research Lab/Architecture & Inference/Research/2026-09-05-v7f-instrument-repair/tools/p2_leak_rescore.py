"""p2_leak_rescore.py — the 0-GPU re-scores pre-registered in SPEC.md §5 (R1, R3, R4, R5).

Subcommands (each writes raw/<name>.json and prints the same numbers):

  data    R5  model-free class-(i) overlaps on the corpus (no model, no torch forward)
  actdiv  R1  MM-E10 action-divergence on the local banked ckpts, with the speed
              channel (a) held as banked at v/30, (b) held at the TRAINED v/10,
              (c) rolled with the action, (d) rolled alone; GS-8 anchored read;
              actchan's nrmse_1 degeneracy conjunction; a clip bootstrap on the ratio
  l3      R4  envpred's L3 rig with the rollout's action feed as banked (TRUE future
              actions), as HOLD-ACTION (the T1 convention) and SHUFFLED; the banked-form
              estimator (v7tiny_probe.probe wrapped by envpred.loeo, reproduced verbatim)
              beside a corrected clip-level one
  drift   R3  E-DEC-59's drift rig (RFF+ridge, band 0:8, K=4) with an ENDPOINT-shuffled
              target control that isolates the class-(ii) arithmetic component

0-GPU BY CONSTRUCTION: CUDA_VISIBLE_DEVICES is set to -1 before torch is imported
(an EMPTY value is DELETED by Windows and would leave the GPU visible — measured on this
box), and the run REFUSES if torch.cuda.is_available() is True. The dev box's RTX 4060 is
busy with a detached T1 read and is never touched.

Provenance idioms carried from the banked probes: MM-C12 stack preflight + tanitad.__file__
stamp (_stackresolve.py); checkpoint md5 stamped per arm; the actdiv compute section is
actdiv_thor.py's verbatim (windows, roll variants, C0 identity, spreads); envpred's clip
selection, scene targets, masks and rollout loop are reproduced line-for-line.
Evidence class of every number: MEASURED (ours; dev-box CPU). Tier: T0-DIAGNOSTIC.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # ⛔ before torch is imported; "" is deleted on Windows

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _stackresolve import preflight_stack, resolve_stack  # noqa: E402

import numpy as np  # noqa: E402

ASSETS = pathlib.Path(r"C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901")
OLD_SP = pathlib.Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
                      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
                      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
ACTDIV_CORPUS = ASSETS / "sp2/cache/physicalai-val-w120-256x640cyl"   # Thor's first 24, md5-manifested
HELDOUT = ASSETS / "sp2/cache/physicalai-val130-heldout"              # 130 held-out clips
LABELS = OLD_SP / "sp2/val130_agents.jsonl"                           # E-DEC-31 join (md5 66efaa94…)
CKPTS = {
    "postrain30k": ASSETS / "v7tiny_postrain30k/ckpt.pt",
    "k8clip05p30k": ASSETS / "v7tiny_k8clip05p30k/ckpt.pt",
    "k60clip05p30k": ASSETS / "v7tiny_k60clip05p30k/ckpt.pt",
    "rdw8p30k": ASSETS / "v7tiny_rdw8p30k/ckpt.pt",
    "postrain30k_freeze": OLD_SP / "v7tiny_postrain30k_freeze/ckpt.pt",
    # ---- 2026-09-05 v7f-instrument-repair: PATH REGISTRATIONS ONLY ----------
    # cmd_l3/cmd_drift call load_arm(arm) with no ckpt_override (only cmd_actdiv
    # honours --ckpt, :288), so an arm must be registered here to be scoreable.
    # NOTHING about the estimator, corpus, or protocol is changed by these lines.
    # Every file below was scp'd from thor:/home/nvidia/v7tiny/<arm>/ckpt.pt and
    # md5-verified against the remote in the same command:
    #   splitp30k          4348cad27dbf1895654c40681d92ea97
    #   postrain30k_freeze 5f5e5c92cd8fb3dc92f7b38580149689
    #   emao14_30k         3a030ba2cff6141fd548307081a592f5
    #   emao14_30k_tauramp a64aa48a23b755bb8aff5dceb7ac13e1
    #   o14fut30k          3e4a7443f1ddf31f7b770471b7a9066d
    "splitp30k": ASSETS / "v7tiny_splitp30k/ckpt.pt",
    "postrain30k_freeze_local": ASSETS / "v7tiny_postrain30k_freeze/ckpt.pt",
    "emao14_30k": ASSETS / "v7tiny_emao14_30k/ckpt.pt",
    "emao14_30k_tauramp": ASSETS / "v7tiny_emao14_30k_tauramp/ckpt.pt",
    "o14fut30k": ASSETS / "v7tiny_o14fut30k/ckpt.pt",
}
N_STACK = 3
DT = 0.1
PROBE_SPEED_SCALE = 30.0        # what actdiv_thor.py:47 / condpath_thor.py:55 feed
WHEELBASE = 2.9                 # physicalai.py legacy constant (kinident.py:48)
LAMBDAS_LIN = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)   # v7tiny_probe.py:58


# ----------------------------------------------------------------------------- common
def md5_of(path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stamp(extra=None) -> dict:
    import torch
    d = {"_evidence_class": "MEASURED (ours; dev-box CPU, GPU hidden)",
         "eval_tier": "T0-DIAGNOSTIC",
         "_spec": "TanitAD Research Lab/Architecture & Inference/Research/"
                  "2026-09-03-p2-probe-leak-audit/SPEC.md",
         "_generated_local": time.strftime("%Y-%m-%d %H:%M:%S"),
         "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if extra:
        d.update(extra)
    return d


def guard_and_stack(stack_req: str | None) -> str:
    stack = resolve_stack(stack_req or None)
    tf = preflight_stack(stack)
    import torch
    if torch.cuda.is_available():
        print("[REFUSED] torch sees a CUDA device — this tool is 0-GPU by construction",
              file=sys.stderr)
        raise SystemExit(2)
    # ⚠️ grad is disabled AFTER each model build, not here: load_trunk_auto ->
    # build_stack_from_args -> assert_isolation runs a real backward (v6.py:5802),
    # so a global no-grad at this point crashes the load (measured on the first run).
    torch.set_num_threads(8)                      # leave CPU headroom for the T1 read's host side
    from tanitad.models.flagship_v15 import SPEED_SCALE
    if float(SPEED_SCALE) != 10.0:
        print(f"[REFUSED] flagship_v15.SPEED_SCALE is {SPEED_SCALE}, the trained "
              f"contract this tool assumes is 10.0", file=sys.stderr)
        raise SystemExit(2)
    return tf


def frames_of(path):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    raw = d["jpeg_buf"].numpy().tobytes()
    lens = d["jpeg_len"].numpy().tolist()
    off = [0]
    for L in lens:
        off.append(off[-1] + int(L))
    return d, raw, off, len(lens)


def decode_frames(raw, off, n):
    import torch
    from PIL import Image
    imgs = []
    for i in range(n):
        im = Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("RGB")
        imgs.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0)
    if not imgs or float(imgs[0].abs().mean()) == 0.0:
        raise SystemExit("[FATAL] a clip decoded to all-zero frames (the jpeg_buf/png trap)")
    return imgs


def encode_imgs(world, imgs):
    """actdiv_thor.py's encode loop, verbatim (n_stack=3 stacked into channels, batch 16)."""
    import torch
    Z, B = [], 16
    n = len(imgs)
    for s in range(0, n, B):
        chunk = []
        for i in range(s, min(s + B, n)):
            idx = [max(i - j, 0) for j in range(N_STACK - 1, -1, -1)]
            chunk.append(torch.cat([imgs[k] for k in idx], 0))
        x = torch.stack(chunk)[:, None]
        Z.append(world.encode_window(x)[:, 0].float().cpu())
    return torch.cat(Z)


def encode_clip(world, path, max_frames):
    d, raw, off, n = frames_of(path)
    n = min(n, max_frames)
    imgs = decode_frames(raw, off, n)
    z = encode_imgs(world, imgs)
    return d, imgs, z, d["actions"].float()[:n], d["poses"].float()[:n, 3]


def load_arm(name: str, ckpt_override: str | None = None):
    import torch
    from tanitad.eval.v6_probe_trunk import load_trunk_auto
    p = pathlib.Path(ckpt_override) if ckpt_override else CKPTS[name]
    if not p.is_file():
        print(f"[REFUSED] {name}: no checkpoint at {p}", file=sys.stderr)
        raise SystemExit(2)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    with torch.enable_grad():                       # the build's isolation check needs autograd
        world, _g, step = load_trunk_auto(ck, "cpu", ckpt_path=str(p))
    for q in world.parameters():
        q.requires_grad_(False)
    world.eval()
    torch.set_grad_enabled(False)                   # every forward from here on is no-grad
    return world, int(step), {"ckpt": str(p), "md5": md5_of(p), "step": int(step)}


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def r_(x, y):
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))


def tstat(d):
    d = np.asarray(d, float)
    return float(d.mean()) / max(float(d.std(ddof=1) / np.sqrt(len(d))), 1e-12)


def boot_mean_ci(d, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# ----------------------------------------------------------------------------- R5 data
def cmd_data(a):
    import torch
    out = stamp({"subcommand": "data (R5)", "corpus_heldout": str(HELDOUT),
                 "corpus_actdiv": str(ACTDIV_CORPUS)})
    clips = sorted(HELDOUT.glob("*.v2ep.pt"))[:a.nclips]
    F, K = 100, 4
    S, A, V, OM, DV1, DV4, DY4, KIN, OM1_IN_TARGET = [], [], [], [], [], [], [], [], []
    for c in clips:
        d = torch.load(c, map_location="cpu", weights_only=False)
        act = np.asarray(d["actions"], dtype=np.float64)
        pos = np.asarray(d["poses"], dtype=np.float64)
        yaw, v = pos[:, 2], pos[:, 3]
        n = min(len(act), len(yaw), F) - K - 1
        if n < 25:
            continue
        i = np.arange(n)
        S.append(act[i, 0]); A.append(act[i, 1]); V.append(v[i])
        om = wrap(yaw[i + 1] - yaw[i]) / DT                     # latentmotion.py:158's omega
        OM.append(om)
        KIN.append(v[i] * np.tan(act[i, 0]) / WHEELBASE)        # E-DEC-57 closed form
        DV1.append(v[i + 1] - v[i]); DV4.append(v[i + K] - v[i])
        DY4.append(wrap(yaw[i + K] - yaw[i]))
    S, A, V, OM, KIN, DV1, DV4, DY4 = map(np.concatenate, (S, A, V, OM, KIN, DV1, DV4, DY4))
    n = int(len(S))
    # share of the K-tick yaw change carried by the first tick (the tick latentmotion's
    # omega input takes from INSIDE the target window): R^2 of a 1-parameter fit
    r_om_dy4 = r_(OM * DT, DY4)
    out["heldout"] = {
        "n_frames": n, "n_clips": len(clips), "F": F, "K": K,
        "r_steer_vs_measured_yawrate": round(r_(S, OM), 4),
        "r_closedform_v_tan_steer_over_L_vs_measured_yawrate": round(r_(KIN, OM), 4),
        "r_accel_vs_dv_1tick": round(r_(A, DV1), 4),
        "r_accel_vs_dv_4tick": round(r_(A, DV4), 4),
        "r_speed_vs_dv_4tick": round(r_(V, DV4), 4),
        "r_omega_first_tick_vs_dyaw_4tick": round(r_om_dy4, 4),
        "share_of_dyaw4_variance_explained_by_first_tick_omega_R2": round(r_om_dy4 ** 2, 4),
        "note": ("latentmotion.py:158 builds omega from yaw[i+1]-yaw[i], the first tick of "
                 "the K=4 transition it scores (class (i), favours the ALTERNATIVE); "
                 "accel is the dataset's own ax (physicalai.py:604-632), not a finite "
                 "difference of v (echo test)"),
    }
    # the actdiv corpus: what the roll variants perturb
    clips2 = sorted(ACTDIV_CORPUS.glob("*.v2ep.pt"))[:24]
    S2, A2, V2 = [], [], []
    for c in clips2:
        d = torch.load(c, map_location="cpu", weights_only=False)
        act = np.asarray(d["actions"], dtype=np.float64)[:60]
        S2.append(act[:, 0]); A2.append(act[:, 1])
        V2.append(np.asarray(d["poses"], dtype=np.float64)[:60, 3])
    S2, A2, V2 = map(np.concatenate, (S2, A2, V2))

    def stats(x):
        return {"min": round(float(x.min()), 4), "mean": round(float(x.mean()), 4),
                "max": round(float(x.max()), 4), "sd": round(float(x.std()), 4)}
    out["actdiv_corpus_first60"] = {
        "n_frames": int(len(S2)), "n_clips": len(clips2),
        "steer_rad": stats(S2), "accel_mps2": stats(A2), "speed_mps": stats(V2),
        "speed_over_10_trained_scale": stats(V2 / 10.0),
        "speed_over_30_probe_scale": stats(V2 / PROBE_SPEED_SCALE),
        "frac_abs_steer_below_0p02": round(float((np.abs(S2) < 0.02).mean()), 4),
    }
    return out


# ----------------------------------------------------------------------------- R1 actdiv
def cmd_actdiv(a):
    import torch
    out = stamp({"subcommand": "actdiv (R1)", "corpus": str(ACTDIV_CORPUS),
                 "banked_reference": {
                     "postrain30k_h1_ratio_thor": 0.005947,
                     "postrain30k_h1_ratio_devbox_cuda_reread": 0.005950,
                     "instrument": "actdiv_thor.py / actdiv_local.py, v/30 held, roll [steer,accel]"},
                 "variants": {
                     "banked_roll_sa_hold_v": "roll [steer,accel] across windows; each window keeps its OWN v (the banked instrument)",
                     "roll_full_tuple": "roll [steer,accel,v] together (a kinematically consistent other window)",
                     "roll_v_only": "roll v across windows; [steer,accel] kept",
                 },
                 "speed_scales": {"10": "TRAINED contract (flagship_v15.SPEED_SCALE, _lift3)",
                                  "30": "what actdiv_thor.py:47 / condpath_thor.py:55 fed"},
                 "arms": {}})
    clips = sorted(ACTDIV_CORPUS.glob("*.v2ep.pt"))[:24]
    if len(clips) < 24:
        print(f"[REFUSED] {len(clips)} clips, the banked instrument used 24", file=sys.stderr)
        raise SystemExit(2)
    N_VAR = 8
    scales = [10.0, PROBE_SPEED_SCALE]
    arms = a.arms.split(",")
    for arm in arms:
        world, step, prov = load_arm(arm, a.ckpt if (a.ckpt and len(arms) == 1) else None)
        W = int(world.window)
        horizons = sorted(int(h) for h in world.stack.cfg.predictor.horizons)
        Z, A, V, CID, ZN = [], [], [], [], []
        t0 = time.time()
        for ci, c in enumerate(clips):
            _d, _imgs, z, act, spd = encode_clip(world, str(c), 60)
            n = min(len(z) - W, len(act) - W, len(spd) - W)
            if n < 4:
                continue
            for i in range(0, n, max(1, n // 5)):               # actdiv_thor.py's window rule
                Z.append(z[i:i + W]); A.append(act[i:i + W]); V.append(spd[i]); CID.append(ci)
                ZN.append(z[i + W] if i + W < len(z) else None)
        zs = torch.stack(Z); aa = torch.stack(A); vv = torch.stack(V).reshape(-1)
        cid = np.asarray(CID)
        print(f"[{arm}] step {step} md5 {prov['md5'][:12]} {zs.shape[0]} windows W={W} "
              f"horizons={horizons} encode {time.time()-t0:.0f}s", flush=True)

        def fwd(a2, vvec, scale):
            a3 = torch.cat([a2, (vvec / scale)[:, None, None].expand(-1, W, -1)], -1)
            return world.predictor(zs, a3)

        arm_out = {**prov, "n_windows": int(zs.shape[0]), "n_clips": int(len(set(CID))),
                   "W": W, "horizons": horizons, "by_scale": {}}
        rng = np.random.default_rng(0)
        for scale in scales:
            base = fwd(aa, vv, scale)
            sc_out = {}
            for h in horizons:
                b_h = base[h].float()
                ident = float((fwd(aa, vv, scale)[h].float() - b_h).abs().max())
                scene_sp = float(b_h.std(dim=0).mean())
                hv = {"scene_spread": round(scene_sp, 6), "C0_identity_max_abs_diff": ident,
                      "C0_passes": ident == 0.0, "variants": {}}
                for vname in ("banked_roll_sa_hold_v", "roll_full_tuple", "roll_v_only"):
                    preds = [b_h]
                    for j in range(1, N_VAR):
                        if vname == "banked_roll_sa_hold_v":
                            o = fwd(torch.roll(aa, shifts=j, dims=0), vv, scale)
                        elif vname == "roll_full_tuple":
                            o = fwd(torch.roll(aa, shifts=j, dims=0),
                                    torch.roll(vv, shifts=j, dims=0), scale)
                        else:
                            o = fwd(aa, torch.roll(vv, shifts=j, dims=0), scale)
                        preds.append(o[h].float())
                    P = torch.stack(preds, 0)                         # [N_VAR, n_win, d]
                    per_win = P.std(dim=0).mean(dim=1).numpy()        # [n_win]
                    act_sp = float(per_win.mean())
                    ratio = act_sp / scene_sp if scene_sp > 1e-9 else None
                    # clip bootstrap of the ratio (backlog L-13 gap, closed for THIS read only)
                    bh = b_h.numpy()
                    uniq = np.unique(cid)
                    rs = []
                    for _ in range(1000):
                        pick = rng.choice(uniq, size=len(uniq), replace=True)
                        sel = np.concatenate([np.where(cid == u)[0] for u in pick])
                        s_sc = float(bh[sel].std(axis=0).mean())
                        rs.append(float(per_win[sel].mean()) / max(s_sc, 1e-12))
                    if h == 1:
                        hv.setdefault("per_window_action_spread_h1", {})[vname] = [round(float(x), 8) for x in per_win]
                        hv["per_window_clip_id"] = [int(x) for x in cid]
                    hv["variants"][vname] = {
                        "action_spread": round(act_sp, 6),
                        "ratio_action_over_scene": round(ratio, 6) if ratio else None,
                        "ratio_clip_bootstrap_ci95": [round(float(np.percentile(rs, 2.5)), 6),
                                                      round(float(np.percentile(rs, 97.5)), 6)],
                        "evidence_note": ("h>=2 heads are UNTRAINED in this recipe (MM-E14); "
                                          "withdrawn as evidence" if h >= 2 else "h1: trained head")}
                    print(f"   v/{scale:g} h={h} {vname:<24} action {act_sp:.6f} scene {scene_sp:.5f} "
                          f"RATIO {ratio:.6f} CI[{hv['variants'][vname]['ratio_clip_bootstrap_ci95'][0]:.6f},"
                          f"{hv['variants'][vname]['ratio_clip_bootstrap_ci95'][1]:.6f}] "
                          f"C0 {'PASS' if ident == 0.0 else 'FAIL'}", flush=True)
                sc_out[f"h{h}"] = hv
            # actchan.py:134-144's degeneracy conjunction: nrmse_1 vs the no-change floor
            keep = [k for k, zn in enumerate(ZN) if zn is not None]
            if keep:
                zn = torch.stack([ZN[k] for k in keep]).float()
                zl = zs[keep, -1].float()
                p1 = base[1][keep].float()
                num = (p1 - zn).norm(dim=1); den = (zn - zl).norm(dim=1).clamp_min(1e-9)
                sc_out["nrmse_1_vs_no_change"] = round(float((num / den).mean()), 4)
                sc_out["nrmse_1_note"] = ">1.10 = DEGENERATE per actchan.py:199 (worse than the no-change predictor)"
            # GS-8 anchored read: displacement from the ZERO-action prediction, monotone in |s|?
            anch = {}
            z0 = fwd(torch.zeros_like(aa), vv, scale)[1].float()
            b1 = base[1].float()
            # comparable to the roll ratio: mean per-dim |displacement| over windows and dims,
            # divided by the same mean per-dim scene std the ratio uses (NOT an L2 norm over
            # 2048 dims divided by a per-dim std — the first run's normalisation, retired)
            scene1 = float(b1.std(dim=0).mean())
            for s in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
                o = fwd(aa * s, vv, scale)[1].float()
                disp = float((o - z0).abs().mean())
                anch[str(s)] = round(disp / max(scene1, 1e-12), 6)
            sc_out["anchored_meanabs_displacement_over_scene_spread_h1"] = anch
            sc_out["anchored_note"] = ("GS-8: displacement of zhat(s*a) from the ZERO-action prediction zhat(0), "
                                       "[steer,accel] scaled by s, v held; read MONOTONICITY in |s|")
            arm_out["by_scale"][str(int(scale))] = sc_out
        out["arms"][arm] = arm_out
        del world, zs, aa
    return out


# ----------------------------------------------------------------------------- R4 l3
def load_labels():
    LAB = {}
    with open(LABELS, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                LAB.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r.get("agents", [])
    return LAB


def scene(ag, m):
    """envpred.py:74-93 verbatim (NaN on an unlabelled frame)."""
    n, rng_ = [], []
    for i in range(m):
        if i not in ag:
            n.append(np.nan); rng_.append(np.nan); continue
        a = ag.get(i, [])
        inl = [x["cx"] for x in a if abs(x.get("cy", 9e9)) < 1.8 and x.get("cx", -1) > 0]
        n.append(float(len(a)))
        rng_.append(min(inl) if inl else np.nan)
    r = np.array(rng_)
    clo = np.full(m, np.nan)
    clo[1:] = r[1:] - r[:-1]
    return np.array(n), r, clo


def pca_fit(X, k):
    mu = X.mean(0, keepdims=True)
    _u, _s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:min(k, Vt.shape[0])].T


def probe_banked(feats, targets, k_pca, seed=0):
    """v7tiny_probe.probe (v7tiny_probe.py:180-225) VERBATIM — the banked L3 estimator:
    fit = FIRST half of the clip list, te = SECOND half; PCA + lambda on the fit clips
    (inner/val split of the fit clips); returns the POOLED cross-clip R2 over te."""
    n = len(feats)
    half = n // 2
    fit, te = list(range(half)), list(range(half, n))
    nv = max(1, len(fit) // 4)
    inner, val = fit[:-nv], fit[-nv:]
    Xin = np.concatenate([feats[i] for i in inner])
    if k_pca and Xin.shape[1] > k_pca:
        mu_x, P = pca_fit(Xin, k_pca)
        red = lambda A: (A - mu_x) @ P  # noqa: E731
    else:
        red = lambda A: A  # noqa: E731
    Xin = red(Xin)
    Yin = np.concatenate([targets[i] for i in inner])
    xm, ym = Xin.mean(0, keepdims=True), Yin.mean(0, keepdims=True)
    Xc, Yc = Xin - xm, Yin - ym
    G, C = Xc.T @ Xc, Xc.T @ Yc
    Ws, vs = {}, {}
    for lam in LAMBDAS_LIN:
        Wm = np.linalg.solve(G + lam * np.eye(G.shape[0]), C)
        Ws[lam] = Wm
        vs[lam] = sum(float((((targets[i]) - ((red(feats[i]) - xm) @ Wm + ym)) ** 2).sum())
                      for i in val)
    lam = min(vs, key=vs.get)
    Wm = Ws[lam]
    errs, tots = [], []
    for i in te:
        p = (red(feats[i]) - xm) @ Wm + ym
        errs.append(float(((targets[i] - p) ** 2).sum()))
        tots.append(float(((targets[i] - ym) ** 2).sum()))
    errs, tots = np.array(errs), np.array(tots)
    return float(1.0 - errs.sum() / tots.sum())


def loeo_banked(X, Y):
    """envpred.py:211-217 verbatim: appends the 'held-out' clip LAST, so probe_banked's
    te half = 11 other clips + this one. The 24 returned scores overlap ~92 %."""
    o = []
    for i in range(len(X)):
        Xf = [X[j] for j in range(len(X)) if j != i]
        Yf = [Y[j] for j in range(len(Y)) if j != i]
        o.append(probe_banked(Xf + [X[i]], Yf + [Y[i]], 128))
    return np.array(o, dtype=np.float64)


def within_clip_r(pred, yte):
    pc = pred - pred.mean(); yc = yte - yte.mean()
    den = float(np.sqrt((pc ** 2).sum() * (yc ** 2).sum()))
    return float((pc * yc).sum() / den) if den > 1e-12 else 0.0


def rand_pca(X, k, seed):
    """randomized range-finder PCA (rangeprobe_rff.py:108-120's idiom), fit rows only."""
    rs = np.random.default_rng(seed)
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    k = min(k, Xc.shape[1], max(Xc.shape[0] - 1, 1))
    Om = rs.standard_normal((Xc.shape[1], k + 10))
    Yq = Xc @ Om
    for _ in range(2):
        Yq = Xc @ (Xc.T @ Yq)
    Q, _ = np.linalg.qr(Yq)
    _, _, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    return mu, Vt[:k].T


def loco_corrected(X, Y, k_pca=128, seed=0, pooled=False):
    """TRUE leave-one-CLIP-out: each clip scored exactly once on a fit that excludes it.
    PCA (randomized, fit rows only), standardisation and lambda (inner CLIP-disjoint quarter
    of the fit clips) all on the fit split; score = within-clip Pearson r (the programme's
    shared metric, rangeprobe_rff.within_clip_r). Returns one score per clip."""
    n = len(X)
    scores = np.full(n, np.nan)
    errs, tots = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        fit = [j for j in range(n) if j != i]
        nv = max(1, len(fit) // 4)
        inner, val = fit[:-nv], fit[-nv:]
        Xin = np.concatenate([X[j] for j in inner]); Yin = np.concatenate([Y[j] for j in inner])
        if Xin.shape[1] > k_pca:
            mu_x, P = rand_pca(Xin, k_pca, seed)
            red = lambda A, mu_x=mu_x, P=P: (A - mu_x) @ P  # noqa: E731
        else:
            red = lambda A: A  # noqa: E731
        Xr = red(Xin)
        xm, ym = Xr.mean(0, keepdims=True), Yin.mean(0, keepdims=True)
        Xc, Yc = Xr - xm, Yin - ym
        G, C = Xc.T @ Xc, Xc.T @ Yc
        best, bw = None, None
        for lam in LAMBDAS_LIN:
            Wm = np.linalg.solve(G + lam * np.eye(G.shape[0]), C)
            e = sum(float(((Y[j] - ((red(X[j]) - xm) @ Wm + ym)) ** 2).sum()) for j in val)
            if best is None or e < best:
                best, bw = e, Wm
        p = (red(X[i]) - xm) @ bw + ym
        scores[i] = within_clip_r(p.ravel(), Y[i].ravel())
        errs[i] = float(((Y[i] - p) ** 2).sum())
        tots[i] = float(((Y[i] - ym) ** 2).sum())      # the banked probe's cross-clip form (vs the FIT mean)
    return (scores, errs, tots) if pooled else scores


def cmd_l3(a):
    import torch
    out = stamp({"subcommand": "l3 (R4)", "corpus": str(HELDOUT), "labels": str(LABELS),
                 "labels_md5": md5_of(LABELS), "min_lead_frames": 20, "K": [1, 3, 6],
                 "rollouts": {
                     "GT": "envpred.py:184-190 — the TRUE FUTURE actions act[i+s:i+s+W] at every rollout step (the banked form)",
                     "HOLD": "the last OBSERVED action repeated (rollout_transitions with future_actions=None; the T1 hold-action convention)",
                     "SHUF": "actionshuf.py:156-164 — the action window drawn from a random other time in the same clip"},
                 "estimators": {
                     "banked": "v7tiny_probe.probe wrapped by envpred.loeo (reproduced verbatim): pooled cross-clip R2 over a 12-clip second half; 24 overlapping scores; t as envpred.py:238-241",
                     "corrected": "true leave-one-clip-out; within-clip Pearson r per clip; paired clip-level t and clip bootstrap"},
                 "arms": {}})
    LAB = load_labels()
    F, N_CLIPS, MIN_LEAD = 100, 24, 20
    KS = (1, 3, 6)

    def lead_frames(cid):
        ag = LAB.get(cid, {})
        return sum(1 for i in range(F)
                   if any(abs(x.get("cy", 9e9)) < 1.8 and x.get("cx", -1) > 0 for x in ag.get(i, [])))

    _all = [c for c in sorted(HELDOUT.glob("*.v2ep.pt"))
            if torch.load(c, map_location="cpu", weights_only=False)["clip_id"] in LAB]
    _kept = [c for c in _all if lead_frames(torch.load(c, map_location="cpu",
                                                       weights_only=False)["clip_id"]) >= MIN_LEAD]
    clips = _kept[:N_CLIPS]
    out["clip_selection"] = {"labelled": len(_all), "lead_matched": len(_kept), "used": len(clips)}
    print(f"[l3] {len(_all)} labelled, {len(_kept)} lead-matched, {len(clips)} used", flush=True)

    for arm in a.arms.split(","):
        world, step, prov = load_arm(arm)
        W = int(world.window)
        arm_out = {**prov, "W": W, "k": {}}
        t0 = time.time()
        per_clip = []
        for c in clips:
            d, imgs, z, act, spd = encode_clip(world, str(c), F)
            per_clip.append((d["clip_id"], imgs, z.float(), act, spd, scene(LAB[d["clip_id"]], min(len(imgs), F))))
        print(f"[{arm}] step {step} encoded {len(per_clip)} clips in {time.time()-t0:.0f}s", flush=True)
        for K in KS:
            COL = {k: [] for k in ("z_t", "zhat_GT", "zhat_HOLD", "zhat_SHUF", "z_t+k_ceiling", "pixels_t")}
            TG = {"n_agents": [], "lead_range_m": [], "lead_closing": []}
            MK = {"n_agents": [], "lead_range_m": [], "lead_closing": []}
            rng_sh = np.random.default_rng(0)
            for (cid, imgs, zt, act, spd, (nag, rr, clo)) in per_clip:
                n = len(imgs)
                starts = [i for i in range(0, len(zt) - W - K, 1) if (i + W - 1 + K) < n]
                if len(starts) < 20:
                    continue
                I = torch.tensor(starts)
                win0 = torch.stack([zt[i:i + W] for i in starts])                     # [n_win, W, d]
                vv = (spd[I] / 10.0).view(-1, 1, 1).expand(-1, W, 1)               # trained scale, as envpred
                # GT rollout (banked): action window slides with the TRUE future
                win = win0.clone()
                for s in range(K):
                    aa = torch.stack([act[i + s:i + s + W] for i in starts])
                    o = world.predictor(win, torch.cat([aa, vv], -1))[1].float()
                    win = torch.cat([win[:, 1:], o[:, None]], 1)
                z_gt = o
                # HOLD rollout: shift the window, append the last OBSERVED action
                win = win0.clone(); aa = torch.stack([act[i:i + W] for i in starts])
                for s in range(K):
                    o = world.predictor(win, torch.cat([aa, vv], -1))[1].float()
                    win = torch.cat([win[:, 1:], o[:, None]], 1)
                    aa = torch.cat([aa[:, 1:], aa[:, -1:]], 1)
                z_hold = o
                # SHUF rollout: a random other time's action window at every step
                win = win0.clone()
                for s in range(K):
                    r0 = rng_sh.integers(0, max(1, len(act) - W), size=len(starts))
                    aa = torch.stack([act[int(r):int(r) + W] for r in r0])
                    o = world.predictor(win, torch.cat([aa, vv], -1))[1].float()
                    win = torch.cat([win[:, 1:], o[:, None]], 1)
                z_shuf = o
                js = [i + W - 1 for i in starts]; tg = [j + K for j in js]
                COL["z_t"].append(zt[js].numpy().astype(np.float64))
                COL["zhat_GT"].append(z_gt.numpy().astype(np.float64))
                COL["zhat_HOLD"].append(z_hold.numpy().astype(np.float64))
                COL["zhat_SHUF"].append(z_shuf.numpy().astype(np.float64))
                COL["z_t+k_ceiling"].append(zt[tg].numpy().astype(np.float64))
                COL["pixels_t"].append(np.stack([torch.nn.functional.adaptive_avg_pool2d(
                    imgs[j][None, -3:], (8, 8)).reshape(-1).numpy() for j in js]).astype(np.float64))
                ya = np.array([nag[t] for t in tg]); yr = np.array([rr[t] for t in tg]); yc = np.array([clo[t] for t in tg])
                TG["n_agents"].append(np.nan_to_num(ya)[:, None]); MK["n_agents"].append(~np.isnan(ya))
                TG["lead_range_m"].append(np.nan_to_num(yr)[:, None]); MK["lead_range_m"].append(~np.isnan(yr))
                TG["lead_closing"].append(np.nan_to_num(yc)[:, None]); MK["lead_closing"].append(~np.isnan(yc))
            COL["constant"] = [np.ones((len(y), 1)) for y in TG["n_agents"]]
            kres = {}
            for tn, Y in TG.items():
                mk = MK[tn]
                keep = [i for i in range(len(Y)) if int(mk[i].sum()) >= 20]
                if len(keep) < 6:
                    kres[tn] = {"skipped": f"{len(keep)} clips with >=20 rows"}; continue
                Yl = [Y[i][mk[i]] for i in keep]
                Cl = {k: [v[i][mk[i]] for i in keep] for k, v in COL.items()}
                nrow = sum(len(y) for y in Yl)
                tres = {"n_clips": len(keep), "n_rows": int(nrow), "banked_form": {}, "corrected": {}}
                # corrected estimator on every column (primary)
                Rfull = {k: loco_corrected(v, Yl, pooled=True) for k, v in Cl.items()}
                Rc = {k: v[0] for k, v in Rfull.items()}
                base = Rc["z_t"]
                # pooled cross-clip R2 over INDEPENDENT LOCO predictions (the banked statistic's
                # form, with the overlap removed) + clip bootstrap of the delta vs z_t
                tres["pooled_crossclip_R2_true_loco"] = {}
                e0, t0_ = Rfull["z_t"][1], Rfull["z_t"][2]
                r2_zt = 1.0 - e0.sum() / t0_.sum()
                rngp = np.random.default_rng(1)
                for k, (_sc, e1, t1) in Rfull.items():
                    r2 = 1.0 - e1.sum() / t1.sum()
                    bs = []
                    for _ in range(2000):
                        jj = rngp.integers(0, len(e1), len(e1))
                        bs.append((1.0 - e1[jj].sum() / t1[jj].sum()) - (1.0 - e0[jj].sum() / t0_[jj].sum()))
                    bs = np.array(bs)
                    tres["pooled_crossclip_R2_true_loco"][k] = {
                        "r2": round(float(r2), 4), "delta_vs_z_t": round(float(r2 - r2_zt), 4),
                        "ci95_clip_boot": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                        "separated_from_z_t": bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0)}
                for k, sc in Rc.items():
                    dd = sc - base
                    tres["corrected"][k] = {"within_clip_r": round(float(sc.mean()), 4),
                                            "delta_vs_z_t": round(float(dd.mean()), 4),
                                            "t_paired_clips": round(tstat(dd), 2) if k != "z_t" else 0.0,
                                            "ci95_clip_boot": [round(x, 4) for x in boot_mean_ci(dd)] if k != "z_t" else [0.0, 0.0],
                                            "n_favouring": int((dd > 0).sum()), "n": int(len(dd))}
                dgh = Rc["zhat_GT"] - Rc["zhat_HOLD"]
                dgs = Rc["zhat_GT"] - Rc["zhat_SHUF"]
                tres["R4_primary"] = {
                    "zhat_GT_minus_zhat_HOLD": {"delta": round(float(dgh.mean()), 4), "t": round(tstat(dgh), 2),
                                                "ci95_clip_boot": [round(x, 4) for x in boot_mean_ci(dgh)],
                                                "n_favouring_GT": int((dgh > 0).sum()), "n": int(len(dgh))},
                    "zhat_GT_minus_zhat_SHUF": {"delta": round(float(dgs.mean()), 4), "t": round(tstat(dgs), 2),
                                                "ci95_clip_boot": [round(x, 4) for x in boot_mean_ci(dgs)],
                                                "n_favouring_GT": int((dgs > 0).sum()), "n": int(len(dgs))}}
                # banked-form estimator on the load-bearing target only (cost)
                if tn == "n_agents" and a.banked_form:
                    Rb = {k: loeo_banked(v, Yl) for k, v in Cl.items()}
                    bb = Rb["z_t"]
                    for k, sc in Rb.items():
                        dd = sc - bb
                        se = float(dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else 0.0
                        tres["banked_form"][k] = {"r2": round(float(sc.mean()), 4),
                                                  "delta_vs_z_t": round(float(dd.mean()), 4),
                                                  "t_as_envpred": round(float(dd.mean()) / max(se, 1e-12), 2),
                                                  "n_favouring": int((dd > 0).sum()), "n": int(len(dd)),
                                                  "note": "24 overlapping 12-clip pooled scores treated as independent (envpred.py:238-241)"}
                kres[tn] = tres
                p = tres["R4_primary"]["zhat_GT_minus_zhat_HOLD"]
                print(f"   K={K} {tn:<13} n={len(keep)}/{nrow} corrected: z_t r {Rc['z_t'].mean():+.4f} "
                      f"GT d {tres['corrected']['zhat_GT']['delta_vs_z_t']:+.4f} (t {tres['corrected']['zhat_GT']['t_paired_clips']:+.2f}) "
                      f"HOLD d {tres['corrected']['zhat_HOLD']['delta_vs_z_t']:+.4f} SHUF d {tres['corrected']['zhat_SHUF']['delta_vs_z_t']:+.4f} "
                      f"| GT-HOLD {p['delta']:+.4f} t {p['t']:+.2f}", flush=True)
            arm_out["k"][str(K)] = kres
        out["arms"][arm] = arm_out
        del world
    return out


# ----------------------------------------------------------------------------- R3 drift
def cmd_drift(a):
    import torch
    sys.path.insert(0, str(HERE))
    import panel_kfold as PK
    from rangeprobe_rff import rff_fold, within_clip_r as wcr
    out = stamp({"subcommand": "drift (R3)", "corpus": str(HELDOUT), "K": 4, "band": [0, 8],
                 "helpers": {"rangeprobe_rff": md5_of(HERE / "rangeprobe_rff.py"),
                             "panel_kfold": md5_of(HERE / "panel_kfold.py")},
                 "targets": {
                     "true": "dz = z_{t+K} - z_t (latentmotion.py:159)",
                     "endpoint_shuffled": "dz' = z_{pi(t)+K} - z_t, pi a within-clip permutation of the FUTURE endpoint only (start kept) — the class-(ii) arithmetic control",
                     "time_shuffled": "the banked control: rows of the projected target permuted within clip (latentmotion.py:272)"},
                 "arms": {}})
    F, K, BAND, K_FOLDS = 100, 4, (0, 8), 10
    clips = sorted(HELDOUT.glob("*.v2ep.pt"))[:a.nclips]
    for arm in a.arms.split(","):
        world, step, prov = load_arm(arm)
        ZT, DZ, DZC = [], [], []
        rng = np.random.default_rng(0)
        t0 = time.time()
        for c in clips:
            d, imgs, z, act, spd = encode_clip(world, str(c), F)
            yaw = np.asarray(d["poses"], dtype=np.float64)[:, 2]
            zt = z.float().numpy().astype(np.float64)
            m = min(len(zt) - K, len(act) - K, len(spd) - K, len(yaw) - K - 1)   # latentmotion.clip_rows
            if m < 30:
                continue
            i = np.arange(m)
            perm = rng.permutation(m)
            ZT.append(zt[i]); DZ.append(zt[i + K] - zt[i]); DZC.append(zt[perm + K] - zt[i])
        del world
        print(f"[{arm}] step {step} {len(ZT)} clips encoded in {time.time()-t0:.0f}s", flush=True)
        ALL = np.concatenate(DZ)
        mu = ALL.mean(0, keepdims=True)
        _, sv, Vt = np.linalg.svd(ALL - mu, full_matrices=False)
        ev = sv ** 2 / (sv ** 2).sum()
        res = {"per_direction": {}}
        TR, CT, SH = [], [], []
        for j in range(*BAND):
            Y = [(dz - mu) @ Vt[j][:, None] for dz in DZ]
            Yc = [(dzc - mu) @ Vt[j][:, None] for dzc in DZC]
            rj = np.random.default_rng(100 + j)
            Ysh = [y.ravel()[rj.permutation(len(y))][:, None] for y in Y]
            tr = PK.kfold_clip_scores(ZT, Y, rff_fold, wcr, K_FOLDS)
            ct = PK.kfold_clip_scores(ZT, Yc, rff_fold, wcr, K_FOLDS)
            sh = PK.kfold_clip_scores(ZT, Ysh, rff_fold, wcr, K_FOLDS)
            TR.append(tr); CT.append(ct); SH.append(sh)
            res["per_direction"][str(j)] = {"var_share": round(float(ev[j]), 4),
                                            "r_true": round(float(tr.mean()), 4),
                                            "r_endpoint_shuffled": round(float(ct.mean()), 4),
                                            "r_time_shuffled": round(float(sh.mean()), 4)}
            print(f"   dir {j} true {tr.mean():+.4f} endpoint-shuf {ct.mean():+.4f} time-shuf {sh.mean():+.4f}", flush=True)
        TR, CT, SH = np.concatenate(TR), np.concatenate(CT), np.concatenate(SH)
        res.update({**prov, "n_clips": len(ZT), "n_rows": int(sum(len(x) for x in ZT)),
                    "drift_r_true_minus_timeshuf": round(float((TR - SH).mean()), 4),
                    "t_true_vs_timeshuf_banked_form": round(tstat(TR - SH), 2),
                    "arithmetic_control_r_endpoint_shuffled": round(float(CT.mean()), 4),
                    "arithmetic_control_t_vs_timeshuf": round(tstat(CT - SH), 2),
                    "share_of_drift_that_is_arithmetic": round(float((CT.mean() - SH.mean()) / max(TR.mean() - SH.mean(), 1e-9)), 4),
                    "t_true_vs_endpoint_shuffled": round(tstat(TR - CT), 2),
                    "banked_reference": {"rdw8p30k_k4_band0_8": {"r": 0.6718, "t": 134.84},
                                         "postrain30k_k4_band0_8_reread_2026-09-02": {"r": 0.6674, "t": 146.85}}})
        out["arms"][arm] = res
    return out


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="P2 probe-leak audit re-scores (0-GPU)")
    ap.add_argument("cmd", choices=("data", "actdiv", "l3", "drift"))
    ap.add_argument("--arms", default="postrain30k,k8clip05p30k,k60clip05p30k,rdw8p30k,postrain30k_freeze")
    ap.add_argument("--ckpt", default="", help="explicit ckpt path (single --arms name only), e.g. o11p30k")
    ap.add_argument("--nclips", type=int, default=80)
    ap.add_argument("--no-banked-form", dest="banked_form", action="store_false")
    ap.add_argument("--stack", default=r"C:\Users\Admin\tanitad-wt\stack")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    tf = guard_and_stack(a.stack)
    t0 = time.time()
    res = {"data": cmd_data, "actdiv": cmd_actdiv, "l3": cmd_l3, "drift": cmd_drift}[a.cmd](a)
    res["_tanitad_imported_from"] = tf
    res["_seconds"] = round(time.time() - t0, 1)
    outp = pathlib.Path(a.out) if a.out else HERE.parent / "raw" / f"{a.cmd}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"-> {outp}  ({res['_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
