"""P9 — the FIRST four-family read of the IDM, on the content-clean comma20 set.

WHAT THIS ANSWERS
    The shipped IDM (`idm_head_v4_steer_ens3.pt`) has never been scored on
    anything but four scalar R2 values. No IDM script imports `four_families`
    (two probes). This runs LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC on
    real predictions, with the paired episode-cluster bootstrap and three
    negative controls.

SUBSTRATE — all local, 0 pod GPU-h
    head    <repo>/TanitAD Research Hub/Benchmarks & Eval/Implementation/
            incoming/2026-07-27-fleet-sync-idm-steer/idm_head_v4_steer_ens3.pt
    encoder C:/Users/Admin/tanitad-data/eval/v1_speedjerk_ckpt.pt   (flagship v1)
    frames  C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f
    split   the CLEAN20 content-clean episodes, joined BY episode_id (content),
            never by filename, from
            .../2026-07-27-anchor-settlement/raw/anchor_resettlement.json

!! ENCODE FIDELITY. Latents are re-encoded here on the dev box, not read from
   the pod's /root/idm2/lat. The program has already flagged that this path is a
   different substrate (`controls.encode_fidelity`: "a plausibility band, not an
   equality check"). Absolute numbers here are therefore NOT bit-comparable to
   the banked pod numbers. Every CONTRAST below is internal to this run -- same
   latents, same windows, same rows -- so the controls and the paired deltas are
   valid regardless.

LABEL PROTOCOL matches the checkpoint's own stamp ("REPAIRED (heading_repair,
   v_min=0.5)"): comma2k19 heading is arctan2 of ENU velocity and is undefined at
   standstill, so it is repaired before the yaw-rate label is derived.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))
sys.path.insert(0, str(REPO / "taniteval"))

HUB = REPO / "TanitAD Research Hub"
HEAD_CKPT = (HUB / "Benchmarks & Eval/Implementation/incoming"
             / "2026-07-27-fleet-sync-idm-steer/idm_head_v4_steer_ens3.pt")
ANCHOR = (HUB / "Benchmarks & Eval/Implementation/incoming"
          / "2026-07-27-anchor-settlement/raw/anchor_resettlement.json")
IDMV3 = HUB / "Architecture & Inference/Implementation/incoming/2026-07-27-idm-v3"
IDMV2 = HUB / "Architecture & Inference/Implementation/incoming/2026-07-26-idm-v2"
ENC_CKPT = Path(r"C:/Users/Admin/tanitad-data/eval/v1_speedjerk_ckpt.pt")
COMMA = Path(r"C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f")

HORIZONS = (5, 10, 15, 20)
K = 4
STRIDE = 2
V_MIN = 0.5
DT = 0.1


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# labels — heading_repair verbatim (idm3_labels.py:57-75)                      #
# --------------------------------------------------------------------------- #
def heading_repair(poses: torch.Tensor, v_min: float = V_MIN):
    yaw = poses[:, 2].numpy().astype(np.float64).copy()
    v = poses[:, 3].numpy().astype(np.float64)
    obs = v >= v_min
    if not obs.any():
        return torch.from_numpy(yaw).float(), torch.from_numpy(obs)
    ux, uy = np.cos(yaw), np.sin(yaw)
    idx = np.where(obs, np.arange(len(yaw)), -1)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.argmax(obs))
    idx[idx < 0] = first
    return torch.from_numpy(np.arctan2(uy[idx], ux[idx])).float(), torch.from_numpy(obs)


def wrap_to_pi(a):
    return a - (2 * math.pi) * torch.floor((a + math.pi) / (2 * math.pi))


def ego_frame(dxy, yaw):
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    return torch.stack([dxy[..., 0] * c - dxy[..., 1] * s,
                        dxy[..., 0] * s + dxy[..., 1] * c], dim=-1)


def build_targets(poses, actions, t, yaw_rep):
    """GT scalars (speed, yaw_rate, steer, long_accel) + ego-frame 2 s waypoints.
    yaw_rate uses the REPAIRED heading, matching the checkpoint's label stamp."""
    speed = poses[t, 3]
    steer = actions[t, 0]
    accel = actions[t, 1]
    yr = wrap_to_pi(yaw_rep[t + 1] - yaw_rep[t - 1]) / (2.0 * DT)
    scal = torch.stack([speed, yr, steer, accel], dim=-1)
    yaw0, xy0 = poses[t, 2], poses[t, :2]
    traj = torch.stack([ego_frame(poses[t + h, :2] - xy0, yaw0) for h in HORIZONS], 1)
    return scal, traj


# --------------------------------------------------------------------------- #
def load_encoder(ckpt_path, device):
    """Delegates to the pod runner's OWN loader (`run_idm_proof.load_encoder`)
    rather than reimplementing it: the v1 checkpoint carries no `cfg`, so the
    architecture comes from `flagship4b_config()`, and a private copy of that
    reconstruction is exactly the kind of drift that silently changes a latent."""
    import run_idm_proof as RIP                                  # noqa: PLC0415
    enc, ro, meta = RIP.load_encoder(str(ckpt_path), device)
    return enc, ro, meta["state_dim"], meta["ckpt_step"]


@torch.no_grad()
def encode(enc, ro, frames_u8, device, batch=32):
    import run_idm_proof as RIP                                  # noqa: PLC0415
    return RIP.encode_frames(enc, ro, frames_u8, device, batch=batch)


#: MEASURED, sha256 of raw bytes on BOTH hosts
#: (.../2026-07-27-anchor-settlement/raw/anchor_overlap.json):
#:   A0_HELDOUT_cm_40_70 x V3_TRAIN = 0   and   A0_UNUSED_cm_70_90 x V3_TRAIN = 0
#: so local ep_00040..ep_00089 are content-clean w.r.t. the head's comma train set.
CLEAN_LO, CLEAN_HI = 40, 90
#: the 9 local episodes that ARE byte-identical to episodes in V3_TRAIN — kept as
#: a LEAK-SENSITIVITY arm, not discarded: if the head scores the same on known
#: memorised episodes as on clean ones, the "leak" framing itself is suspect.
KNOWN_LEAKED = ("ep_00002", "ep_00005", "ep_00007", "ep_00014", "ep_00017",
                "ep_00019", "ep_00025", "ep_00028", "ep_00030")


def split_episodes():
    """-> {"clean": [paths], "leaked": [paths]} with the provenance recorded."""
    clean, leaked = [], []
    for p in sorted(COMMA.glob("ep_*.pt")):
        n = int(p.stem.split("_")[1])
        if CLEAN_LO <= n < CLEAN_HI:
            clean.append(p)
        elif p.stem in KNOWN_LEAKED:
            leaked.append(p)
    return {"clean": clean, "leaked": leaked}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_idm_four_families.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from tanitad.eval import idm_families as FF                  # noqa: PLC0415
    from tanitad.eval.ap_ci import (                             # noqa: PLC0415
        paired_stat_episode_cluster_bootstrap, stat_episode_cluster_bootstrap)
    sys.path.insert(0, str(IDMV2))
    sys.path.insert(0, str(IDMV3))
    import idm3_arms as A                                        # noqa: PLC0415
    # R11: `idm2_lib.py:19` does sys.path.insert(0, "/root/taniteval"), which
    # DEFEATS the estimator pin on a pod. Verify which `ci` we actually got.
    import taniteval.ci as _tci                                  # noqa: PLC0415
    assert str(REPO) in str(Path(_tci.__file__).resolve()), _tci.__file__
    log(f"estimator in use: {_tci.__file__}")

    # ---- 1. leak-audited split, BY CONTENT --------------------------------- #
    split = split_episodes()
    log(f"split: clean={len(split['clean'])} leaked={len(split['leaked'])}")

    # ---- 2. encode + build windows ----------------------------------------- #
    enc, ro, state_dim, enc_step = load_encoder(ENC_CKPT, device)
    log(f"encoder loaded: state_dim={state_dim} step={enc_step}")

    def build(paths):
        Z, S, T, EID = [], [], [], []
        for p in paths:
            d = torch.load(p, map_location="cpu", weights_only=False)
            poses = d["poses"].float()
            actions = d["actions"].float()
            Tn = poses.shape[0]
            z = encode(enc, ro, d["frames_u8"], device)
            lo, hi = max(K, 1), Tn - 1 - max(K, max(HORIZONS))
            t = torch.arange(lo, hi + 1, STRIDE, dtype=torch.long)
            yaw_rep, _ = heading_repair(poses, V_MIN)
            scal, traj = build_targets(poses, actions, t, yaw_rep)
            offs = torch.arange(-K, K + 1)
            Z.append(z[t[:, None] + offs[None, :]])
            S.append(scal)
            T.append(traj)
            EID.extend([p.stem] * len(t))
        return (torch.cat(Z), torch.cat(S).numpy().astype(np.float64),
                torch.cat(T).numpy().astype(np.float64), np.array(EID))

    t0 = time.time()
    Z, Sgt, Tgt, EID = build(split["clean"])
    log(f"CLEAN: windows {Z.shape[0]:,} latent {tuple(Z.shape)} "
        f"episodes {len(split['clean'])} ({time.time()-t0:.0f}s)")
    Zlk, Slk, Tlk, EIDlk = build(split["leaked"])
    log(f"LEAKED: windows {Zlk.shape[0]:,} episodes {len(split['leaked'])}")
    del enc, ro
    torch.cuda.empty_cache()

    # ---- 3. the shipped 3-seed ensemble ------------------------------------ #
    ck = torch.load(HEAD_CKPT, map_location="cpu", weights_only=False)
    hk = ck["head_kwargs"]
    assert hk["class"] == "idm3_arms.IDMHeadV3", hk["class"]
    assert hk["state_dim"] == state_dim, (hk["state_dim"], state_dim)

    def run(zin):
        Ps, Pt = [], []
        for sd in ck["state_dicts"]:
            h = A.IDMHeadV3(state_dim=hk["state_dim"], d_model=hk["d_model"],
                            window=hk["window"], use_ctx=hk["use_ctx"],
                            side_dim=hk["side_dim"], acc_bins=hk["acc_bins"]).to(device)
            h.load_state_dict(sd)
            h.eval()
            s, tj = [], []
            with torch.no_grad():
                for i in range(0, zin.shape[0], 512):
                    o = h(zin[i:i + 512].to(device).float())
                    s.append(o["scalars"].cpu())
                    tj.append(o["traj"].cpu())
            Ps.append(torch.cat(s).numpy().astype(np.float64))
            Pt.append(torch.cat(tj).numpy().astype(np.float64))
            del h
        torch.cuda.empty_cache()
        return np.mean(Ps, 0), np.mean(Pt, 0)          # ck["ensemble_rule"]

    log("predicting: shipped ensemble on the CLEAN split")
    Spred, Tpred = run(Z)
    log("predicting: shipped ensemble on the KNOWN-LEAKED split (sensitivity)")
    Slkp, Tlkp = run(Zlk)

    # ---- 4. NEGATIVE CONTROLS ---------------------------------------------- #
    rng = np.random.default_rng(0)
    log("predicting: NEG-1 latents shuffled across windows")
    perm = rng.permutation(Z.shape[0])
    Sshuf, Tshuf = run(Z[perm])
    log("NEG-2 blind mean predictor (train-mean of THIS set: an upper bound on a"
        " blind arm, so it flatters the control, not us)")
    Sblind = np.repeat(Spred.mean(0, keepdims=True), len(Spred), 0)
    Tblind = np.repeat(Tpred.mean(0, keepdims=True), len(Tpred), 0)
    log("NEG-3 GT trajectory with the yaw sign flipped (right<->left)")
    Tflip = Tgt.copy()
    Tflip[..., 1] *= -1.0

    arms = {
        "shipped_ensemble_v4_steer_ens3": (Spred, Tpred),
        "NEG1_latents_shuffled_across_windows": (Sshuf, Tshuf),
        "NEG2_blind_mean_predictor": (Sblind, Tblind),
        "NEG3_gt_with_lateral_sign_flipped": (Sgt.copy(), Tflip),
    }

    # ---- 5. four families + ADE + CIs -------------------------------------- #
    res = {"meta": {
        "head": str(HEAD_CKPT), "head_kind": ck["kind"], "rung": ck["rung"],
        "seeds": ck["seeds"], "ensemble_rule": ck["ensemble_rule"],
        "label_protocol": ck["label_protocol"],
        "encoder_ckpt": str(ENC_CKPT), "encoder_step": enc_step,
        "state_dim": state_dim, "k": K, "stride": STRIDE,
        "horizons_steps": list(HORIZONS), "waypoint_dt_s": FF.IDM_DT_S,
        "n_windows": int(Z.shape[0]),
        "n_episodes": int(len(split["clean"])),
        "episodes": [p.stem for p in split["clean"]],
        "split": (f"local comma2k19-val-61c46fca8f7f ep_{CLEAN_LO:05d}.."
                  f"ep_{CLEAN_HI-1:05d}"),
        "leak_provenance": (
            "MEASURED sha256 of raw bytes on BOTH hosts (2026-07-27-anchor-"
            "settlement/raw/anchor_overlap.json): A0_HELDOUT_cm_40_70 x V3_TRAIN "
            "= 0 and A0_UNUSED_cm_70_90 x V3_TRAIN = 0. ⚠️ PARTIAL: that clears "
            "these rows against the 42 comma episodes of V3_TRAIN, but the v4 "
            "ensemble trained at rung 757 on cm=121 episodes and no fingerprint "
            "exists for the other 79. Leak status is VERIFIED-AGAINST-42, "
            "UNVERIFIED-AGAINST-79."),
        "leak_sensitivity_arm": {
            "episodes": list(KNOWN_LEAKED),
            "n_windows": int(Zlk.shape[0]),
            "why": ("these 9 local episodes are byte-identical to episodes IN "
                    "V3_TRAIN. Scoring them alongside the clean split turns the "
                    "leak into a measured quantity instead of a caveat.")},
        "n_boot": a.n_boot, "device": device,
        "encode_fidelity_caveat": (
            "latents RE-ENCODED on the dev box, not the pod's /root/idm2/lat. "
            "Absolute values are NOT bit-comparable to banked pod numbers; every "
            "contrast here is internal to this run and is valid."),
    }}

    for name, (Sp, Tp) in arms.items():
        fam = FF.all_families(Tp, Tgt, FF.IDM_DT_S, pred_scalars=Sp, gt_scalars=Sgt)
        fam["ADE_2s_m"] = round(FF.ade(Tp, Tgt), 4)
        fam["scalar_r2"] = {
            nm: round(float(1 - ((Sgt[:, j] - Sp[:, j]) ** 2).sum()
                            / max(((Sgt[:, j] - Sgt[:, j].mean()) ** 2).sum(), 1e-12)), 4)
            for j, nm in enumerate(("speed", "yaw_rate", "steer", "long_accel"))}
        res[name] = fam
        log(f"  {name}: ADE {fam['ADE_2s_m']}  "
            f"lat_BA {fam['TACTICAL']['lateral']['balanced_accuracy']}  "
            f"lon_BA {fam['TACTICAL']['longitudinal']['balanced_accuracy']}  "
            f"speedR2 {fam['scalar_r2']['speed']}")

    # ---- 6. intervals: point estimates + paired vs each control ------------- #
    def ba(Tp, axis):
        """Bootstrap statistic for one arm+axis.

        ``require_all=True``: a resample that happens to contain no turn (turns
        are <1 % of a highway corpus) would otherwise leave one class present
        and score EVERY arm — including a blind constant — at BA 1.0. Those
        draws are dropped and counted, not silently averaged in.
        """
        gm = FF.manoeuvre_classes(Tgt, FF.IDM_DT_S)[axis]
        pm = FF.manoeuvre_classes(Tp, FF.IDM_DT_S)[axis]
        k = 5 if axis == "mixed" else 3
        return lambda sel: FF.balanced_accuracy(
            FF.confusion(pm[sel], gm[sel], k), require_all=True)

    ade_pw = {n: np.linalg.norm(Tp - Tgt, axis=-1).mean(1)
              for n, (_, Tp) in arms.items()}
    sp_pw = {n: np.abs(Sp[:, 0] - Sgt[:, 0]) for n, (Sp, _) in arms.items()}
    yr_pw = {n: np.abs(Sp[:, 1] - Sgt[:, 1]) for n, (Sp, _) in arms.items()}

    ci = {}
    base = "shipped_ensemble_v4_steer_ens3"
    for axis in ("lateral", "longitudinal"):
        ci[f"TACTICAL.{axis}.balanced_accuracy"] = {
            n: stat_episode_cluster_bootstrap(ba(Tp, axis), EID, n_boot=a.n_boot,
                                              name=f"BA_{axis}")
            for n, (_, Tp) in arms.items()}
        # ⚠️ `arms[n][1]`, NOT a loop variable from the comprehension above —
        # a leaked `Tp` silently compared every control against the SAME arm and
        # printed three identical deltas.
        ci[f"TACTICAL.{axis}.paired_vs_controls"] = {
            n: paired_stat_episode_cluster_bootstrap(
                ba(arms[base][1], axis), ba(arms[n][1], axis), EID,
                n_boot=a.n_boot, name=f"dBA_{axis}")
            for n in arms if n != base}

    from taniteval.ci import (episode_cluster_bootstrap,          # noqa: PLC0415
                              paired_episode_cluster_bootstrap)
    for label, pw in (("ADE_2s_m", ade_pw), ("LONGITUDINAL.scalar_speed_mae", sp_pw),
                      ("LATERAL.scalar_yaw_rate_mae", yr_pw)):
        ci[label] = {n: episode_cluster_bootstrap(v, EID, n_boot=a.n_boot)
                     for n, v in pw.items()}
        ci[label + ".paired_vs_controls"] = {
            n: paired_episode_cluster_bootstrap(pw[base], pw[n], EID, n_boot=a.n_boot)
            for n in arms if n != base}
    res["intervals"] = ci

    # ---- 7. leak sensitivity: same head, KNOWN-LEAKED episodes ------------- #
    lk = FF.all_families(Tlkp, Tlk, FF.IDM_DT_S, pred_scalars=Slkp, gt_scalars=Slk)
    lk["ADE_2s_m"] = round(FF.ade(Tlkp, Tlk), 4)
    lk["scalar_r2"] = {
        nm: round(float(1 - ((Slk[:, j] - Slkp[:, j]) ** 2).sum()
                        / max(((Slk[:, j] - Slk[:, j].mean()) ** 2).sum(), 1e-12)), 4)
        for j, nm in enumerate(("speed", "yaw_rate", "steer", "long_accel"))}
    lk["_note"] = ("9 episodes byte-identical to V3_TRAIN. NOT an independent "
                   "result — this is the memorisation reference the clean split "
                   "is read against. It is a DIFFERENT episode set, so the gap is "
                   "descriptive, not a paired test.")
    lk["ADE_2s_m_ci"] = episode_cluster_bootstrap(
        np.linalg.norm(Tlkp - Tlk, axis=-1).mean(1), EIDlk, n_boot=a.n_boot)
    for axis in ("lateral", "longitudinal"):
        gm = FF.manoeuvre_classes(Tlk, FF.IDM_DT_S)[axis]
        pm = FF.manoeuvre_classes(Tlkp, FF.IDM_DT_S)[axis]
        lk[f"BA_{axis}_ci"] = stat_episode_cluster_bootstrap(
            lambda sel, p=pm, g=gm: FF.balanced_accuracy(
                FF.confusion(p[sel], g[sel], 3)),
            EIDlk, n_boot=a.n_boot, name=f"BA_{axis}")
    res["LEAK_SENSITIVITY_known_leaked_episodes"] = lk
    log(f"  LEAKED-arm: ADE {lk['ADE_2s_m']} lat_BA "
        f"{lk['TACTICAL']['lateral']['balanced_accuracy']} lon_BA "
        f"{lk['TACTICAL']['longitudinal']['balanced_accuracy']} "
        f"speedR2 {lk['scalar_r2']['speed']}")

    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
