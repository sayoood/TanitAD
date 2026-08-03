"""P9 — fix the IDM's ONE broken channel: long_accel, by DERIVING it from speed.

THE DEFECT, from primary source
    `…/2026-07-27-fleet-sync-idm-steer/raw/idm5_ensemble.json` — the banked
    HELD-OUT read — gives, on every seed and both domains:
        speed  R2  +0.86    yaw_rate R2 +0.81    steer R2 +0.74
        long_accel R2  -0.2398 (a0) / -0.1862 (seed0) / -0.1510 (seed1)
                       per-domain pai -0.298 / cm -0.254
    A NEGATIVE R2 means the head predicts long_accel WORSE THAN THE TRAINING
    MEAN. Three of four channels work; one is broken.

WHY IT IS AVOIDABLE — MEASURED, not argued
    On the comma2k19 val cache (30 episodes, 8,940 windows) the CAN long_accel
    channel is recoverable from the TRUE speed track by a centred difference at
    R2 0.9021 (pooled corr 0.9506). The head's window is NON-CAUSAL and spans
    t-4..t+4, so v(t-1) and v(t+1) are already inside it. The head simply had no
    reason to connect the two: `long_accel` was a fourth independent linear
    readout of the centre token, unrelated in the architecture and in the loss to
    the `speed` channel it predicts at R2 0.86.

THE FIX  (`stack/scripts/idm_head.py`)
    Read speed at EVERY window position and DERIVE long_accel as the centred
    difference of that sequence. The derived channel is still supervised on its
    own target — differencing multiplies the sequence's error by 1/(2*dt)=5x, so
    a sequence-only loss makes it WORSE (MEASURED on the synthetic contract:
    0.6134 derived vs 0.8755 direct). The physics is a reparameterisation, not a
    substitute for supervision.

THE PROTOCOL
    Both arms are trained FROM SCRATCH on the same TRAIN episodes and scored on
    the same EPISODE-DISJOINT held-out episodes, so the comparison is clean even
    though the shipped checkpoint's own comma numbers are in-corpus (this cache
    IS its training pool — `IDM_FOUR_FAMILIES.md:105`). We make no claim about
    the shipped checkpoint here; the contrast is internal to this run.

usage: python run_idm_derived_accel.py --out results_idm_derived_accel.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))
sys.path.insert(0, str(REPO / "taniteval"))

ENC_CKPT = Path(r"C:/Users/Admin/tanitad-data/eval/v1_speedjerk_ckpt.pt")
COMMA = Path(r"C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f")
#: the content-clean band w.r.t. the shipped head's comma train set, settled by
#: sha256 of raw bytes on both hosts (…/2026-07-27-anchor-settlement/).
CLEAN_LO, CLEAN_HI = 40, 90
K, STRIDE = 4, 2
HORIZONS = (5, 10, 15, 20)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def heading_repair(poses: torch.Tensor, v_min: float = 0.5):
    """comma2k19 heading is arctan2 of ENU velocity and is UNDEFINED at
    standstill. Verbatim from `…/2026-08-03-idm-four-families/
    run_idm_four_families.py:72` (itself `idm3_labels.py:57-75`), so the yaw-rate
    label here is the same one the shipped checkpoint was stamped with."""
    yaw = poses[:, 2].numpy().astype(np.float64).copy()
    v = poses[:, 3].numpy().astype(np.float64)
    obs = v >= v_min
    if not obs.any():
        return torch.from_numpy(yaw).float()
    ux, uy = np.cos(yaw), np.sin(yaw)
    idx = np.where(obs, np.arange(len(yaw)), -1)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.argmax(obs))
    idx[idx < 0] = first
    return torch.from_numpy(np.arctan2(uy[idx], ux[idx])).float()


def build_targets(poses, actions, t, yaw_rep, horizons=HORIZONS):
    """GT scalars + ego-frame waypoints, matching the prior panel's convention:
    yaw_rate from the REPAIRED heading, the trajectory frame from the RAW yaw."""
    import idm_head as ih
    speed, steer, accel = poses[t, 3], actions[t, 0], actions[t, 1]
    yr = ih.wrap_to_pi(yaw_rep[t + 1] - yaw_rep[t - 1]) / (2.0 * ih.DT)
    scal = torch.stack([speed, yr, steer, accel], dim=-1)
    yaw0, xy0 = poses[t, 2], poses[t, :2]
    traj = torch.stack([ih.ego_frame(poses[t + h, :2] - xy0, yaw0)
                        for h in horizons], 1)
    return scal, traj


# --------------------------------------------------------------------------- #
def build_substrate(cache_path: Path, device: str, limit: int = 0):
    """Encode the clean episodes once and window them. -> per-episode dict."""
    import idm_head as ih
    import run_idm_proof as RIP

    eps = [p for p in sorted(COMMA.glob("ep_*.pt"))
           if CLEAN_LO <= int(p.stem.split("_")[1]) < CLEAN_HI]
    if limit:
        eps = eps[:limit]
    log(f"clean episodes: {len(eps)} (ep_{CLEAN_LO:05d}..ep_{CLEAN_HI-1:05d})")
    if cache_path.exists():
        log(f"loading cached latents {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    enc, ro, meta = RIP.load_encoder(str(ENC_CKPT), device)
    state_dim = meta["state_dim"]
    log(f"encoder loaded: state_dim={state_dim} step={meta.get('ckpt_step')}")
    out = {"state_dim": int(state_dim), "episodes": [],
           "encoder": str(ENC_CKPT), "encoder_step": meta.get("ckpt_step")}
    for n, p in enumerate(eps):
        e = torch.load(p, map_location="cpu", weights_only=False)
        fr = e["frames_u8"]
        # ⛔ geometry assertion BEFORE scoring: flagship v1 is 256px SQUARE.
        assert fr.ndim == 4 and fr.shape[2] == 256 and fr.shape[3] == 256, \
            f"{p.name}: expected [T,C,256,256], got {tuple(fr.shape)}"
        z = RIP.encode_frames(enc, ro, fr, device, batch=32).float()
        poses, actions = e["poses"].float(), e["actions"].float()
        t = ih.valid_centers(z.shape[0], K, HORIZONS, STRIDE)
        if t.numel() == 0:
            continue
        offs = torch.arange(-K, K + 1)
        Z = z[t[:, None] + offs[None, :]]
        S, Tj = build_targets(poses, actions, t, heading_repair(poses))
        Q = ih.speed_seq_targets_at(poses, t, K)
        out["episodes"].append({"name": p.stem, "Z": Z.half(), "S": S,
                                "T": Tj, "Q": Q, "n": int(Z.shape[0])})
        if (n + 1) % 10 == 0:
            log(f"  encoded {n+1}/{len(eps)}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache_path)
    log(f"cached latents -> {cache_path}")
    return out


def split_by_episode(sub, hold_every: int = 3):
    """EPISODE-DISJOINT split. Deterministic, no reselection of the corpus."""
    tr, ho = [], []
    for i, e in enumerate(sub["episodes"]):
        (ho if i % hold_every == 0 else tr).append(e)
    return tr, ho


def stack(eps, shuffle_latents_seed: int | None = None):
    Z = torch.cat([e["Z"] for e in eps]).float()
    S = torch.cat([e["S"] for e in eps])
    T = torch.cat([e["T"] for e in eps])
    Q = torch.cat([e["Q"] for e in eps])
    eid = np.concatenate([np.full(e["n"], e["name"]) for e in eps])
    if shuffle_latents_seed is not None:
        # NEGATIVE CONTROL: destroy the latent<->target link, keep everything
        # else. A protocol that still "learns" here is measuring itself.
        g = np.random.default_rng(shuffle_latents_seed)
        Z = Z[torch.from_numpy(g.permutation(len(Z)))]
    return Z, S, T, Q, eid


# --------------------------------------------------------------------------- #
def main() -> int:
    import idm_head as ih
    from tanitad.eval import idm_families as FF
    from tanitad.eval.ap_ci import (paired_stat_episode_cluster_bootstrap,
                                    stat_episode_cluster_bootstrap)
    import taniteval.ci as TCI
    assert str(REPO) in str(Path(TCI.__file__).resolve()), TCI.__file__

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_idm_derived_accel.json")
    ap.add_argument("--latent-cache",
                    default=str(Path.home() / "tanitad-data" / "eval"
                                / "idm_derived_accel_latents.pt"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}  estimator={TCI.__file__}")

    sub = build_substrate(Path(a.latent_cache), device, limit=a.limit)
    D = sub["state_dim"]
    tr_eps, ho_eps = split_by_episode(sub)
    log(f"split: TRAIN {len(tr_eps)} eps / HELDOUT {len(ho_eps)} eps "
        f"(episode-disjoint)")
    Ztr, Str, Ttr, Qtr, _ = stack(tr_eps)
    Zho, Sho, Tho, Qho, eid = stack(ho_eps)
    log(f"windows: train {len(Ztr):,}  heldout {len(Zho):,}  "
        f"clusters {len(np.unique(eid))}")

    arms = {
        "A_BASELINE_direct_accel": dict(speed_seq=False, shuffle=None),
        "B_DERIVED_accel": dict(speed_seq=True, shuffle=None),
        "NEG_shuffled_latents": dict(speed_seq=True, shuffle=7),
    }
    preds, per_arm = {}, {}
    for name, cfg in arms.items():
        seed_preds = []
        for s in range(a.seeds):
            Zs, Ss, Ts, Qs, _ = (stack(tr_eps, cfg["shuffle"] + s) if cfg["shuffle"]
                                 is not None else (Ztr, Str, Ttr, Qtr, None))
            head, _std, meta = ih.fit_head(
                (Zs, Ss, Ts, Qs), state_dim=D, horizons=HORIZONS,
                epochs=a.epochs, seed=s, device=device, log=lambda *_: None,
                speed_seq=cfg["speed_seq"])
            p = ih.predict(head, Zho, device=device)
            seed_preds.append((p["scalars"].numpy(), p["traj"].numpy(),
                               p["scalars_direct"].numpy()))
            log(f"  {name} seed {s} fitted ({meta['params']:,} params)")
        # 3-seed ensemble = mean of per-seed predictions, the shipped recipe
        preds[name] = tuple(np.mean([sp[i] for sp in seed_preds], 0) for i in range(3))
        per_arm[name] = {"params": meta["params"], "seeds": a.seeds,
                         "speed_seq": cfg["speed_seq"],
                         "shuffled_latents": cfg["shuffle"] is not None}

    gt_s = Sho.numpy()
    gt_t = Tho.numpy()

    def r2_fn(pred_col, gt_col):
        def fn(sel):
            g, p = gt_col[sel], pred_col[sel]
            ssr = float(((g - p) ** 2).sum())
            sst = float(((g - g.mean()) ** 2).sum())
            return 1.0 - ssr / max(sst, 1e-12)
        return fn

    res = {
        "_what": "IDM long_accel: DERIVED from the speed sequence vs regressed directly",
        "defect_source": ("…/2026-07-27-fleet-sync-idm-steer/raw/idm5_ensemble.json "
                          "— long_accel R2 NEGATIVE on every seed and both domains"),
        "substrate": {"cache": str(COMMA), "clean_band": [CLEAN_LO, CLEAN_HI],
                      "encoder": sub["encoder"], "encoder_step": sub["encoder_step"],
                      "state_dim": D, "k": K, "stride": STRIDE,
                      "horizons": list(HORIZONS), "geometry": "256x256 asserted"},
        "split": {"train_episodes": [e["name"] for e in tr_eps],
                  "heldout_episodes": [e["name"] for e in ho_eps],
                  "unit": "EPISODE-DISJOINT", "n_train_windows": int(len(Ztr)),
                  "n_heldout_windows": int(len(Zho))},
        "epochs": a.epochs, "n_boot": a.n_boot,
        "estimator": "episode-cluster bootstrap (paired where two arms compared)",
        "arms": per_arm, "scalars": {}, "ade": {}, "four_families": {},
        "paired": {},
    }

    # ---- per-scalar R2, each with its own interval ------------------------- #
    for name, (ps, _pt, pdir) in preds.items():
        res["scalars"][name] = {}
        for j, ch in enumerate(ih.SCALAR_NAMES):
            res["scalars"][name][ch] = stat_episode_cluster_bootstrap(
                r2_fn(ps[:, j], gt_s[:, j]), eid, n_boot=a.n_boot, name=f"r2_{ch}")
            res["scalars"][name][ch]["mae"] = round(
                float(np.abs(ps[:, j] - gt_s[:, j]).mean()), 5)
        if arms[name]["speed_seq"]:
            j = ih.SCALAR_NAMES.index("long_accel")
            res["scalars"][name]["long_accel_DIRECT_readout"] = \
                stat_episode_cluster_bootstrap(
                    r2_fn(pdir[:, j], gt_s[:, j]), eid, n_boot=a.n_boot,
                    name="r2_long_accel_direct")
        log(f"  {name}: " + " ".join(
            f"{c}={res['scalars'][name][c]['point']:+.4f}" for c in ih.SCALAR_NAMES))

    # ---- ADE (one row, never the result) ----------------------------------- #
    for name, (_ps, pt, _d) in preds.items():
        per_win = np.linalg.norm(pt - gt_t, axis=-1).mean(1)
        res["ade"][name] = TCI.episode_cluster_bootstrap(per_win, eid,
                                                         n_boot=a.n_boot)

    # ---- the FOUR FAMILIES, per arm ---------------------------------------- #
    for name, (ps, pt, _d) in preds.items():
        res["four_families"][name] = FF.all_families(
            pt, gt_t, FF.IDM_DT_S, pred_scalars=ps, gt_scalars=gt_s)
        log(f"  four families {name}: unavailable="
            f"{res['four_families'][name]['_unavailable']}")

    # ---- TACTICAL balanced accuracy WITH ITS INTERVAL ---------------------- #
    # The family rule binds each family to an estimator and a CI. Balanced
    # accuracy is set-level (a confusion matrix over the whole selection), so it
    # needs the callable form of the bootstrap, not the per-window reducer.
    AXES = {"lateral": FF.LATERAL_CLASSES,
            "longitudinal": FF.LONGITUDINAL_CLASSES,
            "mixed": ("accelerate", "decelerate", "keep", "left", "right")}
    Gm = FF.manoeuvre_classes(gt_t, FF.IDM_DT_S)
    Pm = {n: FF.manoeuvre_classes(p[1], FF.IDM_DT_S) for n, p in preds.items()}

    def ba_fn(pred_cls, gt_cls, k):
        def fn(sel):
            return FF.balanced_accuracy(FF.confusion(pred_cls[sel], gt_cls[sel], k))
        return fn

    res["tactical_ba_ci"] = {}
    for name in preds:
        res["tactical_ba_ci"][name] = {
            ax: stat_episode_cluster_bootstrap(
                ba_fn(Pm[name][ax], Gm[ax], len(cls)), eid,
                n_boot=a.n_boot, name=f"balanced_accuracy_{ax}")
            for ax, cls in AXES.items()}
        log(f"  TACTICAL BA {name}: " + " ".join(
            f"{ax}={res['tactical_ba_ci'][name][ax]['point']:.4f}" for ax in AXES))

    # ---- PAIRED contrasts, the decision-grade rows -------------------------- #
    def paired(ka, kb, tag):
        out = {}
        for j, ch in enumerate(ih.SCALAR_NAMES):
            out[f"r2_{ch}"] = paired_stat_episode_cluster_bootstrap(
                r2_fn(preds[ka][0][:, j], gt_s[:, j]),
                r2_fn(preds[kb][0][:, j], gt_s[:, j]), eid,
                n_boot=a.n_boot, name=f"d_r2_{ch}")
        aw = np.linalg.norm(preds[ka][1] - gt_t, axis=-1).mean(1)
        bw = np.linalg.norm(preds[kb][1] - gt_t, axis=-1).mean(1)
        out["ade_2s"] = TCI.paired_episode_cluster_bootstrap(aw, bw, eid,
                                                             n_boot=a.n_boot)
        for ax, cls in AXES.items():
            out[f"tactical_ba_{ax}"] = paired_stat_episode_cluster_bootstrap(
                ba_fn(Pm[ka][ax], Gm[ax], len(cls)),
                ba_fn(Pm[kb][ax], Gm[ax], len(cls)), eid,
                n_boot=a.n_boot, name=f"d_balanced_accuracy_{ax}")
        res["paired"][tag] = out
        log(f"  PAIRED {tag}: " + " ".join(
            f"{c}={out['r2_'+c]['delta']:+.4f}"
            f"{'*' if out['r2_'+c]['separated'] else ''}"
            for c in ih.SCALAR_NAMES))

    paired("B_DERIVED_accel", "A_BASELINE_direct_accel", "B_minus_A_THE_FIX")
    paired("B_DERIVED_accel", "NEG_shuffled_latents", "B_minus_NEG_control")
    paired("A_BASELINE_direct_accel", "NEG_shuffled_latents", "A_minus_NEG_control")

    Path(a.out).write_text(json.dumps(res, indent=1))
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
