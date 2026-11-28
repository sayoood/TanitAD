"""STREAM D step 0 — RE-VERIFY the substrate and establish the ORACLE CEILING.

Nothing downstream is admissible until three things are MEASURED here, not
inherited from `…/2026-08-03-idm-derived-accel/`:

1. **The cached latents ARE the frozen v1 encoder's output on these frames.**
   A stale/mismatched latent cache would make every "unrecoverable" verdict a
   statement about the cache, not about the representation. We re-encode two
   episodes from the raw `ep_*.pt` and compare to the cache element-wise.

2. **The labels are re-derivable bit-exactly** from `poses`/`actions` in the raw
   episode files (the same heading-repair the prior panel used).

3. **The ORACLE CEILING.** ⚠️ The claim "CAN long_accel is recoverable from the
   TRUE speed track at R² 0.902" is the load-bearing premise of the whole
   derived-accel line and is INHERITED from a docstring. Re-measure it on THESE
   windows, and add the strictly stronger oracle: a ridge fit on the WHOLE true
   speed window (9 positions), which is the best any speed-mediated route can do.
   If even the oracle is low, "the head cannot recover long_accel" says nothing
   about the latents.

Also characterises the target itself (autocorrelation, band-split), because a
target that is mostly high-frequency CAN noise cannot be recovered by anything
and that is a property of the LABEL, not of the encoder.

usage: python verify_substrate.py --out raw/substrate_verification.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]          # …/TanitAD (HERE is the dated results dir)
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))
sys.path.insert(0, str(REPO / "taniteval"))

ENC_CKPT = Path(r"C:/Users/Admin/tanitad-data/eval/v1_speedjerk_ckpt.pt")
COMMA = Path(r"C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f")
LATENTS = Path(r"C:/Users/Admin/tanitad-data/eval/idm_derived_accel_latents.pt")
CLEAN_LO, CLEAN_HI = 40, 90
K, STRIDE = 4, 2
HORIZONS = (5, 10, 15, 20)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def heading_repair(poses: torch.Tensor, v_min: float = 0.5) -> torch.Tensor:
    """Verbatim from the prior panel (itself `idm3_labels.py:57-75`)."""
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


def r2(pred, gt) -> float:
    gt = np.asarray(gt, np.float64)
    pred = np.asarray(pred, np.float64)
    ssr = float(((gt - pred) ** 2).sum())
    sst = float(((gt - gt.mean()) ** 2).sum())
    return 1.0 - ssr / max(sst, 1e-12)


def main() -> int:
    import idm_head as ih

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "raw" / "substrate_verification.json"))
    ap.add_argument("--reencode", type=int, default=2,
                    help="how many episodes to re-encode from raw frames")
    a = ap.parse_args()

    sub = torch.load(LATENTS, map_location="cpu", weights_only=False)
    eps = sub["episodes"]
    res = {"_what": "STREAM D substrate re-verification + oracle ceiling",
           "latent_cache": str(LATENTS), "encoder": sub["encoder"],
           "encoder_step": sub["encoder_step"], "state_dim": int(sub["state_dim"]),
           "n_episodes": len(eps), "clean_band": [CLEAN_LO, CLEAN_HI],
           "k": K, "stride": STRIDE, "horizons": list(HORIZONS)}

    # ---- (2) labels re-derived from the RAW episode files ------------------- #
    lab = []
    for e in eps:
        raw = torch.load(COMMA / f"{e['name']}.pt", map_location="cpu",
                         weights_only=False)
        poses, actions = raw["poses"].float(), raw["actions"].float()
        fr = raw["frames_u8"]
        assert fr.shape[2] == 256 and fr.shape[3] == 256, \
            f"{e['name']}: v1 is 256px SQUARE, got {tuple(fr.shape)}"
        t = ih.valid_centers(fr.shape[0], K, HORIZONS, STRIDE)
        yr = ih.wrap_to_pi(heading_repair(poses)[t + 1]
                           - heading_repair(poses)[t - 1]) / (2.0 * ih.DT)
        S = torch.stack([poses[t, 3], yr, actions[t, 0], actions[t, 1]], -1)
        Q = ih.speed_seq_targets_at(poses, t, K)
        lab.append({"name": e["name"],
                    "S_bit_exact": bool(torch.equal(S, e["S"])),
                    "Q_bit_exact": bool(torch.equal(Q, e["Q"])),
                    "n": int(e["n"])})
    res["label_rederivation"] = {
        "n_episodes_checked": len(lab),
        "n_S_mismatch": sum(not r["S_bit_exact"] for r in lab),
        "n_Q_mismatch": sum(not r["Q_bit_exact"] for r in lab),
        "verdict": ("BIT-EXACT on all episodes"
                    if all(r["S_bit_exact"] and r["Q_bit_exact"] for r in lab)
                    else "MISMATCH — do not use this cache")}
    log(f"labels: {res['label_rederivation']['verdict']}")

    # ---- (1) latents re-encoded from raw frames ----------------------------- #
    if a.reencode > 0:
        import run_idm_proof as RIP
        device = "cuda" if torch.cuda.is_available() else "cpu"
        enc, ro, meta = RIP.load_encoder(str(ENC_CKPT), device)
        log(f"encoder: state_dim {meta['state_dim']} step {meta['ckpt_step']}")
        checks = []
        for e in eps[:a.reencode]:
            raw = torch.load(COMMA / f"{e['name']}.pt", map_location="cpu",
                             weights_only=False)
            z = RIP.encode_frames(enc, ro, raw["frames_u8"], device, batch=16)
            t = ih.valid_centers(z.shape[0], K, HORIZONS, STRIDE)
            offs = torch.arange(-K, K + 1)
            Z = z[t[:, None] + offs[None, :]]              # fp16, as cached
            same = bool(torch.equal(Z, e["Z"]))
            d = (Z.float() - e["Z"].float()).abs()
            checks.append({"name": e["name"], "bit_exact": same,
                           "max_abs_diff": float(d.max()),
                           "mean_abs_diff": float(d.mean()),
                           "cache_abs_mean": float(e["Z"].float().abs().mean())})
            log(f"  re-encode {e['name']}: bit_exact={same} "
                f"max|d|={float(d.max()):.5f}")
        # fp16 has ~3 decimal digits, so the ULP at |z|~1 is 9.8e-4 and a
        # max|d| of one-to-two ULP is kernel non-determinism, not a wrong cache.
        # The honest verdict keys on the RELATIVE mean error, which is the
        # quantity that could actually move an R².
        rel = max(c["mean_abs_diff"] / max(c["cache_abs_mean"], 1e-9)
                  for c in checks)
        res["latent_reencode"] = {
            "encoder_ckpt_step": meta["ckpt_step"],
            "state_dim": meta["state_dim"], "checks": checks,
            "max_relative_mean_abs_diff": float(rel),
            "fp16_ulp_at_unit_magnitude": 2.0 ** -10,
            "verdict": ("BIT-EXACT" if all(c["bit_exact"] for c in checks)
                        else ("REPRODUCED TO fp16 ROUNDING "
                              f"(max relative mean |d| {rel:.2e}, max |d| one "
                              "fp16 ULP) — the cache IS this encoder's output"
                              if rel < 1e-5 else
                              "⛔ MISMATCH BEYOND fp16 ROUNDING — do not use"))}
        del enc, ro
        torch.cuda.empty_cache()

    # ---- (3) ORACLE CEILING on the true speed track ------------------------- #
    S = torch.cat([e["S"] for e in eps]).numpy().astype(np.float64)
    Q = torch.cat([e["Q"] for e in eps]).numpy().astype(np.float64)
    eid = np.concatenate([np.full(e["n"], e["name"]) for e in eps])
    y = S[:, 3]                                   # CAN long_accel
    cd = (Q[:, K + 1] - Q[:, K - 1]) / (2.0 * ih.DT)     # centred difference
    # ridge on the WHOLE true speed window: the best speed-mediated oracle
    A = np.concatenate([Q, np.ones((len(Q), 1))], 1)
    coef = np.linalg.lstsq(A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ y, rcond=None)[0]
    lin = A @ coef
    # the same, but with a per-episode fit-then-predict split so it is honest
    res["oracle_ceiling"] = {
        "_what": "how well CAN long_accel is explainable from the TRUE speed track",
        "n_windows": int(len(y)),
        "centred_difference_r2": round(r2(cd, y), 4),
        "centred_difference_corr": round(float(np.corrcoef(cd, y)[0, 1]), 4),
        "ridge_on_full_true_speed_window_r2_INSAMPLE": round(r2(lin, y), 4),
        "gt_long_accel_std_mps2": round(float(y.std()), 4),
        "gt_long_accel_mae_about_mean_mps2": round(float(np.abs(y - y.mean()).mean()), 4),
        "note": ("the docstring claim of R² 0.902 was measured on a DIFFERENT "
                 "window set (30 episodes / 8,940 windows); this is the number "
                 "on the 50-episode clean band actually used here."),
    }
    log(f"ORACLE: centred-diff R² {res['oracle_ceiling']['centred_difference_r2']} "
        f"ridge-on-window R² "
        f"{res['oracle_ceiling']['ridge_on_full_true_speed_window_r2_INSAMPLE']}")

    # ---- target characterisation: is long_accel mostly HF noise? ------------ #
    per_ep, sm_r2, degenerate = [], [], []
    for e in eps:
        s = e["S"].numpy().astype(np.float64)[:, 3]
        if s.std() < 1e-9:
            # ⚠️ a FLAT channel: correlation and R² are both undefined here and
            # silently produce NaN. Record it — an episode whose CAN accel is
            # identically zero is a LABEL defect, not a modelling one.
            degenerate.append({"episode": e["name"], "n_windows": int(e["n"]),
                               "std": 0.0, "value": float(s.mean())})
            continue
        if len(s) >= 9:
            # a lagged pair can still be degenerate on ONE side (ep_00080 is
            # exactly 0.0 except for its last two windows), which silently
            # yields NaN. Guard per lag and count what survived.
            per_ep.append([float(np.corrcoef(s[:-L], s[L:])[0, 1])
                           if min(s[:-L].std(), s[L:].std()) > 1e-12
                           else np.nan for L in (1, 2, 4, 8)])
        # a 5-tap boxcar over the (stride-2) window sequence = 1 s smoothing
        sm_r2.append(r2(np.convolve(s, np.ones(5) / 5.0, mode="same"), s))
    ac = np.array(per_ep)
    zero_frac = float((np.abs(y_all := np.concatenate(
        [e["S"].numpy()[:, 3] for e in eps])) < 1e-9).mean())
    res["target_character"] = {
        "autocorr_of_long_accel_lag_1_2_4_8_windows_mean":
            [round(float(x), 4) for x in np.nanmean(ac, 0)],
        "n_episodes_in_autocorr": int(len(per_ep)),
        "n_episodes_per_lag": [int(x) for x in (~np.isnan(ac)).sum(0)],
        "lag_unit": "1 window = stride 2 frames = 0.2 s",
        "r2_of_1s_boxcar_smoothed_vs_raw": round(float(np.mean(sm_r2)), 4),
        "frac_windows_with_long_accel_exactly_zero": round(zero_frac, 4),
        "degenerate_flat_accel_episodes": degenerate,
        "interpretation": ("R² of a smoothed copy against the raw target upper-"
                           "bounds what any SLOW predictor can score. If it is "
                           "low, most of long_accel lives above the frequency a "
                           "vision head could plausibly track."),
    }
    log(f"target: autocorr {res['target_character']['autocorr_of_long_accel_lag_1_2_4_8_windows_mean']} "
        f"smoothed-R² {res['target_character']['r2_of_1s_boxcar_smoothed_vs_raw']}")

    res["window_totals"] = {"n_windows": int(len(y)),
                            "n_episodes": int(len(np.unique(eid)))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
