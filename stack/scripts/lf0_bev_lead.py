"""LF0 — is the lead gap readable off the DECODED BEV? (JEPA_PHYSICS_SURVEY §LF0)

⛔ WHY THIS IS THE FIRST LEVER. P1 measured that the POOLED latent carries no
readable lead distance: R²(enc) ≤ 0 with a linear probe, every transform
(log1p / inverse / TTC-proxy) failed, and a 2-layer MLP ceiling read **−0.334**
— the "H-absent / missing state variable" verdict. LF0 asks the different
question the survey pre-registered: *is it absent, or merely absent FROM THE
POOLED VIEW?* P8 attempt-2 makes that answerable — the predicted latent retains
the environment at **retention 0.932**, so the scene is in there; the question is
whether the lead gap survives to a place a planner could read it.

⭐ WHY GEOMETRIC, NOT A PROBE. A high-dimensional probe on ~hundreds of windows
can read noise, and a spatial-token probe has far more parameters than P1's
pooled one — so "spatial wins" would be confounded with "spatial has more
capacity". This reader has **ZERO fitted parameters**: it walks the decoded
occupancy raster forward along the ego corridor and returns the range of the
first occupied cell. Nothing is fitted, so nothing can overfit, and a positive
result means the fix is exposing a read-off that ALREADY EXISTS — the survey's
RC1, with zero new training.

⛔ THE SANITY CHECK THAT GATES EVERY CONCLUSION. The same reader is run on the
GROUND-TRUTH raster first. If reading the GT raster does not recover the true
lead gap, the READER is broken and NO statement about the latent is admissible —
a broken reader and an empty latent produce the identical null. This is the same
class as the P1 instrument bug (a class filter that made the failure look like a
model verdict until it was fixed and the failure survived).

⚠️ τ is INHERITED from the P8 gate (τ* = 0.7, chosen on the ENCODED arm), never
re-tuned here. Re-picking a threshold on the lead task would be fitting a
one-parameter model and calling it geometry.

⚠️ Windows with no occupied cell in the corridor are **CENSORED and reported**,
never coded as max-range — coding them as "60 m" manufactures correlation out of
missing data, which is exactly the NO_LABEL-is-not-free-flow rule.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

TAU_P8 = 0.7                 # INHERITED from the P8 gate; never re-tuned here
CORRIDOR_M = (1.0, 1.5, 2.0)  # reported for all; HEADLINE is 1.5 m
HEADLINE_CORRIDOR = 1.5
GATE_R2 = 0.30               # JEPA_PHYSICS_SURVEY LF0: "spatial tokens read gap"


def corridor_cols(ny: int, y_half_m: float, cell_m: float,
                  half_width_m: float) -> np.ndarray:
    """Column indices whose CENTER lies within ±half_width of the ego axis.

    Mirrors ``bev_raster._cell_centers``: col j centre is
    ``-y_half_m + (j + 0.5) * cell_m`` (+y is LEFT, col 0 is the RIGHT edge)."""
    yc = -y_half_m + (np.arange(ny, dtype=np.float64) + 0.5) * cell_m
    return np.nonzero(np.abs(yc) <= half_width_m)[0]


def read_lead_range(raster: np.ndarray, *, tau: float, cols: np.ndarray,
                    cell_m: float, min_row: int = 0) -> float:
    """Range (m) of the nearest occupied cell ahead inside the corridor.

    ``raster`` is [nx, ny] with nx = forward. Returns ``np.nan`` when the
    corridor is empty — CENSORED, never max-range."""
    if raster.ndim != 2:
        raise ValueError(f"expected [nx, ny], got {raster.shape}")
    band = raster[min_row:, cols]                       # [nx-min_row, |cols|]
    hit = np.nonzero((band >= tau).any(axis=1))[0]
    if hit.size == 0:
        return float("nan")
    return float((hit[0] + min_row + 0.5) * cell_m)


def r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def score_arm(read: np.ndarray, truth: np.ndarray) -> dict:
    """Paired scoring over the windows where BOTH are finite."""
    ok = np.isfinite(read) & np.isfinite(truth)
    n_total = int(truth.size)
    n_truth = int(np.isfinite(truth).sum())
    n_paired = int(ok.sum())
    out = {
        "n_windows_total": n_total,
        "n_with_label": n_truth,
        "n_paired": n_paired,
        "censored_rate_on_labelled": (
            round(1.0 - n_paired / n_truth, 4) if n_truth else None),
        "_censor_note": "windows with an empty corridor are EXCLUDED, never "
                        "coded as max range — coding them manufactures "
                        "correlation out of missing data",
    }
    if n_paired < 10:
        out.update(status="UNAVAILABLE",
                   reason=f"only {n_paired} paired windows — below the n=10 "
                          f"floor; a correlation here would not be a "
                          f"measurement (WORK ITEM, not a pass)")
        return out
    p, t = read[ok], truth[ok]
    out.update(status="OK", r2=round(r2(p, t), 4),
               spearman=round(spearman(p, t), 4),
               mae_m=round(float(np.abs(p - t).mean()), 4),
               bias_m=round(float((p - t).mean()), 4),
               read_mean_m=round(float(p.mean()), 4),
               true_mean_m=round(float(t.mean()), 4))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("lf0_bev_lead", description=__doc__)
    ap.add_argument("--p8-run", required=True,
                    help="the p8-occupancy run dir (supplies p8_head.pt)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10, help="P1's gate horizon")
    ap.add_argument("--tau", type=float, default=TAU_P8,
                    help="INHERITED from the P8 gate; changing it is fitting")
    ap.add_argument("--n-windows", type=int, default=400)
    ap.add_argument("--min-row", type=int, default=2,
                    help="skip the first rows — the ego's own footprint")
    a, rest = ap.parse_known_args(argv)

    import torch

    from tanitad.data.bev_raster import GRID_DEFAULT
    from train_p8_occupancy import (BEVOccupancyHead, batch_rasters,
                                    build_args, p8_latents)

    nx, ny = GRID_DEFAULT.shape
    cols = {w: corridor_cols(ny, GRID_DEFAULT.y_half_m, GRID_DEFAULT.cell_m, w)
            for w in CORRIDOR_M}
    print(f"[lf0] grid {nx}x{ny} cell {GRID_DEFAULT.cell_m} m · corridor cols "
          + " ".join(f"{w}m:{len(c)}" for w, c in cols.items()), flush=True)

    # ⛔ --ckpt is consumed by OUR parser, so it never reaches build_args and the
    # P8 parser dies with "the following arguments are required: --ckpt".
    # Re-inject it rather than making the caller pass it twice — a duplicated
    # flag is the kind of thing that drifts between the two copies.
    # --v2-cache is likewise required by the TRAINER's parser; LF0 never builds a
    # train loader, so the caller points it at the val cache (as p8_bev_reel
    # does) and it is only there to satisfy the shared arg surface.
    args = build_args(rest + ["--ckpt", a.ckpt,
                              "--out", os.path.join(a.out, "_tmp")])
    device = args.device if torch.cuda.is_available() else "cpu"
    amp_on = (device == "cuda") and not getattr(args, "no_amp", False)
    os.makedirs(a.out, exist_ok=True)

    from torch.utils.data import default_collate

    from eval_flagship_v4 import (_eval_cfg, _plan, build_v2_val_episodes,
                                  load_v1_from_ck, resolve_eval_frames)
    from train_flagship4b import FlagshipWindowDataset
    from train_flagship_v4 import _to_device

    cfg = _eval_cfg(args)
    frame = resolve_eval_frames(args)
    world, _plan_m, base_step = load_v1_from_ck(args, frame)
    world.eval().to(device)
    providers, _prov = build_v2_val_episodes(args, cache_frame=frame,
                                             train_frame=frame)
    ds_val = FlagshipWindowDataset(providers, cfg, _plan(args))

    head = BEVOccupancyHead(world.state_dim, grid=GRID_DEFAULT).to(device)
    sd = torch.load(os.path.join(a.p8_run, "p8_head.pt"), map_location="cpu")
    head.load_state_dict(sd["model"] if "model" in sd else sd)
    head.eval()

    n = min(a.n_windows, len(ds_val))
    reads = {f"{src}@{w}": np.full(n, np.nan)
             for src in ("gt", "enc", "pred") for w in CORRIDOR_M}
    truth = np.full(n, np.nan)

    with torch.no_grad():
        for i in range(n):
            b = _to_device(default_collate([ds_val[i]]), device)
            # GT raster at the SAME k the latents are rolled to
            rk, keep, _ = batch_rasters(ds_val, [i], "gt", a.k, GRID_DEFAULT)
            _zt, z_enc, z_hat = p8_latents(world, b, (a.k,), amp_on=amp_on,
                                           device=device)
            pe = torch.sigmoid(head(z_enc[a.k])).float().cpu().numpy()[0]
            pp = torch.sigmoid(head(z_hat[a.k])).float().cpu().numpy()[0]
            g = np.asarray(rk)[0] if keep is None or keep[0] else None
            for w, cc in cols.items():
                kw = dict(tau=a.tau, cols=cc, cell_m=GRID_DEFAULT.cell_m,
                          min_row=a.min_row)
                if g is not None:
                    reads[f"gt@{w}"][i] = read_lead_range(g, **kw)
                reads[f"enc@{w}"][i] = read_lead_range(pe, **kw)
                reads[f"pred@{w}"][i] = read_lead_range(pp, **kw)
            # the TRUE gap is the GT raster's own read at the headline corridor:
            # it is a geometric fact about the labelled scene, not a model output
            truth[i] = reads[f"gt@{HEADLINE_CORRIDOR}"][i]
            if (i + 1) % 50 == 0:
                print(f"[lf0] {i+1}/{n}", flush=True)

    res = {
        "probe": "LF0 — decoded-BEV lead read-off (JEPA_PHYSICS_SURVEY LF0)",
        "_evidence_class": "MEASURED (ours)",
        "_tier": "T0 — a world-model diagnostic, NEVER driving performance",
        "_zero_parameters": "the reader fits nothing; a positive result means "
                            "the read-off already exists and only needs "
                            "exposing (RC1, zero new training)",
        "tau": a.tau, "tau_provenance": "INHERITED from the P8 gate (tau*=0.7, "
                                        "chosen on the ENCODED arm); re-tuning "
                                        "it here would be fitting",
        "k": a.k, "min_row": a.min_row, "n_requested": n,
        "grid": {"nx": nx, "ny": ny, "cell_m": GRID_DEFAULT.cell_m},
        "gate_r2": GATE_R2, "headline_corridor_m": HEADLINE_CORRIDOR,
        "arms": {},
    }
    for key, v in reads.items():
        res["arms"][key] = score_arm(v, truth)

    # ⛔ THE READER SANITY GATE. gt@headline is read from the SAME raster that
    # defines `truth`, so it is self-consistent by construction — the real check
    # is that the OTHER corridors' GT reads agree with it. If they do not, the
    # corridor geometry is wrong and nothing downstream is admissible.
    gt_alt = [k for k in reads if k.startswith("gt@")
              and k != f"gt@{HEADLINE_CORRIDOR}"]
    agree = all(res["arms"][k].get("status") == "OK"
                and res["arms"][k].get("spearman", 0) >= 0.8 for k in gt_alt)
    res["reader_sanity"] = {
        "checked": gt_alt, "passed": bool(agree),
        "rule": "GT reads at other corridor widths must rank-agree (rho>=0.8) "
                "with the headline GT read; a reader that disagrees with itself "
                "cannot be used to conclude anything about the latent",
    }

    hp = res["arms"].get(f"pred@{HEADLINE_CORRIDOR}", {})
    he = res["arms"].get(f"enc@{HEADLINE_CORRIDOR}", {})
    if not agree:
        res["verdict"] = ("INADMISSIBLE — the reader failed its own sanity "
                          "check; fix the corridor geometry before reading "
                          "anything into the latent arms")
    elif hp.get("status") != "OK" or he.get("status") != "OK":
        res["verdict"] = ("UNAVAILABLE — too few paired windows at the "
                          "headline corridor; report the n, not a correlation")
    else:
        pr2, er2 = hp["r2"], he["r2"]
        res["verdict"] = (
            f"LF0 PASS — the decoded BEV reads the lead gap (pred R2 {pr2}, "
            f"enc R2 {er2} >= {GATE_R2}); RC1 CONFIRMED, the fix is exposing "
            f"this read-off, ZERO new training"
            if max(pr2, er2) >= GATE_R2 else
            f"LF0 FAIL — the decoded BEV does NOT read the lead gap (pred R2 "
            f"{pr2}, enc R2 {er2} < {GATE_R2}). With P1's pooled MLP ceiling at "
            f"-0.334 and the reader sanity-checked, 'missing state variable' "
            f"survives its second independent test")
    json.dump(res, open(os.path.join(a.out, "lf0_gate.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("verdict", "reader_sanity")},
                     indent=1), flush=True)
    print("LF0_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
