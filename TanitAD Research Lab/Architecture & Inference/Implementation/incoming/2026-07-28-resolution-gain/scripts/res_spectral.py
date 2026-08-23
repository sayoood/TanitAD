"""RESOLUTION-GAIN — `A-SPEC`: PROVE the ladder removes information, and prove the low-pass matters.

Two things are asserted by every resolution ladder and almost never measured:

  1. that each rung really has ~`1/k` of the baseline's angular bandwidth, and
  2. that the degradation is a RESOLUTION change and not an ALIASING artifact.

Both are measurable in seconds from the rendered frames, with no encoder and no labels. This script
renders the v5 frame on a sample of real clips, applies every rung, and reports the radially
averaged power spectrum on a CENTRAL crop (chosen to sit well inside the cylindrical observed mask,
so the mask edge — a step discontinuity with power at every frequency — cannot contaminate the
measurement).

The two registered readings:

  * `f95` — the radial frequency (cycles/px) below which 95 % of the AC power lies. If the low-pass
    is real, `f95(D_k) ~ f95(V5)/k`. **Reported as the ratio, so the claim is falsifiable.**
  * `D_1p5` vs `A_1p5_alias` — the SAME nominal factor, one with `antialias=True` and one without.
    Aliasing folds energy back into the band, so the aliased arm must carry MORE high-frequency
    power at the same nominal resolution. If the two are indistinguishable, the whole
    "we removed information rather than adding aliasing" claim is unsupported and must be retracted.

usage:  python res_spectral.py <out_json> [n_clips]
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("TANITAD_STACK",
                                  r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack"))

from res_extract import ARMS, FRAME_V5, degrade                        # noqa: E402
from tanitad.data.calib import cylindrical_rectify                     # noqa: E402
from tanitad.data.physicalai import (discover_r0_clips,                # noqa: E402
                                     intrinsics_for_clip)

ROOT = os.environ.get("TANITAD_PAI_ROOT", r"C:\Users\Admin\tanitad-data\physicalai")
CROP_H, CROP_W = 192, 448          # central crop, comfortably inside the observed mask
N_FRAMES = 6


def radial_spectrum(img: torch.Tensor, nbin: int = 64):
    """[T,3,H,W] uint8 -> (bin centres in cycles/px, mean radial AC power, f95)."""
    H, W = img.shape[-2:]
    y0, x0 = (H - CROP_H) // 2, (W - CROP_W) // 2
    x = img[..., y0:y0 + CROP_H, x0:x0 + CROP_W].float().mean(1)
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    win = (torch.hann_window(CROP_H, device=x.device)[:, None]
           * torch.hann_window(CROP_W, device=x.device)[None, :])
    F = torch.fft.rfft2(x * win)
    p = (F.real ** 2 + F.imag ** 2).mean(0)
    fy = torch.fft.fftfreq(CROP_H, device=p.device).abs()[:, None].expand_as(p)
    fx = torch.fft.rfftfreq(CROP_W, device=p.device)[None, :].expand_as(p)
    r = torch.sqrt(fy ** 2 + fx ** 2).clamp(max=0.5)
    edges = torch.linspace(0, 0.5, nbin + 1, device=p.device)
    idx = torch.bucketize(r.flatten(), edges) - 1
    idx = idx.clamp(0, nbin - 1)
    tot = torch.zeros(nbin, device=p.device).index_add_(0, idx, p.flatten())
    cnt = torch.zeros(nbin, device=p.device).index_add_(0, idx, torch.ones_like(p.flatten()))
    prof = (tot / cnt.clamp_min(1)).cpu().numpy()
    # f95 uses the CUMULATIVE power actually present at each radius (tot, not the mean profile)
    cum = torch.cumsum(tot, 0) / tot.sum().clamp_min(1e-12)
    j = int(torch.searchsorted(cum, torch.tensor(0.95, device=cum.device)).item())
    centres = ((edges[:-1] + edges[1:]) / 2).cpu().numpy()
    # ⭐ `f_cut` is the FALSIFIABLE one. `f95` is a percentile of a renormalised distribution, so it
    # is NOT expected to scale as 1/k (removing high-frequency power concentrates what remains and
    # drives the percentile DOWN further than the cutoff moves). The CUTOFF is what a low-pass
    # actually sets, and it must land at the rung's own Nyquist `0.5/k`.
    pk = float(prof.max())
    above = np.nonzero(prof >= pk * 1e-4)[0]
    f_cut = float(centres[int(above.max())]) if len(above) else 0.0
    return centres, prof, float(centres[min(j, nbin - 1)]), f_cut


def main():
    out_json = sys.argv[1]
    n_clips = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    import av
    clips = discover_r0_clips(ROOT)
    rows, used = [], 0
    for c in clips:
        if used >= n_clips:
            break
        intr = intrinsics_for_clip(c["clip_id"], ROOT)
        if not intr.per_clip:
            continue
        frs = []
        with av.open(str(c["mp4"])) as ct:
            st = ct.streams.video[0]
            st.thread_type = "AUTO"
            for i, fr in enumerate(ct.decode(st)):
                if i % 20 == 0:
                    frs.append(torch.from_numpy(fr.to_ndarray(format="rgb24")).permute(2, 0, 1))
                if len(frs) >= N_FRAMES:
                    break
        native = torch.stack(frs).cuda()
        base = cylindrical_rectify(native, intr, FRAME_V5)
        mask = cylindrical_rectify.last_mask.cuda()
        rig = "B" if intr.cy >= 650 else "A"
        for a, (b, k, aa) in ARMS.items():
            if b != "V5":
                continue
            img = degrade(base, k, aa, mask)
            _cen, _prof, f95, f_cut = radial_spectrum(img)
            rows.append({"clip": used, "rig": rig, "arm": a, "k": round(k, 6),
                         "antialias": aa, "f95_cyc_per_px": round(f95, 6),
                         "f_cut_cyc_per_px": round(f_cut, 6)})
        used += 1
        del native, base

    import pandas as pd
    df = pd.DataFrame(rows)
    g = df.groupby("arm")[["f95_cyc_per_px", "f_cut_cyc_per_px"]].agg(["mean", "std"])
    base_f95 = float(g.loc["V5_640", ("f95_cyc_per_px", "mean")])
    out = {"n_clips": used, "n_frames_per_clip": N_FRAMES,
           "central_crop": [CROP_H, CROP_W],
           "note": "f95 = radial frequency (cyc/px) below which 95 % of AC power lies (a percentile "
                   "of a RENORMALISED distribution, so it is monotone but NOT expected to scale as "
                   "1/k). f_cut = the -40 dB cutoff of the radial power profile — THAT is what a "
                   "low-pass sets and it must land at the rung's own Nyquist 0.5/k.",
           "per_arm": {}}
    for a in g.index:
        k = ARMS[a][1]
        fc = float(g.loc[a, ("f_cut_cyc_per_px", "mean")])
        out["per_arm"][a] = {
            "k": round(k, 6), "antialias": ARMS[a][2],
            "px_per_deg": round((FRAME_V5.f_ref * math.pi / 180.0) / k, 4),
            "f95_mean": round(float(g.loc[a, ("f95_cyc_per_px", "mean")]), 6),
            "f95_sd": round(float(g.loc[a, ("f95_cyc_per_px", "std")]), 6),
            "f95_ratio_to_baseline": round(
                float(g.loc[a, ("f95_cyc_per_px", "mean")]) / base_f95, 4),
            "f_cut_mean": round(fc, 6),
            "f_cut_sd": round(float(g.loc[a, ("f_cut_cyc_per_px", "std")]), 6),
            "f_cut_predicted_0p5_over_k": round(0.5 / k, 4),
            "f_cut_over_prediction": round(fc / (0.5 / k), 4)}
    d15 = out["per_arm"]["D_1p5"]["f_cut_mean"]
    a15 = out["per_arm"]["A_1p5_alias"]["f_cut_mean"]
    out["A_CTRL_antialias_matters"] = {
        "f_cut_D_1p5_lowpassed": d15, "f_cut_A_1p5_aliased": a15,
        "f95_D_1p5_lowpassed": out["per_arm"]["D_1p5"]["f95_mean"],
        "f95_A_1p5_aliased": out["per_arm"]["A_1p5_alias"]["f95_mean"],
        "aliased_carries_more_hf": bool(
            out["per_arm"]["A_1p5_alias"]["f95_mean"] > out["per_arm"]["D_1p5"]["f95_mean"]),
        "ratio_aliased_over_lowpassed_f95": round(
            out["per_arm"]["A_1p5_alias"]["f95_mean"]
            / max(out["per_arm"]["D_1p5"]["f95_mean"], 1e-9), 4),
        "rule": "aliasing folds energy back into the band, so at the SAME nominal factor the "
                "no-low-pass arm must carry MORE high-frequency power. If it does not, the "
                "ladder's 'we removed information' claim is unsupported."}
    json.dump(out, open(out_json, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
