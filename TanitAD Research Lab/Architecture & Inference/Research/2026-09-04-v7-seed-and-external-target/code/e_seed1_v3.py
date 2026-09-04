"""E-SEED-1 v3 — the ONE-VARIABLE arm that separates the two candidate causes.

v2 measures that the seeded trunk carries almost none of DINOv3's speed
decodability. TWO independent defects could explain it and they have OPPOSITE
consequences for the v7 design:

  (N) INPUT NORMALISATION. `V6Stack.encode_window` feeds `.float()/255` -- [0,1].
      DINOv3 was trained on IMAGENET-normalised input. In this repo that
      normalisation exists ONLY inside `O7Distill.target()`
      (train_v6_staged.py:970-972), which feeds the *teacher*, never the trunk.
      ⇒ if this is the cause, the fix is EXACT and FREE: the affine map folds
      into the patch-embed Conv2d, the way LayerScale already folds.

  (A) ARCHITECTURE. DINOv3 is RoPE-only and attends with CLS + 4 register
      tokens; `ViTEncoder` has a learned absolute `pos` table, no RoPE and no
      extra tokens. ⇒ if this is the cause, NO seed of DINOv3 into `ViTEncoder`
      can be faithful, and the design must change class, not normalisation.

ONE VARIABLE: the input normalisation, applied to the KNOWN-GOOD trunk.
  dino_hf      real DINOv3, ImageNet-normalised   (v2's reference, cached)
  dino_hf_01   real DINOv3, input [0,1]           <- the ONLY difference

COMMITTED IN ADVANCE (before the arm was run):
  * if `dino_hf_01` stays within ~20 % of `dino_hf`'s speed R2, normalisation is
    NOT the cause and (A) carries the loss;
  * if `dino_hf_01` falls to the `scratch` floor, normalisation carries most of
    it and the repair is a two-line change to the converter;
  * anything in between is reported as a SPLIT cause with both magnitudes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BANK = HERE / "eseed1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "clone_scripts"))
import e_seed1_arms as E1                                    # noqa: E402
import e_seed1_v2 as V2                                      # noqa: E402


@torch.no_grad()
def extract(imnet: bool):
    frames = np.load(BANK / "frames_u8.npy", mmap_mode="r")
    n = frames.shape[0]
    m = E1.build_hf().to(E1.DEV)
    mean, std = E1.IMNET_MEAN.to(E1.DEV), E1.IMNET_STD.to(E1.DEV)
    g = np.empty((n, 1024), dtype=np.float32)
    B = E1.BATCH
    t0 = time.time()
    for i in range(0, n, B):
        x = torch.from_numpy(np.array(frames[i:i + B])).to(E1.DEV).float() / 255.
        if imnet:
            x = (x - mean) / std
        tok = m(pixel_values=x.to(E1.DT)).last_hidden_state[:, -256:].float()
        g[i:i + B] = tok.mean(1).cpu().numpy()
    del m
    torch.cuda.empty_cache()
    assert np.isfinite(g).all() and float(g.std()) > 0
    print(f"  extracted imnet={imnet} std={g.std():.4f} {time.time()-t0:.0f}s")
    return g


def main() -> int:
    L = np.load(BANK / "labels.npy")
    ep = np.load(BANK / "rows.npy")[:, 0]
    fit, sc = ep < V2.N_FIT_EP, ep >= V2.N_FIT_EP
    p = BANK / "feat_dino_hf_01.npy"
    F = np.load(p) if p.exists() else extract(imnet=False)
    if not p.exists():
        np.save(p, F)
    ref = np.load(BANK / "feat_dino_hf.npy")

    out = {}
    for nm, Farm in (("dino_hf", ref), ("dino_hf_01", F)):
        Z0 = Farm[0::2]
        row = {"participation_sigma2": E1.participation(Z0)}
        for t, y in (("speed", L[:, 0]), ("yaw_rate", L[:, 1])):
            row[t] = V2.probe1(Z0[fit], y[fit], ep[fit], Z0[sc], y[sc], ep[sc])
        out[nm] = row
        print(f"[{nm}] speed R2 {row['speed']['r2']:+.4f} "
              f"{row['speed']['ci95']} lam={row['speed']['lambda']:g}"
              f"{' EDGE' if row['speed']['lambda_at_grid_edge'] else ''} | "
              f"yaw_rate {row['yaw_rate']['r2']:+.4f} "
              f"{row['yaw_rate']['ci95']} | "
              f"participation {row['participation_sigma2']:.3f}")

    a, b = out["dino_hf"]["speed"]["r2"], out["dino_hf_01"]["speed"]["r2"]
    out["_verdict"] = {
        "dino_hf_speed_r2": a, "dino_hf_01_speed_r2": b,
        "retained_fraction": (b / a) if a else None,
        "one_variable": "input normalisation (ImageNet vs [0,1])",
        "evidence_class": "MEASURED (ours; RTX 4060; 90 val episodes)",
    }
    (BANK / "eseed1_v3.json").write_text(json.dumps(out, indent=2))
    print(f"\nRETAINED FRACTION of DINOv3 speed R2 under [0,1] input: "
          f"{b/a:.3f}" if a else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
