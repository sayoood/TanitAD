"""E2 orthogonality/isotropy admissibility on the OPERATIVE flagship-speed ckpt.

Adapts the verified 2026-07-10 instrument (spectral_orthogonality.orthogonality_report)
to the flagship-speed model (WorldModel flagship4b, action_dim=3) on the eval pod's
canonical PhysicalAI val. Drops the pre-reset caveat of the 07-17 verification (which
used the step-6500 base250cam ckpt).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/root")                        # spectral_orthogonality.py
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from spectral_orthogonality import orthogonality_report        # noqa: E402
from tanitad.config import flagship4b_config                   # noqa: E402
from tanitad.data.mixing import load_episode                   # noqa: E402
from tanitad.models.fourbrain import WorldModel                # noqa: E402

CKPT = "/root/models/flagship-speed/ckpt.pt"
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
EPISODES = 40


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = flagship4b_config()
    object.__setattr__(cfg.predictor, "action_dim", 3)
    if getattr(cfg, "tactical_pred", None) is not None:
        object.__setattr__(cfg.tactical_pred, "action_dim", 3)
    world = WorldModel(cfg).to(device).eval()
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    world.load_state_dict(ck["model"], strict=True)
    step = int(ck.get("step", -1))
    print(f"[orth] flagship-speed loaded (step {step}) on {device}")

    eps = [load_episode(str(p), mmap=True)
           for p in sorted(Path(VAL).glob("ep_*.pt"))[:EPISODES]]
    zs = []
    with torch.no_grad():
        for ep in eps:
            fr = (ep.frames.float() / 255.0 if ep.frames.dtype == torch.uint8
                  else ep.frames).to(device)
            z = torch.cat([world.encode(fr[i:i + 32])
                           for i in range(0, fr.shape[0], 32)])
            zs.append(z.cpu())
    Z = torch.cat(zs)
    print(f"[orth] {Z.shape[0]} readout latents, dim {Z.shape[1]}")

    rep = orthogonality_report(Z)
    d = rep.to_dict()
    d["checkpoint_step"] = step
    d["n_episodes"] = len(eps)
    d["model"] = "flagship-speed (WorldModel flagship4b, action_dim=3)"
    d["val"] = "physicalai-val-0c5f7dac3b11"
    out = "/root/taniteval/results/2026-07-18-orth-flagship-speed.json"
    Path(out).write_text(json.dumps(d, indent=2, default=str))
    for k in ("n_samples", "state_dim", "active_k", "iso_ratio_global",
              "iso_ratio_active", "cond_number_active", "rms_offdiag_corr",
              "cov_effective_rank", "participation_ratio_global"):
        print(f"  {k}: {d.get(k)}")
    print("  VERDICT:", d["verdict"])
    print(f"[orth] full report -> {out}")


if __name__ == "__main__":
    main()
