"""Run the orthogonality/isotropy admissibility instrument on a checkpoint.

Loads a training checkpoint + cached validation episodes, encodes every frame
window to its readout latent z, and reports the OrthogonalityReport — the
admissibility gate on the D-021 linear-sizing claim (theory anchor 2605.26379).

Per the same caveat as `run_spectral.py`: on an early/undertrained checkpoint this
is a DIAGNOSTIC (has SIGReg's isotropic target converged yet?), not a decision-grade
admissibility verdict — that needs the trained Stage-0 checkpoint.

Usage:
  python run_orthogonality.py --ckpt C:/.../eval/ckpt_full.pt \
      --cache-dir C:/.../eval --episodes 24 --out orth_step6500.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))                 # spectral_orthogonality
_STACK = Path(__file__).resolve().parents[6] / "stack"                    # worktree/stack
if _STACK.exists():
    sys.path.insert(0, str(_STACK))

from spectral_orthogonality import orthogonality_report                   # noqa: E402
from tanitad.config import base250cam_config                              # noqa: E402
from tanitad.data.mixing import load_episode                              # noqa: E402
from tanitad.instruments.numerics import strict_numerics                  # noqa: E402
from tanitad.models.fourbrain import WorldModel                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", required=True,
                    help="dir containing a *val* episode cache (ep_*.pt); newest is used")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = base250cam_config()
    world = WorldModel(cfg).to(device).eval()
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    state = ck["model"] if "model" in ck else ck
    world.load_state_dict(state)
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    print(f"[orth] checkpoint loaded (step {step}) on {device}")

    root = Path(args.cache_dir)
    # dirs only (a sibling comma_val.tgz would otherwise be picked by the *val* glob)
    val_dirs = sorted(p for p in root.glob("*val*") if p.is_dir())
    if not val_dirs and list(root.glob("ep_*.pt")):
        val_dirs = [root]
    assert val_dirs, f"no val cache dir under {root}"
    src = val_dirs[-1]
    eps = [load_episode(str(p), mmap=True)
           for p in sorted(src.glob("ep_*.pt"))[:args.episodes]]
    print(f"[orth] {len(eps)} val episodes from {src.name}")

    zs = []
    with torch.no_grad(), strict_numerics():
        for ep in eps:
            frames = (ep.frames.float() / 255.0 if ep.frames.dtype == torch.uint8
                      else ep.frames).to(device)
            z = torch.cat([world.encode(frames[i:i + 32])
                           for i in range(0, frames.shape[0], 32)])
            zs.append(z.cpu())
    Z = torch.cat(zs)                                                     # [N, S]
    print(f"[orth] {Z.shape[0]} readout latents, dim {Z.shape[1]}")

    rep = orthogonality_report(Z)
    d = rep.to_dict()
    d["checkpoint_step"] = step
    d["n_episodes"] = len(eps)
    d["DIAGNOSTIC_NOTE"] = ("early-checkpoint isotropy = diagnostic (has SIGReg's target "
                            "converged?); decision-grade admissibility needs the final ckpt")
    out = args.out or (Path(args.ckpt).parent / f"orth_step{step}.json")
    Path(out).write_text(json.dumps(d, indent=2, default=str))
    for k in ("n_samples", "state_dim", "active_k", "iso_ratio_global", "iso_ratio_active",
              "cond_number_active", "rms_offdiag_corr", "cov_effective_rank",
              "participation_ratio_global"):
        print(f"  {k}: {d[k]}")
    print("  VERDICT:", d["verdict"])
    print(f"[orth] full report -> {out}")


if __name__ == "__main__":
    main()
