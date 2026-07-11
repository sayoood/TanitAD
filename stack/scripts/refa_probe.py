"""REF-A gate-matched probe: ADE@1s decodability of the trained REF-A latents.

Runs the SAME probe protocol as scripts/d1_probe_capacity.py (ridge at 3
regularizations + small MLP, episode-parity route split, ADE@1s) on the
REF-A adapter states, so the number lands directly beside the main-model
rows (14k / 23.5k) in one table.

Comparability caveat (recorded in the output): REF-A features exist for the
comma corpus only, so this probe is comma-val-only, while the main-model
d1_probe_capacity rows mixed comma + PhysicalAI val episodes. The clean
apples-to-apples companion is a comma-only re-run of d1_probe_capacity on
pod1 after the 30k record run frees the GPU (same --cache-dirs minus
physicalai).

Usage (pod2):
  python scripts/refa_probe.py \
      --ckpt /workspace/experiments/refa-30k/ckpt.pt \
      --feat-dir /opt/dino_feats/comma2k19-val-*-dinov2-b14 \
      --out /workspace/experiments/refa_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_probe_capacity import STEPS_1S, _ego, fit_eval  # noqa: E402

from tanitad.refs.refa import RefAModel  # noqa: E402


@torch.no_grad()
def collect(model: RefAModel, files, device, window, stride=8, batch=64):
    S, Y, E = [], [], []
    for ei, f in enumerate(files):
        ep = torch.load(f, map_location="cpu", weights_only=True)
        feats, poses = ep["feats_fp16"], ep["poses"]
        T = feats.shape[0]
        starts = list(range(0, T - window - STEPS_1S, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            fw = torch.stack([feats[t:t + window] for t in ch]).to(device)
            st = model.encode_window(fw)[:, -1].cpu()
            last = torch.tensor([t + window - 1 for t in ch])
            wp = _ego(poses[last + STEPS_1S, :2] - poses[last, :2],
                      poses[last, 2])
            S.append(st); Y.append(wp); E.extend([ei] * len(ch))
    return torch.cat(S), torch.cat(Y), torch.tensor(E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--episodes", type=int, default=12)   # match d1_probe count
    ap.add_argument("--adapter", choices=("pool", "grid"), default="pool",
                    help="must match the checkpoint's adapter kind")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RefAModel(adapter_kind=args.adapter)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    step = int(ck.get("step", -1))
    model = model.to(device).eval()

    files = sorted(Path(args.feat_dir).glob("ep_*.pt"))[:args.episodes]
    if not files:
        raise SystemExit(f"no ep_*.pt under {args.feat_dir}")
    S, Y, E = collect(model, files, device, model.pred_cfg.window)

    row = {}
    for kind, a in (("ridge", 1.0), ("ridge", 10.0), ("ridge", 100.0),
                    ("mlp", None)):
        key = f"{kind}{'' if a is None else f'_a{a:g}'}"
        row[key] = fit_eval(S.float(), Y.float(), E, kind, alpha=a or 10.0)
    report = {
        "exp": "refa-probe",
        "protocol": "d1_probe_capacity (ridge x3 + MLP, episode-parity split, ADE@1s)",
        "caveat": "comma-val-only (REF-A features cover comma corpus only); "
                  "main-model rows mixed comma+physicalai — comma-only main "
                  "re-run queued for apples-to-apples",
        "n_episodes": len(files),
        "ade_1s_by_probe": {f"refa_step{step}": row},
    }
    print(json.dumps(report), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("REFA_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
