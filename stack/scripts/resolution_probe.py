"""Resolution-sensitivity probe (H17/UNIFIED_FOV plan). Reuses driving_diagnostic
machinery VERBATIM (no reinvention): _ego, gt_ego_waypoints, split_by_episode,
fit_predict, de_of. Only addition: degrade each window to an effective input
resolution R (downsample->upsample to native) BEFORE encoding, sweep R, and
report route-split ridge ADE@1s per curvature stratum + corpus.

Verdict logic: if ADE is FLAT from 256 down to 128 -> the model uses no detail
beyond ~128px -> resolution is NOT the binding constraint (more px won't help).
If ADE degrades sharply as R drops -> the model uses fine detail -> higher
resolution plausibly helps; far/curve strata should degrade most if so.
Sanity: R=256 (no degrade) must match the diagnostic held-out ridge_a10 ~5.3 m.

Usage (pod1):
  python resolution_probe.py --ckpt /workspace/ckpt27k_flagship.pt \
    --cache-dirs /workspace/data/comma2k19/_epcache /workspace/data/physicalai/_epcache \
    --out /workspace/experiments/resolution_probe.json --episodes 40
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import driving_diagnostic as D             # proven machinery
from tanitad.eval.gates import split_by_episode
from tanitad.data.mixing import load_episode
from tanitad.instruments.numerics import strict_numerics
from tanitad.config import base250cam_config
from tanitad.models.fourbrain import WorldModel

RES_SWEEP = [256, 192, 128, 96, 64]


def degrade(fw: torch.Tensor, R: int) -> torch.Tensor:
    """[B,W,C,H,Wi] -> same shape, but carrying only <=RxR of detail."""
    if R >= fw.shape[-1]:
        return fw
    b, w, c, h, wi = fw.shape
    x = fw.reshape(b * w, c, h, wi)
    x = F.interpolate(x, size=(R, R), mode="bilinear", align_corners=False)
    x = F.interpolate(x, size=(h, wi), mode="bilinear", align_corners=False)
    return x.reshape(b, w, c, h, wi)


@torch.no_grad()
def collect_res(world, episodes, corpora, device, window, R, stride=8, batch=16):
    S, GT, EID, COR, HDG = [], [], [], [], []
    for ep, corp in zip(episodes, corpora):
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 else ep.frames
        T = fr.shape[0]
        for i in range(0, len(range(0, T - window - D.K_MAX, stride)), batch):
            starts = list(range(0, T - window - D.K_MAX, stride))
            ch = starts[i:i + batch]
            if not ch:
                continue
            fw = torch.stack([fr[t:t + window] for t in ch]).to(device)
            fw = degrade(fw, R)
            st = world.encode_window(fw)[:, -1].cpu()
            last = torch.tensor([t + window - 1 for t in ch])
            S.append(st); GT.append(D.gt_ego_waypoints(ep.poses, last))
            EID.extend([ep.episode_id] * len(ch)); COR.extend([corp] * len(ch))
            HDG.append(D.net_heading_change_deg(ep.poses, last))
    return (torch.cat(S).float(), torch.cat(GT).float(), EID, COR,
            torch.cat(HDG).float())


def ade_route_split(states, gt, eid, mask=None):
    """route-split ridge(a=10) ADE@1s (ade_0_2s), mean over 8 splits."""
    idx = torch.arange(len(eid)) if mask is None else torch.nonzero(mask).squeeze(1)
    if len(idx) < 40:
        return None
    sub_eid = [eid[i] for i in idx.tolist()]
    ades = []
    for s in range(8):
        tr, va = split_by_episode(sub_eid, 0.2, s)
        tr, va = idx[tr], idx[va]
        pred, _ = D.fit_predict("ridge", 10.0, states[tr],
                                gt[tr].reshape(len(tr), -1), states[va], 0)
        ades.append(float(D.de_of(pred, gt[va]).mean()))
    return round(sum(ades) / len(ades), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eps, cors = [], []
    for cd in args.cache_dirs:
        corp = "physicalai" if "physicalai" in cd else "comma2k19"
        val = sorted(Path(cd).glob("*val*"))[-1]
        for p in sorted(val.glob("ep_*.pt"))[:args.episodes]:
            eps.append(load_episode(str(p), mmap=True)); cors.append(corp)

    world = WorldModel(base250cam_config())
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    world.load_state_dict(ck["model"] if "model" in ck else ck)
    world = world.to(device).eval()
    window = world.predictor.cfg.window

    report = {"exp": "resolution-sensitivity", "res_sweep": RES_SWEEP,
              "metric": "ridge_a10 ADE@1s (ade_0_2s), 8 route splits", "by_res": {}}
    with strict_numerics():
        for R in RES_SWEEP:
            S, GT, EID, COR, HDG = collect_res(world, eps, cors, device, window, R)
            corr = torch.tensor([c == "comma2k19" for c in COR])
            straight = HDG.abs() < 5.0
            row = {
                "overall": ade_route_split(S, GT, EID),
                "straight": ade_route_split(S, GT, EID, straight),
                "curve": ade_route_split(S, GT, EID, ~straight),
                "comma": ade_route_split(S, GT, EID, corr),
                "physicalai": ade_route_split(S, GT, EID, ~corr),
                "n": len(EID),
            }
            report["by_res"][str(R)] = row
            print(f"R={R}: {row}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("RESOLUTION_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
