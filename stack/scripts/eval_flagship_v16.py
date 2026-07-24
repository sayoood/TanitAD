"""flagship v1.6 evaluation — the TanitEval protocol, RE-ENCODING val frames.

v1.6 unfreezes the trunk, so the cached ``states_val.pt`` (built from the FROZEN
v1 encoder by v15_prep) are STALE — evaluating v1.6 on them would score the wrong
encoder. This script therefore differs from ``eval_flagship_v15.py`` in exactly
one way: it loads the FULL v1.6 world model from the checkpoint (encoder +
readout + predictor, as fine-tuned) and RE-ENCODES the val frames through it,
instead of reading a cached-states file. Everything downstream of ``collect`` is
still the eval pod's own vendored ``taniteval.bench`` — the same 8-split
episode-holdout interval protocol, ``val_frac`` 0.2, CV baseline and strata that
produced 0.4522 (v1), 0.458 (REF-C-XL) and 0.5437 (v1.5 ab) — so the number is
comparable to those rows.

Both the trunk (encoder+predictor for imagination) and the head come from the
SAME v1.6 checkpoint (``ck['model']`` + ``ck['head']``) — a frozen-trunk eval
would silently use the wrong predictor for the imagination conditioning.

Usage (pod2):
  PYTHONPATH=/workspace/TanitAD/stack python3 eval_flagship_v16.py \
    --ckpt /workspace/experiments/flagship-v16-ab-ft/ckpt_best.pt \
    --poses-val /workspace/v15/poses_val.pt \
    --labels-val /workspace/v15/labels_val.pt \
    --val-cache /workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11 \
    --anchors /workspace/v15/anchors256.pt --probes /workspace/v15/probes8.pt \
    --vendor /workspace/v15/evalsrc --key flagship-v16-ab-ft \
    --out /workspace/v15/results/flagship-v16-ab-ft.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

WINDOW = 8
STRIDE = 8
N_EPISODES = 40


def _vendor(path: str):
    p = Path(path)
    need = [p / "driving_diagnostic.py", p / "taniteval" / "bench.py",
            p / "taniteval" / "tanitad_metrics.py"]
    missing = [str(x) for x in need if not x.exists()]
    if missing:
        raise SystemExit(
            "REFUSING to evaluate with substituted metrics — missing vendored "
            f"eval sources: {missing}. Copy them from tanitad-eval so the "
            "numbers stay comparable to the published rows.")
    sys.path.insert(0, str(p))
    import driving_diagnostic as dd                     # noqa: E402
    from taniteval import bench                         # noqa: E402
    return dd, bench


def _load_v16(ckpt: str, device: str, trunk: str | None = None):
    """Rebuild the flagship WorldModel + grounding and load the v1.6 (unfrozen)
    weights. Same config surface as load_frozen_v1, but the source is the v1.6
    checkpoint, not the original frozen trunk.

    CONTROL PATH: a v1.5 head-only checkpoint (`ab`) has no ``ck['model']``. Pass
    ``--trunk`` and it is scored through this SAME re-encoding harness on the
    FROZEN v1 trunk — that is the like-for-like control, and it doubles as a
    validation that the frames path reproduces the published cached-states number
    (`ab` = 0.5437 heldout / 0.5366 full-set).
    """
    from tanitad.config import flagship4b_config
    from tanitad.models.fourbrain import WorldModel
    from tanitad.models.metric_dynamics import HierarchicalGrounding
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "model" not in ck:
        if not trunk:
            raise SystemExit(
                f"{ckpt} has no 'model' key (it is a head-only v1.5 ckpt). Pass "
                "--trunk <frozen v1 ckpt> to score it as the FROZEN control "
                "through this harness.")
        from v15_prep import load_frozen_v1
        world, grounding, _ = load_frozen_v1(trunk, device)
        object.__setattr__(world.encoder.cfg, "grad_checkpoint", False)
        print(f"[v16] CONTROL mode: frozen trunk {trunk} + head {ckpt}",
              flush=True)
        return world, grounding, ck
    cfg = flagship4b_config()
    object.__setattr__(cfg.predictor, "action_dim", 3)
    if cfg.tactical_pred is not None:
        object.__setattr__(cfg.tactical_pred, "action_dim", 3)
    object.__setattr__(cfg.encoder, "grad_checkpoint", False)
    world = WorldModel(cfg)
    world.load_state_dict(ck["model"])                   # STRICT (unfrozen trunk)
    world = world.to(device).eval()
    for p in world.parameters():
        p.requires_grad_(False)
    grounding = None
    if "grounding" in ck:
        grounding = HierarchicalGrounding(world.state_dim).to(device).eval()
        grounding.load_state_dict(ck["grounding"])
    return world, grounding, ck


@torch.no_grad()
def collect(head, world, probes, cfg, poses, labels, eids, val_cache, device,
            steps=None, batch=16, episodes=N_EPISODES, dd=None):
    """Predict the WP_STEPS waypoints for every TanitEval window, RE-ENCODING the
    window frames through the v1.6 trunk."""
    from tanitad.models.flagship_v15 import SPEED_SCALE, imagine_probes
    k_max = max(dd.WP_STEPS)
    P, G, C, EID, SPD, HDG = [], [], [], [], [], []
    head.eval()
    files = sorted(f for f in os.listdir(val_cache)
                   if f.startswith("ep_") and f.endswith(".pt"))[:episodes]
    for e in range(min(episodes, len(poses))):
        po = torch.as_tensor(poses[e], dtype=torch.float32)
        t_len = po.shape[0]
        d = torch.load(os.path.join(val_cache, files[e]), map_location="cpu",
                       weights_only=True, mmap=True)
        frames_ep = d["frames_u8"]                       # [T, 9, 256, 256] u8
        starts = list(range(0, t_len - WINDOW - k_max, STRIDE))
        for b0 in range(0, len(starts), batch):
            ch = starts[b0:b0 + batch]
            last = torch.tensor([t + WINDOW - 1 for t in ch])
            fw = torch.stack([frames_ep[t:t + WINDOW].clone() for t in ch]
                             ).to(device).float().div_(255.0)
            b, w = fw.shape[:2]
            st = world.encode(fw.reshape(b * w, *fw.shape[2:])).reshape(b, w, -1)
            v0 = po[last, 3].to(device)
            ac = torch.stack([labels["actions"][e][t:t + WINDOW] for t in ch]
                             ).to(device)
            ac = torch.cat([ac, (v0 / SPEED_SCALE)[:, None, None]
                            .expand(-1, WINDOW, 1)], dim=-1)
            vb = labels["vt_band"][e][torch.tensor(ch)].to(device)
            rt = labels["route"][e][torch.tensor(ch)].to(device)
            rg = labels["route_graded"][e][torch.tensor(ch)].to(device)
            vs = labels["vt_speed"][e][torch.tensor(ch)].to(device)
            imag = None
            if cfg.cond_imagination:
                imag = imagine_probes(world.predictor, st, ac, probes,
                                      cfg.imag_read, v0 / SPEED_SCALE)
            out = head(st, v0, imagined=imag, vt_band=vb, route=rt,
                       route_graded=rg, vt_speed=vs, steps=steps)
            P.append(out["traj"].float().cpu())
            G.append(dd.gt_ego_waypoints(po, last))
            C.append(dd.baseline_waypoints(po, last)["constant_velocity"])
            EID.extend([eids[e]] * len(ch))
            SPD.append(po[last, 3])
            HDG.append(dd.net_heading_change_deg(po, last))
    return {"pred": torch.cat(P), "gt": torch.cat(G).float(),
            "cv": torch.cat(C).float(), "eid": EID,
            "speed": torch.cat(SPD).float(),
            "head_deg": torch.cat(HDG).float(),
            "wp_steps": list(dd.WP_STEPS),
            "method": ("flagship-v1.6: LP-FT UNFROZEN v1 trunk (re-encoded) + "
                       "REF-C anchored-diffusion head, argmax-conf anchor traj, "
                       f"steps={steps}, {head.decoder.anchors.shape[0]} anchors, "
                       f"cond={'a' if cfg.cond_states else ''}"
                       f"{'b' if cfg.cond_imagination else ''}"
                       f"{'c' if cfg.cond_vtarget else ''}")}


def real_episode_ids(val_cache: str, n: int) -> list[int]:
    files = sorted(f for f in os.listdir(val_cache) if f.startswith("ep_")
                   and f.endswith(".pt"))[:n]
    out = []
    for f in files:
        d = torch.load(os.path.join(val_cache, f), map_location="cpu",
                       weights_only=True, mmap=True)
        out.append(int(d["episode_id"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="v1.6 ckpt (model+head+grounding)")
    ap.add_argument("--poses-val", required=True)
    ap.add_argument("--labels-val", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--vendor", required=True)
    ap.add_argument("--trunk", default=None,
                    help="CONTROL mode: score a head-only v1.5 ckpt (ab) on the "
                         "FROZEN v1 trunk through this same re-encoding harness")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="flagship-v16")
    ap.add_argument("--label-set", choices=("v21", "legacy"), default="v21")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    dd, bench = _vendor(a.vendor)
    from tanitad.models.flagship_v15 import FlagshipV15Head, V15Config
    from tanitad.refs.refc import DecoderConfig

    world, grounding, ck = _load_v16(a.ckpt, a.device, a.trunk)
    cfg = V15Config(**{k: (tuple(v) if isinstance(v, list) else v)
                       for k, v in ck["cfg"].items() if k != "decoder"})
    cfg.decoder = DecoderConfig(**ck["cfg"]["decoder"])
    head = FlagshipV15Head(cfg).to(a.device)
    head.load_state_dict(ck["head"])                     # STRICT
    print(f"[v16] loaded head+trunk step={ck.get('step')} unfreeze="
          f"{ck.get('unfreeze')} cond="
          f"{cfg.cond_states}/{cfg.cond_imagination}/{cfg.cond_vtarget}",
          flush=True)

    prb = torch.load(a.probes, weights_only=False)
    probes = (prb["probes"] if isinstance(prb, dict) else prb).to(a.device)

    pdta = torch.load(a.poses_val, weights_only=False)
    ld = torch.load(a.labels_val, weights_only=False)
    vt_key = "vt_band_v2" if a.label_set == "v21" else "vt_band_raw"
    labels = {"actions": [torch.as_tensor(x, dtype=torch.float32)
                          for x in pdta["actions"]],
              "vt_band": ld[vt_key],
              "vt_speed": ld["vt_v2" if a.label_set == "v21" else "vt_raw"],
              "route": ld["route_v21"] if a.label_set == "v21"
              else ld["route_legacy"],
              "route_graded": ld["route_graded"] if a.label_set == "v21"
              else [torch.zeros_like(x, dtype=torch.float32)
                    for x in ld["route_legacy"]]}
    eids = real_episode_ids(a.val_cache, a.episodes)

    data = collect(head, world, probes, cfg, pdta["poses"], labels, eids,
                   a.val_cache, a.device, steps=a.steps, episodes=a.episodes,
                   batch=16, dd=dd)
    res = bench.run(data)
    res.update({"key": a.key, "method": data["method"], "ckpt": a.ckpt,
                "label_set": a.label_set, "unfreeze": ck.get("unfreeze")})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    # Per-window tensors alongside the JSON (see eval_flagship_v15.py) — without
    # them an arm can never be PAIRED against another, and paired
    # episode-clustered tests are ~2.6x more powerful on trunk-sharing arms
    # (measured, taniteval/recompute_ci.py). 360-review W1.
    wp = Path(a.out).parent / f"windows_{a.key}.pt"
    torch.save({k: data[k] for k in
                ("pred", "gt", "cv", "eid", "speed", "head_deg", "wp_steps")
                if k in data}, wp)
    print(f"[windows] {wp} (enables paired episode-clustered tests)", flush=True)
    m = res["heldout"]["model"]
    print(json.dumps({
        "key": a.key, "n_windows": res["n_windows"],
        "ade@2s_heldout": m["ade@2s"],
        "ade@2s_full": res["full_set"]["model"]["ade@2s"],
        "fde@2s": m["fde@2s"], "miss@2m": m["miss_rate@2m"],
        "beats_cv": res["beats_cv_ade_0_2s"],
        "G1_beat_refc_0.458": bool(m["ade@2s"]["mean"] < 0.458),
        "G2_beat_v1_0.4522": bool(m["ade@2s"]["mean"] < 0.4522),
        "G3_miss_le_0.10": bool(m["miss_rate@2m"]["mean"] <= 0.10),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
