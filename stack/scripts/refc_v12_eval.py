"""REF-C v1.2 — TanitEval-compatible evaluation of the learned re-scorer.

Decodes the FROZEN refc-xl-30k exactly as ``taniteval.refc_eval.collect`` does
(window 8, stride 8, ``model.eval()`` deterministic 2-step denoise, ``v0`` fed,
``nav_cmd=None`` -> `follow`, waypoints read at the shared WP_STEPS) and changes
exactly ONE thing: the selection ``argmax`` runs over the LEARNED score instead
of the frozen t=0 anchor logits. Every other quantity — episodes, GT frame, CV
baseline, strata, the 8-split episode-disjoint jackknife in ``bench.run`` — is
the harness's, so ``refc-v12`` is directly comparable to ``refc-xl-30k``'s
0.458 row.

It ALSO emits the G3 mechanism read on the same 881 windows, which the standard
benchmark row cannot express:

  ``base_ade``          ADE of the plan the FROZEN score picks           (before)
  ``sel_ade``           ADE of the plan the re-scorer picks              (after)
  ``refined_conf_ade``  ADE if REF-C had simply selected on its DISCARDED
                        refined-pass confidence — the free, training-free
                        control that isolates "use the refined logits" from
                        "learn a ranker"
  ``vocab_ade``         best RAW anchor (the quantisation floor, 0.599)
  ``oracle_ade``        best REFINED proposal — GT-informed and UNREACHABLE
  ``*_gap`` / ``*_2x``  the ranking deficit and the >2x-worse-than-oracle rate

Usage (eval pod):
  PYTHONPATH=/root/taniteval:/root/TanitAD/stack python3 \
    /root/TanitAD/stack/scripts/refc_v12_eval.py \
      --head /root/models/refc-v12/head.pt --episodes 40 --tag refc-v12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

for _p in ("/root/taniteval", "/root/TanitAD/stack",
           "/root/TanitAD/stack/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.models.refc_rescorer import (RefCRescorer,  # noqa: E402
                                          RescorerConfig, fan_ade_axes,
                                          fan_ade_from, refc_forward_fan,
                                          select_q)
from refc_v12_cache import load_frozen  # noqa: E402


def paired_delta(base_w: torch.Tensor, sel_w: torch.Tensor, eid: list,
                 n_boot: int = 4000, seed: int = 0) -> dict:
    """Episode-clustered PAIRED bootstrap of ``base - sel`` (positive = better).

    Why paired: v1.2 and the baseline are evaluated on the SAME 881 windows
    through the SAME frozen decoder — only the selection differs. The harness's
    unpaired jackknife CI95 on ADE@2s is ±0.057 m, so an unpaired comparison
    cannot resolve anything smaller than the entire effect we are hunting; the
    per-window difference has vastly more power. Resampling EPISODES (not
    windows) keeps the within-episode correlation honest.
    """
    d = (base_w - sel_w).double()
    eids = sorted(set(eid))
    idx_by_ep = {e: torch.tensor([i for i, x in enumerate(eid) if x == e])
                 for e in eids}
    g = torch.Generator().manual_seed(seed)
    means = []
    for _ in range(n_boot):
        pick = torch.randint(len(eids), (len(eids),), generator=g)
        sel = torch.cat([idx_by_ep[eids[int(p)]] for p in pick])
        means.append(float(d[sel].mean()))
    means = torch.tensor(means)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975])).tolist()
    return {"mean_delta_m": round(float(d.mean()), 5),
            "ci95_lo": round(lo, 5), "ci95_hi": round(hi, 5),
            "significant": bool(lo > 0.0),
            "frac_windows_improved": round(float((d > 1e-9).float().mean()), 4),
            "frac_windows_worsened": round(float((d < -1e-9).float().mean()), 4),
            "frac_windows_unchanged": round(float((d.abs() <= 1e-9).float()
                                                  .mean()), 4),
            "n_episodes": len(eids), "n_boot": n_boot}


def load_head(path: str, device: str) -> tuple[RefCRescorer, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = RescorerConfig(**ck["cfg"])
    head = RefCRescorer(cfg)
    head.load_state_dict(ck["head"])
    return head.to(device).eval(), ck


@torch.no_grad()
def collect(model, head, episodes, device, wp_steps, stride=8, batch=8,
            target="soft") -> dict:
    """Same return contract as ``taniteval.refc_eval.collect`` (+ ``mech``)."""
    from driving_diagnostic import (baseline_waypoints, gt_ego_waypoints,
                                    net_heading_change_deg)
    window = int(model.cfg.window)
    k_max = max(wp_steps)
    horizons = tuple(model.cfg.trajectory.horizons)
    assert horizons == tuple(wp_steps), \
        f"REF-C horizons {horizons} != eval WP_STEPS {tuple(wp_steps)}"
    raw_anchors = model.decoder.anchors.float()                # [N, S, 2]

    S_wp, GT, CV, EID, SPD, HDG = [], [], [], [], [], []
    SEL_W, BASE_W = [], []                 # per-window ADE, for the paired test
    acc = {k: 0.0 for k in ("sel_ade", "base_ade", "refined_conf_ade",
                            "oracle_ade", "oracle_k_ade", "vocab_ade",
                            "chance_ade", "chance_k_ade",
                            "sel_2x", "base_2x", "rank_acc",
                            "sel_along", "sel_cross", "base_along",
                            "base_cross")}
    n_tot = 0
    for ep in episodes:
        fr = ep.feats
        T = fr.shape[0]
        starts = list(range(0, T - window - k_max, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(fr[t:t + window])
                              for t in ch]).to(device).float().div_(255.0)
            v0 = ep.poses[last, 3].to(device).float()
            o = refc_forward_fan(model, fw, nav_cmd=None, v0=v0)
            fan = o["anchor_traj"].float()                     # [b, N, S, 2]
            base = o["anchor_logits"].float()
            ho = head(select_q(o, head.cfg.q_source), base, fan,
                      o["pooled"], o["cond"], v0, target=target)
            b = fan.shape[0]
            ar = torch.arange(b, device=device)
            idx = ho["sel_idx"]                    # GLOBAL anchor index
            traj = fan[ar, idx]
            S_wp.append(traj.cpu())
            gt = gt_ego_waypoints(ep.poses, last)
            GT.append(gt)
            CV.append(baseline_waypoints(ep.poses, last)["constant_velocity"])
            EID.extend([ep.episode_id] * b)
            SPD.append(ep.poses[last, 3])
            HDG.append(net_heading_change_deg(ep.poses, last))

            # ---- mechanism (G3), on these exact windows -------------------
            g = gt.to(device).float()
            fa = fan_ade_from(fan, g)                           # [b, N]
            along, cross = fan_ade_axes(fan, g)
            va = fan_ade_from(raw_anchors[None].expand(b, *raw_anchors.shape),
                              g)
            orc = fa.min(dim=1).values
            bidx = base.argmax(dim=1)
            sel = fa[ar, idx]
            bse = fa[ar, bidx]
            rfc = fa[ar, o["refined_conf"].float().argmax(dim=1)]
            acc["sel_ade"] += float(sel.sum())
            acc["base_ade"] += float(bse.sum())
            acc["refined_conf_ade"] += float(rfc.sum())
            acc["oracle_ade"] += float(orc.sum())
            k_ade = fa.gather(1, ho["topk_idx"])
            acc["oracle_k_ade"] += float(k_ade.min(dim=1).values.sum())
            # chance floors: a no-skill ranker over the fan / inside the top-K
            acc["chance_ade"] += float(fa.mean(dim=1).sum())
            acc["chance_k_ade"] += float(k_ade.mean(dim=1).sum())
            acc["vocab_ade"] += float(va.min(dim=1).values.sum())
            acc["sel_2x"] += float((sel > 2 * orc).float().sum())
            acc["base_2x"] += float((bse > 2 * orc).float().sum())
            acc["rank_acc"] += float((idx == fa.argmin(dim=1)).float().sum())
            acc["sel_along"] += float(along[ar, idx].sum())
            acc["sel_cross"] += float(cross[ar, idx].sum())
            acc["base_along"] += float(along[ar, bidx].sum())
            acc["base_cross"] += float(cross[ar, bidx].sum())
            SEL_W.append(sel.cpu())
            BASE_W.append(bse.cpu())
            n_tot += b

    mech = {k: round(v / max(n_tot, 1), 5) for k, v in acc.items()}
    mech["sel_gap"] = round(mech["sel_ade"] - mech["oracle_ade"], 5)
    mech["base_gap"] = round(mech["base_ade"] - mech["oracle_ade"], 5)
    mech["sel_gap_k"] = round(mech["sel_ade"] - mech["oracle_k_ade"], 5)
    mech["gap_recovered"] = round(
        (mech["base_ade"] - mech["sel_ade"]) / max(mech["base_gap"], 1e-9), 5)
    # fraction of the chance -> oracle span captured INSIDE the top-K: the
    # incumbent's real strength, and the room a re-scorer actually has
    span = mech["chance_k_ade"] - mech["oracle_k_ade"]
    mech["base_span_k"] = round((mech["chance_k_ade"] - mech["base_ade"])
                                / max(span, 1e-9), 5)
    mech["sel_span_k"] = round((mech["chance_k_ade"] - mech["sel_ade"])
                               / max(span, 1e-9), 5)
    mech["topk"] = int(head.cfg.topk)
    mech["n_windows"] = n_tot
    mech["paired"] = paired_delta(torch.cat(BASE_W), torch.cat(SEL_W), EID)
    return {"pred": torch.cat(S_wp).float(), "gt": torch.cat(GT).float(),
            "cv": torch.cat(CV).float(), "eid": EID,
            "speed": torch.cat(SPD).float(),
            "head_deg": torch.cat(HDG).float(), "wp_steps": list(wp_steps),
            "mech": mech,
            "method": ("refc v1.2 — frozen refc-xl-30k anchored-diffusion "
                       "decode (eval-mode, 2 denoise steps, 256 anchors, "
                       "nav=follow), selection by the LEARNED re-scorer over "
                       "the REFINED fan (head-only, frozen decoder)")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--ckpt", default="/root/models/refc-xl-30k/ckpt.pt")
    ap.add_argument("--config", default="xl")
    ap.add_argument("--val", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--tag", default="refc-v12")
    ap.add_argument("--results", default="/root/taniteval/results")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args(argv)

    from taniteval import bench, data
    from driving_diagnostic import WP_STEPS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model, cfg, step = load_frozen(args.ckpt, args.config, None, device)
    head, hck = load_head(args.head, device)
    files = data.list_val_episodes(args.val, args.episodes)
    assert files, f"no val episodes under {args.val}"
    eps = data.load_frames(files)
    win = collect(model, head, eps, device, WP_STEPS, batch=args.batch,
                  target=hck.get("target", "soft"))
    res = bench.run(win)                       # 8-split episode-disjoint jack
    res["method"] = win["method"]
    res["mechanism"] = win["mech"]
    res["model"] = {"key": args.tag, "name": "REF-C v1.2 (learned re-scorer)",
                    "arch": "refc+rescorer", "encoder": "frozen refc-xl-30k",
                    "speed_input": True,
                    "rescorer": {"target": hck.get("target"),
                                 "tau": hck.get("tau"),
                                 "margin_scale": hck.get("margin_scale"),
                                 "head_step": hck.get("step"),
                                 "dev": hck.get("dev")}}
    res["ckpt_step"] = step
    res["wall_s"] = round(time.time() - t0, 1)
    out = Path(args.results) / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    torch.save({k: win[k] for k in ("pred", "gt", "cv", "eid", "speed",
                                    "head_deg", "wp_steps")},
               Path(args.results) / f"windows_{args.tag}.pt")
    print(json.dumps({"tag": args.tag, "mechanism": win["mech"],
                      "heldout": res.get("heldout", {}).get("model", {}),
                      "full_set": res.get("full_set", res.get("model")),
                      "wall_s": res["wall_s"]}, indent=2, default=str),
          flush=True)
    return res


if __name__ == "__main__":
    main()
