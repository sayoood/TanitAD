"""flagship v1.5 evaluation — the TanitEval protocol, run off-pod.

v1.5 has its OWN trajectory decoder (like REF-C), not a grounded operative
rollout, so it needs its own ``collect`` — but everything downstream of collect
must be the code that produced 0.4522 (v1) and 0.458 (REF-C-XL FINAL), or the
comparison is not a comparison. This script therefore VENDORS the eval pod's
own modules (``taniteval/bench.py``, ``taniteval/tanitad_metrics.py``,
``scripts/driving_diagnostic.py``) rather than the repo's copies, which have
drifted, and calls ``bench.run`` unmodified: same 8-split episode-disjoint
interval protocol, same ``val_frac`` 0.2, same CV baseline, same strata.

Window protocol, copied from ``taniteval/refc_eval.py::collect``:
    first 40 val episodes (``sorted(glob('ep_*.pt'))[:40]``), window 8,
    stride 8, ``starts = range(0, T - window - K_MAX, stride)``,
    ``last = t + window - 1``, waypoints at steps 5/10/15/20.

``--vendor`` must point at a directory holding the eval pod's
``driving_diagnostic.py`` + a ``taniteval`` package (bench, tanitad_metrics,
rollout). Fails loud if they are missing — a silently-substituted local metric
is exactly the failure this guards against.

Usage (pod2):
  PYTHONPATH=/workspace/TanitAD/stack python3 eval_flagship_v15.py \
    --ckpt /workspace/experiments/flagship-v15-abc/ckpt_best.pt \
    --states-val /workspace/v15/states_val.pt \
    --poses-val /workspace/v15/poses_val.pt \
    --labels-val /workspace/v15/labels_val.pt \
    --val-cache <epcache>/physicalai-val-0c5f7dac3b11 \
    --trunk /workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt \
    --anchors /workspace/v15/anchors256.pt --probes /workspace/v15/probes8.pt \
    --vendor /workspace/v15/evalsrc --key flagship-v15-abc \
    --out /workspace/v15/results/flagship-v15-abc.json
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
    """Import the eval pod's metric stack. Fails loud rather than falling back."""
    p = Path(path)
    need = [p / "driving_diagnostic.py", p / "taniteval" / "bench.py",
            p / "taniteval" / "tanitad_metrics.py"]
    missing = [str(x) for x in need if not x.exists()]
    if missing:
        raise SystemExit(
            "REFUSING to evaluate with substituted metrics — missing vendored "
            f"eval sources: {missing}. Copy them from tanitad-eval "
            "(/root/taniteval/taniteval/, /root/TanitAD/stack/scripts/"
            "driving_diagnostic.py) so the numbers stay comparable to the "
            "published rows.")
    sys.path.insert(0, str(p))
    import driving_diagnostic as dd                     # noqa: E402
    from taniteval import bench                         # noqa: E402
    return dd, bench


@torch.no_grad()
def collect(head, predictor, probes, cfg, states, poses, labels, eids,
            device, steps=None, batch=64, episodes=N_EPISODES, dd=None):
    """Predict the WP_STEPS waypoints for every TanitEval window."""
    from tanitad.models.flagship_v15 import SPEED_SCALE, imagine_probes
    k_max = max(dd.WP_STEPS)
    P, G, C, EID, SPD, HDG = [], [], [], [], [], []
    head.eval()
    for e in range(min(episodes, len(states))):
        st_ep = torch.as_tensor(states[e])
        po = torch.as_tensor(poses[e], dtype=torch.float32)
        t_len = po.shape[0]
        starts = list(range(0, t_len - WINDOW - k_max, STRIDE))
        for b0 in range(0, len(starts), batch):
            ch = starts[b0:b0 + batch]
            last = torch.tensor([t + WINDOW - 1 for t in ch])
            st = torch.stack([st_ep[t:t + WINDOW] for t in ch]).float().to(device)
            v0 = po[last, 3].to(device)
            ac = torch.stack([labels["actions"][e][t:t + WINDOW] for t in ch]
                             ).to(device)
            ac = torch.cat([ac, (v0 / SPEED_SCALE)[:, None, None]
                            .expand(-1, WINDOW, 1)], dim=-1)
            vb = labels["vt_band"][e][torch.tensor(ch)].to(device)
            rt = labels["route"][e][torch.tensor(ch)].to(device)
            rg = labels["route_graded"][e][torch.tensor(ch)].to(device)
            # vt_speed feeds the LONGITUDINAL SELECTION term. Training optimises
            # the score WITH it, so omitting it here would silently evaluate a
            # different ranker than the one that was trained.
            vs = labels["vt_speed"][e][torch.tensor(ch)].to(device)
            imag = None
            if cfg.cond_imagination:
                imag = imagine_probes(predictor, st, ac, probes, cfg.imag_read,
                                      v0 / SPEED_SCALE)
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
            "method": ("flagship-v1.5: FROZEN v1 trunk (encoder+predictor) + "
                       "REF-C anchored-diffusion head, argmax-conf anchor "
                       f"trajectory, steps={steps}, "
                       f"{head.decoder.anchors.shape[0]} anchors, cond="
                       f"{'a' if cfg.cond_states else ''}"
                       f"{'b' if cfg.cond_imagination else ''}"
                       f"{'c' if cfg.cond_vtarget else ''}")}


def real_episode_ids(val_cache: str, n: int) -> list[int]:
    """The episode_id ints the eval pod's interval estimator clusters on.

    The split is ``split_by_episode`` over these ids; using file indices instead
    would produce a DIFFERENT episode partition and therefore a different
    heldout mean. mmap keeps the 117 MB frame tensors off the heap.
    """
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--states-val", required=True)
    ap.add_argument("--poses-val", required=True)
    ap.add_argument("--labels-val", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--vendor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="flagship-v15")
    ap.add_argument("--label-set", choices=("v21", "legacy"), default="v21")
    ap.add_argument("--steps", type=int, default=None,
                    help="decoder mode: omit = trained truncated denoise, "
                         "0 = the classifier floor")
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    dd, bench = _vendor(a.vendor)
    from tanitad.models.flagship_v15 import FlagshipV15Head, V15Config
    from v15_prep import load_frozen_v1

    trunk, _g, _s = load_frozen_v1(a.trunk, a.device)
    predictor = trunk.predictor
    del trunk.encoder
    torch.cuda.empty_cache()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = V15Config(**{k: (tuple(v) if isinstance(v, list) else v)
                       for k, v in ck["cfg"].items() if k != "decoder"})
    from tanitad.refs.refc import DecoderConfig
    cfg.decoder = DecoderConfig(**ck["cfg"]["decoder"])
    head = FlagshipV15Head(cfg).to(a.device)
    head.load_state_dict(ck["head"])                      # STRICT
    print(f"[v15] loaded head step={ck.get('step')} cond="
          f"{cfg.cond_states}/{cfg.cond_imagination}/{cfg.cond_vtarget}",
          flush=True)

    prb = torch.load(a.probes, weights_only=False)
    probes = (prb["probes"] if isinstance(prb, dict) else prb).to(a.device)

    sd = torch.load(a.states_val, weights_only=False)
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

    data = collect(head, predictor, probes, cfg, sd["states"], pdta["poses"],
                   labels, eids, a.device, steps=a.steps,
                   episodes=a.episodes, dd=dd)
    res = bench.run(data)
    res["key"] = a.key
    res["method"] = data["method"]
    res["ckpt"] = a.ckpt
    res["trunk"] = a.trunk
    res["label_set"] = a.label_set
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    # Persist the per-window tensors next to the JSON, exactly as the eval pod's
    # runner does (results/windows_<key>.pt). WITHOUT this, no arm evaluated by
    # this script can ever be PAIRED against another — which is why the v1.5
    # a->ab imagination delta had to be combined in quadrature (invalid: the
    # arms are not independent) instead of paired. ~96 KB. 360-review W1.
    wp = Path(a.out).parent / f"windows_{a.key}.pt"
    torch.save({k: data[k] for k in
                ("pred", "gt", "cv", "eid", "speed", "head_deg", "wp_steps")
                if k in data}, wp)
    print(f"[windows] {wp} (per-window pred/gt/cv/eid — enables paired "
          f"episode-clustered tests)", flush=True)
    m = res["heldout"]["model"]
    print(json.dumps({
        "key": a.key, "n_windows": res["n_windows"],
        "ade@2s_heldout": m["ade@2s"], "ade@2s_full": res["full_set"]["model"]["ade@2s"],
        "fde@2s": m["fde@2s"], "miss@2m": m["miss_rate@2m"],
        "beats_cv": res["beats_cv_ade_0_2s"],
        # Gate constants read from Project Steering/MODEL_REGISTRY.md.
        # G1 moved on 2026-07-20: REF-C-XL FINISHED at step 29,999 and its
        # FINAL score is 0.458 (the 0.470 in circulation is the 28k provisional,
        # and 0.5645 is the ~16k snapshot). Always gate against the final.
        "G1_beat_refc_xl_final_0.458": bool(m["ade@2s"]["mean"] < 0.458),
        "G2_beat_v1_0.4522": bool(m["ade@2s"]["mean"] < 0.4522),
        "G3_miss_le_0.10": bool(m["miss_rate@2m"]["mean"] <= 0.10),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
