"""TOLERANCE-BAND RE-SCORE — is the recovery-FT ADE-cost a knife-edge-metric artifact?

D2/RefcCL/LOWOOD-CL all found: recovery-FT reduces corridor departures but closed_ade2s
goes SEPARATED-worse. closed_ade2s is L2 to the EXACT GT path (knife-edge) — it penalizes
BENIGN in-lane recovery wiggle. This re-scores the SAME rollouts with a TOLERANCE-BAND
metric (deviation BEYOND a lane-tolerance band, not exact-path L2) — NO new training.

Reuses lowood_lanekeep.cl_realfootage: it already returns per-window `lat_traj` [N,K]
(per-step |XTE|) and `de` [N,4] (2D deviation at the 4 waypoints). We compute:
  band_ade2d(band)  = mean over the 4 waypoints of max(0, de - band)   (2D, the direct
                      tolerance-band analog of closed_ade2s)
  band_xte_ade(band)= mean over K steps of max(0, |XTE| - band)         (LATERAL only —
                      the direct test of "benign in-lane recovery")
  corridor_departure(band) = fraction of steps with |XTE| > band        (reproduced)
Paired base-vs-FT episode-cluster bootstrap on all, band grid {0.5, 1.0, 1.5} m.

PRE-REGISTERED (per the coordinator, both committed):
  WIN   for a config: corridor_departure still SEP-lower (dCDR CI∌0) AND the ADE penalty
        DISAPPEARS under the fair metric (d band_ade2d CI∋0 = within-noise of base) ->
        the "trade" was a knife-edge-ADE artifact; the lever IS promotable under a fair metric.
  BOUND departure ↓ but d band_ade2d STILL separated-worse -> the trade is REAL even under
        a fair metric -> the map/tolerance instrument + renderer paths are genuinely needed.

Run (eval): PYTHONPATH=...:/root/taniteval python3 tolerance_rescore.py \
  --base-ckpt /root/models/refc-base-30k/ckpt.pt \
  --arms naive:/workspace/refc-recovery-ft/ckpt.pt,g2:/workspace/refc-ft-g2/ckpt.pt,\
lowoodcl:/workspace/refc-lowood-cl/ckpt.pt --holdout-slice 28:40 --out tolerance_rescore.json
CPU smoke: python3 tolerance_rescore.py --smoke --out /tmp/tr_smoke.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_ROOTS = ["/root/TanitAD/stack", "/root/TanitAD/stack/scripts",
          "/workspace/TanitAD/stack", "/workspace/TanitAD/stack/scripts",
          "/root/taniteval", "/root/TanitAD/taniteval", "/workspace/refc_cl_d2",
          str(_HERE.parent), str(_HERE.parents[1] / "2026-07-23-refc-planner-closedloop")]
for _up in (5, 6):
    try:
        r = _HERE.parents[_up]
        _ROOTS += [str(r / "stack"), str(r / "stack" / "scripts"), str(r / "taniteval")]
    except IndexError:
        pass
for _p in _ROOTS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

BANDS = [0.5, 1.0, 1.5]
PRIMARY_BAND = 1.0
JUNCTION_DEG = 10.0


def _find_lk():
    for cand in (str(_HERE.parent), "/workspace/refc_cl_d2", "/root/TanitAD",
                 "/workspace/TanitAD"):
        if not Path(cand).exists():
            continue
        hits = list(Path(cand).rglob("lowood_lanekeep.py"))
        if hits:
            spec = importlib.util.spec_from_file_location("lowood_lanekeep", str(hits[0]))
            LK = importlib.util.module_from_spec(spec); spec.loader.exec_module(LK)
            return LK, str(hits[0])
    raise FileNotFoundError("lowood_lanekeep.py not found")


def band_metrics(pw, band):
    """From a cl_realfootage per-window dict -> per-window band arrays [N]."""
    lat = pw["lat_traj"].numpy()                 # [N,K] per-step |XTE|
    de = pw["de"].numpy()                        # [N,4] 2D dev at waypoints
    band_ade2d = np.maximum(0.0, de - band).mean(axis=1)
    band_xte = np.maximum(0.0, lat - band).mean(axis=1)
    dep = (lat > band).mean(axis=1)              # corridor_departure at this band
    return band_ade2d, band_xte, dep


def _paired(a, b, eid):
    from taniteval.ci import paired_episode_cluster_bootstrap
    return paired_episode_cluster_bootstrap(np.asarray(a, float), np.asarray(b, float),
                                            eid, n_boot=2000)


def rescore(LK, base_pw, ft_pw, band):
    eid = base_pw["eid"]
    assert eid == ft_pw["eid"], "arms not window-aligned"
    ba, bx, bd = band_metrics(base_pw, band)
    fa, fx, fd = band_metrics(ft_pw, band)
    return {
        "band_m": band,
        "base_band_ade2d": round(float(ba.mean()), 4),
        "ft_band_ade2d": round(float(fa.mean()), 4),
        "delta_band_ade2d_base_minus_ft": _paired(ba, fa, eid),
        "delta_band_xte_ade_base_minus_ft": _paired(bx, fx, eid),
        "delta_corridor_departure_base_minus_ft": _paired(bd, fd, eid),
    }


def run(args, device):
    LK, path = _find_lk()
    print(f"[rescore] instrument: {path}", flush=True)
    from tanitad.data.mixing import load_episode
    eps = sorted(Path(args.val_dir).glob("ep_*.pt"))
    a, b = (int(x) if x else None for x in args.holdout_slice.split(":"))
    episodes = [load_episode(str(p), mmap=True) for p in eps[a:b]]
    base, bstep = LK.load_refc(args.base_ckpt, "base", device)
    base_pw = LK.cl_realfootage("refc", base, episodes, device, None,
                                stride=args.stride, batch=args.batch)
    print(f"[rescore] base step {bstep}: closed_ade2s={float(base_pw['ade'].mean()):.3f}",
          flush=True)
    res = {"_design": "tolerance-band re-score of existing rollouts; band_ade2d = "
                      "max(0, 2D-dev - band) (fair metric forgiving in-lane recovery); "
                      "positive delta = FT better.",
           "base_ckpt": args.base_ckpt, "holdout_slice": args.holdout_slice,
           "primary_band_m": PRIMARY_BAND, "bands_m": BANDS,
           "base_closed_ade2s": round(float(base_pw["ade"].mean()), 4),
           "arms": {}, "evidence_class": "MEASURED (this re-score run)"}
    for spec in args.arms.split(","):
        name, ck = spec.split(":")
        ft, fstep = LK.load_refc(ck, "base", device)
        ft_pw = LK.cl_realfootage("refc", ft, episodes, device, None,
                                  stride=args.stride, batch=args.batch)
        arm = {"ckpt": ck, "ft_step": fstep,
               "ft_closed_ade2s": round(float(ft_pw["ade"].mean()), 4),
               "bands": {f"{bd:g}": rescore(LK, base_pw, ft_pw, bd) for bd in BANDS}}
        res["arms"][name] = arm
        prim = arm["bands"][f"{PRIMARY_BAND:g}"]
        dcdr = prim["delta_corridor_departure_base_minus_ft"]
        dade = prim["delta_band_ade2d_base_minus_ft"]
        dxte = prim["delta_band_xte_ade_base_minus_ft"]
        # WIN = departure SEP-lower AND band_ade2d NOT separated-worse
        dep_win = dcdr["delta"] > 0 and dcdr["separated"]
        ade_ok = not (dade["separated"] and dade["delta"] < 0)
        verdict = "WIN" if (dep_win and ade_ok) else (
            "ade-forgiven-but-dep-not-sep" if ade_ok else "BOUND")
        arm["primary_verdict"] = verdict
        print(f"[rescore] {name:9s} band{PRIMARY_BAND}: "
              f"dCDR={dcdr['delta']:+.4f}{'S' if dcdr['separated'] else '.'} "
              f"d_band_ade2d={dade['delta']:+.4f}{'S' if dade['separated'] else '.'} "
              f"(base {prim['base_band_ade2d']}->ft {prim['ft_band_ade2d']}) "
              f"d_band_xte={dxte['delta']:+.4f}{'S' if dxte['separated'] else '.'} "
              f"-> {verdict}", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print(f"[rescore] wrote {args.out}")
    print("TOLERANCE_RESCORE_DONE", flush=True)


def smoke(args):
    import types
    LK = types.SimpleNamespace()
    rng = np.random.default_rng(0)
    n_ep, per = 8, 20
    eid = sum(([str(e)] * per for e in range(n_ep)), [])
    N = n_ep * per
    # base: small deviations; ft: bigger 2D dev but mostly in-lane lateral wiggle
    base_pw = {"eid": eid, "lat_traj": torch.tensor(rng.uniform(0, 0.6, (N, 20)), dtype=torch.float),
               "de": torch.tensor(rng.uniform(0, 0.7, (N, 4)), dtype=torch.float)}
    ft_pw = {"eid": eid, "lat_traj": torch.tensor(rng.uniform(0, 0.9, (N, 20)), dtype=torch.float),
             "de": torch.tensor(rng.uniform(0, 1.0, (N, 4)), dtype=torch.float)}
    base_pw["ade"] = base_pw["de"].mean(1); ft_pw["ade"] = ft_pw["de"].mean(1)
    for bd in BANDS:
        r = rescore(LK, base_pw, ft_pw, bd)
        print(f"band {bd}: base_ade2d {r['base_band_ade2d']} ft {r['ft_band_ade2d']} "
              f"d_band_ade2d {r['delta_band_ade2d_base_minus_ft']['delta']:+.4f} "
              f"sep={r['delta_band_ade2d_base_minus_ft']['separated']}")
    Path(args.out).write_text(json.dumps({"smoke": True}, indent=2))
    print("TOLERANCE_RESCORE_SMOKE_OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="/root/models/refc-base-30k/ckpt.pt")
    ap.add_argument("--arms", default="naive:/workspace/refc-recovery-ft/ckpt.pt,"
                    "g2:/workspace/refc-ft-g2/ckpt.pt,"
                    "lowoodcl:/workspace/refc-lowood-cl/ckpt.pt")
    ap.add_argument("--val-dir", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--holdout-slice", default="28:40")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        return smoke(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args, device)


if __name__ == "__main__":
    main()
