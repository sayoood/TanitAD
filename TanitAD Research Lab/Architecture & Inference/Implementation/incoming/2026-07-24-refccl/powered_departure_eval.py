"""MORE-POWERED DEPARTURE EVAL — 2-fold cross-fit of the naive recovery recipe to n=40.

The re-score left ONE residual question: the naive departure win (+0.0089 S in the D2
eval, +0.0083 n.s. in the re-score) is boundary-significant at n=12 held-out. The
existing ckpts (trained on eps 0:28) CAP at 12 held-out eps, so re-eval alone cannot
add power. The standard fix (DAgger cross-fit): train the naive recipe on each half of
the 40-ep val and evaluate on the OTHER half -> pool -> ALL 40 eps held-out, each scored
by a model that never trained on it. n=12 -> n=40 = ~1.8x power (sqrt(40/12)).

  foldA: train on lo (0:20) -> eval on hi (20:40)
  foldB: train on hi (20:40) -> eval on lo (0:20)
  pooled held-out = base-vs-FT over all 40 eps, paired episode-cluster bootstrap.

Scored under BOTH corridor_departure_rate AND the fair band_ade2d(1.0) (tolerance_rescore).

PRE-REGISTERED (both committed):
  WIN   at n=40: corridor_departure dCDR CI-excludes-0 (departures really cut) AND
        band_ade2d(1.0) CI-includes-0 (no fair-metric ADE cost) -> the closed-loop lever
        IS a real net win -> PROMOTABLE, the direction REOPENS decisively.
  BOUND departure win vanishes / not separated at n=40 -> it was n=12 noise -> the lever
        is not a net win on road-keeping -> the direction closes honestly.

Reuses lowood_lanekeep.cl_realfootage + tolerance_rescore.band_metrics. This file EVALS
the two fold ckpts (produced by recovery_aug_ft.py); the chain trains them first.

CPU smoke: python3 powered_departure_eval.py --smoke --out /tmp/pde_smoke.json
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
_ROOTS = ["/root/TanitAD/stack", "/root/TanitAD/stack/scripts", "/root/taniteval",
          "/root/TanitAD/taniteval", "/workspace/refc_cl_d2", str(_HERE.parent),
          str(_HERE.parents[1] / "2026-07-23-refc-planner-closedloop")]
for _up in (5, 6):
    try:
        r = _HERE.parents[_up]
        _ROOTS += [str(r / "stack"), str(r / "stack" / "scripts"), str(r / "taniteval")]
    except IndexError:
        pass
for _p in _ROOTS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

BAND = 1.0
JUNCTION_DEG = 10.0


def _find(name):
    for cand in (str(_HERE.parent), "/workspace/refc_cl_d2", "/root/TanitAD", "/workspace/TanitAD"):
        if not Path(cand).exists():
            continue
        hits = list(Path(cand).rglob(name))
        if hits:
            spec = importlib.util.spec_from_file_location(name[:-3], str(hits[0]))
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            return m
    raise FileNotFoundError(name)


def _cat(pw_lo, pw_hi, eid_offset):
    """Concatenate two per-window dicts; offset hi's episode ids so lo/hi stay distinct."""
    out = {}
    for k, v in pw_lo.items():
        if k == "eid":
            out[k] = list(v) + [str(int(e) + eid_offset) for e in pw_hi["eid"]]
        elif torch.is_tensor(v):
            out[k] = torch.cat([v, pw_hi[k]])
    return out


def _paired(a, b, eid):
    from taniteval.ci import paired_episode_cluster_bootstrap
    return paired_episode_cluster_bootstrap(np.asarray(a, float), np.asarray(b, float),
                                            eid, n_boot=2000)


def run(args, device):
    LK = _find("lowood_lanekeep.py")
    TR = _find("tolerance_rescore.py")
    from tanitad.data.mixing import load_episode
    eps = sorted(Path(args.val_dir).glob("ep_*.pt"))
    la, lb = (int(x) if x else None for x in args.lo_slice.split(":"))
    ha, hb = (int(x) if x else None for x in args.hi_slice.split(":"))
    lo = [load_episode(str(p), mmap=True) for p in eps[la:lb]]
    hi = [load_episode(str(p), mmap=True) for p in eps[ha:hb]]
    n_lo, n_hi = len(lo), len(hi)
    base, bstep = LK.load_refc(args.base_ckpt, "base", device)
    ftA, _ = LK.load_refc(args.ft_a_ckpt, "base", device)   # trained lo -> eval hi
    ftB, _ = LK.load_refc(args.ft_b_ckpt, "base", device)   # trained hi -> eval lo

    def roll(m, e):
        return LK.cl_realfootage("refc", m, e, device, None, stride=args.stride, batch=args.batch)
    print(f"[pde] rolling base/foldB on lo ({n_lo} eps) + base/foldA on hi ({n_hi} eps)...", flush=True)
    b_lo, b_hi = roll(base, lo), roll(base, hi)
    fB_lo, fA_hi = roll(ftB, lo), roll(ftA, hi)             # cross-fit: held-out each
    base_pw = _cat(b_lo, b_hi, n_lo)
    ft_pw = _cat(fB_lo, fA_hi, n_lo)
    assert base_pw["eid"] == ft_pw["eid"], "cross-fit windows not aligned"
    eid = base_pw["eid"]
    n_ep = len(set(eid))

    b_dep = (base_pw["lat_traj"].numpy() > 1.75).mean(axis=1)
    f_dep = (ft_pw["lat_traj"].numpy() > 1.75).mean(axis=1)
    b_band = TR.band_metrics(base_pw, BAND)[0]
    f_band = TR.band_metrics(ft_pw, BAND)[0]
    b_ade, f_ade = base_pw["ade"].numpy(), ft_pw["ade"].numpy()
    dcdr = _paired(b_dep, f_dep, eid)
    dband = _paired(b_band, f_band, eid)
    dade = _paired(b_ade, f_ade, eid)

    dep_win = dcdr["delta"] > 0 and dcdr["separated"]
    ade_ok = not (dband["separated"] and dband["delta"] < 0)
    verdict = "WIN (departures cut + no fair-metric ADE cost) -> PROMOTABLE" if (dep_win and ade_ok) \
        else "BOUND (departure win not separated at n=%d)" % n_ep
    res = {"_design": "2-fold cross-fit naive recovery -> n=%d held-out; paired base-vs-FT" % n_ep,
           "n_episodes_heldout": n_ep, "n_windows": len(eid),
           "power_vs_n12": round((n_ep / 12) ** 0.5, 2),
           "base_ckpt": args.base_ckpt, "fold_ckpts": [args.ft_a_ckpt, args.ft_b_ckpt],
           "delta_corridor_departure_base_minus_ft": dcdr,
           "delta_band_ade2d_1.0_base_minus_ft": dband,
           "delta_closed_ade2s_exactL2_base_minus_ft": dade,
           "abs": {"base_dep": round(float(b_dep.mean()), 4), "ft_dep": round(float(f_dep.mean()), 4),
                   "base_band_ade2d": round(float(b_band.mean()), 4),
                   "ft_band_ade2d": round(float(f_band.mean()), 4)},
           "verdict": verdict, "evidence_class": "MEASURED (this cross-fit eval)"}
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print(f"[pde] n={n_ep} eps ({len(eid)} win, power {res['power_vs_n12']}x vs n12)", flush=True)
    print(f"[pde] dCDR={dcdr['delta']:+.4f} [{dcdr['lo']:+.4f},{dcdr['hi']:+.4f}] sep={dcdr['separated']}", flush=True)
    print(f"[pde] d_band_ade2d(1.0)={dband['delta']:+.4f} [{dband['lo']:+.4f},{dband['hi']:+.4f}] sep={dband['separated']}", flush=True)
    print(f"[pde] VERDICT: {verdict}", flush=True)
    print(f"[pde] wrote {args.out}"); print("POWERED_DEPARTURE_DONE", flush=True)


def smoke(args):
    import types
    TR = types.SimpleNamespace()

    def band_metrics(pw, band):
        de = pw["de"].numpy()
        return (np.maximum(0.0, de - band).mean(axis=1),)
    TR.band_metrics = band_metrics
    rng = np.random.default_rng(0)
    n, per = 40, 15
    eid = sum(([str(e)] * per for e in range(n)), [])
    N = n * per
    base_pw = {"eid": eid, "lat_traj": torch.tensor(rng.uniform(0, 1.2, (N, 20)), dtype=torch.float),
               "de": torch.tensor(rng.uniform(0, 0.7, (N, 4)), dtype=torch.float)}
    ft_pw = {"eid": eid, "lat_traj": torch.tensor(rng.uniform(0, 1.0, (N, 20)), dtype=torch.float),
             "de": torch.tensor(rng.uniform(0, 0.75, (N, 4)), dtype=torch.float)}
    b_dep = (base_pw["lat_traj"].numpy() > 1.0).mean(1); f_dep = (ft_pw["lat_traj"].numpy() > 1.0).mean(1)
    d = _paired(b_dep, f_dep, eid)
    print(json.dumps({"n_ep": n, "dCDR": {"delta": d["delta"], "sep": d["separated"]}}, indent=2))
    Path(args.out).write_text(json.dumps({"smoke": True}, indent=2))
    print("POWERED_DEPARTURE_SMOKE_OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="/root/models/refc-base-30k/ckpt.pt")
    ap.add_argument("--ft-a-ckpt", default="/workspace/refc-naive-foldA/ckpt.pt")
    ap.add_argument("--ft-b-ckpt", default="/workspace/refc-naive-foldB/ckpt.pt")
    ap.add_argument("--val-dir", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--lo-slice", default="0:20")
    ap.add_argument("--hi-slice", default="20:40")
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
