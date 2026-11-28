"""P2b eval — PAIRED held-out corridor_departure_rate: base REF-C vs recovery-FT.

Reuses the abe82f1f instrument (`lowood_lanekeep.py`) VERBATIM as the closed-loop
rollout + corridor metric — no instrument code is modified. Runs both arms on the
IDENTICAL held-out episodes (episode-DISJOINT from the FT slice) in one process and
reports the PAIRED episode-cluster bootstrap Δ (base − FT): positive => the FT LOWERS
corridor departures (better). peak_xte is the high-deviation GUARD (the FT must not
buy recovery by over-steering — the Gate-1 side-effect).

This is the head-to-head with Gate-1: Gate-1's real-junction recovery FT gave
held-out Δ≈0 (memorised, n≈15 scenes). This asks whether SYNTHETIC in-envelope
recovery — generated from EVERY held-in window — GENERALISES to held-out episodes.

Run (eval pod, gpu_lock refc-cl-improve, after abe82f1f frees eval):
  PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts \
    python3 eval_corridor_split.py \
      --base-ckpt /root/models/refc-base-30k/ckpt.pt \
      --ft-ckpt /workspace/refc-recovery-ft/ckpt.pt \
      --val-dir /root/valdata/physicalai-val-0c5f7dac3b11 \
      --holdout-slice 28:40 --out corridor_split_results.json
CPU smoke (paired-metric math on synthetic per-window arrays; no rollout):
  python3 eval_corridor_split.py --smoke --out /tmp/corridor_smoke.json
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
          "/root/taniteval", "/root/TanitAD/taniteval", "/workspace/taniteval",
          "/workspace/TanitAD/taniteval"]
for _up in (5, 6):                       # local repo layout depth (crash-safe)
    try:
        _r = _HERE.parents[_up]
        _ROOTS += [str(_r / "stack"), str(_r / "stack" / "scripts"), str(_r / "taniteval")]
    except IndexError:
        pass
for _p in _ROOTS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

THRESHOLDS = [1.0, 1.75, 2.5]
PRIMARY = 1.75
JUNCTION_DEG = 10.0


def _find_instrument():
    cands = [str(_HERE.parent), "/workspace/refc_cl_d2", "/root/TanitAD",
             "/workspace/TanitAD"]
    for _up in (5, 6):
        try:
            cands.append(str(_HERE.parents[_up]))
        except IndexError:
            pass
    for cand in cands:
        if not Path(cand).exists():
            continue
        hits = list(Path(cand).rglob("lowood_lanekeep.py"))
        if hits:
            spec = importlib.util.spec_from_file_location("lowood_lanekeep", str(hits[0]))
            LK = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(LK)
            return LK, str(hits[0])
    raise FileNotFoundError(
        "lowood_lanekeep.py (abe82f1f instrument) not found — pass its dir on "
        "PYTHONPATH or copy it beside this script.")


def _paired(a, b, eid, reduce="mean"):
    from taniteval.ci import paired_episode_cluster_bootstrap
    return paired_episode_cluster_bootstrap(np.asarray(a, float), np.asarray(b, float),
                                            eid, n_boot=2000)


def paired_block(LK, pw_base, pw_ft, mask, primary):
    m = np.flatnonzero(mask)
    if not len(m):
        return None
    eid = [pw_base["eid"][i] for i in m]
    b_dep = LK.corridor_stats(pw_base["lat_traj"], primary)[0].numpy()[m]
    f_dep = LK.corridor_stats(pw_ft["lat_traj"], primary)[0].numpy()[m]
    b_pk, f_pk = pw_base["peak_lat"].numpy()[m], pw_ft["peak_lat"].numpy()[m]
    b_ade, f_ade = pw_base["ade"].numpy()[m], pw_ft["ade"].numpy()[m]
    return {
        "n_windows": int(len(m)), "n_episodes": int(len(set(eid))),
        "delta_corridor_departure_rate_base_minus_ft": _paired(b_dep, f_dep, eid),
        "delta_peak_xte_m_base_minus_ft": _paired(b_pk, f_pk, eid),
        "delta_closed_ade2s_m_base_minus_ft": _paired(b_ade, f_ade, eid),
    }


def run(args, device):
    LK, path = _find_instrument()
    print(f"[eval] instrument: {path}", flush=True)
    from tanitad.data.mixing import load_episode
    eps_paths = sorted(Path(args.val_dir).glob("ep_*.pt"))
    a, b = (int(x) if x else None for x in args.holdout_slice.split(":"))
    hol_paths = eps_paths[a:b]
    episodes = [load_episode(str(p), mmap=True) for p in hol_paths]
    ood = LK.OODMap(args.p1_json) if args.p1_json and Path(args.p1_json).exists() else None
    base, bstep = LK.load_refc(args.base_ckpt, args.refc_preset, device)
    ft, fstep = LK.load_refc(args.ft_ckpt, args.refc_preset, device)
    print(f"[eval] base step {bstep} vs FT step {fstep} | holdout {args.holdout_slice} "
          f"({len(episodes)} eps)", flush=True)
    pw_base = LK.cl_realfootage("refc", base, episodes, device, ood,
                                stride=args.stride, batch=args.batch)
    pw_ft = LK.cl_realfootage("refc", ft, episodes, device, ood,
                              stride=args.stride, batch=args.batch)
    assert pw_base["eid"] == pw_ft["eid"], "arms not window-aligned"
    junc = np.abs(pw_base["head_deg"].numpy()) >= JUNCTION_DEG
    spd = pw_base["speed"].numpy()
    long_ = (~junc) & (spd >= np.median(spd))
    res = {
        "_design": "paired held-out corridor_departure: base REF-C vs recovery-FT, "
                   "same held-out episodes, episode-cluster bootstrap. Δ(base-ft)>0 "
                   "and CI-excludes-0 => FT reduces departures (WIN).",
        "_honest_frame": "LANE-KEEPING / on-policy drift at low OOD; NOT off-road/"
                         "collision (map/agent-free source; AlpaSim is the 3.2x-OOD "
                         "renderer this escapes).",
        "base_ckpt": args.base_ckpt, "ft_ckpt": args.ft_ckpt,
        "base_step": bstep, "ft_step": fstep,
        "holdout_slice": args.holdout_slice, "primary_threshold_m": PRIMARY,
        "per_arm": {
            "base": LK.summarize(pw_base, THRESHOLDS, PRIMARY, JUNCTION_DEG),
            "ft": LK.summarize(pw_ft, THRESHOLDS, PRIMARY, JUNCTION_DEG)},
        "paired_base_minus_ft": {
            "note": "positive delta => FT LOWER (better) than base; separated == CI "
                    "excludes 0",
            "overall": paired_block(LK, pw_base, pw_ft, np.ones(len(junc), bool), PRIMARY),
            "junction": paired_block(LK, pw_base, pw_ft, junc, PRIMARY),
            "longitudinal": paired_block(LK, pw_base, pw_ft, long_, PRIMARY)},
        "evidence_class": "MEASURED (this eval run)",
    }
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    ov = res["paired_base_minus_ft"]["overall"]
    d = ov["delta_corridor_departure_rate_base_minus_ft"]
    pk = ov["delta_peak_xte_m_base_minus_ft"]
    print(f"[eval] OVERALL Δcorridor_departure(base-ft)={d['delta']:+.4f} "
          f"[{d['lo']:+.4f},{d['hi']:+.4f}] separated={d['separated']}", flush=True)
    print(f"[eval] GUARD  Δpeak_xte(base-ft)={pk['delta']:+.4f} "
          f"[{pk['lo']:+.4f},{pk['hi']:+.4f}] (FT must NOT increase peak_xte: "
          f"Δ>=0 desired)", flush=True)
    verdict = ("WIN" if d["separated"] and d["delta"] > 0
               else "HURT" if d["separated"] and d["delta"] < 0 else "NULL")
    print(f"[eval] PRE-REGISTERED VERDICT: {verdict}", flush=True)
    print(f"[eval] wrote {args.out}")
    print("CORRIDOR_SPLIT_DONE", flush=True)


def smoke(args):
    """Validate the paired-metric math on synthetic per-window arrays (a FT that
    reduces lateral drift should yield a positive, separated Δ)."""
    import types
    LK = types.SimpleNamespace()

    def corridor_stats(lat_abs, thr, dt=0.1):
        over = lat_abs > thr
        return (over.float().mean(1), over.any(1).float(),
                torch.zeros(lat_abs.shape[0]))
    LK.corridor_stats = corridor_stats
    rng = np.random.default_rng(0)
    n_ep, per = 8, 20
    eid = sum(([str(e)] * per for e in range(n_ep)), [])
    N = n_ep * per
    base_lat = torch.tensor(rng.uniform(0, 3.0, size=(N, 20)), dtype=torch.float)
    ft_lat = base_lat * 0.6                                  # FT halves the drift
    pw_base = {"eid": eid, "lat_traj": base_lat,
               "peak_lat": base_lat.max(1).values, "ade": base_lat.mean(1)}
    pw_ft = {"eid": eid, "lat_traj": ft_lat,
             "peak_lat": ft_lat.max(1).values, "ade": ft_lat.mean(1)}
    blk = paired_block(LK, pw_base, pw_ft, np.ones(N, bool), PRIMARY)
    d = blk["delta_corridor_departure_rate_base_minus_ft"]
    print(json.dumps({"delta_departure_base_minus_ft": d,
                      "delta_peak_xte": blk["delta_peak_xte_m_base_minus_ft"]},
                     indent=2, default=str))
    assert d["delta"] > 0 and d["separated"], "synthetic FT-helps not detected"
    Path(args.out).write_text(json.dumps({"smoke": True, "block": blk},
                                         indent=2, default=str))
    print("CORRIDOR_SPLIT_SMOKE_OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="/root/models/refc-base-30k/ckpt.pt")
    ap.add_argument("--ft-ckpt", default="/workspace/refc-recovery-ft/ckpt.pt")
    ap.add_argument("--refc-preset", default="base")
    ap.add_argument("--val-dir", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--holdout-slice", default="28:40",
                    help="python slice over sorted ep_*.pt (DISJOINT from --ft-slice)")
    ap.add_argument("--p1-json", default="",
                    help="optional P1 OOD envelope (lowood_flagship_ci.json)")
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
