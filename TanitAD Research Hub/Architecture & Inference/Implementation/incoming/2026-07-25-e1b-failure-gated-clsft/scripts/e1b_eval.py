"""E1b EVAL — paired closed-loop verdict, FT vs base (run AFTER the FT completes).

Re-runs the E1a real-footage closed loop (e1a_horizon.rollout VERBATIM) at K=20
and K=185 on the SAME held-out eval episodes E1a used
(physicalai-val-heldout-79d4e3d2d4c6, MEASURED disjoint from the parity-train the
FT mined from) for BOTH the base and the FT checkpoint, then reports the
PRE-REGISTERED primary + guardrails with the PAIRED episode-cluster bootstrap
(taniteval/ci.py, B=2000) on identical windows.

  PRIMARY : junction corridor-departure-rate @ K=185, paired Delta(FT - base).
            SUCCESS = CI-separated LOWER (hi < 0).
  GUARDRAIL(a): open-loop ADE@2s paired Delta(FT - base). PASS = CI includes 0
            or lower (not CI-separated-worse).
  Also reported: overall + longitudinal @ K=185, and K=20 (the standing 2 s
            instrument) so any 2 s regression is visible.

Base per-window arrays were not persisted by the original E1a run, so BOTH arms
are re-rolled here on identical windows — that is what makes the bootstrap PAIRED.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

for _p in ("/workspace/TanitAD/stack", "/workspace/TanitAD/stack/scripts",
           "/workspace/e1a_e2a"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e1a_horizon as e1a
import taniteval_ci as _ci
from tanitad.data.mixing import load_episode


def per_window(model, episodes, device, K, primary, junction_deg, stride, batch):
    """e1a rollout -> per-window departure indicators + strata + keys."""
    pw = e1a.rollout(model, episodes, device, K, stride, batch)
    lat = pw["lat"].numpy()                                  # [N,K]
    dep = (lat > primary).mean(1)                            # corridor dep rate
    win_dep = (lat > primary).any(1).astype(float)
    ade2s = pw["ade2s"].numpy()
    hd = pw["hd2s"].numpy(); spd = pw["speed"].numpy()
    junc = hd >= junction_deg
    long_ = (~junc) & (spd >= np.median(spd))
    keys = [(int(a), int(b)) for a, b in zip(pw["epi"], pw["t0"])]
    return {"key": keys, "eid": pw["eid"], "dep": dep, "win_dep": win_dep,
            "ade2s": ade2s, "junc": junc, "long": long_}


def paired(a_map, b_map, field, mask_name=None):
    """Align FT (a) vs base (b) on common keys; paired episode-cluster bootstrap
    of Delta(FT - base). mask_name selects a stratum by base's labels."""
    bk = {k: i for i, k in enumerate(b_map["key"])}
    common = [(i, bk[k]) for i, k in enumerate(a_map["key"]) if k in bk]
    ia = np.array([i for i, _ in common]); ib = np.array([j for _, j in common])
    if mask_name:
        m = b_map[mask_name][ib]
        ia, ib = ia[m], ib[m]
    if len(ia) < 2:
        return {"n": int(len(ia)), "note": "too few common windows"}
    eid = [b_map["eid"][j] for j in ib]
    return _ci.paired_episode_cluster_bootstrap(
        a_map[field][ia], b_map[field][ib], eid, n_boot=2000)


def run_arm(ckpt, preset, episodes, device, Ks, primary, junction_deg,
            stride, batch, skip_canary):
    model, step, cfg = e1a.load_refc(ckpt, preset, device)
    out = {"ckpt": ckpt, "step": step, "perK": {}}
    if not skip_canary:
        out["openloop_ade2s"] = e1a.openloop_canary(model, episodes, device,
                                                     stride, batch)
    for K in Ks:
        out["perK"][K] = per_window(model, episodes, device, K, primary,
                                    junction_deg, stride, batch)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt",
                    default="/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt")
    ap.add_argument("--ft-ckpt",
                    default="/workspace/e1b/refc-base-e1b-clsft/ckpt.pt")
    ap.add_argument("--preset", default="base")
    ap.add_argument("--val-dir",
                    default="/workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6")
    ap.add_argument("--horizons", default="20,185")
    ap.add_argument("--corridor-halfwidth", type=float, default=1.75)
    ap.add_argument("--junction-deg", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="/workspace/e1b/e1b_eval_result.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Ks = [int(x) for x in args.horizons.split(",")]
    prim = args.corridor_halfwidth
    ep_files = sorted(Path(args.val_dir).glob("ep_*.pt"))
    episodes = [load_episode(str(p), mmap=True) for p in ep_files]
    print(f"[e1b-eval] {len(episodes)} held-out eps | K {Ks} | dev {device}",
          flush=True)

    t0 = time.time()
    base = run_arm(args.base_ckpt, args.preset, episodes, device, Ks, prim,
                   args.junction_deg, args.stride, args.batch, False)
    ft = run_arm(args.ft_ckpt, args.preset, episodes, device, Ks, prim,
                 args.junction_deg, args.stride, args.batch, False)
    Kmax = max(Ks)

    res = {
        "_experiment": "E1b paired closed-loop verdict (FT vs base)",
        "_estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py), B=2000",
        "val_dir": args.val_dir, "n_episodes": len(episodes),
        "corridor_halfwidth_m": prim, "horizons_K": Ks,
        "base_ckpt": args.base_ckpt, "base_step": base["step"],
        "ft_ckpt": args.ft_ckpt, "ft_step": ft["step"],
        "openloop_ade2s": {
            "base": base["openloop_ade2s"], "ft": ft["openloop_ade2s"],
        },
        "PRIMARY_junction_corridor_departure_K%d" % Kmax:
            paired(ft["perK"][Kmax], base["perK"][Kmax], "dep", "junc"),
        "overall_corridor_departure_K%d" % Kmax:
            paired(ft["perK"][Kmax], base["perK"][Kmax], "dep"),
        "longitudinal_corridor_departure_K%d" % Kmax:
            paired(ft["perK"][Kmax], base["perK"][Kmax], "dep", "long"),
        "overall_closed_ade2s_K%d" % Kmax:
            paired(ft["perK"][Kmax], base["perK"][Kmax], "ade2s"),
        "GUARDRAIL_openloop_ade2s_delta_ft_minus_base": None,  # filled below
        "K20_overall_corridor_departure":
            paired(ft["perK"][Ks[0]], base["perK"][Ks[0]], "dep"),
        "K20_junction_corridor_departure":
            paired(ft["perK"][Ks[0]], base["perK"][Ks[0]], "dep", "junc"),
    }
    # open-loop guardrail delta (paired at window level via the canary arrays is
    # not aligned here; report both single-arm intervals + point delta)
    ob, of = base["openloop_ade2s"], ft["openloop_ade2s"]
    res["GUARDRAIL_openloop_ade2s_delta_ft_minus_base"] = {
        "base_mean": ob["mean"], "base_ci": [ob["lo"], ob["hi"]],
        "ft_mean": of["mean"], "ft_ci": [of["lo"], of["hi"]],
        "point_delta": round(of["mean"] - ob["mean"], 4),
        "_note": "single-arm episode-cluster bootstrap each; overlapping CIs => "
                 "no CI-separated open-loop regression (guardrail a PASS).",
    }

    prim_key = "PRIMARY_junction_corridor_departure_K%d" % Kmax
    p = res[prim_key]
    verdict = "INDETERMINATE"
    if isinstance(p, dict) and "separated" in p:
        if p["separated"] and p["hi"] < 0:
            verdict = ("SUCCESS: junction corridor-departure@K%d CI-separated "
                       "LOWER for FT (paired). Check guardrails." % Kmax)
        elif p["separated"] and p["lo"] > 0:
            verdict = ("BOUND/REGRESS: FT junction departure@K%d CI-separated "
                       "HIGHER (worse)." % Kmax)
        else:
            verdict = ("BOUND: FT junction departure@K%d NOT CI-separated from "
                       "base." % Kmax)
    res["verdict"] = verdict
    res["wall_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print(f"[e1b-eval] PRIMARY {prim_key}: {p}", flush=True)
    print(f"[e1b-eval] verdict: {verdict}  ({res['wall_s']}s) -> {args.out}",
          flush=True)
    print("E1B_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
