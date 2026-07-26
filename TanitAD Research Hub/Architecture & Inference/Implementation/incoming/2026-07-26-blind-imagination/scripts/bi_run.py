#!/usr/bin/env python3
"""E-IMAG driver — the blind-imagination horizon sweep. Runs on pod2 (A40).

Three stages, in the pre-registered order (`PRE_REGISTRATION.md` §8, §2, §7):

  gate   the REPRODUCTION GATE. Tensor-identity against the unmodified
         ``taniteval.rollout.collect`` at K=20, then v1's committed
         ``ade_0_2s`` on BOTH deployments (40 eps -> 0.4271, 600 eps -> 0.4108).
         Nothing else runs until this passes.
  sweep  E-IMAG-1/2. One fixed window set at K_max, every arm x every action
         regime, dense per-window paths dumped so any bar can be recomputed with
         no GPU.
  peek   E-IMAG-4. Uniform peek-every-T' and the ORACLE peek, same window set.

Usage (pod2):
    PYTHONPATH=/root/TanitAD/stack:/root/taniteval OMP_NUM_THREADS=8 \
      python3 bi_run.py gate  --out <dir>
    ... bi_run.py sweep --out <dir> --episodes 600 --kmax 185
    ... bi_run.py peek  --out <dir> --episodes 600 --kmax 185

Never touches pod1 / pod3 / the eval pod. Reads the val cache, writes nothing
into it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

for _p in ("/root/taniteval", "/root/TanitAD/stack", "/root/TanitAD/stack/scripts"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from taniteval import blindimag as bi          # noqa: E402
from taniteval import ci as _ci                # noqa: E402
from taniteval import data as tdata            # noqa: E402
from taniteval import loaders, registry, rollout  # noqa: E402

VAL_DIR = ("/workspace/data/physicalai_phase0/_epcache/"
           "physicalai-val-0c5f7dac3b11")
WP_IDX = [4, 9, 14, 19]                        # 0.5 / 1 / 1.5 / 2 s


def _entry(key="flagship-30k"):
    for m in registry.MODELS:
        if m["key"] == key:
            return m
    raise KeyError(key)


def _load(key, device):
    h = loaders.load(_entry(key), device=device)
    assert h["step_readout"] is not None, f"{key} has no grounded step readout"
    return h


def _ade_sparse(pred_dense, gt_dense):
    """The program's ``ade_0_2s``: per-window mean over the 4 sparse waypoints."""
    de = torch.linalg.norm(pred_dense[:, WP_IDX] - gt_dense[:, WP_IDX], dim=-1)
    return de.mean(dim=1).double().numpy()


# =========================================================================== #
# stage: gate                                                                  #
# =========================================================================== #
def stage_gate(args):
    dev = "cuda"
    out = {"block": bi.BLOCK, "version": bi.VERSION, "stage": "gate",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "torch": torch.__version__, "python": sys.version.split()[0]}
    h = _load(args.arm, dev)
    model, sro = h["model"], h["step_readout"]
    out["arm"] = args.arm
    out["ckpt_step"] = h.get("step")

    for n_ep, expect_ade, label, cached in ((40, 0.4271, "40ep_canonical", False),
                                            (600, 0.4108, "600ep", True)):
        files = tdata.list_val_episodes(VAL_DIR, n_ep)
        eps = tdata.load_frames(files)
        parity = tdata.last_val_parity()
        t_ref, ref = 0.0, None
        if not cached:
            # --- the harness's own path, UNMODIFIED, for tensor identity --- #
            t0 = time.time()
            ref = rollout.collect(model, sro, eps, dev, window=8, fwd_k=20,
                                  stride=8, batch=args.batch, speed_input=True)
            t_ref = time.time() - t0
        # --- the new instrument -------------------------------------------- #
        t0 = time.time()
        poses_by_ep = {i: e.poses for i, e in enumerate(eps)}
        cache = None
        if cached:
            cache = {}
            for i, ep in enumerate(eps):
                cache[i] = bi.encode_episode_states(model, ep, dev,
                                                    batch=args.enc_batch)
                if i % 100 == 0:
                    print(f"  [gate:encode] {i}/{len(eps)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        recs = bi.build_windows(model, eps, dev, 20, stride=8,
                                speed_input=True, states_cache=cache)
        preds, gts, eids = [], [], []
        for b in bi.batch_windows(recs, poses_by_ep, 20, batch=args.batch):
            r = bi.blind_rollout(model.predictor, b["states"], b["actions"],
                                 sro, 20, state_source="imagination",
                                 action_source="true_future",
                                 future_actions=b["future_actions"])
            preds.append(r["waypoints"].cpu().float())
            gts.append(b["gt_pos"].float())
            eids += b["eid"]
        t_new = time.time() - t0
        pred = torch.cat(preds)
        gt = torch.cat(gts)

        # (2) tensor identity against the harness (40-ep leg only)
        d_pred = d_gt = ade_ref = None
        if ref is not None:
            d_pred = float((pred[:, WP_IDX] - ref["pred"]).abs().max())
            d_gt = float((gt[:, WP_IDX] - ref["gt"]).abs().max())
            ade_ref = round(float(torch.linalg.norm(
                ref["pred"] - ref["gt"], dim=-1).mean(dim=1).mean()), 6)
        # (3)/(4) the committed number
        per_win = _ade_sparse(pred, gt)
        ci = _ci.episode_cluster_bootstrap(per_win, eids, n_boot=2000, seed=0)
        out[label] = {
            "n_windows": int(pred.shape[0]),
            "n_episodes_requested": n_ep,
            "n_episode_clusters": ci["n_episodes"],
            "encode_path": "cached_per_frame" if cached else "encode_window",
            "max_abs_diff_pred_vs_rollout_collect_m": d_pred,
            "max_abs_diff_gt_m": d_gt,
            "ade_0_2s_new_instrument": round(float(per_win.mean()), 6),
            "ade_0_2s_rollout_collect": ade_ref,
            "ade_0_2s_expected_committed": expect_ade,
            "reproduces_committed": bool(
                abs(float(per_win.mean()) - expect_ade) < 5e-4),
            "ci95_episode_cluster_bootstrap": ci,
            "val_parity": parity,
            "seconds_rollout_collect": round(t_ref, 1),
            "seconds_new_instrument": round(t_new, 1),
        }
        print(f"[gate:{label}] n={pred.shape[0]} "
              f"ade={per_win.mean():.6f} (expect {expect_ade}) "
              f"maxdiff={d_pred}", flush=True)
        del eps, recs, preds, gts, ref, cache
        torch.cuda.empty_cache()

    out["GATE_PASS"] = bool(
        out["40ep_canonical"]["reproduces_committed"]
        and out["600ep"]["reproduces_committed"]
        and out["40ep_canonical"]["max_abs_diff_pred_vs_rollout_collect_m"] < 1e-4)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "gate_reproduction.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nGATE_PASS = {out['GATE_PASS']}", flush=True)
    return 0 if out["GATE_PASS"] else 3


# =========================================================================== #
# the arm table — pre-registered, no arm added after a result is seen           #
# =========================================================================== #
SWEEP_ARMS = [
    # (name, state_source, action_source)
    ("a_imagination__true",      "imagination",   "true_future"),
    ("b_frozenlast__true",       "frozen_last",   "true_future"),
    ("c_fullobs__true",          "full_obs",      "true_future"),
    ("c2_observedpair__true",    "observed_pair", "true_future"),
    ("a_imagination__own",       "imagination",   "own_kinematic"),
    ("b_frozenlast__own",        "frozen_last",   "own_kinematic"),
    ("c_fullobs__own",           "full_obs",      "own_kinematic"),
    ("a_imagination__gtkin",     "imagination",   "gt_kinematic"),
    ("b_frozenlast__gtkin",      "frozen_last",   "gt_kinematic"),
    ("c_fullobs__gtkin",         "full_obs",      "gt_kinematic"),
    ("a_imagination__hold",      "imagination",   "hold_last"),
    ("b_frozenlast__hold",       "frozen_last",   "hold_last"),
]
# --------------------------------------------------------------------------- #
# E-IMAG-3 SENSITIVITIES — amendment A2, reported separately, NEVER mixed into
# the primary comparison (which was fixed before these existed).
#
# ⭐ The readout-level swap is a ZERO-TRAINING lever and it exists because of a
# fact read out of the checkpoint's own config: `HierarchicalGrounding` holds
# THREE step readouts, and `flagship_losses.grounding_losses` trains ALL THREE on
# the SAME operative imagination rollout — but over different lengths
# (`op_fwd_k=4`, `tac_fwd_k=16`, `str_fwd_k=20`; trainer defaults, and v1's
# launch command overrides none of them). Every grounded number in this program
# uses `step["op"]`, which was calibrated over **4 steps = 0.4 s** and is then
# read at k = 20. `step["str"]` was calibrated at exactly 20.
# --------------------------------------------------------------------------- #
EXTRA_ARMS = [
    # (name, state_source, action_source, update_speed_channel, readout_level)
    ("a_imagination__own_vupd",   "imagination", "own_kinematic", True,  "op"),
    ("a_imagination__true__roTAC", "imagination", "true_future",  False, "tac"),
    ("a_imagination__true__roSTR", "imagination", "true_future",  False, "str"),
    ("a_imagination__own__roSTR",  "imagination", "own_kinematic", False, "str"),
    ("a_imagination__hold__roSTR", "imagination", "hold_last",    False, "str"),
    ("b_frozenlast__true__roSTR",  "frozen_last", "true_future",  False, "str"),
    ("c2_observedpair__true__roSTR", "observed_pair", "true_future", False, "str"),
]


# --------------------------------------------------------------------------- #
# T_BLIND RUNG 1, the R1 PLANNER row — additive, inert for every other arm      #
# --------------------------------------------------------------------------- #
#: Arms whose dense ``fed_actions`` are kept (the action-amplitude statistics
#: that separate a gain change from a no-op). Empty => nothing kept => the dump
#: is byte-identical to the pre-planner one.
KEEP_FED: tuple = ()


def _plan_fn_for(model):
    """v1's DEPLOYED plan step, and nothing else.

    ⭐ This is ``closedloop.closed_loop_rollout``'s own two lines::

        ctx    = model.strategic_policy(win_s, nav_follow)["ctx"]
        w_look = model.tactical_policy(win_s, ctx)["waypoints"][LOOKAHEAD_STEP]

    with ``nav_cmd = follow`` (index 0), exactly as that harness does when no
    ``plan_fn`` is injected. ``blindimag`` deliberately does not import
    ``fourbrain``; the planner is handed in, so the rollout module stays a pure
    function of the predictor + readout.
    """
    from taniteval.closedloop import LOOKAHEAD_STEP

    def f(win_s, v):
        nav = torch.zeros(win_s.shape[0], dtype=torch.long, device=win_s.device)
        ctx = model.strategic_policy(win_s, nav)["ctx"]
        return model.tactical_policy(win_s, ctx)["waypoints"][LOOKAHEAD_STEP]
    return f


def _run_arms(model, sro, recs, poses_by_ep, k, batch, arms, extra=(),
              peek_cfgs=(), verbose=True, readouts=None):
    """One pass over the window set per arm. Returns {arm: dense pred [N,k,2]}
    plus the shared GT/floors and bookkeeping (computed once)."""
    store = {name: [] for name, *_ in arms}
    store.update({name: [] for name, *_ in extra})
    store.update({name: [] for name, *_ in peek_cfgs})
    psis = {name: [] for name in store}
    spds = {name: [] for name in store}
    peekm = {name: [] for name in store}
    feds = {name: [] for name in store if name in KEEP_FED}
    _pf = None                       # built lazily, only if a planner arm exists
    shared = {"gt": [], "gt_yaw": [], "cv": [], "hold_v0": [], "eid": [],
              "speed": [], "head_deg": [], "t0": [], "ep_i": []}
    nb = (len(recs) + batch - 1) // batch
    for bi_i, b in enumerate(bi.batch_windows(recs, poses_by_ep, k, batch=batch)):
        shared["gt"].append(b["gt_pos"].float())
        shared["gt_yaw"].append(b["gt_yaw"].float())
        shared["cv"].append(b["cv"].float())
        shared["hold_v0"].append(b["hold_v0"].float())
        shared["speed"].append(b["speed"].float())
        shared["head_deg"].append(b["head_deg"].float())
        shared["eid"] += b["eid"]
        shared["t0"] += b["t0"]
        shared["ep_i"] += b["ep_i"]
        gtd = b["gt_dpose"].to(b["states"].device)
        vl = b["v_last"].to(b["states"].device)
        for name, ss, asrc, *rest in list(arms) + list(extra):
            lvl = rest[1] if len(rest) > 1 else "op"
            kw = {}
            if str(asrc).partition("|")[0].strip() == "planner":
                if _pf is None:
                    _pf = _plan_fn_for(model)
                kw = {"plan_fn": _pf,
                      "gt_pos": b["gt_pos"].to(b["states"].device)}
            r = bi.blind_rollout(
                model.predictor, b["states"], b["actions"],
                sro if lvl == "op" else readouts[lvl], k,
                state_source=ss, action_source=asrc,
                future_actions=b["future_actions"],
                obs_states=b["obs_states"], gt_step_dpose=gtd, v_last=vl,
                update_speed_channel=bool(rest[0]) if rest else False, **kw)
            store[name].append(r["waypoints"].cpu().float())
            psis[name].append(r["psi"].cpu().float())
            spds[name].append(r["pred_speed"].cpu().float())
            if name in feds:
                feds[name].append(r["fed_actions"].cpu().float())
        for name, base_action, period, obar in peek_cfgs:
            r = bi.blind_rollout(
                model.predictor, b["states"], b["actions"], sro, k,
                state_source="imagination", action_source=base_action,
                future_actions=b["future_actions"],
                obs_states=b["obs_states"], gt_step_dpose=gtd, v_last=vl,
                peek_period=period, peek_oracle_bar=obar)
            store[name].append(r["waypoints"].cpu().float())
            psis[name].append(r["psi"].cpu().float())
            spds[name].append(r["pred_speed"].cpu().float())
            peekm[name].append(r["peek_mask"].cpu())
        if verbose and bi_i % 2 == 0:
            print(f"  [rollout] batch {bi_i + 1}/{nb}", flush=True)
    out = {"pred": {k2: torch.cat(v) for k2, v in store.items()},
           "psi": {k2: torch.cat(v) for k2, v in psis.items()},
           "pred_speed": {k2: torch.cat(v) for k2, v in spds.items()},
           "fed_actions": {k2: torch.cat(v) for k2, v in feds.items() if v},
           "peek_mask": {k2: torch.cat(v) for k2, v in peekm.items() if v}}
    for k2 in ("gt", "gt_yaw", "cv", "hold_v0", "speed", "head_deg"):
        out[k2] = torch.cat(shared[k2])
    for k2 in ("eid", "t0", "ep_i"):
        out[k2] = shared[k2]
    return out


def _prepare(args, dev):
    h = _load(args.arm, dev)
    files = tdata.list_val_episodes(VAL_DIR, args.episodes)
    eps = tdata.load_frames(files)
    parity = tdata.last_val_parity()
    print(f"[prep] {len(eps)} episodes; encoding all frames once "
          f"(this is what makes the full-obs arm free)", flush=True)
    t0 = time.time()
    cache, poses_by_ep = {}, {}
    for i, ep in enumerate(eps):
        cache[i] = bi.encode_episode_states(h["model"], ep, dev,
                                            batch=args.enc_batch)
        poses_by_ep[i] = ep.poses
        if i % 50 == 0:
            print(f"  [encode] episode {i}/{len(eps)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    t_enc = time.time() - t0
    print(f"[prep] encoding done in {t_enc:.0f}s", flush=True)
    recs = bi.build_windows(h["model"], eps, dev, args.kmax, stride=args.stride,
                            speed_input=True, states_cache=cache, verbose=True)
    print(f"[prep] {len(recs)} windows at K={args.kmax}", flush=True)
    return h, eps, cache, poses_by_ep, recs, parity, t_enc


def _dump(res, path, meta, names=None):
    def sel(d):
        return d if names is None else {k: v for k, v in d.items() if k in names}
    torch.save({"pred": sel(res["pred"]), "psi": sel(res["psi"]),
                "pred_speed": sel(res["pred_speed"]),
                "fed_actions": sel(res.get("fed_actions", {})),
                "peek_mask": sel(res.get("peek_mask", {})),
                "gt": res["gt"], "gt_yaw": res["gt_yaw"], "cv": res["cv"],
                "hold_v0": res["hold_v0"], "speed": res["speed"],
                "head_deg": res["head_deg"], "eid": res["eid"],
                "t0": res["t0"], "ep_i": res["ep_i"], "meta": meta}, path)
    print(f"[dump] {path} ({Path(path).stat().st_size / 1e6:.1f} MB)", flush=True)


def stage_sweep(args):
    """E-IMAG-1/2 AND E-IMAG-4 in ONE process — the 600-episode encoding is the
    dominant cost and is paid once."""
    dev = "cuda"
    outd = Path(args.out)
    outd.mkdir(parents=True, exist_ok=True)
    h, eps, cache, poses_by_ep, recs, parity, t_enc = _prepare(args, dev)
    periods = [int(x) for x in args.periods.split(",")]
    bars = [float(x) for x in args.oracle_bars.split(",")]
    bases = [b for b in args.peek_bases.split(",") if b]
    peek_cfgs = []
    for base in bases:
        tag = "own" if base == "own_kinematic" else "hold"
        peek_cfgs += [(f"peek_{tag}_uniform_T{p}", base, p, None) for p in periods]
        peek_cfgs += [(f"peek_{tag}_oracle_e{b:g}", base, None, b) for b in bars]
    t0 = time.time()
    res = _run_arms(h["model"], h["step_readout"], recs, poses_by_ep,
                    args.kmax, args.batch, SWEEP_ARMS, EXTRA_ARMS, peek_cfgs,
                    readouts=h["grounding"].step)
    t_roll = time.time() - t0
    meta = {"block": bi.BLOCK, "version": bi.VERSION, "stage": "sweep+peek",
            "arm_ckpt": args.arm, "ckpt_step": h.get("step"),
            "kmax": args.kmax, "stride": args.stride,
            "episodes_requested": args.episodes,
            "n_windows": len(recs),
            "n_episode_clusters": len(set(res["eid"])),
            "val_parity": parity, "wheelbase_used": bi.WHEELBASE,
            "speed_scale": bi.SPEED_SCALE, "dt_s": bi.DT,
            "arms": {n: {"state_source": s, "action_source": a}
                     for n, s, a in SWEEP_ARMS},
            "extra_arms": {n: {"state_source": s, "action_source": a,
                               "update_speed_channel": bool(u),
                               "readout_level": lv}
                           for n, s, a, u, lv in EXTRA_ARMS},
            "trained_rollout_lengths_steps": {"op": 4, "tac": 16, "str": 20},
            "peek": {n: {"base_action_source": ba, "uniform_period": p,
                         "oracle_bar_m": o} for n, ba, p, o in peek_cfgs},
            "uniform_periods": periods, "oracle_bars": bars,
            "peek_bases": bases,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seconds_encode": round(t_enc, 1),
            "seconds_rollout": round(t_roll, 1)}
    sweep_names = [n for n, *_ in SWEEP_ARMS] + [n for n, *_ in EXTRA_ARMS]
    peek_names = [n for n, *_ in peek_cfgs]
    _dump(res, str(outd / f"perwindow_sweep_K{args.kmax}.pt"), meta, sweep_names)
    _dump(res, str(outd / f"perwindow_peek_K{args.kmax}.pt"), meta, peek_names)
    (outd / f"sweep_meta_K{args.kmax}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["gate", "sweep"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="flagship-30k")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--kmax", type=int, default=185)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--enc-batch", type=int, default=32)
    ap.add_argument("--periods", default="2,3,5,10,15,20,30,45,60,90")
    ap.add_argument("--oracle-bars", default="0.02,0.05,0.1,0.2,0.5")
    ap.add_argument("--peek-bases", default="own_kinematic,hold_last")
    a = ap.parse_args()
    torch.manual_seed(0)
    return {"gate": stage_gate, "sweep": stage_sweep}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
