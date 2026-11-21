"""E1c FRONTIER EVALUATOR — the whole training trajectory, not two endpoints.

For BASE and every pre-registered checkpoint, on the SAME 44 held-out episodes
E1a and E1b used (byte-level disjoint from the mining/replay corpus):

  * HELD-OUT OPEN LOOP  (guardrails)  — ADE@2s, anchor_acc, anchor_ce,
    anchor_traj_l1, + the M1 lateral/longitudinal split.        ~60 s / arm
  * CLOSED LOOP @ K=185 (the primary) — corridor-departure overall AND
    junction (+ longitudinal), window-departure, peak/mean |XTE|, peak |dpsi|,
    OOD envelope, out-of-envelope fraction, closed ADE@2s.      ~735 s / arm
  * CLOSED LOOP @ K=20  — the standing 2 s instrument, REPORTED NON-DECIDING,
    run only for base / the endpoint / the selected checkpoint.

Both arms of every comparison are rolled on IDENTICAL windows, which is what
makes the episode-cluster bootstrap PAIRED.  The base arm is re-rolled here and
therefore reproduces E1a/E1b's base row a third time as a built-in control.

The rollout body is E1a's `e1a_horizon.rollout` reused VERBATIM via E1b's
declared additive-capture copy; `per_window`, `lateral_block` and `align` are
imported from `e1b_eval` UNCHANGED so E1c's numbers are measured by literally
the same code as E1b's.  The verdict comes from `e1c_common`, whose deciding
path was validated against synthetic guardrail failures by `e1c_selftest.py`
BEFORE this run.

ESTIMATOR: paired episode-cluster bootstrap (taniteval/ci.py, B=2000),
resampling the held-out EPISODES.  `overlapping_holdout_se` is used NOWHERE.

Results are written INCREMENTALLY after every checkpoint (--resume skips the
ones already banked), because a 4 h evaluation must not be all-or-nothing.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

for _p in ("/workspace/e1c", "/workspace/e1b"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e1b_eval as EB                     # noqa: E402  (binds the capture copy)
e1a = EB.e1a
import taniteval_ci as CI                 # noqa: E402  (md5 == taniteval/ci.py)
import e1c_common as C                    # noqa: E402
from tanitad.data.mixing import load_episode          # noqa: E402
from tanitad.instruments.numerics import strict_numerics   # noqa: E402
from driving_diagnostic import gt_ego_waypoints       # noqa: E402

BOOT, PAIRED = C.make_stat(CI)


def _ser(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def load_point(base_ckpt, delta_path, preset, device):
    """base + trainable-only delta -> the full checkpoint, verified."""
    model, base_step, cfg = e1a.load_refc(base_ckpt, preset, device)
    if delta_path is None:
        return model, 0, base_step
    d = torch.load(delta_path, map_location=device, weights_only=False)
    inc = model.load_state_dict(d["trainable"], strict=False)
    assert not inc.unexpected_keys, f"unexpected keys in {delta_path}: " \
                                    f"{inc.unexpected_keys[:5]}"
    bad = [k for k in inc.missing_keys if not k.startswith("encoder.")]
    assert not bad, f"delta is missing NON-encoder keys {bad[:5]} — not lossless"
    return model, int(d["step"]), base_step


def arm_arrays(model, episodes, device, Ks, prim, junc_deg, stride, batch, ood,
               label):
    out, t = {}, time.time()
    out["ol"] = C.openloop_full(model, episodes, device, e1a.W, e1a.WP_STEPS,
                                gt_ego_waypoints, stride=stride, batch=batch)
    print(f"[e1c-eval] {label}: open-loop n={len(out['ol']['eid'])} "
          f"ADE@2s={out['ol']['ade2s'].mean():.4f} "
          f"acc={out['ol']['anchor_acc'].mean():.4f} "
          f"l1={out['ol']['anchor_traj_l1'].mean():.4f} "
          f"({time.time() - t:.0f}s)", flush=True)
    for K in Ks:
        t2 = time.time()
        m = EB.per_window(model, episodes, device, K, prim, junc_deg, stride,
                          batch, ood)
        out[f"K{K}"] = m
        print(f"[e1c-eval] {label}: K={K} n={len(m['eid'])} "
              f"dep={m['dep'].mean():.4f} "
              f"junc_dep={m['dep'][m['junc']].mean():.4f} "
              f"peakXTE={m['peak_xte'].mean():.3f} "
              f"OODpeak={m['ood_peak'].mean():.3f} ({time.time() - t2:.0f}s)",
              flush=True)
    return out


def k20_block(ft, base):
    """The standing 2 s instrument — reported, NON-DECIDING."""
    return {nm: {f: {"base": C.arm_field(base, f, BOOT, mk),
                     "ft": C.arm_field(ft, f, BOOT, mk),
                     "paired_delta_ft_minus_base":
                         C.paired_field(ft, base, f, PAIRED, mk)}
                 for f in ("dep", "win_dep", "peak_xte", "ade2s")}
            for nm, mk in C.STRATA.items()}


def m1_blocks(ft, base, tag):
    ia, ib, eid = C.common_index(ft["ol"], base["ol"])
    out = {"openloop": EB.lateral_block(
        ft["ol"]["pred"][ia], ft["ol"]["gt"][ia],
        base["ol"]["pred"][ib], base["ol"]["gt"][ib], eid,
        f"open-loop 4 knots (0.5-2.0 s), held-out 44 — {tag}")}
    ia2, ib2, ceid = EB.align(ft["K185"], base["K185"])
    out["closedloop_K185"] = EB.lateral_block(
        ft["K185"]["pred2s"][ia2], ft["K185"]["gt2s"][ia2],
        base["K185"]["pred2s"][ib2], base["K185"]["gt2s"][ib2], ceid,
        f"closed-loop ADE@2s knots inside the K=185 rollout — {tag}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt",
                    default="/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt")
    ap.add_argument("--ft-dir", default="/workspace/e1c/refc-base-e1c-clsft")
    ap.add_argument("--preset", default="base")
    ap.add_argument("--val-dir",
                    default="/workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6")
    ap.add_argument("--p1-json", default="/workspace/e1a_e2a/lowood_flagship_ci.json")
    ap.add_argument("--corridor-halfwidth", type=float, default=1.75)
    ap.add_argument("--junction-deg", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", default="", help="override the frontier steps")
    ap.add_argument("--k20-steps", default="4000")
    ap.add_argument("--order-by-gate", default="",
                    help="heldout_gate.jsonl from training. ONLY reorders the "
                         "compute so checkpoints that already passed the "
                         "in-training held-out open-loop gate are rolled FIRST "
                         "(a 4 h eval must not be all-or-nothing). Changes no "
                         "pre-registered criterion and no reported ordering.")
    ap.add_argument("--episodes", type=int, default=999)
    ap.add_argument("--out", default="/workspace/e1c/e1c_frontier_result.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prim, jd = args.corridor_halfwidth, args.junction_deg
    ood = e1a.OODMap(args.p1_json)
    ft_dir = Path(args.ft_dir)
    steps = sorted([int(x) for x in args.steps.split(",") if x] if args.steps
                   else [int(p.stem.split("step")[1]) for p in
                         sorted(ft_dir.glob("delta_step*.pt"))])
    k20 = {int(x) for x in args.k20_steps.split(",") if x}

    # COMPUTE ORDER ONLY (never a reporting order, never a criterion): roll the
    # checkpoints that already cleared the in-training held-out gate first, so a
    # truncated run still holds the decision-relevant points.
    eval_order = list(steps)
    if args.order_by_gate and Path(args.order_by_gate).exists():
        gk = {}
        for ln in Path(args.order_by_gate).read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                gk[int(r["step"])] = bool(r.get("gate_ok", True))
        eval_order = sorted(steps, key=lambda s: (not gk.get(s, True), s))
        print(f"[e1c-eval] compute order (gate-passing first): {eval_order}",
              flush=True)
    ep_files = sorted(Path(args.val_dir).glob("ep_*.pt"))[:args.episodes]
    episodes = [load_episode(str(p), mmap=True) for p in ep_files]
    print(f"[e1c-eval] {len(episodes)} held-out eps | frontier steps {steps} | "
          f"K20 at {sorted(k20)} | dev {device}", flush=True)

    outp = Path(args.out)
    res = {}
    if args.resume and outp.exists():
        res = json.loads(outp.read_text())
        print(f"[e1c-eval] resuming; already banked: "
              f"{sorted(res.get('points', {}))}", flush=True)
    res.setdefault("_experiment", "E1c frontier: held-out-gated closed-loop SFT")
    res.setdefault("_estimator",
                   "paired_episode_cluster_bootstrap / episode_cluster_bootstrap "
                   "(taniteval/ci.py), B=%d, resampling the held-out EPISODES. "
                   "overlapping_holdout_se is used NOWHERE." % C.B_BOOT)
    res.setdefault("_prereg", "PRE_REGISTRATION_E1C.md §4 — a checkpoint is a "
                              "SUCCESS POINT iff P1 & P2 (corridor-departure@K185 "
                              "overall AND junction, paired, CI-separated LOWER) "
                              "and Ga/Gb1/Gb2/Gc all hold on HELD-OUT data.")
    res.update({"val_dir": args.val_dir, "n_episodes": len(episodes),
                "corridor_halfwidth_m": prim, "junction_deg": jd,
                "stride": args.stride, "base_ckpt": args.base_ckpt,
                "ft_dir": str(ft_dir), "frontier_steps": steps,
                "_compute_order": eval_order,
                "M_frontier_points": len(steps) + 1,
                "bonferroni_alpha": C.BONFERRONI_ALPHA})
    res.setdefault("points", {})

    t0 = time.time()
    with strict_numerics():
        # ---------------- BASE (rolled once, reused for every pairing) --------
        base_model, _, base_step = load_point(args.base_ckpt, None, args.preset,
                                              device)
        print(f"[e1c-eval] BASE: step {base_step} | anchors "
              f"{tuple(base_model.decoder.anchors.shape)}", flush=True)
        base = arm_arrays(base_model, episodes, device, [185, 20], prim, jd,
                          args.stride, args.batch, ood, "BASE")
        base_anchors = base_model.decoder.anchors.detach().cpu().clone()
        del base_model; torch.cuda.empty_cache()
        res["base_step"] = base_step

        def _m(a):
            a = np.asarray(a, float)
            return round(float(a.mean()), 6) if a.size else None   # NaN-safe JSON

        res["base_reproduction_check"] = {
            "_what": "E1a/E1b reported these exact base figures; the base arm is "
                     "re-rolled here as a control.",
            "K185_n_windows": int(len(base["K185"]["eid"])),
            "K185_n_junction_windows": int(base["K185"]["junc"].sum()),
            "K185_overall_dep": _m(base["K185"]["dep"]),
            "K185_junction_dep": _m(base["K185"]["dep"][base["K185"]["junc"]]),
            "K185_peak_xte": _m(base["K185"]["peak_xte"]),
            "K185_ood_peak": _m(base["K185"]["ood_peak"]),
            "K20_overall_dep": _m(base["K20"]["dep"]),
            "openloop_ade2s": _m(base["ol"]["ade2s"]),
            "openloop_anchor_acc": _m(base["ol"]["anchor_acc"]),
            "openloop_anchor_traj_l1": _m(base["ol"]["anchor_traj_l1"]),
            "e1b_reference": {"K185_overall_dep": 0.5877, "K185_junction_dep": 0.8414,
                              "K185_peak_xte": 38.9445, "K185_ood_peak": 1.2664,
                              "K20_overall_dep": 0.0053, "openloop_ade2s": 0.4747,
                              "openloop_anchor_acc": 0.6815,
                              "openloop_anchor_traj_l1": 0.1775}}
        outp.write_text(json.dumps(res, indent=2, default=_ser))

        # ---------------- FRONTIER --------------------------------------------
        for st_i in eval_order:
            if str(st_i) in res["points"] and args.resume:
                print(f"[e1c-eval] step {st_i}: already banked, skip", flush=True)
                continue
            dp = ft_dir / f"delta_step{st_i:05d}.pt"
            model, dstep, _ = load_point(args.base_ckpt, dp, args.preset, device)
            assert dstep == st_i, f"{dp} says step {dstep}"
            assert torch.equal(model.decoder.anchors.detach().cpu(), base_anchors), \
                "decoder.anchors drifted — the anchor set must be constant"
            Ks = [185, 20] if st_i in k20 else [185]
            arm = arm_arrays(model, episodes, device, Ks, prim, jd, args.stride,
                             args.batch, ood, f"step{st_i}")
            del model; torch.cuda.empty_cache()

            stt = C.frontier_point_stats(st_i, arm["K185"], base["K185"],
                                         arm["ol"], base["ol"], BOOT, PAIRED,
                                         label=str(dp))
            ev = C.evaluate_point(stt)
            stt["EVAL"] = ev
            stt["M1_lateral_split"] = m1_blocks(arm, base, f"step {st_i}")
            if "K20" in arm:
                stt["closed_loop_K20_nondeciding"] = k20_block(arm["K20"],
                                                               base["K20"])
            res["points"][str(st_i)] = stt
            print(f"[e1c-eval] step {st_i}: SUCCESS_POINT={ev['SUCCESS_POINT']} "
                  f"failed={ev['failed']} | dep_ov={ev['_dep_overall_delta']} "
                  f"dep_ju={ev['_dep_junction_delta']} "
                  f"ade={ev['_openloop_ade2s_delta']} "
                  f"{ev['_openloop_ade2s_ci']}", flush=True)
            res["wall_s"] = round(time.time() - t0, 1)
            outp.write_text(json.dumps(res, indent=2, default=_ser))

        # ---------------- VERDICT ---------------------------------------------
        evals = [res["points"][str(s)]["EVAL"] for s in steps
                 if str(s) in res["points"]]
        v = C.render_verdict(evals)
        res["VERDICT"] = v
        res["FRONTIER_TABLE"] = [
            {"step": e["step"],
             "dep_overall_delta": e["_dep_overall_delta"],
             "dep_junction_delta": e["_dep_junction_delta"],
             "openloop_ade2s_delta": e["_openloop_ade2s_delta"],
             "openloop_ade2s_ci": e["_openloop_ade2s_ci"],
             "ood_peak_ft": e["_ood_peak_ft"],
             "P1": e["P1_dep_overall_separated_lower"],
             "P2": e["P2_dep_junction_separated_lower"],
             "Ga": e["Ga_openloop_ade2s_ok"], "Gb1": e["Gb1_anchor_acc_ok"],
             "Gb2": e["Gb2_anchor_traj_l1_ok"], "Gc": e["Gc_ood_in_band"],
             "SUCCESS_POINT": e["SUCCESS_POINT"],
             "multiplicity_robust": e["multiplicity_robust"],
             "noninferior_0p05": e["noninferior_0p05"],
             "failed": e["failed"]}
            for e in evals]
        outp.write_text(json.dumps(res, indent=2, default=_ser))

        # ---------------- K=20 for the winner, if not already run -------------
        wstep = v.get("winner_step")
        if wstep is not None and wstep not in k20:
            print(f"[e1c-eval] running the non-deciding K=20 block for the "
                  f"selected step {wstep}", flush=True)
            model, _, _ = load_point(args.base_ckpt,
                                     ft_dir / f"delta_step{wstep:05d}.pt",
                                     args.preset, device)
            arm = arm_arrays(model, episodes, device, [20], prim, jd,
                             args.stride, args.batch, ood, f"step{wstep}-K20")
            del model; torch.cuda.empty_cache()
            res["points"][str(wstep)]["closed_loop_K20_nondeciding"] = \
                k20_block(arm["K20"], base["K20"])

    res["wall_s"] = round(time.time() - t0, 1)
    outp.write_text(json.dumps(res, indent=2, default=_ser))
    print(f"[e1c-eval] VERDICT: {v['verdict']} (winner step {v.get('winner_step')})",
          flush=True)
    print(f"[e1c-eval] {v['text']}", flush=True)
    print(f"[e1c-eval] wall {res['wall_s']}s -> {outp}", flush=True)
    print("E1C_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
