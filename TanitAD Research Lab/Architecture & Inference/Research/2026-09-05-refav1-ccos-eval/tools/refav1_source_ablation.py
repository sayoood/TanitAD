"""D-REFAV1-CCOS-EVAL — GATE 2b of `stack/tanitad/eval/echo_gate.py` (`source_ablation_test`),
in refav1 form: does the planner USE the scene, or only carry a wire to it?

Hand `plan()` the WRONG scene (the feature window of a different eval window, derangement fixed
by seed) while holding the measured v0 and the true nav; and the WRONG ego (v0 of a different
window) while holding the scene. Degradation = relative rise of the arm's ADE against the GT on
the SAME windows (paired episode-cluster bootstrap). A model that reads the scene is hurt by the
wrong scene; one that echoes its own dynamics is not.

CONTROLS that must read KNOWN values (printed beside the arm):
  * `ha0` under scene derangement: EXACTLY 0.0 (it never reads the scene)
  * the shipped `cos` arm: a straight line on 282/282 — its scene degradation must read ~0 by
    construction (the deliberate-regression form of this test)
Stratified subset (every --every-th window of the banked 282-window grid → one window per
episode cluster), each window planned 3× per metric. GPU.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys, time
import numpy as np
import torch


def _load_arm(wt):
    sys.path.insert(0, os.path.join(wt, "stack")); sys.path.insert(0, os.path.join(wt, "taniteval"))
    p = os.path.join(wt, "taniteval", "tools", "refav1_arm.py")
    spec = importlib.util.spec_from_file_location("refav1_arm_abl", p)
    m = importlib.util.module_from_spec(spec); sys.modules["refav1_arm_abl"] = m; spec.loader.exec_module(m)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt", default=r"C:\Users\Admin\tanitad-wt")
    for f in ("ckpt", "cache", "episodes", "out"):
        ap.add_argument(f"--{f}", required=True)
    ap.add_argument("--config", default=None); ap.add_argument("--labels", default=None); ap.add_argument("--nav", default=None)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--window-stride", type=int, default=40)
    ap.add_argument("--every", type=int, default=12, help="every n-th window of the 282 grid")
    ap.add_argument("--metrics", default="cos,ccos"); ap.add_argument("--weights", default=None, help="w_jerk,w_kappa,w_vend for every metric (else shipped)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-boot", type=int, default=2000)
    a0 = ap.parse_args(argv)
    ra = _load_arm(a0.wt)
    from taniteval import ci as CI
    dev = a0.device
    model, cfg, prov = ra.load_model(a0.ckpt, a0.config, dev, False)
    K = ra.K_TRAJ_DEFAULT; DT = ra.DT

    class _A: pass
    a = _A(); a.cache, a.episodes, a.labels, a.nav = a0.cache, a0.episodes, a0.labels, a0.nav
    a.lru, a.plan_seed = 8, 0; a.plan_n_samples = a.plan_n_iters = a.plan_n_elites = None
    ld = ra.build_loader(a, cfg, max(10, int(cfg.op_steps)), ra.episode_names(a0.cache))
    pc = ra._plan_cfg(cfg, a)
    stride = max(1, int(a0.window_stride))
    grid = [(wi, ei, t) for wi, (ei, t) in enumerate(ld.windows) if (t - (ld.W - 1)) % stride == 0]
    sel = grid[::a0.every]
    rng = np.random.default_rng(a0.seed)
    perm = rng.permutation(len(sel))
    while np.any(perm == np.arange(len(sel))):      # a derangement: no window keeps its own
        perm = rng.permutation(len(sel))
    weights = tuple(float(x) for x in a0.weights.split(",")) if a0.weights else None
    metrics = a0.metrics.split(",")
    # gather inputs once
    W = []
    for (wi, ei, t) in sel:
        nm = ld.names[ei]; F_ep, v_ep, kap_ep = ld._episode(nm)
        # the GT exactly as run_dump builds it: RAW 10 Hz poses of the v2ep file -> gt_waypoints(poses, t, k)
        o = torch.load(os.path.join(a0.episodes, f"{nm}.v2ep.pt"), map_location="cpu", weights_only=False)
        poses = o["poses"].float()
        if float(poses[2 * t, 3]) != float(v_ep[2 * t]):
            raise RuntimeError("loader v0 != poses[2t, 3] — wrong episode file")
        ld._order, ld._cursor = [wi], 0
        b = ld.batch(1)
        nid = ld._nav_id.get(ld.clip_id[nm]) if ld._nav_on else None
        W.append({"wi": wi, "ei": ei, "t": t, "eid": nm, "feats": b["feats"].to(dev), "v0": float(v_ep[2 * t]),
                  "nav": (torch.tensor([0 if nid is None else int(nid)], device=dev) if ld._nav_on else None),
                  "g": ra.gt_waypoints(poses, t, K)[0].float().cpu().numpy()})
    rec_units = "kappa"
    res = {"tool": "refav1_source_ablation.py", "model": prov, "n_windows": len(W), "n_episodes": len({w["eid"] for w in W}),
           "every": a0.every, "derangement_seed": a0.seed, "tier": "T1", "estimator": "paired_episode_cluster_bootstrap",
           "weights": weights or "shipped", "metrics": {}}
    t0 = time.time()
    for metric in metrics:
        ade = {"true": [], "scene_deranged": [], "ego_deranged": []}
        ade_ha0 = {"true": [], "scene_deranged": [], "ego_deranged": []}
        for i, w in enumerate(W):
            j = int(perm[i])
            cases = {"true": (w["feats"], w["v0"]), "scene_deranged": (W[j]["feats"], w["v0"]), "ego_deranged": (w["feats"], W[j]["v0"])}
            for cname, (feats, v0) in cases.items():
                with torch.no_grad():
                    r = model.plan(feats, v0=v0, nav_cmd=w["nav"], plan_cfg=pc, model_action_units=rec_units,
                                   cost_metric=metric, cost_weights=weights)
                # the path is integrated from the TRUE measured v0 of THIS window for the ego case too?
                # No: the arm's own convention — the unicycle integrates from the v0 the planner was GIVEN
                # (that is what "wrong ego" means for a planner: it plans AND rolls from the wrong state).
                P = ra.paths_from_controls(r.controls.detach(), v0, DT, K, action_units=rec_units)
                ade[cname].append(float(np.linalg.norm(P.float().cpu().numpy()[..., :2] - w["g"][..., :2], axis=-1).mean()))
                H = ra.paths_from_controls(ra.hold_v0_controls(K).to(dev), v0, DT, K)
                ade_ha0[cname].append(float(np.linalg.norm(H.float().cpu().numpy()[..., :2] - w["g"][..., :2], axis=-1).mean()))
            if (i + 1) % 5 == 0:
                print(f"[{metric}] {i + 1}/{len(W)} {time.time() - t0:.0f}s", flush=True)
        eid = [w["eid"] for w in W]
        blk = {"ade": {k: float(np.mean(v)) for k, v in ade.items()}, "ade_ha0": {k: float(np.mean(v)) for k, v in ade_ha0.items()},
               "sources": {}}
        for src in ("scene_deranged", "ego_deranged"):
            r = CI.paired_episode_cluster_bootstrap(np.array(ade[src]), np.array(ade["true"]), eid, n_boot=a0.n_boot, seed=a0.seed)
            rc = CI.paired_episode_cluster_bootstrap(np.array(ade_ha0[src]), np.array(ade_ha0["true"]), eid, n_boot=a0.n_boot, seed=a0.seed)
            base = float(np.mean(ade["true"]))
            blk["sources"][src] = {"delta_ade": r.get("delta"), "ci": [r.get("lo"), r.get("hi")], "separated": r.get("separated"),
                                   "degradation_rel": (r.get("delta") or 0.0) / base if base > 0 else None,
                                   "control_ha0_delta": rc.get("delta"), "control_ha0_ci": [rc.get("lo"), rc.get("hi")]}
        s, e = blk["sources"]["scene_deranged"], blk["sources"]["ego_deranged"]
        reads_scene = bool(s["separated"] and (s["delta_ade"] or 0) > 0)
        reads_ego = bool(e["separated"] and (e["delta_ade"] or 0) > 0)
        blk["verdict"] = ("READS_BOTH" if reads_scene and reads_ego else "ECHOING (ego only)" if reads_ego and not reads_scene
                          else "IGNORES_EGO (scene only)" if reads_scene else "READS_NEITHER")
        res["metrics"][metric] = blk
        print(f"[{metric}] verdict {blk['verdict']}  ade true {blk['ade']['true']:.4f} scene-deranged {blk['ade']['scene_deranged']:.4f} "
              f"(Δ {s['delta_ade']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]) ego-deranged {blk['ade']['ego_deranged']:.4f} "
              f"(Δ {e['delta_ade']:+.4f} [{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}]); ha0 control scene Δ {s['control_ha0_delta']:+.4f}", flush=True)
    res["wallclock_s"] = time.time() - t0
    json.dump(res, open(a0.out, "w", encoding="utf-8"), indent=1, default=float)
    print(f"[done] -> {a0.out} ({res['wallclock_s']:.0f}s)")


if __name__ == "__main__":
    main()
