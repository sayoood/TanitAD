"""A6 DENSE capture (SPEC_ADDENDUM_A6/A7): derived from run_route.py (md5 3e36a379...) -- the model build, hooks, per-window
seeding, forward and capture loop are UNCHANGED; only the window set differs (split `a6_dense`).

a6_dense = EVERY eval window whose GT 6-s heading change is >= 30 deg (SPEC sec.3 GT-turn rule: slot-60 valid, GT path length
>= 5 m, |terminal heading| >= 30 deg) at stride 1, PLUS an equal-count sample (np.random.default_rng(0), without replacement)
of all OTHER windows. The list is written to --window-list on the first pass and READ BACK by the next pass (same set for
every sampler seed).

--- original header ---
refcv7-r101-s0 route-following capture -- ONE pass, ONE window set, ONE inference seed (SPEC.md sec. 2).

The model is built by the launch tree's own eval loader (`tanitad.eval.refcv7_loader.build_model`, STRICT, step
asserted) and every window goes through the trainer's own `compute_losses_v3` in eval mode under no_grad, batch 1,
`torch.manual_seed(seed)` immediately before each window (the reel's forward). NOTHING in the model changes:
every capture is a forward hook or a call-through wrapper that returns the original's value untouched.

Window sets: `eval_diag` / `train_diag` (the map-box diagnostics grid, 8 per episode) and `reel` (every window of
the 12 reel clips, in time order, ids from the reel's clip_selection.json).

Output: <out>/<tag>.npz (per-window arrays), <tag>.packs.pkl (the trainer's detection packs, sha12 only),
<tag>.json (record + identity controls). Env (set by the caller BEFORE import): REFCV6_REPO, REFCV6_KIT,
PYTHONPATH=<tree>/stack, REFCV6_REMAP_OVERRIDES for train_diag. Prints sha12 only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
EXPECT_STEP = 50400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["eval_diag", "train_diag", "reel", "a6_dense"])
    ap.add_argument("--window-list", default="")
    ap.add_argument("--ckpt", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0/ckpt.pt")
    ap.add_argument("--run-dir", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0")
    ap.add_argument("--reel-selection", default="/home/nvidia/refcv7_post/video/clip_selection.json")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--k-per-ep", type=int, default=8)
    ap.add_argument("--max-windows", type=int, default=0)
    a = ap.parse_args()

    import torch
    from tanitad.eval import refcv7_loader as L
    tr = L.trainer()                                  # bootstraps sys.path (scripts/ for refb_labels)
    import refb_labels
    from tanitad.refs import refcv6_selection as v6sel
    from tanitad.refs import refc_select as sl
    from tanitad.data import v7_labels as v7l
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    config = L.load_config(Path(a.run_dir) / "config.json")
    model, cfg, args, mrec = L.build_model(config, a.ckpt, device="cuda", strict=True)
    sd = mrec["state_dict"]
    if int(sd.get("step") or -1) != EXPECT_STEP or sd["missing"] or sd["unexpected"]:
        raise SystemExit(f"[route] bad load: step {sd.get('step')} missing {sd['missing'][:3]} "
                         f"unexpected {sd['unexpected'][:3]}")
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config)
    sha_of_eid = dict(e_ds._vis1_sha12)
    W = int(cfg.core.window)
    if tuple(int(h) for h in cfg.core.trajectory.horizons) != HORIZONS:
        raise SystemExit("[route] horizons drifted")
    print(f"[route] built: step {sd['step']} episodes {len(e_eps)} windows {len(e_ds)} "
          f"build {time.time() - t_start:.0f}s", flush=True)

    # ---- the window set ----------------------------------------------------------------------- #
    by_ep = {}
    for i, (e_i, t) in enumerate(e_ds.index):
        by_ep.setdefault(int(e_i), []).append((int(t), i))
    idx = []
    if a.split == "a6_dense":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import route_metrics as rm
        wl_path = Path(a.window_list)
        by_key = {(sha_of_eid[int(e_ds.episodes[e].episode_id)], t): i for e in by_ep for (t, i) in by_ep[e]}
        if wl_path.exists():
            wl = json.loads(wl_path.read_text(encoding="utf-8"))
            idx = [by_key[(s_, t_)] for s_, t_, _f in wl["windows"]]
            print(f"[route] a6_dense: window list READ BACK from {wl_path} ({len(idx)} windows)", flush=True)
        else:
            t_sel = time.time()
            allw = []                                   # (dataset index, episode, t)
            gts, gvs = [], []
            for e_i in sorted(by_ep):
                ep = e_ds.episodes[e_i]
                P = ep.poses.float()
                Tn = P.shape[0]
                ts_i = sorted(by_ep[e_i])
                tt = torch.tensor([t for t, _i in ts_i], dtype=torch.long)
                pl = P[tt + W - 1]                       # == the item's `pose_last` (V3Dataset: NOW = t + w - 1)
                ix = tt[:, None] + W + torch.arange(60)[None, :]
                fut = P[ix.clamp(max=Tn - 1)]
                fv = (ix <= (Tn - 1)).numpy()
                g = refb_labels.waypoint_targets(pl, fut, HORIZONS).numpy()
                gts.append(g)
                gvs.append(np.stack([fv[:, h - 1] for h in HORIZONS], 1))
                allw += [(i, e_i, t) for t, i in ts_i]
            gt_all = np.concatenate(gts)
            gv_all = np.concatenate(gvs)
            cls, _th = rm.gt_class(gt_all, gv_all)
            is_turn = np.isin(cls, ["turnL", "turnR"])
            n_turn = int(is_turn.sum())
            rest = np.nonzero(~is_turn)[0]
            rng = np.random.default_rng(0)
            pick = np.sort(rng.choice(rest, size=n_turn, replace=False))
            sel = np.sort(np.concatenate([np.nonzero(is_turn)[0], pick]))
            flag = {int(j): ("turn" if is_turn[j] else "rest") for j in sel}
            idx = [allw[j][0] for j in sel]
            wl = {"split": "a6_dense", "n_windows_total_eval": len(allw), "n_turn_windows": n_turn,
                  "n_rest_sampled": int(len(pick)), "rest_sample_seed": 0,
                  "class_counts_total": {c: int((cls == c).sum()) for c in ("turnL", "turnR", "straight", "gentle", "unclassified")},
                  "class_counts_selected": {c: int((cls[sel] == c).sum()) for c in ("turnL", "turnR", "straight", "gentle", "unclassified")},
                  "windows": [[sha_of_eid[int(e_ds.episodes[allw[j][1]].episode_id)], int(allw[j][2]), flag[int(j)]] for j in sel],
                  "select_seconds": round(time.time() - t_sel, 1)}
            wl_path.parent.mkdir(parents=True, exist_ok=True)
            wl_path.write_text(json.dumps(wl), encoding="utf-8")
            np.savez_compressed(wl_path.with_suffix(".gt.npz"), gt=gt_all[sel], gt_valid=gv_all[sel])
            print(f"[route] a6_dense SELECTED: {n_turn} turn windows + {len(pick)} sampled of {len(rest)} others "
                  f"(of {len(allw)} total) in {time.time() - t_sel:.0f}s; list -> {wl_path}", flush=True)
    elif a.split in ("eval_diag", "train_diag"):
        K = int(a.k_per_ep)
        for e_i in sorted(by_ep, key=lambda e: sha_of_eid[int(e_ds.episodes[e].episode_id)]):
            ts = sorted(by_ep[e_i])
            for j in range(K):
                idx.append(ts[int((j + 0.5) * len(ts) / K)][1])
    else:
        sel = json.loads(Path(a.reel_selection).read_text(encoding="utf-8"))
        want = [c["sha12"] for c in sel["chosen"]]
        e_of_sha = {sha_of_eid[int(e_ds.episodes[e].episode_id)]: e for e in by_ep}
        for s in want:
            if s not in e_of_sha:
                raise SystemExit(f"[route] reel clip {s} not in the eval split")
            idx += [i for _t, i in sorted(by_ep[e_of_sha[s]])]
    if a.max_windows:
        idx = idx[:a.max_windows]
    win_sha = [sha_of_eid[int(e_ds.episodes[e_ds.index[i][0]].episode_id)] for i in idx]
    win_t = [int(e_ds.index[i][1]) for i in idx]
    digest = hashlib.sha256(json.dumps(list(zip(win_sha, win_t))).encode()).hexdigest()
    print(f"[route] split {a.split}: {len(idx)} windows over {len(set(win_sha))} episodes; digest {digest[:16]}",
          flush=True)

    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(e_ds, idx), batch_size=1, shuffle=False,
                                     num_workers=a.workers, prefetch_factor=(2 if a.workers > 0 else None))

    # ---- read-only capture -------------------------------------------------------------------- #
    CAP: dict = {}
    dec = model.core.decoder

    def _hook_out(_m, _i, out):
        CAP["n_fwd"] = CAP.get("n_fwd", 0) + 1
        if "out" not in CAP:                          # the FIRST model forward of this window
            CAP["out"] = out
    hooks = [model.register_forward_hook(_hook_out)]

    orig_ag = dec._apply_grafts

    def _ag(base, terms, state, surface, patience):
        res = orig_ag(base, terms, state, surface, patience)
        if "out" in CAP:
            return res
        CAP.setdefault("grafts", {})[surface] = {
            "base": base.detach().float().clone(), "terms": [t.detach().float().clone() for t in terms],
            "res": res[0].detach().float().clone()}
        return res
    dec._apply_grafts = _ag

    orig_nc = v6sel.nav_compliance_prior

    def _nc(cand, nav_cmd, *, tau_rad, stall_m=0.05):
        r = orig_nc(cand, nav_cmd, tau_rad=tau_rad, stall_m=stall_m)
        if "out" in CAP:
            return r
        CAP["navc"] = (r[0].detach().float().clone(), r[1].detach().clone(), float(tau_rad))
        return r
    v6sel.nav_compliance_prior = _nc

    orig_sc = sl.apply_seam_clamp

    def _sc(base, graft, **kw):
        r = orig_sc(base, graft, **kw)
        if "out" in CAP:
            return r
        CAP.setdefault("seam", {})[kw.get("surface")] = {
            "base": base.detach().float().clone(), "graft": graft.detach().float().clone(),
            "res": r[0].detach().float().clone()}
        return r
    sl.apply_seam_clamp = _sc

    orig_cf = v6sel.SpeedCeilingFilter.forward

    def _cf(self, cand, v_limit_ms):
        r = orig_cf(self, cand, v_limit_ms)
        if "out" in CAP:
            return r
        CAP["ceil"] = (r[0].detach().clone(), v_limit_ms.detach().float().clone())
        return r
    v6sel.SpeedCeilingFilter.forward = _cf

    def _mk(name):
        def h(_m, _i, o):
            if "out" not in CAP:
                CAP[name] = o.detach().float().clone()
        return h
    for nm, mod in (("beh", getattr(model, "tac_behaviour_gate_v6", None)),
                    ("tac8_lat", getattr(dec, "tac8_lat_to_anchor", None)),
                    ("tac8_lon", getattr(dec, "tac8_lon_to_anchor", None))):
        if mod is None:
            raise SystemExit(f"[route] expected module {nm} is not built")
        hooks.append(mod.register_forward_hook(_mk(nm)))

    gates = {"navc_gate": float(dec.navc_gate.detach()) if dec.navc_gate is not None else None,
             "navc_tau_rad": float(dec.navc_tau_rad),
             "goal_gate_e9": float(model.goal_gate.detach().reshape(-1)[0]),
             "seam_clamp_decoder": float(dec.sel.seam_clamp),
             "seam_clamp_e9": float(model.cfg.seam_clamp),
             "speed_ceiling_filter": bool(getattr(dec, "speed_ceiling_filter", False)),
             "graft_behaviour_sel": bool(dec.graft_behaviour_sel),
             "refcv7_wta_built": model.refcv7_wta is not None}
    print(f"[route] gates {json.dumps(gates)}", flush=True)

    # ---- the loop ----------------------------------------------------------------------------- #
    keys1 = {}
    packs = {"box3d": [], "agent": []}
    ident = {"max_model_forwards_per_window": 0, "traj_vs_fan_max_abs": 0.0, "n_e9_argmax_mismatch": 0, "n_core_argmax_mismatch": 0,
             "n_core_ceil_absent": 0, "graft_rank_recon_max_abs": 0.0, "e9_recon_max_abs": 0.0}
    terms_shape = {}
    gpu_s = 0.0
    t_loop = time.time()
    for k, b in enumerate(dl):
        CAP.clear()
        torch.cuda.synchronize()
        tg = time.time()
        torch.manual_seed(int(a.seed))
        with torch.no_grad():
            extra = tr.compute_losses_v3(model, b, "cuda", mode=args.mode, ablate_frames=args.ablate_frames)
        torch.cuda.synchronize()
        gpu_s += time.time() - tg
        if "out" not in CAP or "grafts" not in CAP or "seam" not in CAP:
            raise SystemExit(f"[route] capture incomplete at window {k}: {sorted(CAP)}")
        out = CAP["out"]
        ident["max_model_forwards_per_window"] = max(ident["max_model_forwards_per_window"], CAP["n_fwd"])
        fan = out["anchor_traj"][0].detach().float().cpu().numpy()
        traj = out["traj"][0].detach().float().cpu().numpy()
        si = int(out["sel_idx"][0])
        sib = int(out["sel_idx_base"][0])
        ident["traj_vs_fan_max_abs"] = max(ident["traj_vs_fan_max_abs"], float(np.abs(traj - fan[si]).max()))
        s_core = out["sel_score"][0].detach().float().cpu().numpy()
        s_e9 = out["sel_score_v3"][0].detach().float().cpu().numpy()
        reach = out["reach_keep"][0].detach().cpu().numpy().astype(bool)
        # C1: argmax identities
        r_e9 = np.where(reach, s_e9, -np.inf)
        if int(np.argmax(r_e9)) != si:
            ident["n_e9_argmax_mismatch"] += 1
        if "ceil" in CAP:
            ceil_keep = CAP["ceil"][0][0].cpu().numpy().astype(bool)
            v_lim = float(CAP["ceil"][1].reshape(-1)[0])
        else:
            ceil_keep = np.ones_like(reach)
            v_lim = float("inf")
            ident["n_core_ceil_absent"] += 1
        r_core = np.where(reach & ceil_keep, s_core, -np.inf)
        if int(np.argmax(r_core)) != sib:
            ident["n_core_argmax_mismatch"] += 1
        g = CAP["grafts"]
        rk = g["rank"]
        ident["graft_rank_recon_max_abs"] = max(ident["graft_rank_recon_max_abs"],
                                                float((rk["res"][0] - out["sel_score"][0].float()).abs().max()))
        se = CAP["seam"]["goal_sel"]
        ident["e9_recon_max_abs"] = max(ident["e9_recon_max_abs"],
                                        float((se["res"][0] - out["sel_score_v3"][0].float()).abs().max()))
        for surf in ("conf", "refined", "rank"):
            if surf in g:
                terms_shape[surf] = len(g[surf]["terms"])
        item_nav = int(b["nav_cmd"][0])
        pl = b["pose_last"].float()
        fut = b["future_poses_ext"].float()
        fv = b["future_valid_ext"][0].bool().numpy()
        gt_slots = refb_labels.waypoint_targets(pl, fut, HORIZONS)[0].numpy()
        slot_valid = np.array([bool(fv[h - 1]) for h in HORIZONS])
        bs = out["perception"]["box_slots"]
        rec = {
            "fan": fan.astype(np.float32), "traj": traj.astype(np.float32),
            "sel_idx": si, "sel_idx_base": sib, "s_core": s_core, "s_e9": s_e9, "reach": reach,
            "ceil_keep": ceil_keep, "v_lim": v_lim,
            "conf_base": g["conf"]["base"][0].cpu().numpy() if "conf" in g else np.full(fan.shape[0], np.nan),
            "refined_base": g["refined"]["base"][0].cpu().numpy() if "refined" in g else np.full(fan.shape[0], np.nan),
            "refined_res": g["refined"]["res"][0].cpu().numpy() if "refined" in g else np.full(fan.shape[0], np.nan),
            "rank_base": rk["base"][0].cpu().numpy(), "rank_res": rk["res"][0].cpu().numpy(),
            "rank_terms": np.stack([t[0].cpu().numpy() for t in rk["terms"]]) if rk["terms"] else
            np.zeros((0, fan.shape[0]), np.float32),
            "e9_base": se["base"][0].cpu().numpy(), "e9_graft": se["graft"][0].cpu().numpy(),
            "navc": (CAP["navc"][0][0].cpu().numpy() if "navc" in CAP else np.full(fan.shape[0], np.nan)),
            "navc_mask": (CAP["navc"][1][0].cpu().numpy().astype(bool) if "navc" in CAP
                          else np.zeros(fan.shape[0], bool)),
            "beh": CAP["beh"][0].cpu().numpy() if "beh" in CAP else np.full(fan.shape[0], np.nan),
            "tac8_lat": CAP["tac8_lat"][0].cpu().numpy() if "tac8_lat" in CAP else np.full(fan.shape[0], np.nan),
            "tac8_lon": CAP["tac8_lon"][0].cpu().numpy() if "tac8_lon" in CAP else np.full(fan.shape[0], np.nan),
            "p_lat": out["tacv6_lat_logits"][0].float().softmax(-1).cpu().numpy(),
            "p_lon": out["tacv6_lon_logits"][0].float().softmax(-1).cpu().numpy(),
            "p_goal": out["tacv6_goal_logits"][0].float().sigmoid().cpu().numpy(),
            "goal_point": out["goal_point_tac"][0].detach().float().cpu().numpy(),
            # ADDITIVE 2026-10-04 (after eval_s0/train_s0 started): the tactical goal head at every tau (2/4/6 s)
            "g_tac": (out["g_tac"][0].detach().float().cpu().numpy() if "g_tac" in out
                      else np.full((3, 2), np.nan, np.float32)),
            # ADDITIVE 2026-10-04: the residual prior the plan is composed on (refcv7 NEW-1, ha0_ext_pose)
            "prior_path": (out["residual_prior_path"][0].detach().float().cpu().numpy().reshape(-1, 2)
                           if "residual_prior_path" in out else np.full((8, 2), np.nan, np.float32)),
            "goal_dist": (out["goal_dist"][0].detach().float().cpu().numpy().reshape(-1) if "goal_dist" in out
                          else np.full(1, np.nan, np.float32)),
            "gt": gt_slots.astype(np.float32), "gt_valid": slot_valid,
            "lat_gt": int(b["lat_v7"][0]), "lon_gt": int(b["lon_v7"][0]),
            "nav": item_nav, "nav_valid": bool(b["nav_valid"][0]) if "nav_valid" in b else True,
            "v0": float(pl[0, 3]), "v_max_raw": float(b["v_max_ms"][0]), "v_max_valid": float(b["v_max_valid"][0]),
            "box_logit": bs["presence_logit"][0].float().cpu().numpy(),
            "box_xylw": bs["box"][0].float().cpu().numpy(),
            "box_yaw": bs["yaw"][0].float().cpu().numpy().reshape(-1),
            "box_cls": bs["cls_logits"][0].float().argmax(-1).cpu().numpy().astype(np.int16),
            "agent_label": bool(b["agent_label"][0]),
        }
        for kk, v in rec.items():
            keys1.setdefault(kk, []).append(v)
        for hd in ("box3d", "agent"):
            for pk in extra.get(f"_det_pack_{hd}") or []:
                pk = dict(pk)
                pk["sha12"] = sha_of_eid.get(int(pk["ep"])) if pk.get("ep") is not None else None
                pk.pop("ep", None)
                pk["win"] = k
                packs[hd].append(pk)
        if k % 100 == 0 or k in (9, 49) or k == len(idx) - 1:
            el = time.time() - t_loop
            print(f"[route] ZZ{k + 1}-{len(idx)}ZZ {el:.0f}s gpu {gpu_s:.0f}s "
                  f"cuda_max {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        del extra, out
        CAP.clear()
    for h in hooks:
        h.remove()
    dec._apply_grafts = orig_ag
    v6sel.nav_compliance_prior = orig_nc
    sl.apply_seam_clamp = orig_sc
    v6sel.SpeedCeilingFilter.forward = orig_cf

    arrs = {}
    for kk, v in keys1.items():
        try:
            arrs[kk] = np.stack([np.asarray(x) for x in v])
        except ValueError:
            arrs[kk] = np.asarray(v, dtype=object)
    arrs["win_sha12"] = np.asarray(win_sha)
    arrs["win_t"] = np.asarray(win_t)
    np.savez_compressed(out_dir / f"{a.tag}.npz", **arrs)
    with open(out_dir / f"{a.tag}.packs.pkl", "wb") as fh:
        pickle.dump(packs, fh, protocol=4)
    names = {"tac_lat": list(v7l.HEADS["tac_lat"]), "tac_lon": list(v7l.HEADS["tac_lon"]),
             "goal": list(v7l.TAC_GOAL_TOKENS)}
    rec = {"tag": a.tag, "split": a.split, "seed": a.seed, "ckpt": a.ckpt, "step": sd["step"],
           "n_windows": len(idx), "n_episodes": len(set(win_sha)),
           "window_sha12_t_sha256": digest, "gates": gates, "terms_per_surface": terms_shape,
           "identity": ident, "names": names, "nav_commands": ["follow", "left", "right", "straight"],
           "n_packs": {k_: len(v) for k_, v in packs.items()},
           "loader_departures": mrec.get("departures"),
           "state_dict": {k_: sd[k_] for k_ in ("strict", "missing", "unexpected", "step")},
           "compute": {"gpu_forward_s": round(gpu_s, 1), "wall_s": round(time.time() - t_start, 1),
                       "loop_s": round(time.time() - t_loop, 1),
                       "cuda_max_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                       "device": torch.cuda.get_device_name(0)},
           "ignore_id": int(v7l.IGNORE_ID)}
    (out_dir / f"{a.tag}.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(f"[route] DONE {a.tag}: {json.dumps(ident)} gpu {gpu_s:.0f}s wall {time.time() - t_start:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
