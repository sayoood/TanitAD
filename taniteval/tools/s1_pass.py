#!/usr/bin/env python3
"""s1_pass.py -- PREREG_S1's dev-box arms (S1 AMENDMENT + ERRATA, 2026-09-19): the data path,
and the checkpoint-free HUMAN ROUND-TRIP control (S1A.6).

``--mode roundtrip`` (NO checkpoint, no frames decoded except a 3-window equivalence probe):
  1. rebuilds the run's config from its ``config.json`` through the TRAINER's own build path
     (``refcv3_arm.rebuild_config``), with no weights;
  2. builds the trainer's window dataset over the held-out cache (``refcv3_arm.build_corpus``) and
     ASSERTS ``len(ds)`` equals the count the trainer printed (A8/A3 ``train.log``: *"held-out
     eval: 62 episodes -> 10600 windows"*). This is a known value: if it differs, these are
     not the trainer's windows;
  3. draws the trainer's own 1,000 (``torch.Generator().manual_seed(12345)``,
     ``randperm(len(ds))[:eval_batches*batch]``);
  4. applies item 19's eligibility rules (``ddv2_rl_refcv5.Ctx._select``: t0 >= 1, a full
     ROUTE_TICKS future, agent labels on every raw frame t0..t0+AGENT_TICKS-1) and counts drops
     with their reasons;
  5. builds each window's human, route, recorded tracks and SAM3 map exactly as item 19's
     ``Ctx.fetch`` does. ⭐ This is PROVEN rather than asserted: on the first 3 eligible windows,
     the light (pose-only) item is compared field by field with ``Ctx.fetch`` itself, called
     unbound;
  6. scores the human TWO ways: directly, and cut to the fan's slots and re-expanded by S1A.2's
     spline. The pre-registered bar is **NC and DAC agreement on >= 99 % of eligible windows**,
     and the residual is printed as the waypoint representation's own error floor.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import sys
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "taniteval" / "tools", REPO / "stack" / "scripts", REPO / "stack"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import refcv3_arm as ARM                    # noqa: E402  the trainer-faithful corpus + config
import ddv2_rl_refcv5 as DD                 # noqa: E402  item 19's own module
import s1_gate as S                         # noqa: E402
from tanitad.rl import pdm_proxy as P       # noqa: E402

ROUNDTRIP_BAR = 0.99                        # S1A.6, fixed before data


def sha12(x) -> str:
    return hashlib.sha256(str(x).encode()).hexdigest()[:12]


def trainer_windows(ds, n: int, seed: int = 12345) -> list[int]:
    """The trainer's OWN held-out draw (``refc_v3_train.py``: ``g_ev``)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(len(ds), generator=g)[:n].tolist()


class Corpus:
    def __init__(self, config_path: str, cache: str, labels: str, agents: str,
                 maps: str | None, lru: int = 2):
        self.config_path, self.cache = config_path, cache
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.cfg, self.targs, self.cfg_src = ARM.rebuild_config(self.config)
        ns = types.SimpleNamespace(episodes=cache, lru=lru, episodes_n=0, labels=labels,
                                   nav_source="none")
        (self.eps, _, self.clip_ids, self.ds, _, self.join, _,
         self.raw_off) = ARM.build_corpus(ns, self.cfg, {})
        self.W = int(self.cfg.core.window)
        self.horizons = tuple(int(h) for h in self.cfg.core.trajectory.horizons)
        # item 19's OWN loaders, called unbound on a shim: same code, not a copy
        self.shim = types.SimpleNamespace(clip_ids=self.clip_ids, map_root=maps,
                                          map_store={} if maps else None)
        if maps:
            from tanitad.data import semantic_map_gt as _smg
            self.shim._smg = _smg
        self.agents = DD.Ctx._load_agents(self.shim, agents)
        self.shim.agents = self.agents
        self.map_for = functools.partial(DD.Ctx.map_for, self.shim)
        self._poses = {}
        import refb_labels
        self.rb = refb_labels

    def poses(self, e_i: int) -> torch.Tensor:
        """Per-episode pose cache: the 1,000 windows arrive in RANDOM episode order, and
        reading through a small provider LRU would reload an episode on nearly every window."""
        if e_i not in self._poses:
            self._poses[e_i] = self.eps[e_i].poses.float().clone()
        return self._poses[e_i]

    def eligibility(self, wi: int):
        e_i, t = self.ds.index[wi]
        t0 = t + self.W - 1
        cid = str(self.clip_ids[e_i])
        n_prov = int(self.poses(e_i).shape[0])
        if t0 < 1:
            return "t0"
        if t0 + DD.ROUTE_TICKS > n_prov - 1:
            return "future"
        r0 = t0 + self.raw_off
        if any((cid, r0 + k) not in self.agents for k in range(DD.AGENT_TICKS)):
            return "agents"
        return None

    def light_item(self, wi: int) -> dict:
        """``Ctx.fetch`` minus frames / ego_state / nav: the pose-derived half, identical by
        construction and PROVEN identical by ``fetch_equivalence``."""
        e_i, t = self.ds.index[wi]
        t0 = t + self.W - 1
        cid = str(self.clip_ids[e_i])
        pa = self.poses(e_i)
        pose_last = pa[t0]
        fut = pa[t0 + 1:t0 + 1 + DD.ROUTE_TICKS]
        human = P.ego_states_from_poses(pose_last[None], fut[None])[0]
        rst = P.ego_states_from_poses(pose_last[None], fut[None],
                                      P.ProxyConfig(n_ticks=DD.ROUTE_TICKS))[0]
        rcx = rst[:, 0] + P.PROXY.rear_axle_to_center * torch.cos(rst[:, 2])
        rcy = rst[:, 1] + P.PROXY.rear_axle_to_center * torch.sin(rst[:, 2])
        r0 = t0 + self.raw_off
        tracks = P.AgentTracks.from_frames(
            [self.agents[(cid, r0 + k)] for k in range(DD.AGENT_TICKS)],
            pa[t0:t0 + DD.AGENT_TICKS])
        dmap = self.map_for(cid, r0)
        return {"pose_last": pose_last, "human": human, "fut": fut,
                "route": torch.stack([rcx, rcy], dim=-1), "tracks": tracks,
                "gt_wp": self.rb.waypoint_targets(pose_last[None], fut[None], self.horizons)[0],
                "map_drivable": None if dmap is None else dmap[0],
                "map_seen": None if dmap is None else dmap[1],
                "sha12": sha12(cid), "t0": int(t0)}

    def fetch_equivalence(self, wis: list[int]) -> dict:
        """Item 19's ``Ctx.fetch``, called UNBOUND on a shim carrying exactly what it reads,
        against ``light_item``, field by field. Decodes frames for these windows only."""
        from tanitad.refs import refc_v3
        shim = types.SimpleNamespace(ds=self.ds, W=self.W, clip_ids=self.clip_ids,
                                     eps=self.eps, rb=self.rb, agents=self.agents,
                                     raw_off=self.raw_off, v3=refc_v3, map_for=self.map_for,
                                     horizons=self.horizons)
        rep = []
        # `fetch` reads the nav id through `ds._nav_by_sid`, which only a v7.2-nav build
        # populates. Nav is NOT a compared field (the round-trip runs no model), so an empty
        # map stands in for the probe and the original value is restored.
        nav_saved = getattr(self.ds, "_nav_by_sid", None)
        if nav_saved is None:
            self.ds._nav_by_sid = {}
        try:
            refs = {wi: DD.Ctx.fetch(shim, (wi, *self.ds.index[wi])) for wi in wis}
        finally:
            self.ds._nav_by_sid = nav_saved
        for wi in wis:
            ref = refs[wi]
            got = self.light_item(wi)
            ok = {
                "human": bool(torch.allclose(ref["human"], got["human"], atol=1e-5)),
                "route": bool(torch.allclose(ref["route"], got["route"], atol=1e-5)),
                "tracks_xy": bool(torch.allclose(ref["tracks"].xy, got["tracks"].xy, atol=1e-5)),
                "tracks_valid": bool(torch.equal(ref["tracks"].valid, got["tracks"].valid)),
                "tracks_static": bool(torch.equal(ref["tracks"].static, got["tracks"].static)),
                "gt_wp": bool(torch.allclose(ref["gt_wp"], got["gt_wp"], atol=1e-5)),
                "map": ((ref["map_drivable"] is None) == (got["map_drivable"] is None)) and (
                    ref["map_drivable"] is None
                    or bool(torch.equal(ref["map_drivable"], got["map_drivable"]))),
                "t0": ref["t0"] == got["t0"], "sha12": ref["sha12"] == got["sha12"],
                "v0": abs(float(ref["v0"]) - float(got["pose_last"][3])) < 1e-6}
            rep.append({"sha12": got["sha12"], "t0": got["t0"], **ok})
        return {"windows": rep, "all_equal": all(all(v for k, v in r.items()
                                                     if k not in ("sha12", "t0"))
                                                 for r in rep)}


def _dac(states, it):
    if it["map_drivable"] is None:
        return None
    return P.dac_from_drivable(states, it["map_drivable"], it["map_seen"])


def roundtrip(corp: Corpus, wis: list[int]) -> dict:
    rows = []
    for wi in wis:
        it = corp.light_item(wi)
        rt = S.candidate_states(it["gt_wp"][None].numpy(), corp.horizons, it["pose_last"])
        dac_h = _dac(it["human"][None], it)
        out = P.score_candidates(rt, it["human"], it["tracks"], it["route"],
                                 dac_cand=_dac(rt, it),
                                 dac_human=None if dac_h is None else dac_h[0])
        h = out["human"]
        rows.append({"sha12": it["sha12"], "t0": it["t0"],
                     "nc_h": h["nc"], "nc_rt": float(out["nc"][0]),
                     "dac_h": h["dac"], "dac_rt": float(out["dac"][0]),
                     "pdms_h": h["pdms"], "pdms_rt": float(out["pdms"][0]),
                     "max_pos_err_m": float((rt[0, :, :2] - it["human"][:, :2]).norm(dim=-1).max()),
                     "has_map": it["map_drivable"] is not None})
    n = len(rows)
    nc_eq = sum(r["nc_h"] == r["nc_rt"] for r in rows)
    dac_eq = sum(r["dac_h"] == r["dac_rt"] for r in rows)
    both = sum(r["nc_h"] == r["nc_rt"] and r["dac_h"] == r["dac_rt"] for r in rows)
    pe = np.array([r["max_pos_err_m"] for r in rows])
    dp = np.array([abs(r["pdms_h"] - r["pdms_rt"]) for r in rows])
    return {"n": n, "nc_agree": nc_eq / n, "dac_agree": dac_eq / n, "both_agree": both / n,
            "bar": ROUNDTRIP_BAR, "passed": both / n >= ROUNDTRIP_BAR,
            "human_nc_ne_1_frac": sum(r["nc_h"] != 1 for r in rows) / n,
            "has_map_frac": sum(r["has_map"] for r in rows) / n,
            "max_pos_err_m": {"median": float(np.median(pe)), "p95": float(np.quantile(pe, 0.95)),
                              "max": float(pe.max())},
            "abs_pdms_diff": {"median": float(np.median(dp)), "p95": float(np.quantile(dp, 0.95)),
                              "max": float(dp.max())},
            "rows": rows}


# ============================================================================ the model pass
BOX_THRESH_M = 2.0                                     # S1A.11: box3d_ap's default, BEV


def attach_agent_gt(corp: Corpus, targs) -> dict:
    """The trainer's OWN held-out GT calls, in its order (``refc_v3_train.py`` ~6524-6590):
    the reader with rates AND track ids, ``enable_agent_join`` at the TRAIN pad, then the 3-D
    widening by track id. After this, ``ds[wi]`` carries exactly the ``agent_*`` fields the
    trainer's eval batches carry."""
    tr = ARM.trainer()
    from train_p8_occupancy import JoinFileReader as _JFR
    from tanitad.data import agent_cuboid_gt as _cub
    rd = _JFR(targs.agent_join, episode_ids={int(e.episode_id) for e in corp.eps},
              with_rates=not bool(getattr(targs, "agent_join_no_rates", False)),
              with_track_ids=bool(getattr(targs, "join3d", None)))
    st = corp.ds.enable_agent_join(rd, pad=int(targs.agent_pad), allow_legacy_ids=bool(
        getattr(targs, "agent_join_allow_legacy_ids", False)))
    rec = {"agent_join": st, "pad": int(targs.agent_pad)}
    if getattr(targs, "join3d", None):
        clip_tab, n_stack = tr._clip_table_for_caches([corp.cache])
        corp.ds.map_clip_of_ep, corp.ds.map_n_stack = clip_tab, n_stack
        rec["join3d"] = corp.ds.enable_join3d(
            _cub.open_join3d(targs.join3d, clips=set(clip_tab.values())))
    return rec


class ModelPass:
    """ONE eval-mode forward per window, called exactly as ``refcv3_arm``'s own forward
    (``refcv3_arm.py`` ~2150-2240): the trainer's frame ingest, v0 = the measured
    ``pose_last[3]``, the per-clip lift geometry keyed on ``stable_episode_id``, the
    ego block only where the build consumes it. Nav = the dataset item's OWN ``nav_cmd``,
    which is the v1 derivation because ``nav_from_v7`` is off (S1A.10, asserted)."""

    def __init__(self, corp: Corpus, ckpt: str, device: str = "cpu"):
        self.corp, self.dev = corp, device
        self.model, self.cfg, self.targs, self.prov = ARM.load_model(
            ckpt, corp.config_path, device, False)
        self.model.eval()
        self.tr = ARM.trainer()
        self.steps = int(self.prov["decoder_steps"])
        self.feed_ego = bool(getattr(self.cfg, "ego_state_inject", False))
        perc = getattr(self.model, "_perception", None)
        self.box = getattr(perc, "box_dec", None)
        if self.box is None:
            raise SystemExit("⛔ this checkpoint has no box head: PRED and the box read "
                             "cannot run")
        from tanitad.models.agent_slots import AGENT_CLASSES
        self.classes = AGENT_CLASSES
        if bool(getattr(corp.ds, "nav_from_v7", False)):
            raise SystemExit("⛔ the dataset feeds v7.2 nav; S1A.10 pre-registered v1")

    def forward(self, item: dict, clip_id: str) -> dict:
        from tanitad.refs import refc_v3 as v3
        from tanitad.data.v2_dataset import stable_episode_id
        fr = self.tr.frames_to_device(item["frames"][None], self.dev)
        v0 = item["pose_last"][3:4].float().to(self.dev)
        nav = item["nav_cmd"].reshape(1).long().to(self.dev)
        es = None
        if self.feed_ego:
            es = v3.ego_state_from_batch({"pose_last": item["pose_last"].float()[None],
                                          "actions": item["actions"].float()[None]},
                                         device=self.dev)
        geom = {}
        lb = getattr(self.model, "_lift_bank", None)
        if lb is not None:
            pg, pv = lb.for_episodes([int(stable_episode_id(clip_id))], device=self.dev)
            geom = {"perception_grid": pg, "perception_valid": pv}
        with torch.no_grad():
            return self.model(fr, nav_cmd=nav, v0=v0, steps=self.steps, ego_state=es, **geom)


def pass_window(corp: Corpus, mp: ModelPass, wi: int) -> dict:
    from tanitad.models.box3d_head import box3d_match_rows
    from tanitad.refs import refc_tactical as tac
    import refb_labels as rb
    item = corp.ds[wi]
    e_i, _t = corp.ds.index[wi]
    cid = str(corp.clip_ids[e_i])
    li = corp.light_item(wi)
    pl = li["pose_last"]
    if not torch.allclose(item["pose_last"].float(), pl, atol=1e-5):
        raise SystemExit("⛔ the dataset item's pose_last is not item 19's: index contract")
    out = mp.forward(item, cid)
    fan = out["anchor_traj"][0].float().cpu()
    rank = (out["sel_score_v3"] if "sel_score_v3" in out else out["sel_score"])[0].float().cpu()
    keep = out.get("reach_keep")
    keep = None if keep is None else keep[0].cpu()
    states = S.candidate_states(fan.numpy(), corp.horizons, pl)
    dac_c = _dac(states, li)
    dac_h = _dac(li["human"][None], li)
    dec = {k: v[0].float().cpu() for k, v in out["perception"]["box_slots"].items()
           if torch.is_tensor(v) and v.dim() >= 2}
    gate = {"GATE_ORACLE": li["tracks"], "GATE_CONST": S.empty_tracks(pl),
            "GATE_PRED": S.pred_tracks(dec, pl, mp.classes),
            "ORACLE_CV": S.cv_tracks_from_recorded(li["tracks"], pl)}
    m = S.evaluate_window(states, rank, keep, li["human"], li["route"], li["tracks"], gate,
                          dac_c, None if dac_h is None else dac_h[0])
    # families on every arm's pick (RANDOM has no pick; its families are not defined)
    world = S.candidate_world_poses(fan.numpy(), corp.horizons, pl, P.PROXY.n_ticks, P.PROXY.dt)
    hz = tac.LABEL_HORIZON
    h_lat, h_lon = tac.window_factored_labels(pl[None], li["fut"][None, :hz])
    nav = int(item["nav_cmd"])
    for arm, v in m.items():
        if arm.startswith("_") or "idx" not in v:
            continue
        k = v["idx"]
        v.update(S.pick_families(states[k], li["human"]))
        c_lat, c_lon = tac.window_factored_labels(pl[None], world[k:k + 1, :hz])
        v["tac_lat_agree"] = float(c_lat[0] == h_lat[0])
        v["tac_lon_agree"] = float(c_lon[0] == h_lon[0])
        v["nc_half"] = float(v["nc"] == 0.5)
        if nav in (rb.NAV_LEFT, rb.NAV_RIGHT):          # STRATEGIC, reported PAUSED
            dyaw = float(states[k, -1, 2])
            v["strat_comply"] = float(dyaw > 0 if nav == rb.NAV_LEFT else dyaw < 0)
    # the box read (S1A.4/S1A.11): the trainer's own GT fields on this window
    box = {"labelled": bool(item.get("agent_label", torch.tensor(False)))}
    if box["labelled"]:
        tgt = {"box": item["agent_box"][None].float(), "valid": item["agent_valid"][None],
               "cz": item.get("agent_cz", torch.zeros(item["agent_valid"].shape))[None].float()}
        per, ngt = box3d_match_rows({k: v[None] for k, v in dec.items()}, tgt,
                                    dist_thresh_m=BOX_THRESH_M, use_z=False)
        rates, rmask = item["agent_rates"].float(), item["agent_rates_mask"].bool()
        err, floor = [], []
        for conf, hit, i, j in per[0]:
            if hit and bool(rmask[j]):
                vg = rates[j, :2]
                err.append(float((dec["rates"][i, :2] - vg).norm()))
                floor.append(float(vg.norm()))
        box.update({"rows": [[round(r[0], 6), r[1]] for r in per[0]], "n_gt": ngt[0],
                    "n_pred": len(per[0]), "vel_err": err, "vel_floor": floor})
    return {"sha12": li["sha12"], "t0": li["t0"], "ep": sha12(cid), "nav": nav,
            "model_sel_idx": int(out["sel_idx"][0]), "arms": m, "box": box,
            "fan_hash": hashlib.sha256(fan.numpy().tobytes()).hexdigest()[:16]}


# ============================================================================ the verdict
def _ci():
    from taniteval.ci import episode_cluster_bootstrap, paired_episode_cluster_bootstrap
    return episode_cluster_bootstrap, paired_episode_cluster_bootstrap


def analyze(rows: list[dict], ranges: tuple[float, float], n_boot: int = 2000) -> dict:
    """S1A.7 / S1A.11 exactly as pre-registered: per-arm rates, paired deltas vs BASE, the
    quotable share of the ceiling, the box read's two PASS tests, and the controls."""
    from tanitad.models.box3d_head import ap_from_rows, random_ap_base_rate
    ecb, pecb = _ci()
    eid = [r["ep"] for r in rows]
    n = len(rows)
    arms = ("BASE", "RANDOM", "GATE_ORACLE", "GATE_CONST", "GATE_PRED", "ORACLE_CV")
    col = {a: np.array([r["arms"][a]["collided"] for r in rows]) for a in arms}
    res = {"n_windows": n, "n_episodes": len(set(eid)),
           "collided_rate": {a: float(col[a].mean()) for a in arms},
           "tier": {a: S.TIER[a] for a in arms}}
    res["paired_vs_BASE"] = {a: pecb(col[a], col["BASE"], eid, n_boot=n_boot)
                             for a in arms if a != "BASE"}
    # the quotable share of the ceiling, with its interval: (pred - base) / (oracle - base)
    base, orc, prd = col["BASE"], col["GATE_ORACLE"], col["GATE_PRED"]
    cat = np.concatenate([base, orc, prd])

    def share(idx):
        ii = np.asarray(idx).astype(np.int64)
        b, o, p = cat[ii].mean(), cat[ii + n].mean(), cat[ii + 2 * n].mean()
        return float((p - b) / (o - b)) if o != b else float("nan")
    res["share_of_ceiling"] = ecb(np.arange(n, dtype=np.float64), eid, reduce=share,
                                  n_boot=n_boot)
    # controls that must read known values
    res["controls"] = {
        "base_equals_model_pick": all(r["arms"]["BASE"]["idx"] == r["model_sel_idx"]
                                      for r in rows),
        "const_recovery_exactly_0": all(r["arms"]["GATE_CONST"]["idx"]
                                        == r["arms"]["BASE"]["idx"] for r in rows),
        "random_n_candidates": sorted({r["arms"]["RANDOM"]["n_candidates"] for r in rows}),
        "fan_collision_free_share": float(np.mean([r["arms"]["_fan"]["collision_free_share"]
                                                   for r in rows])),
        "oracle_changed_a_pick": any(r["arms"]["GATE_ORACLE"]["idx"] != r["arms"]["BASE"]["idx"]
                                     for r in rows)}
    # the box read
    lab = [i for i, r in enumerate(rows) if r["box"]["labelled"]]
    xr, yr = ranges
    if lab:
        n_gt = sum(rows[i]["box"]["n_gt"] for i in lab)
        rows_all = [tuple(x) for i in lab for x in rows[i]["box"]["rows"]]
        ap = ap_from_rows(rows_all, n_gt)["ap"]
        w = np.array([rows[i]["box"]["n_pred"] for i in lab], dtype=np.float64)
        rates = np.array([random_ap_base_rate(rows[i]["box"]["n_gt"], rows[i]["box"]["n_pred"],
                                              BOX_THRESH_M, (xr, 2 * yr, 0.0)) for i in lab])
        base_rate = float((w * rates).sum() / max(w.sum(), 1.0))
        lab_rows = [rows[i] for i in lab]

        def ap_red(idx):
            sel = [lab_rows[int(k)] for k in idx]
            return ap_from_rows([tuple(x) for r in sel for x in r["box"]["rows"]],
                                sum(r["box"]["n_gt"] for r in sel))["ap"]
        ap_ci = ecb(np.arange(len(lab), dtype=np.float64), [r["ep"] for r in lab_rows],
                    reduce=ap_red, n_boot=n_boot)
        e = np.array([sum(r["box"]["vel_err"]) for r in lab_rows])
        f = np.array([sum(r["box"]["vel_floor"]) for r in lab_rows])
        c = np.array([len(r["box"]["vel_err"]) for r in lab_rows], dtype=np.float64)
        cat2 = np.concatenate([f, e, c])
        m2 = len(lab_rows)

        def vgain(idx):
            ii = np.asarray(idx).astype(np.int64)
            cc = cat2[ii + 2 * m2].sum()
            return float((cat2[ii].sum() - cat2[ii + m2].sum()) / cc) if cc > 0 else float("nan")
        v_ci = ecb(np.arange(m2, dtype=np.float64), [r["ep"] for r in lab_rows], reduce=vgain,
                   n_boot=n_boot)
        res["box_read"] = {
            "labelled_windows": len(lab), "unlabelled_windows": n - len(lab), "n_gt": n_gt,
            "ap_bev": ap, "ap_ci": ap_ci, "random_base_rate": base_rate,
            "ap_pass": bool(ap_ci["lo"] > base_rate),
            "vel_pairs": int(c.sum()), "vel_mae_pred": float(e.sum() / max(c.sum(), 1)),
            "vel_mae_zero_floor": float(f.sum() / max(c.sum(), 1)),
            "vel_gain_ci": v_ci, "vel_pass": bool(v_ci["lo"] > 0),
            "extent_m": [xr, 2 * yr], "dist_thresh_m": BOX_THRESH_M}
        pred_quotable = res["box_read"]["ap_pass"] and res["box_read"]["vel_pass"]
    else:
        res["box_read"] = {"labelled_windows": 0, "note": "no labelled window"}
        pred_quotable = False
    # families per arm (means over windows; curvature over its masked set only)
    fam = {}
    for a in arms:
        if a == "RANDOM":
            continue
        d = {}
        for key in ("along_2s", "cross_2s", "along_4s", "cross_4s", "speed_err_4s",
                    "heading_err_4s", "curv_err_2s", "curv_floor_2s", "tac_lat_agree",
                    "tac_lon_agree", "strat_comply", "dac", "ep", "ttc", "comfort", "pdms",
                    "nc_half"):
            vals = [r["arms"][a].get(key) for r in rows]
            vals = [x for x in vals if x is not None]
            d[key] = {"mean": float(np.mean(vals)) if vals else None, "n": len(vals)}
        fam[a] = d
    res["families"] = fam
    # the verdict (S1A.7), and the two sentences it must carry
    dp = res["paired_vs_BASE"]["GATE_PRED"]
    harm = []
    # ⛔ EVERY S1A.8 family, not only the PDMS sub-scores. STRATEGIC is evaluated and
    # reported but PAUSED (PI item 25): never counted, never blocking.
    worse = {"pdms": "lower", "dac": "lower", "ttc": "lower", "ep": "lower",
             "comfort": "lower", "tac_lat_agree": "lower", "tac_lon_agree": "lower",
             "along_2s": "higher", "cross_2s": "higher", "along_4s": "higher",
             "cross_4s": "higher", "speed_err_4s": "higher", "heading_err_4s": "higher",
             "curv_err_2s": "higher"}
    for key, worse_if in worse.items():
        keep_w = [i for i, r in enumerate(rows)
                  if r["arms"]["GATE_PRED"].get(key) is not None
                  and r["arms"]["BASE"].get(key) is not None]
        if not keep_w:
            continue
        a_ = np.array([rows[i]["arms"]["GATE_PRED"][key] for i in keep_w], dtype=np.float64)
        b_ = np.array([rows[i]["arms"]["BASE"][key] for i in keep_w], dtype=np.float64)
        dd = pecb(a_, b_, [eid[i] for i in keep_w], n_boot=n_boot)
        res.setdefault("harm_checks", {})[key] = dd
        if dd["separated"] and ((dd["delta"] < 0) if worse_if == "lower" else (dd["delta"] > 0)):
            harm.append(key)
    sc = [r["arms"]["GATE_PRED"].get("strat_comply") for r in rows]
    sb = [r["arms"]["BASE"].get("strat_comply") for r in rows]
    res["strategic_PAUSED"] = {
        "pred": float(np.mean([x for x in sc if x is not None])) if any(
            x is not None for x in sc) else None,
        "base": float(np.mean([x for x in sb if x is not None])) if any(
            x is not None for x in sb) else None,
        "n_turn_windows": sum(x is not None for x in sc),
        "ruling": "PI item 25: evaluated, reported PAUSED, never counted as a pass"}
    if not res["controls"]["base_equals_model_pick"] or \
            not res["controls"]["const_recovery_exactly_0"]:
        verdict = "VOID (a known-value control failed)"
    elif not pred_quotable:
        verdict = ("PRED NOT QUOTABLE -- the box read failed its pre-registered bar; "
                   "the next lever is box-head quality (PREREG_S1 §10)")
    elif harm:
        verdict = "FAIL-HARM (%s separated-worse)" % ", ".join(harm)
    elif dp["separated"] and dp["delta"] < 0:
        verdict = "SUPPORTED (dev-box scale)"
    else:
        verdict = "REFUTED as configured (collided-selection rate did not separate)"
    sh = res["share_of_ceiling"]["mean"]
    if isinstance(sh, float) and not math.isnan(sh) and sh < 0.10:
        verdict += (" | recovery %.1f %% < 10 %% of the ceiling: PREREG_S1 §10's "
                    "pre-committed reading -- the next work is perception quality" % (100 * sh))
    res["verdict"] = verdict
    res["verdict_text"] = [
        "Internal validity holds (gate vs base on the same fan); EXTERNAL validity to a "
        "trained planner does NOT: A8's selector is weak (A3 measured anchor_acc 0.092 vs "
        "chance 1/128 = 0.0078).",
        "nav = v1, the trained derivation; on PhysicalAI it is derived from the ego's own "
        "future path, so it is optimistic by construction.",
        "The waypoint representation's own floor (S1A.6 round-trip) is ~1 % of windows: "
        "absolute collision rates are +-1 %, between-arm deltas are unaffected."]
    return res


def _windows(corp, a):
    if len(corp.ds) != a.expect_windows:
        raise SystemExit(f"⛔ len(ds) = {len(corp.ds)} != the trainer's {a.expect_windows}: "
                         f"these are NOT the trainer's held-out windows")
    wis = trainer_windows(corp.ds, a.n)
    drops, elig = Counter(), []
    for wi in wis:
        why = corp.eligibility(wi)
        if why:
            drops[why] += 1
        else:
            elig.append(wi)
    return wis, elig, drops


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("roundtrip", "pass", "analyze"), default="roundtrip")
    ap.add_argument("--config", help="the run's config.json")
    ap.add_argument("--cache", help="the held-out v2 cache (halfB)")
    ap.add_argument("--labels")
    ap.add_argument("--agents")
    ap.add_argument("--maps", default=None)
    ap.add_argument("--expect-windows", type=int,
                    help="the trainer's printed held-out window count (known value)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True, help="roundtrip/verdict JSON")
    ap.add_argument("--ckpt", help="pass: the checkpoint (A8 5k)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rows", help="pass/analyze: per-window JSONL, appended as it goes")
    ap.add_argument("--limit", type=int, default=0, help="pass: first K eligible windows (smoke)")
    ap.add_argument("--replicate", type=int, default=0,
                    help="pass: re-run the first K windows; selections must be IDENTICAL (S1A.3)")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.mode == "analyze":
        meta = json.loads(Path(a.rows + ".meta.json").read_text(encoding="utf-8"))
        rows = _read_rows(Path(a.rows))
        res = analyze(rows, tuple(meta["box_ranges"]), n_boot=a.n_boot)
        res["_meta"] = meta
        Path(a.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        print("S1 VERDICT:", res["verdict"], "| collided", {k: round(v, 4) for k, v in
                                                             res["collided_rate"].items()})
        return 0
    corp = Corpus(a.config, a.cache, a.labels, a.agents, a.maps)
    wis, elig, drops = _windows(corp, a)
    if a.mode == "pass":
        gt = attach_agent_gt(corp, corp.targs)
        mp = ModelPass(corp, a.ckpt, a.device)
        todo = elig[:a.limit] if a.limit else elig
        rows_p = Path(a.rows)
        done = {(r["sha12"], r["t0"]) for r in _read_rows(rows_p)} if rows_p.exists() else set()
        times = []
        with open(rows_p, "a", encoding="utf-8") as fh:
            for k, wi in enumerate(todo):
                e_i, t = corp.ds.index[wi]
                key = (sha12(corp.clip_ids[e_i]), int(t + corp.W - 1))
                if key in done:
                    continue
                tw = time.time()
                row = pass_window(corp, mp, wi)
                times.append(time.time() - tw)
                fh.write(json.dumps(row) + chr(10))
                fh.flush()
                if k % 25 == 0:
                    print("[s1] %d/%d windows, %.2f s/window" % (k + 1, len(todo),
                                                                float(np.mean(times))), flush=True)
        rep = None
        if a.replicate:
            first = _read_rows(rows_p)[:a.replicate]
            by_key = {(r["sha12"], r["t0"]): r for r in first}
            diffs = 0
            for wi in todo[:a.replicate]:
                r2 = pass_window(corp, mp, wi)
                r1 = by_key.get((r2["sha12"], r2["t0"]))
                if r1 is None:
                    continue
                for arm, v in r2["arms"].items():
                    if "idx" in v and v["idx"] != r1["arms"][arm]["idx"]:
                        diffs += 1
                    if "collided" in v and v["collided"] != r1["arms"][arm]["collided"]:
                        diffs += 1
            rep = {"windows": len(by_key), "selection_or_nc_diffs": diffs,
                   "passed": diffs == 0}
        r = mp.box.ranges
        meta = {"_what": "PREREG_S1 dev-box one-pass rows (S1 AMENDMENT + ERRATA + S1A.10/11)",
                "ckpt": a.ckpt, "config": a.config, "cfg_source": corp.cfg_src,
                "device": a.device, "decoder_steps": mp.steps, "feed_ego": mp.feed_ego,
                "windows": {"len_ds": len(corp.ds), "drawn": len(wis), "eligible": len(elig),
                            "dropped": dict(drops), "run": len(todo)},
                "agent_gt": {k: (v if not isinstance(v, dict) else
                                 {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))})
                             for k, v in gt.items()},
                "box_ranges": [float(r.x_fwd_m), float(r.y_half_m)],
                "s_per_window": float(np.mean(times)) if times else None,
                "replicate": rep, "wall_s": round(time.time() - t0, 1)}
        Path(a.rows + ".meta.json").write_text(json.dumps(meta, indent=1, default=str),
                                               encoding="utf-8")
        print("[s1] pass done: %d rows, %s s/window, replicate %s" % (
            len(_read_rows(rows_p)), meta["s_per_window"], rep), flush=True)
        return 0
    eq = corp.fetch_equivalence(elig[:3])
    if not eq["all_equal"]:
        raise SystemExit("⛔ the light item differs from item 19's Ctx.fetch: %s" % eq)
    rt = roundtrip(corp, elig)
    rec = {"_what": "PREREG_S1 S1A.6 human round-trip control (checkpoint-free)",
           "_evidence_class": "MEASURED (ours)", "_tier": "T0 (the human's own recorded future)",
           "config": a.config, "cfg_source": corp.cfg_src, "cache": a.cache,
           "windows": {"len_ds": len(corp.ds), "expected": a.expect_windows,
                       "drawn": len(wis), "eligible": len(elig), "dropped": dict(drops),
                       "episodes": len({corp.ds.index[w][0] for w in elig})},
           "horizons": list(corp.horizons), "raw_offset": corp.raw_off,
           "fetch_equivalence": eq, "roundtrip": rt, "wall_s": round(time.time() - t0, 1)}
    Path(a.out).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    r = rt
    print("S1A.6 ROUND-TRIP: %s | both-agree %.4f (bar %.2f) | NC %.4f | DAC %.4f | "
          "n=%d eligible of %d drawn (dropped %s) | pos err median %.3f m p95 %.3f m | "
          "human nc!=1 %.4f" % ("PASS" if r["passed"] else "⛔ FAIL", r["both_agree"],
                                 ROUNDTRIP_BAR, r["nc_agree"], r["dac_agree"], r["n"],
                                 len(wis), dict(drops), r["max_pos_err_m"]["median"],
                                 r["max_pos_err_m"]["p95"], r["human_nc_ne_1_frac"]))
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
