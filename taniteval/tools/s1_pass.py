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
        return {"pose_last": pose_last, "human": human,
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("roundtrip",), default="roundtrip")
    ap.add_argument("--config", required=True, help="the run's config.json (no weights read)")
    ap.add_argument("--cache", required=True, help="the held-out v2 cache (halfB)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--agents", required=True)
    ap.add_argument("--maps", default=None)
    ap.add_argument("--expect-windows", type=int, required=True,
                    help="the trainer's printed held-out window count (known value)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    corp = Corpus(a.config, a.cache, a.labels, a.agents, a.maps)
    if len(corp.ds) != a.expect_windows:
        raise SystemExit(f"⛔ len(ds) = {len(corp.ds)} != the trainer's {a.expect_windows}: "
                         f"these are NOT the trainer's held-out windows")
    wis = trainer_windows(corp.ds, a.n)
    drops = Counter()
    elig = []
    for wi in wis:
        why = corp.eligibility(wi)
        if why:
            drops[why] += 1
        else:
            elig.append(wi)
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
