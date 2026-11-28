#!/usr/bin/env python
"""ddv2_rl_refcv5.py — DiffusionDriveV2's RL stage (released arithmetic) on refcv5-v2, dev-box scale.

  diagnose  ZERO training. D1 parity of the binding on the REAL model; D2 how often the release's
            clamps bind; D3 how far the release's 10-label chain lands from the DEPLOYED 2-step
            fan (clamps on / off); D4 the cost of one RL step; D5 the known-value controls
            (the human's own proxy scores; human-as-candidate identity; the GT-bar admission rate)
  train     one arm: ``--arm rl`` (DDv2 advantage) | ``--arm norl`` (the same stage with A == 0,
            i.e. the release's IL term alone at lambda = 1.0 — the length-matched control).
            ⭐ LEVER L1 / D9 (PI 2026-09-16), two flags and nothing else:
            ``--il-form {release,matched,lambda}`` and ``--grad-clip`` (default 100 — AMENDMENT
            A-1, 2026-09-16, pre-run: 1.0 was MEASURED to bind on 600/600 steps of all three
            banked arms, i.e. an every-step rescale rather than a spike guard; ``0`` = the
            release's no-clipping recipe). ``--il-form release --grad-clip 0`` reproduces the
            2026-09-15 validation EXACTLY. See ``tanitad.rl.ddv2_il``.
  heldout   T0 read of a checkpoint on held-out windows with the DEPLOYED sampler (paired
            inference noise per window): proxy scores of the selected plan and of the fan

⭐ Everything the model sees is built by the T1 harness's own code
(``taniteval/tools/refcv3_arm.py``: ``load_model``, ``build_corpus``, ``trainer().frames_to_device``,
``refc_v3.ego_state_from_batch``, ``refb_labels.waypoint_targets``), so a window here is the
window the T1 panel scores. The DDv2 arithmetic is ``tanitad.rl.ddv2_rl`` (pinned bitwise to
the release), the binding ``tanitad.rl.ddv2_refc_chain`` (pinned bitwise to ``_sample``), the
reward ``tanitad.rl.pdm_proxy`` (a PDMS-SHAPED proxy — never quote it as PDMS).

⛔ Tier: every number this script prints is T0 (training-side / teacher-forced inputs). T1 is the
harness's roll of a saved checkpoint, and it is SELF-ACTION OPEN LOOP — never closed loop.
⛔ Clip ids are never written: windows are named by ``sha256(clip_id)[:12]``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import lzma
import math
import os
import pathlib
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.rl import ddv2_il as L  # noqa: E402  (lever L1 / D9 — the two declared changes)
from tanitad.rl import ddv2_refc_chain as C  # noqa: E402
from tanitad.rl import ddv2_rl as D  # noqa: E402
from tanitad.rl import pdm_proxy as P  # noqa: E402

#: ``--il-form`` -> :class:`tanitad.rl.ddv2_il.IlSettings` form + lambda scale.
#: ``lambda``'s 0.1 multiplies the release's advantage-derived 0.1 / 1.0 row weights, and
#: MEASURED D6 says essentially every cold-start row carries a positive advantage — so the
#: weight that binds becomes 0.1 * 0.1 = **lambda ~ 0.01**, the RESULT.md LEVER-1 alternative.
IL_FORM_FLAGS = {"release": (L.RELEASE_IL_FORM, 1.0),
                 "matched": ("matched_anchor", 1.0),
                 "lambda": ("release_all_modes_lambda", 0.1)}

DEFAULTS = {
    "ckpt": "C:/Users/Admin/refcv5v2_final/ckpt.pt",
    "config": "C:/Users/Admin/refcv5v2_final/config.json",
    "episodes": "C:/Users/Admin/tanitad-data/refav1-eval141/eps",
    "labels": "C:/Users/Admin/refcv5cmp/data/s2_labels_v7.2_eval.jsonl.gz",
    "agents": "C:/Users/Admin/tanitad-caches/b1-agent-join-20260906/b1eval_agents.jsonl.xz",
    "split": os.path.join(REPO, "TanitAD Research Lab", "Architecture & Inference", "Research",
                          "2026-09-15-ddv2-rl-prep", "raw", "SPLIT_eval141_sha12.json"),
    "ckpt_md5_prefix": "9405ec73",
}
AGENT_TICKS = P.PROXY.n_ticks + 1 + max(P.PROXY.ttc_offsets)        # 50
ROUTE_TICKS = 60


def sha12(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:12]


def md5_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
# setup: the harness's own model and corpus                                     #
# --------------------------------------------------------------------------- #
def load_harness():
    path = os.path.join(REPO, "taniteval", "tools", "refcv3_arm.py")
    spec = importlib.util.spec_from_file_location("refcv3_arm_for_ddv2", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refcv3_arm_for_ddv2"] = mod
    spec.loader.exec_module(mod)
    return mod


class Ctx:
    def __init__(self, a, split_name: str):
        self.a = a
        self.dev = a.device
        self.arm = load_harness()
        self.model, self.cfg, _, self.prov = self.arm.load_model(a.ckpt, a.config, self.dev, False)
        if self.prov["state_dict_load"]["missing_keys"] or self.prov["state_dict_load"]["unexpected_keys"]:
            raise SystemExit("non-strict load; refusing")
        self.dec = self.model.core.decoder
        ns = types.SimpleNamespace(episodes=a.episodes, lru=8, episodes_n=0, labels=a.labels,
                                   nav_source="v72")
        (self.eps, _, self.clip_ids, self.ds, _, self.join, self.nav_src,
         self.raw_off) = self.arm.build_corpus(ns, self.cfg, self.prov)
        # ---- the reward's DAC term ------------------------------------------- #
        # ⛔ STATED LOUDLY EITHER WAY. A reward silently missing its road-boundary
        # constraint is what produced H-DDV2RL-2's FAIL-HARM, and the run record
        # said nothing about it. Now it always does.
        self.map_root = getattr(a, "map_gt_root", None) or None
        self.map_store = None
        if self.map_root:
            from tanitad.data import semantic_map_gt as _smg
            self._smg = _smg
            self.map_store = {}          # clip_id -> ClipMapGT, opened lazily
            print(f"[rl] ⭐ DAC IS LIVE: SAM3 map GT from {self.map_root}. The reward's "
                  f"road-boundary multiplier reads the map; `cand_dac_mean` below 1.0 "
                  f"means candidates are being penalised for leaving the drivable area.",
                  flush=True)
        else:
            print("[rl] ⛔ DAC IS DEAD: no --map-gt-root, so `score_candidates` defaults "
                  "the road-boundary multiplier to ONES. `cand_dac_mean` will read "
                  "EXACTLY 1.0 on every step and the policy is unconstrained by the road "
                  "edge -- this is the H-DDV2RL-2 configuration, and it is a legal "
                  "ABLATION but never a deployable arm.", flush=True)

    def map_for(self, clip_id: str, raw_frame: int):
        """``(drivable_frac [120,64], seen [120,64])`` at a RAW frame, or ``None``.

        ⛔ The key is the RAW frame (``t0 + raw_off``) -- the same index the agent
        lookup already uses. Passing the stacked-row ``t0`` would label every window
        ``raw_off`` frames early, silently.
        """
        if self.map_store is None:
            return None
        g = self.map_store.get(clip_id)
        if g is None:
            p = pathlib.Path(self.map_root) / f"{self._smg.sha12(clip_id)}{self._smg.GT_SUFFIX}"
            if not p.is_file():
                self.map_store[clip_id] = False
                return None
            g = self.map_store[clip_id] = self._smg.open_path(p, clip_id)
        if g is False:
            return None
        mf = g.read(np.asarray([int(raw_frame)]))
        ch = self._smg.CHANNELS.index("drivable")
        return (torch.from_numpy(np.asarray(mf.cart[0][ch], dtype=np.float32)),
                torch.from_numpy(np.asarray(mf.seen[0], dtype=bool)))
        self.tr = self.arm.trainer()
        from tanitad.refs import refc_v3 as v3mod
        import refb_labels
        self.v3, self.rb = v3mod, refb_labels
        self.horizons = tuple(int(h) for h in self.cfg.core.trajectory.horizons)
        self.W = int(self.cfg.core.window)
        self.steps = int(self.prov["decoder_steps"])
        with open(a.split, encoding="utf-8") as fh:
            split = json.load(fh)
        allow = set(split["held_out_sha12" if split_name == "held_out" else "rl_train_sha12"])
        self.agents = self._load_agents(a.agents)
        self.windows = self._select(allow, int(a.stride))
        self.norm = torch.tensor(tuple(self.dec.cfg.control_norm), device=self.dev)
        self.table = D.diffusers_alphas_cumprod(self.dev)

    def _load_agents(self, path):
        want = {str(c) for c in self.clip_ids}
        out = {}
        with lzma.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["clip_id"] in want:
                    out[(r["clip_id"], int(r["frame"]))] = r["agents"]
        return out

    def _select(self, allow, stride):
        """The harness's global stride over ``ds.index``, restricted to the split, a full 6 s
        future, and agent labels on every raw frame the proxy reads (t0 .. t0+49)."""
        out, dropped = [], {"split": 0, "future": 0, "agents": 0, "t0": 0}
        for wi, (e_i, t) in enumerate(self.ds.index):
            if wi % stride:
                continue
            cid = str(self.clip_ids[e_i])
            if sha12(cid) not in allow:
                dropped["split"] += 1
                continue
            t0 = t + self.W - 1
            n_prov = int(self.eps[e_i].poses.shape[0])
            if t0 < 1:
                dropped["t0"] += 1
                continue
            if t0 + ROUTE_TICKS > n_prov - 1:
                dropped["future"] += 1
                continue
            r0 = t0 + self.raw_off
            if any((cid, r0 + k) not in self.agents for k in range(AGENT_TICKS)):
                dropped["agents"] += 1
                continue
            out.append((wi, e_i, t))
        self.dropped = dropped
        return out

    # ---- one window's tensors ------------------------------------------------ #
    def fetch(self, w):
        wi, e_i, t = w
        item = self.ds[wi]
        t0 = t + self.W - 1
        cid = str(self.clip_ids[e_i])
        ep = self.eps[e_i]
        pose_last = item["pose_last"].float()
        fut = item["future_poses_ext"].float()
        fv = item["future_valid_ext"]
        if not bool(fv[:ROUTE_TICKS].all()):
            raise RuntimeError(f"window {sha12(cid)}@{t0}: future not valid over {ROUTE_TICKS}")
        poses = ep.poses[t0:t0 + AGENT_TICKS].float()
        if not torch.allclose(poses[1:1 + P.PROXY.n_ticks], fut[:P.PROXY.n_ticks], atol=1e-5):
            raise RuntimeError("future_poses_ext is not provider poses[t0+1:] — index contract broken")
        gt_wp = self.rb.waypoint_targets(pose_last[None], fut[None], self.horizons)[0]
        human = P.ego_states_from_poses(pose_last[None], fut[None])[0]
        rcfg = P.ProxyConfig(n_ticks=ROUTE_TICKS)
        route_st = P.ego_states_from_poses(pose_last[None], fut[None], rcfg)[0]
        rcx = route_st[:, 0] + P.PROXY.rear_axle_to_center * torch.cos(route_st[:, 2])
        rcy = route_st[:, 1] + P.PROXY.rear_axle_to_center * torch.sin(route_st[:, 2])
        r0 = t0 + self.raw_off
        frames = [self.agents[(cid, r0 + k)] for k in range(AGENT_TICKS)]
        tracks = P.AgentTracks.from_frames(frames, poses)
        nid = getattr(self.ds, "_nav_by_sid", {}).get(int(ep.episode_id))
        es = self.v3.ego_state_from_batch({"pose_last": pose_last[None],
                                           "actions": item["actions"].float()[None]}, device="cpu")
        # ⛔ THE MAP IS KEYED ON r0, THE RAW FRAME -- the same index the agent lookup
        # above already uses. Keying it on the stacked-row `t0` would label every
        # window `raw_off` frames early and nothing would report it.
        dmap = self.map_for(cid, r0)
        return {"frames": item["frames"], "v0": float(pose_last[3]), "nav": 0 if nid is None else int(nid),
                "ego_state": es[0], "gt_wp": gt_wp, "human": human,
                "route": torch.stack([rcx, rcy], dim=-1), "tracks": tracks,
                "map_drivable": None if dmap is None else dmap[0],
                "map_seen": None if dmap is None else dmap[1],
                "sha12": sha12(cid), "t0": int(t0)}

    def capture(self, items, seed=None):
        """Batched deployed forward (eval, no grad) under the sampler capture."""
        fr = self.tr.frames_to_device(torch.stack([x["frames"] for x in items]), self.dev)
        nav = torch.tensor([x["nav"] for x in items], device=self.dev)
        v0 = torch.tensor([x["v0"] for x in items], dtype=torch.float32, device=self.dev)
        es = torch.stack([x["ego_state"] for x in items]).to(self.dev)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        with C.capture_sampler_inputs(self.dec) as rec, torch.no_grad():
            out = self.model(fr, nav_cmd=nav, v0=v0, steps=self.steps, ego_state=es)
        if len(rec) != 1:
            raise RuntimeError(f"expected ONE sampler call per forward, got {len(rec)}")
        return rec[0], out


def score_batch(ctx, u, items, v):
    """Controls ``u [B, M, S, 2]`` (metric) -> per-window proxy dicts (candidates + human)."""
    states = P.ego_states_from_controls(u, v, ctx.horizons,
                                        control_units=ctx.dec.anchor_control_units,
                                        alat_v_floor=ctx.dec.anchor_alat_v_floor,
                                        kappa_cap=ctx.dec.anchor_kappa_cap)
    # ⛔ THE ARTIFACT-SUPPLIED-BUT-READ-BY-NOTHING REFUSAL. An operator who passes
    # --map-gt-root and whose items carry no map would get a run whose record NAMES the
    # SAM3 corpus and whose reward never saw one cell of it -- the exact shape of
    # H-DDV2RL-2, with a config that says otherwise. MEASURED 2026-09-17: a mutation
    # that silently drops the map in `fetch` leaves every unit test green, because the
    # unit tests build their items by hand. This guard is what catches it.
    if getattr(ctx, "map_root", None) and not any(
            it.get("map_drivable") is not None for it in items):
        raise SystemExit(
            "[rl] ⛔ --map-gt-root is set but NOT ONE item in this batch carries a map, "
            "so DAC would silently fall back to ONES and the reward would have no "
            "road-boundary term -- while config.json named a map corpus. Refusing. "
            "Check that the clips have GT files and that `fetch` still attaches "
            "`map_drivable` / `map_seen`.")
    outs = []
    for b, it in enumerate(items):
        # ---- the road-boundary multiplier ---------------------------------- #
        # ⛔ Passed EXPLICITLY, because `score_candidates` defaults DAC to ONES and
        # for the whole of H-DDV2RL-2 nothing ever passed it. The human is scored
        # with the SAME map so its DAC is comparable -- `pdms` normalises EP
        # pairwise against the human, so a human scored without the constraint and
        # candidates scored with it would be two different reward scales.
        dk = {}
        if it.get("map_drivable") is not None:
            fr = it["map_drivable"].to(u.device)
            sn = it["map_seen"].to(u.device)
            hs = it["human"].to(u.device)
            dk = {"dac_cand": P.dac_from_drivable(states[b], fr, sn),
                  "dac_human": P.dac_from_drivable(hs[None], fr, sn)[0]}
        outs.append(P.score_candidates(states[b], it["human"].to(u.device),
                                       it["tracks"].to(u.device), it["route"].to(u.device),
                                       **dk))
    return outs, states


def trainable_params(dec):
    names = ("traj_proj", "time_mlp", "layers", "control_head")
    out = []
    for n, p in dec.named_parameters():
        if n.split(".")[0] in names:
            out.append((n, p))
    return out


# --------------------------------------------------------------------------- #
# diagnose                                                                     #
# --------------------------------------------------------------------------- #
def cmd_diagnose(a):
    ctx = Ctx(a, "rl_train")
    rec = {"_tier": "T0 zero-training diagnostics", "_evidence_class": "MEASURED (ours)",
           "ckpt_step": ctx.prov["step"], "n_windows_eligible": len(ctx.windows),
           "dropped": ctx.dropped, "stride": a.stride}
    log(f"[diag] eligible train windows {len(ctx.windows)} dropped {ctx.dropped}")
    g = torch.Generator().manual_seed(0)
    pick = [ctx.windows[int(i)] for i in torch.randperm(len(ctx.windows), generator=g)[:a.n_windows]]
    d1, d2n, d3, d5 = [], [], [], []
    anc = C.anchor_state(ctx.dec, 1)[0]
    rec["D2_anchor_out_of_box_frac"] = {
        "a_lon": float((anc[..., 0].abs() > 1).float().mean()),
        "a_lat": float((anc[..., 1].abs() > 1).float().mean())}
    for k, w in enumerate(pick):
        it = ctx.fetch(w)
        inp, out = ctx.capture([it], seed=1000 + k)
        fan_ref, u0_ref, _, _ = inp.out
        eps = inp.replay_eps(ctx.dec)
        with torch.no_grad():
            fan, u0 = C.native_sample(ctx.dec, inp, eps)
        d1.append({"u0_bitwise": bool(torch.equal(u0, u0_ref)), "fan_bitwise": bool(torch.equal(fan, fan_ref)),
                   "traj_is_fan_sel": bool(torch.allclose(out["traj"], fan_ref[0, int(out["sel_idx"][0])][None], atol=1e-5))
                   if "sel_idx" in out else None})
        un = u0_ref / ctx.norm
        d2n.append([float((un[..., 0].abs() > 1).float().mean()), float((un[..., 1].abs() > 1).float().mean())])
        # D3: the release chain from the SAME start state, eta = 0 (no exploration)
        x0n = C.anchor_state(ctx.dec, 1)
        ctx.dec.sched.to(ctx.dev)
        x_t = ctx.dec.sched.add_noise(x0n, eps, torch.tensor(int(ctx.dec.cfg.sampler_infer_t), device=ctx.dev)).float()
        variants = {}
        for name, clamp in (("release_clamps_on", True), ("clamps_off", False)):
            consts = D.DDV2 if clamp else D.DDV2Constants(clip_sample=False)
            fn = C.make_x0_fn(ctx.dec, inp, input_clamp=clamp)
            with torch.no_grad():
                roll = D.rollout_chain(fn, x_t, ctx.table, eta=0.0, consts=consts, keep_x0=True)
            u = roll["chain"][..., -1] * ctx.norm
            path = C.state_to_path(ctx.dec, roll["chain"][..., -1], inp.v)
            dend = (path[0, :, -1] - fan_ref[0, :, -1]).norm(dim=-1)
            mnade = (path[0] - it["gt_wp"].to(ctx.dev)[None]).norm(dim=-1).mean(-1).min()
            variants[name] = {"endpoint_dev_m_mean": float(dend.mean()), "endpoint_dev_m_p90": float(dend.quantile(0.9)),
                              "endpoint_dev_m_max": float(dend.max()), "minADE_m": float(mnade),
                              **C.clamp_rates(roll["chain"], roll["x0"])}
            sc, _ = score_batch(ctx, u, [it], inp.v)
            variants[name]["fan_pdms_mean"] = float(sc[0]["pdms"].mean())
            variants[name]["fan_pdms_best"] = float(sc[0]["pdms"].max())
        nat_sc, _ = score_batch(ctx, u0_ref, [it], inp.v)
        variants["deployed"] = {"minADE_m": float((fan_ref[0] - it["gt_wp"].to(ctx.dev)[None]).norm(dim=-1).mean(-1).min()),
                                "fan_pdms_mean": float(nat_sc[0]["pdms"].mean()),
                                "fan_pdms_best": float(nat_sc[0]["pdms"].max())}
        d3.append(variants)
        h = nat_sc[0]["human"]
        # identity control: the human itself as a candidate scores exactly the human
        idc = P.score_candidates(it["human"][None].to(ctx.dev), it["human"].to(ctx.dev),
                                 it["tracks"].to(ctx.dev), it["route"].to(ctx.dev))
        d5.append({"human": h, "identity_equal": abs(float(idc["pdms"][0]) - idc["human"]["pdms"]) < 1e-6})
        log(f"[diag {k + 1}/{len(pick)}] {it['sha12']}@{it['t0']} D1 {d1[-1]} human pdms {h['pdms']:.3f} "
            f"nc {h['nc']} dev_on {variants['release_clamps_on']['endpoint_dev_m_mean']:.2f} m "
            f"dev_off {variants['clamps_off']['endpoint_dev_m_mean']:.2f} m")
    n = len(pick)

    def mean_of(key, sub):
        return sum(v[sub][key] for v in d3) / n
    rec["D1_parity"] = {"n": n, "u0_bitwise_all": all(x["u0_bitwise"] for x in d1),
                        "fan_bitwise_all": all(x["fan_bitwise"] for x in d1),
                        "traj_is_fan_at_sel_idx_all": all(bool(x["traj_is_fan_sel"]) for x in d1
                                                          if x["traj_is_fan_sel"] is not None)}
    rec["D2_deployed_u0_out_of_box_frac"] = {"a_lon": sum(x[0] for x in d2n) / n, "a_lat": sum(x[1] for x in d2n) / n}
    rec["D3_chain_vs_deployed"] = {
        sub: {k: sum(v[sub][k] for v in d3) / n for k in d3[0][sub]}
        for sub in ("release_clamps_on", "clamps_off", "deployed")}
    rec["D5_known_value_controls"] = {
        "human_nc_eq_1_frac": sum(1 for x in d5 if x["human"]["nc"] == 1.0) / n,
        "human_ttc_eq_1_frac": sum(1 for x in d5 if x["human"]["ttc"] == 1.0) / n,
        "human_comfort_eq_1_frac": sum(1 for x in d5 if x["human"]["comfort"] == 1.0) / n,
        "human_pdms_mean": sum(x["human"]["pdms"] for x in d5) / n,
        "identity_control_all_equal": all(x["identity_equal"] for x in d5)}
    # D4 + D6: one full RL step's cost and the GT-bar admission rate at the cold start (no update)
    for _, p in trainable_params(ctx.dec):           # load_model froze everything
        p.requires_grad_(True)
    d4 = []
    for bsz in a.cost_batches:
        items = [ctx.fetch(w) for w in pick[:bsz]]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t_a = time.time()
        inp, _ = ctx.capture(items, seed=7)
        t_b = time.time()
        stats = rl_step(ctx, inp, items, ChainArm("rl", C.ChainSettings()), torch.Generator(device=ctx.dev).manual_seed(3),
                        apply=False)
        torch.cuda.synchronize()
        d4.append({"batch": bsz, "capture_s": round(t_b - t_a, 3), "rl_step_s": round(time.time() - t_b, 3),
                   "peak_mem_MB": round(torch.cuda.max_memory_allocated() / 2 ** 20), **stats["timing"],
                   "frac_admitted_by_bar": stats["frac_admitted_by_bar"],
                   "frac_positive_after_bar": stats["frac_positive_after_bar"],
                   "frac_constraint_fail": stats["frac_constraint_fail"], "rows_with_positive": stats["rows_with_positive"],
                   "reward_mean": stats["reward_mean"], "human_pdms_mean": stats["human_pdms_mean"]})
        log(f"[diag D4] batch {bsz}: {d4[-1]}")
    rec["D4_D6_rl_step"] = d4
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    log(json.dumps({k: v for k, v in rec.items() if k != "D4_D6_rl_step"}, indent=1))
    log(f"[diag] wrote {a.out}")


# --------------------------------------------------------------------------- #
# the RL step                                                                  #
# --------------------------------------------------------------------------- #
class ChainArm:
    def __init__(self, kind: str, settings: C.ChainSettings, il: L.IlSettings | None = None):
        if kind not in ("rl", "norl"):
            raise ValueError(kind)
        # ⛔ DEFAULT = THE RELEASE. `il=None` is the 2026-09-15 recipe exactly (all-modes IL,
        # no gradient clipping), so a call site that does not name the lever does not get it.
        self.kind, self.settings = kind, settings
        self.il = L.IlSettings(grad_clip=None) if il is None else il


def rl_step(ctx, inp, items, arm: ChainArm, gen, *, apply: bool, opt=None) -> dict:
    """One DDv2 step on a captured batch. ``apply=False`` computes and discards the gradient."""
    st = arm.settings
    consts = st.consts
    b = len(items)
    tim = {}
    t = time.time()
    fn = C.make_x0_fn(ctx.dec, inp, input_clamp=st.input_clamp)
    start, _ = D.truncated_start(C.anchor_state(ctx.dec, b), st.groups, ctx.table,
                                 trunc_t=consts.trunc_t, generator=gen)
    with torch.no_grad():
        roll = D.rollout_chain(fn, start, ctx.table, eta=consts.eta, consts=consts, generator=gen,
                               keep_x0=True)
    torch.cuda.synchronize()
    tim["rollout_s"] = round(time.time() - t, 3)
    t = time.time()
    u = roll["chain"][..., -1] * ctx.norm
    scores, _ = score_batch(ctx, u, items, inp.v)
    n = ctx.dec.anchors.shape[0]
    reward = torch.stack([s["pdms"] for s in scores]).view(b, st.groups, n)
    fail = torch.stack([s["constraint_fail"] for s in scores]).view(b, st.groups, n)
    reward_gt = torch.tensor([s["human"]["pdms"] for s in scores], device=reward.device)
    adv_out = D.intra_anchor_advantage(reward, reward_gt, fail, consts=consts)
    adv = adv_out["advantage"]
    torch.cuda.synchronize()
    tim["reward_s"] = round(time.time() - t, 3)
    t = time.time()
    T = len(roll["labels"])
    w = D.step_loss_weights(adv.reshape(b, -1), T, consts=consts)
    # ⭐ LEVER L1, CHANGE 1b — the cheaper alternative arm. A pure post-hoc scale of the IL
    # coefficient the RELEASE derived; `coef_rl` is returned by identity, so a lambda arm
    # cannot move the policy gradient. A no-op (same dict object) when lambda_scale == 1.0.
    w = L.apply_lambda_scale(w, arm.il)
    if arm.kind == "norl":
        # ⭐ THE CONTROL REMOVES ONLY THE POLICY-GRADIENT TERM. The IL weights stay the ones the
        # release derives from THIS batch's advantage (0.1 on rows with a positive, 1.0 else,
        # rl.py:1113-1117) — zeroing A instead would ALSO move lambda to 1.0 on every row and
        # confound "REINFORCE on/off" with a 10x IL-weight change (MEASURED D6: every cold-start
        # row carries a positive, so the RL arm runs at lambda = 0.1 nearly always).
        w["coef_rl"] = torch.zeros_like(w["coef_rl"])
    gt = torch.stack([x["gt_wp"] for x in items]).to(ctx.dev)
    # ⭐ LEVER L1, CHANGE 1 — the match is over the DECODED BANK this forward used
    # (`inp.bank` IS `out["anchor_bank"]`, refc.py:2615), the trainer's own object
    # (refc_v3_train.py:2383-2386), and it is computed ONCE per step: it depends on the GT and
    # the bank, neither of which moves across the 10 rollout steps.
    # ⚠️ COMPUTED ON EVERY ARM, USED BY ONE. `imitation_term` dispatches on `arm.il.form`, so
    # the release arms are unaffected; having `a_star` lets every arm log BOTH IL statistics, so
    # `il_mean_m` — which is not comparable across forms, the matched anchor being the nearest
    # by construction — is never the only number a cross-arm reading has.
    a_star = L.matched_anchor_index(inp.bank, gt)
    params = [p for _, p in trainable_params(ctx.dec)]
    for p in params:
        p.grad = None
    loss_total, il_total, rl_part, spread = 0.0, 0.0, 0.0, None
    il_all, il_match = 0.0, 0.0
    for i in range(T):
        lp, x0 = D.chain_step_logprob(fn, roll["chain"], i, ctx.table, labels=roll["labels"],
                                      eta=consts.eta, consts=consts)
        path = C.state_to_path(ctx.dec, x0, inp.v)
        il_i = L.imitation_term(path, gt, arm.il, a_star=a_star, n_anchors=n)
        li = D.per_step_loss(lp, il_i, w, i)
        li.backward()
        loss_total += float(li)
        il_total += float(il_i) / T
        rl_part += float(li) - float(w["il_coef"]) * float(il_i)
        with torch.no_grad():                       # the two IL statistics, on every arm
            il_all += float(L.all_modes_il(path, gt)) / T
            il_match += float(L.matched_anchor_il(path, gt, a_star, n)) / T
            if i == T - 1:
                # the LAST step's x̂0 is the chain's answer — the canary is read there
                spread = L.fan_endpoint_spread(path, n)
    torch.cuda.synchronize()
    tim["grad_pass_s"] = round(time.time() - t, 3)
    # ⭐ LEVER L1, CHANGE 2 — ONCE, after all T backwards have accumulated and before the step.
    # `grad_norm` stays the PRE-clip norm, so it remains comparable with the 2026-09-15 logs
    # (RL-s0 147, RL-s1 15,712).
    gstats = L.clip_gradients(params, arm.il)
    gnorm = gstats["grad_norm"]
    if apply:
        opt.step()
    for p in params:
        p.grad = None
    hum = [s["human"] for s in scores]
    # ⛔ "dac" BELONGS IN THIS LIST AND WAS MISSING UNTIL 2026-09-17. `score_candidates`
    # returns it, `pdms` multiplies by it, and `frac_constraint_fail` is
    # `(nc != 1) | (dac != 1)` -- yet 600 steps x 3 arms logged NOTHING about the term,
    # so a reward whose road-boundary constraint was identically 1 left no trace in the
    # telemetry. MEASURED: `cand_dac_mean` is absent from every banked L1 metrics file
    # while `cand_nc_mean` / `cand_ep_mean` / `cand_ttc_mean` are present and varying.
    # ⇒ A SUB-SCORE THAT IS NOT LOGGED CANNOT BE SEEN TO BE CONSTANT. With this key in
    # place, a dead DAC reads `cand_dac_mean` EXACTLY 1.0 on every step, which is a
    # visible defect instead of an invisible one.
    sub = {k: float(torch.stack([s[k] for s in scores]).mean())
           for k in ("nc", "dac", "ep", "ttc", "comfort")}
    return {"loss": loss_total, "rl_part": rl_part, "rl_coef_abs_sum": float(w["coef_rl"].abs().sum()),
            "il_mean_m": il_total, "il_coef": float(w["il_coef"]), "grad_norm": gnorm,
            # ---- LEVER L1 / D9 telemetry -------------------------------------------- #
            "il_form": arm.il.form, "il_lambda_scale": arm.il.lambda_scale,
            "il_all_modes_m": il_all, "il_matched_anchor_m": il_match,
            "grad_norm_clipped": gstats["grad_norm_clipped"], "grad_clipped": gstats["grad_clipped"],
            "grad_clip": gstats["grad_clip"], "chain_endpoint_spread_m": spread,
            **{f"match_{k}": v for k, v in L.anchor_match_diagnostics(a_star, n).items()},
            "reward_mean": float(reward.mean()), "reward_best_mean": float(reward.flatten(1).amax(1).mean()),
            "human_pdms_mean": float(reward_gt.mean()),
            "human_nc_eq_1_frac": sum(1 for h in hum if h["nc"] == 1.0) / b,
            **{f"cand_{k}_mean": v for k, v in sub.items()},
            "frac_positive_before_bar": float(adv_out["frac_positive_before_bar"]),
            "frac_admitted_by_bar": float(adv_out["frac_admitted_by_bar"]),
            "frac_positive_after_bar": float(adv_out["frac_positive_after_bar"]),
            "frac_constraint_fail": float(adv_out["frac_constraint_fail"]),
            "frac_nonzero_adv_used": float((w["coef_rl"] != 0).float().mean()),
            "rows_with_positive": int((w["il_weight_b"] < 0.5).sum()),
            "il_weight_mean": float(w["il_weight_b"].float().mean()),
            **C.clamp_rates(roll["chain"], roll["x0"]), "timing": tim}


# --------------------------------------------------------------------------- #
# train                                                                        #
# --------------------------------------------------------------------------- #
def lr_at(step, total, lr, min_lr=1e-6, warmup_frac=0.10):
    """P suppl. §7: 'a cosine learning rate schedule with a 10 % linear warmup' (over steps)."""
    wu = max(1, int(round(warmup_frac * total)))
    if step < wu:
        return lr * (step + 1) / wu
    prog = (step - wu) / max(1, total - wu)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * prog))


def cmd_train(a):
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    ctx = Ctx(a, "rl_train")
    if not ctx.windows:
        raise SystemExit("no eligible windows")
    os.makedirs(a.out_dir, exist_ok=True)
    settings = C.ChainSettings(groups=a.groups, input_clamp=not a.no_input_clamp,
                               consts=D.DDV2Constants(clip_sample=not a.no_clip_sample))
    form, lam = IL_FORM_FLAGS[a.il_form]
    il = L.IlSettings(form=form, lambda_scale=lam,
                      grad_clip=(None if float(a.grad_clip) == 0.0 else float(a.grad_clip)))
    arm = ChainArm(a.arm, settings, il)
    for p in ctx.model.parameters():
        p.requires_grad_(False)
    named = trainable_params(ctx.dec)
    for _, p in named:
        p.requires_grad_(True)
    n_train = sum(p.numel() for _, p in named)
    base = {n: p.detach().clone() for n, p in named}
    opt = torch.optim.AdamW([p for _, p in named], lr=a.lr, weight_decay=a.weight_decay)
    run = {"arm": a.arm, "seed": a.seed, "steps": a.steps, "batch": a.batch, "lr": a.lr,
           "weight_decay": a.weight_decay, "schedule": "linear warmup 10% + cosine to 1e-6 (P suppl. §7, over steps)",
           "grad_clip": il.grad_clip, "il": il.to_dict(), "il_form_flag": a.il_form,
           "precision": "fp32", "chain": settings.to_dict(),
           "trainable": {"prefixes": ["traj_proj", "time_mlp", "layers", "control_head"],
                         "n_params": n_train, "n_tensors": len(named)},
           "proxy": P.PROXY.to_dict(), "n_train_windows": len(ctx.windows), "dropped": ctx.dropped,
           "stride": a.stride, "ckpt": a.ckpt, "ckpt_step": ctx.prov["step"],
           "_tier": "T0 training-side", "_evidence_class": "MEASURED (ours)"}
    with open(os.path.join(a.out_dir, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(run, fh, indent=1)
    log(f"[train] arm {a.arm} seed {a.seed} windows {len(ctx.windows)} trainable {n_train:,} "
        f"steps {a.steps} batch {a.batch}")
    log(f"[train] LEVER L1: il_form={il.form} lambda_scale={il.lambda_scale} "
        f"effective_lambda={il.effective_lambda()} grad_clip={il.grad_clip} "
        f"is_release={il.is_release}")
    order_gen = torch.Generator().manual_seed(10_000 + a.seed)
    chain_gen = torch.Generator(device=ctx.dev).manual_seed(20_000 + a.seed)
    perm, ptr = torch.randperm(len(ctx.windows), generator=order_gen).tolist(), 0
    mpath = os.path.join(a.out_dir, "metrics.jsonl")
    t_start = time.time()
    with open(mpath, "w", encoding="utf-8") as mf:
        for step in range(a.steps):
            if ptr + a.batch > len(perm):
                perm, ptr = torch.randperm(len(ctx.windows), generator=order_gen).tolist(), 0
            ws = [ctx.windows[j] for j in perm[ptr:ptr + a.batch]]
            ptr += a.batch
            lr = lr_at(step, a.steps, a.lr)
            for gp in opt.param_groups:
                gp["lr"] = lr
            t0 = time.time()
            items = [ctx.fetch(w) for w in ws]
            t_fetch = time.time() - t0
            inp, _ = ctx.capture(items)
            stats = rl_step(ctx, inp, items, arm, chain_gen, apply=True, opt=opt)
            with torch.no_grad():
                dnorm = float(torch.sqrt(sum(((p - base[n]) ** 2).sum() for n, p in named)))
            stats.update({"step": step, "lr": lr, "fetch_s": round(t_fetch, 3),
                          "param_delta_norm": dnorm, "wall_s": round(time.time() - t_start, 1),
                          "finite": bool(math.isfinite(stats["loss"]) and math.isfinite(stats["grad_norm"]))})
            mf.write(json.dumps(stats) + "\n")
            mf.flush()
            if not stats["finite"]:
                raise SystemExit(f"[train] non-finite at step {step}: {stats}")
            if step % 10 == 0 or step == a.steps - 1:
                log(f"[train {a.arm} s{a.seed}] step {step} lr {lr:.2e} loss {stats['loss']:.4f} "
                    f"il {stats['il_mean_m']:.3f} m R {stats['reward_mean']:.3f} Rbest {stats['reward_best_mean']:.3f} "
                    f"H {stats['human_pdms_mean']:.3f} pos {stats['frac_positive_after_bar']:.4f} "
                    f"fail {stats['frac_constraint_fail']:.3f} g {stats['grad_norm']:.3f}"
                    f"{'->clip' if stats['grad_clipped'] else ''} d {dnorm:.4f} "
                    f"spread {stats['chain_endpoint_spread_m']:.2f} m "
                    f"t {stats['timing']} wall {stats['wall_s']}s")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ctx.model.state_dict()
    for k in ck["model"]:
        ck["model"][k] = sd[k].detach().cpu()
    ck["ddv2_rl"] = run
    torch.save(ck, os.path.join(a.out_dir, "ckpt.pt"))
    with open(a.config, encoding="utf-8") as fh:
        cfgj = json.load(fh)
    cfgj["ddv2_rl_posttrain"] = run
    with open(os.path.join(a.out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfgj, fh, indent=1)
    log(f"[train] done in {time.time() - t_start:.0f}s -> {a.out_dir}")


# --------------------------------------------------------------------------- #
# heldout T0 read                                                              #
# --------------------------------------------------------------------------- #
def cmd_heldout(a):
    ctx = Ctx(a, "held_out")
    rows = []
    for k, w in enumerate(ctx.windows):
        it = ctx.fetch(w)
        inp, out = ctx.capture([it], seed=500_000 + k)      # PAIRED inference noise per window
        fan, u0, _, _ = inp.out
        sel = int(out["sel_idx"][0])
        sc, _ = score_batch(ctx, u0, [it], inp.v)
        s = sc[0]
        gt = it["gt_wp"].to(ctx.dev)
        traj = out["traj"][0]
        ends = fan[0, :, -1]
        rows.append({"sha12": it["sha12"], "t0": it["t0"], "sel_pdms": float(s["pdms"][sel]),
                     "sel_nc": float(s["nc"][sel]), "sel_ttc": float(s["ttc"][sel]), "sel_ep": float(s["ep"][sel]),
                     "sel_comfort": float(s["comfort"][sel]), "fan_pdms_mean": float(s["pdms"].mean()),
                     "fan_pdms_best": float(s["pdms"].max()), "fan_nc_fail_frac": float((s["nc"] != 1).float().mean()),
                     "human_pdms": s["human"]["pdms"], "sel_ade_m": float((traj - gt).norm(dim=-1).mean()),
                     "sel_fde_m": float((traj[-1] - gt[-1]).norm()),
                     "fan_minade_m": float((fan[0] - gt[None]).norm(dim=-1).mean(-1).min()),
                     "fan_endpoint_spread_m": float(torch.cdist(ends, ends).mean()),
                     "traj_matches_fan_sel": bool(torch.allclose(traj, fan[0, sel], atol=1e-4))})
        if k % 50 == 0:
            log(f"[heldout] {k}/{len(ctx.windows)} {rows[-1]}")
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"ckpt": a.ckpt, "n_windows": len(rows), "rows": rows, "stride": a.stride,
                   "_tier": "T0 (deployed sampler, recorded future; proxy reward)",
                   "_evidence_class": "MEASURED (ours)"}, fh)
    keys = [k for k in rows[0] if k not in ("sha12", "t0", "traj_matches_fan_sel")]
    log(json.dumps({k: sum(r[k] for r in rows) / len(rows) for k in keys}, indent=1))
    log(f"[heldout] traj==fan[sel] on {sum(r['traj_matches_fan_sel'] for r in rows)}/{len(rows)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("diagnose", "train", "heldout"):
        s = sub.add_parser(name)
        for k, v in DEFAULTS.items():
            if k != "ckpt_md5_prefix":
                s.add_argument(f"--{k}", default=v)
        s.add_argument("--device", default="cuda")
        s.add_argument("--stride", type=int, default=5)
        # ⛔ WITHOUT THIS, THE REWARD HAS NO ROAD-BOUNDARY TERM AT ALL.
        # `pdms = NC x DAC x (...)`, DAC is a BINARY MULTIPLIER, and
        # `score_candidates` defaults it to ONES. Until 2026-09-17 no caller
        # anywhere passed it -- `dac_from_drivable` had never had a production
        # caller -- so DAC was identically 1 for every candidate in every anchor
        # group. GRPO's advantage is computed WITHIN a group, so a term with zero
        # within-group variance contributes EXACTLY ZERO to the gradient: the
        # constraint was not weak, it was ABSENT. (`H-DDV2RL-2` = FAIL-HARM.)
        s.add_argument("--map-gt-root", default=None,
                       help="SAM3 map GT root. Makes the reward's DAC term LIVE. "
                            "Omitted, DAC is identically 1 and the policy is free "
                            "to leave the road -- the run record and the per-step "
                            "cand_dac_mean both say so.")
        if name == "diagnose":
            s.add_argument("--n-windows", type=int, default=12)
            s.add_argument("--cost-batches", type=int, nargs="+", default=[2, 4])
            s.add_argument("--out", required=True)
        if name == "train":
            s.add_argument("--arm", choices=("rl", "norl"), required=True)
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--steps", type=int, required=True)
            s.add_argument("--batch", type=int, default=4)
            s.add_argument("--groups", type=int, default=D.DDV2.group_size)
            s.add_argument("--lr", type=float, default=D.DDV2.lr)
            s.add_argument("--weight-decay", type=float, default=D.DDV2.weight_decay)
            s.add_argument("--no-input-clamp", action="store_true")
            s.add_argument("--no-clip-sample", action="store_true")
            # ⭐ LEVER L1 / D9. ⛔ DEFAULTS ARE THE LEVER, not the release: this script's only
            # remaining purpose is running L1, and the 2026-09-15 release arms are already
            # banked. `--il-form release --grad-clip 0` reproduces them exactly, and `run.json`
            # records `il.is_release` either way so no record can be ambiguous about which ran.
            s.add_argument("--il-form", choices=tuple(IL_FORM_FLAGS), default="matched",
                           help="release = the all-modes L1 that collapsed the fan 93%%; "
                                "matched = the trainer's nearest-anchor L1 (mode preserving); "
                                "lambda = the release's form at lambda ~ 0.01")
            s.add_argument("--grad-clip", type=float, default=L.GRAD_CLIP,
                           help="clip_grad_norm_ max-norm, default 100 (A-1: 1.0 bound on "
                                "600/600 banked steps = an every-step rescale; 100 binds on "
                                "1/600 stable, 276/600 diverged); 0 = the release (no clipping)")
            s.add_argument("--out-dir", required=True)
        if name == "heldout":
            s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.ckpt == DEFAULTS["ckpt"]:
        got = md5_file(a.ckpt)
        if not got.startswith(DEFAULTS["ckpt_md5_prefix"]):
            raise SystemExit(f"cold-start md5 {got} does not start with {DEFAULTS['ckpt_md5_prefix']}")
    {"diagnose": cmd_diagnose, "train": cmd_train, "heldout": cmd_heldout}[a.cmd](a)


if __name__ == "__main__":
    main()
