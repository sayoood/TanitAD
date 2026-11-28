"""refcv6 T1 roll: `taniteval/tools/refcv3_arm.py::run_dump`, fed exactly as the trainer feeds refcv6.

refcv3_arm owns the dump contract, the nav controls, the model-free controls, the lead-block join and
the four-family analysis. This module does not copy any of it. It patches three of its seams:

  * `load_model`   -> `refcv6_loader.build_model` (the trainer-exact build: lift bank WITH
                      `equalize_bottom_rows`, the model-level attributes, strict load);
  * `build_corpus` -> the trainer's own V3Dataset over the kit eval cache, every refcv6 channel
                      enabled exactly as `train()` enables it for the eval split (ego history, v6
                      max-speed sidecar, v7 labels + nav, 22-token goal targets). Future FRAMES are
                      not decoded (refcv3_arm's own EvalV3Windows rule: only the LAW target reads
                      them). The window index is restricted to an explicit (clip, ws) list, so
                      refcv6 is rolled on the IDENTICAL windows as the banked baselines;
  * the MODEL CALL -> `Refcv6Proxy`, which adds the two inputs refcv3_arm's call never passes and
                      the refcv6 build REFUSES without: `ego_poses` (= `pose_hist`, the observed
                      window, as `compute_losses_v3` sets it through `set_ego_window`) and
                      `v_max_ms` / `v_max_valid` (the eval sidecar value). It also banks the
                      per-row refcv6 tactical outputs, and on the batched call runs the two extra
                      arms refcv3_arm has no slot for, each under a FORKED RNG so the main arms'
                      DDIM draws are unchanged:
                        - `os_vmaxzero`: max-speed WITHHELD (`v_max_valid = 0`). This is the
                          robustness ablation the E16 seam stamp names as an eval obligation.
                        - `obey30`: the refcv6 acceptance OBEDIENCE forward (ceiling forced to
                          30 km/h). It runs only on windows whose GT max over the valid 6 s
                          future exceeds 40 km/h.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcv6_loader as L  # noqa: E402

L.bootstrap()
import torch  # noqa: E402

_RA3 = None


def ra3():
    """`taniteval/tools/refcv3_arm.py` by path (it runs its own bootstrap on import)."""
    global _RA3
    if _RA3 is None:
        p = L.TANITEVAL / "tools" / "refcv3_arm.py"
        spec = importlib.util.spec_from_file_location("refcv3_arm", str(p))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["refcv3_arm"] = mod
        spec.loader.exec_module(mod)
        _RA3 = mod
    return _RA3


EXTRA_ROW_KEYS = ("tacv6_goal_logits", "tacv6_goal_conf", "tacv6_lat_logits",
                  "tacv6_lon_logits", "lat_logits_tac", "lon_logits_tac", "traj", "sel_idx")
KMH = 1.0 / 3.6


class Ctx:
    item = None
    index = None


def make_dataset_cls(tr, ctx: Ctx):
    class Refcv6EvalWindows(tr.V3Dataset):
        u8_frames = True

        def _window_u8(self, i: int) -> dict:          # refcv3_arm.make_eval_dataset_class, verbatim
            e_i, t = self.index[i]
            ep = self.episodes[e_i]
            w = self.window
            return {
                "frames": ep.frames[t:t + w],
                "actions": ep.actions[t:t + w],
                "future_frames": torch.zeros(0, dtype=torch.uint8),
                "future_actions": ep.actions[t + w:t + w + self.max_horizon],
                "future_poses": ep.poses[t + w:t + w + self.max_horizon],
                "pose_last": ep.poses[t + w - 1],
                "episode_id": ep.episode_id,
            }

        #: ⭐ DECODE PREFETCH (a speed lever, same items): refcv3_arm.run_dump asks for windows in
        #: index order, and the roll is CPU-bound on PNG decode (GPU util 3-20 % MEASURED on the
        #: step-1000 roll). One background thread builds item i+1 while the model runs on item i.
        #: ⛔ ALL dataset access goes through one lock, so the payload LRU (an OrderedDict,
        #: `v2_dataset.py:_payload`) is never touched by two threads at once: the parallelism
        #: is decode || GPU forward, never decode || decode.
        _pool = None
        _lock = None
        _pending: dict = {}

        def _build(self, i: int):
            with type(self)._lock:
                return super().__getitem__(i)

        def __getitem__(self, i: int):
            import threading
            from concurrent.futures import ThreadPoolExecutor
            cls = type(self)
            if cls._lock is None:
                cls._lock = threading.Lock()
                cls._pool = ThreadPoolExecutor(max_workers=1)
            fut = cls._pending.pop(i, None)
            item = fut.result() if fut is not None else self._build(i)
            for j in list(cls._pending):            # stale prefetches (out-of-order access)
                if j != i + 1:
                    cls._pending.pop(j).result()
            if i + 1 < len(self.index) and (i + 1) not in cls._pending:
                cls._pending[i + 1] = cls._pool.submit(self._build, i + 1)
            ctx.item = item
            ctx.index = tuple(self.index[i])
            return item
    return Refcv6EvalWindows


class Refcv6Proxy:
    """Stands in for the model inside refcv3_arm.run_dump. Delegates every attribute."""

    def __init__(self, model, ctx: Ctx, conds_fed, horizons, *, obedience=True, vmaxzero=True):
        d = self.__dict__
        d["model"], d["ctx"], d["conds_fed"] = model, ctx, list(conds_fed)
        d["horizons"] = tuple(int(h) for h in horizons)
        d["obedience"], d["vmaxzero"] = bool(obedience), bool(vmaxzero)
        d["extras"] = {}
        d["n_calls"] = 0

    def __getattr__(self, k):
        return getattr(self.__dict__["model"], k)

    def __setattr__(self, k, v):
        setattr(self.__dict__["model"], k, v)

    def _inputs(self, n, dev, *, vmax_override=None, valid_override=None):
        it = self.ctx.item
        ph = it["pose_hist"].to(dev)
        vm = it["v_max_ms"].reshape(1).to(dev)
        vv = it["v_max_valid"].reshape(1).to(dev)
        if vmax_override is not None:
            vm = torch.full_like(vm, float(vmax_override))
        if valid_override is not None:
            vv = torch.full_like(vv, float(valid_override))
        return {"ego_poses": ph[None].expand(n, *ph.shape).contiguous(),
                "ego_n_past": int(ph.shape[0]),
                "v_max_ms": vm.expand(n).contiguous(), "v_max_valid": vv.expand(n).contiguous()}

    def __call__(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
        model = self.__dict__["model"]
        n, dev = int(frames.shape[0]), frames.device
        kw.update(self._inputs(n, dev))
        out = model(frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **kw)
        self.__dict__["n_calls"] += 1
        key = self.ctx.index
        rec = self.extras.setdefault(key, {})
        conds = ["nav_zero"] if nav_cmd is None else self.conds_fed[:n]
        for r, c in enumerate(conds):
            rec[c] = {k: out[k][r].detach().float().cpu().numpy() for k in EXTRA_ROW_KEYS
                      if k in out and torch.is_tensor(out[k])}
        if nav_cmd is not None:
            it = self.ctx.item
            rec["_window"] = {
                "tac_goal_y": it["tac_goal_y"].numpy() if "tac_goal_y" in it else None,
                "tac_goal_w": it["tac_goal_w"].numpy() if "tac_goal_w" in it else None,
                "lat_v7": int(it["lat_v7"]) if "lat_v7" in it else -100,
                "lon_v7": int(it["lon_v7"]) if "lon_v7" in it else -100,
                "v_max_ms": float(it["v_max_ms"]), "v_max_valid": float(it["v_max_valid"]),
                "nav_fed": [int(x) for x in nav_cmd.reshape(-1).tolist()],
                "v0": float(v0.reshape(-1)[0]) if v0 is not None else None}
            fut = it["future_poses_ext"][:, 3].double().numpy()
            fv = it["future_valid_ext"].numpy().astype(bool)
            rec["_window"]["gt_max_speed_6s"] = float(fut[fv].max()) if fv.any() else float("nan")
            # --- the two extra arms, one row each, under a FORKED RNG ---------------
            f1 = frames[:1]
            nv1 = nav_cmd.reshape(-1)[:1]
            v01 = v0.reshape(-1)[:1] if v0 is not None else None
            es1 = ego_state[:1] if ego_state is not None else None
            geo = {k: kw[k][:1] for k in ("perception_grid", "perception_valid") if k in kw}
            devs = [dev] if dev.type == "cuda" else []
            if self.vmaxzero:
                with torch.random.fork_rng(devices=devs):
                    o = model(f1, nav_cmd=nv1, v0=v01, steps=steps, ego_state=es1,
                              **geo, **self._inputs(1, dev, valid_override=0.0))
                rec["vmax_zero"] = {k: o[k][0].detach().float().cpu().numpy()
                                    for k in EXTRA_ROW_KEYS if k in o and torch.is_tensor(o[k])}
            gmax = rec["_window"]["gt_max_speed_6s"]
            if self.obedience and np.isfinite(gmax) and gmax * 3.6 > 40.0:
                with torch.random.fork_rng(devices=devs):
                    o = model(f1, nav_cmd=nv1, v0=v01, steps=steps, ego_state=es1,
                              **geo, **self._inputs(1, dev, vmax_override=30.0 * KMH,
                                                     valid_override=1.0))
                from tanitad.refs.refcv6_selection import planned_max_speed
                pm = planned_max_speed(o["traj"][:1, None].float(), horizons=self.horizons)
                pm_all = planned_max_speed(o["anchor_traj"][:1].float(), horizons=self.horizons)
                rec["obey30"] = {"traj": o["traj"][0].detach().float().cpu().numpy(),
                                 "planned_max_ms": float(pm.reshape(-1)[0]),
                                 "n_fan_compliant": int((pm_all <= 30.0 * KMH).sum()),
                                 "n_fan": int(pm_all.shape[-1])}
        return out


class FrameMemo:
    """EXACT per-window memo of the trunk BACKBONE (a same-function speed lever, recorded).

    refcv3_arm rolls every nav condition of one window as rows of one batch, then a separate
    nav-null call; this module adds the vmax-zero and obedience calls. That is 5-6 forwards of the
    SAME 10 frames per window, and `--trunk-dedup-frames` shares frames only between consecutive
    rows of ONE sample. With BN frozen (`trunk_frozen_bn`, asserted at build) a frame's features
    depend on that frame alone, so a frame whose pixels are `torch.equal` to one already computed
    for this window reuses its (s16, s32) features. Identity is decided by exact tensor equality,
    never by the fingerprint alone. The cache is dropped when the window changes.
    """

    def __init__(self, trunk, ctx: Ctx):
        if not bool(getattr(trunk, "memory_levers", {}).get("frozen_bn", False)):
            raise SystemExit("[FrameMemo] refused: the trunk's BN is not frozen")
        self.trunk, self.ctx = trunk, ctx
        self.orig = trunk._backbone
        self.key = None
        self.frames = self.f16 = self.f32 = self.fp = None
        self.hits = self.misses = 0
        trunk._backbone = self

    @staticmethod
    def _fp(x):
        return x.flatten(1)[:, ::7919].double().sum(1)

    def __call__(self, x):
        if self.ctx.index != self.key:
            self.key = self.ctx.index
            self.frames = self.f16 = self.f32 = self.fp = None
        n = int(x.shape[0])
        src = [-1] * n
        if self.frames is not None:
            fx = self._fp(x)
            for i in range(n):
                for j in (self.fp == fx[i]).nonzero().flatten().tolist():
                    if torch.equal(self.frames[j], x[i]):
                        src[i] = j
                        break
        miss = [i for i in range(n) if src[i] < 0]
        if miss:
            # ⭐ duplicates WITHIN this call (refcv3_arm batches the nav conditions of ONE window
            # as rows with identical frames) are computed once too -- same exact-equality rule.
            fpm = self._fp(x[miss])
            uniq, dup_of = [], {}
            for a_i, i in enumerate(miss):
                hit = None
                for b_i, j in enumerate(uniq):
                    if bool(fpm[a_i] == fpm[miss.index(j)]) and torch.equal(x[i], x[j]):
                        hit = j
                        break
                if hit is None:
                    uniq.append(i)
                else:
                    dup_of[i] = hit
            xm = x[uniq]
            f16m, f32m = self.orig(xm)
            base = 0 if self.frames is None else int(self.frames.shape[0])
            if self.frames is None:
                self.frames, self.f16, self.f32, self.fp = xm, f16m, f32m, self._fp(xm)
            else:
                self.frames = torch.cat([self.frames, xm])
                self.f16 = torch.cat([self.f16, f16m])
                self.f32 = torch.cat([self.f32, f32m])
                self.fp = torch.cat([self.fp, self._fp(xm)])
            for k, i in enumerate(uniq):
                src[i] = base + k
            for i, j in dup_of.items():
                src[i] = src[j]
            miss = uniq
        self.hits += n - len(miss)
        self.misses += len(miss)
        idx = torch.tensor(src, device=self.f16.device)
        return [self.f16.index_select(0, idx), self.f32.index_select(0, idx)]


def install_frame_memo(model, ctx: Ctx):
    from tanitad.models.timm_trunk import TimmResNetTrunk
    trunks = [m for m in model.modules() if isinstance(m, TimmResNetTrunk)]
    if len(trunks) != 1:
        raise SystemExit(f"[FrameMemo] expected exactly one TimmResNetTrunk, found {len(trunks)}")
    return FrameMemo(trunks[0], ctx)


def s2_window_list(baseline_dump: str, kit_clip_ids: set) -> dict:
    """{clip_id: sorted ws list} from a banked baseline dump, restricted to the kit clips."""
    man = json.load(open(os.path.join(baseline_dump, "manifest.json"), encoding="utf-8"))
    out = {}
    for e in man["episodes"]:
        if e["clip_id"] not in kit_clip_ids:
            continue
        z = np.load(os.path.join(baseline_dump, f"ep{e['file_index']:03d}.npz"))
        out[e["clip_id"]] = sorted(int(x) for x in z["ws"])
    return out


def install_patches(*, config_path: str, windows: dict | None, obedience=True, vmaxzero=True,
                    with_navflip=True, frame_memo=True):
    """Patch refcv3_arm's `load_model` / `build_corpus`. Returns the shared state dict."""
    R = ra3()
    tr = L.trainer()
    ctx = Ctx()
    st = {"ctx": ctx, "proxy": None, "model_record": None, "dataset_record": None}

    def load_model(ckpt_path, config_path_arg=None, device="cpu", allow_nonstrict=False):
        cfgp = config_path_arg or config_path
        config = L.load_config(cfgp)
        model, cfg, targs, mrec = L.build_model(config, ckpt_path, device)
        st["model_record"] = mrec
        st["config"], st["cfg"], st["targs"], st["model"] = config, cfg, targs, model
        from tanitad.refs import refc_v3 as v3
        _ash = config.get("anchors") if isinstance(config.get("anchors"), dict) else {}
        n_anchors = int(model.state_dict()["core.decoder.anchors"].shape[0])
        steps = int(mrec["decoder_steps"])
        prov = {"ckpt": ckpt_path, "step": mrec["state_dict"]["step"], "config_json": cfgp,
                "rebuilt_from": "refcv6_loader.build_model (train() replayed; lift bank WITH "
                                "equalize_bottom_rows; strict)",
                "config_cross_checks": {"param_breakdown_equal": mrec["param_breakdown"]["equal"]},
                "state_dict_load": mrec["state_dict"], "eval_time_cfg_overrides": {},
                "param_breakdown": mrec["param_breakdown"]["rebuilt"],
                "arm": "hier" if cfg.hier else "flat", "hier": bool(cfg.hier),
                "tac_vocab_version": cfg.tac_vocab_version,
                "horizons": list(cfg.core.trajectory.horizons),
                "n_anchors": n_anchors, "n_anchors_provenance": "checkpoint anchor buffer",
                "window": int(cfg.core.window), "decoder_steps": steps,
                "decoder_mode": mrec["mode"],
                "nav_from_v7_trained": bool(config.get("nav_from_v7", False)),
                "nav_cmd_derivation_trained": config.get("nav_cmd_derivation"),
                "train_labels_manifest": config.get("v7_labels"),
                "eval_labels_md5_at_train": (((config.get("nav_from_v7_stats") or {})
                                              .get("eval") or {}).get("label_md5")),
                "u8_batches_trained": config.get("u8_batches"),
                "refcv6_loader": {k: mrec[k] for k in ("argv_remap", "departures",
                                                       "trunk_memory_levers_built",
                                                       "trunk_memory_levers_run", "sampler",
                                                       "anchor_file_vs_ckpt_buffers")}}
        conds = ["nav_true", "nav_shuffled"] + (["nav_flipped"] if with_navflip else [])
        proxy = Refcv6Proxy(model, ctx, conds, cfg.core.trajectory.horizons,
                            obedience=obedience, vmaxzero=vmaxzero)
        st["proxy"] = proxy
        st["memo"] = install_frame_memo(model, ctx) if frame_memo else None
        prov["frame_memo"] = ("ON: exact per-window backbone memo (torch.equal frames, BN frozen)"
                              if frame_memo else "OFF")
        return proxy, cfg, targs, prov

    def build_corpus(a, cfg, prov):
        from tanitad.data import v7_labels as v7l
        from tanitad.data.v2_dataset import load_or_build_manifest
        model, targs, config = st["model"], st["targs"], st["config"]
        if os.path.normcase(os.path.abspath(a.episodes)) != os.path.normcase(
                os.path.abspath(targs.eval_cache)):
            raise SystemExit(f"[refcv6_roll] --episodes {a.episodes} is not the run's eval cache "
                             f"{targs.eval_cache}")
        dcls = make_dataset_cls(tr, ctx)
        ds, eps, drec = L.build_eval_dataset(model, cfg, targs, config,
                                             with_perception_targets=False, dataset_cls=dcls)
        ds.u8_frames = True
        st["dataset_record"] = drec
        man = load_or_build_manifest(a.episodes, verbose=False)
        files = list(man["files"])
        clip_ids = [str(c) for c in man["clip_id"]]
        n_stack = [int(x) for x in man["n_stack"]]
        if len(set(n_stack)) != 1:
            raise SystemExit("[refcv6_roll] mixed n_stack")
        W = int(cfg.core.window)
        if windows is not None:
            keep = set()
            ci = {c: i for i, c in enumerate(clip_ids)}
            for c, wss in windows.items():
                for w0 in wss:
                    keep.add((ci[c], int(w0) - (W - 1)))
            before = len(ds.index)
            ds.index = [tuple(x) for x in ds.index if tuple(x) in keep]
            if len(ds.index) != len(keep):
                missing = len(keep) - len(ds.index)
                raise SystemExit(f"[refcv6_roll] {missing} requested windows are not windows of "
                                 f"the kit dataset (index {before})")
            st["window_restriction"] = {"n_requested": len(keep), "n_index_before": before,
                                        "n_index_after": len(ds.index),
                                        "source": "banked baseline grid (clip, ws), kit clips"}
        labels, lman = v7l.load_v7_labels(a.labels, allow_oracle_nav=True)
        join = {"labels": {"path": a.labels, "md5": lman.md5, "n_records": lman.n_records,
                           "n_episodes": len(eps),
                           "n_joined": sum(1 for e in eps if int(e.episode_id) in ds.v7_by_sid),
                           "join_key": "stable_episode_id(clip_id) (the trainer's key)"},
                "frames": {"n_stack": n_stack[0], "provider_to_raw_frame_offset": n_stack[0] - 1,
                           "rule": "v2_dataset.py:36-38"},
                "refcv6_channels": {k: drec.get(k) for k in ("nav", "max_speed_v6", "labels")},
                "window_restriction": st.get("window_restriction")}
        join["nav"] = {"source": "v72", "stats": drec.get("nav"),
                       "trained_on": prov.get("nav_cmd_derivation_trained")}
        return eps, files, clip_ids, ds, lman, join, "v72", n_stack[0] - 1

    R.load_model = load_model
    R.build_corpus = build_corpus
    for arm, tier, meaning in (
            ("stop", "T1", "STOP: zero displacement at every instant (model-free)"),
            ("os_vmaxzero", "T1", "os with the max-speed input WITHHELD (v_max_valid = 0)")):
        R.ARM_TIERS.setdefault(arm, tier)
        if hasattr(R, "ARM_MEANING"):
            R.ARM_MEANING.setdefault(arm, meaning)
    return st


def save_extras(st: dict, path: str, clip_ids: list):
    """The refcv6 sidecar, aligned to the dump by (clip sha12 in the JSON, clip index, ws)."""
    proxy = st["proxy"]
    W = int(st["cfg"].core.window)
    keys = sorted(proxy.extras)
    arrs: dict = {"clip_index": [], "ws": []}
    rows: dict = {}
    for (e_i, t) in keys:
        rec = proxy.extras[(e_i, t)]
        if "_window" not in rec:
            continue
        arrs["clip_index"].append(e_i)
        arrs["ws"].append(t + W - 1)
        wd = rec["_window"]
        for k in ("lat_v7", "lon_v7", "v_max_ms", "v_max_valid", "gt_max_speed_6s", "v0"):
            rows.setdefault(k, []).append(wd.get(k) if wd.get(k) is not None else np.nan)
        for k in ("tac_goal_y", "tac_goal_w"):
            v = wd.get(k)
            rows.setdefault(k, []).append(v if v is not None else np.full(22, np.nan, np.float32))
        for c in ("nav_true", "nav_shuffled", "nav_flipped", "nav_zero", "vmax_zero"):
            for k in EXTRA_ROW_KEYS:
                v = (rec.get(c) or {}).get(k)
                rows.setdefault(f"{c}.{k}", []).append(v)
        ob = rec.get("obey30")
        rows.setdefault("obey30.planned_max_ms", []).append(ob["planned_max_ms"] if ob else np.nan)
        rows.setdefault("obey30.n_fan_compliant", []).append(ob["n_fan_compliant"] if ob else -1)
        rows.setdefault("obey30.traj", []).append(ob["traj"] if ob else None)
    out = {"clip_index": np.asarray(arrs["clip_index"], np.int64),
           "ws": np.asarray(arrs["ws"], np.int64)}
    for k, v in rows.items():
        proto = next((x for x in v if x is not None), None)
        if proto is None:
            continue
        proto = np.asarray(proto)
        filled = [np.asarray(x, np.float32) if x is not None
                  else np.full(proto.shape, np.nan, np.float32) for x in v]
        out[k] = np.stack(filled)
    np.savez_compressed(path, **out)
    meta = {"n_windows": int(len(out["ws"])), "keys": sorted(out),
            "clip_sha12_by_index": {int(i): L.sha12(clip_ids[int(i)])
                                    for i in sorted(set(out["clip_index"].tolist()))},
            "proxy_calls": int(proxy.n_calls)}
    json.dump(meta, open(path + ".json", "w", encoding="utf-8"), indent=1)
    return meta
