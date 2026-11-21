"""refcv8 R1 -- cache the frozen trunk of refcv7-r101-s0 @ 50,400 and run the IDENTITY controls (DESIGN.md sec. 2-6).

ONE process, ONE split. Built by the launch tree's own eval loader (`tanitad.eval.refcv7_loader.build_model`, STRICT,
step asserted). Per batch:
  phase FULL : the full model through `r1_heads.forward_like_trainer` (compute_losses_v3's own pre-forward block);
               read-only hooks record pooled_seq (input of core.strategic), kv (output of decoder.feat_proj), the
               perception branch output (bev_feats) and the agent slots; the model output is kept.
  phase EXACT: (eval, --identity on) the SAME window through the head-only path fed the FULL-precision captured tensors.
  phase STORE: (eval, --identity on) the head-only path fed exactly what is written to disk (kv fp16, pooled fp16,
               bev_feats int8-affine per window x channel, slots fp16 except presence fp32).
`torch.manual_seed(seed)` immediately before EVERY phase of every window (batch 1 on eval -- the reel's / the route
package's forward). Nothing in the model changes: hooks and instance-attribute stubs only (r1_heads.py).

Output <out>/<split>/: selection.json (window set, sha12 + t + class + flags), shard_XXXX.pt (atomic), identity.json,
record.json. Resumable: a shard whose file exists is skipped. ⛔ Prints sha12 only.
Env (set by the caller BEFORE import): REFCV6_REPO, REFCV6_KIT, PYTHONPATH=<tree>/stack, REFCV6_REMAP_OVERRIDES (train).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r1_lib as L1  # noqa: E402

EXPECT_STEP = 50400
BIG_KEYS = ("frames", "future_frames")
SMALL_MAX_NUMEL = 4096


def select_windows(e_ds, split, sha_of_eid, seed, k_per_clip, turn_k):
    """-> (idx list sorted by (sha12, t), per-window info list). Rules in DESIGN.md sec. 4 (fixed before any R1 number)."""
    W = int(e_ds.window)
    by_ep = {}
    for i, (e_i, t) in enumerate(e_ds.index):
        by_ep.setdefault(int(e_i), []).append((int(t), i))
    cls_of = {}
    gt_cpu = {}
    for e_i, lst in by_ep.items():
        ts, gt, gv = L1.episode_window_gt(e_ds.episodes[e_i].poses.numpy(), W)
        cls, _ = L1.gt_class(gt, gv)
        pos = {int(t): j for j, t in enumerate(ts)}
        for t, i in lst:
            j = pos[t]
            cls_of[i] = str(cls[j])
            gt_cpu[i] = gt[j]
    eps_sorted = sorted(by_ep, key=lambda e: sha_of_eid[int(e_ds.episodes[e].episode_id)])
    flags = {}
    rng = np.random.default_rng(seed)
    if split == "eval":
        grid = []
        for e_i in eps_sorted:                                   # run_route.py's eval_diag rule, K = 8
            ts = sorted(by_ep[e_i])
            for j in range(8):
                grid.append(ts[int((j + 0.5) * len(ts) / 8)][1])
        order = [i for e_i in eps_sorted for _t, i in sorted(by_ep[e_i])]
        turn = [i for i in order if cls_of[i] in ("turnL", "turnR")]
        rest = [i for i in order if cls_of[i] not in ("turnL", "turnR")]
        samp = [rest[j] for j in sorted(rng.choice(len(rest), min(len(turn), len(rest)), replace=False))]
        for i in grid:
            flags.setdefault(i, set()).add("grid")
        for i in turn:
            flags.setdefault(i, set()).add("turn")
        for i in samp:
            flags.setdefault(i, set()).add("sample")
    else:
        for e_i in eps_sorted:
            lst = sorted(by_ep[e_i])
            tw = [i for _t, i in lst if cls_of[i] in ("turnL", "turnR")]
            ow = [i for _t, i in lst if cls_of[i] not in ("turnL", "turnR")]
            nt = min(turn_k, len(tw))
            pt = [tw[j] for j in sorted(rng.choice(len(tw), nt, replace=False))] if nt else []
            no = min(k_per_clip - nt, len(ow))
            po = [ow[j] for j in sorted(rng.choice(len(ow), no, replace=False))] if no else []
            for i in pt:
                flags.setdefault(i, set()).add("turn")
            for i in po:
                flags.setdefault(i, set()).add("other")
    idx = sorted(flags, key=lambda i: (sha_of_eid[int(e_ds.episodes[e_ds.index[i][0]].episode_id)], e_ds.index[i][1]))
    info = [{"i": int(i), "sha12": sha_of_eid[int(e_ds.episodes[e_ds.index[i][0]].episode_id)],
             "t": int(e_ds.index[i][1]), "cls": cls_of[i], "flags": sorted(flags[i])} for i in idx]
    return idx, info, gt_cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["eval", "train"])
    ap.add_argument("--ckpt", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0/ckpt.pt")
    ap.add_argument("--run-dir", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--identity", choices=["on", "off"], default="on")
    ap.add_argument("--k-per-clip", type=int, default=10)
    ap.add_argument("--turn-k", type=int, default=5)
    ap.add_argument("--shard", type=int, default=256)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--box3d", choices=["auto", "on", "off"], default="auto",
                    help="cache the perception box3d decoder's 300 slots (WP-D option B3, MM 2026-10-04); auto = on for "
                         "the TRAIN split only (the eval capture was already running when this was asked)")
    ap.add_argument("--box3d-only", action="store_true",
                    help="a short pass that writes ONLY box3d slots (box3d_XXXX.pt, keyed by b_r1_i) for the windows "
                         "of the trunk-carrying shards -- the eval side of B3, no identity phases, no trunk tensors")
    ap.add_argument("--budget-s", type=float, default=2280.0,
                    help="stop before a shard that would end past this many seconds (shared-lock courtesy)")
    ap.add_argument("--device", default="cuda", help="cpu = a GPU-free SMOKE of the code path only (never a capture)")
    a = ap.parse_args()
    if a.device != "cuda" and not (a.probe or 0 < a.max_windows <= 8):
        raise SystemExit("[r1] --device cpu is for smokes only (--probe, or --max-windows <= 8)")
    if a.identity == "on" and a.batch != 1:
        raise SystemExit("[r1] identity needs batch 1 (the per-window seed of the banked capture)")

    import torch
    from tanitad.eval import refcv7_loader as L
    tr = L.trainer()
    import refb_labels
    from tanitad.data import v7_labels as v7l
    from tanitad.refs import refc
    import r1_heads as H
    out_dir = Path(a.out) / a.split
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- vocabulary / constants the arms rely on, asserted against the tree --------------------------- #
    if tuple(v7l.HEADS["tac_lat"]) != L1.LAT_V7 or tuple(v7l.HEADS["tac_lon"]) != L1.LON_V7:
        raise SystemExit("[r1] the v7 lat/lon vocabulary differs from r1_lib's literal copy")
    if list(refc.NAV_COMMANDS) != ["follow", "left", "right", "straight"]:
        raise SystemExit(f"[r1] NAV_COMMANDS {refc.NAV_COMMANDS} differs from r1_lib.NAV_INDEX")

    config = L.load_config(Path(a.run_dir) / "config.json")
    model, cfg, args, mrec = L.build_model(config, a.ckpt, device=a.device, strict=True)
    sd = mrec["state_dict"]
    if int(sd.get("step") or -1) != EXPECT_STEP or sd["missing"] or sd["unexpected"]:
        raise SystemExit(f"[r1] bad load: step {sd.get('step')}")
    if model.training:
        raise SystemExit("[r1] the loader returned a model in train mode")

    class R1DS(tr.V3Dataset):
        def __getitem__(self, i):
            item = super().__getitem__(i)
            e_i, t = self.index[i]
            ep = self.episodes[e_i]
            item["map_ep"] = torch.tensor(int(ep.episode_id), dtype=torch.long)   # == V3Dataset:3937/3944
            item["r1_i"] = torch.tensor(int(i), dtype=torch.long)
            item["r1_t_now"] = torch.tensor(float(self._now_s(ep, t)), dtype=torch.float64)
            return item

    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=False,
                                             dataset_cls=R1DS)
    sha_of_eid = dict(getattr(e_ds, "_vis1_sha12", None) or {})
    if not sha_of_eid:
        man = __import__("tanitad.data.v2_dataset", fromlist=["x"])
        m_ = man.load_or_build_manifest(args.eval_cache, verbose=False)
        sha_of_eid = {int(man.stable_episode_id(str(c))): L1.sha12(str(c)) for c in m_["clip_id"]}
    W = int(cfg.core.window)
    if tuple(int(h) for h in cfg.core.trajectory.horizons) != L1.HORIZONS:
        raise SystemExit("[r1] horizons drifted")
    print(f"[r1] built: step {sd['step']} episodes {len(e_eps)} windows {len(e_ds)} W {W} "
          f"build {time.time() - t_start:.0f}s", flush=True)

    idx, info, gt_cpu = select_windows(e_ds, a.split, sha_of_eid, a.seed, a.k_per_clip, a.turn_k)
    counts_full = {}
    for r in info:
        for f in r["flags"]:
            counts_full[f] = counts_full.get(f, 0) + 1
        counts_full["cls_" + r["cls"]] = counts_full.get("cls_" + r["cls"], 0) + 1
    print(f"[r1] FULL selection {a.split}: {len(idx)} windows, counts {json.dumps(counts_full)}", flush=True)
    if a.max_windows:
        idx, info = idx[:a.max_windows], info[:a.max_windows]
    digest = hashlib.sha256(json.dumps([(r["sha12"], r["t"]) for r in info]).encode()).hexdigest()
    counts = {}
    for r in info:
        for f in r["flags"]:
            counts[f] = counts.get(f, 0) + 1
        counts["cls_" + r["cls"]] = counts.get("cls_" + r["cls"], 0) + 1
    sel = {"split": a.split, "n_windows": len(idx), "n_episodes": len({r["sha12"] for r in info}),
           "digest_sha12_t": digest, "counts": counts, "seed": a.seed, "k_per_clip": a.k_per_clip,
           "turn_k": a.turn_k, "windows": info}
    if not a.probe and not a.box3d_only:
        (out_dir / "selection.json").write_text(json.dumps(sel), encoding="utf-8")
    print(f"[r1] selection {a.split}: {len(idx)} windows / {sel['n_episodes']} episodes, counts "
          f"{json.dumps(counts)} digest {digest[:16]}", flush=True)

    # ---- read-only capture -------------------------------------------------------------------------- #
    CAP = {"phase": None}

    def h_model(_m, _i, out):
        k = "out_" + str(CAP["phase"])
        if k not in CAP:
            CAP[k] = out
    def h_strat(_m, inp, _o):
        if CAP["phase"] == "full" and "pooled" not in CAP:
            CAP["pooled"] = inp[0].detach().clone()
    def h_fp(_m, _i, o):
        if CAP["phase"] == "full" and "kv" not in CAP:
            CAP["kv"] = o.detach().clone()
    def h_perc(_m, _i, o):
        if CAP["phase"] == "full" and "pout" not in CAP:
            CAP["pout"] = o
    def h_ag(_m, _i, o):
        if CAP["phase"] == "full" and "slots" not in CAP:
            CAP["slots"] = o
    hooks = [model.register_forward_hook(h_model), model.core.strategic.register_forward_hook(h_strat),
             model.core.decoder.feat_proj.register_forward_hook(h_fp),
             model._perception.register_forward_hook(h_perc), model.core.agent_head.register_forward_hook(h_ag)]
    probe_shapes = {}
    if a.probe:
        def mk(name):
            def h(_m, _i, o):
                if CAP["phase"] == "full" and name not in probe_shapes:
                    def shp(x):
                        if torch.is_tensor(x):
                            return [list(x.shape), str(x.dtype)]
                        if isinstance(x, dict):
                            return {k: shp(v) for k, v in x.items()}
                        if isinstance(x, (list, tuple)):
                            return [shp(v) for v in x[:4]]
                        return type(x).__name__
                    probe_shapes[name] = shp(o)
            return h
        for nm, mod in list(model._map_hires.named_children()) + [("_map_hires", model._map_hires)]:
            hooks.append(mod.register_forward_hook(mk("map_hires." + nm)))
        for nm, mod in model._perception.named_children():
            hooks.append(mod.register_forward_hook(mk("perception." + nm)))

    def _sync():
        if a.device == "cuda":
            torch.cuda.synchronize()

    def _maxmem():
        return torch.cuda.max_memory_allocated() / 2**30 if a.device == "cuda" else float("nan")

    feed = H.Feed()
    S = int(a.shard)
    if S % a.batch:
        raise SystemExit("[r1] --shard must be a multiple of --batch")
    # ⭐ codec v2 (2026-10-04, after I-S FAILED on the per-channel BEV codec): the trunk tensors are stored ONLY for
    # the windows an arm reads -- every TRAIN window, and the SCORED eval windows (turn / sample). The eval-grid-only
    # windows (I-0's join to the route bank) get their identity rows and full-model outputs in `gridonly_*` shards
    # without trunk tensors (they would cost 1.07 GB that the budget does not have).
    flag_of = {r["i"]: r["flags"] for r in info}
    keep_trunk = [i for i in idx if a.split == "train" or ("turn" in flag_of[i] or "sample" in flag_of[i])]
    no_trunk = [i for i in idx if i not in set(keep_trunk)]
    shard_idx = [keep_trunk[k:k + S] for k in range(0, len(keep_trunk), S)]
    shard_kind = ["shard"] * len(shard_idx)
    shard_idx += [no_trunk[k:k + S] for k in range(0, len(no_trunk), S)]
    shard_kind += ["gridonly"] * (len(shard_idx) - len(shard_kind))

    box3d_on = (a.box3d == "on") or (a.box3d == "auto" and a.split == "train") or a.box3d_only
    if a.box3d_only:
        if a.identity == "on":
            raise SystemExit("[r1] --box3d-only runs with --identity off")
        shard_idx = shard_idx[:shard_kind.count("shard")]
        shard_kind = ["box3d"] * len(shard_idx)

    def shard_path(k):
        return out_dir / f"{shard_kind[k]}_{k:04d}.pt"

    def meta_path(k):
        return out_dir / f"{shard_kind[k]}_{k:04d}.meta.json"

    missing = [k for k in range(len(shard_idx)) if a.probe or not (shard_path(k).exists() and meta_path(k).exists())]
    print(f"[r1] shards: {len(shard_idx)} of <= {S} windows; missing {len(missing)}; budget {a.budget_s:.0f}s",
          flush=True)
    gpu_s = {"full": 0.0, "exact": 0.0, "store": 0.0}
    t_loop = time.time()

    def run_phase(name, b):
        CAP["phase"] = name
        _sync()
        tg = time.time()
        torch.manual_seed(int(a.seed))
        with torch.no_grad():
            H.forward_like_trainer(model, b, a.device, args.mode, tr)
        _sync()
        gpu_s[name] += time.time() - tg
        CAP["phase"] = None
        return CAP["out_" + name]

    def summarize(o):
        return {"fan": o["anchor_traj"].float(), "traj": o["traj"].float(), "sel": o["sel_idx"],
                "selb": o["sel_idx_base"], "s_core": o["sel_score"].float(), "s_e9": o["sel_score_v3"].float(),
                "p_lat": o["tacv6_lat_logits"].float().softmax(-1), "p_lon": o["tacv6_lon_logits"].float().softmax(-1),
                "p_goal": o["tacv6_goal_logits"].float().sigmoid(), "g_tac": o["g_tac"].float(),
                "lat_tac": o["lat_logits_tac"].float().softmax(-1), "lon_tac": o["lon_logits_tac"].float().softmax(-1)}

    est_shard_s = S * (0.6 if a.identity == "on" else 0.4)
    first_box3d = [True]
    n_done_proc = 0
    partial = False
    first_batch = True
    probe_rows = []
    for k in missing:
        if (not a.probe) and (time.time() - t_start) + 1.25 * est_shard_s > a.budget_s:
            partial = True
            print(f"[r1] budget: stopping before shard {k} ({time.time() - t_start:.0f}s elapsed, "
                  f"est {est_shard_s:.0f}s/shard) -- the chain re-runs the pass", flush=True)
            break
        t_sh = time.time()
        g0 = dict(gpu_s)
        acc = {}
        ident_rows = []
        sel_gt_maxdiff = 0.0

        def push(kk, v):
            acc.setdefault(kk, []).append(v)
        dl = torch.utils.data.DataLoader(torch.utils.data.Subset(e_ds, shard_idx[k]), batch_size=a.batch,
                                         shuffle=False, num_workers=a.workers,
                                         prefetch_factor=(2 if a.workers > 0 else None))
        n_sh = 0
        for b in dl:
            CAP.clear()
            CAP["phase"] = None
            o_full = run_phase("full", b)
            for kk in ("pooled", "kv", "pout", "slots"):
                if kk not in CAP:
                    raise SystemExit(f"[r1] capture incomplete in shard {k}: {kk} missing ({sorted(CAP)})")
            kv, pooled = CAP["kv"], CAP["pooled"]
            bev = CAP["pout"]["bev_feats"].detach()
            slots = H.slots_to_cache(CAP["slots"])
            if first_batch:
                first_batch = False
                pkeys = sorted((o_full.get("perception") or {}).keys())
                print(f"[r1] kv {tuple(kv.shape)} {kv.dtype} pooled {tuple(pooled.shape)} bev {tuple(bev.shape)} "
                      f"slots {{{', '.join(f'{x}: {tuple(v.shape)}' for x, v in slots.items())}}} perception keys "
                      f"{pkeys}", flush=True)
            # ---- the storage codec (what goes to disk) ------------------------------------------------- #
            kv16 = kv.half()
            pooled16 = pooled.half()
            q, lo, sc = L1.quant_u8_row(bev)
            slots_store = {x: (v.float() if x == "presence_logit" else (v.half() if v.is_floating_point() else v)
                               ).contiguous() for x, v in slots.items()}
            bb = int(kv.shape[0])
            # ---- GT (the tree's own target function) --------------------------------------------------- #
            pl = b["pose_last"].float()
            fut = b["future_poses_ext"].float()
            gt = refb_labels.waypoint_targets(pl, fut, L1.HORIZONS)
            gv = torch.stack([b["future_valid_ext"][:, h - 1] for h in L1.HORIZONS], 1)
            for j in range(bb):
                sel_gt_maxdiff = max(sel_gt_maxdiff,
                                     float(np.abs(gt_cpu[int(b["r1_i"][j])] - gt[j].numpy()).max()))
            # ---- identity phases (eval) ---------------------------------------------------------------- #
            if a.identity == "on":
                sf = summarize(o_full)
                saved = H.install(model, feed)
                try:
                    feed.cur = {"kv": kv, "pooled": pooled, "bev": bev, "slots": slots,
                                "bev_pad": (o_full.get("perception") or {}).get("bev_pad")}
                    sx = summarize(run_phase("exact", b))
                    # ⭐ CONTIGUOUS, exactly as the arms harness reads them back from disk (a strided view can take
                    # a different kernel and break I-D bit-identity -- MEASURED on Thor CPU, s_e9 5.7e-6)
                    feed.cur = {"kv": kv16.float().contiguous(), "pooled": pooled16.float().contiguous(),
                                "bev": L1.dequant_u8(q.contiguous(), lo.contiguous(), sc.contiguous()).contiguous(),
                                "slots": {x: (v.float() if v.is_floating_point() else v).contiguous()
                                          for x, v in slots_store.items()},
                                "bev_pad": (o_full.get("perception") or {}).get("bev_pad")}
                    ss = summarize(run_phase("store", b))
                finally:
                    H.uninstall(model, saved)
                    feed.cur = None
                for j in range(bb):
                    row = {"i": int(b["r1_i"][j])}
                    reach = o_full["reach_keep"][j].bool()
                    se9 = sf["s_e9"][j].masked_fill(~reach, float("-inf"))
                    top2 = torch.topk(se9, 2).values
                    row["e9_top2_margin"] = float(top2[0] - top2[1])
                    for tag, s2 in (("x", sx), ("s", ss)):
                        for x in ("fan", "traj", "s_core", "s_e9", "p_lat", "p_lon", "p_goal", "g_tac", "lat_tac",
                                  "lon_tac"):
                            row[f"{tag}_d_{x}"] = float((s2[x][j] - sf[x][j]).abs().max())
                        row[f"{tag}_eq_sel"] = bool(int(s2["sel"][j]) == int(sf["sel"][j]))
                        row[f"{tag}_eq_selb"] = bool(int(s2["selb"][j]) == int(sf["selb"][j]))
                        row[f"{tag}_s_sel"] = int(s2["sel"][j])
                    ident_rows.append(row)
                for x, v in (("fan", sf["fan"]), ("traj", sf["traj"]), ("sel", sf["sel"]), ("selb", sf["selb"]),
                             ("s_core", sf["s_core"]), ("s_e9", sf["s_e9"]), ("reach", o_full["reach_keep"].bool()),
                             ("p_lat", sf["p_lat"]), ("p_lon", sf["p_lon"]), ("p_goal", sf["p_goal"]),
                             ("g_tac", sf["g_tac"]), ("lat_tac", sf["lat_tac"]), ("lon_tac", sf["lon_tac"])):
                    push("full_" + x, v.cpu())
                if "residual_prior_path" in o_full:
                    push("full_prior_path", o_full["residual_prior_path"].float().reshape(bb, -1, 2).cpu())
                # the STORED-tensor head path's own outputs: what H0 replayed from disk must equal (I-D)
                for x in ("fan", "sel", "selb", "s_e9", "p_lat", "p_lon"):
                    push("store_" + x, ss[x].cpu())
            # ---- box3d slots (B3) ------------------------------------------------------------------------ #
            if box3d_on:
                bs3 = (CAP["pout"].get("box_slots") or {})
                kept3 = {x: v for x, v in bs3.items() if torch.is_tensor(v) and v.dim() >= 2
                         and int(v.shape[0]) == bb and int(v.shape[1]) == 300}
                if not kept3:
                    raise SystemExit(f"[r1] --box3d: no [B, 300, ...] tensor in box_slots ({sorted(bs3)})")
                if first_box3d[0]:
                    first_box3d[0] = False
                    print("[r1] box3d slots " + ", ".join(f"{x} {tuple(v.shape)} {v.dtype}" for x, v in kept3.items())
                          + f"; dropped {sorted(set(bs3) - set(kept3))}", flush=True)
                for x, v in kept3.items():
                    v = v.detach()
                    push("box3d_" + x, (v.float() if (x == "presence_logit" or not v.is_floating_point())
                                        else v.half()).contiguous().cpu() if v.is_floating_point() else v.cpu())
            if shard_kind[k] == "box3d":
                push("b_r1_i", b["r1_i"])
                push("win_sha12", [sha_of_eid[int(e_ds.episodes[e_ds.index[int(i)][0]].episode_id)]
                                   for i in b["r1_i"]])
                n_sh += bb
                n_done_proc += bb
                del o_full
                CAP.clear()
                continue
            # ---- the cache rows ------------------------------------------------------------------------ #
            if shard_kind[k] == "shard":
                push("kv", kv16.cpu())
                push("pooled", pooled16.cpu())
                push("bev_q", q.cpu())
                push("bev_lo", lo.cpu())
                push("bev_scale", sc.cpu())
                for x, v in slots_store.items():
                    push("slot_" + x, v.cpu())
            push("gt", gt.float())
            push("gt_valid", gv.bool())
            for x, v in b.items():
                if x in BIG_KEYS or not torch.is_tensor(v):
                    continue
                if v[0].numel() <= SMALL_MAX_NUMEL:
                    push("b_" + x, v)
            push("win_sha12", [sha_of_eid[int(e_ds.episodes[e_ds.index[int(i)][0]].episode_id)] for i in b["r1_i"]])
            n_sh += bb
            n_done_proc += bb
            del o_full
            CAP.clear()
            if a.probe and n_done_proc >= max(1, a.max_windows):
                break
        print(f"[r1] ZZ{k + 1}-{len(shard_idx)}ZZ shard {k}: {n_sh} windows {time.time() - t_sh:.0f}s gpu full "
              f"{gpu_s['full'] - g0['full']:.0f}s exact {gpu_s['exact'] - g0['exact']:.0f}s store "
              f"{gpu_s['store'] - g0['store']:.0f}s cuda_max {_maxmem():.2f} GiB", flush=True)
        if a.probe:
            probe_rows = ident_rows
            break
        rec_sh = {}
        for kk, v in acc.items():
            rec_sh[kk] = torch.cat(v, 0) if torch.is_tensor(v[0]) else [x for xs in v for x in xs]
        tmp = str(shard_path(k)) + ".tmp"
        torch.save(rec_sh, tmp)
        os.replace(tmp, shard_path(k))
        meta = {"shard": k, "kind": shard_kind[k], "codec": "kv fp16 | pooled fp16 | bev int8 per (channel, row) | "
                "slots fp16 (presence fp32)", "n": n_sh, "ident_rows": ident_rows, "sel_gt_maxdiff": sel_gt_maxdiff,
                "gpu_s": {x: round(gpu_s[x] - g0[x], 2) for x in gpu_s}, "wall_s": round(time.time() - t_sh, 1),
                "bytes": os.path.getsize(shard_path(k))}
        meta_path(k).write_text(json.dumps(meta), encoding="utf-8")
        est_shard_s = max(est_shard_s * 0.5, time.time() - t_sh)
    for h in hooks:
        h.remove()
    if partial:
        print(f"[r1] PARTIAL {a.split}: {n_done_proc} windows this process; re-run to continue", flush=True)
        return 3

    # ---- summary over ALL shards (this process and earlier ones) ------------------------------------------ #
    if a.probe:
        rows_all, metas = probe_rows, []
    else:
        metas = [json.loads(meta_path(k).read_text()) for k in range(len(shard_idx))]
        rows_all = [r for m in metas for r in m["ident_rows"]]
    ident = {"n_windows": len(rows_all)}
    if rows_all:
        for tag in ("x", "s"):
            for x in ("fan", "traj", "s_core", "s_e9", "p_lat", "p_lon", "p_goal", "g_tac", "lat_tac", "lon_tac"):
                v = np.array([r[f"{tag}_d_{x}"] for r in rows_all])
                ident[f"{tag}_maxabs_{x}"] = float(v.max())
                ident[f"{tag}_p99abs_{x}"] = float(np.percentile(v, 99))
            ident[f"{tag}_n_sel_differ"] = int(sum(not r[f"{tag}_eq_sel"] for r in rows_all))
            ident[f"{tag}_n_selbase_differ"] = int(sum(not r[f"{tag}_eq_selb"] for r in rows_all))
            flip_m = [r["e9_top2_margin"] for r in rows_all if not r[f"{tag}_eq_sel"]]
            ident[f"{tag}_flip_e9_top2_margins"] = sorted(round(x, 6) for x in flip_m)[:50]
        ident["e9_top2_margin_p01_p05_p50"] = [float(np.percentile([r["e9_top2_margin"] for r in rows_all], q))
                                               for q in (1, 5, 50)]
    gsum = {x: round(sum(m["gpu_s"][x] for m in metas), 1) for x in gpu_s} if metas else gpu_s
    rec = {"split": a.split, "seed": a.seed, "ckpt": a.ckpt, "step": sd["step"], "n_windows": len(idx),
           "selection_digest": digest, "selection_counts": counts, "W": W,
           "cpu_selection_gt_vs_tree_gt_maxabs_m": max([m["sel_gt_maxdiff"] for m in metas] or [0.0]),
           "identity": ident, "probe_shapes": probe_shapes,
           "loader_departures": mrec.get("departures"),
           "state_dict": {k_: sd[k_] for k_ in ("strict", "missing", "unexpected", "step")},
           "dataset": {k_: v for k_, v in drec.items() if k_ in ("n_episodes", "n_windows", "labels", "nav",
                                                                 "max_speed_v6")},
           "compute": {"gpu_s_all_shards": gsum, "wall_s_all_shards": round(sum(m["wall_s"] for m in metas), 1),
                       "cache_bytes": int(sum(m["bytes"] for m in metas)),
                       "cuda_max_memory_allocated_gib_last_process": round(_maxmem(), 3),
                       "device": torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu"},
           "batch": a.batch, "shard": a.shard, "n_shards": len(shard_idx)}
    name = "probe.json" if a.probe else ("record_box3d.json" if a.box3d_only else "record.json")
    (out_dir / name).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    if rows_all and not a.probe:
        (out_dir / "identity_rows.json").write_text(json.dumps(rows_all), encoding="utf-8")
    print(f"[r1] DONE {a.split}: identity {json.dumps({x: v for x, v in ident.items() if 'flip' not in x})} "
          f"wall {time.time() - t_start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
