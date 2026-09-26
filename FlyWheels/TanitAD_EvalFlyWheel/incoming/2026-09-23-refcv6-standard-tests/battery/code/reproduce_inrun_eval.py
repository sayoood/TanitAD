"""GATE G0 -- reproduce refcv6's own in-run eval row from the checkpoint (SPEC.md §2).

    python reproduce_inrun_eval.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_step1000.pt \
        --config D:/refcv6_eval_kit/ckpt/config.json --metrics <metrics.jsonl> \
        --seeds 0,1,2,3,4,5,6,7 --out raw/g0_step1000.json [--mutation m1_no_equalize]

The trainer's own `compute_losses_v3` is called on the trainer's own eval dataset (rebuilt by
`refcv6_loader`) over the SAME 128 windows in the SAME 8 batches of 16. The only departures are
recorded: (1) the forward is micro-batched INSIDE the single model call (`microbatch.py`), (2) only
the LAW frame of `future_frames` reaches the device, (3) `--trunk-compile` is off, (4) the DDIM draw
is seeded per inference seed (the run's draw came from its training RNG and cannot be replayed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcv6_loader as L  # noqa: E402

L.bootstrap()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from microbatch import MicroBatchForward  # noqa: E402

EXCLUDED = {"eval_goal_gate_grad"}
COUNT_EXACT = {"eval_batches", "eval_windows", "eval_slot_valid_frac", "eval_tac_label_rows",
               "eval_tac_label_v7", "eval_nav_injected", "eval_ego_injected",
               "eval_box3d_visible_filter", "eval_map_gt_drivable_frac"}
MATCHED = {f"eval_{k}" for k in ("agent_centre", "agent_cls", "agent_presence", "agent_size",
                                 "agent_yaw", "box3d", "box3d_centre", "box3d_cls", "box3d_h",
                                 "box3d_occ", "box3d_presence", "box3d_rates", "box3d_size",
                                 "box3d_yaw", "box3d_z")}
T_995 = {2: 63.657, 3: 9.925, 4: 5.841, 5: 4.604, 6: 4.032, 7: 3.707, 8: 3.499, 9: 3.355,
         10: 3.250, 12: 3.106, 16: 2.947}   # t(0.995, K-1) keyed by K


def term_class(k: str) -> str:
    if k in EXCLUDED:
        return "EXCLUDED"
    # ⚠️ CORRECTED 2026-09-24 00:07, BEFORE any G0 output existed: a raw substring test for "n_"
    # also matched `lon_tac` / `tacv6_lon_ce` ("lo[n_]..."). SPEC §2 names the count keys by
    # example (`agent_n_*`, `box3d_n_*`, `map_n_*`, `n_map_cells*`, `tacv6_n_*`): a COUNT key has a
    # standalone `n` TOKEN between underscores. That is the rule implemented here.
    toks = k[len("eval_"):].split("_")
    if k in COUNT_EXACT or "n" in toks or k.startswith("eval_agent_rows_"):
        return "COUNT"
    if k in MATCHED:
        return "MATCHED"
    return "SMOOTH_OR_STOCHASTIC"


class LawOnly:
    """Stands in for the collated `future_frames` [B, 20, C, H, W] uint8: holds ONLY frame
    `law_idx` (the one `compute_losses_v3` reads, `refc_v3_train.py:3714`)."""

    def __init__(self, u8_frame, law_idx):
        self.u8 = u8_frame            # [B, C, H, W] uint8
        self.law_idx = int(law_idx)


class LawOnlyDevice:
    def __init__(self, dev_frame, law_idx):
        self.x = dev_frame
        self.law_idx = law_idx

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == slice(None)
                and int(key[1]) == self.law_idx):
            raise KeyError(f"LawOnlyDevice holds only [:, {self.law_idx}], asked {key}")
        return self.x


def patch_frames_to_device(tr):
    orig = tr.frames_to_device

    def f(x, device):
        if isinstance(x, LawOnly):
            return LawOnlyDevice(orig(x.u8, device), x.law_idx)
        return orig(x, device)
    tr.frames_to_device = f
    return orig


def make_g0_dataset_cls(tr, law_ahead: int):
    """The trainer's V3Dataset with ONE change: `future_frames` decodes only the first `LAW_AHEAD`
    stacks (`refb_train.py:226` decodes 20). `compute_losses_v3` reads ONLY index `LAW_AHEAD-1`
    (`refc_v3_train.py:3714`), which is the same frame either way; the other 15 are decode cost."""
    class G0Windows(tr.V3Dataset):
        def _window_u8(self, i: int) -> dict:
            e_i, t = self.index[i]
            ep = self.episodes[e_i]
            w = self.window
            return {
                "frames": ep.frames[t:t + w],
                "actions": ep.actions[t:t + w],
                "future_frames": ep.frames[t + w:t + w + law_ahead],
                "future_actions": ep.actions[t + w:t + w + self.max_horizon],
                "future_poses": ep.poses[t + w:t + w + self.max_horizon],
                "pose_last": ep.poses[t + w - 1],
                "episode_id": ep.episode_id,
            }
    return G0Windows


def iter_batches(e_ds, perm, batch, n_batches, law_idx):
    """Collate lazily (RAM: one 16-window batch at a time), exactly the trainer's DataLoader."""
    import torch.utils.data as tud
    dl = tud.DataLoader(tud.Subset(e_ds, perm), batch_size=batch, shuffle=False,
                        num_workers=0, drop_last=True)
    for i, eb in enumerate(dl):
        if i >= n_batches:
            break
        ff = eb["future_frames"]
        eb["future_frames"] = LawOnly(ff[:, law_idx].clone(), law_idx)
        del ff
        yield eb


def collate_batches(e_ds, perm, batch, n_batches, law_idx):
    import torch.utils.data as tud
    dl = tud.DataLoader(tud.Subset(e_ds, perm), batch_size=batch, shuffle=False,
                        num_workers=0, drop_last=True)
    out = []
    t0 = time.time()
    for i, eb in enumerate(dl):
        if i >= n_batches:
            break
        ff = eb["future_frames"]
        eb["future_frames"] = LawOnly(ff[:, law_idx].clone(), law_idx)
        del ff
        out.append(eb)
        print(f"[g0] collated batch {i + 1}/{n_batches} ({time.time() - t0:.0f}s)", flush=True)
    return out


def sub_batch(eb, n):
    o = {}
    for k, v in eb.items():
        if torch.is_tensor(v):
            o[k] = v[:n]
        elif isinstance(v, LawOnly):
            o[k] = LawOnly(v.u8[:n], v.law_idx)
        elif isinstance(v, (list, tuple)):
            o[k] = v[:n]
        else:
            o[k] = v
    return o


def run_eval(tr, model, batches, device, mode, ablate_frames, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    acc, nb, per_batch = {}, 0, []
    bsz = None
    with torch.no_grad():
        for eb in (batches() if callable(batches) else batches):
            bsz = int(eb["frames"].shape[0])
            el = tr.compute_losses_v3(model, eb, device, mode=mode, ablate_frames=ablate_frames)
            row = {}
            for k, v in el.items():
                if torch.is_tensor(v) and v.ndim == 0:
                    acc[k] = acc.get(k, 0.0) + float(v.detach())
                    row[k] = float(v.detach())
                elif isinstance(v, (int, float, bool)):
                    acc[k] = acc.get(k, 0.0) + float(v)
                    row[k] = float(v)
            per_batch.append(row)
            nb += 1
    erow = {f"eval_{k}": round(v / nb, 5) for k, v in acc.items()}
    erow.update(eval_batches=nb, eval_windows=nb * int(bsz or 0))
    return erow, per_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--micro", default="3,3,3,3,4")
    ap.add_argument("--mutation", default="none", choices=("none", "m1_no_equalize"))
    ap.add_argument("--mutation-seed", type=int, default=0)
    ap.add_argument("--skip-wrapper-control", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t_all = time.time()
    tr = L.trainer()
    device = "cuda"
    spec = HERE.parent / "SPEC.md"
    rec = {"tool": "reproduce_inrun_eval.py", "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest()
           if spec.exists() else None, "ckpt": a.ckpt, "ckpt_md5": L.md5_file(a.ckpt),
           "config": a.config, "micro": a.micro, "seeds": a.seeds,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn": torch.backends.cudnn.allow_tf32}}
    config = L.load_config(a.config)
    model, cfg, args, mrec = L.build_model(config, a.ckpt, device)
    rec["model"] = mrec
    step = int(a.step if a.step is not None else mrec["state_dict"]["step"])
    rec["step"] = step
    print(f"[g0] model built in {mrec['build_s']} s; strict missing={len(mrec['state_dict']['missing'])} "
          f"unexpected={len(mrec['state_dict']['unexpected'])} step={step}; param_breakdown equal="
          f"{mrec['param_breakdown']['equal']}; levers={mrec['trunk_memory_levers_built']}", flush=True)
    # the in-run row
    rows = [json.loads(l) for l in open(a.metrics, encoding="utf-8") if l.strip()]
    ev = [r for r in rows if r.get("step") == step and "eval_loss" in r]
    if len(ev) != 1:
        raise SystemExit(f"[g0] {len(ev)} eval rows at step {step} in {a.metrics}")
    inrun = ev[0]
    rec["inrun_row"] = inrun
    # the eval dataset + the fixed subset
    t_ds = time.time()
    law_idx = int(tr.LAW_AHEAD) - 1
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=True,
                                             dataset_cls=make_g0_dataset_cls(tr, int(tr.LAW_AHEAD)))
    rec["dataset"] = drec
    perm = L.inrun_eval_perm(e_ds, int(args.eval_batches), int(args.batch))
    rec["perm_first8"] = perm[:8]
    rec["perm_sha256"] = hashlib.sha256(json.dumps(perm).encode()).hexdigest()
    print(f"[g0] eval dataset: {drec['n_episodes']} episodes -> {drec['n_windows']} windows "
          f"({time.time() - t_ds:.0f}s); subset {len(perm)}", flush=True)
    def batches():
        return iter_batches(e_ds, perm, int(args.batch), int(args.eval_batches), law_idx)
    # ⭐ SPEED: the 8 collated batches are the SAME for every seed; keep them in RAM when there
    # is room (~4.4 GB), else re-collate per seed. Same tensors either way (decode is exact).
    try:
        import gpu_gate as _gg
        _free = float(_gg.gate().get("free_ram_gb", 0.0))
    except Exception:                                      # noqa: BLE001
        _free = 0.0
    rec["batch_cache"] = {"free_ram_gb_at_decision": _free, "cached": _free >= 14.0}
    if _free >= 14.0:
        _cached = list(batches())

        def batches():                                     # noqa: F811
            return iter(_cached)
    first = next(iter(batches()))
    patch_frames_to_device(tr)
    # seam-clamp patience: a TRAINING-dynamics refusal; outputs are unchanged by it
    sp = {}
    if getattr(cfg, "seam_fail_patience", 0):
        sp["cfg"] = cfg.seam_fail_patience
    rec["seam_fail_patience_trained"] = sp
    mode = getattr(args, "mode", "diffusion")
    abl = bool(getattr(args, "ablate_frames", False))
    torch.cuda.reset_peak_memory_stats()
    sizes = [int(x) for x in a.micro.split(",")]
    # ---- wrapper control (DDIM draw zeroed in both paths) ------------------------------ #
    if not a.skip_wrapper_control:
        sb = sub_batch(first, 4)
        orig_randn_like = torch.randn_like
        torch.randn_like = lambda x, *aa, **kk: torch.zeros_like(x)
        try:
            torch.manual_seed(0)
            with torch.no_grad():
                plain = tr.compute_losses_v3(model, sb, device, mode=mode, ablate_frames=abl)
            mbc = MicroBatchForward(model, [1, 3]).install()
            try:
                with torch.no_grad():
                    wrapped = tr.compute_losses_v3(model, sb, device, mode=mode, ablate_frames=abl)
            finally:
                mbc.remove()
        finally:
            torch.randn_like = orig_randn_like
        wc = {}
        worst = 0.0
        for k, v in plain.items():
            if torch.is_tensor(v) and v.ndim == 0 or isinstance(v, (int, float)):
                pv = float(v)
                wv = float(wrapped[k])
                rel = abs(pv - wv) / max(abs(pv), 1e-8)
                wc[k] = {"plain": pv, "wrapped": wv, "rel": rel}
                if abs(pv) > 1e-6:
                    worst = max(worst, rel)
        rec["wrapper_control"] = {"batch": 4, "micro": [1, 3], "ddim_eps": "zeros (both paths)",
                                  "terms": wc, "max_rel_nonzero": worst,
                                  "pass_rel_1e-3": worst <= 1e-3,
                                  "flagged_merge_rules": mbc.flagged()}
        print(f"[g0] wrapper control: max rel diff {worst:.3e} over {len(wc)} terms -> "
              f"{'PASS' if worst <= 1e-3 else 'FAIL'}; flagged merge rules {mbc.flagged()}", flush=True)
    # ---- the seeds ---------------------------------------------------------------------- #
    mb = MicroBatchForward(model, sizes).install()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip() != ""]
    by_seed = {}
    for s in seeds:
        t_s = time.time()
        erow, pb = run_eval(tr, model, batches, device, mode, abl, s)
        by_seed[s] = {"row": erow, "per_batch": pb, "wall_s": round(time.time() - t_s, 1)}
        print(f"[g0] seed {s}: eval_loss {erow.get('eval_loss')} eval_traj {erow.get('eval_traj')} "
              f"lat_tac {erow.get('eval_lat_tac')} ({time.time() - t_s:.0f}s, peak "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB)", flush=True)
    rec["merge_rules_flagged"] = mb.flagged()
    rec["merge_rules_all"] = mb.rules
    rec["by_seed"] = {str(k): v for k, v in by_seed.items()}
    rec["cuda_max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    # ---- mutation M1 (the gate must be able to FAIL) ------------------------------------ #
    if a.mutation == "m1_no_equalize":
        _perc = tr._perc
        pcfg = _perc.PerceptionBranchConfig(w_map=model._w_map, w_box3d=model._w_box3d)
        _single, _table = tr._read_rig_extrinsics(str(args.agent_rig_extrinsics))
        saved = model._lift_bank
        model._lift_bank = _perc.LiftGeometryBank(
            _table, frame=_perc.frame_for_model(model), stride=int(pcfg.stride),
            heights_m=tuple(pcfg.heights_m))                 # refcv3_arm.py:1101, verbatim
        erow_m, pb_m = run_eval(tr, model, batches, device, mode, abl, a.mutation_seed)
        model._lift_bank = saved
        rec["mutation_m1"] = {"what": "lift bank WITHOUT equalize_bottom_rows (refcv3_arm.py:1101)",
                              "seed": a.mutation_seed, "row": erow_m, "per_batch": pb_m}
        print(f"[g0] M1: eval_loss {erow_m.get('eval_loss')} map {erow_m.get('eval_map')}", flush=True)
    mb.remove()
    # ---- the verdict -------------------------------------------------------------------- #
    rec["verdict"] = judge(inrun, by_seed, rec.get("mutation_m1"), rec)
    rec["wall_s"] = round(time.time() - t_all, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    v = rec["verdict"]
    print(json.dumps({k: v[k] for k in ("G0", "reasons", "n_terms", "by_class_counts")}, indent=1))
    print(f"[g0] wrote {a.out} ({rec['wall_s']} s)")


def judge(inrun, by_seed, m1, rec):
    seeds = sorted(by_seed)
    K = len(seeds)
    terms = sorted(k for k in inrun if k.startswith("eval_") and k not in ("eval_batches",))
    res, reasons = {}, []
    cls_count = {}
    for k in terms:
        c = term_class(k)
        vals = [by_seed[s]["row"].get(k) for s in seeds]
        have = [v for v in vals if v is not None]
        r = {"inrun": inrun[k], "class": c, "seed_values": vals}
        if c == "EXCLUDED":
            r["verdict"] = "EXCLUDED"
            res[k] = r
            cls_count[c] = cls_count.get(c, 0) + 1
            continue
        if len(have) != K:
            r["verdict"] = "MISSING"
            reasons.append(f"{k}: not produced by the reproduction")
            res[k] = r
            continue
        mean = statistics.fmean(have)
        sd = statistics.pstdev(have) if K < 2 else statistics.stdev(have)
        spread = (max(have) - min(have)) / max(abs(mean), 1e-12)
        if c == "SMOOTH_OR_STOCHASTIC":
            c = "STOCHASTIC" if spread > 1e-5 else "SMOOTH"
        r.update(mean=mean, sd=sd, rel_spread=spread, cls=c)
        x = float(inrun[k])
        if c == "COUNT":
            ok = abs(mean - x) <= 1e-5 + 1e-9
            r["tol"] = "abs<=1e-5"
        elif c == "MATCHED":
            rel = abs(mean - x) / max(abs(x), 1e-12)
            r["rel_dev"] = rel
            ok = rel <= 0.15
            r["tol"] = "rel<=0.15"
        elif c == "SMOOTH":
            rel = abs(mean - x) / max(abs(x), 1e-12)
            r["rel_dev"] = rel
            ok = (abs(mean - x) <= 1e-3) if abs(x) < 0.1 else (rel <= 0.01)
            r["tol"] = "abs<=1e-3 (|x|<0.1) else rel<=0.01"
        else:
            t = T_995.get(K, 3.499)
            half = t * sd * math.sqrt(1 + 1 / K) + 0.01 * abs(mean)
            r.update(pi_lo=mean - half, pi_hi=mean + half)
            ok = (mean - half) <= x <= (mean + half)
            r["tol"] = f"99% PI t={t} + 1% |mean|"
        r["verdict"] = "OK" if ok else "OUT"
        if not ok:
            reasons.append(f"{k} [{c}] in-run {x} vs seeds mean {mean:.6g} (sd {sd:.3g})")
        cls_count[c] = cls_count.get(c, 0) + 1
        res[k] = r
    med = {}
    for c in ("SMOOTH", "MATCHED"):
        devs = [v["rel_dev"] for v in res.values() if v.get("cls") == c and "rel_dev" in v
                and abs(float(v["inrun"])) >= 0.1]
        med[c] = statistics.median(devs) if devs else None
    if med["SMOOTH"] is not None and med["SMOOTH"] > 0.002:
        reasons.append(f"SMOOTH median rel dev {med['SMOOTH']:.4f} > 0.002")
    if med["MATCHED"] is not None and med["MATCHED"] > 0.05:
        reasons.append(f"MATCHED median rel dev {med['MATCHED']:.4f} > 0.05")
    md = rec["model"]
    if md["state_dict"]["missing"] or md["state_dict"]["unexpected"]:
        reasons.append("strict load not clean")
    if not md["param_breakdown"]["equal"]:
        reasons.append("param_breakdown differs from config.json")
    anc_bad = [k for k, v in md["anchor_file_vs_ckpt_buffers"].items() if v["max_abs_diff"] != 0.0]
    if anc_bad:
        reasons.append(f"anchor file != ckpt anchor buffers: {anc_bad}")
    wc = rec.get("wrapper_control")
    if wc is None:
        reasons.append("wrapper control not run")
    elif not wc["pass_rel_1e-3"]:
        reasons.append(f"wrapper control failed: max rel {wc['max_rel_nonzero']:.3e}")
    m1_out = None
    if m1 is not None:
        m1_out = []
        for k, r in res.items():
            if r.get("cls") != "SMOOTH":
                continue
            y = m1["row"].get(k)
            x = float(r["inrun"])
            if y is None:
                continue
            bad = (abs(y - x) > 1e-3) if abs(x) < 0.1 else (abs(y - x) / abs(x) > 0.01)
            if bad:
                m1_out.append({"term": k, "inrun": x, "m1": y})
        if not m1_out:
            reasons.append("M1 (no equalize) stayed inside the SMOOTH tolerance: the gate has no "
                           "power -> VOID")
    else:
        reasons.append("M1 not run")
    g0 = "PASS" if not reasons else "FAIL"
    if m1 is not None and not m1_out and all(r.startswith("M1") for r in reasons):
        g0 = "VOID"
    return {"G0": g0, "reasons": reasons, "n_terms": len(res), "by_class_counts": cls_count,
            "medians": med, "terms": res, "m1_terms_out_of_tolerance": m1_out}


if __name__ == "__main__":
    main()
