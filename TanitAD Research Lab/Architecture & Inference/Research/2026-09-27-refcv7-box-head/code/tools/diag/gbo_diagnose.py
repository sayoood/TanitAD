#!/usr/bin/env python3
"""gbo_diagnose.py -- the Master Mind's three G-BOX-OVERFIT DIAGNOSTICS (2026-09-27 ~09:20 Berlin). NON-BINDING.

MAIN failed its early (non-binding) G-BOX-OVERFIT: no slot ever reached sigma >= 0.5 and AP@2 m ended at 0.013, below
its own ``memory_zeros`` control (0.037). Before any new lever, three diagnostics, one GPU job, in this order:

  audit     MAIN's config, the SAME seed and the first 200 steps of the SAME batch schedule, logged at step 1 and
            every 20: positive presence targets; |dL/d presence_logit| for matched vs unmatched (vs IGNORE-exempt)
            slots, per decoder layer; the grad norm on the memory entering the box decoder (and per parameter
            group); the presence logit of matched vs unmatched slots. Checks: the IGNORE mask never zeroes a
            MATCHED slot (raw mask count + every matched slot carries a non-zero presence gradient), and every
            layer's presence targets come from the matching run on THAT layer's predictions (object identity).
  oneframe  MAIN's config on ONE frame (the frame-set frame with the most VIS-1 positives), 500 steps: the
            matched slots' presence must rise past 0.5 (train-mode forward every 25 steps, eval-rule score every
            100), plus the stability of the Hungarian assignment between logs.
  control   the refcv6 HEAD on the same harness, 16 frames, schedule and 2,000 steps: BCE with NO_OBJECT_W 0.1 (the
            legacy loss path, bit-identical), 100 queries, no deep supervision, prior 0.05, refcv6 targets (the
            visible_target_filter rows, no VIS-1 split) -- SCORED under the same VIS-1 eval rule as MAIN
            (the harness's ``targets_filter_only`` scoring switch).

It imports the harness (``stack/scripts/g_box_overfit.py``) of the tree it is pointed at, UNMODIFIED, and changes
no product code: the instrumentation is forward hooks and call-through wrappers installed only around the logged
steps, removed afterwards. The control's two departures from the launch argv (the four A9 flags off; the box
decoder built at 100 queries through a wrapper around ``build_perception_branch``) are recorded in the output.

usage: python gbo_diagnose.py --tree <tree> --launch-argv <json> --audit-dir <audit> --out <json>
                              [--parts audit,oneframe,control] [--device cuda]
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[diag {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_harness(tree: Path):
    p = tree / "stack" / "scripts" / "g_box_overfit.py"
    spec = importlib.util.spec_from_file_location("g_box_overfit_diag", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["g_box_overfit_diag"] = mod
    spec.loader.exec_module(mod)
    return mod, hashlib.md5(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------------------- #
# the probe: hooks on the box decoder + call-through wrappers on the refined loss's pieces                 #
# --------------------------------------------------------------------------------------------------------- #
class Probe:
    """Captures, for ONE training step: the box decoder's output (grads retained on every layer's presence logit),
    the memory tensor entering it (grad retained), and -- only inside ``refined_box3d_losses`` -- every
    ``match_slots`` and ``presence_term`` call in order (one per supervised layer)."""

    def __init__(self, SP, AS):
        self.SP, self.AS = SP, AS
        self.active = False
        self.in_box = False
        self._orig = {}
        self._hooks = []
        self.reset()

    def reset(self):
        self.pred, self.mem, self.matches, self.terms = None, None, [], []

    def install(self, box_dec):
        SP, AS, pr = self.SP, self.AS, self

        def pre(_m, args):
            if pr.active and args and args[0].requires_grad:
                args[0].retain_grad()
                pr.mem = args[0]

        def post(_m, _args, out):
            if pr.active and isinstance(out, dict) and out["presence_logit"].requires_grad:
                for d in [out, *(out.get("aux") or [])]:
                    d["presence_logit"].retain_grad()
                pr.pred = out
        self._hooks = [box_dec.register_forward_pre_hook(pre), box_dec.register_forward_hook(post)]
        self._orig = {"refined_box3d_losses": SP.refined_box3d_losses, "presence_term": SP.presence_term,
                      "match_slots": AS.match_slots}
        o = self._orig

        def rbl(*a, **k):
            pr.in_box = True
            try:
                return o["refined_box3d_losses"](*a, **k)
            finally:
                pr.in_box = False

        def pt(logit, match, *, mode, ignore_w=None):
            r = o["presence_term"](logit, match, mode=mode, ignore_w=ignore_w)
            if pr.active and pr.in_box:
                pr.terms.append({"logit": logit, "match": match, "mode": mode,
                                 "ignore_w": None if ignore_w is None else ignore_w.detach().clone(),
                                 "n_matched": int(r["n_matched"]), "n_exempt": int(r["n_exempt"]),
                                 "loss": float(r["loss"].detach())})
            return r

        def ms(pred, tgt, *, presence_cost="sigmoid"):
            m = o["match_slots"](pred, tgt, presence_cost=presence_cost)
            if pr.active and pr.in_box:
                pr.matches.append({"logit_id": id(pred["presence_logit"]), "match": m, "cost": presence_cost})
            return m
        SP.refined_box3d_losses, SP.presence_term, AS.match_slots = rbl, pt, ms

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        if self._orig:
            self.SP.refined_box3d_losses = self._orig["refined_box3d_losses"]
            self.SP.presence_term = self._orig["presence_term"]
            self.AS.match_slots = self._orig["match_slots"]
        self._orig = {}


def _f(x):
    return None if x is None else float(x)


def grad_norm(params) -> float:
    import torch
    s = 0.0
    for p in params:
        if p.grad is not None:
            s += float(p.grad.detach().float().pow(2).sum())
    return s ** 0.5


def summarise_step(pr: Probe, model, step: int, row: dict, prev_rows=None) -> dict:
    """One logged step, read AFTER backward and BEFORE the optimizer step."""
    import torch
    out = {"step": step, "loss_box3d": float(row["box3d"].detach())}
    for k in ("box3d_presence", "box3d_centre", "box3d_size", "box3d_cls", "box3d_yaw", "box3d_z", "box3d_h"):
        if k in row and torch.is_tensor(row[k]):
            out[k] = float(row[k].detach())
    layers = [*(pr.pred.get("aux") or []), pr.pred]
    L = len(layers)
    if len(pr.terms) != L or len(pr.matches) != L:
        out["ERROR"] = f"captured {len(pr.terms)} presence terms / {len(pr.matches)} matchings for {L} layers"
        return out
    per = []
    last_rows = None
    for i, (lay, term, mt) in enumerate(zip(layers, pr.terms, pr.matches)):
        g = lay["presence_logit"].grad
        logit = lay["presence_logit"].detach()
        d = {"layer": i}
        # the loss read THIS layer's logits (same values; the trainer's row selection is the identity here)
        d["logit_is_decoder_output"] = bool(term["logit"].shape == logit.shape
                                            and torch.equal(term["logit"].detach(), logit))
        # ⭐ check (ii): the presence targets of layer i come from the matching run on layer i's OWN predictions
        d["targets_from_own_layer_match"] = bool(mt["logit_id"] == id(term["logit"]) and mt["match"] is term["match"])
        matched = torch.zeros_like(logit, dtype=torch.bool)
        for b, r in enumerate(term["match"]["rows"]):
            if r.numel():
                matched[b, r.to(logit.device)] = True
        iw = term["ignore_w"]
        raw_zero = (iw == 0) if iw is not None else torch.zeros_like(matched)
        exempt = raw_zero & ~matched
        unm = ~matched & ~exempt
        d["n_pos_targets"] = int(matched.sum())
        d["n_exempt"] = int(exempt.sum())
        d["n_unmatched"] = int(unm.sum())
        # ⭐ check (i): a MATCHED slot inside an IGNORE radius (raw mask 0) is RESTORED to weight 1 by presence_term;
        # the effective proof is its gradient: every matched slot must carry a non-zero presence gradient
        d["n_matched_raw_ignore_zero"] = int((raw_zero & matched).sum())
        if g is None:
            d["ERROR"] = "no gradient reached this layer's presence logit"
        else:
            ga = g.detach().abs().float()
            d["n_matched_zero_grad"] = int((ga[matched] == 0).sum())
            d["grad_abs_mean_matched"] = _f(ga[matched].mean()) if bool(matched.any()) else None
            d["grad_abs_mean_unmatched"] = _f(ga[unm].mean()) if bool(unm.any()) else None
            d["grad_abs_mean_exempt"] = _f(ga[exempt].mean()) if bool(exempt.any()) else None
            d["grad_abs_sum_matched"] = _f(ga[matched].sum())
            d["grad_abs_sum_unmatched"] = _f(ga[unm].sum())
            d["exempt_grad_max"] = _f(ga[exempt].max()) if bool(exempt.any()) else None
            # sign: a matched slot's presence must be pushed UP (dL/dlogit < 0)
            d["frac_matched_grad_negative"] = _f((g.detach()[matched] < 0).float().mean()) if bool(matched.any()) else None
        p = torch.sigmoid(logit.float())
        d["logit_mean_matched"] = _f(logit[matched].float().mean()) if bool(matched.any()) else None
        d["logit_mean_unmatched"] = _f(logit[unm].float().mean()) if bool(unm.any()) else None
        d["p_matched_median"] = _f(p[matched].median()) if bool(matched.any()) else None
        d["p_matched_max"] = _f(p[matched].max()) if bool(matched.any()) else None
        d["p_unmatched_p99"] = _f(p[unm].quantile(0.99)) if bool(unm.any()) else None
        d["p_unmatched_max"] = _f(p[unm].max()) if bool(unm.any()) else None
        d["n_conf"] = int((p >= 0.5).sum())
        d["presence_term"] = term["loss"]
        rows = [r.clone().cpu() for r in term["match"]["rows"]]
        cols = [c.clone().cpu() for c in term["match"]["cols"]]
        d["_rows"], d["_cols"] = rows, cols
        per.append(d)
    # cross-layer: the fraction of target rows matched to the SAME slot as on the last layer (re-matching is real)
    lr_, lc_ = per[-1]["_rows"], per[-1]["_cols"]
    for d in per:
        same, tot = 0, 0
        for r, c, R, C in zip(d["_rows"], d["_cols"], lr_, lc_):
            a = {int(cc): int(rr) for rr, cc in zip(r.tolist(), c.tolist())}
            bb = {int(cc): int(rr) for rr, cc in zip(R.tolist(), C.tolist())}
            for k in bb:
                tot += 1
                same += int(a.get(k) == bb[k])
        d["frac_same_slot_as_last_layer"] = same / tot if tot else None
    out["layers"] = [{k: v for k, v in d.items() if not k.startswith("_")} for d in per]
    out["_last_assign"] = [{int(c): int(r) for r, c in zip(R.tolist(), C.tolist())} for R, C in zip(lr_, lc_)]
    # the memory entering the box decoder
    mem = pr.mem
    if mem is not None and mem.grad is not None:
        mg = mem.grad.detach().float()
        mv = mem.detach().float()
        out["mem_grad_norm"] = _f(mg.norm())
        out["mem_grad_abs_mean"] = _f(mg.abs().mean())
        out["mem_abs_mean"] = _f(mv.abs().mean())
        out["mem_std_total"] = _f(mv.std())
        out["mem_std_between_frames"] = _f(mv.std(dim=0).mean()) if mv.shape[0] > 1 else None
    else:
        out["mem_grad_norm"] = None
    br = model._perception
    groups = {"trunk": list(model.core.encoder.parameters()),
              "lift": list(br.lift.parameters()) if getattr(br, "lift", None) is not None else [],
              "bev_encoder": list(br.map_branch.parameters()) if getattr(br, "map_branch", None) is not None else [],
              "box_memory": list(br.box_mem.parameters()), "box_decoder": list(br.box_dec.parameters())}
    out["grad_norm"] = {k: grad_norm(v) for k, v in groups.items()}
    out["grad_norm"]["box_decoder_queries"] = grad_norm([br.box_dec.queries])
    out["grad_norm"]["box_decoder_mem_pos"] = grad_norm([br.box_dec.mem_pos])
    return out


def assignment_stability(prev: list, cur: list):
    if prev is None:
        return None
    same, tot = 0, 0
    for a, b in zip(prev, cur):
        for k, v in b.items():
            tot += 1
            same += int(a.get(k) == v)
    return same / tot if tot else None


# --------------------------------------------------------------------------------------------------------- #
# the parts                                                                                                #
# --------------------------------------------------------------------------------------------------------- #
def part_audit(G, ad, SP, AS, steps=200, every=20):
    import torch
    ad.setup("main", G.SEED)
    params = ad.params("main")
    opt = torch.optim.AdamW(params, lr=G.LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sched = G.batch_schedule(G.N_FRAMES, G.BATCH, steps, G.SEED)       # = the first `steps` of MAIN's schedule
    pr = Probe(SP, AS)
    pr.install(ad.model._perception.box_dec)
    rows, prev = [], None
    t0 = time.time()
    try:
        for step, idx in enumerate(sched, start=1):
            opt.zero_grad(set_to_none=True)
            pr.reset()
            pr.active = step == 1 or step % every == 0
            batch = ad._batch(idx)
            if pr.active and "agent_label" in batch and not bool(batch["agent_label"].all()):
                raise SystemExit("[diag] an unlabelled window in the batch: the row selection is not the identity")
            row = ad._loss_row(batch)
            L = row["box3d"]
            L.backward()
            if pr.active:
                s = summarise_step(pr, ad.model, step, row)
                s["frames"] = list(idx)
                s["assign_stability_vs_prev_log"] = None      # different frames per step here: not comparable
                rows.append({k: v for k, v in s.items() if not k.startswith("_")})
                ly = s.get("layers") or [{}]
                log(f"audit step {step}: npos {ly[-1].get('n_pos_targets')} |g| matched "
                    f"{ly[-1].get('grad_abs_mean_matched')} unmatched {ly[-1].get('grad_abs_mean_unmatched')} "
                    f"mem-grad {s.get('mem_grad_norm')} logit m/u {ly[-1].get('logit_mean_matched')}/"
                    f"{ly[-1].get('logit_mean_unmatched')} loss {s['loss_box3d']:.3f}")
            pr.active = False
            opt.step()
    finally:
        pr.remove()
        ad.teardown()
    return {"steps": steps, "every": every, "rows": rows, "wall_s": round(time.time() - t0, 1),
            "schedule": "g_box_overfit.batch_schedule(16, 4, steps, seed 0) -- the first steps of MAIN's"}


def part_oneframe(G, ad, SP, AS, steps=500, every=25, eval_every=100):
    import numpy as np
    import torch
    all_items = ad.items
    npos = [int(x[2]) for x in ad.per_frame]
    fi = int(np.argmax(npos))                      # the frame with the most VIS-1 positives (ties -> first)
    ad.setup("main", G.SEED)
    params = ad.params("main")                     # the probe backward uses frames 0..3 (as the harness does)
    opt = torch.optim.AdamW(params, lr=G.LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    ad.items = [all_items[fi]]
    pr = Probe(SP, AS)
    pr.install(ad.model._perception.box_dec)
    rows, evals, prev = [], [], None
    t0 = time.time()

    def score():
        packs, pres = ad.evaluate()
        s = G.score_packs(packs)
        pk = packs[0]
        pm = 1.0 / (1.0 + np.exp(-pk["logit"][pk["matched"]].astype(np.float64)))
        return {**{k: s[k] for k in ("ap2m", "prec", "rec", "n_conf", "n_pos", "auroc_matched", "centre_p50_m",
                                     "size_p50_m", "z_p50_m")},
                "presence_term": pres, "eval_p_matched_median": float(np.median(pm)) if pm.size else None,
                "eval_p_matched_min": float(pm.min()) if pm.size else None,
                "eval_p_matched_max": float(pm.max()) if pm.size else None,
                "eval_frac_matched_ge_0p5": float((pm >= 0.5).mean()) if pm.size else None}
    try:
        evals.append({"step": 0, **score()})
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            pr.reset()
            pr.active = step == 1 or step % every == 0
            row = ad._loss_row(ad._batch([0]))
            row["box3d"].backward()
            if pr.active:
                s = summarise_step(pr, ad.model, step, row)
                cur = s.pop("_last_assign", None)
                s["assign_stability_vs_prev_log"] = assignment_stability(prev, cur)
                prev = cur
                rows.append(s)
                ly = s.get("layers") or [{}]
                log(f"oneframe step {step}: p_matched median {ly[-1].get('p_matched_median')} max "
                    f"{ly[-1].get('p_matched_max')} | unmatched max {ly[-1].get('p_unmatched_max')} | assign "
                    f"stable {s['assign_stability_vs_prev_log']} | loss {s['loss_box3d']:.3f}")
            pr.active = False
            opt.step()
            if step % eval_every == 0 or step == steps:
                e = {"step": step, **score()}
                evals.append(e)
                log(f"oneframe eval {step}: AP {e['ap2m']:.3f} P {e['prec']:.3f} R {e['rec']:.3f} conf "
                    f"{e['n_conf']:.0f}/{e['n_pos']:.0f} p_matched median {e['eval_p_matched_median']}")
    finally:
        pr.remove()
        ad.items = all_items
        ad.teardown()
    return {"frame_index": fi, "frame": {"sha12": ad.per_frame[fi][0], "t": int(ad.per_frame[fi][1]),
                                         "n_pos": int(ad.per_frame[fi][2]), "n_ign": int(ad.per_frame[fi][3])},
            "steps": steps, "rows": rows, "evals": evals, "wall_s": round(time.time() - t0, 1),
            "batch": "the one frame, every step (batch 1)", "lr": G.LR}


def part_control(G, ad, tr, steps=None):
    """The refcv6 head under the SAME harness loop (run_arm): legacy flags, 100 box queries, VIS-1 scoring."""
    a_main = ad.args
    a2 = copy.copy(a_main)
    a2.slot_presence_loss, a2.slot_presence_prior = "bce", 0.05
    a2.slot_deep_supervision, a2.slot_vis1, a2.vis1_sidecar = False, False, None
    ad.args = a2
    orig = tr._perc.build_perception_branch
    built = {}

    def bpb(model, cfg):
        cfg2 = dataclasses.replace(cfg, n_queries=100)
        built["cfg"] = {k: getattr(cfg2, k) for k in ("n_queries", "presence_loss", "presence_prior",
                                                      "deep_supervision", "vis1") if hasattr(cfg2, k)}
        return orig(model, cfg2)
    tr._perc.build_perception_branch = bpb
    try:
        res = G.run_arm(ad, "targets_filter_only", steps=steps or G.STEPS)
    finally:
        tr._perc.build_perception_branch = orig
        ad.args = a_main
    res.pop("controls_step0_packs", None)
    res["arm"] = "refcv6_control"
    res["verdict_note"] = ("informative (diagnostic): the harness's targets_filter_only scoring switch -- trained on "
                           "the refcv6 targets, scored under the VIS-1 eval rule at sigma >= 0.5")
    res["built_box_cfg"] = built.get("cfg")
    res["departures_from_launch_argv"] = {
        "slot_presence_loss": "bce", "slot_presence_prior": 0.05, "slot_deep_supervision": False,
        "slot_vis1": False, "vis1_sidecar": "None for the BUILD (the 16 frames' VIS-1 blocks were joined once, "
                                            "for scoring only)",
        "box n_queries": "100 (a wrapper around build_perception_branch; the argv has no box-query flag)",
        "agent head": "rebuilt legacy by the same args (not in the box loss)"}
    return res


RUNGS = ("main", "frozen_trunk", "lr_2e-5", "bce_presence", "no_deep_sup", "q100", "refcv6_head")


def _pack_stats(pk, prev_matched=None):
    """Eval-rule statistics of ONE frame's pack (works for every loss path, legacy included)."""
    import numpy as np
    lg = pk["logit"].astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-lg))
    m = pk["matched"].astype(bool)
    ex = pk["exempt"].astype(bool)
    um = ~m & ~ex
    xy = pk["xy"].astype(np.float64)
    gxy = pk["gt_xy"][pk["pos"].astype(bool)].astype(np.float64) if "pos" in pk else pk["gt_xy"].astype(np.float64)
    out = {"p_matched_median": float(np.median(p[m])) if m.any() else None,
           "p_matched_max": float(p[m].max()) if m.any() else None,
           "p_unmatched_max": float(p[um].max()) if um.any() else None,
           "n_conf": int((p >= 0.5).sum()),
           "slot_centre_spread_m": float(xy.std(axis=0).mean()),
           "matched_set_overlap_vs_prev": (None if prev_matched is None else
                                           float(len(set(np.nonzero(m)[0]) & prev_matched) / max(1, int(m.sum()))))}
    if gxy.size:
        d = np.sqrt(((gxy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))           # [A, Q]
        ds = np.sort(d, axis=1)
        out["tgt_to_nearest_slot_m"] = float(ds[:, 0].mean())
        out["tgt_to_2nd_slot_m"] = float(ds[:, 1].mean()) if ds.shape[1] > 1 else None
        out["slots_within_1m_of_a_target"] = int((d <= 1.0).any(axis=0).sum())
    return out, set(np.nonzero(m)[0])


def memory_frame_share(ad, all_items):
    """Eval mode, no grad: the memory entering the box decoder for ALL 16 frames; the frame-specific share =
    (std over frames, per token and channel, averaged) / (std over everything) -- the audit's metric (there over the
    4 frames of a batch), here over the full frame set."""
    import torch
    mems = []
    br = ad.model._perception
    h = br.box_dec.register_forward_pre_hook(lambda _m, args: mems.append(args[0].detach().float()))
    items = ad.items
    ad.items = all_items
    was_training = ad.model.training
    ad.model.eval()
    try:
        with torch.no_grad():
            for k in range(0, len(all_items), 4):
                ad._loss_row(ad._batch(list(range(k, min(k + 4, len(all_items))))))
    finally:
        h.remove()
        ad.items = items
        ad.model.train(was_training)
    m = torch.cat(mems, 0)
    between, total = float(m.std(dim=0).mean()), float(m.std())
    return {"mem_between_frames_std": between, "mem_total_std": total,
            "mem_frame_share": between / total if total > 0 else None, "mem_abs_mean": float(m.abs().mean())}


class AssignProbe:
    """Every ``match_slots`` call inside ``box3d_loss_row`` (legacy AND refined paths); the LAST one is the last
    layer's assignment. Plus the box decoder's train-mode output."""

    def __init__(self, PB, AS):
        self.PB, self.AS, self.active, self.inside = PB, AS, False, False
        self.calls, self.pred, self._orig, self._h = [], None, {}, None

    def install(self, box_dec):
        from tanitad.models import box3d_head as _B3        # the legacy path imported match_slots BY NAME
        self._B3 = _B3
        o = self._orig = {"box3d_loss_row": self.PB.box3d_loss_row, "match_slots": self.AS.match_slots,
                          "b3_match_slots": _B3.match_slots}
        pr = self

        def row(*a, **k):
            pr.inside = True
            try:
                return o["box3d_loss_row"](*a, **k)
            finally:
                pr.inside = False

        def ms(pred, tgt, *, presence_cost="sigmoid"):
            m = o["match_slots"](pred, tgt, presence_cost=presence_cost)
            if pr.active and pr.inside:
                pr.calls.append(m)
            return m

        def post(_m, _a, out):
            if pr.active:
                pr.pred = out
        self.PB.box3d_loss_row, self.AS.match_slots, _B3.match_slots = row, ms, ms
        self._h = box_dec.register_forward_hook(post)

    def remove(self):
        if self._orig:
            self.PB.box3d_loss_row = self._orig["box3d_loss_row"]
            self.AS.match_slots = self._orig["match_slots"]
            self._B3.match_slots = self._orig["b3_match_slots"]
        if self._h is not None:
            self._h.remove()
        self._orig, self._h = {}, None

    def last_assign(self):
        if not self.calls:
            return None
        m = self.calls[-1]
        return [{int(c): int(r) for r, c in zip(R.tolist(), C.tolist())} for R, C in zip(m["rows"], m["cols"])]


def part_ladder(G, ad, tr, rungs=RUNGS, steps=500, every=25):
    """One-frame attribution ladder: MAIN's one-frame failure re-run with ONE variable changed per rung."""
    import numpy as np
    import torch
    all_items = ad.items
    fi = int(np.argmax([int(x[2]) for x in ad.per_frame]))
    out = {"frame_index": fi, "frame": {"sha12": ad.per_frame[fi][0], "t": int(ad.per_frame[fi][1]),
                                        "n_pos": int(ad.per_frame[fi][2])}, "steps": steps, "every": every,
           "rungs": {}}
    orig_bpb = tr._perc.build_perception_branch
    from tanitad.models import agent_slots as _AS
    PB = tr._perc
    for v in rungs:
        t0 = time.time()
        a_main = ad.args
        a2 = copy.copy(a_main)
        arm, param_arm, lr, q100 = "main", "main", G.LR, False
        if v == "main":
            pass
        elif v == "frozen_trunk":
            param_arm = "frozen_trunk"
        elif v == "lr_2e-5":
            lr = 2e-5
        elif v == "bce_presence":
            a2.slot_presence_loss = "bce"
        elif v == "no_deep_sup":
            a2.slot_deep_supervision = False
        elif v == "q100":
            q100 = True
        elif v == "refcv6_head":
            a2.slot_presence_loss, a2.slot_presence_prior = "bce", 0.05
            a2.slot_deep_supervision, a2.slot_vis1, a2.vis1_sidecar = False, False, None
            arm, q100 = "targets_filter_only", True
        else:
            raise SystemExit(f"unknown rung {v!r}")
        ad.args = a2
        if q100:
            tr._perc.build_perception_branch = (
                lambda model, cfg: orig_bpb(model, dataclasses.replace(cfg, n_queries=100)))
        rows = []
        ap_ = AssignProbe(PB, _AS)
        try:
            ad.setup(arm, G.SEED)
            params = ad.params(param_arm)
            opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
            ap_.install(ad.model._perception.box_dec)
            ms0 = memory_frame_share(ad, all_items)
            ad.items = [all_items[fi]]
            prev, losses, prev_assign = None, [], None
            pk0, _ = ad.evaluate()
            st, prev = _pack_stats(pk0[0])
            rows.append({"step": 0, **st, **{k: G.score_packs(pk0)[k] for k in ("ap2m", "centre_p50_m")}, **ms0})
            for step in range(1, steps + 1):
                opt.zero_grad(set_to_none=True)
                logged = step % every == 0
                ap_.active, ap_.calls, ap_.pred = logged, [], None
                L = ad._loss_row(ad._batch([0]))["box3d"]
                L.backward()
                ap_.active = False
                opt.step()
                losses.append(float(L.detach()))
                if logged:
                    cur = ap_.last_assign()
                    stab = None
                    if cur is not None and prev_assign is not None:
                        tot = sum(len(b) for b in cur)
                        same = sum(int(a.get(k) == v2) for a, b in zip(prev_assign, cur) for k, v2 in b.items())
                        stab = same / tot if tot else None
                    tm = {}
                    if cur is not None and ap_.pred is not None:
                        lg = ap_.pred["presence_logit"].detach().float()[0]
                        pp = torch.sigmoid(lg)
                        idx = torch.tensor(sorted(cur[0].values()), dtype=torch.long, device=pp.device)
                        mask = torch.zeros_like(pp, dtype=torch.bool)
                        if idx.numel():
                            mask[idx] = True
                        tm = {"train_p_matched_median": float(pp[mask].median()) if bool(mask.any()) else None,
                              "train_p_matched_max": float(pp[mask].max()) if bool(mask.any()) else None,
                              "train_p_unmatched_max": float(pp[~mask].max()) if bool((~mask).any()) else None,
                              "train_slot_centre_spread_m": float(
                                  ap_.pred["box"].detach().float()[0, :, :2].std(dim=0).mean())}
                    prev_assign = cur
                    pks, pres = ad.evaluate()
                    st, prev = _pack_stats(pks[0], prev)
                    sc = G.score_packs(pks)
                    msr = memory_frame_share(ad, all_items)
                    rows.append({"step": step, "loss_mean": float(np.mean(losses[-every:])), **st,
                                 "ap2m": sc["ap2m"], "centre_p50_m": sc["centre_p50_m"], "presence_term": pres,
                                 "assign_stability_25": stab, **tm, **msr})
            last = rows[-1]
            stabs = [r["assign_stability_25"] for r in rows if r.get("assign_stability_25") is not None]
            log(f"ladder {v}: p_matched median {last['p_matched_median']} max {last['p_matched_max']} "
                f"unmatched max {last['p_unmatched_max']} conf {last['n_conf']} AP {last['ap2m']:.3f} "
                f"centre {last['centre_p50_m']:.3f} spread {last['slot_centre_spread_m']:.2f} "
                f"assign-stab mean {np.mean(stabs) if stabs else None} mem share "
                f"{rows[0]['mem_frame_share']:.3f}->{last['mem_frame_share']:.3f} loss {last['loss_mean']:.3f}")
            out["rungs"][v] = {"rows": rows, "n_trainable_params": int(sum(p.numel() for p in params)), "lr": lr,
                               "arm_for_setup": arm, "param_arm": param_arm, "q100": q100,
                               "wall_s": round(time.time() - t0, 1),
                               "max_p_matched_median": max((r["p_matched_median"] or 0.0) for r in rows),
                               "first_step_p_matched_median_ge_0p5": next(
                                   (r["step"] for r in rows if (r["p_matched_median"] or 0.0) >= 0.5), None)}
        except Exception as exc:                                         # noqa: BLE001 -- one rung, recorded
            import traceback
            out["rungs"][v] = {"ERROR": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(),
                               "rows": rows}
            log(f"ladder {v}: ERROR {type(exc).__name__}: {exc}")
        finally:
            ap_.remove()
            tr._perc.build_perception_branch = orig_bpb
            ad.args = a_main
            ad.items = all_items
            if getattr(ad, "model", None) is not None:
                ad.teardown()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--launch-argv", required=True)
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="audit,oneframe,control")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--audit-steps", type=int, default=200)
    ap.add_argument("--oneframe-steps", type=int, default=500)
    ap.add_argument("--control-steps", type=int, default=None)
    ap.add_argument("--rungs", default=",".join(RUNGS))
    ap.add_argument("--ladder-steps", type=int, default=500)
    a = ap.parse_args(argv)
    tree = Path(a.tree).resolve()
    G, gmd5 = load_harness(tree)
    import torch
    audit = Path(a.audit_dir)
    fs_bytes = (audit / "raw" / "visibility" / "gbo_frameset.json").read_bytes()
    frameset = json.loads(fs_bytes.decode("utf-8"))
    launch_argv = json.loads(Path(a.launch_argv).read_text(encoding="utf-8"))
    rec = {"tool": "gbo_diagnose.py", "binding": False,
           "purpose": "Master Mind 2026-09-27 ~09:20: diagnose MAIN's early G-BOX-OVERFIT FAIL before any new lever",
           "tree": str(tree), "harness_md5": gmd5, "launch_argv_sha256":
               hashlib.sha256(json.dumps(launch_argv).encode()).hexdigest(),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": a.device,
           "script_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(), "parts": {}}
    tr = G._load_by_path("refc_v3_train_for_gbo", G.SCRIPTS / "refc_v3_train.py")
    import tanitad
    if not os.path.abspath(tanitad.__file__).startswith(str(tree)):
        raise SystemExit(f"[diag] tanitad imported from {tanitad.__file__}, not {tree}")
    os.environ["REFCV6_REPO"] = str(G.STACK.parent)
    L = G._load_by_path("refcv6_loader_for_gbo", audit / "code" / "refcv6_loader.py")
    from tanitad.models import agent_slots as AS
    from tanitad.models import slot_presence as SP
    ad = G.TrainerAdapter(tr, L, launch_argv, frameset, a.device)
    rec["C3"] = G.control_c3(ad.per_frame, frameset, hashlib.md5(fs_bytes).hexdigest())
    log(f"C3 {'PASS' if rec['C3']['pass'] else 'FAIL'}: {rec['C3']['totals']}")
    if not rec["C3"]["pass"]:
        Path(a.out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        raise SystemExit("[diag] C3 failed")
    for part in [x for x in a.parts.split(",") if x]:
        t = time.time()
        log(f"part {part} starts")
        try:
            if part == "audit":
                rec["parts"][part] = part_audit(G, ad, SP, AS, steps=a.audit_steps)
            elif part == "oneframe":
                rec["parts"][part] = part_oneframe(G, ad, SP, AS, steps=a.oneframe_steps)
            elif part == "control":
                rec["parts"][part] = part_control(G, ad, tr, steps=a.control_steps)
            elif part == "ladder":
                rec["parts"][part] = part_ladder(G, ad, tr, rungs=[x for x in a.rungs.split(",") if x],
                                                 steps=a.ladder_steps)
            else:
                raise SystemExit(f"unknown part {part!r}")
        except Exception as exc:                                           # noqa: BLE001 -- recorded, then raised
            import traceback
            rec["parts"][part] = {"ERROR": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
            Path(a.out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
            raise
        rec["parts"][part]["part_wall_s"] = round(time.time() - t, 1)
        if torch.cuda.is_available():
            rec["parts"][part]["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        Path(a.out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        log(f"part {part} done ({rec['parts'][part]['part_wall_s']} s)")
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    log("DIAG DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
