#!/usr/bin/env python3
"""g_box_overfit.py -- G-BOX-OVERFIT (SPEC_REFCV7 §14 A9, §15.1 A10): can the refcv7 LAUNCH box path memorise 16
real TRAIN frames?

The protocol is the box-head audit's ``raw/PREREG_G_BOX_OVERFIT.md`` (md5 ``594c71196cc5bbd527fee40b2cb0e3f1``)
VERBATIM, with A10's one reconciliation: rows with vis_frac < 0.05 (the prereg's "DROPPED", 49 rows) are IGNORE, so
control C3 reads **113 POSITIVE / 77 IGNORE**, and the loss mask and the DontCare scoring use all 77. Rows removed by
``visible_target_filter`` stay outside the target set.

LITERALS (changing one after a run is a goalpost move):
  * frames: ``raw/visibility/gbo_frameset.json`` (md5 b291404c...), 16 TRAIN windows; NOW = t + 7;
  * model: the LAUNCH box path (the launch argv's perception branch, reading the launch trunk and BEV), built fresh
    at seed 0 through the trainer's own pin; trainable = every tensor the box loss reaches (GATING) or box decoder +
    box memory only (``frozen_trunk``, informative);
  * loss: EXACTLY the launch box loss -- the trainer's own ``compute_losses_v3`` term ``box3d`` (focal presence,
    per-layer supervision, VIS-1 POSITIVES + the 77-row IGNORE mask, the launch class weights, z and h);
  * ⭐ A13 (SPEC_REFCV7 §18): the LAUNCH optimiser AS BUILT -- the trainer's own ``build_optimizer`` on the launch
    argv (``--opt dd --lr 1e-4``: AdamW, DD's two groups, the trunk at 0.5 x lr, weight decay 1e-4) and
    ``clip_grad_norm_`` 10.0 every step, held at those PEAK lrs, constant, no warm-up. (The early runs' AdamW 2e-4
    on every tensor, no clip, contradicted the prereg's own claim to test the launch box path.) The config is the
    refcv7 canonical argv + the A9 flags, so the box head reads NEW-2's pooled BEV (``--bev-source
    map_hires_pool``) and the replay builds the 10 cm branch and attaches its fine store as ``train()`` does;
  * batch 4 frames per step by a seeded ``randperm`` over the 16 (seed 0); N = 2,000 steps (500 passes);
  * log every 100 steps; score on all 16 frames in eval mode under the eval rule (P0 / A9: greedy 2 m BEV,
    POSITIVE ∪ IGNORE, DontCare) at the declared gate sigma(presence) >= 0.5.

PASS at step 2,000 iff ALL of: (1) AP@2 m >= 0.90; (2) P >= 0.90 AND R >= 0.90; (3) |Σ confident - 113| / 113 <=
0.10; (4) Hungarian-pair median centre error <= 0.30 m, median |dl|+|dw| <= 0.30 m, median |dz| <= 0.15 m; (5) greedy-
TP class accuracy >= 0.90; (6) the loss finite at every step and the presence term <= 0.25 x its step-0 value.
MUST FAIL: ``memory_zeros`` fails (1); ``presence_w0`` fails (2) and (3) -- if either passes the result is VOID.
CONTROLS: C1 GT-as-slots reads AP 1.0, P = R = 1.0, Σ confident = 113 exactly; C2 constant presence reads AUROC 0.5
exactly; C3 113 / 77 with per-frame POS and IGN + DROP of the frame set and its md5; C4 step 0 is reported.

``--binding`` is OFF by default: an early run stamps ``"binding": false`` and a ``commit`` field that can never
equal a launch sha, so the launch gate cannot consume it (Master Mind, 2026-09-27). A11 (SPEC_REFCV7 §16): a
BINDING record binds through the code closure that ``stack/scripts/closure_run.py`` records around this UNMODIFIED
script (every in-tree module in ``sys.modules`` -- ``_load_by_path`` registers the trainer there, and the audit
loader is VENDORED as ``tanitad.eval.refcv6_loader`` so it lies in stack/, the tree the closure is verified on --
the data-contract sha256s, the argv, this record's sha256, the exit status), not through ``--commit``.
``--launch-argv`` takes the CANONICAL run file (``stack/ops/runs.d/<run>.argv.json``, an object with ``"argv"``) or a
plain list; the record's ``launch_argv_sha256`` is over the argv list as compact JSON (the gate's form).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
STACK = HERE.parents[1]
SCRIPTS = STACK / "scripts"
for _p in (str(STACK), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

PREREG_MD5 = "594c71196cc5bbd527fee40b2cb0e3f1"
FRAMESET_MD5 = "b291404c36f83c3e397e3b90367e8e7b"
SELECTOR_MD5 = "f9f93f92b60a979b581b8f78be2ec6db"
N_FRAMES, N_POS, N_IGN = 16, 113, 77
STEPS, BATCH, LR, LOG_EVERY, SEED = 2000, 4, 2e-4, 100, 0
BARS = {"ap2m": 0.90, "prec": 0.90, "rec": 0.90, "count_rel": 0.10, "centre_p50_m": 0.30,
        "size_p50_m": 0.30, "z_p50_m": 0.15, "cls_acc": 0.90, "presence_ratio": 0.25}
ARMS = ("main", "memory_zeros", "presence_w0", "frozen_trunk", "targets_filter_only")
MUST_FAIL = {"memory_zeros": ("1",), "presence_w0": ("2", "3")}
INFORMATIVE = ("frozen_trunk", "targets_filter_only")
#: A14 (SPEC_REFCV7 §19) + A14.1 (§19.1): the ONE-FRAME test. The frame = the frame-set frame with the most VIS-1
#: POSITIVES (ties -> first): 384cb23868d0 t = 149, 13 POSITIVE -- the ladder's frame. PASS iff BOTH at step 500:
#: (i) the median sigmoid presence of the Hungarian-matched slots (eval rule) >= 0.50; (ii) the mean share of targets
#: keeping their matched ANCHOR over the final 100 steps, in 25-step windows, >= 0.80 -- the anchor is the selected CELL
#: under HQS (``--slot-query-select heatmap``: a top-K rank index churns as scores move, the cell does not) and the
#: QUERY otherwise (``learned_ref``, and the ``unanchored`` red arm = the MAIN decoder, run as its own invocation).
ONE_FRAME_STEPS, ONE_FRAME_EVERY = 500, 25
ONE_FRAME_BARS = {"p_matched_median": 0.50, "assign_keep_final100": 0.80}
ONE_FRAME_ARMS = ("main", "anchors_removed")
#: A13: the trainer's ``clip_grad_norm_(model.parameters(), 10.0)`` in ``train()`` -- a LITERAL there, so a test reads
#: it back out of the trainer's source (``trainer_clip_literal``) and holds this constant to it.
CLIP_NORM = 10.0
OPTIMIZER_RULE = ("A13 (SPEC_REFCV7 §18): the trainer's build_optimizer(model, launch args) -- --opt dd: AdamW, "
                  "DD's two groups (core.encoder.* at --encoder-lr-mult x lr, the rest at --lr), --weight-decay -- "
                  "held at those PEAK lrs, constant, no warm-up; clip_grad_norm_(model.parameters(), 10.0) every step")


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def log(msg: str) -> None:
    print(f"[gbo {time.strftime('%H:%M:%S')}] {msg}", flush=True)


GPU_JOB_PATTERNS = ("gmo_early", "gate_", "launch_gate", "refc_v3_train.py", "g_box_overfit.py")


def other_gpu_jobs(patterns=GPU_JOB_PATTERNS, proc_root: str = "/proc", me: int | None = None) -> list:
    """Other python processes whose command line names a GPU job (Thor's nvidia-smi reports no per-process usage),
    INCLUDING another harness run (a concurrent +R6 arm shares the GPU exactly as a foreign job does). Stamped on
    every log row as ``gpu_shared_with``: a shared GPU makes s/step informative only (Master Mind 2026-09-27). Only
    this process's own pid is excluded; nothing is ever signalled."""
    out = []
    me = os.getpid() if me is None else int(me)
    try:
        for d in sorted(os.listdir(proc_root)):
            if not d.isdigit() or int(d) == me:
                continue
            try:
                with open(os.path.join(proc_root, d, "cmdline"), "rb") as fh:
                    cmd = fh.read().replace(b"\x00", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            if "python" in cmd and any(p in cmd for p in patterns):
                out.append(f"pid {d}: {cmd[:120]}")
    except OSError:
        return [f"<unreadable {proc_root}>"]
    return out


# --------------------------------------------------------------------------------------------------------- #
# the literal pieces (pure; unit-tested)                                                                   #
# --------------------------------------------------------------------------------------------------------- #
def trainer_clip_literal(src: str) -> float:
    """The max-norm literal of the ONE ``clip_grad_norm_(model.parameters(), <x>)`` call inside the trainer's
    ``train()``; refuses zero or several (a second clip site would make "the launch's clip" ambiguous)."""
    import ast
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train"]
    if len(fn) != 1:
        raise SystemExit(f"[gbo] {len(fn)} top-level train() functions in the trainer source")
    vals = []
    for n in ast.walk(fn[0]):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) == "clip_grad_norm_":
            if len(n.args) < 2 or not isinstance(n.args[1], ast.Constant):
                raise SystemExit("[gbo] a clip_grad_norm_ call in train() without a literal max-norm")
            vals.append(float(n.args[1].value))
    if len(vals) != 1:
        raise SystemExit(f"[gbo] {len(vals)} clip_grad_norm_ calls in train(); A13 reads exactly one")
    return vals[0]


def optimizer_spec(opt, clip) -> dict:
    """What an optimiser IS, comparably: its class, every group's name / lr / weight decay / betas / eps and tensor
    ids, and the clip applied before each step (None = no clip)."""
    groups = []
    for g in opt.param_groups:
        groups.append({"name": g.get("name"), "lr": float(g["lr"]), "weight_decay": float(g.get("weight_decay", 0.0)),
                       "betas": [float(x) for x in g.get("betas", ())], "eps": float(g.get("eps", 0.0)),
                       "n_tensors": len(g["params"]), "param_ids": sorted(id(q) for q in g["params"])})
    return {"class": type(opt).__name__, "groups": groups, "clip": None if clip is None else float(clip)}


def spec_mismatches(got: dict, want: dict) -> list:
    """Every difference between two ``optimizer_spec``s, named; [] means equal."""
    out = []
    if got["class"] != want["class"]:
        out.append(f"class {got['class']} != {want['class']}")
    if got["clip"] != want["clip"]:
        out.append(f"clip {got['clip']} != {want['clip']}")
    if len(got["groups"]) != len(want["groups"]):
        out.append(f"groups {len(got['groups'])} != {len(want['groups'])}")
        return out
    for i, (a, b) in enumerate(zip(got["groups"], want["groups"])):
        for k in ("name", "lr", "weight_decay", "betas", "eps", "n_tensors", "param_ids"):
            if a[k] != b[k]:
                out.append(f"group {i} {k}: {a[k] if k != 'param_ids' else len(a[k])} != "
                           f"{b[k] if k != 'param_ids' else len(b[k])}")
    return out


def public_spec(spec: dict) -> dict:
    """``optimizer_spec`` for a record (tensor ids dropped: they are process-local)."""
    return {**spec, "groups": [{k: v for k, v in g.items() if k != "param_ids"} for g in spec["groups"]]}


def launch_optimizer(tr, model, args):
    """A13: the LAUNCH optimiser AS BUILT -- ``tr.build_optimizer(model, args)``, never a re-spelling of it -- held
    at its peak lrs (no LambdaLR is attached), with the trainer's clip. -> (optimizer, spec)."""
    opt = tr.build_optimizer(model, args)
    return opt, optimizer_spec(opt, CLIP_NORM)


def batch_schedule(n_frames: int = N_FRAMES, batch: int = BATCH, steps: int = STEPS, seed: int = SEED) -> list:
    """The harness's seeded ``randperm`` over the frames, ``batch`` per step, passes back to back."""
    g = torch.Generator().manual_seed(int(seed))
    out = []
    while len(out) < steps:
        perm = torch.randperm(int(n_frames), generator=g).tolist()
        for k in range(0, int(n_frames), int(batch)):
            out.append(perm[k:k + int(batch)])
    return out[:steps]


def ece_10bin(p, y) -> float:
    """10-bin ECE of sigma(presence) vs the Hungarian-matched label -- INFORMATIVE (the prereg logs it; A10: no
    calibration map is fitted and no check reads this)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    bins = np.minimum((p * 10).astype(int), 9)
    e = 0.0
    for k in range(10):
        m = bins == k
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def score_packs(packs: list) -> dict:
    """The prereg's per-log-point numbers from the eval rule's packs (``detection_metrics`` for everything it
    defines; the size / z medians and the ECE here)."""
    from tanitad.eval import detection_metrics as DM
    m = DM.summarise(packs, "box3d")
    sz = np.concatenate([pk.get("pair_size_err", np.zeros(0)) for pk in packs]) if packs else np.zeros(0)
    zz = np.concatenate([pk.get("pair_z_err", np.zeros(0)) for pk in packs]) if packs else np.zeros(0)
    P = np.concatenate([1.0 / (1.0 + np.exp(-pk["logit"].astype(np.float64)))[~pk["exempt"]] for pk in packs]) \
        if packs else np.zeros(0)
    M = np.concatenate([pk["matched"][~pk["exempt"]] for pk in packs]) if packs else np.zeros(0, bool)
    pr = DM.pr_equal_gate(packs) if packs else {"gate": float("nan"), "precision": float("nan"),
                                                 "recall": float("nan")}
    return {"ap2m": m["eval_box3d_ap2m"], "prec": m["eval_box3d_prec@gate"], "rec": m["eval_box3d_rec@gate"],
            "n_conf": m["eval_box3d_n_conf"], "n_pos": m["eval_box3d_n_pos"], "n_ignore": m["eval_box3d_n_ignore"],
            "conf_ratio": m["eval_box3d_conf_ratio"], "centre_p50_m": m["eval_box3d_centre_err_p50"],
            "size_p50_m": float(np.median(sz)) if sz.size else float("nan"),
            "z_p50_m": float(np.median(zz)) if zz.size else float("nan"),
            "cls_acc": m["eval_box3d_cls_acc_tp"], "auroc_matched": m["eval_box3d_auroc_matched"],
            "presence_ece_matched_informative": ece_10bin(P, M),
            # INFORMATIVE (Master Mind 2026-09-27): the TRAIN P = R gate on these frames -- AP is threshold-free and
            # shows whether RANKING is learning even while no slot crosses the declared 0.5
            "pr_equal_gate_informative": pr["gate"], "pr_equal_prec_informative": pr["precision"],
            "pr_equal_rec_informative": pr["recall"]}


def criteria(s: dict, *, presence_step0: float, presence_end: float, finite: bool) -> dict:
    """PREREG §5, literally. Each entry: value(s), bar, pass. NaN never passes."""
    def ge(v, b):
        return bool(v == v and v >= b)

    def le(v, b):
        return bool(v == v and v <= b)
    cnt = abs(float(s["n_conf"]) - N_POS) / N_POS
    ratio = (presence_end / presence_step0) if presence_step0 and presence_step0 == presence_step0 else float("nan")
    c = {"1": {"what": "AP@2 m BEV >= 0.90", "value": s["ap2m"], "pass": ge(s["ap2m"], BARS["ap2m"])},
         "2": {"what": "precision >= 0.90 AND recall >= 0.90 at the declared gate",
               "value": [s["prec"], s["rec"]], "pass": ge(s["prec"], BARS["prec"]) and ge(s["rec"], BARS["rec"])},
         "3": {"what": "|sum confident - 113| / 113 <= 0.10", "value": [s["n_conf"], cnt],
               "pass": le(cnt, BARS["count_rel"])},
         "4": {"what": "median centre <= 0.30 m, |dl|+|dw| <= 0.30 m, |dz| <= 0.15 m (Hungarian pairs)",
               "value": [s["centre_p50_m"], s["size_p50_m"], s["z_p50_m"]],
               "pass": le(s["centre_p50_m"], BARS["centre_p50_m"]) and le(s["size_p50_m"], BARS["size_p50_m"])
               and le(s["z_p50_m"], BARS["z_p50_m"])},
         "5": {"what": "greedy-TP class accuracy >= 0.90", "value": s["cls_acc"], "pass": ge(s["cls_acc"], BARS["cls_acc"])},
         "6": {"what": "loss finite every step AND presence(2000) <= 0.25 x presence(0)",
               "value": [bool(finite), presence_step0, presence_end, ratio],
               "pass": bool(finite) and le(ratio, BARS["presence_ratio"])}}
    c["PASS"] = all(v["pass"] for k, v in c.items() if k != "PASS")
    return c


def arm_verdict(arm: str, crit: dict) -> str:
    """``main`` -> PASS / FAIL; a must-fail arm -> 'failed as required' or VOID; informative arms -> reported."""
    if arm == "main":
        return "PASS" if crit["PASS"] else "FAIL"
    if arm in MUST_FAIL:
        failed = all(not crit[k]["pass"] for k in MUST_FAIL[arm])
        return "failed as required" if failed else "VOID (a must-fail arm passed its criteria)"
    return "informative: " + ("would pass" if crit["PASS"] else "would fail")


def gt_as_slot_packs(packs: list) -> list:
    """C1: every POSITIVE written into the slot format (presence +20, the true class), all other slots empty (-20)
    and parked at (1000, 1000) m; IGNORE rows get no slot."""
    out = []
    for pk in packs:
        n = len(pk["logit"])
        pos = np.nonzero(pk["pos"])[0]
        q = copy.deepcopy(pk)
        q["logit"] = np.full(n, -20.0, np.float32)
        q["xy"] = np.full((n, 2), 1000.0, np.float32)
        q["cls"] = np.zeros(n, np.int16)
        q["matched"] = np.zeros(n, bool)
        q["exempt"] = np.zeros(n, bool)
        for s, g in enumerate(pos.tolist()[:n]):
            q["logit"][s] = 20.0
            q["xy"][s] = pk["gt_xy"][g]
            q["cls"][s] = pk["gt_cls"][g]
            q["matched"][s] = True
        q["cls_corr"] = q["cls"].copy()
        q["pair_err"] = np.zeros(len(pos))
        q["pair_size_err"] = np.zeros(len(pos))
        q["pair_z_err"] = np.zeros(len(pos))
        out.append(q)
    return out


def constant_presence_packs(packs: list) -> list:
    out = []
    for pk in packs:
        q = copy.deepcopy(pk)
        q["logit"] = np.zeros_like(pk["logit"])
        out.append(q)
    return out


def control_c1_c2(packs: list) -> dict:
    from tanitad.eval import detection_metrics as DM
    s1 = score_packs(gt_as_slot_packs(packs))
    c1 = {"ap2m": s1["ap2m"], "prec": s1["prec"], "rec": s1["rec"], "n_conf": s1["n_conf"],
          "pass": s1["ap2m"] == 1.0 and s1["prec"] == 1.0 and s1["rec"] == 1.0 and s1["n_conf"] == float(N_POS)}
    m2 = DM.summarise(constant_presence_packs(packs), "box3d")
    c2 = {"auroc_matched": m2["eval_box3d_auroc_matched"], "pass": m2["eval_box3d_auroc_matched"] == 0.5}
    return {"C1": c1, "C2": c2}


def control_c3(per_frame: list, frameset: dict, frameset_md5: str) -> dict:
    """``per_frame`` = [(sha12, t, n_pos, n_ign)] in the frame set's order, from ``vis1.vis1_split`` on the
    harness's OWN target blocks. A10: POS and IGN + DROP of the frame set; totals 113 / 77."""
    want = [(f["sha12"], int(f["t"]), int(f["positive"]), int(f["ignore"]) + int(f["dropped"]))
            for f in frameset["frames"]]
    got = [(str(a), int(b), int(c), int(d)) for (a, b, c, d) in per_frame]
    tot = (sum(x[2] for x in got), sum(x[3] for x in got))
    return {"per_frame_equal": got == want, "totals": list(tot), "want_totals": [N_POS, N_IGN],
            "frameset_md5": frameset_md5, "frameset_md5_ok": frameset_md5 == FRAMESET_MD5,
            "mismatch": [(g, w) for g, w in zip(got, want) if g != w][:8],
            "pass": got == want and tot == (N_POS, N_IGN) and frameset_md5 == FRAMESET_MD5}


# --------------------------------------------------------------------------------------------------------- #
# the loop (adapter-agnostic)                                                                              #
# --------------------------------------------------------------------------------------------------------- #
def run_arm(adapter, arm: str, *, steps: int = STEPS, log_every: int = LOG_EVERY, lr: float | None = None,
            seed: int = SEED, n_frames: int = N_FRAMES, batch: int = BATCH) -> dict:
    """One arm. ``adapter`` provides: ``setup(arm, seed)``, ``params(arm)``, ``optimizer(arm, params, lr) ->
    (optimizer, spec)``, ``clip()`` (applied before every step when the spec carries a clip), ``loss(idx) ->
    tensor``, ``evaluate() -> (packs, presence)``, ``teardown()``. ``lr`` is for TOY adapters only: the real adapter
    REFUSES it (A13 -- the launch optimiser, as built)."""
    adapter.setup(arm, seed)
    t0 = time.time()
    sched = batch_schedule(n_frames, batch, steps, seed)
    packs0, pres0 = adapter.evaluate()
    rows = [{"step": 0, **score_packs(packs0), "presence": pres0}]
    params = adapter.params(arm)
    opt, ospec = adapter.optimizer(arm, params, lr=lr)
    do_clip = ospec.get("clip") is not None
    finite = True
    losses = []
    t_steps = time.time()
    for step, idx in enumerate(sched, start=1):
        opt.zero_grad(set_to_none=True)
        L = adapter.loss(idx)
        if not bool(torch.isfinite(L).all()):
            finite = False
            log(f"{arm}: NON-FINITE loss at step {step}")
            break
        L.backward()
        if do_clip:
            adapter.clip()
        opt.step()
        losses.append(float(L.detach()))
        if step % log_every == 0 or step == steps:
            pk, pres = adapter.evaluate()
            row = {"step": step, "loss_mean_since_last": float(np.mean(losses[-log_every:])),
                   **score_packs(pk), "presence": pres,
                   "s_per_step": (time.time() - t_steps) / step}
            if torch.cuda.is_available():
                row["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            row["gpu_shared_with"] = other_gpu_jobs()
            rows.append(row)
            log(f"{arm} step {step}: AP {row['ap2m']:.3f} P {row['prec']:.3f} R {row['rec']:.3f} "
                f"conf {row['n_conf']:.0f} centre {row['centre_p50_m']:.3f} pres {pres:.4f}")
    final = rows[-1]
    crit = criteria(final, presence_step0=pres0, presence_end=final["presence"], finite=finite and
                    final["step"] == steps)
    res = {"arm": arm, "rows": rows, "criteria": crit, "verdict": arm_verdict(arm, crit),
           "n_trainable_tensors": len(params), "n_trainable_params": int(sum(p.numel() for p in params)),
           "wall_s": round(time.time() - t0, 1), "s_per_step": final.get("s_per_step"),
           "peak_mem_gb": final.get("peak_mem_gb"), "optimizer": public_spec(ospec),
           "controls_step0_packs": packs0}
    adapter.teardown()
    return res


def one_frame_index(per_frame) -> int:
    """The frame with the most VIS-1 POSITIVES (ties -> the first): the ladder's frame."""
    return int(np.argmax([int(x[2]) for x in per_frame]))


def keep_share(prev: list | None, cur: list | None):
    """The share of targets whose matched slot index is unchanged between two assignments ({target: slot} per
    frame), or None when either is missing."""
    if prev is None or cur is None:
        return None
    tot = sum(len(b) for b in cur)
    same = sum(int(a.get(k) == v) for a, b in zip(prev, cur) for k, v in b.items())
    return same / tot if tot else None


def one_frame_verdict(rows: list, final_p: float | None, steps: int = ONE_FRAME_STEPS,
                      every: int = ONE_FRAME_EVERY, key: str = "keep") -> dict:
    """A14/A14.1's literals from the logged rows: (i) ``final_p`` (the eval-rule median presence of the matched slots at
    the last step) >= 0.50; (ii) the mean of the ``key`` shares logged in the final 100 steps >= 0.80 (``key`` =
    ``"anchor_keep"`` under HQS -- the matched CELL -- and ``"keep"`` -- the matched QUERY -- otherwise). NaN never
    passes."""
    fin = [r[key] for r in rows if r.get(key) is not None and r["step"] > steps - 100]
    m = float(np.mean(fin)) if fin else float("nan")
    i_ok = bool(final_p is not None and final_p == final_p and final_p >= ONE_FRAME_BARS["p_matched_median"])
    ii_ok = bool(m == m and m >= ONE_FRAME_BARS["assign_keep_final100"])
    return {"i_p_matched_median": final_p, "i_pass": i_ok, "ii_keep_final100": m, "ii_key": key,
            "ii_windows": len(fin), "ii_pass": ii_ok, "PASS": i_ok and ii_ok}


def run_one_frame(adapter, arm: str, *, steps: int = ONE_FRAME_STEPS, every: int = ONE_FRAME_EVERY) -> dict:
    """A14's one-frame test for one arm: the ladder's frame, batch 1, the adapter's (A13) optimiser and clip; the
    last layer's assignment captured inside ``box3d_loss_row`` at every logged TRAINING step; the eval rule every
    ``every`` steps."""
    from tanitad.models import agent_slots as _AS
    from tanitad.models import box3d_head as _B3
    adapter.setup(arm, SEED)
    br_ = getattr(getattr(adapter, "model", None), "_perception", None)
    anchor_is_cell = getattr(br_, "box_heat", None) is not None and arm != "anchors_removed"
    params = adapter.params(arm)
    opt, ospec = adapter.optimizer(arm, params, lr=None)
    do_clip = ospec.get("clip") is not None
    fi = one_frame_index(adapter.per_frame)
    all_items = adapter.items
    adapter.items = [all_items[fi]]
    PB = adapter.tr._perc
    cap = {"on": False, "inside": False, "calls": [], "pred": None}
    o_row, o_ms, o_b3 = PB.box3d_loss_row, _AS.match_slots, _B3.match_slots

    def row(*aa, **kk):
        cap["inside"] = True
        try:
            return o_row(*aa, **kk)
        finally:
            cap["inside"] = False

    def ms(pred, tgt, *, presence_cost="sigmoid"):
        mm = o_ms(pred, tgt, presence_cost=presence_cost)
        if cap["on"] and cap["inside"]:
            cap["calls"].append(mm)
            cap["pred"] = pred
        return mm
    PB.box3d_loss_row, _AS.match_slots, _B3.match_slots = row, ms, ms
    rows, prev, prev_anchor, t0 = [], None, None, time.time()
    try:
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            cap["on"], cap["calls"], cap["pred"] = step % every == 0, [], None
            L = adapter.loss([0])
            if not bool(torch.isfinite(L).all()):
                rows.append({"step": step, "non_finite": True})
                break
            L.backward()
            cap["on"] = False
            if do_clip:
                adapter.clip()
            opt.step()
            if step % every:
                continue
            cur = None
            anchor_of = None
            if cap["calls"]:
                m = cap["calls"][-1]
                cur = [{int(c): int(r) for r, c in zip(R.tolist(), C.tolist())} for R, C in zip(m["rows"], m["cols"])]
                anc = cap["pred"].get("anchors") if cap["pred"] is not None else None
                if anc is not None:            # HQS: which anchor CELL each target's slot sat on (informative)
                    a0 = anc.detach().float().cpu()
                    anchor_of = [{k: (round(float(a0[b, v, 0]), 3), round(float(a0[b, v, 1]), 3))
                                  for k, v in cur[b].items()} for b in range(len(cur))]
            packs, pres = adapter.evaluate()
            pk = packs[0]
            pm = 1.0 / (1.0 + np.exp(-pk["logit"][pk["matched"]].astype(np.float64)))
            sc = score_packs(packs)
            rows.append({"step": step, "loss": float(L.detach()), "keep": keep_share(prev, cur),
                         "anchor_keep": keep_share(prev_anchor, anchor_of),
                         "p_matched_median": float(np.median(pm)) if pm.size else None,
                         "p_matched_max": float(pm.max()) if pm.size else None,
                         "n_conf": sc["n_conf"], "ap2m": sc["ap2m"], "centre_p50_m": sc["centre_p50_m"],
                         "presence": pres})
            prev, prev_anchor = cur, anchor_of
            log(f"one-frame {arm} step {step}: p_matched median {rows[-1]['p_matched_median']} keep "
                f"{rows[-1]['keep']} anchor-keep {rows[-1]['anchor_keep']} conf {sc['n_conf']:.0f} "
                f"AP {sc['ap2m']:.3f} loss {rows[-1]['loss']:.3f}")
    finally:
        PB.box3d_loss_row, _AS.match_slots, _B3.match_slots = o_row, o_ms, o_b3
        adapter.items = all_items
        adapter.teardown()
    final_p = rows[-1].get("p_matched_median") if rows and rows[-1]["step"] == steps else None
    key = "anchor_keep" if anchor_is_cell else "keep"
    return {"arm": arm, "anchor_identity": "cell" if anchor_is_cell else "query",
            "frame_index": fi, "frame": {"sha12": adapter.per_frame[fi][0],
                                                     "t": int(adapter.per_frame[fi][1]),
                                                     "n_pos": int(adapter.per_frame[fi][2])},
            "rows": rows, "verdict": one_frame_verdict(rows, final_p, steps, every, key=key),
            "optimizer": public_spec(ospec), "wall_s": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------------------------------------- #
# the REAL adapter: the launch box path through the trainer's own loss                                     #
# --------------------------------------------------------------------------------------------------------- #
def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: A11: the audit's loader, VENDORED into stack/ (the closure is verified on Thor's stack/taniteval/tools tree)
LOADER_MODULE = "tanitad.eval.refcv6_loader"
AUDIT_LOADER_BLOB = "11808258cd5c90647fae03f5859774b5541e9300"   # 2026-09-26-box-head-audit/code/refcv6_loader.py
VENDOR_END = "== END VENDOR PROVENANCE ==\n"
ARGV_HASH_FORM = 'sha256(json.dumps(argv, separators=(",", ":")))'


def git_blob_sha1(data: bytes) -> str:
    """``git hash-object --no-filters`` of these bytes."""
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def vendored_audit_blob(source: str) -> str:
    """The git blob of the AUDIT file a vendored copy was made from: strip the provenance block (the only addition,
    from the opening quotes to the END line) and re-hash the rest. A copy without the block is refused."""
    if not source.startswith('"""VENDORED COPY') or VENDOR_END not in source:
        raise SystemExit("[gbo] the vendored loader carries no provenance block")
    return git_blob_sha1(('"""' + source[source.index(VENDOR_END) + len(VENDOR_END):]).encode("utf-8"))


def load_vendored_loader():
    """-> (the loader module, its record). The module reads REFCV6_REPO at import: the env is set first, and a copy
    imported earlier against another repo is REFUSED. Its code must re-hash to the audit blob."""
    os.environ["REFCV6_REPO"] = str(STACK.parent)
    L = importlib.import_module(LOADER_MODULE)
    if os.path.normcase(os.path.abspath(str(L.STACK))) != os.path.normcase(os.path.abspath(str(STACK))):
        raise SystemExit(f"[gbo] the loader resolved the repo's stack to {L.STACK}, not {STACK} "
                         "(imported before REFCV6_REPO was set?)")
    raw = Path(L.__file__).read_bytes()
    blob = vendored_audit_blob(raw.decode("utf-8"))
    if blob != AUDIT_LOADER_BLOB:
        raise SystemExit(f"[gbo] the vendored loader is not the audit's: blob {blob} != {AUDIT_LOADER_BLOB}")
    return L, {"module": LOADER_MODULE, "file": str(L.__file__), "file_sha256": hashlib.sha256(raw).hexdigest(),
               "audit_blob": blob}


def load_launch_argv(path) -> tuple:
    """-> (argv, record fields). ``path`` is the CANONICAL run file (``stack/ops/runs.d/<run>.argv.json``, an OBJECT
    whose ``"argv"`` is the list) or a plain JSON list of strings; anything else is refused. The sha256 is over the
    ARGV only, as compact JSON (the launch gate's form, A11), never over a wrapping object."""
    raw = Path(path).read_bytes()
    obj = json.loads(raw.decode("utf-8"))
    shape = "list"
    if isinstance(obj, dict):
        if "argv" not in obj:
            raise SystemExit(f"[gbo] {path}: a JSON object without an 'argv' key")
        obj, shape = obj["argv"], "object['argv']"
    if not isinstance(obj, list) or not all(isinstance(x, str) for x in obj):
        raise SystemExit(f"[gbo] {path}: the argv must be a list of strings")
    argv = list(obj)
    return argv, {"launch_argv_file": str(path), "launch_argv_file_sha256": hashlib.sha256(raw).hexdigest(),
                  "launch_argv_shape": shape, "launch_argv_sha256_form": ARGV_HASH_FORM,
                  "launch_argv_sha256": hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode("utf-8"))
                  .hexdigest()}


def harness_build_argv(launch_argv: list) -> tuple:
    """-> (the argv the harness BUILDS from, record fields): the launch argv minus ``--trunk-compile`` -- DECLARED,
    the harness runs eager (Master Mind 2026-09-27); ``trunk_compile`` in the record is what the harness RAN."""
    return [x for x in launch_argv if x != "--trunk-compile"], {
        "trunk_compile": False, "trunk_compile_in_launch_argv": "--trunk-compile" in launch_argv,
        "trunk_compile_why": ("harness: the launch argv's --trunk-compile (torch.compile of the backbone call only) is "
                              "dropped for G-BOX-OVERFIT -- eager numerics, no compile warm-up; the binding run "
                              "decides whether the launch path needs it (Master Mind 2026-09-27)")}


class TrainerAdapter:
    """The refcv7 LAUNCH box path: the model the launch argv builds (seed 0, fresh -- ImageNet trunk init, no
    checkpoint), the 16 TRAIN windows through the trainer's dataset (the audit loader's recipe with the TRAIN
    override, as ``bha_dump --set train``) with the VIS-1 sidecar joined, and the trainer's OWN ``box3d`` loss term.

    ⛔ The model build replays ``refc_v3_train.train()``'s build lines (as the audit loader does) PLUS the A9 box
    refinement, and is then checked by ``declared_vs_built.check`` against the launch argv: a replay that drifted
    from the launch would be NAMED, not trained."""

    def __init__(self, tr, loader, launch_argv: list, frameset: dict, device: str, pad: int = 400):
        self.tr, self.L, self.device = tr, loader, device
        self.argv = list(launch_argv)
        self.args = tr.build_parser().parse_args(self.argv)
        if not getattr(self.args, "slot_vis1", False) or not getattr(self.args, "vis1_sidecar", None):
            raise SystemExit("[gbo] the launch argv carries no --slot-vis1 / --vis1-sidecar: the prereg scores "
                             "VIS-1 positives and cannot run without the sidecar")
        self.config = {"argv": self.argv, "agent_join_stats": {"train": {"agent_pad": int(pad)}}}
        self.frameset = frameset
        self._hooks = []
        self._patched = []
        self.items, self.per_frame = self._build_items()

    # -- data -------------------------------------------------------------------------------------------- #
    def _build_items(self):
        tr, L = self.tr, self.L
        from tanitad.data import vis1 as V
        from tanitad.data import v2_dataset as v2d
        a = copy.copy(self.args)
        a.eval_cache = a.v2_cache[0] if isinstance(a.v2_cache, (list, tuple)) else a.v2_cache
        a.eval_labels = a.v7_labels
        a.speed_max_sidecar_v6_eval = getattr(a, "speed_max_sidecar_v6", None)
        man = v2d.load_or_build_manifest(a.eval_cache, verbose=False)
        by12 = {hashlib.sha256(str(c).encode()).hexdigest()[:12]: str(c) for c in man["clip_id"]}
        want = [(f["sha12"], int(f["t"])) for f in self.frameset["frames"]]
        miss = [s for s, _ in want if s not in by12]
        if miss:
            raise SystemExit(f"[gbo] {len(miss)} frame-set clips are not in the TRAIN cache: {miss[:4]}")
        keep = {by12[s] for s, _ in want}
        _orig = v2d.build_v2_providers

        def clip_of(ep):
            fp = ep.frames
            return Path(fp._cache.files[fp._clip]).name.split(".")[0]

        def _only(*aa, **kk):
            eps = _orig(*aa, **kk)
            ks = [e for e in eps if clip_of(e) in keep]
            if len(ks) != len(keep):
                raise SystemExit(f"[gbo] provider filter kept {len(ks)} of {len(keep)}")
            return ks
        v2d.build_v2_providers = _only
        try:
            cfg = tr._pin_trainer_cfg(tr.v3.refc_v3_sized_config(a.size, hier=a.arm == "hier"), a)
            # ⭐ A13 / refcv7 A6: under --map-hires on, train() attaches ONLY the 10 cm fine store (no 0.5 m
            # MapGTStore: a /3 file past the old window has no /2 twin). The vendored (refcv6-era) loader would
            # attach the 0.5 m store for any --map-gt-root, so it is handed a copy WITHOUT it, and the fine store is
            # attached below exactly as train() attaches it.
            hires = bool(getattr(tr, "_map_hires_on", lambda _a: False)(a))
            a_loader = copy.copy(a)
            if hires:
                a_loader.map_gt_root = None
            e_ds, e_eps, self.dataset_record = L.build_eval_dataset(None, cfg, a_loader, self.config,
                                                                    with_perception_targets=True)
            if hires:
                _e_clip, _e_ns = tr._clip_table_for_caches([a.eval_cache])
                self.dataset_record["map_hires"] = e_ds.enable_map_hires(
                    tr._sem_fine.FineMapGTStore(Path(a.map_gt_root), max_open=int(getattr(a, "map_lru", 4) or 4),
                                                extent=tr._mhr.declared_extent(a)),
                    _e_clip, _e_ns, min_coverage=getattr(a, "map_min_coverage", None))
        finally:
            v2d.build_v2_providers = _orig
        sc = V.VIS1Sidecar(a.vis1_sidecar)
        self.sidecar_stamp = sc.stamp()
        self.vis1_stats = e_ds.enable_vis1(sc, split="train")
        pos_of = {}
        for i, (e_i, t) in enumerate(e_ds.index):
            eid = int(e_eps[e_i].episode_id)
            pos_of[(e_ds._vis1_sha12[eid], int(t))] = i
        items, per_frame = [], []
        for s12, t in want:
            if (s12, t) not in pos_of:
                raise SystemExit(f"[gbo] frame ({s12}, t={t}) is not a window of the TRAIN dataset")
            it = e_ds[pos_of[(s12, t)]]
            items.append(it)
            b = torch.utils.data.default_collate([it])
            tgt = {"box": b["agent_box"], "valid": b["agent_valid"]}
            sp = V.vis1_split(tgt, n_full=b["agent_vis_full"], n_vis=b["agent_vis_px"],
                              vis_known=b["agent_vis_known"])
            per_frame.append((s12, t, int(sp["pos"]["valid"].sum()), int(sp["ignore"].sum())))
        return items, per_frame

    def _batch(self, idx):
        return torch.utils.data.default_collate([self.items[i] for i in idx])

    # -- model -------------------------------------------------------------------------------------------- #
    def _build_model(self):
        tr, a, dev = self.tr, self.args, self.device
        from tanitad.refs import refc
        from tanitad.refs import refc_v3 as v3
        tr._check_nav_from_v7_args(a)
        tr._check_max_speed_args(a)
        tr._check_goal_point_args(a)
        tr.check_effective_weights(a)
        art = tr._read_anchor_artifact(a)
        cfg = tr._pin_trainer_cfg(v3.refc_v3_smoke_config(a.arm == "hier") if a.smoke else
                                  v3.refc_v3_sized_config(a.size, hier=a.arm == "hier"), a)
        tr._check_anchor_artifact_against_cfg(art, cfg, a)
        if a.graft_lan or a.goal_str:
            cfg.core.lan = refc.LanConfig(k=len(a.lan_arclengths))
        m = v3.RefCV3Model(cfg).to(dev)
        m._w_agent = float(getattr(a, "w_agent", tr.AGENT_WEIGHT_DEFAULT))
        m._w_bev_aux = float(getattr(a, "w_bev_aux", 0.0))
        m._bev_shuffle = bool(getattr(a, "bev_aux_shuffle", False))
        m._cls_class_weight, m._cls_class_weight_stamp = None, None
        if str(getattr(a, "agent_cls_weight", "off")) != "off":
            _n, _line = tr.CLS_WEIGHT_CHOICES[str(a.agent_cls_weight)]
            _line, _pop, _src = tr._cls_weight_expectation(a, _line)
            _cw, _cws = tr._agent_slots.load_cls_class_weight(_n, expect_corpus_line=_line, target_population=_pop)
            m._cls_class_weight, m._cls_class_weight_stamp = _cw.to(dev), _cws
        m._w_map = float(getattr(a, "w_map", 0.0) or 0.0)
        m._w_box3d = float(getattr(a, "w_box3d", 0.0) or 0.0)
        m._map_lift_valid_mask = bool(getattr(a, "map_lift_valid_mask", True))
        m._box3d_visible_filter = bool(getattr(a, "box3d_visible_filter", True))
        m._vis1 = bool(getattr(a, "slot_vis1", False))
        m._perception, m._lift_bank = None, None
        if m._w_box3d <= 0.0:
            raise SystemExit("[gbo] the launch argv builds no box head (--w-box3d 0)")
        _perc = tr._perc
        # ---- refcv7 NEW-2 + A6/A7: the 10 cm branch, replayed from train() (BUILT FIRST: under A6 the perception
        # branch's pool reads its 0.25 m encoder); off => nothing, exactly as train() ------------------------- #
        m._w_map_hires, m._map_hires, m._lift_bank_hires = 0.0, None, None
        m._map_hires_class_weight = m._map_hires_class_weight_stamp = None
        if bool(getattr(tr, "_map_hires_on", lambda _a: False)(a)):
            _mhr = tr._mhr
            _hext = _mhr.declared_extent(a)
            m.core.encoder.enable_s8_tap()
            _hcw, _hcws = _mhr.load_class_weights(a.map_hires_class_weights, extent=_hext)
            _hcfg = _mhr.MapHiresConfig(
                w_map_hires=float(a.w_map_hires), x_max_m=float(_hext.x_max_m), y_half_m=float(_hext.y_half_m),
                grad_ckpt=bool(_mhr.declared_grad_ckpt(a)), class_weights_sha256=str(_hcws["sha256"]),
                decision_rule=str(getattr(a, "map_hires_decision_rule", _mhr.DECISION_RULES[0])))
            m._map_hires = _mhr.build_map_hires_branch(m, _hcfg).to(dev)
            m._w_map_hires = float(a.w_map_hires)
            m._map_hires_class_weight, m._map_hires_class_weight_stamp = _hcw.to(dev), _hcws
            _hpe, _htable = tr._read_rig_extrinsics(str(getattr(a, "agent_rig_extrinsics", "")))
            if _htable is None:
                raise SystemExit("[gbo] --map-hires on needs a PER-CLIP extrinsics table (as train() refuses)")
            m._lift_bank_hires = _mhr.HiresLiftGeometryBank(
                _htable, frame=_perc.frame_for_model(m), cfg=_hcfg,
                equalize_bottom_rows=int(getattr(a, "equalize_bottom_rows", 0) or 0))
        _bev_src = str(getattr(a, "bev_source", None) or _perc.BEV_SOURCES[0]) \
            if hasattr(_perc, "BEV_SOURCES") else None
        if _bev_src is None:
            _pcfg = _perc.PerceptionBranchConfig(w_map=m._w_map, w_box3d=m._w_box3d)
        else:
            _pcfg = _perc.PerceptionBranchConfig(
                w_map=m._w_map, w_box3d=m._w_box3d, bev_source=_bev_src,
                planner_crop_m=tuple(float(v) for v in (getattr(a, "bev_planner_crop_m", None)
                                                        or tr._mhr.PLANNER_CROP_DEFAULT)))
        _kw = dict(tr._slot_refine_kwargs(a))
        _fields = {f.name for f in dataclasses.fields(_pcfg)}
        if "dn_groups" in _fields:                          # +R6 (A10.1), when that patch is in
            _kw["dn_groups"] = int(getattr(a, "slot_dn_groups", 0) or 0)
        if "query_select" in _fields:                       # A14 (HQS), when that patch is in
            _kw["query_select"] = str(getattr(a, "slot_query_select", "learned") or "learned")
        _pcfg = dataclasses.replace(_pcfg, **_kw)
        m._dn_gen = torch.Generator().manual_seed(int(getattr(a, "seed", 0)) + 7919)
        m._perception = _perc.build_perception_branch(m, _pcfg).to(dev)
        if m._w_map > 0.0:
            _pe, _ptable = tr._read_rig_extrinsics(str(getattr(a, "agent_rig_extrinsics", "")))
            m._lift_bank = _perc.LiftGeometryBank(_ptable, frame=_perc.frame_for_model(m), stride=int(_pcfg.stride),
                                                  equalize_bottom_rows=int(getattr(a, "equalize_bottom_rows", 0) or 0))
        m._w_tac_goal = float(getattr(a, "w_tac_goal", 0.0) or 0.0)
        m._tac_goal_pos_weight = m._tac_goal_class_mask = None
        m._w_tac_v6 = float(getattr(a, "w_tac_v6", 0.0) or 0.0)
        m._w_r7_wta = float(getattr(a, "w_r7_wta", 0.0) or 0.0)
        m._w_r7_scorer = float(getattr(a, "w_r7_scorer", 0.0) or 0.0)
        m._r7_n_perturb = int(getattr(a, "r7_n_perturb", 0) or 0)
        m._r7_nav_tau_rad = float(getattr(a, "r7_nav_tau_rad", 0.0) or 0.0)
        m._r7_calls = 0
        m._w_u0 = float(getattr(a, "w_u0", tr.U0_WEIGHT_DEFAULT))
        m._w_goal_point = float(getattr(a, "goal_point_w", tr.GOAL_POINT_WEIGHT_DEFAULT))
        m._rig_camera, _cam = tr._build_rig_camera(cfg, a)
        if a.anchors:
            m.core.decoder.load_anchors(art.anchors.to(dev), None if art.controls is None else art.controls.to(dev))
        tr._apply_withheld_bank(m, a, [], dev)
        # ⭐ the replay is CHECKED against the launch argv (the harness trains the declared model or refuses)
        from tanitad.train import declared_vs_built as dvb
        bad = list(dvb.check(m, a))
        self.dvb_mismatches = [str(x) for x in bad]
        if bad:
            raise SystemExit("[gbo] the replayed build is NOT the declared launch model:\n  - "
                             + "\n  - ".join(self.dvb_mismatches))
        return m

    # -- the adapter protocol --------------------------------------------------------------------------- #
    def setup(self, arm: str, seed: int) -> None:
        torch.manual_seed(int(seed))
        self.model = self._build_model()
        self.model.train()
        self.arm = arm
        br = self.model._perception
        if arm == "memory_zeros":
            self._hooks.append(br.box_mem.register_forward_hook(lambda _m, _i, o: torch.zeros_like(o)))
            if getattr(br, "box_heat", None) is not None:
                # A14: under HQS the image also reaches the box head through the heatmap's anchors -- the must-fail
                # arm blinds that input too, or it would not mean "the box head cannot see the image"
                self._hooks.append(br.box_heat.register_forward_pre_hook(
                    lambda _m, args: (torch.zeros_like(args[0]),) + tuple(args[1:])))
        if arm == "anchors_removed":
            # A14's red arm: the heatmap's anchors replaced by LEARNED reference points (one trainable (x, y) per
            # query, uniform over the planner grid at seed 0), everything else the same
            from tanitad.models import slot_query_select as _sqs
            if getattr(br, "box_heat", None) is None:
                raise SystemExit("[gbo] anchors_removed is HQS's red arm: this build has no heatmap")
            grid = br.cfg.planner_grid
            k = int(br.box_dec.n_queries)
            g = torch.Generator().manual_seed(SEED)
            ref = torch.stack([torch.rand(k, generator=g) * float(grid.x_fwd_m),
                               (torch.rand(k, generator=g) * 2 - 1) * float(grid.y_half_m)], dim=-1)
            br.box_hqs_refpts = torch.nn.Parameter(ref.to(next(br.parameters()).device))
            self._patched.append((_sqs, "choose_anchors", _sqs.choose_anchors))
            _sqs.choose_anchors = lambda branch, heat: (
                branch.box_hqs_refpts[None].expand(int(heat.shape[0]), -1, -1),
                torch.zeros(int(heat.shape[0]), int(branch.box_dec.n_queries), device=heat.device))
        if arm == "presence_w0":
            from tanitad.models import slot_presence as SP
            from tanitad.models import agent_slots as AS
            self._patched.append((SP, "FOCAL_PRESENCE_W", SP.FOCAL_PRESENCE_W))
            SP.FOCAL_PRESENCE_W = 0.0
            self._patched.append((AS, "SLOT_LOSS_W", AS.SLOT_LOSS_W))
            AS.SLOT_LOSS_W = {**AS.SLOT_LOSS_W, "presence": 0.0}
        if arm == "targets_filter_only":
            self._vis1_cfg = br.cfg
            br.cfg = dataclasses.replace(br.cfg, vis1=False)
            self.model._vis1 = False

    def teardown(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        for mod, name, val in reversed(self._patched):
            setattr(mod, name, val)
        self._patched = []
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _loss_row(self, batch):
        return self.tr.compute_losses_v3(self.model, batch, self.device, mode=getattr(self.args, "mode", "diffusion"),
                                         ablate_frames=False)

    def params(self, arm: str) -> list:
        """Every tensor the box loss reaches (a probe backward), or box decoder + memory for ``frozen_trunk``."""
        br = self.model._perception
        if arm == "frozen_trunk":
            ps = list(br.box_dec.parameters()) + list(br.box_mem.parameters())
            keep = {id(p) for p in ps}
            for p in self.model.parameters():
                if id(p) not in keep:
                    p.requires_grad_(False)
            return [p for p in ps if p.requires_grad]
        self.model.zero_grad(set_to_none=True)
        L = self._loss_row(self._batch(list(range(BATCH))))["box3d"]
        L.backward()
        ps = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        self.model.zero_grad(set_to_none=True)
        return ps

    def optimizer(self, arm: str, params, lr=None):
        """A13: the LAUNCH optimiser AS BUILT over the whole model (tensors the box loss does not reach get no
        gradient and AdamW leaves them untouched). ``frozen_trunk`` freezes AFTER the groups are built, so the
        launch's partition is the one used. An lr override is REFUSED."""
        if lr is not None:
            raise SystemExit("[gbo] A13: the harness trains with the LAUNCH optimiser; an lr override is refused")
        frozen = [q for q in self.model.parameters() if not q.requires_grad]
        for q in frozen:
            q.requires_grad_(True)
        try:
            opt, spec = launch_optimizer(self.tr, self.model, self.args)
        finally:
            for q in frozen:
                q.requires_grad_(False)
        self.optimizer_spec = spec
        return opt, spec

    def clip(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_NORM)

    def loss(self, idx):
        return self._loss_row(self._batch(idx))["box3d"]

    def evaluate(self):
        """Eval mode, no grad, all 16 frames in their fixed order: the P0 packs and the mean presence term."""
        br = self.model._perception
        restore = None
        if self.arm == "targets_filter_only":          # the SCORING is VIS-1 for every arm
            restore = (br.cfg, self.model._vis1)
            br.cfg, self.model._vis1 = self._vis1_cfg, True
        self.model.eval()
        packs, pres = [], []
        try:
            with torch.no_grad():
                for k in range(0, len(self.items), BATCH):
                    row = self._loss_row(self._batch(list(range(k, min(k + BATCH, len(self.items))))))
                    packs += list(row.get("_det_pack_box3d") or [])
                    pres.append(float(row["box3d_presence"]))
        finally:
            self.model.train()
            if restore is not None:
                br.cfg, self.model._vis1 = restore
        if len(packs) != len(self.items):
            raise SystemExit(f"[gbo] the eval rule produced {len(packs)} packs for {len(self.items)} frames")
        return packs, float(np.mean(pres))


# --------------------------------------------------------------------------------------------------------- #
# main                                                                                                     #
# --------------------------------------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--launch-argv", required=True,
                    help="JSON: the canonical run file (an object with 'argv') or a plain list of strings")
    ap.add_argument("--audit-dir", required=True,
                    help="…/2026-09-26-box-head-audit (the prereg + the frame set; its loader is VENDORED in stack/)")
    ap.add_argument("--out", required=True, help="the harness record (JSON)")
    ap.add_argument("--arms", default="main,memory_zeros,presence_w0")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--binding", action="store_true", help="a PASS record the launch gate may consume")
    ap.add_argument("--candidate", required=True, help="what was run (tree / patch description)")
    ap.add_argument("--commit", default=None, help="--binding only: the commit the run's tree was archived "
                    "from (A11: the launch gate binds the record through closure_run.py's code closure)")
    ap.add_argument("--check-build", action="store_true",
                    help="build the data (C3) and the model ONCE (G-DVB self-check, parameter counts), train nothing")
    ap.add_argument("--one-frame", action="store_true",
                    help="A14's ONE-FRAME test (arms main + anchors_removed unless --arms says otherwise)")
    a = ap.parse_args(argv)
    if a.binding and not a.commit:
        raise SystemExit("[gbo] --binding needs --commit (the launch commit)")
    if a.steps != STEPS and a.binding:
        raise SystemExit("[gbo] a binding run uses the prereg's N = 2,000 steps")
    t_all = time.time()
    audit = Path(a.audit_dir)
    prereg = (audit / "raw" / "PREREG_G_BOX_OVERFIT.md").read_bytes()
    fs_bytes = (audit / "raw" / "visibility" / "gbo_frameset.json").read_bytes()
    rec = {"tool": "g_box_overfit.py", "binding": bool(a.binding),
           "commit": (a.commit if a.binding else f"NON-BINDING candidate: {a.candidate}"),
           "candidate": a.candidate, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "prereg_md5": md5_bytes(prereg), "prereg_md5_ok": md5_bytes(prereg) == PREREG_MD5,
           "frameset_md5": md5_bytes(fs_bytes), "steps": a.steps, "device": a.device,
           "literals": {"optimizer": OPTIMIZER_RULE, "clip": CLIP_NORM, "batch": BATCH, "seed": SEED, "bars": BARS,
                        "n_pos": N_POS, "n_ign": N_IGN}}
    if not rec["prereg_md5_ok"]:
        log(f"⚠ prereg md5 {rec['prereg_md5']} != {PREREG_MD5} (line endings? recorded, not refused)")
    frameset = json.loads(fs_bytes.decode("utf-8"))
    launch_argv, argv_rec = load_launch_argv(a.launch_argv)
    rec.update(argv_rec)
    build_argv, tc_rec = harness_build_argv(launch_argv)
    rec.update(tc_rec)
    tr = _load_by_path("refc_v3_train_for_gbo", SCRIPTS / "refc_v3_train.py")
    import tanitad
    if not os.path.normcase(os.path.abspath(tanitad.__file__)).startswith(os.path.normcase(str(STACK))):
        raise SystemExit(f"[gbo] tanitad imported from {tanitad.__file__}, not this tree")
    L, rec["loader"] = load_vendored_loader()
    ad = TrainerAdapter(tr, L, build_argv, frameset, a.device)
    rec["sidecar"] = ad.sidecar_stamp
    rec["C3"] = control_c3(ad.per_frame, frameset, rec["frameset_md5"])
    log(f"C3 {'PASS' if rec['C3']['pass'] else 'FAIL'}: totals {rec['C3']['totals']} vs {rec['C3']['want_totals']}")
    if not rec["C3"]["pass"]:
        Path(a.out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        raise SystemExit("[gbo] C3 failed: the harness's VIS-1 recount is not the frame set's; nothing trains")
    if a.check_build:
        ad.setup("main", SEED)
        br = ad.model._perception
        _opt, _ospec = ad.optimizer("main", None)
        rec["check_build_optimizer"] = public_spec(_ospec)
        rec["check_build_optimizer_vs_trainer_clip"] = {
            "harness": CLIP_NORM, "trainer": trainer_clip_literal((SCRIPTS / "refc_v3_train.py").read_text(
                encoding="utf-8"))}
        rec["check_build"] = {"dvb_mismatches": ad.dvb_mismatches,
                              "box_decoder_params": int(sum(p.numel() for p in br.box_dec.parameters())),
                              "box_memory_params": int(sum(p.numel() for p in br.box_mem.parameters())),
                              "box_n_queries": int(br.box_dec.n_queries),
                              "box_deep_supervision": bool(br.box_dec.deep_supervision),
                              "box_presence_prior": float(br.box_dec.presence_prior),
                              "box_presence_loss": str(br.cfg.presence_loss), "box_vis1": bool(br.cfg.vis1),
                              "agent_n_queries": int(ad.model.core.agent_head.n_queries)}
        ad.teardown()
        Path(a.out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        log(f"check-build: {rec['check_build']}")
        return 0
    if a.one_frame:
        arms = [x for x in (a.arms if a.arms != "main,memory_zeros,presence_w0" else "main").split(",") if x]
        rec["one_frame"] = {"literals": {"steps": ONE_FRAME_STEPS, "every": ONE_FRAME_EVERY, "bars": ONE_FRAME_BARS},
                            "arms": {}}
        for arm in arms:
            if arm not in ONE_FRAME_ARMS:
                raise SystemExit(f"[gbo] unknown one-frame arm {arm!r}")
            rec["one_frame"]["arms"][arm] = run_one_frame(ad, arm)
            Path(a.out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
            log(f"one-frame {arm}: {rec['one_frame']['arms'][arm]['verdict']}")
        v = {k: x["verdict"] for k, x in rec["one_frame"]["arms"].items()}
        main_ok = bool(v.get("main", {}).get("PASS"))
        red = v.get("anchors_removed")
        # A14.1: the red arm is its own invocation (`unanchored` = the MAIN decoder); this record's RESULT is the
        # verdict of the configuration it ran (``anchors_removed``, when run, stays informative)
        rec["one_frame"]["RESULT"] = ("INCOMPLETE" if "main" not in v else "PASS" if main_ok else "FAIL")
        rec["one_frame"]["query_select"] = str(getattr(ad.args, "slot_query_select", "learned") or "learned")
        if red is not None:
            rec["one_frame"]["anchors_removed_informative"] = red
        rec["wall_s"] = round(time.time() - t_all, 1)
        Path(a.out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        log(f"ONE-FRAME RESULT {rec['one_frame']['RESULT']}")
        return 0
    rec["arms"] = {}
    for arm in [x for x in a.arms.split(",") if x]:
        if arm not in ARMS:
            raise SystemExit(f"[gbo] unknown arm {arm!r}")
        res = run_arm(ad, arm, steps=a.steps)
        packs0 = res.pop("controls_step0_packs")
        if arm == "main":
            rec["C1_C2"] = control_c1_c2(packs0)
            rec["C4_step0"] = res["rows"][0]
        rec["arms"][arm] = res
        rec["dvb_mismatches"] = getattr(ad, "dvb_mismatches", [])
        Path(a.out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        log(f"arm {arm}: {res['verdict']} ({res['wall_s']} s, {res['s_per_step']} s/step)")
    main_ok = rec["arms"].get("main", {}).get("verdict") == "PASS"
    void = any(str(v.get("verdict", "")).startswith("VOID") for v in rec["arms"].values())
    ctl_ok = rec["C3"]["pass"] and all(v["pass"] for v in (rec.get("C1_C2") or {}).values())
    rec["RESULT"] = ("VOID" if void else ("PASS" if (main_ok and ctl_ok and all(
        m in rec["arms"] for m in MUST_FAIL)) else "FAIL" if "main" in rec["arms"] else "INCOMPLETE"))
    rec["wall_s"] = round(time.time() - t_all, 1)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    log(f"RESULT {rec['RESULT']} (binding={rec['binding']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
