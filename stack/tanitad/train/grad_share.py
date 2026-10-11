"""In-run GRADIENT-SHARE instrument (refcv8 WP-B DESIGN sec. 3.6 X4) -- WP-D's P-GRAD statistic, inside the trainer.

P-GRAD (``TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/RESULT.md``
sec. 7.2, MEASURED on refcv7-50,400, frozen batch, all weights 1.0) found the trunk update paid for by agent 50.1 %,
box3d 35.7 %, the planner auxiliaries 14.2 %, the trajectory 0.02 %, the 10 cm map 0.01 %, the tactical decoder
-0.05 %. A loss-budget arm is sized and judged on THAT statistic -- never on a loss-value share -- so the trainer must
measure it in-run, on the run's own batches and weights.

The statistic (per parameter group, on ONE batch, at the weights the terms enter ``loss`` with):

* ``proj_t = <g_t, g_tot> / ||g_tot||^2`` -- the share of the update term ``t`` "pays for" (sums to 1 over the terms
  plus ``rest`` = total minus the named terms);
* ``norm_t = ||g_t||`` and ``norm_total``;
* the LINEARITY CONTROL ``lin_rel_err = ||sum_t g_t - g_tot|| / ||g_tot||`` (P-GRAD's bar: <= 1e-4) -- a reading whose
  control misses is logged with the control, never silently.

``torch.autograd.grad`` throughout: ``.grad`` is never touched, so the optimiser step that follows is the step the
trainer would have taken without the instrument (pinned: ``tests/test_refcv8_grad_share.py``). No RNG is consumed.

COST (MEASURED 2026-10-11, the R2 ladder's V0 on Thor): one reading is ONE retained ``autograd.grad`` pass per term
plus one for the total, and under ``--trunk-chunk-ckpt`` / ``--map-hires-grad-ckpt`` EVERY pass re-runs the
checkpointed forward; inside the fp32 replay (TF32 off, eager) the first reading at step 250 ran > 20 min and pushed
the arm past its 2400 s cap with no checkpoint. Two levers, both recorded in the row so a reading carries its cost:
``keep=`` (``--grad-share-terms``) names the terms a reading must SEPARATE -- every other term folds into ``rest``,
so the kept terms' ``proj`` / ``norm`` and every ``norm_total`` / ``lin_rel_err`` are UNCHANGED and the pass count
drops to ``len(keep) + 2``; and the timing keys ``{prefix}n_passes`` / ``{prefix}measure_s`` / ``{prefix}pass_s_max``
(+ ``{prefix}fp32_fwd_s`` / ``{prefix}fp32_replay_s`` on a replay). ``TANITAD_GS_TIMING=1`` also prints every pass to
stderr, so a stall is localised while it happens instead of after a timeout.
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Callable, Iterable, Mapping, Optional

import torch
from torch import Tensor

__all__ = ["TERMS", "GROUP_PREFIXES", "term_tensors", "param_groups", "measure", "bf16_levers", "fp32_replay",
           "parse_keep", "parse_at", "due"]

#: the named terms, in the order P-GRAD reports them; ``rest`` is derived
TERMS = ("traj", "agent", "box3d", "map_hires", "tac_v6", "r8")
#: parameter groups (name prefix on ``model.named_parameters()``); P-GRAD's ``trunk`` = the encoder stem..temporal fuse
GROUP_PREFIXES = {"trunk": ("core.encoder.",),
                  "bev025": ("_map_hires.lift.", "_map_hires.encoder."),
                  "bev_pool": ("_perception.bev_pool.",)}


def parse_keep(spec) -> Optional[tuple]:
    """``--grad-share-terms`` -> the kept term tuple (TERMS order), or ``None`` = every term. Accepts a comma string
    or an iterable; an empty spec / ``"all"`` is ``None``. Refuses an unknown term (a typo would otherwise fold the
    term it meant into ``rest`` and the reading would be silently about something else)."""
    if spec is None:
        return None
    items = [x.strip() for x in (spec.split(",") if isinstance(spec, str) else spec)]
    items = [x for x in items if x]
    if not items or items == ["all"]:
        return None
    bad = [x for x in items if x not in TERMS]
    if bad:
        raise ValueError(f"grad-share terms {bad} are not in TERMS {TERMS}")
    return tuple(t for t in TERMS if t in items)


def parse_at(spec, every: int, steps: int = 0) -> Optional[frozenset]:
    """``--grad-share-at`` -> the frozenset of ROW steps (``step + 1``, the step a metrics row is stamped with) on which
    a reading is taken, or ``None`` = every ``every``-th row (the tip's schedule). Refuses a spec with the instrument
    off, a non-integer / non-positive step, a step that is not a multiple of ``every`` (it would never be due -- the
    reading the consumer waits for would silently never exist), and a step past ``steps`` when ``steps > 0``.
    COST is why it exists: a reading inside the fp32 replay is ~16 min on Thor (MEASURED 2026-10-11), so a ladder arm
    reads only where its consumer's window looks."""
    if spec is None:
        return None
    items = [x.strip() for x in (spec.split(",") if isinstance(spec, str) else [str(v) for v in spec])]
    items = [x for x in items if x]
    if not items:
        return None
    if int(every) <= 0:
        raise ValueError(f"names steps {items} but --grad-share-every is {every}: the instrument is off")
    out = set()
    for x in items:
        try:
            v = int(x)
        except ValueError:
            raise ValueError(f"step {x!r} is not an integer") from None
        if v <= 0:
            raise ValueError(f"step {v} is not positive")
        if v % int(every) != 0:
            raise ValueError(f"step {v} is not a multiple of --grad-share-every {every}: it would never be due")
        if int(steps) > 0 and v > int(steps):
            raise ValueError(f"step {v} is past --steps {steps}: it would never be due")
        out.add(v)
    return frozenset(out)


def due(row_step: int, every: int, at: Optional[frozenset] = None) -> bool:
    """Is a reading due on the metrics row stamped ``row_step`` (= ``step + 1``)? ``every <= 0`` is the instrument off;
    ``at`` (``parse_at``) restricts the tip's every-N schedule to the named rows."""
    return int(every) > 0 and row_step % int(every) == 0 and (at is None or row_step in at)


def term_tensors(losses: Mapping[str, Tensor], weights: Mapping[str, float], keep: Optional[Iterable[str]] = None) -> dict:
    """``{term: weighted attached tensor}`` for every term with a non-zero weight that is PRESENT and attached in this
    batch's loss dict, plus ``rest = loss - sum(named)``. A zero weight or an absent / detached key is not a term
    (its gradient is the zero vector, and a share of it is not a measurement). ``keep`` (None = all): a term outside
    it is not separated -- it stays inside ``rest`` -- which leaves every kept term's tensor, and therefore its
    ``proj`` / ``norm``, exactly what the full reading gives."""
    keep = None if keep is None else set(parse_keep(keep if isinstance(keep, str) else list(keep)) or TERMS)
    out: dict = {}
    for t in TERMS:
        if keep is not None and t not in keep:
            continue
        w = float(weights.get(t, 0.0) or 0.0)
        v = losses.get(t)
        if w == 0.0 or v is None or not torch.is_tensor(v) or not v.requires_grad:
            continue
        out[t] = w * v
    total = losses["loss"]
    named = None
    for v in out.values():
        named = v if named is None else named + v
    out["rest"] = total if named is None else total - named
    return out


def param_groups(model, prefixes: Mapping[str, tuple] = GROUP_PREFIXES) -> dict:
    """``{group: [params]}`` over the parameters that REQUIRE grad, non-empty groups only."""
    g: dict = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for name, pre in prefixes.items():
            if n.startswith(pre):
                g.setdefault(name, []).append(p)
                break
    return g


def _grads(t: Tensor, params: list) -> list:
    gr = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if x is None else x for x, p in zip(gr, params)]


def _sync() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _say(msg: str) -> None:
    if os.environ.get("TANITAD_GS_TIMING", "") == "1":
        print(f"[grad_share {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def measure(terms: Mapping[str, Tensor], total: Tensor, groups: Mapping[str, list], *,
            prefix: str = "gs_") -> dict:
    """One reading -> flat ``{key: float}`` for the metrics row (exact, never rounded: a 1e-4 share rounded to 5 dp is
    the "no signal" misreading). Keys: ``{prefix}{group}_proj_{term}``, ``{prefix}{group}_norm_{term}``,
    ``{prefix}{group}_norm_total``, ``{prefix}{group}_lin_rel_err``, ``{prefix}n_terms``, and its cost
    ``{prefix}n_passes`` / ``{prefix}measure_s`` / ``{prefix}pass_s_max`` (wall seconds, CUDA-synchronised)."""
    if not total.requires_grad:
        return {f"{prefix}skipped_no_graph": 1.0}
    flat, owner = [], []
    for gname, ps in groups.items():
        for p in ps:
            flat.append(p)
            owner.append(gname)
    if not flat:
        return {f"{prefix}skipped_no_params": 1.0}
    _sync()
    t_m0 = time.perf_counter()
    _say(f"pass total: start ({len(flat)} params, terms {list(terms)})")
    g_tot = _grads(total, flat)
    _sync()
    pass_s = [time.perf_counter() - t_m0]
    _say(f"pass total: {pass_s[-1]:.2f}s")
    tot2 = {g: 0.0 for g in groups}
    for gname, x in zip(owner, g_tot):
        tot2[gname] += float(torch.sum(x.double() * x.double()))
    acc = [torch.zeros_like(x, dtype=torch.float64) for x in g_tot]
    row: dict = {f"{prefix}n_terms": float(len(terms))}
    for t, v in terms.items():
        if not v.requires_grad:
            continue
        t_p0 = time.perf_counter()
        _say(f"pass {t}: start")
        gt = _grads(v, flat)
        _sync()
        pass_s.append(time.perf_counter() - t_p0)
        _say(f"pass {t}: {pass_s[-1]:.2f}s")
        dot = {g: 0.0 for g in groups}
        nrm = {g: 0.0 for g in groups}
        for i, (gname, x) in enumerate(zip(owner, gt)):
            xd = x.double()
            dot[gname] += float(torch.sum(xd * g_tot[i].double()))
            nrm[gname] += float(torch.sum(xd * xd))
            acc[i] += xd
        del gt
        for g in groups:
            row[f"{prefix}{g}_proj_{t}"] = (dot[g] / tot2[g]) if tot2[g] > 0 else float("nan")
            row[f"{prefix}{g}_norm_{t}"] = math.sqrt(nrm[g])
    err2 = {g: 0.0 for g in groups}
    for gname, a, x in zip(owner, acc, g_tot):
        d = a - x.double()
        err2[gname] += float(torch.sum(d * d))
    for g in groups:
        row[f"{prefix}{g}_norm_total"] = math.sqrt(tot2[g])
        row[f"{prefix}{g}_lin_rel_err"] = (math.sqrt(err2[g]) / math.sqrt(tot2[g])) if tot2[g] > 0 else 0.0
    _sync()
    row[f"{prefix}n_passes"] = float(len(pass_s))
    row[f"{prefix}measure_s"] = time.perf_counter() - t_m0
    row[f"{prefix}pass_s_max"] = max(pass_s)
    _say(f"measure: {row[f'{prefix}measure_s']:.2f}s over {len(pass_s)} passes")
    return row


def weights_of(model, traj_weight: float) -> dict:
    """The weights each term enters ``loss`` with, read off the model the trainer set them on. ``r8`` is already the
    WEIGHTED refcv8 sum (``refcv8_train.r8_losses``), so it enters at 1.0."""
    return {"traj": float(traj_weight), "agent": float(getattr(model, "_w_agent", 0.0) or 0.0),
            "box3d": float(getattr(model, "_w_box3d", 0.0) or 0.0),
            "map_hires": float(getattr(model, "_w_map_hires", 0.0) or 0.0),
            "tac_v6": float(getattr(model, "_w_tac_v6", 0.0) or 0.0),
            "r8": 1.0 if bool(getattr(model, "r8_enabled", False)) else 0.0}


# ---------------------------------------------------------------------------------------------------------------- #
# the fp32 REPLAY (MEASURED 2026-10-05, the Thor L3 smoke: under `--trunk-bf16` the trunk group's linearity control  #
# read 5.9e-3 against the 1e-4 bar -- a bf16 backward makes the gradient of the sum differ from the sum of the        #
# gradients by ~ bf16 epsilon; the fp32 groups read 9.3e-5 / 6.0e-5). The statistic is read on a REPLAY of the SAME  #
# step's forward with every bf16 lever OFF; everything the replay could disturb is restored, so the training step  #
# that follows is the step the trainer would have taken without the instrument.                                    #
# ---------------------------------------------------------------------------------------------------------------- #
def bf16_levers(model) -> list:
    """Every module whose ``memory_levers['bf16']`` is ON (the timm trunk's autocast lever)."""
    out = []
    for m in model.modules():
        lv = getattr(m, "memory_levers", None)
        if isinstance(lv, dict) and bool(lv.get("bf16", False)):
            out.append(m)
    return out


def _dedicated_generators(model) -> list:
    """Every refcv8 dedicated generator reachable from the model (``R8Generator`` duck type: ``_g`` dict of
    torch.Generator) -- their states are snapshotted and restored around the replay."""
    seen, out = set(), []
    for obj in [model] + list(model.modules()):
        for v in list(vars(obj).values()):
            g = getattr(v, "_g", None)
            if isinstance(g, dict) and all(isinstance(x, torch.Generator) for x in g.values()) and id(v) not in seen:
                seen.add(id(v))
                out.append(v)
    return out


def _eager():
    """``torch.compiler.set_stance("force_eager")`` as a context (every torch.compile'd callable runs EAGERLY inside;
    the stance is restored on exit); a no-op context on a torch without stances."""
    st = getattr(getattr(torch, "compiler", None), "set_stance", None)
    if st is None:
        import contextlib
        return contextlib.nullcontext()
    return st("force_eager")


def fp32_replay(model, run_losses: Callable[[], dict], groups: Mapping[str, list], weights: Mapping[str, float], *,
                prefix: str = "gs_", keep: Optional[Iterable[str]] = None) -> dict:
    """``run_losses()`` re-runs THIS step's forward + losses; it is called with every bf16 lever OFF, inside a forked
    global RNG, with the dedicated generators' states and every BUFFER (BN running stats, ...) snapshotted and restored
    afterwards. -> ``measure(...)`` of that fp32 graph, plus ``{prefix}fp32_replay: 1.0`` and the number of levers
    switched. ``.grad`` is never touched (``torch.autograd.grad`` only)."""
    _sync()
    t_r0 = time.perf_counter()
    levers = bf16_levers(model)
    gens = _dedicated_generators(model)
    gstate = [{k: g.get_state() for k, g in v._g.items()} for v in gens]
    bufs = {n: b.detach().clone() for n, b in model.named_buffers()}
    devs = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    # ⛔ TF32 OFF inside the replay too (MEASURED 2026-10-05, the L4 G-SMOKE on Thor: with bf16 off but cuDNN's TF32
    # default ON the trunk linearity read 4.2e-4; P-GRAD's 1e-4 bar was measured with BOTH TF32 switches off,
    # pgrad.py:68-69). Restored in `finally`, so the training step that follows runs exactly as configured.
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    try:
        for m in levers:
            m.memory_levers["bf16"] = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        # ⛔ EAGER inside the replay (MEASURED 2026-10-05, the L5 S3 re-read on Thor with --trunk-compile): the
        # switched precision state is a dynamo guard, so the compiled backbone RECOMPILED inside the replay and the
        # inductor build died (gcc on Thor) -- killing the training run at its first gs-due step. The replay is an
        # instrument: it runs the same modules eagerly, never compiles, never touches the compiled cache.
        with torch.random.fork_rng(devices=devs), _eager():
            _say(f"fp32 replay: forward start ({len(levers)} bf16 levers off, TF32 off, eager)")
            t_f0 = time.perf_counter()
            losses = run_losses()
            _sync()
            fwd_s = time.perf_counter() - t_f0
            _say(f"fp32 replay: forward {fwd_s:.2f}s")
            row = measure(term_tensors(losses, weights, keep), losses["loss"], groups, prefix=prefix)
        del losses
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32[0], tf32[1]
        torch.set_float32_matmul_precision(tf32[2])
        for m in levers:
            m.memory_levers["bf16"] = True
        for v, st in zip(gens, gstate):
            for k in [k for k in v._g if k not in st]:
                del v._g[k]              # a generator FIRST created by the replay: dropped, re-created fresh on use
            for k, s_ in st.items():
                v._g[k].set_state(s_)
        with torch.no_grad():
            for n, b in model.named_buffers():
                if n in bufs:
                    b.copy_(bufs[n])
    row[f"{prefix}fp32_replay"] = 1.0
    row[f"{prefix}fp32_replay_levers"] = float(len(levers))
    row[f"{prefix}fp32_replay_tf32_off"] = 1.0
    row[f"{prefix}fp32_replay_eager"] = 1.0
    row[f"{prefix}fp32_fwd_s"] = fwd_s
    _sync()
    row[f"{prefix}fp32_replay_s"] = time.perf_counter() - t_r0
    _say(f"fp32 replay: total {row[f'{prefix}fp32_replay_s']:.2f}s")
    return row

