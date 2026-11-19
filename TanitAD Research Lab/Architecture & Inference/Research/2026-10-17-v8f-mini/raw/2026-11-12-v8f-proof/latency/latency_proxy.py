#!/usr/bin/env python3
"""latency_proxy.py -- a LATENCY PROXY of the v8F-mini parts that are specific to the v8proof study (step S0 of COORD_D5_SCOPING_2026-11-19.md).

Runs ON THE POD GPU (also on cpu).  A proxy on the pod GPU: not a Thor number, not a product latency; the image encoder is NOT included (the study works on cached tokens).
Tier: none (a cost measurement, not a driving metric).  LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC: not applicable (a cost measurement).

  python latency_proxy.py --tree <tree root> --ckpt <ckpt_last.pt> --out-dir <new folder> --tracks T --eval-cache C --eval-grid G --eval-join J
         [--k-list 1,8,16,32] [--horizon-steps 0] [--extra-horizons 30] [--n 200] [--warmup 20] [--device cuda|cpu] [--n-windows 80] [--variants eager,fp16,compile]
         [--lib-dirs a:b] [--stub]

Timed regions (every region is bracketed by a device synchronisation before the start and after the end; torch.cuda.synchronize() on cuda):
  (a) one predictor step at batch 1 (level-1 predictor, context = the predictor's own);
  (b) the imagined rollout to the horizon for K candidates as ONE batch, and as K sequential batch-1 rollouts (the stack's own rollout, as imag_dump.roll_states);
  (c) the lead head on one state, the lead head at 1.0 s / 2.0 s on a rolled state, the lead head on cached tokens (as head_values), the trajectory heads (forward_infer at batch 1);
  (d) one full planning tick of ImagPlanner as it runs today (the pinned model planner + the rule + the imagined rollouts of the whole candidate set), with the predictor calls
      per tick COUNTED by a wrapper in a separate, untimed pass.
Self-checks gate the output: s1 (timer vs a known 50 ms sleep, and vs an asynchronous stub), s2 (batched vs sequential rollout states), s3 (doubled horizon takes 1.6 to 2.4 times as
long), s4 (counted predictor calls per tick above 0 and equal on two ticks of one input), s5 (the statistics of a known list).  A failed check: no timing table, exit 2.
Everything open is in LATENCY_DEFS.md.  No torch is imported at module level (the stub path and the check runner work without it).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import platform
import random
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))

FIRST_LINE = "Latency proxy on the pod GPU (not Thor, not a product latency; the image encoder is not included). Tier: none (a cost measurement, not a driving metric)."
FAMILY_LINE = "LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC: not applicable (a cost measurement)"
DT_S = 0.2                      # one cache / planner step (imag_plan.DT); the stub path uses it too
BUDGET_MS = 100.0               # the PI's end-to-end ruling (Thor, 4 cameras): shown only as an arithmetic share, never a verdict
S1_TOL_MS = 5.0
S2_TOL = 1e-4
S3_LO, S3_HI = 1.6, 2.4


class NotRun(Exception):
    """A variant or section that cannot run here; the text is recorded in the output."""


# ============================================================================================ statistics and the timer
def percentile(sorted_vals, q):
    """Linear interpolation between order statistics (numpy's default 'linear' method), q in [0, 100]."""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("no samples")
    k = (n - 1) * (q / 100.0)
    f = int(math.floor(k))
    c = min(f + 1, n - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def stats_ms(samples_s):
    """Samples in SECONDS -> p50 / p95 / mean / min / max in MILLISECONDS."""
    v = sorted(float(x) * 1000.0 for x in samples_s)
    return {"p50": percentile(v, 50), "p95": percentile(v, 95), "mean": sum(v) / len(v), "min": v[0], "max": v[-1], "n": len(v)}


def time_fn(fn, n, warmup, sync, clock=None):
    """Time `fn()` n times after `warmup` untimed calls.  Each timed region is: sync(); t0; fn(); sync(); t1.  Returns (list of seconds, seconds of the very first call)."""
    clk = clock if clock is not None else time.perf_counter
    first = None
    for _ in range(int(warmup)):
        t0 = clk()
        fn()
        sync()
        t1 = clk()
        if first is None:
            first = t1 - t0
    out = []
    for _ in range(int(n)):
        sync()
        t0 = clk()
        fn()
        sync()
        t1 = clk()
        out.append(t1 - t0)
    if first is None:
        first = out[0]
    return out, first


def measure(fn, n, warmup, sync):
    samples, first = time_fn(fn, n, warmup, sync)
    d = stats_ms(samples)
    d["first_call_ms"] = float(first) * 1000.0
    return d


class Counter:
    """Counts the calls of a wrapped callable (and the leading size of one argument).  Used for the predictor, the rollout and the lead head."""

    def __init__(self):
        self.n = 0
        self.rows = []

    def reset(self):
        self.n = 0
        self.rows = []

    def wrap(self, fn, rows_arg=0):
        def wrapped(*a, **k):
            self.n += 1
            try:
                self.rows.append(int(a[rows_arg].shape[0]))
            except Exception:
                pass
            return fn(*a, **k)
        return wrapped


def _flat(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        for y in x:
            yield from _flat(y)
    else:
        yield float(x)


def max_abs_diff(a, b):
    fa, fb = list(_flat(a)), list(_flat(b))
    if len(fa) != len(fb):
        return float("inf")
    return max((abs(x - y) for x, y in zip(fa, fb)), default=0.0)


def max_abs(a):
    return max((abs(x) for x in _flat(a)), default=0.0)


class AsyncDevice:
    """A stub device whose work is asynchronous: launch() returns at once, sync() waits until the work is done (what a GPU does).  Tests the synchronisation discipline of the timer."""

    def __init__(self):
        self.deadline = 0.0

    def launch(self, seconds):
        self.deadline = time.perf_counter() + float(seconds)

    def sync(self):
        rem = self.deadline - time.perf_counter()
        if rem > 0:
            time.sleep(rem)


# ============================================================================================ the self-checks
def check_s1(sync):
    samples, _ = time_fn(lambda: time.sleep(0.05), 20, 3, sync)
    st = stats_ms(samples)
    return {"id": "s1", "name": "timer measures a known 50 ms sleep", "pass": bool(abs(st["p50"] - 50.0) <= S1_TOL_MS), "value_p50_ms": st["p50"], "bar": f"|p50 - 50| <= {S1_TOL_MS} ms"}


def check_s1a():
    dev = AsyncDevice()
    samples, _ = time_fn(lambda: dev.launch(0.05), 20, 3, dev.sync)
    st = stats_ms(samples)
    return {"id": "s1a", "name": "timer waits for an asynchronous device (50 ms of async work)", "pass": bool(abs(st["p50"] - 50.0) <= S1_TOL_MS), "value_p50_ms": st["p50"],
            "bar": f"|p50 - 50| <= {S1_TOL_MS} ms"}


def check_s2(subject, K, H):
    a = subject.rollout_states(K, H, "batch")
    b = subject.rollout_states(K, H, "seq")
    d = max_abs_diff(a, b)
    nz = max_abs(a)
    return {"id": "s2", "name": "batched and sequential rollouts give the same states", "pass": bool(d <= S2_TOL and nz > 0.0), "value_max_abs_diff": d, "max_abs_state": nz, "K": K, "H": H,
            "bar": f"max abs diff <= {S2_TOL} and states not all zero"}


def check_s3(subject, K, H):
    f1 = subject.rollout_fn(K, H, "batch")
    f2 = subject.rollout_fn(K, 2 * H, "batch")
    s_1 = stats_ms(time_fn(f1, 30, 5, subject.sync)[0])
    s_2 = stats_ms(time_fn(f2, 30, 5, subject.sync)[0])
    ratio = s_2["p50"] / s_1["p50"] if s_1["p50"] > 0 else float("inf")
    return {"id": "s3", "name": "a doubled horizon takes between 1.6 and 2.4 times as long", "pass": bool(S3_LO <= ratio <= S3_HI), "value_ratio": ratio, "p50_ms_H": s_1["p50"],
            "p50_ms_2H": s_2["p50"], "K": K, "H": H, "bar": f"{S3_LO} <= ratio <= {S3_HI}"}


def check_s4(subject):
    spec = subject.tick_primary()
    idx = None
    for i in range(spec["n_inputs"]):
        if spec["count"](i)["predictor_calls"] > 0:
            idx = i
            break
    if idx is None:
        return {"id": "s4", "name": "counted predictor calls per tick", "pass": False, "detail": "no input of the set produced a tick with a predictor call", "bar": "calls > 0 and equal on two ticks"}
    c1 = spec["count"](idx)["predictor_calls"]
    c2 = spec["count"](idx)["predictor_calls"]
    return {"id": "s4", "name": "counted predictor calls per tick", "pass": bool(c1 > 0 and c1 == c2), "value": [c1, c2], "input_index": idx, "bar": "calls > 0 and equal on two ticks of one input"}


def check_s5():
    st = stats_ms([i / 1000.0 for i in range(1, 101)])
    ok = abs(st["p50"] - 50.5) < 1e-9 and abs(st["p95"] - 95.05) < 1e-9 and abs(st["mean"] - 50.5) < 1e-9 and st["n"] == 100
    sk = stats_ms([0.001, 0.001, 0.001, 0.001, 0.1])             # a skewed list: p50 1, p95 80.2, mean 20.8 (p50 and mean differ, p95 and max differ)
    ok2 = abs(sk["p50"] - 1.0) < 1e-9 and abs(sk["p95"] - 80.2) < 1e-9 and abs(sk["mean"] - 20.8) < 1e-9 and abs(sk["max"] - 100.0) < 1e-9 and abs(sk["min"] - 1.0) < 1e-9
    return {"id": "s5", "name": "statistics of two known lists (1..100 ms; 1,1,1,1,100 ms)", "pass": bool(ok and ok2), "value": {k: st[k] for k in ("p50", "p95", "mean", "n")},
            "value_skewed": {k: sk[k] for k in ("p50", "p95", "mean", "max")}, "bar": "p50 50.5, p95 95.05, mean 50.5, n 100; skewed list p50 1, p95 80.2, mean 20.8, max 100"}


def run_self_checks(subject, k_for_s2, k_for_s3, h):
    out = []
    for name, thunk in (("s1", lambda: check_s1(subject.sync)), ("s1a", check_s1a), ("s2", lambda: check_s2(subject, k_for_s2, h)),
                        ("s3", lambda: check_s3(subject, k_for_s3, h)), ("s4", lambda: check_s4(subject)), ("s5", check_s5)):
        try:
            out.append(thunk())
        except Exception as e:                                   # a check that cannot run is a failed check
            out.append({"id": name, "pass": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    return out


# ============================================================================================ a stub subject (pure python; the check runner and --stub)
class StubSubject:
    """The same interface as RealSubject, with no model: a recurrence in pure python whose cost is a sleep per predictor call (so the timer has work to respond to)."""
    label = "stub"

    def __init__(self, step_s=0.001, dim=4, n_inputs=4, horizon=10, n_cands=3):
        self.step_s, self.dim, self.n_inputs, self.horizon, self.n_cands = step_s, dim, n_inputs, horizon, n_cands
        self.counter = Counter()
        self.pred = self.counter.wrap(self._pred_raw)
        self._acts = {}

    sync = staticmethod(lambda: None)

    def _pred_raw(self, z, a):
        time.sleep(self.step_s)                                  # one call = one unit of work, whatever the number of rows
        return [[math.tanh(0.7 * zi + 0.2 * ai[0] - 0.1 * ai[1] + 0.05 * i) for i, zi in enumerate(zr)] for zr, ai in zip(z, a)]

    def _A(self, K, H):
        key = (K, H)
        if key not in self._acts:
            r = random.Random(1234 + 1000 * K + H)
            self._acts[key] = [[[r.gauss(0, 1.0), r.gauss(0, 0.05)] for _ in range(H)] for _ in range(K)]
        return self._acts[key]

    def _roll(self, z0, A):                                      # z0 [K][D], A [K][H][2] -> [K][H][D]
        K, H = len(z0), len(A[0])
        z, out = [list(r) for r in z0], [[] for _ in range(K)]
        for h in range(H):
            z = self.pred(z, [A[k][h] for k in range(K)])
            for k in range(K):
                out[k].append(list(z[k]))
        return out

    def info(self):
        return {"subject": "stub", "note": "no model: a pure-python recurrence; the numbers are NOT a measurement of the model"}

    def planner_horizon(self):
        return self.horizon, "stub"

    def step_fn(self):
        z, a = [[0.1] * self.dim], [[0.5, 0.0]]
        return lambda: self.pred(z, a)

    def rollout_fn(self, K, H, mode):
        z0, A = [[0.1 * (k + 1)] * self.dim for k in range(K)], self._A(K, H)
        if mode == "batch":
            return lambda: self._roll(z0, A)

        def seq():
            for k in range(K):
                self._roll(z0[k:k + 1], A[k:k + 1])
        return seq

    def rollout_states(self, K, H, mode):
        z0, A = [[0.1 * (k + 1)] * self.dim for k in range(K)], self._A(K, H)
        if mode == "batch":
            return self._roll(z0, A)
        return [self._roll(z0[k:k + 1], A[k:k + 1])[0] for k in range(K)]

    def head_fns(self):
        return {"c1_lead_head_one_state": lambda: time.sleep(0.0005), "c4_forward_infer_b1": lambda: time.sleep(0.002)}

    def tick_primary(self):
        return self.tick_set()["stub-tick"]

    def tick_set(self):
        def fn(i):
            time.sleep(0.002)
            if i % 2 == 0:                                       # even inputs have a lead: the planner rolls its candidate set
                z0 = [[0.1] * self.dim for _ in range(self.n_cands)]
                self._roll(z0, self._A(self.n_cands, self.horizon))

        def count(i):
            self.counter.reset()
            fn(i)
            return {"predictor_calls": self.counter.n, "rollout_calls": 0, "head_calls": 0, "predictor_rows": list(self.counter.rows)}
        return {"stub-tick": {"n_inputs": self.n_inputs, "fn": fn, "count": count, "n_cands": self.n_cands, "label": "stub"}}


# ============================================================================================ the real subject (torch, the tree, the study's tools)
def _add_paths(tree, lib_dirs):
    stack = os.path.abspath(os.path.join(tree, "stack")).replace(os.sep, "/")
    if stack not in sys.path:
        sys.path.insert(0, stack)
    cands = list(lib_dirs) + [HERE, os.path.join(HERE, "..", "imagplan"), os.path.join(HERE, "..", "distrule"), os.path.join(HERE, "..", "imaghead")]
    for d in cands:
        d = os.path.abspath(d)
        if os.path.isdir(d) and d not in sys.path:
            sys.path.append(d)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class RealSetup:
    """Everything built once: the model as the T1 tool builds it, the real windows of the eval grid as the T1 tool builds them, one batch per window at tick 0."""


def build_real_setup(args):
    import torch
    _add_paths(args.tree, [d for d in (args.lib_dirs or "").split(os.pathsep) if d])
    import numpy as np
    import imag_plan as IP
    import v8f_t1_eval_imagplan as IPW
    import v8f_t1_eval_rule as W
    import imag_dump as IH
    S = RealSetup()
    S.torch, S.IP, S.IPW, S.W, S.IH = torch, IP, IPW, W, IH
    T, pinned_path = W.load_pinned(args.tree)
    S.T, S.pinned_path, S.pinned_sha = T, pinned_path, W.sha256_file(pinned_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("REFUSED: --device cuda but torch.cuda.is_available() is False")
    S.dev = torch.device(args.device)
    # the model: the five lines of T.cmd_run (lines 505-513) through the study's loader imag_dump.load_model (the same construction, strict state dict, refusals for a missing lead head)
    S.cfg, S.model, S.stk, S.ck = IH.load_model(torch, T.M, args.ckpt, require_head=True)
    if S.cfg.kin_mode != "yaw":
        raise SystemExit("REFUSED: kin_mode is not yaw (T.cmd_run refuses it too)")
    S.model.to(S.dev).eval()
    S.ctx = IH.predictor_context(S.stk)
    rest = ["run", "--ckpt", args.ckpt, "--tag", "latency", "--tracks", args.tracks, "--eval-cache", args.eval_cache, "--eval-grid", args.eval_grid, "--eval-join", args.eval_join,
            "--out-dir", args.out_dir, "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal", "--device", args.device]
    S.a = T.build_parser().parse_args(rest)                      # the argv of the chain's dump step (pod_imagplan_run.sh) without the dump flags
    tracks = T.D.load_native_tracks(S.a.tracks)
    J = T.D.load_join(S.a.eval_join)
    store = T.D.ClipStore(S.a.eval_cache, "eval", lru=S.a.lru, grid=S.ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    wins = T.D.eval_grid_windows(S.a.eval_grid, J)
    nw = min(int(args.n_windows), len(wins))
    sel = [wins[i] for i in sorted(set(int(round(x)) for x in np.linspace(0, len(wins) - 1, nw)))]
    S.n_grid_windows, S.n_windows = len(wins), len(sel)
    S.store, S.tracks, S.J, S.sel = store, tracks, J, sel
    proxy = T._OneClipStore()
    ds = T.D.V8FMiniWindows(proxy, J, [], hist=S.cfg.hist, hmax=S.cfg.hmax, max_age_t=S.cfg.max_age_t, max_age_s=S.cfg.max_age_s, window_start=S.cfg.window_start)
    S.planners = {}
    S.states = None
    S.planner_specs = {"cand-imag/scale (card default set)": ("cand-imag", "scale"), "cand-imagkin/decel (note ADD40_N3, N4)": ("cand-imagkin", "decel")}
    for label, (source, family) in S.planner_specs.items():
        cset = IP.parse_decel_set(IP.DECEL_CARD) if family == "decel" else IP.parse_cand_set(IP.CAND_CARD)
        opts = {"cset": cset, "floor": "card", "fallback": "zero", "const_gap": None, "shift_m": 0.0, "noise_seed": -1, "action_def": "logged", "family": family}
        made, restore = IPW.install(T, source, S.a, opts)        # patches T.model_planner / make_states for this block only
        try:
            S.planners[label] = T.model_planner(S.model, S.dev)  # an ImagPlanner around the pinned model planner
            if S.states is None:
                S.states = T.make_states(store, tracks, J, sel, S.a.nav)
        finally:
            restore()
        S.planners[label].n_cands = len(IP.effective_set(cset, "zero", family))
    a_t, a_s = T.ages_for(0, S.a.schedule)
    S.batches = []
    for st in S.states:                                          # tick 0 of every window: the item the rollout loop builds, batch 1
        it = T.build_item(ds, proxy, st, 0, a_t, a_s, S.a.scene, S.a.limit, None)
        full = T.D.collate([it])
        S.batches.append({k: full[k].to(S.dev) for k in T.MODEL_KEYS})
    with torch.no_grad():
        f = S.batches[0]["feats"]
        S.feats_shape = list(f.shape)
        S.hist_lead = int(S.model.wm.lead_head.hist)
        z = S.model.wm.z1(S.model.adapter(f[:, -S.hist_lead:].float()) + S.model.pos[None, None])      # [1, hist, D]  (the formula of head_values)
        S.Z5, S.z0 = z, z[:, -1]                                  # z0 [1, D] = the level-1 state of the tick frame (the formula of ImagPlanner._z0)
    S.D = int(S.z0.shape[-1])
    return S


class RealSubject:
    """The timed callables of one variant (eager | fp16 | compile) over a RealSetup."""

    def __init__(self, S, mode="eager"):
        self.S, self.mode, self.torch = S, mode, S.torch
        self.label = f"real-{mode}"
        self._acts = {}
        self._compiled = {}
        torch = self.torch
        if mode == "fp16":
            if S.dev.type != "cuda":
                raise NotRun("fp16 autocast: not run on a cpu device")
            with torch.autocast(device_type=S.dev.type, dtype=torch.float16):
                if not torch.is_autocast_enabled():
                    raise NotRun("fp16 autocast did not turn on")
        elif mode == "compile":
            if not hasattr(torch, "compile"):
                raise NotRun("this torch has no torch.compile")
            try:
                torch._dynamo.config.cache_size_limit = 64       # K x H shapes recompile; the default limit of 8 would fall back to eager silently
            except Exception:
                pass
        elif mode != "eager":
            raise ValueError(mode)

    def sync(self):
        if self.S.dev.type == "cuda":
            self.torch.cuda.synchronize()

    def _cm(self):
        if self.mode == "fp16":
            return self.torch.autocast(device_type=self.S.dev.type, dtype=self.torch.float16)
        return contextlib.nullcontext()

    def planner_horizon(self):
        return int(self.S.IP.N_ROLL), "imag_plan.N_ROLL (imag_plan.py:41): 10 rollout steps of 0.2 s = 2.0 s, the horizon the planner rolls"

    def _A(self, K, H):
        key = (K, H)
        if key not in self._acts:
            torch, S = self.torch, self.S
            g = torch.Generator().manual_seed(1234 + 1000 * K + H)
            raw = torch.randn(K, H, 2, generator=g) * torch.tensor([1.0, 0.05])      # raw level-1 actions (m/s2, rad/s) of the size the logged ones have
            with torch.no_grad():
                self._acts[key] = S.model.wm.scale_actions(raw.to(S.dev))
        return self._acts[key]

    def _roll_callable(self):
        S = self.S
        stk, ctx = S.stk, S.ctx

        def roll(z, A):
            return stk.rollout(1, z, A, context=ctx)
        if self.mode == "compile":
            if "roll" not in self._compiled:
                self._compiled["roll"] = self.torch.compile(roll)
            return self._compiled["roll"]
        return roll

    def _pred_callable(self):
        pred = self.S.stk.predictors[0]
        if self.mode == "compile":
            if "pred" not in self._compiled:
                self._compiled["pred"] = self.torch.compile(pred)
            return self._compiled["pred"]
        return pred

    def info(self):
        S, torch = self.S, self.torch
        d = {"subject": "real", "variant": self.mode, "torch": torch.__version__, "torch_cuda": getattr(torch.version, "cuda", None), "python": platform.python_version(),
             "device": str(S.dev), "torch_threads": torch.get_num_threads(), "gpu_name": torch.cuda.get_device_name(0) if S.dev.type == "cuda" else None,
             "cudnn": torch.backends.cudnn.version() if S.dev.type == "cuda" else None, "ckpt": S.args_ckpt, "ckpt_sha256": S.ckpt_sha, "pinned_tool_sha256": S.pinned_sha,
             "pinned_tool_expected_sha256": S.W.PINNED_SHA256, "state_dim_D": S.D, "state_shape_batch1": [1, S.D], "predictor_context": S.ctx,
             "cached_tokens_batch1_shape": S.feats_shape, "tokens_per_frame": S.feats_shape[2], "frames_in_batch": S.feats_shape[1], "d_feat": S.feats_shape[3],
             "lead_head_hist": S.hist_lead, "n_grid_windows": S.n_grid_windows, "n_windows_used": S.n_windows,
             "cfg": {k: str(getattr(S.cfg, k)) for k in ("kin_mode", "wm", "wm_state", "wm_dim", "hist", "hmax", "d", "d_feat", "n_tok", "lead_sup", "lead_hist") if hasattr(S.cfg, k)}}
        m = S.model
        d["params_total"] = int(sum(p.numel() for p in m.parameters()))
        d["params_top_level"] = {n: int(sum(p.numel() for p in c.parameters())) for n, c in m.named_children()}
        d["params_top_level"]["(direct parameters, e.g. pos, time)"] = int(sum(p.numel() for p in m.parameters(recurse=False)))
        d["params_wm"] = {n: int(sum(p.numel() for p in c.parameters())) for n, c in m.wm.named_children()}
        d["params_wm_stack"] = {n: int(sum(p.numel() for p in c.parameters())) for n, c in S.stk.named_children()}
        d["planner_candidates"] = {k: v.n_cands for k, v in S.planners.items()}
        return d

    # ---- (a)
    def step_fn(self):
        torch, S = self.torch, self.S
        pred = self._pred_callable()
        zc = S.z0[:, None, :].expand(1, S.ctx, -1).contiguous()
        with torch.no_grad():
            e = S.stk.act_embed(self._A(1, S.ctx))

        def fn():
            with torch.no_grad(), self._cm():
                pred(zc, e)[:, -1]
        return fn

    # ---- (b)
    def rollout_fn(self, K, H, mode):
        torch, S = self.torch, self.S
        roll = self._roll_callable()
        z0K, AK = S.z0.expand(K, -1).contiguous(), self._A(K, H)
        if mode == "batch":
            def fn():
                with torch.no_grad(), self._cm():
                    roll(z0K, AK)
            return fn
        zs, As = [z0K[k:k + 1].contiguous() for k in range(K)], [AK[k:k + 1].contiguous() for k in range(K)]

        def seq():
            with torch.no_grad(), self._cm():
                for z, A in zip(zs, As):
                    roll(z, A)
        return seq

    def rollout_states(self, K, H, mode):
        torch, S = self.torch, self.S
        roll = self._roll_callable()
        z0K, AK = S.z0.expand(K, -1).contiguous(), self._A(K, H)
        with torch.no_grad(), self._cm():
            if mode == "batch":
                r = roll(z0K, AK)
            else:
                r = torch.cat([roll(z0K[k:k + 1].contiguous(), AK[k:k + 1].contiguous()) for k in range(K)], 0)
        return r.double().cpu().numpy()

    # ---- (c)
    def head_fns(self):
        if self.mode == "compile":
            raise NotRun("the compile variant covers (a) and (b) only")
        torch, S = self.torch, self.S
        m, IH, W, T = S.model, S.IH, S.W, S.T
        roll = self._roll_callable()
        with torch.no_grad():
            Rr = roll(S.z0, self._A(1, S.IP.N_ROLL))             # [1, 10, D]: a rolled state sequence for the head reads at 1.0 s and 2.0 s
        b0 = S.batches[0]

        def c1():
            with torch.no_grad(), self._cm():
                m.wm.lead_head(S.Z5)

        def c2():
            with torch.no_grad(), self._cm():
                IH.head_reads_rolled(torch, m, T.M, Rr)

        def c3():
            with self._cm():
                W.head_values(m, b0["feats"], b0["frame_ok"])

        def c4():
            with torch.no_grad(), self._cm():
                m.forward_infer(b0)
        return {"c1_lead_head_one_state": c1, "c2_lead_head_at_1s_2s_on_rolled_state (2 head calls)": c2,
                "c3_lead_head_on_cached_tokens (head_values: adapter, z1, head, host copy)": c3, "c4_trajectory_heads (forward_infer, S+T+O readers and all heads, batch 1)": c4}

    # ---- (d)
    def tick_set(self):
        if self.mode == "compile":
            raise NotRun("the planning tick has host-side numpy work and is not compiled")
        S, torch = self.S, self.torch
        out = {}
        for label, pl in S.planners.items():
            def fn(i, pl=pl):
                pl.bind([S.states[i]])
                with self._cm():
                    pl(S.batches[i])

            def count(i, pl=pl, fn=fn):
                stk, lead = S.stk, S.model.wm.lead_head
                pred = stk.predictors[0]
                cp, cr, ch = Counter(), Counter(), Counter()
                pred.forward = cp.wrap(pred.forward)
                stk.rollout = cr.wrap(stk.rollout, rows_arg=1)
                lead.forward = ch.wrap(lead.forward)
                try:
                    fn(i)
                finally:
                    del pred.forward
                    del stk.rollout
                    del lead.forward
                return {"predictor_calls": cp.n, "rollout_calls": cr.n, "head_calls": ch.n, "predictor_rows": list(cp.rows)}
            out[label] = {"n_inputs": len(S.states), "fn": fn, "count": count, "n_cands": pl.n_cands, "label": label}
        return out

    def tick_primary(self):
        return self.tick_set()[next(iter(self.S.planners))]


# ============================================================================================ one variant, all regions
def _section(fn):
    try:
        return fn()
    except NotRun as e:
        return {"status": "NOT RUN", "error": str(e)}
    except Exception as e:
        return {"status": "NOT RUN", "error": f"{type(e).__name__}: {str(e)[:300]}"}


def _cycle(fn, idxs):
    it = [0]

    def f():
        i = idxs[it[0] % len(idxs)]
        it[0] += 1
        fn(i)
    return f


def collect(subject, k_list, horizons, n, warmup, plan_h):
    sync = subject.sync
    res = {}
    res["a_predictor_step_batch1"] = _section(lambda: measure(subject.step_fn(), n, warmup, sync))
    b = {}

    def do_b():
        for H in horizons:
            for K in k_list:
                rb = measure(subject.rollout_fn(K, H, "batch"), n, warmup, sync)
                rs = measure(subject.rollout_fn(K, H, "seq"), n, warmup, sync)
                secs = H * DT_S
                b[f"H{H}_K{K}"] = {"H_steps": H, "horizon_s": secs, "K": K, "batch": rb, "seq_total": rs, "seq_mean_per_rollout_ms": rs["mean"] / K,
                                   "rtf_batch_p50": secs / (rb["p50"] / 1000.0), "rtf_seq_p50": secs / (rs["p50"] / 1000.0), "seq_over_batch_p50": rs["p50"] / rb["p50"],
                                   "is_planner_horizon": H == plan_h}
        return b
    res["b_rollout"] = _section(do_b)

    def do_c():
        return {name: measure(fn, n, warmup, sync) for name, fn in subject.head_fns().items()}
    res["c_heads"] = _section(do_c)

    def do_d():
        d = {}
        for name, spec in subject.tick_set().items():
            counts = [spec["count"](i) for i in range(spec["n_inputs"])]
            act = [i for i, c in enumerate(counts) if c["predictor_calls"] > 0]
            calls = sorted(set(c["predictor_calls"] for c in counts))
            entry = {"n_inputs": spec["n_inputs"], "n_candidates": spec["n_cands"], "n_inputs_with_rollout": len(act), "fraction_with_rollout": len(act) / spec["n_inputs"],
                     "predictor_calls_per_tick_values": calls, "predictor_calls_per_tick_counted": {str(v): sum(1 for c in counts if c["predictor_calls"] == v) for v in calls},
                     "rollout_calls_per_tick_values": sorted(set(c["rollout_calls"] for c in counts)), "head_calls_per_tick_values": sorted(set(c["head_calls"] for c in counts)),
                     "predictor_rows_per_call_on_a_rolling_tick": (counts[act[0]]["predictor_rows"] if act else None),
                     "count_method": "wrapper on the predictor's forward, on stack.rollout and on the lead head's forward, installed only for an untimed pass (never during the timed pass)"}
            entry["tick_all_inputs"] = measure(_cycle(spec["fn"], list(range(spec["n_inputs"]))), n, warmup, sync)
            if act:
                entry["tick_inputs_with_rollout"] = measure(_cycle(spec["fn"], act), n, warmup, sync)
            else:
                entry["tick_inputs_with_rollout"] = {"status": "NOT RUN", "error": "no input of the set made the planner roll the world model"}
            d[name] = entry
        return d
    res["d_planning_tick"] = _section(do_d)
    return res


# ============================================================================================ the writers
def _f(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def _row(name, st):
    if not isinstance(st, dict) or "p50" not in st:
        return f"| {name} | NOT RUN: {st.get('error', '') if isinstance(st, dict) else st} | | | |"
    return f"| {name} | {_f(st['p50'])} | {_f(st['p95'])} | {_f(st['mean'])} | {st['n']} |"


def render_md(R):
    L = [FIRST_LINE, FAMILY_LINE, "", f"Generated {R['generated_utc']} by latency_proxy.py (sha256 {R['tool_sha256'][:12]}). All times in ms; p50 / p95 / mean over n timed repetitions after warm-up; "
         "every timed region is bracketed by a device synchronisation.", ""]
    info = R["info"]
    L += ["## Setup", "", f"* subject: {info.get('subject')}; device {info.get('device')}; GPU {info.get('gpu_name')}; torch {info.get('torch')} (cuda {info.get('torch_cuda')}); cudnn {info.get('cudnn')}",
          f"* checkpoint sha256 {info.get('ckpt_sha256')}; pinned tool sha256 {info.get('pinned_tool_sha256')} (expected {info.get('pinned_tool_expected_sha256')})",
          f"* state shape at batch 1 {info.get('state_shape_batch1')}; predictor context {info.get('predictor_context')}; cached tokens at batch 1 {info.get('cached_tokens_batch1_shape')} "
          f"(tokens per frame {info.get('tokens_per_frame')}, frames {info.get('frames_in_batch')}, feature width {info.get('d_feat')})",
          f"* planner horizon {R['planner_horizon_steps']} steps of {DT_S} s = {R['planner_horizon_steps'] * DT_S:.1f} s; source: {R['planner_horizon_source']}",
          f"* parameters {info.get('params_total')}; per top-level module {json.dumps(info.get('params_top_level'))}; world-model parts {json.dumps(info.get('params_wm'))}; world-model stack {json.dumps(info.get('params_wm_stack'))}",
          f"* windows: {info.get('n_windows_used')} of the {info.get('n_grid_windows')} grid windows; n {R['args']['n']}, warm-up {R['args']['warmup']}, K list {R['args']['k_list']}", "", "## Self-checks (all passed, or no table is written)", ""]
    for c in R["checks"]:
        L.append(f"* {c['id']} {c.get('name', '')}: {'PASS' if c['pass'] else 'FAIL'} ({json.dumps({k: v for k, v in c.items() if k not in ('id', 'name', 'pass')})})")
    for vname, v in R["variants"].items():
        L += ["", f"## Variant: {vname}", ""]
        if v.get("status") == "NOT RUN":
            L.append(f"NOT RUN: {v.get('error')}")
            continue
        L += ["### (a) one predictor step at batch 1", "", "| region | p50 | p95 | mean | n |", "|---|---|---|---|---|", _row("predictor step, batch 1", v["a_predictor_step_batch1"]), ""]
        b = v["b_rollout"]
        L += ["### (b) imagined rollout, K candidates", ""]
        if "status" in b:
            L.append(f"NOT RUN: {b.get('error')}")
        else:
            L += ["| H (steps / s) | K | batch p50 | batch p95 | batch mean | K x batch-1 total p50 | total p95 | per rollout mean | seq / batch | real-time factor (batch p50) |", "|---|---|---|---|---|---|---|---|---|---|"]
            for key, r in b.items():
                tag = " (planner horizon)" if r["is_planner_horizon"] else ""
                L.append(f"| {r['H_steps']} / {r['horizon_s']:.1f}{tag} | {r['K']} | {_f(r['batch']['p50'])} | {_f(r['batch']['p95'])} | {_f(r['batch']['mean'])} | {_f(r['seq_total']['p50'])} | "
                         f"{_f(r['seq_total']['p95'])} | {_f(r['seq_mean_per_rollout_ms'])} | {_f(r['seq_over_batch_p50'], 2)} | {_f(r['rtf_batch_p50'], 1)} |")
            L.append("")
            L.append("real-time factor = simulated seconds rolled / wall seconds spent (above 1 = faster than real time).")
        c = v["c_heads"]
        L += ["", "### (c) heads", "", "| region | p50 | p95 | mean | n |", "|---|---|---|---|---|"]
        if "status" in c:
            L.append(f"| NOT RUN: {c.get('error')} | | | | |")
        else:
            L += [_row(k, s) for k, s in c.items()]
        d = v["d_planning_tick"]
        L += ["", "### (d) one full planning tick of the planner tool as it runs today (batch 1)", ""]
        if "status" in d:
            L.append(f"NOT RUN: {d.get('error')}")
        else:
            for name, e in d.items():
                L += [f"**{name}**: {e['n_candidates']} candidates; the planner rolled the world model on {e['n_inputs_with_rollout']} of {e['n_inputs']} inputs ({100 * e['fraction_with_rollout']:.0f} %); "
                      f"predictor calls per tick (counted) {e['predictor_calls_per_tick_counted']}; predictor rows per call on a rolling tick {e['predictor_rows_per_call_on_a_rolling_tick']}; "
                      f"rollout calls {e['rollout_calls_per_tick_values']}; lead-head calls {e['head_calls_per_tick_values']}.", "",
                      "| region | p50 | p95 | mean | n |", "|---|---|---|---|---|", _row("tick, every input", e["tick_all_inputs"]), _row("tick, inputs where it rolls", e["tick_inputs_with_rollout"]), ""]
                st = e["tick_inputs_with_rollout"]
                if "p50" in st:
                    L += [f"Arithmetic share of the PI's 100 ms end-to-end budget (Thor, 4 cameras; the encoder is not in this tick, so this is NOT a pass or fail): p50 {100 * st['p50'] / BUDGET_MS:.0f} %, p95 {100 * st['p95'] / BUDGET_MS:.0f} %.", ""]
        if "variant_notes" in v:
            L += ["", "Notes: " + json.dumps(v["variant_notes"])]
    L += ["", "Definitions and open choices: LATENCY_DEFS.md.", ""]
    return "\n".join(L)


def write_outputs(out_dir, R, stub=False):
    base = "LATENCY_PROXY_STUB" if stub else "LATENCY_PROXY"
    os.makedirs(out_dir, exist_ok=True)
    jp, mp = os.path.join(out_dir, base + ".json"), os.path.join(out_dir, base + ".md")
    for p, text in ((jp, json.dumps(R, indent=1)), (mp, render_md(R))):
        with open(p, "x", encoding="utf-8", newline="\n") as fh:          # mode x: an existing result is never overwritten
            fh.write(text)
    return jp, mp


def write_checks_failed(out_dir, checks, subject_label):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "SELF_CHECKS_FAILED.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"first_line": FIRST_LINE, "family_line": FAMILY_LINE, "subject": subject_label, "note": "a self-check failed: NO timing table was written", "checks": checks}, fh, indent=1)
    return p


# ============================================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="latency proxy of the v8F-mini study-specific parts (pod GPU proxy)")
    ap.add_argument("--tree", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k-list", default="1,8,16,32")
    ap.add_argument("--horizon-steps", type=int, default=0, help="rollout steps; 0 = the planner's own (imag_plan.N_ROLL = 10)")
    ap.add_argument("--extra-horizons", default="30", help="comma list of extra horizons (steps) timed beside the planner's; 30 = the 6 s of the plan; '' for none")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--n-windows", type=int, default=80, help="real eval windows used for the tick (tick 0 of each, batch 1)")
    ap.add_argument("--variants", default="eager,fp16,compile")
    ap.add_argument("--tracks", default="")
    ap.add_argument("--eval-cache", default="")
    ap.add_argument("--eval-grid", default="")
    ap.add_argument("--eval-join", default="")
    ap.add_argument("--lib-dirs", default="", help="folders (os.pathsep separated) with imag_plan.py v8f_t1_eval_imagplan.py dist_rule.py v8f_t1_eval_rule.py imag_dump.py; default: next to this file and ../imagplan ../distrule ../imaghead")
    ap.add_argument("--stub", action="store_true", help="no model: run the whole flow on the pure-python stub (the check runner uses it); writes LATENCY_PROXY_STUB.*")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    if not k_list or min(k_list) < 1:
        raise SystemExit("REFUSED: --k-list must hold positive integers")
    t_start = time.time()
    if args.stub:
        S = None
        base = StubSubject()

        def mk(mode):
            if mode != "eager":
                raise NotRun(f"{mode}: no such variant for the stub")
            return base
        ckpt_sha = "stub"
    else:
        for need in ("tree", "ckpt", "tracks", "eval_cache", "eval_grid", "eval_join"):
            if not getattr(args, need):
                raise SystemExit(f"REFUSED: --{need.replace('_', '-')} is required without --stub")
        S = build_real_setup(args)
        S.args_ckpt = os.path.abspath(args.ckpt).replace(os.sep, "/")
        S.ckpt_sha = sha256_file(args.ckpt)
        ckpt_sha = S.ckpt_sha

        def mk(mode):
            return RealSubject(S, mode)
    eager = mk("eager")
    plan_h, plan_src = eager.planner_horizon()
    H = args.horizon_steps if args.horizon_steps > 0 else plan_h
    horizons = [H] + [int(x) for x in args.extra_horizons.split(",") if x.strip() and int(x) != H]
    k_s2 = max(k_list)
    k_s3 = 8 if 8 in k_list else k_list[0]
    checks = run_self_checks(eager, k_s2, k_s3, H)
    print("self-checks:", " ".join(f"{c['id']}={'PASS' if c['pass'] else 'FAIL'}" for c in checks), flush=True)
    if not all(c["pass"] for c in checks):
        p = write_checks_failed(args.out_dir, checks, eager.label)
        print("ZZ LATENCY CHECKS FAILED, no timing table; wrote", p, flush=True)
        return 2
    variants = {}
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    if "eager" not in names:
        names = ["eager"] + names
    for vn in names:
        def one(vn=vn):
            subj = eager if vn == "eager" else mk(vn)
            r = collect(subj, k_list, [H] if vn == "compile" else horizons, args.n, args.warmup, H)      # compile: the planner's horizon only (the unrolled 30-step graph is a long compile)
            if vn != "eager" and not args.stub:
                try:                                              # information only: how far the variant's states are from the eager ones
                    a = eager.rollout_states(k_s3, H, "batch")
                    b = subj.rollout_states(k_s3, H, "batch")
                    r["variant_notes"] = {"max_abs_state_diff_vs_eager_K%d_H%d" % (k_s3, H): max_abs_diff(a, b)}
                except Exception as e:
                    r["variant_notes"] = {"state_diff_vs_eager": f"NOT RUN: {type(e).__name__}: {str(e)[:200]}"}
            return r
        print("variant", vn, flush=True)
        variants[vn] = _section(one)
    info = eager.info()
    rc = 0
    if variants.get("eager", {}).get("status") == "NOT RUN":
        rc = 1
    for sect in ("a_predictor_step_batch1", "b_rollout", "c_heads", "d_planning_tick"):
        if isinstance(variants.get("eager", {}).get(sect), dict) and variants["eager"][sect].get("status") == "NOT RUN":
            rc = 1                                                 # an eager section that could not run is a failed measurement, reported in the files
    R = {"first_line": FIRST_LINE, "family_line": FAMILY_LINE, "tier": "none (a cost measurement, not a driving metric)", "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "tool": os.path.abspath(__file__).replace(os.sep, "/"), "tool_sha256": sha256_file(os.path.abspath(__file__)), "argv": list(argv if argv is not None else sys.argv[1:]),
         "args": {"n": args.n, "warmup": args.warmup, "k_list": k_list, "horizon_steps": H, "horizons_timed": horizons, "device": args.device, "n_windows": args.n_windows, "variants": names},
         "planner_horizon_steps": H, "planner_horizon_source": ("--horizon-steps" if args.horizon_steps > 0 else plan_src), "ckpt_sha256": ckpt_sha, "info": info, "checks": checks,
         "variants": variants, "seconds_total": round(time.time() - t_start, 1)}
    try:
        jp, mp = write_outputs(args.out_dir, R, stub=args.stub)
    except FileExistsError as e:
        print(f"REFUSED: output exists, never overwritten: {e}", flush=True)
        return 4
    print("wrote", jp, mp, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
