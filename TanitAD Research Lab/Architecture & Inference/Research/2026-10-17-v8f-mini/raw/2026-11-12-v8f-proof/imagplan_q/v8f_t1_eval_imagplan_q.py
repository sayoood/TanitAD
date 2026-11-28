#!/usr/bin/env python3
"""v8f_t1_eval_imagplan.py -- the IMAGPLAN wrapper (card PREREG_V8F_PROOF_ADD40) around the pinned T1-kin tool `v8f_t1_eval.py`, in the form of `v8f_t1_eval_rule.py`.

Tier T1-kin (kinematic rollout on the model's own actions, logged frames); a diagnostic, never driving performance.

  python v8f_t1_eval_imagplan.py --tree <repo root> [--imag-plan {off,cand-imag,cand-cv,cand-label}] [--imag-log NPZ] [--cand-set 1.0,0.9,...,0.0]
         [--const-gap M] [--imag-shift-m X] [--floor {card,cap-only}] [--fallback {zero,none}] [--noise-after-tick SEED] [--action-def {logged,executed}] [--p45-log NPZ]  run <the argv of the pinned tool's `run`>

The pinned tool is imported from --tree by file path and is NOT edited.  Its module globals `model_planner`, `rollout`, `make_states` are replaced for the call of its own
`cmd_run` (exactly the seam of v8f_t1_eval_rule.py).  `--imag-plan off` (default) installs nothing = the pinned tool bit for bit.  Every open choice: IMAGPLAN_DEFS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import imag_plan as IP  # noqa: E402  (also puts ../distrule and ../imaghead on sys.path when they exist)
import dist_rule as R  # noqa: E402
import imag_dump as IH  # noqa: E402
import v8f_t1_eval_rule as W  # noqa: E402
import qhead_model as QM  # noqa: E402  (note ADD40_N7: the trained quantile head of card ADD42, one version, never a copy of its code)

PINNED_SHA256 = W.PINNED_SHA256
MODES = ("off",) + IP.SOURCES + ("cand-cvq",)      # cand-cvq: note ADD40_N7 (cand-cv with the gap min(g_head, q_head))
HEAD_P_MIN = W.HEAD_P_MIN
CHUNK = 256
P45_OFFSETS = (-2.0, -1.0, 1.0)      # m/s2: the acceleration offsets of the reader of card ADD36 (imag_dump.OFFSETS without 0), replayed in the p4/p5 log (note ADD40_N4)
P45_BASES = ("lp", "pl", "pp", "sm")  # baselines of the second diagnostic of note ADD40_N4 (see _log_p45)
LOG_SCALARS = ("ci", "t0", "j", "k", "tick", "active", "present", "src_bad", "g_src", "c_src", "p_src", "v0", "v_l", "v_L", "a_l", "a_m", "g0", "chosen_idx", "chosen_c", "any_safe",
               "n_lift", "n_nonfinite", "changed", "n_yaw_lim", "g_head", "q_head", "g_used")
LOG_VECTORS = ("safe", "gap_h", "v_h", "a_h", "imag_raw")      # [C] or [C, 2] per row


class IPLog:
    def __init__(self):
        self.rows = {k: [] for k in LOG_SCALARS + LOG_VECTORS}

    def add(self, **cols):
        for k in self.rows:
            self.rows[k].append(np.asarray(cols[k]))

    def arrays(self):
        return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in self.rows.items()}

    def save(self, path, meta):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **self.arrays(), **{k: np.asarray(v) for k, v in meta.items()})


def sha256_file(path):
    return W.sha256_file(path)


class ImagPlanner(W.RulePlanner):
    """The model planner followed by the IMAGPLAN rule.  The per-window tick state (v0, a_l, a_m, v_l, source values, gate) is the one of RulePlanner.__call__ (v8f_t1_eval_rule.py:189-213);
    a test asserts the values equal those RulePlanner logs on the same ticks."""

    def __init__(self, base, model, source, m, fk, T, label=None, log=None, cset=IP.CAND_CARD, floor="card", fallback="zero", const_gap=None, shift_m=0.0, p45=None, action_def="logged", family="scale", qhead=None, q_const=None):
        if family not in IP.FAMILIES:
            raise ValueError(f"family must be one of {IP.FAMILIES}")
        if source not in IP.SOURCES + ("cand-cvq",):
            raise ValueError(f"source must be one of {IP.SOURCES + ('cand-cvq',)}")
        if source != "cand-label" and getattr(getattr(model, "wm", None), "lead_head", None) is None:
            raise SystemExit(f"REFUSED: --imag-plan {source} needs a checkpoint trained with lead_sup on (no wm.lead_head in this model)")
        if source in IP.IMAG_SOURCES and int(model.wm.stack.cfg[0].action_dim) != 2:
            raise SystemExit(f"REFUSED: level-1 action width {model.wm.stack.cfg[0].action_dim}: cand-imag reads acc_yaw arms only (as imag_dump.py)")
        super().__init__(base, model, "label" if source == "cand-label" else "head", m, fk, label=label, log=W.RuleLog(), floor="card")
        self.q_on = source == "cand-cvq"
        if self.q_on and qhead is None:
            raise SystemExit("REFUSED: cand-cvq needs the quantile head file (IP_QHEAD_FILE)")
        if (not self.q_on) and (qhead is not None or q_const is not None):
            raise SystemExit("REFUSED: a quantile head or constant with a source other than cand-cvq would change nothing")
        self.qhead, self.q_const, self.q_dev = qhead, q_const, None
        self.cand_source, self.T = ("cand-cv" if self.q_on else source), T      # the rule of cand-cv, unchanged (note N7 section 2)
        self.cset, self.floor_c, self.fallback = tuple(cset), floor, fallback
        self.const_gap, self.shift_m, self.action_def = const_gap, float(shift_m), action_def
        self.family = family
        self.ilog = log if log is not None else IPLog()
        self.p45 = p45                      # None or a dict of lists: the p4 / p5 log
        self.torch = None
        if model is not None:
            import torch
            self.torch = torch

    # ---------------------------------------------------------------- model parts
    def _z0(self, batch, idx):
        """Level-1 state of the tick frame of rows `idx`: z1 of the last frame of the batch (the formula of `head_values`, per-frame, so independent of the window)."""
        torch, model = self.torch, self.model
        f = batch["feats"][torch.as_tensor(np.asarray(idx), dtype=torch.long, device=batch["feats"].device)][:, -1:]
        with torch.no_grad():
            return model.wm.z1(model.adapter(f.float()) + model.pos[None, None])[:, 0]

    def _roll_reads(self, z0, A_raw):
        """z0 [n, D], raw actions [n, 10, 2] -> head reads (logit, gap m, closing m/s) at 1.0 s, 2.0 s [n, 2, 3] as numpy float64: the tool of imag_dump.py (roll_states, head_reads_rolled)."""
        torch, model = self.torch, self.model
        M, stk = self.T.M, model.wm.stack
        ctx = IH.predictor_context(stk)
        out = []
        with torch.no_grad():
            A = model.wm.scale_actions(torch.as_tensor(np.asarray(A_raw, np.float32), device=z0.device))
            for i in range(0, z0.shape[0], CHUNK):
                R_ = IH.roll_states(torch, stk, z0[i:i + CHUNK].float(), A[i:i + CHUNK], ctx)
                out.append(IH.head_reads_rolled(torch, model, M, R_).double().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2, 3))

    def _imag_fn(self, batch, st, plan=None):
        def fn(cand, ia):
            Ba, C = cand.shape[:2]
            A = np.zeros((Ba * C, IP.N_ROLL, 2))
            ds = IP.effective_set(self.cset, self.fallback, self.family) if self.family == "decel" else None
            for k, b in enumerate(ia):
                base = None
                for ci in range(C):
                    if self.family == "decel":              # note ADD40_N3: the PLAN's actions with the acceleration channel of candidate d (the plan's rows are computed once per window)
                        if base is None:
                            act0, nl0 = IP.candidate_actions(self.T, st[int(b)], plan[int(b)], IP.N_ROLL, self.action_def, return_n=True)
                            base = (act0, nl0, np.asarray(IP.candidate_track_rows(self.T, st[int(b)], plan[int(b)], IP.N_ROLL)[0::2, 3], np.float64))
                        A[k * C + ci], n_yl = IP.decel_actions(self.T, st[int(b)], plan[int(b)], ds[ci], IP.N_ROLL, self.action_def, return_n=True, base=base)
                    else:
                        A[k * C + ci], n_yl = IP.candidate_actions(self.T, st[int(b)], cand[k, ci], IP.N_ROLL, self.action_def, return_n=True)
                    self._yl[int(b)] += n_yl                # steps of this tick whose yaw rate was limited (note ADD40_N2), summed over the candidates
            z0 = self._z0(batch, ia).repeat_interleave(C, 0)
            rd = self._roll_reads(z0, A)
            return rd[:, :, 1].reshape(Ba, C, 2)
        return fn

    def _read_q(self, feats, g_head):
        """q_head in metres for each row of the batch, read at the TICK frame: the quantile head on z1 of the last `hist` frames of the batch (the expression of the banked head read);
        or the constant IP_QHEAD_CONST (controls q1, q2) in place of the head output."""
        g = np.asarray(g_head)
        if self.q_const is not None:
            return np.full(g.shape, float(self.q_const), np.float32)
        hist = int(self.model.wm.lead_head.hist)
        if self.qhead.hist != hist:
            raise SystemExit("REFUSED: the quantile head reads %d frames, the banked head %d" % (self.qhead.hist, hist))
        dev = feats.device
        if self.q_dev != str(dev):
            self.qhead = self.qhead.to(dev).eval()
            self.q_dev = str(dev)
        with self.torch.no_grad():
            f = feats[:, -hist:]
            z1 = self.model.wm.z1(self.model.adapter(f.float()) + self.model.pos[None, None])
            o = self.qhead(z1)[:, 0, 0].float().cpu().numpy()
        return (o * QM.GAP_SCALE_M).astype(np.float32)

    # ---------------------------------------------------------------- p4 / p5 (tick 0 only, every window, any gate)
    def _log_p45(self, batch, st, plan=None):
        T, B = self.T, len(st)
        a_gt, a_gt_exec, a_cache, a_track = (np.full((B, IP.N_ROLL, 2), np.nan) for _ in range(4))
        a_plan = np.full((B, IP.N_ROLL, 2), np.nan)
        gap_logged = np.full((B, 2), np.nan)
        gap_off = np.full((B, len(P45_OFFSETS), 2), np.nan)
        hyb = {nm: np.full((B, 1 + len(P45_OFFSETS), 2), np.nan) for nm in P45_BASES}
        for b, s in enumerate(st):
            j, n = int(s.j), IP.N_ROLL
            gt = np.asarray(s.d["gt"][s.t0], np.float64)                                              # the logged future path as the plan, c = 1.0 (the pinned `gt` of the window)
            a_gt[b] = IP.candidate_actions(T, s, gt, n, self.action_def)
            a_gt_exec[b] = IP.candidate_actions(T, s, gt, n, "executed")
            if plan is not None:
                a_plan[b] = IP.candidate_actions(T, s, plan[b], n, self.action_def)                   # the model's own plan as actions (the d = 0 candidate of the decel family)
            if j + n <= len(s.logged_a):
                a_cache[b] = np.stack([np.nan_to_num(s.logged_a[j:j + n]), np.nan_to_num(s.logged_ch1[j:j + n])], -1)
                a_track[b] = np.nan_to_num(s.logged_chan[j:j + n, 1:3])
        if self.cand_source in IP.IMAG_SOURCES:
            ok = np.isfinite(a_cache).all((1, 2))
            if ok.any():
                idx = np.nonzero(ok)[0]
                rd = self._roll_reads(self._z0(batch, idx), a_cache[idx])
                gap_logged[idx] = rd[:, :, 1]
                # note ADD40_N4 (added by the lead): the stimulus of the reader of card ADD36 through THIS tool's state path -- the logged actions with a constant
                # acceleration offset (m/s2) on all ten steps, yaw channel untouched. Raw offset d = d / WM_ACC_SCALE after scale_actions, as imag_dump.offset_actions.
                for oi, dd in enumerate(P45_OFFSETS):
                    a_off = np.array(a_cache[idx], np.float64, copy=True)
                    a_off[:, :, 0] += dd
                    gap_off[idx, oi] = self._roll_reads(self._z0(batch, idx), a_off)[:, :, 1]
            # note ADD40_N4, second diagnostic (added by the lead): which part of the PLAN's action sequence changes the response? The same offsets on four other baselines:
            # lp = logged acceleration + the plan's yaw channel, pl = the plan's acceleration + logged yaw, pp = the plan's actions, sm = the plan's actions with both channels
            # replaced by their mean over the ten steps. Index 0 = the baseline itself, 1.. = the offsets of P45_OFFSETS. Tick 0, every window; changes no decision of the tool.
            okp = ok & np.isfinite(a_plan).all((1, 2))
            if okp.any():
                ip = np.nonzero(okp)[0]
                zp = self._z0(batch, ip)
                L_, P_ = np.asarray(a_cache[ip], np.float64), np.asarray(a_plan[ip], np.float64)
                bases = {"lp": np.stack([L_[..., 0], P_[..., 1]], -1), "pl": np.stack([P_[..., 0], L_[..., 1]], -1), "pp": P_,
                         "sm": np.array(np.broadcast_to(P_.mean(1, keepdims=True), P_.shape))}
                for nm in P45_BASES:
                    A0 = bases[nm]
                    hyb[nm][ip, 0] = self._roll_reads(zp, A0)[:, :, 1]
                    for oi, dd in enumerate(P45_OFFSETS):
                        A1 = np.array(A0, np.float64, copy=True)
                        A1[:, :, 0] += dd
                        hyb[nm][ip, 1 + oi] = self._roll_reads(zp, A1)[:, :, 1]
        rec = dict(ci=[s.ci for s in st], t0=[s.t0 for s in st], act_gt=a_gt, act_gt_exec=a_gt_exec, act_cache=a_cache, act_track=a_track, imag_gap_logged=gap_logged,
                   imag_gap_logged_off=gap_off, act_plan=a_plan)
        rec.update({"imag_gap_hyb_" + nm: hyb[nm] for nm in P45_BASES})
        for k, v in rec.items():
            self.p45.setdefault(k, []).append(np.asarray(v))

    # ---------------------------------------------------------------- the tick
    def __call__(self, batch):
        plan = self.base(batch)
        st = self.states
        if st is None or len(st) != plan.shape[0]:
            raise RuntimeError("ImagPlanner not bound to the states of this batch (rollout shim missing)")
        B = len(st)
        v0 = batch["p10f"][:, -1, 0].double().cpu().numpy()
        ci = np.array([s.ci for s in st], np.int64)
        t0 = np.array([s.t0 for s in st], np.int64)
        j = np.array([s.j for s in st], np.int64)
        k = j - t0
        g_log = np.stack([np.asarray(s.d["gt"][s.t0], np.float64) for s in st])
        pos = np.stack([self.fk.gt_waypoints(s.track[0::2], np.array([s.t0]), np.array([int(kk)]))[0, 0] for s, kk in zip(st, k)])[:, None, :]
        a_m = np.where(k > 0, R.progress(g_log, pos)[:, 0], 0.0)
        a_l = np.where(k > 0, R.logged_arc(g_log)[np.arange(B), np.clip(k - 1, 0, 29)], 0.0)
        v_l = np.array([s.logged_v[int(s.j)] if int(s.j) < len(s.logged_v) else np.nan for s in st], np.float64)
        p_src = np.full(B, np.nan)
        g_gate, gh_c, qh_c = None, np.full(B, np.nan), np.full(B, np.nan)
        if self.source == "label":
            rd = [self.label.read(int(c), int(jj)) for c, jj in zip(ci, j)]
            present = np.array([r[0] for r in rd], bool)
            g_src = np.array([r[1] for r in rd], np.float64)
            c_src = np.array([r[2] for r in rd], np.float64)
        else:
            p, g_h, c_h, ok = W.head_values(self.model, batch["feats"], batch["frame_ok"])
            gh_c = np.asarray(g_h, np.float64)
            if self.q_on:
                q_h = self._read_q(batch["feats"], g_h)
                g_h = QM.used_gap(g_h, q_h)                      # g_used = min(g_head, q_head)
                g_gate, qh_c = gh_c, np.asarray(q_h, np.float64)  # the gate (gap at most 60 m) stays on the head gap: the active set is that of cand-cv (note N7 control q5)
            p_src, g_src, c_src = p.astype(np.float64), g_h.astype(np.float64), c_h.astype(np.float64)
            present = (p >= HEAD_P_MIN) & ok
        with np.errstate(invalid="ignore"):
            src_bad = present & ~(np.isfinite(g_src) & np.isfinite(c_src) & np.isfinite(v_l))
            present = present & ~src_bad
            v_L = np.maximum(0.0, v_l + c_src)
            g0 = g_src + a_l - a_m
            active = present & np.isfinite(g_src) & np.isfinite(g0) & np.isfinite(v_L) & np.isfinite(v0) & ((g_src if g_gate is None else g_gate) <= IP.G_MAX_M)
        if self.p45 is not None and bool((k == 0).all()):
            self._log_p45(batch, st, plan)
        self._yl = np.zeros(B, np.int64)
        imag_fn = self._imag_fn(batch, st, plan) if (self.cand_source in IP.IMAG_SOURCES and active.any() and self.const_gap is None) else None
        out = IP.imag_plan_rule(plan, v0, active, g_src, a_l - a_m, v_L, self.cand_source, self.cset, self.floor_c, self.fallback, imag_fn, self.const_gap, self.shift_m, family=self.family)
        self.ilog.add(ci=ci, t0=t0, j=j, k=k, tick=k // max(self.m, 1), active=active, present=present, src_bad=src_bad, g_src=g_src, c_src=c_src, p_src=p_src, v0=v0, v_l=v_l, v_L=v_L,
                      a_l=a_l, a_m=a_m, g0=g0, chosen_idx=out.chosen_idx, chosen_c=out.chosen_c, any_safe=out.any_safe, n_lift=out.n_lift, n_nonfinite=out.n_nonfinite, changed=out.changed,
                      n_yaw_lim=self._yl.copy(), g_head=gh_c, q_head=qh_c, g_used=g_src, safe=out.safe, gap_h=out.gap_h, v_h=out.v_h, a_h=out.a_h, imag_raw=out.imag_raw)
        return out.plan


# ============================================================================================ installing the seam
def apply_noise_after_tick(states, seed, m, H=30):
    """Control p3: every camera token of a window after its LAST tick frame (t0 + H - m) is replaced by noise (the form of the DISTRULE r4 test).  Only copies the window's `feats`."""
    rng = np.random.default_rng(int(seed))
    for s in states:
        f = np.array(s.d["feats"], copy=True)
        lo = int(s.t0) + H - int(m) + 1
        if lo < len(f):
            f[lo:] = rng.normal(0.0, 1.0, f[lo:].shape).astype(f.dtype)
        s.d["feats"] = f
    return states


def attach_logged_actions(T, states, store, tracks, J, want_chan):
    for s in states:
        base = store.get(str(J["clip"][s.ci]))
        s.logged_a = np.asarray(base["a"], np.float64)
        s.logged_ch1 = np.asarray(base["ch1"], np.float64)
        if want_chan:
            s.logged_chan = T.D.ego_channels_native(tracks[str(J["clip"][s.ci])])[0::2]
    return states


def load_qhead_env(mode, ckpt, env=None):
    """Note ADD40_N7: IP_QHEAD_FILE (the head file; required for cand-cvq, refused for any other mode) and IP_QHEAD_CONST (metres; replaces the head OUTPUT, controls q1 = 1e9, q2 = 0).
    Checks of the rule wrapper v8f_t1_eval_rule_qhead.py: not an evaluation-split path, fit record next to the file, checkpoint sha256 of the record equals the run's, head sha256 in the
    record, and (note N7 section 2) coverage on the check clips in [0.15, 0.25] (h5 true) -- else the head is not read.  Returns (head or None, const or None, meta dict)."""
    env = os.environ if env is None else env
    f, c = env.get("IP_QHEAD_FILE", ""), env.get("IP_QHEAD_CONST", "")
    if mode != "cand-cvq":
        if f or c != "":
            raise SystemExit("REFUSED: IP_QHEAD_FILE / IP_QHEAD_CONST with mode %s would change nothing" % mode)
        return None, None, {}
    if not f:
        raise SystemExit("REFUSED: cand-cvq needs IP_QHEAD_FILE")
    if not os.path.isfile(f):
        raise SystemExit("REFUSED: no head file " + f)
    if QM.is_eval_path(f):
        raise SystemExit("REFUSED: a head file of the evaluation split: " + f)
    rec_p = f[:-4] + ".json" if f.endswith(".npz") else f + ".json"
    if not os.path.isfile(rec_p):
        raise SystemExit("REFUSED: no fit record " + rec_p)
    rec = json.load(open(rec_p, encoding="utf-8"))
    sha_head = QM.sha256_file(f)
    if sha_head not in (rec.get("head_file_sha256") or {}).values():
        raise SystemExit("REFUSED: the head file does not match the sha256 of its fit record")
    if rec.get("h5_check_coverage_in_0.15_0.25") is not True:
        raise SystemExit("REFUSED: the fit record does not say check coverage inside [0.15, 0.25]: the head is not read")
    ck_rec = (rec.get("states_record") or {}).get("ckpt_sha256")
    if not ck_rec:
        raise SystemExit("REFUSED: the fit record has no checkpoint sha256")
    if ckpt and os.path.isfile(ckpt) and QM.sha256_file(ckpt) != ck_rec:
        raise SystemExit("REFUSED: the head was fitted on states of checkpoint %s, this run is another" % ck_rec)
    const = float(c) if c != "" else None
    return QM.load_head(f), const, {"qhead_file": f, "qhead_sha256": sha_head, "qhead_fit_record": rec_p, "qhead_const": np.nan if const is None else const,
                                    "qhead_model_sha256": QM.sha256_file(QM.__file__)}


def install(T, source, a, opts, J=None, log=None, p45=None):
    """Replace model_planner / rollout / make_states of the pinned module T for one call of T.cmd_run.  Returns (planners list, restore function)."""
    orig = {k: getattr(T, k) for k in ("model_planner", "rollout", "make_states")}
    made = []
    label = W.LabelSource(J) if source == "cand-label" else None
    if source != "cand-label":
        assert abs(T.M.LEAD_SUP_GAP_SCALE_M - W.LEAD_GAP_SCALE_M) == 0 and abs(T.M.LEAD_SUP_CLOS_SCALE_MS - W.LEAD_CLOS_SCALE_MS) == 0

    def model_planner(model, device, keep_aux=False, keep_heads=False):
        base = orig["model_planner"](model, device, keep_aux=keep_aux, keep_heads=keep_heads)
        rp = ImagPlanner(base, model, source, a.replan_steps, T.fk, T, label=label, log=log, cset=opts["cset"], floor=opts["floor"], fallback=opts["fallback"],
                         const_gap=opts["const_gap"], shift_m=opts["shift_m"], p45=p45, action_def=opts["action_def"], family=opts["family"], qhead=opts["qhead"], q_const=opts["q_const"])
        made.append(rp)
        return rp

    def make_states(store, tracks, J_, wins, nav_mode):
        out = orig["make_states"](store, tracks, J_, wins, nav_mode)
        W.attach_logged(out, store, J_)
        attach_logged_actions(T, out, store, tracks, J_, p45 is not None)
        if opts["noise_seed"] >= 0:
            apply_noise_after_tick(out, opts["noise_seed"], a.replan_steps)
        return out

    def rollout(planner, states, *args, **kw):
        if isinstance(planner, ImagPlanner):
            planner.bind(states)
        return orig["rollout"](planner, states, *args, **kw)

    T.model_planner, T.rollout, T.make_states = model_planner, rollout, make_states

    def restore():
        for k, v in orig.items():
            setattr(T, k, v)
    return made, restore


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--imag-plan", choices=MODES, default="off")
    ap.add_argument("--imag-log", default="")
    ap.add_argument("--cand-family", choices=IP.FAMILIES, default="scale", help="scale = speed-scaled candidates (card ADD40); decel = the plan under a constant deceleration d (note ADD40_N3)")
    ap.add_argument("--cand-set", default=None, help="comma list: scale: c in [0,1] (default CAND_CARD; 1.0 alone = control p1); decel: d >= 0 m/s2 (default DECEL_CARD)")
    ap.add_argument("--const-gap", type=float, default=None, help="every forecast gap is this many metres after the gate (control p1: 100)")
    ap.add_argument("--imag-shift-m", type=float, default=0.0, help="added to every imagined gap (control p6: -20); cand-imag only")
    ap.add_argument("--floor", choices=IP.FLOORS, default="card", help="card = max(c a_P, a_B) as written; cap-only = min(a_P, .) never above the plan (IMAGPLAN_DEFS E1, not the card)")
    ap.add_argument("--fallback", choices=IP.FALLBACKS, default="zero", help="zero = c 0.0 when no candidate is safe (card); none = the plan unchanged (identity probe p1 only)")
    ap.add_argument("--action-def", choices=IP.ACTION_DEFS, default="logged", help="logged = the corpus' definition of the logged actions (default); executed = the pinned tool's own write (IMAGPLAN_DEFS A2)")
    ap.add_argument("--noise-after-tick", type=int, default=-1, help="control p3: SEED (>= 0) replaces every token after the window's last tick by noise")
    ap.add_argument("--p45-log", default="", help="NPZ with the tick-0 action sequences (p4) and the imagined gap under logged actions (p5) of every window")
    mine, rest = ap.parse_known_args(argv)
    T, path = W.load_pinned(mine.tree)
    a = T.build_parser().parse_args(rest)
    if mine.cand_family == "decel":
        cset = IP.parse_decel_set(IP.DECEL_CARD if mine.cand_set is None else mine.cand_set)
        if mine.floor != "card":
            raise SystemExit("REFUSED: --floor applies to the scale family only (a decel candidate is never ahead of the plan: nothing to lift)")
    else:
        cset = IP.parse_cand_set(IP.CAND_CARD if mine.cand_set is None else mine.cand_set)
    if a.mode != "run":
        if mine.imag_plan != "off":
            raise SystemExit("REFUSED: --imag-plan applies to the `run` mode of the pinned tool only")
        return T.main(rest)
    qh, qc, qmeta = load_qhead_env(mine.imag_plan, a.ckpt)
    if mine.imag_plan == "off":
        extras = [n for n, bad in (("--imag-log", mine.imag_log), ("--const-gap", mine.const_gap is not None), ("--imag-shift-m", mine.imag_shift_m != 0.0), ("--noise-after-tick", mine.noise_after_tick >= 0),
                                   ("--p45-log", mine.p45_log), ("--floor", mine.floor != "card"), ("--fallback", mine.fallback != "zero"), ("--action-def", mine.action_def != "logged"),
                                   ("--cand-set", mine.cand_family == "scale" and cset != IP.CAND_CARD), ("--cand-family", mine.cand_family != "scale")) if bad]
        if extras:
            raise SystemExit(f"REFUSED: {', '.join(extras)} with --imag-plan off would change nothing (a flag that silently does nothing is worse than none)")
        return T.cmd_run(a)
    if mine.imag_shift_m != 0.0 and mine.imag_plan not in IP.IMAG_SOURCES:
        raise SystemExit("REFUSED: --imag-shift-m shifts the imagined gap: cand-imag and cand-imagkin only")
    pin = sha256_file(path)
    if pin != PINNED_SHA256:
        print(f"WARNING: pinned tool sha256 {pin} != {PINNED_SHA256}", file=sys.stderr, flush=True)
    J = T.D.load_join(a.eval_join) if mine.imag_plan == "cand-label" else None
    log = IPLog()
    p45 = {} if mine.p45_log else None
    opts = {"cset": cset, "floor": mine.floor, "fallback": mine.fallback, "action_def": mine.action_def, "const_gap": mine.const_gap, "shift_m": mine.imag_shift_m, "noise_seed": mine.noise_after_tick, "family": mine.cand_family, "qhead": qh, "q_const": qc}
    made, restore = install(T, mine.imag_plan, a, opts, J, log, p45)
    try:
        rc = T.cmd_run(a)
    finally:
        restore()
        if mine.imag_log:
            meta = {"mode": mine.imag_plan, "D": IP.D, "TAU": IP.TAU, "B_MAX": IP.B_MAX, "G_MAX": IP.G_MAX_M, "family": mine.cand_family, "cand_set": np.asarray(IP.effective_set(cset, mine.fallback, mine.cand_family)), "cand_set_arg": np.asarray(cset),
                    "knot_h": np.asarray(IP.KNOT_H), "replan_steps": a.replan_steps, "floor": mine.floor, "fallback": mine.fallback, "action_def": mine.action_def, "const_gap": np.nan if mine.const_gap is None else mine.const_gap,
                    "imag_shift_m": mine.imag_shift_m, "noise_after_tick": mine.noise_after_tick, **qmeta, "pinned_path": path, "pinned_sha256": pin, "ckpt": a.ckpt, "tag": a.tag,
                    "eval_join": a.eval_join, "join_sha256": sha256_file(a.eval_join) if os.path.isfile(a.eval_join) else "",
                    "imag_plan_sha256": sha256_file(IP.__file__), "wrapper_sha256": sha256_file(os.path.abspath(__file__)), "dist_rule_sha256": sha256_file(R.__file__),
                    "imag_dump_sha256": sha256_file(IH.__file__), "rule_wrapper_sha256": sha256_file(W.__file__), "argv": json.dumps(list(argv if argv is not None else sys.argv[1:]))}
            log.save(mine.imag_log, meta)
            print("wrote", mine.imag_log, flush=True)
        if mine.p45_log and p45:
            os.makedirs(os.path.dirname(os.path.abspath(mine.p45_log)), exist_ok=True)
            np.savez(mine.p45_log, **{k: np.concatenate(v) for k, v in p45.items()}, ckpt=a.ckpt, pinned_sha256=pin)
            print("wrote", mine.p45_log, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
