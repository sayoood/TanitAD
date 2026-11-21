"""AMEND_HWMPLAN_4 (L2-IMG), study-local shadow around HWMPlanner.plan_batch (MM ruling M1 = option (b)): the code of record is IMPORTED unchanged and is never written.
Modes (flag --l2-image):
  off    : the code of record. plan_unit() calls planner.plan_windows itself; nothing is installed. (C-OFF checks it bit for bit.)
  prim   : level 2's optimisation variable is the primitive sequence p (B, S, h2*c2 = 30, 2) in the LEVEL-1 prior box; its macro is M2(p) = macro_enc_2(act_embed(p)), the frozen encoders' own path
           (HJEPAStack.encode). Level 3 and level 1 are the library's. Knots 11-30 are the chosen level-2 p itself (no decode).
  prim23 : prim, and level 3's variable is the same 30 primitives with macro M3(M2(p)).
Level 1 sub-goal source (flag --l1-sub): own (the library's) or donor (RS, the deliberate regression: level 1 plans against the level-2 rollout of a donor window, one clip-disjoint cyclic shift per batch).
Constants of AMEND_HWMPLAN_4 section 2.2, fixed there: candidate 0 = HOLD_c[:, :30]; candidates 1.. = HOLD_c + prior_std_1 * N(0, 1) clipped to the level-1 box; N from a SEPARATE generator
manual_seed(2**40 + seed*100003 + batch) at level 2 and 2**41 + ... at level 3 (the planner's own generator is consumed exactly as under off); lr 0.1 (the level-1 value); the level-2/3 prior box is not applied.
Every shadow is an INSTANCE attribute on the stack, scoped to one call, restored (to the previous attribute or removed) on return or on raise.
Test-only mutation hooks (`mutate=`) exist for the deliberate-regression arms of the controls; the run wrapper refuses them.
Tier T0 (oracle goal): a world-model diagnostic, never driving performance."""
import contextlib, hashlib, math, sys
sys.dont_write_bytecode = True
import numpy as np
import torch
import hwm_plan as H
import plantest_lib as P

MODES = ("off", "prim", "prim23")
L1SUB = ("own", "donor")
MUTATIONS = (None, "no_shadow", "decode_knots", "noise_from_planner_gen", "frozen_p")
DEFAULT_MODE = "off"                          # the mutation of C-OFF sets this to "prim"; resolve_mode(None) must read the code of record
_MISSING = object()
IMG_TOL = 1e-6                                # AMEND_HWMPLAN_4 section 8.2 C-PRIM-ID


def resolve_mode(mode):
    mode = DEFAULT_MODE if mode is None else mode
    if mode not in MODES:
        raise ValueError(f"--l2-image must be one of {MODES}, got {mode!r}")
    return mode


def M2(st, geom, p):
    """p (..., n, 2) scaled primitives -> level-2 macros (..., n // c2, E2): the grouping of HJEPAStack.encode (hjepa.py:669-700)."""
    c2 = geom.c2
    e = st.act_embed(p)
    return st.macro_encoders[0](e.reshape(*p.shape[:-2], p.shape[-2] // c2, c2, e.shape[-1]))


def M3(st, m2):
    """level-2 macros (..., n2, E2) -> level-3 macros (..., n2 // s3, E3)."""
    s3 = int(st.cfg[2].stride)
    return st.macro_encoders[1](m2.reshape(*m2.shape[:-2], m2.shape[-2] // s3, s3, m2.shape[-1]))


def gen_hash(g):
    return hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest()[:16]


def donor_shift(eids):
    """The smallest d >= 1 with eids[(i + d) % B] != eids[i] for every i (a bijection inside the batch). A batch with none is refused (MM ruling M3)."""
    B = len(eids)
    for d in range(1, B):
        if all(eids[(i + d) % B] != eids[i] for i in range(B)):
            return (np.arange(B) + d) % B, d
    raise RuntimeError("no clip-disjoint cyclic shift in this batch: the RS donor rule refuses it")


@contextlib.contextmanager
def instance_attrs(obj, **attrs):
    """Instance-level attributes on obj for the duration of the block; the previous instance attribute is restored (or removed) on exit, on return or on raise."""
    saved = {}
    try:
        for k, v in attrs.items():
            saved[k] = obj.__dict__.get(k, _MISSING)
            obj.__dict__[k] = v
        yield
    finally:
        for k, old in saved.items():
            if old is _MISSING:
                obj.__dict__.pop(k, None)
            else:
                obj.__dict__[k] = old


class PrimShadow:
    """Replaces st._optimize for ONE hier plan call. The library visits the levels top down (3, 2, 1); the call of a level in `levels` optimises primitives and returns the macro of the best iterate."""

    def __init__(self, st, geom, hc, seed, batch, levels=(2,), mutate=None, planner_gen=None, probe=None):
        self.st, self.geom, self.hc, self.levels, self.mutate, self.planner_gen = st, geom, hc, tuple(levels), mutate, planner_gen
        self.n, self.store, self.probe = 0, {}, probe
        self.np_ = geom.h2 * geom.c2
        self.gens = {2: torch.Generator().manual_seed(2 ** 40 + int(seed) * 100003 + int(batch)), 3: torch.Generator().manual_seed(2 ** 41 + int(seed) * 100003 + int(batch))}
        self.orig = None

    def macro_fn(self, level):
        st, g = self.st, self.geom
        return (lambda p: M2(st, g, p)) if level == 2 else (lambda p: M3(st, M2(st, g, p)))

    def opt(self, cost_fn, init, **kw):
        self.n += 1
        if self.n > 3:
            raise RuntimeError("more than three _optimize calls in one hier plan: the level order (3, 2, 1) no longer holds")
        level = (3, 2, 1)[self.n - 1]
        if level not in self.levels:
            return self.orig(cost_fn, init, **kw)
        st = self.st
        lo1, hi1 = st.prior_lo_1, st.prior_hi_1
        b, S = init.shape[:2]
        sd = self.hc[:, :self.np_].to(init.dtype)
        gen = self.planner_gen if self.mutate == "noise_from_planner_gen" else self.gens[level]
        noise = torch.randn(b, S, self.np_, 2, generator=gen, dtype=init.dtype)
        p0 = sd.unsqueeze(1) + st.prior_std_1.to(init.dtype) * noise
        p0[:, 0] = sd
        p0 = torch.max(torch.min(p0, hi1), lo1)
        kw2 = dict(kw)
        kw2.update(lr=float(P.LR[1]), lo=lo1, hi=hi1, project=None)
        mp = self.macro_fn(level)
        if self.mutate == "frozen_p":                                       # deliberate regression of C-REACH: the cost does not depend on p (gradient exactly 0)
            def cost_p(p):
                return cost_fn(mp(p).detach() + 0.0 * p.sum())
        else:
            def cost_p(p):
                return cost_fn(mp(p))
        best_p, best_cost, init_cost, hist, nit = self.orig(cost_p, p0, **kw2)
        with torch.no_grad():
            best_m = mp(best_p)
        self.store[level] = dict(best_p=best_p.detach(), best_cost=best_cost.detach(), init_p=p0.detach().clone())
        if self.probe is not None:
            self.probe[level] = dict(cost_fn=cost_fn, mp=mp, init_p=p0.detach().clone(), best_p=best_p.detach(), best_cost=best_cost.detach())
        return best_m, best_cost, init_cost, hist, nit

    def chosen_p(self, level):
        s = self.store[level]
        ar = torch.arange(s["best_p"].shape[0])
        return s["best_p"][ar, s["best_cost"].argmin(1)]


def _check_plain_config(planner):
    g, s = planner.geom, planner.solver
    ok = (s.seed_mode == "hold" and float(s.lam) == 0.0 and planner.vocab is None and planner.constraints is None
          and all(int(c) == 1 for c in H.ctx_tuple(s.plan_ctx)) and g.stitch_mode == "decode")
    if not ok:
        raise RuntimeError(f"l2img needs the LON-PLAN levers (hold seed, lam 0, plan_ctx 1, vocab off, constr 0, stitch decode): {s} {g}")
    assert geom_ok(g), g


def geom_ok(g):
    return g.h2 * g.c2 >= g.n_traj and g.n_op <= g.n_traj


def _scale_back(wm, p_raw):
    out = np.array(p_raw, np.float64)
    out[..., 0] /= H.ACC_SCALE
    out[..., 1] /= float(wm.ch1_scale)
    return torch.from_numpy(out).float()


def image_residual(st, geom, wm, p_raw, macro, level):
    """FRESH recompute: ||M(p) - macro|| / ||macro|| per window (level 2: M2; level 3: M3 o M2), p in raw units as banked."""
    with torch.no_grad():
        p = _scale_back(wm, p_raw)
        mm = M2(st, geom, p)
        if level == 3:
            mm = M3(st, mm)
        macro = torch.as_tensor(macro)
        sc = macro.pow(2).sum(-1).mean(-1).clamp_min(1e-12)
        return ((mm - macro).pow(2).sum(-1).mean(-1) / sc).sqrt().numpy()


def plan_unit(planner, mode, goal_fn, z_now, A_raw0, idx, seed, batch0, eids=None, l1_sub="own", mutate=None, record=False, probe=None):
    """One batch of windows idx (<= P.BATCH) of one arm. Returns the dict of HWMPlanner.plan_windows (plan_raw, cost1, iters1, idx, a1_raw, decode_resid) and, for prim / prim23, the extras banked in the plan file:
    l2_prim_raw (n, 30, 2), cost2, cost3, iters2, iters3, image_resid (level 2), image_resid3 (level 3; nan under prim), [l3_prim_raw under prim23], gen_hash (1,), and m2_b0 / m3_b0 for batch 0.
    mode off with l1_sub own and nothing requested is planner.plan_windows verbatim. record=True adds the batch's chosen macros / costs / generator hash from a passive wrapper of plan_batch (no value changes)."""
    mode = resolve_mode(mode)
    if mutate not in MUTATIONS:
        raise ValueError(mutate)
    if l1_sub not in L1SUB:
        raise ValueError(l1_sub)
    if l1_sub == "donor" and mode == "off":
        raise ValueError("--l1-sub donor is the regression of the prim arm: it needs --l2-image prim")
    idx = np.asarray(idx)
    if len(idx) > P.BATCH:
        raise ValueError(f"one batch at a time (<= {P.BATCH} windows), got {len(idx)}")
    g = planner.geom
    if mode == "off" and not record and mutate is None:
        return planner.plan_windows("hier", goal_fn, z_now, A_raw0, idx=idx, seed=seed, batch0=batch0)
    cap = {}
    if mode == "off" or mutate == "no_shadow":
        orig_pb = planner.plan_batch

        def pb(m, zn, gl, hc, gen, **kw):
            r = orig_pb(m, zn, gl, hc, gen, **kw)
            cap.update(res=r, gen=gen_hash(gen))
            return r
        with instance_attrs(planner, plan_batch=pb):
            r = planner.plan_windows("hier", goal_fn, z_now, A_raw0, idx=idx, seed=seed, batch0=batch0)
        res = cap["res"]
        extras = dict(cost2=res.costs[2].numpy().astype(np.float64), cost3=res.costs[3].numpy().astype(np.float64),
                      iters2=np.full(len(idx), float(res.iters[2])), iters3=np.full(len(idx), float(res.iters[3])), gen_hash=np.array([cap["gen"]]))
        if batch0 == 0:
            extras.update(m2_b0=res.level_actions[2].detach().numpy(), m3_b0=res.level_actions[3].detach().numpy())
        if mutate == "no_shadow":                                           # mutation (a): flag reads prim, the shadow is not installed: the "primitives" are the decoded knots
            extras.update(l2_prim_raw=r["plan_raw"].copy(), image_resid=np.zeros(len(idx)))
        r.update(extras)
        return r
    # ---- prim / prim23 (and rs): the library's plan_batch under the shadow
    _check_plain_config(planner)
    wm, st = planner.wm, planner.st
    hc = planner.hold_c(A_raw0)[idx]
    gen = P.make_gen(seed, batch0)
    levels = (2, 3) if mode == "prim23" else (2,)
    sh = PrimShadow(st, g, hc, seed, batch0, levels=levels, mutate=mutate, planner_gen=gen, probe=probe)
    attrs = {}
    if l1_sub == "donor":
        dn, d = donor_shift(np.asarray(eids)[idx])
        dn = torch.as_tensor(dn)
        o_roll = st.rollout

        def roll(level, z0, actions, *, context=1):
            rr = o_roll(level, z0, actions, context=context)
            return rr[dn] if (level == 2 and actions.dim() == 3) else rr
        attrs["rollout"] = roll
    sh.orig = st._optimize
    attrs["_optimize"] = sh.opt
    with instance_attrs(st, **attrs):
        res = planner.plan_batch("hier", z_now[idx], goal_fn(idx), hc, gen, goal_level=int(getattr(goal_fn, "level", 1)), ii=idx)
    if sh.n != 3:
        raise RuntimeError(f"the solver made {sh.n} _optimize calls, expected 3")
    a1 = res.actions.detach()
    m2 = res.level_actions[2].detach()
    m3 = res.level_actions[3].detach()
    pc = sh.chosen_p(2)                                                      # (B, 30, 2): the chosen level-2 primitives
    traj = torch.zeros(len(idx), g.n_traj, 2)
    traj[:, :g.n_op] = a1[:, :g.n_op]
    if mutate == "decode_knots":                                             # mutation (b): knots 11-30 from the decode, the identity breaks
        lo, hi = planner.level1_box()
        traj, _ = H.stitch(st, g, a1, m2, hc, lo, hi, H.Q.prior_std_orig(st), planner.solver)
    else:
        traj[:, g.n_op:] = pc[:, g.n_op:g.n_traj]
    with torch.no_grad():
        sc2 = m2.pow(2).sum(-1).mean(-1).clamp_min(1e-12)
        ir2 = ((M2(st, g, pc) - m2).pow(2).sum(-1).mean(-1) / sc2).sqrt().numpy()
        ir3 = np.full(len(idx), np.nan)
        if mode == "prim23":
            p3 = sh.chosen_p(3)
            sc3 = m3.pow(2).sum(-1).mean(-1).clamp_min(1e-12)
            ir3 = ((M3(st, M2(st, g, p3)) - m3).pow(2).sum(-1).mean(-1) / sc3).sqrt().numpy()
    out = dict(plan_raw=H.to_raw(wm, traj.numpy().astype(np.float64)), cost1=res.costs[1].numpy().astype(np.float64), iters1=np.full(len(idx), float(res.iters[1])), idx=idx,
               a1_raw=H.to_raw(wm, a1.numpy().astype(np.float64)), l2_prim_raw=H.to_raw(wm, pc.numpy().astype(np.float64)),
               cost2=res.costs[2].numpy().astype(np.float64), cost3=res.costs[3].numpy().astype(np.float64), iters2=np.full(len(idx), float(res.iters[2])), iters3=np.full(len(idx), float(res.iters[3])),
               image_resid=ir2, image_resid3=ir3, gen_hash=np.array([gen_hash(gen)]))
    if mode == "prim23":
        out["l3_prim_raw"] = H.to_raw(wm, sh.chosen_p(3).numpy().astype(np.float64))
    if batch0 == 0:
        out.update(m2_b0=m2.numpy(), m3_b0=m3.numpy())
    return out


MERGE_FIRST = ("m2_b0", "m3_b0")


def merge_units(parts):
    """parts: [(batch0, dict)] -> one dict, as run_hwm.cmd_plan merges (concatenate along windows; wall_s summed); the *_b0 arrays come from batch 0 only."""
    parts = sorted(parts, key=lambda x: x[0])
    keys = [k for k in parts[0][1] if k != "wall_s"]
    out = {}
    for k in keys:
        if k in MERGE_FIRST:
            out[k] = parts[0][1][k]
        else:
            out[k] = np.concatenate([np.atleast_1d(r[k]) for _, r in parts])
    out["wall_s"] = np.array(sum(float(r["wall_s"]) for _, r in parts))
    return out
