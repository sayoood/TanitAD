"""HWM-PLAN: the multi-hierarchy world-model planner (SPEC_HWM section 4), as an evaluation harness. Zero training. CPU.
Top-down at each replan: L3 plans macro-actions to a goal and hands subgoals to L2; L2 plans to 6 s and hands subgoals to L1; L1 plans primitive actions to 2 s.
The output is ONE 6 s trajectory: knots 1-10 are L1's actions, knots 11-30 are L2's macro-actions decoded to primitive actions; a kinematic model with friction and speed bounds integrates it.
Strides and levels are read from the BUILT model (HJEPAStack.cfg / WMAux.built_strides), so the same code runs on M2b (2,2) and on the ladder (3,5).
The solver (plan_topdown2: PLANTEST2's copy of HJEPAStack.plan_topdown with SEED / REG hooks) is imported, not copied; its settings are parameters (SolverCfg).
Admissibility: the goal path takes GoalInputs only (latents up to NOW, nav, measured ego history); no situation-classifier output exists in this module (test_hwm_plan.py scans the source)."""
import contextlib, math, os, sys
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, NamedTuple, Optional, Tuple
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ("plantest", "plantest2", "wmact"):
    sys.path.insert(0, os.path.join(HERE, "..", _d))
import numpy as np
import torch
import plantest_lib as P            # PLANTEST's library, unchanged (apply_prior, make_gen, planted world)
import plantest2_lib as Q           # PLANTEST2: plan_topdown2 with the SEED / REG hooks

DT = 0.2                            # one level-1 step, seconds
N_TRAJ, N_OP = 30, 10               # 6 s and 2 s in level-1 steps (R-TRAJ, R-OP)
MU, G_ACC, VMAX = 0.7, 9.81, 40.0   # kinematic bounds: friction coefficient (as v8f_mini.project_speed), gravity, top speed (this file's choice, ESTIMATED)
ACC_SCALE = 2.0                     # tanitad.models.v8f_mini.WM_ACC_SCALE


# ---------------------------------------------------------------------------------------------- geometry, read from the BUILT model
@dataclass(frozen=True)
class Geometry:
    """All counts in level-1 steps (0.2 s) unless named. s2, s3: strides of levels 2 and 3 (states of the level below). K1: level-2 states the level-1 plan matches; K2: level-3 states the level-2 plan matches;
    T3: level-3 steps of the top plan (the goal lies T3 level-3 steps ahead); n_traj: knots of the output trajectory; n_op: knots taken from level 1."""
    s2: int
    s3: int
    K1: int
    K2: int
    T3: int
    n_traj: int = N_TRAJ
    n_op: int = N_OP
    stitch_mode: str = "decode"     # "decode" (today): knots 1-n_op from level 1, the rest from level 2 decoded. "l1_full" (G3): level 1 plans all n_traj knots against all level-2 sub-goals, no decode

    @property
    def c2(self): return self.s2
    @property
    def c3(self): return self.s2 * self.s3
    @property
    def h1(self): return self.K1 * self.c2           # level-1 planned steps
    @property
    def h2(self): return self.K2 * self.s3           # level-2 planned steps
    @property
    def goal_frame(self): return self.T3 * self.c3   # goal frame, level-1 steps ahead of NOW
    @property
    def n_hold(self): return max(self.goal_frame, self.h1, self.h2 * self.c2)
    @property
    def steps_s(self): return (DT, DT * self.c2, DT * self.c3)

    def check(self):
        assert self.K1 >= 1 and self.K2 >= 1 and self.T3 >= self.K2 and self.K1 <= self.h2, self
        assert self.h1 >= self.n_op and self.h2 * self.c2 >= self.n_traj and self.n_traj >= self.n_op >= 1, self
        assert self.stitch_mode in ("decode", "l1_full"), self.stitch_mode
        if self.stitch_mode == "l1_full":
            assert self.h1 >= self.n_traj, ("l1_full: level 1 must plan every knot", self)
        return self

    @staticmethod
    def from_stack(st, wm=None, goal_steps=None, n_traj=N_TRAJ, n_op=N_OP, stitch="decode"):
        """Strides from the BUILT stack (never from a config). goal_steps: level-3 steps to the goal (default K2: the goal lies at the end of the level-2 horizon; 4 on the ladder is 12 s, G2).
        stitch: "decode" (today) or "l1_full" (G3: K1 = ceil(n_traj / s2), level 1 plans every knot against the level-2 states)."""
        assert st.n_levels == 3 and all(st.cfg[i].window == 1 for i in range(3)), "HWM needs three levels with window 1"
        s2, s3 = int(st.cfg[1].stride), int(st.cfg[2].stride)
        assert int(st.macro_encoders[0].stride) == s2 and int(st.macro_encoders[1].stride) == s3, "macro-encoder stride != LevelConfig stride"
        if wm is not None and hasattr(wm, "built_strides"):
            assert list(wm.built_strides()["stride"]) == [s2, s3], (wm.built_strides(), s2, s3)
        K1 = math.ceil((n_traj if stitch == "l1_full" else n_op) / s2)
        K2 = math.ceil(math.ceil(n_traj / s2) / s3)
        return Geometry(s2, s3, K1, K2, int(goal_steps or K2), n_traj, n_op, stitch).check()

    @staticmethod
    def plantest(st):
        """PLANTEST's geometry on (2,2): top plan 3 steps (2.4 s), K = (3, 6) -> level 2 plans 6 steps, level 1 plans 12 (the whole trajectory comes from level 1)."""
        g = Geometry(int(st.cfg[1].stride), int(st.cfg[2].stride), 6, 3, 3, 12, 12)
        assert (g.s2, g.s3) == (2, 2) and g.h1 == P.H1 and g.h2 == 6
        return g.check()


@dataclass(frozen=True)
class SolverCfg:
    """Parameters that PLANTEST2's result will choose. Defaults = PLANTEST's solver values (library plantest_lib) except seed_mode, which is a placeholder the Master Mind sets."""
    seed_mode: str = "hold"       # "hold": candidate 0 = HOLD at level 1, upper levels seeded from the macro encoding of HOLD (PLANTEST2 SEED); "prior": PLANTEST's initialisation
    lam: float = 0.0              # REG(lambda): cost on level-1 actions, lambda * mean squared deviation from HOLD in prior-std units (needs seed_mode "hold")
    S: int = P.S_CAND
    n_iters: int = P.N_ITERS
    lr: tuple = (P.LR[1], P.LR[2], P.LR[3])
    lam_sub: float = P.LAM
    patience: int = P.PATIENCE
    rel_tol: float = P.REL_TOL
    last_k: Optional[int] = None
    decode_lam: float = 0.0       # macro decode: optional pull toward HOLD in prior-std units (0.0: the inversion starts at HOLD and is not pulled back; 0.01 raised the primitive error on a random-init model)
    decode_iters: int = 300
    decode_lr: float = 0.1
    plan_ctx: object = 1          # G1: rollout context used by the planner, an int (all levels) or a 3-tuple. 1 = H-JEPA's planning context (today); the ladder trains with --wm-ctx 3 3 3

    @staticmethod
    def plantest():
        return SolverCfg(seed_mode="prior", lam=0.0, decode_lam=0.0)

    def lr_dict(self): return {1: self.lr[0], 2: self.lr[1], 3: self.lr[2]}


# ---------------------------------------------------------------------------------------------- goals (information-disjoint from any situation classifier)
class GoalInputs(NamedTuple):
    """The ONLY things a goal predictor may read at inference: level-1 latents up to NOW, the navigation command, the measured ego history and the vision-only context token. No situation-classifier output, no future."""
    z_hist: torch.Tensor                    # (B, H, D) level-1 latents of frames <= NOW (index -1 = NOW)
    nav: Optional[torch.Tensor] = None
    ego_hist: Optional[torch.Tensor] = None  # (B, H, 2) measured (a, yaw rate), frames <= NOW
    ctx: Optional[torch.Tensor] = None       # (B, C) the vision-only strategic token (stok: detected context, derived goals) at NOW; detached by the reader; None = the off arm (G5)


class OracleGoal:
    """HWM-ORACLE: the encoded true-future level-1 latent at goal_frame. T0 diagnostic ceiling, never a driving number."""
    tier, name = "T0", "oracle"

    def __init__(self, z_future: torch.Tensor, goal_frame: int):
        self.z, self.goal_frame = z_future, int(goal_frame)

    def __call__(self, idx, inputs: Optional[GoalInputs] = None):
        return self.z[torch.as_tensor(np.asarray(idx))]


class ShuffledGoal(OracleGoal):
    """G-RAND: the oracle goal of a window of a different episode (donor), for the goal-plumbing controls."""
    name = "rand"

    def __init__(self, z_future, goal_frame, donor):
        super().__init__(z_future, goal_frame); self.donor = np.asarray(donor)

    def __call__(self, idx, inputs=None):
        return self.z[torch.as_tensor(self.donor[np.asarray(idx)])]


class PredictedGoal:
    """HWM-PLAN goal slot: fn(GoalInputs) -> (B, D) latent at goal_frame. level 1 (default): a level-1 latent that the planner encodes up to level 3; level 3: a goal head's own level-3 latent (G4), used as the top goal as it is.
    The goal head is trained by goal_head.py; nothing is trained here."""
    tier, name = "T1-kin", "plan"

    def __init__(self, fn: Callable, goal_frame: int, z_now: torch.Tensor, nav=None, ego=None, ctx=None, level: int = 1):
        assert level in (1, 3), level
        self.fn, self.goal_frame, self.z_now, self.nav, self.ego, self.ctx, self.level = fn, int(goal_frame), z_now, nav, ego, ctx, int(level)

    def __call__(self, idx, inputs: Optional[GoalInputs] = None):
        ii = torch.as_tensor(np.asarray(idx))
        pick = lambda t: None if t is None else t[ii]
        gi = GoalInputs(self.z_now[ii].unsqueeze(1), pick(self.nav), pick(self.ego), pick(self.ctx))
        out = self.fn(gi)
        assert out.dim() == 2 and out.shape[0] == len(ii) and (self.level == 3 or out.shape[1] == self.z_now.shape[-1]), out.shape
        return out


def stub_goal_fn(gi: GoalInputs):
    """STUB: the goal is the current latent (G-STAY). It exists so the plumbing runs; a result from it is not a HWM-PLAN result."""
    return gi.z_hist[:, -1]


# ---------------------------------------------------------------------------------------------- HOLD seeds in the planner's units
def hold_full(wm, A_raw0, lo, hi, n):
    """HOLD (the first logged action a0 = measured acceleration and yaw rate at NOW, held n steps) in scaled planning units, clipped to the level-1 box. Only A_raw[:, 0] is read."""
    a0 = torch.as_tensor(np.asarray(A_raw0)).float().reshape(-1, 1, 2)
    h = wm.scale_actions(a0).repeat(1, n, 1)
    return torch.max(torch.min(h, hi), lo)


def hold_seeds_g(st, hold_c, z_now, geom, mode):
    """Level-1 seed = HOLD_c (hier: h1 steps; flat: goal_frame steps); hier upper levels from the macro encoding of the HOLD sequence (the macro path reads the actions only)."""
    n = geom.n_hold
    if mode == "flat":
        return {1: hold_c[:, :geom.goal_frame]}
    seeds = {1: hold_c[:, :geom.h1]}
    with torch.no_grad():
        z_seq = z_now.unsqueeze(1).expand(-1, n + 1, -1)
        enc = st.encode(z_seq, hold_c[:, :n])
    seeds[2], seeds[3] = enc.act[1][:, :geom.h2].detach(), enc.act[2][:, :geom.T3].detach()
    return seeds


# ---------------------------------------------------------------------------------------------- macro-action decode (L2 -> primitives)
def decode_macro(st, macro, ref, lo, hi, std, cfg: SolverCfg):
    """Invert the macro encoder: primitive actions p (B, n, c2, 2), scaled units, with macro_enc(act_embed(p)) = macro. HJEPAStack has no learned decoder; this is deterministic (AdamW, fixed iterations, init = ref = HOLD_c),
    no sampling. macro (B, n, E); ref (B, n*c2, 2). Returns (p (B, n*c2, 2), relative residual (B,))."""
    b, n, _ = macro.shape
    c2 = int(st.cfg[1].stride)
    mac = st.macro_encoders[0]
    p = ref[:, :n * c2].clone().detach().requires_grad_(True)
    opt = torch.optim.AdamW([p], lr=cfg.decode_lr, weight_decay=0.0)
    tgt = macro.detach()
    scale = tgt.pow(2).sum(-1).mean(-1).clamp_min(1e-12)

    def enc(pp):
        e = st.act_embed(pp)
        return mac(e.reshape(b, n, c2, e.shape[-1]))

    with torch.enable_grad():
        for _ in range(cfg.decode_iters):
            opt.zero_grad(set_to_none=True)
            fit = (enc(p) - tgt).pow(2).sum(-1).mean(-1) / scale
            reg = cfg.decode_lam * ((p - ref[:, :n * c2]) / std).pow(2).mean((-1, -2))
            (fit + reg).sum().backward()
            opt.step()
            with torch.no_grad():
                p.copy_(torch.max(torch.min(p, hi), lo))
    with torch.no_grad():
        resid = ((enc(p) - tgt).pow(2).sum(-1).mean(-1) / scale).sqrt()
    return p.detach(), resid


def stitch(st, geom, a1, macro2, hold_c, lo, hi, std, cfg: SolverCfg, decode=None):
    """Knots 1..n_op from level 1; knots n_op+1..n_traj from level 2's macro-actions decoded to primitives. Returns (traj (B, n_traj, 2) scaled, decode residual (B,) or None)."""
    b = a1.shape[0]
    traj = torch.zeros(b, geom.n_traj, 2, dtype=a1.dtype)
    traj[:, :geom.n_op] = a1[:, :geom.n_op]
    if geom.n_traj == geom.n_op:
        return traj, None
    j0 = geom.n_op // geom.c2                                   # first level-2 step that holds a step beyond n_op
    j1 = math.ceil(geom.n_traj / geom.c2)
    prim, resid = (decode or decode_macro)(st, macro2[:, j0:j1], hold_c[:, j0 * geom.c2:j1 * geom.c2], lo, hi, std, cfg)
    off = geom.n_op - j0 * geom.c2
    traj[:, geom.n_op:] = prim[:, off:off + geom.n_traj - geom.n_op]
    return traj, resid


# ---------------------------------------------------------------------------------------------- solver shadows (G1 context, G6 vocabulary seeds, G7 constraints); plantest2_lib stays untouched
_MISSING = object()


def ctx_tuple(plan_ctx):
    """plan_ctx: an int (all three levels) or a 3-tuple."""
    c = tuple(int(x) for x in plan_ctx) if isinstance(plan_ctx, (tuple, list)) else (int(plan_ctx),) * 3
    assert len(c) == 3 and min(c) >= 1, plan_ctx
    return c


@contextlib.contextmanager
def shadow_solver(st, plan_ctx=1, levels=None, init_fix=None, cost_wrap=None):
    """Instance-level shadows on the stack for the duration of one plan call (the class and plantest2_lib are not touched, the previous instance attribute is restored on exit):
    * st.rollout passes context = plan_ctx[level - 1] (G1; the library's solver calls rollout(level, z0, a) with the default context 1). Not installed when every level is 1.
    * st._optimize: the i-th call is level levels[i] (the solver visits the levels top down: [3, 2, 1] hier, [1] flat); init_fix(level, init) may return a replacement of the initial candidates, cost_wrap(level, cost_fn) a replacement cost.
    With plan_ctx all 1 and no hooks nothing is installed, so the call is the library's own."""
    ctxs = ctx_tuple(plan_ctx)
    saved = {}

    def put(name, fn):
        saved.setdefault(name, st.__dict__.get(name, _MISSING))
        st.__dict__[name] = fn

    if any(c != 1 for c in ctxs):
        orig_roll = st.rollout

        def roll(level, z0, actions, *, context=1):
            return orig_roll(level, z0, actions, context=ctxs[level - 1])

        put("rollout", roll)
    if init_fix is not None or cost_wrap is not None:
        assert levels is not None
        orig_opt, n = st._optimize, [0]

        def opt(cost_fn, init, **kw):
            level = levels[n[0]]
            n[0] += 1
            if cost_wrap is not None:
                cost_fn = cost_wrap(level, cost_fn) or cost_fn
            if init_fix is not None:
                init = init_fix(level, init)
            return orig_opt(cost_fn, init, **kw)

        put("_optimize", opt)
    try:
        yield
    finally:
        for name, old in saved.items():
            if old is _MISSING:
                st.__dict__.pop(name, None)
            else:
                st.__dict__[name] = old


# ---------------------------------------------------------------------------------------------- G7: decoded kinematics of a candidate, and T's constraints as a cost
class KinReadout:
    """Differentiable decode of a level-2 macro-action to its mean raw (acceleration, yaw rate) over its c2 level-1 steps: a ridge-regression readout fitted on TRAIN windows (the macro encoder has no learned decoder; decode_macro is an
    iterative inversion that cannot sit inside a cost). r2 (2,) is the fit quality per channel on the fitting windows: a number to read, not assume."""

    def __init__(self, W: torch.Tensor, b: torch.Tensor, r2, c2: int):
        self.W, self.b, self.r2, self.c2 = W, b, r2, int(c2)

    def __call__(self, macro: torch.Tensor) -> torch.Tensor:
        return macro @ self.W + self.b


def fit_kin_readout(st, wm, Z1, A_raw, ridge: float = 1e-3) -> KinReadout:
    """Z1 (N, T, D) TRAIN level-1 latents, A_raw (N, T, 2) raw logged actions of the same windows (as fit_priors). Target of macro step t = mean of the raw actions of its c2 steps."""
    A_raw = torch.as_tensor(A_raw).float()
    c2 = int(st.cfg[1].stride)
    with torch.no_grad():
        enc = st.encode(Z1, wm.scale_actions(A_raw)[:, :-1])
        m = enc.act[1]
        n = m.shape[1]
        Y = A_raw[:, :n * c2].reshape(A_raw.shape[0], n, c2, 2).mean(2).reshape(-1, 2).double()
        X = m.reshape(-1, m.shape[-1]).double()
        mu, ym = X.mean(0), Y.mean(0)
        Xc, Yc = X - mu, Y - ym
        G = Xc.T @ Xc + ridge * Xc.shape[0] * torch.eye(X.shape[1], dtype=torch.double)
        W = torch.linalg.solve(G, Xc.T @ Yc)
        b = ym - mu @ W
        res = Y - (X @ W + b)
        r2 = 1.0 - res.pow(2).sum(0) / Yc.pow(2).sum(0).clamp_min(1e-12)
    return KinReadout(W.float(), b.float(), r2.numpy(), c2)


@dataclass
class Constraints:
    """T's constraints for the windows (G7), per window: v0 (measured speed at NOW, m/s), v_cap (speed cap, m/s), alat_cap (lateral acceleration cap v |w|, m/s^2), brake_cap (deceleration cap, m/s^2, positive). weight 0.0 = no effect.
    readout: KinReadout, needed for the level-2 term. The planner adds weight * constraint_penalty(level, candidates) to the cost of level 2 and level 1 (hier) or level 1 (flat)."""
    v0: np.ndarray
    v_cap: np.ndarray
    alat_cap: np.ndarray
    brake_cap: np.ndarray
    weight: float = 0.0
    readout: Optional[KinReadout] = None

    def batch(self, ii):
        t = lambda x: torch.as_tensor(np.asarray(x, np.float64)[np.asarray(ii)]).float()
        return dict(v0=t(self.v0), v_cap=t(self.v_cap), alat=t(self.alat_cap), brake=t(self.brake_cap))


def constraint_penalty(level, a, cb, ch1_scale, c2, readout=None):
    """Penalty (B, S) of the candidates a (B, S, H, A) of level 1 (scaled primitive actions) or level 2 (macro-actions, decoded by the readout), through the kinematic model of the harness (differentiable): per step, the squared
    relative exceedance of the speed cap (state after the step), of the lateral acceleration v |w| (state before the step) and of the deceleration; mean over steps. Exactly 0.0 for a plan inside every cap."""
    if level == 1:
        acc, yaw = a[..., 0] * ACC_SCALE, a[..., 1] * float(ch1_scale)
    elif level == 2:
        assert readout is not None, "the level-2 constraint term needs a KinReadout"
        r = readout(a)
        acc, yaw = r[..., 0].repeat_interleave(readout.c2, dim=-1), r[..., 1].repeat_interleave(readout.c2, dim=-1)
    else:
        raise ValueError(level)
    v = cb["v0"].to(a.dtype)[:, None].expand(a.shape[0], a.shape[1])
    vc, lc, bc = (cb[k].to(a.dtype)[:, None] for k in ("v_cap", "alat", "brake"))
    pen = 0.0
    for k in range(acc.shape[-1]):
        lat = torch.relu(v * yaw[..., k].abs() / lc - 1.0).pow(2)
        brk = torch.relu(-acc[..., k] / bc - 1.0).pow(2)
        v = torch.clamp(v + acc[..., k] * DT, 0.0, VMAX)
        pen = pen + lat + brk + torch.relu(v / vc - 1.0).pow(2)
    return pen / acc.shape[-1]


# ---------------------------------------------------------------------------------------------- G6: the tactical vocabulary as L2 candidates
def _class_names():
    from tanitad.refs.refc_tactical import LAT_CLASSES, LON_CLASSES
    return tuple(LAT_CLASSES), tuple(LON_CLASSES)


@dataclass
class Vocabulary:
    """Kinematic templates per (lat, lon) class of the tactical vocabulary (the programme's factored classes, tanitad.refs.refc_tactical: lateral {lane_keep, turn_left, turn_right} x longitudinal {brake_stop, steady, accelerate} = 9),
    raw (acceleration, yaw rate) per level-1 step, at least n_hold steps. Template amplitudes are this file's choice (ESTIMATED): acceleration -2.5 / 0 / +1.5 m/s^2, yaw rate 0 / +0.12 / -0.12 rad/s, held over the whole horizon."""
    names: tuple
    templates: np.ndarray            # (n_classes, n, 2) raw

    @staticmethod
    def default(n: int, acc=(-2.5, 0.0, 1.5), yaw=(0.0, 0.12, -0.12)):
        lat, lon = _class_names()
        names, tpl = [], []
        for i, la in enumerate(lat):
            for j, lo in enumerate(lon):
                names.append(f"{la}/{lo}")
                tpl.append(np.tile([acc[j], yaw[i]], (n, 1)))
        return Vocabulary(tuple(names), np.asarray(tpl, np.float64))

    def order(self, scores, ii, m):
        """Class order per window, (len(ii), m): by descending per-window score when scores (n, n_classes) exist (the top classes of a banked posterior), else the fixed order for every window (all classes)."""
        if scores is None:
            return np.tile(np.arange(m), (len(ii), 1))
        return np.argsort(-np.asarray(scores)[np.asarray(ii)], axis=1, kind="stable")[:, :m]


def vocab_macro_seeds(st, wm, vocab: Vocabulary, z_ref, n, h2):
    """Macro encoding of every template through the macro encoders: (n_classes, h2, E2) in the planner's level-2 units. The macro path reads the actions only (as hold_seeds_g), so one reference latent serves all windows."""
    lo, hi = st.prior_lo_1, st.prior_hi_1
    out = []
    with torch.no_grad():
        for tpl in vocab.templates:
            a = torch.max(torch.min(wm.scale_actions(torch.as_tensor(tpl[:n]).float().reshape(1, n, 2)), hi), lo)
            enc = st.encode(z_ref[:1].reshape(1, 1, -1).expand(1, n + 1, -1), a)
            out.append(enc.act[1][0, :h2])
    return torch.stack(out)


# ---------------------------------------------------------------------------------------------- the planner
class HWMPlanner:
    """mode 'hier': HWM (L3 -> L2 -> L1, stitched). mode 'flat': FLAT-WM, level 1 only, horizon goal_frame, the same goal and the same capacity."""

    def __init__(self, wm, geom: Optional[Geometry] = None, solver: Optional[SolverCfg] = None, plan_fn=None, vocab: Optional[Vocabulary] = None, vocab_scores=None, constraints: Optional[Constraints] = None):
        """vocab (G6): level-2 candidates 1..m are replaced by the macro encodings of the vocabulary templates (candidate 0 stays the seed); vocab_scores (n, n_classes) per-window class scores (top classes first) or None (all classes, fixed order).
        constraints (G7): T's constraints as a cost term at levels 2 and 1. All three extras default to None = today's planner, bit for bit."""
        self.wm, self.st = wm, wm.stack
        self.geom = geom or Geometry.from_stack(self.st, wm)
        self.solver = solver or SolverCfg()
        self.plan_fn = plan_fn or Q.plan_topdown2
        self.vocab, self.vocab_scores, self.constraints = vocab, vocab_scores, constraints
        self.last_aux = {}

    def level1_box(self):
        return self.st.prior_lo_1, self.st.prior_hi_1

    def hold_c(self, A_raw0):
        lo, hi = self.level1_box()
        return hold_full(self.wm, A_raw0, lo, hi, self.geom.n_hold)

    def plan_batch(self, mode, z_now, goal_l1, hold_c, gen, goal_level=1, ii=None):
        """z_now (B, D); goal_l1 (B, D) level-1 latent at goal_frame (goal_level 1) or a level-3 latent (goal_level 3, a goal head's own output; hier only); hold_c (B, n_hold, 2) scaled (None with seed_mode 'prior').
        ii: the window indices of the batch (needed by vocab scores and constraints). Inputs are state + goal + HOLD only."""
        st, g, s = self.st, self.geom, self.solver
        lr = s.lr_dict()
        if s.seed_mode == "prior":
            assert s.lam == 0.0, "REG needs the HOLD seed"
            seeds = reg = None
        else:
            seeds = hold_seeds_g(st, hold_c, z_now, g, mode)
            reg = dict(lam=float(s.lam), ref=seeds[1], std=Q.prior_std_orig(st)) if s.lam != 0.0 else None
        if goal_level == 3:
            assert mode == "hier", "a level-3 goal has no level-1 pre-image: the flat arm needs a level-1 goal"
            goal = goal_l1
        else:
            with torch.no_grad():
                goal = st.initial_latents(goal_l1, 3)[2] if mode == "hier" else goal_l1
        levels = [3, 2, 1] if mode == "hier" else [1]
        init_fix, cost_wrap = self._hooks(mode, z_now, ii)
        with shadow_solver(st, s.plan_ctx, levels, init_fix, cost_wrap):
            if mode == "hier":
                sub_k = {2: g.K2, 1: g.K1}
                res = self.plan_fn(st, z_now, goal, top_horizon=g.T3, goal_level=3, num_candidates=s.S, n_iters=s.n_iters, lr=lr, lam=s.lam_sub, subgoal_K=sub_k,
                                   patience=s.patience, rel_tol=s.rel_tol, last_k=s.last_k, generator=gen, seeds=seeds, reg=reg)
            elif mode == "flat":
                res = self.plan_fn(st, z_now, goal, top_horizon=g.goal_frame, goal_level=1, num_candidates=s.S, n_iters=s.n_iters, lr=lr, lam=0.0, subgoal_K=None,
                                   patience=s.patience, rel_tol=s.rel_tol, last_k=s.last_k, generator=gen, seeds=seeds, reg=reg)
            else:
                raise ValueError(mode)
        return res

    def _hooks(self, mode, z_now, ii):
        """G6 / G7 hooks for shadow_solver, or (None, None) when neither is on (then nothing is installed)."""
        g, s = self.geom, self.solver
        init_fix = cost_wrap = None
        if self.vocab is not None and mode == "hier":
            m = min(len(self.vocab.names), s.S - 1)
            order = self.vocab.order(self.vocab_scores, np.arange(z_now.shape[0]) if ii is None else ii, m)
            seeds_c = vocab_macro_seeds(self.st, self.wm, self.vocab, z_now, g.n_hold, g.h2)             # (n_classes, h2, E2)
            lo2, hi2 = self.st.prior_lo_2, self.st.prior_hi_2
            vs = torch.max(torch.min(seeds_c[torch.as_tensor(order)], hi2), lo2)                         # (B, m, h2, E2)
            self.last_aux["vocab_m"] = m

            def init_fix(level, init, _vs=vs, _m=m):
                if level != 2:
                    return init
                init = init.clone()
                init[:, 1:1 + _m] = _vs.to(init.dtype)
                return init
        c = self.constraints
        if c is not None:
            assert ii is not None, "constraints are per window: plan_batch needs ii"
            cb = c.batch(ii)
            lv = (2, 1) if mode == "hier" else (1,)

            def cost_wrap(level, cost_fn, _cb=cb, _lv=lv):
                if level not in _lv:
                    return None
                w = float(c.weight)
                return lambda a: cost_fn(a) + w * constraint_penalty(level, a, _cb, self.wm.ch1_scale, g.c2, c.readout)
        return init_fix, cost_wrap

    def plan_windows(self, mode, goal_fn, z_now, A_raw0, idx=None, seed=0, batch=P.BATCH, batch0=0):
        """Plan the windows idx in fixed consecutive batches (generator of batch j = make_gen(seed, batch0 + j), identical across goals and arms; batch0 lets a worker plan a slice of the batches and reproduce the whole-run plans). Returns dict: plan_raw (n, n_traj, 2), a1_raw (hier), resid, costs."""
        g, s = self.geom, self.solver
        n = z_now.shape[0]
        idx = np.arange(n) if idx is None else np.asarray(idx)
        hc_all = self.hold_c(A_raw0) if s.seed_mode == "hold" else None
        lo, hi = self.level1_box()
        std = Q.prior_std_orig(self.st)
        out = dict(traj=[], a1=[], resid=[], cost1=[], iters1=[])
        for j, s0 in enumerate(range(0, len(idx), batch)):
            ii = idx[s0:s0 + batch]
            zn = z_now[ii]
            hc = None if hc_all is None else hc_all[ii]
            res = self.plan_batch(mode, zn, goal_fn(ii), hc, P.make_gen(seed, batch0 + j), goal_level=int(getattr(goal_fn, "level", 1)), ii=ii)
            a1 = res.actions.detach()
            if mode == "hier" and g.stitch_mode == "l1_full":
                traj = a1[:, :g.n_traj]                                            # G3: level 1 planned every knot against the level-2 states; no decode
                out["a1"].append(a1)
            elif mode == "hier":
                if hc is None:
                    hc = hold_full(self.wm, A_raw0[ii], lo, hi, g.n_hold)       # decode reference only (HOLD_c); the PLANTEST initialisation itself reads no logged action
                traj, resid = stitch(self.st, g, a1, res.level_actions[2].detach(), hc, lo, hi, std, s)
                out["a1"].append(a1)
                if resid is not None:
                    out["resid"].append(resid)
            else:
                traj = a1[:, :g.n_traj]
            out["traj"].append(traj)
            out["cost1"].append(res.costs[1]); out["iters1"].append(torch.full((len(ii),), float(res.iters[1])))
        r = dict(plan_raw=to_raw(self.wm, torch.cat(out["traj"]).numpy().astype(np.float64)), cost1=torch.cat(out["cost1"]).numpy().astype(np.float64),
                 iters1=torch.cat(out["iters1"]).numpy().astype(np.float64), idx=idx)
        if out["a1"]:
            r["a1_raw"] = to_raw(self.wm, torch.cat(out["a1"]).numpy().astype(np.float64))
        if out["resid"]:
            r["decode_resid"] = torch.cat(out["resid"]).numpy().astype(np.float64)
        return r


def to_raw(wm, a):
    out = np.array(a, np.float64)
    out[..., 0] *= ACC_SCALE
    out[..., 1] *= float(wm.ch1_scale)
    return out


# ---------------------------------------------------------------------------------------------- the kinematic model (friction and speed bounds)
def integrate(v0, plan, dt=DT, vmax=VMAX, mu=MU, bounds=True):
    """Unicycle with bounds: acceleration clipped to +-mu g, yaw rate clipped so that v |w| <= mu g, speed in [0, vmax]; position by the speed at the step start (as PLANTEST's unicycle).
    plan (n, K, 2) raw (a, yaw rate). Returns v (n, K+1), psi (n, K+1), xy (n, K+1, 2) (index k = after k steps) and the fraction of steps where a bound acted."""
    n, K, _ = plan.shape
    v = np.array(v0, np.float64)
    psi, x, y = np.zeros(n), np.zeros(n), np.zeros(n)
    V, PSI, XY = [v.copy()], [psi.copy()], [np.stack([x, y], 1)]
    acted = np.zeros((n, K), bool)
    amax = mu * G_ACC
    for t in range(K):
        a, w = plan[:, t, 0], plan[:, t, 1]
        if bounds:
            ac = np.clip(a, -amax, amax)
            wm_ = amax / np.maximum(v, 1.0)
            wc = np.clip(w, -wm_, wm_)
            acted[:, t] = (ac != a) | (wc != w)
            a, w = ac, wc
        x = x + v * np.cos(psi) * dt
        y = y + v * np.sin(psi) * dt
        v = np.clip(v + a * dt, 0.0, vmax) if bounds else np.maximum(v + a * dt, 0.0)
        psi = psi + w * dt
        V.append(v.copy()); PSI.append(psi.copy()); XY.append(np.stack([x, y], 1))
    return np.stack(V, 1), np.stack(PSI, 1), np.stack(XY, 1), acted.mean()


def hold_plan(A_raw0, n):
    return np.repeat(np.asarray(A_raw0, np.float64).reshape(-1, 1, 2), n, axis=1)


def prior_plan(wm, st, nwin, n):
    m = st.prior_mean_1.detach().numpy().astype(np.float64)
    return np.tile(np.array([m[0] * ACC_SCALE, m[1] * float(wm.ch1_scale)]), (nwin, n, 1))
