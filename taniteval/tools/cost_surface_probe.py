#!/usr/bin/env python3
"""R10 — the refav1 planner COST SURFACE, decomposed term by term and attributed.

⛔ WHY THIS EXISTS. ``D-REFAV1-PAIRED-READ-VOID`` measured refav1 step-1,000's
closed-loop plan as the CONSTANT-VELOCITY straight line on 140/140 eval windows under
both banked checkpoints — the deployed trajectory cannot discriminate recipes. The
Master Mind's reading is that the flat plan is a THREE-factor, compounding property of
the COST and not of the world model:

  (i)   the lateral channel is intrinsically ~40-300x less potent than the longitudinal
        one in the imagination (``D-ACTDIV-ANCHORED-REFAV1``);
  (ii)  the planner proposes in TRUE curvature and hands it to a STEER-trained predictor
        unconverted, under-actuating every turn by arctan(L*k)/k ~ 1/2.9
        (``C-STEER-CURVATURE-INTERFACE``);
  (iii) ``_cost_chunk`` charges an explicit ``0.05*k^2`` at the FULL proposed curvature
        while the imagined benefit arrives at the under-actuated one.

That reading is a HYPOTHESIS. This tool measures it. It sweeps the planner's own
candidate space on a grid at the planner's own horizon, records EVERY cost term
separately, and derives all four attribution arms by ARITHMETIC on ONE set of rollouts
per convention -- so the arms can never disagree about the forward pass.

⭐ THE COST, READ FROM SOURCE (``stack/tanitad/refs/refa_v1.py:1795-1820``):

  T1  :1807-1813   ``1 - cos(z_terminal, goal)``           weight 1.0   sees k THROUGH THE MODEL
  T2  :1814-1815   ``0.02 * mean((a_{k+1}-a_k)/dt)^2``     weight 0.02  channel 0 only
  T3  :1816        ``0.05 * mean(k^2)``                    weight 0.05  sees k RAW
  T4  :1817-1819   ``0.10 * (v_end - target_speed)^2``     weight 0.10  DEAD -- ``target_speed``
                                                                        is never passed by the T1
                                                                        adapter (two probes over
                                                                        refav1_arm.py: 0 hits)

so the LIVE cost is THREE terms. All four are recorded; T4 reads exactly 0.0 unless
``--target-speed`` is given, which is itself a control.

⭐ THE FOURTH FACTOR THE BRIEF DOES NOT CONTAIN, and why the strata exist.
``plan()``'s default goal is the tactical predictor rolled under
``canonical_controls(lat, lon, v0)`` (``refa_v1.py:1647-1684`` / ``:128-164``).
``LANE_KEEP`` gives k == 0; ``CRUISE`` gives a == 0; ``ADAPT_SPEED_FOR_CURVE`` gives
a == 0 whenever v0 <= 8. When the canonical controls are all-zero the goal rollout IS
the ``cv`` candidate's rollout -- the same forward pass -- so T1 = 1 - cos(x, x) = 0
EXACTLY, and since T1, T2, T3 >= 0 the flat plan is the GLOBAL MINIMUM by construction,
whatever the lateral potency, the boundary or the penalty do. Every table is therefore
stratified by ``zero_goal``, and the DISCRIMINATING stratum is reported first.

CONTROLS THAT MUST READ KNOWN VALUES (pre-registered; a failure VOIDS the panel)
--------------------------------------------------------------------------------
  C0  DECOMPOSITION IDENTITY  c_goal + c_jerk + c_kappa + c_vend == c_total, <= 1e-6.
  C1  SHIPPED-COST GATE       this file's re-scoring of cv / hold_v0 / decel_1.5 /
                              proposal must equal ``PlanResult.baseline_costs`` from a
                              REAL ``model.plan()`` call. Run TWICE: with the default
                              (imagined) goal, and with a SUPPLIED oracle goal so the
                              costs are O(1) and the agreement is not trivially met by
                              two numbers that are both ~0.
  C2  BANKED-WINNER GATE      a real ``plan()`` re-run must reproduce the banked ``cl``
                              trajectory to < 3.8e-6 m max abs (the previous agent's gate).
  C3  ZERO-MODEL              a wrapper feeding the ZERO action for every candidate ->
                              c_goal EXACTLY constant across the grid (peak-to-peak 0.0),
                              so total(k) - total(0) == 0.05*k^2 exactly and argmin sits
                              at k = 0. This measures T3's contribution BY CONSTRUCTION.
  C4  CONSTANT-COST           all weights zeroed -> c_total == 0 everywhere and argmin is
                              the first grid index (a deterministic tie), so the argmin
                              machinery cannot manufacture structure.
  C5  CHANNEL-0 INVARIANCE    conventions A and B must agree EXACTLY on c_jerk and on the
                              a-marginal at k = 0.
  C6  n AND d PRINTED         every table carries its n (windows) and its grid size.
  C7  CONVERSION AGREEMENT    our one-line map == ``kinematic.as_command(.., "steer")``
                              to 1e-7 when that helper is importable (never depended on).

THE TWO CONVENTIONS
-------------------
  A  "as shipped"  -- controls handed to ``augment_actions`` unconverted.
  B  "repaired"    -- ``steer = arctan(L * kappa)`` applied to channel 1 IMMEDIATELY
                      before ``augment_actions`` and nowhere else; the search space,
                      ``_clip``, T3's k^2 and any integrated path stay in curvature.

L is the ENCODING wheelbase, not a vehicle property: ``physicalai.signals_at`` wrote
``steer = arctan(L*curvature)`` with the value the BUILD passed and every shipped cache
passed 2.9. Default ``--wheelbase 2.9``; ``--wheelbase-sweep`` reports the sensitivity
over {2.73, 3.085, 3.216} (the release's real vehicle wheelbases, none of them 2.9).

⭐ A FIFTH FACTOR, FOUND ON THE FIRST SMOKE WINDOW AND THEREFORE INSTRUMENTED HERE.
The shipped goal term is ``1 - cosine_similarity(...)`` evaluated in FLOAT32, and near
cos = 1 the representable steps are ``spacing(1f)/2 = 5.9604645e-08`` apart. On the
first window measured, EVERY one of the 189 grid cells' goal terms was an exact small
integer multiple of that step (1, 2, 3, 4 x) and the whole grid's range was 1.19e-07 --
while the explicit curvature penalty at ``GOAL_KAPPA_TURN`` is 3.2e-04, i.e. **2,700x
larger than the entire modelled signal**. So the probe also evaluates the SAME cosine of
the SAME float32 fields in float64 (``*_f64`` arms), and reports the terminal-field
displacement, which needs no cosine at all:

  (iv)  FLOAT32 SATURATION of the goal cosine -- the arithmetic destroys the modelled
        benefit before any of factors (i)-(iii) get to act on it.

TIER: T0. The cost surface is a WM/cost diagnostic anchored at t0. ``turn_frac`` is a
property of the COST, never of the car; any driving claim needs the T1 adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# the shipped weights, mirrored ONCE with their source line so a drift in       #
# refa_v1.py shows up as a C1 FAILURE rather than as a silently different cost. #
# --------------------------------------------------------------------------- #
W_JERK = 0.02           #: refa_v1.py:1815
W_KAPPA = 0.05          #: refa_v1.py:1816
W_VEND = 0.10           #: refa_v1.py:1819 (inactive: target_speed is never passed)
TURN_THRESHOLD = 0.04   #: 1/m -- half of GOAL_KAPPA_TURN (refa_v1.py:116)
GT_TURN_DEG = 5.0       #: deg over the plan horizon -- the "the human turns" stratum
BANKED_GATE_M = 3.8e-6  #: the previous agent's reproduction gate
L_SWEEP = (2.73, 3.085, 3.216)
#: ⭐ the float32 resolution of ``1 - cos`` NEAR 1. ``cosine_similarity`` returns
#: float32; the representable steps just below 1.0 are spaced by ``spacing(1f)/2 =
#: 5.9604645e-08``, so a goal term whose whole range across the candidate set is a small
#: integer multiple of this number carries NO information -- it is the arithmetic's
#: resolution, not the model's response. MEASURED on the first smoke window: every
#: c_goal value on the 189-cell grid was an exact multiple of it (1, 2, 3, 4 x).
F32_COS_ULP = 5.9604645e-08


def _p(*a):
    print(*a, flush=True)


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_refav1_arm():
    """Import the sibling ``refav1_arm.py`` BY FILE so its strict loader, its loader
    construction and its integrator are REUSED, never re-derived (the pattern
    ``actdiv_anchored.py:996-1001`` uses)."""
    p = os.path.join(_HERE, "refav1_arm.py")
    if not os.path.exists(p):
        raise SystemExit(f"[cost_surface] sibling refav1_arm.py not found at {p}")
    spec = importlib.util.spec_from_file_location("refav1_arm_for_costsurface", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# candidates                                                                    #
# --------------------------------------------------------------------------- #
def build_grid(horizon: int, kappa_max: float, a_max: float, n_k: int, n_a: int,
               device, dtype):
    """CONSTANT-over-horizon candidates on the (a, kappa) product grid, row-major in a.

    Constant candidates make T2 (jerk) identically zero -- a STRUCTURAL fact, not an
    omission: jerk is a first difference of a constant, so the comfort term cannot move
    the grid minimum. It is still recorded, and the NAMED candidates are non-constant so
    T2 is exercised there.
    """
    import torch
    # ⛔ the axes are built in numpy float64 and the ZERO IS FORCED EXACT.
    # ``torch.linspace(-0.2, 0.2, 21)[10]`` is 1.49e-08, NOT 0 -- and that one
    # non-zero made controls C3 and C5 read FALSE on the first smoke run
    # (arctan(2.9 * 1.49e-8) != 1.49e-8, so conventions A and B differed by one
    # float32 ULP at "kappa = 0"). A control that must read a KNOWN value needs the
    # known point to be exactly on the grid.
    ks_np = np.linspace(-kappa_max, kappa_max, n_k)
    as_np = np.linspace(-a_max, a_max, n_a)
    ks_np[int(np.argmin(np.abs(ks_np)))] = 0.0
    as_np[int(np.argmin(np.abs(as_np)))] = 0.0
    ks = torch.tensor(ks_np, device=device, dtype=dtype)
    as_ = torch.tensor(as_np, device=device, dtype=dtype)
    A, K = torch.meshgrid(as_, ks, indexing="ij")                   # [n_a, n_k]
    flat = torch.stack([A.reshape(-1), K.reshape(-1)], dim=-1)      # [G, 2]
    ctrl = flat[:, None, :].expand(-1, horizon, -1).contiguous()    # [G, H, 2]
    return ctrl, as_.detach().cpu().numpy(), ks.detach().cpu().numpy()


def named_candidates(pc, v0: float, proposal, goal_controls, device):
    """The planner's own named candidates: the injected baselines
    (``refa_v1_plan._baseline_controls:167-186``) plus the canonical-goal seed that
    ``plan()`` adds at ``refa_v1.py:1790-1793``. Non-constant, so T2 is live."""
    from tanitad.refs.refa_v1_plan import _baseline_controls
    out = dict(_baseline_controls(pc, v0, device, proposal))
    if goal_controls is not None:
        out["goal_canonical"] = goal_controls[:pc.horizon]
    return {k: v.to(device).float() for k, v in out.items()}


def to_steer(controls, wheelbase: float):
    """GEOMETRY -> COMMAND at the model boundary: ``steer = arctan(L * kappa)``.
    Channel 0 (accel) is unit-invariant and passes through untouched."""
    import torch
    return torch.cat([controls[..., :1],
                      torch.atan(float(wheelbase) * controls[..., 1:2]),
                      controls[..., 2:]], dim=-1)


# --------------------------------------------------------------------------- #
# the cost, re-implemented term by term -- gated against the shipped one (C1)    #
# --------------------------------------------------------------------------- #
class WindowContext:
    """Everything ``RefAV1.plan`` computes once per window, computed once here too.

    Mirrors ``refa_v1.py:1731-1793`` statement for statement; control C1 is what proves
    the mirror is faithful, and it is the only reason this re-implementation is
    admissible at all.
    """

    def __init__(self, model, cfg, feats, v0: float, nav_cmd, pc, goal_field=None):
        import torch
        self.model, self.cfg, self.pc = model, cfg, pc
        self.v0 = float(v0)
        self.dev = feats.device
        self.v0_t = torch.as_tensor([float(v0)], dtype=torch.float32, device=self.dev)
        field = model.encode(feats)                                    # :1739
        self.last = model._last_state(field)                           # :1740
        pooled_win = field.mean(dim=-2)                                # :1741
        pooled = pooled_win[:, -1]                                     # :1742
        self.brains = model._run_brains(pooled_win, nav_cmd)           # :1744
        self.intent = None if self.brains is None else self.brains["intent"]
        modes = model.proposal(pooled).reshape(cfg.proposal_k, cfg.plan_steps,
                                               cfg.a_dim)              # :1746
        if model.proposal_score is not None:                           # :1748
            modes = modes[model.proposal_score(pooled)[0].argsort(descending=True)]
        self.proposal = modes[0]                                       # :1757
        self.coarse = cfg.plan_level == "tactical"                     # :1764
        self.pred = model.tactical if self.coarse else model.operative
        self.z0 = model._tac_field(self.last) if self.coarse else self.last
        self.goal_lat = self.goal_lon = None
        self.goal_controls = None
        if goal_field is not None:                                     # :1771
            self.goal_t, self.goal_source = model._tac_field(goal_field), "supplied"
        elif self.brains is not None:
            self.goal_t, ga = model._imagine_tactical_goal(
                self.last, self.brains, self.v0_t)                     # :1772
            self.goal_source = "tactical_imagined"
            self.goal_lat, self.goal_lon = ga["lat"][0], ga["lon"][0]
            self.goal_controls = ga["controls"][0]
        else:
            self.goal_t, self.goal_source = None, "none"
        gc = self.goal_controls
        self.zero_goal = bool(gc is not None and float(gc.abs().max()) == 0.0)

    def score(self, controls, *, units: str = "kappa", wheelbase: float = 2.9,
              zero_model: bool = False, chunk: int = 64, target_speed=None,
              exact: bool = False):
        """``[n, H, 2]`` controls -> EVERY cost term, separately.

        ``units="kappa"`` is convention A (the shipped pass-through);
        ``units="steer"`` is convention B (the one-line boundary conversion).
        ``zero_model=True`` feeds the ZERO action for every candidate (control C3).

        ⭐ ``exact=True`` additionally returns ``c_goal_f64`` -- the SAME cosine of the
        SAME float32 terminal fields, evaluated in float64 -- and the relative terminal
        displacement ``disp``. This is not a cosmetic upgrade: the shipped term is
        ``1 - cosine_similarity(...)`` in float32, whose representable steps just below
        1.0 are 5.96e-08 apart, and the measured surface sits ENTIRELY inside 4 of those
        steps. ``c_goal_f64`` says whether a real modelled response exists BELOW that
        resolution; ``disp`` measures the same response without a cosine at all, so a
        saturating similarity cannot hide it.
        """
        import torch
        import torch.nn.functional as F
        model, pc = self.model, self.pc
        n = controls.shape[0]
        cg = torch.zeros(n, device=self.dev)
        zt_cpu = [] if exact else None
        for i in range(0, n, chunk):
            c = controls[i:i + chunk]
            m = c.shape[0]
            z = self.z0.expand(m, -1, -1)                              # :1799
            c_model = c if units == "kappa" else to_steer(c, wheelbase)
            acts = model.augment_actions(c_model, self.v0_t.expand(m))  # :1803
            if zero_model:
                acts = torch.zeros_like(acts)
            zk = self.pred.rollout(z, acts, intent=self.intent,
                                   last_only=True)                     # :1805
            if self.goal_t is None:
                continue
            zt = zk if self.pred is model.tactical else model._tac_field(zk)  # :1810
            g = self.goal_t.expand(m, -1, -1)                          # :1811
            cg[i:i + m] = 1.0 - F.cosine_similarity(
                zt.flatten(1), g.flatten(1), dim=-1)                   # :1812
            if exact:
                zt_cpu.append(zt.flatten(1).float().cpu().numpy())
        if controls.shape[1] > 1:
            jerk = (controls[:, 1:, 0] - controls[:, :-1, 0]) / pc.dt  # :1814
            cj = W_JERK * jerk.pow(2).mean(-1)                         # :1815
        else:
            cj = torch.zeros(n, device=self.dev)
        ck = W_KAPPA * controls[..., 1].pow(2).mean(-1)                # :1816
        cv = torch.zeros(n, device=self.dev)
        if target_speed is not None:                                   # :1817
            v_end = self.v0 + controls[..., 0].sum(-1) * pc.dt         # :1818
            cv = W_VEND * (v_end - float(target_speed)).pow(2)         # :1819
        out = {"c_goal": cg, "c_jerk": cj, "c_kappa": ck, "c_vend": cv,
               "c_total": cg + cj + ck + cv}
        if exact:
            out["_zt"] = (np.concatenate(zt_cpu, axis=0) if zt_cpu else None)
            out["_goal"] = (self.goal_t.flatten(1).float().cpu().numpy()
                            if self.goal_t is not None else None)
        return out


# --------------------------------------------------------------------------- #
# per-window analysis                                                          #
# --------------------------------------------------------------------------- #
def _argmin_cell(total, n_k: int, a_axis, k_axis):
    """argmin over the grid -> (a*, kappa*, flat index). Ties resolve to the LOWEST
    index, i.e. (a_min, kappa_min) -- deterministic, and control C4 reads exactly that."""
    j = int(np.argmin(total))
    return float(a_axis[j // n_k]), float(k_axis[j % n_k]), j


def analyse_window(ctx, grid, a_axis, k_axis, named, *, wheelbase, chunk,
                   target_speed=None, do_zero_model=False, save_surface=False):
    import torch
    n_a, n_k = int(a_axis.size), int(k_axis.size)
    row0 = int(np.argmin(np.abs(a_axis)))       # the a = 0 row
    col0 = int(np.argmin(np.abs(k_axis)))       # the kappa = 0 column
    out: dict = {}
    raw: dict = {}
    for units, tag in (("kappa", "A"), ("steer", "B")):
        s = ctx.score(grid, units=units, wheelbase=wheelbase, chunk=chunk,
                      target_speed=target_speed, exact=True)
        zt, gv = s.pop("_zt", None), s.pop("_goal", None)
        raw[tag] = {k: v.detach().float().cpu().numpy().astype(np.float64)
                    for k, v in s.items()}
        # ⭐ the SAME cosine of the SAME float32 fields, in float64 -- the read that
        # says whether a modelled response exists BELOW the shipped arithmetic's
        # resolution; and the terminal displacement, which needs no cosine at all.
        if zt is not None and gv is not None:
            z64 = zt.astype(np.float64)
            g64 = gv.astype(np.float64).reshape(1, -1)
            cs = (z64 @ g64[0]) / (np.linalg.norm(z64, axis=1) *
                                   np.linalg.norm(g64) + 1e-300)
            raw[tag]["c_goal_f64"] = 1.0 - cs
            anchor = z64[row0 * n_k + col0]           # the (a=0, kappa=0) cell
            raw[tag]["disp"] = (np.linalg.norm(z64 - anchor, axis=1) /
                                max(np.linalg.norm(anchor), 1e-300))
        else:
            raw[tag]["c_goal_f64"] = raw[tag]["c_goal"].copy()
            raw[tag]["disp"] = np.zeros_like(raw[tag]["c_goal"])

    # ---- the attribution arms are ARITHMETIC on those two scorings ------------
    # float32 = the SHIPPED arithmetic (what the planner actually does);
    # _f64     = the same cost with an exact cosine (factor (iv): saturation).
    for tag in ("A", "B"):
        r = raw[tag]
        for gkey, gsfx in (("c_goal", ""), ("c_goal_f64", "_f64")):
            for pen, penname in ((True, "full"), (False, "nopen")):
                tot = r[gkey] + r["c_jerk"] + r["c_vend"] + \
                    (r["c_kappa"] if pen else 0.0)
                a_s, k_s, j = _argmin_cell(tot, n_k, a_axis, k_axis)
                arm = f"{tag}_{penname}{gsfx}"
                out[f"{arm}_a"] = a_s
                out[f"{arm}_kappa"] = k_s
                out[f"{arm}_cost"] = float(tot[j])
                out[f"{arm}_turn"] = bool(abs(k_s) >= TURN_THRESHOLD)
                out[f"{arm}_cost_at_origin"] = float(tot[row0 * n_k + col0])
                out[f"{arm}_surface_ptp"] = float(tot.max() - tot.min())

    # ---- ⛔ THE SATURATION READ: is the shipped goal term below its own ULP? ---
    for tag in ("A", "B"):
        g32 = raw[tag]["c_goal"]
        q = g32 / F32_COS_ULP
        out[f"{tag}_goal_ptp_f32"] = float(g32.max() - g32.min())
        out[f"{tag}_goal_ptp_in_ulps"] = float((g32.max() - g32.min()) / F32_COS_ULP)
        out[f"{tag}_goal_is_ulp_quantized"] = bool(
            np.max(np.abs(q - np.round(q))) < 1e-3)
        out[f"{tag}_goal_n_distinct_f32"] = int(np.unique(g32).size)
        g64 = raw[tag]["c_goal_f64"]
        out[f"{tag}_goal_ptp_f64"] = float(g64.max() - g64.min())
        out[f"{tag}_f64_over_f32_ptp"] = float(
            (g64.max() - g64.min()) / max(g32.max() - g32.min(), 1e-300))
        d = raw[tag]["disp"]
        G = d.reshape(n_a, n_k)
        out[f"{tag}_disp_kappa_max"] = float(G[row0].max())
        out[f"{tag}_disp_a_max"] = float(G[:, col0].max())
        out[f"{tag}_disp_ratio_kappa_over_a"] = float(
            G[row0].max() / max(G[:, col0].max(), 1e-300))
        Gk = raw[tag]["c_goal_f64"].reshape(n_a, n_k)
        kt = int(np.argmin(np.abs(k_axis - 0.08)))
        out[f"{tag}_goal_gain_at_kturn_f64"] = float(Gk[row0, col0] - Gk[row0, kt])

    # ---- surface shape --------------------------------------------------------
    kt = int(np.argmin(np.abs(k_axis - 0.08)))          # GOAL_KAPPA_TURN
    for tag in ("A", "B"):
        G = raw[tag]["c_goal"].reshape(n_a, n_k)
        gk = G[row0]                                    # T1 along kappa at a = 0
        ga = G[:, col0]                                 # T1 along a at kappa = 0
        out[f"{tag}_goal_kappa_range"] = float(gk.max() - gk.min())
        out[f"{tag}_goal_a_range"] = float(ga.max() - ga.min())
        out[f"{tag}_goal_at_origin"] = float(G[row0, col0])
        # ⭐ THE SHARP FORM OF THE SATURATION: how many float32 steps of the cosine
        # does each AXIS span? A term resolved over ~2 steps carries ~1 bit.
        out[f"{tag}_goal_kappa_range_in_ulps"] = float(
            (gk.max() - gk.min()) / F32_COS_ULP)
        out[f"{tag}_goal_a_range_in_ulps"] = float(
            (ga.max() - ga.min()) / F32_COS_ULP)
        G64 = raw[tag]["c_goal_f64"].reshape(n_a, n_k)
        out[f"{tag}_goal_kappa_range_f64"] = float(G64[row0].max() - G64[row0].min())
        out[f"{tag}_goal_a_range_f64"] = float(G64[:, col0].max() - G64[:, col0].min())
        out[f"{tag}_potency_ratio_f64"] = float(
            out[f"{tag}_goal_kappa_range_f64"] /
            max(out[f"{tag}_goal_a_range_f64"], 1e-300))
        # the modelled BENEFIT a turn can buy, against the explicit charge it pays
        out[f"{tag}_goal_gain_at_kturn"] = float(gk[col0] - gk[kt])
        out[f"{tag}_penalty_at_kturn"] = float(W_KAPPA * k_axis[kt] ** 2)
    out["potency_ratio_A"] = out["A_goal_kappa_range"] / max(out["A_goal_a_range"], 1e-12)
    out["potency_ratio_B"] = out["B_goal_kappa_range"] / max(out["B_goal_a_range"], 1e-12)
    out["B_over_A_kappa_range"] = (out["B_goal_kappa_range"] /
                                   max(out["A_goal_kappa_range"], 1e-15))

    # ---- C0: the decomposition identity --------------------------------------
    ident = 0.0
    for tag in ("A", "B"):
        r = raw[tag]
        ident = max(ident, float(np.max(np.abs(
            r["c_goal"] + r["c_jerk"] + r["c_kappa"] + r["c_vend"] - r["c_total"]))))
    out["C0_identity_max_abs"] = ident

    # ---- C4: the constant-cost control ---------------------------------------
    zero_tot = 0.0 * raw["A"]["c_goal"] + 0.0 * raw["A"]["c_jerk"] + \
        0.0 * raw["A"]["c_kappa"]
    out["C4_all_zero"] = bool(np.all(zero_tot == 0.0))
    out["C4_argmin_index"] = int(np.argmin(zero_tot))

    # ---- C5: channel-0 invariance --------------------------------------------
    out["C5_jerk_max_abs_diff"] = float(np.max(np.abs(raw["A"]["c_jerk"] -
                                                      raw["B"]["c_jerk"])))
    out["C5_a_marginal_max_abs_diff"] = float(np.max(np.abs(
        raw["A"]["c_goal"].reshape(n_a, n_k)[:, col0] -
        raw["B"]["c_goal"].reshape(n_a, n_k)[:, col0])))

    # ---- C3: the zero-model control ------------------------------------------
    if do_zero_model:
        z = {k: v.detach().float().cpu().numpy().astype(np.float64)
             for k, v in ctx.score(grid, units="kappa", wheelbase=wheelbase,
                                   chunk=chunk, zero_model=True,
                                   target_speed=target_speed).items()}
        out["C3_zero_model_goal_ptp"] = float(z["c_goal"].max() - z["c_goal"].min())
        tot = z["c_total"].reshape(n_a, n_k)
        delta = tot[row0] - tot[row0][col0]
        out["C3_penalty_recovery_max_abs"] = float(np.max(np.abs(
            delta - W_KAPPA * (k_axis ** 2))))
        _a, k_s, _ = _argmin_cell(z["c_total"], n_k, a_axis, k_axis)
        out["C3_zero_model_argmin_kappa"] = k_s

    # ---- the named candidates: T2 is live here --------------------------------
    names = list(named)
    if names:
        stack = torch.stack([named[k] for k in names])
        for units, tag in (("kappa", "A"), ("steer", "B")):
            s = ctx.score(stack, units=units, wheelbase=wheelbase, chunk=chunk,
                          target_speed=target_speed)
            for kk, v in s.items():
                arr = v.detach().float().cpu().numpy().astype(np.float64)
                for nm, val in zip(names, arr):
                    out[f"named_{tag}_{nm}_{kk}"] = float(val)

    surf = None
    if save_surface:
        surf = {tag: {k: raw[tag][k].reshape(n_a, n_k).tolist()
                      for k in ("c_goal", "c_goal_f64", "c_kappa", "c_total", "disp")}
                for tag in ("A", "B")}
    return out, surf


def _gt_turn_deg(poses, t: int, k: int) -> float:
    """|heading change| in DEGREES over the plan horizon, from the RAW 10 Hz poses
    (frame 2(t+j)) -- the indexing ``refav1_arm.gt_waypoints:436-441`` uses. This reads
    GROUND TRUTH, so it may only ever SELECT windows and never enter a cost."""
    f0, f1 = 2 * t, 2 * (t + k)
    if f1 >= poses.shape[0]:
        return float("nan")
    y0, y1 = float(poses[f0, 2]), float(poses[f1, 2])
    return math.degrees(abs(math.atan2(math.sin(y1 - y0), math.cos(y1 - y0))))


def _banked_gate(dump_dir: str, fi: int, t: int, res, v0: float, dt: float, k: int, ra):
    """C2 -- a real ``plan()`` re-run must reproduce the BANKED ``cl`` trajectory."""
    p = os.path.join(dump_dir, f"ep{fi:03d}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    if "cl" not in d.files or "ws" not in d.files:
        return None
    j = np.flatnonzero(d["ws"].astype(np.int64) == t)
    if j.size != 1:
        return None
    banked = d["cl"][int(j[0])]
    mine = ra.paths_from_controls(res.controls, v0, float(dt), k)
    mine = mine[0].detach().float().cpu().numpy()
    m = float(np.max(np.abs(mine - banked)))
    return {"ep": fi, "t": int(t), "max_abs_m": m, "gate_m": BANKED_GATE_M,
            "pass": bool(m < BANKED_GATE_M), "plan_source": str(res.source)}


def _c1_gate(ctx, model, feats, v0, nav_t, pc, ra, dev, *, goal_field, label,
             wheelbase, chunk, target_speed):
    """C1 -- our re-scoring vs the SHIPPED ``_cost_chunk`` via ``baseline_costs``."""
    import torch
    from tanitad.refs.refa_v1_plan import _baseline_controls
    res = model.plan(feats, v0=v0, nav_cmd=nav_t, plan_cfg=pc, goal_field=goal_field)
    bcs = _baseline_controls(pc, v0, dev, ctx.proposal)
    bn = list(bcs)
    mine = ctx.score(torch.stack([bcs[x] for x in bn]), units="kappa",
                     wheelbase=wheelbase, chunk=chunk,
                     target_speed=target_speed)["c_total"]
    mine = mine.detach().float().cpu().numpy()
    worst, worst_nm, scale = 0.0, "", 0.0
    for nmm, val in zip(bn, mine):
        ship = float(res.baseline_costs[nmm])
        scale = max(scale, abs(ship))
        e = abs(float(val) - ship) / max(1.0, abs(ship))
        if e > worst:
            worst, worst_nm = e, nmm
    return res, {"label": label, "rel_err": worst, "worst_candidate": worst_nm,
                 "cost_scale": scale, "pass": bool(worst <= 1e-6),
                 "shipped": {k: float(v) for k, v in res.baseline_costs.items()},
                 "mine": {k: float(v) for k, v in zip(bn, mine)}}


# --------------------------------------------------------------------------- #
# the run                                                                      #
# --------------------------------------------------------------------------- #
def run(a) -> dict:
    import torch
    ra = _load_refav1_arm()
    dev = a.device
    t_start = time.time()

    model, cfg, prov = ra.load_model(a.ckpt, a.config, dev, False)
    H = int(cfg.plan_steps)
    pc = ra._plan_cfg(cfg, argparse.Namespace(
        plan_seed=a.plan_seed, plan_n_samples=a.plan_n_samples,
        plan_n_iters=a.plan_n_iters, plan_n_elites=a.plan_n_elites))
    names = ra.episode_names(a.cache)
    if a.episodes_n:
        names = names[:int(a.episodes_n)]
    k_loader = max(int(a.horizon_k), int(cfg.op_steps))
    ld = ra.build_loader(argparse.Namespace(
        cache=a.cache, episodes=a.episodes, lru=a.lru, labels=a.labels, nav=a.nav),
        cfg, k_loader, names)
    stride = max(1, int(a.window_stride))
    sel = [(wi, ei, t) for wi, (ei, t) in enumerate(ld.windows)
           if (t - (ld.W - 1)) % stride == 0]
    if not sel:
        raise SystemExit("[cost_surface] the stride selected zero windows")
    nav_true = np.zeros(len(sel), dtype=np.int64)
    if ld._nav_on:
        for i, (wi, ei, t) in enumerate(sel):
            nid = ld._nav_id.get(ld.clip_id[ld.names[ei]])
            nav_true[i] = 0 if nid is None else int(nid)

    grid, a_axis, k_axis = build_grid(H, pc.kappa_max, pc.a_max, a.n_kappa,
                                      a.n_accel, dev, torch.float32)
    _p(f"[model] {a.ckpt} step={prov['step']} a_dim={cfg.a_dim} "
       f"speed_channel={getattr(cfg, 'speed_channel', None)} "
       f"plan_level={cfg.plan_level} H={H} dt={cfg.op_dt}")
    _p(f"[grid] n_kappa={k_axis.size} x n_accel={a_axis.size} = {grid.shape[0]} "
       f"candidates; kappa in +-{pc.kappa_max}, a in +-{pc.a_max}; L={a.wheelbase}")
    _p(f"[pop] {len(sel)} windows over {len({e for _, e, _ in sel})} episodes "
       f"stride={stride} device={dev}")

    # ---- C7: our one-line map vs the sibling stream's helper ------------------
    try:
        from tanitad.models.kinematic import STEER_WHEELBASE_M, as_command
        probe = torch.randn(7, H, 2, device=dev) * 0.1
        d = float((as_command(probe, "steer", a.wheelbase) -
                   to_steer(probe, a.wheelbase)).abs().max())
        c7 = {"available": True, "max_abs_diff": d, "pass": bool(d <= 1e-7),
              "kinematic_STEER_WHEELBASE_M": float(STEER_WHEELBASE_M)}
    except Exception as ex:                                        # noqa: BLE001
        c7 = {"available": False, "pass": None,
              "reason": f"{type(ex).__name__}: {ex}",
              "note": "the probe implements the conversion itself; this is a "
                      "cross-check, never a dependency"}
    _p(f"[C7] {c7}")

    by_ep: dict[int, list] = {}
    for i, (wi, ei, t) in enumerate(sel):
        by_ep.setdefault(ei, []).append((i, wi, t))

    rows: list[dict] = []
    surfaces: list[dict] = []
    gates = {"C1": [], "C2": []}
    for fi, ei in enumerate(sorted(by_ep)):
        nm = ld.names[ei]
        o = torch.load(ld.episode_dir / f"{nm}.v2ep.pt", map_location="cpu",
                       weights_only=False)
        poses = o["poses"].float()
        _F, v_ep, _k = ld._episode(nm)
        for (i, wi, t) in by_ep[ei]:
            ld._order, ld._cursor = [wi], 0
            b = ld.batch(1)
            feats = b["feats"].to(dev)
            v0 = float(v_ep[2 * t])
            if b.get("v0") is not None and abs(float(b["v0"][0]) - v0) > 1e-6:
                raise RuntimeError("loader v0 != poses[2t, 3] — loader drift")
            nav_t = (torch.tensor([int(nav_true[i])], device=dev)
                     if ld._nav_on else None)
            with torch.no_grad():
                ctx = WindowContext(model, cfg, feats, v0, nav_t, pc)
                named = named_candidates(pc, v0, ctx.proposal, ctx.goal_controls, dev)
                r, surf = analyse_window(
                    ctx, grid, a_axis, k_axis, named, wheelbase=a.wheelbase,
                    chunk=a.chunk, target_speed=a.target_speed,
                    do_zero_model=(len(rows) < a.zero_model_n),
                    save_surface=(len(surfaces) < a.save_surfaces_n))
                if surf is not None:
                    surfaces.append({"ep_file": fi, "t": int(t), "v0": v0,
                                     "goal_lat": ctx.goal_lat,
                                     "goal_lon": ctx.goal_lon, "surface": surf})
                if a.wheelbase_sweep and len(rows) < a.wheelbase_sweep_n:
                    for L in L_SWEEP:
                        tot = ctx.score(grid, units="steer", wheelbase=L,
                                        chunk=a.chunk, target_speed=a.target_speed
                                        )["c_total"].detach().float().cpu().numpy()
                        _a2, _k2, _ = _argmin_cell(tot, int(k_axis.size), a_axis, k_axis)
                        r[f"Lsweep_{L}_kappa"] = _k2
                        r[f"Lsweep_{L}_turn"] = bool(abs(_k2) >= TURN_THRESHOLD)
                if len(gates["C1"]) < 2 * a.gate_n:
                    # (a) the DEFAULT (imagined) goal — the shipped path
                    res, g1 = _c1_gate(ctx, model, feats, v0, nav_t, pc, ra, dev,
                                       goal_field=None, label="imagined_goal",
                                       wheelbase=a.wheelbase, chunk=a.chunk,
                                       target_speed=a.target_speed)
                    g1.update({"ep": fi, "t": int(t)})
                    gates["C1"].append(g1)
                    if a.banked_dump:
                        gp = _banked_gate(a.banked_dump, fi, int(t), res, v0,
                                          cfg.op_dt, int(a.horizon_k), ra)
                        if gp is not None:
                            gates["C2"].append(gp)
                    # (b) a SUPPLIED oracle goal — costs are O(1), so the agreement
                    #     is not trivially met by two numbers that are both ~0
                    fut = b["future_feats"].to(dev)
                    go = model.adapter(model.std(fut[:, :H]))[:, H - 1]
                    ctx_o = WindowContext(model, cfg, feats, v0, nav_t, pc,
                                          goal_field=go)
                    _res2, g2 = _c1_gate(ctx_o, model, feats, v0, nav_t, pc, ra, dev,
                                         goal_field=go, label="supplied_oracle_goal",
                                         wheelbase=a.wheelbase, chunk=a.chunk,
                                         target_speed=a.target_speed)
                    g2.update({"ep": fi, "t": int(t)})
                    gates["C1"].append(g2)
            r.update({"ep_file": fi, "ep_index": int(ei), "ep_name": nm,
                      "t": int(t), "v0": v0, "nav": int(nav_true[i]),
                      "goal_lat": ctx.goal_lat, "goal_lon": ctx.goal_lon,
                      "goal_source": ctx.goal_source, "zero_goal": ctx.zero_goal,
                      "goal_kappa_max": (float(ctx.goal_controls[:, 1].abs().max())
                                         if ctx.goal_controls is not None else None),
                      "gt_turn_deg": _gt_turn_deg(poses, t, int(a.horizon_k))})
            rows.append(r)
        _p(f"  [{fi + 1}/{len(by_ep)}] {nm[:14]} {len(by_ep[ei])} windows "
           f"{time.time() - t_start:.0f}s")

    # ---- provenance: the md5 of every source this run actually imported -------
    import tanitad
    stack_root = os.path.dirname(os.path.dirname(os.path.abspath(tanitad.__file__)))
    tree = os.path.dirname(stack_root)
    src_md5 = {}
    for rel in ("stack/tanitad/refs/refa_v1.py", "stack/tanitad/refs/refa_v1_plan.py",
                "stack/tanitad/models/kinematic.py", "taniteval/tools/refav1_arm.py"):
        pth = os.path.join(tree, *rel.split("/"))
        src_md5[rel] = _md5(pth) if os.path.exists(pth) else f"NOT FOUND at {pth}"
    src_md5["taniteval/tools/cost_surface_probe.py"] = _md5(os.path.abspath(__file__))
    src_md5["_tanitad_imported_from"] = os.path.abspath(tanitad.__file__)

    return {"tool": "taniteval/tools/cost_surface_probe.py",
            "tier": "T0",
            "tier_note": "cost/WM diagnostic anchored at t0; turn_frac is a property of "
                         "the COST, never of the car. Any driving claim needs the T1 "
                         "adapter (taniteval/tools/refav1_arm.py).",
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": prov, "device": str(dev),
            "grid": {"n_kappa": int(k_axis.size), "n_accel": int(a_axis.size),
                     "n_candidates": int(grid.shape[0]),
                     "kappa_axis": [float(x) for x in k_axis],
                     "accel_axis": [float(x) for x in a_axis],
                     "horizon": H, "dt": float(cfg.op_dt),
                     "turn_threshold_kappa": TURN_THRESHOLD,
                     "gt_turn_deg_threshold": GT_TURN_DEG},
            "plan_cfg": {k: getattr(pc, k) for k in
                         ("n_samples", "n_iters", "n_elites", "horizon", "dt",
                          "a_max", "kappa_max", "seed")},
            "weights": {"goal": 1.0, "jerk": W_JERK, "kappa": W_KAPPA, "vend": W_VEND,
                        "target_speed_passed": a.target_speed is not None},
            "wheelbase": float(a.wheelbase), "wheelbase_sweep": list(L_SWEEP),
            "controls": {"C7_as_command_agreement": c7},
            "gates": gates, "source_md5": src_md5,
            "n_windows": len(rows),
            "wallclock_s": round(time.time() - t_start, 1),
            "surfaces": surfaces, "rows": rows}


# --------------------------------------------------------------------------- #
# analysis — PAIRED EPISODE-CLUSTER BOOTSTRAP (never overlapping_holdout_se)     #
# --------------------------------------------------------------------------- #
def _boot_mean(vals: np.ndarray, ep_ids: np.ndarray, n_boot: int, seed: int) -> dict:
    """Mean of a per-window quantity with a PAIRED EPISODE-CLUSTER bootstrap.

    Episodes are the cluster (windows inside one episode are not independent). A DELTA
    between two arms is passed in as a per-window difference, which is what makes the
    interval paired: both arms are recomputed on the same resample by construction.
    ⛔ `overlapping_holdout_se` is never used: it is not a valid SE and it BIASES the
    point estimate (CLAUDE.md, 27-dump measurement).
    """
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return {"point": None, "lo": None, "hi": None, "n": 0}
    eps = np.unique(ep_ids)
    idx = [np.flatnonzero(ep_ids == e) for e in eps]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(eps), size=(n_boot, len(eps)))
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sel = np.concatenate([idx[j] for j in draws[b]])
        out[b] = vals[sel].mean()
    return {"point": float(vals.mean()),
            "lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5)),
            "n": int(vals.size), "n_episodes": int(len(eps)), "n_boot": int(n_boot),
            "estimator": "paired episode-cluster bootstrap (2.5/97.5 pct over "
                         "episode resamples)"}


#: every per-window scalar that gets a median in the summary. Hoisted to a module
#: constant so the test fixture can enumerate it and a later addition cannot leave a
#: stale synthetic row silently passing.
ROW_MEDIAN_KEYS = tuple(
    f"{t}{k}" for k in (
        "_goal_kappa_range", "_goal_a_range", "_goal_gain_at_kturn",
        "_goal_ptp_in_ulps", "_goal_ptp_f64", "_goal_n_distinct_f32",
        "_disp_kappa_max", "_disp_a_max", "_disp_ratio_kappa_over_a",
        "_goal_gain_at_kturn_f64", "_goal_kappa_range_in_ulps",
        "_goal_a_range_in_ulps", "_goal_kappa_range_f64", "_goal_a_range_f64",
        "_potency_ratio_f64", "_f64_over_f32_ptp")
    for t in ("A", "B")
) + ("potency_ratio_A", "potency_ratio_B", "B_over_A_kappa_range")


#: the SHIPPED-arithmetic arms (what the planner actually does) and the exact-cosine
#: arms (factor (iv): is the shipped float32 cosine destroying a real response?)
ARMS32 = ("A_full", "A_nopen", "B_full", "B_nopen")
ARMS64 = ("A_full_f64", "A_nopen_f64", "B_full_f64", "B_nopen_f64")
ARMS = ARMS32 + ARMS64


def summarize(res: dict, n_boot: int = 10000, seed: int = 0) -> dict:
    rows = res["rows"]
    strata = {
        "all": lambda r: True,
        "live_goal": lambda r: not r["zero_goal"],
        "zero_goal": lambda r: bool(r["zero_goal"]),
        "goal_turn": lambda r: float(r.get("goal_kappa_max") or 0.0) > 0.0,
        "human_turns": lambda r: (r.get("gt_turn_deg") is not None
                                  and np.isfinite(r.get("gt_turn_deg", np.nan))
                                  and r["gt_turn_deg"] >= GT_TURN_DEG),
        "human_turns_live_goal": lambda r: (
            not r["zero_goal"] and r.get("gt_turn_deg") is not None
            and np.isfinite(r.get("gt_turn_deg", np.nan))
            and r["gt_turn_deg"] >= GT_TURN_DEG),
    }
    out: dict = {"strata": {}}
    for sname, f in strata.items():
        sub = [r for r in rows if f(r)]
        blk: dict = {"n": len(sub), "n_episodes": len({r["ep_file"] for r in sub})}
        if sub:
            ep = np.array([r["ep_file"] for r in sub])
            turn = {arm: np.array([1.0 if r[f"{arm}_turn"] else 0.0 for r in sub])
                    for arm in ARMS}
            for arm in ARMS:
                blk[f"turn_frac_{arm}"] = _boot_mean(turn[arm], ep, n_boot, seed)
            blk["share_iii_penalty"] = _boot_mean(
                turn["A_nopen"] - turn["A_full"], ep, n_boot, seed)
            blk["share_ii_boundary"] = _boot_mean(
                turn["B_full"] - turn["A_full"], ep, n_boot, seed)
            blk["share_ii_plus_iii"] = _boot_mean(
                turn["B_nopen"] - turn["A_full"], ep, n_boot, seed)
            blk["share_iv_f32_saturation"] = _boot_mean(
                turn["A_full_f64"] - turn["A_full"], ep, n_boot, seed)
            blk["share_ii_iii_iv_all"] = _boot_mean(
                turn["B_nopen_f64"] - turn["A_full"], ep, n_boot, seed)
            blk["share_ii_boundary_f64"] = _boot_mean(
                turn["B_full_f64"] - turn["A_full_f64"], ep, n_boot, seed)
            blk["share_iii_penalty_f64"] = _boot_mean(
                turn["A_nopen_f64"] - turn["A_full_f64"], ep, n_boot, seed)
            blk["residual_after_B_nopen_f64"] = _boot_mean(
                1.0 - turn["B_nopen_f64"], ep, n_boot, seed)
            blk["median_abs_kappa_star"] = {
                arm: float(np.median([abs(r[f"{arm}_kappa"]) for r in sub]))
                for arm in ARMS}
            blk["median_surface_ptp"] = {
                arm: float(np.median([r[f"{arm}_surface_ptp"] for r in sub]))
                for arm in ARMS}
            for key in ROW_MEDIAN_KEYS:
                v = np.array([r[key] for r in sub], dtype=np.float64)
                blk[f"median_{key}"] = float(np.median(v))
            blk["frac_goal_ulp_quantized_A"] = float(np.mean(
                [1.0 if r["A_goal_is_ulp_quantized"] else 0.0 for r in sub]))
            blk["frac_goal_ptp_below_2ulp_A"] = float(np.mean(
                [1.0 if r["A_goal_ptp_in_ulps"] <= 2.0 else 0.0 for r in sub]))
            blk["penalty_at_kturn"] = float(np.median(
                [r["A_penalty_at_kturn"] for r in sub]))
            for tag in ("A", "B"):
                blk[f"frac_gain_exceeds_penalty_{tag}"] = float(np.mean(
                    [1.0 if r[f"{tag}_goal_gain_at_kturn"] >
                     r[f"{tag}_penalty_at_kturn"] else 0.0 for r in sub]))
                blk[f"frac_gain_exceeds_penalty_{tag}_f64"] = float(np.mean(
                    [1.0 if r[f"{tag}_goal_gain_at_kturn_f64"] >
                     r[f"{tag}_penalty_at_kturn"] else 0.0 for r in sub]))
                blk[f"median_gain_over_penalty_{tag}_f64"] = float(np.median(
                    [r[f"{tag}_goal_gain_at_kturn_f64"] /
                     max(r[f"{tag}_penalty_at_kturn"], 1e-300) for r in sub]))
        out["strata"][sname] = blk

    # ---- the controls, aggregated --------------------------------------------
    def _mx(key):
        v = [r[key] for r in rows if key in r]
        return float(max(v)) if v else float("nan")

    zm = [r for r in rows if "C3_zero_model_goal_ptp" in r]
    c1 = res["gates"]["C1"]
    c1o = [g for g in c1 if g["label"] == "supplied_oracle_goal"]
    out["controls"] = {
        "C0_decomposition_identity": {
            "max_abs": _mx("C0_identity_max_abs"), "threshold": 1e-6,
            "pass": bool(_mx("C0_identity_max_abs") <= 1e-6),
            "reads": "c_goal + c_jerk + c_kappa + c_vend == c_total"},
        "C1_shipped_cost_gate": {
            "n": len(c1), "n_oracle_goal": len(c1o),
            "worst_rel_err": (max(g["rel_err"] for g in c1) if c1 else None),
            "worst_rel_err_oracle": (max(g["rel_err"] for g in c1o) if c1o else None),
            "max_cost_scale_oracle": (max(g["cost_scale"] for g in c1o) if c1o else None),
            "pass": bool(c1 and all(g["pass"] for g in c1)),
            "reads": "our re-scoring == PlanResult.baseline_costs (the SHIPPED "
                     "_cost_chunk), rel err <= 1e-6; run with an O(1) oracle goal so "
                     "the agreement is not trivially met by two ~0 numbers"},
        "C2_banked_winner_gate": {
            "n": len(res["gates"]["C2"]),
            "worst_max_abs_m": (max(g["max_abs_m"] for g in res["gates"]["C2"])
                                if res["gates"]["C2"] else None),
            "gate_m": BANKED_GATE_M,
            "pass": bool(res["gates"]["C2"] and
                         all(g["pass"] for g in res["gates"]["C2"])),
            "reads": "a real plan() re-run reproduces the banked cl trajectory"},
        "C3_zero_model": ({
            "n": len(zm),
            "goal_ptp_max": _mx("C3_zero_model_goal_ptp"),
            "penalty_recovery_max_abs": _mx("C3_penalty_recovery_max_abs"),
            "argmin_kappa_all_zero": bool(all(
                r["C3_zero_model_argmin_kappa"] == 0.0 for r in zm)),
            "pass": bool(_mx("C3_zero_model_goal_ptp") == 0.0 and
                         _mx("C3_penalty_recovery_max_abs") <= 1e-9 and
                         all(r["C3_zero_model_argmin_kappa"] == 0.0 for r in zm)),
            "reads": "a predictor that ignores actions -> c_goal EXACTLY constant in "
                     "kappa, so total(k)-total(0) == 0.05*k^2 exactly and argmin is 0"}
            if zm else {"n": 0, "pass": False, "reason": "not run"}),
        "C4_constant_cost": {
            "all_zero": bool(all(r["C4_all_zero"] for r in rows)),
            "argmin_index_always_0": bool(all(r["C4_argmin_index"] == 0
                                              for r in rows)),
            "pass": bool(all(r["C4_all_zero"] for r in rows) and
                         all(r["C4_argmin_index"] == 0 for r in rows)),
            "reads": "all weights zeroed -> total == 0 everywhere, argmin = index 0"},
        "C5_channel0_invariance": {
            "jerk_max_abs_diff": _mx("C5_jerk_max_abs_diff"),
            "a_marginal_max_abs_diff": _mx("C5_a_marginal_max_abs_diff"),
            "pass": bool(_mx("C5_jerk_max_abs_diff") == 0.0 and
                         _mx("C5_a_marginal_max_abs_diff") == 0.0),
            "reads": "A and B must agree EXACTLY on channel 0"},
        "C6_n_and_d": {"n_windows": len(rows),
                       "n_candidates": res["grid"]["n_candidates"],
                       "pass": True},
        "C7_as_command_agreement": res["controls"]["C7_as_command_agreement"],
    }
    # ⭐ THE SATURATION READ -- not a control, a RESULT, but it decides how every
    # argmin below may be read: an argmin taken on a surface whose whole range is a
    # couple of float32 ULPs is a coin flip, not a decision.
    out["f32_saturation"] = {
        "f32_cos_ulp": F32_COS_ULP,
        "n": len(rows),
        "frac_goal_ulp_quantized_A": float(np.mean(
            [1.0 if r["A_goal_is_ulp_quantized"] else 0.0 for r in rows])),
        "median_goal_ptp_in_ulps_A": float(np.median(
            [r["A_goal_ptp_in_ulps"] for r in rows])),
        "max_goal_ptp_in_ulps_A": float(np.max(
            [r["A_goal_ptp_in_ulps"] for r in rows])),
        "median_n_distinct_f32_A": float(np.median(
            [r["A_goal_n_distinct_f32"] for r in rows])),
        "median_goal_ptp_f64_A": float(np.median(
            [r["A_goal_ptp_f64"] for r in rows])),
        "median_goal_KAPPA_range_in_ulps_A": float(np.median(
            [r["A_goal_kappa_range_in_ulps"] for r in rows])),
        "median_goal_A_range_in_ulps_A": float(np.median(
            [r["A_goal_a_range_in_ulps"] for r in rows])),
        "median_goal_KAPPA_range_in_ulps_B": float(np.median(
            [r["B_goal_kappa_range_in_ulps"] for r in rows])),
        "median_f64_over_f32_ptp_A": float(np.median(
            [r["A_f64_over_f32_ptp"] for r in rows])),
        "median_penalty_at_kturn": float(np.median(
            [r["A_penalty_at_kturn"] for r in rows])),
        "reads": "the shipped goal term is float32 `1 - cosine_similarity`; its "
                 "representable steps just below 1.0 are 5.96e-08 apart. If the whole "
                 "grid's range is a small number of those steps, the modelled benefit "
                 "of ANY action is below the arithmetic's resolution and the cost is, "
                 "to float precision, jerk + curvature alone."}
    must_pass = ("C0_decomposition_identity", "C1_shipped_cost_gate",
                 "C3_zero_model", "C4_constant_cost", "C5_channel0_invariance")
    out["controls"]["PANEL_VOID"] = not all(
        out["controls"][k].get("pass") for k in must_pass)
    out["controls"]["must_pass"] = list(must_pass)

    # ---- factor (0): the degenerate goal --------------------------------------
    out["goal_degeneracy"] = {
        "n": len(rows),
        "n_zero_goal": int(sum(1 for r in rows if r["zero_goal"])),
        "frac_zero_goal": float(np.mean([1.0 if r["zero_goal"] else 0.0
                                         for r in rows])) if rows else None,
        "goal_lat_counts": _counts(rows, "goal_lat"),
        "goal_lon_counts": _counts(rows, "goal_lon"),
        "reads": "a zero canonical goal makes the cv candidate the EXACT global "
                 "minimum (T1 = 1-cos(x,x) = 0, and T1,T2,T3 >= 0), independently of "
                 "factors (i)-(iii)"}

    # ---- the named candidates -------------------------------------------------
    nm_keys = sorted({k[len("named_A_"):-len("_c_total")] for r in rows for k in r
                      if k.startswith("named_A_") and k.endswith("_c_total")})
    out["named_candidates"] = {}
    for nm in nm_keys:
        blk = {"n": 0}
        for tag in ("A", "B"):
            for term in ("c_total", "c_goal", "c_jerk", "c_kappa"):
                key = f"named_{tag}_{nm}_{term}"
                v = [r[key] for r in rows if key in r]
                if v:
                    blk[f"{tag}_{term}_median"] = float(np.median(v))
                    blk["n"] = len(v)
        sub = [r for r in rows if f"named_A_{nm}_c_total" in r
               and "named_A_cv_c_total" in r]
        if sub:
            for tag in ("A", "B"):
                blk[f"{tag}_beats_cv_frac"] = float(np.mean(
                    [1.0 if r[f"named_{tag}_{nm}_c_total"] <
                     r[f"named_{tag}_cv_c_total"] else 0.0 for r in sub]))
        out["named_candidates"][nm] = blk

    # ---- the wheelbase sensitivity -------------------------------------------
    lk = [k for k in (rows[0] if rows else {}) if k.startswith("Lsweep_")
          and k.endswith("_turn")]
    if lk:
        out["wheelbase_sensitivity"] = {
            k: float(np.mean([1.0 if r[k] else 0.0 for r in rows if k in r]))
            for k in sorted(lk)}
        out["wheelbase_sensitivity"]["B_full_at_2.9"] = \
            out["strata"]["all"].get("turn_frac_B_full", {}).get("point")
    return out


def _counts(rows, key):
    c: dict = {}
    for r in rows:
        c[str(r.get(key))] = c.get(str(r.get(key)), 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="R10 — refav1 planner cost-surface decomposition + attribution")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--nav", default=None)
    ap.add_argument("--banked-dump", default=None,
                    help="t1_dump dir for the C2 banked-winner gate")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lru", type=int, default=6)
    ap.add_argument("--episodes-n", type=int, default=0)
    ap.add_argument("--window-stride", type=int, default=10)
    ap.add_argument("--horizon-k", type=int, default=10)
    ap.add_argument("--n-kappa", type=int, default=21)
    ap.add_argument("--n-accel", type=int, default=9)
    ap.add_argument("--wheelbase", type=float, default=2.9,
                    help="ENCODING wheelbase L_enc (kinematic.STEER_WHEELBASE_M)")
    ap.add_argument("--wheelbase-sweep", action="store_true")
    ap.add_argument("--wheelbase-sweep-n", type=int, default=140)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--target-speed", type=float, default=None)
    ap.add_argument("--zero-model-n", type=int, default=140)
    ap.add_argument("--gate-n", type=int, default=5)
    ap.add_argument("--save-surfaces-n", type=int, default=6)
    ap.add_argument("--plan-seed", type=int, default=0)
    ap.add_argument("--plan-n-samples", type=int, default=None)
    ap.add_argument("--plan-n-iters", type=int, default=None)
    ap.add_argument("--plan-n-elites", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    res = run(a)
    res["summary"] = summarize(res, n_boot=a.n_boot)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    s = res["summary"]
    _p("")
    _p(f"[out] {a.out}  ({res['n_windows']} windows, {res['wallclock_s']} s)")
    _p(f"[controls] PANEL_VOID = {s['controls']['PANEL_VOID']}")
    for kk in s["controls"]["must_pass"] + ["C2_banked_winner_gate",
                                            "C7_as_command_agreement"]:
        _p(f"   {kk}: pass={s['controls'][kk].get('pass')}")
    gd = s["goal_degeneracy"]
    _p(f"[goal] zero-canonical-goal windows: {gd['n_zero_goal']}/{gd['n']}  "
       f"lat={gd['goal_lat_counts']}")
    sat = s["f32_saturation"]
    _p(f"[f32] goal-term range across the WHOLE grid: median "
       f"{sat['median_goal_ptp_in_ulps_A']:.2f} ULPs (max "
       f"{sat['max_goal_ptp_in_ulps_A']:.2f}); ULP-quantized on "
       f"{sat['frac_goal_ulp_quantized_A'] * 100:.0f}% of windows; float64 range "
       f"{sat['median_goal_ptp_f64_A']:.3e} vs the kappa penalty at k_turn "
       f"{sat['median_penalty_at_kturn']:.3e}")
    _p(f"[f32] per AXIS at the origin: kappa spans "
       f"{sat['median_goal_KAPPA_range_in_ulps_A']:.2f} ULPs (A) / "
       f"{sat['median_goal_KAPPA_range_in_ulps_B']:.2f} (B), accel spans "
       f"{sat['median_goal_A_range_in_ulps_A']:.2f} ULPs")
    for sname, blk in s["strata"].items():
        if not blk["n"]:
            _p(f"[{sname}] n=0")
            continue
        _p(f"[{sname}] n={blk['n']} (eps {blk['n_episodes']})")
        for grp, lbl in ((ARMS32, "f32(shipped)"), (ARMS64, "f64(exact)  ")):
            _p(f"    turn_frac {lbl} " + "  ".join(
                f"{arm}={blk['turn_frac_' + arm]['point']:.3f}" for arm in grp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
