"""TanitEval — **REF-C v1.0**: training-free COST re-ranking of REF-C's own fan.

THE QUESTION (one number): how much of REF-C's *selection* gap is recoverable
without training anything?

REF-C-XL (`refc-xl-30k`, step 29,999, ADE@2s 0.458) denoises ALL 256 anchors but
SELECTS with the t=0 classifier confidence over the UN-refined anchors — the
denoise passes' own confidences are discarded (`refs/refc.py::
AnchoredDiffusionDecoder.forward` returns `_, off`). Measured consequence
(`plan_fan.py`, ep11): selected 1.110 m vs oracle-in-fan 0.295 m, 65 % of frames
pick a plan >2x worse than one already in the fan. Flagship v1.5 reproduced the
same pathology on a different trunk.

**REF-C v1.0 shares its weights with `refc-xl-30k` BYTE-FOR-BYTE.** Nothing is
trained, nothing is fine-tuned, no parameter is added. The ONLY thing that
changes is the decode-time selection policy: instead of `argmax(anchor_logits)`
we re-rank the 256 already-refined proposals with the P2 planner cost. That is
what makes v1.0 a clean control for REF-C v1.2 (frozen decoder + a LEARNED
re-scorer): any delta measured here is attributable purely to the selection
rule, so v1.2's delta over v1.0 is exactly the value of *learning* the ranker.

REF-C hands us 256 refined candidates for free — a CEM population without running
CEM. So this is P2's machinery applied to REF-C's proposals:

    J = w_v (v_hat - v_target)^2 + w_c (accel^2 + jerk^2) + w_s steer_rate^2
        - w_p progress                              [planner_p2.cost_fn, VERBATIM]

WHAT IS REUSED, NEVER REWRITTEN
  * `planner_p2.cost_fn` + `planner_p2.W`   — the cost and its engineered weights
  * `pathspeed.step_speed / heading_deg`    — planned-speed + heading geometry
  * `tanitad.lake.vtarget.vtarget_v2`       — the FIXED VTARGET mint (5 s lookahead
      floor + explicit `valid` mask; the old `planner_p2.vtarget_for` path
      silently fell back to hold-speed on a chunk of windows)
  * `refc_eval.collect`'s exact decode call — nav=follow, v0 through the
      measurement encoder, `steps = cfg.decoder.diffusion_steps`
  * `bench.run`                             — the canonical 881-window harness,
      8 overlapping random episode holdouts (DEPRECATED, not a jackknife),
      so every number stays comparable
  * `closedloop.WHEELBASE`                  — bicycle constant for the steer read

THE ONE MODELLING ASSUMPTION (stated, not hidden). REF-C's fan is 4 waypoints
(0.5/1/1.5/2 s), but `cost_fn`'s comfort/smoothness terms are defined at the
10 Hz operative tick. We therefore DENSIFY each 4-point proposal to a 20-step
dt=0.1 path with a cubic spline through (0,0) + the 4 waypoints, CLAMPED at
t=0 to the observed entry speed v0 (so the first-step deceleration is real
information, not a spline artefact) and natural at t=2 s. The densifier is a
fixed linear operator (`spline_operator`), self-tested against a direct solve.
Sensitivity to this choice is reported: `natural` start-condition variant and a
`speed+progress only` (w_c = w_s = 0) term ablation, which needs no derivative
of the interpolant at all.

Run (GPU pass once, then all sweeps are CPU):
    python3 -m taniteval.refc_rerank dump      [--episodes 40]
    python3 -m taniteval.refc_rerank analyze
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from driving_diagnostic import (WP_STEPS, baseline_waypoints,  # noqa: E402
                                gt_ego_waypoints, net_heading_change_deg)
from taniteval import bench, data, loaders  # noqa: E402
from taniteval import pathspeed as ps  # noqa: E402
from taniteval import planner_p2 as p2  # noqa: E402
from taniteval.closedloop import WHEELBASE  # noqa: E402
from taniteval.registry import MODELS  # noqa: E402

RES = Path("/root/taniteval/results")
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
DUMP = RES / "fan_refc-xl-30k.pt"

DT = 0.1
K_MAX = max(WP_STEPS)                    # 20 steps = 2 s
WINDOW, STRIDE = 8, 8                    # canonical val protocol
BASE_KEY = "refc-xl-30k"                 # the trained baseline (weights)
KEY = "refc-v10"                         # THIS experiment
NAME = "REF-C v1.0 (cost re-rank, training-free)"

KNOT_T = np.array([0.0, 0.5, 1.0, 1.5, 2.0])          # (0,0) + the 4 horizons
QUERY_T = np.arange(1, K_MAX + 1) * DT                # 0.1 .. 2.0 s
V_FLOOR = 0.5                            # m/s: steer read is meaningless below
BRAKE_A = -0.5                           # m/s^2: GT braking-window threshold
EPS = 1e-8


# ======================================================================== #
# VTARGET — the FIXED mint (tanitad.lake.vtarget), imported not copied      #
# ======================================================================== #
def load_vtarget():
    """`vtarget_v2` from the lake. Falls back to a by-path load of the SAME
    file when `tanitad.lake.__init__` cannot import (it pulls the catalog /
    parquet stack, which the eval pod does not need). Returns (fn, source)."""
    try:
        from tanitad.lake.vtarget import vtarget_v2
        return vtarget_v2, "tanitad.lake.vtarget (package import)"
    except Exception as e:                                    # noqa: BLE001
        for p in (Path("/root/TanitAD/stack/tanitad/lake/vtarget.py"),
                  Path("/root/vendor/vtarget.py")):
            if p.exists():
                spec = importlib.util.spec_from_file_location("_lake_vt", p)
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                return m.vtarget_v2, f"{p} (by-path; pkg import: {type(e).__name__})"
        raise RuntimeError(
            "tanitad.lake.vtarget not importable and no file copy found — "
            "REFUSING to fall back to planner_p2.vtarget_for (that is the "
            "defective mint this experiment is required to avoid)") from e


# ======================================================================== #
# Densifier — 4 waypoints -> a 20-step dt=0.1 path (fixed linear operator)  #
# ======================================================================== #
def _spline_solve(y: np.ndarray, d0: float | None) -> np.ndarray:
    """Cubic spline through (KNOT_T, y); clamped y'(0)=d0 (or natural if None),
    natural at the right end. Returns the values at QUERY_T."""
    h = float(KNOT_T[1] - KNOT_T[0])
    n = len(KNOT_T)
    A = np.zeros((n, n))
    b = np.zeros(n)
    if d0 is None:
        A[0, 0] = 1.0                                       # natural left
    else:
        A[0, 0], A[0, 1] = 2 * h, h                         # clamped left
        b[0] = 6.0 * ((y[1] - y[0]) / h - d0)
    for k in range(1, n - 1):
        A[k, k - 1], A[k, k], A[k, k + 1] = h, 4 * h, h
        b[k] = 6.0 * (y[k + 1] - 2 * y[k] + y[k - 1]) / h
    A[n - 1, n - 1] = 1.0                                   # natural right
    M = np.linalg.solve(A, b)                               # second derivatives
    out = np.empty(len(QUERY_T))
    for i, t in enumerate(QUERY_T):
        k = min(int(np.searchsorted(KNOT_T, t, side="right") - 1), n - 2)
        a_, b_ = KNOT_T[k + 1] - t, t - KNOT_T[k]
        out[i] = (M[k] * a_ ** 3 / (6 * h) + M[k + 1] * b_ ** 3 / (6 * h)
                  + (y[k] / h - M[k] * h / 6) * a_
                  + (y[k + 1] / h - M[k + 1] * h / 6) * b_)
    return out


def spline_operator(clamped: bool = True) -> torch.Tensor:
    """The densifier as a fixed [K_MAX, 6] matrix over (y_0..y_4, d0).

    The spline is linear in its knot values and its clamped end-derivative, so
    the whole densification is one matmul. Built by feeding basis vectors and
    VERIFIED against a direct solve on random input (fails loud)."""
    cols = []
    for k in range(len(KNOT_T)):
        e = np.zeros(len(KNOT_T))
        e[k] = 1.0
        cols.append(_spline_solve(e, 0.0 if clamped else None))
    cols.append(_spline_solve(np.zeros(len(KNOT_T)), 1.0) if clamped
                else np.zeros(len(QUERY_T)))
    M = np.stack(cols, axis=1)                                   # [K, 6]
    rng = np.random.default_rng(0)
    for _ in range(8):
        y = rng.normal(size=len(KNOT_T))
        y[0] = 0.0
        d0 = float(rng.normal()) if clamped else None
        ref = _spline_solve(y, d0)
        got = M @ np.concatenate([y, [d0 if clamped else 0.0]])
        assert np.allclose(ref, got, atol=1e-9), \
            f"spline operator != direct solve (max {np.abs(ref-got).max():.2e})"
    return torch.tensor(M, dtype=torch.float32)


def densify(fan: torch.Tensor, v0: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """[B,N,4,2] waypoints + [B] entry speed -> [B,N,K_MAX,2] dt=0.1 path.

    Knot 0 is the observed pose (0,0); the clamped start derivative is
    (v0, 0) — the ego frame's x axis IS the heading at the window end."""
    B, N = fan.shape[:2]
    zeros = torch.zeros(B, N, 1, 2, dtype=fan.dtype)
    knots = torch.cat([zeros, fan], dim=2)                       # [B,N,5,2]
    d0 = torch.zeros(B, N, 1, 2, dtype=fan.dtype)
    d0[..., 0, 0] = v0.view(B, 1).expand(B, N)                   # x' (0) = v0
    ctrl = torch.cat([knots, d0], dim=2)                         # [B,N,6,2]
    return torch.einsum("kc,bnci->bnki", M, ctrl)                # [B,N,K,2]


def path_to_controls(traj: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """[.,K,2] dense ego path + [.] entry speed -> exec_act [.,K,2] (steer,accel).

    The exact inverse of `closedloop.bicycle_integrate`:
      v_i    = ||p_i - p_{i-1}|| / dt                  (pathspeed.step_speed)
      a_i    = (v_i - v_{i-1}) / dt,  v_{-1} = v0      (observed entry speed —
               NOT pathspeed.step_accel's a_0 = 0 convention, which would blind
               the cost to exactly the first-step braking we are testing for)
      psi_i  = heading of the step tangent, psi_{-1} = 0 (ego forward)
      delta_i= atan(L * (psi_i - psi_{i-1}) / dt / max(v_i, V_FLOOR))"""
    v = ps.step_speed(traj)                                      # [.,K]
    v_prev = torch.cat([v0.view(-1, 1), v[:, :-1]], dim=1)
    accel = (v - v_prev) / DT
    psi = ps.heading_deg(traj) * (math.pi / 180.0)               # [.,K]
    psi_prev = torch.cat([torch.zeros_like(psi[:, :1]), psi[:, :-1]], dim=1)
    yaw_rate = (psi - psi_prev) / DT
    steer = torch.atan(WHEELBASE * yaw_rate / v.clamp_min(V_FLOOR))
    return torch.stack([steer, accel], dim=-1)                   # [.,K,2]


# ======================================================================== #
# PASS 1 (GPU) — decode the canonical val once, keep the FULL fan           #
# ======================================================================== #
def _entry(key):
    e = [m for m in MODELS if m["key"] == key]
    assert e, f"unknown model {key}"
    return e[0]


@torch.no_grad()
def dump(episodes=40, device="cuda", batch=8, out=DUMP):
    """One decode pass over the canonical val keeping every proposal.

    The model call is IDENTICAL to `refc_eval.collect` (same window/stride/
    batching, nav=follow, v0 to the measurement encoder, same `steps`), so the
    lam=0 row of the sweep must reproduce the published `refc-xl-30k` row."""
    vtarget_v2, vt_src = load_vtarget()
    e = _entry(BASE_KEY)
    t0 = time.time()
    L = loaders.load(e, device)
    model = L["model"]
    assert not getattr(model.cfg, "refc1", False), "refc1 ckpt: not time-wp comparable"
    horizons = tuple(model.cfg.trajectory.horizons)
    assert horizons == tuple(WP_STEPS), f"horizons {horizons} != {tuple(WP_STEPS)}"
    steps = model.cfg.decoder.diffusion_steps
    window = int(model.cfg.window)
    assert window == WINDOW, f"trained window {window} != protocol {WINDOW}"

    files = data.list_val_episodes(VAL, episodes)
    eps = data.load_frames(files)
    FAN, LOG, SEL, GT, CV, EID, SPD, HDG, V0, VT, VTOK, VTLK, AGT = (
        [], [], [], [], [], [], [], [], [], [], [], [], [])
    for ep in eps:
        fr, poses = ep.feats, ep.poses.float()
        T = fr.shape[0]
        starts = list(range(0, T - window - K_MAX, STRIDE))
        last_all = np.array([t + window - 1 for t in starts])
        vt, ok, look, _vs = vtarget_v2(poses[:, 3].numpy(), last_all)
        VT.append(torch.tensor(vt, dtype=torch.float32))
        VTOK.append(torch.tensor(ok))
        VTLK.append(torch.tensor(look))
        # GT longitudinal accel over the scored horizon — the braking stratum
        AGT.append((poses[last_all + K_MAX, 3] - poses[last_all, 3]) / (K_MAX * DT))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(fr[t:t + window])
                              for t in ch]).to(device).float().div_(255.0)
            v0 = poses[last, 3].to(device)
            o = model(fw, nav_cmd=None, v0=v0, steps=steps)
            logits = o["anchor_logits"].float().cpu()            # [b,N]
            fan = o["anchor_traj"].float().cpu()                 # [b,N,4,2]
            sel = o["sel_idx"].cpu()
            assert torch.equal(sel, logits.argmax(dim=1)), (
                "sel_idx != argmax(anchor_logits) — the ckpt is not scoring "
                "with the returned logits (grounded_selector?); the lam=0 row "
                "would not be the published baseline. Refusing.")
            wp = torch.stack([o["waypoints"][k] for k in WP_STEPS],
                             dim=1).cpu().float()                # [b,4,2]
            assert torch.allclose(wp, fan[torch.arange(len(ch)), sel], atol=1e-5), \
                "model waypoints != anchor_traj[sel_idx] — fan is not the decode"
            FAN.append(fan)
            LOG.append(logits)
            SEL.append(sel)
            GT.append(gt_ego_waypoints(ep.poses, last))
            CV.append(baseline_waypoints(ep.poses, last)["constant_velocity"])
            EID.extend([ep.episode_id] * len(ch))
            SPD.append(ep.poses[last, 3])
            HDG.append(net_heading_change_deg(ep.poses, last))
            V0.append(poses[last, 3])
        print(f"[dump] ep{ep.episode_id} {len(starts)} windows "
              f"({time.time()-t0:.0f}s)", flush=True)

    d = dict(fan=torch.cat(FAN), logits=torch.cat(LOG), sel=torch.cat(SEL),
             gt=torch.cat(GT).float(), cv=torch.cat(CV).float(), eid=EID,
             speed=torch.cat(SPD).float(), head_deg=torch.cat(HDG).float(),
             v0=torch.cat(V0).float(), v_target=torch.cat(VT),
             vt_valid=torch.cat(VTOK), vt_lookahead=torch.cat(VTLK),
             a_gt=torch.cat(AGT).float(), wp_steps=list(WP_STEPS),
             ckpt=e["ckpt"], ckpt_step=L["step"], steps=steps,
             vtarget_source=vt_src, n_anchors=int(torch.cat(LOG).shape[1]),
             wall_s=round(time.time() - t0, 1))
    RES.mkdir(parents=True, exist_ok=True)
    torch.save(d, out)
    print(f"[dump] {d['fan'].shape[0]} windows x {d['n_anchors']} anchors -> "
          f"{out} ({d['wall_s']}s)", flush=True)
    return d


# ======================================================================== #
# PASS 2 (CPU) — cost, sweeps, strata                                      #
# ======================================================================== #
def compute_cost(d, clamped=True, w=None, chunk=64):
    """J [B,N] for every proposal, via `planner_p2.cost_fn` VERBATIM."""
    w = dict(p2.W) if w is None else w
    M = spline_operator(clamped)
    fan, v0, vt = d["fan"], d["v0"], d["v_target"]
    B, N = fan.shape[:2]
    out = torch.empty(B, N)
    for i in range(0, B, chunk):
        f, v, t = fan[i:i + chunk], v0[i:i + chunk], vt[i:i + chunk]
        b = f.shape[0]
        traj = densify(f, v, M).reshape(b * N, K_MAX, 2)
        v0e = v.repeat_interleave(N)
        act = path_to_controls(traj, v0e)
        out[i:i + chunk] = p2.cost_fn(traj, act, t.repeat_interleave(N),
                                      w).reshape(b, N)
    return out


def _de(pred, gt):
    """Per-window ADE@2s = mean over the 4 horizons of the L2 error."""
    return torch.linalg.norm(pred - gt, dim=-1).mean(-1)


def _pick(fan, idx):
    return fan[torch.arange(fan.shape[0]), idx]


def _wins(d, pred):
    return {"pred": pred, "gt": d["gt"], "cv": d["cv"], "eid": d["eid"],
            "speed": d["speed"], "head_deg": d["head_deg"],
            "wp_steps": d["wp_steps"]}


def _strata_labels(d):
    q = torch.quantile(d["speed"], torch.tensor([1 / 3, 2 / 3]))   # bench parity
    spd = np.array(["low" if float(s) < float(q[0]) else
                    "high" if float(s) >= float(q[1]) else "med"
                    for s in d["speed"]])
    a = d["a_gt"].numpy()
    lon = np.where(a <= BRAKE_A, "braking",
                   np.where(a >= -BRAKE_A, "accelerating", "steady"))
    return spd, lon, np.array([f"{s}/{l}" for s, l in zip(spd, lon)])


def _score_row(d, idx, J, tag, extra=None):
    fan, gt = d["fan"], d["gt"]
    pred = _pick(fan, idx)
    de_sel = _de(pred, gt)
    de_all = torch.linalg.norm(fan - gt[:, None], dim=-1).mean(-1)  # [B,N]
    de_or = de_all.min(1).values
    res = bench.run(_wins(d, pred))
    row = dict(tag=tag,
               ade2s_heldout=res["heldout"]["model"]["ade_0_2s"]["mean"],
               ci95=res["heldout"]["model"]["ade_0_2s"]["ci95"],
               ade2s_full=round(float(de_sel.mean()), 4),
               fde2s_full=round(res["full_set"]["model"]["fde@2s"], 4),
               miss2m_full=round(res["full_set"]["model"]["miss_rate@2m"], 4),
               tms=round(res["full_set"]["model"]["tms_openloop"], 4),
               frac_sel_2x_worse=round(float((de_sel > 2 * de_or).float().mean()), 4),
               conf_rank_mean=round(float(
                   (d["logits"] > d["logits"].gather(1, idx[:, None])
                    ).float().sum(1).mean()), 2))
    if extra:
        row.update(extra)
    row["_de"] = de_sel
    row["_bench"] = res
    return row


def sweep(d, J, lams, tag_fmt="lam={lam}"):
    logp = torch.log_softmax(d["logits"], dim=1)
    Jz = (J - J.mean(1, keepdim=True)) / (J.std(1, keepdim=True) + EPS)
    rows = []
    for lam in lams:
        idx = (logp - lam * Jz).argmax(1)
        rows.append(_score_row(d, idx, J, tag_fmt.format(lam=lam),
                               dict(lam=lam)))
    return rows


def topk_sweep(d, J, ks):
    rows = []
    for k in ks:
        top = d["logits"].topk(k, dim=1).indices                   # [B,k]
        pick = J.gather(1, top).argmin(1)
        idx = top.gather(1, pick[:, None]).squeeze(1)
        rows.append(_score_row(d, idx, J, f"topK={k}", dict(topk=k)))
    return rows


def stratum_table(d, base_de, new_de):
    spd, lon, cross = _strata_labels(d)
    gt, fan = d["gt"], d["fan"]
    de_or = torch.linalg.norm(fan - gt[:, None], dim=-1).mean(-1).min(1).values
    out = {}
    for name, labels in (("speed", spd), ("longitudinal", lon), ("cross", cross)):
        t = {}
        for lab in sorted(set(labels.tolist())):
            m = torch.tensor(labels == lab)
            if int(m.sum()) < 5:
                continue
            t[lab] = dict(n=int(m.sum()),
                          base=round(float(base_de[m].mean()), 4),
                          rerank=round(float(new_de[m].mean()), 4),
                          delta=round(float((new_de[m] - base_de[m]).mean()), 4),
                          oracle=round(float(de_or[m].mean()), 4),
                          gap_recovered_pct=round(100.0 * float(
                              (base_de[m].mean() - new_de[m].mean())
                              / (base_de[m].mean() - de_or[m].mean() + EPS)), 1))
        out[name] = t
    return out


def analyze(dump_path=DUMP, out=None):
    d = torch.load(dump_path, map_location="cpu", weights_only=False)
    fan, gt, logits = d["fan"], d["gt"], d["logits"]
    B, N = fan.shape[:2]
    de_all = torch.linalg.norm(fan - gt[:, None], dim=-1).mean(-1)      # [B,N]
    de_or = de_all.min(1).values
    base_idx = logits.argmax(1)
    assert torch.equal(base_idx, d["sel"]), "argmax != recorded sel_idx"

    J = compute_cost(d, clamped=True)
    Jn = compute_cost(d, clamped=False)
    w_vp = dict(p2.W)
    w_vp["c"], w_vp["s"] = 0.0, 0.0
    Jvp = compute_cost(d, clamped=True, w=w_vp)

    lams = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
            5.0, 10.0, 30.0, 100.0, 1e6]
    rows = sweep(d, J, lams)
    base = rows[0]
    assert int(base["conf_rank_mean"]) == 0, "lam=0 is not pure-confidence"
    best = min(rows, key=lambda r: r["ade2s_full"])
    tk = topk_sweep(d, J, [1, 2, 3, 4, 8, 16, 32, 64, 128, N])
    best_tk = min(tk, key=lambda r: r["ade2s_full"])
    alt_natural = sweep(d, Jn, lams)
    alt_vp = sweep(d, Jvp, lams)

    ade_base = float(base["_de"].mean())
    ade_or = float(de_or.mean())

    def rec(x):                       # fraction of the selection gap recovered
        return round(100.0 * (ade_base - x) / (ade_base - ade_or), 2)

    res = dict(
        key=KEY, name=NAME, kind="training-free decode-time re-rank",
        weights_from=BASE_KEY, weights_note=(
            "byte-for-byte the refc-xl-30k checkpoint; zero parameters added, "
            "zero training steps. The only change is the selection rule."),
        ckpt=d["ckpt"], ckpt_step=d["ckpt_step"], diffusion_steps=d["steps"],
        n_windows=B, n_anchors=N, vtarget_source=d["vtarget_source"],
        cost_weights=dict(p2.W),
        vtarget=dict(
            frac_invalid=round(float((~d["vt_valid"]).float().mean()), 4),
            frac_invalid_pct=round(100.0 * float((~d["vt_valid"]).float().mean()), 2),
            fallback="v0-hold (smoothed current speed) where valid=False",
            mean_lookahead_steps=round(float(d["vt_lookahead"].float().mean()), 1),
            mean_vtarget=round(float(d["v_target"].mean()), 3),
            mean_v0=round(float(d["v0"].mean()), 3),
            mean_vt_minus_v0=round(float((d["v_target"] - d["v0"]).mean()), 3)),
        headline=dict(
            ade2s_full_baseline=round(ade_base, 4),
            ade2s_full_oracle_in_fan=round(ade_or, 4),
            ade2s_full_best_rerank=round(best["ade2s_full"], 4),
            ade2s_full_pure_cost=round(rows[-1]["ade2s_full"], 4),
            ade2s_full_best_topk=round(best_tk["ade2s_full"], 4),
            ade2s_heldout_baseline=base["ade2s_heldout"],
            ade2s_heldout_best_rerank=best["ade2s_heldout"],
            ade2s_heldout_oracle=None,
            gap_recovered_pct_best=rec(best["ade2s_full"]),
            gap_recovered_pct_pure_cost=rec(rows[-1]["ade2s_full"]),
            gap_recovered_pct_best_topk=rec(best_tk["ade2s_full"]),
            best_lam=best["lam"], best_topk=best_tk["topk"],
            frac_sel_2x_worse_before=base["frac_sel_2x_worse"],
            frac_sel_2x_worse_after=best["frac_sel_2x_worse"],
            frac_sel_2x_worse_pure_cost=rows[-1]["frac_sel_2x_worse"]),
        curve_lambda=[{k: v for k, v in r.items() if not k.startswith("_")}
                      for r in rows],
        curve_topk=[{k: v for k, v in r.items() if not k.startswith("_")}
                    for r in tk],
        sensitivity=dict(
            natural_spline=[{k: v for k, v in r.items() if not k.startswith("_")}
                            for r in alt_natural],
            speed_progress_only=[{k: v for k, v in r.items()
                                  if not k.startswith("_")} for r in alt_vp]),
        strata_best=stratum_table(d, base["_de"], best["_de"]),
        strata_pure_cost=stratum_table(d, base["_de"], rows[-1]["_de"]),
        bench_best=best["_bench"],
        protocol=dict(val=VAL, window=WINDOW, stride=STRIDE,
                      wp_steps=list(WP_STEPS), n_splits=8, val_frac=0.2,
                      statistic="overlapping_holdout_se, 8 random 20% holdouts "
                                "(DEPRECATED, not a jackknife) (heldout) "
                                "+ plain mean over all windows (full_set)",
                      claim_strength="open-loop / weak (arXiv:2605.00066)"))
    # heldout oracle, for the honest ceiling statement
    res["headline"]["ade2s_heldout_oracle"] = bench.run(
        _wins(d, _pick(fan, de_all.argmin(1))))["heldout"]["model"]["ade_0_2s"]["mean"]

    out = out or (RES / f"{KEY}.json")
    Path(out).write_text(json.dumps(res, indent=2, default=str))
    h = res["headline"]
    print(f"\n[{KEY}] {NAME}\n"
          f"  baseline (argmax-conf)  ADE@2s full {h['ade2s_full_baseline']:.4f} "
          f"| heldout {h['ade2s_heldout_baseline']:.4f}\n"
          f"  oracle-in-fan           ADE@2s full {h['ade2s_full_oracle_in_fan']:.4f} "
          f"| heldout {h['ade2s_heldout_oracle']:.4f}\n"
          f"  best blend  lam={h['best_lam']:<8} ADE@2s full "
          f"{h['ade2s_full_best_rerank']:.4f}  -> {h['gap_recovered_pct_best']}% "
          f"of the gap\n"
          f"  pure cost               ADE@2s full {h['ade2s_full_pure_cost']:.4f}"
          f"  -> {h['gap_recovered_pct_pure_cost']}%\n"
          f"  best top-K={h['best_topk']:<12}ADE@2s full "
          f"{h['ade2s_full_best_topk']:.4f}  -> {h['gap_recovered_pct_best_topk']}%\n"
          f"  frac_sel_2x_worse {h['frac_sel_2x_worse_before']} -> "
          f"{h['frac_sel_2x_worse_after']}\n"
          f"  VTARGET invalid (v0-hold fallback): "
          f"{res['vtarget']['frac_invalid_pct']}%\n  -> {out}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "analyze", "diag"])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(a.episodes, a.device, a.batch)
    elif a.cmd == "diag":
        diag()
    else:
        analyze()




# ======================================================================== #
# DIAGNOSTICS — why the re-rank behaves as it does (CPU, from the dump)     #
# ======================================================================== #
def _spearman_rows(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean over rows of Spearman rho between a[i] and b[i] ([B,N] each)."""
    ra = a.argsort(1).argsort(1).float()
    rb = b.argsort(1).argsort(1).float()
    ra = ra - ra.mean(1, keepdim=True)
    rb = rb - rb.mean(1, keepdim=True)
    rho = (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1) + EPS)
    return round(float(rho.mean()), 4)


def _pct_rank(score: torch.Tensor, idx: torch.Tensor) -> float:
    """Mean percentile rank (0 = best) of idx under `score` (lower = better)."""
    v = score.gather(1, idx[:, None])
    return round(float((score < v).float().mean(1).mean()), 4)


def diag(dump_path=DUMP, out=None):
    """The three questions the headline negative raises:
       (1) is the fan's confidence even contestable?      -> sharpness
       (2) does the cost carry ANY ranking signal?        -> rank correlation
       (3) is the oracle reachable in principle?          -> min-of-K + a
           GT-informed LONGITUDINAL-only ranker (the ceiling for any cost that
           only knows target speed) and the confidence-gated oracle (the ceiling
           for a learned re-scorer over the top-K, i.e. REF-C v1.2)."""
    d = torch.load(dump_path, map_location="cpu", weights_only=False)
    fan, gt, logits = d["fan"], d["gt"], d["logits"]
    B, N = fan.shape[:2]
    de_all = torch.linalg.norm(fan - gt[:, None], dim=-1).mean(-1)      # [B,N]
    de_or = de_all.min(1).values
    or_idx = de_all.argmin(1)
    base_idx = logits.argmax(1)
    p = torch.softmax(logits, 1)
    J = compute_cost(d, clamped=True)
    w_vp = dict(p2.W)
    w_vp["c"], w_vp["s"] = 0.0, 0.0
    Jvp = compute_cost(d, clamped=True, w=w_vp)

    # --- (3a) min-of-K oracle: how much of the 0.16 m is a sampling artefact?
    g = torch.Generator().manual_seed(0)
    ok = {}
    kk = 1
    while kk <= N:
        acc = []
        for _ in range(16 if kk < N else 1):
            sub = torch.randint(0, N, (B, kk), generator=g)
            acc.append(float(de_all.gather(1, sub).min(1).values.mean()))
        ok[kk] = round(float(np.mean(acc)), 4)
        kk *= 2

    # --- (3b) oracle WITHIN the top-K confidence set (the v1.2 ceiling) ------
    ow = {}
    kk = 1
    while kk <= N:
        top = logits.topk(kk, 1).indices
        ow[kk] = round(float(de_all.gather(1, top).min(1).values.mean()), 4)
        kk *= 2

    # --- (3c) GT-informed rankers: longitudinal-only vs lateral-only --------
    flat = fan.reshape(B * N, fan.shape[2], 2)
    gtr = gt[:, None].expand(-1, N, -1, -1).reshape(B * N, fan.shape[2], 2)
    along, cross = ps.frenet_residual(flat, gtr)
    along = along.abs().mean(-1).reshape(B, N)
    cross = cross.abs().mean(-1).reshape(B, N)
    lon_idx, lat_idx = along.argmin(1), cross.argmin(1)
    # GT mean speed over the scored horizon; the ceiling of a perfect
    # target-speed tracker that knows nothing about the path
    v_gt = ps.arclength(gt)[:, -1] / (K_MAX * DT)                       # [B]
    v_fan = ps.arclength(flat)[:, -1].reshape(B, N) / (K_MAX * DT)
    vsp_idx = (v_fan - v_gt[:, None]).abs().argmin(1)

    res = dict(
        key=KEY, n_windows=B, n_anchors=N,
        confidence_sharpness=dict(
            top1_prob_mean=round(float(p.max(1).values.mean()), 4),
            top1_prob_median=round(float(p.max(1).values.median()), 4),
            top1_prob_p10=round(float(p.max(1).values.quantile(0.1)), 4),
            top1_minus_top2_logit=round(float(
                (logits.topk(2, 1).values[:, 0]
                 - logits.topk(2, 1).values[:, 1]).mean()), 3),
            entropy_nats=round(float(
                -(p * p.clamp_min(1e-12).log()).sum(1).mean()), 4),
            n_modes_gt_1pct=round(float((p > 0.01).float().sum(1).mean()), 2)),
        rank_signal=dict(
            spearman_cost_vs_ade=_spearman_rows(J, de_all),
            spearman_cost_vp_vs_ade=_spearman_rows(Jvp, de_all),
            spearman_negconf_vs_ade=_spearman_rows(-logits, de_all),
            oracle_pct_rank_under_cost=_pct_rank(J, or_idx),
            oracle_pct_rank_under_negconf=_pct_rank(-logits, or_idx),
            selected_pct_rank_under_cost=_pct_rank(J, base_idx),
            note="pct rank 0 = ranked first (best); 0.5 = chance"),
        oracle_min_of_K=ok,
        oracle_within_topK_conf=ow,
        gt_informed_ceilings=dict(
            baseline_ade=round(float(de_all.gather(
                1, base_idx[:, None]).mean()), 4),
            oracle_full_ade=round(float(de_or.mean()), 4),
            longitudinal_only_ranker_ade=round(float(de_all.gather(
                1, lon_idx[:, None]).mean()), 4),
            lateral_only_ranker_ade=round(float(de_all.gather(
                1, lat_idx[:, None]).mean()), 4),
            gt_speed_matched_ranker_ade=round(float(de_all.gather(
                1, vsp_idx[:, None]).mean()), 4),
            note="ALL of these read the ground truth — they are unreachable "
                 "ceilings, reported to say WHICH information a GT-free ranker "
                 "would have to recover"),
        error_decomposition=dict(
            selected_along=round(float(along.gather(1, base_idx[:, None]).mean()), 4),
            selected_cross=round(float(cross.gather(1, base_idx[:, None]).mean()), 4),
            oracle_along=round(float(along.gather(1, or_idx[:, None]).mean()), 4),
            oracle_cross=round(float(cross.gather(1, or_idx[:, None]).mean()), 4)),
        vtarget_quality=dict(
            corr_vtarget_vs_gt_mean_speed=round(float(np.corrcoef(
                d["v_target"].numpy(), v_gt.numpy())[0, 1]), 4),
            corr_v0_vs_gt_mean_speed=round(float(np.corrcoef(
                d["v0"].numpy(), v_gt.numpy())[0, 1]), 4),
            mae_vtarget_vs_gt_mean_speed=round(float(
                (d["v_target"] - v_gt).abs().mean()), 4),
            mae_v0_vs_gt_mean_speed=round(float((d["v0"] - v_gt).abs().mean()), 4),
            note="VTARGET is a 10-20 s free-flow SET-SPEED, not a 2 s mean "
                 "speed; this quantifies how far apart the two are on this val"))
    # --- (4) is the cost better than CHANCE inside the top-K conf set? ------
    ref = {}
    kk = 1
    while kk <= 32:
        top = logits.topk(kk, 1).indices
        de_k = de_all.gather(1, top)                                   # [B,k]
        Jk = J.gather(1, top)
        pick = de_k.gather(1, Jk.argmin(1)[:, None]).squeeze(1)
        ref[kk] = dict(oracle=round(float(de_k.min(1).values.mean()), 4),
                       chance_mean=round(float(de_k.mean()), 4),
                       anti_oracle=round(float(de_k.max(1).values.mean()), 4),
                       cost_pick=round(float(pick.mean()), 4),
                       conf_pick=round(float(de_k[:, 0].mean()), 4))
        kk *= 2
    res["topK_reference"] = ref
    res["topK_reference_note"] = (
        "cost_pick vs chance_mean is the ONLY fair read on whether the cost "
        "carries selection signal at the top of the confidence ranking; "
        "conf_pick is the deployed baseline (always the top-1).")

    out = out or (RES / f"diag_{KEY}.json")
    Path(out).write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
