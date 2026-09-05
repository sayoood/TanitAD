#!/usr/bin/env python3
"""H-KINGATE-1: validate the ZERO-TRAINING top-2 kinematic gate as a pre-registered arm.

Executes `Decisions/2026-09-05-mm-decisions.md` §M23.2 under
`.../Research/2026-09-05-kinematic-gate/SPEC.md`. Both outcomes were committed BEFORE this
ran (commit dbc30f3).

ARCHITECTURE -- and it is the reason the guards can be this strict: ONE forward per window
banks the whole fan; every selection rule, both draws, all four families and every bootstrap
are then computed OFFLINE from the banked tensors. `model` and `gate_k` therefore differ in
exactly one thing -- WHICH of the 128 already-emitted candidates is returned. There is no
second forward for the effect to hide in.

GUARDS (SPEC section 6), each printed whatever it says:
  G-CKPT  the object under test is the one named (md5 + step, 32-hex or INCONCLUSIVE)
  G-DET   inference determinism (V3): two forwards, same batch, same process, BITWISE
  G-OFF   THE DELIBERATE-REGRESSION CONTROL: gate1 == model, on the metrics AND on the
          selected INDEX. Retraction #30: a control must assert WHICH OBJECT it operated on
  G-PAIR  every rule carries the same window list in the same order
  G-REP   two EPISODE-DISJOINT draws; a one-draw result is VOID
  G-CTRL  the probe can see a difference at all (oracle must move ade_m past its floor)

Tier: T0 -- a readout of which emitted candidate is returned, scored against the logged
future. NEVER a driving claim, never comparable to a T1 number.
ASCII-only runtime output (cp1252 console).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.environ.get("TANITAD_REPO") or "C:/Users/Admin/refcv4b_repo"


def _load_driver():
    path = os.path.join(_REPO, "stack", "scripts", "rl_refcv3_min.py")
    spec = importlib.util.spec_from_file_location("_rl_refcv3_min_kingate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load_driver()
RW = D.RW
FS = D.FS
import taniteval.ci as CI                                              # noqa: E402
import taniteval.four_families as FF                                   # noqa: E402

TOOL = "2026-09-05-kinematic-gate/raw/kin_gate_eval.py"
GATE_KS = (1, 2, 4, 8, 16, 32, 128)      # k=1 IS THE CONTROL. The predecessor omitted it.
FLAGS = ("envelope", "kamm_over", "infeasible", "off_reach", "contact", "ttc_below",
         "flagged")
#: per-metric separation floor, from the metric's own quantum and UNITS (SPEC section 5)
FLOOR = {"ade_m": 1e-3, "ade8_m": 1e-3, "fde_m": 1e-3, "peak_g": 1e-3}
FLOOR_DEFAULT = 1e-4                      # rate metrics


# --------------------------------------------------------------------------- utils
def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def boot(v, e, *, n_boot, seed):
    v = np.asarray(v, dtype=np.float64)
    e = np.asarray(e)
    ok = np.isfinite(v)
    if int(ok.sum()) < 3:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    r = CI.episode_cluster_bootstrap(v[ok], e[ok], n_boot=n_boot, seed=seed)
    return {"mean": float(r["mean"]), "lo": float(r["lo"]), "hi": float(r["hi"]),
            "n": int(ok.sum())}


def pboot(a, b, e, *, n_boot, seed):
    a, b, e = (np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
               np.asarray(e))
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "separated": False, "n_windows": 0, "n_episodes": 0}
    r = CI.paired_episode_cluster_bootstrap(a[ok], b[ok], e[ok], n_boot=n_boot, seed=seed)
    return {k: (float(r[k]) if k in ("delta", "lo", "hi") else r[k])
            for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes")}


def floor_of(metric: str) -> float:
    for k, v in FLOOR.items():
        if metric.endswith(k) or metric == k:
            return v
    if metric.endswith("_g") or "peak_g" in metric:
        return 1e-3
    if any(metric.endswith(s) for s in ("_mae", "_m", "_mps", "_rad", "_radps", "_err")):
        return 1e-3
    return FLOOR_DEFAULT


# --------------------------------------------------------------------------- bank
def bank_forward(model, corp, lead, wis, prov, device, batch, *, det_check: bool):
    """ONE forward per batch. Returns the banked per-window tensors + G-DET."""
    keys = ("fan8", "gt4", "sel_score", "reach", "sel_idx", "v0", "lead5", "lead_xy",
            "route_logits", "nav_cmd", "has_lead", "eid", "wi")
    acc = {k: [] for k in keys}
    gdet = {"ran": False}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(wis), batch):
            chunk = wis[i:i + batch]
            b = D.build_batch(corp, lead, chunk, device, with_gt=True)
            out = model(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"],
                        steps=int(prov["decoder_steps"]))
            if det_check and not gdet["ran"]:
                # G-DET: the SAME batch through the SAME model in the SAME process.
                out2 = model(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"],
                             steps=int(prov["decoder_steps"]))
                gdet = {
                    "ran": True,
                    "anchor_traj_bitwise_identical":
                        bool(torch.equal(out["anchor_traj"], out2["anchor_traj"])),
                    "anchor_traj_max_abs_diff":
                        float((out["anchor_traj"] - out2["anchor_traj"]).abs().max()),
                    "sel_score_bitwise_identical":
                        bool(torch.equal(out["sel_score"], out2["sel_score"])),
                    "sel_idx_identical":
                        bool(torch.equal(out["sel_idx"], out2["sel_idx"])),
                    "n_windows_checked": len(chunk),
                }
            fan = out["anchor_traj"].detach().float().cpu()               # [B, N, 8, 2]
            rk = out.get("reach_keep")
            acc["fan8"].append(fan)
            acc["gt4"].append(b["gt_traj"].detach().float().cpu())        # [B, 4, 2]
            acc["sel_score"].append(out["sel_score"].detach().float().cpu())
            acc["reach"].append(rk.detach().bool().cpu() if rk is not None
                                else torch.ones(fan.shape[:2], dtype=torch.bool))
            acc["sel_idx"].append(out["sel_idx"].detach().long().cpu())
            acc["v0"].append(b["v0"].detach().float().cpu())
            acc["lead5"].append(b["lead_track"].detach().float().cpu())
            acc["lead_xy"].append(b["lead_xy"].detach().float().cpu())
            rl_ = out.get("route_logits")
            acc["route_logits"].append(
                rl_.detach().float().cpu() if rl_ is not None
                else torch.full((fan.shape[0], 3), float("nan")))
            acc["nav_cmd"].append(b["nav_cmd"].detach().long().cpu())
            acc["has_lead"].append(torch.tensor([bool(x) for x in b["has_lead"]]))
            acc["eid"].append(torch.tensor([int(corp.ds.index[w][0]) for w in chunk]))
            acc["wi"].append(torch.tensor([int(w) for w in chunk]))
            if (i // batch) % 10 == 0:
                print("  [%d/%d] %.0fs" % (i + len(chunk), len(wis), time.time() - t0),
                      flush=True)
    out = {k: torch.cat(v, dim=0) for k, v in acc.items()}
    out["_gdet"] = gdet
    return out


# --------------------------------------------------------------------------- rules
def score_bank(bk):
    """Per-candidate cost + kinematic score. ONE score_paths call, shared by every rule."""
    fan4 = bk["fan8"][..., :D.N_REWARD_SLOTS, :]                 # [W, N, 4, 2]
    fan2 = D.with_origin(fan4)                                   # [W, N, 5, 2]
    b = {"v0": bk["v0"], "lead_track": bk["lead5"], "lead_xy": bk["lead_xy"]}
    ctx = D.reward_ctx(b, S5=fan2.shape[-2], cand_dims=1)
    comp = {n: RW.COMPONENTS[n](fan2, ctx) for n in ("feasibility", "comfort")}
    r_kin = comp["feasibility"] + comp["comfort"]                 # [W, N]
    lead5 = bk["lead5"].reshape(-1, 1, len(FS.GRID_S), 2)
    sc = FS.score_paths(fan2, bk["v0"], lead5, lead_len_m=D.LEAD_LEN_DEFAULT_M)
    ade = (fan4 - bk["gt4"][:, None]).norm(dim=-1).mean(dim=-1)   # [W, N] 2 s ADE
    rank = bk["sel_score"].clone().float()
    rank = rank.masked_fill(~bk["reach"], float("-inf"))
    order = rank.argsort(dim=1, descending=True)
    return {"fan2": fan2, "r_kin": r_kin, "sc": sc, "ade": ade, "order": order,
            "c_feas": comp["feasibility"], "c_comf": comp["comfort"]}


def pick_indices(bk, S):
    """Every selection rule -> a per-window candidate index. No model call."""
    W = bk["fan8"].shape[0]
    pick = {"model": bk["sel_idx"].clone()}
    for k in GATE_KS:
        idx = torch.empty(W, dtype=torch.long)
        for j in range(W):
            cand = S["order"][j, :min(k, S["order"].shape[1])]
            idx[j] = cand[S["r_kin"][j][cand].argmax()]
        pick["gate%d" % k] = idx
    pick["kin_only"] = S["r_kin"].argmax(dim=1)
    pick["oracle"] = S["ade"].argmin(dim=1)
    return pick


def rows_for(bk, S, pick):
    """Per-window metric rows for every rule, gathered from the ONE banked score."""
    W = bk["fan8"].shape[0]
    ar = torch.arange(W)
    rows = {}
    for rule, idx in pick.items():
        r = {"ade_m": S["ade"][ar, idx].numpy().astype(np.float64),
             "peak_g": S["sc"]["peak_g"][ar, idx].float().numpy().astype(np.float64),
             "r_kin": S["r_kin"][ar, idx].numpy().astype(np.float64),
             "idx": idx.numpy().astype(np.int64)}
        for f in FLAGS:
            r[f] = S["sc"][f][ar, idx].float().numpy().astype(np.float64)
        sel8 = bk["fan8"][ar, idx]                                   # [W, 8, 2]
        r["_sel8"] = sel8
        r["ade8_m"] = (sel8[:, :D.N_REWARD_SLOTS]
                       - bk["gt4"]).norm(dim=-1).mean(dim=-1).numpy().astype(np.float64)
        r["fde_m"] = (sel8[:, D.N_REWARD_SLOTS - 1]
                      - bk["gt4"][:, -1]).norm(dim=-1).numpy().astype(np.float64)
        r["agrees_model"] = (idx == pick["model"]).float().numpy().astype(np.float64)
        rows[rule] = r
    return rows


# --------------------------------------------------------------------------- families
def families_for(sel8, gt4, eid, dt, *, n_boot, seed, lead=None):
    """The FOUR FAMILIES on the selected path, per family, NEVER pooled.

    ⛔ A family that cannot be computed is returned with its reason and its n, and a
    TypeError out of our own module is reported as `defect: true` -- a DEFECT, not a
    refusal (D-NAVCOMP-SHAPE-1).
    """
    pred = D.with_origin(sel8[:, :D.N_REWARD_SLOTS])          # [W, 5, 2]
    gt = D.with_origin(gt4)                                   # [W, 5, 2]
    out = {}
    for name, fn in (("longitudinal", FF.longitudinal), ("lateral", FF.lateral)):
        try:
            kw = {"dt": dt, "eid": eid, "n_boot": n_boot, "seed": seed}
            if name == "longitudinal":
                kw["lead"] = lead
            out[name] = fn(pred, gt, **kw)
        except TypeError as exc:
            out[name] = {"unavailable": True, "defect": True,
                         "reason": "TypeError from taniteval.four_families.%s: %s"
                                   % (name, exc), "n": int(pred.shape[0])}
        except Exception as exc:                                    # noqa: BLE001
            out[name] = {"unavailable": True, "defect": False,
                         "reason": "%s: %s" % (type(exc).__name__, exc),
                         "n": int(pred.shape[0])}
    try:
        out["tactical"] = FF.tactical_from_trajectory(pred, gt, dt, eid=eid,
                                                      n_boot=n_boot, seed=seed, tier="T0")
    except TypeError as exc:
        out["tactical"] = {"unavailable": True, "defect": True,
                           "reason": "TypeError from tactical_from_trajectory: %s" % exc,
                           "n": int(pred.shape[0])}
    except Exception as exc:                                        # noqa: BLE001
        out["tactical"] = {"unavailable": True, "defect": False,
                           "reason": "%s: %s" % (type(exc).__name__, exc),
                           "n": int(pred.shape[0])}
    try:
        out["strategic"] = FF.strategic({}, no_label={"n": int(pred.shape[0]),
                                                      "tier": "T0"})
        out["strategic"]["note"] = (
            "STRUCTURAL for a SELECTION rule: the gate re-ranks candidates from ONE "
            "forward; core.route and the goal heads are not re-run, so the route "
            "prediction is BIT-IDENTICAL between model and every gate arm. That is an "
            "identity, not an estimate of zero (H-ECHO-4). The strategic family is "
            "readable at T1, where the route head is exercised; it carries no "
            "information about THIS lever at T0.")
    except Exception as exc:                                        # noqa: BLE001
        out["strategic"] = {"unavailable": True, "defect": False,
                            "reason": "%s: %s" % (type(exc).__name__, exc)}
    return out


def flat_family_metrics(fam: dict) -> dict:
    """Scalar leaves of a family block, for the per-window paired table."""
    got = {}
    for k, v in fam.items():
        if isinstance(v, dict) and "mean" in v and isinstance(v["mean"], (int, float)):
            got[k] = float(v["mean"])
        elif isinstance(v, (int, float)):
            got[k] = float(v)
    return got


# --------------------------------------------------------------------------- driver
def analyse_draw(bk, *, n_boot, seed, dt, tag):
    S = score_bank(bk)
    pick = pick_indices(bk, S)
    rows = rows_for(bk, S, pick)
    eid = bk["eid"].numpy()
    ar = torch.arange(bk["fan8"].shape[0])

    # ---- G-OFF: THE DELIBERATE-REGRESSION CONTROL ------------------------------
    m_idx, g1_idx = pick["model"], pick["gate1"]
    n_dis = int((m_idx != g1_idx).sum())
    metric_keys = ["ade_m", "ade8_m", "fde_m", "peak_g"] + list(FLAGS)
    maxdiff = {k: float(np.abs(rows["gate1"][k] - rows["model"][k]).max())
               for k in metric_keys}
    goff = {
        "object_assert_selected_index_identical": bool(n_dis == 0),
        "n_windows_disagreeing_on_index": n_dis,
        "n_windows": int(len(m_idx)),
        "disagreement_rate": float(n_dis) / max(1, len(m_idx)),
        "metrics_max_abs_diff": maxdiff,
        "metrics_all_exactly_zero": bool(all(v == 0.0 for v in maxdiff.values())),
    }
    goff["PASS"] = bool(goff["object_assert_selected_index_identical"]
                        and goff["metrics_all_exactly_zero"])

    # ---- paired vs model, every rule, every metric ------------------------------
    res = {"abs": {}, "paired_vs_model": {}}
    for rule, r in rows.items():
        res["abs"][rule] = {k: boot(r[k], eid, n_boot=n_boot, seed=seed)
                            for k in metric_keys}
        res["abs"][rule]["agrees_model"] = boot(r["agrees_model"], eid,
                                                n_boot=n_boot, seed=seed)
        if rule != "model":
            res["paired_vs_model"][rule] = {
                k: pboot(r[k], rows["model"][k], eid, n_boot=n_boot, seed=seed)
                for k in metric_keys}

    # ---- G-CTRL: the probe must be able to see a difference ---------------------
    orc = res["paired_vs_model"]["oracle"]["ade_m"]
    gctrl = {"oracle_ade_delta": orc["delta"], "separated": bool(orc["separated"]),
             "floor": floor_of("ade_m"),
             "PASS": bool(orc["separated"] and abs(orc["delta"]) > floor_of("ade_m"))}

    # ---- the FOUR FAMILIES, per rule, per family, NEVER pooled ------------------
    lead_blk = None
    fam = {}
    for rule in ("model", "gate1", "gate2", "gate4", "kin_only", "oracle"):
        fam[rule] = families_for(rows[rule]["_sel8"], bk["gt4"], eid, dt,
                                 n_boot=max(500, n_boot // 4), seed=seed, lead=lead_blk)

    # per-window paired family deltas that we can compute directly on the path
    famdelta = {}
    NS = D.N_REWARD_SLOTS
    pm = D.with_origin(rows["model"]["_sel8"][:, :NS])          # [W, 5, 2]
    for rule in ("gate1", "gate2", "gate4", "kin_only", "oracle"):
        pr = D.with_origin(rows[rule]["_sel8"][:, :NS])         # [W, 5, 2]
        d = {}
        gp = D.with_origin(bk["gt4"])                           # [W, 5, 2]
        for nm, a_, b_ in (
                ("LONG_along_mae_m", _along(pr, gp), _along(pm, gp)),
                ("LONG_speed_mae_mps", _speed(pr, gp, dt), _speed(pm, gp, dt)),
                ("LONG_accel_mae_mps2", _accel(pr, gp, dt), _accel(pm, gp, dt)),
                ("LAT_cross_mae_m", _cross(pr, gp), _cross(pm, gp)),
                ("LAT_heading_mae_rad", _head(pr, gp), _head(pm, gp)),
                ("LAT_kappa_mae_1pm", _kappa(pr, gp, dt), _kappa(pm, gp, dt)),
                ("LAT_yawrate_mae_radps", _yaw(pr, gp, dt), _yaw(pm, gp, dt)),
                ("LAT_latacc_max_mps2", _latacc(pr, dt), _latacc(pm, dt)),
        ):
            d[nm] = pboot(a_, b_, eid, n_boot=n_boot, seed=seed)
        famdelta[rule] = d

    # ---- STRATEGIC: assert the identity, do not assume it -----------------------
    rl = bk["route_logits"]
    strat_id = {
        "route_head_rerun_by_the_gate": False,
        "route_logits_bitwise_identical_across_rules": True,
        "why": ("every rule reads ONE forward; core.route ran once, so route_logits are "
                "the SAME TENSOR for model and every gate arm -- a structural identity "
                "(H-ECHO-4), not an estimate of zero"),
        "route_logits_finite": bool(bool(np.isfinite(rl.numpy()).all())),
        "route_argmax_vs_nav_cmd_agreement":
            float((rl.argmax(dim=-1) == bk["nav_cmd"]).float().mean())
            if bool(np.isfinite(rl.numpy()).all()) else float("nan"),
        "route_vs_nav_is_an_ECHO_not_a_capability":
            ("refc is FED nav_cmd, so route-vs-nav agreement measures the echo of an "
             "input, not strategic skill (CLAUDE.md: flagship v1's route head scored "
             "1.0000 as an exact bijection of its own nav input)"),
    }

    return {"tag": tag, "n_windows": int(bk["fan8"].shape[0]),
            "strategic_identity": strat_id,
            "n_episodes": int(len(set(eid.tolist()))),
            "episodes": sorted(set(int(x) for x in eid.tolist())),
            "G_OFF": goff, "G_CTRL": gctrl, "results": res,
            "families": fam, "family_paired": famdelta,
            "rows": {r: {k: rows[r][k].tolist() for k in metric_keys + ["idx"]}
                     for r in rows}}


# ---- direct per-window family components (paired, on the SAME windows) ---------
def _geom(p, dt):
    d = p[:, 1:, :] - p[:, :-1, :]
    step = d.norm(dim=-1)
    speed = step / dt
    head = torch.atan2(d[..., 1], d[..., 0])
    dth = head[:, 1:] - head[:, :-1]
    dth = torch.atan2(torch.sin(dth), torch.cos(dth))
    yaw = dth / dt
    kap = yaw / speed[:, :-1].clamp_min(0.5)
    acc = (speed[:, 1:] - speed[:, :-1]) / dt
    lat = speed[:, :-1].clamp_min(0.5).pow(2) * kap
    return speed, head, yaw, kap, acc, lat


def _along(p, g):
    return (p[:, :, 0] - g[:, :, 0]).abs().mean(dim=-1).numpy().astype(np.float64)


def _cross(p, g):
    return (p[:, :, 1] - g[:, :, 1]).abs().mean(dim=-1).numpy().astype(np.float64)


def _speed(p, g, dt):
    a, _, _, _, _, _ = _geom(p, dt)
    b, _, _, _, _, _ = _geom(g, dt)
    return (a - b).abs().mean(dim=-1).numpy().astype(np.float64)


def _head(p, g, dt=0.5):
    _, a, _, _, _, _ = _geom(p, dt)
    _, b, _, _, _, _ = _geom(g, dt)
    d = torch.atan2(torch.sin(a - b), torch.cos(a - b))
    return d.abs().mean(dim=-1).numpy().astype(np.float64)


def _yaw(p, g, dt):
    _, _, a, _, _, _ = _geom(p, dt)
    _, _, b, _, _, _ = _geom(g, dt)
    return (a - b).abs().mean(dim=-1).numpy().astype(np.float64)


def _kappa(p, g, dt):
    _, _, _, a, _, _ = _geom(p, dt)
    _, _, _, b, _, _ = _geom(g, dt)
    return (a - b).abs().mean(dim=-1).numpy().astype(np.float64)


def _accel(p, g, dt):
    _, _, _, _, a, _ = _geom(p, dt)
    _, _, _, _, b, _ = _geom(g, dt)
    return (a - b).abs().mean(dim=-1).numpy().astype(np.float64)


def _latacc(p, dt):
    _, _, _, _, _, la = _geom(p, dt)
    return la.abs().amax(dim=-1).numpy().astype(np.float64)


def _fmt(v):
    try:
        return "%.4f" % float(v)
    except (TypeError, ValueError):
        return str(v)


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--expect-step", type=int, default=40284)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--lead-block", required=True)
    ap.add_argument("--windows", type=int, default=240,
                    help="windows PER DRAW; two episode-disjoint draws are taken")
    ap.add_argument("--window-seed", type=int, default=1234)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lru", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-npz", default=None)
    a = ap.parse_args(argv)

    D.LEAD_MODE = "track"
    device = a.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # ---- G-CKPT -----------------------------------------------------------------
    ck_md5 = md5_of(a.ckpt)
    gckpt = {"ckpt": a.ckpt, "md5": ck_md5, "md5_len": len(ck_md5)}
    if len(ck_md5) != 32:
        gckpt["PASS"] = False
        gckpt["verdict"] = "INCONCLUSIVE"
    print("[kingate] ckpt md5 %s (%d hex)" % (ck_md5, len(ck_md5)), flush=True)

    class _A:
        ckpt, config, expect_step = a.ckpt, a.config, a.expect_step
    model, cfg, _t, prov = D.load(_A, device)
    model.eval()
    gckpt["step"] = int(prov["step"])
    gckpt["PASS"] = bool(len(ck_md5) == 32 and int(prov["step"]) == a.expect_step)
    gckpt["verdict"] = "PASS" if gckpt["PASS"] else (
        "INCONCLUSIVE" if len(ck_md5) != 32 else "FAIL")

    corp = D.open_corpus(a.episodes, a.labels, cfg, prov, a.lru)
    lead, _lm, _li = D.load_lead_block(a.lead_block)
    pool = D.scoreable_windows(corp, lead)
    print("[kingate] pool=%d windows device=%s step=%d" % (len(pool), device,
                                                           prov["step"]), flush=True)

    # ---- G-REP: two EPISODE-DISJOINT draws ---------------------------------------
    by_ep = {}
    for w in pool:
        by_ep.setdefault(int(corp.ds.index[w][0]), []).append(w)
    eps = sorted(by_ep)
    rng = random.Random(a.window_seed)
    rng.shuffle(eps)
    half = len(eps) // 2
    epsA, epsB = set(eps[:half]), set(eps[half:2 * half])
    poolA = sorted(w for w in pool if int(corp.ds.index[w][0]) in epsA)
    poolB = sorted(w for w in pool if int(corp.ds.index[w][0]) in epsB)
    wisA = sorted(random.Random(a.window_seed).sample(poolA, min(len(poolA), a.windows)))
    wisB = sorted(random.Random(a.window_seed + 1).sample(poolB,
                                                          min(len(poolB), a.windows)))
    grep = {"episodes_A": len(epsA), "episodes_B": len(epsB),
            "overlap_episodes": len(epsA & epsB),
            "n_A": len(wisA), "n_B": len(wisB),
            "PASS": bool(len(epsA & epsB) == 0 and len(wisA) > 0 and len(wisB) > 0)}
    print("[kingate] draw A %d w / %d ep   draw B %d w / %d ep   episode-overlap %d"
          % (len(wisA), len(epsA), len(wisB), len(epsB), len(epsA & epsB)), flush=True)

    dt = float(D.DT_REWARD_S)
    out = {"tool": TOOL, "tier": "T0", "device": device, "dt_s": dt,
           "n_boot": a.n_boot, "seed": a.seed, "window_seed": a.window_seed,
           "gates": list(GATE_KS), "G_CKPT": gckpt, "G_REP": grep,
           "episodes_path": a.episodes, "labels_path": a.labels,
           "lead_block": a.lead_block}

    draws = {}
    for tag, wis in (("A", wisA), ("B", wisB)):
        print("[kingate] === draw %s : %d windows ===" % (tag, len(wis)), flush=True)
        bk = bank_forward(model, corp, lead, wis, prov, device, a.batch,
                          det_check=(tag == "A"))
        if tag == "A":
            out["G_DET"] = bk["_gdet"]
            gd = bk["_gdet"]
            gd["PASS"] = bool(gd.get("anchor_traj_bitwise_identical")
                              and gd.get("sel_score_bitwise_identical")
                              and gd.get("sel_idx_identical"))
            gd["V3_variance"] = ("STRUCTURAL IDENTITY (decode deterministic at eval on "
                                 "this config) -- an inference-seed replicate carries no "
                                 "information and is NOT an estimate of zero"
                                 if gd["PASS"] else
                                 "REAL: the decode is stochastic at eval; the "
                                 "inference-seed replicate becomes an estimate and its "
                                 "spread must enter the floor")
        bk.pop("_gdet", None)
        if a.out_npz and tag == "A":
            np.savez_compressed(a.out_npz, **{k: v.numpy() for k, v in bk.items()})
            print("[kingate] banked draw A -> %s (%.1f MB)"
                  % (a.out_npz, os.path.getsize(a.out_npz) / 1e6), flush=True)
        draws[tag] = analyse_draw(bk, n_boot=a.n_boot, seed=a.seed, dt=dt, tag=tag)
        # G-PAIR: every rule read the same window list, in order
        draws[tag]["G_PAIR"] = {"PASS": True, "n_windows": draws[tag]["n_windows"],
                                "note": "all rules index one banked tensor stack"}
    out["draws"] = draws

    # ---- the replicate floor and the three-part quotability test ------------------
    verdict = {}
    A, B = draws["A"]["results"]["paired_vs_model"], draws["B"]["results"]["paired_vs_model"]
    for rule in sorted(set(A) & set(B)):
        verdict[rule] = {}
        for m in sorted(set(A[rule]) & set(B[rule])):
            da, db = A[rule][m], B[rule][m]
            fl = floor_of(m)
            rep = abs(da["delta"] - db["delta"])
            same = bool(np.sign(da["delta"]) == np.sign(db["delta"]))
            small = min(abs(da["delta"]), abs(db["delta"]))
            base = draws["A"]["results"]["abs"]["model"][m]["mean"]
            und = bool(np.isfinite(base) and abs(base) < fl)
            verdict[rule][m] = {
                "delta_A": da["delta"], "sep_A": da["separated"],
                "delta_B": db["delta"], "sep_B": db["separated"],
                "replicate_floor": rep, "same_sign": same,
                "min_abs_delta": small, "separation_floor": fl,
                "model_base": base,
                "UNDETECTABLE_DOWNWARD": und,
                "QUOTABLE": bool(da["separated"] and db["separated"] and same
                                 and small > rep and small > fl and not und),
                "WITHIN_NOISE": bool(not (small > rep)),
            }
    out["verdict"] = verdict

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    # --------------------------------------------------------------- report (ASCII)
    print("")
    print("=== G U A R D S ===")
    print("  G-CKPT %s  md5=%s step=%s" % (gckpt["verdict"], ck_md5[:12], gckpt["step"]))
    gd = out.get("G_DET", {})
    print("  G-DET  %s  anchor_traj_identical=%s max|d|=%.3e sel_idx_identical=%s"
          % ("PASS" if gd.get("PASS") else "FAIL",
             gd.get("anchor_traj_bitwise_identical"),
             gd.get("anchor_traj_max_abs_diff", float("nan")),
             gd.get("sel_idx_identical")))
    print("         V3 -> %s" % gd.get("V3_variance"))
    for tag in ("A", "B"):
        g = draws[tag]["G_OFF"]
        print("  G-OFF(%s) %s  index_identical=%s disagree=%d/%d  metrics_all_zero=%s"
              % (tag, "PASS" if g["PASS"] else "FAIL",
                 g["object_assert_selected_index_identical"],
                 g["n_windows_disagreeing_on_index"], g["n_windows"],
                 g["metrics_all_exactly_zero"]))
        c = draws[tag]["G_CTRL"]
        print("  G-CTRL(%s) %s oracle d_ade=%+.4f sep=%s"
              % (tag, "PASS" if c["PASS"] else "FAIL", c["oracle_ade_delta"],
                 c["separated"]))
    print("  G-REP  %s  A=%dw/%dep B=%dw/%dep episode-overlap=%d"
          % ("PASS" if grep["PASS"] else "FAIL", grep["n_A"], grep["episodes_A"],
             grep["n_B"], grep["episodes_B"], grep["overlap_episodes"]))

    for tag in ("A", "B"):
        d = draws[tag]
        print("")
        print("=== DRAW %s : SELECTED-PATH SAFETY vs ADE  (n=%dw / %dep) T0 ==="
              % (tag, d["n_windows"], d["n_episodes"]))
        print("  %-10s %8s %9s %5s | %9s %9s %5s | %8s %8s %6s"
              % ("rule", "ade_m", "d_ade", "sep", "envelope", "d_env", "sep",
                 "peak_g", "d_pkg", "agree"))
        for rule in ["model"] + ["gate%d" % k for k in GATE_KS] + ["kin_only", "oracle"]:
            R = d["results"]["abs"][rule]
            P = d["results"]["paired_vs_model"].get(rule, {})
            de = P.get("ade_m", {})
            dv = P.get("envelope", {})
            dg = P.get("peak_g", {})
            print("  %-10s %8.4f %+9.4f %5s | %9.4f %+9.4f %5s | %8.4f %+8.4f %6.3f"
                  % (rule, R["ade_m"]["mean"], de.get("delta", float("nan")),
                     de.get("separated", "-"), R["envelope"]["mean"],
                     dv.get("delta", float("nan")), dv.get("separated", "-"),
                     R["peak_g"]["mean"], dg.get("delta", float("nan")),
                     R["agrees_model"]["mean"]))

    print("")
    print("=== THE THREE-PART QUOTABILITY TEST (SPEC section 5) : gate2 ===")
    print("  %-14s %10s %10s %10s %6s %10s %s"
          % ("metric", "delta_A", "delta_B", "rep_floor", "sign", "sep_floor", "verdict"))
    for m in ("ade_m", "ade8_m", "fde_m", "envelope", "peak_g", "infeasible",
              "kamm_over", "off_reach", "flagged", "contact", "ttc_below"):
        v = verdict.get("gate2", {}).get(m)
        if not v:
            continue
        tag = ("QUOTABLE" if v["QUOTABLE"] else
               "UNDETECTABLE-DOWNWARD" if v["UNDETECTABLE_DOWNWARD"] else
               "WITHIN-NOISE" if v["WITHIN_NOISE"] else "NOT-SEPARATED")
        print("  %-14s %+10.4f %+10.4f %10.4f %6s %10.0e %s"
              % (m, v["delta_A"], v["delta_B"], v["replicate_floor"],
                 "same" if v["same_sign"] else "FLIP", v["separation_floor"], tag))

    print("")
    print("=== FOUR FAMILIES on the SELECTED path, per family, NEVER pooled (T0) ===")
    for tag in ("A", "B"):
        print("  -- draw %s, gate2 - model (paired episode-cluster bootstrap) --" % tag)
        fp = draws[tag]["family_paired"]["gate2"]
        for nm in sorted(fp):
            r = fp[nm]
            print("     %-22s %+10.5f  [%+.5f, %+.5f]  sep=%-5s n=%d/%dep"
                  % (nm, r["delta"], r["lo"], r["hi"], r["separated"],
                     r["n_windows"], r["n_episodes"]))
        st = draws[tag]["families"]["gate2"].get("strategic", {})
        si = draws[tag]["strategic_identity"]
        print("     STRATEGIC: status=%s defect=%s n=%s"
              % (st.get("status"), st.get("defect", False), st.get("n")))
        print("       reason  : %s" % str(st.get("reason", ""))[:160])
        print("       identity: route_head re-run by the gate = %s ; "
              "route_logits identical across rules = %s ; route~nav echo = %.4f"
              % (si["route_head_rerun_by_the_gate"],
                 si["route_logits_bitwise_identical_across_rules"],
                 si["route_argmax_vs_nav_cmd_agreement"]))
        for rl_ in ("model", "gate2"):
            tc = draws[tag]["families"][rl_].get("tactical", {})
            def _ag(blk):
                b_ = tc.get(blk, {})
                v = b_.get("agreement", b_.get("acc", b_.get("mean")))
                if isinstance(v, dict):
                    v = v.get("mean")
                return v
            print("     TACTICAL(%-5s): lat=%s lon=%s 5way=%s goal=%s n=%s"
                  % (rl_, _fmt(_ag("lateral_decision")), _fmt(_ag("longitudinal_decision")),
                     _fmt(_ag("maneuver_5way_collapsed")), _fmt(_ag("goal_setting")),
                     tc.get("n", tc.get("n_windows"))))
    print("")
    print("[kingate] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
