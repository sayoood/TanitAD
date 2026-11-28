#!/usr/bin/env python3
"""box_quality.py — the P-arm scorer of `PREREG_PERCEPTION_BOX_QUALITY.md` (`E-PERCEP-BOX-1`).

Every P arm is decided by `box3d_centre` on a target population, so that read must be an
INSTRUMENT, not analysis code written per arm.

⛔ BOTH POPULATIONS, ALWAYS, AND THE PRIMARY IS THE GATE'S. `summarise` returns
`near_forward` (x in [0, 60] m, |y| <= 16 m — what a collision gate acts on) AND `all_360`
(the 32-nearest, 360° set the loss trains on) in one object, and :func:`headline` REFUSES to
render one without the other. MEASURED on A8 and the reason this is enforced in code rather
than asked for in prose: 6.06 m near-forward vs 12.04 m over 360°, so quoting only the 360°
number understates the head ~2x on the gate's own question, and quoting only the near-forward
number overstates the detector.

⛔ THE MATCHER IS THE TRAINING ONE (`agent_slots.match_slots`, Hungarian over EVERY valid
target), not the AP's 2 m greedy matcher: at a ~10 m error the greedy matcher pairs only the
lucky hits and flatters the head. It is also what `box3d_centre` itself is computed through,
so the number is comparable to the trainer's own log.

⛔ CONTROLS THAT MUST READ KNOWN VALUES, emitted on every call, never optional:
  * `n_matched == n_target` — an identity while targets <= queries (`match_slots` matches
    `min(n_target, n_query)`; 32 padded targets against 100 queries). If it ever fails, the
    matcher changed and no centre number from that run is readable.
  * the velocity term against its ZERO-velocity floor (A8: +2.01 m/s [+0.87, +3.03]) — a
    regression is FAIL-HARM for the arm, not a footnote.
Pinned by `stack/tests/test_box_quality.py` and mutation-proven by
`stack/scripts/mutate_box_quality.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "stack", _ROOT / "taniteval" / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tanitad.models.agent_slots import match_slots        # noqa: E402

#: the gate's population — PREREG_PERCEPTION_BOX_QUALITY §1
NEAR_X = (0.0, 60.0)
NEAR_Y_HALF = 16.0
#: the two populations every report carries. ⛔ `POPULATIONS[0]` is the PRIMARY.
POPULATIONS = ("near_forward", "all_360")
BAR_M = 2.0


def population_mask(gx, gy, which: str):
    """The ONE definition of each population, so no caller can invent a second one."""
    gx, gy = np.asarray(gx, float), np.asarray(gy, float)
    if which == "near_forward":
        return (gx >= NEAR_X[0]) & (gx <= NEAR_X[1]) & (np.abs(gy) <= NEAR_Y_HALF)
    if which == "all_360":
        return np.ones(gx.shape, dtype=bool)
    if which == "far_or_behind":
        return ~population_mask(gx, gy, "near_forward")
    raise ValueError(f"unknown population {which!r}; known: {POPULATIONS + ('far_or_behind',)}")


def match_pairs(pred: dict, tgt: dict) -> dict:
    """One window -> per-matched-pair arrays through the TRAINING matcher.

    Returns ``dx, dy`` (abs metres), ``gx, gy`` (the TARGET's centre, which decides the
    population), ``vel_err`` / ``vel_floor`` (L2 over (v_rel_x, v_rel_y) against a
    zero-velocity predictor, on pairs whose GT rate is observed), and the counts
    ``n_target`` / ``n_matched`` the identity control reads.
    """
    m = match_slots(pred, tgt)
    rows, cols = m["rows"][0].tolist(), m["cols"][0].tolist()
    dx, dy, gx, gy, ve, vf = [], [], [], [], [], []
    rates = tgt.get("rates")
    rmask = tgt.get("rates_mask")
    for i, j in zip(rows, cols):
        p = pred["box"][0, i, :2]
        g = tgt["box"][0, j, :2]
        dx.append(abs(float(p[0] - g[0])))
        dy.append(abs(float(p[1] - g[1])))
        gx.append(float(g[0]))
        gy.append(float(g[1]))
        if rates is not None and rmask is not None and bool(rmask[0, j]):
            vg = rates[0, j, :2].float()
            ve.append(float((pred["rates"][0, i, :2].float() - vg).norm()))
            vf.append(float(vg.norm()))
    return {"dx": dx, "dy": dy, "gx": gx, "gy": gy, "vel_err": ve, "vel_floor": vf,
            "n_target": int(m["n_target"][0]), "n_matched": len(rows),
            "n_dropped": int(m["n_dropped"][0])}


def _block(dx, dy, mask) -> dict:
    if not mask.any():
        return {"n": 0, "centre_l1_m": None, "abs_dx_m": None, "abs_dy_m": None,
                "x_share_of_l1": None, "meets_bar": None}
    x, y = dx[mask], dy[mask]
    tot = float(x.sum() + y.sum())
    l1 = float((x + y).mean())
    return {"n": int(mask.sum()), "centre_l1_m": round(l1, 4),
            "abs_dx_m": round(float(x.mean()), 4), "abs_dy_m": round(float(y.mean()), 4),
            "median_l1_m": round(float(np.median(x + y)), 4),
            "x_share_of_l1": round(float(x.sum() / tot), 4) if tot > 0 else None,
            "meets_bar": bool(l1 < BAR_M), "bar_m": BAR_M}


def summarise(windows: list[dict], eid: list[str] | None = None, n_boot: int = 2000) -> dict:
    """Per-window pair records -> the report. ⛔ BOTH populations, always, plus the controls.

    ``eid`` (one episode id per window) turns on the episode-cluster bootstrap; without it the
    point estimates are still emitted and the intervals say why they are absent.
    """
    dx = np.array([v for w in windows for v in w["dx"]], dtype=float)
    dy = np.array([v for w in windows for v in w["dy"]], dtype=float)
    gx = np.array([v for w in windows for v in w["gx"]], dtype=float)
    gy = np.array([v for w in windows for v in w["gy"]], dtype=float)
    out = {"_what": "box-head centre error per target population (PREREG_PERCEPTION_BOX_QUALITY)",
           "_primary": POPULATIONS[0], "n_windows": len(windows), "n_pairs": int(dx.size)}
    for pop in POPULATIONS + ("far_or_behind",):
        out[pop] = _block(dx, dy, population_mask(gx, gy, pop))
    # ---- controls, emitted on EVERY call --------------------------------------------- #
    n_t = sum(w["n_target"] for w in windows)
    n_m = sum(w["n_matched"] for w in windows)
    ve = np.array([v for w in windows for v in w["vel_err"]], dtype=float)
    vf = np.array([v for w in windows for v in w["vel_floor"]], dtype=float)
    gain = float(vf.mean() - ve.mean()) if ve.size else None
    out["controls"] = {
        "n_target": n_t, "n_matched": n_m, "n_dropped": sum(w["n_dropped"] for w in windows),
        "matched_equals_target": bool(n_t == n_m),
        "identity_rule": ("match_slots matches min(n_target, n_query); with <= 32 padded targets "
                          "against 100 queries this is an IDENTITY. False ⇒ the matcher changed "
                          "and no centre number from this run is readable."),
        "vel_pairs": int(ve.size),
        "vel_mae_pred_mps": round(float(ve.mean()), 4) if ve.size else None,
        "vel_mae_zero_floor_mps": round(float(vf.mean()), 4) if ve.size else None,
        "vel_gain_mps": round(gain, 4) if gain is not None else None,
        "vel_beats_zero_floor": (None if gain is None else bool(gain > 0)),
        "vel_rule": "a regression below the zero-velocity floor is FAIL-HARM for the arm"}
    if eid is not None:
        from taniteval.ci import episode_cluster_bootstrap
        idx_by_w = np.cumsum([0] + [len(w["dx"]) for w in windows])

        def reducer(pop):
            def _r(sel):
                take = np.concatenate([np.arange(idx_by_w[int(k)], idx_by_w[int(k) + 1])
                                       for k in sel]) if len(sel) else np.array([], int)
                if take.size == 0:
                    return float("nan")
                m = population_mask(gx[take], gy[take], pop)
                return float((dx[take][m] + dy[take][m]).mean()) if m.any() else float("nan")
            return _r
        for pop in POPULATIONS:
            out[pop]["ci"] = episode_cluster_bootstrap(
                np.arange(len(windows), dtype=float), eid, reduce=reducer(pop), n_boot=n_boot)
    return out


def headline(report: dict) -> str:
    """⛔ REFUSES to render one population without the other (the prereg's §1 rule, in code)."""
    missing = [p for p in POPULATIONS if p not in report]
    if missing:
        raise ValueError(f"a box-quality report must carry BOTH populations; missing {missing}. "
                         f"Quoting one alone misstates the head: near-forward is the gate's "
                         f"question, 360° is the loss's.")
    nf, a3 = report["near_forward"], report["all_360"]
    return ("box3d_centre  PRIMARY near-forward %s m (n=%s, bar %s) | SECONDARY 360° %s m (n=%s)"
            % (nf["centre_l1_m"], nf["n"], BAR_M, a3["centre_l1_m"], a3["n"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--agents", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=200, help="windows to score")
    ap.add_argument("--expect-windows", type=int, default=10600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import s1_pass as SP
    corp = SP.Corpus(a.config, a.cache, a.labels, a.agents, None)
    if len(corp.ds) != a.expect_windows:
        raise SystemExit(f"⛔ len(ds) {len(corp.ds)} != the trainer's {a.expect_windows}")
    SP.attach_agent_gt(corp, corp.targs)
    mp = SP.ModelPass(corp, a.ckpt, a.device)
    wis = [w for w in SP.trainer_windows(corp.ds, 1000) if corp.eligibility(w) is None][:a.n]
    windows, eid = [], []
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        out = mp.forward(item, str(corp.clip_ids[e_i]))
        dec = {k: v[0].float().cpu() for k, v in out["perception"]["box_slots"].items()
               if torch.is_tensor(v) and v.dim() >= 2}
        pred = {k: v[None] for k, v in dec.items()}
        tgt = {"box": item["agent_box"][None].float(), "valid": item["agent_valid"][None],
               "cls": item["agent_cls"][None], "rates": item["agent_rates"][None].float(),
               "rates_mask": item["agent_rates_mask"][None].bool()}
        windows.append(match_pairs(pred, tgt))
        eid.append(SP.sha12(corp.clip_ids[e_i]))
    rep = summarise(windows, eid)
    rep["_provenance"] = {"ckpt": a.ckpt, "config": a.config, "cache": a.cache,
                          "device": a.device, "windows_scored": len(windows)}
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(headline(rep))
    print("controls:", {k: rep["controls"][k] for k in ("matched_equals_target",
                                                        "vel_gain_mps", "vel_beats_zero_floor")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
