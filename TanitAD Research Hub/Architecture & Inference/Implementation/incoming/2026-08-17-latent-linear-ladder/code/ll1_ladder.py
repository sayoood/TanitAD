"""LL1 — the LATENT LINEAR LADDER: pc6's ridge readout, run on a GRADED
SPECTRUM of targets instead of on the lead gap alone.

⛔ WHAT THIS IS AND IS NOT.
`PROBE_POSITIVE_CONTROL.md` §2.3 measured ONE row of a ladder: under a ridge the
v6 latent beats the random-latent null by ~1.8 m on the lead gap with a POSITIVE
correlation. A single quantity gives a single verdict. This runs the SAME ridge,
on the SAME caches/split/estimator, over a spectrum from *must-be-present*
(the ego's own speed) to *hard* (lead closing speed), so the answer is WHERE the
latent stops carrying information, not merely whether one number moves.

⛔ T0-DIAGNOSTIC. A frozen-latent readout is a world-model diagnostic and is
NEVER driving performance.

⛔ EQUIVALENCE TO THE BANKED NUMBERS IS PROVED, NOT ASSERTED.
`ridge_fit` is IMPORTED from `pc6_linear_readout` (no second implementation of
the solve). Everything else — z-score with the PROBE-TRAIN mean/sd, bias column,
alpha chosen by MAE on an EPISODE-DISJOINT inner split of the PROBE-TRAIN clips,
scoring on the same eval windows with `paired_episode_cluster_bootstrap` — is
replicated and then GATED: `--gate-pc6` asserts that target `lead_gap` at seed 0
reproduces the banked `pc6_ridge_*.json` to 1e-4. That is the same discipline as
pc5's "R0 reproduces every banked headline to +/-0.008 m".

THE LADDER, and why each rung is on it
  ego_v0         the SANITY ANCHOR. The ego's own speed, and the latent is built
                 from a window that contains it. If a linear readout cannot
                 recover this, the cache or the readout is broken and every
                 other row is meaningless.
  ego_accel      own-motion, one derivative harder; the LONGITUDINAL family's
                 own control variable.
  ego_yawrate    own-motion, rotational.
  ego_curv       yawrate / speed — the LATERAL family's actual target. Only
                 defined above a speed floor (see EGO_CURV_V_FLOOR).
  n_agents_grid  scene density, in-grid.
  n_agents_all   scene density, whole frame (in-grid + out-of-grid count).
  lead_present   binary: is there an in-corridor agent within 30 m at all?
  nearest_any    min cx over ALL in-grid agents — "nearest object", WITHOUT the
                 corridor selection. Separates *seeing an object* from
                 *selecting the lead*.
  lead_gap       the banked ~1.8 m result (PROBE_POSITIVE_CONTROL §2.3).
  lead_closing   -v_rel_x of the GT lead: the LONGITUDINAL family's real target.
  lead_inv_ttc   closing / gap  (1/s).  ⚠️ INVERSE TTC IS THE REGRESSED
                 QUANTITY, NOT TTC. TTC is unbounded (it diverges as closing ->
                 0) and undefined for a receding lead, so a mean abs error on it
                 is dominated by a handful of near-zero denominators and is not
                 a sane regression target. The inverse is bounded, signed and is
                 the standard risk surrogate. Stated rather than quietly
                 substituted.

CONTROLS ON EVERY ROW — a row without its null is not a result
  C-CONST     the PROBE-TRAIN median (pc6's own constant).
  C-EPMEAN    leave-one-out mean of the window's OWN eval episode — the
              EPISODE-IDENTITY ORACLE. A readout that merely recognises which
              episode it is in cannot beat this.
  NULL        the same target, same windows, on the window-matched
              RANDOM-LATENT cache. Run as a separate invocation with
              --cache cache_nullmatched.
  C-V0        ⭐ THE TRIVIAL-PROXY CHECK: the identical ridge with the latent
              replaced by the single scalar `v0`. It answers "is this quantity
              recoverable from something trivial?" — e.g. lead gap being a
              stand-in for ego speed. A positive that is really a proxy is worse
              than a null.

AND THE ANTI-EPISODE-IDENTITY STATISTIC, because a rich latent is MORE prone to
episode recognition than a slot head, not less: `corr_within_ep` is the
correlation of prediction and truth after BOTH are demeaned by their own eval
episode. Within-episode gap SD is only 3.9 m, so a readout that merely
identifies the episode scores well globally and at ~0 here.
"""
from __future__ import annotations

# ⚠️ pyarrow BEFORE torch on this box (0xC0000005 segfault, proven A/B).
import pyarrow  # noqa: F401  # isort: skip

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp2_probe as SP                                          # noqa: E402
from pc6_linear_readout import ridge_fit                        # noqa: E402
from taniteval.ci import (episode_cluster_bootstrap,             # noqa: E402
                          paired_episode_cluster_bootstrap)

# Below this speed a curvature (yawrate / v) is numerically meaningless: a
# stationary car with any yaw jitter reads as an infinite radius of curvature.
EGO_CURV_V_FLOOR = 2.0            # m/s
TTC_EPS = 1e-3

LADDER = ["ego_v0", "ego_accel", "ego_yawrate", "ego_curv",
          "n_agents_grid", "n_agents_all", "lead_present",
          "nearest_any", "lead_gap", "lead_closing", "lead_inv_ttc"]

UNITS = {"ego_v0": "m/s", "ego_accel": "m/s^2", "ego_yawrate": "rad/s",
         "ego_curv": "1/m", "n_agents_grid": "agents", "n_agents_all": "agents",
         "lead_present": "prob", "nearest_any": "m", "lead_gap": "m",
         "lead_closing": "m/s", "lead_inv_ttc": "1/s"}

RUNG = {"ego_v0": "EGO (anchor)", "ego_accel": "EGO", "ego_yawrate": "EGO",
        "ego_curv": "EGO", "n_agents_grid": "SCENE", "n_agents_all": "SCENE",
        "lead_present": "OBJECT", "nearest_any": "OBJECT",
        "lead_gap": "OBJECT", "lead_closing": "OBJECT-DYNAMICS",
        "lead_inv_ttc": "OBJECT-DYNAMICS"}


# ---------------------------------------------------------------------------
# ego kinematics, derived from the EPISODE POSES on the join's own clock
# ---------------------------------------------------------------------------
def load_ego(ep_dir: Path, join_file: Path, clips: set[str]) -> dict:
    """clip_id -> {"poses": [T,4] (x, y, yaw, v), "t_join": {frame_idx: t_s}}.

    ⚠️ The frame clock is read from the JOIN's own ``t_s``, never assumed to be
    0.1 s (the episode grid is ~0.1007 s — `build_obstacle_join.py` §CLOCK;
    MEASURED here over 25 660 intervals: 0.1004-0.1011, median 0.100700,
    sd 1.15e-4).
    """
    ts: dict[str, dict[int, float]] = {}
    with open(join_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            c = str(r["clip_id"])
            if c in clips:
                ts.setdefault(c, {})[int(r["frame_idx"])] = float(r["t_s"])
    out = {}
    for c in sorted(clips):
        f = ep_dir / f"{c}.v2ep.pt"
        if not f.exists():
            raise SystemExit(f"[ll1] missing episode file for clip {c}: {f}")
        blob = torch.load(f, map_location="cpu", weights_only=False)
        out[c] = {"poses": np.asarray(blob["poses"], dtype=np.float64),
                  "t_join": ts.get(c, {})}
    return out


def bind_pose_grid(rows, ego, span: int = 4) -> dict:
    """⭐ DISCOVER the row->pose index offset FROM THE DATA, and prove it.

    ⛔ THE FIRST VERSION OF THIS FILE ASSUMED offset 0, because `window_frame`
    returns ``t + window - 1`` and `_contract.py` sets ``pose_last =
    ep.poses[t + window - 1]``. READING THE CODE GAVE THE WRONG ANSWER: the
    episode's `poses` array is on the RAW frame grid while the dataset's frames
    are 3-STACKS (`n_stack = 3`), so the dataset's frame 0 is raw frame 2. The
    assumption was caught only because ``v0`` is banked per row and can be
    checked — max mismatch 0.667 m/s on 4 869 of 5 617 rows.

    So the offset is SCANNED and accepted only on EXACT equality (the banked
    ``v0`` IS ``poses[frame_idx + off][3]``, same float32), on EVERY row. A
    near-match is rejected: an off-by-one in a finite difference is exactly the
    error this check exists to stop.
    """
    scan = {}
    for off in range(-span, span + 1):
        worst, n_ok = 0.0, 0
        for r in rows:
            p = ego[r["clip_id"]]["poses"]
            k = int(r["frame_idx"]) + off
            if not (0 <= k < p.shape[0]):
                worst = float("inf")
                break
            worst = max(worst, abs(p[k, 3] - float(r["v0"])))
            n_ok += 1
        scan[off] = worst
        if worst == 0.0 and n_ok == len(rows):
            # per-clip median spacing on the JOIN grid, then a pose-grid clock
            for c, e in ego.items():
                tj = e["t_join"]
                ks = sorted(tj)
                dt = (float(np.median(np.diff([tj[k] for k in ks])
                                      / np.diff(ks))) if len(ks) >= 2
                      else float("nan"))
                T = e["poses"].shape[0]
                t = np.full(T, np.nan)
                for k, v in tj.items():          # pose index = join index+off
                    if 0 <= k + off < T:
                        t[k + off] = v
                good = np.nonzero(~np.isnan(t))[0]
                for j in range(T):
                    if np.isnan(t[j]) and good.size:
                        t[j] = t[good[0]] + (j - good[0]) * dt
                e["t"], e["dt_median"] = t, dt
            return {"pose_index_offset": off, "rule": "poses[frame_idx + off]",
                    "accepted_on": "EXACT equality of poses[.,3] and banked v0",
                    "n_rows_checked": len(rows),
                    "max_abs_v0_mismatch": 0.0,
                    "scan_max_abs_mismatch": {str(k): (None if v == float("inf")
                                                       else round(v, 6))
                                              for k, v in scan.items()},
                    "n_stack_note": "offset == n_stack - 1 == 2"}
    raise SystemExit("[ll1] ⛔ POSE ALIGNMENT FAILED — no offset in "
                     f"[-{span},{span}] reproduces the banked v0 exactly: "
                     f"{ {k: (None if v == float('inf') else round(v, 6)) for k, v in scan.items()} }")


def _central(vals: np.ndarray, t: np.ndarray, k: int, wrap: bool = False):
    """Central difference at k, one-sided at the ends. None if not derivable."""
    T = vals.size
    lo, hi = max(0, k - 1), min(T - 1, k + 1)
    if hi == lo:
        return None
    dt = t[hi] - t[lo]
    if not np.isfinite(dt) or abs(dt) < 1e-6:
        return None
    d = float(vals[hi] - vals[lo])
    if wrap:
        d -= 2.0 * math.pi * math.floor((d + math.pi) / (2.0 * math.pi))
    return d / dt


def target_of(row: dict, ego: dict, name: str, pose_off: int = 0):
    """(value, ok). ``ok=False`` means the quantity is UNDEFINED on this window
    and the window is dropped from BOTH train and eval, with its n reported."""
    ag = row["agents"]
    if name == "ego_v0":
        return float(row["v0"]), True
    if name in ("ego_accel", "ego_yawrate", "ego_curv"):
        e = ego[row["clip_id"]]
        p, t, k = e["poses"], e["t"], int(row["frame_idx"]) + pose_off
        if k < 0 or k >= p.shape[0]:
            return 0.0, False
        if name == "ego_accel":
            a = _central(p[:, 3], t, k)
            return (float(a), True) if a is not None else (0.0, False)
        yr = _central(p[:, 2], t, k, wrap=True)
        if yr is None:
            return 0.0, False
        if name == "ego_yawrate":
            return float(yr), True
        v = float(row["v0"])
        if v < EGO_CURV_V_FLOOR:
            return 0.0, False
        return float(yr / v), True
    if name == "n_agents_grid":
        return float(ag.shape[0]), True
    if name == "n_agents_all":
        return float(ag.shape[0] + int(row["n_out_of_grid"])), True
    if name == "lead_present":
        return (1.0 if SP.gt_lead_gap(ag) is not None else 0.0), True
    if name == "nearest_any":
        if ag.numel() == 0:
            return 0.0, False
        cx = ag[:, 0]
        m = cx > 0
        if not bool(m.any()):
            return 0.0, False
        return float(cx[m].min()), True
    g = SP.gt_lead_gap(ag)
    if g is None:
        return 0.0, False
    if name == "lead_gap":
        return float(g), True
    j = SP.gt_lead_row(ag)
    if j is None or not bool(row["rates_mask"][j]):
        return 0.0, False
    closing = -float(row["rates"][j, 0])          # +ve = closing
    if name == "lead_closing":
        return closing, True
    if name == "lead_inv_ttc":
        return closing / max(float(g), TTC_EPS), True
    raise SystemExit(f"[ll1] unknown target {name}")


# ---------------------------------------------------------------------------
def auc_binary(score: np.ndarray, y: np.ndarray):
    """Rank AUC; None when the target is not binary or is degenerate."""
    pos, neg = y > 0.5, y <= 0.5
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1, dtype=np.float64)
    s = np.sort(score)
    i = 0
    while i < s.size:                                     # average ties
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            m = (score >= s[i]) & (score <= s[i])
            ranks[m] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n1, n0 = float(pos.sum()), float(neg.sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def demean_by(v: np.ndarray, key: np.ndarray) -> np.ndarray:
    out = np.array(v, dtype=np.float64, copy=True)
    for c in np.unique(key):
        m = key == c
        out[m] -= out[m].mean()
    return out


def loo_epmean(y: np.ndarray, key: np.ndarray, fallback: float) -> np.ndarray:
    """pc6's C-EPMEAN, verbatim: leave-one-out mean of the own episode."""
    out = np.full(y.size, fallback, dtype=np.float64)
    for c in np.unique(key):
        pos = np.nonzero(key == c)[0]
        tot = float(np.sum(y[pos]))
        for k in pos:
            out[k] = ((tot - y[k]) / (pos.size - 1)) if pos.size > 1 \
                else fallback
    return out


# ---------------------------------------------------------------------------
def _solve(Z, y, alpha, mode):
    """``pc6`` — verbatim pc6: the ones-column is INSIDE the penalty.
    ``centred`` — THE REPAIR: y is centred and the intercept is not penalised.

    ⛔ THE DEFECT THE REPAIR FIXES. pc6's `ridge_fit` builds
    ``X.T @ X + alpha * np.eye(d)`` on a design matrix whose LAST COLUMN IS THE
    BIAS, so the intercept is shrunk like any other coefficient. As alpha grows
    the prediction therefore collapses toward **ZERO, NOT TOWARD THE MEAN** —
    MEASURED on a synthetic target of mean 7.0 with no signal: pc6's MAE goes
    0.0001 -> 1.15 -> 6.63 -> 7.00 across alpha 1e-2..1e8, i.e. the fit ends up
    predicting 0 and scoring the target's own magnitude.
    ⇒ **THE READOUT IS STRUCTURALLY UNABLE TO FALL BACK TO A CONSTANT**, which
    is exactly the baseline K1 scores it against. Large alphas are unusable, so
    the alpha sweep is silently truncated and a no-signal arm is pushed to an
    ARBITRARILY BAD score rather than to a tie. Both matter for reading "K1
    fails": under `pc6` a failure can mean *"no signal"* OR *"the instrument
    cannot express the null hypothesis"*, and those are different facts.
    """
    if mode == "pc6":
        return ridge_fit(Z, y, alpha), 0.0
    mu = float(y.mean())
    return ridge_fit(Z[:, :-1], y - mu, alpha), mu


def _apply(Z, w, b, mode):
    return Z @ w if mode == "pc6" else Z[:, :-1] @ w + b


def partial_corr(a, b, z):
    """corr(a, b) with the linear effect of ``z`` removed from BOTH.

    ⭐ THE TRIVIAL-PROXY TEST, in its sharpest form. A positive that is really a
    proxy is worse than a null, and MEASURED here the lead gap is recoverable
    from the EGO SPEED SCALAR ALONE at r +0.683 — better than from the whole
    2 048-dim latent. So every latent correlation must be re-asked as: *does
    anything survive once ego speed is partialled out?*
    """
    a, b, z = (np.asarray(x, float) for x in (a, b, z))
    Z = np.stack([z, np.ones_like(z)], 1)
    ra = a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    rb = b - Z @ np.linalg.lstsq(Z, b, rcond=None)[0]
    return corr(ra, rb)


def fit_one(Ztr, ytr, ctr, Zev, yev, eev, cev, alphas, inner_frac, seed,
            n_boot, want_auc, v0ev=None, mode="pc6"):
    """pc6's fit, verbatim (``mode='pc6'``), on an arbitrary target."""
    rng = np.random.default_rng(seed)
    clips = np.array(sorted(set(ctr.tolist())))
    rng.shuffle(clips)
    n_in = max(1, int(round(len(clips) * inner_frac)))
    inner = set(clips[:n_in].tolist())
    m_in = np.array([c in inner for c in ctr])
    best, best_mae, tried = None, np.inf, {}
    for al in alphas:
        w, b = _solve(Ztr[~m_in], ytr[~m_in], al, mode)
        mae = float(np.abs(_apply(Ztr[m_in], w, b, mode) - ytr[m_in]).mean())
        tried[f"{al:g}"] = round(mae, 6)
        if mae < best_mae:
            best, best_mae = al, mae
    w, b = _solve(Ztr, ytr, best, mode)
    pred = _apply(Zev, w, b, mode)

    const_v = float(np.median(ytr))
    epmean = loo_epmean(yev, cev, const_v)
    e_arm, e_con, e_ep = (np.abs(pred - yev), np.abs(const_v - yev),
                          np.abs(epmean - yev))
    arm = episode_cluster_bootstrap(e_arm, eev, n_boot=n_boot)
    con = episode_cluster_bootstrap(e_con, eev, n_boot=n_boot)
    k1 = paired_episode_cluster_bootstrap(e_arm, e_con, eev, n_boot=n_boot)
    k5 = paired_episode_cluster_bootstrap(e_arm, e_ep, eev, n_boot=n_boot)
    sst = float(((yev - yev.mean()) ** 2).sum())
    r2 = float(1.0 - ((pred - yev) ** 2).sum() / sst) if sst > 0 else float("nan")
    r2c = float(1.0 - ((const_v - yev) ** 2).sum() / sst) if sst > 0 else float("nan")
    out = {"alpha_chosen": best, "alpha_inner_mae": tried,
           "alpha_at_grid_edge": bool(best in (alphas[0], alphas[-1])),
           "inner_split_clips": n_in,
           "err": round(float(e_arm.mean()), 4),
           "err_lo": arm["lo"], "err_hi": arm["hi"],
           "err_median": round(float(np.median(e_arm)), 4),
           "c_const_value": round(const_v, 4),
           "c_const_err": round(float(e_con.mean()), 4),
           "c_const_lo": con["lo"], "c_const_hi": con["hi"],
           "c_epmean_err": round(float(e_ep.mean()), 4),
           "skill_vs_const": round(1.0 - float(e_arm.mean())
                                   / max(float(e_con.mean()), 1e-12), 4),
           "R2": round(r2, 4), "R2_const": round(r2c, 4),
           "K1_delta": k1["delta"], "K1_lo": k1["lo"], "K1_hi": k1["hi"],
           "K1_separated": k1["separated"],
           "K1_PASSES": bool(k1["separated"] and k1["delta"] < 0),
           "K5_delta": k5["delta"], "K5_lo": k5["lo"], "K5_hi": k5["hi"],
           "K5_separated": k5["separated"],
           "K5_PASSES": bool(k5["separated"] and k5["delta"] < 0),
           "corr": round(corr(pred, yev), 4),
           "corr_within_ep": round(corr(demean_by(pred, cev),
                                        demean_by(yev, cev)), 4),
           # r^2 is the CEILING an optimally-rescaled linear readout could
           # reach: the fit is over-dispersed (it emits full variance at low
           # correlation), so MAE can lose to a constant while r is positive.
           # Quoting both stops "K1 fails" from being read as "r is zero".
           "r2_ceiling": round(float(corr(pred, yev) ** 2), 4),
           "pred_sd": round(float(pred.std()), 4),
           "gt_sd": round(float(yev.std()), 4)}
    if v0ev is not None:
        out["corr_partial_v0"] = round(partial_corr(pred, yev, v0ev), 4)
    if want_auc:
        out["auc"] = (lambda a: round(a, 4) if a is not None else None)(
            auc_binary(pred, yev))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--episodes-dir", required=True)
    ap.add_argument("--join-file", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", nargs="+", default=LADDER)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5])
    ap.add_argument("--inner-frac", type=float, default=0.25)
    ap.add_argument("--proxy-v0", action="store_true",
                    help="C-V0: replace the latent with the scalar v0")
    ap.add_argument("--features", choices=["cells", "tokens_mean"],
                    default="cells",
                    help="`cells` = the [16,128] operative readout (the "
                         "ladder's subject). `tokens_mean` = the ENCODER's "
                         "[640,768] ViT patch tokens MEAN-POOLED over the "
                         "16x40 grid -> 768 features. ⚠️ Pooling is a real "
                         "choice: it is defensible for GLOBAL quantities "
                         "(ego speed, density) and DESTROYS the spatial "
                         "structure a per-object quantity needs, so only the "
                         "EGO rows of a tokens run are load-bearing. Its job "
                         "is to localise the anchor's negative: is ego motion "
                         "absent from the ENCODER, or present there and "
                         "discarded by the readout?")
    ap.add_argument("--randomise-features", type=int, default=None,
                    help="matched-random NULL for ANY feature set: replace X "
                         "with N(mu, sd) per feature, mu/sd from the real "
                         "features. Same construction as pA_null_matched.py. "
                         "The int is the seed.")
    ap.add_argument("--fit-mode", choices=["pc6", "centred"], default="pc6",
                    help="pc6 = verbatim incumbent (intercept INSIDE the "
                         "penalty); centred = THE REPAIR (see _solve)")
    ap.add_argument("--gate-pc6", default=None,
                    help="path to pc6_ridge_<arm>.json; asserts lead_gap@seed0 "
                         "reproduces it")
    a = ap.parse_args(argv)

    blob = torch.load(a.cache, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], blob["meta"]
    decl = json.loads(Path(a.split_json).read_text("utf-8"))
    ev_c, tr_c = set(decl["eval_clips"]), set(decl["train_clips"])
    clips_used = {r["clip_id"] for r in rows}
    need_ego = any(t in ("ego_accel", "ego_yawrate", "ego_curv")
                   for t in a.targets)
    ego = load_ego(Path(a.episodes_dir), Path(a.join_file),
                   clips_used) if need_ego else {}

    align = bind_pose_grid(rows, ego) if need_ego else None
    pose_off = align["pose_index_offset"] if align else 0

    idx_tr = [i for i, r in enumerate(rows) if r["clip_id"] in tr_c]
    idx_ev = [i for i, r in enumerate(rows) if r["clip_id"] in ev_c]

    def feats(idx):
        if a.proxy_v0:
            return np.array([[float(rows[i]["v0"])] for i in idx],
                            dtype=np.float64)
        if a.features == "tokens_mean":
            if rows[0].get("tokens") is None:
                raise SystemExit("[ll1] ⛔ --features tokens_mean but this "
                                 "cache banked no tokens (meta "
                                 f"tokens_banked={meta.get('tokens_banked')})")
            return np.stack([rows[i]["tokens"].numpy().astype(np.float64)
                             .mean(0) for i in idx])
        return np.stack([rows[i]["cells"].numpy().reshape(-1).astype(np.float64)
                         for i in idx])

    Xtr_all, Xev_all = feats(idx_tr), feats(idx_ev)
    if a.randomise_features is not None:
        g = np.random.default_rng(a.randomise_features)
        mu_r, sd_r = Xtr_all.mean(0), Xtr_all.std(0)
        Xtr_all = g.normal(mu_r, np.maximum(sd_r, 1e-12), Xtr_all.shape)
        Xev_all = g.normal(mu_r, np.maximum(sd_r, 1e-12), Xev_all.shape)
    ctr_all = np.array([rows[i]["clip_id"] for i in idx_tr])
    cev_all = np.array([rows[i]["clip_id"] for i in idx_ev])
    eev_all = np.array([rows[i]["episode_uid"] for i in idx_ev])

    res = {"_evidence_class":
           "MEASURED (ours; pc6 ridge readout on a graded target ladder — a "
           "DIFFERENT instrument from the F-18 slot probe and never to be "
           "quoted as one)",
           "eval_tier": "T0-DIAGNOSTIC",
           "arm": a.label, "run_stamp": meta.get("run_stamp"),
           "step": meta.get("step"),
           "features": ("v0 scalar (C-V0 PROXY)" if a.proxy_v0
                        else "tokens MEAN-POOLED over the 16x40 grid "
                             f"{list(rows[0]['tokens'].shape)}"
                        if a.features == "tokens_mean"
                        else f"cells {list(rows[0]['cells'].shape)} flattened")
           + ("" if a.randomise_features is None
              else f"  [MATCHED-RANDOM NULL, seed {a.randomise_features}]"),
           "feature_set": a.features,
           "randomise_features_seed": a.randomise_features,
           "n_features": int(Xtr_all.shape[1]) + 1,
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap",
           "forbidden": "overlapping_holdout_se",
           "fit_mode": a.fit_mode, "alphas": a.alphas,
           "n_boot": a.n_boot, "seeds": a.seeds,
           "ego_curv_v_floor_ms": EGO_CURV_V_FLOOR,
           "pose_alignment_check": align,
           "cache": str(a.cache), "targets": {}}

    for tname in a.targets:
        tv, tok = zip(*[target_of(rows[i], ego, tname, pose_off)
                        for i in idx_tr])
        mtr = np.array(tok)
        ytr = np.array(tv, dtype=np.float64)[mtr]
        tv, tok = zip(*[target_of(rows[i], ego, tname, pose_off)
                        for i in idx_ev])
        mev = np.array(tok)
        yev = np.array(tv, dtype=np.float64)[mev]
        Xtr, Xev = Xtr_all[mtr], Xev_all[mev]
        ctr, cev, eev = ctr_all[mtr], cev_all[mev], eev_all[mev]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd < 1e-12] = 1.0
        Ztr = np.concatenate([(Xtr - mu) / sd, np.ones((Xtr.shape[0], 1))], 1)
        Zev = np.concatenate([(Xev - mu) / sd, np.ones((Xev.shape[0], 1))], 1)
        want_auc = tname == "lead_present"
        v0ev = (None if tname == "ego_v0" else
                np.array([float(rows[i]["v0"]) for i in idx_ev])[mev])
        per_seed = {}
        for s in a.seeds:
            per_seed[str(s)] = fit_one(Ztr, ytr, ctr, Zev, yev, eev, cev,
                                       a.alphas, a.inner_frac, s, a.n_boot,
                                       want_auc, v0ev, a.fit_mode)
        errs = [per_seed[str(s)]["err"] for s in a.seeds]
        k1s = [per_seed[str(s)]["K1_delta"] for s in a.seeds]
        res["targets"][tname] = {
            "unit": UNITS[tname], "rung": RUNG[tname],
            "n_train": int(Ztr.shape[0]), "n_eval": int(Zev.shape[0]),
            "n_eval_clusters": int(len(np.unique(eev))),
            "n_train_dropped_undefined": int((~mtr).sum()),
            "n_eval_dropped_undefined": int((~mev).sum()),
            "gt_mean": round(float(yev.mean()), 4),
            "gt_sd": round(float(yev.std()), 4),
            "gt_within_ep_sd": round(float(demean_by(yev, cev).std()), 4),
            "seed_err_range": round(float(max(errs) - min(errs)), 4),
            "seed_K1_range": round(float(max(k1s) - min(k1s)), 4),
            "per_seed": per_seed}
        p = per_seed[str(a.seeds[0])]
        print("  %-15s %-22s err=%9.4f %-6s const=%9.4f K1=%+9.4f %-8s "
              "R2=%+7.4f r=%+.3f r_wep=%+.3f r_pv0=%+.3f seedrng=%.4f n=%d/%d"
              % (tname, a.label[:22], p["err"], UNITS[tname],
                 p["c_const_err"], p["K1_delta"],
                 "K1 PASS" if p["K1_PASSES"] else "K1 fail", p["R2"],
                 p["corr"], p["corr_within_ep"],
                 p.get("corr_partial_v0", float("nan")),
                 res["targets"][tname]["seed_err_range"],
                 int(Zev.shape[0]), int(len(np.unique(eev)))), flush=True)

    # ---- the equivalence GATE against the banked pc6 numbers ---------------
    if a.gate_pc6 and a.fit_mode != "pc6":
        raise SystemExit("[ll1] ⛔ --gate-pc6 is only meaningful in "
                         "--fit-mode pc6; the repair CHANGES the numbers "
                         "on purpose and must not be gated against them.")
    if a.gate_pc6:
        want = json.loads(Path(a.gate_pc6).read_text("utf-8"))
        got = res["targets"]["lead_gap"]["per_seed"]["0"]
        checks = [("ridge_err_m", want["ridge_err_m"], got["err"], 1e-4),
                  ("c_const_err_m", want["c_const_err_m"],
                   got["c_const_err"], 1e-4),
                  ("K1_delta", want["K1_delta"], got["K1_delta"], 1e-3),
                  ("corr_pred_gt", want["corr_pred_gt"], got["corr"], 1e-3),
                  ("alpha_chosen", want["alpha_chosen"],
                   got["alpha_chosen"], 0.0),
                  ("n_eval_windows", want["n_eval_windows"],
                   res["targets"]["lead_gap"]["n_eval"], 0.0),
                  ("n_train_windows", want["n_train_windows"],
                   res["targets"]["lead_gap"]["n_train"], 0.0)]
        bad = [(k, w, g) for k, w, g, tol in checks if abs(w - g) > tol]
        res["pc6_equivalence_gate"] = {
            "against": str(a.gate_pc6),
            "checks": {k: {"banked": w, "ours": g, "tol": tol}
                       for k, w, g, tol in checks},
            "PASSED": not bad}
        print(f"  [gate] pc6 equivalence: "
              f"{'PASS' if not bad else 'FAIL ' + repr(bad)}", flush=True)
        if bad:
            Path(a.out).write_text(json.dumps(res, indent=1), "utf-8")
            raise SystemExit(f"[ll1] ⛔ pc6 EQUIVALENCE GATE FAILED: {bad}")

    Path(a.out).write_text(json.dumps(res, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
