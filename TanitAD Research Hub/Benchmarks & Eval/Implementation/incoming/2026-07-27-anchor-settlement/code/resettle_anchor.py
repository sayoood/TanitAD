"""Re-settle the comma heading-repair ANCHOR (+0.0114 -> +0.3308) on its own
substrate, after the content-overlap probe found it partly IN-TRAIN.

WHAT THIS ANSWERS
-----------------
1. Does the published anchor reproduce from the persisted artifacts?  (a pin —
   if it does not, nothing below is about the anchor.)
2. What is the anchor once the CONTENT-VERIFIED in-train episodes are removed?
   `fingerprint_comma_cache.py` found `76b:ep_00018` and `76b:ep_00039`
   bit-identical (poses AND raw frames) to `61c:ep_00008` / `61c:ep_00020`,
   which are inside `idm_head_v1`'s own `cm_[0:40]` comma TRAINING set.
3. What does the anchor's substrate read under STRICT ADMISSIBILITY — the
   `observable` mask that `hold_heading_through_standstill` returns and that no
   caller in the repo consumes?
4. PhysicalAI, per corpus and never pooled: the same three protocols, so the
   claim "PhysicalAI is unaffected" is measured here rather than inherited.

⛔ NOTHING IS RETRAINED and nothing is re-encoded. The persisted A0 predictions
(`a0_preds.npy`, the EXACT array the anchor was computed from) are re-used, so
every contrast below is on identical predictions and differs only by which
windows/labels enter the statistic.

IMPORT DISCIPLINE (the brief's trap, and it is real here)
--------------------------------------------------------
`idm2_lib.py:19` and `idm3_a0.py` run an unconditional
`sys.path.insert(0, "/root/taniteval")`.  On this host that file is
`ef925f06febd20a99f5901491fcf75cb` while HEAD's is
`c92618a02b36f8191a581fb74a491a8d` — a DIFFERENT ci.py.  `stack_check` cannot
see this (it checks the tanitad stack, not which taniteval got imported).
Therefore this script imports NEITHER module, pins the HEAD taniteval by path,
and ASSERTS the md5 of the ci.py it actually loaded.

ESTIMATOR: `taniteval.ci.episode_cluster_bootstrap` (callable-`r2` reducer) and
`paired_episode_cluster_bootstrap`, B = 2000, unit = the episode.
⛔ `overlapping_holdout_se` is never called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# pinned imports — HEAD's taniteval ONLY, and prove it                         #
# --------------------------------------------------------------------------- #
HEAD_TANITEVAL = "/workspace/TanitAD-head/taniteval"
CI_MD5_HEAD = "c92618a02b36f8191a581fb74a491a8d"

sys.path.insert(0, HEAD_TANITEVAL)
from taniteval import ci as tci                                    # noqa: E402


def md5_of(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while (b := f.read(1 << 22)):
            h.update(b)
    return h.hexdigest()


_ci_md5 = md5_of(tci.__file__)
assert _ci_md5 == CI_MD5_HEAD, (
    f"⛔ ESTIMATOR PIN FAILED — imported {tci.__file__} (md5 {_ci_md5}), "
    f"expected HEAD's {CI_MD5_HEAD}. A stale /root/taniteval is on sys.path.")

VAL_GT = Path("/workspace/idm3/out/val_gt_v3.npy")
A0_PREDS = Path("/workspace/idm3/out/a0_preds.npy")
LAT = Path("/root/idm2/lat")
HEAD_PT = Path("/root/idmval/idm_head_v1.pt")
HEAD_MD5 = "fa4462f0b898b036be729c790278b823"       # card /weights_md5

K_BUILD, STRIDE, HORIZONS = 8, 2, (5, 10, 15, 20)
V_MIN = 0.5
N_BOOT = 2000
CH = {"speed": 0, "yaw_rate": 1, "steer": 2, "long_accel": 3}

# CONTENT-VERIFIED in-train episodes of the anchor's val set.
# Source: raw/anchor_overlap.json — sha256 of raw pose bytes AND of the raw
# frame bytes, both matching; `episode_id` agrees as a cross-check only.
INTRAIN_TAGS = ("cm_00018", "cm_00039")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# metrics — copied verbatim from `idm2_lib.chan_metrics` / `spearman` rather    #
# than imported, because importing that module would insert the stale pod path #
# --------------------------------------------------------------------------- #
def spearman(a, b) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def chan_metrics(pred, gt) -> dict:
    p = np.asarray(pred, np.float64); g = np.asarray(gt, np.float64)
    err = p - g
    mad = float(np.median(np.abs(g - np.median(g))))
    return {"r2": float(1.0 - (err ** 2).sum()
                        / max(((g - g.mean()) ** 2).sum(), 1e-12)),
            "rho": spearman(p, g),
            "mae": float(np.abs(err).mean()),
            "medae": float(np.median(np.abs(err))),
            "nmedae": float(np.median(np.abs(err)) / max(mad, 1e-12)),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "gt_std": float(g.std()), "gt_mad": mad,
            "n_impossible_gt1p5": int((np.abs(g) > 1.5).sum()),
            "n": int(g.size)}


def boot_r2(pred, gt, eid) -> dict:
    p = np.asarray(pred, np.float64); g = np.asarray(gt, np.float64)

    def _r2(idx):
        i = idx.astype(np.int64); gg, pp = g[i], p[i]
        return float(1.0 - ((pp - gg) ** 2).sum()
                     / max(((gg - gg.mean()) ** 2).sum(), 1e-12))
    _r2.__name__ = "r2"
    return tci.episode_cluster_bootstrap(np.arange(p.size, dtype=np.float64),
                                         eid, reduce=_r2, n_boot=N_BOOT, seed=0)


def boot_delta_r2(pred, gt_a, gt_b, eid) -> dict:
    """CI on R2(vs gt_a) - R2(vs gt_b), resampling EPISODES jointly (paired)."""
    p = np.asarray(pred, np.float64)
    ga = np.asarray(gt_a, np.float64); gb = np.asarray(gt_b, np.float64)

    def _d(idx):
        i = idx.astype(np.int64)
        out = []
        for g in (ga, gb):
            gg, pp = g[i], p[i]
            out.append(1.0 - ((pp - gg) ** 2).sum()
                       / max(((gg - gg.mean()) ** 2).sum(), 1e-12))
        return float(out[0] - out[1])
    _d.__name__ = "delta_r2"
    r = tci.episode_cluster_bootstrap(np.arange(p.size, dtype=np.float64),
                                      eid, reduce=_d, n_boot=N_BOOT, seed=0)
    r["separated"] = bool(r["lo"] > 0 or r["hi"] < 0)
    return r


def boot_mae_delta(pred, gt_a, gt_b, eid) -> dict:
    """Paired CI on MAE(vs gt_a) - MAE(vs gt_b): per-window |err| both ways."""
    a = np.abs(np.asarray(pred, np.float64) - np.asarray(gt_a, np.float64))
    b = np.abs(np.asarray(pred, np.float64) - np.asarray(gt_b, np.float64))
    return tci.paired_episode_cluster_bootstrap(a, b, eid, n_boot=N_BOOT,
                                                seed=0, reduce="mean")


# --------------------------------------------------------------------------- #
# the repair + the admissibility mask — INDEPENDENT reimplementation, then      #
# PINNED against the labels the anchor was actually scored on                   #
# --------------------------------------------------------------------------- #
def hold_heading(yaw: np.ndarray, v: np.ndarray, v_min: float):
    """`comma2k19.hold_heading_through_standstill`, reimplemented. Returns
    (yaw_fixed, observable). NO-OP on a wholly-stationary segment — deliberately,
    and that no-op is the entire subject of the admissibility question."""
    yaw = np.asarray(yaw, np.float64); v = np.asarray(v, np.float64)
    obs = v >= v_min
    if not obs.any():
        return yaw.copy(), obs
    idx = np.where(obs, np.arange(yaw.size), -1)
    np.maximum.accumulate(idx, out=idx)
    idx[idx < 0] = int(np.argmax(obs))
    return np.arctan2(np.sin(yaw)[idx], np.cos(yaw)[idx]), obs


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def valid_centers(T: int, k: int, horizons, stride: int) -> np.ndarray:
    lo = max(k, 1)
    hi = T - 1 - max(k, max(horizons))
    if hi < lo:
        return np.empty(0, np.int64)
    return np.arange(lo, hi + 1, stride, dtype=np.int64)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/anchor_resettlement.json")
    args = ap.parse_args()

    assert md5_of(HEAD_PT) == HEAD_MD5, "⛔ deployed head md5 mismatch"

    gt = np.load(VAL_GT, allow_pickle=True).item()
    S = gt["S"].astype(np.float64)            # REPAIRED labels (v_min 0.5)
    Sleg = gt["S_leg"].astype(np.float64)     # LEGACY labels
    eid = np.asarray(gt["eid"]); dom = np.asarray(gt["dom"])
    A0 = np.load(A0_PREDS, allow_pickle=True).item()["S"].astype(np.float64)
    assert A0.shape[0] == S.shape[0] == len(eid), (A0.shape, S.shape, len(eid))
    log(f"windows {A0.shape[0]}  episodes {len(set(eid))}")

    # ---- rebuild the admissibility mask, and PIN the repair ---------------- #
    obs_win = np.zeros(len(eid), bool)
    repaired_mine = np.full(len(eid), np.nan)
    legacy_mine = np.full(len(eid), np.nan)
    per_ep = {}
    import torch
    for tag in sorted(set(eid)):
        m = np.where(eid == tag)[0]
        assert np.array_equal(m, np.arange(m[0], m[0] + m.size)), \
            f"{tag}: windows are not contiguous — mask alignment unsafe"
        d = torch.load(LAT / f"{tag}.pt", weights_only=False)
        po = d["poses"].float().numpy().astype(np.float64)
        T = po.shape[0]
        t = valid_centers(T, K_BUILD, HORIZONS, STRIDE)
        assert t.size == m.size, f"{tag}: {t.size} centers vs {m.size} windows"
        yaw_raw, v = po[:, 2], po[:, 3]
        if tag.startswith("cm_"):
            yaw_fix, obs = hold_heading(yaw_raw, v, V_MIN)
        else:                       # PhysicalAI heading is quaternion-derived:
            yaw_fix, obs = yaw_raw.copy(), np.ones(T, bool)   # always defined
        legacy_mine[m] = wrap_to_pi(yaw_raw[t + 1] - yaw_raw[t - 1]) / (2 * 0.1)
        repaired_mine[m] = wrap_to_pi(yaw_fix[t + 1] - yaw_fix[t - 1]) / (2 * 0.1)
        obs_win[m] = obs[t - 1] & obs[t] & obs[t + 1]
        per_ep[tag] = {"T": T, "n_windows": int(m.size),
                       "n_observable_frames": int(obs.sum()),
                       "v_max": float(v.max()), "v_mean": float(v.mean()),
                       "n_windows_admissible": int(obs_win[m].sum()),
                       "episode_id": int(d["episode_id"]), "src": d.get("src", "")}

    pin = {
        "repaired_max_abs_delta_vs_val_gt": float(np.abs(repaired_mine - S[:, 1]).max()),
        "legacy_max_abs_delta_vs_val_gt": float(np.abs(legacy_mine - Sleg[:, 1]).max()),
        "reading": "an INDEPENDENT reimplementation of the repair reproduces the "
                   "labels the anchor was actually scored against; if these are "
                   "~0 the admissibility mask below is aligned to the same "
                   "windows and the same repair",
    }
    log(f"PIN repaired delta {pin['repaired_max_abs_delta_vs_val_gt']:.3e}  "
        f"legacy delta {pin['legacy_max_abs_delta_vs_val_gt']:.3e}")

    # ---- subsets ---------------------------------------------------------- #
    is_cm = dom == "cm"
    is_pai = dom == "pai"
    intrain = np.isin(eid, INTRAIN_TAGS)
    subsets = {
        "cm_ALL22_the_published_anchor": is_cm,
        "cm_CLEAN20_content_verified_disjoint": is_cm & ~intrain,
        "cm_INTRAIN2_content_verified_in_train": is_cm & intrain,
        "pai_ALL14": is_pai,
    }

    def block(mask, extra_mask=None, label="repaired"):
        m = mask if extra_mask is None else (mask & extra_mask)
        if m.sum() == 0:
            return {"n": 0}
        g = {"legacy": Sleg, "repaired": S}[label][:, 1][m]
        p = A0[:, 1][m]
        out = chan_metrics(p, g)
        out["r2_ci"] = boot_r2(p, g, eid[m])
        out["n_episodes"] = int(len(set(eid[m])))
        return out

    res = {
        "what": "the comma heading-repair ANCHOR re-settled after a content-based "
                "overlap probe found it partly IN-TRAIN",
        "date": "2026-07-27",
        "agent": "anchor-settlement",
        "evidence_class": "MEASURED (ours; tanitad-eval, this script)",
        "tier": "decision-grade for the anchor substrate; nothing retrained, "
                "nothing re-encoded — the persisted A0 predictions are re-used",
        "host": __import__("socket").gethostname(),
        "estimator": "taniteval.ci episode_cluster_bootstrap (callable r2 "
                     "reducer) / paired_episode_cluster_bootstrap, B=2000, "
                     "unit = the episode. overlapping_holdout_se NOT used.",
        "ci_py_imported": tci.__file__, "ci_py_md5": _ci_md5,
        "pins": {"head_md5": HEAD_MD5, "val_gt": md5_of(VAL_GT),
                 "a0_preds": md5_of(A0_PREDS), "label_repair_pin": pin},
        "substrate": {
            "cache": "comma2k19-val-76b6e94a97a1 (64 segs) + "
                     "physicalai-val-0c5f7dac3b11 (40 eps)",
            "split": "idm2_lib.split_tags(val_every=3) — every 3rd episode",
            "k_build": K_BUILD, "stride": STRIDE, "v_min": V_MIN,
            "n_windows_total": int(A0.shape[0]),
            "n_windows_cm": int(is_cm.sum()), "n_windows_pai": int(is_pai.sum()),
            "intrain_tags": list(INTRAIN_TAGS),
            "n_windows_intrain": int(intrain.sum()),
        },
        "label_protocols": {
            "legacy": "poses[:,2] as cached (arctan2 of ENU velocity) — the "
                      "protocol every pre-2026-07-27 comma number used",
            "repaired": f"hold_heading_through_standstill, v_min={V_MIN}; NO "
                        "window dropped — this is the ANCHOR's protocol",
            "strict_admissible": f"repaired AND the `observable` mask required at "
                                 f"t-1, t, t+1 (v>={V_MIN}); windows whose centred "
                                 "heading difference is UNDEFINED are dropped",
        },
        "per_episode": per_ep,
        "results": {},
        "contrasts": {},
    }

    for name, mask in subsets.items():
        res["results"][name] = {
            "n_windows": int(mask.sum()), "n_episodes": int(len(set(eid[mask]))),
            "legacy": block(mask, None, "legacy"),
            "repaired": block(mask, None, "repaired"),
            "strict_admissible": block(mask, obs_win, "repaired"),
            "strict_admissible_kept": int((mask & obs_win).sum()),
            "strict_admissible_dropped": int((mask & ~obs_win).sum()),
        }
        m = mask
        res["contrasts"][name] = {
            "delta_r2_repaired_minus_legacy": boot_delta_r2(
                A0[m, 1], S[m, 1], Sleg[m, 1], eid[m]),
            "delta_mae_repaired_minus_legacy": boot_mae_delta(
                A0[m, 1], S[m, 1], Sleg[m, 1], eid[m]),
        }
        r = res["results"][name]
        log(f"{name:42s} n={r['n_windows']:5d}  legacy R2 {r['legacy']['r2']:+.6f}"
            f"  repaired {r['repaired']['r2']:+.6f}"
            f"  strict {r['strict_admissible'].get('r2', float('nan')):+.6f}")

    # ---- PhysicalAI: the "unaffected" claim, MEASURED not inherited -------- #
    n_pai_changed = int((np.abs(S[is_pai, 1] - Sleg[is_pai, 1]) > 1e-12).sum())
    vspeed = S[:, 0]
    pai_gate = is_pai & (vspeed >= V_MIN)
    res["physicalai_control"] = {
        "n_pai_windows_changed_by_repair": n_pai_changed,
        "must_be_zero": True,
        "r2_bit_identical": bool(res["results"]["pai_ALL14"]["legacy"]["r2"]
                                 == res["results"]["pai_ALL14"]["repaired"]["r2"]),
        "strict_admissible_dropped": res["results"]["pai_ALL14"]["strict_admissible_dropped"],
        "note": "PhysicalAI heading comes from an orientation quaternion, which "
                "is standstill-robust, so the producer-level admissibility mask "
                "is all-True by construction and drops 0 windows.",
        "DIAGNOSTIC_same_speed_gate": {
            "what": "if the SAME v>=v_min gate were applied to PhysicalAI as a "
                    "blunt speed filter, what would it do? Separates 'the mask "
                    "removes undefined labels' from 'the mask removes slow "
                    "windows'.",
            "n_kept": int(pai_gate.sum()), "n_dropped": int((is_pai & ~pai_gate).sum()),
            **({"r2": chan_metrics(A0[pai_gate, 1], S[pai_gate, 1])["r2"]}
               if pai_gate.sum() else {}),
        },
    }

    # ---- the residual defect, per subset ---------------------------------- #
    res["residual_defect"] = {}
    for name, mask in subsets.items():
        rep = S[mask, 1]
        res["residual_defect"][name] = {
            "n_impossible_legacy": int((np.abs(Sleg[mask, 1]) > 1.5).sum()),
            "n_impossible_repaired": int((np.abs(rep) > 1.5).sum()),
            "n_impossible_strict_admissible": int(
                (np.abs(S[mask & obs_win, 1]) > 1.5).sum()),
            "gt_std_repaired": float(rep.std()),
            "gt_std_strict_admissible": float(S[mask & obs_win, 1].std())
            if (mask & obs_win).sum() else None,
        }

    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
