"""D-LAN-COV — DOES THE LAN LEAK GUARD LEAVE ANY SIGNAL ON REAL DRIVING DATA?

⚠️ NON-PARITY. The local cache is ``physicalai-train-14231cd29c74`` (400 eps),
NOT the sacred parity corpus ``physicalai-train-e438721ae894`` (2376 eps,
skip-hash f09e44db). This probe answers a MECHANISM question — "does the
speed-driven leak guard mask every anchor on real poses?" — which any real
driving poses can answer. ⛔ It must NEVER be quoted as a cross-arm result, and
it does not re-select the parity corpus.

PRE-REGISTERED, both outcomes committed before the run. The bar is not mine: it
is written into the instrument by its own author, ``refc_train.lan_stats``:

    "The number that matters: ``any_valid_frac``. The 4-way ``nav_cmd`` it
     replaces is valid on 0.21-0.25 of windows (MEASURED, all four arms) — if
     LAN is not materially higher, the input is not fixed and the experiment
     should not run."

  H4  LAN's any_valid_frac on real windows is MATERIALLY ABOVE 0.25.
      SUPPORTED  => the route input is genuinely fixed; the strategic-goal arm
                    may run, and `goal_str` will carry gradient.
      REFUTED    => the input is NOT fixed. Per the instrument's own rule the
                    experiment must NOT run; min_lead_m / arclengths_m need
                    re-derivation FIRST. This outcome is reported just as
                    loudly as the other.

  H5  The synthetic-corpus lan_valid_frac == 0.0 (MEASURED, D-LAN-PF) is an
      ARTIFACT of the synthetic episodes, not a property of LAN.
      SUPPORTED  => real any_valid_frac > 0.
      REFUTED    => real any_valid_frac ~ 0 too; the guard is the cause.

Controls (programme 6.2 — a guard must be shown ABLE TO FAIL):
  * CONSTANT-ONLY control: min_lead_m -> +inf must drive any_valid_frac to 0.
    If it does not, the probe is not measuring the guard.
  * NO-GUARD floor: min_lead_m -> 0 and t_pred_s -> 0 gives the ceiling the
    guard is trading against. The gap between floor and measured IS the cost.
Stratified by ego speed, because the guard is speed-driven by construction
(horizon_lead_m(v0, t_pred_s)); a single pooled scalar would hide the trade.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

STACK = Path("C:/Users/Admin/tanitad-wt/stack")
sys.path.insert(0, str(STACK))
sys.path.insert(0, str(STACK / "scripts"))

from tanitad.data.mixing import load_episode             # noqa: E402
from tanitad.data.lan import (LanConfig, horizon_lead_m,  # noqa: E402
                              lan_window_features)

CACHE = Path("C:/Users/Admin/tanitad-data/physicalai/_epcache/"
             "physicalai-train-14231cd29c74")
N_EPISODES = 60
WINDOW = 8            # refc window; the anchor index is t + WINDOW - 1
SEED = 0
NAV_CMD_BASELINE = (0.21, 0.25)


def collect(cfg: LanConfig, t_pred_s: float, poses_list, idx):
    k = cfg.k
    per_anchor = np.zeros(k)
    any_valid = 0
    speeds = []
    valid_by_speed: dict[str, list[int]] = {}
    for e_i, t in idx:
        p = poses_list[e_i]
        f = lan_window_features(p, t, cfg, t_pred_s=t_pred_s)
        v = f.reshape(k, -1)[:, 3]
        per_anchor += v
        ok = int(bool(v.any()))
        any_valid += ok
        v0 = float(p[t, 3])
        speeds.append(v0)
        band = ("0-5" if v0 < 5 else "5-10" if v0 < 10 else
                "10-15" if v0 < 15 else "15-20" if v0 < 20 else "20+")
        valid_by_speed.setdefault(band, []).append(ok)
    n = max(len(idx), 1)
    return {
        "n_windows": n,
        "any_valid_frac": round(any_valid / n, 4),
        "per_anchor_valid_frac": [round(x / n, 4) for x in per_anchor],
        "speed_mean_mps": round(float(np.mean(speeds)), 3),
        "speed_median_mps": round(float(np.median(speeds)), 3),
        "any_valid_frac_by_speed": {
            b: {"n": len(v), "frac": round(sum(v) / len(v), 4)}
            for b, v in sorted(valid_by_speed.items())},
    }


def main() -> None:
    files = sorted(CACHE.glob("ep_*.pt"))[:N_EPISODES]
    if not files:
        raise SystemExit(f"no ep_*.pt under {CACHE}")
    poses_list = []
    for f in files:
        ep = load_episode(str(f), mmap=True)
        poses_list.append(np.asarray(ep.poses, dtype=np.float64))

    rng = np.random.default_rng(SEED)
    idx = []
    for e_i, p in enumerate(poses_list):
        # every legal window anchor in this episode
        hi = p.shape[0] - 1
        lo = WINDOW - 1
        if hi <= lo:
            continue
        for t in range(lo, hi):
            idx.append((e_i, t))
    rng.shuffle(idx)
    idx = idx[:8000]

    cfg = LanConfig()          # the shipped default: 20/40/80/160 m, lead 5 m
    out: dict = {
        "probe": "D-LAN-COV",
        "evidence_class": "MEASURED",
        "PARITY": "NON-PARITY — cache physicalai-train-14231cd29c74 "
                  "(400 eps) is NOT physicalai-train-e438721ae894/f09e44db. "
                  "Mechanism probe only; never quotable as a cross-arm result.",
        "cache": str(CACHE),
        "n_episodes_loaded": len(poses_list),
        "window": WINDOW,
        "seed": SEED,
        "cfg": {"arclengths_m": list(cfg.arclengths_m),
                "min_lead_m": cfg.min_lead_m, "lat_clip": cfg.lat_clip},
        "nav_cmd_baseline_valid_frac": list(NAV_CMD_BASELINE),
        "episode_len_median": float(np.median([p.shape[0]
                                               for p in poses_list])),
    }

    out["MEASURED_shipped_default"] = collect(cfg, 2.0, poses_list, idx)

    # ---- controls -----------------------------------------------------------
    out["CONTROL_constant_only_lead_inf"] = collect(
        LanConfig(arclengths_m=cfg.arclengths_m, min_lead_m=1e9), 2.0,
        poses_list, idx)
    out["CONTROL_no_guard_floor"] = collect(
        LanConfig(arclengths_m=cfg.arclengths_m, min_lead_m=0.0), 0.0,
        poses_list, idx)

    # ---- what the guard is actually asking for, in metres ------------------
    leads = [horizon_lead_m(v0=float(poses_list[e][t, 3]), t_pred_s=2.0,
                            cfg=cfg) for e, t in idx[:2000]]
    out["leak_guard_lead_m"] = {
        "mean": round(float(np.mean(leads)), 2),
        "median": round(float(np.median(leads)), 2),
        "p90": round(float(np.percentile(leads, 90)), 2),
        "max": round(float(np.max(leads)), 2),
        "shortest_arclength_m": cfg.arclengths_m[0],
        "frac_windows_where_lead_exceeds_shortest_anchor":
            round(float(np.mean([x > cfg.arclengths_m[0] for x in leads])), 4),
        "frac_windows_where_lead_exceeds_LONGEST_anchor":
            round(float(np.mean([x > cfg.arclengths_m[-1] for x in leads])), 4),
    }

    # ---- verdicts -----------------------------------------------------------
    m = out["MEASURED_shipped_default"]["any_valid_frac"]
    ctrl0 = out["CONTROL_constant_only_lead_inf"]["any_valid_frac"]
    floor = out["CONTROL_no_guard_floor"]["any_valid_frac"]
    out["CONTROL_able_to_fail"] = (ctrl0 == 0.0)
    out["H4_verdict"] = "SUPPORTED" if m > NAV_CMD_BASELINE[1] else "REFUTED"
    out["H4_margin_over_nav_cmd"] = round(m - NAV_CMD_BASELINE[1], 4)
    out["H5_verdict"] = "SUPPORTED" if m > 0.0 else "REFUTED"
    out["guard_cost_frac"] = round(floor - m, 4)
    out["INTERPRETATION"] = (
        "any_valid_frac is the fraction of windows on which the strategic goal "
        "head receives ANY gradient. It bounds the LAN experiment from above.")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
