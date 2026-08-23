"""S3 characterisation driver -- mines the decision points, measures coverage /
class balance / the ttm distribution, runs the circularity firewall, and emits
the power statement.

Runs on ANY PhysicalAI epcache (``ep_*.pt`` with a ``poses`` [T,4] tensor).
Prints the resolved cache and its PARITY STATUS on every run, and REFUSES the
78.5 %-leaked val split by name. The decision-grade run is a ``--cache-dir``
swap to the parity caches -- zero code change.

CPU only. No GPU, no pod SSH, no training launched.

Usage::

    python run_s3_characterisation.py \
        --train-cache <dir with ep_*.pt> --test-cache <dir with ep_*.pt> \
        --out-dir . [--limit N] [--horizon-s 12] [--min-ttm-s 1.0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s3_labels as S3                                          # noqa: E402
from s3_blind_baseline import (blind_conditioning_baseline,     # noqa: E402
                               refusal_verdict)

import refb_labels                                              # noqa: E402
from tanitad.lake.vocab import VTARGET_TOKENS                   # noqa: E402
from tanitad.lake.vtarget import vtarget_v2                     # noqa: E402

PARITY_TRAIN = "physicalai-train-e438721ae894"
PARITY_VAL = "physicalai-val-0c5f7dac3b11"
LEAKED_VAL = "physicalai-val-f1b378f295ae"          # 78.5 % into parity train
N_VT = len(VTARGET_TOKENS)


# ===========================================================================
# cache resolution -- loud, and it refuses the leaked split
# ===========================================================================
def resolve_cache(path: str, role: str) -> dict:
    p = Path(path)
    if LEAKED_VAL in p.as_posix():
        raise SystemExit(
            f"[S3] REFUSING {role}={p}: '{LEAKED_VAL}' leaks 78.5 % into the "
            f"parity train set. Use '{PARITY_VAL}'.")
    files = sorted(p.glob("ep_*.pt"))
    if not files:
        raise SystemExit(f"[S3] no ep_*.pt under {p}")
    name = p.name
    parity = (name == PARITY_TRAIN if role == "train"
              else name == PARITY_VAL)
    info = {"role": role, "dir": p.as_posix(), "cache_name": name,
            "n_episode_files": len(files),
            "is_parity_cache": bool(parity),
            "parity_expected": PARITY_TRAIN if role == "train" else PARITY_VAL}
    print(f"[S3] {role} cache = {p}  ({len(files)} ep_*.pt)  "
          f"PARITY={'YES' if parity else 'NO -- distributional only'}",
          flush=True)
    return info


# ===========================================================================
# X_cond -- EXACTLY the model's inference-time conditioning channels
# ===========================================================================
# MEASURED (stack/tanitad/models/flagship_v4.py:198-203, fed at eval by
# stack/scripts/eval_flagship_v4.py:333 via train_flagship_v4._goal_inputs):
#   forward(states, v0, imagined, vt_band, route, route_graded, vt_speed, ...)
# `states` are the pixels (excluded by definition of a BLIND baseline). The
# remaining channels are what a scene-blind head gets -- and four of them are
# themselves FUTURE-DERIVED, which is the real circularity risk for S3.
COND_GROUPS = {
    # B1: ego + in-window kinematics the encoder trivially recovers
    "sensor": ["v0", "win_v_mean", "win_dv", "win_kappa_mean",
               "win_kappa_abs_mean", "win_a_abs_mean"],
    # B2 adds: the STRATEGIC conditioning (future-derived, 25 s lookahead)
    "route": ["route_0", "route_1", "route_2", "route_3", "route_graded"],
    # B3 adds: the SPEED-TARGET conditioning (future-derived, >=5 s lookahead)
    "vtarget": ["vt_band_n", "vt_speed", "vt_valid"],
    # B4 adds: the censoring/clock channel -- the S3-specific artifact probe
    "clock": ["obs_h_steps_n"],
}
BLIND_ARMS = {
    "B1_sensor_only": ["sensor"],
    "B2_plus_route": ["sensor", "route"],
    "B3_FULL_CONDITIONING": ["sensor", "route", "vtarget"],
    "B4_plus_clock": ["sensor", "route", "vtarget", "clock"],
}


def episode_cond(poses: torch.Tensor, rows: list[dict]) -> None:
    """Attach the conditioning channels to every admissible row, in place.

    ``route``/``route_graded`` come from ``route_from_future_v3`` at the
    DEFAULT 25 s lookahead and ``vt_band``/``vt_speed`` from ``vtarget_v2``
    with ``min_lookahead=50`` -- i.e. minted exactly as ``v4_labels.mint_episode``
    mints the fields the head is fed.
    """
    keep = [r for r in rows if r.get("m1") and r.get("m4")]
    if not keep:
        return
    v = poses[:, 3].numpy().astype(np.float64)
    last_ix = np.array([r["L"] for r in keep], dtype=np.int64)
    vt2, vt_ok, _look, _ = vtarget_v2(v, last_ix, min_lookahead=50)
    for i, r in enumerate(keep):
        L = r["L"]
        r3 = refb_labels.route_from_future_v3(poses, L)     # 25 s, as shipped
        rc = int(r3["route"])
        for c in range(4):
            r[f"route_{c}"] = 1.0 if rc == c else 0.0
        r["route_graded"] = float(r3["graded_route"])
        r["vt_speed"] = float(vt2[i])
        r["vt_valid"] = 1.0 if bool(vt_ok[i]) else 0.0
        r["vt_band_n"] = (float(vt2[i]) / 30.0) if bool(vt_ok[i]) else -1.0
        r["v0"] = float(poses[L, 3])
        r["obs_h_steps_n"] = float(r["obs_h_steps"]) / 200.0


def mine_cache(cache_dir: str, horizon_s: float, min_ttm_s: float,
               limit: int = 0) -> list[dict]:
    files = sorted(Path(cache_dir).glob("ep_*.pt"))
    if limit:
        files = files[:limit]
    out, t0 = [], time.time()
    for i, f in enumerate(files):
        ep = torch.load(f, weights_only=False, map_location="cpu")
        poses = torch.as_tensor(ep["poses"], dtype=torch.float32)
        rows = S3.mine_episode(poses, eid=f.stem, horizon_s=horizon_s,
                               min_ttm_s=min_ttm_s)
        episode_cond(poses, rows)
        out.extend(rows)
        if i % 25 == 0:
            print(f"  [{Path(cache_dir).name}] {i}/{len(files)} eps  "
                  f"{time.time() - t0:.0f}s  rows={len(out)}", flush=True)
    print(f"  [{Path(cache_dir).name}] DONE {len(files)} eps  "
          f"{time.time() - t0:.0f}s  rows={len(out)}", flush=True)
    return out


# ===========================================================================
# reporting
# ===========================================================================
def _dist(vals) -> dict:
    a = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    q = np.percentile(a, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    return {"n": int(a.size), "mean": round(float(a.mean()), 3),
            "sd": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 3),
            "deciles_s": [round(float(x), 3) for x in q],
            "hist_1s_bins": np.histogram(
                a, bins=np.arange(0, 13, 1.0))[0].tolist()}


def coverage_block(rows: list[dict], axis: str, horizon_s: float) -> dict:
    n_all = len(rows)
    m1 = sum(1 for r in rows if r["m1"])
    m14 = sum(1 for r in rows if r["m1"] and r["m4"])
    adm = [r for r in rows if r.get(f"{axis}_admissible")]
    m2r = sum(1 for r in rows if r.get("m1") and r.get("m4")
              and not r.get(f"m2_{axis}", True))
    m3r = sum(1 for r in rows if r.get("m1") and r.get("m4")
              and r.get(f"m2_{axis}", True) and not r.get(f"m3_{axis}", True))
    bands = np.array([r[f"band_{axis}"] for r in adm], dtype=np.int64)
    cnt = np.bincount(bands, minlength=S3.N_BANDS) if bands.size else \
        np.zeros(S3.N_BANDS, dtype=np.int64)
    ev = [r[f"ttm_{axis}"] for r in adm
          if r.get(f"ttm_{axis}_ok") and np.isfinite(r[f"ttm_{axis}"])]
    eps_all = {r["eid"] for r in rows}
    eps_adm = {r["eid"] for r in adm}
    eps_event = {r["eid"] for r in adm if r[f"band_{axis}"] != S3.IX_NONE}
    return {
        "axis": axis,
        "horizon_s": horizon_s,
        "n_windows_total": n_all,
        "n_after_M1_observable_horizon": m1,
        "n_after_M1_M4_moving": m14,
        "n_admissible": len(adm),
        "rejected_by_M2_already_begun": m2r,
        "rejected_by_M3_window_executing": m3r,
        "coverage_of_all_windows": round(len(adm) / max(1, n_all), 4),
        "coverage_of_M1M4_windows": round(len(adm) / max(1, m14), 4),
        "class_counts": {S3.BAND_NAMES[i]: int(cnt[i])
                         for i in range(S3.N_BANDS)},
        "class_balance": {S3.BAND_NAMES[i]: round(float(cnt[i] / max(1, cnt.sum())), 4)
                          for i in range(S3.N_BANDS)},
        "majority_class": S3.BAND_NAMES[int(cnt.argmax())] if cnt.sum() else None,
        "majority_rate": round(float(cnt.max() / max(1, cnt.sum())), 4),
        "event_rate": round(float(1.0 - cnt[S3.IX_NONE] / max(1, cnt.sum())), 4),
        "ttm_distribution_event_windows": _dist(ev),
        "equal_mass_quartile_edges_s": (
            [round(float(x), 3) for x in np.percentile(ev, [25, 50, 75])]
            if len(ev) >= 4 else None),
        "n_episodes_total": len(eps_all),
        "n_episodes_with_decision_point": len(eps_adm),
        "n_episodes_with_an_EVENT": len(eps_event),
        "episode_yield_rate": round(len(eps_adm) / max(1, len(eps_all)), 4),
    }


# speed strata -- the corpus's own regimes (parity_profile.py: stopped / city /
# hw), so an S3 result is never a single pooled number (spec HP-2 needs strata).
SPEED_STRATA = (("city_lt8", 0.0, 8.0), ("mid_8_15", 8.0, 15.0),
                ("hw_ge15", 15.0, 1e9))


def stratum_block(rows: list[dict], axis: str) -> dict:
    """Per-speed-regime coverage, class balance and episode clusters.

    A pooled S3 number cannot answer HP-2 ("does the advantage concentrate at
    decision points, or is it uniform?"), which is the question S3 exists to
    help answer. Strata are reported at MINT time so nobody has to re-derive
    them at score time.
    """
    out = {}
    for name, lo, hi in SPEED_STRATA:
        sel = [r for r in rows if r.get(f"{axis}_admissible")
               and lo <= r.get("win_v_mean", -1.0) < hi]
        c = (np.bincount(np.array([r[f"band_{axis}"] for r in sel],
                                  dtype=np.int64), minlength=S3.N_BANDS)
             if sel else np.zeros(S3.N_BANDS, dtype=np.int64))
        eps = {r["eid"] for r in sel}
        ev = [r[f"ttm_{axis}"] for r in sel
              if r.get(f"ttm_{axis}_ok") and np.isfinite(r[f"ttm_{axis}"])]
        out[name] = {
            "speed_mps": [lo, None if hi > 1e8 else hi],
            "n_windows": len(sel),
            "n_episode_clusters": len(eps),
            "meets_single_arm_bar_40": bool(len(eps) >= 40),
            "meets_two_arm_bar_200": bool(len(eps) >= 200),
            "class_balance": {S3.BAND_NAMES[i]: round(float(c[i] / max(1, c.sum())), 4)
                              for i in range(S3.N_BANDS)},
            "majority_rate": round(float(c.max() / max(1, c.sum())), 4),
            "median_ttm_s": round(float(np.median(ev)), 3) if ev else None,
        }
    return out


def power_block(cov_test: dict, cov_train: dict) -> dict:
    """Episode-clusters against the standing bars (spec §0.4)."""
    yld = cov_test["episode_yield_rate"]
    yld_ev = cov_test["n_episodes_with_an_EVENT"] / max(
        1, cov_test["n_episodes_total"])
    def proj(n):
        return {"n_val_episodes": n,
                "projected_clusters_with_decision_point": int(round(n * yld)),
                "projected_clusters_with_an_event": int(round(n * yld_ev))}
    return {
        "bars": {"single_arm_min_clusters": 40, "two_arm_min_clusters": 200},
        "measured_on_test_cache": {
            "n_episodes": cov_test["n_episodes_total"],
            "clusters_with_decision_point":
                cov_test["n_episodes_with_decision_point"],
            "clusters_with_an_event": cov_test["n_episodes_with_an_EVENT"],
            "episode_yield_rate": yld},
        "projections": {
            "published_eval_val_40ep": proj(40),
            "pod1_valdata_12ep": proj(12),
            "training_pod_val_600ep": proj(600),
            "parity_train_2376ep": proj(2376)},
        "train_cache_clusters": cov_train["n_episodes_with_decision_point"],
    }


def option_set_block(rows: list[dict], axis: str, horizon_s: float,
                     edges_secondary=None) -> dict:
    """The option set, justified from the data's OWN distribution.

    Reports the PRIMARY banding (the spec's own 2/5/10 s edges) and the
    pre-registered SECONDARY equal-mass quartile banding side by side, each with
    its majority-class rate -- the number the primary metric must beat. A band
    table without the underlying distribution is not an admissible S3 report,
    so the full decile/histogram is carried too.
    """
    adm = [r for r in rows if r.get(f"{axis}_admissible")]
    ev = np.array([r[f"ttm_{axis}"] for r in adm
                   if r.get(f"ttm_{axis}_ok")
                   and np.isfinite(r[f"ttm_{axis}"])], dtype=np.float64)
    def _bal(edges):
        b = np.array([S3.band_of(r[f"ttm_{axis}"], r[f"ttm_{axis}_ok"],
                                 edges=edges, horizon_s=horizon_s)
                      for r in adm], dtype=np.int64)
        c = np.bincount(b, minlength=S3.N_BANDS)
        names = [f"[{lo},{hi})" for lo, hi in
                 zip((S3.MIN_TTM_S,) + tuple(edges), tuple(edges) + (horizon_s,))]
        return {"edges_s": list(edges),
                "interval_names": names + ["t_none"],
                "counts": [int(x) for x in c],
                "balance": [round(float(x), 4) for x in c / max(1, c.sum())],
                "majority_rate": round(float(c.max() / max(1, c.sum())), 4),
                "qwk_of_majority_baseline": 0.0,     # exact, by construction
                "n": int(c.sum())}
    sec = (edges_secondary if edges_secondary is not None else
           (tuple(np.percentile(ev, [25, 50, 75])) if ev.size >= 4 else None))
    med = float(np.median(ev)) if ev.size else float("nan")
    return {
        "axis": axis, "horizon_s": horizon_s, "min_ttm_s": S3.MIN_TTM_S,
        "PRIMARY_spec_edges": _bal(S3.BAND_EDGES_S),
        "SECONDARY_equal_mass_quartiles": (_bal(tuple(sec)) if sec else None),
        "full_distribution_event_windows": _dist(ev),
        "mae_median_constant_baseline": {
            "median_const_s": round(med, 4),
            "mae_of_median_constant_s": round(
                float(np.abs(ev - med).mean()), 4) if ev.size else None,
            "note": "the MAE-optimal constant; ttm_MAE_skill_s is measured "
                    "against THIS, never against 0"},
    }


def build_X(rows: list[dict], groups: list[str]) -> tuple[np.ndarray, list[str]]:
    names: list[str] = []
    for g in groups:
        names += COND_GROUPS[g]
    X = np.array([[float(r.get(n, 0.0)) for n in names] for r in rows],
                 dtype=np.float64)
    return X, names


def run_firewall(tr_rows, te_rows, axis: str, seed: int = 0) -> dict:
    """The MANDATORY pre-flight. B0 majority -> B1 -> B2 -> B3 -> B4."""
    tr = [r for r in tr_rows if r.get(f"{axis}_admissible")]
    te = [r for r in te_rows if r.get(f"{axis}_admissible")]
    ytr = np.array([r[f"band_{axis}"] for r in tr], dtype=np.int64)
    yte = np.array([r[f"band_{axis}"] for r in te], dtype=np.int64)
    eid_te = [r["eid"] for r in te]
    score = lambda a, b: S3.band_metrics(a, b)            # noqa: E731

    out = {"axis": axis, "n_train": int(ytr.size), "n_test": int(yte.size),
           "n_test_episodes": len(set(eid_te)), "arms": {}}
    preds: dict[str, np.ndarray] = {}
    for label, groups in BLIND_ARMS.items():
        Xtr, names = build_X(tr, groups)
        Xte, _ = build_X(te, groups)
        r = blind_conditioning_baseline(Xtr, ytr, Xte, yte, eid_te,
                                        S3.N_BANDS, score,
                                        feature_names=names, label=label,
                                        seed=seed)
        preds[label] = r.pop("pred")
        r["blind_qwk_ci"] = S3.qwk_bootstrap(yte, preds[label], eid_te,
                                             n_boot=2000)
        out["arms"][label] = r
    q = {k: v["blind"]["qwk"] for k, v in out["arms"].items()}
    b1, b2 = q["B1_sensor_only"], q["B2_plus_route"]
    b3, b4 = q["B3_FULL_CONDITIONING"], q["B4_plus_clock"]

    # The OPERATIVE blind floor is the MAX over the non-clock arms: a firewall
    # must not be passed by an MLP that happened to overfit one feature set.
    floor_label = max(("B1_sensor_only", "B2_plus_route",
                       "B3_FULL_CONDITIONING"), key=lambda k: q[k])
    out["operative_blind_floor"] = {
        "arm": floor_label, "qwk": q[floor_label],
        "rule": "max over B1/B2/B3 -- skill = QWK(model) - this. Using B3 alone "
                "would let a capacity artefact in one arm lower the bar."}
    out["verdict_R1_circular"] = refusal_verdict(q[floor_label], ceiling=1.0)

    # paired intervals on the two attribution deltas (same windows, same
    # resampled episodes each draw -- never a quadrature combination)
    out["paired_leak_B2_minus_B1"] = S3.paired_qwk_bootstrap(
        yte, preds["B2_plus_route"], preds["B1_sensor_only"], eid_te)
    out["paired_leak_B3_minus_B1"] = S3.paired_qwk_bootstrap(
        yte, preds["B3_FULL_CONDITIONING"], preds["B1_sensor_only"], eid_te)
    out["paired_clock_B4_minus_B3"] = S3.paired_qwk_bootstrap(
        yte, preds["B4_plus_clock"], preds["B3_FULL_CONDITIONING"], eid_te)
    out["deltas"] = {"B2_minus_B1": round(b2 - b1, 4),
                     "B3_minus_B1": round(b3 - b1, 4),
                     "B4_minus_B3": round(b4 - b3, 4)}
    out["pre_registered_skill_bars"] = {
        "S3_as_specified_route_and_vt_GIVEN": q[floor_label],
        "S3_W_withheld_conditioning_pixels_and_v0_only":
            q["B1_sensor_only"],
        "note": "a model must clear these, not 0. PRE_REGISTRATION_S3.md §5.3/§7 R2."}
    return out


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser("run_s3_characterisation")
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--test-cache", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--horizon-s", type=float, default=S3.H_S3_S)
    ap.add_argument("--min-ttm-s", type=float, default=S3.MIN_TTM_S)
    ap.add_argument("--tag", default="primary")
    a = ap.parse_args(argv)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prov = {"artifact": "S3 manoeuvre-initiation-timing characterisation",
            "date": "2026-07-26", "horizon_s": a.horizon_s,
            "min_ttm_s": a.min_ttm_s, "limit": a.limit, "tag": a.tag,
            "band_names": list(S3.BAND_NAMES),
            "band_edges_s": list(S3.BAND_EDGES_S),
            "estimator": "episode_cluster_bootstrap (taniteval/ci.py) B=2000; "
                         "overlapping_holdout_se NEVER used",
            "caches": [resolve_cache(a.train_cache, "train"),
                       resolve_cache(a.test_cache, "val")]}

    t0 = time.time()
    tr_rows = mine_cache(a.train_cache, a.horizon_s, a.min_ttm_s, a.limit)
    te_rows = mine_cache(a.test_cache, a.horizon_s, a.min_ttm_s, a.limit)
    prov["mine_seconds"] = round(time.time() - t0, 1)

    cov = {"provenance": prov}
    for axis in ("lat", "lon"):
        cov[f"train_{axis}"] = coverage_block(tr_rows, axis, a.horizon_s)
        cov[f"val_{axis}"] = coverage_block(te_rows, axis, a.horizon_s)
    (out / f"s3_coverage_{a.tag}.json").write_text(json.dumps(cov, indent=2))
    print(json.dumps({k: v for k, v in cov.items() if k.startswith("val_")},
                     indent=2)[:2500], flush=True)

    opt = {"provenance": prov}
    for axis in ("lat", "lon"):
        opt[f"train_{axis}"] = option_set_block(tr_rows, axis, a.horizon_s)
        opt[f"val_{axis}"] = option_set_block(te_rows, axis, a.horizon_s)
    (out / f"s3_option_set_{a.tag}.json").write_text(json.dumps(opt, indent=2))

    pw = {"provenance": prov,
          "lat": power_block(cov["val_lat"], cov["train_lat"]),
          "lon": power_block(cov["val_lon"], cov["train_lon"]),
          "strata_val": {ax: stratum_block(te_rows, ax) for ax in ("lat", "lon")},
          "strata_train": {ax: stratum_block(tr_rows, ax) for ax in ("lat", "lon")}}
    (out / f"s3_power_{a.tag}.json").write_text(json.dumps(pw, indent=2))

    fw = {"provenance": prov,
          "what_X_cond_is": {
              "source": "MEASURED stack/tanitad/models/flagship_v4.py:198-203 "
                        "+ stack/scripts/eval_flagship_v4.py:333",
              "channels": COND_GROUPS,
              "note": "route/route_graded/vt_band/vt_speed are THEMSELVES "
                      "future-derived and are fed at inference"},
          "lat": run_firewall(tr_rows, te_rows, "lat"),
          "lon": run_firewall(tr_rows, te_rows, "lon")}
    (out / f"s3_blind_baseline_{a.tag}.json").write_text(json.dumps(fw, indent=2))
    print(json.dumps({ax: {"R1": fw[ax]["verdict_R1_circular"],
                           "floor": fw[ax]["operative_blind_floor"],
                           "deltas": fw[ax]["deltas"],
                           "leak_B3_B1_ci": {
                               k: fw[ax]["paired_leak_B3_minus_B1"][k]
                               for k in ("delta", "lo", "hi", "separated")},
                           "clock_B4_B3_ci": {
                               k: fw[ax]["paired_clock_B4_minus_B3"][k]
                               for k in ("delta", "lo", "hi", "separated")}}
                      for ax in ("lat", "lon")}, indent=2), flush=True)
    print(f"[S3] wrote {out}/s3_"
          f"{{coverage,option_set,power,blind_baseline}}_{a.tag}.json",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
