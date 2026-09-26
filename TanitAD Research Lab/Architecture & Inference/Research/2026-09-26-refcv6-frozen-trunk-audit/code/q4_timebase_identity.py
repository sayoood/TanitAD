"""Q4 (advisory class D) -- the DATA-SIDE time base of refcv6, against an INDEPENDENT reference.

The question the 2026-09-22 review left open (DIFFUSION_PAPER_REVIEW.md s8 item 3):
*"Whether the corpus's true frame rate is 0.1 s -- settles it: one identity against a
LOG-RECORDED speed, per the advisory's own prescription, on any --v2-cache episode."*

What every consumer assumes: one cache row = 0.1 s (EgoHistoryConfig.dt, anchor_dt, v7_dt,
horizons in ticks, the 4 literal sites). What OWNS the rate: the cache builder,
`v2_compressed._resampled` / `physicalai.build_episode`:

    n_target = int(span_s * TARGET_HZ); t_query = np.linspace(t0, tN, n_target)

so the true step is span / (n_target - 1), which is > 0.1 s for every clip (floor + the n-1).

THE IDENTITY. poses[:, :2] (x, y) and poses[:, 3] (v = hypot(vx, vy)) are both interpolated
from the egomotion log at the SAME query times, and the log's velocity knows nothing about our
cadence. So, over moving rows, the true step is

    dt_true = sum_i |xy_{i+1} - xy_i|  /  sum_i (v_i + v_{i+1}) / 2          (identity 1)

Two more, weaker, from other log channels: yaw vs v*kappa (kappa = tan(steer)/WHEELBASE, the
builder's own inversion) and dv vs the log's own ax.

CONTROLS (each must read a KNOWN value, written as a literal):
  C1 analytic: a synthetic constant-speed arc sampled at a known dt -> the estimator returns it
  C2 mutation: the SAME real tracks subsampled every 2nd row -> the estimator must read 2x
     (the advisory's class-D row 2: a 0.2 s displacement divided by a 0.1 s step)
  C3 stationary rows are excluded (v > 2 m/s), and the count of used steps is reported.

Also derived: where the trainer's LABEL lookup `(t + w - 1) * v7_dt` lands against the label's
own timeline (egomotion_source: a TRUE 10 Hz grid from the recording start, anchor RAW_T0_S 8.0,
band +-2.0 s), versus where the row really is: raw index (row + n_stack - 1) at dt_true.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.bootstrap()
import torch  # noqa: E402

SCRATCH = C.SCRATCH
MANIFESTS = {
    "eval139 (kit, local)": C.KIT / "data/refcv6-b1-416x1024-eval139/_v2manifest.pt",
    "train (Thor view, scp md5 3c9f8bc8)": SCRATCH / "refcv6-b1-416x1024-train___v2manifest.pt",
}
V_MIN = 2.0          # m/s: moving rows only
WHEELBASE = 2.9      # physicalai.WHEELBASE (physicalai.py:63); literal, cross-checked below
DT_ASSUMED = 0.1     # the literal every consumer uses


def dt_identity_1(p: np.ndarray, stride: int = 1):
    """sum |dxy| / sum vbar over moving steps -> (dt_est, n_steps). p = [T, 4] (x, y, yaw, v)."""
    q = p[::stride]
    d = np.hypot(np.diff(q[:, 0]), np.diff(q[:, 1]))
    vb = 0.5 * (q[1:, 3] + q[:-1, 3])
    m = (q[1:, 3] > V_MIN) & (q[:-1, 3] > V_MIN)
    if m.sum() < 10:
        return float("nan"), int(m.sum())
    return float(d[m].sum() / vb[m].sum()), int(m.sum())


def dt_identity_2(p: np.ndarray, a: np.ndarray):
    """|dyaw| vs v*kappa, kappa = tan(steer)/WHEELBASE -> dt_est over turning moving steps."""
    yaw = np.unwrap(p[:, 2])
    dyaw = np.diff(yaw)
    kap = np.tan(a[:, 0]) / WHEELBASE
    kb = 0.5 * (kap[1:] + kap[:-1])
    vb = 0.5 * (p[1:, 3] + p[:-1, 3])
    rate = vb * kb
    m = (np.abs(kb) > 0.02) & (vb > 3.0) & (np.sign(dyaw) == np.sign(rate))
    if m.sum() < 10:
        return float("nan"), int(m.sum())
    return float(np.abs(dyaw[m]).sum() / np.abs(rate[m]).sum()), int(m.sum())


def dt_identity_3(p: np.ndarray, a: np.ndarray):
    """dv vs the log's own ax -> dt_est over accelerating steps (noisy; reported, not relied on)."""
    dv = np.diff(p[:, 3])
    ab = 0.5 * (a[1:, 1] + a[:-1, 1])
    m = (np.abs(ab) > 0.5) & (np.sign(dv) == np.sign(ab)) & (p[1:, 3] > V_MIN)
    if m.sum() < 10:
        return float("nan"), int(m.sum())
    return float(np.abs(dv[m]).sum() / np.abs(ab[m]).sum()), int(m.sum())


def controls() -> dict:
    out = {}
    # C1 analytic: constant speed 12 m/s on a radius-80 m arc, sampled at KNOWN dt
    for dt_known in (0.1, 0.1008403361, 0.2):
        t = np.arange(200) * dt_known
        R, v = 80.0, 12.0
        th = v * t / R
        p = np.stack([R * np.sin(th), R * (1 - np.cos(th)), th, np.full_like(t, v)], 1)
        a = np.stack([np.full_like(t, math.atan(WHEELBASE / R)), np.zeros_like(t)], 1)
        e1, _ = dt_identity_1(p)
        e2, _ = dt_identity_2(p, a)
        out[f"C1_arc_dt_{dt_known}"] = {"dt_known": dt_known, "id1": e1, "id2": e2,
                                        "id1_rel_err": e1 / dt_known - 1.0,
                                        "id2_rel_err": e2 / dt_known - 1.0}
    return out


def label_mapping(dt_true: float, n_stack: int, w: int, T_out: int) -> dict:
    """Trainer lookup t = row * 0.1 vs true (row + n_stack - 1) * dt_true; band |t - 8| <= 2."""
    t0, half = 8.0, 2.0
    rows = np.arange(w - 1, T_out)
    t_tr = rows * DT_ASSUMED
    t_true = (rows + n_stack - 1) * dt_true
    adm_tr = np.abs(t_tr - t0) <= half
    adm_true = np.abs(t_true - t0) <= half
    r_anchor_tr = int(round(t0 / DT_ASSUMED))
    return {
        "trainer_lookup": "t_now = (t + w - 1) * v7_dt, v7_dt = 0.1 (refc_v3_train.py:3009-3010, :3029-3030)",
        "true_time_of_row": "(row + n_stack - 1) * dt_true (v2_dataset.py:36: poses = payload['poses'][n_stack-1:])",
        "row_trainer_calls_anchor": r_anchor_tr,
        "true_time_at_that_row_s": float((r_anchor_tr + n_stack - 1) * dt_true),
        "offset_at_anchor_s": float((r_anchor_tr + n_stack - 1) * dt_true - t0),
        "offset_range_over_rows_s": [float((t_true - t_tr).min()), float((t_true - t_tr).max())],
        "n_rows_admitted_trainer": int(adm_tr.sum()),
        "n_rows_admitted_true": int(adm_true.sum()),
        "n_admitted_by_trainer_but_truly_outside": int((adm_tr & ~adm_true).sum()),
        "n_truly_inside_but_ignored_by_trainer": int((~adm_tr & adm_true).sum()),
        "frac_trainer_admitted_rows_outside_true_band": float((adm_tr & ~adm_true).sum()
                                                              / max(adm_tr.sum(), 1)),
    }


def run_manifest(name: str, path: Path) -> dict:
    m = torch.load(str(path), map_location="cpu", weights_only=False)
    n = len(m["poses"])
    res = {"name": name, "n_clips": n, "n_stack": sorted(set(int(x) for x in m["n_stack"])),
           "T_out_hist": {}, "id1": [], "id2": [], "id3": [], "id1_stride2": [],
           "n_steps_id1": 0, "n_steps_id2": 0, "n_steps_id3": 0, "n_clips_moving": 0}
    per_clip = []
    for i in range(n):
        p = m["poses"][i].double().numpy()
        a = m["actions"][i].double().numpy()
        T = int(p.shape[0])
        res["T_out_hist"][T] = res["T_out_hist"].get(T, 0) + 1
        e1, k1 = dt_identity_1(p)
        e2, k2 = dt_identity_2(p, a)
        e3, k3 = dt_identity_3(p, a)
        e1s, _ = dt_identity_1(p, stride=2)
        if not math.isnan(e1):
            res["id1"].append(e1)
            res["n_steps_id1"] += k1
            res["n_clips_moving"] += 1
            res["id1_stride2"].append(e1s)
        if not math.isnan(e2):
            res["id2"].append(e2)
            res["n_steps_id2"] += k2
        if not math.isnan(e3):
            res["id3"].append(e3)
            res["n_steps_id3"] += k3
        per_clip.append({"clip": C.sha12(m["clip_id"][i]), "T_out": T, "dt_id1": e1,
                         "dt_id2": e2, "dt_id3": e3, "steps_id1": k1})

    def q(v):
        v = np.asarray([x for x in v if not math.isnan(x)])
        if v.size == 0:
            return None
        return {"n": int(v.size), "median": float(np.median(v)), "mean": float(v.mean()),
                "p05": float(np.quantile(v, 0.05)), "p95": float(np.quantile(v, 0.95)),
                "min": float(v.min()), "max": float(v.max())}
    res["summary"] = {"id1_disp_vs_logspeed": q(res["id1"]),
                      "id2_yaw_vs_v_kappa": q(res["id2"]),
                      "id3_dv_vs_log_ax": q(res["id3"]),
                      "C2_mutation_stride2_id1": q(res["id1_stride2"])}
    # the builder formula's prediction for the dominant T_out
    Tdom = max(res["T_out_hist"], key=res["T_out_hist"].get)
    nq = Tdom + (res["n_stack"][0] - 1)
    res["builder_prediction_for_dominant_T"] = {
        "T_out": Tdom, "n_target": nq,
        "dt_true_interval_if_span_in_[n/10,(n+1)/10)": [nq / 10.0 / (nq - 1), (nq + 1) / 10.0 / (nq - 1)],
        "formula": "v2_compressed.py:119-120 n_target=int(span*10); linspace(t0,tN,n_target)"}
    med = res["summary"]["id1_disp_vs_logspeed"]["median"]
    res["label_mapping_at_measured_dt"] = label_mapping(med, res["n_stack"][0], 8, Tdom)
    res["per_clip"] = per_clip
    for k in ("id1", "id2", "id3", "id1_stride2"):
        res.pop(k)
    res["T_out_hist"] = {str(k): v for k, v in sorted(res["T_out_hist"].items())}
    return res


if __name__ == "__main__":
    # light job (two pose manifests, < 0.5 GB); the 8 GB floor is for model builds
    if C.ram_available_gb() < 1.5:
        raise SystemExit("[audit:RAM] < 1.5 GB available even for a light job")
    from tanitad.data import physicalai as pai
    wb_ok = float(pai.WHEELBASE) == WHEELBASE
    out = {"what": "Q4 data-side time base: row step vs the egomotion log's own velocity",
           "evidence_class": "MEASURED (ours, dev box CPU)",
           "dt_assumed_by_consumers": DT_ASSUMED,
           "wheelbase_literal_matches_physicalai": wb_ok,
           "v_min_ms": V_MIN,
           "builder_source": "v2_compressed.py:108-126 (linspace grid, n_target=int(span*10)) "
                             "and physicalai.py:596-649 (signals_at: x, y, v=hypot(vx,vy), ax from the log)",
           "controls": controls(), "manifests": {}}
    for name, path in MANIFESTS.items():
        if not Path(path).exists():
            out["manifests"][name] = {"error": f"missing {path}"}
            continue
        out["manifests"][name] = run_manifest(name, Path(path))
        s = out["manifests"][name]["summary"]
        print(name, "id1", s["id1_disp_vs_logspeed"], "\n   id2", s["id2_yaw_vs_v_kappa"],
              "\n   id3", s["id3_dv_vs_log_ax"], "\n   C2 stride2", s["C2_mutation_stride2_id1"],
              flush=True)
        print("   T_out hist", out["manifests"][name]["T_out_hist"])
        print("   builder prediction", out["manifests"][name]["builder_prediction_for_dominant_T"])
        print("   label mapping", out["manifests"][name]["label_mapping_at_measured_dt"])
    for k, v in out["controls"].items():
        print(k, v)
    C.write_json("q4_timebase_identity.json", out)
