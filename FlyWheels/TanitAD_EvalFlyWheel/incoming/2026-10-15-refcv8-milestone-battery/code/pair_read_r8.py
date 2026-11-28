"""pair_read_r8.py -- the PAIRED FOUR-FAMILY READ of refcv8 `os` against refcv7-50,400 `os`, per inference seed, S2 and S6.

    python pair_read_r8.py --root E:/refcv8_eval_kit/battery/step5000_unc --clock standalone \
        --g0-json E:/refcv8_eval_kit/battery/step5000/g0.json --out-json <...>/PAIRED_READ_step5000.json --out-md <...>.md

Reads the PANELS the refcv8 battery wrote (`panel_s{s}` = S2 grid, `s6_s{s}` = the 6-s subset), each of which carries the arms
`os`, `ha0_ext` (the echo), the model-free controls and `b_refcv7_50400_s{s}` (the banked refcv7-50,400 `os` plan, same
inference seed, put there by `build_panel_dump(baselines=)` after it asserted GT / model-free equality).  Per window it
recomputes the battery's own components (`refav1_arm._components`, four_families' geometry -- not re-derived) plus the SPEC
section 5 headline cells the shared component set lacks (curvature MAE masked, |along| and |cross| at the final instant,
target-speed accuracy), and scores them with `taniteval.ci.paired_episode_cluster_bootstrap` (B 2,000, seed 0, 95 % percentile),
refcv8 minus refcv7.  ESTIMATOR / QUESTION: episode-cluster bootstrap over the 139 eval episodes with the models FIXED --
it answers "another draw of EPISODES", NOT "another training run" and NOT "another inference run" (the inference floor is
reported beside it: |mean(os, seed 0) - mean(os, seed 1)|).

Bars (SPEC_REFCV8_DRAFT section 5), worded as the SPEC words them:
  BAR-R8-E1  S2: os - ha0_ext ADE 0-2 s < 0, SEPARATED, at BOTH seeds.             (inherits BAR-R7-1)
  BAR-R8-E3  S6: os - ha0_ext ADE 0-6 s < 0, SEPARATED, at BOTH seeds.             (inherits BAR-R7-3)
  BAR-R8-NR  per family, refcv8 - refcv7 is NOT SEPARATED WORSE (error metric: CI lower <= 0) at BOTH seeds, for:
             LONGITUDINAL speed MAE 0-2 s, |along-track| at 2 s; LATERAL heading MAE 0-2 s, curvature MAE 0-2 s,
             yaw-rate MAE (masked); ADE 0-2 s (S2) and ADE 0-6 s (S6).
  NOT PROVEN rule (E1/E3): met and separated but |delta| within 2x the inference floor => NOT PROVEN, never PASS.
STRATEGIC: NOT APPLICABLE with n = 0 and the reason when the run's argv carries --no-strategic (the layer is off).
Every output is stamped with its tier, its clock, and -- when G0 is not PASS -- PROVISIONAL.

CLOCKS: on the STANDALONE (uncorrected) pose clock the refcv8 GT and the model-free arms are bit-identical to refcv7's, so the
paired read is valid there.  On the CORRECTED clock the GT differs by the pose shift (<= 35 ms of motion), a paired
difference against refcv7 compares two different eval sets, and this tool reports E1 / E3 only (`--no-ref`).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

N_BOOT = 2000
SEED = 0
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"
QUESTION = ("episode-cluster bootstrap, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, "
            "NOT another inference run (the inference floor is reported beside it)")

#: (family, key, label, unit, lower_is_better, NR-headline on these surfaces)
METRICS = [
    ("ADE", "ade_m", "ADE (mean over the surface's instants)", "m", True, ("S2", "S6")),
    ("ADE", "fde_m", "FDE (final instant)", "m", True, ()),
    ("LONGITUDINAL", "LON_speed_mae_mps", "speed MAE", "m/s", True, ("S2",)),
    ("LONGITUDINAL", "LON_along_mae_m", "along-track MAE", "m", True, ()),
    ("LONGITUDINAL", "LON_along_abs_final_m", "|along-track| at the final instant", "m", True, ("S2",)),
    ("LONGITUDINAL", "LON_speed_within_0p5", "target-speed accuracy (+-0.5 m/s, fraction of steps)", "frac", False, ()),
    ("LONGITUDINAL", "LON_accel_mae_mps2", "accel MAE", "m/s2", True, ()),
    ("LATERAL", "LAT_heading_mae_deg", "heading MAE", "deg", True, ("S2",)),
    ("LATERAL", "LAT_curvature_mae_1pm_valid", "curvature MAE (masked)", "1/m", True, ("S2",)),
    ("LATERAL", "LAT_yaw_rate_mae_radps_valid", "yaw-rate MAE (masked)", "rad/s", True, ("S2",)),
    ("LATERAL", "LAT_cross_mae_m", "cross-track MAE", "m", True, ()),
    ("LATERAL", "LAT_cross_abs_final_m", "|cross-track| at the final instant", "m", True, ()),
    ("TACTICAL", "TAC_traj_lat_correct", "trajectory-derived lateral decision correct", "frac", False, ()),
    ("TACTICAL", "TAC_traj_lon_correct", "trajectory-derived longitudinal decision correct", "frac", False, ()),
]
HORIZON = {"S2": "0-2 s (instants 0.5-2.0 s, dt 0.5)", "S6": "0-6 s (instants 1-6 s, dt 1.0)"}
FINAL = {"S2": "2 s", "S6": "6 s"}


# ------------------------------------------------------------------------------------------ pure statistics
def paired_cell(new, ref, eid, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """new - ref, paired episode-cluster bootstrap on the windows finite on BOTH sides (the drop is counted)."""
    from taniteval import ci
    new, ref = np.asarray(new, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    if new.shape != ref.shape:
        raise ValueError(f"paired cell needs aligned arms: {new.shape} vs {ref.shape}")
    keep = np.isfinite(new) & np.isfinite(ref)
    if keep.sum() == 0:
        return {"status": "REFUSED", "reason": "no window is finite on both sides"}
    e = [x for x, k in zip(eid, keep) if k]
    r = ci.paired_episode_cluster_bootstrap(new[keep], ref[keep], e, n_boot=n_boot, seed=seed)
    r["n_dropped_nonfinite"] = int((~keep).sum())
    r["mean_new"] = float(new[keep].mean())
    r["mean_ref"] = float(ref[keep].mean())
    return r


def components(ra, P_arr, G_arr, dt: float) -> dict:
    """The battery's per-window components (`refav1_arm._components`) + the SPEC section 5 cells it lacks.
    Masked cells are per-window means over the steps where BOTH paths have a tangent (four_families' own `pair_valid`),
    NaN for a window with none -- the battery's A3 convention, extended to curvature."""
    import torch
    from taniteval import four_families as ff
    c = ra._components(P_arr, G_arr, dt)
    pt, gt = torch.as_tensor(P_arr).float(), torch.as_tensor(G_arr).float()
    Pg, Gg = ff._seq_geometry(pt, dt), ff._seq_geometry(gt, dt)
    m = Pg["pair_valid"] & Gg["pair_valid"]
    nv = m.sum(1)

    def masked(err):
        return torch.where(nv > 0, (err * m).sum(1) / nv.clamp_min(1), torch.full_like(err[:, 0], float("nan"))).numpy()

    c["LAT_yaw_rate_mae_radps_valid"] = masked((Pg["yaw_rate"] - Gg["yaw_rate"]).abs())
    c["LAT_curvature_mae_1pm_valid"] = masked((Pg["curvature"] - Gg["curvature"]).abs())
    c["LON_along_abs_final_m"] = (Pg["along"][:, -1] - Gg["along"][:, -1]).abs().numpy()
    c["LAT_cross_abs_final_m"] = (Pg["cross"][:, -1] - Gg["cross"][:, -1]).abs().numpy()
    c["LON_speed_within_0p5"] = ((Pg["speed"] - Gg["speed"]).abs() <= 0.5).float().mean(1).numpy()
    return c


# ------------------------------------------------------------------------------------------ the bars (pure)
def bar_E(cells_by_seed: dict, floor_m) -> dict:
    """BAR-R8-E1 / E3: os - echo < 0 AND separated (CI upper < 0) at EVERY inference seed; NOT PROVEN if within 2x floor."""
    per = {}
    for s, c in cells_by_seed.items():
        if c is None or "delta" not in c:
            per[s] = {"status": "ABSENT"}
            continue
        per[s] = {"delta": c["delta"], "lo": c["lo"], "hi": c["hi"], "n_windows": c.get("n_windows"),
                  "n_episodes": c.get("n_episodes"), "pass_separated": bool(c["delta"] < 0 and c["hi"] < 0),
                  "within_2x_inference_floor": (None if floor_m is None else bool(abs(c["delta"]) <= 2.0 * floor_m))}
    if len(per) < 2 or any(x.get("status") == "ABSENT" for x in per.values()):
        verdict = "NOT EVALUABLE (a cell is absent, or fewer than 2 inference seeds)"
    elif not all(x["pass_separated"] for x in per.values()):
        verdict = "FAILED"
    elif floor_m is None:
        verdict = "NOT PROVEN (no inference-seed floor)"
    elif any(x["within_2x_inference_floor"] for x in per.values()):
        verdict = "NOT PROVEN (|delta| within 2x the inference-seed floor)"
    else:
        verdict = "PASS (single training seed; training-seed floor unmeasured)"
    return {"per_inference_seed": per, "inference_seed_floor": floor_m, "verdict": verdict}


def bar_NR_metric(cells_by_seed: dict) -> dict:
    """One NR headline (an ERROR metric, refcv8 - refcv7): OK iff the CI lower bound is <= 0 at EVERY inference seed."""
    per = {}
    for s, c in cells_by_seed.items():
        if c is None or "delta" not in c:
            per[s] = {"status": "ABSENT"}
            continue
        per[s] = {"delta": c["delta"], "lo": c["lo"], "hi": c["hi"], "n_windows": c.get("n_windows"),
                  "separated_worse": bool(c["lo"] > 0), "point_worse": bool(c["delta"] > 0)}
    if len(per) < 2 or any(x.get("status") == "ABSENT" for x in per.values()):
        verdict = "NOT EVALUABLE"
    elif any(x["separated_worse"] for x in per.values()):
        verdict = "FAIL (separated worse)"
    else:
        verdict = "PASS (not separated worse at both seeds)"
    return {"per_inference_seed": per, "verdict": verdict}


def rollup_NR(per_metric: dict, family_of: dict) -> dict:
    fams: dict = {}
    for k, v in per_metric.items():
        fams.setdefault(family_of[k], []).append((k, v["verdict"]))
    out = {}
    for f, items in fams.items():
        if any(vd.startswith("FAIL") for _, vd in items):
            out[f] = "FAIL: " + ", ".join(k for k, vd in items if vd.startswith("FAIL"))
        elif any(vd.startswith("NOT EVALUABLE") for _, vd in items):
            out[f] = "NOT EVALUABLE"
        else:
            out[f] = "PASS"
    return out


# ------------------------------------------------------------------------------------------ panels
def load_panel_keyed(panel_dir: str, sha12):
    """(manifest, arrays, eid, keys): P.load_panel's layout + the (sha12(clip), ws) identity of every row."""
    man = json.load(open(os.path.join(panel_dir, "manifest.json"), encoding="utf-8"))
    cid_of = {int(e["file_index"]): str(e["clip_id"]) for e in man["episodes"]}
    files = sorted(glob.glob(os.path.join(panel_dir, "ep*.npz")))
    arrs: dict = {}
    eid, keys = [], []
    for fi, f in enumerate(files):
        z = np.load(f)
        file_index = int(os.path.basename(f)[2:5])
        n = z["g"].shape[0]
        eid += [fi] * n
        keys += [(sha12(cid_of[file_index]), int(w)) for w in z["ws"]]
        for k in z.files:
            if k in ("eid", "clip_index", "ws", "v0"):
                continue
            arrs.setdefault(k, []).append(z[k])
    return man, {k: np.concatenate(v) for k, v in arrs.items()}, np.asarray(eid), keys


def windows_check(keys: list, baseline_windows: dict, sha12) -> dict:
    """SET equality of the panel's (clip, ws) identities with a banked baseline's, counts printed."""
    want = {(sha12(c), int(w)) for c, v in baseline_windows.items() for w in v}
    have = set(keys)
    return {"equal": have == want, "n_panel": len(have), "n_baseline": len(want), "n_panel_rows": len(keys),
            "only_in_panel": len(have - want), "only_in_baseline": len(want - have),
            "duplicate_rows": len(keys) - len(have)}


def read_surface(ra, sha12, panel_dir: str, ref_arm, n_boot: int, baseline_windows: dict | None) -> dict:
    man, A, eid, keys = load_panel_keyed(panel_dir, sha12)
    dt = float(man["grid"]["dt_s"])
    out = {"panel_dir": panel_dir, "n_windows": int(len(eid)), "n_episodes": int(len(set(eid.tolist()))), "dt_s": dt,
           "instants_s": man["grid"].get("instants_s"), "arms": sorted(A)}
    if baseline_windows is not None:
        out["windows_check"] = windows_check(keys, baseline_windows, sha12)
    need = [a for a in ("os", "ha0_ext", ref_arm) if a and a in A]
    comps = {a: components(ra, A[a], A["g"], dt) for a in need}
    out["means"] = {a: {k: (float(np.nanmean(v)) if np.isfinite(v).any() else None) for k, v in c.items()}
                    for a, c in comps.items()}
    out["rows"] = {}
    for fam, key, label, unit, lower, _nr in METRICS:
        row = {"family": fam, "label": label, "unit": unit, "lower_is_better": lower}
        if "os" in comps and "ha0_ext" in comps:
            row["os_minus_ha0_ext"] = paired_cell(comps["os"][key], comps["ha0_ext"][key], eid, n_boot)
        if ref_arm and ref_arm in comps:
            row["os_minus_refcv7_50400"] = paired_cell(comps["os"][key], comps[ref_arm][key], eid, n_boot)
        out["rows"][key] = row
    return out


def controls(ra, sha12, panel_dir, ref_panel_dir, n_boot) -> dict:
    """VOID-gate controls vs the banked refcv7 panel: GT and the model-free arms must be BIT-IDENTICAL, paired delta 0.0 CI [0,0]."""
    man, A, eid, keys = load_panel_keyed(panel_dir, sha12)
    man2, B, eid2, keys2 = load_panel_keyed(ref_panel_dir, sha12)
    pos = {k: i for i, k in enumerate(keys2)}
    if not set(keys) <= set(keys2) or len(set(keys)) != len(keys):
        return {"status": "WINDOWS DIFFER", "n_a": len(keys), "n_b": len(keys2)}
    rows = [pos[k] for k in keys]
    dt = float(man["grid"]["dt_s"])
    full = set(keys) == set(keys2)
    out = {"status": "OK" if full else f"OK-SUBSET ({len(keys)} of {len(keys2)} baseline windows: a restricted run)", "arms": {}}
    for arm in ("g", "ha", "ha0", "ha0_ext", "stop"):
        if arm not in A or arm not in B:
            out["arms"][arm] = {"status": "ABSENT"}
            continue
        d = float(np.abs(A[arm].astype(np.float64) - B[arm][rows].astype(np.float64)).max())
        rec = {"max_abs_diff": d, "bit_identical": bool(d == 0.0)}
        if arm != "g":
            ca = components(ra, A[arm], A["g"], dt)["ade_m"]
            cb = components(ra, B[arm][rows], B["g"][rows], dt)["ade_m"]
            rec["paired_ade_delta"] = paired_cell(ca, cb, eid, n_boot)
        out["arms"][arm] = rec
    out["all_bit_identical"] = all(v.get("bit_identical") is True for v in out["arms"].values())
    return out


# ------------------------------------------------------------------------------------------ assembly
def _jload(path):
    if not path or not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def amended_g0(g: dict, wrapfix: dict | None, fp64: dict | None) -> dict:
    """The REGISTERED amended reading of a refcv8 milestone G0 (PI 2026-10-17; AMENDMENT_W_R8_1.md, AMENDMENT_W_R8_2.md).
    PASS only when ALL hold: as coded the G0 is FAIL with M1 detected and the wrapper clause is its ONLY gate reason; the
    W-R8-1 fixed probe (`wrapper_clause_r8fix_gpu.json`) is ok for THIS checkpoint (md5) and reads CLAUSE PASS -- or reads
    FAIL and the W-R8-2 float64 re-check (`wrapper_fp64/result.json`) of THIS step has every term <= 1e-9 with W1 and W2
    detected.  Anything missing or unreadable is NOT a pass (INCONCLUSIVE = FAIL)."""
    v = g.get("verdict") or {}
    reasons = v.get("reasons") or []
    m1 = ((v.get("mutation_detection") or {}).get("m1") or {}).get("detected")
    out = {"verdict": None, "amendments": [], "why": None}
    if m1 is not True:
        out["why"] = "M1 not detected (a G0 that cannot see a deliberate defect qualifies nothing)"
        return out
    if v.get("G0") == "PASS":
        out.update(verdict="PASS", amendments=[v.get("amendment")], why="PASS as coded")
        return out
    if not (len(reasons) == 1 and str(reasons[0]).startswith("wrapper clause FAIL")):
        out["why"] = f"{len(reasons)} gate reason(s), not the wrapper clause alone"
        return out
    if not wrapfix:
        out["why"] = "the W-R8-1 fixed wrapper probe has not been run for this checkpoint"
        return out
    if wrapfix.get("ok") is not True or wrapfix.get("ckpt_md5") != g.get("ckpt_md5"):
        out["why"] = "the W-R8-1 probe is not ok or is for another checkpoint (md5)"
        return out
    if wrapfix.get("CLAUSE") == "PASS":
        out.update(verdict="PASS", amendments=[v.get("amendment"), "W-R8-1"],
                   why=f"wrapper clause PASS under the fixed probe (max rel {wrapfix.get('max_wrapper_rel')})")
        return out
    crit = (fp64 or {}).get("criteria") or {}
    w12 = (fp64 or {}).get("F64_W1_W2_detected")
    if (fp64 and fp64.get("step") == g.get("step") and crit.get("F64_rel_le_1e-9") is True
            and fp64.get("F64_n_terms_wrapper_gt_1e-9") == 0 and isinstance(w12, list) and w12 and all(x is True for x in w12)):
        out.update(verdict="PASS", amendments=[v.get("amendment"), "W-R8-1", "W-R8-2"],
                   why=(f"fixed probe FAIL by fp32 noise (max rel {wrapfix.get('max_wrapper_rel')}); float64 re-check: every "
                        f"term <= 1e-9 (max {fp64.get('F64_max_wrapper_rel')}), W1/W2 detected"))
        return out
    out["why"] = ("fixed probe reads FAIL and no qualifying W-R8-2 float64 re-check for this step "
                  "(run raw/step20000/wrapper_fp64/wrapper_fp64_probe.py)")
    return out


def g0_stamp(g0_json: str | None, wrapfix_json: str | None = None, fp64_json: str | None = None) -> dict:
    if not g0_json or not os.path.exists(g0_json):
        return {"G0": "NOT AVAILABLE", "provisional": True, "why": "no g0.json"}
    g = json.load(open(g0_json, encoding="utf-8"))
    v = g.get("verdict") or {}
    m1 = ((v.get("mutation_detection") or {}).get("m1") or {}).get("detected")
    am = amended_g0(g, _jload(wrapfix_json), _jload(fp64_json))
    ok = am["verdict"] == "PASS"
    return {"G0": v.get("G0"), "amendment": v.get("amendment"), "M1_detected": m1, "ckpt_md5": g.get("ckpt_md5"),
            "G0_amended": am["verdict"] or "NOT PASS", "amendments": am["amendments"], "amended_why": am["why"],
            "wrapfix_json": wrapfix_json if _jload(wrapfix_json) else None, "fp64_json": fp64_json if _jload(fp64_json) else None,
            "provisional": not ok,
            "stamp": ("" if ok else
                      "PROVISIONAL: G0 is not PASS as coded nor under the registered amendments (" + str(am["why"]) + "): "
                      "per SPEC_REFCV8 section 5 no refcv8 battery number counts as bar evidence")}


def read_label(root: str, config_json: str | None) -> dict:
    """step / weights / total steps of the read, from the pass dir name (`step<N>[ema]_<unc|corr>`) and the run's argv."""
    import re
    m = re.match(r"step(\d+)(ema)?_(unc|corr)$", os.path.basename(os.path.normpath(root)))
    total = None
    if config_json and os.path.exists(config_json):
        argv = json.load(open(config_json, encoding="utf-8")).get("argv") or []
        if "--steps" in argv:
            total = int(argv[argv.index("--steps") + 1])
    total = total or 48852
    if not m:
        return {"step": None, "weights": None, "total": total, "pass": None}
    return {"step": int(m.group(1)), "weights": "EMA" if m.group(2) else "raw", "total": total, "pass": m.group(3)}


def read_stamp(lab: dict) -> str:
    if lab.get("step") is None:
        return "step unknown (pass dir not named step<N>[ema]_<pass>): read the root"
    if lab["step"] >= lab["total"]:
        return f"FINAL checkpoint (step {lab['step']:,} of {lab['total']:,}): the pre-registered bar verdict"
    return f"EARLY READ at step {lab['step']:,} of {lab['total']:,}: not the pre-registered final-checkpoint bar verdict"


def strategic_stamp(config_json: str | None) -> dict:
    off = None
    if config_json and os.path.exists(config_json):
        argv = json.load(open(config_json, encoding="utf-8")).get("argv") or []
        off = "--no-strategic" in argv
    if off:
        return {"status": "NOT APPLICABLE", "n": 0, "reason": "the strategic layer is OFF (--no-strategic in the run's argv; "
                "PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. "
                "Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic."}
    return {"status": "NOT EVALUATED HERE", "n": 0, "reason": "argv does not name --no-strategic (or no config): no strategic "
            "instrument is part of this read"}


def build(root: str, seeds, clock: str, ref: bool, g0_json, config_json, r7base: str, n_boot: int, sha12, ra,
          wrapfix_json=None, fp64_json=None) -> dict:
    lab = read_label(root, config_json)
    res = {"tool": "pair_read_r8.py", "tier": TIER, "estimator": QUESTION, "clock": clock, "n_boot": n_boot,
           "bootstrap_seed": SEED, "root": root, "label": lab, "g0": g0_stamp(g0_json, wrapfix_json, fp64_json),
           "strategic": strategic_stamp(config_json),
           "stamps": ["single training seed; training-seed floor at full scale unmeasured (SPEC_REFCV8 section 9)",
                      read_stamp(lab),
                      f"pose clock: {clock}",
                      "privileged input: the per-frame nav and the road-type speed label are ego-future derived "
                      "(optimistic by construction); LEGAL / NAVSHUF rows are NOT part of this read"],
           "surfaces": {}}
    sys.path.insert(0, str(HERE))
    import r8_battery_patch as BP
    for surf, pat in (("S2", "panel_s{s}"), ("S6", "s6_s{s}")):
        res["surfaces"][surf] = {"horizon": HORIZON[surf], "seeds": {}}
        for s in seeds:
            pd = os.path.join(root, pat.format(s=s))
            if not os.path.exists(os.path.join(pd, "manifest.json")):
                res["surfaces"][surf]["seeds"][str(s)] = {"status": "ABSENT", "panel_dir": pd}
                continue
            refarm = BP.R7_BASE_LABEL.format(s=s) if ref else None
            bw = BP.windows_of(os.path.join(r7base, ("dump_s%d" if surf == "S2" else "s6_s%d") % s)) if ref else None
            res["surfaces"][surf]["seeds"][str(s)] = read_surface(ra, sha12, pd, refarm, n_boot, bw)
    # inference floor per metric: |mean(os, s0) - mean(os, s1)|
    floors = {}
    for surf, sd in res["surfaces"].items():
        a, b = sd["seeds"].get("0", {}), sd["seeds"].get("1", {})
        if "means" in a and "means" in b:
            floors[surf] = {k: (abs(a["means"]["os"][k] - b["means"]["os"][k])
                                if a["means"]["os"].get(k) is not None and b["means"]["os"].get(k) is not None else None)
                            for k in a["means"]["os"]}
    res["inference_floor"] = floors
    cell = lambda surf, s, key, which: (res["surfaces"][surf]["seeds"].get(str(s), {}).get("rows", {}).get(key, {}).get(which))
    res["bars"] = {
        "BAR-R8-E1": {"surface": "S2", "metric": "ade_m", "statement": "os - ha0_ext ADE 0-2 s < 0, separated, at both seeds",
                      **bar_E({s: cell("S2", s, "ade_m", "os_minus_ha0_ext") for s in seeds},
                              (floors.get("S2") or {}).get("ade_m"))},
        "BAR-R8-E3": {"surface": "S6", "metric": "ade_m", "statement": "S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds",
                      **bar_E({s: cell("S6", s, "ade_m", "os_minus_ha0_ext") for s in seeds},
                              (floors.get("S6") or {}).get("ade_m"))}}
    fam_of = {k: f for f, k, *_ in METRICS}
    if ref:
        nr = {}
        for fam, key, label, unit, lower, surfs in METRICS:
            for surf in surfs:
                nr[f"{surf}:{key}"] = {"family": fam, "label": f"{label} [{HORIZON[surf]}]",
                                       **bar_NR_metric({s: cell(surf, s, key, "os_minus_refcv7_50400") for s in seeds})}
        res["bars"]["BAR-R8-NR"] = {"statement": "refcv8 - refcv7-50,400 is not separated worse (error metric: CI lower <= 0) "
                                    "at both seeds, per headline", "metrics": nr,
                                    "per_family": rollup_NR(nr, {k: v["family"] for k, v in nr.items()})}
        if "S2" in res["surfaces"] and "0" in res["surfaces"]["S2"]["seeds"]:
            res["controls"] = {}
            for s in seeds:
                pd = os.path.join(root, "panel_s%d" % s)
                if os.path.exists(os.path.join(pd, "manifest.json")):
                    res["controls"]["S2_s%d" % s] = controls(ra, sha12, pd, os.path.join(r7base, "panel_s%d" % s), n_boot)
    else:
        res["bars"]["BAR-R8-NR"] = {"verdict": "NOT PAIRABLE on this clock (refcv7's GT is on the uncorrected clock; the paired "
                                    "difference would compare two eval sets)", "metrics": {}}
    return res


def render_md(res: dict) -> str:
    g0 = res["g0"]
    lab = res.get("label") or {}
    what = (f"step-{lab['step']:,} {lab['weights']} weights" if lab.get("step") is not None else "unlabelled pass")
    L = [f"# refcv8 {what}: paired four-family read (refcv8 `os` vs refcv7-50,400 `os`)", "",
         f"* tier: **{res['tier']}**; clock: **{res['clock']}**; estimator: {res['estimator']}; B {res['n_boot']}, seed {res['bootstrap_seed']}.",
         f"* G0 as coded: **{g0.get('G0')}** (amendment {g0.get('amendment')}), M1 detected: {g0.get('M1_detected')}; "
         f"under the registered amendments: **{g0.get('G0_amended', 'n/a')}** ({' + '.join(a for a in (g0.get('amendments') or []) if a) or '-'}"
         f"{'; ' + g0['amended_why'] if g0.get('amended_why') else ''}). "
         + (f"**{g0.get('stamp') or g0.get('why') or 'PROVISIONAL'}**" if g0.get("provisional") else "G0 qualifies the instrument."), ""]
    for st in res["stamps"]:
        L.append(f"* {st}")
    L += ["", "## Bars (SPEC_REFCV8_DRAFT section 5)", "", "| bar | verdict |", "|---|---|"]
    for b in ("BAR-R8-E1", "BAR-R8-E3"):
        x = res["bars"][b]
        L.append(f"| {b} ({x['statement']}) | {x['verdict']} |")
        for s, c in sorted(x["per_inference_seed"].items()):
            if "delta" in c:
                L.append(f"| &nbsp;&nbsp;seed {s} | delta {c['delta']} [{c['lo']}, {c['hi']}] n {c['n_windows']}/{c['n_episodes']} ep; "
                         f"floor {x['inference_seed_floor']} |")
    nr = res["bars"]["BAR-R8-NR"]
    L.append(f"| BAR-R8-NR | {nr.get('verdict') or json.dumps(nr.get('per_family'))} |")
    L += ["", "## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)", "",
          "| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |", "|---|---|---|---|---|---|---|---|---|"]
    for surf, sd in res["surfaces"].items():
        for s, d in sorted(sd["seeds"].items()):
            if "rows" not in d:
                L.append(f"| - | {surf} | {s} | ABSENT | | | | | |")
                continue
            for key, row in d["rows"].items():
                c = row.get("os_minus_refcv7_50400") or row.get("os_minus_ha0_ext")
                tag = "" if row.get("os_minus_refcv7_50400") else " (vs echo)"
                if not c or "delta" not in c:
                    continue
                L.append(f"| {row['family']} | {surf} {HORIZON[surf].split(' (')[0]} | {s} | {row['label']} [{row['unit']}]{tag} | {c['mean_new']:.5g} | {c['mean_ref']:.5g} | "
                         f"{c['delta']} [{c['lo']}, {c['hi']}] | {c['separated']} | {c['n_windows']}/{c['n_episodes']} |")
    L += ["", f"STRATEGIC: **{res['strategic']['status']}**, n = {res['strategic']['n']}: {res['strategic']['reason']}", ""]
    if "controls" in res:
        L += ["## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)", ""]
        for k, c in res["controls"].items():
            L.append(f"* {k}: all bit-identical = {c.get('all_bit_identical')}; status {c.get('status')}")
    L += ["", "Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 "
          "tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="battery work dir of the pass (holds panel_s*/ s6_s*/)")
    ap.add_argument("--clock", choices=("standalone", "corrected"), required=True)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--g0-json", default=None)
    ap.add_argument("--config", default="E:/refcv8_eval_kit/ckpt/config.json")
    ap.add_argument("--r7base", default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--wrapfix-json", default=None, help="W-R8-1 fixed wrapper probe of THIS checkpoint (absent = not run)")
    ap.add_argument("--fp64-json", default=None, help="W-R8-2 float64 re-check result.json of THIS step (absent = not run)")
    ap.add_argument("--restamp", default=None, help="re-stamp a banked PAIRED_READ json: label, read stamp and G0 only; "
                    "every number is copied unchanged (no torch, no bootstrap)")
    a = ap.parse_args()
    if a.restamp:
        res = json.load(open(a.restamp, encoding="utf-8"))
        root = res.get("root") or a.root
        lab = read_label(root, a.config)
        res["label"] = lab
        res["stamps"] = [read_stamp(lab) if str(x).startswith(("EARLY READ", "FINAL checkpoint", "step unknown")) else x
                         for x in res["stamps"]]
        if a.g0_json:
            res["g0"] = g0_stamp(a.g0_json, a.wrapfix_json, a.fp64_json)
        res["restamped"] = {"at": time.strftime("%FT%T"), "by": "pair_read_r8.py --restamp",
                            "what": "label, read stamp and G0 stamp only; every number unchanged"}
        json.dump(res, open(a.out_json, "w", encoding="utf-8"), indent=1, default=str)
        if a.out_md:
            open(a.out_md, "w", encoding="utf-8").write(render_md(res))
        print(json.dumps({"restamped": a.out_json, "label": lab, "G0_amended": res["g0"].get("G0_amended"),
                          "provisional": res["g0"].get("provisional")}))
        return
    import r8_battery_patch as BP
    L, P = BP.apply()
    ra = P.RR.ra3().ra
    res = build(a.root, [int(x) for x in a.seeds.split(",")], a.clock, a.clock == "standalone", a.g0_json, a.config,
                a.r7base or BP.R7_BASE_ROOT, a.n_boot, L.sha12, ra, a.wrapfix_json, a.fp64_json)
    res["finished"] = time.strftime("%FT%T")
    json.dump(res, open(a.out_json, "w", encoding="utf-8"), indent=1, default=str)
    if a.out_md:
        open(a.out_md, "w", encoding="utf-8").write(render_md(res))
    print(json.dumps({"bars": {k: v.get("verdict") or v.get("per_family") for k, v in res["bars"].items()}}, indent=1))


if __name__ == "__main__":
    main()
