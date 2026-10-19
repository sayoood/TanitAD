"""kdraw_e1_r8.py -- does a K-draw inference ensemble of refcv8 pass BAR-R8-E1 on the ceiling-COMPLIANT S2 set?  (PREREG_KDRAW_E1.md)
CPU only, banked dumps only.  NEW file; nothing existing is edited.

ONE COMMAND (works with whatever seeds are rolled; the sets that are incomplete are reported PENDING, never guessed):

    sh <PKG>/code/run_kdraw_analysis_r8.sh [step30000ema_unc]
  == CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' \
       python -X utf8 kdraw_e1_r8.py run --tree step30000ema_unc          # -> <PKG>/raw/kdraw/KDRAW_E1.json (+ .md)

Arms (per window, on the 4,754-window S2 grid, scored on the compliant mask): SINGLE seed s; ENS-K = mean of the K draws' `os` plans; MEDOID-K = the
single draw whose mean L2 (over the 4 S2 waypoints) to the K-mean is smallest.  K=4 on the DISJOINT sets A={0,1,2,3}, B={4,5,6,7}; K=8 = all; ENS-2={0,1}
is a control.  Cell = paired episode-cluster bootstrap (B 2000, seed 0, 95 % percentile, `pair_read_r8.paired_cell`) of arm minus the echo `ha0_ext`
on `components()['ade_m']` -- the SAME function the banked read used, computed on all windows then subset by the mask (as `read_masked` does).
D-R8-E1-REF (PI 2026-10-19): BAR-R8-E1 is now scored against `ha0_ext_pose` (refcv8's own causal prior).  Every cell and verdict above is the PRE-REGISTERED read
against `ha0_ext` and is UNCHANGED; the same arms scored against the pose echo are added BESIDE them under `pose_reference` (E1_pose cells, `bars_pose`, own controls:
decisions + the echo rebuilt, the 2026-09-26 model-free known value, the A0K0 single-draw literals).  The block can only add a key: a failure inside it is recorded, never raised.
TIER: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  ESTIMATOR: answers "another draw of EPISODES" only; the disjoint
seed sets A/B are the inference-run replicate; one training seed.  Verdict rules are the ones committed in PREREG_KDRAW_E1.md section 3 and are coded below.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PKG = HERE.parent
OUT = PKG / "raw" / "kdraw"
ARCHIVE = "C:/Users/Admin/refcv8_eval_archive/battery"
KIT = "E:/refcv8_eval_kit"
MASK_SHA = "4ebc6b8604f064f69e9e94c15e099062f68339a8ef38d666e8a29d29cb1c4129"      # literal (compliant mask of the banked windows)
N_BOOT, BOOT_SEED = 2000, 0
SET_A, SET_B, SET_ALL, SET_2 = (0, 1, 2, 3), (4, 5, 6, 7), tuple(range(8)), (0, 1)
SPREAD_M = 1.0
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"
QUESTION = ("paired episode-cluster bootstrap, models FIXED, B 2000, seed 0: answers 'another draw of EPISODES' only; the disjoint seed sets A/B are the "
            "inference-run replicate; NOT another training run (single training seed)")
#: literals (MEASURED, raw/e1_decomp/E1_DECOMP.json); written as literals, never derived from the code under test
LIT_K1 = {0: (-0.0085, -0.0229, 0.0065), 1: (-0.0076, -0.0216, 0.0073)}
LIT_K2 = (-0.0138, -0.0279, 0.0012)
TEST_MUTATION = os.environ.get("KDRAW_TEST_MUTATION", "")          # "", "ens_first", "medoid_is_mean": recorded in the output; tests only
EXPECT = {"n_windows": 4159, "n_episodes": 138, "n_dump": 4754}
#: literals (MEASURED, raw/a0k0_headroom/A0K0_HEADROOM.json, uncapped os - ha0_ext_pose, S2, compliant set): the D-R8-E1-REF known values of the single draws
LIT_POSE_K1 = {0: (-0.0136, -0.0286, 0.0018), 1: (-0.0127, -0.0272, 0.0024)}
#: literal (MEASURED 2026-09-26, measure_prior_vs_echo.json B_ade_0_2s): ha0_ext_pose - echo, S2, all 4,754 windows
LIT_R0926 = (0.0055, 0.0036, 0.0076)


# ============================================================================================================== pure functions (unit-tested)
def assert_disjoint(a, b) -> None:
    if set(a) & set(b):
        raise SystemExit(f"[kdraw_e1] seed sets overlap: {sorted(set(a) & set(b))}")


def ensemble_mean(plans: np.ndarray) -> np.ndarray:
    """plans [K, N, H, 2] -> [N, H, 2]; float64 mean cast to float32 (K=1 returns the draw bit-for-bit)."""
    return np.mean(np.asarray(plans, dtype=np.float64), axis=0).astype(np.float32)


def medoid_index(plans: np.ndarray) -> np.ndarray:
    """[N] index of the draw with the smallest mean-over-waypoints L2 distance to the K-mean plan (ties -> lowest index)."""
    p = np.asarray(plans, dtype=np.float64)
    mu = p.mean(axis=0)
    d = np.linalg.norm(p - mu[None], axis=-1).mean(axis=-1)           # [K, N]
    return np.argmin(d, axis=0)


def medoid_plan(plans: np.ndarray) -> np.ndarray:
    idx = medoid_index(plans)
    return np.asarray(plans)[idx, np.arange(plans.shape[1])]


def spread_2s(plans: np.ndarray) -> np.ndarray:
    """[N] largest pairwise L2 distance between the K draws' 2-s endpoints (last waypoint)."""
    e = np.asarray(plans, dtype=np.float64)[:, :, -1, :]
    k = e.shape[0]
    s = np.zeros(e.shape[1])
    for i in range(k):
        for j in range(i + 1, k):
            s = np.maximum(s, np.linalg.norm(e[i] - e[j], axis=-1))
    return s


def arm_verdict(cell_a: dict | None, cell_b: dict | None, floor: float | None) -> dict:
    """PREREG section 3.  PASS: delta<0 AND hi<0 AND |delta| > 2*floor on BOTH sets.  PARTIAL: not PASS, delta<0 on both.  FAIL: delta>=0 on >=1 set."""
    if cell_a is None or cell_b is None or floor is None or "delta" not in cell_a or "delta" not in cell_b:
        return {"verdict": "PENDING", "why": "a seed set is not fully rolled"}
    cs = (cell_a, cell_b)
    sep = [bool(c["delta"] < 0 and c["hi"] < 0) for c in cs]
    neg = [bool(c["delta"] < 0) for c in cs]
    within = [bool(abs(c["delta"]) <= 2.0 * floor) for c in cs]
    base = {"separated_by_set": sep, "delta_negative_by_set": neg, "within_2x_floor_by_set": within, "floor_arm_m": floor}
    if all(sep) and not any(within):
        return {"verdict": "PASS", **base}
    if all(neg):
        return {"verdict": "PARTIAL", "n_sets_separated": int(sum(sep)), "only_miss_is_not_proven_rule": bool(all(sep) and any(within)), **base}
    return {"verdict": "FAIL", **base}


def overall_verdict(ens: dict, med: dict, blend_flag: bool) -> dict:
    vs = {"ENS-4": ens["verdict"], "MEDOID-4": med["verdict"]}
    if "PASS" in vs.values():
        o = "PASS"
        if ens["verdict"] == "PASS" and med["verdict"] != "PASS" and blend_flag:
            o = "PASS WITH BLEND FLAG"
    elif "PENDING" in vs.values():
        o = "PENDING"
    elif "PARTIAL" in vs.values():
        o = "PARTIAL"
    else:
        o = "FAIL"
    return {"overall": o, "per_arm": vs}


def blend_flag(cell_ens_minus_med_curv: dict | None) -> bool:
    """PREREG section 5: raised iff, on the disagreement subset, ENS curvature MAE minus MEDOID curvature MAE has CI lower bound > 0."""
    return bool(cell_ens_minus_med_curv and "lo" in cell_ens_minus_med_curv and cell_ens_minus_med_curv["lo"] > 0)


def fc(c: dict | None, dp: int = 4) -> str:
    if not c or "delta" not in c:
        return "n/a"
    return f"{c['delta']:+.{dp}f} [{c['lo']:+.{dp}f}, {c['hi']:+.{dp}f}]"


def lit_equal(c: dict, lit: tuple, tol: float = 5e-5 + 1e-7) -> bool:
    return bool(abs(c["delta"] - lit[0]) <= tol and abs(c["lo"] - lit[1]) <= tol and abs(c["hi"] - lit[2]) <= tol)


# ============================================================================================================== real-data loading
def load_dump(dump_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(dump_dir, "ep*.npz")))
    if not files:
        raise FileNotFoundError(dump_dir)
    cols = {k: [] for k in ("os", "g", "ha0_ext", "ws", "clip_index", "v0")}
    eid = []
    for fi, f in enumerate(files):
        z = np.load(f)
        n = len(z["ws"])
        for k in cols:
            cols[k].append(np.repeat(z[k], n) if k == "clip_index" else z[k])
        eid += [fi] * n
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["eid"] = np.asarray(eid)
    out["n_files"] = len(files)
    return out


def load_panel_arms(panel_dir: str, names: tuple) -> dict:
    files = sorted(glob.glob(os.path.join(panel_dir, "ep*.npz")))
    arr = {k: [] for k in names + ("ws", "os", "ha0_ext", "g")}
    for f in files:
        z = np.load(f)
        for k in arr:
            arr[k].append(z[k])
    return {k: np.concatenate(v) for k, v in arr.items()}


def seed_dump_dir(tree: str, s: int, kd_root: str) -> str:
    return f"{ARCHIVE}/{tree}/dump_s{s}" if s in (0, 1) else f"{kd_root}/dump_s{s}"


def seed_available(tree: str, s: int, kd_root: str) -> tuple[bool, str]:
    d = seed_dump_dir(tree, s, kd_root)
    if s in (0, 1):
        return os.path.exists(os.path.join(d, "manifest.json")), d
    vj = os.path.join(d, "KDRAW_VERIFY.json")
    if not os.path.exists(vj):
        return False, d
    try:
        return bool(json.load(open(vj, encoding="utf-8")).get("ok")), d
    except Exception:                                     # noqa: BLE001
        return False, d


# ============================================================================================================== the analysis
def pose_beside(NR, PR, tree, ref, ECHO, G, dt, eid, mask, comps_of, cell, singles, single_c, arms, plans_of) -> dict:
    """Scores the SAME single draws and arms against `ha0_ext_pose` (D-R8-E1-REF).  The pose echo is built by a0k0_headroom_r8.load_decisions + references (reused,
    no second derivation) from the panel's decisions.  Adds `E1_pose` to singles / arms IN PLACE (a new key; no existing key is touched) and returns the block."""
    import a0k0_headroom_r8 as AK
    pdir = f"{ARCHIVE}/{tree}/panel_s0"
    hz = [int(h) for h in json.load(open(f"{pdir}/manifest.json", encoding="utf-8"))["grid"]["horizons_steps"]]
    dec = AK.load_decisions(NR, pdir, ref["ws"], ref["v0"])
    REFS, V = AK.references(dec, ECHO, hz)
    POSE = REFS["HP"]
    pose_c = comps_of(POSE)
    ctl = {"decisions_check": dec["chk"], "V1_ha0_ext_rebuilt_vs_panel_max_abs_m": V["V1_ha0_ext_rebuilt_vs_panel_max_abs_m"]}
    r09 = PR.paired_cell(pose_c["ade_m"], comps_of(ECHO)["ade_m"], eid, N_BOOT, BOOT_SEED)          # model-free, mask-free: all 4,754 S2 windows
    ctl["R0926_pose_minus_echo_all_windows"] = {"got": fc(r09), "lit": LIT_R0926, "ok": lit_equal(r09, LIT_R0926), "n_windows": r09.get("n_windows")}
    for s in sorted(singles):
        singles[s]["E1_pose"] = cell(single_c[s]["ade_m"], pose_c["ade_m"])
    for s in (0, 1):
        if s in singles:
            ctl[f"K1_pose_seed{s}"] = {"got": fc(singles[s]["E1_pose"]), "lit": LIT_POSE_K1[s], "ok": lit_equal(singles[s]["E1_pose"], LIT_POSE_K1[s])}
    for name, a in arms.items():
        if "E1" in a and name in plans_of:
            a["E1_pose"] = cell(comps_of(plans_of[name])["ade_m"], pose_c["ade_m"])
    ctl["PASS"] = bool(dec["chk"]["PASS"] and V["V1_ha0_ext_rebuilt_vs_panel_max_abs_m"] <= AK.V1_TOL_M and ctl["R0926_pose_minus_echo_all_windows"]["ok"]
                       and all(v["ok"] for k, v in ctl.items() if k.startswith("K1_pose")))
    return {"reference": "ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)", "beside": "the pre-registered cells above are vs ha0_ext and are unchanged", "arms": True,
            "echo_pose_mean_ade_compliant": float(pose_c["ade_m"][mask].mean()), "controls": ctl}


def analyse(tree: str, kd_root: str, out_dir: Path) -> dict:
    t0 = time.time()
    import clamp_posthoc_r8 as CP
    NR, BP, PR, ra, sha12 = CP.get_readers()          # FIRST: r8_repoint must configure the loader environment before anything imports the loader
    import compliant_reread_r8 as CR
    rc = CP.run_cfg()
    res = {"tool": "kdraw_e1_r8.py", "generated": time.strftime("%FT%T%z"), "tier": TIER, "estimator": QUESTION, "n_boot": N_BOOT,
           "boot_seed": BOOT_SEED, "tree": tree, "kd_root": kd_root, "prereg": "raw/kdraw/PREREG_KDRAW_E1.md",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "status": "EXPLORATORY read of a banked milestone (EMA step in the tree name); the registered read is the FINAL checkpoint (48,852)",
           "bar_of_record": "single-draw `os` (BAR-R8-E1); a K-draw arm is a NEW arm with inference cost up to xK",
           "test_mutation": TEST_MUTATION or None}
    avail, dumps = {}, {}
    for s in range(8):
        ok, d = seed_available(tree, s, kd_root)
        avail[s] = {"available": ok, "dump": d}
        if ok:
            dumps[s] = load_dump(d)
    res["seeds"] = avail
    need = {"A": set(SET_A) <= set(dumps), "B": set(SET_B) <= set(dumps)}
    if 0 not in dumps or 1 not in dumps:
        raise SystemExit("[kdraw_e1] banked seeds 0 / 1 missing: nothing to anchor the controls on")
    ref = dumps[0]
    # ---- window identity / mask --------------------------------------------------------------------------------------------------------------
    raw, lim, linfo, dump0 = CP.load_seed_inputs(tree, 0, rc)
    comp = CR.compliant_mask(raw["gt_max_speed_6s"], lim)
    clip_x, ws_x = raw["clip_index"].astype(np.int64), np.load(dump0)["ws"].astype(np.int64)
    order_ok = bool(np.array_equal(clip_x, ref["clip_index"].astype(np.int64)) and np.array_equal(ws_x, ref["ws"].astype(np.int64)))
    sha = CR.mask_sha(clip_x, ws_x, comp)
    mask = comp.copy()
    eid = ref["eid"]
    ctl = {"C-MASK": {"sha": sha, "sha_equal_expected": sha == MASK_SHA, "n_compliant": int(mask.sum()), "n_dump": int(len(mask)),
                      "n_episodes": int(len(np.unique(eid[mask]))), "dump_row_order_equals_extras_order": order_ok}}
    ctl["C-MASK"]["PASS"] = bool(ctl["C-MASK"]["sha_equal_expected"] and ctl["C-MASK"]["n_compliant"] == EXPECT["n_windows"] and
                                 ctl["C-MASK"]["n_dump"] == EXPECT["n_dump"] and ctl["C-MASK"]["n_episodes"] == EXPECT["n_episodes"] and order_ok)
    # panel baselines (refcv7-50,400 per inference seed) + the equality of the dump-derived arrays with the banked panel
    pan = load_panel_arms(f"{ARCHIVE}/{tree}/panel_s0", ("b_refcv7_50400_s0", "b_refcv7_50400_s1"))
    ctl["C-PANEL"] = {"ws_equal": bool(np.array_equal(pan["ws"], ref["ws"])), "g_equal": bool(np.array_equal(pan["g"], ref["g"])),
                      "echo_equal": bool(np.array_equal(pan["ha0_ext"], ref["ha0_ext"])), "os_s0_equal": bool(np.array_equal(pan["os"], dumps[0]["os"]))}
    ctl["C-PANEL"]["PASS"] = bool(all(ctl["C-PANEL"].values()))
    dt = float(json.load(open(f"{dump0.rsplit('/', 1)[0]}/manifest.json", encoding="utf-8"))["grid"]["dt_s"])
    G, ECHO = ref["g"], ref["ha0_ext"]
    # C-ID / C-DIST
    bad_id = {}
    for s, d in dumps.items():
        bad = [k for k in ("ws", "eid", "clip_index", "g", "ha0_ext", "v0") if not np.array_equal(d[k], ref[k])]
        if bad:
            bad_id[s] = bad
    ctl["C-ID"] = {"seeds_checked": sorted(dumps), "mismatches": bad_id, "PASS": not bad_id}
    pair_l2 = {}
    for i in sorted(dumps):
        for j in sorted(dumps):
            if i < j:
                pair_l2[f"{i}-{j}"] = float(np.linalg.norm(dumps[i]["os"].astype(np.float64) - dumps[j]["os"].astype(np.float64), axis=-1).mean())
    ctl["C-DIST"] = {"mean_pair_plan_l2_m": pair_l2, "min": (min(pair_l2.values()) if pair_l2 else None), "PASS": bool(pair_l2 and min(pair_l2.values()) > 1e-4)}
    # C-CAP (capped == uncapped on the compliant windows, every available seed)
    try:
        V, _vrec = CP.load_vendored()                      # NOT CR.tools_load(): that re-runs get_readers() and r8_repoint refuses a second apply
        capres = {}
        for s in sorted(dumps):
            z = np.load(f"{avail[s]['dump']}/refcv6_extras.npz")
            lim_s, _ = CP.fed_ceiling(z["v_max_ms"], z["v_max_valid"], rc)
            plan = z["nav_true.traj"]
            capped, _diag = CP.apply_clamp(V, plan, lim_s, z["v0"].astype(np.float32), rc)
            ch = np.abs(capped.numpy().astype(np.float64) - plan.astype(np.float64)).max(axis=(1, 2)) > 0
            capres[s] = {"n_changed_all_windows": int(ch.sum()), "n_changed_on_compliant": int((ch & mask).sum())}
        ctl["C-CAP"] = {"by_seed": capres, "PASS": bool(all(v["n_changed_on_compliant"] == 0 for v in capres.values()))}
    except (Exception, SystemExit) as exc:                  # noqa: BLE001 -- printed UNVERIFIED, not hidden
        ctl["C-CAP"] = {"PASS": None, "UNVERIFIED": f"{type(exc).__name__}: {str(exc)[:200]}"}
    # C-DET (roller determinism, written by kdraw_roll_r8.py control)
    cj = Path(kd_root) / "KDRAW_CONTROL.json"
    if cj.exists():
        c = json.load(open(cj, encoding="utf-8"))
        ctl["C-DET"] = {"file": str(cj), "PASS": (c.get("verdict") or {}).get("C1_bit_identical"), "verdict": c.get("verdict"), "prefix": {k: c["prefix"][k] for k in ("n_clips", "n_windows")},
                        "cases": {k: {"mode": v.get("mode"), "os_bit_identical_all": (v.get("compare") or {}).get("os_bit_identical_all"),
                                      "os": ((v.get("compare") or {}).get("arms") or {}).get("os")} for k, v in c["cases"].items()}}
    else:
        ctl["C-DET"] = {"PASS": None, "UNVERIFIED": "KDRAW_CONTROL.json absent (the roller control has not run)"}
    # ---- components ------------------------------------------------------------------------------------------------------------------------
    def comps_of(plan):
        return PR.components(ra, np.ascontiguousarray(plan, dtype=np.float32), G, dt)
    echo_c = comps_of(ECHO)
    r7 = {0: np.asarray(pan["b_refcv7_50400_s0"], dtype=np.float32), 1: np.asarray(pan["b_refcv7_50400_s1"], dtype=np.float32)}
    r7["ens2"] = ensemble_mean(np.stack([r7[0], r7[1]]))
    r7_c = {k: comps_of(v) for k, v in r7.items()}
    em, eg = eid[mask], eid[mask]

    def cell(arm_v, ref_v):
        return PR.paired_cell(np.asarray(arm_v)[mask], np.asarray(ref_v)[mask], eg, N_BOOT, BOOT_SEED)

    singles, single_c = {}, {}
    for s in sorted(dumps):
        single_c[s] = comps_of(dumps[s]["os"])
        singles[s] = {"E1": cell(single_c[s]["ade_m"], echo_c["ade_m"]), "mean_ade_compliant": float(single_c[s]["ade_m"][mask].mean())}
    # C-K1
    k1 = {s: lit_equal(singles[s]["E1"], LIT_K1[s]) for s in (0, 1)}
    ctl["C-K1"] = {"seed0": {"got": fc(singles[0]["E1"]), "lit": LIT_K1[0], "ok": k1[0]}, "seed1": {"got": fc(singles[1]["E1"]), "lit": LIT_K1[1], "ok": k1[1]}, "PASS": all(k1.values())}
    # ---- arms -------------------------------------------------------------------------------------------------------------------------------
    arms, plans_of = {}, {}

    def add_arm(name: str, seeds: tuple, kind: str, full_nr: bool):
        if not set(seeds) <= set(dumps):
            arms[name] = {"seeds": list(seeds), "status": "PENDING (seeds not rolled: " + ",".join(str(x) for x in sorted(set(seeds) - set(dumps))) + ")"}
            return
        st = np.stack([dumps[s]["os"] for s in seeds])
        plan = ensemble_mean(st) if kind == "ENS" else medoid_plan(st).astype(np.float32)
        if TEST_MUTATION == "ens_first" and kind == "ENS":                    # deliberate-regression hooks, exercised ONLY by test_kdraw_e1_r8.py
            plan = st[0].astype(np.float32)
        if TEST_MUTATION == "medoid_is_mean" and kind == "MEDOID":
            plan = ensemble_mean(st)
        c = comps_of(plan)
        plans_of[name] = plan
        a = {"seeds": list(seeds), "kind": kind, "K": len(seeds), "E1": cell(c["ade_m"], echo_c["ade_m"]), "mean_ade_compliant": float(c["ade_m"][mask].mean())}
        if kind == "ENS":                                                                  # C-TRI: the triangle inequality, every window
            per = np.stack([single_c[s]["ade_m"] for s in seeds]).mean(0)
            a["triangle_ok_every_window"] = bool((c["ade_m"] <= per + 1e-6).all())
        if full_nr:
            fam_of = {key: fam for fam, key, _l, _u, _lo, _nr in PR.METRICS}
            rows, per_metric = {}, {}
            for fam, key, label, unit, lower, nr_surf in PR.METRICS:
                row = {"family": fam, "label": label, "unit": unit, "lower_is_better": lower,
                       "vs_echo": cell(c[key], echo_c[key]), "vs_r7_s0": cell(c[key], r7_c[0][key]), "vs_r7_s1": cell(c[key], r7_c[1][key]),
                       "vs_r7_ens2": cell(c[key], r7_c["ens2"][key])}
                rows[key] = row
                if "S2" in nr_surf:
                    per_metric[key] = PR.bar_NR_metric({"0": row["vs_r7_s0"], "1": row["vs_r7_s1"]})
            a["NR_rows"] = rows
            a["NR_headline_verdicts"] = {k: v["verdict"] for k, v in per_metric.items()}
            a["NR_family_rollup"] = PR.rollup_NR(per_metric, fam_of)
        arms[name] = a

    add_arm("ENS-2", SET_2, "ENS", False)
    for nm, ss in (("4A", SET_A), ("4B", SET_B), ("8", SET_ALL)):
        add_arm(f"ENS-{nm}", ss, "ENS", True)
        add_arm(f"MEDOID-{nm}", ss, "MEDOID", True)
    if "E1" in arms["ENS-2"]:
        ctl["C-K2"] = {"got": fc(arms["ENS-2"]["E1"]), "lit": LIT_K2, "PASS": lit_equal(arms["ENS-2"]["E1"], LIT_K2)}
    ctl["C-TRI"] = {"PASS": bool(all(a.get("triangle_ok_every_window", True) for a in arms.values() if isinstance(a, dict)))}
    # ---- D-R8-E1-REF: the same arms against ha0_ext_pose, BESIDE the pre-registered read (additive; a failure here is recorded, never raised) -------
    try:
        pose_ref = pose_beside(NR, PR, tree, ref, ECHO, G, dt, eid, mask, comps_of, cell, singles, single_c, arms, plans_of)
    except (Exception, SystemExit) as exc:                  # noqa: BLE001 -- printed UNVERIFIED, not hidden
        pose_ref = {"status": f"UNAVAILABLE: {type(exc).__name__}: {str(exc)[:200]}"}
    # ---- verdicts ---------------------------------------------------------------------------------------------------------------------------
    def floor_of(n):
        if "E1" not in arms.get(f"{n}-4A", {}) or "E1" not in arms.get(f"{n}-4B", {}):
            return None
        return abs(arms[f"{n}-4A"]["mean_ade_compliant"] - arms[f"{n}-4B"]["mean_ade_compliant"])
    bars = {}
    for n in ("ENS", "MEDOID"):
        f = floor_of(n)
        bars[f"{n}-4"] = arm_verdict(arms[f"{n}-4A"].get("E1"), arms[f"{n}-4B"].get("E1"), f)
        bars[f"{n}-4"]["cells"] = {"A": fc(arms[f"{n}-4A"].get("E1")), "B": fc(arms[f"{n}-4B"].get("E1"))}
    # ---- mode disagreement ------------------------------------------------------------------------------------------------------------------
    md = {}
    flag_any = False
    for nm, ss in (("4A", SET_A), ("4B", SET_B), ("8", SET_ALL)):
        if not set(ss) <= set(dumps):
            md[nm] = {"status": "PENDING"}
            continue
        st = np.stack([dumps[s]["os"] for s in ss])
        sp = spread_2s(st)
        sub = mask & (sp > SPREAD_M)
        rec = {"seeds": list(ss), "frac_spread_gt_1m_compliant": float((sp[mask] > SPREAD_M).mean()), "frac_spread_gt_1m_all_windows": float((sp > SPREAD_M).mean()),
               "n_disagree_compliant": int(sub.sum()), "spread_p50_p90_p99_compliant_m": [float(x) for x in np.percentile(sp[mask], [50, 90, 99])]}
        ec, mc = comps_of(plans_of[f"ENS-{nm}"]), comps_of(plans_of[f"MEDOID-{nm}"])
        for key in ("LAT_curvature_mae_1pm_valid", "LAT_heading_mae_deg", "ade_m"):
            single_mean = np.stack([single_c[s][key] for s in ss]).mean(0)
            rec[key] = {"all_compliant": {"ENS": float(np.nanmean(ec[key][mask])), "MEDOID": float(np.nanmean(mc[key][mask])), "mean_of_singles": float(np.nanmean(single_mean[mask]))},
                        "disagree_subset": ({"ENS": float(np.nanmean(ec[key][sub])), "MEDOID": float(np.nanmean(mc[key][sub])), "mean_of_singles": float(np.nanmean(single_mean[sub]))}
                                            if sub.sum() >= 1 else None)}
        if sub.sum() >= 30 and len(np.unique(eid[sub])) >= 5:
            rec["ENS_minus_MEDOID_curvature_on_disagree"] = PR.paired_cell(ec["LAT_curvature_mae_1pm_valid"][sub], mc["LAT_curvature_mae_1pm_valid"][sub], eid[sub], N_BOOT, BOOT_SEED)
            rec["ENS_minus_MEDOID_ade_on_disagree"] = PR.paired_cell(ec["ade_m"][sub], mc["ade_m"][sub], eid[sub], N_BOOT, BOOT_SEED)
            rec["blend_flag"] = blend_flag(rec["ENS_minus_MEDOID_curvature_on_disagree"])
            flag_any = flag_any or (nm in ("4A", "4B") and rec["blend_flag"])
        else:
            rec["ENS_minus_MEDOID_curvature_on_disagree"] = {"status": f"UNDERPOWERED n={int(sub.sum())}"}
            rec["blend_flag"] = False
        md[nm] = rec
    ov = overall_verdict(bars["ENS-4"], bars["MEDOID-4"], flag_any)
    # dose-response (nested, descriptive)
    dose = {}
    for K, nm in ((1, "SINGLE-0"), (2, "ENS-2"), (4, "ENS-4A"), (8, "ENS-8")):
        c = singles[0]["E1"] if K == 1 else arms.get(nm, {}).get("E1")
        dose[str(K)] = {"arm": nm, "E1": c, "mean_ade_compliant": (singles[0]["mean_ade_compliant"] if K == 1 else arms.get(nm, {}).get("mean_ade_compliant"))}
    hard = [k for k in ("C-MASK", "C-PANEL", "C-ID", "C-DIST", "C-K1", "C-TRI") if ctl.get(k, {}).get("PASS") is False]
    if "C-K2" in ctl and ctl["C-K2"]["PASS"] is False:
        hard.append("C-K2")
    if isinstance(pose_ref, dict) and "arms" in pose_ref:
        for n in ("ENS", "MEDOID"):
            f = floor_of(n)
            pose_ref.setdefault("bars_pose", {})[f"{n}-4"] = arm_verdict(arms[f"{n}-4A"].get("E1_pose"), arms[f"{n}-4B"].get("E1_pose"), f)
            pose_ref["bars_pose"][f"{n}-4"]["cells"] = {"A": fc(arms[f"{n}-4A"].get("E1_pose")), "B": fc(arms[f"{n}-4B"].get("E1_pose"))}
    res.update({"controls": ctl, "void_controls_failed": hard, "singles": {str(k): v for k, v in singles.items()}, "arms": arms, "bars": bars, "overall": ov,
                "mode_disagreement": md, "dose_response": dose, "echo_mean_ade_compliant": float(echo_c["ade_m"][mask].mean()),
                "unverified": [k for k, v in ctl.items() if isinstance(v, dict) and v.get("UNVERIFIED")],
                "pose_reference": pose_ref, "wall_s": round(time.time() - t0, 1)})
    if hard:
        res["overall"] = {"overall": "VOID (controls failed: " + ", ".join(hard) + ")", "per_arm": ov["per_arm"], "would_have_been": ov["overall"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "KDRAW_E1.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (out_dir / "KDRAW_E1.md").write_text(render(res), encoding="utf-8")
    return res


# ============================================================================================================== markdown
def render(r: dict) -> str:
    L = [f"# KDRAW_E1 -- K-draw ensemble vs the echo on the ceiling-compliant S2 set ({r['tree']})", "",
         f"Tier: {r['tier']}.  Estimator: {r['estimator']}.", f"Status: {r['status']}.", f"Bar of record: {r['bar_of_record']}.", "",
         f"## OVERALL: **{r['overall']['overall']}**  (per arm {r['overall']['per_arm']})", ""]
    if r.get("void_controls_failed"):
        L += [f"**VOID: controls failed -- {r['void_controls_failed']}**", ""]
    L += ["## Seeds", "", "| seed | available | dump |", "|---|---|---|"]
    for s, v in r["seeds"].items():
        L.append(f"| {s} | {v['available']} | {v['dump']} |")
    L += ["", "## Controls", "", "| control | result |", "|---|---|"]
    for k, v in r["controls"].items():
        if isinstance(v, dict):
            L.append(f"| {k} | PASS={v.get('PASS')}{' UNVERIFIED: ' + v['UNVERIFIED'] if v.get('UNVERIFIED') else ''} |")
    L += ["", f"Echo `ha0_ext` mean ADE (compliant) {r['echo_mean_ade_compliant']:.4f} m.", "", "## E1 cells: arm - echo, ADE 0-2 s, compliant windows", "",
          "| arm | seeds | E1 delta [95% CI] | mean ADE | n win / ep |", "|---|---|---|---|---|"]
    for s, v in r["singles"].items():
        c = v["E1"]
        L.append(f"| SINGLE | {s} | {fc(c)} | {v['mean_ade_compliant']:.4f} | {c.get('n_windows')} / {c.get('n_episodes')} |")
    for n, a in r["arms"].items():
        if "E1" in a:
            c = a["E1"]
            L.append(f"| {n} | {a['seeds']} | {fc(c)} | {a['mean_ade_compliant']:.4f} | {c.get('n_windows')} / {c.get('n_episodes')} |")
        else:
            L.append(f"| {n} | {a['seeds']} | {a['status']} | | |")
    L += ["", "## Verdicts (PREREG section 3)", ""]
    for n, b in r["bars"].items():
        L.append(f"* **{n}: {b['verdict']}**  cells A {b['cells']['A']}  B {b['cells']['B']}  detail: { {k: v for k, v in b.items() if k not in ('cells', 'verdict')} }")
    pr = r.get("pose_reference") or {}
    if isinstance(pr, dict) and pr.get("arms"):
        L += ["", f"## D-R8-E1-REF: the same arms against `ha0_ext_pose` (BESIDE the pre-registered read above; reference {pr['reference']})", "",
              f"Pose-echo mean ADE (compliant) {pr['echo_pose_mean_ade_compliant']:.4f} m.  Controls PASS = **{pr['controls']['PASS']}** ({ {k: v for k, v in pr['controls'].items() if k != 'decisions_check'} }).", "",
              "| arm | seeds | E1_pose delta [95% CI] |", "|---|---|---|"]
        for s, v in r["singles"].items():
            L.append(f"| SINGLE | {s} | {fc(v.get('E1_pose'))} |")
        for n, a in r["arms"].items():
            if "E1_pose" in a:
                L.append(f"| {n} | {a['seeds']} | {fc(a['E1_pose'])} |")
        for n, b in (pr.get("bars_pose") or {}).items():
            L.append(f"\n* **{n} (pose): {b['verdict']}**  cells A {b['cells']['A']}  B {b['cells']['B']}")
    elif isinstance(pr, dict) and pr.get("status"):
        L += ["", f"## D-R8-E1-REF pose read: {pr['status']}"]
    L += ["", "## Mode disagreement (spread of the K 2-s endpoints > 1.0 m)", ""]
    for n, m in r["mode_disagreement"].items():
        if "frac_spread_gt_1m_compliant" in m:
            cu = m["LAT_curvature_mae_1pm_valid"]
            L.append(f"* K-set {n}: {100 * m['frac_spread_gt_1m_compliant']:.1f} % of compliant windows (n={m['n_disagree_compliant']}); curvature MAE all/disagree "
                     f"ENS {cu['all_compliant']['ENS']:.5f}/{(cu['disagree_subset'] or {}).get('ENS', float('nan')):.5f}, MEDOID {cu['all_compliant']['MEDOID']:.5f}/"
                     f"{(cu['disagree_subset'] or {}).get('MEDOID', float('nan')):.5f}; ENS-MEDOID curvature on disagree {fc(m['ENS_minus_MEDOID_curvature_on_disagree'], 6)}; blend flag {m['blend_flag']}")
        else:
            L.append(f"* K-set {n}: {m.get('status')}")
    L += ["", "## NR vs refcv7-50,400 (S2 headline metrics; OK = CI lower <= 0 at BOTH refcv7 seeds)", ""]
    for n, a in r["arms"].items():
        if "NR_family_rollup" in a:
            L.append(f"* {n}: {a['NR_family_rollup']}  (STRATEGIC: NOT APPLICABLE, n = 0)")
    L += ["", "## Dose-response (nested sets, descriptive)", ""]
    for K, d in r["dose_response"].items():
        L.append(f"* K={K} ({d['arm']}): {fc(d['E1'])}  mean ADE {d['mean_ade_compliant']}")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--tree", default="step30000ema_unc")
    ap.add_argument("--kd-root", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[kdraw_e1] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    out = Path(a.out_dir) if a.out_dir else OUT
    if TEST_MUTATION and out.resolve() == OUT.resolve():
        raise SystemExit("[kdraw_e1] KDRAW_TEST_MUTATION is set: refusing to write into the production raw/kdraw directory")
    r = analyse(a.tree, a.kd_root or f"{KIT}/kdraw/{a.tree}", out)
    print("OVERALL", r["overall"], "VOID" if r["void_controls_failed"] else "", "WROTE", out / "KDRAW_E1.json", f"{r['wall_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
