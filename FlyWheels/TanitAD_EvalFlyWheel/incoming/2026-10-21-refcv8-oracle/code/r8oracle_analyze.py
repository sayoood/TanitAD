"""r8oracle_analyze.py -- the R8-ORACLE analysis, MECHANICAL from PREREG_R8_ORACLE.md (sealed; refuses to run if the seal breaks).
CPU only, no model, no GPU.  Writes R8ORACLE.json + RESULT.md.

    PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' CUDA_VISIBLE_DEVICES=-1 \
      python -X utf8 r8oracle_analyze.py --selftest                        # C9 (decision function) + C8 (calibration on the BANKED trees)
      python -X utf8 r8oracle_analyze.py --root <run dir> --seeds 0,1       # the real read (oracle_s{seed}.npz + dump_s{seed}/ in <run dir>)
      python -X utf8 r8oracle_analyze.py --root <smoke dir> --seeds 1 --smoke --plain-dump <smoke dir>/plain_dump_s1   # controls only

Every interval: paired episode-cluster bootstrap (taniteval.ci, via pair_read_r8.paired_cell), B 2000, seed 0, models FIXED -- answers
'another draw of EPISODES' only; NOT another training run (one training seed); the INFERENCE question is answered by seeds 0 and 1 + the
floor F = max(|lo|, |hi|) of OWN(s0) - OWN(s1).  Tier: T0 (GT at inference) -- a diagnostic, never driving performance.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREREG = ROOT / "PREREG_R8_ORACLE.md"
SEAL = ROOT / "PREREG_R8_ORACLE.sha256"
AMEND = ROOT / "PREREG_R8_ORACLE.AMENDMENT-1.md"
AMEND_SEAL = ROOT / "PREREG_R8_ORACLE.AMENDMENT-1.sha256"
BANK = Path("C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf")
DEP = "C:/Users/Admin/refcv8_pf/wise_ship/step48852ema_pf/pfpkg/raw/retime_deploy/DEPLOY8_POS_step48852ema_pf_s{seed}.npz"
PKG_CODE = Path("C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code")
SLOTS = {"S2": [0, 1, 2, 3], "S6": [1, 3, 4, 5, 6, 7]}
DT = {"S2": 0.5, "S6": 1.0}
MAIN = ("GT_CONS", "GT_HYP", "GT_BOTH")
BESIDE = ("GT_SPEED_ONLY", "OWN_ARGMAX")
SWAP = ("GT_HYP_SWAP", "GT_BOTH_SWAP")          # AMENDMENT 1 (beside; decides nothing)
THR = 0.6                    # committed share threshold (prereg section 5)
SPEED_FRAC = 0.6             # beside: share_SPEED >= 0.6 * share_CONS
C1_TOL, C6_TOL = 1e-5, 1e-3
N_BOOT, BOOT_SEED = 2000, 0
PRIMARY = ("S6", "ade_m")
#: C8 literals, banked FORWARD_DEPLOY8_step48852ema_pf.json `os_minus_ha0_ext_pose` (compliant): (delta, lo, hi, n_windows, n_episodes)
C8_LIT = {("S2", "0"): (-0.0069, -0.0246, 0.0112, 4159, 138), ("S2", "1"): (-0.0048, -0.0210, 0.0124, 4159, 138),
          ("S6", "0"): (-0.7285, -1.0809, -0.3758, 3251, 138), ("S6", "1"): (-0.7287, -1.0635, -0.3826, 3251, 138)}
TIER = "T0 (GT tactical decision and/or constraints fed at inference): a planner DIAGNOSTIC, never driving performance"
QUESTION = ("paired episode-cluster bootstrap (taniteval.ci), B 2000, seed 0, models FIXED: 'another draw of EPISODES' only; NOT another "
            "training run (one training seed, H-ESTIM-SEED-1); inference: seeds 0 and 1 + the floor F")


def check_seal() -> str:
    got = []
    for doc, seal in ((PREREG, SEAL), (AMEND, AMEND_SEAL)):
        want = seal.read_text(encoding="utf-8").split()[0].strip()
        g = hashlib.sha256(doc.read_bytes()).hexdigest()
        if len(want) != 64 or g != want:
            raise SystemExit(f"[r8oracle-analyze] {doc.name} sha256 {g} != sealed {want}")
        got.append(g)
    return "+".join(got)


# ============================================================================================ pure decision logic (C9 pins it)
def reading(per_seed: dict) -> str:
    """per_seed[s] = {"H_point", "H_separated", "floor", "share_CONS", "share_HYP"} -> the committed reading (prereg section 5)."""
    if not per_seed:
        return "INCOMPLETE"
    for v in per_seed.values():
        if any(v.get(k) is None for k in ("H_point", "H_separated", "floor")):
            return "INCOMPLETE"
        if not (bool(v["H_separated"]) and v["H_point"] > 0 and v["H_point"] > v["floor"]):
            return "NO-HEADROOM"
    for v in per_seed.values():
        if v.get("share_CONS") is None or v.get("share_HYP") is None or not all(map(np.isfinite, (v["share_CONS"], v["share_HYP"]))):
            return "INCOMPLETE"
    sc = all(v["share_CONS"] >= THR for v in per_seed.values())
    sh = all(v["share_HYP"] >= THR for v in per_seed.values())
    if sc and sh:
        return "REDUNDANT"
    if sc:
        return "CONSTRAINT-DOMINANT"
    if sh:
        return "DECISION-DOMINANT"
    return "MIXED"


def decompose(own, cons, hyp, both, eid, n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    """H = OWN - GT_BOTH, c = OWN - GT_CONS, d = OWN - GT_HYP, I = H - c - d (lower-is-better metric); shares on the SAME draws."""
    from taniteval import ci
    own, cons, hyp, both = (np.asarray(x, np.float64) for x in (own, cons, hyp, both))
    keep = np.isfinite(own) & np.isfinite(cons) & np.isfinite(hyp) & np.isfinite(both)
    own, cons, hyp, both = own[keep], cons[keep], hyp[keep], both[keep]
    e = [x for x, k in zip(eid, keep) if k]
    H, c, d = float((own - both).mean()), float((own - cons).mean()), float((own - hyp).mean())
    uniq, idx = ci.episode_index(e)
    sC, sH, sI = [], [], []
    for sel in ci._draws(uniq, idx, n_boot, seed):
        h = (own[sel] - both[sel]).mean()
        cc = (own[sel] - cons[sel]).mean()
        dd = (own[sel] - hyp[sel]).mean()
        sC.append(cc / h if h != 0 else np.nan)
        sH.append(dd / h if h != 0 else np.nan)
        sI.append((h - cc - dd) / h if h != 0 else np.nan)

    def q(v):
        v = np.asarray(v, np.float64)
        v = v[np.isfinite(v)]
        return [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])] if v.size else [None, None]
    sh = lambda x: (x / H) if H != 0 else float("nan")  # noqa: E731
    return {"H": H, "c": c, "d": d, "I": H - c - d, "share_CONS": sh(c), "share_HYP": sh(d), "share_INT": sh(H - c - d),
            "share_CONS_ci": q(sC), "share_HYP_ci": q(sH), "share_INT_ci": q(sI), "n_windows": int(keep.sum()),
            "n_episodes": int(len(uniq)), "n_dropped_nonfinite": int((~keep).sum()), "n_boot": n_boot,
            "estimator": "ratio of resampled means, SAME episode-cluster draws as taniteval.ci._draws(seed 0) for every term"}


def selftest_c9() -> dict:
    """C9: the decision function on synthetic arms with KNOWN answers (literals, never derived from the code under test)."""
    rng = np.random.default_rng(1)
    n = 600
    eid = np.repeat(np.arange(60), 10)
    own = 2.0 + rng.normal(0, 0.05, n)
    both = own - 1.0
    cons = own - 0.9                     # c = 0.9 of H = 1.0
    hyp = own - 0.1                      # d = 0.1
    out = {}
    D = decompose(own, cons, hyp, both, eid, n_boot=200)
    out["shares_known"] = bool(abs(D["share_CONS"] - 0.9) < 1e-9 and abs(D["share_HYP"] - 0.1) < 1e-9 and abs(D["H"] - 1.0) < 1e-9)
    ps = {"0": {"H_point": D["H"], "H_separated": True, "floor": 0.01, "share_CONS": D["share_CONS"], "share_HYP": D["share_HYP"]}}
    ps["1"] = dict(ps["0"])
    out["cons_dominant"] = reading(ps) == "CONSTRAINT-DOMINANT"
    sw = {s: dict(v, share_CONS=v["share_HYP"], share_HYP=v["share_CONS"]) for s, v in ps.items()}
    out["swap_gives_decision_dominant"] = reading(sw) == "DECISION-DOMINANT"
    D0 = decompose(own, cons, hyp, own, eid, n_boot=200)          # OWN in place of GT_BOTH
    ps0 = {s: {"H_point": D0["H"], "H_separated": False, "floor": 0.01, "share_CONS": D0["share_CONS"], "share_HYP": D0["share_HYP"]}
           for s in ("0", "1")}
    out["own_as_both_H_exactly_0"] = D0["H"] == 0.0
    out["own_as_both_no_headroom"] = reading(ps0) == "NO-HEADROOM"
    mixed = {s: dict(v, share_CONS=0.5, share_HYP=0.5) for s, v in ps.items()}
    out["mixed"] = reading(mixed) == "MIXED"
    red = {s: dict(v, share_CONS=0.8, share_HYP=0.7) for s, v in ps.items()}
    out["redundant"] = reading(red) == "REDUNDANT"
    one_seed = {"0": ps["0"], "1": dict(ps["1"], share_CONS=0.5)}
    out["needs_both_seeds"] = reading(one_seed) == "MIXED"
    floor_bites = {s: dict(v, floor=2.0) for s, v in ps.items()}
    out["floor_bites"] = reading(floor_bites) == "NO-HEADROOM"
    out["ALL_PASS"] = all(out.values())
    return out


# ============================================================================================ data
def load_tree(dump_dir: Path, decisions=True) -> dict:
    """{(clip_index, ws): row dict} from a battery dump tree (ep*.npz + decisions/ep*.npz)."""
    rows = {}
    for f in sorted(glob.glob(str(dump_dir / "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        ci_ = int(z["clip_index"][0])
        for j, w in enumerate(z["ws"]):
            rows[(ci_, int(w))] = {"os": z["os"][j], "g": z["g"][j], "eid": int(z["eid"][0]) if "eid" in z.files else ci_}
        if decisions:
            fd = dump_dir / "decisions" / os.path.basename(f)
            if fd.exists():
                d = np.load(fd, allow_pickle=True)
                for j, w in enumerate(d["ws"]):
                    r = rows.setdefault((ci_, int(w)), {})
                    r["plan8"] = d["plan_full_nav_true"][j]
                    r["gt_ext"] = d["gt_future_ext"][j]
    return rows


def load_panel(pdir: Path) -> dict:
    rows = {}
    for f in sorted(glob.glob(str(pdir / "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        ci_ = int(z["clip_index"][0])
        for j, w in enumerate(z["ws"]):
            rows[(ci_, int(w))] = {"g": z["g"][j], "os": z["os"][j], "v0": float(z["v0"][j]), "file": os.path.basename(f)}
    return rows


def load_dep(seed: str) -> dict:
    z = np.load(DEP.format(seed=seed), allow_pickle=True)
    out = {}
    for S in ("S2", "S6"):
        out[S] = {(int(c), int(w)): {"prior8": z[f"{S}_prior8"][i], "comp": bool(z[f"{S}_compliant"][i]), "eid": int(z[f"{S}_eid"][i]),
                                     "os": z[f"{S}_os"][i]}
                  for i, (c, w) in enumerate(zip(z[f"{S}_clip"], z[f"{S}_ws"]))}
    return out


def load_oracle(p: Path) -> dict:
    z = np.load(p, allow_pickle=True)
    o = {k: z[k] for k in z.files}
    o["meta"] = json.loads(str(z["meta_json"]))
    o["key"] = [(int(c), int(w)) for c, w in zip(o["clip_index"], o["ws"])]
    o["row"] = {k: i for i, k in enumerate(o["key"])}
    return o


def ade(P, G):
    return np.linalg.norm(np.asarray(P, np.float64) - np.asarray(G, np.float64), axis=-1).mean(-1)


def cls_thr(P, v0, dt=1.0, thr=1.5):
    """an3.py's rule, verbatim in effect: v_end from the last two S6 points; decel if dv <= -1.5, accel if >= +1.5."""
    v_end = np.linalg.norm(P[:, -1] - P[:, -2], axis=-1) / dt
    dv = v_end - v0
    return np.where(dv <= -thr, "decel", np.where(dv >= thr, "accel", "cruise")).astype(object)


_PR = {}


def pr_mod():
    """pair_read_r8 (paired_cell, components, METRICS) + refav1_arm (`ra`), exactly as pair_read_r8.main obtains them."""
    if not _PR:
        sys.path.insert(0, str(PKG_CODE))
        import pair_read_r8 as PR
        import r8_battery_patch as BP
        L, P = BP.apply()
        _PR.update(PR=PR, ra=P.RR.ra3().ra)
    return _PR["PR"], _PR["ra"]


def cell(new, ref, eid):
    PR, _ = pr_mod()
    return PR.paired_cell(new, ref, eid, N_BOOT, BOOT_SEED)


# ============================================================================================ C8 calibration (banked trees only)
def calibrate_c8(seeds=("0", "1")) -> dict:
    out = {}
    for s in seeds:
        dep = load_dep(s)
        pan = {"S2": load_panel(BANK / f"panel_s{s}"), "S6": load_panel(BANK / f"s6_s{s}")}
        for S in ("S2", "S6"):
            keys = [k for k, v in dep[S].items() if v["comp"]]
            new = ade(np.stack([pan[S][k]["os"] for k in keys]), np.stack([pan[S][k]["g"] for k in keys]))
            ref = ade(np.stack([dep[S][k]["prior8"][SLOTS[S]] for k in keys]), np.stack([pan[S][k]["g"] for k in keys]))
            eid = [dep[S][k]["eid"] for k in keys]
            c = cell(new, ref, eid)
            lit = C8_LIT[(S, s)]
            ok = (round(c["delta"], 4) == lit[0] and round(c["lo"], 4) == lit[1] and round(c["hi"], 4) == lit[2]
                  and c["n_windows"] == lit[3] and c["n_episodes"] == lit[4])
            out[f"{S}_s{s}"] = {"mine": {k: c[k] for k in ("delta", "lo", "hi", "n_windows", "n_episodes")}, "literal": list(lit), "pass": bool(ok)}
    out["ALL_PASS"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


# ============================================================================================ the read
def read_seed(root: Path, s: str, arms: list, smoke: bool, plain_dump: Path | None) -> dict:
    o = load_oracle(root / f"oracle_s{s}.npz")
    run = load_tree(root / f"dump_s{s}")
    bank = load_tree(BANK / f"dump_s{s}") if not smoke else {k: v for k, v in load_tree(BANK / f"dump_s{s}").items() if k in o["row"]}
    pan = {"S2": load_panel(BANK / f"panel_s{s}"), "S6": load_panel(BANK / f"s6_s{s}")}
    dep = load_dep(s)
    R = {"n_windows_oracle": len(o["key"]), "meta": o["meta"], "controls": {}}
    C = R["controls"]
    # ---- C2 in-run identity
    d2 = [float(np.abs(o["traj_OWN"][o["row"][k]] - run[k]["plan8"]).max()) for k in o["key"] if k in run and "plan8" in run[k]]
    C["C2_own_capture_eq_dump"] = {"n": len(d2), "max_abs": max(d2) if d2 else None, "pass": bool(d2) and max(d2) == 0.0 and len(d2) == len(o["key"])}
    # ---- C1 OWN reproduces the battery (dump os + plan_full) / CPU smoke: vs the plain child on the same device
    ref_tree = load_tree(plain_dump) if plain_dump is not None else bank
    keys1 = [k for k in o["key"] if k in run and k in ref_tree]
    if keys1:
        dos = max(float(np.abs(run[k]["os"] - ref_tree[k]["os"]).max()) for k in keys1)
        dpl = max(float(np.abs(run[k]["plan8"] - ref_tree[k]["plan8"]).max()) for k in keys1 if "plan8" in ref_tree[k])
    else:
        dos = dpl = None
    n_over = sum(1 for k in keys1 if float(np.abs(run[k]["os"] - ref_tree[k]["os"]).max()) > C1_TOL) if keys1 else None
    binding = (plain_dump is not None) or (s == "1")
    C["C1_own_reproduces"] = {"against": str(plain_dump) if plain_dump is not None else str(BANK / f"dump_s{s}"), "n": len(keys1),
                              "max_abs_os": dos, "max_abs_plan8": dpl, "tol": C1_TOL, "binding": binding, "n_windows_over_tol": n_over,
                              "s2_ade_run_minus_ref": (cell(ade(np.stack([run[k]["os"] for k in keys1]), np.stack([run[k]["g"] for k in keys1])),
                                                            ade(np.stack([ref_tree[k]["os"] for k in keys1]), np.stack([ref_tree[k]["g"] for k in keys1])),
                                                            [k[0] for k in keys1]) if (keys1 and not smoke) else None),
                              "pass": (bool(keys1) and dos is not None and max(dos, dpl) <= C1_TOL) if binding else None,
                              "note": None if binding else "seed 0: banked s0 is the FULL arm-set roll (different sampler stream); judged vs the floor"}
    if smoke and plain_dump is not None and bank:
        kb = [k for k in o["key"] if k in run and k in bank]
        C["C1_beside_cpu_vs_banked_gpu"] = {"n": len(kb), "max_abs_os": max(float(np.abs(run[k]["os"] - bank[k]["os"]).max()) for k in kb) if kb else None,
                                            "note": "cross-device (CPU fp32 trunk vs banked CUDA bf16): NOT bound to 1e-5"}
    # ---- C3 seam identity at self
    if "SELF_NAN" in arms:
        m = o["has_SELF_NAN"] & o["forced_SELF_NAN"]
        dt_ = np.abs(o["traj_SELF_NAN"][m] - o["traj_OWN"][m])
        C["C3_self_nan_eq_own"] = {"n": int(m.sum()), "max_abs_traj": float(dt_.max()) if m.any() else None,
                                   "sel_equal": bool((o["sel_SELF_NAN"][m] == o["sel_OWN"][m]).all()) if m.any() else None,
                                   "force_dict_reached": bool(o["spy_fd_SELF_NAN"][m].all()) if m.any() else None,
                                   "force_min": sorted(set(int(x) for x in o["spy_fmin_SELF_NAN"][m])) if m.any() else None,
                                   "pass": bool(m.any()) and float(dt_.max()) == 0.0 and bool((o["sel_SELF_NAN"][m] == o["sel_OWN"][m]).all())}
    # ---- C3b (amendment 1): the swap is the identity where the own argmax already IS the GT tag on both axes
    if "GT_HYP_SWAP" in arms:
        m = o["forced_GT_HYP_SWAP"] & (o["spy_hl_OWN"] == o["gt_gl"]) & (o["spy_ho_OWN"] == o["gt_go"])
        C["C3b_swap_identity_where_own_eq_gt"] = {
            "n": int(m.sum()), "max_abs_traj": float(np.abs(o["traj_GT_HYP_SWAP"][m] - o["traj_OWN"][m]).max()) if m.any() else None,
            "sel_equal": bool((o["sel_GT_HYP_SWAP"][m] == o["sel_OWN"][m]).all()) if m.any() else None,
            "pass": (float(np.abs(o["traj_GT_HYP_SWAP"][m] - o["traj_OWN"][m]).max()) == 0.0
                     and bool((o["sel_GT_HYP_SWAP"][m] == o["sel_OWN"][m]).all())) if m.any() else None,
            "voids_if_failed": "GT_HYP_SWAP, GT_BOTH_SWAP only"}
    # ---- C4 GT-as-prediction readback
    c4 = {}
    gt_c = o["gt_cons_force"]
    for arm in [a for a in ("GT_CONS", "GT_BOTH", "GT_SPEED_ONLY", "GT_BOTH_SWAP") if a in arms]:
        m = o[f"forced_{arm}"]
        want = gt_c.copy()
        if arm == "GT_SPEED_ONLY":
            want[:, :3] = np.nan
        fin = np.isfinite(want) & m[:, None]
        err = np.abs(o[f"spy_cons_{arm}"][fin] - want[fin])
        c4[arm] = {"n_windows": int(m.sum()), "n_entries": int(fin.sum()), "max_abs_err": float(err.max()) if err.size else None,
                   "pass": bool(err.size) and float(err.max()) == 0.0}
    for arm in [a for a in ("GT_HYP", "GT_BOTH", "GT_HYP_SWAP", "GT_BOTH_SWAP") if a in arms]:
        m = o[f"forced_{arm}"]
        ml, mo = m & (o["gt_gl"] >= 0), m & (o["gt_go"] >= 0)
        al = float((o[f"spy_hl_{arm}"][ml] == o["gt_gl"][ml]).mean()) if ml.any() else None
        ao = float((o[f"spy_ho_{arm}"][mo] == o["gt_go"][mo]).mean()) if mo.any() else None
        c4[f"{arm}_class"] = {"n_lat": int(ml.sum()), "lat_agree": al, "n_lon": int(mo.sum()), "lon_agree": ao,
                              "pass": al == 1.0 and ao == 1.0}
    fin = np.isfinite(gt_c)
    c4["gt_vs_gt_mae"] = {"value": float(np.abs(gt_c[fin] - gt_c[fin]).mean()) if fin.any() else None, "pass": bool(fin.any())}
    c4["ALL_PASS"] = all(v["pass"] for v in c4.values())
    C["C4_gt_readback"] = c4
    # ---- C6 prior invariance + parity
    pm = {arm: float(np.nanmax(o[f"prior_maxdiff_{arm}"][o[f"forced_{arm}"]])) if o[f"forced_{arm}"].any() else None for arm in arms}
    kp = [k for k in o["key"] if k in dep["S2"]]
    par = max(float(np.abs(o["prior_OWN"][o["row"][k]] - dep["S2"][k]["prior8"]).max()) for k in kp) if kp else None
    C["C6_prior"] = {"max_abs_across_arms": pm, "parity_vs_banked_prior8_max_abs": par, "n_parity": len(kp), "tol": C6_TOL,
                     "pass": all(v == 0.0 for v in pm.values() if v is not None) and par is not None and par <= C6_TOL}
    # ---- C7 GT parity
    kg = [k for k in o["key"] if k in run and k in bank]
    g7 = max(float(np.abs(run[k]["g"] - bank[k]["g"]).max()) for k in kg) if kg else None
    C["C7_gt_parity"] = {"n": len(kg), "max_abs_g": g7, "pass": bool(kg) and g7 == 0.0}
    # ---- forcing census + descriptive
    cen = {arm: {"forced": int(o[f"forced_{arm}"].sum()), "rolled": int(o[f"has_{arm}"].sum())} for arm in arms}
    has = o["gt_gl"] >= 0
    joint_eq = (o["spy_hl_OWN"] == o["gt_gl"]) & (o["spy_ho_OWN"] == o["gt_go"])
    R["census"] = {"arms": cen, "n_gt_hyp": int(has.sum()), "own_argmax_eq_gt_lat": float((o["spy_hl_OWN"][has] == o["gt_gl"][has]).mean()) if has.any() else None,
                   "own_argmax_eq_gt_lon": float((o["spy_ho_OWN"][o["gt_go"] >= 0] == o["gt_go"][o["gt_go"] >= 0]).mean()) if (o["gt_go"] >= 0).any() else None,
                   "own_argmax_eq_gt_joint": float(joint_eq[has].mean()) if has.any() else None,
                   "wall_by_arm_s": o["meta"].get("wall_by_arm_s")}
    # own head readback where its argmax IS the GT class (the cons the planner gets from its own head, vs GT)
    names = ("psi_term", "t_onset", "prog6", "v_end")
    hr = {}
    for f_, nm in enumerate(names):
        okc = (o["spy_hl_OWN"] == o["gt_gl"]) if f_ < 2 else (o["spy_ho_OWN"] == o["gt_go"])
        m = okc & np.isfinite(gt_c[:, f_])
        hr[nm] = {"n": int(m.sum()), "mae_norm": float(np.abs(o["spy_cons_OWN"][m, f_] - gt_c[m, f_]).mean()) if m.any() else None}
    R["own_head_readback_on_argmax_eq_gt"] = hr
    # ---- metrics per arm
    if smoke:
        R["smoke_means"] = {}
        for S in ("S2", "S6"):
            ks = [k for k in o["key"] if k in pan[S]]
            if not ks:
                continue
            G = np.stack([pan[S][k]["g"] for k in ks])
            R["smoke_means"][S] = {arm: float(ade(o[f"traj_{arm}"][[o["row"][k] for k in ks]][:, SLOTS[S]], G).mean())
                                   for arm in ["OWN"] + arms if o[f"has_{arm}"][[o["row"][k] for k in ks]].all()}
        return R
    PR, ra = pr_mod()
    R["surfaces"] = {}
    per_window = {}
    for S in ("S2", "S6"):
        ks = [k for k in o["key"] if k in pan[S]]
        G = np.stack([pan[S][k]["g"] for k in ks]).astype(np.float32)
        rows = [o["row"][k] for k in ks]
        eid = [int(pan[S][k]["file"][2:5]) for k in ks]
        comps = {arm: PR.components(ra, o[f"traj_{arm}"][rows][:, SLOTS[S]].astype(np.float32), G, DT[S]) for arm in ["OWN"] + arms}
        per_window[S] = {"keys": ks, "eid": eid, "comps": comps}
        sur = {"n_windows": len(ks), "n_episodes": len(set(eid)), "means": {}, "rows": {}}
        for arm in ["OWN"] + arms:
            sur["means"][arm] = {k: (float(np.nanmean(v)) if np.isfinite(v).any() else None) for k, v in comps[arm].items()}
        for fam, key, label, unit, lower, _nr in PR.METRICS:
            sur["rows"][key] = {"family": fam, "label": label, "unit": unit, "lower_is_better": lower,
                                **{f"{arm}_minus_OWN": cell(comps[arm][key], comps["OWN"][key], eid) for arm in arms}}
        # E1 / E3 vs ha0_ext_pose on compliant windows
        kc = [k for k in ks if k in dep[S] and dep[S][k]["comp"]]
        rc = [o["row"][k] for k in kc]
        Gc = np.stack([pan[S][k]["g"] for k in kc])
        prior = ade(np.stack([dep[S][k]["prior8"][SLOTS[S]] for k in kc]), Gc)
        ec = [dep[S][k]["eid"] for k in kc]
        sur["E_vs_ha0_ext_pose_compliant"] = {arm: cell(ade(o[f"traj_{arm}"][rc][:, SLOTS[S]], Gc), prior, ec) for arm in ["OWN"] + arms}
        # false-decel on GT cruise (an3) -- S6 only
        if S == "S6":
            v0 = np.array([pan[S][k]["v0"] for k in ks], np.float64)
            gtc = cls_thr(G.astype(np.float64), v0)
            fd = {}
            for arm in ["OWN"] + arms:
                pc = cls_thr(o[f"traj_{arm}"][rows][:, SLOTS[S]].astype(np.float64), v0)
                fd[arm] = {"fd": np.where(gtc == "cruise", (pc == "decel").astype(float), np.nan),
                           "md": np.where(gtc == "decel", (pc != "decel").astype(float), np.nan)}
            sur["false_decel_on_gt_cruise"] = {"n_gt_cruise": int((gtc == "cruise").sum()), "n_gt_decel": int((gtc == "decel").sum()),
                                               "rate": {arm: float(np.nanmean(fd[arm]["fd"])) for arm in fd},
                                               "missed_decel_rate": {arm: float(np.nanmean(fd[arm]["md"])) for arm in fd},
                                               "minus_OWN": {arm: cell(fd[arm]["fd"], fd["OWN"]["fd"], eid) for arm in arms}}
        sur["STRATEGIC"] = {"status": "NOT APPLICABLE", "n": 0, "reason": "SHIP-R8 clause (6) --no-strategic; route checkpoint off (--r8-no-rc)"}
        # decomposition (prereg section 5)
        if all(a in arms for a in MAIN):
            dec = {}
            for key in ("ade_m",):
                D = decompose(comps["OWN"][key], comps["GT_CONS"][key], comps["GT_HYP"][key], comps["GT_BOTH"][key], eid)
                D["H_cell"] = cell(comps["OWN"][key], comps["GT_BOTH"][key], eid)
                D["c_cell"] = cell(comps["OWN"][key], comps["GT_CONS"][key], eid)
                D["d_cell"] = cell(comps["OWN"][key], comps["GT_HYP"][key], eid)
                if "GT_SPEED_ONLY" in arms:
                    sp = float(np.nanmean(comps["OWN"][key] - comps["GT_SPEED_ONLY"][key]))
                    D["s"] = sp
                    D["share_SPEED"] = sp / D["H"] if D["H"] != 0 else float("nan")
                    D["s_cell"] = cell(comps["OWN"][key], comps["GT_SPEED_ONLY"][key], eid)
                if "OWN_ARGMAX" in arms:
                    sharp = float(np.nanmean(comps["OWN"][key] - comps["OWN_ARGMAX"][key]))
                    D["sharp"] = sharp
                    D["correction"] = D["d"] - sharp
                    D["sharp_cell"] = cell(comps["OWN"][key], comps["OWN_ARGMAX"][key], eid)
                dec[key] = D
            if all(a in arms for a in SWAP):
                Dsw = decompose(comps["OWN"]["ade_m"], comps["GT_CONS"]["ade_m"], comps["GT_HYP_SWAP"]["ade_m"], comps["GT_BOTH_SWAP"]["ade_m"], eid)
                Dsw["H_cell"] = cell(comps["OWN"]["ade_m"], comps["GT_BOTH_SWAP"]["ade_m"], eid)
                Dsw["d_cell"] = cell(comps["OWN"]["ade_m"], comps["GT_HYP_SWAP"]["ade_m"], eid)
                dec["ade_m_SWAP_beside"] = Dsw
            sur["decomposition"] = dec
        R["surfaces"][S] = sur
    R["_per_window"] = per_window
    return R


def build(root: Path, seeds: list, arms: list, smoke: bool, plain_dump: Path | None) -> dict:
    res = {"tool": "r8oracle_analyze.py", "generated": time.strftime("%FT%T%z"), "prereg_sha256": check_seal(), "tier": TIER,
           "estimator": QUESTION, "root": str(root), "seeds": seeds, "arms": arms, "smoke": smoke,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    res["C9_selftest"] = selftest_c9()
    if not smoke:
        res["C8_calibration"] = calibrate_c8(tuple(seeds))
    per = {s: read_seed(root, s, arms, smoke, plain_dump) for s in seeds}
    pw = {s: per[s].pop("_per_window", None) for s in seeds}
    res["per_seed"] = per
    if smoke or len(seeds) < 2:
        res["verdict"] = "NOT EVALUATED (smoke / one seed)"
        return res
    # inference-seed floor F = max(|lo|, |hi|) of OWN(s0) - OWN(s1), per surface, ade_m, same windows
    floor = {}
    for S in ("S2", "S6"):
        k0 = {k: i for i, k in enumerate(pw["0"][S]["keys"])}
        common = [k for k in pw["1"][S]["keys"] if k in k0]
        i1 = {k: i for i, k in enumerate(pw["1"][S]["keys"])}
        a0 = pw["0"][S]["comps"]["OWN"]["ade_m"][[k0[k] for k in common]]
        a1 = pw["1"][S]["comps"]["OWN"]["ade_m"][[i1[k] for k in common]]
        c = cell(a0, a1, [pw["0"][S]["eid"][k0[k]] for k in common])
        floor[S] = {"cell_own_s0_minus_s1": c, "F": max(abs(c["lo"]), abs(c["hi"]))}
    res["inference_floor"] = floor
    readings = {}
    for S in ("S6", "S2"):
        ps = {}
        for s in seeds:
            D = per[s]["surfaces"][S].get("decomposition", {}).get("ade_m")
            if D is None:
                continue
            ps[s] = {"H_point": D["H"], "H_separated": D["H_cell"]["separated"], "floor": floor[S]["F"],
                     "share_CONS": D["share_CONS"], "share_HYP": D["share_HYP"]}
        r = reading(ps) if len(ps) == len(seeds) else "INCOMPLETE"
        speed = None
        if r not in ("NO-HEADROOM", "INCOMPLETE") and all("share_SPEED" in per[s]["surfaces"][S]["decomposition"]["ade_m"] for s in seeds):
            speed = all(per[s]["surfaces"][S]["decomposition"]["ade_m"]["share_SPEED"] >= SPEED_FRAC * per[s]["surfaces"][S]["decomposition"]["ade_m"]["share_CONS"]
                        for s in seeds)
        ps_sw = {}
        for s in seeds:
            Dsw = per[s]["surfaces"][S].get("decomposition", {}).get("ade_m_SWAP_beside")
            if Dsw is not None:
                ps_sw[s] = {"H_point": Dsw["H"], "H_separated": Dsw["H_cell"]["separated"], "floor": floor[S]["F"],
                            "share_CONS": Dsw["share_CONS"], "share_HYP": Dsw["share_HYP"]}
        r_sw = (reading(ps_sw) if len(ps_sw) == len(seeds) else None)
        readings[S] = {"reading": r, "inputs": ps, "beside_speed_carries_constraint_share": speed,
                       "BESIDE_amendment1_swap_reading": r_sw, "inputs_swap": ps_sw}
    ctrl_fail = []
    for s in seeds:
        C = per[s]["controls"]
        if C["C1_own_reproduces"]["binding"] and not C["C1_own_reproduces"]["pass"]:
            ctrl_fail.append(f"C1_s{s}")
        for k in ("C2_own_capture_eq_dump", "C3_self_nan_eq_own", "C6_prior", "C7_gt_parity"):
            if k in C and not C[k]["pass"]:
                ctrl_fail.append(f"{k}_s{s}")
        if not C["C4_gt_readback"]["ALL_PASS"]:
            ctrl_fail.append(f"C4_s{s}")
        if C.get("C3b_swap_identity_where_own_eq_gt", {}).get("pass") is False:
            res.setdefault("swap_arms_void", []).append(f"C3b_s{s}")
    if not res["C9_selftest"]["ALL_PASS"]:
        ctrl_fail.append("C9")
    if not res["C8_calibration"]["ALL_PASS"]:
        ctrl_fail.append("C8")
    res["controls_failed"] = ctrl_fail
    c3_void = any(k.startswith("C3") for k in ctrl_fail)
    prim = readings["S6"]["reading"]
    res["readings"] = readings
    res["verdict"] = ("INCONCLUSIVE-CONTROL (" + ", ".join(ctrl_fail) + f"; uncontrolled primary would read {prim})") if ctrl_fail else prim
    res["c3_void_constraint_arms"] = c3_void
    return res


# ============================================================================================ RESULT.md (mechanical)
def _c(c, dp=4):
    if not isinstance(c, dict) or "delta" not in c:
        return "n/a"
    return f"{c['delta']:+.{dp}f} [{c['lo']:+.{dp}f}, {c['hi']:+.{dp}f}]{' sep' if c.get('separated') else ''}"


def _f(x, dp=4):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{dp}f}"


def render(res: dict) -> str:
    L = [f"# R8-ORACLE: where is refcv8's tactical headroom? (EMA-48,852, per-frame nav) -- {res['tier'].split(':')[0]}", "",
         f"**Verdict (prereg section 5, S6 ADE primary, both inference seeds): {res['verdict']}**", "",
         f"Tier: {res['tier']}.  Estimator: {res['estimator']}.  Prereg sha256 `{res['prereg_sha256']}` (sealed before any forced number). "
         f"Generated {res['generated']} by `r8oracle_analyze.py` sha256 `{res['script_sha256'][:16]}`.", ""]
    if "readings" in res:
        L += ["## 1. Decomposition (headroom H = OWN - GT_BOTH; shares on the same episode draws)", "",
              "| surface | seed | H (m) | c = OWN-GT_CONS | d = OWN-GT_HYP | share_CONS [CI] | share_HYP [CI] | share_INT | floor F | reading |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for S in ("S6", "S2"):
            for s in res["seeds"]:
                D = res["per_seed"][s]["surfaces"][S].get("decomposition", {}).get("ade_m")
                if D is None:
                    continue
                L.append(f"| {S}{' (PRIMARY)' if S == 'S6' else ' (beside)'} | {s} | {_c(D['H_cell'])} | {_c(D['c_cell'])} | {_c(D['d_cell'])} | "
                         f"{_f(D['share_CONS'], 3)} {D['share_CONS_ci']} | {_f(D['share_HYP'], 3)} {D['share_HYP_ci']} | {_f(D['share_INT'], 3)} | "
                         f"{_f(res['inference_floor'][S]['F'])} | {res['readings'][S]['reading']} |")
        L += ["", "Beside (not deciding): GT_SPEED_ONLY share and the one-hot SHARPENING part of GT_HYP (OWN - OWN_ARGMAX):", "",
              "| surface | seed | s = OWN-GT_SPEED_ONLY | share_SPEED | sharp = OWN-OWN_ARGMAX | correction = d - sharp |", "|---|---|---|---|---|---|"]
        for S in ("S6", "S2"):
            for s in res["seeds"]:
                D = res["per_seed"][s]["surfaces"][S].get("decomposition", {}).get("ade_m", {})
                L.append(f"| {S} | {s} | {_c(D.get('s_cell'))} | {_f(D.get('share_SPEED'), 3)} | {_c(D.get('sharp_cell'))} | {_f(D.get('correction'))} |")
        L += ["", f"Speed carries most of the constraint share (beside rule, S6): {res['readings']['S6']['beside_speed_carries_constraint_share']}", "",
              "BESIDE (amendment 1, decides nothing): the same decomposition with the SWAP decision oracle (GT class, own confidence profile):", "",
              "| surface | seed | H_sw = OWN-GT_BOTH_SWAP | d_sw = OWN-GT_HYP_SWAP | share_CONS | share_HYP_sw | share_INT | reading (beside) |",
              "|---|---|---|---|---|---|---|---|"]
        for S in ("S6", "S2"):
            for s in res["seeds"]:
                D = res["per_seed"][s]["surfaces"][S].get("decomposition", {}).get("ade_m_SWAP_beside")
                if D is None:
                    continue
                L.append(f"| {S} | {s} | {_c(D['H_cell'])} | {_c(D['d_cell'])} | {_f(D['share_CONS'], 3)} {D['share_CONS_ci']} | "
                         f"{_f(D['share_HYP'], 3)} {D['share_HYP_ci']} | {_f(D['share_INT'], 3)} | {res['readings'][S].get('BESIDE_amendment1_swap_reading')} |")
        if res.get("swap_arms_void"):
            L.append(f"**Swap arms VOID (C3b failed: {res['swap_arms_void']}).**")
        L.append("")
    L += ["## 2. Per arm (both seeds)", "", "| seed | arm | S2 ADE | S6 ADE | E1 S2 - ha0_ext_pose (compliant) | E3 S6 - ha0_ext_pose (compliant) | false-decel on GT cruise | - OWN |",
          "|---|---|---|---|---|---|---|---|"]
    for s in res["seeds"]:
        P = res["per_seed"][s]
        if "surfaces" not in P:
            continue
        S2, S6 = P["surfaces"]["S2"], P["surfaces"]["S6"]
        for arm in ["OWN"] + res["arms"]:
            fdm = S6["false_decel_on_gt_cruise"]
            L.append(f"| {s} | {arm} | {_f(S2['means'][arm]['ade_m'])} | {_f(S6['means'][arm]['ade_m'])} | {_c(S2['E_vs_ha0_ext_pose_compliant'][arm])} | "
                     f"{_c(S6['E_vs_ha0_ext_pose_compliant'][arm])} | {_f(fdm['rate'][arm])} | {_c(fdm['minus_OWN'].get(arm)) if arm != 'OWN' else '-'} |")
    L += ["", "## 3. Four families, arm - OWN (paired; per family, never pooled). STRATEGIC: NOT APPLICABLE, n 0 (clause 6, --no-strategic).", ""]
    for s in res["seeds"]:
        P = res["per_seed"][s]
        if "surfaces" not in P:
            continue
        for S in ("S2", "S6"):
            L += [f"### seed {s}, {S} ({P['surfaces'][S]['n_windows']} windows / {P['surfaces'][S]['n_episodes']} episodes)", "",
                  "| family | metric | " + " | ".join(res["arms"]) + " |", "|---|---|" + "---|" * len(res["arms"])]
            for key, row in P["surfaces"][S]["rows"].items():
                L.append(f"| {row['family']} | {row['label']} ({row['unit']}) | " + " | ".join(_c(row.get(f"{a}_minus_OWN")) for a in res["arms"]) + " |")
            L.append("")
    L += ["## 4. Controls", "", "| control | seed | result | pass |", "|---|---|---|---|"]
    for s in res["seeds"]:
        for k, v in res["per_seed"][s]["controls"].items():
            ok = v.get("pass", v.get("ALL_PASS"))
            L.append(f"| {k} | {s} | `{json.dumps({kk: vv for kk, vv in v.items() if kk not in ('pass',)}, default=str)[:300]}` | {ok} |")
    if "C8_calibration" in res:
        L.append(f"| C8 calibration (banked os - ha0_ext_pose, literals) | both | `{json.dumps(res['C8_calibration'], default=str)[:400]}` | {res['C8_calibration']['ALL_PASS']} |")
    L.append(f"| C9 decision-function mutations | - | `{json.dumps(res['C9_selftest'])[:300]}` | {res['C9_selftest']['ALL_PASS']} |")
    L += ["", "## 5. Census", ""]
    for s in res["seeds"]:
        L.append(f"* seed {s}: `{json.dumps(res['per_seed'][s]['census'], default=str)}`")
        L.append(f"* seed {s} own head readback (normalised MAE where its argmax IS the GT class): `{json.dumps(res['per_seed'][s]['own_head_readback_on_argmax_eq_gt'])}`")
    L += ["", "## 6. What this is not", "",
          "T0: GT enters at inference; no row here is driving performance. One training seed. The v8F-mini shares (0.88-0.90 constraints / "
          "0.07-0.12 decisions) are INHERITED from a different model and are compared, never replicated. A GT_CONS force writes the GT path's "
          "constraints into the planner's ARGMAX hypothesis (the seam's semantics); GT_HYP is a ONE-HOT force (OWN_ARGMAX isolates sharpening)."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--arms", default=None, help="default: the arms recorded in the oracle npz meta (common to all seeds)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plain-dump")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    a = ap.parse_args()
    seal = check_seal()
    if a.selftest:
        r = {"prereg_sha256": seal, "C9": selftest_c9(), "C8": calibrate_c8()}
        out = Path(a.out_json or ROOT / "SELFTEST.json")
        json.dump(r, open(out, "w", encoding="utf-8"), indent=1, default=str)
        print(f"[selftest] C9 {r['C9']['ALL_PASS']} C8 {r['C8']['ALL_PASS']} -> {out} {'ZZSELFOKZZ' if r['C9']['ALL_PASS'] and r['C8']['ALL_PASS'] else 'ZZSELFBADZZ'}")
        return 0 if (r["C9"]["ALL_PASS"] and r["C8"]["ALL_PASS"]) else 1
    root = Path(a.root)
    seeds = [s for s in a.seeds.split(",") if s]
    if a.arms:
        arms = [x for x in a.arms.split(",") if x]
    else:                                                       # default: the arms the roll actually ran (its meta), common to every seed
        per = [json.loads(str(np.load(root / f"oracle_s{s}.npz", allow_pickle=True)["meta_json"]))["arms"] for s in seeds]
        arms = [x for x in per[0] if all(x in p for p in per)]
    res = build(root, seeds, arms, a.smoke, Path(a.plain_dump) if a.plain_dump else None)
    oj = Path(a.out_json or root / ("R8ORACLE_SMOKE.json" if a.smoke else "R8ORACLE.json"))
    json.dump(res, open(oj, "w", encoding="utf-8"), indent=1, default=str)
    om = Path(a.out_md or root / ("RESULT_SMOKE.md" if a.smoke else "RESULT.md"))
    om.write_text(render(res), encoding="utf-8")
    print(f"[r8oracle-analyze] verdict={res['verdict']} -> {oj} {om}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
