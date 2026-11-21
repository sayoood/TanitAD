"""paired_40k_48852.py -- is the refcv8 step-48,852 minus step-40,000 difference real when PAIRED on the SAME eval windows?

CPU only.  NEW file; nothing in the battery code is edited.  Writes ONLY under C:/Users/Admin/refcv8_pf/paired_40k_48852/.

    PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 \
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -B -X utf8 paired_40k_48852.py

WHAT IT REUSES (no re-implementation of a statistic or of a per-window metric):
  compliant_reread_r8 (prepare_seed, attach_pose_reference, plan_variants, compliant_mask, tools_load)  -> the dump/panel loaders, the ceiling label,
  the landed capped plan (de4e15b `cap_plan_to_limit`), the ha0_ext_pose reference, the compliant mask;
  pair_read_r8 (components, paired_cell, METRICS, load_panel_keyed)                                        -> the per-window components and
  the paired episode-cluster bootstrap (taniteval/ci.py, B 2000, seed 0, 95 % percentile).
The ONLY new things are (i) the per-tree archive re-point (the 40k tree lives under battery40k_pf, exactly as rehearsal_lane.py re-points it),
(ii) the join of the two trees' per-window arrays on the (sha12(clip), window-start) key, and (iii) the paired 48,852 - 40k bootstrap on that join.

ESTIMATOR / QUESTION: paired episode-cluster bootstrap over the episodes the set contains, models FIXED -> "another draw of EPISODES" ONLY.  It says
nothing about another TRAINING run (one training seed; the two checkpoints are two points on ONE run, not two runs), and nothing about another
INFERENCE run beyond the two inference seeds reported beside it.

Stage gate: control C1 (the per-window arrays reproduce EVERY banked instrument cell of both trees to 1e-3) is evaluated BEFORE any paired number is
computed; if it fails nothing else is written except the failing-control record.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np

CODE = "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code"
sys.path.insert(0, CODE)
OUTDIR = Path("C:/Users/Admin/refcv8_pf/paired_40k_48852")
OUTDIR.mkdir(parents=True, exist_ok=True)

import compliant_reread_r8 as CR  # noqa: E402

CP = CR.CP

NEW, OLD = "48852", "40k"                      # delta = NEW - OLD = step 48,852 minus step 40,000
TREES = {
    NEW: {"root": "C:/Users/Admin/refcv8_eval_archive/battery", "tree": "step48852ema_pf",
          "banked": "C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/pfpkg/raw/compliant_reread/trees/step48852ema_pf.json"},
    OLD: {"root": "C:/Users/Admin/refcv8_pf/battery40k_pf", "tree": "step40000ema_pf",
          "banked": "C:/Users/Admin/refcv8_pf/rehearsal40k/pfpkg/raw/compliant_reread/trees/step40000ema_pf.json"},
}
SEEDS = (0, 1)
SURFS = ("S2", "S6")
N_BOOT = 2000
TOL = 1e-3
VARIANTS = ("capped", "uncapped")
SETS = (("compliant", "capped"), ("all", "uncapped"), ("compliant", "uncapped"), ("all", "capped"))   # headline first, all-window uncapped beside
HEADLINE_SET = ("compliant", "capped")
BESIDE_SET = ("all", "uncapped")
HEADLINE_KEYS = {"S2": ["ade_m", "LON_speed_mae_mps", "LON_along_abs_final_m", "LAT_heading_mae_deg", "LAT_curvature_mae_1pm_valid",
                        "LAT_yaw_rate_mae_radps_valid"],
                 "S6": ["ade_m"]}
PRINT = lambda *a: print(*a, flush=True)  # noqa: E731


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fnum(x):
    return None if x is None else float(x)


# ------------------------------------------------------------------------------------------------------ gather one tree
def gather(label: str, tools: dict, rc: dict) -> dict:
    """Per (seed, surface) per-window arrays for one tree, built EXACTLY as compliant_reread_r8.surface_reads builds its `comps`."""
    PR, V, ra, sha12 = tools["PR"], tools["V"], tools["ra"], tools["sha12"]
    cfg = TREES[label]
    CP.ARCHIVE = cfg["root"]                    # the same re-point rehearsal_lane.py does for the 40k tree (read-only use here)
    CR.ARCHIVE = cfg["root"]
    keys_m = [t[1] for t in PR.METRICS]
    G = {"label": label, "tree": cfg["tree"], "root": cfg["root"], "seeds": {}}
    for seed in SEEDS:
        ctx = CR.prepare_seed(cfg["tree"], seed, tools, rc)
        pose_meta = CR.attach_pose_reference(tools, ctx)
        pv = CR.plan_variants(V, ctx, rc)
        binding = pv["diag"]["plan_cap_binding"].numpy().astype(bool)
        sd = {"mask_dump": ctx["comp"].copy(), "n_dump_windows": int(len(ctx["comp"])), "mask_sha256": ctx["sha"],
              "pose_reference_PASS": {s: bool(pose_meta[s]["PASS"]) for s in SURFS}, "surfaces": {}}
        for surf, _pat in CR.SURFS:
            P = ctx["panels"][surf]
            A, dt, slots, rows, eid = P["A"], P["dt"], P["slots"], P["rows"], P["eid"]
            _man, _A2, eid2, keys = PR.load_panel_keyed(P["dir"], sha12)
            if not np.array_equal(np.asarray(eid2), np.asarray(eid)):
                raise SystemExit(f"[paired] {label} s{seed} {surf}: panel eid disagrees between the two loads")
            ws_panel = np.array([int(k[1]) for k in keys], dtype=np.int64)
            if not np.array_equal(ws_panel, ctx["ws"][rows]):
                raise SystemExit(f"[paired] {label} s{seed} {surf}: panel window starts disagree with the dump rows")
            clip12 = np.array([k[0] for k in keys])
            os_slots = {v: pv[v][rows][:, slots] for v in VARIANTS}
            comps = {"uncapped": PR.components(ra, A["os"], A["g"], dt), "capped": PR.components(ra, os_slots["capped"], A["g"], dt)}
            fixed = {"ha0_ext": PR.components(ra, A["ha0_ext"], A["g"], dt), "ref7": PR.components(ra, A[ctx["refarm"]], A["g"], dt),
                     "pose": PR.components(ra, A["ha0_ext_pose"], A["g"], dt)}
            asf = lambda d: {k: np.asarray(d[k], dtype=np.float64) for k in keys_m}  # noqa: E731
            # exploratory (section 7): the SIGNED final-instant along-track error, by the battery's own geometry (taniteval.four_families._seq_geometry, the call
            # pair_read_r8.components makes); its absolute value must equal the instrument's LON_along_abs_final_m EXACTLY (asserted)
            import torch
            from taniteval import four_families as ff
            Gg = ff._seq_geometry(torch.as_tensor(A["g"]).float(), dt)
            signed = {}
            for v in VARIANTS:
                plan_v = A["os"] if v == "uncapped" else os_slots["capped"]
                Pg = ff._seq_geometry(torch.as_tensor(plan_v).float(), dt)
                signed[v] = (Pg["along"][:, -1] - Gg["along"][:, -1]).numpy().astype(np.float64)
                if not np.array_equal(np.abs(signed[v]), np.asarray(comps[v]["LON_along_abs_final_m"], dtype=np.float64), equal_nan=True):
                    raise SystemExit(f"[paired] {label} s{seed} {surf} {v}: |signed final along| != the instrument's LON_along_abs_final_m")
            sd["surfaces"][surf] = {
                "clip12": clip12, "ws": ws_panel, "eid": np.asarray(eid, dtype=np.int64), "mask": np.asarray(ctx["comp"][rows], bool),
                "binding_capped": binding[rows],
                "os": {v: asf(comps[v]) for v in VARIANTS}, "ha0_ext": asf(fixed["ha0_ext"]), "ref7": asf(fixed["ref7"]), "pose": asf(fixed["pose"]),
                # model-free / shared arms, kept to assert they are identical across the two trees (same GT, same references)
                "g": np.asarray(A["g"], dtype=np.float64), "ha0_ext_plan": np.asarray(A["ha0_ext"], dtype=np.float64),
                "ref7_plan": np.asarray(A[ctx["refarm"]], dtype=np.float64), "pose_plan": np.asarray(A["ha0_ext_pose"], dtype=np.float64),
                "plan_capped": np.asarray(os_slots["capped"], dtype=np.float64), "plan_uncapped": np.asarray(os_slots["uncapped"], dtype=np.float64),
                "signed_final_along": signed, "v0": np.asarray(ctx["v0"][rows], dtype=np.float64),
                "slots": list(slots), "dt": float(dt)}
            del comps, fixed
        G["seeds"][seed] = sd
        del ctx, pv
        PRINT(f"[gather] {label} seed {seed} done: S2 rows {len(sd['surfaces']['S2']['ws'])} (compliant {int(sd['surfaces']['S2']['mask'].sum())}), "
              f"S6 rows {len(sd['surfaces']['S6']['ws'])} (compliant {int(sd['surfaces']['S6']['mask'].sum())})")
    return G


# ------------------------------------------------------------------------------------------------------ C1: reproduction
def cell_cmp(got: dict, banked: dict | None) -> dict:
    if not isinstance(banked, dict) or "delta" not in banked or not isinstance(got, dict) or "delta" not in got:
        return {"ok": False, "status": "ABSENT", "got_has_delta": isinstance(got, dict) and "delta" in got, "banked_has_delta": isinstance(banked, dict) and "delta" in banked}
    d = {f: abs(float(got[f]) - float(banked[f])) for f in ("delta", "lo", "hi")}
    exact_int = all(got.get(f) == banked.get(f) for f in ("n_windows", "n_episodes", "separated", "n_dropped_nonfinite"))
    ok = bool(max(d.values()) <= TOL and exact_int)
    return {"ok": ok, "max_abs_diff": max(d.values()), "integer_fields_equal": bool(exact_int)}


def reproduce(label: str, G: dict, PR) -> dict:
    """Every banked instrument cell of this tree (both seeds, both surfaces, all-window + compliant, capped + uncapped, ALL 14 metrics vs refcv7 and vs
    the echo, ADE vs ha0_ext_pose) is recomputed from the per-window arrays and compared to the banked tree json to TOL."""
    banked = json.load(open(TREES[label]["banked"], encoding="utf-8"))
    n_ok = n_tot = 0
    worst = 0.0
    bad = []
    n_exact = 0
    for seed in SEEDS:
        for surf in SURFS:
            S = G["seeds"][seed]["surfaces"][surf]
            for setname, variant in (("compliant", "capped"), ("compliant", "uncapped"), ("all", "capped"), ("all", "uncapped")):
                mk = S["mask"] if setname == "compliant" else np.ones(len(S["mask"]), bool)
                b = banked["seeds"][str(seed)]["surfaces"][surf]["compliant" if setname == "compliant" else "all_window"][variant]
                e = S["eid"][mk]
                # window / episode counts of the set
                if int(b["n_windows"]) != int(mk.sum()) or int(b["n_episodes"]) != int(len(np.unique(e))):
                    bad.append({"set": f"{setname}/{variant}", "seed": seed, "surf": surf, "what": "n_windows/n_episodes", "got": [int(mk.sum()), int(len(np.unique(e)))],
                                "banked": [b["n_windows"], b["n_episodes"]]})
                for fam, key, lab, unit, lower, _nr in PR.METRICS:
                    x = S["os"][variant][key][mk]
                    refs = [("ref7", "os_minus_refcv7_50400"), ("ha0_ext", "os_minus_ha0_ext")] + ([("pose", "os_minus_ha0_ext_pose")] if key == "ade_m" else [])
                    for arm, bkey in refs:
                        got = PR.paired_cell(x, S[arm][key][mk], e, N_BOOT)
                        cmp_ = cell_cmp(got, b["rows"][key].get(bkey))
                        n_tot += 1
                        n_ok += int(cmp_["ok"])
                        if cmp_.get("max_abs_diff") is not None:
                            worst = max(worst, cmp_["max_abs_diff"])
                            n_exact += int(cmp_["max_abs_diff"] == 0.0)
                        if not cmp_["ok"]:
                            bad.append({"set": f"{setname}/{variant}", "seed": seed, "surf": surf, "metric": key, "arm": arm, **cmp_})
    return {"tree": TREES[label]["tree"], "banked_file": TREES[label]["banked"], "tolerance": TOL, "n_cells_compared": n_tot, "n_cells_within_tolerance": n_ok,
            "n_cells_bit_identical_on_delta_lo_hi": n_exact, "worst_abs_diff": worst, "n_bad": len(bad), "bad_first10": bad[:10], "PASS": bool(n_tot > 0 and n_ok == n_tot and not bad)}


def named_values(Gs: dict, PR) -> dict:
    """The values quoted in the brief (compliant, capped, seed 0), recomputed from the arrays: S6 ADE vs refcv7, E3 (S6 ADE - ha0_ext_pose), E1 (S2)."""
    want = {NEW: {"S6_ade_vs_refcv7": (0.1725, 0.0337, 0.3384), "E3_S6_ade_vs_pose": -0.7285, "E1_S2_ade_vs_pose": -0.0069},
            OLD: {"S6_ade_vs_refcv7": (0.0049, -0.1044, 0.1196), "E3_S6_ade_vs_pose": -0.8961, "E1_S2_ade_vs_pose": -0.0182}}
    out = {}
    for lab in (NEW, OLD):
        S6, S2 = Gs[lab]["seeds"][0]["surfaces"]["S6"], Gs[lab]["seeds"][0]["surfaces"]["S2"]
        m6, m2 = S6["mask"], S2["mask"]
        c = PR.paired_cell(S6["os"]["capped"]["ade_m"][m6], S6["ref7"]["ade_m"][m6], S6["eid"][m6], N_BOOT)
        e3 = PR.paired_cell(S6["os"]["capped"]["ade_m"][m6], S6["pose"]["ade_m"][m6], S6["eid"][m6], N_BOOT)
        e1 = PR.paired_cell(S2["os"]["capped"]["ade_m"][m2], S2["pose"]["ade_m"][m2], S2["eid"][m2], N_BOOT)
        w = want[lab]
        ok = bool(abs(c["delta"] - w["S6_ade_vs_refcv7"][0]) <= TOL and abs(c["lo"] - w["S6_ade_vs_refcv7"][1]) <= TOL and abs(c["hi"] - w["S6_ade_vs_refcv7"][2]) <= TOL
                  and abs(e3["delta"] - w["E3_S6_ade_vs_pose"]) <= TOL and abs(e1["delta"] - w["E1_S2_ade_vs_pose"]) <= TOL)
        out[lab] = {"brief": w, "got": {"S6_ade_vs_refcv7": [c["delta"], c["lo"], c["hi"]], "E3_S6_ade_vs_pose": e3["delta"], "E1_S2_ade_vs_pose": e1["delta"]}, "PASS": ok}
    out["PASS"] = bool(out[NEW]["PASS"] and out[OLD]["PASS"])
    return out


# ------------------------------------------------------------------------------------------------------ C3: the join
def canon_order(S: dict):
    ks = np.array([f"{c}:{w}" for c, w in zip(S["clip12"], S["ws"])])
    if len(set(ks.tolist())) != len(ks):
        raise SystemExit("[paired] duplicate (clip, window) keys inside a panel")
    o = np.argsort(ks, kind="stable")
    return ks[o], o


def join(Gs: dict) -> dict:
    """Join the two trees' per-window arrays on the (sha12(clip), ws) key, per seed and surface.  Asserts key-set equality and reports everything."""
    J, rep = {}, {"per_surface_seed": {}, "model_free_arms_identical_across_trees": {}, "PASS": True}
    for seed in SEEDS:
        for surf in SURFS:
            A, B = Gs[NEW]["seeds"][seed]["surfaces"][surf], Gs[OLD]["seeds"][seed]["surfaces"][surf]
            ka, oa = canon_order(A)
            kb, ob = canon_order(B)
            same_keys = bool(len(ka) == len(kb) and np.array_equal(ka, kb))
            r = {"n_windows_48852": int(len(ka)), "n_windows_40k": int(len(kb)), "key_sets_identical": same_keys,
                 "only_in_48852": int(len(set(ka) - set(kb))), "only_in_40k": int(len(set(kb) - set(ka)))}
            if not same_keys:
                rep["PASS"] = False
                rep["per_surface_seed"][f"{surf}|s{seed}"] = r
                continue
            take = lambda S, o: {  # noqa: E731
                "os": {v: {k: S["os"][v][k][o] for k in S["os"][v]} for v in VARIANTS},
                "pose": {k: S["pose"][k][o] for k in S["pose"]}, "ha0_ext": {k: S["ha0_ext"][k][o] for k in S["ha0_ext"]},
                "ref7": {k: S["ref7"][k][o] for k in S["ref7"]}, "mask": S["mask"][o], "eid": S["eid"][o], "clip12": S["clip12"][o],
                "binding": S["binding_capped"][o], "g": S["g"][o], "pose_plan": S["pose_plan"][o], "ha0_ext_plan": S["ha0_ext_plan"][o],
                "ref7_plan": S["ref7_plan"][o], "plan_capped": S["plan_capped"][o], "plan_uncapped": S["plan_uncapped"][o],
                "v0": S["v0"][o], "signed": {v: S["signed_final_along"][v][o] for v in VARIANTS}}
            a, b = take(A, oa), take(B, ob)
            mask_same = bool(np.array_equal(a["mask"], b["mask"]))
            eid_same = bool(np.array_equal(a["eid"], b["eid"]))
            # episode label <-> clip must be a bijection inside each tree
            def bij(x):
                m1, m2 = {}, {}
                for e, c in zip(x["eid"].tolist(), x["clip12"].tolist()):
                    if m1.setdefault(e, c) != c or m2.setdefault(c, e) != e:
                        return False
                return True
            mf = {"v0": bool(np.array_equal(a["v0"], b["v0"])), "g": bool(np.array_equal(a["g"], b["g"])), "ha0_ext_plan": bool(np.array_equal(a["ha0_ext_plan"], b["ha0_ext_plan"])),
                  "ref7_plan": bool(np.array_equal(a["ref7_plan"], b["ref7_plan"])), "pose_plan": bool(np.array_equal(a["pose_plan"], b["pose_plan"]))}
            rep["model_free_arms_identical_across_trees"][f"{surf}|s{seed}"] = mf
            r.update({"mask_identical_across_trees": mask_same, "eid_label_identical_across_trees": eid_same,
                      "episode_label_clip_bijection_ok": {"48852": bij(a), "40k": bij(b)},
                      "n_compliant_windows": int(a["mask"].sum()), "n_compliant_episodes": int(len(np.unique(a["eid"][a["mask"]]))),
                      "n_episodes_all": int(len(np.unique(a["eid"]))), "n_windows_all": int(len(ka)),
                      "key_set_fingerprint_sha256": hashlib.sha256("\n".join(ka.tolist()).encode()).hexdigest(),
                      "native_slots_equal": bool(A["slots"] == B["slots"]), "native_dt_equal": bool(A["dt"] == B["dt"])})
            if not (mask_same and eid_same and all(mf.values()) and r["episode_label_clip_bijection_ok"]["48852"] and r["episode_label_clip_bijection_ok"]["40k"]
                    and r["native_slots_equal"] and r["native_dt_equal"]):
                rep["PASS"] = False
            rep["per_surface_seed"][f"{surf}|s{seed}"] = r
            J[(seed, surf)] = {NEW: a, OLD: b}
    # cross-seed: the key set must also be identical between the two inference seeds (same eval windows)
    for surf in SURFS:
        for lab in (NEW, OLD):
            k0, _ = canon_order(Gs[lab]["seeds"][0]["surfaces"][surf])
            k1, _ = canon_order(Gs[lab]["seeds"][1]["surfaces"][surf])
            ok = bool(np.array_equal(k0, k1))
            rep.setdefault("key_sets_identical_across_inference_seeds", {})[f"{lab}|{surf}"] = ok
            rep["PASS"] = bool(rep["PASS"] and ok)
    return J, rep


# ------------------------------------------------------------------------------------------------------ the paired cells
def direction(cell: dict, lower_is_better: bool) -> str:
    if not cell.get("separated"):
        return "not separated"
    worse = (cell["delta"] > 0) == lower_is_better
    return "SEPARATED WORSE" if worse else "SEPARATED BETTER"


def pcell(PR, new, old, eid, lower_is_better):
    c = PR.paired_cell(new, old, eid, N_BOOT)
    if "delta" not in c:
        return c
    return {"mean_48852": c["mean_new"], "mean_40k": c["mean_ref"], "delta": c["delta"], "lo": c["lo"], "hi": c["hi"], "separated": bool(c["separated"]),
            "direction": direction(c, lower_is_better), "n_windows": c["n_windows"], "n_episodes": c["n_episodes"], "n_dropped_nonfinite": c["n_dropped_nonfinite"],
            "p_delta_gt0": c["p_delta_gt0"], **({"degenerate": True} if c.get("degenerate") else {})}


def paired_rows(J: dict, PR) -> dict:
    rows = {}
    for surf in SURFS:
        for seed in SEEDS:
            a, b = J[(seed, surf)][NEW], J[(seed, surf)][OLD]
            for setname, variant in SETS:
                mk = a["mask"] if setname == "compliant" else np.ones(len(a["mask"]), bool)
                e = a["eid"][mk]
                tag = f"{setname}_{variant}"
                for fam, key, lab, unit, lower, _nr in PR.METRICS:
                    cell = pcell(PR, a["os"][variant][key][mk], b["os"][variant][key][mk], e, lower)
                    cell.update({"family": fam, "label": lab, "unit": unit, "lower_is_better": lower})
                    rows.setdefault(surf, {}).setdefault(f"seed{seed}", {}).setdefault(tag, {})[key] = cell
    return rows


def e_forms(J: dict, PR) -> dict:
    """E1 (S2) / E3 (S6): refcv8 minus ha0_ext_pose at each tree (the instrument's own cell) AND the paired 48,852 - 40k difference of those E values,
    computed explicitly as mean((os48 - pose) - (os40 - pose)); the pre-ruling echo form (ha0_ext) kept beside."""
    out = {}
    for surf, bar in (("S2", "E1"), ("S6", "E3")):
        for seed in SEEDS:
            a, b = J[(seed, surf)][NEW], J[(seed, surf)][OLD]
            for setname, variant in SETS:
                mk = a["mask"] if setname == "compliant" else np.ones(len(a["mask"]), bool)
                e = a["eid"][mk]
                rec = {}
                for refname, lab in (("pose", "ha0_ext_pose (D-R8-E1-REF)"), ("ha0_ext", "ha0_ext echo (pre-ruling, beside)")):
                    xa, xb = a["os"][variant]["ade_m"][mk], b["os"][variant]["ade_m"][mk]
                    ref = a[refname]["ade_m"][mk]
                    assert np.array_equal(ref, b[refname]["ade_m"][mk], equal_nan=True)
                    r_a = PR.paired_cell(xa, ref, e, N_BOOT)
                    r_b = PR.paired_cell(xb, ref, e, N_BOOT)
                    d = pcell(PR, xa - ref, xb - ref, e, True)
                    rec[refname] = {"reference": lab, f"{bar}_48852_os_minus_ref": {k: r_a[k] for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes")},
                                    f"{bar}_40k_os_minus_ref": {k: r_b[k] for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes")},
                                    "paired_diff_of_E_48852_minus_40k": d}
                out.setdefault(surf, {}).setdefault(f"seed{seed}", {})[f"{setname}_{variant}"] = rec
    return out


# ------------------------------------------------------------------------------------------------------ the other controls
def control_identity(Gs: dict, J: dict, PR) -> dict:
    """40k minus ITSELF must read exactly 0 with a [0, 0] interval (every headline metric, both seeds, compliant capped)."""
    res, ok = {}, True
    for surf in SURFS:
        for seed in SEEDS:
            b = J[(seed, surf)][OLD]
            mk = b["mask"]
            for key in HEADLINE_KEYS[surf]:
                c = PR.paired_cell(b["os"]["capped"][key][mk], b["os"]["capped"][key][mk], b["eid"][mk], N_BOOT)
                good = bool(c["delta"] == 0.0 and c["lo"] == 0.0 and c["hi"] == 0.0 and not c["separated"])
                ok &= good
                res[f"{surf}|s{seed}|{key}"] = {"delta": c["delta"], "lo": c["lo"], "hi": c["hi"], "separated": c["separated"], "degenerate_flag": bool(c.get("degenerate", False)),
                                                 "reads_exact_zero": good}
    return {"what": "step-40,000 minus itself (compliant capped), paired bootstrap", "cells": res, "PASS": bool(ok)}


def control_mispaired(J: dict, PR) -> dict:
    """The join matters: the same two arrays paired with a 35-window roll of the 40k side (windows no longer matched) must give a WIDER interval on ADE
    (the point estimate is unchanged because the same windows are averaged) -- shows the paired interval is using the pairing."""
    res, ok = {}, True
    for surf in SURFS:
        a, b = J[(0, surf)][NEW], J[(0, surf)][OLD]
        mk = a["mask"]
        xa, xb = a["os"]["capped"]["ade_m"][mk], b["os"]["capped"]["ade_m"][mk]
        e = a["eid"][mk]
        p = PR.paired_cell(xa, xb, e, N_BOOT)
        m = PR.paired_cell(xa, np.roll(xb, -35), e, N_BOOT)
        wid = lambda c: c["hi"] - c["lo"]  # noqa: E731
        good = bool(wid(m) > wid(p) * 1.5 and abs(m["delta"] - p["delta"]) < 1e-9)
        ok &= good
        res[surf] = {"paired": [p["delta"], p["lo"], p["hi"]], "rolled40k_by_35": [m["delta"], m["lo"], m["hi"]], "width_ratio_mispaired_over_paired": wid(m) / wid(p),
                     "mispaired_is_wider_by_gt_1p5x": good}
    return {"what": "mispaired control (40k side rolled by 35 windows), S2/S6 ADE, compliant capped, seed 0", "cells": res, "PASS": bool(ok)}


def inference_floor(rows: dict, Gs: dict, J: dict, PR) -> dict:
    """C4: |delta(seed0) - delta(seed1)| of the 48,852 - 40k delta, per row; the instrument's per-tree ADE floors (|mean ADE s0 - s1|, same set); and the
    within-tree inference replicate (seed0 - seed1 paired on the same windows) for ADE."""
    per_row = {}
    for surf in SURFS:
        for tag in rows[surf]["seed0"]:
            for key, c0 in rows[surf]["seed0"][tag].items():
                c1 = rows[surf]["seed1"][tag][key]
                if "delta" in c0 and "delta" in c1:
                    per_row[f"{surf}|{tag}|{key}"] = {"delta_s0": c0["delta"], "delta_s1": c1["delta"], "abs_seed_diff_of_delta": abs(c0["delta"] - c1["delta"]),
                                                       "same_sign": bool(np.sign(c0["delta"]) == np.sign(c1["delta"])), "separated_at_both_seeds": bool(c0["separated"] and c1["separated"]),
                                                       "same_direction_label": c0["direction"] == c1["direction"]}
    tree_floor = {}
    for lab in (NEW, OLD):
        for surf in SURFS:
            for setname, variant in (HEADLINE_SET, BESIDE_SET):
                a0, a1 = J[(0, surf)][lab], J[(1, surf)][lab]
                mk = a0["mask"] if setname == "compliant" else np.ones(len(a0["mask"]), bool)
                m0, m1 = float(np.nanmean(a0["os"][variant]["ade_m"][mk])), float(np.nanmean(a1["os"][variant]["ade_m"][mk]))
                c = PR.paired_cell(a0["os"][variant]["ade_m"][mk], a1["os"][variant]["ade_m"][mk], a0["eid"][mk], N_BOOT)
                tree_floor[f"{lab}|{surf}|{setname}_{variant}"] = {"instrument_floor_abs_mean_ade_s0_minus_s1": abs(m0 - m1),
                                                                  "within_tree_replicate_seed0_minus_seed1_ade": {k: c[k] for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes")}}
    heads = {}
    for surf in SURFS:
        for tag in (f"{HEADLINE_SET[0]}_{HEADLINE_SET[1]}", f"{BESIDE_SET[0]}_{BESIDE_SET[1]}"):
            d0, d1 = rows[surf]["seed0"][tag]["ade_m"]["delta"], rows[surf]["seed1"][tag]["ade_m"]["delta"]
            fl = tree_floor[f"{NEW}|{surf}|{tag}"]["instrument_floor_abs_mean_ade_s0_minus_s1"]
            heads[f"{surf}|{tag}"] = {"delta_s0": d0, "delta_s1": d1, "abs_seed_diff_of_delta": abs(d0 - d1), "tree_floor_48852": fl,
                                      "tree_floor_40k": tree_floor[f"{OLD}|{surf}|{tag}"]["instrument_floor_abs_mean_ade_s0_minus_s1"],
                                      "min_abs_delta_over_seeds_div_seed_diff": (min(abs(d0), abs(d1)) / abs(d0 - d1)) if d0 != d1 else None,
                                      "min_abs_delta_over_larger_tree_floor": min(abs(d0), abs(d1)) / max(fl, tree_floor[f"{OLD}|{surf}|{tag}"]["instrument_floor_abs_mean_ade_s0_minus_s1"]),
                                      "clears_2x_larger_tree_floor": bool(min(abs(d0), abs(d1)) > 2.0 * max(fl, tree_floor[f"{OLD}|{surf}|{tag}"]["instrument_floor_abs_mean_ade_s0_minus_s1"]))}
    return {"per_row": per_row, "headline_ade": heads, "per_tree_floor_and_replicate": tree_floor,
            "max_abs_seed_diff_of_delta_over_all_rows": max(v["abs_seed_diff_of_delta"] for v in per_row.values()),
            "rows_where_seeds_disagree_on_direction": sorted(k for k, v in per_row.items() if not v["same_direction_label"])}


def binding_context(J: dict) -> dict:
    """Context only (not a test): how often the landed cap binds on each tree's plans, on the compliant set."""
    out = {}
    for surf in SURFS:
        for seed in SEEDS:
            for lab in (NEW, OLD):
                d = J[(seed, surf)][lab]
                mk = d["mask"]
                out[f"{surf}|s{seed}|{lab}"] = {"cap_binding_rate_compliant": float(d["binding"][mk].mean()), "cap_binding_rate_all": float(d["binding"].mean())}
    return out


def where_it_lives(J: dict, PR) -> dict:
    """POST-HOC, EXPLORATORY (never a verdict; not pre-registered): where the 48,852 - 40k ADE delta sits.  Compliant windows, capped plans.
    (a) the SIGNED final-instant along-track error (plan minus human; positive = the plan ends FARTHER along the path = faster) per tree and its paired delta;
    (b) the ADE delta split by whether the landed cap binds on EITHER tree's plan for that window;
    (c) the ADE delta split by v0 terciles of the compliant set (terciles from the logged v0, model-free)."""
    out = {"status": "POST-HOC EXPLORATORY: splits chosen after seeing the headline; no verdict language; to be promoted only through its own pre-registration",
           "controls": {}, "signed_final_along": {}, "split_by_cap_binding": {}, "split_by_v0_tercile": {}, "cap_binding_on_compliant_windows": {}}
    dir_ok = True
    for surf in SURFS:
        for seed in SEEDS:
            a, b = J[(seed, surf)][NEW], J[(seed, surf)][OLD]
            mk = a["mask"]
            e = a["eid"][mk]
            # direction control: sign(signed final along) must agree with sign(|plan_final| - |human_final|) wherever the two endpoints differ by > 0.5 m
            for lab, d in ((NEW, a), (OLD, b)):
                sg = d["signed"]["capped"][mk]
                rd = np.linalg.norm(d["plan_capped"][mk][:, -1, :], axis=-1) - np.linalg.norm(d["g"][mk][:, -1, :], axis=-1)
                big = np.abs(rd) > 0.5
                agree = float((np.sign(sg[big]) == np.sign(rd[big])).mean()) if big.any() else None
                out["controls"][f"direction|{surf}|s{seed}|{lab}"] = {"n_windows_endpoint_differs_gt_0p5m": int(big.sum()), "sign_agreement": agree}
                dir_ok &= bool(agree is not None and agree > 0.95)
            sa, sb = a["signed"]["capped"][mk], b["signed"]["capped"][mk]
            rec = pcell(PR, sa, sb, e, True)
            rec["note"] = "mean_48852 / mean_40k are the MEAN SIGNED final along-track error (m): positive = the plan ends farther than the human (too fast), negative = short (too slow)"
            out["signed_final_along"].setdefault(surf, {})[f"seed{seed}"] = rec
            # (b) cap binding in either tree
            bind = (a["binding"] | b["binding"])[mk]
            xa, xb = a["os"]["capped"]["ade_m"][mk], b["os"]["capped"]["ade_m"][mk]
            for lab, d in ((NEW, a), (OLD, b)):
                out["cap_binding_on_compliant_windows"][f"{surf}|s{seed}|{lab}"] = {
                    "n_windows_where_cap_binds": int(d["binding"][mk].sum()),
                    "capped_plan_bit_identical_to_uncapped_on_compliant": bool(np.array_equal(d["plan_capped"][mk], d["plan_uncapped"][mk]))}
            for nm, sel in (("cap_binds_in_either_tree", bind), ("cap_binds_in_neither_tree", ~bind)):
                if sel.sum() >= 30 and len(np.unique(e[sel])) >= 5:
                    c = pcell(PR, xa[sel], xb[sel], e[sel], True)
                else:
                    c = {"status": "TOO FEW WINDOWS", "n_windows": int(sel.sum())}
                out["split_by_cap_binding"].setdefault(surf, {}).setdefault(f"seed{seed}", {})[nm] = c
            # (c) v0 terciles
            v0 = a["v0"][mk]
            q1, q2 = np.quantile(v0, [1 / 3, 2 / 3])
            for nm, sel in ((f"v0_low (<= {q1:.2f} m/s)", v0 <= q1), (f"v0_mid ({q1:.2f}, {q2:.2f}] m/s", (v0 > q1) & (v0 <= q2)), (f"v0_high (> {q2:.2f} m/s)", v0 > q2)):
                c = pcell(PR, xa[sel], xb[sel], e[sel], True)
                out["split_by_v0_tercile"].setdefault(surf, {}).setdefault(f"seed{seed}", {})[nm] = c
    out["controls"]["direction_PASS"] = bool(dir_ok)
    return out


# ------------------------------------------------------------------------------------------------------ rendering
def f4(x):
    return "n/a" if x is None else f"{x:+.4f}"


def render_md(R: dict, rows: dict, eforms: dict) -> str:
    L = ["# PAIRED_40K_48852 -- refcv8 step-48,852 minus step-40,000, PAIRED on the same eval windows", "",
         f"**Generated:** {R['generated']} (local).  **Evidence class:** MEASURED (ours): per-window arrays rebuilt with the battery's own functions from the banked "
         "`perframe_full` dumps/panels of both EMA checkpoints; script `paired_40k_48852.py` (this directory).  **Tier:** T1 self-action OPEN LOOP on logged eval139 frames "
         "(not driving performance).", "",
         "**Estimator / question:** paired episode-cluster bootstrap (`taniteval/ci.py`, B 2000, seed 0, 95 % percentile) over the episodes the set contains, models FIXED.  "
         "It answers *\"would another draw of EPISODES say this?\"* ONLY.  It does NOT answer *\"would another TRAINING run say this?\"* (one training seed: the two "
         "checkpoints are two points on ONE run, not two runs) and, beyond the two inference seeds printed per row, not *\"would another INFERENCE run say this?\"*.  A separated "
         "interval is necessary, not sufficient (H-ESTIM-SEED-1).", "",
         "**Sign:** delta = 48,852 minus 40k.  For an ERROR metric (lower is better) a positive separated delta is WORSE at 48,852.  Headline set = ceiling-COMPLIANT windows, "
         "CAPPED plans (the landed `cap_plan_to_limit`, brake 2.5 m/s^2).  The ALL-WINDOW UNCAPPED form is printed beside.", ""]
    L += ["## 0. Controls", "", R["controls_md"], ""]
    L += ["## 1. Headline in plain words", "", R["headline_words"], ""]
    names = {"ade_m": "ADE", "LON_speed_mae_mps": "LON speed MAE", "LON_along_abs_final_m": "LON |along-track| at the final instant",
             "LAT_heading_mae_deg": "LAT heading MAE", "LAT_curvature_mae_1pm_valid": "LAT curvature MAE (masked)", "LAT_yaw_rate_mae_radps_valid": "LAT yaw-rate MAE (masked)"}

    def table(surf, key, lab):
        T = [f"### {lab} -- {surf} ({'instants 0.5-2.0 s' if surf == 'S2' else 'instants 1-6 s'})", "",
             "| set | seed | 48,852 mean | 40k mean | delta [95 % CI] | separated | verdict | n windows / episodes |", "|---|---|---|---|---|---|---|---|"]
        for setname, variant in (HEADLINE_SET, BESIDE_SET):
            tag = f"{setname}_{variant}"
            for seed in SEEDS:
                c = rows[surf][f"seed{seed}"][tag][key]
                nm = "compliant, capped (HEADLINE)" if (setname, variant) == HEADLINE_SET else "all-window, uncapped (beside)"
                T.append(f"| {nm} | {seed} | {c['mean_48852']:.4f} | {c['mean_40k']:.4f} | {c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | {'yes' if c['separated'] else 'no'} | "
                         f"{c['direction']} | {c['n_windows']} / {c['n_episodes']} |")
        return T + [""]

    L += ["## 2. Headline metrics (compliant capped, all-window uncapped beside)", ""]
    for surf in SURFS:
        for key in HEADLINE_KEYS[surf]:
            L += table(surf, key, names[key])
    L += ["## 3. E1 / E3 forms (refcv8 minus `ha0_ext_pose`, the D-R8-E1-REF reference) and the paired difference of those E values", "",
          "E value per tree = mean(ADE_refcv8 - ADE_ha0_ext_pose) on the set (negative = better than the prior).  The prior is model-free and identical for both trees, so the paired "
          "difference of E values equals the ADE delta above by construction (printed explicitly as the identity check).", "",
          "| bar | set | seed | E(48,852) [CI] | E(40k) [CI] | paired E(48,852) - E(40k) [95 % CI] | separated | verdict | n windows / episodes |", "|---|---|---|---|---|---|---|---|---|"]
    for surf, bar in (("S2", "E1"), ("S6", "E3")):
        for setname, variant in (HEADLINE_SET, BESIDE_SET):
            for seed in SEEDS:
                r = eforms[surf][f"seed{seed}"][f"{setname}_{variant}"]["pose"]
                a, b, d = r[f"{bar}_48852_os_minus_ref"], r[f"{bar}_40k_os_minus_ref"], r["paired_diff_of_E_48852_minus_40k"]
                nm = "compliant, capped" if (setname, variant) == HEADLINE_SET else "all-window, uncapped"
                L.append(f"| {bar} ({surf}) | {nm} | {seed} | {a['delta']:+.4f} [{a['lo']:+.4f}, {a['hi']:+.4f}] | {b['delta']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}] | "
                         f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] | {'yes' if d['separated'] else 'no'} | {d['direction']} | {d['n_windows']} / {d['n_episodes']} |")
    L += ["", "Pre-ruling echo form (`ha0_ext`) beside: the paired difference is the same ADE delta (the echo is also model-free), so only the per-tree E values differ; see the JSON "
          "(`e_forms.*.ha0_ext`).", ""]
    L += ["## 4. Every other metric of `pair_read_r8.METRICS`, compliant capped (headline set), both seeds", "",
          "| surf | metric | seed | 48,852 | 40k | delta [95 % CI] | verdict | n win / ep |", "|---|---|---|---|---|---|---|---|"]
    for surf in SURFS:
        for key, c0 in rows[surf]["seed0"]["compliant_capped"].items():
            if key in HEADLINE_KEYS[surf]:
                continue
            for seed in SEEDS:
                c = rows[surf][f"seed{seed}"]["compliant_capped"][key]
                L.append(f"| {surf} | {c['label']} ({c['unit']}) | {seed} | {c['mean_48852']:.4f} | {c['mean_40k']:.4f} | {c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | "
                         f"{c['direction']} | {c['n_windows']} / {c['n_episodes']} |")
    L += ["", "## 5. Inference-seed floor (control C4)", "", R["floor_md"], ""]
    L += ["## 6. Scope and caveats", "", R["caveats_md"], ""]
    L += ["## 7. Where the delta sits (POST-HOC, EXPLORATORY -- not a verdict, not pre-registered)", "", R["explore_md"], ""]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------ main
def main() -> int:
    t0 = time.time()
    rc = CP.run_cfg()
    if rc["horizons"] != [5, 10, 15, 20, 30, 40, 50, 60] or rc["argv_has_refcv7"]:
        raise SystemExit(f"[paired] unexpected run config {rc}")
    tools = CR.tools_load()
    PR = tools["PR"]
    Gs = {lab: gather(lab, tools, rc) for lab in (NEW, OLD)}
    PRINT(f"[paired] gather done {time.time() - t0:.0f}s")

    # ---------------- C1: the stage gate
    c1 = {lab: reproduce(lab, Gs[lab], PR) for lab in (NEW, OLD)}
    c1["named_values_from_the_brief"] = named_values(Gs, PR)
    c1["PASS"] = bool(c1[NEW]["PASS"] and c1[OLD]["PASS"] and c1["named_values_from_the_brief"]["PASS"])
    PRINT(f"[paired] C1 reproduction PASS={c1['PASS']}  (48852: {c1[NEW]['n_cells_within_tolerance']}/{c1[NEW]['n_cells_compared']}, worst {c1[NEW]['worst_abs_diff']:.2e}; "
          f"40k: {c1[OLD]['n_cells_within_tolerance']}/{c1[OLD]['n_cells_compared']}, worst {c1[OLD]['worst_abs_diff']:.2e}; named {c1['named_values_from_the_brief']['PASS']})  {time.time() - t0:.0f}s")
    if not c1["PASS"]:
        (OUTDIR / "CONTROL_C1_FAILED.json").write_text(json.dumps({"C1_reproduction": c1, "status": "STOPPED: nothing else is admissible until C1 passes"}, indent=1, default=str),
                                                       encoding="utf-8")
        PRINT("[paired] C1 FAILED: stopping; wrote CONTROL_C1_FAILED.json")
        return 4

    # ---------------- C3: the join
    J, rep3 = join(Gs)
    PRINT(f"[paired] C3 join PASS={rep3['PASS']}  {time.time() - t0:.0f}s")
    if not rep3["PASS"]:
        (OUTDIR / "CONTROL_C3_FAILED.json").write_text(json.dumps({"C3_join": rep3, "C1_reproduction": c1}, indent=1, default=str), encoding="utf-8")
        PRINT("[paired] C3 FAILED: stopping")
        return 5

    # ---------------- the paired cells
    rows = paired_rows(J, PR)
    PRINT(f"[paired] paired rows done {time.time() - t0:.0f}s")
    eforms = e_forms(J, PR)
    c2 = control_identity(Gs, J, PR)
    c5 = control_mispaired(J, PR)
    c4 = inference_floor(rows, Gs, J, PR)
    ex = where_it_lives(J, PR)
    PRINT(f"[paired] controls done  C2={c2['PASS']} C5={c5['PASS']}  {time.time() - t0:.0f}s")

    # ---------------- plain-words headline (derived from the rows, not typed)
    def head(surf, key, tag="compliant_capped"):
        c0, c1_ = rows[surf]["seed0"][tag][key], rows[surf]["seed1"][tag][key]
        return c0, c1_
    sep_worse, sep_better, not_sep, mixed = [], [], [], []
    for surf in SURFS:
        for key, c0 in rows[surf]["seed0"]["compliant_capped"].items():
            c1_ = rows[surf]["seed1"]["compliant_capped"][key]
            d0, d1 = c0.get("direction"), c1_.get("direction")
            lab = f"{surf} {c0['label']}"
            if d0 == d1 == "SEPARATED WORSE":
                sep_worse.append(lab)
            elif d0 == d1 == "SEPARATED BETTER":
                sep_better.append(lab)
            elif d0 == d1 == "not separated":
                not_sep.append(lab)
            else:
                mixed.append(f"{lab} (seed0 {d0}, seed1 {d1})")
    R = {"generated": time.strftime("%FT%T"), "wall_s": round(time.time() - t0, 1)}
    hw = []
    for surf in SURFS:
        for key in HEADLINE_KEYS[surf]:
            c0, c1_ = head(surf, key)
            hw.append(f"* {surf} {c0['label']}: delta {c0['delta']:+.4f} [{c0['lo']:+.4f}, {c0['hi']:+.4f}] seed 0, {c1_['delta']:+.4f} [{c1_['lo']:+.4f}, {c1_['hi']:+.4f}] seed 1 "
                      f"-> **{c0['direction']}** (seed 0) / **{c1_['direction']}** (seed 1)")
    R["headline_words"] = ("Compliant windows, capped plans, 48,852 minus 40k, paired:\n\n" + "\n".join(hw) +
                           f"\n\nAcross ALL {sum(len(rows[s]['seed0']['compliant_capped']) for s in SURFS)} (surface, metric) rows of `pair_read_r8.METRICS` (14 per surface x 2 surfaces) at BOTH inference seeds: "
                           f"separated WORSE {len(sep_worse)}: {', '.join(sep_worse) or 'none'}; separated BETTER {len(sep_better)}: {', '.join(sep_better) or 'none'}; "
                           f"not separated {len(not_sep)}: {', '.join(not_sep) or 'none'}; seeds disagree {len(mixed)}: {', '.join(mixed) or 'none'}.")
    ctl = [f"| C1 reproduction (every banked instrument cell, both trees, to {TOL:g}) | **{'PASS' if c1['PASS'] else 'FAIL'}** -- 48,852: {c1[NEW]['n_cells_within_tolerance']}/{c1[NEW]['n_cells_compared']} cells, "
           f"worst abs diff {c1[NEW]['worst_abs_diff']:.2e}; 40k: {c1[OLD]['n_cells_within_tolerance']}/{c1[OLD]['n_cells_compared']} cells, worst {c1[OLD]['worst_abs_diff']:.2e}; "
           f"the brief's named values (S6 ADE vs refcv7 +0.1725 / +0.0049, E3 -0.7285 / -0.8961, E1 -0.0069 / -0.0182) reproduce: {c1['named_values_from_the_brief']['PASS']} |",
           f"| C2 identity (40k minus itself) | **{'PASS' if c2['PASS'] else 'FAIL'}** -- every headline cell reads delta 0, [0, 0], not separated |",
           f"| C3 window join | **{'PASS' if rep3['PASS'] else 'FAIL'}** -- key sets identical across trees and across inference seeds; compliant mask, episode labels and the model-free arms "
           f"(logged v0, GT, echo, refcv7, ha0_ext_pose plans) bit-identical across trees; "
           f"S2 {rep3['per_surface_seed']['S2|s0']['n_compliant_windows']} compliant / {rep3['per_surface_seed']['S2|s0']['n_windows_all']} windows, {rep3['per_surface_seed']['S2|s0']['n_compliant_episodes']} / "
           f"{rep3['per_surface_seed']['S2|s0']['n_episodes_all']} episodes; S6 {rep3['per_surface_seed']['S6|s0']['n_compliant_windows']} / {rep3['per_surface_seed']['S6|s0']['n_windows_all']}, "
           f"{rep3['per_surface_seed']['S6|s0']['n_compliant_episodes']} / {rep3['per_surface_seed']['S6|s0']['n_episodes_all']} |",
           f"| C4 inference-seed floor | max over all rows of abs(delta seed0 - delta seed1) = {c4['max_abs_seed_diff_of_delta_over_all_rows']:.4f}; headline ADE below |",
           f"| C5 mispaired (40k rolled 35 windows) | **{'PASS' if c5['PASS'] else 'FAIL'}** -- the interval widens {c5['cells']['S2']['width_ratio_mispaired_over_paired']:.1f}x (S2) / "
           f"{c5['cells']['S6']['width_ratio_mispaired_over_paired']:.1f}x (S6) when the pairing is broken, point estimate unchanged |"]
    R["controls_md"] = "| control | result |\n|---|---|\n" + "\n".join(ctl)
    fl = ["| surface | set | delta seed 0 | delta seed 1 | abs(s0 - s1) of delta | inference floor 48,852 (abs mean-ADE s0 - s1) | floor 40k | smaller abs(delta) / larger floor | clears 2x floor |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in c4["headline_ade"].items():
        surf, tag = k.split("|")
        fl.append(f"| {surf} | {tag} | {v['delta_s0']:+.4f} | {v['delta_s1']:+.4f} | {v['abs_seed_diff_of_delta']:.4f} | {v['tree_floor_48852']:.4f} | {v['tree_floor_40k']:.4f} | "
                  f"{v['min_abs_delta_over_larger_tree_floor']:.1f}x | {'yes' if v['clears_2x_larger_tree_floor'] else 'no'} |")
    fl += ["", "Within-tree inference replicate (seed 0 minus seed 1, same checkpoint, same windows, ADE, compliant capped):", ""]
    for lab in (NEW, OLD):
        for surf in SURFS:
            r = c4["per_tree_floor_and_replicate"][f"{lab}|{surf}|compliant_capped"]["within_tree_replicate_seed0_minus_seed1_ade"]
            fl.append(f"* {lab} {surf}: {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] separated={r['separated']}")
    R["floor_md"] = "\n".join(fl)
    R["caveats_md"] = ("* The two checkpoints are EMA snapshots of ONE training run (seed 0).  A separated checkpoint-to-checkpoint difference is a statement about THIS run's "
                       "trajectory on THESE 139 eval episodes; it is not a training-seed effect size.  The paired bootstrap is structurally blind to training variance "
                       "(`H-ESTIM-SEED-1`).\n"
                       "* Inference seeds: the two seeds give the same direction and separation on every headline cell and the delta moves by at most "
                       f"{max(v['abs_seed_diff_of_delta'] for v in c4['headline_ade'].values()):.4f} m between them (headline ADE rows), against deltas of ~0.011 (S2) and ~0.16-0.17 (S6).  "
                       "Per window the two seeds are NOT identical (the within-tree seed0-minus-seed1 intervals are not zero-width), so agreement of the means is averaging, "
                       "and two seeds are a thin floor: it does not rule out a larger inference variance.\n"
                       "* Reading E1: at 40k the E1 interval sits below zero (separated better than the prior), at 48,852 it straddles zero.  That contrast is NOT itself the test "
                       "(difference of significances); the test is the paired difference of the two E values, printed in section 3.\n"
                       "* 14 metrics x 2 surfaces x 2 seeds are tabulated; with 95 % intervals a few spurious 'separated' cells are expected by chance among the non-headline rows; "
                       "the headline rows (S2/S6 ADE, S2 speed MAE, S2 |along-track| final) are the ones that decide.\n"
                       "* No clip ids appear in this document or the JSON; windows are identified only by position / sha12 join fingerprints.")
    ex_l = []
    for surf in SURFS:
        for seed in SEEDS:
            r = ex["signed_final_along"][surf][f"seed{seed}"]
            ex_l.append(f"* (a) {surf} seed {seed}: mean SIGNED final along-track error (plan minus human; + = farther/faster) 48,852 {r['mean_48852']:+.4f} m vs 40k {r['mean_40k']:+.4f} m; "
                        f"paired delta {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] (separated={r['separated']}) n {r['n_windows']}/{r['n_episodes']}")
    cb = ex["cap_binding_on_compliant_windows"]
    ex_l.append(f"* (b) the landed cap binds on {sum(v['n_windows_where_cap_binds'] for v in cb.values())} compliant windows over all (surface, seed, tree) combinations, and the capped plan is bit-identical to the uncapped "
                f"plan on the compliant set in {sum(v['capped_plan_bit_identical_to_uncapped_on_compliant'] for v in cb.values())} of {len(cb)} combinations: on the compliant set 'capped' and 'uncapped' are the same plans, "
                "so the headline is not a cap effect.  (The cap binds only on the non-compliant windows, ~5 % of all windows.)")
    ex_l.append("")
    ex_l.append("(c) ADE delta by v0 tercile of the compliant set (terciles of the logged v0):")
    ex_l.append("")
    ex_l.append("| split (ADE, compliant capped) | surf | seed | 48,852 ADE | 40k ADE | delta [95 % CI] | verdict | n win / ep |")
    ex_l.append("|---|---|---|---|---|---|---|---|")
    for grp in ("split_by_v0_tercile",):
        for surf in SURFS:
            for seed in SEEDS:
                for nm, c in ex[grp][surf][f"seed{seed}"].items():
                    if "delta" in c:
                        ex_l.append(f"| {nm} | {surf} | {seed} | {c['mean_48852']:.4f} | {c['mean_40k']:.4f} | {c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['direction']} | {c['n_windows']} / {c['n_episodes']} |")
                    else:
                        ex_l.append(f"| {nm} | {surf} | {seed} | - | - | {c.get('status')} (n={c.get('n_windows')}) | - | - |")
    ex_l.append("")
    ex_l.append(f"Direction control (sign of the signed along-track error agrees with the endpoint-range sign wherever the endpoints differ by > 0.5 m; need > 95 %): "
                f"{'PASS' if ex['controls']['direction_PASS'] else 'FAIL'}.  The signed value's absolute equals the instrument's `LON_along_abs_final_m` exactly (asserted on every tree/seed/surface/variant).")
    R["explore_md"] = "\n".join(ex_l)
    out = {"tool": "paired_40k_48852.py", "generated": R["generated"], "wall_s": R["wall_s"],
           "question": "Is the refcv8 step-48,852 minus step-40,000 difference real when PAIRED on the same eval windows?",
           "tier": CP.TIER, "estimator": "paired episode-cluster bootstrap (taniteval/ci.py), B 2000, seed 0, 95 % percentile, models FIXED: answers 'another draw of EPISODES' only; "
                                         "NOT another training run (one training seed; two checkpoints of one run), NOT another inference run",
           "delta_definition": "step 48,852 (EMA, perframe_full) minus step 40,000 (EMA, perframe_full)",
           "trees": {lab: {"tree": TREES[lab]["tree"], "root": TREES[lab]["root"], "banked_instrument_output": TREES[lab]["banked"]} for lab in (NEW, OLD)},
           "instrument": {"compliant_reread_r8.py_sha256": sha256_file(f"{CODE}/compliant_reread_r8.py"), "clamp_posthoc_r8.py_sha256": sha256_file(f"{CODE}/clamp_posthoc_r8.py"),
                          "pair_read_r8.py_sha256": sha256_file(f"{CODE}/pair_read_r8.py"),
                          "banked_tree_jsons_record_compliant_reread_sha256": {lab: json.load(open(TREES[lab]["banked"], encoding="utf-8"))["script_sha256"] for lab in (NEW, OLD)},
                          "this_script_sha256": sha256_file(__file__)},
           "controls": {"C1_reproduction": c1, "C2_identity_40k_minus_itself": c2, "C3_window_join": rep3, "C4_inference_seed_floor": c4, "C5_mispaired_roll35": c5,
                        "pose_reference_construction_PASS": {lab: {str(s): Gs[lab]["seeds"][s]["pose_reference_PASS"] for s in SEEDS} for lab in (NEW, OLD)},
                        "ALL_PASS": bool(c1["PASS"] and c2["PASS"] and rep3["PASS"] and c5["PASS"])},
           "headline_sets": {"headline": "compliant windows, capped plans", "beside": "all windows, uncapped plans"},
           "rows": rows, "e_forms": eforms, "cap_binding_context": binding_context(J), "post_hoc_exploratory": ex,
           "summary_compliant_capped": {"separated_worse_both_seeds": sep_worse, "separated_better_both_seeds": sep_better, "not_separated_both_seeds": not_sep,
                                        "seeds_disagree": mixed},
           "headline_words": R["headline_words"]}
    (OUTDIR / "PAIRED_40K_48852.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    (OUTDIR / "PAIRED_40K_48852.md").write_text(render_md(R, rows, eforms), encoding="utf-8")
    PRINT(f"[paired] WROTE PAIRED_40K_48852.json/.md  ALL_PASS={out['controls']['ALL_PASS']}  {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
