"""read_4d.py -- SHIP-R8 AMENDMENT 4d reads (R1 / R2 / R3, the REPLICATED word) + the seed-0 reproduction.  Pure json / npz + the battery's own paired_cell; CPU only.
NEW file (EvalFlyWheel 2026-10-26).  Writes ONLY out_4d/READ_4D.json and out_4d/READ_4D.md (inside ship_r8_sub_fs/); reads ship_r8_sub/ (landed verdict) read-only.

    python read_4d.py            # needs out_4d_fs0/, out_4d_fs1/, out_4d_fs2/ (run_4d_read.sh 0|1|2) to be complete

REPRODUCTION (seed 0): the registered stage re-read from THIS code copy (out_4d_fs0/) vs the landed out/: every leaf of the forward blocks / bars / controls, of the verdict
clauses (+ beside), of the TAC cells, and every array of the DEPLOY8_POS packages, bit for bit.  The ONLY exclusions are named in the record (wall-clock fields, the code hashes
that differ by design, the 4d additions, output-directory path tokens).
FD / MD exactly as refcv9_l2c/code/h2cn_fdmd_diag.py: l2_common.lon_indicators(S6_pos, E, compliant) + PR.paired_cell (episode-cluster bootstrap, B 2000, seed 0).
ADE per window exactly as the battery's arm_metrics (PR.components(ra, float32 P, float32 g, dt)["ade_m"]); S2 mirrors l2_common.load_eval_pf's panel loader on dump_s{seed}.
CONTROLS (a failed control voids the reads): the recomputed E1/E3 cells of every H2CN arm equal the verdict's registered-code cells; RETIME_DEPLOY8 positions bit-equal across the
three out dirs AND the banked registered package; row alignment; an identity pairing reads exactly 0; a mispaired control widens the CI (pairing does work).
"""
from __future__ import annotations

import copy
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np

W = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub_fs")
LAND = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub")
TREE = "step48852ema_pf"
OUT = W / "out_4d"
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
sys.path.insert(0, str(W))
import l2_common as L2  # noqa: E402
import ship_r8_sub_common as C  # noqa: E402

SEEDS_INF = (0, 1)
FIT = (1, 2)
LABEL = {0: "H2CN_DEPLOY8", 1: "H2CN_FS1_DEPLOY8", 2: "H2CN_FS2_DEPLOY8"}

# named exclusions of the seed-0 reproduction (keys whose value differs by construction, or exists only on the 4d side)
SKIP_TIME = {"generated", "wall_s", "passed_epoch", "passed_iso", "first_eval_output_epoch", "first_eval_output_iso", "registration_gate_passed_epoch", "K_REG_detail"}
SKIP_CODE = {"script_sha256", "common_sha256"}
SKIP_4D = {"4d", "fit_seed", "stage_file", "stage_sha256_read", "internal_arm_key", "amendment_4d_sha256_now", "forward_amendment_4d_sha256", "registration_file", "freeze_info",
           "K_PERSURF_required"}
SKIP_PATHS = {"inputs", "inputs_sha256"}
SKIP_ALL = SKIP_TIME | SKIP_CODE | SKIP_4D | SKIP_PATHS


def jl(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p) -> str:
    return C.sha256_file(p)


def norm(x):
    if isinstance(x, str):
        return (x.replace("\\", "/").replace("/out_4d_fs0/", "/OUT/").replace("/out/", "/OUT/").replace("ship_r8_sub_fs", "ship_r8_sub"))
    return x


def exact_equal(a, b, path="", skip=()):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k in skip:
                continue
            out += [f"{path}/{k}: one side only"] if (k not in a or k not in b) else exact_equal(a[k], b[k], f"{path}/{k}", skip)
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += exact_equal(x, y, f"{path}[{i}]", skip)
        return out
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if norm(a) == norm(b) else [f"{path}: {a!r} != {b!r}"]


def leaves(x, skip=()):
    if isinstance(x, dict):
        return sum(leaves(v, skip) for k, v in x.items() if k not in skip)
    if isinstance(x, (list, tuple)):
        return sum(leaves(v, skip) for v in x)
    return 1


def cmp_json(a, b, skip=SKIP_ALL) -> dict:
    mm = exact_equal(a, b, "", skip)
    return {"n_leaves": leaves(a, skip), "n_mismatches": len(mm), "first_mismatches": mm[:8]}


def arms_dir(s):
    return W / f"out_4d_fs{s}" / "arms" / "h2cn"


def npz(s, arm, sd):
    return np.load(arms_dir(s) / arm / f"DEPLOY8_POS_{TREE}_s{sd}.npz", allow_pickle=True)


# ------------------------------------------------------------------------------------------------------------------------------ seed-0 reproduction
def repro_seed0() -> dict:
    rec = {"what": "the REGISTERED seed-0 stage re-read from this code copy (out_4d_fs0/) vs the landed ship_r8_sub/out/ -- exact leaf / bit equality",
           "exclusions_named": {"wall_clock": sorted(SKIP_TIME), "code_hashes_differ_by_design": sorted(SKIP_CODE), "4d_additions_one_side_only": sorted(SKIP_4D),
                                "path_strings_naming_run_dirs_and_their_hashes": sorted(SKIP_PATHS), "path_token_normalisation": "/out_4d_fs0/ and /out/ -> /OUT/, ship_r8_sub_fs -> ship_r8_sub"}}
    fo = jl(LAND / "out/forward" / f"FORWARD_H2CN_{TREE}.json")
    fn = jl(W / "out_4d_fs0/forward" / f"FORWARD_H2CN_{TREE}.json")
    rec["forward"] = cmp_json(fo, fn)
    rec["forward"]["h2cn_stage_sha256_equal"] = bool(fo["h2cn_stage"]["sha256"] == fn["h2cn_stage"]["sha256"] == fo["h2cn_stage"]["registered"] == fn["h2cn_stage"]["registered"])
    rec["forward"]["h2cn_stage_sha256"] = fn["h2cn_stage"]["sha256"]
    rec["forward"]["K_REG_both_true"] = bool(fo["controls"]["K_REG"] is True and fn["controls"]["K_REG"] is True)
    rec["forward"]["ALL_PASS_both_true"] = bool(fo["controls"]["ALL_PASS"] is True and fn["controls"]["ALL_PASS"] is True)
    vo = jl(LAND / "out/ship_r8/SHIP_R8_H2CN_DEPLOY8.json")
    vn = jl(W / "out_4d_fs0/ship_r8/SHIP_R8_H2CN_DEPLOY8.json")
    rec["verdict"] = {"overall": [vo["overall"], vn["overall"]], "per_clause_equal": bool(vo["per_clause"] == vn["per_clause"]),
                      **cmp_json({k: vo[k] for k in ("overall", "per_clause", "clauses", "beside", "next_step", "tier", "estimator")},
                                 {k: vn[k] for k in ("overall", "per_clause", "clauses", "beside", "next_step", "tier", "estimator")})}
    rec["tac"] = {}
    for arm in ("H2CN_DEPLOY8", "H2CN_NEAR", "RETIME_DEPLOY8"):
        ta = jl(LAND / "out/arms/h2cn" / arm / f"TAC_{TREE}.json")
        tb = jl(arms_dir(0) / arm / f"TAC_{TREE}.json")
        rec["tac"][arm] = cmp_json(ta, tb)
    rec["positions"] = {}
    n_arr = n_bad = 0
    for arm in ("H2CN_DEPLOY8", "H2CN_NEAR", "RETIME_DEPLOY8"):
        for sd in SEEDS_INF:
            za = np.load(LAND / "out/arms/h2cn" / arm / f"DEPLOY8_POS_{TREE}_s{sd}.npz", allow_pickle=True)
            zb = npz(0, arm, sd)
            ka, kb = sorted(k for k in za.files if k != "meta_json"), sorted(k for k in zb.files if k != "meta_json")
            bad = [] if ka == kb else ["KEYSET"]
            for k in ka:
                if k in kb:
                    n_arr += 1
                    x, y = za[k], zb[k]
                    if not (x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()):
                        bad.append(k)
            n_bad += len(bad)
            rec["positions"][f"{arm}|s{sd}"] = {"n_arrays": len(ka), "bit_equal": not bad, "differ": bad}
    rec["positions_summary"] = {"n_arrays_compared": n_arr, "n_arrays_differ": n_bad}
    rec["PASS"] = bool(rec["forward"]["n_mismatches"] == 0 and rec["forward"]["h2cn_stage_sha256_equal"] and rec["forward"]["K_REG_both_true"] and rec["forward"]["ALL_PASS_both_true"]
                       and rec["verdict"]["n_mismatches"] == 0 and rec["verdict"]["per_clause_equal"] and vo["overall"] == vn["overall"]
                       and all(v["n_mismatches"] == 0 for v in rec["tac"].values()) and n_bad == 0 and n_arr > 0)
    return rec


# ------------------------------------------------------------------------------------------------------------------------------ windows / per-window metrics
def load_s2(sd):
    root = L2.ARCH / L2.EVAL_TREE / f"dump_s{sd}"
    man = L2.jload(root / "manifest.json")
    files = sorted(glob.glob(str(root / "ep*.npz")))
    g, ws, clip = [], [], []
    for f in files:
        a = np.load(f)
        g.append(a["g"]); ws.append(a["ws"]); clip += [int(a["clip_index"][0])] * len(a["ws"])
    return {"g": np.concatenate(g).astype(np.float64), "ws": np.concatenate(ws).astype(np.int64), "clip": np.asarray(clip, np.int64), "dt": float(man["grid"]["dt_s"]),
            "n_files": len(files)}


def metrics(z, E6, P2, st):
    """per-window ADE on S2 / S6 (battery components), FD / MD on S6 (h2cn_fdmd_diag), for the positions package z."""
    PR, ra = st["PR"], st["ra"]
    c2 = PR.components(ra, np.asarray(z["S2_pos"], np.float32), P2["g"].astype(np.float32), P2["dt"])
    c6 = PR.components(ra, np.asarray(z["S6_pos"], np.float32), E6["g"].astype(np.float32), E6["dt"])
    fd, md, _gt, _pc = L2.lon_indicators(z["S6_pos"], E6, np.asarray(E6["compliant"], bool))
    return {"ade2": np.asarray(c2["ade_m"], np.float64), "ade6": np.asarray(c6["ade_m"], np.float64), "fd": fd, "md": md}


def ref_metrics(z, E6, P2, st):
    PR, ra = st["PR"], st["ra"]
    p2 = np.asarray(z["S2_prior8"][:, z["S2_slots"]], np.float32)
    p6 = np.asarray(z["S6_prior8"][:, z["S6_slots"]], np.float32)
    return {"ade2": np.asarray(PR.components(ra, p2, P2["g"].astype(np.float32), P2["dt"])["ade_m"], np.float64),
            "ade6": np.asarray(PR.components(ra, p6, E6["g"].astype(np.float32), E6["dt"])["ade_m"], np.float64)}


def cell(PR, a, b, eid, mask=None):
    a, b, eid = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(eid)
    if mask is not None:
        a, b, eid = a[mask], b[mask], eid[mask]
    c = PR.paired_cell(a, b, list(eid))
    out = {k: c.get(k) for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "n_dropped_nonfinite", "mean_new", "mean_ref", "status") if k in c}
    out["separated_below_0"] = bool(c.get("separated") is True and c.get("hi") is not None and c["hi"] < 0)
    out["separated_above_0"] = bool(c.get("separated") is True and c.get("lo") is not None and c["lo"] > 0)
    return out


def f4(c):
    if c is None or c.get("delta") is None:
        return "n/a"
    return f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" + (" SEP-" if c["separated_below_0"] else (" SEP+" if c["separated_above_0"] else " ns")) + f" n={c['n_windows']}"


# ------------------------------------------------------------------------------------------------------------------------------ main
def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    seal4d = C.check_amendment_4d_seal()
    for s in (0, 1, 2):
        for need in (W / f"out_4d_fs{s}/forward/FORWARD_H2CN_{TREE}.json", W / f"out_4d_fs{s}/ship_r8/SHIP_R8_{LABEL[s]}.json"):
            if not need.is_file():
                raise SystemExit(f"[read_4d] REFUSING: {need} is missing (run_4d_read.sh {s} first)")
    st = L2.stack()
    PR = st["PR"]
    res = {"tool": "read_4d.py", "amendment_4d": seal4d, "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)",
           "estimator": "paired episode-cluster bootstrap, B 2000, bootstrap seed 0 (pair_read_r8.paired_cell -> taniteval.ci) over EPISODES with the models FIXED. It answers 'another draw of "
                        "episodes' only. ONE refcv8 training seed; inference seeds 0 and 1 are never pooled; fit seeds 0/1/2 are never pooled (each read separately). R3 is the fit-seed floor made "
                        "visible, never a lever."}
    # ---- stages (sha256 of the file each read used)
    stages = {}
    for s in (1, 2):
        pt = C.stage_paths(s)["pt"]
        fwd = jl(W / f"out_4d_fs{s}/forward/FORWARD_H2CN_{TREE}.json")
        reg = C.read_seal(W / f"REGISTERED_H2CN_FS{s}_STAGE.sha256")
        stages[str(s)] = {"stage_file": str(pt), "sha256_now": sha(pt), "registered": reg, "forward_record": fwd["h2cn_stage"]["sha256"],
                          "all_equal": bool(len(reg) == 64 and sha(pt) == reg == fwd["h2cn_stage"]["sha256"] == fwd["h2cn_stage"]["registered"])}
    fwd0 = jl(W / f"out_4d_fs0/forward/FORWARD_H2CN_{TREE}.json")
    reg0 = C.read_seal(LAND / "REGISTERED_H2CN_STAGE.sha256")
    stages["0"] = {"stage_file": str(LAND / "H2CN_STAGE_FROZEN.pt"), "sha256_now": sha(LAND / "H2CN_STAGE_FROZEN.pt"), "registered": reg0, "forward_record": fwd0["h2cn_stage"]["sha256"],
                   "all_equal": bool(len(reg0) == 64 and sha(LAND / "H2CN_STAGE_FROZEN.pt") == reg0 == fwd0["h2cn_stage"]["sha256"] == fwd0["h2cn_stage"]["registered"]),
                   "note": "the REGISTERED stage (Amendment 4), read-only; H2CN_FS0_STAGE_FROZEN.pt is the repro copy and is not scored"}
    res["stages"] = stages
    # ---- reproduction of the landed seed-0 verdict
    res["repro_seed0"] = repro_seed0()
    print(f"[read_4d] seed-0 reproduction PASS={res['repro_seed0']['PASS']}", flush=True)
    # ---- R1: SHIP-R8 verdicts
    vd = {s: jl(W / f"out_4d_fs{s}/ship_r8/SHIP_R8_{LABEL[s]}.json") for s in (0, 1, 2)}
    res["R1"] = {}
    for s in (0, 1, 2):
        v = vd[s]
        res["R1"][str(s)] = {"arm": v["arm"], "overall": v["overall"], "per_clause": v["per_clause"], "stage_sha256_read": v.get("stage_sha256_read"),
                             "non_pass_clauses": {str(c["id"]): {"status": c["status"], "reason": c["reason"]} for c in v["clauses"] if c["status"] not in ("PASS", "N/A")},
                             "clause_reasons": {str(c["id"]): c["reason"] for c in v["clauses"]},
                             "E1_cells_vs_ha0_ext_pose": v["clauses"][2]["evidence"].get("cells"), "E3_cells_vs_ha0_ext_pose": v["clauses"][3]["evidence"].get("cells"),
                             "E1_separated_below_0": v["clauses"][2]["evidence"].get("separated_below_0"), "E3_separated_below_0": v["clauses"][3]["evidence"].get("separated_below_0"),
                             "RETIME_DEPLOY8_E1_cells": v["beside"]["arms"]["RETIME_DEPLOY8"]["E1"], "RETIME_DEPLOY8_E3_cells": v["beside"]["arms"]["RETIME_DEPLOY8"]["E3"],
                             "compliance_row_exceed_note": "beside (no clause reads it): see forward compliance_row"}
    # ---- per-window metrics, per inference seed
    ctl = {"alignment": {}, "retime_bit_equal": {}, "e1e3_recompute": {}, "identity_pairing": {}, "mispaired": {}}
    R2, R3 = {str(s): {} for s in FIT}, {str(s): {} for s in FIT}
    beside0 = {}
    for sd in SEEDS_INF:
        E6, P2 = L2.load_eval_pf(sd), load_s2(sd)
        banked = np.load(L2.DEPLOY_DIR / f"DEPLOY8_POS_{L2.EVAL_TREE}_s{sd}.npz", allow_pickle=True)
        z = {s: {a: npz(s, a, sd) for a in ("H2CN_DEPLOY8", "RETIME_DEPLOY8")} for s in (0, 1, 2)}
        # alignment (every package vs the battery panels)
        al = {}
        for s in (0, 1, 2):
            for a, zz in z[s].items():
                al[f"fs{s}|{a}"] = bool(np.array_equal(zz["S6_ws"], E6["ws"]) and np.array_equal(zz["S6_clip"], E6["clip"]) and np.array_equal(zz["S2_ws"], P2["ws"])
                                        and np.array_equal(zz["S2_clip"], P2["clip"]) and np.array_equal(np.asarray(zz["S6_compliant"], bool), np.asarray(E6["compliant"], bool))
                                        and np.array_equal(zz["S6_eid"], E6["eid"]))
        ctl["alignment"][f"s{sd}"] = al
        # RETIME_DEPLOY8 bit-equal across the three out dirs AND the banked registered package
        rt = {f"fs{s}_vs_banked_registered_S6_pos": bool(z[s]["RETIME_DEPLOY8"]["S6_pos"].tobytes() == banked["S6_pos"].tobytes() and z[s]["RETIME_DEPLOY8"]["S2_pos"].tobytes() == banked["S2_pos"].tobytes())
              for s in (0, 1, 2)}
        ctl["retime_bit_equal"][f"s{sd}"] = rt
        M = {s: {a: metrics(z[s][a], E6, P2, st) for a in z[s]} for s in (0, 1, 2)}
        REF = ref_metrics(z[0]["H2CN_DEPLOY8"], E6, P2, st)
        m2c, m6c = np.asarray(z[0]["H2CN_DEPLOY8"]["S2_compliant"], bool), np.asarray(E6["compliant"], bool)
        eid2, eid6 = np.asarray(z[0]["H2CN_DEPLOY8"]["S2_eid"]), np.asarray(E6["eid"])
        # CONTROL: my E1 / E3 recompute == the verdict's registered-code cell, for every H2CN arm and RETIME (delta to 4 dp, means exactly)
        rc = {}
        for s in (0, 1, 2):
            e1 = cell(PR, M[s]["H2CN_DEPLOY8"]["ade2"], REF["ade2"], eid2, m2c)
            e3 = cell(PR, M[s]["H2CN_DEPLOY8"]["ade6"], REF["ade6"], eid6, m6c)
            v1, v3 = res["R1"][str(s)]["E1_cells_vs_ha0_ext_pose"][f"s{sd}"], res["R1"][str(s)]["E3_cells_vs_ha0_ext_pose"][f"s{sd}"]
            rc[f"fs{s}"] = {"E1_mine": e1, "E1_verdict": v1, "E3_mine": e3, "E3_verdict": v3,
                            "E1_equal": bool(round(e1["delta"], 4) == v1["delta"] and abs(e1["mean_new"] - v1["mean_new"]) < 1e-9 and abs(e1["mean_ref"] - v1["mean_ref"]) < 1e-9 and e1["n_windows"] == v1["n_windows"]),
                            "E3_equal": bool(round(e3["delta"], 4) == v3["delta"] and abs(e3["mean_new"] - v3["mean_new"]) < 1e-9 and abs(e3["mean_ref"] - v3["mean_ref"]) < 1e-9 and e3["n_windows"] == v3["n_windows"])}
            r1_ = cell(PR, M[s]["RETIME_DEPLOY8"]["ade2"], REF["ade2"], eid2, m2c)
            r3_ = cell(PR, M[s]["RETIME_DEPLOY8"]["ade6"], REF["ade6"], eid6, m6c)
            rv1 = res["R1"][str(s)]["RETIME_DEPLOY8_E1_cells"][f"s{sd}"]
            rv3 = res["R1"][str(s)]["RETIME_DEPLOY8_E3_cells"][f"s{sd}"]
            rc[f"fs{s}"]["RETIME_E1_equal"] = bool(round(r1_["delta"], 4) == rv1["delta"] and abs(r1_["mean_new"] - rv1["mean_new"]) < 1e-9)
            rc[f"fs{s}"]["RETIME_E3_equal"] = bool(round(r3_["delta"], 4) == rv3["delta"] and abs(r3_["mean_new"] - rv3["mean_new"]) < 1e-9)
        ctl["e1e3_recompute"][f"s{sd}"] = rc
        # CONTROL: identity pairing reads exactly 0 with CI [0, 0]; mispaired pairing keeps the mean difference but widens the CI
        idc = cell(PR, M[1]["H2CN_DEPLOY8"]["ade6"], M[1]["H2CN_DEPLOY8"]["ade6"], eid6, m6c)
        ctl["identity_pairing"][f"s{sd}"] = {"cell": idc, "PASS": bool(idc["delta"] == 0.0 and idc["lo"] == 0.0 and idc["hi"] == 0.0)}
        a_ = cell(PR, M[1]["H2CN_DEPLOY8"]["ade6"], M[0]["H2CN_DEPLOY8"]["ade6"], eid6, m6c)
        sh = np.roll(M[0]["H2CN_DEPLOY8"]["ade6"][m6c], 7)
        mp = PR.paired_cell(M[1]["H2CN_DEPLOY8"]["ade6"][m6c], sh, list(eid6[m6c]))
        ctl["mispaired"][f"s{sd}"] = {"aligned_halfwidth": (a_["hi"] - a_["lo"]) / 2, "mispaired_halfwidth": (mp["hi"] - mp["lo"]) / 2, "aligned_delta": a_["delta"], "mispaired_delta": mp["delta"],
                                      "PASS": bool((mp["hi"] - mp["lo"]) > 1.5 * (a_["hi"] - a_["lo"]))}
        # R2 / R3 per fit seed (and seed 0 beside)
        def vs_retime(s):
            A, Rt = M[s]["H2CN_DEPLOY8"], M[s]["RETIME_DEPLOY8"]
            return {"ADE_S2_compliant": cell(PR, A["ade2"], Rt["ade2"], eid2, m2c), "ADE_S6_compliant": cell(PR, A["ade6"], Rt["ade6"], eid6, m6c),
                    "FD_S6_compliant": cell(PR, A["fd"], Rt["fd"], eid6), "MD_S6_compliant": cell(PR, A["md"], Rt["md"], eid6)}
        beside0[f"s{sd}"] = vs_retime(0)
        for s in FIT:
            R2[str(s)][f"s{sd}"] = {**vs_retime(s), "E1_vs_ha0_ext_pose": res["R1"][str(s)]["E1_cells_vs_ha0_ext_pose"][f"s{sd}"], "E3_vs_ha0_ext_pose": res["R1"][str(s)]["E3_cells_vs_ha0_ext_pose"][f"s{sd}"],
                                    "RETIME_E1": res["R1"][str(s)]["RETIME_DEPLOY8_E1_cells"][f"s{sd}"], "RETIME_E3": res["R1"][str(s)]["RETIME_DEPLOY8_E3_cells"][f"s{sd}"]}
            A, B0 = M[s]["H2CN_DEPLOY8"], M[0]["H2CN_DEPLOY8"]
            R3[str(s)][f"s{sd}"] = {"ADE_S2_compliant": cell(PR, A["ade2"], B0["ade2"], eid2, m2c), "ADE_S6_compliant": cell(PR, A["ade6"], B0["ade6"], eid6, m6c),
                                    "ADE_S2_all_windows": cell(PR, A["ade2"], B0["ade2"], eid2), "ADE_S6_all_windows": cell(PR, A["ade6"], B0["ade6"], eid6),
                                    "FD_S6_compliant": cell(PR, A["fd"], B0["fd"], eid6), "MD_S6_compliant": cell(PR, A["md"], B0["md"], eid6)}
        print(f"[read_4d] inference seed {sd} done ({time.time() - t0:.0f} s)", flush=True)
    res["R2"] = {"seed0_beside_vs_RETIME_DEPLOY8": beside0, "per_fit_seed": R2}
    res["R3_vs_H2CN_DEPLOY8_seed0"] = R3
    # ---- controls roll-up
    c_all = {"stages_registered_equal": all(v["all_equal"] for v in stages.values()),
             "alignment": all(all(d.values()) for d in ctl["alignment"].values()),
             "retime_bit_equal": all(all(d.values()) for d in ctl["retime_bit_equal"].values()),
             "e1e3_recompute_equal": all(all(v for k, v in d.items() if k.endswith("_equal")) for sd_ in ctl["e1e3_recompute"].values() for d in sd_.values()),
             "identity_pairing_zero": all(v["PASS"] for v in ctl["identity_pairing"].values()),
             "mispaired_widens_ci": all(v["PASS"] for v in ctl["mispaired"].values()),
             "repro_seed0": bool(res["repro_seed0"]["PASS"]),
             "forward_controls_ALL_PASS": all(jl(W / f"out_4d_fs{s}/forward/FORWARD_H2CN_{TREE}.json")["controls"]["ALL_PASS"] is True for s in (0, 1, 2)),
             "K_RETIME_identity_each_fit_seed": all(jl(W / f"out_4d_fs{s}/forward/FORWARD_H2CN_{TREE}.json")["controls"]["K_RETIME_identity_vs_registered_forward_deploy8"] is True for s in (0, 1, 2)),
             "RETIME_TAC_identity_each_fit_seed": all(jl(arms_dir(s) / "RETIME_DEPLOY8" / f"TAC_RECORD_{TREE}.json").get("identity_vs_registered_TAC", {}).get("pass") is True for s in (0, 1, 2)),
             "K_DISJ_each_fit_seed": all(jl(W / f"out_4d_fs{s}/forward/FORWARD_H2CN_{TREE}.json")["controls"]["K_DISJ"] is True for s in (0, 1, 2)),
             "K_REG_each_fit_seed": all(jl(W / f"out_4d_fs{s}/forward/FORWARD_H2CN_{TREE}.json")["controls"]["K_REG"] is True for s in (0, 1, 2))}
    c_all["ALL_PASS"] = bool(all(c_all.values()))
    res["controls"] = {"summary": c_all, "detail": ctl}
    # ---- the word (Amendment 4d section 3, mechanically)
    conds, fails = {}, []
    for s in FIT:
        conds[str(s)] = {"R1_overall": res["R1"][str(s)]["overall"], "R1_PASS": res["R1"][str(s)]["overall"] == "PASS"}
        if not conds[str(s)]["R1_PASS"]:
            fails.append(f"fit seed {s}: R1 OVERALL {res['R1'][str(s)]['overall']} (clauses {sorted(res['R1'][str(s)]['non_pass_clauses'])})")
        for sd in SEEDS_INF:
            fd_ok = bool(R2[str(s)][f"s{sd}"]["FD_S6_compliant"]["separated_below_0"])
            e3_ok = bool(res["R1"][str(s)]["E3_separated_below_0"][f"s{sd}"])
            conds[str(s)][f"s{sd}"] = {"FD_vs_RETIME_separated_below": fd_ok, "E3_vs_ha0_ext_pose_separated_below": e3_ok}
            if not fd_ok:
                fails.append(f"fit seed {s}, inference seed {sd}: FD vs RETIME_DEPLOY8 NOT separated below ({f4(R2[str(s)][f's{sd}']['FD_S6_compliant'])})")
            if not e3_ok:
                fails.append(f"fit seed {s}, inference seed {sd}: E3 vs ha0_ext_pose NOT separated below")
    word = "REPLICATED" if (not fails and c_all["ALL_PASS"]) else "NOT REPLICATED"
    if not c_all["ALL_PASS"]:
        word = "VOID (a control failed: " + ", ".join(k for k, v in c_all.items() if v is False) + ")"
    res["word"] = {"word": word, "rule": "Amendment 4d s3: REPLICATED iff R1 OVERALL PASS at BOTH fit seeds AND, at both fit seeds and both inference seeds, FD vs RETIME_DEPLOY8 is separated below "
                                          "AND E3 vs ha0_ext_pose is separated below; otherwise NOT REPLICATED naming the clause and the seed", "conditions": conds, "failures": fails}
    res["deviations"] = [
        "Internal arm key: the replicate stages are scored under the forward's internal arm key H2CN_DEPLOY8 (renaming it would touch every script); out_4d_fs{1,2}/arms/h2cn/H2CN_DEPLOY8 holds the "
        "FIT-SEED stage, named by sha256 in the forward (h2cn_stage), in every DEPLOY8_POS npz meta (h2cn_stage_sha256) and in SHIP_R8_H2CN_FS{s}_DEPLOY8.json (stage_sha256_read).",
        "The seed-0 arm is the REGISTERED stage ship_r8_sub/H2CN_STAGE_FROZEN.pt (read-only), re-read from this code copy into out_4d_fs0/; H2CN_FS0_STAGE_FROZEN.pt (the repro copy) is not scored.",
        "K-PERSURF (S2 H2cn - H0 ADE == -0.0130 / -0.0136 within 1e-4) is the SEED-0 fit's confirmed cell: required for the seed-0 reproduction, recorded but NOT required for a replicate "
        "(forward controls.K_PERSURF_required false; verdict clause 1 states it). Its replicate values are in each forward's K_PERSURF_detail.",
        "Clause 1 for a replicate adds the Amendment 4d seal (re-hashed now == the forward's record) and uses REGISTERED_H2CN_FS{s}_STAGE.sha256 / H2CN_FS{s}_FREEZE_INFO.json.",
        "S2 per-window ADE (R2 / R3) has no existing loader: load_s2 mirrors l2_common.load_eval_pf's panel part on dump_s{seed}; it is validated by the E1 recompute control (equals the verdict's registered-code cell).",
        "R3 ADE is reported on the compliant set (the clauses' set) and on all windows; FD / MD on the S6 compliant set as h2cn_fdmd_diag.",
    ]
    res["generated"] = time.strftime("%FT%T%z")
    res["wall_s"] = round(time.time() - t0, 1)
    jp = OUT / "READ_4D.json"
    jp.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    (OUT / "READ_4D.md").write_text(render(res), encoding="utf-8")
    chk = json.loads(jp.read_text(encoding="utf-8"))                              # assert on the ARTIFACT
    assert chk["word"]["word"] and chk["controls"]["summary"]["ALL_PASS"] in (True, False)
    print(f"READ_4D {chk['word']['word']} controls_ALL_PASS={chk['controls']['summary']['ALL_PASS']}", flush=True)
    print("ZZ4DREADDONEZZ", flush=True)
    return 0


def render(r: dict) -> str:
    L = [f"# SHIP-R8 AMENDMENT 4d -- the H2cn stage re-trained at fit seeds 1 and 2", "",
         f"**{r['word']['word']}**", "", f"tier: {r['tier']}", f"estimator: {r['estimator']}", "",
         "## stages (sha256 of the file each read used)", ""]
    for s, v in r["stages"].items():
        L.append(f"* fit seed {s}: `{v['sha256_now']}` registered==file==forward record: {v['all_equal']}")
    L += ["", "## R1 -- SHIP-R8 (1)-(7)+(1b), both inference seeds inside clauses 2-5", "", "| fit seed | arm | OVERALL | per clause |", "|---|---|---|---|"]
    for s, v in r["R1"].items():
        L.append(f"| {s} | {v['arm']} | **{v['overall']}** | {v['per_clause']} |")
    for s, v in r["R1"].items():
        if v["non_pass_clauses"]:
            L.append(f"\nfit seed {s} non-PASS clauses: {json.dumps(v['non_pass_clauses'])}")
    L += ["", "## R2 -- vs RETIME_DEPLOY8 on the same draw (S6 compliant for FD / MD; ADE compliant)", "",
          "| fit seed | inf seed | E1 vs ha0_ext_pose (RETIME) | E3 vs ha0_ext_pose (RETIME) | ADE S2 vs RETIME | ADE S6 vs RETIME | FD vs RETIME | MD vs RETIME |", "|---|---|---|---|---|---|---|---|"]
    for sd in SEEDS_INF:
        b = r["R2"]["seed0_beside_vs_RETIME_DEPLOY8"][f"s{sd}"]
        e1, e3 = r["R1"]["0"]["E1_cells_vs_ha0_ext_pose"][f"s{sd}"], r["R1"]["0"]["E3_cells_vs_ha0_ext_pose"][f"s{sd}"]
        L.append(f"| 0 (beside) | {sd} | {e1['delta']:+.4f} [{e1['lo']:+.4f}, {e1['hi']:+.4f}] ({r['R1']['0']['RETIME_DEPLOY8_E1_cells'][f's{sd}']['delta']:+.4f}) | "
                 f"{e3['delta']:+.4f} [{e3['lo']:+.4f}, {e3['hi']:+.4f}] ({r['R1']['0']['RETIME_DEPLOY8_E3_cells'][f's{sd}']['delta']:+.4f}) | {f4(b['ADE_S2_compliant'])} | {f4(b['ADE_S6_compliant'])} | {f4(b['FD_S6_compliant'])} | {f4(b['MD_S6_compliant'])} |")
    for s in FIT:
        for sd in SEEDS_INF:
            c = r["R2"]["per_fit_seed"][str(s)][f"s{sd}"]
            e1, e3 = c["E1_vs_ha0_ext_pose"], c["E3_vs_ha0_ext_pose"]
            L.append(f"| {s} | {sd} | {e1['delta']:+.4f} [{e1['lo']:+.4f}, {e1['hi']:+.4f}] ({c['RETIME_E1']['delta']:+.4f}) | {e3['delta']:+.4f} [{e3['lo']:+.4f}, {e3['hi']:+.4f}] ({c['RETIME_E3']['delta']:+.4f}) | "
                     f"{f4(c['ADE_S2_compliant'])} | {f4(c['ADE_S6_compliant'])} | {f4(c['FD_S6_compliant'])} | {f4(c['MD_S6_compliant'])} |")
    L += ["", "## R3 -- each replicate PAIRED vs H2CN_DEPLOY8 (fit seed 0) on the same draw: the fit-seed floor (never a lever)", "",
          "| fit seed | inf seed | ADE S2 compl | ADE S6 compl | ADE S2 all | ADE S6 all | FD S6 compl | MD S6 compl |", "|---|---|---|---|---|---|---|---|"]
    for s in FIT:
        for sd in SEEDS_INF:
            c = r["R3_vs_H2CN_DEPLOY8_seed0"][str(s)][f"s{sd}"]
            L.append(f"| {s} | {sd} | {f4(c['ADE_S2_compliant'])} | {f4(c['ADE_S6_compliant'])} | {f4(c['ADE_S2_all_windows'])} | {f4(c['ADE_S6_all_windows'])} | {f4(c['FD_S6_compliant'])} | {f4(c['MD_S6_compliant'])} |")
    L += ["", "## controls", "", json.dumps(r["controls"]["summary"], indent=1), "", "### seed-0 reproduction", "",
          json.dumps({k: v for k, v in r["repro_seed0"].items() if k not in ("positions", "exclusions_named")}, indent=1, default=str),
          "", "## the word (Amendment 4d section 3)", "", f"**{r['word']['word']}**", ""] + [f"* {f}" for f in r["word"]["failures"]] + \
        ["", "## deviations (disclosed)", ""] + [f"* {d}" for d in r["deviations"]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
