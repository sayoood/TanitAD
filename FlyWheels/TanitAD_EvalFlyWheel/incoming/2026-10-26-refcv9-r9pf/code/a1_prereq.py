"""a1_prereq.py -- PREREG_REFCV9_INMODEL_PA PHASE A1: prerequisites + K-DISJ + K-LABSRC-R9, and the calibration-row cache.  CPU only.
NEW file (EvalFlyWheel 2026-10-26).  NO eval139 R9 output is computed here (the eval z_tac bank is only VERIFIED: hashes, meta, seed-independence).

Writes result/R9PF_A1_PREREQ.json, and F:/refcv9_r9pf_scratch/calib_rows.npz (+ .sha256).  If a prerequisite fails: result/R9PF_PA_VERDICT.json = BLOCKED.

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 R8_NO_POSE_SYNC=1 python -X utf8 a1_prereq.py
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402

import numpy as np  # noqa: E402


def blocked(reason: str, rec: dict) -> None:
    RC.RESULT.mkdir(parents=True, exist_ok=True)
    RC.jdump({"verdict": "BLOCKED", "reason": reason, "phase": "A1", "a1_record": rec, "generated": time.strftime("%FT%T%z")},
             RC.RESULT / "R9PF_PA_VERDICT.json")
    RC.jdump(rec, RC.RESULT / "R9PF_A1_PREREQ.json")
    raise SystemExit(f"[a1] BLOCKED: {reason}")


def bank_meta(p: Path) -> dict:
    z = np.load(p, allow_pickle=False)
    m = json.loads(str(z["meta_json"]))
    return z, m


def main() -> int:
    t0 = time.time()
    rec = {"tool": "a1_prereq.py", "prereg": RC.check_prereg(), "script_sha256": RC.sha256_file(__file__),
           "common_sha256": RC.sha256_file(HERE / "r9pf_common.py")}
    # ---------------------------------------------------------------- R9 code = the banked patch on a COPY
    R = RC.r9p()
    pr = subprocess.run(["patch", "-p2", "-R", "--dry-run"], stdin=open(RC.SCRATCH / "REFCV9_INMODEL.tip.patch", "rb"), cwd=str(RC.SCRATCH / "tree"),
                        capture_output=True, text=True)
    rec["r9_code"] = {"copy": str(RC.R9_TREE), "refcv9_progress_sha256": RC._R9["sha256"],
                      "patch_file": str(RC.SCRATCH / "REFCV9_INMODEL.tip.patch"), "patch_sha256": RC.sha256_file(RC.SCRATCH / "REFCV9_INMODEL.tip.patch"),
                      "patch_landed_at": "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-26-refcv9-inmodel/REFCV9_INMODEL.patch (commit 4b5f7fb; git blob 0a165b98 == F:/refcv9_inmodel copy)",
                      "copy_reverse_dry_run_rc": pr.returncode, "copy_carries_patch": pr.returncode == 0,
                      "scalar_names": list(R.SCALAR_NAMES), "regimes": list(R.REGIMES), "dv_threshold": R.DV_THRESHOLD_MS}
    if pr.returncode != 0:
        blocked("the tree copy does not carry the banked R9 patch", rec)
    # ---------------------------------------------------------------- eval z_tac bank at EMA-48,852 (both seeds) + seed independence
    ev = {}
    zz = {}
    for s in (0, 1):
        p = RC.EVAL_BANK / f"ztac_eval_s{s}.npz"
        z, m = bank_meta(p)
        argv = m.get("child_argv", [])
        ck = argv[argv.index("--ckpt") + 1] if "--ckpt" in argv else None
        sd = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else None
        zt = z["z_tac"]
        ev[f"s{s}"] = {"file": str(p), "sha256": RC.sha256_file(p), "ckpt": ck, "ckpt_is_ema_48852": ck == RC.CKPT, "seed_argv": sd, "final": m.get("final"),
                       "n_windows": m.get("n_windows"), "d_tac": m.get("d_tac"), "rows": int(len(zt)), "finite": bool(np.isfinite(zt).all()),
                       "counters": m.get("counters"), "os_only": "--os-only" in argv}
        zz[s] = z
    k0 = list(zip(zz[0]["e_i"].tolist(), zz[0]["t"].tolist()))
    k1 = {k: j for j, k in enumerate(zip(zz[1]["e_i"].tolist(), zz[1]["t"].tolist()))}
    j1 = np.array([k1.get(k, -1) for k in k0])
    aligned = bool((j1 >= 0).all() and len(k0) == len(k1))
    dz = np.abs(zz[0]["z_tac"].astype(np.float64) - zz[1]["z_tac"][j1].astype(np.float64)) if aligned else np.array([np.inf])
    dtr = np.abs(zz[0]["traj_own"].astype(np.float64) - zz[1]["traj_own"][j1].astype(np.float64)) if aligned else np.array([np.inf])
    ev["seed_independence"] = {"aligned_keys": aligned, "n": len(k0), "max_abs_z_s0_minus_s1": float(dz.max()),
                               "discriminating_control_traj_own_differs_across_seeds_max_abs_m": float(dtr.max()),
                               "frac_windows_traj_differs": float((dtr.reshape(len(dtr), -1).max(1) > 0).mean()) if aligned else None}
    ev["seed_independence"]["PASS"] = bool(aligned and dz.max() == 0.0 and dtr.max() > 0.0)
    # the bank's own plan == the battery tree dump of the same seed (s1) / the r8oracle dump (s0, as L1 K4)
    refs = {0: Path("C:/Users/Admin/refcv8_pf/r8oracle/run_48852ema_pf/dump_s0/refcv6_extras.npz"),
            1: Path("C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/dump_s1/refcv6_extras.npz")}
    for s, rp in refs.items():
        try:
            e = np.load(rp, allow_pickle=False)
            ex_t = e["nav_true.traj"]
            W = int(json.loads(str(zz[s]["meta_json"]))["window_W"])
            ws_b = zz[s]["t"].astype(np.int64) + W - 1
            idx = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(e["clip_index"], e["ws"]))}
            jj = np.array([idx.get((int(c), int(w)), -1) for c, w in zip(zz[s]["e_i"], ws_b)])
            ok = bool((jj >= 0).all())
            ev[f"s{s}"]["own_plan_vs_dump"] = {"dump": str(rp), "all_joined": ok,
                                               "max_abs_m": float(np.abs(ex_t[jj].astype(np.float64) - zz[s]["traj_own"].astype(np.float64)).max()) if ok else None}
        except Exception as ex:      # noqa: BLE001
            ev[f"s{s}"]["own_plan_vs_dump"] = {"dump": str(rp), "error": f"{type(ex).__name__}: {ex}"}
    ev["PASS"] = bool(all(ev[f"s{s}"]["ckpt_is_ema_48852"] and ev[f"s{s}"]["final"] is True and ev[f"s{s}"]["n_windows"] == 4754 and ev[f"s{s}"]["rows"] == 4754
                          and ev[f"s{s}"]["finite"] and ev[f"s{s}"]["d_tac"] == 512 and ev[f"s{s}"]["seed_argv"] == s for s in (0, 1))
                      and ev["seed_independence"]["PASS"])
    rec["eval_ztac_bank"] = ev
    if not ev["PASS"]:
        blocked("the eval-side z_tac bank at EMA-48,852 could not be verified (see a1_record.eval_ztac_bank)", rec)
    # ---------------------------------------------------------------- fresh z_tac bank
    zf, mf = bank_meta(RC.FRESH_BANK)
    argv = mf.get("child_argv", [])
    fb = {"file": str(RC.FRESH_BANK), "sha256": RC.sha256_file(RC.FRESH_BANK), "sha256_expected_H2CN_FREEZE_INFO": RC.FRESH_BANK_SHA,
          "ckpt": argv[argv.index("--ckpt") + 1] if "--ckpt" in argv else None, "final": mf.get("final"), "n_windows": mf.get("n_windows"),
          "rows": int(len(zf["z_tac"])), "finite": bool(np.isfinite(zf["z_tac"]).all())}
    fb["PASS"] = bool(fb["sha256"] == RC.FRESH_BANK_SHA and fb["ckpt"] == RC.CKPT and fb["final"] is True and fb["rows"] == fb["n_windows"] and fb["finite"])
    rec["fresh_ztac_bank"] = fb
    if not fb["PASS"]:
        blocked("the fresh z_tac bank does not match H2cn's banked fresh bank", rec)
    # ---------------------------------------------------------------- calibration rows (H2cn's loader, kept with the pose window)
    RC.setup_paths()
    import torch
    import ship_r8_sub_common as C
    import l1c_fit_score as L
    F, AL, PS = L.F, L.AL, L.PS
    RT = L.RT
    import retime_trainfit_r8 as TF
    from tanitad.models import kinematic_prior as KP
    from tanitad.models.ego_history import ego_channels_from_poses
    tools = RT.tools_load()
    TF.TRAIN_ROOT, TF.TRAIN_TREE = str(C.FRESH_TREE_ROOT), C.FRESH_TREE
    _cp, CR2 = TF._patch_archive(TF.TRAIN_ROOT)
    ctx = CR2.prepare_seed(TF.TRAIN_TREE, 0, tools, tools["rc"])
    first, sur = {}, {}
    for surf in ("S2", "S6"):
        S = TF.train_surface_inputs(tools, ctx, surf, first, TF.TRAIN_TREE)
        rows = ctx["panels"][surf]["rows"]
        lim = ctx["lim"][rows]
        m = np.asarray(S["mask"], bool)
        sur[surf] = {"HP": np.asarray(S["X"]["HP"])[m], "G": np.asarray(S["X"]["G"])[m], "O": np.asarray(S["X"]["O"])[m], "win": np.asarray(S["dec"]["win"])[m],
                     "X": PS.feats(S["dec"]["win"][m], lim[m]), "eid": np.asarray([str(e) for e in S["eid"]])[m], "clip": np.asarray(S["clip"], np.int64)[m],
                     "ws": np.asarray(S["ws"], np.int64)[m], "v0": np.asarray(S["v0"], np.float64)[m], "sha12": np.asarray(S["clip_sha12"])[m],
                     "lim": np.asarray(lim)[m], "checks": {"decisions_PASS": bool(S["dec"]["chk"]["PASS"]), "validation_PASS": bool(S["V"]["PASS"])},
                     "n_all": int(len(m)), "plan8": np.asarray(ctx["plan"][rows])[m]}
    del ctx
    ref_counts = {"S2": 9135, "S6": 7147}
    rec["calibration_load"] = {s: {"n_compliant": int(len(sur[s]["ws"])), "n_all": sur[s]["n_all"], "expected_H2CN_FREEZE_INFO": ref_counts[s],
                                   "n_episodes": int(len(np.unique(sur[s]["eid"]))), "n_clips": int(len(np.unique(sur[s]["sha12"]))), **sur[s]["checks"]}
                               for s in sur}
    # ---- K-DISJ (before any fit) + its mutation arm
    disj = C.fresh_disjointness({s: np.unique(sur[s]["sha12"]) for s in sur})
    disj_mut = C.fresh_disjointness(None, inject_eval_clip=True)
    k_disj = bool(disj["PASS"] and not disj_mut["PASS"] and disj_mut["intersections"]["fresh12_x_eval139"] == 1)
    rec["K-DISJ"] = {"PASS": k_disj, "check": disj, "mutation_arm": disj_mut, "mutation_red": bool(not disj_mut["PASS"])}
    # ---- the calibration rows: S6 compliant (all 8 slots + a defined regime), slots 0 / 2 from the S2 row of the same window
    s2k = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(sur["S2"]["clip"], sur["S2"]["ws"]))}
    j2 = np.array([s2k.get((int(c), int(w)), -1) for c, w in zip(sur["S6"]["clip"], sur["S6"]["ws"])])
    sub = bool((j2 >= 0).all())
    if not sub:
        rec["calibration_rows"] = {"S6_subset_of_S2": False, "n_missing": int((j2 < 0).sum())}
        blocked("an S6 compliant fresh row has no S2 compliant row (the 8-slot GT cannot be assembled)", rec)
    A6, A2 = sur["S6"], {k: (v[j2] if isinstance(v, np.ndarray) and len(v) == len(sur["S2"]["ws"]) else v) for k, v in sur["S2"].items()}
    G8 = np.zeros((len(j2), 8, 2), np.float64)
    G8[:, SLOTS6] = A6["G"]
    G8[:, [0, 2]] = A2["G"][:, [0, 2]]
    pw = torch.from_numpy(np.ascontiguousarray(A6["win"], dtype=np.float32))
    a0, kp = KP.prior_controls("ha0_ext_pose", pw)
    v0p = ego_channels_from_poses(pw, RT.W_OBS, dt=RT.DT)[:, -1, 0]
    prior8 = KP.prior_path(a0, kp, v0p, RC.HZ8).numpy().astype(np.float32)
    x9 = RC.x9_inmodel(A6["win"], A6["lim"])
    Zf, jf = F.join(F.load_bank(RC.FRESH_BANK), A6["clip"], A6["ws"])
    fbk = F.load_bank(RC.FRESH_BANK)
    cr = {"S6_subset_of_S2": sub, "n_rows": int(len(j2)), "n_episodes": int(len(np.unique(A6["eid"]))), "n_clips": int(len(np.unique(A6["sha12"]))),
          "G_overlap_slots_1_3_S2_eq_S6_bit": bool(np.array_equal(A2["G"][:, [1, 3]], A6["G"][:, [0, 1]])),
          "win_S2_eq_S6_bit": bool(np.array_equal(A2["win"], A6["win"])), "v0_S2_eq_S6": bool(np.array_equal(A2["v0"], A6["v0"])),
          "prior8_at_S6_slots_eq_RT_HP_bit": bool(np.array_equal(prior8[:, SLOTS6], A6["HP"])),
          "prior8_at_S2_slots_eq_RT_HP_bit": bool(np.array_equal(prior8[:, [0, 1, 2, 3]], A2["HP"])),
          "plan8_at_S6_slots_eq_panel_os_bit": bool(np.array_equal(A6["plan8"][:, SLOTS6], A6["O"])),
          "x9_inmodel_vs_progress_scalar_feats_maxabs": float(np.abs(x9.astype(np.float64) - A6["X"][:, 1:]).max()),
          "x9_maxabs_scale": float(np.abs(A6["X"][:, 1:]).max()),
          "bank_v0_vs_panel_maxabs": float(np.nanmax(np.abs(fbk["v0_in"][jf] - A6["v0"]))), "z_rows": int(len(Zf)),
          "fit_rows_sha12_subset_of_fresh_S6": bool(set(A6["sha12"].tolist()) <= set(sur["S6"]["sha12"].tolist()))}
    cr["PASS"] = bool(cr["G_overlap_slots_1_3_S2_eq_S6_bit"] and cr["win_S2_eq_S6_bit"] and cr["v0_S2_eq_S6"] and cr["prior8_at_S6_slots_eq_RT_HP_bit"]
                      and cr["prior8_at_S2_slots_eq_RT_HP_bit"] and cr["x9_inmodel_vs_progress_scalar_feats_maxabs"] <= 1e-4 * max(1.0, cr["x9_maxabs_scale"])
                      and cr["bank_v0_vs_panel_maxabs"] < 1e-3 and cr["n_rows"] == ref_counts["S6"])
    rec["calibration_rows"] = cr
    # ---- K-LABSRC-R9: the fit's target (refcv9_progress.gt_regime, the in-model loss's own function) vs an INDEPENDENT recomputation from the
    #      RAW GT future (decisions/gt_future_ext steps 50 / 60, extras v0) -- plus the MD/FD cells' plan_lon_class on the S6 panel GT;
    #      mutation: labels from the fresh v9 lon_v7 must go RED.
    r_fit, ok_fit = R.gt_regime(torch.from_numpy(G8), torch.ones(len(G8), 8, dtype=torch.bool), torch.from_numpy(A6["v0"]), RC.HZ8)
    r_fit = r_fit.numpy().astype(np.int64)
    files = sorted(glob.glob(str(RC.FRESH_DUMP / "decisions" / "ep*.npz")))
    ex = np.load(RC.FRESH_DUMP / "refcv6_extras.npz", allow_pickle=False)
    gws, ge, gv = [], [], []
    for f in files:
        d = np.load(f)
        gws.append(d["ws"]); ge.append(d["gt_future_ext"][:, [49, 59], :2]); gv.append(d["gt_future_valid_ext"][:, [49, 59]].min(1))
    gws, ge, gv = np.concatenate(gws).astype(np.int64), np.concatenate(ge).astype(np.float64), np.concatenate(gv)
    order_ok = bool(len(gws) == len(ex["ws"]) and np.array_equal(gws, ex["ws"]))
    ek = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(ex["clip_index"], ex["ws"]))}
    je = np.array([ek.get((int(c), int(w)), -1) for c, w in zip(A6["clip"], A6["ws"])])
    joined = bool((je >= 0).all())
    r_ind, dv_ind = RC.regime_independent(ge[je, 0], ge[je, 1], ex["v0"][je].astype(np.float64))
    pass  # s6_lon_decomp lives in the battery code dir (on sys.path via setup_paths)
    import s6_lon_decomp as S6M
    pc = S6M.plan_lon_class(np.asarray(A6["G"], np.float64), A6["v0"], 1.0)
    r_pc = np.where(pc == "decel", 0, np.where(pc == "accel", 2, 1)).astype(np.int64)
    r_mut = RC.regime_from_lon_v7(ex["lon_v7"][je])
    lab = {"source_fit": "refcv9_progress.gt_regime(G8 slots 50/60, panel v0) -- the in-model loss's own target function",
           "source_independent_raw": "decisions/gt_future_ext[:, 49/59] (world frame) + extras v0, numpy float64, l2_common threshold",
           "source_independent_plan_class": "s6_lon_decomp.plan_lon_class on the S6 panel GT (the MD/FD cells' own class)",
           "decisions_order_eq_extras": order_ok, "all_rows_joined": joined, "gt_valid_at_50_60_all": bool((gv[je] > 0.5).all()),
           "n_rows": int(len(r_fit)), "ok_all": bool(ok_fit.all()),
           "class_counts_fit": {c: int((r_fit == i).sum()) for i, c in enumerate(R.REGIMES)},
           "mismatch_fit_vs_raw": int((r_fit != r_ind).sum()), "mismatch_fit_vs_plan_class": int((r_fit != r_pc).sum()),
           "rows_within_1e-4_of_threshold": int((np.abs(np.abs(dv_ind) - 1.5) < 1e-4).sum()),
           "mutation_lon_v7": {"mismatch_vs_raw": int((r_mut != r_ind).sum()), "n_unmapped": int((r_mut < 0).sum()),
                               "class_counts": {c: int((r_mut == i).sum()) for i, c in enumerate(R.REGIMES)}}}
    lab["mutation_red"] = bool(lab["mutation_lon_v7"]["mismatch_vs_raw"] > 0)
    lab["PASS"] = bool(order_ok and joined and lab["gt_valid_at_50_60_all"] and lab["ok_all"] and lab["mismatch_fit_vs_raw"] == 0
                       and lab["mismatch_fit_vs_plan_class"] == 0 and lab["mutation_red"])
    rec["K-LABSRC-R9"] = lab
    # ---- cache the calibration rows (F:), deterministic content
    out = RC.SCRATCH / "calib_rows.npz"
    np.savez(out, z=Zf.astype(np.float32), x9=x9.astype(np.float32), prior8=prior8, G8=G8, plan8=A6["plan8"].astype(np.float32), v0=A6["v0"],
             r_star=r_fit, eid=A6["eid"].astype("U64"), clip=A6["clip"], ws=A6["ws"], sha12=A6["sha12"].astype("U12"), lim=A6["lim"].astype(np.float32),
             lon_logits_v7=ex["nav_true.tacv6_lon_logits"][je].astype(np.float32))
    rec["calib_cache"] = {"file": str(out), "sha256": RC.sha256_file(out), "n_rows": int(len(r_fit))}
    (RC.SCRATCH / "calib_rows.sha256").write_text(rec["calib_cache"]["sha256"] + "  calib_rows.npz\n", encoding="utf-8")
    rec["ALL_PASS"] = bool(rec["r9_code"]["copy_carries_patch"] and ev["PASS"] and fb["PASS"] and k_disj and cr["PASS"] and lab["PASS"])
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["generated"] = time.strftime("%FT%T%z")
    RC.jdump(rec, RC.RESULT / "R9PF_A1_PREREQ.json")
    if not rec["ALL_PASS"]:
        failed = [k for k, v in (("K-DISJ", k_disj), ("calibration_rows", cr["PASS"]), ("K-LABSRC-R9", lab["PASS"])) if not v]
        blocked(f"A1 control(s) failed: {failed}", rec)
    print(f"[a1] ALL_PASS rows {len(r_fit)} counts {lab['class_counts_fit']} wall {rec['wall_s']} s  ZZA1OKZZ", flush=True)
    return 0


SLOTS6 = [1, 3, 4, 5, 6, 7]

if __name__ == "__main__":
    sys.exit(main())
