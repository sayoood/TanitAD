"""ship_r8_sub_forward.py -- the registered SHIP-R8 battery device (deploy8_forward_r8.py) with the arm positions replaced per SHIP-R8 AMENDMENT 4
(--mode h2cn: the frozen H2cn progress stage) or AMENDMENT 5 (--mode factored: native os-only path x self-sharpened progress, then RETIME_DEPLOY8).
CPU only, zero GPU.  NEW file (EvalFlyWheel 2026-10-26).  The battery code is IMPORTED and monkeypatched exactly as its sibling builds do (pf_lane.configure
re-points it at the REGISTERED per-frame table; deploy8_forward_r8's own four-hook device replaces only the ARM POSITIONS); nothing is edited.

    python ship_r8_sub_forward.py --mode h2cn --tree step48852ema_pf                       # ONLY after the MM confirmed the stage registration
    python ship_r8_sub_forward.py --mode factored --tree step48852ema_pf                   # identity / control arms only (seed-1 replicate pending)
    python ship_r8_sub_forward.py --mode factored --tree step48852ema_pf --scored           # the scored FACTORED_DEPLOY8 read (gated: SSGUARD seed-1 REPLICATED)
    python ship_r8_sub_forward.py --mode h2cn|factored --tree step48852ema_pf --mutate <kind>   # deliberate regression: an identity control MUST go RED

ARMS
  h2cn:     H2CN_DEPLOY8 (SCORED, Amendment 4 s2) | H2CN_NEAR, H2CN_PERSURFACE, RETIME_DEPLOY8, RETIME_DEPLOY_PERSURFACE, CONST_G1 (beside / controls)
  factored: FACTORED_DEPLOY8 (SCORED, only with --scored) | OSONLY_DEPLOY8, FACTORED_SSOS_DEPLOY8 (= FACTORED with P_ss := P_os), RETIME_DEPLOY8,
            RETIME_DEPLOY_PERSURFACE, CONST_G1
CONTROLS (any failure WITHHOLDS: WITHHELD_<mode>_<tree>.json, no FORWARD file, rc 4)
  deploy8_forward_r8's own (a)-(f): RT.forward ALL_PASS (incl. every arm's on-path geometry); CONST_G1 - os exactly 0 on every row; 8-slot known values
  (g == 1 bit-equal plan8, g == 0 == the independent walker); RETIME_DEPLOY_PERSURFACE S2 == FORWARD_TRAINFIT RETIME_TRAINFIT S2 exactly; row alignment;
  the deploy table == the REGISTERED per-frame train gains at its knots.
  + K-RETIME   RETIME_DEPLOY8 re-read through this code path == the REGISTERED FORWARD_DEPLOY8_<tree>.json arm blocks and bars EXACTLY
  h2cn:  K-REG (registered stage sha256 == the file; the registration gate record precedes the first eval-side stage output), K-DISJ (fresh-clip
         disjointness, re-asserted), K-PERSURF (H2CN_PERSURFACE S2 H2cn - H0 ADE == -0.0130 / -0.0136 within 1e-4, l1c_h2cn_confirm's cell),
         K-ATPROG (the per-surface positions == progress_headroom.at_progress bit for bit)
  factored: K-SSOS (FACTORED with P_ss := P_os == OSONLY_DEPLOY8 bit for bit: composed plan, positions, every cell), K-TRACE (trace final / 4,754 /
         hook counters / sha256 recorded), K-FEED (the os-only rolls' own manifests read perframe_full)
MUTATIONS: stage (h2cn: +0.05 m on every S2 output of the frozen S2 net -> K-PERSURF RED), knot (+0.01 on knot 1 -> K-RETIME RED),
           draw (factored: P_ss := the battery tree's own plan, the MAPSEL K_STAGE draw mismatch -> K-SSOS RED).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ship_r8_sub_common as C  # noqa: E402

import numpy as np  # noqa: E402

SCRIPT_SHA = C.sha256_file(__file__)            # hashed at IMPORT: the code that actually ran
COMMON_SHA = C.sha256_file(HERE / "ship_r8_sub_common.py")
PFPKG = C.PF_HOLD / "pfpkg"
REG_PF = Path("C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256")
REG_STAGE_DEFAULT = C.WORK / "REGISTERED_H2CN_STAGE.sha256"
# AMENDMENT 4d: the stage under test is an ARGUMENT now (set in main()).  Defaults = the pre-4d behaviour.  FIT_SEED in (1, 2) = a REPLICATE stage.
STAGE_FILE = C.STAGE_PT
FREEZE_INFO = C.WORK / "H2CN_FREEZE_INFO.json"
FIT_SEED = None
L1_BANK = Path("C:/Users/Admin/refcv8_pf/refcv9_l1/bank")
SS_BANK = Path("F:/refcv9_selfsharp")
H2CN_RESULT = Path("C:/Users/Admin/refcv8_pf/refcv9_l1c/result_h2cn")
N1 = {0: -0.0130, 1: -0.0136}                 # Amendment 4 s3 (= l1c_h2cn_confirm N1, K-R reproduced exactly)
TOL_PERSURF = 1e-4
HZ8 = [5, 10, 15, 20, 30, 40, 50, 60]
T8 = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
SLOTS = C.SLOTS
SURFS = ("S2", "S6")
SETS = ("compliant", "all")
EIGHT = {"H8": "H2CN_DEPLOY8", "HN": "H2CN_NEAR", "D8": "RETIME_DEPLOY8", "G1": "CONST_G1", "OS": "OSONLY_DEPLOY8", "FS": "FACTORED_SSOS_DEPLOY8", "FX": "FACTORED_DEPLOY8",
         "HC": "H2CN_CAP8"}
PERSURF = {"HP": "H2CN_PERSURFACE"}
SCORED = {"h2cn": "H2CN_DEPLOY8", "factored": "FACTORED_DEPLOY8"}
CAP_MARGIN = 1e-6                                # Amendment 4b s1: the (1 - 1e-6) on every interval's advance


def pf_lane():
    sys.path.insert(0, str(C.PF_HOLD / "code"))
    import pf_lane as PL
    return PL


def sha12(x) -> str:
    return hashlib.sha256(str(x).encode()).hexdigest()[:12]


def running_max(s):
    s0 = np.maximum(np.asarray(s, np.float64), 0.0)
    sr = np.maximum.accumulate(s0, axis=1)
    return sr, (sr != s0).any(1)


def h2cn_progress(AL, prior8, f2, f6):
    """Amendment 4 s2: slots 0..3 arc(prior8[:, :4]) + f_S2; slots 4..7 arc(prior8[:, S6])[j] + f_S6[j], j = 2..5; then max(s, 0) and the running max."""
    a4 = AL.arc(prior8[:, :4])
    a6 = AL.arc(prior8[:, SLOTS["S6"]])
    s = np.concatenate([a4 + f2, a6[:, 2:] + f6[:, 2:]], axis=1)
    return running_max(s)


def cap_progress(s8, lim, scale: float = 1.0):
    """Amendment 4b s1: s_cap[k] = min(s8[k], s_cap[k-1] + lim * (t[k] - t[k-1]) * (1 - 1e-6)) on the 8-slot grid T8, s_cap[-1] = 0, t[-1] = 0.
    `lim` is the window's FED ceiling (float32 as the compliance row reads it), widened EXACTLY to float64 before use; a non-finite ceiling leaves the window
    unclamped (+inf advance).  `scale` multiplies lim and is 1.0 for the registered arm; 1.05 exists ONLY for the deliberate-regression arm (--mutate cap105)."""
    s8 = np.asarray(s8, np.float64)
    lim = np.asarray(lim, np.float64) * scale
    fin = np.isfinite(lim)
    out = np.empty_like(s8)
    prev, t_prev = np.zeros(len(s8), np.float64), 0.0
    for k in range(s8.shape[1]):
        dt = float(T8[k] - t_prev)
        adv = np.where(fin, lim * dt * (1.0 - CAP_MARGIN), np.inf)
        out[:, k] = np.minimum(s8[:, k], prev + adv)
        prev, t_prev = out[:, k], float(T8[k])
    return out


def compose(RT, P_path, P_prog):
    """Amendment 5 s1: P = eval_polyline(polyline_geometry(P_path), arc(P_prog)) on the full 8 slots, cast to the path's dtype (the model plan's float32)."""
    s = RT.polyline_geometry(np.asarray(P_prog, np.float64))["L"][:, 1:]
    return RT.eval_polyline(RT.polyline_geometry(P_path), s).astype(np.asarray(P_path).dtype)


def diag_progress(RT, plan, pos64, pos32, s, what):
    """retime_diag's schema for an arm that carries its own progress s (no gain): on-path distance against the polyline the arm moves along."""
    geo = RT.polyline_geometry(plan)
    d = RT.dist_to_path(pos64, geo)
    d32 = RT.dist_to_path(pos32.astype(np.float64), geo)
    LS = geo["L"][:, -1:]
    return {"n_windows": int(len(s)), "max_dist_to_path_f64_m": float(d.max()), "max_dist_to_path_after_cast_m": float(d32.max()),
            "float32_rounding_bound_m": float(np.finfo(pos32.dtype).eps * np.abs(pos64).max()),
            "frac_nonmonotone_windows": float((np.diff(s, axis=1) < -1e-12).any(1).mean()) if s.shape[1] > 1 else 0.0,
            "frac_extrapolated_points": float((s > LS + 1e-9).mean()), "frac_fallback_dir_windows": float(geo["fallback"].mean()),
            "max_abs_g_minus_1": None, "mean_abs_g_minus_1": None, "geometry_over": what}


def load_factored_inputs(seed: int, scored: bool) -> dict:
    sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
    import l2_score as S2c
    b1 = S2c.load_l1_base(str(L1_BANK), seed)
    rec = {"l1_base": {"file": str(L1_BANK / f"ztac_eval_s{seed}.npz")}}
    if b1 is None or b1.get("incomplete"):
        raise SystemExit(f"[sub-fwd] the L1 os-only bank for seed {seed} is absent or not final (refusing)")
    rec["l1_base"].update(sha256=C.sha256_file(rec["l1_base"]["file"]), final=b1["meta"].get("final"), n_windows=b1["meta"].get("n_windows"),
                          counters=b1["meta"].get("counters"))
    out = {"b1": b1, "rec": rec}
    tp = SS_BANK / f"trace_selfsharp_s{seed}.npz"
    if scored:
        ts = S2c.load_trace(tp)
        if ts is None or ts.get("incomplete"):
            raise SystemExit(f"[sub-fwd] {tp} is absent or not final: the FACTORED read is gated on the seed-{seed} SELFSHARP roll (refusing)")
        rec["selfsharp"] = {"file": str(tp), "sha256": C.sha256_file(tp), "final": ts["meta"].get("final"), "n_windows": ts["meta"].get("n_windows"),
                            "counters": ts["meta"].get("counters")}
        out["ts"] = ts
    else:
        rec["selfsharp"] = {"file": str(tp), "present": tp.exists(), "not_read": "identity-only run: FACTORED_DEPLOY8 is not computed (gated on the seed-1 replicate)"}
    return out


def trace_checks(rec: dict, scored: bool) -> dict:
    b = rec["l1_base"]
    cb = b.get("counters") or {}
    ok_b = bool(b.get("final") is True and b.get("n_windows") == 4754 and cb.get("own_calls") == 4754 and cb.get("batch_gt1") == 0 and cb.get("hook_calls_unarmed") == 0
                and len(b.get("sha256", "")) == 64)
    out = {"l1_base_final_4754_batch1_none_unarmed": ok_b}
    if scored:
        t = rec["selfsharp"]
        ct = t.get("counters") or {}
        out["selfsharp_final_4754_all_forced_none_unarmed_batch1"] = bool(
            t.get("final") is True and t.get("n_windows") == 4754 and ct.get("calls") == 4754 and ct.get("hook_applied") == 4754 and ct.get("hook_calls_unarmed") == 0
            and ct.get("batch_gt1") == 0 and ct.get("multi_hook_in_call") == 0 and len(t.get("sha256", "")) == 64)
    return out


def feed_checks(seed: int, scored: bool) -> dict:
    """Amendment 5 (1b): the os-only rolls use the per-frame feed, asserted from the rolls' OWN manifests (dump manifest nav_feed.mode + roll json pf)."""
    out = {}
    rolls = {"l1_os_only": (L1_BANK / f"eval_dump_s{seed}" / "manifest.json", L1_BANK / f"eval_roll_s{seed}.json")}
    if scored:
        rolls["selfsharp"] = (SS_BANK / f"dump_selfsharp_s{seed}" / "manifest.json", SS_BANK / f"roll_selfsharp_s{seed}.json")
    for nm, (man, rj) in rolls.items():
        r = {"manifest": str(man), "roll_json": str(rj)}
        j = {}
        try:
            j = json.load(open(rj, encoding="utf-8"))
            r["roll_pf_nav_feed"], r["roll_pf_ok"] = (j.get("pf") or {}).get("nav_feed"), (j.get("pf") or {}).get("ok")
        except Exception as e:      # noqa: BLE001
            r["roll_json_error"] = type(e).__name__
        if man.exists():                                                 # the dump's own manifest
            try:
                m = json.load(open(man, encoding="utf-8"))
                r["manifest_source"] = "dump manifest.json"
                r["manifest_nav_feed_mode"] = (m.get("nav_feed") or {}).get("mode")
                r["manifest_n_windows"] = (m.get("grid") or {}).get("n_windows")
            except Exception as e:      # noqa: BLE001
                r["manifest_error"] = type(e).__name__
        else:                                                            # dump not kept: the roll json's census of that manifest (written by the roll)
            mn = ((j.get("pf") or {}).get("census") or {}).get("manifest_nav_feed") or {}
            r["manifest_source"] = "roll json pf.census.manifest_nav_feed (the dump directory was not kept)"
            r["manifest_nav_feed_mode"], r["manifest_n_windows"] = mn.get("mode"), mn.get("n_windows")
        r["perframe_full"] = bool(r.get("manifest_nav_feed_mode") == "perframe_full" and r.get("roll_pf_nav_feed") == "perframe_full" and r.get("roll_pf_ok") is True
                                  and r.get("manifest_n_windows") == 4754)
        out[nm] = r
    return out


def forward(mode: str, tree: str, seeds: tuple, scored: bool, mutate: str | None, out_root: Path, reg_stage_file: Path, cap8: bool = False) -> int:
    t_start = time.time()
    seals = C.check_amendment_seals()
    if cap8:
        seals = {**seals, "4b": C.check_amendment_4b_seal()}              # AMENDMENT 4b: refuses unless the sealed text is the file (only when --cap8)
    if FIT_SEED is not None:
        seals = {**seals, "4d": C.check_amendment_4d_seal()}              # AMENDMENT 4d: refuses unless the sealed text is the file (only with --fit-seed)
    C.setup_paths()
    PL = pf_lane()
    reg_pf = PL.read_registered(REG_PF)
    PL.verify_installed(PFPKG, reg_pf)                                   # the installed per-frame table IS the registered one (else SystemExit)
    M = PL.mods()
    PL.configure(M, PFPKG, reg_pf)
    RT, CR, TFM, D8, CP = M["RT"], M["CR"], M["TFM"], M["D8"], M["CP"]
    RS = D8.RS
    import l1_fit_score as F
    AL, PH, PS = F.AL, F.PH, F.PS
    if AL.RT is not RT:
        raise SystemExit("[sub-fwd] two copies of a0k0_headroom_retime_r8 are loaded (the L1 helpers and the battery lane would disagree)")
    tag = f"{mode}" + ("_cap8" if cap8 else "") + ("_scored" if scored else "") + (f"_MUTATION_{mutate}" if mutate else "")
    work = out_root / ("mutation" if mutate else "work") / tag
    work.mkdir(parents=True, exist_ok=True)
    RT.OUT = work                                                         # RT.forward writes ONLY here (never into the registered package)
    CR.OUT, CP.OUT = work / "cr_unused", work / "cp_unused"
    if tuple(seeds) != (0, 1) or tuple(RT.SEEDS) != (0, 1):
        raise SystemExit("[sub-fwd] RT.forward reads BOTH inference seeds (its bars read seed 0 AND seed 1: a0k0_headroom_retime_r8.forward); a one-seed run is impossible on this device")
    train_path = TFM.TF / "FROZEN_GAINS_RETIME_train.json"
    train_fz = json.load(open(train_path, encoding="utf-8"))
    TFM.assert_frozen_is_train(train_fz)
    if TFM.sha256_file(train_path) != reg_pf:
        raise SystemExit("[sub-fwd] REFUSING: the installed train gains do not hash to the registered per-frame sha256")
    table = RS.load_deploy_table()                                        # sha-verified (sidecar)
    V_cmp, vrec = CP.load_vendored()                                      # the compliance row's planned_max_speed (compliant_reread_r8.tools_load's V)
    stage_meta = {"stage_file": str(D8.RS_FILE), "stage_sha256": hashlib.sha256(Path(D8.RS_FILE).read_bytes()).hexdigest(), "table_file": str(RS.DEPLOY_TABLE_PATH),
                  "table_sha256": table["sha256"], "registered_pf_table_sha256": reg_pf}
    k_all = [(f"S2.RETIME[{i}]", train_fz["surfaces"]["S2"]["all"]["RETIME"][i]) for i in range(4)] + \
            [(f"S6.RETIME[{i}]", train_fz["surfaces"]["S6"]["all"]["RETIME"][i]) for i in range(2, 6)]
    ctl_table = {"knots_equal_frozen_train_exactly": bool(table["knots_s"].tolist() == T8.tolist() and table["gains"].tolist() == [v for _n, v in k_all]),
                 "knot_sources": [n for n, _v in k_all]}
    table_d8 = D8.mutate_table(table, 1, 0.01) if mutate == "knot" else table
    if mutate == "knot":
        table_d8["sha256"] = table["sha256"]
    # ---- mode inputs ----------------------------------------------------------------------------------------------------------------------------
    reg_rec, preds, zbank, fac = {}, None, {}, {}
    if mode == "h2cn":
        reg_sha = C.read_seal(reg_stage_file)
        if len(reg_sha) != 64:
            raise SystemExit(f"[sub-fwd] REFUSING: no registered H2cn stage sha256 in {reg_stage_file} (Amendment 4 s1: the stage is registered BEFORE any eval-side output)")
        preds, blob, stage_sha = C.load_stage(STAGE_FILE, reg_sha)           # refuses unless sha256(file) == registered
        gate = {"registered_sha256": reg_sha, "registration_file": str(reg_stage_file), "registration_file_mtime_epoch": os.path.getmtime(reg_stage_file),
                "stage_file": str(STAGE_FILE), "stage_file_sha256": stage_sha, "stage_file_mtime_epoch": os.path.getmtime(STAGE_FILE),
                "passed_epoch": time.time(), "passed_iso": time.strftime("%FT%T%z"),
                "rule": "SHIP-R8 AMENDMENT 4 s1: the frozen stage's sha256 is registered BEFORE any eval-side stage output; this record is written before the first one"}
        gp = work / "REGISTRATION_GATE_H2CN.json"
        gp.write_text(json.dumps(gate, indent=1), encoding="utf-8")
        reg_rec = {"gate": gate, "gate_file": str(gp), "first_eval_output_epoch": None}
        if mutate == "stage":
            import torch
            net_mut = C.make_net(blob["surfaces"]["S2"]["d_in"], blob["surfaces"]["S2"]["d_out"])
            sd = {k: v.clone() for k, v in blob["surfaces"]["S2"]["state_dict"].items()}
            sd["2.bias"] = sd["2.bias"] + 0.05
            net_mut.load_state_dict(sd)
            st = {k: blob["surfaces"]["S2"]["st"][k].numpy().astype(np.float64) for k in ("xmu", "xsd", "zmu", "zsd")}

            def pred_mut(Xe, Ze, _n=net_mut, _st=st):
                with torch.no_grad():
                    return _n(torch.from_numpy(F.mlp_inputs(Xe, Ze, _st))).numpy()
            preds = {**preds, "S2": pred_mut}
        for s in seeds:
            zbank[s] = F.load_bank(L1_BANK / f"ztac_eval_s{s}.npz")
        arm_order = ["H2CN_DEPLOY8", "H2CN_NEAR", "H2CN_PERSURFACE", "RETIME_DEPLOY8", "RETIME_DEPLOY_PERSURFACE", "CONST_G1"] + (["H2CN_CAP8"] if cap8 else [])
    else:
        for s in seeds:
            fac[s] = load_factored_inputs(s, scored)
        arm_order = (["FACTORED_DEPLOY8"] if scored else []) + ["OSONLY_DEPLOY8", "FACTORED_SSOS_DEPLOY8", "RETIME_DEPLOY8", "RETIME_DEPLOY_PERSURFACE", "CONST_G1"]
    # ---- the device (deploy8_forward_r8.forward's four hooks, arms replaced) ----------------------------------------------------------------------
    cache, cur = {}, {"seed": None, "surf": None}
    stash, kv, align, store, extra, compl = {}, {}, {}, {}, {}, {}
    orig_tl, orig_si, orig_ps = RT.tools_load, RT.surface_inputs, CR.prepare_seed
    orig_apply, orig_diag, orig_fw = RT.retime_apply, RT.retime_diag, RT.frozen_weights
    const1, const0 = {"knots_s": [0.5], "gains": [1.0]}, {"knots_s": [0.5], "gains": [0.0]}
    KIND_OF = {"H2CN_DEPLOY8": "H8", "H2CN_NEAR": "HN", "RETIME_DEPLOY8": "D8", "CONST_G1": "G1", "OSONLY_DEPLOY8": "OS", "FACTORED_SSOS_DEPLOY8": "FS",
               "FACTORED_DEPLOY8": "FX", "H2CN_PERSURFACE": "HP", "H2CN_CAP8": "HC"}
    classes = {k: type(f"_Rows{k}", (D8._Rows,), {"KIND": k}) for k in KIND_OF.values()}

    def tools_load_once():
        if "t" not in cache:
            cache["t"] = orig_tl()
        return cache["t"]

    def prepare_seed_rec(tree_, seed, tools, rc):
        cur["seed"] = int(seed)
        ctx = orig_ps(tree_, seed, tools, rc)
        cur["lim"] = ctx["lim"]
        return ctx

    def surface_inputs_hook(tools, ctx, surf, first, tree_):
        S = orig_si(tools, ctx, surf, first, tree_)
        seed, P = cur["seed"], ctx["panels"][surf]
        rows, pdir = P["rows"], S["dir"]
        grid = json.load(open(os.path.join(pdir, "manifest.json"), encoding="utf-8"))["grid"]
        slots, hz = [int(x) for x in grid["slots"]], [int(x) for x in grid["horizons_steps"]]
        inst = np.asarray(grid["instants_s"], np.float64)
        a = {"slots_equal_expected": slots == SLOTS[surf], "slot_horizons_consistent_with_model": [HZ8[i] for i in slots] == hz,
             "instants_equal_T8_at_slots": bool(np.array_equal(inst, T8[slots]))}
        plan8 = np.ascontiguousarray(ctx["plan"][rows])
        import torch
        from tanitad.models import kinematic_prior as KP
        from tanitad.models.ego_history import ego_channels_from_poses
        _m, files, _e = tools["NR"].panel_layout(pdir)
        dd = [np.load(os.path.join(pdir, "decisions", os.path.basename(f))) for f in files]
        dec_plan = np.concatenate([d["plan_full_nav_true"] for d in dd])
        dec_ws = np.concatenate([d["ws"] for d in dd]).astype(np.int64)
        a["dump_traj_eq_decisions_plan_full_nav_true"] = bool(plan8.shape == dec_plan.shape and np.array_equal(plan8, dec_plan))
        a["decisions_ws_eq_rows_ws"] = bool(np.array_equal(dec_ws, S["ws"]))
        a["panel_os_eq_plan8_at_slots"] = bool(np.array_equal(S["X"]["O"], plan8[:, slots]))
        pw = torch.from_numpy(np.ascontiguousarray(S["dec"]["win"], dtype=np.float32))
        a0, kp = KP.prior_controls("ha0_ext_pose", pw)
        v0 = ego_channels_from_poses(pw, RT.W_OBS, dt=RT.DT)[:, -1, 0]
        prior8 = KP.prior_path(a0, kp, v0, HZ8).numpy().astype(np.float32)
        a["prior8_at_slots_eq_RT_HP"] = bool(np.array_equal(prior8[:, slots], S["X"]["HP"]))
        g8 = RS.gain_at(T8, table_d8)
        st = {"plan8": plan8, "prior8": prior8, "g8": g8, "slots": slots, "in8": {}, "s8": {}}
        # RETIME_DEPLOY8 exactly as deploy8_forward_r8 (the registered arm) + the known values
        st["D8"] = (RS.retime_plan(plan8, prior8, T8, table_d8, cast=True), RS.retime_plan(plan8, prior8, T8, table_d8, cast=False))
        g1_32, g1_64 = RS.retime_plan(plan8, prior8, T8, const1, cast=True), RS.retime_plan(plan8, prior8, T8, const1, cast=False)
        g0 = RS.retime_plan(plan8, prior8, T8, const0, cast=False)
        st["G1"] = (g1_32, g1_64)
        kv[f"{surf}|s{seed}"] = {"n_windows": int(len(plan8)), "g1_bit_equal_plan8": bool(g1_32.tobytes() == plan8.tobytes()),
                                 "g1_max_abs_m": float(np.max(np.abs(g1_32.astype(np.float64) - plan8.astype(np.float64)))),
                                 "g0_vs_walker_max_abs_m": float(np.max(np.abs(g0 - TFM.walker_positions(plan8, prior8))))}
        ex = {}
        if mode == "h2cn":
            lim = cur["lim"][rows]
            X = PS.feats(S["dec"]["win"], lim)
            Z, _j = F.join(zbank[seed], S["clip"], S["ws"])
            if reg_rec["first_eval_output_epoch"] is None:
                reg_rec["first_eval_output_epoch"] = time.time()
            f2, f6 = preds["S2"](X, Z), preds["S6"](X, Z)
            s8, changed = h2cn_progress(AL, prior8, f2, f6)
            geo = RT.polyline_geometry(plan8)
            p64 = RT.eval_polyline(geo, s8)
            st["H8"] = (p64.astype(plan8.dtype), p64)
            st["s8"]["H8"] = s8
            s_ret = RS.retime_progress(geo["L"][:, 1:], RS.prior_arc(prior8), g8)
            sN, changedN = running_max(np.concatenate([np.maximum(AL.arc(prior8[:, :4]) + f2, 0.0), s_ret[:, 4:]], axis=1))
            pN = RT.eval_polyline(geo, sN)
            st["HN"] = (pN.astype(plan8.dtype), pN)
            st["s8"]["HN"] = sN
            O, HP = S["X"]["O"], S["X"]["HP"]
            fs = f2 if surf == "S2" else f6
            sP = np.maximum(AL.arc(HP) + fs, 0.0)
            pP64 = RT.eval_polyline(RT.polyline_geometry(O), sP)
            pP32 = PH.at_progress(O, AL.arc(HP) + fs)                      # L1C's own call (the identity's code path)
            st["HP"] = (pP32, pP64)
            st["sP"] = sP
            m = np.asarray(S["mask"], bool)
            Xm = PS.feats(S["dec"]["win"][m], lim[m])
            ex = {"running_max_changed_windows_H2CN_DEPLOY8": int(changed.sum()), "running_max_changed_windows_H2CN_NEAR": int(changedN.sum()), "n_windows": int(len(s8)),
                  "persurface_at_progress_bit_equal_f64_cast": bool(pP32.tobytes() == pP64.astype(O.dtype).tobytes()),
                  "feats_all_rows_masked_bit_equal_feats_on_compliant_rows": bool(np.array_equal(X[m], Xm)),
                  "frac_progress_clipped_at_0_H8": float((np.concatenate([AL.arc(prior8[:, :4]) + f2, AL.arc(prior8[:, SLOTS['S6']])[:, 2:] + f6[:, 2:]], 1) < 0).any(1).mean())}
            if cap8:
                # AMENDMENT 4b: H2CN_CAP8 = the registered s8, clamped on the 8-slot grid to the window's FED ceiling (`lim`, exactly the compliance row's input)
                scale = 1.05 if mutate == "cap105" else 1.0                     # 1.05 = the deliberate-regression arm only
                sC = cap_progress(s8, lim, scale)
                pC64 = RT.eval_polyline(geo, sC)
                st["HC"] = (pC64.astype(plan8.dtype), pC64)
                st["s8"]["HC"] = sC
                # K-CAP-ID: lim := +inf everywhere -> bit-equal to H2CN_DEPLOY8 (the progress, the float64 positions, the cast positions)
                sI = cap_progress(s8, np.full(len(s8), np.inf))
                pI64 = RT.eval_polyline(geo, sI)
                pI32 = pI64.astype(plan8.dtype)
                limf64 = np.asarray(lim, np.float64)
                nf = ~np.isfinite(limf64)
                changed_c = (sC != s8).any(1)
                changed_pos = (st["HC"][0] != st["H8"][0]).any((1, 2))
                ex["cap8"] = {
                    "lim_scale": scale, "margin": CAP_MARGIN, "n_windows": int(len(s8)), "n_compliant": int(m.sum()),
                    "n_changed_by_clamp": int(changed_c.sum()), "share_changed_by_clamp": float(changed_c.mean()),
                    "n_changed_by_clamp_compliant": int(changed_c[m].sum()), "share_changed_by_clamp_compliant": float(changed_c[m].mean()) if m.any() else None,
                    "n_cast_positions_differ_from_H2CN_DEPLOY8": int(changed_pos.sum()),
                    "n_nonfinite_lim": int(nf.sum()), "n_lim_le_0": int((limf64 <= 0).sum()),
                    "lim_min_finite": float(limf64[~nf].min()) if (~nf).any() else None, "lim_max_finite": float(limf64[~nf].max()) if (~nf).any() else None,
                    "nonfinite_lim_windows_bit_equal_H2CN_DEPLOY8": bool(st["HC"][0][nf].tobytes() == st["H8"][0][nf].tobytes() and st["HC"][1][nf].tobytes() == st["H8"][1][nf].tobytes()),
                    "id_progress_bit_equal": bool(sI.tobytes() == s8.tobytes()), "id_pos64_bit_equal": bool(pI64.tobytes() == st["H8"][1].tobytes()),
                    "id_pos_cast_bit_equal": bool(pI32.tobytes() == st["H8"][0].tobytes()),
                    "mono_s_cap_nondecreasing_from_0": bool((np.diff(np.concatenate([np.zeros((len(sC), 1)), sC], axis=1), axis=1) >= 0).all() and np.isfinite(sC).all()),
                    "s_cap_min": float(sC.min()), "max_progress_removed_m": float((s8 - sC).max())}
            store_extra = {"X": X, "Z_rows": _j, "f2": f2, "f6": f6, "eid": np.asarray([str(e) for e in S["eid"]]), "mask": m, "G": S["X"]["G"], "O": O, "HP": HP, "dt": S["dt"]}
            extra[(seed, surf)] = store_extra
        else:
            b = fac[seed]
            keys = list(zip(np.asarray(S["clip"], np.int64).tolist(), np.asarray(S["ws"], np.int64).tolist()))
            jb = np.array([b["b1"]["idx"].get(k, -1) for k in keys])
            a["os_only_join_complete"] = bool((jb >= 0).all())
            if (jb < 0).any():
                raise SystemExit(f"[sub-fwd] {int((jb < 0).sum())} {surf} windows have no os-only bank entry")
            P_os8 = np.ascontiguousarray(b["b1"]["plan8"][jb]).astype(np.float32)
            st["in8"]["OS"] = P_os8
            st["OS"] = (RS.retime_plan(P_os8, prior8, T8, table, cast=True), RS.retime_plan(P_os8, prior8, T8, table, cast=False))
            P_ssos_src = plan8 if mutate == "draw" else P_os8                # MUTATION: the battery tree's own plan (a different inference draw) as P_ss
            P_fs = compose(RT, P_os8, P_ssos_src)
            st["in8"]["FS"] = P_fs
            st["FS"] = (RS.retime_plan(P_fs, prior8, T8, table, cast=True), RS.retime_plan(P_fs, prior8, T8, table, cast=False))
            ex = {"composed_ssos_bit_equal_P_os8": bool(P_fs.tobytes() == P_os8.tobytes()),
                  "osonly_vs_tree_plan_windows_differ": int((np.abs(P_os8.astype(np.float64) - plan8.astype(np.float64)).max(axis=(1, 2)) > 0).sum()), "n_windows": int(len(P_os8))}
            if scored:
                jt = np.array([b["ts"]["idx"].get(k, -1) for k in keys])
                if (jt < 0).any():
                    raise SystemExit(f"[sub-fwd] {int((jt < 0).sum())} {surf} windows have no SELFSHARP trace entry")
                P_ss8 = np.ascontiguousarray(b["ts"]["plan8"][jt]).astype(np.float32)
                P_fx = compose(RT, P_os8, P_ss8)
                st["in8"]["FX"] = P_fx
                st["FX"] = (RS.retime_plan(P_fx, prior8, T8, table, cast=True), RS.retime_plan(P_fx, prior8, T8, table, cast=False))
                ex["selfsharp_vs_osonly_windows_differ"] = int((np.abs(P_ss8.astype(np.float64) - P_os8.astype(np.float64)).max(axis=(1, 2)) > 0).sum())
        # the COMPLIANCE row (A-R8-CLAMP-2 s2.3), as registered: compliant_reread_r8.pmax + compliance_sets on the panel grid and the full 8-slot plan
        rc_ = tools["rc"]
        lim_r, v0_r = cur["lim"][rows], ctx["v0"][rows]
        eid_r, mk = np.asarray(S["eid"]), np.asarray(S["mask"], bool)
        crow = {"panel_grid_horizons_ticks": hz, "full_plan_horizons_ticks": [int(h) for h in rc_["horizons"]], "arms": {}}
        for k_, nm_ in [("os", "os_model_plan")] + [(k2, EIGHT[k2]) for k2 in ("H8", "HN", "D8", "OS", "FS", "FX", "HC") if k2 in st]:
            full = plan8 if k_ == "os" else st[k_][0]
            crow["arms"][nm_] = {"panel_grid": CR.compliance_sets(CR.pmax(V_cmp, full[:, slots], hz, rc_["tick_s"]), lim_r, v0_r, eid_r, {"all_windows": None, "compliant_set": mk}),
                                 "full_plan_0_6s": CR.compliance_sets(CR.pmax(V_cmp, full, [int(h) for h in rc_["horizons"]], rc_["tick_s"]), lim_r, v0_r, eid_r,
                                                                      {"all_windows": None, "compliant_set": mk})}
        compl[f"{surf}|s{seed}"] = crow
        extra[f"{surf}|s{seed}"] = ex
        align[f"{surf}|s{seed}"] = a
        stash[(seed, surf)] = st
        cur["surf"] = surf
        store[(seed, surf)] = {"os": S["X"]["O"], "plan8": plan8, "prior8": prior8, "clip": S["clip"], "ws": S["ws"], "eid": np.asarray(S["eid"]),
                               "compliant": np.asarray(S["mask"], bool), "slots": np.asarray(slots), "instants": inst, "rows": np.asarray(rows)}
        return S

    def frozen_weights_hook(_fz, surf, _v0):
        n = len(stash[(cur["seed"], surf)]["plan8"])
        T = len(SLOTS[surf])
        t_s = np.array(C.INSTANTS[surf])
        out = {}
        for arm in arm_order:
            if arm == "RETIME_DEPLOY_PERSURFACE":
                out[arm] = RS.gain_at(t_s, table)
            else:
                out[arm] = D8.rows_gain(n, T, classes[KIND_OF[arm]])
        return out

    def retime_apply_hook(O, B, Gn, cast=True):
        if isinstance(Gn, D8._Rows):
            st = stash[(cur["seed"], cur["surf"])]
            rows = np.asarray(Gn)[:, 0].astype(np.int64)
            arr = st[Gn.KIND][0 if cast else 1]
            pos = (arr if Gn.KIND == "HP" else arr[:, st["slots"]])[rows]
            if not np.array_equal(np.asarray(O), np.asarray(store[(cur["seed"], cur["surf"])]["os"])[rows]):
                raise SystemExit("[sub-fwd] the rows RT.forward selected do not carry the os the positions were built from (row misalignment)")
            return pos
        return orig_apply(O, B, Gn, cast)

    def retime_diag_hook(O, B, Gn):
        if isinstance(Gn, D8._Rows):
            st = stash[(cur["seed"], cur["surf"])]
            rows = np.asarray(Gn)[:, 0].astype(np.int64)
            k = Gn.KIND
            if k in ("D8", "G1"):
                g8 = st["g8"] if k == "D8" else np.ones(8)
                return D8.diag8(st["plan8"][rows], st["prior8"][rows], st[k][1][rows], st[k][0][rows], g8, st["plan8"].dtype)
            if k in ("OS", "FS", "FX"):
                d = D8.diag8(st["in8"][k][rows], st["prior8"][rows], st[k][1][rows], st[k][0][rows], RS.gain_at(T8, table), st["plan8"].dtype)
                d["geometry_over"] = "the 8-slot stage call on the arm's own input plan (os-only / composed): positions vs that plan's 8-point polyline"
                return d
            if k in ("H8", "HN", "HC"):
                return diag_progress(RT, st["plan8"][rows], st[k][1][rows], st[k][0][rows], st["s8"][k][rows],
                                     "the 8-slot H2cn call: positions at all 8 slots against the 8-point model-plan polyline (progress after max(s,0) + running max)"
                                     + (" then clamped to the fed ceiling (Amendment 4b)" if k == "HC" else ""))
            if k == "HP":
                return diag_progress(RT, np.asarray(O), st["HP"][1][rows], st["HP"][0][rows], st["sP"][rows],
                                     "the per-surface H2cn form: positions against the surface's own os polyline")
        return orig_diag(O, B, Gn)

    RT.tools_load, RT.surface_inputs, RT.frozen_weights = tools_load_once, surface_inputs_hook, frozen_weights_hook
    RT.retime_apply, RT.retime_diag, CR.prepare_seed = retime_apply_hook, retime_diag_hook, prepare_seed_rec
    RT.FROZEN_JSON = train_path
    try:
        rc = RT.forward(tree)
    finally:
        RT.retime_apply, RT.retime_diag, CR.prepare_seed = orig_apply, orig_diag, orig_ps
        RT.tools_load, RT.surface_inputs, RT.frozen_weights = orig_tl, orig_si, orig_fw
    p = work / f"FORWARD_RETIME_{tree}.json"
    res = json.load(open(p, encoding="utf-8"))
    os.remove(p)
    tools = cache["t"]
    # ---- controls ---------------------------------------------------------------------------------------------------------------------------------
    controls = {"rt_forward": bool(res["controls"]["ALL_PASS"] and rc == 0), "table_matches_frozen_train": bool(ctl_table["knots_equal_frozen_train_exactly"]), "table": ctl_table}
    try:
        controls["const_g1_cell_rows_checked_exactly_zero"] = TFM.assert_const_g1_cells(res)
        controls["const_g1_exactly_os"] = True
    except SystemExit as e:
        controls["const_g1_exactly_os"], controls["const_g1_error"] = False, str(e)
    bad = [f"{k}: g1_bit_equal={v['g1_bit_equal_plan8']} g1_max={v['g1_max_abs_m']:.3g} g0_walker_max={v['g0_vs_walker_max_abs_m']:.3g}" for k, v in kv.items()
           if not (v["g1_bit_equal_plan8"] and v["g1_max_abs_m"] == 0.0 and v["g0_vs_walker_max_abs_m"] <= 1e-6)]
    controls["known_values_8slot"] = bool(kv and not bad)
    controls["known_value_arrays_8slot"] = kv
    controls["row_alignment"] = bool(align and all(all(v for v in a.values()) for a in align.values()))
    controls["row_alignment_detail"] = align
    tfp = TFM.TF / f"FORWARD_TRAINFIT_{tree}.json"
    ref_tf = json.load(open(tfp, encoding="utf-8")) if tfp.exists() else None
    ident = D8.s2_identity(res, ref_tf)
    ident["reference_controls_ALL_PASS"] = (ref_tf or {}).get("controls", {}).get("ALL_PASS")
    controls["persurface_identity_vs_forward_trainfit"] = bool(ident["pass"] and ident["reference_controls_ALL_PASS"] is True)
    controls["persurface_identity_vs_forward_trainfit_detail"] = ident
    # K-RETIME: the registered arm re-read through THIS code path == the registered FORWARD_DEPLOY8 blocks EXACTLY
    regp = D8.DEP / f"FORWARD_DEPLOY8_{tree}.json"
    regd8 = json.load(open(regp, encoding="utf-8"))
    mism, n = [], 0
    for s in seeds:
        for surf in SURFS:
            for stn in SETS:
                for arm in ("RETIME_DEPLOY8", "RETIME_DEPLOY_PERSURFACE", "CONST_G1"):
                    a_ = res["seeds"][str(s)]["surfaces"][surf]["sets"][stn]["arms"][arm]
                    b_ = regd8["seeds"][str(s)]["surfaces"][surf]["sets"][stn]["arms"][arm]
                    mism += [f"s{s}|{surf}|{stn}|{arm}{m}" for m in D8.exact_equal(a_, b_)]
                    n += D8.count_leaves(a_)
    bars_eq = {k: not D8.exact_equal(res["bars"][k], regd8["bars"][k]) for k in ("compliant|RETIME_DEPLOY8", "all|RETIME_DEPLOY8")}
    controls["K_RETIME_identity_vs_registered_forward_deploy8"] = bool(not mism and n > 0 and all(bars_eq.values()))
    controls["K_RETIME_detail"] = {"registered_file": str(regp), "registered_sha256": C.sha256_file(regp), "n_leaves_compared": n, "n_mismatches": len(mism),
                                   "first_mismatches": mism[:10], "bars_equal": bars_eq, "seeds": list(seeds)}
    need = ["rt_forward", "const_g1_exactly_os", "known_values_8slot", "persurface_identity_vs_forward_trainfit", "row_alignment", "table_matches_frozen_train",
            "K_RETIME_identity_vs_registered_forward_deploy8"]
    if mode == "h2cn":
        g = reg_rec["gate"]
        controls["K_REG"] = bool(g["registered_sha256"] == g["stage_file_sha256"] and reg_rec["first_eval_output_epoch"] is not None
                                 and g["passed_epoch"] < reg_rec["first_eval_output_epoch"] and g["registration_file_mtime_epoch"] < reg_rec["first_eval_output_epoch"])
        controls["K_REG_detail"] = {**reg_rec, "first_eval_output_iso": time.strftime("%FT%T%z", time.localtime(reg_rec["first_eval_output_epoch"] or 0))}
        dj = C.fresh_disjointness(None)
        fi = json.loads(FREEZE_INFO.read_text(encoding="utf-8")) if FREEZE_INFO.exists() else {}
        controls["K_DISJ"] = bool(dj["PASS"] and (fi.get("controls") or {}).get("K-DISJ") is True and (fi.get("stage_file") or {}).get("sha256") == g["stage_file_sha256"])
        controls["K_DISJ_detail"] = {"now": {k: dj[k] for k in ("intersections", "consistency", "PASS")}, "at_freeze_K-DISJ": (fi.get("controls") or {}).get("K-DISJ"),
                                     "freeze_info_stage_sha256": (fi.get("stage_file") or {}).get("sha256")}
        # K-PERSURF: L1C's S2 cell (l1c_h2cn_confirm: F.comp / F.cell, H0 = RT.retime_apply(O, HP, per-frame S2 all RETIME), compliant rows)
        gains_s2 = np.asarray(train_fz["surfaces"]["S2"]["all"]["RETIME"], np.float64)
        kp, s6_id = {}, {}
        for s in seeds:
            e = extra[(s, "S2")]
            m = e["mask"]
            Pn = stash[(s, "S2")]["HP"][0][m]
            H0 = RT.retime_apply(e["O"][m], e["HP"][m], gains_s2)
            c = F.cell(tools, F.comp(tools, Pn, e["G"][m], e["dt"])["ade_m"], F.comp(tools, H0, e["G"][m], e["dt"])["ade_m"], e["eid"][m])
            kp[str(s)] = {"cell": c, "N1": N1[s], "absdiff": abs(c["delta"] - N1[s]), "pass": bool(abs(c["delta"] - N1[s]) <= TOL_PERSURF)}
            bk = H2CN_RESULT / f"arms_S6_s{s}.npz"
            if bk.exists():
                z = np.load(bk)
                e6 = extra[(s, "S6")]
                m6 = e6["mask"]
                st6 = store[(s, "S6")]
                key = {(int(c_), int(w_)): j for j, (c_, w_) in enumerate(zip(z["clip"], z["ws"]))}
                jj = np.array([key.get((int(c_), int(w_)), -1) for c_, w_ in zip(st6["clip"][m6], st6["ws"][m6])])
                mine = stash[(s, "S6")]["HP"][0][m6]
                if (jj >= 0).all() and len(jj) == len(z["ws"]):
                    dif = np.abs(mine.astype(np.float64) - z["arm_H2cn"][jj].astype(np.float64))
                    s6_id[str(s)] = {"n": int(len(jj)), "bit_equal": bool(np.array_equal(mine, z["arm_H2cn"][jj])), "max_abs_m": float(dif.max())}
                else:
                    s6_id[str(s)] = {"aligned": False}
        controls["K_PERSURF_identity_S2_vs_N1"] = bool(kp and all(v["pass"] for v in kp.values()))
        controls["K_PERSURF_detail"] = {"cells_H2cn_minus_H0_S2_compliant_ADE": kp, "tolerance": TOL_PERSURF,
                                        "beside_S6_positions_vs_l1c_banked_arms_S6_H2cn": s6_id}
        controls["K_ATPROG_persurface_bit_equal"] = bool(all(v.get("persurface_at_progress_bit_equal_f64_cast") for k, v in extra.items() if isinstance(k, str)))
        # AMENDMENT 4d: K-PERSURF's reference (-0.0130 / -0.0136) is the SEED-0 confirmed stage's cell.  A REPLICATE stage (fit seed 1/2) is a different fit: its S2 cell is
        # recorded (descriptive, see K_PERSURF_detail) and is NOT required.  The seed-0 reproduction (FIT_SEED None / 0) keeps it required.
        replicate = FIT_SEED not in (None, 0)
        controls["K_PERSURF_required"] = bool(not replicate)
        need += ["K_REG", "K_DISJ"] + ([] if replicate else ["K_PERSURF_identity_S2_vs_N1"]) + ["K_ATPROG_persurface_bit_equal"]
        if cap8:
            # AMENDMENT 4b s2.  A failed control voids the clauses: it is in `need`, so ALL_PASS reads False and the run WITHHOLDS (no FORWARD file, no verdict).
            cx = {k: v["cap8"] for k, v in extra.items() if isinstance(k, str)}
            controls["K_CAP_ID"] = bool(len(cx) == 4 and all(v["id_progress_bit_equal"] and v["id_pos64_bit_equal"] and v["id_pos_cast_bit_equal"]
                                                             and v["nonfinite_lim_windows_bit_equal_H2CN_DEPLOY8"] for v in cx.values()))
            controls["K_CAP_MONO"] = bool(len(cx) == 4 and all(v["mono_s_cap_nondecreasing_from_0"] for v in cx.values()))
            comply = {}
            for key, crow in compl.items():
                cc = crow["arms"]["H2CN_CAP8"]
                for gname in ("panel_grid", "full_plan_0_6s"):
                    for sname in ("all_windows", "compliant_set"):
                        c_ = cc[gname][sname]
                        er = c_.get("exceed_rate_strict") or {}
                        comply[f"{key}|{gname}|{sname}"] = {"n_windows": c_.get("n_windows"), "n_exceed_strict": c_.get("n_exceed_strict"),
                                                            "exceed_rate_strict_mean": er.get("mean"), "exceed_rate_strict_lo": er.get("lo"), "exceed_rate_strict_hi": er.get("hi"),
                                                            "residual_excess_ms_on_exceeding_windows": c_.get("residual_excess_ms_on_exceeding_windows")}
            controls["K_CAP_COMPLY"] = bool(len(comply) == 16 and all((v["n_windows"] or 0) > 0 and v["n_exceed_strict"] == 0 and v["exceed_rate_strict_mean"] == 0.0
                                                                     for v in comply.values()))
            controls["K_CAP_detail"] = {"rule": "Amendment 4b s2: K-CAP-ID (lim := +inf -> bit-equal H2CN_DEPLOY8), K-CAP-COMPLY (compliance row exceed_rate_strict == 0 on every set / grid / "
                                                "surface / seed), K-CAP-MONO (s_cap non-decreasing); clamp-touched counts are reported, not pass/fail",
                                        "per_surface_seed": cx, "compliance_of_H2CN_CAP8": comply, "lim_scale_used": 1.05 if mutate == "cap105" else 1.0}
            need += ["K_CAP_ID", "K_CAP_COMPLY", "K_CAP_MONO"]
    else:
        ssos_pos, mism2, n2 = {}, [], 0
        for s in seeds:
            for surf in SURFS:
                st = stash[(s, surf)]
                ssos_pos[f"{surf}|s{s}"] = {"composed_bit_equal_P_os8": extra[f"{surf}|s{s}"]["composed_ssos_bit_equal_P_os8"],
                                            "pos32_bit_equal": bool(st["FS"][0].tobytes() == st["OS"][0].tobytes()),
                                            "pos64_bit_equal": bool(st["FS"][1].tobytes() == st["OS"][1].tobytes())}
                for stn in SETS:
                    a_ = res["seeds"][str(s)]["surfaces"][surf]["sets"][stn]["arms"]["FACTORED_SSOS_DEPLOY8"]
                    b_ = res["seeds"][str(s)]["surfaces"][surf]["sets"][stn]["arms"]["OSONLY_DEPLOY8"]
                    mism2 += [f"s{s}|{surf}|{stn}{m}" for m in D8.exact_equal(a_, b_)]
                    n2 += D8.count_leaves(a_)
        controls["K_SSOS_identity_bit_for_bit"] = bool(all(all(v.values()) for v in ssos_pos.values()) and not mism2 and n2 > 0)
        controls["K_SSOS_detail"] = {"positions": ssos_pos, "cells_n_leaves": n2, "cells_n_mismatches": len(mism2), "first_mismatches": mism2[:10]}
        tr = {str(s): trace_checks(fac[s]["rec"], scored) for s in seeds}
        controls["K_TRACE"] = bool(all(all(v.values()) for v in tr.values()))
        controls["K_TRACE_detail"] = {"checks": tr, "inputs": {str(s): fac[s]["rec"] for s in seeds}}
        fd = {str(s): feed_checks(s, scored) for s in seeds}
        controls["K_FEED_perframe_full"] = bool(all(r["perframe_full"] for v in fd.values() for r in v.values()))
        controls["K_FEED_detail"] = fd
        need += ["K_SSOS_identity_bit_for_bit", "K_TRACE", "K_FEED_perframe_full"]
    controls["ALL_PASS"] = bool(all(controls.get(k) is True for k in need))
    controls["required"] = need
    res["controls_rt"] = res["controls"]
    res["controls"] = controls
    res.update({"tool": "ship_r8_sub_forward.py", "mode": mode, "scored_read": bool(scored), "tree_name": tree, "n_boot": RT.N_BOOT, "bootstrap_seed": 0, "seeds_run": list(seeds),
                "role": ("MUTATION RUN (deliberate regression; NOT a result)" if mutate else
                         (f"SHIP-R8 AMENDMENT 4b: arm positions replaced; scored arm H2CN_CAP8 (H2CN_DEPLOY8 re-run beside it for the reproduction)" if cap8 else
                          f"SHIP-R8 AMENDMENT {'4' if mode == 'h2cn' else '5'}: arm positions replaced; scored arm {SCORED[mode]}") +
                         ("" if (mode == "h2cn" or scored) else " NOT COMPUTED (identity / control arms only; the scored read is gated on the seed-1 replicate)")),
                "mutation": mutate, "amendments": seals, "arm_order": arm_order, "per_window_extra": {k: v for k, v in extra.items() if isinstance(k, str)},
                "compliance_row": {"rule": "A-R8-CLAMP-2 s2.3 (compliant_reread_r8.pmax + compliance_sets): exceed = planned_max_speed(arm plan) > refcv8's FED ceiling, strict; "
                                           "cluster rate over episodes; panel grid = the surface's slots, full plan = all 8 slots; BESIDE the clauses (no SHIP-R8 clause reads it)",
                                   "vendored_selection": vrec,
                                   "cells": compl},
                "deploy8": {"model_slot_horizons_ticks": HZ8, "slot_instants_s": T8.tolist(), "surface_slots": SLOTS, "deploy_table_gains": [float(x) for x in table["gains"]],
                            **stage_meta},
                "frozen_gains_sha256": res.get("frozen_gains_sha256"), "registered_pf_table_sha256": reg_pf,
                "tier": RT.TIER, "estimator": RT.QUESTION,
                "script_sha256": SCRIPT_SHA, "common_sha256": COMMON_SHA,
                "wall_s": round(time.time() - t_start, 1), "generated": time.strftime("%FT%T%z")})
    if mode == "h2cn":
        res["h2cn_stage"] = {"file": str(STAGE_FILE), "sha256": reg_rec["gate"]["stage_file_sha256"], "registered": reg_rec["gate"]["registered_sha256"],
                             "registration_file": str(reg_stage_file), "freeze_info": str(FREEZE_INFO), "fit_seed": FIT_SEED}
    if cap8:
        res["cap8"] = True
    if mutate:
        red = {"stage": "K_PERSURF_identity_S2_vs_N1", "knot": "K_RETIME_identity_vs_registered_forward_deploy8", "draw": "K_SSOS_identity_bit_for_bit",
               "cap105": "K_CAP_COMPLY"}[mutate]
        rec = {"tree": tree, "mode": mode, **({"cap8": True} if cap8 else {}), "mutation": mutate, "target_control": red, "target_control_value": controls.get(red), "ALL_PASS": controls["ALL_PASS"],
               "mutation_detected": bool(controls.get(red) is False and not controls["ALL_PASS"]),
               "detail": controls.get({"stage": "K_PERSURF_detail", "knot": "K_RETIME_detail", "draw": "K_SSOS_detail", "cap105": "K_CAP_detail"}[mutate]),
               "other_controls": {k: controls.get(k) for k in need}, "generated": res["generated"], "wall_s": res["wall_s"]}
        mp = out_root / "mutation" / f"MUTATION_{mode}_{mutate}_{tree}.json"
        mp.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        chk = json.loads(mp.read_text(encoding="utf-8"))
        print(f"MUTATION {mp} detected={chk['mutation_detected']} ({red}={chk['target_control_value']})", flush=True)
        print(f"ZZSUBMUT_{'RED' if chk['mutation_detected'] else 'GREEN'}ZZ {mode} {mutate}", flush=True)
        shutil.rmtree(work, ignore_errors=True)
        return 0 if chk["mutation_detected"] else 5
    fdir = out_root / "forward"
    fdir.mkdir(parents=True, exist_ok=True)
    out, wh = fdir / f"FORWARD_{tag.upper()}_{tree}.json", fdir / f"WITHHELD_{tag.upper()}_{tree}.json"
    if not controls["ALL_PASS"]:
        RT.jdump({"tree": tree, "mode": mode, "WITHHELD": {k: v for k, v in controls.items() if v is False or k.endswith("_error") or k.endswith("_detail")},
                  "generated": res["generated"]}, wh)
        if out.exists():
            os.remove(out)
        print(f"WITHHELD: a control did not pass -> {wh}\nZZSUBWITHHELDZZ {mode}", flush=True)
        return 4
    if wh.exists():
        os.remove(wh)
    RT.jdump(res, out)
    # ---- per-arm npz packages (the consumer contract of deploy8_forward_r8, one arm per directory, for the TAC device) -------------------------------
    arms_npz = [a for a in arm_order if a in ("H2CN_DEPLOY8", "H2CN_NEAR", "RETIME_DEPLOY8", "FACTORED_DEPLOY8", "OSONLY_DEPLOY8", "H2CN_CAP8")]
    for arm in arms_npz:
        ad = out_root / "arms" / tag / arm
        ad.mkdir(parents=True, exist_ok=True)
        k = KIND_OF[arm]
        for seed in seeds:
            arrs = {}
            for surf in SURFS:
                s_ = store[(seed, surf)]
                st = stash[(seed, surf)]
                p32, p64 = st[k]
                arrs.update({f"{surf}_pos": p32[:, s_["slots"]], f"{surf}_pos64": p64[:, s_["slots"]], f"{surf}_pos8": p32, f"{surf}_os": s_["os"],
                             f"{surf}_plan8": st["in8"].get(k, s_["plan8"]), f"{surf}_prior8": s_["prior8"], f"{surf}_clip": s_["clip"], f"{surf}_ws": s_["ws"],
                             f"{surf}_eid": s_["eid"], f"{surf}_compliant": s_["compliant"], f"{surf}_slots": s_["slots"], f"{surf}_instants_s": s_["instants"], f"{surf}_rows": s_["rows"]})
            meta = {"tool": "ship_r8_sub_forward.py", "tree": tree, "inference_seed": seed, "arm": arm, "tier": RT.TIER, "mode": mode, "table_sha256": table["sha256"],
                    "stage_sha256": stage_meta["stage_sha256"], "model_slot_horizons_ticks": HZ8, "slot_instants_s": T8.tolist(), "surface_slots": SLOTS,
                    "consumer_contract": D8.CONTRACT.replace("RETIME_DEPLOY8", arm),
                    "note": "<S>_os is the battery panel's os (row alignment); <S>_plan8 is the plan the arm was built on (the model plan, or the os-only / composed plan)"}
            if mode == "h2cn":
                meta["h2cn_stage_sha256"] = reg_rec["gate"]["stage_file_sha256"]
            arrs["meta_json"] = np.array(json.dumps(meta))
            np.savez(ad / f"DEPLOY8_POS_{tree}_s{seed}.npz", **arrs)
        # the TAC device's cross-check inputs: this run's FORWARD file under the battery's name + the registered per-surface file (K6)
        shutil.copyfile(out, ad / f"FORWARD_DEPLOY8_{tree}.json")
        regps = D8.DEP / f"FORWARD_DEPLOY_{tree}.json"
        if regps.exists():
            shutil.copyfile(regps, ad / f"FORWARD_DEPLOY_{tree}.json")
    chk = json.load(open(out, encoding="utf-8"))                          # assert on the ARTIFACT
    assert chk["controls"]["ALL_PASS"] is True
    for arm in arms_npz:
        for seed in seeds:
            z = np.load(out_root / "arms" / tag / arm / f"DEPLOY8_POS_{tree}_s{seed}.npz")
            assert z["S2_pos"].shape[1:] == (4, 2) and z["S6_pos"].shape[1:] == (6, 2) and json.loads(str(z["meta_json"]))["arm"] == arm
    shutil.rmtree(work, ignore_errors=True)
    print(f"WROTE {out} controls=ALL_PASS wall={res['wall_s']}s", flush=True)
    print(f"ZZSUBFWDOKZZ {mode}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("h2cn", "factored"))
    ap.add_argument("--tree", default="step48852ema_pf")
    ap.add_argument("--scored", action="store_true", help="factored: also compute FACTORED_DEPLOY8 (gated on the SSGUARD seed-1 replicate)")
    ap.add_argument("--mutate", default=None, choices=("stage", "knot", "draw", "cap105"))
    ap.add_argument("--cap8", action="store_true", help="h2cn: AMENDMENT 4b -- also build H2CN_CAP8 (the stage clamped to the fed ceiling) + K-CAP-ID / K-CAP-COMPLY / K-CAP-MONO")
    ap.add_argument("--registered-stage", default=str(REG_STAGE_DEFAULT))
    ap.add_argument("--out-root", default=str(C.WORK / "out"))
    ap.add_argument("--stage-file", default=None, help="AMENDMENT 4d: the stage under test (default: with --fit-seed 1/2 the replicate's H2CN_FS{s}_STAGE_FROZEN.pt)")
    ap.add_argument("--freeze-info", default=None, help="AMENDMENT 4d: the stage's freeze record (K-DISJ at the freeze; default with --fit-seed 1/2: H2CN_FS{s}_FREEZE_INFO.json)")
    ap.add_argument("--fit-seed", type=int, default=None, choices=(0, 1, 2),
                    help="AMENDMENT 4d: 1/2 = a REPLICATE stage (K-PERSURF descriptive, registration file default REGISTERED_H2CN_FS{s}_STAGE.sha256); 0 = the REGISTERED seed-0 stage "
                         "(--stage-file/--registered-stage/--freeze-info must be given: the read-only ship_r8_sub/ files) reproduced from this copy")
    a = ap.parse_args()
    global STAGE_FILE, FREEZE_INFO, FIT_SEED
    FIT_SEED = a.fit_seed
    if a.fit_seed in (1, 2):
        sp = C.stage_paths(a.fit_seed)
        STAGE_FILE = Path(a.stage_file) if a.stage_file else sp["pt"]
        FREEZE_INFO = Path(a.freeze_info) if a.freeze_info else sp["info"]
        if a.registered_stage == str(REG_STAGE_DEFAULT):
            a.registered_stage = str(C.WORK / f"REGISTERED_H2CN_FS{a.fit_seed}_STAGE.sha256")
    else:
        if a.stage_file:
            STAGE_FILE = Path(a.stage_file)
        if a.freeze_info:
            FREEZE_INFO = Path(a.freeze_info)
        if a.fit_seed == 0 and not (a.stage_file and a.freeze_info and a.registered_stage != str(REG_STAGE_DEFAULT)):
            raise SystemExit("[sub-fwd] --fit-seed 0 needs --stage-file, --freeze-info and --registered-stage (the registered seed-0 files in ship_r8_sub/)")
    if a.fit_seed is not None and Path(a.out_root).resolve() == (C.WORK / "out").resolve():
        raise SystemExit("[sub-fwd] REFUSING: a --fit-seed read must write to its own --out-root (out_4d_fs{s}/)")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[sub-fwd] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    seeds = (0, 1)
    if a.mode == "h2cn" and a.scored:
        raise SystemExit("--scored is the factored mode's gate; the h2cn mode always computes its scored arm (after the registration)")
    if a.mutate == "stage" and a.mode != "h2cn" or a.mutate == "draw" and a.mode != "factored":
        raise SystemExit(f"--mutate {a.mutate} does not apply to --mode {a.mode}")
    if a.cap8 and a.mode != "h2cn":
        raise SystemExit("--cap8 applies to --mode h2cn only")
    if a.mutate == "cap105" and not a.cap8:
        raise SystemExit("--mutate cap105 needs --cap8")
    if a.cap8 and Path(a.out_root).resolve() == (C.WORK / "out").resolve():
        raise SystemExit("[sub-fwd] REFUSING: --cap8 must write to a SEPARATE --out-root (the landed Amendment 4 / 5 outputs under out/ are never overwritten)")
    return forward(a.mode, a.tree, seeds, a.scored, a.mutate, Path(a.out_root), Path(a.registered_stage), a.cap8)


if __name__ == "__main__":
    sys.exit(main())
