"""ship_r8_hsgate_forward.py -- PREREG_HSGATE (2331e1f3...) PHASE 2: a COPY of ship_r8_sub/ship_r8_4c_forward.py (left untouched; built by
make_hsgate_forward.py, diff ship_r8_hsgate_forward.DIFF.txt) with ONE switch added, --hsgate (mode h2cn): FACTORED_DEPLOY8 (Amendment 5, the factored
branch's code verbatim) is built on the SAME rows beside H2CN_DEPLOY8; the REGISTERED gate (HSGATE_FROZEN.json, sha256 == REGISTERED_HSGATE.sha256, checked
BEFORE the first eval-side output) picks per window -> HSGATE8; identity arms HSGATE8_G0 (g := 0) / HSGATE8_G1 (g := 1); controls K_REG_HSGATE,
K_FIT_HSGATE, K_ID0, K_ID1, K_EVALFEAT, K_LANDED (H2CN / FACTORED positions bit-equal the LANDED packages), K_GATE_SURF, K_TRACE, K_FEED.  --hsgate refuses
any --out-root outside C:/Users/Admin/refcv8_pf/refcv9_hsgate and F:/refcv9_hsgate_scratch.

ship_r8_4c_forward.py -- SHIP-R8 AMENDMENT 4c: a COPY of ship_r8_sub_forward.py (left untouched: it is pinned by MANIFEST_4B) with ONE switch added,
--h2cnl (mode h2cn): two more arms built from the SAME forward, H2CNL_DEPLOY8 (the registered H2cn-L stage, H2CNL_STAGE_FROZEN.pt) and H2CNL_MUT8 (the
registered mutation stage, H2CNL_MUT_STAGE.pt, fed the TRUE eval logits), plus K_REG_H2CNL / K_REG_H2CNL_MUT / K_FREEZE_H2CNL / K_JOIN_EVAL.  Every 4c
addition is NAMED (arms HL/HM, per_window_extra[*].h2cnl, the four controls) so the H2CN_DEPLOY8 reproduction can drop exactly them.  Diff:
ship_r8_4c_forward.DIFF.txt.  --h2cnl refuses out/ and out_4b/ as --out-root.

ship_r8_sub_forward.py -- the registered SHIP-R8 battery device (deploy8_forward_r8.py) with the arm positions replaced per SHIP-R8 AMENDMENT 4
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
SUBDIR = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub")                # READ-ONLY: the copied device's helpers are imported from there
sys.path.insert(0, str(SUBDIR))
sys.path.insert(0, str(HERE))
import ship_r8_sub_common as C  # noqa: E402
import hsgate_common as HG  # noqa: E402

import numpy as np  # noqa: E402

SCRIPT_SHA = C.sha256_file(__file__)            # hashed at IMPORT: the code that actually ran
COMMON_SHA = C.sha256_file(SUBDIR / "ship_r8_sub_common.py")
HSG_COMMON_SHA = C.sha256_file(HERE / "hsgate_common.py")
REG_HSG = HG.WORK / "REGISTERED_HSGATE.sha256"                     # written by the Master Mind, NEVER here
HSG_FROZEN, HSG_FIT_INFO = HG.WORK / "HSGATE_FROZEN.json", HG.WORK / "HSGATE_FIT_INFO.json"
PFPKG = C.PF_HOLD / "pfpkg"
REG_PF = Path("C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256")
REG_STAGE_DEFAULT = C.WORK / "REGISTERED_H2CN_STAGE.sha256"
import h2cnl_freeze as HL  # noqa: E402   AMENDMENT 4c: the H2cn-L stage loader (load_stage_l), the extra-column builder (lon6_softmax) and the 4c seal
REG_L_DEFAULT = C.WORK / "REGISTERED_H2CNL_STAGE.sha256"
REG_M_DEFAULT = C.WORK / "REGISTERED_H2CNL_MUT_STAGE.sha256"
EVAL_EXTRAS = "C:/Users/Admin/refcv8_eval_archive/battery/{tree}/dump_s{seed}/refcv6_extras.npz"
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
         "HC": "H2CN_CAP8", "HL": "H2CNL_DEPLOY8", "HM": "H2CNL_MUT8", "HS": "HSGATE8", "Z0": "HSGATE8_G0", "Z1": "HSGATE8_G1"}
PERSURF = {"HP": "H2CN_PERSURFACE"}
SCORED = {"h2cn": "H2CN_DEPLOY8", "factored": "FACTORED_DEPLOY8"}
CAP_MARGIN = 1e-6                                # Amendment 4b s1: the (1 - 1e-6) on every interval's advance


def bit_eq(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return bool(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())


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


def load_eval_extras(tree: str, seed: int) -> dict:
    """AMENDMENT 4c: the battery tree's own extras of inference seed `seed` (the SAME forward the plan came from), indexed on (clip_index, ws)."""
    p = Path(EVAL_EXTRAS.format(tree=tree, seed=seed))
    z = np.load(p, allow_pickle=False)
    e = {"file": str(p), "sha256": C.sha256_file(p), **{k: z[k] for k in ("clip_index", "ws", "v0", "nav_true.traj", HL.LOGIT_KEY)}}
    idx = {}
    for j, k in enumerate(zip(e["clip_index"].tolist(), e["ws"].tolist())):
        idx.setdefault(k, []).append(j)
    o = np.lexsort((e["ws"], e["clip_index"]))
    dws = np.diff(e["ws"][o])[np.diff(e["clip_index"][o]) == 0]
    e.update(_idx=idx, n_dup=int(sum(len(v) - 1 for v in idx.values())), stride=int(np.median(dws)))
    return e


def join_eval_extras(e: dict, clip, ws, v0, plan8, ws_shift: int = 0) -> tuple:
    """AMENDMENT 4c K-JOIN (eval side): every eval row joins EXACTLY one extras row on (clip index, ws + ws_shift); independent cross-checks: extras v0 == the
    battery ctx v0 and extras nav_true.traj == the 8-slot plan the arms re-time (bit for bit: the logits come from the SAME forward)."""
    hits = [e["_idx"].get((int(c), int(w) + ws_shift), []) for c, w in zip(clip, ws)]
    j = np.asarray([h[0] if len(h) == 1 else -1 for h in hits], np.int64)
    ok = j >= 0
    jj = np.where(ok, j, 0)
    rec = {"rows": int(len(j)), "extras_rows": int(len(e["ws"])), "extras_duplicate_keys": e["n_dup"], "missing": int(sum(len(h) == 0 for h in hits)),
           "multi": int(sum(len(h) > 1 for h in hits)), "ws_shift": int(ws_shift),
           "ws_equal": bool(ok.all() and np.array_equal(e["ws"][jj], np.asarray(ws, np.int64) + ws_shift)),
           "clip_equal": bool(ok.all() and np.array_equal(e["clip_index"][jj], np.asarray(clip, np.int64))),
           "v0_maxabs": float(np.abs(e["v0"][jj].astype(np.float64) - np.asarray(v0, np.float64))[ok].max()) if ok.any() else None,
           "traj_bit_equal_plan8": bool(ok.all() and e["nav_true.traj"][jj].tobytes() == np.ascontiguousarray(plan8).tobytes()),
           "traj_vs_plan8_maxabs": float(np.abs(e["nav_true.traj"][jj].astype(np.float64) - np.asarray(plan8, np.float64))[ok].max()) if ok.any() else None}
    rec["PASS"] = bool(rec["extras_duplicate_keys"] == 0 and rec["missing"] == 0 and rec["multi"] == 0 and rec["ws_equal"] and rec["clip_equal"]
                       and rec["v0_maxabs"] is not None and rec["v0_maxabs"] <= 1e-5 and rec["traj_bit_equal_plan8"])
    rec["red_by_crosscheck_alone"] = bool(rec["v0_maxabs"] is not None and (rec["v0_maxabs"] > 1e-5 or rec["traj_vs_plan8_maxabs"] > 1e-5) and rec["missing"] < rec["rows"] // 2)
    return rec, j


def forward(mode: str, tree: str, seeds: tuple, scored: bool, mutate: str | None, out_root: Path, reg_stage_file: Path, cap8: bool = False,
            h2cnl: bool = False, work_root: Path | None = None, hsgate: bool = False) -> int:
    t_start = time.time()
    seals = C.check_amendment_seals()
    if cap8:
        seals = {**seals, "4b": C.check_amendment_4b_seal()}              # AMENDMENT 4b: refuses unless the sealed text is the file (only when --cap8)
    if h2cnl:
        seals = {**seals, "4c": HL.check_amendment_4c_seal()}              # AMENDMENT 4c: refuses unless the sealed text is the file (only when --h2cnl)
    if hsgate:
        seals = {**seals, "PREREG_HSGATE": HG.check_prereg()}              # PREREG_HSGATE: refuses unless the sealed text is the file
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
    tag = f"{mode}" + ("_cap8" if cap8 else "") + ("_h2cnl" if h2cnl else "") + ("_hsgate" if hsgate else "") + ("_scored" if scored else "") + (f"_MUTATION_{mutate}" if mutate else "")
    work = (work_root or out_root) / ("mutation" if mutate else "work") / tag
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
        preds, blob, stage_sha = C.load_stage(C.STAGE_PT, reg_sha)           # refuses unless sha256(file) == registered
        gate = {"registered_sha256": reg_sha, "registration_file": str(reg_stage_file), "registration_file_mtime_epoch": os.path.getmtime(reg_stage_file),
                "stage_file": str(C.STAGE_PT), "stage_file_sha256": stage_sha, "stage_file_mtime_epoch": os.path.getmtime(C.STAGE_PT),
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
        lstages, exl = {}, {}
        if h2cnl:
            L2 = HL.load_l2_common()
            for kk, regf, ptf in (("HL", REG_L_DEFAULT, HL.STAGE_L_PT), ("HM", REG_M_DEFAULT, HL.MUT_PT)):
                rs_ = C.read_seal(regf)
                if len(rs_) != 64:
                    raise SystemExit(f"[sub-fwd] REFUSING: no registered {EIGHT[kk]} stage sha256 in {regf} (Amendment 4c s1: registered BEFORE any eval-side output)")
                pr_, bl_, sh_ = HL.load_stage_l(ptf, rs_)                       # refuses unless sha256(file) == registered
                g_ = {"arm": EIGHT[kk], "registered_sha256": rs_, "registration_file": str(regf), "registration_file_mtime_epoch": os.path.getmtime(regf),
                      "stage_file": str(ptf), "stage_file_sha256": sh_, "stage_file_mtime_epoch": os.path.getmtime(ptf), "stage_kind": bl_.get("kind"),
                      "stage_is_mutation": bool(bl_.get("mutation") is not None), "passed_epoch": time.time(), "passed_iso": time.strftime("%FT%T%z"),
                      "rule": "SHIP-R8 AMENDMENT 4c s1: the frozen stage's sha256 is registered BEFORE any eval-side output; this record is written before the first one"}
                (work / f"REGISTRATION_GATE_{EIGHT[kk]}.json").write_text(json.dumps(g_, indent=1), encoding="utf-8")
                lstages[kk] = {"preds": pr_, "gate": g_}
            if lstages["HL"]["gate"]["stage_is_mutation"] or not lstages["HM"]["gate"]["stage_is_mutation"]:
                raise SystemExit("[sub-fwd] the H2CNL / H2CNL_MUT stage files carry the wrong mutation flag")
            for s in seeds:
                exl[s] = load_eval_extras(tree, s)
        hsg, landed = {}, {}
        if hsgate:
            # PREREG_HSGATE s1: the gate is frozen and its sha256 REGISTERED (REGISTERED_HSGATE.sha256, written by the MM) BEFORE any SCORE-side output
            reg_g = C.read_seal(REG_HSG)
            got_g = C.sha256_file(HSG_FROZEN) if HSG_FROZEN.exists() else ""
            if len(reg_g) != 64 or len(got_g) != 64:
                raise SystemExit(f"[hsgate-fwd] INCONCLUSIVE gate registration: registered '{reg_g}' file '{got_g}' -- refusing")
            if reg_g != got_g:
                raise SystemExit(f"[hsgate-fwd] the gate file hashes to {got_g}, the REGISTERED gate is {reg_g} -- refusing")
            g_rec = {"arm": "HSGATE8", "registered_sha256": reg_g, "registration_file": str(REG_HSG), "registration_file_mtime_epoch": os.path.getmtime(REG_HSG),
                     "gate_file": str(HSG_FROZEN), "gate_file_sha256": got_g, "gate_file_mtime_epoch": os.path.getmtime(HSG_FROZEN),
                     "passed_epoch": time.time(), "passed_iso": time.strftime("%FT%T%z"),
                     "rule": "PREREG_HSGATE s1: the gate is frozen and its sha256 registered by the Master Mind before any SCORE-side output; this record is written before the first one"}
            (work / "REGISTRATION_GATE_HSGATE8.json").write_text(json.dumps(g_rec, indent=1), encoding="utf-8")
            L2h, HLh = HG.load_l2_common()
            hsg = {"gate": json.loads(HSG_FROZEN.read_text(encoding="utf-8")), "rec": g_rec, "L2": L2h, "HL": HLh}
            for s in seeds:
                fac[s] = load_factored_inputs(s, True)                          # Amendment 5's scored inputs (L1 os-only bank + SELFSHARP trace)
                if s not in exl:
                    exl[s] = load_eval_extras(tree, s)
                landed[s] = {"H8": np.load(str(HG.LANDED_H8).format(seed=s), allow_pickle=False), "FX": np.load(str(HG.LANDED_FX).format(seed=s), allow_pickle=False),
                             "files": {"H8": str(HG.LANDED_H8).format(seed=s), "FX": str(HG.LANDED_FX).format(seed=s)}}
        arm_order = ["H2CN_DEPLOY8", "H2CN_NEAR", "H2CN_PERSURFACE", "RETIME_DEPLOY8", "RETIME_DEPLOY_PERSURFACE", "CONST_G1"] + (["H2CN_CAP8"] if cap8 else []) + \
                    (["H2CNL_DEPLOY8", "H2CNL_MUT8"] if h2cnl else []) + (["HSGATE8", "FACTORED_DEPLOY8", "HSGATE8_G0", "HSGATE8_G1"] if hsgate else [])
    else:
        for s in seeds:
            fac[s] = load_factored_inputs(s, scored)
        arm_order = (["FACTORED_DEPLOY8"] if scored else []) + ["OSONLY_DEPLOY8", "FACTORED_SSOS_DEPLOY8", "RETIME_DEPLOY8", "RETIME_DEPLOY_PERSURFACE", "CONST_G1"]
    # ---- the device (deploy8_forward_r8.forward's four hooks, arms replaced) ----------------------------------------------------------------------
    cache, cur = {}, {"seed": None, "surf": None}
    stash, kv, align, store, extra, compl = {}, {}, {}, {}, {}, {}
    gate_keys = {}
    orig_tl, orig_si, orig_ps = RT.tools_load, RT.surface_inputs, CR.prepare_seed
    orig_apply, orig_diag, orig_fw = RT.retime_apply, RT.retime_diag, RT.frozen_weights
    const1, const0 = {"knots_s": [0.5], "gains": [1.0]}, {"knots_s": [0.5], "gains": [0.0]}
    KIND_OF = {"H2CN_DEPLOY8": "H8", "H2CN_NEAR": "HN", "RETIME_DEPLOY8": "D8", "CONST_G1": "G1", "OSONLY_DEPLOY8": "OS", "FACTORED_SSOS_DEPLOY8": "FS",
               "FACTORED_DEPLOY8": "FX", "H2CN_PERSURFACE": "HP", "H2CN_CAP8": "HC", "H2CNL_DEPLOY8": "HL", "H2CNL_MUT8": "HM", "HSGATE8": "HS", "HSGATE8_G0": "Z0", "HSGATE8_G1": "Z1"}
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
            if h2cnl:
                # AMENDMENT 4c: H2CNL_DEPLOY8 / H2CNL_MUT8 = Amendment 4's 8-slot call with X := concat[X, softmax(lon6(logits))] of the SAME forward
                e_ = exl[seed]
                v0_ctx = ctx["v0"][rows]
                rj, jL = join_eval_extras(e_, S["clip"], S["ws"], v0_ctx, plan8, 0)
                rjm, _jm = join_eval_extras(e_, S["clip"], S["ws"], v0_ctx, plan8, e_["stride"])
                if not rj["PASS"]:
                    raise SystemExit(f"[sub-fwd] K-JOIN (eval) failed on {surf} seed {seed}: {rj}")
                P6 = HL.lon6_softmax(L2, e_[HL.LOGIT_KEY][jL])
                X16 = np.concatenate([X, P6], 1)
                hl = {"join": rj, "join_mutation_next_window": rjm, "ws_stride": e_["stride"], "extras_file_sha256": e_["sha256"],
                      "x16_base_cols_bit_equal_X": bool(X16[:, :X.shape[1]].tobytes() == np.ascontiguousarray(X).tobytes() and X.shape[1] == HL.N_X_BASE),
                      "lon6_softmax_rows_sum_to_1_maxdev": float(np.abs(P6.sum(1) - 1.0).max()), "lon6_softmax_mean": P6.mean(0).tolist()}
                for kk in ("HL", "HM"):
                    pr_ = lstages[kk]["preds"]
                    g2, g6 = pr_["S2"](X16, Z), pr_["S6"](X16, Z)
                    sL, chL = h2cn_progress(AL, prior8, g2, g6)
                    pL = RT.eval_polyline(geo, sL)
                    st[kk] = (pL.astype(plan8.dtype), pL)
                    st["s8"][kk] = sL
                    hl[f"running_max_changed_windows_{EIGHT[kk]}"] = int(chL.sum())
                    hl[f"frac_progress_clipped_at_0_{EIGHT[kk]}"] = float((np.concatenate([AL.arc(prior8[:, :4]) + g2, AL.arc(prior8[:, SLOTS['S6']])[:, 2:] + g6[:, 2:]], 1) < 0).any(1).mean())
                    hl[f"n_windows_cast_positions_differ_from_H2CN_DEPLOY8_{EIGHT[kk]}"] = int((st[kk][0] != st["H8"][0]).any((1, 2)).sum())
                    hl[f"mean_abs_progress_diff_vs_H2CN_DEPLOY8_m_{EIGHT[kk]}"] = float(np.abs(sL - s8).mean())
                ex["h2cnl"] = hl
            if hsgate:
                # PREREG_HSGATE s1: FACTORED_DEPLOY8 exactly as the factored branch builds it (Amendment 5), on the SAME rows as H2CN_DEPLOY8; HSGATE8 picks per window
                b = fac[seed]
                keys = list(zip(np.asarray(S["clip"], np.int64).tolist(), np.asarray(S["ws"], np.int64).tolist()))
                jb = np.array([b["b1"]["idx"].get(k, -1) for k in keys])
                jt = np.array([b["ts"]["idx"].get(k, -1) for k in keys])
                if (jb < 0).any() or (jt < 0).any():
                    raise SystemExit(f"[hsgate-fwd] {int((jb < 0).sum())} / {int((jt < 0).sum())} {surf} windows have no os-only bank / SELFSHARP trace entry")
                P_os8 = np.ascontiguousarray(b["b1"]["plan8"][jb]).astype(np.float32)
                P_ss8 = np.ascontiguousarray(b["ts"]["plan8"][jt]).astype(np.float32)
                P_fx = compose(RT, P_os8, P_ss8)
                st["in8"]["FX"] = P_fx
                st["FX"] = (RS.retime_plan(P_fx, prior8, T8, table, cast=True), RS.retime_plan(P_fx, prior8, T8, table, cast=False))
                sFX = RT.retime_progress(RT.polyline_geometry(P_fx)["L"][:, 1:], RT.prior_arc(prior8), g8)
                e_ = exl[seed]
                v0_ctx = ctx["v0"][rows]
                rjh, jH = join_eval_extras(e_, S["clip"], S["ws"], v0_ctx, plan8, 0)
                rjhm, _jhm = join_eval_extras(e_, S["clip"], S["ws"], v0_ctx, plan8, e_["stride"])
                if not rjh["PASS"]:
                    raise SystemExit(f"[hsgate-fwd] K-JOIN (eval extras) failed on {surf} seed {seed}: {rjh}")
                Xg, parts = HG.features(hsg["L2"], hsg["HL"], AL, e_[HG.LOGIT_KEY][jH], e_[HG.V0_KEY][jH], st["FX"][0][:, HG.S6], st["H8"][0][:, HG.S6])
                gw = np.asarray(HG.gate_decide(hsg["gate"], Xg), bool)
                gm = gw[:, None, None]
                st["HS"] = (np.where(gm, st["FX"][0], st["H8"][0]), np.where(gm, st["FX"][1], st["H8"][1]))
                st["in8"]["HS"] = np.where(gm, P_fx, plan8)
                st["s8"]["HS"] = np.where(gw[:, None], sFX, s8)
                st["Z0"], st["Z1"] = st["H8"], st["FX"]                          # the identity arms: g := 0 / g := 1 through the SAME device
                st["in8"]["Z1"] = P_fx
                st["s8"]["Z0"] = s8
                st["gate"] = gw
                lk = {}
                for kk in ("H8", "FX"):
                    z_ = landed[seed][kk]
                    lk[kk] = {"rows_aligned": bool(np.array_equal(z_[f"{surf}_clip"], np.asarray(S["clip"], np.int64)) and np.array_equal(z_[f"{surf}_ws"], np.asarray(S["ws"], np.int64))),
                              "pos8_bit_equal": bit_eq(z_[f"{surf}_pos8"], st[kk][0]), "pos64_bit_equal": bit_eq(z_[f"{surf}_pos64"], st[kk][1][:, slots]),
                              "plan8_bit_equal": bit_eq(z_[f"{surf}_plan8"], st["in8"].get(kk, plan8)), "prior8_bit_equal": bit_eq(z_[f"{surf}_prior8"], prior8)}
                mh = np.asarray(S["mask"], bool)
                ex["hsgate"] = {"join": rjh, "join_mutation_next_window": rjhm, "ws_stride": e_["stride"], "extras_file": e_["file"], "extras_file_sha256": e_["sha256"],
                                "landed": lk, "n_windows": int(len(gw)), "n_compliant": int(mh.sum()), "n_F_all": int(gw.sum()), "n_F_compliant": int(gw[mh].sum()),
                                "F_rate_all_windows": float(gw.mean()), "F_rate_compliant": (float(gw[mh].mean()) if mh.any() else None),
                                "d_mean": float(parts["d"].mean()), "features_finite": bool(np.isfinite(Xg).all()),
                                "windows_positions_differ_HSGATE8_vs_H2CN_DEPLOY8": int((st["HS"][0] != st["H8"][0]).any((1, 2)).sum()),
                                "windows_positions_differ_HSGATE8_vs_FACTORED_DEPLOY8": int((st["HS"][0] != st["FX"][0]).any((1, 2)).sum())}
                gate_keys[(seed, surf)] = (np.asarray(S["clip"], np.int64), np.asarray(S["ws"], np.int64), gw)
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
        for k_, nm_ in [("os", "os_model_plan")] + [(k2, EIGHT[k2]) for k2 in ("H8", "HN", "D8", "OS", "FS", "FX", "HC", "HL", "HM", "HS") if k2 in st]:
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
            if k in ("OS", "FS", "FX", "Z1"):
                d = D8.diag8(st["in8"][k][rows], st["prior8"][rows], st[k][1][rows], st[k][0][rows], RS.gain_at(T8, table), st["plan8"].dtype)
                d["geometry_over"] = "the 8-slot stage call on the arm's own input plan (os-only / composed): positions vs that plan's 8-point polyline"
                return d
            if k == "HS":
                return diag_progress(RT, st["in8"]["HS"][rows], st["HS"][1][rows], st["HS"][0][rows], st["s8"]["HS"][rows],
                                     "HSGATE8: per window the chosen arm's positions against THAT arm's 8-point polyline (FACTORED: the composed plan; H2CN: the model plan)")
            if k in ("H8", "HN", "HC", "HL", "HM", "Z0"):
                return diag_progress(RT, st["plan8"][rows], st[k][1][rows], st[k][0][rows], st["s8"][k][rows],
                                     "the 8-slot H2cn call: positions at all 8 slots against the 8-point model-plan polyline (progress after max(s,0) + running max)"
                                     + (" then clamped to the fed ceiling (Amendment 4b)" if k == "HC" else "")
                                     + (" with X + the LON6 softmax of the same forward (Amendment 4c)" if k in ("HL", "HM") else ""))
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
        fi = json.loads((C.WORK / "H2CN_FREEZE_INFO.json").read_text(encoding="utf-8")) if (C.WORK / "H2CN_FREEZE_INFO.json").exists() else {}
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
        need += ["K_REG", "K_DISJ", "K_PERSURF_identity_S2_vs_N1", "K_ATPROG_persurface_bit_equal"]
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
        if h2cnl:
            # AMENDMENT 4c s2.  In `need`: a failure WITHHOLDS (no FORWARD file, no verdict).
            fe = reg_rec["first_eval_output_epoch"]
            kr = {}
            for kk in ("HL", "HM"):
                g_ = lstages[kk]["gate"]
                kr[kk] = bool(len(g_["registered_sha256"]) == 64 and g_["registered_sha256"] == g_["stage_file_sha256"] and fe is not None
                              and g_["passed_epoch"] < fe and g_["registration_file_mtime_epoch"] < fe)
            controls["K_REG_H2CNL"], controls["K_REG_H2CNL_MUT"] = kr["HL"], kr["HM"]
            controls["K_REG_H2CNL_detail"] = {EIGHT[kk]: {**lstages[kk]["gate"], "first_eval_output_epoch": fe,
                                                          "first_eval_output_iso": time.strftime("%FT%T%z", time.localtime(fe or 0))} for kk in ("HL", "HM")}
            fil = json.loads(HL.INFO_L.read_text(encoding="utf-8")) if HL.INFO_L.exists() else {}
            fc = fil.get("controls") or {}
            controls["K_FREEZE_H2CNL"] = bool(fc.get("ALL_PASS") is True and fil.get("smoke") is False
                                              and (fil.get("stage_file") or {}).get("sha256") == lstages["HL"]["gate"]["stage_file_sha256"]
                                              and (fil.get("mutation_stage_file") or {}).get("sha256") == lstages["HM"]["gate"]["stage_file_sha256"])
            controls["K_FREEZE_H2CNL_detail"] = {"freeze_info": str(HL.INFO_L), "freeze_info_sha256": C.sha256_file(HL.INFO_L) if HL.INFO_L.exists() else None,
                                                 "ALL_PASS": fc.get("ALL_PASS"), "smoke": fil.get("smoke"), "K-DISJ": fc.get("K-DISJ"), "K-JOIN_PASS": fc.get("K-JOIN_PASS"),
                                                 "K-H2CN-REPRO": {s_: (fc.get("K-H2CN-REPRO") or {}).get(s_, {}).get("PASS") for s_ in SURFS},
                                                 "K-FIT": {k_: v_.get("PASS") for k_, v_ in (fc.get("K-FIT") or {}).items()},
                                                 "K-RELOAD": {k_: (v_.get("bit_equal") if isinstance(v_, dict) else v_) for k_, v_ in (fc.get("K-RELOAD") or {}).items()},
                                                 "K-DETERM": fc.get("K-DETERM"), "K-MUT": {k_: v_.get("PASS") for k_, v_ in (fc.get("K-MUT") or {}).items()},
                                                 "stage_sha256": (fil.get("stage_file") or {}).get("sha256"), "mutation_stage_sha256": (fil.get("mutation_stage_file") or {}).get("sha256")}
            hj = {k: v["h2cnl"] for k, v in extra.items() if isinstance(k, str)}
            controls["K_JOIN_EVAL"] = bool(len(hj) == 4 and all(v["join"]["PASS"] and not v["join_mutation_next_window"]["PASS"]
                                                                 and v["join_mutation_next_window"]["red_by_crosscheck_alone"] and v["x16_base_cols_bit_equal_X"] for v in hj.values()))
            controls["K_JOIN_EVAL_detail"] = {k: {"join": v["join"], "mutation_next_window": v["join_mutation_next_window"], "x16_base_cols_bit_equal_X": v["x16_base_cols_bit_equal_X"]}
                                              for k, v in hj.items()}
            need += ["K_REG_H2CNL", "K_REG_H2CNL_MUT", "K_FREEZE_H2CNL", "K_JOIN_EVAL"]
        if hsgate:
            # PREREG_HSGATE s2.  In `need`: a failure WITHHOLDS (no FORWARD file, no verdict, no bars).
            fe = reg_rec["first_eval_output_epoch"]
            gr = hsg["rec"]
            controls["K_REG_HSGATE"] = bool(len(gr["registered_sha256"]) == 64 and gr["registered_sha256"] == gr["gate_file_sha256"] == C.sha256_file(HSG_FROZEN)
                                            and fe is not None and gr["passed_epoch"] < fe and gr["registration_file_mtime_epoch"] < fe)
            controls["K_REG_HSGATE_detail"] = {**gr, "first_eval_output_epoch": fe, "first_eval_output_iso": time.strftime("%FT%T%z", time.localtime(fe or 0))}
            fit = json.loads(HSG_FIT_INFO.read_text(encoding="utf-8")) if HSG_FIT_INFO.exists() else {}
            fcs = fit.get("controls") or {}
            controls["K_FIT_HSGATE"] = bool(fit.get("ALL_PASS") is True and fit.get("smoke_partial_trace") is False and fcs.get("K-RELOAD") is True
                                            and (fit.get("gate_file") or {}).get("sha256") == gr["gate_file_sha256"])
            controls["K_FIT_HSGATE_detail"] = {"fit_info": str(HSG_FIT_INFO), "fit_info_sha256": (C.sha256_file(HSG_FIT_INFO) if HSG_FIT_INFO.exists() else None),
                                               "ALL_PASS": fit.get("ALL_PASS"), **{k_: (v_.get("PASS") if isinstance(v_, dict) else v_) for k_, v_ in fcs.items()
                                                                                    if k_ in ("K-FRESH-SS", "K-JOIN", "K-OOF_PASS", "K-PATH", "K-PRIOR", "K-DETERM", "K-RELOAD")}}
            mm0, mm1, n0, n1, pid = [], [], 0, 0, {}
            for s in seeds:
                for surf in SURFS:
                    st_ = stash[(s, surf)]
                    pid[f"{surf}|s{s}"] = {"G0_pos32": bit_eq(st_["Z0"][0], st_["H8"][0]), "G0_pos64": bit_eq(st_["Z0"][1], st_["H8"][1]),
                                           "G1_pos32": bit_eq(st_["Z1"][0], st_["FX"][0]), "G1_pos64": bit_eq(st_["Z1"][1], st_["FX"][1])}
                    for stn in SETS:
                        A_ = res["seeds"][str(s)]["surfaces"][surf]["sets"][stn]["arms"]
                        mm0 += [f"s{s}|{surf}|{stn}{m}" for m in D8.exact_equal(A_["HSGATE8_G0"], A_["H2CN_DEPLOY8"])]
                        mm1 += [f"s{s}|{surf}|{stn}{m}" for m in D8.exact_equal(A_["HSGATE8_G1"], A_["FACTORED_DEPLOY8"])]
                        n0 += D8.count_leaves(A_["HSGATE8_G0"])
                        n1 += D8.count_leaves(A_["HSGATE8_G1"])
            bars_id = {}
            for stn in SETS:
                for a_, b_ in (("HSGATE8_G0", "H2CN_DEPLOY8"), ("HSGATE8_G1", "FACTORED_DEPLOY8")):
                    ka, kb = f"{stn}|{a_}", f"{stn}|{b_}"
                    if ka in res.get("bars", {}) and kb in res.get("bars", {}):
                        bars_id[ka] = not D8.exact_equal(res["bars"][ka], res["bars"][kb])
            controls["K_ID0"] = bool(not mm0 and n0 > 0 and all(v["G0_pos32"] and v["G0_pos64"] for v in pid.values())
                                     and all(v for k_, v in bars_id.items() if "G0" in k_))
            controls["K_ID1"] = bool(not mm1 and n1 > 0 and all(v["G1_pos32"] and v["G1_pos64"] for v in pid.values())
                                     and all(v for k_, v in bars_id.items() if "G1" in k_))
            controls["K_ID_detail"] = {"positions": pid, "G0_cells_n_leaves": n0, "G0_cells_n_mismatches": len(mm0), "G0_first_mismatches": mm0[:10],
                                       "G1_cells_n_leaves": n1, "G1_cells_n_mismatches": len(mm1), "G1_first_mismatches": mm1[:10], "bars_equal": bars_id}
            hx = {k: v["hsgate"] for k, v in extra.items() if isinstance(k, str)}
            kls = json.loads(HG.FRESH_KLABSRC.read_text(encoding="utf-8"))
            controls["K_EVALFEAT"] = bool(kls.get("key") == HG.LOGIT_KEY == HL.LOGIT_KEY and (kls.get("K_LABSRC") or {}).get("pass") is True
                                          and exl[0]["sha256"] == (kls.get("K_LABSRC") or {}).get("extras_sha256")
                                          and all(Path(exl[s]["file"]).resolve() == Path(EVAL_EXTRAS.format(tree=tree, seed=s)).resolve() for s in seeds)
                                          and len(hx) == 4 and all(v["join"]["PASS"] and not v["join_mutation_next_window"]["PASS"]
                                                                   and v["join_mutation_next_window"]["red_by_crosscheck_alone"] and v["features_finite"] for v in hx.values()))
            controls["K_EVALFEAT_detail"] = {"klabsrc_file": str(HG.FRESH_KLABSRC), "klabsrc_sha256": C.sha256_file(HG.FRESH_KLABSRC), "klabsrc_key": kls.get("key"),
                                             "klabsrc_extras_sha256_s0": (kls.get("K_LABSRC") or {}).get("extras_sha256"),
                                             "eval_extras": {str(s): {"file": exl[s]["file"], "sha256": exl[s]["sha256"]} for s in seeds},
                                             "keys_used": [HG.LOGIT_KEY, HG.V0_KEY],
                                             "joins": {k: {"join": v["join"], "mutation_next_window": v["join_mutation_next_window"]} for k, v in hx.items()}}
            controls["K_LANDED"] = bool(len(hx) == 4 and all(all(all(x.values()) for x in v["landed"].values()) for v in hx.values()))
            controls["K_LANDED_detail"] = {k: v["landed"] for k, v in hx.items()}
            gs = {}
            for s in seeds:
                c2, w2, g2 = gate_keys[(s, "S2")]
                c6, w6, g6 = gate_keys[(s, "S6")]
                k2 = {(int(a__), int(b__)): bool(c__) for a__, b__, c__ in zip(c2, w2, g2)}
                miss = sum((int(a__), int(b__)) not in k2 for a__, b__ in zip(c6, w6))
                dif = sum(k2.get((int(a__), int(b__))) != bool(c__) for a__, b__, c__ in zip(c6, w6, g6))
                gs[str(s)] = {"n_S6": int(len(g6)), "S6_rows_missing_in_S2": int(miss), "decisions_differ": int(dif)}
            controls["K_GATE_SURF"] = bool(all(v["S6_rows_missing_in_S2"] == 0 and v["decisions_differ"] == 0 for v in gs.values()))
            controls["K_GATE_SURF_detail"] = gs
            trh = {str(s): trace_checks(fac[s]["rec"], True) for s in seeds}
            controls["K_TRACE"] = bool(all(all(v.values()) for v in trh.values()))
            controls["K_TRACE_detail"] = {"checks": trh, "inputs": {str(s): fac[s]["rec"] for s in seeds}}
            fdh = {str(s): feed_checks(s, True) for s in seeds}
            controls["K_FEED_perframe_full"] = bool(all(r["perframe_full"] for v in fdh.values() for r in v.values()))
            controls["K_FEED_detail"] = fdh
            need += ["K_REG_HSGATE", "K_FIT_HSGATE", "K_ID0", "K_ID1", "K_EVALFEAT", "K_LANDED", "K_GATE_SURF", "K_TRACE", "K_FEED_perframe_full"]
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
    res.update({"tool": "ship_r8_hsgate_forward.py", "mode": mode, "scored_read": bool(scored), "tree_name": tree, "n_boot": RT.N_BOOT, "bootstrap_seed": 0, "seeds_run": list(seeds),
                "role": ("MUTATION RUN (deliberate regression; NOT a result)" if mutate else
                         (f"SHIP-R8 AMENDMENT 4b: arm positions replaced; scored arm H2CN_CAP8 (H2CN_DEPLOY8 re-run beside it for the reproduction)" if cap8 else
                          f"SHIP-R8 AMENDMENT 4c: arm positions replaced; scored arm H2CNL_DEPLOY8, mutation arm H2CNL_MUT8 beside (H2CN_DEPLOY8 re-run beside them for the reproduction)" if h2cnl else
                          f"PREREG_HSGATE PHASE 2: scored arm HSGATE8 (per-window gate between FACTORED_DEPLOY8 and H2CN_DEPLOY8); identity arms HSGATE8_G0 / HSGATE8_G1 beside" if hsgate else
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
        res["h2cn_stage"] = {"file": str(C.STAGE_PT), "sha256": reg_rec["gate"]["stage_file_sha256"], "registered": reg_rec["gate"]["registered_sha256"]}
    if cap8:
        res["cap8"] = True
    if h2cnl:
        res["h2cnl"] = True
        for kk, nm in (("HL", "h2cnl_stage"), ("HM", "h2cnl_mut_stage")):
            g_ = lstages[kk]["gate"]
            res[nm] = {"file": g_["stage_file"], "sha256": g_["stage_file_sha256"], "registered": g_["registered_sha256"], "registration_file": g_["registration_file"]}
        res["h2cnl_eval_extras"] = {str(s): {"file": exl[s]["file"], "sha256": exl[s]["sha256"]} for s in seeds}
        res["h2cnl_x_extra"] = HL.X_EXTRA
    if hsgate:
        res["hsgate"] = {"gate_file": str(HSG_FROZEN), "gate_sha256": hsg["rec"]["gate_file_sha256"], "registered": hsg["rec"]["registered_sha256"],
                         "registration_file": str(REG_HSG), "prereg_sha256": seals["PREREG_HSGATE"]["sha256"], "hsgate_common_sha256": HSG_COMMON_SHA,
                         "F_rate": {k: {kk: v["hsgate"][kk] for kk in ("n_windows", "n_compliant", "n_F_all", "n_F_compliant", "F_rate_all_windows", "F_rate_compliant")}
                                    for k, v in extra.items() if isinstance(k, str)},
                         "landed_inputs": {str(s): {k: {"file": f, "sha256": C.sha256_file(f)} for k, f in landed[s]["files"].items()} for s in seeds}}
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
    arms_npz = [a for a in arm_order if a in ("H2CN_DEPLOY8", "H2CN_NEAR", "RETIME_DEPLOY8", "FACTORED_DEPLOY8", "OSONLY_DEPLOY8", "H2CN_CAP8", "H2CNL_DEPLOY8", "H2CNL_MUT8", "HSGATE8")]
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
            meta = {"tool": "ship_r8_hsgate_forward.py", "tree": tree, "inference_seed": seed, "arm": arm, "tier": RT.TIER, "mode": mode, "table_sha256": table["sha256"],
                    "stage_sha256": stage_meta["stage_sha256"], "model_slot_horizons_ticks": HZ8, "slot_instants_s": T8.tolist(), "surface_slots": SLOTS,
                    "consumer_contract": D8.CONTRACT.replace("RETIME_DEPLOY8", arm),
                    "note": "<S>_os is the battery panel's os (row alignment); <S>_plan8 is the plan the arm was built on (the model plan, or the os-only / composed plan)"}
            if mode == "h2cn":
                meta["h2cn_stage_sha256"] = reg_rec["gate"]["stage_file_sha256"]
            if arm == "HSGATE8":
                meta["hsgate_gate_sha256"] = hsg["rec"]["gate_file_sha256"]
                meta["note_gate"] = "<S>_gate: the registered gate's per-window decision (True = FACTORED_DEPLOY8's positions, False = H2CN_DEPLOY8's)"
                for surf_ in SURFS:
                    arrs[f"{surf_}_gate"] = np.asarray(stash[(seed, surf_)]["gate"], bool)
            if arm in ("H2CNL_DEPLOY8", "H2CNL_MUT8"):
                meta["h2cnl_stage_sha256"] = lstages["HL" if arm == "H2CNL_DEPLOY8" else "HM"]["gate"]["stage_file_sha256"]
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
    ap.add_argument("--h2cnl", action="store_true", help="h2cn: AMENDMENT 4c -- also build H2CNL_DEPLOY8 + H2CNL_MUT8 (the registered H2cn-L stages) + their controls")
    ap.add_argument("--work-root", default=None, help="where the transient RT.forward work dir lives (default: --out-root); removed at the end")
    ap.add_argument("--hsgate", action="store_true", help="h2cn: PREREG_HSGATE -- also build FACTORED_DEPLOY8 + HSGATE8 (+ G0 / G1 identity arms) + their controls")
    a = ap.parse_args()
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
    if a.h2cnl and (a.mode != "h2cn" or a.cap8):
        raise SystemExit("--h2cnl applies to --mode h2cn only, without --cap8")
    if a.h2cnl and Path(a.out_root).resolve() in ((C.WORK / "out").resolve(), (C.WORK / "out_4b").resolve()):
        raise SystemExit("[sub-fwd] REFUSING: --h2cnl must write to a NEW --out-root (the landed out/ and out_4b/ are never overwritten)")
    if a.hsgate:
        if a.mode != "h2cn" or a.cap8 or a.h2cnl or a.mutate:
            raise SystemExit("--hsgate applies to --mode h2cn only, without --cap8 / --h2cnl / --mutate")
        okroots = (HG.WORK.resolve(), HG.SCRATCH.resolve())
        for p_ in [Path(a.out_root)] + ([Path(a.work_root)] if a.work_root else []):
            rp = p_.resolve()
            if not any(rp == r_ or r_ in rp.parents for r_ in okroots):
                raise SystemExit(f"[hsgate-fwd] REFUSING: {p_} is outside {okroots}")
    else:
        raise SystemExit("[hsgate-fwd] this copy runs ONLY with --hsgate (the 4c reads live in ship_r8_sub/ and are never re-run from here)")
    return forward(a.mode, a.tree, seeds, a.scored, a.mutate, Path(a.out_root), Path(a.registered_stage), a.cap8, a.h2cnl, Path(a.work_root) if a.work_root else None,
                   a.hsgate)


if __name__ == "__main__":
    sys.exit(main())
