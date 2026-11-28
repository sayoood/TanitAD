"""stoporc_read.py -- PREREAD_V8F_STOPORC (theory/stopbal/oracle/PREREAD_V8F_STOPORC.md, frozen; sha256 in PREREAD_V8F_STOPORC.sha256).  CPU, zero training.

Does O plan a stop when T's code says stop?  Per arm, per window of the STOPDEC first-tick batches (same batch build, STOPBAL tree, same windows):
  OWN    = forward_infer plan (T's own code)                       IDENT = forward_diag(OWN code)  (control 1: equals OWN, max abs 0.0)
  F-GT   = OWN code with goal[STOP_POINT] := 1.0 and the lon part := one-hot (STOP if the GT allowed lon set holds STOP, else the allowed class T ranks highest)
  F-GOAL = goal entry only;  F-LON = lon part only (same rule);  MUT = the red-arm mutant that forces nothing (F-GT := OWN).
  Lat, the other goal entries and the constraint cells stay T's own.  Plan of a code = forward_diag(batch, code)["traj30"].
Stage `run` writes one npz per arm to the scratch folder (per-window rows, before any analysis); stage `analyse` reads them, runs the controls and writes RESULT_STOPORC.json.
Usage:  python stoporc_read.py --stage run --tags BAL-NULL_s0 ... [--smoke N]
        python stoporc_read.py --stage analyse --out RESULT_STOPORC.json
The pure logic (force_codes, check_forced, analyse_cells, outcome) needs no torch and no model: the unit tests call it directly.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.dirname(os.path.dirname(HERE))
READ = os.path.join(THEORY, "stopbal", "read")
SCRATCH = os.environ.get("STORC_SCRATCH", "F:/TanitAD-scratch/stoporc_build")
TREE = os.environ.get("STOPBAL_TREE", "F:/TanitAD-scratch/stopbal_read/tree")
T1 = os.environ.get("STOPBAL_DUMPS", "F:/TanitAD-scratch/stopbal_read/t1")
STOPDEC_OUT = os.path.join(READ, "out", "stopdec")
if READ not in sys.path:
    sys.path.insert(0, READ)

N_G, N_LAT, N_LON, N_CONS = 22, 5, 7, 26                  # verified against the tree source in run_arm (assert)
CODE_DIM = N_G + N_LAT + N_LON + N_CONS
STOP_GI = 6                                                # GOAL22 index of STOP_POINT (verified: D.GOAL22.index)
LON_STOP_I = 2                                             # LON9 index of STOP (verified: D.LON9.index)
LON_SL = slice(N_G + N_LAT, N_G + N_LAT + N_LON)           # code[:, 27:34]
MODES = ("fgt", "fgoal", "flon", "mutant")
FORCED = ("fgt", "fgoal", "flon")
TLSTOP_SP = 1                                              # strat_tlstop level of STOP_POINT
EXPECT = {"n_stop": 117, "n_ctl": 2234}
NB, SEED = 2000, 0
BANDS = {"operative_1_10": (0, 10), "tactical_11_30": (10, 30)}
METRICS = ("ade_m", "fde_m", "speed_mae_mps", "accel_mae_mps2", "heading_mae_deg", "yaw_rate_mae_degps")
FAMILY = {"ade_m": "PRIMARY", "fde_m": "PRIMARY", "speed_mae_mps": "LONGITUDINAL", "accel_mae_mps2": "LONGITUDINAL", "heading_mae_deg": "LATERAL", "yaw_rate_mae_degps": "LATERAL"}
CELLS = [f"{b}:{m}" for b in BANDS for m in METRICS]
PRIMARY_CELL = "tactical_11_30:ade_m"
DECIDE = ("BAL-NULL_s0", "BAL-NULL_s1")
FLOOR_ARM = "BAL-NULL_rep"
LEVERS = ("BAL-G_s0", "BAL-G_s1", "BAL-L_s0", "BAL-L_s1")
ALL_TAGS = (*DECIDE, FLOOR_ARM, *LEVERS)
TACTICAL_FAMILY = {"status": "n = 0", "reason": "T's decision is the forced input of this read"}
STRATEGIC_FAMILY = {"status": "PAUSED", "reason": "PI item 25 (PI_DECISION_QUEUE.md): the strategic layer is switched off temporarily; never a pass, never omitted"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_ci():
    """taniteval/ci.py is pure numpy: loaded from its file (tree: env STOPBAL_TREE, then the STOPBAL export, then the OTBAND export)."""
    for t in (os.environ.get("STOPBAL_TREE"), "F:/TanitAD-scratch/stopbal_read/tree", "F:/TanitAD-scratch/otband_read/tree"):
        if t and os.path.exists(t + "/taniteval/taniteval/ci.py"):
            spec = importlib.util.spec_from_file_location("stoporc_ci", t + "/taniteval/taniteval/ci.py")
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            m.SOURCE = t + "/taniteval/taniteval/ci.py"
            return m
    raise RuntimeError("taniteval/ci.py not found in any read tree")


# ------------------------------------------------------------------------------------------------------------------ the forced codes (numpy, float32)
def lon_target(own_lon, lon_al):
    """-> (target index [N], case [N] in {0: STOP allowed, 1: other allowed class (T's highest prob), 2: none allowed (lon part left as T's own)})."""
    al = np.asarray(lon_al, bool)
    masked = np.where(al, own_lon, -np.inf)
    alt = masked.argmax(1)
    has_stop = al[:, LON_STOP_I]
    tgt = np.where(has_stop, LON_STOP_I, alt)
    case = np.where(has_stop, 0, np.where(al.any(1), 1, 2))
    return tgt.astype(np.int64), case.astype(np.int64)


def force_codes(own, lon_al, mode):
    """own [N, 60] float32 (T's code), lon_al [N, 7] bool -> forced code [N, 60] (a copy).  mode in MODES (see the module docstring)."""
    assert mode in MODES, mode
    own = np.asarray(own)
    c = own.copy()
    if mode == "mutant":
        return c
    if mode in ("fgt", "fgoal"):
        c[:, STOP_GI] = 1.0
    if mode in ("fgt", "flon"):
        tgt, case = lon_target(own[:, LON_SL], lon_al)
        oh = np.zeros((len(own), N_LON), own.dtype)
        oh[np.arange(len(own)), tgt] = 1.0
        c[:, LON_SL] = np.where((case < 2)[:, None], oh, own[:, LON_SL])
    return c


def allowed_cols(mode):
    m = np.zeros(CODE_DIM, bool)
    if mode in ("fgt", "fgoal"):
        m[STOP_GI] = True
    if mode in ("fgt", "flon"):
        m[LON_SL] = True
    return m


def check_forced(own, forced, lon_al, mode, sel=None):
    """Control 2 on [sel] windows: forced differs from own ONLY in the named entries, takes the stated values there, every other entry is equal bit for bit.  -> (counts, problems)."""
    own, forced, lon_al = np.asarray(own), np.asarray(forced), np.asarray(lon_al, bool)
    if sel is not None:
        own, forced, lon_al = own[sel], forced[sel], lon_al[sel]
    allowed = allowed_cols(mode)
    prob = []
    if own.shape != forced.shape or own.dtype != forced.dtype:
        return {}, [f"{mode}: shape/dtype {own.shape}/{own.dtype} vs {forced.shape}/{forced.dtype}"]
    differ = own.view(np.uint32) != forced.view(np.uint32) if own.dtype == np.float32 else own != forced      # bit comparison
    n_bad = int(differ[:, ~allowed].sum())
    if n_bad:
        prob.append(f"{mode}: {n_bad} entries differ outside the named entries")
    if mode in ("fgt", "fgoal") and not (forced[:, STOP_GI] == 1.0).all():
        prob.append(f"{mode}: goal[STOP_POINT] is not 1.0 on every window")
    tgt, case = lon_target(own[:, LON_SL], lon_al)
    if mode in ("fgt", "flon"):
        oh = np.zeros((len(own), N_LON), own.dtype)
        oh[np.arange(len(own)), tgt] = 1.0
        want = np.where((case < 2)[:, None], oh, own[:, LON_SL])
        if not np.array_equal(forced[:, LON_SL], want):
            prob.append(f"{mode}: the lon part is not the stated one-hot")
    if mode == "mutant" and not np.array_equal(forced.view(np.uint32), own.view(np.uint32)):
        prob.append("mutant: the forced code is not OWN bit for bit")
    gd, ld = differ[:, STOP_GI], differ[:, LON_SL].any(1)
    ncols = differ.sum(1)
    counts = {"n_windows": int(len(own)), "goal_entry_changed": int(gd.sum()), "lon_part_changed": int(ld.sum()), "both_changed": int((gd & ld).sum()), "neither_changed": int((~gd & ~ld).sum()),
              "changed_entries_per_window": {str(k): int((ncols == k).sum()) for k in np.unique(ncols)},
              "lon_case_counts": {"stop_in_allowed_set": int((case == 0).sum()), "other_allowed_class_by_T_prob": int((case == 1).sum()), "no_allowed_class_lon_left_own": int((case == 2).sum())},
              "n_goal_already_ge_1": int((own[:, STOP_GI] >= 1.0).sum())}
    return counts, prob


# ------------------------------------------------------------------------------------------------------------------ estimator and reading logic (numpy)
def paired(CI, a, b, eid):
    """forced minus own.  point is the unrounded mean difference on finite pairs; the interval and `separated` come from taniteval.ci (B 2,000, seed 0)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return {"status": "NOT-APPLICABLE", "n_windows": 0}
    r = CI.paired_episode_cluster_bootstrap(a[m], b[m], np.asarray(eid)[m], n_boot=NB, seed=SEED)
    return {"point": float((a[m] - b[m]).mean()), "delta_ci_rounded": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]),
            "mean_own": float(b[m].mean()), "mean_forced": float(a[m].mean()), "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(np.asarray(eid)[m])))}


def analyse_cells(CI, cells, sets, eid):
    """cells[variant][cell] -> [N] vectors, variants include 'own' and the forced ones present; sets {'stop': mask, 'ctl': mask}.  -> {forced: {set: {cell: paired dict}}}."""
    out = {}
    for v in FORCED + ("mutant",):
        if v not in cells:
            continue
        out[v] = {}
        for sn, msk in sets.items():
            idx = np.where(msk)[0]
            out[v][sn] = {c: paired(CI, cells[v][c][idx], cells["own"][c][idx], np.asarray(eid)[idx]) for c in CELLS}
    return out


def is_gain(d, floor):
    """the primary's gain: separated, better side (lower ADE), above the floor.  No floor => never a gain."""
    if floor is None or not np.isfinite(floor) or "point" not in d:
        return False
    return bool(d["separated"] and d["point"] < 0 and abs(d["point"]) > floor)


def outcome(arms, controls_ok=True):
    """arms = {tag: {'primary': paired dict (stop set, tactical ADE, F-GT minus OWN), 'reach': paired dict (control set, same cell)}} for the three NULL arms.  Frozen table of the card."""
    need = (*DECIDE, FLOOR_ARM)
    miss = [t for t in need if t not in arms or "point" not in arms[t]["primary"] or "separated" not in arms[t]["reach"]]
    if miss:
        return {"outcome": "NOT-EVALUATED", "reason": f"missing or empty cells for {miss}"}
    floor = abs(arms[FLOOR_ARM]["primary"]["point"] - arms[DECIDE[0]]["primary"]["point"])
    gain = {t: is_gain(arms[t]["primary"], floor) for t in DECIDE}
    reach = {t: bool(arms[t]["reach"]["separated"]) for t in DECIDE}
    g = list(gain.values())
    r = list(reach.values())
    if all(g):
        o = "O-FOLLOWS"
    elif not any(g) and all(r):
        o = "O-MISREADS"
    elif not any(g) and not any(r):
        o = "O-IGNORES"
    else:
        o = "SEEDS-DISAGREE"
    if not controls_ok:
        o_final = "INVALID-CONTROLS"
    else:
        o_final = o
    return {"outcome": o_final, "rule_outcome_if_controls_held": o, "floor_m": float(floor), "floor_rule": "|delta(NULL_rep) - delta(NULL_s0)| on the primary cell, read at seed 0, applied to both seeds",
            "primary_gain_by_seed": gain, "reaches_O_separated_by_seed": reach}


# ------------------------------------------------------------------------------------------------------------------ the model pass (CPU)
def run_arm(tag, smoke=0):
    import stopdec_arm as SA
    SA.install()
    sys.path.insert(0, os.path.join(THEORY, "stopdec"))
    sys.path.insert(0, os.path.join(THEORY, "speedsrc"))
    import speedsrc_read as SS
    import torch
    TK, M, D, BL = SS.TK, SS.M, SS.D, SS.BL
    import tanitad
    assert os.path.abspath(tanitad.__file__).replace("\\", "/").startswith(TREE), tanitad.__file__
    assert (M.N_GOAL, M.N_LAT, M.N_LON, M.N_CONS) == (N_G, N_LAT, N_LON, N_CONS) and D.GOAL22.index("STOP_POINT") == STOP_GI and D.LON9.index("STOP") == LON_STOP_I
    assert SS.GI["STOP_POINT"] == STOP_GI
    E = TK.load_eval_module()
    A, ck_path, ddir = SS.tag_paths(tag)
    got = sha256(ck_path)
    man = os.path.dirname(ck_path) + "/BANK_MANIFEST.json"
    man_ok = None
    if os.path.exists(man):
        man_ok = json.load(open(man, encoding="utf-8"))["files"]["ckpt_last.pt"]["sha256"] == got
        assert man_ok, (tag, "checkpoint sha256 differs from its bank manifest")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    model.eval()
    tracks = D.load_native_tracks(A["tracks"])
    J = D.load_join(A["eval-join"])
    store = D.ClipStore(A["eval-cache"], "eval", lru=32, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(A["eval-grid"], J)
    if smoke:
        wins = [wins[i] for i in np.linspace(0, len(wins) - 1, smoke).round().astype(int)]
    proxy = E._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    a_t, a_s = E.ages_for(0, A["schedule"])
    N = len(wins)
    o = {"g": np.zeros((N, 30, 2)), "ci": np.zeros(N, np.int64), "ws": np.zeros(N, np.int64), "has": np.zeros(N, bool), "v0": np.zeros(N), "n_present": np.zeros(N, np.int64),
         "cons_gt": np.full((N, M.N_CONS), np.nan), "tl_hit": np.zeros(N, bool), "stop_hit": np.zeros(N, bool), "lon_al": np.zeros((N, N_LON), bool),
         "own_code": np.zeros((N, CODE_DIM), np.float32), "plan_own": np.zeros((N, 30, 2), np.float32), "plan_ident": np.zeros((N, 30, 2), np.float32)}
    for m in MODES:
        o[f"code_{m}"] = np.zeros((N, CODE_DIM), np.float32)
        o[f"plan_{m}"] = np.zeros((N, 30, 2), np.float32)
    t0 = time.time()
    for s in range(0, N, 32):
        chunk = wins[s:s + 32]
        SS.wait_free(3.0, 3600)
        st = E.make_states(store, tracks, J, chunk, A["nav"])
        items = [E.build_item(ds, proxy, x, 0, a_t, a_s, A["scene"], A["limit"], None) for x in st]
        full = D.collate(items)
        batch = {k: full[k] for k in E.MODEL_KEYS}
        lt = full["lab_t"]
        has = lt["has"].bool()
        sl = slice(s, s + len(chunk))
        with torch.no_grad():
            inf = model.forward_infer(batch)
            own = inf["code"]
            assert own.shape[1] == CODE_DIM and own.dtype == torch.float32, (own.shape, own.dtype)
            own_np = own.numpy().copy()
            lon_al = lt["lon_al"].bool().numpy()
            o["plan_own"][sl] = inf["traj30"].float().numpy()
            o["plan_ident"][sl] = model.forward_diag(batch, own)["traj30"].float().numpy()
            for m in MODES:
                c = force_codes(own_np, lon_al, m)
                o[f"code_{m}"][sl] = c
                o[f"plan_{m}"][sl] = model.forward_diag(batch, torch.from_numpy(c))["traj30"].float().numpy()
            present = ~torch.isnan(lt["cons"])
            pm = (present[:, SS.SP_COLS] & has[:, None]).numpy()
            gy, gw = lt["goal_y"], lt["goal_w"]
            hit = ((gy > 0.5) & (gw > 0.5)).numpy() & has[:, None].numpy()
        o["own_code"][sl] = own_np
        o["lon_al"][sl] = lon_al
        o["has"][sl] = has.numpy()
        o["v0"][sl] = batch["p10f"][:, -1, 0].double().numpy()
        o["n_present"][sl] = pm.sum(1)
        o["cons_gt"][sl] = np.where(has.numpy()[:, None], lt["cons"].double().numpy(), np.nan)
        o["tl_hit"][sl] = hit[:, [SS.GI[g] for g in SS.TL_GOALS]].any(1)
        o["stop_hit"][sl] = hit[:, STOP_GI]
        for i, (ci, tt, _, _) in enumerate(chunk):
            o["g"][s + i] = store.get(str(J["clip"][ci]))["gt"][tt]
            o["ci"][s + i], o["ws"][s + i] = ci, tt
        print(f"  {tag} [{s + len(chunk)}/{N}] {time.time() - t0:.0f}s free {TK.free_gb():.1f} GB", flush=True)
    # tie the eval windows to the dump and read the dump's own plan (control 1b)
    if smoke:
        o["eid"], o["ol_vs_own_max_abs_m"] = o["ci"].copy(), np.nan
    else:
        Z = BL.load_dir(ddir)
        key_d = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(Z["clip_index"], Z["ws"]))}
        key_m = [(int(c), int(w)) for c, w in zip(o["ci"], o["ws"])]
        assert set(key_d) == set(key_m) and len(key_d) == len(key_m), "window set differs from the dump"
        idx = np.array([key_d[x] for x in key_m])
        o["eid"] = Z["eid"][idx]
        o["ol_vs_own_max_abs_m"] = float(np.abs(o["plan_own"].astype(np.float64) - Z["ol"][idx]).max())
        o["g_vs_dump_max_abs_m"] = float(np.abs(o["g"] - Z["g"][idx]).max())
    o["ckpt_sha256"], o["manifest_ok"], o["tree"], o["tanitad_file"] = got, bool(man_ok), TREE, tanitad.__file__
    return o


def run_stage(tags, smoke):
    os.makedirs(SCRATCH, exist_ok=True)
    for tag in tags:
        print(f"arm {tag}", flush=True)
        o = run_arm(tag, smoke)
        np.savez(os.path.join(SCRATCH, f"{'smoke_' if smoke else ''}stoporc_{tag}.npz"), **{k: np.asarray(v) for k, v in o.items()})
        print(f"banked {tag}", flush=True)
        if smoke:                                           # a code check on N windows, never a result
            allw = np.ones(len(o["ci"]), bool)
            pr = [p for m in MODES for p in check_forced(o["own_code"], o[f"code_{m}"], o["lon_al"], m, allw)[1]]
            print("SMOKE-CONTROLS", json.dumps({"ident_vs_own_max_abs_m": float(np.abs(o["plan_ident"].astype(np.float64) - o["plan_own"]).max()),
                                                "mutant_plan_vs_own_max_abs_m": float(np.abs(o["plan_mutant"].astype(np.float64) - o["plan_own"]).max()),
                                                "fgt_plan_vs_own_max_abs_m": float(np.abs(o["plan_fgt"].astype(np.float64) - o["plan_own"]).max()), "forced_code_problems": pr}), flush=True)


# ------------------------------------------------------------------------------------------------------------------ analysis
def window_sets(z):
    ok = z["has"] & (z["n_present"] > 0)
    import speedsrc_read as SS
    S, _ = SS.build_strata(z["g"], z["v0"], z["cons_gt"], z["tl_hit"], z["stop_hit"])
    sp = (S["tlstop"] == TLSTOP_SP) & ok
    return ok, sp, ok & ~sp, S


def arm_cells(z, BL):
    cells = {}
    for v in ("own", "ident") + FORCED + ("mutant",):
        plan = z[f"plan_{v}"].astype(np.float64)
        cells[v] = {}
        for bn, (a, b) in BANDS.items():
            r = BL.band_rows(plan, z["g"], a, b)
            for m in METRICS:
                cells[v][f"{bn}:{m}"] = r[m]
    return cells


def analyse_arm(tag, z, CI, BL):
    ok, sp, ctl, S = window_sets(z)
    sd = dict(np.load(os.path.join(STOPDEC_OUT, f"stopdec_{tag}.npz")))
    cx = {"windows_equal_stopdec": bool(np.array_equal(sd["clip_index"], z["ci"]) and np.array_equal(sd["ws"], z["ws"]) and np.array_equal(sd["has"], z["has"])),
          "tlstop_equal_stopdec": bool(np.array_equal(sd["strat_tlstop"], S["tlstop"])),
          "stop_point_set_equal_stopdec": bool(np.array_equal(sp, (sd["strat_tlstop"] == TLSTOP_SP) & sd["has"] & (np.isfinite(sd["err_mps"]).sum(1) > 0))),
          "own_goal_prob_max_abs_vs_stopdec": float(np.abs(z["own_code"][:, :N_G].astype(np.float64) - sd["goal_prob"]).max()),
          "own_lon_prob_max_abs_vs_stopdec": float(np.abs(z["own_code"][:, LON_SL].astype(np.float64) - sd["lon_prob"]).max())}
    cells = arm_cells(z, BL)
    eid = z["eid"]
    res = analyse_cells(CI, cells, {"stop": sp, "ctl": ctl}, eid)
    ctrl = {"n_windows": int(len(z["ci"])), "n_stop_point": int(sp.sum()), "n_control_set": int(ctl.sum()), "n_stop_point_episodes": int(len(np.unique(eid[sp]))),
            "ckpt_sha256": str(z["ckpt_sha256"]), "manifest_ok": bool(z["manifest_ok"]), "tanitad_file": str(z["tanitad_file"]),
            "ident_vs_own_max_abs_m": float(np.abs(z["plan_ident"].astype(np.float64) - z["plan_own"]).max()),
            "own_plan_vs_dump_ol_max_abs_m": float(z["ol_vs_own_max_abs_m"]), "stopdec_cross_check": cx, "mutant_code_equals_own": {}, "forced_code_checks": {}, "problems": []}
    P = ctrl["problems"]
    if ctrl["ident_vs_own_max_abs_m"] != 0.0:
        P.append(f"control 1: IDENT vs OWN max abs {ctrl['ident_vs_own_max_abs_m']}")
    if ctrl["own_plan_vs_dump_ol_max_abs_m"] == ctrl["own_plan_vs_dump_ol_max_abs_m"] and ctrl["own_plan_vs_dump_ol_max_abs_m"] > 1e-4:
        P.append(f"own plan vs dump ol {ctrl['own_plan_vs_dump_ol_max_abs_m']}")
    if (ctrl["n_stop_point"], ctrl["n_control_set"]) != (EXPECT["n_stop"], EXPECT["n_ctl"]):
        P.append(f"window counts {ctrl['n_stop_point']} / {ctrl['n_control_set']} != {EXPECT['n_stop']} / {EXPECT['n_ctl']}")
    if not (cx["windows_equal_stopdec"] and cx["tlstop_equal_stopdec"] and cx["stop_point_set_equal_stopdec"]):
        P.append("windows / strata differ from the STOPDEC npz")
    if cx["own_goal_prob_max_abs_vs_stopdec"] > 1e-5 or cx["own_lon_prob_max_abs_vs_stopdec"] > 1e-5:
        P.append("own code differs from the STOPDEC npz probabilities")
    for m in MODES:
        for sn, msk in (("stop", sp), ("ctl", ctl)):
            cnt, pr = check_forced(z["own_code"], z[f"code_{m}"], z["lon_al"], m, msk)
            ctrl["forced_code_checks"][f"{m}:{sn}"] = cnt
            P.extend(f"control 2 [{sn}]: {p}" for p in pr)
    red = res["mutant"]
    ctrl["red_arm_mutant"] = {"primary_delta_point": red["stop"][PRIMARY_CELL].get("point"), "reaches_O_separated": red["ctl"][PRIMARY_CELL].get("separated"),
                              "all_mutant_cells_exactly_zero": bool(all(red[s][c].get("point") == 0.0 for s in red for c in CELLS))}
    if not (ctrl["red_arm_mutant"]["primary_delta_point"] == 0.0 and ctrl["red_arm_mutant"]["reaches_O_separated"] is False and ctrl["red_arm_mutant"]["all_mutant_cells_exactly_zero"]):
        P.append("control 3: the mutant arm does not read exactly 0.0 / not separated")
    ctrl["pass"] = not P
    # context beside the cells (descriptive): own plan's tactical ADE on the two sets
    own_tac = cells["own"][PRIMARY_CELL]
    arm = {"controls": ctrl, "forced_minus_own": {v: res[v] for v in FORCED}, "mutant_minus_own": res["mutant"],
           "primary": res["fgt"]["stop"][PRIMARY_CELL], "reach": res["fgt"]["ctl"][PRIMARY_CELL],
           "own_tactical_ade_mean": {"stop": float(np.nanmean(own_tac[sp])), "ctl": float(np.nanmean(own_tac[ctl]))}}
    return arm


def analyse_stage(out_json, tags):
    sys.path.insert(0, READ)
    import stopdec_arm as SA
    SA.install()
    sys.path.insert(0, os.path.join(THEORY, "stopdec"))
    sys.path.insert(0, os.path.join(THEORY, "speedsrc"))
    import speedsrc_read as SS
    BL = SS.BL
    CI = BL.CI
    arms, banked = {}, {}
    for tag in tags:
        p = os.path.join(SCRATCH, f"stoporc_{tag}.npz")
        if not os.path.exists(p):
            banked[tag] = "per-arm npz absent"
            continue
        z = dict(np.load(p, allow_pickle=False))
        arms[tag] = analyse_arm(tag, z, CI, BL)
        banked[tag] = {"scratch_npz_sha256": sha256(p)}
        print(f"analysed {tag}: controls pass {arms[tag]['controls']['pass']}", flush=True)
    sets_ok = {a["controls"]["n_stop_point"] for a in arms.values()} | set()
    ctl_ok = all(a["controls"]["pass"] for a in arms.values()) and len(arms) == len(tags)
    cross = {"stop_point_counts_equal_across_arms": len(sets_ok) == 1, "counts": sorted(sets_ok)}
    ctl_ok = ctl_ok and cross["stop_point_counts_equal_across_arms"]
    oc = outcome({t: {"primary": arms[t]["primary"], "reach": arms[t]["reach"]} for t in arms if t in (*DECIDE, FLOOR_ARM)}, controls_ok=ctl_ok)
    floor = oc.get("floor_m")
    beside = {}
    for t in LEVERS:
        if t in arms and floor is not None:
            beside[t] = {"primary_is_gain_by_the_same_rule": is_gain(arms[t]["primary"], floor), "reaches_O_separated": arms[t]["reach"].get("separated")}
    res = {"tool": "stoporc_read.py", "card": "theory/stopbal/oracle/PREREAD_V8F_STOPORC.md", "card_sha256_expected": "49eb7d7be6bcdfc655fcb2f7c1c21e6583248c0d3a75f87321fd776806c34535",
           "tier": "T1-kin FIRST TICK, oracle diagnostic (zero training), never driving performance", "estimator": f"taniteval.ci paired episode-cluster bootstrap, B={NB}, seed {SEED}; answers 'another draw of episodes' only (not another training run, not another inference run: the planner is deterministic)",
           "ci_source": getattr(CI, "SOURCE", getattr(CI, "__file__", "?")), "tree": TREE, "evidence_class": "MEASURED (this file; per-arm rows were banked to scratch npz, sha256 in `banked`)",
           "definitions": {"STOP_POINT_goal_index": STOP_GI, "LON9_STOP_index": LON_STOP_I, "code_layout": {"goal": [0, N_G], "lat": [N_G, N_G + N_LAT], "lon": [N_G + N_LAT, N_G + N_LAT + N_LON], "cons": [N_G + N_LAT + N_LON, CODE_DIM]},
                           "bands": {k: list(v) for k, v in BANDS.items()}, "primary_cell": PRIMARY_CELL, "delta": "forced minus OWN (negative = lower error)"},
           "families": {"PRIMARY/ADE-FDE, LONGITUDINAL, LATERAL": "in forced_minus_own per cell", "TACTICAL": TACTICAL_FAMILY, "STRATEGIC": STRATEGIC_FAMILY},
           "banked": banked, "cross_arm": cross, "outcome": oc, "levers_beside": beside, "arms": arms}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({"outcome": oc, "controls_ok": ctl_ok}, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("run", "analyse"), required=True)
    ap.add_argument("--tags", nargs="+", default=list(ALL_TAGS))
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "RESULT_STOPORC.json"))
    a = ap.parse_args()
    if a.stage == "run":
        run_stage(a.tags, a.smoke)
    else:
        analyse_stage(a.out, a.tags)


if __name__ == "__main__":
    sys.path.insert(0, READ)
    main()
