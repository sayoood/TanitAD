"""refcv6 battery PANEL (SPEC.md §3). Every number here comes through ONE instrument.

1. `build_panel_dump`: the refcv6 dump plus STOP plus the banked baselines' `os` rows, added as
   arms (`b_refcv4b`, `b_refcv5v2_s0`, `b_refcv5v2_s1`) on the SAME windows. Before any row is
   copied, the pairing is VERIFIED per window, and every one of these is a VOID gate:
   (clip, ws) identical, `g` bit-identical, and the model-free arms `ha`/`ha0`/`ha0_ext`
   bit-identical.
2. `analyze`: `refcv3_arm.analyze_refcv3` over that dump. Each baseline is read by the same code
   and the same lead-block join as refcv6.
3. `cross_paired`: per-family paired cells (`refav1_arm._components` + `_paired_families`,
   imported), refcv6 `os` against every control and every baseline.
4. `tactical_v6`: the v6 behaviour decoder's lat/lon heads and the z_tac v7 heads against the
   v8 labels (in-band windows), with confusion matrices and kappa, plus the 22-token goal
   selection per class (AUROC / AP / P / R at 0.5, n_pos / n_neg, UNSCOREABLE under n_pos 200).
5. `acceptance`: the frozen refcv6 instruments (T-FLIP from the nav-compliance block, OBEDIENCE).
6. `build_s6_dump`: the 6 s read from the banked full plans. The controls are re-integrated on
   the 6 s grid.
"""
from __future__ import annotations

import glob
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcv6_loader as L  # noqa: E402

L.bootstrap()
import torch  # noqa: E402
import refcv6_roll as RR  # noqa: E402

BASELINES = {
    "b_refcv4b": "C:/Users/Admin/refcv5cmp/out/refcv4b_devbox_dump",
    "b_refcv5v2_s0": "C:/Users/Admin/refcv5cmp/out/refcv5-v2_dump",
    "b_refcv5v2_s1": "C:/Users/Admin/refcv5cmp/out/refcv5-v2-seed1_dump",
}
LEAD_BLOCK = str(L.REPO / "TanitAD Research Lab" / "Benchmarks & Evals" / "Research" /
                 "2026-09-02-b1-eval-lead-block" / "raw" / "b1_eval_lead_block.npz")
MODEL_FREE = ("ha", "ha0", "ha0_ext")
TIERS = {"os": "T1", "os_navshuf": "T1", "os_navzero": "T1", "os_navflip": "T1",
         "oracle_sel": "T0", "stop": "T1", "os_vmaxzero": "T1", "ha": "T1", "ha0": "T1",
         "ha0_ext": "T1", **{k: "T1" for k in BASELINES}}


def tiers_arg(arms) -> str:
    return ",".join(f"{a}={TIERS[a]}" for a in arms if a in TIERS)


def _man(d):
    return json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))


def _eps_by_clip(d):
    return {e["clip_id"]: e["file_index"] for e in _man(d)["episodes"]}


# --------------------------------------------------------------------------------------- #
def build_panel_dump(r6_dump: str, out_dir: str, extras_npz: str | None,
                     baselines: dict = BASELINES) -> dict:
    os.makedirs(os.path.join(out_dir, "decisions"), exist_ok=True)
    man = _man(r6_dump)
    base_idx = {lab: _eps_by_clip(d) for lab, d in baselines.items()}
    ex = np.load(extras_npz) if extras_npz and os.path.exists(extras_npz) else None
    ex_key = None
    if ex is not None:
        ex_key = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(ex["clip_index"], ex["ws"]))}
    slots = man["grid"]["slots"]
    rec = {"n_episodes": 0, "n_windows": 0, "pairing": {}, "vmaxzero_rows": 0}
    for lab in baselines:
        rec["pairing"][lab] = {"clips_missing": 0, "ws_equal_clips": 0, "g_bit_equal_windows": 0,
                               "g_max_abs_diff": 0.0, "model_free_bit_equal": {a: 0 for a in MODEL_FREE},
                               "model_free_max_abs_diff": {a: 0.0 for a in MODEL_FREE}}
    for e in man["episodes"]:
        fi, cid, e_i = e["file_index"], e["clip_id"], e["episode_index"]
        z = dict(np.load(os.path.join(r6_dump, f"ep{fi:03d}.npz")))
        g, ws = z["g"], z["ws"]
        z["stop"] = np.zeros_like(g)
        for lab, d in baselines.items():
            p = rec["pairing"][lab]
            if cid not in base_idx[lab]:
                p["clips_missing"] += 1
                raise SystemExit(f"[panel] baseline {lab} lacks clip {L.sha12(cid)}")
            zb = dict(np.load(os.path.join(d, f"ep{base_idx[lab][cid]:03d}.npz")))
            pos = {int(w): i for i, w in enumerate(zb["ws"])}
            miss = [int(w) for w in ws if int(w) not in pos]
            if miss:
                raise SystemExit(f"[panel] {lab}: {len(miss)} refcv6 windows of clip "
                                 f"{L.sha12(cid)} are not in the banked baseline -- not the same "
                                 f"windows; refusing to pair")
            rows = [pos[int(w)] for w in ws]
            p["ws_equal_clips"] += int(len(rows) == len(zb["ws"]))
            p.setdefault("n_windows_paired", 0)
            p["n_windows_paired"] += len(rows)
            dg = np.abs(zb["g"][rows].astype(np.float64) - g.astype(np.float64)).max(axis=(1, 2))
            p["g_bit_equal_windows"] += int((dg == 0).sum())
            p["g_max_abs_diff"] = max(p["g_max_abs_diff"], float(dg.max()))
            for a in MODEL_FREE:
                da = np.abs(zb[a][rows].astype(np.float64) - z[a].astype(np.float64)).max(axis=(1, 2))
                p["model_free_bit_equal"][a] += int((da == 0).sum())
                p["model_free_max_abs_diff"][a] = max(p["model_free_max_abs_diff"][a], float(da.max()))
            z[lab] = zb["os"][rows].astype(np.float32)
        if ex is not None and "vmax_zero.traj" in ex.files:
            rows = [ex_key.get((int(e_i), int(w))) for w in ws]
            if any(r is None for r in rows):
                raise SystemExit(f"[panel] extras lack windows of clip {L.sha12(cid)}")
            tr_ = ex["vmax_zero.traj"][rows][:, slots]
            if np.isfinite(tr_).all():
                z["os_vmaxzero"] = tr_.astype(np.float32)
                rec["vmaxzero_rows"] += len(rows)
        np.savez_compressed(os.path.join(out_dir, f"ep{fi:03d}.npz"), **z)
        shutil.copyfile(os.path.join(r6_dump, "decisions", f"ep{fi:03d}.npz"),
                        os.path.join(out_dir, "decisions", f"ep{fi:03d}.npz"))
        rec["n_episodes"] += 1
        rec["n_windows"] += int(len(ws))
    arms = list(man["arms"]) + ["stop"] + list(baselines) + (
        ["os_vmaxzero"] if rec["vmaxzero_rows"] == rec["n_windows"] and rec["n_windows"] else [])
    man["arms"] = arms
    man["tiers"] = {a: TIERS.get(a, man.get("tiers", {}).get(a)) for a in arms}
    man.setdefault("arm_meaning", {}).update({
        "stop": "STOP: zero displacement at every instant (model-free control)",
        "os_vmaxzero": "refcv6 os with the max-speed input WITHHELD (v_max_valid = 0)",
        **{lab: f"banked `os` of {lab[2:]} ({d}), 256x640, SAME windows (verified)"
           for lab, d in baselines.items()}})
    man["panel"] = {"built_from": r6_dump, "baselines": baselines, "pairing": rec["pairing"]}
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8"),
              indent=1, default=str)
    for a in ("ABLATION.txt",):
        if os.path.exists(os.path.join(r6_dump, a)):
            shutil.copyfile(os.path.join(r6_dump, a), os.path.join(out_dir, a))
    # VOID gates 3/4
    void = []
    for lab, p in rec["pairing"].items():
        if p["g_bit_equal_windows"] != rec["n_windows"]:
            void.append(f"{lab}: g not bit-identical on {rec['n_windows'] - p['g_bit_equal_windows']} windows")
        for a in MODEL_FREE:
            if p["model_free_bit_equal"][a] != rec["n_windows"]:
                void.append(f"{lab}: {a} not bit-identical on "
                            f"{rec['n_windows'] - p['model_free_bit_equal'][a]} windows")
    rec["void_gates_3_4"] = {"pass": not void, "failures": void}
    rec["arms"] = arms
    return rec


def analyze(panel_dir: str, out_json: str, labels: str, n_boot=2000, seed=0) -> dict:
    R = RR.ra3()
    arms = _man(panel_dir)["arms"]
    for a in arms:
        R.ARM_TIERS.setdefault(a, TIERS.get(a, "T1"))
    argv = ["--analyze-only", panel_dir, "--out", out_json, "--labels", labels,
            "--n-boot", str(n_boot), "--seed", str(seed), "--arm", "refcv6",
            "--tiers", tiers_arg(arms)]
    if os.path.exists(LEAD_BLOCK):
        argv += ["--lead-block", LEAD_BLOCK]
    R.main(argv)
    return json.load(open(out_json, encoding="utf-8"))


# --------------------------------------------------------------------------------------- #
def load_panel(panel_dir: str):
    man = _man(panel_dir)
    files = sorted(glob.glob(os.path.join(panel_dir, "ep*.npz")))
    arrs, eid = {}, []
    for fi, f in enumerate(files):
        z = np.load(f)
        n = z["g"].shape[0]
        eid += [fi] * n
        for k in z.files:
            if k in ("eid", "clip_index", "ws", "v0"):
                continue
            arrs.setdefault(k, []).append(z[k])
    return man, {k: np.concatenate(v) for k, v in arrs.items()}, np.asarray(eid)


#: SPEC AMENDMENT A3. The shared `refav1_arm._components` scores yaw-rate over EVERY step, while
#: `four_families` itself masks it with `pred.pair_valid & gt.pair_valid` (a stopped or crawling
#: step has no path tangent). MEASURED at step 5000 seed 0: refcv4b 0.2034 rad/s unmasked vs
#: 0.0318 on valid steps. The `_valid` cell below is the one the renderer shows; the shared cell
#: stays in the JSON, labelled DEFECTIVE, and is never quoted.
YAW_VALID = "LAT_yaw_rate_mae_radps_valid"
YAW_SHARED = "LAT_yaw_rate_mae_radps"
A3_NOTE = ("SPEC A3: `LAT_yaw_rate_mae_radps` (shared refav1_arm._components) is scored on steps "
           "with no path tangent and is DEFECTIVE -- quote `LAT_yaw_rate_mae_radps_valid` "
           "(four_families' own pred&gt pair_valid mask, per-window mean over valid steps).")


def yaw_rate_valid(P: np.ndarray, G: np.ndarray, dt: float) -> np.ndarray:
    """A3: per-window yaw-rate MAE (rad/s) over the steps where BOTH the prediction's and the
    GT's `pair_valid` hold -- four_families' own geometry and its own mask (four_families.py
    `both_pair`), never re-derived. NaN for a window with no valid step pair."""
    from taniteval import four_families as ff
    pt, gt = torch.as_tensor(P).float(), torch.as_tensor(G).float()
    Pg, Gg = ff._seq_geometry(pt, dt), ff._seq_geometry(gt, dt)
    m = Pg["pair_valid"] & Gg["pair_valid"]
    err = (Pg["yaw_rate"] - Gg["yaw_rate"]).abs()
    nv = m.sum(1)
    return torch.where(nv > 0, (err * m).sum(1) / nv.clamp_min(1),
                       torch.full_like(err[:, 0], float("nan"))).numpy()


def _paired_yaw_valid(comps: dict, a_: str, b_: str, eid, n_boot: int, seed: int) -> dict:
    """b − a on the A3 cell, the SAME estimator `_paired_families` uses (paired episode-cluster
    bootstrap, windows with a non-finite value on either side dropped and counted)."""
    from taniteval import ci as _ci
    a_v, b_v = comps[a_][YAW_VALID], comps[b_][YAW_VALID]
    keep = np.isfinite(a_v) & np.isfinite(b_v)
    if keep.sum() == 0:
        return {"status": "REFUSED", "reason": "no window has a valid step pair for both arms",
                "amendment": "A3"}
    e = [x for x, kp in zip(eid, keep) if kp]
    r = _ci.paired_episode_cluster_bootstrap(b_v[keep], a_v[keep], e, n_boot=n_boot, seed=seed)
    r["n_dropped_nonfinite"] = int((~keep).sum())
    r["amendment"] = "A3"
    return r


def _attach_a3(block: dict, comps: dict, a_: str, b_: str, eid, n_boot: int, seed: int) -> dict:
    lat = block.setdefault("families", {}).setdefault("lateral", {})
    lat[YAW_VALID] = _paired_yaw_valid(comps, a_, b_, eid, n_boot, seed)
    block["A3_note"] = A3_NOTE
    return block


def cross_paired(panel_dir: str, pairs, n_boot=2000, seed=0) -> dict:
    R = RR.ra3()
    ra = R.ra
    man, A, eid = load_panel(panel_dir)
    dt = float(man["grid"]["dt_s"])
    need = sorted({x for p in pairs for x in p[:2] if x in A})
    comps = {x: ra._components(A[x], A["g"], dt) for x in need}
    for x in comps:                                    # SPEC A3
        comps[x][YAW_VALID] = yaw_rate_valid(A[x], A["g"], dt)
    tiers = {x: TIERS.get(x, "T1") for x in need}
    out = {}
    for a_, b_, nm in pairs:
        if a_ in comps and b_ in comps:
            out[nm] = _attach_a3(ra._paired_families(comps, a_, b_, list(eid), tiers, n_boot, seed),
                                 comps, a_, b_, list(eid), n_boot, seed)
        else:
            out[nm] = {"status": "ABSENT", "missing": [x for x in (a_, b_) if x not in comps]}
    return {"n_windows": int(len(eid)), "n_episodes": int(len(set(eid.tolist()))),
            "dt_s": dt, "pairs": out, "A3_note": A3_NOTE,
            "means": {x: {k: float(np.nanmean(v)) for k, v in comps[x].items()} for x in comps}}


def seed_replicate(panel_a: str, panel_b: str, n_boot=2000, seed=0) -> dict:
    """refcv6 `os` at inference seed A vs seed B on the SAME windows (the inference-run question)."""
    R = RR.ra3()
    ra = R.ra
    ma, A, eid = load_panel(panel_a)
    mb, B, eidb = load_panel(panel_b)
    if not (np.array_equal(eid, eidb) and np.array_equal(A["g"], B["g"])):
        raise SystemExit("[panel] seed replicate: the two dumps are not the same windows")
    dt = float(ma["grid"]["dt_s"])
    comps = {"os_A": ra._components(A["os"], A["g"], dt), "os_B": ra._components(B["os"], B["g"], dt)}
    comps["os_A"][YAW_VALID] = yaw_rate_valid(A["os"], A["g"], dt)      # SPEC A3
    comps["os_B"][YAW_VALID] = yaw_rate_valid(B["os"], B["g"], dt)
    blk = _attach_a3(ra._paired_families(comps, "os_B", "os_A", list(eid),
                                         {"os_A": "T1", "os_B": "T1"}, n_boot, seed),
                     comps, "os_B", "os_A", list(eid), n_boot, seed)
    ident = float(np.abs(A["os"].astype(np.float64) - B["os"].astype(np.float64)).max())
    return {"direction": "os(seed A) - os(seed B)", "max_abs_path_diff_m": ident,
            "families": blk["families"]}


def void_gate_checks(panel_dir: str, analysis: dict) -> dict:
    """SPEC §3.5, gates 1-2 and 5: controls that MUST read a known value."""
    from taniteval import four_families as ff
    man, A, eid = load_panel(panel_dir)
    dt = float(man["grid"]["dt_s"])
    out = {}
    # (2) STOP: zero displacement; its ADE must equal the mean GT distance exactly
    st = A["stop"].astype(np.float64)
    g = A["g"].astype(np.float64)
    ade_stop = np.linalg.norm(st - g, axis=-1).mean(1)
    ade_gt = np.linalg.norm(g, axis=-1).mean(1)
    out["stop"] = {"max_abs_displacement_m": float(np.abs(st).max()),
                   "ade_stop_minus_mean_gt_distance_max_abs": float(np.abs(ade_stop - ade_gt).max()),
                   "pass": bool(np.abs(st).max() == 0.0 and np.abs(ade_stop - ade_gt).max() == 0.0)}
    # (1) ha0: its own curvature exactly 0 under the real (masked) estimator; tactical kappa 0
    geo = ff._seq_geometry(torch.as_tensor(A["ha0"]).float(), dt)
    kap = geo.get("curvature")
    val = geo.get("pair_valid")               # [n, H-1] -- the mask four_families scores with
    kmax = (float((kap.abs() * val).max()) if kap is not None and val is not None and val.any()
            else None)
    tac = ((analysis.get("arms") or {}).get("ha0") or {}).get("four_families", {}).get("tactical", {})
    latk = (tac.get("lateral_decision") or {}).get("kappa")
    lonk = (tac.get("longitudinal_decision") or {}).get("kappa")
    out["ha0"] = {"max_abs_own_curvature_1pm": kmax, "tactical_lat_kappa": latk,
                  "tactical_lon_kappa": lonk,
                  "pass": bool(kmax == 0.0 and (latk in (0.0, None)) and (lonk in (0.0, None)))}
    # (5) profiles (recorded; degenerate is expected at an early checkpoint)
    rc = analysis.get("refcv3") or {}
    sp = rc.get("selection_profile") or {}
    tp = ((rc.get("trivial_profile") or {}).get("arms") or {}).get("os") or {}
    out["profiles"] = {"selection_degenerate": sp.get("degenerate"),
                       "selection_modal_frac": sp.get("modal_frac"),
                       "n_distinct_selected": sp.get("n_distinct_selected"),
                       "os_trivial_frac": tp.get("trivial_frac")}
    out["pass_1_2"] = bool(out["stop"]["pass"] and out["ha0"]["pass"])
    return out


# --------------------------------------------------------------------------------------- #
def _auroc(score, y):
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, np.float64)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), np.float64)
    ss = s[order]
    i = 0
    while i < len(ss):                                     # average ranks over ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _ap(score, y):
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return None
    o = np.argsort(-np.asarray(score, np.float64), kind="mergesort")
    yy = y[o]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def _kappa(t, p, n):
    cm = np.zeros((n, n), np.int64)
    for a, b in zip(t, p):
        cm[a, b] += 1
    tot = cm.sum()
    if tot == 0:
        return None, cm
    po = np.trace(cm) / tot
    pe = (cm.sum(0) * cm.sum(1)).sum() / tot ** 2
    return (float((po - pe) / (1 - pe)) if pe < 1 else None), cm


def tactical_v6(extras_npz: str, clip_eid: dict, n_boot=2000, seed=0) -> dict:
    from taniteval import ci as _ci
    from tanitad.data import v7_labels as v7l
    from tanitad.models import vocab_v7
    ex = np.load(extras_npz)
    eid = np.asarray([clip_eid[int(c)] for c in ex["clip_index"]])
    out = {"n_windows": int(len(eid)), "estimator": "episode_cluster_bootstrap (accuracy); kappa and "
           "per-class rates are full-set", "tier": "T1 (UNRULED)"}
    names = {"lat": list(v7l.HEADS["tac_lat"]), "lon": list(v7l.HEADS["tac_lon"])}
    for head in ("lat", "lon"):
        lab = ex[f"{head}_v7"].astype(np.int64)
        ok = lab >= 0
        blk = {"classes": names[head], "n_in_band": int(ok.sum()),
               "n_episodes": int(len(set(eid[ok].tolist())))}
        for surf, key in (("v6_behaviour_decoder", f"nav_true.tacv6_{head}_logits"),
                          ("z_tac_v7_heads", f"nav_true.{head}_logits_tac"),
                          ("v6_behaviour_decoder_NAVZERO", f"nav_zero.tacv6_{head}_logits")):
            if key not in ex.files:
                blk[surf] = {"status": "ABSENT", "key": key}
                continue
            pred = np.nanargmax(np.nan_to_num(ex[key], nan=-1e9), axis=1)
            t, p = lab[ok], pred[ok]
            corr = (t == p).astype(np.float64)
            kap, cm = _kappa(t, p, len(names[head]))
            ci = _ci.episode_cluster_bootstrap(corr, list(eid[ok]), n_boot=n_boot, seed=seed) \
                if ok.sum() else None
            maj = int(np.bincount(t, minlength=len(names[head])).argmax()) if ok.sum() else None
            blk[surf] = {"acc": ci, "kappa": kap, "confusion_true_by_pred": cm.tolist(),
                         "majority_class_rate": float((t == maj).mean()) if ok.sum() else None,
                         "per_class": {names[head][c]: {"n_true": int((t == c).sum()),
                                                        "n_pred": int((p == c).sum()),
                                                        "recall": (float(((t == c) & (p == c)).sum() / (t == c).sum())
                                                                   if (t == c).sum() else None)}
                                       for c in range(len(names[head]))}}
        out[head] = blk
    # ---- 22-token goal selection ------------------------------------------------------ #
    toks = list(getattr(v7l, "TAC_GOAL_TOKENS", None) or vocab_v7.TACTICAL_GOAL_TOKENS_V7)
    floor = int(getattr(vocab_v7, "GOAL_MIN_N_FOR_METRIC", 200))
    gy, gw = ex["tac_goal_y"], ex["tac_goal_w"]
    goal = {"tokens": toks, "scoreability_floor_n_pos": floor, "per_class": {}}
    for surf, key in (("nav_true", "nav_true.tacv6_goal_logits"),
                      ("nav_zero", "nav_zero.tacv6_goal_logits")):
        if key not in ex.files:
            continue
        lg = ex[key]
        pc = {}
        for j, tk in enumerate(toks):
            sup = np.isfinite(gw[:, j]) & (gw[:, j] > 0) & np.isfinite(lg[:, j])
            y = gy[sup, j] > 0.5
            s = 1.0 / (1.0 + np.exp(-lg[sup, j].astype(np.float64)))
            npos, nneg = int(y.sum()), int((~y).sum())
            pr = s >= 0.5
            pc[tk] = {"n_supervised": int(sup.sum()), "n_pos": npos, "n_neg": nneg,
                      "n_episodes": int(len(set(eid[sup].tolist()))),
                      "auroc": _auroc(s, y), "ap": _ap(s, y),
                      "prevalence": (npos / max(1, npos + nneg)),
                      "precision@0.5": (float((pr & y).sum() / pr.sum()) if pr.sum() else None),
                      "recall@0.5": (float((pr & y).sum() / y.sum()) if y.sum() else None),
                      "status": ("UNSCOREABLE (n_pos < %d)" % floor) if npos < floor else "SCOREABLE"}
        goal["per_class"][surf] = pc
    out["goal_22"] = goal
    return out


def acceptance(analysis: dict, extras_npz: str, clip_eid: dict, panel_dir: str,
               n_boot=2000, seed=0) -> dict:
    from taniteval import refcv6_acceptance as acc
    out = {}
    nc = (((analysis.get("refcv3") or {}).get("strategic") or {}).get("nav_compliance") or {})
    # `taniteval.nav_compliance.nav_compliance_report` stores `compliance_arm(...)` per readout
    # under `readouts[<name>]` (nav_compliance.py:862); T-FLIP reads the PLAN readout.
    blk = (nc.get("readouts") or {}).get("plan")
    if isinstance(blk, dict):
        out["tflip"] = acc.tflip_verdict(blk)
        out["tflip"]["nav_compliance_status"] = nc.get("status")
        out["tflip"]["nav_compliance_reason"] = nc.get("reason")
    else:
        out["tflip"] = {"test": "T-FLIP", "verdict": "NOT_RUN",
                        "reason": "no readouts.plan block in the analysis",
                        "nav_compliance_status": nc.get("status"),
                        "nav_compliance_reason": nc.get("reason")}
    ex = np.load(extras_npz)
    pm = ex["obey30.planned_max_ms"] if "obey30.planned_max_ms" in ex.files else None
    if pm is not None:
        sel = np.isfinite(pm)
        eid = np.asarray([clip_eid[int(c)] for c in ex["clip_index"]])
        rows_empty = int((ex["obey30.n_fan_compliant"][sel] == 0).sum())
        # ⭐ THE COST IS PART OF THE RESULT (refcv6_acceptance docstring): ADE over the S2 grid of
        # the forced-30 km/h plan vs the same window's unforced `os`, on the SAME GT.
        man = _man(panel_dir)
        slots = man["grid"]["slots"]
        gt_by, os_by = {}, {}
        for e in man["episodes"]:
            z = np.load(os.path.join(panel_dir, f"ep{e['file_index']:03d}.npz"))
            for j, w in enumerate(z["ws"]):
                gt_by[(int(e["episode_index"]), int(w))] = z["g"][j]
                os_by[(int(e["episode_index"]), int(w))] = z["os"][j]
        idx = np.nonzero(sel)[0]
        ade_f, ade_b = [], []
        for i in idx:
            k = (int(ex["clip_index"][i]), int(ex["ws"][i]))
            g = gt_by.get(k)
            if g is None:
                ade_f.append(np.nan)
                ade_b.append(np.nan)
                continue
            tf = ex["obey30.traj"][i][slots]
            ade_f.append(float(np.linalg.norm(tf - g, axis=-1).mean()))
            ade_b.append(float(np.linalg.norm(os_by[k] - g, axis=-1).mean()))
        out["obedience"] = acc.speed_obedience_report(
            pm[sel], ex["gt_max_speed_6s"][sel], eid[sel], rows_empty=rows_empty,
            ade_forced=np.asarray(ade_f), ade_baseline=np.asarray(ade_b),
            n_boot=n_boot, seed=seed)
    else:
        out["obedience"] = {"verdict": "NOT_RUN"}
    return out


def tzero_from_tactical(tac: dict) -> dict:
    """The frozen T-ZERO diagnostic (`refcv6_acceptance.tzero_nonturn_report`) on the 22-token goal
    set: per class recall@0.5 under the true nav vs nav withheld, on the SAME windows."""
    from taniteval import refcv6_acceptance as acc
    pc = (tac.get("goal_22") or {}).get("per_class") or {}
    t, z = pc.get("nav_true") or {}, pc.get("nav_zero") or {}
    if not t or not z:
        return {"test": "T-ZERO", "verdict": "NOT_RUN", "reason": "no nav_true / nav_zero goal block"}
    conv = lambda d: {k: {"n_pos": v.get("n_pos"), "recall": v.get("recall@0.5")}  # noqa: E731
                      for k, v in d.items()}
    return acc.tzero_nonturn_report(conv(t), conv(z))


# --------------------------------------------------------------------------------------- #
def build_s6_dump(src_dump: str, out_dir: str, baselines: dict = BASELINES,
                  kit_eval: str | None = None) -> dict:
    """The 6 s read (SPEC §3.1 S6) from the banked FULL plans (`plan_full_*` in the decisions)."""
    R = RR.ra3()
    import refb_labels
    from tanitad.data.v2_dataset import build_v2_providers, load_or_build_manifest
    man = _man(src_dump)
    hz = list(man["model"]["horizons"])
    g6 = R.grid_slots(hz, "6s")
    s6 = g6["slots"]
    kit_eval = kit_eval or man["corpus"]["episodes"]
    kman = load_or_build_manifest(kit_eval, verbose=False)
    provs = dict(zip([str(c) for c in kman["clip_id"]],
                     build_v2_providers([kit_eval], lru_size=2, verbose=False)))
    base_idx = {lab: _eps_by_clip(d) for lab, d in baselines.items()}
    os.makedirs(os.path.join(out_dir, "decisions"), exist_ok=True)
    n_tot = n_keep = 0
    new_eps = []
    for e in man["episodes"]:
        fi, cid = e["file_index"], e["clip_id"]
        z = dict(np.load(os.path.join(src_dump, f"ep{fi:03d}.npz")))
        dz = dict(np.load(os.path.join(src_dump, "decisions", f"ep{fi:03d}.npz")))
        fv = dz["gt_future_valid_ext"] > 0.5
        keep = fv[:, max(g6["horizons_steps"]) - 1]
        n_tot += len(keep)
        if not keep.any():
            continue
        ws = z["ws"][keep]
        pl = torch.as_tensor(dz["pose_last"][keep]).float()
        fut = torch.as_tensor(dz["gt_future_ext"][keep]).float()
        g = refb_labels.waypoint_targets(pl, fut, hz)[:, s6].numpy().astype(np.float32)
        out = {"g": g, "ws": ws, "v0": z["v0"][keep]}
        for c, arm in (("nav_true", "os"), ("nav_shuffled", "os_navshuf"), ("nav_zero", "os_navzero"),
                       ("nav_flipped", "os_navflip")):
            k = f"plan_full_{c}"
            if k in dz:
                out[arm] = dz[k][keep][:, s6].astype(np.float32)
        ep = provs[cid]
        v_ep = ep.poses[:, 3].float()
        kap_ep = ep.actions[:, 0].float()
        ha, ha0, hx = [], [], []
        for t0, v0 in zip(ws.tolist(), out["v0"].tolist()):
            n_f = g6["n_frames"]
            hold = R.hold_controls(v_ep, kap_ep, int(t0))
            ha.append(R.integrate_select(hold[None].expand(n_f, 2), float(v0), g6, action_units="steer"))
            ha0.append(R.integrate_select(R.hold_v0_controls(n_f), float(v0), g6, action_units="steer"))
            ext = R.ra.hold_ext_controls(None, v_ep, kap_ep, int(t0), dt=0.1, stride=1)
            hx.append(R.integrate_select(ext[None].expand(n_f, 2), float(v0), g6, action_units="steer"))
        out["ha"] = np.concatenate(ha).astype(np.float32)
        out["ha0"] = np.concatenate(ha0).astype(np.float32)
        out["ha0_ext"] = np.concatenate(hx).astype(np.float32)
        out["stop"] = np.zeros_like(g)
        for lab, d in baselines.items():
            bfi = base_idx[lab][cid]
            bd = np.load(os.path.join(d, "decisions", f"ep{bfi:03d}.npz"))
            bws = np.load(os.path.join(d, f"ep{bfi:03d}.npz"))["ws"]
            pos = {int(w): i for i, w in enumerate(bws)}
            rows = [pos[int(w)] for w in ws]
            out[lab] = bd["plan_full_nav_true"][rows][:, s6].astype(np.float32)
            # the baseline's GT through ITS OWN banked future must equal ours (VOID gate 4, 6 s)
            gb = refb_labels.waypoint_targets(torch.as_tensor(bd["pose_last"][rows]).float(),
                                              torch.as_tensor(bd["gt_future_ext"][rows]).float(),
                                              hz)[:, s6].numpy()
            if not np.array_equal(gb.astype(np.float32), g):
                raise SystemExit(f"[s6] {lab}: 6 s GT differs on clip {L.sha12(cid)}")
        nf = len(new_eps)
        out["eid"] = np.array([nf])
        out["clip_index"] = z["clip_index"]
        np.savez_compressed(os.path.join(out_dir, f"ep{nf:03d}.npz"), **out)
        dsub = {}
        N = len(keep)
        for k, v in dz.items():
            if k == "ep_poses":
                dsub[k] = v
            elif hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == N:
                dsub[k] = v[keep]
            else:
                dsub[k] = v
        np.savez_compressed(os.path.join(out_dir, "decisions", f"ep{nf:03d}.npz"), **dsub)
        new_eps.append({**e, "file_index": nf, "n_windows": int(keep.sum())})
        n_keep += int(keep.sum())
    m2 = dict(man)
    m2["episodes"] = new_eps
    m2["arms"] = [a for a in ("os", "ha", "ha0", "ha0_ext", "os_navshuf", "os_navzero",
                              "os_navflip", "stop", *baselines)]
    m2["tiers"] = {a: TIERS[a] for a in m2["arms"]}
    m2["grid"] = {**g6, "n_windows": n_keep, "n_windows_skipped": n_tot - n_keep,
                  "skip_reasons": {"6s_future_beyond_clip": n_tot - n_keep},
                  "n_episodes": len(new_eps), "window_stride": man["grid"].get("window_stride"),
                  "obs_window": man["grid"].get("obs_window"), "ws_is": man["grid"].get("ws_is"),
                  "source": "S6 = S2 windows with a valid 60-step future; paths = plan_full_* "
                            "index-selected at slots %s" % s6}
    json.dump(m2, open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8"),
              indent=1, default=str)
    return {"n_windows": n_keep, "n_dropped_6s_future": n_tot - n_keep, "n_episodes": len(new_eps),
            "grid": g6}
