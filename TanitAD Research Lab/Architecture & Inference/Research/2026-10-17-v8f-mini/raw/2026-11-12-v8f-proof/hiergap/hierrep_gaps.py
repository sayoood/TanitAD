r"""hierrep_gaps.py -- fills the two NAMED GAP rows of the HIERREP read (HIERREP_DEFS R14 a, b).  CPU only, no model pass.  Definitions: GAPS_DEFS.md (written first).
Tier T1-kin (kinematic rollout on the model's own actions, logged scene), a diagnostic and never driving performance.  Evidence class MEASURED once run.
These rows complete the four metric families; they are not card criteria.  STRATEGIC: PAUSED (PI item 25).

  python hierrep_gaps.py tables --lead-npz F --join-npz F --out-dir D          builds lead_table.npz and label_table.npz (dev box, once)
  python hierrep_gaps.py arm  --dump DIR --ticks FILE --tables-dir D --out PREFIX [--identity]      one arm -> PREFIX.npz + PREFIX.json
  python hierrep_gaps.py read --t1-root DIR --tables-dir D --out-dir D [--pairs HIER:FLAT,DAG:FLATDAG] [--seeds 0,..,5] [--map file.json] [--no-card-counts]
Layout (default): <t1-root>/<ARM>_s<S>/dump_<ARM>_s<S>_m10_causal_heads/ep*.npz and <t1-root>/<ARM>_s<S>/ticks/ticks_<ARM>_s<S>_m10_causal_heads.npz.
--map JSON {"ARM": {"seed": "tagstem"}} selects the flat layout of an older folder: <t1-root>/dump_<stem>_m10_causal_heads and <t1-root>/ticks/ticks_<stem>_m10_causal_heads.npz.
Exit codes: 0 read written and controls pass, 2 a control failed (files written, headed READ NOT VALID), 3 refusal (input error / completeness).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

os.environ.setdefault("TANITEVAL_ROOT", "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/taniteval")
STAGEB = os.environ.get("HIERREP_STAGEB", "F:/TanitAD-scratch/v8proof/pod/stageB")
if STAGEB not in sys.path:
    sys.path.insert(0, STAGEB)
import hierrep_read as RH  # noqa: E402  (read_cell, boot_two_level, seed_arrays, classify, kinematics, load_dump, sign_flip_p, t_interval, N_SEEDS_CARD)

DT = 0.2
TIER = "T1-kin (kinematic rollout on the model's own actions, logged scene)"
STAMP = "these rows complete the four metric families; they are not card criteria"
N_KNOTS = 30
TG_FLOOR_V = 1.0       # m/s, floor of the label builder's time gap
TG_LIMIT_S = 1.0
CROSS_M = 2.0
ALIGN_MAX_MPS = 0.5
STOP_I = 6             # GOAL22 index of STOP_POINT
LON_STOP_I = 2
A_KEYS = ("gap_abs", "gap_signed", "tg_abs", "tg_share")
A_ROWS = (("gap_abs", "mean absolute gap error", "m"), ("gap_signed", "mean signed gap error (rolled minus logged; negative = closer)", "m"),
          ("tg_abs", "mean absolute time-gap error", "s"), ("tg_share", "share of units with time gap < 1.0 s where the logged ego kept >= 1.0 s", "share"))
GAPS_NOT_MEASURABLE = (
    "A: the lead does not react to the rolled ego (logged scene replays the logged lead)",
    "A: only the along-path offset of the rolled ego enters the gap; lateral departure is counted (offset > 2.0 m), not scored",
    "A: knots with no lead in the lead file have no gap; lead identity and class are not in the file",
    "B: the label is multi-hot, so no argmax-selection confusion exists; token-wise rows only",
    "B: ticks 1 and 2 decide about frames seen from the rolled pose while the label belongs to the logged frame; tick 0 is printed beside the pooled row",
    "STRATEGIC: PAUSED (PI item 25), never a pass, never omitted")


# ------------------------------------------------------------------------------------------------ tables (dev box, once)
def sha12(cid):
    return hashlib.sha256(str(cid).encode("utf-8")).hexdigest()[:12]


def build_tables(lead_npz, join_npz, out_dir, n_raw=200):
    J = np.load(join_npz, allow_pickle=True)
    clips = [str(c) for c in J["clip"]]
    L = np.load(lead_npz, allow_pickle=True)
    sh = np.asarray(L["sha12"]).astype(str)
    C = len(clips)
    has, lead, gap, v = (np.zeros((C, n_raw), np.int8), np.zeros((C, n_raw), np.int8), np.full((C, n_raw), np.nan), np.full((C, n_raw), np.nan))
    k = np.asarray(L["k"], np.int64)
    for c, cid in enumerate(clips):
        m = sh == sha12(cid)
        kk = k[m]
        ok = kk < n_raw
        has[c, kk[ok]] = np.asarray(L["has"])[m][ok]
        lead[c, kk[ok]] = np.asarray(L["lead"])[m][ok]
        gap[c, kk[ok]] = np.asarray(L["gap"], np.float64)[m][ok]
        v[c, kk[ok]] = np.asarray(L["v"], np.float64)[m][ok]
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(out_dir + "/lead_table.npz", clip=np.asarray(clips), has=has, lead=lead, gap=gap, v=v)
    np.savez_compressed(out_dir + "/label_table.npz", clip=np.asarray(clips), off=J["off"], n=J["n"], has=J["has"], goal_y=J["goal_y"], goal_w=J["goal_w"], lon_cls=J["lon_cls"])
    return {"n_clips": C, "n_lead_frames": int(((has == 1) & (lead == 1)).sum()), "n_clips_with_lead": int((((has == 1) & (lead == 1)).any(1)).sum())}


def load_tables(d):
    out = {}
    for name in ("lead_table", "label_table"):
        with np.load(f"{d}/{name}.npz") as z:
            out[name] = {k: z[k] for k in z.files}
    return out


# ------------------------------------------------------------------------------------------------ part A
def _tangents(g):
    """unit tangent of the logged track at each knot, [n,30,2]; previous kept on a standstill step, (1,0) before any."""
    p = np.concatenate([np.zeros_like(g[:, :1]), g], axis=1)
    d = p[:, 1:] - p[:, :-1]
    nrm = np.linalg.norm(d, axis=-1)
    u = np.zeros_like(d)
    cur = np.tile(np.array([1.0, 0.0]), (len(g), 1))
    for k in range(g.shape[1]):
        ok = nrm[:, k] > 1e-6
        cur = np.where(ok[:, None], d[:, k] / np.where(ok, nrm[:, k], 1.0)[:, None], cur)
        u[:, k] = cur
    return u


def _speed(p):
    q = np.concatenate([np.zeros_like(p[:, :1]), p], axis=1)
    return np.linalg.norm(q[:, 1:] - q[:, :-1], axis=-1) / DT


def part_a_units(g, cl, ws, ci, lead_tab):
    """Per (window, knot) arrays of the A rules.  Returns dict of [n,30] arrays (valid mask included)."""
    g, cl = np.asarray(g, np.float64), np.asarray(cl, np.float64)
    ws, ci = np.asarray(ws, np.int64), np.asarray(ci, np.int64)
    raw = 2 * (ws[:, None] + np.arange(1, N_KNOTS + 1)[None, :])
    inb = raw < lead_tab["gap"].shape[1]
    rr = np.where(inb, raw, 0)
    cc = np.broadcast_to(ci[:, None], raw.shape)
    has, lead, gap, vfile = lead_tab["has"][cc, rr], lead_tab["lead"][cc, rr], lead_tab["gap"][cc, rr], lead_tab["v"][cc, rr]
    fin = np.isfinite(g).all(-1) & np.isfinite(cl).all(-1)
    valid = inb & (has == 1) & (lead == 1) & np.isfinite(gap) & fin
    u = _tangents(np.where(np.isfinite(g), g, 0.0))
    diff = np.where(np.isfinite(cl - g), cl - g, 0.0)
    e = (diff * u).sum(-1)
    cross = np.abs(diff[..., 0] * (-u[..., 1]) + diff[..., 1] * u[..., 0])
    gap_roll = gap - e
    v_log, v_roll = _speed(np.where(np.isfinite(g), g, 0.0)), _speed(np.where(np.isfinite(cl), cl, 0.0))
    tg_log = np.maximum(gap, 0.0) / np.maximum(v_log, TG_FLOOR_V)
    tg_roll = np.maximum(gap_roll, 0.0) / np.maximum(v_roll, TG_FLOOR_V)
    return {"valid": valid, "gap_log": gap, "gap_roll": gap_roll, "e": e, "cross": cross, "tg_log": tg_log, "tg_roll": tg_roll, "v_log": v_log, "v_roll": v_roll, "v_file": vfile}


def aggregate_a(U, inv, n_ep):
    v = U["valid"]
    out = {}

    def acc(name, val, mask):
        m = mask & v
        out[name + "_sum"] = np.bincount(inv, weights=np.where(m, val, 0.0).sum(1), minlength=n_ep)
        out[name + "_cnt"] = np.bincount(inv, weights=m.sum(1).astype(np.float64), minlength=n_ep)
    allm = np.ones_like(v)
    acc("gap_abs", np.abs(U["gap_roll"] - U["gap_log"]), allm)
    acc("gap_signed", U["gap_roll"] - U["gap_log"], allm)
    acc("tg_abs", np.abs(U["tg_roll"] - U["tg_log"]), allm)
    den = U["tg_log"] >= TG_LIMIT_S
    acc("tg_share", (U["tg_roll"] < TG_LIMIT_S).astype(np.float64), den)
    out["n_win_with_unit"] = np.bincount(inv, weights=v.any(1).astype(np.float64), minlength=n_ep)
    out["n_through"] = np.bincount(inv, weights=((U["gap_roll"] < 0) & v).sum(1).astype(np.float64), minlength=n_ep)
    out["n_cross2"] = np.bincount(inv, weights=((U["cross"] > CROSS_M) & v).sum(1).astype(np.float64), minlength=n_ep)
    out["align_absdiff"] = np.abs(U["v_file"] - U["v_log"])[v & np.isfinite(U["v_file"])]
    return out


# ------------------------------------------------------------------------------------------------ part B
def _conf(pred, lab):
    return np.array([(pred & lab).sum(), (~pred & lab).sum(), (pred & ~lab).sum(), (~pred & ~lab).sum()], np.float64)


def part_b(tk, lab_tab, ep_ids):
    """Per-episode counts and the labelled rows of the stop-goal decision, the lon STOP class and goal macro-F1 counts."""
    n_ep = len(ep_ids)
    pos_of = {int(e): i for i, e in enumerate(ep_ids)}
    ci = np.asarray(tk["ci"], np.int64)
    ep = np.array([pos_of[int(c)] for c in ci], np.int64)
    s = np.asarray(tk["j"], np.int64) - np.asarray(tk["age_t"], np.int64)
    off, nn = lab_tab["off"][ci], lab_tab["n"][ci]
    inb = (s >= 0) & (s < nn)
    row = np.where(inb, off + s, 0)
    has = inb & lab_tab["has"][row].astype(bool)
    gy = ((lab_tab["goal_y"][row].astype(np.int64)[:, None] >> np.arange(22)) & 1).astype(bool)
    gw = ((lab_tab["goal_w"][row].astype(np.int64)[:, None] >> np.arange(22)) & 1).astype(bool) & has[:, None]
    prob = np.asarray(tk["t_goal_prob"], np.float64)
    tick = np.asarray(tk["tick"], np.int64)
    out = {"n_rows": int(len(ci)), "n_rows_in_clip": int(inb.sum()), "n_rows_has": int(has.sum())}
    # stop goal
    lm = gw[:, STOP_I]
    lab = gy[:, STOP_I]
    pred = prob[:, STOP_I] >= 0.5
    stop_all, stop_t0 = np.zeros((n_ep, 4)), np.zeros((n_ep, 4))
    for e in range(n_ep):
        m = lm & (ep == e)
        stop_all[e] = _conf(pred[m], lab[m])
        m0 = m & (tick == 0)
        stop_t0[e] = _conf(pred[m0], lab[m0])
    out["stop_all"], out["stop_t0"] = stop_all, stop_t0
    out["row_ep"], out["row_score"], out["row_label"], out["row_tick"] = ep[lm], prob[lm, STOP_I], lab[lm].astype(np.int8), tick[lm]
    # lon STOP class
    lonc = lab_tab["lon_cls"][row].astype(np.int64)
    lmask = has & (lonc >= 0)
    lpred = np.asarray(tk["t_lon_logits"]).argmax(-1) == LON_STOP_I
    lstop = np.zeros((n_ep, 4))
    for e in range(n_ep):
        m = lmask & (ep == e)
        lstop[e] = _conf(lpred[m], lonc[m] == LON_STOP_I)
    out["lon_stop"] = lstop
    # goal macro F1 counts [E, 22, 3] = tp, fp, fn
    gp = prob > 0.5
    gf = np.zeros((n_ep, 22, 3))
    for e in range(n_ep):
        m = ep == e
        w, y, p = gw[m], gy[m], gp[m]
        gf[e, :, 0] = (w & p & y).sum(0)
        gf[e, :, 1] = (w & p & ~y).sum(0)
        gf[e, :, 2] = (w & ~p & y).sum(0)
    out["goal_f1"] = gf.reshape(n_ep, 66)
    return out


def ba_stat(c):
    c = np.asarray(c, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        tpr = c[..., 0] / (c[..., 0] + c[..., 1])
        tnr = c[..., 3] / (c[..., 3] + c[..., 2])
    return 0.5 * (tpr + tnr)


def f1_stat(c):
    c = np.asarray(c, np.float64).reshape(c.shape[:-1] + (22, 3))
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(tp + fn > 0, np.where(tp == 0, 0.0, 2 * tp / (2 * tp + fp + fn)), np.nan)
    cnt = np.isfinite(f).sum(-1)
    s = np.where(np.isfinite(f), f, 0.0).sum(-1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def auc_prep(score, label, ep):
    o = np.argsort(score, kind="stable")
    s, y, e = score[o], label[o].astype(bool), ep[o]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]]) if len(s) else np.array([], np.int64)
    return {"y": y, "ep": e, "starts": starts}


def auc_w(P, w):
    """Weighted AUROC; w [..., R] row weights in the sorted order of P.  Ties count one half; NaN with no positive or no negative weight."""
    if len(P["y"]) == 0:
        return np.full(w.shape[:-1], np.nan)
    wp, wn = w * P["y"], w * (~P["y"])
    gp, gn = np.add.reduceat(wp, P["starts"], axis=-1), np.add.reduceat(wn, P["starts"], axis=-1)
    below = np.cumsum(gn, axis=-1) - gn
    num = (gp * (below + 0.5 * gn)).sum(-1)
    den = wp.sum(-1) * wn.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# ------------------------------------------------------------------------------------------------ one arm
def run_arm(dump_dir, ticks_file, tables, expect_files=141, identity=False, tick_rows=True):
    t0 = time.time()
    D = RH.load_dump(dump_dir, expect_files)
    ep_ids = np.unique(D["eid"])
    inv = np.searchsorted(ep_ids, D["eid"])
    n_ep = len(ep_ids)
    cl = D["g"] if identity else D["cl"]
    U = part_a_units(D["g"], cl, D["ws"], D["clip_index"], tables["lead_table"])
    out = {"ep_ids": ep_ids, "windows_sha": np.array([hashlib.sha256(np.ascontiguousarray(D[k]).tobytes()).hexdigest() for k in ("g", "ws", "eid", "clip_index", "v0")])}
    for k, v in aggregate_a(U, inv, n_ep).items():
        out["A_" + k] = v
    meta = {"n_windows": int(len(D["g"])), "n_episodes": int(n_ep), "n_units": int(U["valid"].sum()), "identity": bool(identity)}
    if ticks_file and not identity:
        with np.load(ticks_file) as z:
            tk = {k: z[k] for k in ("ci", "j", "age_t", "tick", "t_goal_prob", "t_lon_logits")}
        B = part_b(tk, tables["label_table"], ep_ids)
        for k, v in B.items():
            if isinstance(v, np.ndarray):
                out["B_" + k] = v
            else:
                meta["B_" + k] = v
        meta["ticks_file"] = os.path.basename(ticks_file)
    meta["seconds"] = round(time.time() - t0, 2)
    return out, meta


def cmd_arm(a):
    tables = load_tables(a.tables_dir)
    out, meta = run_arm(a.dump, a.ticks, tables, a.expect_files, a.identity)
    np.savez_compressed(a.out + ".npz", **out)
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print("arm", json.dumps(meta))
    return 0


# ------------------------------------------------------------------------------------------------ the estimators
def _label_hi(d, lo, hi, n_req):
    d = np.asarray(d, np.float64)
    if len(d) < n_req or not np.all(np.isfinite(d)):
        return "OPEN", ["INCOMPLETE"] if len(d) < n_req else ["NON-FINITE"]
    if np.all(d > 0) and lo > 0:
        return "HIER", []
    if np.all(d < 0) and hi < 0:
        return "FLAT", []
    return "OPEN", []


def _cell_from(d, Dstar, n_req, higher, extra):
    Dstar = Dstar[np.isfinite(Dstar)]
    lo, hi = (np.percentile(Dstar, [2.5, 97.5]) if len(Dstar) else (np.nan, np.nan))
    p, na = RH.sign_flip_p(d) if np.all(np.isfinite(d)) else (float("nan"), 0)
    tlo, thi = RH.t_interval(d)
    if higher:
        lab, fl = _label_hi(d, lo, hi, n_req)
    else:
        lab, fl = RH.label_of(d, lo, hi, n_req)
    c = {"d_s": [float(x) for x in d], "D": float(np.mean(d)), "A_lo": float(lo), "A_hi": float(hi), "A_excludes_0": bool(lo > 0 or hi < 0), "A_n_boot_finite": int(len(Dstar)),
         "B_signflip_p": p, "B_n_assignments": na, "C_t_lo": tlo, "C_t_hi": thi, "label": lab, "flags": fl, "n_seeds": int(len(d)), "direction": "higher is better" if higher else "lower is better"}
    c.update(extra)
    return c


def ratio_cell(H, F, n_boot, n_req):
    """H, F: lists over seeds of (sum[E], cnt[E]).  The unit set is logged-defined, so counts must be equal."""
    for (sh, ch), (sf, cf) in zip(H, F):
        if not np.array_equal(ch, cf):
            raise ValueError("unit counts differ between arms: the unit set is not logged-defined")
    sE = np.asarray([h[0] - f[0] for h, f in zip(H, F)])
    cE = np.asarray([h[1] for h in H])
    c = RH.read_cell(sE, cE, n_boot, n_req)
    c["direction"] = "lower is better"
    return c


def stat_cell(H, F, stat, n_boot, n_req, higher=True, chunk=100):
    """H, F: [S,E,K] counts.  Slot value = stat(drawn-episode sums of the hierarchical arm) - stat(same for flat); same draws as RH.boot_two_level."""
    H, F = np.asarray(H, np.float64), np.asarray(F, np.float64)
    S, E, K = H.shape
    d = stat(H.sum(1)) - stat(F.sum(1))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, S, size=(n_boot, S))
    counts = rng.multinomial(E, np.full(E, 1.0 / E), size=(n_boot, S)).astype(np.float64)
    Dstar = np.empty(n_boot)
    for b0 in range(0, n_boot, chunk):
        sl = slice(b0, min(n_boot, b0 + chunk))
        nh = np.einsum("bse,bsek->bsk", counts[sl], H[idx[sl]])
        nf = np.einsum("bse,bsek->bsk", counts[sl], F[idx[sl]])
        with np.errstate(invalid="ignore"):
            v = stat(nh) - stat(nf)
            Dstar[sl] = np.nanmean(np.where(np.isfinite(v), v, np.nan), axis=1) if np.isfinite(v).any() else np.nan
    return _cell_from(d, Dstar, n_req, higher, {"n_episodes_with_data": int((H.sum((0, 2)) + F.sum((0, 2)) > 0).sum())})


def auc_cell(Ph, Pf, n_boot, n_req, n_ep, chunk=250):
    """Ph, Pf: lists over seeds of prepared AUC rows (sorted); episode-weighted, paired slot draws."""
    S = len(Ph)
    d = np.array([float(auc_w(Ph[s], np.ones(len(Ph[s]["y"])))) - float(auc_w(Pf[s], np.ones(len(Pf[s]["y"])))) for s in range(S)])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, S, size=(n_boot, S))
    counts = rng.multinomial(n_ep, np.full(n_ep, 1.0 / n_ep), size=(n_boot, S)).astype(np.float64)
    V = np.full((n_boot, S), np.nan)
    for s in range(S):
        bb, jj = np.nonzero(idx == s)
        for c0 in range(0, len(bb), chunk):
            b, j = bb[c0:c0 + chunk], jj[c0:c0 + chunk]
            cw = counts[b, j]
            V[b, j] = auc_w(Ph[s], cw[:, Ph[s]["ep"]]) - auc_w(Pf[s], cw[:, Pf[s]["ep"]])
    with np.errstate(invalid="ignore"):
        Dstar = np.nanmean(V, axis=1) if np.isfinite(V).any() else np.full(n_boot, np.nan)
    return _cell_from(d, Dstar, n_req, True, {})


# ------------------------------------------------------------------------------------------------ layouts
def arm_paths(t1_root, arm, s, mapping):
    if mapping is not None:
        stem = mapping[arm][str(s)]
        return f"{t1_root}/dump_{stem}_m10_causal_heads", f"{t1_root}/ticks/ticks_{stem}_m10_causal_heads.npz"
    d = f"{t1_root}/{arm}_s{s}"
    return f"{d}/dump_{arm}_s{s}_m10_causal_heads", f"{d}/ticks/ticks_{arm}_s{s}_m10_causal_heads.npz"


def all_rows(H, F, n_ep, nb, n_req, nbc=None):
    """H, F: lists over seeds of per-arm dicts.  -> {row key: cell}.  Same function for the read and for g1 (nb differs)."""
    cells = {}
    for key, name, unit in A_ROWS:
        cells[f"A:{key}"] = ratio_cell([(h["A_" + key + "_sum"], h["A_" + key + "_cnt"]) for h in H], [(f["A_" + key + "_sum"], f["A_" + key + "_cnt"]) for f in F], nb, n_req)
    for key, stat in (("stop_all", ba_stat), ("stop_t0", ba_stat), ("lon_stop", ba_stat), ("goal_f1", f1_stat)):
        cells[f"B:{key}"] = stat_cell([h["B_" + key] for h in H], [f["B_" + key] for f in F], stat, nb, n_req)
    for key, tsel in (("stop_auroc_all", None), ("stop_auroc_t0", 0)):
        Ph, Pf = [], []
        for h, f in zip(H, F):
            for src, dst in ((h, Ph), (f, Pf)):
                m = np.ones(len(src["B_row_tick"]), bool) if tsel is None else src["B_row_tick"] == tsel
                dst.append(auc_prep(src["B_row_score"][m], src["B_row_label"][m], src["B_row_ep"][m]))
        cells[f"B:{key}"] = auc_cell(Ph, Pf, nb, n_req, n_ep)
    return cells


def row_stats(arm_list, key):
    return [x[key] for x in arm_list]


def counts_block(H, F):
    """n units / windows / episodes per row family, from the hierarchical arm of each seed (logged-defined, equal for both arms)."""
    cnt = np.asarray([h["A_gap_abs_cnt"] for h in H])
    A = {"n_units_per_seed": [int(x) for x in cnt.sum(1)], "n_windows_with_unit_per_seed": [int(h["A_n_win_with_unit"].sum()) for h in H],
         "n_episodes_with_unit": int((cnt.sum(0) > 0).sum()),
         "n_tg_denominator_per_seed": [int(h["A_tg_share_cnt"].sum()) for h in H],
         "n_through_lead_hier_per_seed": [int(h["A_n_through"].sum()) for h in H], "n_through_lead_flat_per_seed": [int(f["A_n_through"].sum()) for f in F],
         "n_cross_gt2m_hier_per_seed": [int(h["A_n_cross2"].sum()) for h in H], "n_cross_gt2m_flat_per_seed": [int(f["A_n_cross2"].sum()) for f in F]}
    b = H[0]
    B = {"n_stop_rows_all": int(b["B_stop_all"].sum()), "n_stop_pos_all": int(b["B_stop_all"][:, :2].sum()), "n_stop_rows_tick0": int(b["B_stop_t0"].sum()),
         "n_stop_pos_tick0": int(b["B_stop_t0"][:, :2].sum()), "n_episodes_with_stop_pos": int((b["B_stop_all"][:, :2].sum(1) > 0).sum()),
         "n_lon_rows": int(b["B_lon_stop"].sum()), "n_lon_stop_pos": int(b["B_lon_stop"][:, :2].sum())}
    return A, B


# ------------------------------------------------------------------------------------------------ read
def cmd_read(a):
    seeds_all = [int(x) for x in a.seeds.split(",")]
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",")]
    mapping = None
    if a.map:
        with open(a.map, encoding="utf-8") as f:
            mapping = json.load(f)
    tables = load_tables(a.tables_dir)
    n_req = len(seeds_all) if a.no_card_counts else RH.N_SEEDS_CARD
    res = {"tier": TIER, "evidence_class": "MEASURED (ours, T1-kin dumps)", "stamp": STAMP, "STRATEGIC": "PAUSED (PI item 25)", "not_measurable": list(GAPS_NOT_MEASURABLE),
           "estimator": f"two-level bootstrap seeds then episodes (B {a.n_boot}, rng 0, paired draws; same engine as hierrep_read.py), exact sign-flip over seed differences, t interval (decides nothing); AUROC episode-weighted",
           "relaxed": ("--no-card-counts: six-seed label gate relaxed to the seeds given; these tags are not HIERREP arms" if a.no_card_counts else None),
           "controls": {}, "pairs": {}}
    t_start = time.time()
    # ---- g3 completeness and load
    arms = {}
    refW = None
    problems = []
    for hier, flat in pairs:
        for arm in (hier, flat):
            for s in seeds_all:
                if (arm, s) in arms:
                    continue
                try:
                    d, tf = arm_paths(a.t1_root, arm, s, mapping)
                except KeyError:
                    problems.append(f"{arm} seed {s}: not in --map")
                    continue
                n = len(glob.glob(d + "/ep*.npz"))
                if n != a.expect_files:
                    problems.append(f"{arm} seed {s}: {n} ep files in {d}, expected {a.expect_files}")
                    continue
                if not os.path.isfile(tf):
                    problems.append(f"{arm} seed {s}: tick file {tf} absent (part B needs it)")
                    continue
                if a.runs_root:
                    ok, why, _ = RH.arm_status(a.t1_root, a.runs_root, arm, s, a.expect_files)
                    if not ok:
                        problems.append(f"{arm} seed {s}: {why}")
                        continue
                arms[(arm, s)] = (d, tf)
    res["controls"]["g3_completeness"] = {"PASS": not problems, "problems": problems, "expect_files": a.expect_files}
    if problems:
        _write(res, a.out_dir)
        print("REFUSED (g3):", "; ".join(problems[:6]))
        return 3
    data = {}
    for (arm, s), (d, tf) in arms.items():
        out, meta = run_arm(d, tf, tables, a.expect_files)
        if refW is None:
            refW = out["windows_sha"]
            res["window_sha256"] = [str(x) for x in refW]
            ref_ep = out["ep_ids"]
        if not np.array_equal(out["windows_sha"], refW) or not np.array_equal(out["ep_ids"], ref_ep):
            res["controls"]["g3_completeness"] = {"PASS": False, "problems": [f"windows differ: {arm} seed {s}"]}
            _write(res, a.out_dir)
            print(f"REFUSED (g3): windows differ between arms: {arm} seed {s}")
            return 3
        data[(arm, s)] = out
        data[(arm, s)]["_meta"] = meta
    n_ep = len(ref_ep)
    # ---- row sets equal across arms (labels depend only on windows and ticks)
    for (arm, s), o in data.items():
        r0 = data[next(iter(data))]
        if not (np.array_equal(o["B_row_ep"], r0["B_row_ep"]) and np.array_equal(o["B_row_label"], r0["B_row_label"]) and np.array_equal(o["B_row_tick"], r0["B_row_tick"])):
            res["controls"]["g3_completeness"] = {"PASS": False, "problems": [f"labelled rows differ: {arm} seed {s}"]}
            _write(res, a.out_dir)
            print("REFUSED (g3): labelled rows differ between arms")
            return 3
    # ---- g2 identity (first arm, through the same code path with the logged track as the rolled one)
    first = next(iter(arms))
    d0 = arms[first][0]
    idout, _ = run_arm(d0, None, tables, a.expect_files, identity=True)
    zeros = {k: float(np.abs(idout["A_" + k + "_sum"]).max()) for k in A_KEYS}
    res["controls"]["g2_identity"] = {"PASS": all(v == 0.0 for v in zeros.values()), "max_abs_sum_per_metric": zeros, "arm_used": f"{first[0]} seed {first[1]}", "n_units": int(idout["A_gap_abs_cnt"].sum())}
    # ---- g4 alignment
    al = np.concatenate([o["A_align_absdiff"] for o in data.values()])
    med = float(np.median(al)) if len(al) else float("nan")
    res["controls"]["g4_alignment"] = {"PASS": bool(len(al) > 0 and med <= ALIGN_MAX_MPS), "median_abs_dv_mps": med, "n_units": int(len(al)), "limit_mps": ALIGN_MAX_MPS}
    # ---- pairs
    for hier, flat in pairs:
        S = [s for s in seeds_all]
        H = [data[(hier, s)] for s in S]
        F = [data[(flat, s)] for s in S]
        cells = all_rows(H, F, n_ep, a.n_boot, n_req)
        cb, bb = counts_block(H, F)
        res["pairs"][f"{hier}_vs_{flat}"] = {"seeds": S, "n_episodes": int(n_ep), "n_windows": int(H[0]["_meta"]["n_windows"]), "A_counts": cb, "B_counts": bb, "cells": cells}
        # ---- g1 shuffled arm labels, 20 draws, reduced B
        labs = {k: [] for k in cells}
        for k in range(20):
            rng = np.random.default_rng(2000 + k)
            sw = rng.random(len(S)) < 0.5
            Hs = [F[i] if sw[i] else H[i] for i in range(len(S))]
            Fs = [H[i] if sw[i] else F[i] for i in range(len(S))]
            cc = all_rows(Hs, Fs, n_ep, a.n_boot_control, n_req)
            for key, c in cc.items():
                labs[key].append(c["label"])
        per_row = {key: int(sum(x == "OPEN" for x in v)) for key, v in labs.items()}
        informative = len(S) >= 4
        ok_rows = {key: n >= 18 for key, n in per_row.items()}
        res["controls"][f"g1_shuffle_{hier}_vs_{flat}"] = {"PASS": (bool(all(ok_rows.values())) if informative else None), "status": ("PASS" if all(ok_rows.values()) else "FAIL") if informative else "NOT-INFORMATIVE (n seeds < 4)",
                                                         "n_open_of_20_per_row": per_row, "n_boot_control": a.n_boot_control, "nominal_rate_note": "rows must read OPEN in >= 18 of 20 draws"}
    flags = [v.get("PASS") for v in res["controls"].values() if isinstance(v, dict) and "PASS" in v]
    res["controls"]["ALL_PASS"] = bool(all(f is not False for f in flags))
    res["seconds"] = round(time.time() - t_start, 1)
    _write(res, a.out_dir)
    print(f"read written | pairs {list(res['pairs'])} | seeds {seeds_all} | controls {'PASS' if res['controls']['ALL_PASS'] else 'FAILED'} | {res['seconds']} s")
    return 0 if res["controls"]["ALL_PASS"] else 2


def _write(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir + "/RESULT_HIERGAP.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(out_dir + "/VERDICT_HIERGAP.md", "w", encoding="utf-8") as f:
        f.write(verdict(res))


def _fmt(c, nd=4):
    return f"{c['D']:+.{nd}f} [{c['A_lo']:+.{nd}f}, {c['A_hi']:+.{nd}f}]"


def verdict(res):
    L = []
    P = L.append
    C = res["controls"]
    P("# VERDICT_HIERGAP (HIERREP gap rows; MEASURED once run; " + res["tier"] + "; a diagnostic, never driving performance)")
    P("")
    P(res["stamp"] + ". STRATEGIC: PAUSED (PI item 25), never a pass.")
    if C.get("ALL_PASS") is not True:
        P("")
        P("**READ NOT VALID: a control failed or the input was refused (see Controls).** Tables, if any, are printed for the record only.")
    if res.get("relaxed"):
        P("")
        P("Relaxation in force: " + res["relaxed"])
    P("")
    P("Estimator: " + res["estimator"] + ".")
    P("Differences are hierarchical minus flat. Part A rows: lower is better (negative = hierarchy better). Part B rows: higher is better (positive = hierarchy better). Label per R10 (HIER / FLAT / OPEN).")
    for name, pr in res.get("pairs", {}).items():
        P("")
        P(f"## {name}: seeds {pr['seeds']} (n = {len(pr['seeds'])}), {pr['n_windows']} windows, {pr['n_episodes']} episodes (T1-kin dump)")
        A, B = pr["A_counts"], pr["B_counts"]
        P(f"Part A units (window, knot with a lead): per seed {A['n_units_per_seed']}, windows with a unit {A['n_windows_with_unit_per_seed']}, episodes with a unit {A['n_episodes_with_unit']}; time-gap denominator (logged tg >= 1.0 s) {A['n_tg_denominator_per_seed']}; rolled through the lead (hier / flat) {A['n_through_lead_hier_per_seed']} / {A['n_through_lead_flat_per_seed']}; cross-path offset > 2.0 m (hier / flat) {A['n_cross_gt2m_hier_per_seed']} / {A['n_cross_gt2m_flat_per_seed']}.")
        P(f"Part B rows (seed 0, hierarchical arm): stop-goal rows all ticks {B['n_stop_rows_all']} ({B['n_stop_pos_all']} positive), tick 0 {B['n_stop_rows_tick0']} ({B['n_stop_pos_tick0']} positive), episodes with a positive {B['n_episodes_with_stop_pos']}; lon rows {B['n_lon_rows']} ({B['n_lon_stop_pos']} STOP).")
        P("")
        P("| row | family | D [95 % interval A] | all d_s same sign | sign-flip p | label | n seeds |")
        P("|---|---|---|---|---|---|---|")
        fam = {"A:gap_abs": "LONGITUDINAL (m)", "A:gap_signed": "LONGITUDINAL (m)", "A:tg_abs": "LONGITUDINAL (s)", "A:tg_share": "LONGITUDINAL (share)"}
        for k, c in pr["cells"].items():
            same = all(x > 0 for x in c["d_s"]) or all(x < 0 for x in c["d_s"])
            P(f"| {k} | {fam.get(k, 'TACTICAL')} | {_fmt(c)} | {same} | {c['B_signflip_p']:.4f} | {c['label']} {','.join(c['flags'])} | {c['n_seeds']} |")
    P("")
    P("## Controls")
    for k, v in C.items():
        if k == "ALL_PASS":
            continue
        P(f"* {k}: " + json.dumps(v, default=str)[:900])
    P(f"* ALL_PASS: {C.get('ALL_PASS')}")
    P("")
    P("## Named gaps that remain (not passes)")
    for g in res["not_measurable"]:
        P("* " + g)
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("tables")
    t.add_argument("--lead-npz", required=True)
    t.add_argument("--join-npz", required=True)
    t.add_argument("--out-dir", required=True)
    r = sp.add_parser("arm")
    r.add_argument("--dump", required=True)
    r.add_argument("--ticks", default="")
    r.add_argument("--tables-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--identity", action="store_true")
    r.add_argument("--expect-files", type=int, default=141)
    d = sp.add_parser("read")
    d.add_argument("--t1-root", required=True)
    d.add_argument("--tables-dir", required=True)
    d.add_argument("--out-dir", required=True)
    d.add_argument("--pairs", default="HIER:FLAT,DAG:FLATDAG")
    d.add_argument("--seeds", default="0,1,2,3,4,5")
    d.add_argument("--map", default="")
    d.add_argument("--runs-root", default="")
    d.add_argument("--n-boot", type=int, default=2000)
    d.add_argument("--n-boot-control", type=int, default=500)
    d.add_argument("--expect-files", type=int, default=141)
    d.add_argument("--no-card-counts", action="store_true")
    a = ap.parse_args()
    if a.cmd == "tables":
        print("tables", json.dumps(build_tables(a.lead_npz, a.join_npz, a.out_dir)))
        return 0
    return cmd_arm(a) if a.cmd == "arm" else cmd_read(a)


if __name__ == "__main__":
    sys.exit(main())
