"""measure_ckpt120_coverage.py -- PREREG_V8F_CKPT120 design measurement (CPU, read-only, zero training, no model output read).

What it measures, for the eval-141 grid (2,399 windows) and the train cache (4,572 clips):
  * RC-A at arcs 30/50/80/100/120/150 m recomputed with the v9 builder's own functions (build_v9_labels.py at a4357e79,
    git blob edf5424d...) from the egomotion logs: validity = rc_point inside s_hi_valid AND chord sagitta <= 0.25 m,
    exactly rc_fields(tag "") with a longer lookahead.
  * CONTROL R1: the recomputed 30/50/80 validity must equal the join's rc_ok (eval) and the release's rc_A*_valid (train)
    on every labelled step, and the recomputed (x, y, psi) must match the release to float32 precision.
  * CONTROL R2: the vectorised sagitta must equal B.chord_sagitta_max on every eval query (exact).
  * CONTROL R3: the S_CKPT scorable n at 80 m (join rc_ok) must reproduce the STRAT1 reader's 244 windows / 25 episodes.
  * per arc: S_CKPT-scorable windows (text TURN side, d_text + 20 <= arc, rc_ok at arc), per side, episodes, the best
    constant (majority side) level, and how many of them the model's nav input announces (token on: d <= max(30, 6 v)).
  * K-PERFECT (labels only, no model): paired episode-cluster bootstrap (taniteval ci.py at a4357e79, blob 4f17fe11...)
    of a planted perfect follower, a nav-informed follower (right exactly where the token is on, else straight) and a
    nav-or-majority follower, each minus the best constant, at every arc: can the bar be passed at all?
  * future-time coverage (log end - NOW >= 6/10/15/20/30 s) and the time-to-arc 120 / v.
Run:  python measure_ckpt120_coverage.py --builder <bv9.py> --ci <ci.py> --out <json>   (paths default below)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time

import numpy as np

ARCS = (30.0, 50.0, 80.0, 100.0, 120.0, 150.0)
T_FUT = (6.0, 10.0, 15.0, 20.0, 30.0)
COMPLETE_M = 20.0
SIG, NSIG, SAG_MAX = 8.0, 3.0, 0.25
BLOB_BUILDER = "edf5424d613e90844f0ed984c3a2677c6b6c02e7"
BLOB_CI = "4f17fe11458d593978c3d303c48ee2aece356158"
MD5_V9_TRAIN = "83e30bd3791f0767707eade5f4671759"
MD5_V9_EVAL = "6b5c7f207cffc3b7eb3cd527fd433599"
IN = "F:/TanitAD-scratch/strat1/read/inputs"
EGO = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/{}.parquet"


def git_blob(path):
    b = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def fhash(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def load_mod(name, path, blob):
    got = git_blob(path)
    if got != blob:
        raise SystemExit(f"{path}: git blob {got} != pinned {blob}")
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class SparseMax:
    def __init__(self, a):
        a = np.asarray(a, np.float64)
        self.t = [a]
        k = 1
        while (1 << k) <= len(a):
            p = self.t[-1]
            h = 1 << (k - 1)
            self.t.append(np.maximum(p[:-h], p[h:]))
            k += 1

    def query(self, lo, hi):  # inclusive, lo <= hi, vectorised
        n = hi - lo + 1
        k = np.floor(np.log2(np.maximum(n, 1))).astype(int)
        out = np.empty(len(lo))
        for kk in np.unique(k):
            m = k == kk
            t = self.t[kk]
            out[m] = np.maximum(t[lo[m]], t[hi[m] - (1 << kk) + 1])
        return out


class ClipGeom:
    def __init__(self, B, path):
        self.B = B
        self.log = B.load_log(path)
        self.tr = B.track10(self.log)
        self.p8 = B.smooth_path(self.tr, SIG)
        self.s_nat = np.asarray(B.s_at(self.tr, self.log.ts), np.float64)
        lg = self.log
        c = np.hypot(np.diff(lg.x), np.diff(lg.y)) * np.abs(np.diff(lg.yaw)) / 8.0
        self.nch = len(c)
        self.sm = SparseMax(c) if len(c) else None

    def sag(self, s_lo, s_hi):
        s = self.s_nat
        i0 = np.searchsorted(s, s_lo, "left")
        i1 = np.searchsorted(s, s_hi, "right") - 1
        cnt = i1 - i0 + 1
        out = np.full(len(s_lo), np.inf)
        ok = cnt >= 2
        if ok.any() and self.sm is not None:
            lo = np.maximum(i0[ok] - 1, 0)
            hiend = np.minimum(i1[ok] + 2, len(s))
            out[ok] = self.sm.query(lo, hiend - 2)
        return out

    def rc(self, now, arcs=ARCS):
        """-> ok [n, A], x, y, psi_deg [n, A], v_now [n], fut_s [n] (log end - now)"""
        B, p8 = self.B, self.p8
        now = np.asarray(now, np.float64)
        x0, y0, psi0, v0 = (np.asarray(a, np.float64) for a in B.sample(self.log, now))
        s_now = np.asarray(B.s_at(self.tr, now), np.float64)
        A = len(arcs)
        ok = np.zeros((len(now), A), bool)
        X, Y, P = (np.full((len(now), A), np.nan) for _ in range(3))
        for m, L in enumerate(arcs):
            st = s_now + L
            ins = (st <= p8.s_hi_valid) & (st >= p8.s[0])
            px, py, pp = np.interp(st, p8.s, p8.x), np.interp(st, p8.s, p8.y), np.interp(st, p8.s, p8.psi)
            c, s = np.cos(psi0), np.sin(psi0)
            dx, dy = px - x0, py - y0
            xl, yl = c * dx + s * dy, -s * dx + c * dy
            rel = (pp - psi0 + math.pi) % (2 * math.pi) - math.pi
            sg = self.sag(s_now - NSIG * SIG, s_now + L + NSIG * SIG)
            good = ins & (sg <= SAG_MAX)
            ok[:, m] = good
            X[good, m], Y[good, m], P[good, m] = xl[good], yl[good], np.degrees(rel[good])
        return ok, X, Y, P, v0, self.log.t_end - now


def side_table(vocab):
    sd, th = [], []
    for n in vocab:
        n = str(n)
        if not n.startswith("TURN"):
            sd.append(0), th.append(np.nan)
            continue
        th.append(10.0 if "SLIGHT" in n else 25.0)
        sd.append(1 if n.endswith("_L") else -1 if n.endswith("_R") else 0)
    return np.array(sd, np.int8), np.array(th)


def q(a, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    return {str(x): (round(float(np.quantile(a, x)), 2) if a.size else None) for x in qs}


def arc_block(CI, ok_arc, side, d, on, eid, v, arc):
    sc = ok_arc & (side != 0) & np.isfinite(d) & (d + COMPLETE_M <= arc)
    s = side[sc]
    nL, nR = int((s == 1).sum()), int((s == -1).sum())
    blk = {"n_windows": int(sc.sum()), "n_episodes": int(len(np.unique(eid[sc]))), "n_left": nL, "n_right": nR}
    if sc.sum() == 0:
        return blk, sc
    maj = 1 if nL >= nR else -1
    bc = (s == maj).astype(float)
    on_s = on[sc]
    blk.update({"majority_side": "left" if maj == 1 else "right", "best_constant_level": round(float(bc.mean()), 4),
                "minority_frac": round(1 - float(bc.mean()), 4),
                "token_on": int(on_s.sum()), "token_on_left": int((on_s & (s == 1)).sum()), "token_on_right": int((on_s & (s == -1)).sum()),
                "token_off_episodes_only": int(len(set(eid[sc][~on_s]) - set(eid[sc][on_s]))),
                "d_text_q": q(d[sc]), "t_to_turn_s_q": q(d[sc] / np.maximum(v[sc], 0.1)),
                "n_new_d_text_gt_60": int((d[sc] > 60.0).sum()), "n_new_d_text_gt_60_token_on": int(((d[sc] > 60.0) & on_s).sum())})
    e = eid[sc]
    perfect = np.ones(sc.sum())
    navinf = on_s.astype(float)
    nav_or_maj = np.where(on_s, 1.0, bc)
    kp = {}
    for name, a in (("perfect", perfect), ("nav_informed", navinf), ("nav_or_majority", nav_or_maj)):
        r = CI.paired_episode_cluster_bootstrap(a, bc, e, n_boot=2000, seed=0)
        kp[name] = {k: r[k] for k in ("delta", "lo", "hi", "separated")}
    # balanced form: mean over sides of the per-side level; a constant reads exactly 0.5
    def bal(vals, sides):
        m = [vals[sides == z].mean() for z in (1, -1) if (sides == z).any()]
        return float(np.mean(m)) if m else float("nan")
    uniq = np.unique(e)
    idx = {u: np.nonzero(e == u)[0] for u in uniq}
    rng = np.random.default_rng(0)
    bal_rows = {}
    for name, a in (("perfect", perfect), ("nav_informed", navinf), ("nav_or_majority", nav_or_maj)):
        pt = bal(a, s) - 0.5
        draws = []
        for _ in range(2000):
            sel = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])
            draws.append(bal(a[sel], s[sel]) - 0.5)
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        bal_rows[name] = {"delta": round(pt, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4), "separated": bool(lo > 0 or hi < 0)}
    blk["K_PERFECT_minus_best_constant"] = kp
    blk["K_PERFECT_balanced_minus_half"] = bal_rows
    return blk, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builder", required=True)
    ap.add_argument("--ci", required=True)
    ap.add_argument("--v9-train", default="E:/Projects/TanitAD-artifacts/v9labels/v9_labels_train.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-every", type=int, default=1, help="process every k-th train clip (1 = all)")
    ap.add_argument("--allow-release-md5", default="", help="measure train on a v9 release other than the one of record (scoped in the output)")
    ap.add_argument("--dump-npz", default="", help="write the eval-grid per-window ground truth here")
    a = ap.parse_args()
    t0 = time.time()
    B = load_mod("bv9", a.builder, BLOB_BUILDER)
    CI = load_mod("ci_a43", a.ci, BLOB_CI)
    inp = {k: f"{IN}/{k}" for k in ("join_eval141.npz", "navtok_table.npz", "navtext_nav.npz", "eval_grid_windows.json")}
    rep = {"tool": "measure_ckpt120_coverage.py", "evidence_class": "MEASURED (ours; labels and logs only, no model output, zero training)",
           "tier": "n/a (label geometry and coverage)", "builder_blob": BLOB_BUILDER, "ci_blob": BLOB_CI,
           "inputs_sha256": {k: fhash(p, "sha256") for k, p in inp.items()}, "arcs_m": list(ARCS), "complete_m": COMPLETE_M}
    J = np.load(inp["join_eval141.npz"])
    NT = np.load(inp["navtok_table.npz"], allow_pickle=True)
    NX = np.load(inp["navtext_nav.npz"], allow_pickle=True)
    G = json.load(open(inp["eval_grid_windows.json"]))
    sd, th = side_table(NT["vocab"])
    # ---------------------------------------------------------------- eval
    clips = list(J["clip"])
    assert list(NT["eval_clip"]) == clips == list(NX["eval_clip"])
    off = J["off"]
    rows, eid = [], []
    for c in G["order"]:
        ci = clips.index(c)
        for t in G["windows"][c]:
            rows.append(off[ci] + int(t))
            eid.append(ci)
    rows, eid = np.array(rows), np.array(eid)
    rep["eval_grid"] = {"n_windows": int(len(rows)), "n_episodes": int(len(np.unique(eid)))}
    has = J["has"][rows]
    now = J["now_s"][rows]
    N = len(rows)
    ok = np.zeros((N, len(ARCS)), bool)
    P = np.full((N, len(ARCS)), np.nan)
    X = np.full((N, len(ARCS)), np.nan)
    Y = np.full((N, len(ARCS)), np.nan)
    v = np.full(N, np.nan)
    fut = np.full(N, np.nan)
    sag_mismatch, sag_n = 0, 0
    for ci in np.unique(eid):
        m = (eid == ci) & has
        if not m.any():
            continue
        g = ClipGeom(B, EGO.format(clips[ci]))
        o, x, y, p, vv, ff = g.rc(now[m])
        ok[m], X[m], Y[m], P[m], v[m], fut[m] = o, x, y, p, vv, ff
        s_now = np.asarray(B.s_at(g.tr, now[m]))
        for L in ARCS:   # CONTROL R2
            vec = g.sag(s_now - NSIG * SIG, s_now + L + NSIG * SIG)
            ref = np.array([B.chord_sagitta_max(g.tr, g.log, sn - NSIG * SIG, sn + L + NSIG * SIG, g.s_nat) for sn in s_now])
            sag_mismatch += int((~((vec == ref) | (np.isinf(vec) & np.isinf(ref)))).sum())
            sag_n += len(ref)
    rcok_join = J["rc_ok"][rows]
    rc_join = J["rc"][rows]
    r1 = {}
    for m, L in enumerate((30.0, 50.0, 80.0)):
        k = ARCS.index(L)
        r1[f"{int(L)}_valid_mismatch"] = int((ok[has, k] != rcok_join[has, m]).sum())
        both = has & ok[:, k] & rcok_join[:, m]
        r1[f"{int(L)}_xy_maxabs_m"] = float(np.nanmax(np.abs(np.stack([X[both, k] - rc_join[both, m, 0], Y[both, k] - rc_join[both, m, 1]])))) if both.any() else None
        r1[f"{int(L)}_psi_maxabs_deg"] = float(np.nanmax(np.abs(P[both, k] - np.degrees(rc_join[both, m, 2])))) if both.any() else None
    rep["CONTROL_R1_eval"] = {"n_has": int(has.sum()), **r1, "pass": all(r1[f"{L}_valid_mismatch"] == 0 for L in (30, 50, 80))}
    rep["CONTROL_R2_sagitta"] = {"n_queries": sag_n, "mismatch": sag_mismatch, "pass": sag_mismatch == 0}
    cls = NT["eval_cls_id"][rows].astype(int)
    text_ok = NT["eval_flag"][rows] == 0
    side = np.where(text_ok, sd[cls], 0).astype(np.int8)
    d = np.where(side != 0, NX["eval_nav_d"][rows].astype(np.float64), np.nan)
    nav = NX["eval_nav"][rows]
    on = (nav[:, 1] + nav[:, 2]) > 0
    on_side_agree = bool(np.all(np.where(nav[on, 1] > 0, 1, -1) == side[on]))
    rep["eval_token_side_agrees_with_text_side"] = on_side_agree
    E = {"rc_ok_rate_over_has": {str(int(L)): round(float(ok[has, k].mean()), 4) for k, L in enumerate(ARCS)},
         "abs_psi_ge_25deg_rate_over_rc_ok": {str(int(L)): round(float((np.abs(P[ok[:, k], k]) >= 25).mean()), 4) for k, L in enumerate(ARCS)},
         "future_log_ge_s_rate_over_has": {str(int(T)): round(float((fut[has] >= T).mean()), 4) for T in T_FUT},
         "v_now_q": q(v[has]), "t_to_120m_s_q": q(120.0 / np.maximum(v[has], 0.1)),
         "t_to_120m_in_6_30s_rate": round(float(((120.0 / np.maximum(v[has], 0.1) >= 6) & (120.0 / np.maximum(v[has], 0.1) <= 30)).mean()), 4)}
    if a.dump_npz:   # per-window ground truth for the power simulation (labels only; no model output)
        np.savez_compressed(a.dump_npz, rows=rows, eid=eid, side=side, d_text=d, token_on=on, rc_ok=ok, v_now=v, arcs=np.array(ARCS), has=has)
    blocks = {}
    for k, L in enumerate(ARCS):
        blk, _ = arc_block(CI, ok[:, k], side, d, on, eid, v, L)
        blocks[str(int(L))] = blk
    E["scorable_by_arc_recomputed_rc"] = blocks
    jb = {}
    for m, L in enumerate((30.0, 50.0, 80.0)):
        blk, _ = arc_block(CI, rcok_join[:, m], side, d, on, eid, v, L)
        jb[str(int(L))] = {kk: blk[kk] for kk in ("n_windows", "n_episodes", "n_left", "n_right") if kk in blk} | {"best_constant_level": blk.get("best_constant_level")}
    E["scorable_by_arc_join_rc_ok"] = jb
    rep["CONTROL_R3_reader_80m"] = {"expected": {"n_windows": 244, "n_episodes": 25, "best_constant": 0.7664},
                                    "got": jb["80"], "pass": jb["80"]["n_windows"] == 244 and jb["80"]["n_episodes"] == 25}
    # dense eval: every labelled cache step (not only the 2,399-window grid) -- same episodes, more (correlated) windows
    allr = np.nonzero(J["has"])[0]
    ce = np.searchsorted(off, allr, "right") - 1
    okd = np.zeros((len(allr), len(ARCS)), bool)
    vd = np.full(len(allr), np.nan)
    for ci in np.unique(ce):
        m = ce == ci
        g = ClipGeom(B, EGO.format(clips[ci]))
        o, _, _, _, vv, _ = g.rc(J["now_s"][allr[m]])
        okd[m], vd[m] = o, vv
    clsd = NT["eval_cls_id"][allr].astype(int)
    sided = np.where(NT["eval_flag"][allr] == 0, sd[clsd], 0).astype(np.int8)
    dd = np.where(sided != 0, NX["eval_nav_d"][allr].astype(np.float64), np.nan)
    ond = (NX["eval_nav"][allr, 1] + NX["eval_nav"][allr, 2]) > 0
    dense = {}
    for k, L in enumerate(ARCS):
        sc = okd[:, k] & (sided != 0) & np.isfinite(dd) & (dd + COMPLETE_M <= L)
        dense[str(int(L))] = {"n_steps": int(sc.sum()), "n_episodes": int(len(np.unique(ce[sc]))), "n_left": int((sc & (sided == 1)).sum()),
                              "n_right": int((sc & (sided == -1)).sum()), "token_on": int((sc & ond).sum())}
    E["dense_all_labelled_steps"] = dense
    rep["eval"] = E
    rep["eval_t_elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps({k: rep[k] for k in ("CONTROL_R1_eval", "CONTROL_R2_sagitta", "CONTROL_R3_reader_80m")}, indent=1), flush=True)
    # ---------------------------------------------------------------- train
    md5 = fhash(a.v9_train, "md5")
    rep["v9_train_md5"] = md5
    rep["v9_train_is_release_of_record"] = md5 == MD5_V9_TRAIN
    if md5 != MD5_V9_TRAIN and md5 != a.allow_release_md5:
        raise SystemExit(f"v9 train release md5 {md5} != {MD5_V9_TRAIN} (pass --allow-release-md5 {md5} to measure on it, scoped)")
    R = np.load(a.v9_train, allow_pickle=True)
    rsha = [str(s) for s in R["clip__sha12"]]
    r0s, nrs, k0s = R["clip__row0"], R["clip__n_rows"], R["clip__k0"]
    by = {s: (int(r0s[i]), int(nrs[i]), int(k0s[i])) for i, s in enumerate(rsha)}
    now_r = R["row__now_s"]
    val = {L: R[f"row__rc_A{L}_valid"] for L in (30, 50, 80)}
    rx = {L: (R[f"row__rc_A{L}_x"], R[f"row__rc_A{L}_y"], R[f"row__rc_A{L}_psi"]) for L in (30, 50, 80)}
    tclips, toff, tn = list(NT["train_clip"]), NT["train_off"], NT["train_n"]
    tcls, tflag, tnd, tnav = NT["train_cls_id"], NT["train_flag"], NX["train_nav_d"], NX["train_nav"]
    acc = {"n_steps": 0, "n_has": 0, "ok": np.zeros(len(ARCS), np.int64), "turn25": np.zeros(len(ARCS), np.int64),
           "fut": np.zeros(len(T_FUT), np.int64), "sc": np.zeros(len(ARCS), np.int64), "sc_on": np.zeros(len(ARCS), np.int64),
           "sc_L": np.zeros(len(ARCS), np.int64), "sc_R": np.zeros(len(ARCS), np.int64), "mis": {30: 0, 50: 0, 80: 0},
           "xy": {30: 0.0, 50: 0.0, 80: 0.0}, "psi": {30: 0.0, 50: 0.0, 80: 0.0}, "missing_log": 0, "missing_release": 0}
    acc_ex = {"n_has": 0, "ok": np.zeros(len(ARCS), np.int64)}
    for ci in range(0, len(tclips), a.train_every):
        cid = tclips[ci]
        n = int(tn[ci])
        acc["n_steps"] += n
        meta = by.get(hashlib.sha256(cid.encode()).hexdigest()[:12])
        if meta is None:
            acc["missing_release"] += 1
            continue
        r0, nr, k0 = meta
        j = np.arange(n)
        k = 2 * j
        ins = (k >= k0) & (k < k0 + nr)
        if not ins.any():
            continue
        rr = r0 + (k[ins] - k0)
        try:
            g = ClipGeom(B, EGO.format(cid))
        except FileNotFoundError:
            acc["missing_log"] += 1
            continue
        o, x, y, p, vv, ff = g.rc(now_r[rr])
        acc["n_has"] += int(ins.sum())
        acc["ok"] += o.sum(0)
        acc["turn25"] += (o & (np.abs(np.nan_to_num(p)) >= 25)).sum(0)
        acc["fut"] += np.array([(ff >= T).sum() for T in T_FUT])
        if not (540 <= ci < 600):
            acc_ex["n_has"] += int(ins.sum())
            acc_ex["ok"] += o.sum(0)
        for L in (30, 50, 80):
            kk = ARCS.index(float(L))
            rv = val[L][rr] == 1
            acc["mis"][L] += int((rv != o[:, kk]).sum())
            b2 = rv & o[:, kk]
            if b2.any():
                acc["xy"][L] = max(acc["xy"][L], float(np.max(np.abs(np.stack([x[b2, kk] - rx[L][0][rr][b2], y[b2, kk] - rx[L][1][rr][b2]])))))
                acc["psi"][L] = max(acc["psi"][L], float(np.max(np.abs(p[b2, kk] - rx[L][2][rr][b2]))))
        tr_rows = toff[ci] + j[ins]
        tok = tflag[tr_rows] == 0
        s_ = np.where(tok, sd[tcls[tr_rows].astype(int)], 0)
        d_ = np.where(s_ != 0, tnd[tr_rows].astype(np.float64), np.nan)
        on_ = (tnav[tr_rows, 1] + tnav[tr_rows, 2]) > 0
        for kk, L in enumerate(ARCS):
            sc = o[:, kk] & (s_ != 0) & np.isfinite(d_) & (d_ + COMPLETE_M <= L)
            acc["sc"][kk] += int(sc.sum())
            acc["sc_on"][kk] += int((sc & on_).sum())
            acc["sc_L"][kk] += int((sc & (s_ == 1)).sum())
            acc["sc_R"][kk] += int((sc & (s_ == -1)).sum())
        if ci % 500 == 0:
            print(f"[train] clip {ci}/{len(tclips)} t={time.time() - t0:.0f}s", flush=True)
    nh = max(acc["n_has"], 1)
    rep["train"] = {"n_clips_processed": len(range(0, len(tclips), a.train_every)), "train_every": a.train_every,
                    "n_steps": acc["n_steps"], "n_labelled_steps": acc["n_has"], "missing_release": acc["missing_release"], "missing_log": acc["missing_log"],
                    "rc_ok_steps": {str(int(L)): int(acc["ok"][k]) for k, L in enumerate(ARCS)},
                    "rc_ok_rate_over_labelled": {str(int(L)): round(float(acc["ok"][k] / nh), 4) for k, L in enumerate(ARCS)},
                    "rc_ok_rate_excl_540_600": {str(int(L)): round(float(acc_ex["ok"][k] / max(acc_ex["n_has"], 1)), 4) for k, L in enumerate(ARCS)},
                    "abs_psi_ge_25deg_steps": {str(int(L)): int(acc["turn25"][k]) for k, L in enumerate(ARCS)},
                    "abs_psi_ge_25deg_rate_over_rc_ok": {str(int(L)): round(float(acc["turn25"][k] / max(acc["ok"][k], 1)), 4) for k, L in enumerate(ARCS)},
                    "future_log_ge_s_rate": {str(int(T)): round(float(acc["fut"][i] / nh), 4) for i, T in enumerate(T_FUT)},
                    "text_turn_scorable_steps": {str(int(L)): {"n": int(acc["sc"][k]), "left": int(acc["sc_L"][k]), "right": int(acc["sc_R"][k]), "token_on": int(acc["sc_on"][k])} for k, L in enumerate(ARCS)}}
    rep["CONTROL_R1_train"] = {**{f"{L}_valid_mismatch": acc["mis"][L] for L in (30, 50, 80)},
                               **{f"{L}_xy_maxabs_m": acc["xy"][L] for L in (30, 50, 80)},
                               **{f"{L}_psi_maxabs_deg": acc["psi"][L] for L in (30, 50, 80)},
                               "pass": all(acc["mis"][L] == 0 for L in (30, 50, 80))}
    rep["t_elapsed_s"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: rep[k] for k in ("CONTROL_R1_train", "t_elapsed_s")}, indent=1))


if __name__ == "__main__":
    main()
