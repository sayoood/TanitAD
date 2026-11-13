"""domprem.py -- PREREG_V8F_PROOF_ADD14 (DOMPREM). CPU, zero training, zero rollouts.
Evidence class MEASURED; tier T1-kin (kinematic rollout on the model's own actions, logged frames): a diagnostic, never driving performance.
STRATEGIC family: PAUSED (PI item 25).

Usage: python domprem.py run [--out-dir DIR] [--nboot 2000] [--smoke N]
Runs controls d1..d5 first, then the read on pair 1 and pair 2, seeds 0 and 1, then the card outcomes; writes RESULT_DOMPREM.json and VERDICT_DOMPREM.md.
Exit 0 = every control reads its known value; 2 = a control failed (the read is then not valid; the numbers are still printed).
Definitions and reading choices: DOMPREM_DEFS.md.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
HIER = "F:/TanitAD-scratch/v8proof/hier"
sys.path.insert(0, HIER)
import hierwhere as H  # noqa: E402  (imports the pinned taniteval tree, see HIER_DEFS R1)

np = H.np
H.torch.set_num_threads(2)
CI = H.CI

CARD = "F:/TanitAD-scratch/v8proof/PREREG_V8F_PROOF_ADD14.md"
CARD_SHA = "70b48463a5372b0918c4b341549f5ca7688004f66b5f28737e9abd34765d8c36"
ART = H.ART
PAIRS = {
    "pair1": {"H": ART + "/v8f-dagger-t1/dump_DAG_s{s}_m10_causal_heads", "F": ART + "/v8f-flatdag-t1/dump_FLATDAG_s{s}_m10_causal_heads"},
    "pair2": {"H": ART + "/v8f-dagger-t1/dump_DAG0_s{s}_m10_causal_heads", "F": ART + "/v8f-flatdag-t1/dump_FLATDAG0_s{s}_m10_causal_heads"},
}
KNOWN = {"pair1": HIER + "/RESULT_HIERWHERE.json", "pair2": HIER + "/pair2/read/RESULT_HIERWHERE.json"}
SEEDS = (0, 1)
EDGES = (2.0, 8.0, 15.0)                      # card section 3: fixed v0 edges, m/s
CTX = (("POOLED", "all windows"), ("v0_0_2", "v0 in [0,2)"), ("v0_2_8", "v0 in [2,8)"), ("v0_8_15", "v0 in [8,15)"), ("v0_15_inf", "v0 in [15,inf)"))
MIN_WIN, MIN_EP = 100, 20
TOL = 2.0
TOLS = (2.0, 1.0, 4.0)
CLS = ("all", "crit", "bulk")
KS = H.KEEP_STRAIGHT
NB = 2000
EPS = 1e-9


# ------------------------------------------------------------------------------------------------ best single plan: exact maximum, angular sweep
def best_disc(P, tol, eps=EPS):
    """P [n,2].  Exact maximum number of points one closed disc of radius tol can hold (a point on the boundary counts as inside, tolerance eps).
    Angular sweep, O(n^2 log n): for each point i, the discs of radius tol that have i on the boundary have centres p_i + tol*(cos t, sin t);
    point j is inside iff |t - phi_j| <= acos(d_ij / (2 tol)).  The maximum overlap of those arcs (plus i and its coincident points) is the answer.
    Returns (count, centre)."""
    P = np.asarray(P, np.float64).reshape(-1, 2)
    n = len(P)
    if n == 0:
        return 0, (float("nan"), float("nan"))
    best, ctr = 1, P[0].copy()
    r2 = 2.0 * tol
    twopi = 2.0 * math.pi
    for i in range(n):
        d = P - P[i]
        dist = np.hypot(d[:, 0], d[:, 1])
        near = dist <= r2 + eps
        near[i] = False
        dup = near & (dist <= 1e-12)
        real = near & ~dup
        base = 1 + int(dup.sum())
        k = int(real.sum())
        if base + k <= best:
            continue
        if k == 0:
            best, ctr = base, P[i].copy()
            continue
        phi = np.arctan2(d[real, 1], d[real, 0])
        w = np.arccos(np.clip(dist[real] / r2, -1.0, 1.0)) + eps / (4.0 * tol)   # angular widening: moves a centre by at most 2 tol * eps / (4 tol) = eps / 2
        s = np.mod(phi - w, twopi)
        e = s + 2.0 * w
        wrap = e > twopi
        c0 = int(wrap.sum())
        ang = np.concatenate([s, np.where(wrap, e - twopi, e)])
        typ = np.concatenate([np.zeros(k, np.int64), np.ones(k, np.int64)])          # 0 = start, 1 = end; starts first on ties
        # a wrapped arc is active at angle 0: it ENDS at e - 2pi, and (re)STARTS at s.  Non-wrapped: start s, end e.
        ang[k:] = np.where(wrap, e - twopi, e)
        order = np.lexsort((typ, ang))
        delta = np.where(typ[order] == 0, 1, -1)
        run = c0 + np.cumsum(delta)
        run = np.where(typ[order] == 0, run, -1)                                      # candidate maxima only right after a start
        j = int(np.argmax(run))
        m = max(int(run[j]), c0)
        if base + m > best:
            best = base + m
            sa = ang[order]
            th = (0.5 * (sa[j] + sa[j + 1]) if j + 1 < len(sa) else sa[j]) if int(run[j]) >= c0 else 0.0   # middle of the maximal overlap (robust to the eps widening)
            ctr = P[i] + tol * np.array([math.cos(th), math.sin(th)])
    return int(best), (float(ctr[0]), float(ctr[1]))


# ------------------------------------------------------------------------------------------------ contexts
def ctx_index(v0):
    """0..3 for [0,2), [2,8), [8,15), [15,inf); a value equal to an edge goes to the UPPER bin (np.digitize, right=False)."""
    return np.digitize(np.asarray(v0, np.float64), EDGES)


def ctx_masks(v0):
    idx = ctx_index(v0)
    return [np.ones(len(idx), bool)] + [idx == k for k in range(4)]


# ------------------------------------------------------------------------------------------------ statistics (pure functions of an index selection)
def maxshare(cell, sub, K):
    return float(np.bincount(cell[sub], minlength=K).max() / len(sub)) if len(sub) else float("nan")


def mean_at(v, sub):
    return float(v[sub].mean()) if len(sub) else float("nan")


def margin(right, cell, sub, K):
    """m = acc - p_max, with p_max the share of the largest logged cell of the SAME selection."""
    return mean_at(right, sub) - maxshare(cell, sub, K) if len(sub) else float("nan")


def share_hit_among_right(right, hit, sub):
    den = right[sub].sum()
    return float((right[sub] * hit[sub]).sum() / den) if den > 0 else float("nan")


def separation(g6, cell, sub, tol):
    """(D, rho, S) for the two largest logged cells (ties: lower cell index) of the selection."""
    if len(sub) == 0:
        return float("nan"), float("nan"), float("nan")
    cnt = np.bincount(cell[sub], minlength=12)
    order = np.argsort(-cnt, kind="stable")
    a, b = int(order[0]), int(order[1])
    if cnt[b] == 0:
        return float("nan"), float("nan"), float("nan")
    pa, pb = g6[sub][cell[sub] == a], g6[sub][cell[sub] == b]
    ma, mb = pa.mean(0), pb.mean(0)
    D = float(np.linalg.norm(ma - mb))
    rho = float(max(np.percentile(np.linalg.norm(pa - ma, axis=1), 90), np.percentile(np.linalg.norm(pb - mb, axis=1), 90)))
    return D, rho, D / (2 * tol + 2 * rho)


class Memo:
    """Per-draw memo: boot passes the SAME selection object to every statistic of a draw."""

    def __init__(self):
        self.sel, self.d = None, {}

    def get(self, sel, key, fn):
        if self.sel is not sel:
            self.sel, self.d = sel, {}
        if key not in self.d:
            self.d[key] = fn()
        return self.d[key]


MEMO = Memo()


def sub_of(sel, cid, mask):
    return MEMO.get(sel, ("s", cid), lambda: sel[mask[sel]])


def subc_of(sel, cid, mask, cname, carr):
    if cname == "all":
        return sub_of(sel, cid, mask)
    return MEMO.get(sel, ("c", cid, cname), lambda: (lambda s: s[carr[s]])(sub_of(sel, cid, mask)))


def nums_of(eid, m):
    return int(m.sum()), int(len(np.unique(np.asarray(eid)[m])))


# ------------------------------------------------------------------------------------------------ one arm's per-window features
def arm_feats(cl, g):
    lon, lat = H.modes(cl)
    fde = np.linalg.norm(np.asarray(cl, np.float64)[:, 29] - np.asarray(g, np.float64)[:, 29], axis=-1)
    return {"lon": lon, "lat": lat, "cell": H.joint(lon, lat), "fde": fde}


def load_pair(pair, s, smoke=0):
    a, b = H.load_arm(PAIRS[pair]["H"].format(s=s), smoke), H.load_arm(PAIRS[pair]["F"].format(s=s), smoke)
    for k in ("g", "ws", "eid", "clip_index", "v0"):
        assert np.array_equal(a[k], b[k]), f"{pair} s{s}: arms differ in {k}"
    return a, b


def world(a, b):
    """everything the read needs for one (pair, seed)."""
    g = np.asarray(a["g"], np.float64)
    lg, tg = H.modes(g)
    W = {"g": g, "g6": g[:, 29], "v0": np.asarray(a["v0"], np.float64), "eid": a["eid"], "lg": lg, "tg": tg, "jg": H.joint(lg, tg),
         "H": arm_feats(a["cl"], g), "F": arm_feats(b["cl"], g)}
    W["crit"] = W["jg"] != KS
    W["cls"] = {"all": np.ones(len(g), bool), "crit": W["crit"], "bulk": ~W["crit"]}
    W["masks"] = ctx_masks(W["v0"])
    return W


def rights(W, arm):
    A = W[arm]
    return {"joint": (A["cell"] == W["jg"]).astype(np.float64), "lon": (A["lon"] == W["lg"]).astype(np.float64), "lat": (A["lat"] == W["tg"]).astype(np.float64)}


# ------------------------------------------------------------------------------------------------ the read for one (pair, seed)
def read_world(W, nb):
    eid = W["eid"]
    R = {"H": rights(W, "H"), "F": rights(W, "F")}
    jg, lg, tg, g6 = W["jg"], W["lg"], W["tg"], W["g6"]
    hits = {arm: {tol: (W[arm]["fde"] <= tol).astype(np.float64) for tol in TOLS} for arm in ("H", "F")}
    stats, meta, plans = {}, {}, {}
    for cid, (cname, clabel) in enumerate(CTX):
        mask = W["masks"][cid]
        n, nep = nums_of(eid, mask)
        small = bool(n < MIN_WIN or nep < MIN_EP)
        meta[cname] = {"label": clabel, "n_windows": n, "n_episodes": nep, "TOO-SMALL": small}
        if small:
            continue

        def mk(fn, cls="all", cid=cid, mask=mask):
            return lambda sel: fn(subc_of(sel, cid, mask, cls, W["cls"][cls]))
        P = lambda nm: f"{cname}|{nm}"          # noqa: E731
        stats[P("pmax")] = mk(lambda s: maxshare(jg, s, 12))
        stats[P("pmax_lon")] = mk(lambda s: maxshare(lg, s, 4))
        stats[P("pmax_lat")] = mk(lambda s: maxshare(tg, s, 3))
        for arm in ("H", "F"):
            r = R[arm]
            stats[P(arm + "|acc")] = mk(lambda s, r=r: mean_at(r["joint"], s))
            stats[P(arm + "|acc_lon")] = mk(lambda s, r=r: mean_at(r["lon"], s))
            stats[P(arm + "|acc_lat")] = mk(lambda s, r=r: mean_at(r["lat"], s))
            stats[P(arm + "|m")] = mk(lambda s, r=r: margin(r["joint"], jg, s, 12))
            stats[P(arm + "|m_lon")] = mk(lambda s, r=r: margin(r["lon"], lg, s, 4))
            stats[P(arm + "|m_lat")] = mk(lambda s, r=r: margin(r["lat"], tg, s, 3))
            stats[P(arm + "|e")] = mk(lambda s, r=r, arm=arm: share_hit_among_right(r["joint"], hits[arm][TOL], s))
        stats[P("acc_diff")] = mk(lambda s: mean_at(R["H"]["joint"], s) - mean_at(R["F"]["joint"], s))
        sep = {}
        for j, nm in enumerate(("D", "rho", "S")):
            stats[P(nm)] = mk(lambda s, j=j: separation(g6, jg, s, TOL)[j])
        for tol in TOLS:
            cnt, ctr = best_disc(g6[mask], tol)
            inside = (np.linalg.norm(g6 - np.asarray(ctr), axis=1) <= tol + EPS).astype(np.float64)
            assert int(inside[mask].sum()) == cnt, (cname, tol, cnt, int(inside[mask].sum()))
            plans[(cname, tol)] = {"count": cnt, "share": cnt / n, "centre": list(ctr), "n": n}
            for cls in CLS:
                stats[P(f"hit{tol}|{cls}|H")] = mk(lambda s, tol=tol: mean_at(hits["H"][tol], s), cls)
                stats[P(f"hit{tol}|{cls}|F")] = mk(lambda s, tol=tol: mean_at(hits["F"][tol], s), cls)
                stats[P(f"hit{tol}|{cls}|diff")] = mk(lambda s, tol=tol: mean_at(hits["H"][tol], s) - mean_at(hits["F"][tol], s), cls)
                stats[P(f"hit{tol}|{cls}|plan")] = mk(lambda s, inside=inside: mean_at(inside, s), cls)
                stats[P(f"hit{tol}|{cls}|H_minus_plan")] = mk(lambda s, tol=tol, inside=inside: mean_at(hits["H"][tol], s) - mean_at(inside, s), cls)
    t0 = time.time()
    res = H.boot(eid, stats, nb)
    print(f"   boot: {len(stats)} statistics x {nb} draws in {time.time() - t0:.0f}s", flush=True)
    row = lambda k: {kk: res[k][kk] for kk in ("point", "lo", "hi")}   # noqa: E731
    out = {}
    for cid, (cname, clabel) in enumerate(CTX):
        mask = W["masks"][cid]
        e = dict(meta[cname])
        if e["TOO-SMALL"]:
            out[cname] = e
            continue
        P = lambda nm: f"{cname}|{nm}"  # noqa: E731
        e["pmax"], e["pmax_lon"], e["pmax_lat"] = row(P("pmax")), row(P("pmax_lon")), row(P("pmax_lat"))
        e["acc_diff"] = row(P("acc_diff"))
        e["separation"] = {nm: row(P(nm)) for nm in ("D", "rho", "S")}
        e["separation"]["SEPARATED"] = bool(e["separation"]["S"]["point"] > 1.0)
        for arm in ("H", "F"):
            ar = {nm: row(P(arm + "|" + nm)) for nm in ("acc", "acc_lon", "acc_lat", "m", "m_lon", "m_lat", "e")}
            ok = (R[arm]["joint"] > 0) & mask
            ar["r90_fde6_m"] = float(np.percentile(W[arm]["fde"][ok], 90)) if ok.any() else float("nan")
            ar["n_right"] = int(ok.sum())
            ar["right_and_hit_tol2"] = int((ok & (hits[arm][TOL] > 0)).sum())
            e[arm] = ar
        e["hit"] = {}
        for tol in TOLS:
            e["hit"][f"tol{tol}"] = {"best_plan_fit_on_context": plans[(cname, tol)]}
            for cls in CLS:
                cm = mask & W["cls"][cls]
                nw, ne = nums_of(eid, cm)
                e["hit"][f"tol{tol}"][cls] = {"n_windows": nw, "n_episodes": ne, **{k: row(P(f"hit{tol}|{cls}|{k}")) for k in ("H", "F", "diff", "plan", "H_minus_plan")}}
        out[cname] = e
    return out


# ------------------------------------------------------------------------------------------------ controls
def point_table(W, arm_cell, arm_fde):
    """per context: (n, acc, hit rate at each tol, lon acc, lat acc) of an arbitrary arm, by the same masks and helpers as the read."""
    right = (arm_cell == W["jg"]).astype(np.float64)
    out = {}
    for cid, (cname, _) in enumerate(CTX):
        m = W["masks"][cid]
        sub = np.flatnonzero(m)
        out[cname] = {"n": int(len(sub)), "acc": mean_at(right, sub), "hit": {f"tol{t}": mean_at((arm_fde <= t).astype(np.float64), sub) for t in TOLS}}
    return out


def synth_d4():
    rng = np.random.default_rng(0)

    def cluster(n, c, r):
        ang = rng.uniform(0, 2 * math.pi, n)
        rad = r * np.sqrt(rng.uniform(0, 1, n))
        return np.stack([c[0] + rad * np.cos(ang), c[1] + rad * np.sin(ang)], 1)
    two = np.concatenate([cluster(60, (0.0, 0.0), 1.0), cluster(40, (10.0, 0.0), 1.0)])
    one = cluster(100, (3.0, -2.0), 1.5)
    return two, one


def run_controls(data, nb, smoke):
    C = []

    def add(cid, what, observed, known, ok):
        C.append({"id": cid, "what": what, "observed": observed, "known": known, "PASS": bool(ok)})
        print(f"  {cid:12s} {'PASS' if ok else 'FAIL'}  {what}: observed {observed}  known {known}", flush=True)
    # ---- d1
    for pair in ("pair1", "pair2"):
        kn = json.load(open(KNOWN[pair], encoding="utf-8"))
        for s in SEEDS:
            W = data[(pair, s)]["W"]
            n = len(W["jg"])
            ks = kn["modes"][f"s{s}"]
            cnt = {f"{H.LON[i]}/{H.LAT[j]}": int(((W["lg"] == i) & (W["tg"] == j)).sum()) for i in range(4) for j in range(3)}
            ok = (not smoke) and cnt == ks["logged_joint_counts"] and int(W["crit"].sum()) == ks["n_SELECTION_CRITICAL"] == 1305 and int((~W["crit"]).sum()) == ks["n_BULK"] == 1094
            add(f"d1_{pair}_s{s}_counts", "class counts SELECTION-CRITICAL / BULK and 12 logged joint counts", [int(W["crit"].sum()), int((~W["crit"]).sum())], [ks["n_SELECTION_CRITICAL"], ks["n_BULK"]] + ["12 joint counts equal" if cnt == ks["logged_joint_counts"] else "joint counts DIFFER"], ok)
            lon_ok = {H.LON[i]: int((W["lg"] == i).sum()) for i in range(4)} == ks["logged_lon_counts"] and {H.LAT[j]: int((W["tg"] == j).sum()) for j in range(3)} == ks["logged_lat_counts"]
            add(f"d1_{pair}_s{s}_modes", "logged longitudinal and lateral class counts", "equal" if lon_ok else "DIFFER", "equal", lon_ok and not smoke)
            for arm, key in (("H", "DAG"), ("F", "FLAT-DAG")):
                kp = kn["seeds"][f"s{s}"]["R2"]["joint"][key]["point"]            # error rate of the banked read, 6 dp
                kcount = int(round((1 - kp) * n))
                mine = int((W[arm]["cell"] == W["jg"]).sum())
                ok = (not smoke) and mine == kcount and abs((1 - kp) * n - kcount) < 0.01
                add(f"d1_{pair}_s{s}_{arm}", f"arm {arm} RIGHT windows (share {mine / n:.6f})", mine, f"{kcount} (banked error rate {kp})", ok)
    # ---- d2, d3, d5 on every (pair, seed)
    for (pair, s), D in data.items():
        W = D["W"]
        n = len(W["jg"])
        g = W["g"]
        # d2
        t = point_table(W, W["jg"], np.linalg.norm(g[:, 29] - g[:, 29], axis=1))
        bad = [(c, v) for c, v in t.items() if v["n"] > 0 and not (v["acc"] == 1.0 and all(h == 1.0 for h in v["hit"].values()))]
        lon_lat = bool(np.all(H.modes(g)[0] == W["lg"]) and np.all(H.modes(g)[1] == W["tg"]))
        add(f"d2_{pair}_s{s}", "cl := g, acc and hit (tol 2, 1, 4) in every context", "all 1.0" if not bad else bad, "all 1.0", (not bad) and lon_lat)
        # d3a: constant (KEEP, STRAIGHT) arm
        k = np.arange(1, 31)[None, :, None] * 0.2 * 5.0
        const = np.concatenate([np.repeat(k, n, 0), np.zeros((n, 30, 1))], axis=2)
        cl_, ct_ = H.modes(const)
        assert np.all(H.joint(cl_, ct_) == KS), "constant arm is not (KEEP, STRAIGHT)"
        t = point_table(W, H.joint(cl_, ct_), np.linalg.norm(const[:, 29] - g[:, 29], axis=1))
        diffs = {}
        for cid, (cname, _) in enumerate(CTX):
            m = W["masks"][cid]
            if m.sum():
                diffs[cname] = abs(t[cname]["acc"] - float((W["jg"][m] == KS).sum()) / float(m.sum()))
        add(f"d3a_{pair}_s{s}", "constant (KEEP, STRAIGHT) arm: acc minus share of that cell, max over contexts", max(diffs.values()), 0.0, max(diffs.values()) == 0.0)
        # d3b: permuted cells inside each context (rng seed 0) against the interval of sum of squared cell shares (bootstrap, same draws for all)
        stats, pts = {}, {}
        for cid, (cname, _) in enumerate(CTX):
            m = W["masks"][cid]
            nn, ne = nums_of(W["eid"], m)
            if nn < MIN_WIN or ne < MIN_EP:
                continue
            idx = np.flatnonzero(m)
            perm = np.random.default_rng(0).permutation(len(idx))
            right = np.zeros(n)
            right[idx] = (W["jg"][idx][perm] == W["jg"][idx]).astype(np.float64)
            pts[cname] = float(right[idx].mean())
            stats[cname] = (lambda sel, cid=cid, m=m: (lambda sub: float(((np.bincount(W["jg"][sub], minlength=12) / len(sub)) ** 2).sum()) if len(sub) else float("nan"))(sub_of(sel, cid, m)))
        rr = H.boot(W["eid"], stats, nb)
        inside = {c: (rr[c]["lo"] <= pts[c] <= rr[c]["hi"]) for c in pts}
        add(f"d3b_{pair}_s{s}", "permuted-cells arm acc inside the interval of the sum of squared shares, per context",
            {c: [round(pts[c], 4), round(rr[c]["lo"], 4), round(rr[c]["hi"], 4)] for c in pts}, "acc in [lo, hi]", all(inside.values()))
        # d5: logged future shifted by 3.0 m across the track (ego-frame lateral axis) at every knot
        sh = g.copy()
        sh[..., 1] += 3.0
        t = point_table(W, H.joint(*H.modes(sh)), np.linalg.norm(sh[:, 29] - g[:, 29], axis=1))
        h2 = {c: v["hit"]["tol2.0"] for c, v in t.items() if v["n"] > 0}
        add(f"d5_{pair}_s{s}", "g shifted 3.0 m across the track: hit rate at tol 2.0 in every context", h2, "0.0 everywhere", all(v == 0.0 for v in h2.values()))
    # ---- d4 (once)
    two, one = synth_d4()
    c2, _ = best_disc(two, TOL)
    c1, _ = best_disc(one, TOL)
    add("d4", "best single plan on two clusters (60, 40; separated) and on one cluster of radius 1.5 < tol (100 points)", [c2, c1], [60, 100], c2 == 60 and c1 == 100)
    return C


# ------------------------------------------------------------------------------------------------ outcomes (code, from the numbers)
def label_premise(r):
    if r["point"] > 0 and r["lo"] > 0:
        return "PREMISE-HOLDS"
    if r["point"] < 0 and r["hi"] < 0:
        return "PREMISE-FAILS"
    return "UNDECIDED"


def outcomes(R):
    """R[pair][s][ctx] -> labels and card outcomes."""
    O = {}
    for pair in ("pair1", "pair2"):
        lab = {}
        for s in SEEDS:
            for cname, _ in CTX:
                e = R[pair][f"s{s}"][cname]
                lab[(s, cname)] = "TOO-SMALL" if e["TOO-SMALL"] else label_premise(e["H"]["m"])
        pooled = all(lab[(s, "POOLED")] == "PREMISE-HOLDS" for s in SEEDS)
        v0_hold = [cn for cn, _ in CTX[1:] if all(lab[(s, cn)] == "PREMISE-HOLDS" for s in SEEDS)]
        q1 = "DOM-PREMISE-MET" if pooled else ("DOM-PREMISE-PARTIAL" if v0_hold else "DOM-PREMISE-NOT-MET")
        q2 = {}
        for cname, _ in CTX:
            es = [R[pair][f"s{s}"][cname] for s in SEEDS]
            if es[0]["TOO-SMALL"]:
                q2[cname] = {"label": "TOO-SMALL"}
                continue
            q2[cname] = {"S": [e["separation"]["S"]["point"] for e in es], "sep": ["SEPARATED" if e["separation"]["SEPARATED"] else "NOT-SEPARATED" for e in es],
                         "r90_H": [e["H"]["r90_fde6_m"] for e in es], "exec": "EXEC-OK" if all(e["H"]["r90_fde6_m"] <= TOL for e in es) else "EXEC-WIDE"}
        d = [R[pair][f"s{s}"]["POOLED"]["hit"]["tol2.0"]["crit"]["diff"] for s in SEEDS]
        q3 = "HIT-HIER" if all(x["point"] > 0 and x["lo"] > 0 for x in d) else ("HIT-FLAT" if all(x["point"] < 0 and x["hi"] < 0 for x in d) else "HIT-UNDECIDED")
        O[pair] = {"premise_labels": {f"s{s}|{cn}": lab[(s, cn)] for (s, cn) in lab}, "Q1": q1, "Q1_v0_contexts_holding_at_both_seeds": v0_hold, "Q2": q2, "Q3": q3,
                   "decides": pair == "pair1"}
    return O


def follows(R, O):
    """the paragraph committed by the card for the pair-1 Q1 outcome, from the numbers."""
    q1 = O["pair1"]["Q1"]
    rows = []
    for s in SEEDS:
        for cname, _ in CTX:
            e = R["pair1"][f"s{s}"][cname]
            if e["TOO-SMALL"]:
                continue
            lack = e["pmax"]["point"] - e["H"]["acc"]["point"]
            dl = e["pmax_lon"]["point"] - e["H"]["acc_lon"]["point"]
            dt = e["pmax_lat"]["point"] - e["H"]["acc_lat"]["point"]
            rows.append({"seed": s, "ctx": cname, "lack_joint": lack, "lack_lon": dl, "lack_lat": dt, "carried_by": "longitudinal" if dl > dt else "lateral"})
    bound = []
    for s in SEEDS:
        for cname, _ in CTX:
            e = R["pair1"][f"s{s}"][cname]
            if e["TOO-SMALL"]:
                continue
            n = e["n_windows"]
            bound.append({"seed": s, "ctx": cname, "bound_right_and_hit": e["H"]["right_and_hit_tol2"] / n, "hit_H_tol2": e["hit"]["tol2.0"]["all"]["H"]["point"],
                          "hit_F_tol2": e["hit"]["tol2.0"]["all"]["F"]["point"], "best_plan": e["hit"]["tol2.0"]["best_plan_fit_on_context"]["share"]})
    return {"outcome": q1, "lack_rows": rows, "bound_rows": bound}


# ------------------------------------------------------------------------------------------------ verdict
def f3(x):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.3f}" if False else f"{x:.3f}"


def fr(r, signed=False):
    p, lo, hi = r["point"], r["lo"], r["hi"]
    if signed:
        return f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def write_verdict(res, path):
    L = []
    w = L.append
    R, O, F = res["results"], res["outcomes"], res["follows"]
    cok = res["controls_all_pass"]
    w("# VERDICT_DOMPREM (PREREG_V8F_PROOF_ADD14, does v8F-mini meet the premise of the dominance theorem?)")
    w("")
    w("Evidence class MEASURED. Tier T1-kin (kinematic rollout on the model's own actions, logged frames), zero training, CPU. A diagnostic, never driving performance.")
    w("STRATEGIC family: PAUSED (PI item 25).")
    w(f"Estimator: `taniteval.ci` episode-cluster bootstrap, B {res['nboot']}, seed 0, same draws for every statistic of a call; p_max recomputed in every draw. One training run per arm per seed: a separated interval is necessary and not sufficient.")
    w(f"Card sha256 {res['card_sha256']}; domprem.py sha256 {res['domprem_sha256']}.")
    w("")
    if not cok:
        w("**READ NOT VALID: at least one control did not read its known value (see the controls block). The numbers below are printed as computed and decide nothing.**")
        w("")
    w("## Controls (run first)")
    w("")
    w("| control | what | observed | known | result |")
    w("|---|---|---|---|---|")
    for c in res["controls"]:
        w(f"| {c['id']} | {c['what']} | {json.dumps(c['observed'])} | {json.dumps(c['known'])} | {'PASS' if c['PASS'] else 'FAIL'} |")
    w("")
    w(f"Controls overall: **{'ALL PASS' if cok else 'FAILED'}**")
    w("")
    w("## Card outcomes (pair 1 decides; pair 2 is printed with the same labels and decides nothing)")
    w("")
    w(f"* Q1 (pair 1): **{O['pair1']['Q1']}**  (v0 contexts with PREMISE-HOLDS at both seeds: {O['pair1']['Q1_v0_contexts_holding_at_both_seeds'] or 'none'})")
    w(f"* Q3 (pair 1, SELECTION-CRITICAL windows, pooled contexts, tol 2.0 m): **{O['pair1']['Q3']}**")
    w(f"* Pair 2, same labels (decides nothing): Q1 {O['pair2']['Q1']}; Q3 {O['pair2']['Q3']}")
    w("")
    for pair in ("pair1", "pair2"):
        title = "Pair 1 (on-policy data in both arms)" if pair == "pair1" else "Pair 2 (no on-policy data)"
        w(f"## {title}")
        for s in SEEDS:
            w(f"")
            w(f"### Q1 premise, selection: {pair}, seed {s}")
            w("")
            w("| context | n win / ep | p_max | acc HIER [95 %] | m HIER [95 %] | label | acc FLAT [95 %] | m FLAT [95 %] | acc HIER - FLAT [95 %] |")
            w("|---|---|---|---|---|---|---|---|---|")
            for cname, clabel in CTX:
                e = R[pair][f"s{s}"][cname]
                if e["TOO-SMALL"]:
                    w(f"| {clabel} | {e['n_windows']} / {e['n_episodes']} | TOO-SMALL | | | TOO-SMALL | | | |")
                    continue
                w(f"| {clabel} | {e['n_windows']} / {e['n_episodes']} | {fr(e['pmax'])} | {fr(e['H']['acc'])} | {fr(e['H']['m'], True)} | {O[pair]['premise_labels'][f's{s}|{cname}']} | {fr(e['F']['acc'])} | {fr(e['F']['m'], True)} | {fr(e['acc_diff'], True)} |")
        w("")
        w(f"### Four families: {pair} (mode accuracy and margin over the largest class of the same selection, per seed)")
        w("")
        w("| seed | context | LONGITUDINAL acc HIER [95 %] | LON margin HIER [95 %] | LON margin FLAT [95 %] | LATERAL acc HIER [95 %] | LAT margin HIER [95 %] | LAT margin FLAT [95 %] | TACTICAL (joint) m HIER [95 %] |")
        w("|---|---|---|---|---|---|---|---|---|")
        for s in SEEDS:
            for cname, clabel in CTX:
                e = R[pair][f"s{s}"][cname]
                if e["TOO-SMALL"]:
                    w(f"| {s} | {clabel} | TOO-SMALL | | | | | | |")
                    continue
                w(f"| {s} | {clabel} | {fr(e['H']['acc_lon'])} | {fr(e['H']['m_lon'], True)} | {fr(e['F']['m_lon'], True)} | {fr(e['H']['acc_lat'])} | {fr(e['H']['m_lat'], True)} | {fr(e['F']['m_lat'], True)} | {fr(e['H']['m'], True)} |")
        w("")
        w("STRATEGIC: PAUSED (PI item 25), not a pass and not omitted.")
        w("")
        w(f"### Q2 premise, geometry: {pair} (printed, no card outcome)")
        w("")
        w("| seed | context | D (m) [95 %] | rho (m) [95 %] | S = D / (2 tol + 2 rho) [95 %] | separation | e HIER (hits among RIGHT) [95 %] | r90 HIER (m) | n RIGHT |")
        w("|---|---|---|---|---|---|---|---|---|")
        for s in SEEDS:
            for cname, clabel in CTX:
                e = R[pair][f"s{s}"][cname]
                if e["TOO-SMALL"]:
                    w(f"| {s} | {clabel} | TOO-SMALL | | | | | | |")
                    continue
                sp = e["separation"]
                w(f"| {s} | {clabel} | {fr(sp['D'])} | {fr(sp['rho'])} | {fr(sp['S'])} | {'SEPARATED' if sp['SEPARATED'] else 'NOT-SEPARATED'} | {fr(e['H']['e'])} | {e['H']['r90_fde6_m']:.3f} | {e['H']['n_right']} |")
        w("")
        w("| context | EXEC (r90 HIER <= tol 2.0 at both seeds) |")
        w("|---|---|")
        for cname, clabel in CTX:
            q = O[pair]["Q2"][cname]
            w(f"| {clabel} | {q.get('exec', q.get('label'))} |")
        w("")
        w(f"### Q3 hit rates at tol 2.0 m (primary): {pair}")
        w("")
        w("| seed | context | windows | n win / ep | hit HIER [95 %] | hit FLAT [95 %] | HIER - FLAT [95 %] | best single plan (context-fitted disc) [95 %] | HIER - best plan [95 %] |")
        w("|---|---|---|---|---|---|---|---|---|")
        for s in SEEDS:
            for cname, clabel in CTX:
                e = R[pair][f"s{s}"][cname]
                if e["TOO-SMALL"]:
                    w(f"| {s} | {clabel} | | TOO-SMALL | | | | | |")
                    continue
                for cls in CLS:
                    h = e["hit"]["tol2.0"][cls]
                    w(f"| {s} | {clabel} | {cls} | {h['n_windows']} / {h['n_episodes']} | {fr(h['H'])} | {fr(h['F'])} | {fr(h['diff'], True)} | {fr(h['plan'])} | {fr(h['H_minus_plan'], True)} |")
        w("")
        w("Exact best single plan share (maximum logged 6 s endpoints in one disc of radius tol, fitted on all windows of the context; upper bound for a context-constant plan): ")
        w("")
        w("| seed | context | tol 2.0 | tol 1.0 | tol 4.0 |")
        w("|---|---|---|---|---|")
        for s in SEEDS:
            for cname, clabel in CTX:
                e = R[pair][f"s{s}"][cname]
                if e["TOO-SMALL"]:
                    continue
                w(f"| {s} | {clabel} | " + " | ".join(f"{e['hit'][f'tol{t}']['best_plan_fit_on_context']['count']} of {e['n_windows']} ({e['hit'][f'tol{t}']['best_plan_fit_on_context']['share']:.3f})" for t in TOLS) + " |")
        w("")
        w(f"### Sensitivity, tol 1.0 m and 4.0 m (decide nothing): {pair}, hit rate HIER / FLAT / HIER - FLAT [95 %]")
        w("")
        w("| seed | context | windows | tol 1.0 HIER | tol 1.0 FLAT | tol 1.0 diff [95 %] | tol 4.0 HIER | tol 4.0 FLAT | tol 4.0 diff [95 %] |")
        w("|---|---|---|---|---|---|---|---|---|")
        for s in SEEDS:
            for cname, clabel in CTX:
                e = R[pair][f"s{s}"][cname]
                if e["TOO-SMALL"]:
                    continue
                for cls in (CLS if cname == "POOLED" else ("all",)):
                    a, b = e["hit"]["tol1.0"][cls], e["hit"]["tol4.0"][cls]
                    w(f"| {s} | {clabel} | {cls} | {a['H']['point']:.3f} | {a['F']['point']:.3f} | {fr(a['diff'], True)} | {b['H']['point']:.3f} | {b['F']['point']:.3f} | {fr(b['diff'], True)} |")
        w("")
    w("## What follows (committed in the card for the outcome reached)")
    w("")
    if F["outcome"] == "DOM-PREMISE-NOT-MET":
        w("Outcome DOM-PREMISE-NOT-MET: the theorem does not apply to v8F-mini today. Per context, the accuracy the hierarchy lacks (p_max minus acc_HIER, point values, pair 1) and which of the two terms carries it. The longitudinal and lateral columns are p_max of that mode minus the mode accuracy of the hierarchy; the term with the larger lack is named as carrying it. The next lever is the accuracy of the tactical decision on that term, not a gate and not the operative level.")
    elif F["outcome"] in ("DOM-PREMISE-MET", "DOM-PREMISE-PARTIAL"):
        w(f"Outcome {F['outcome']}: the theorem applies in the contexts where PREMISE-HOLDS at both seeds. Its bound (the decided plan hits at least the RIGHT windows that execute within tol) is compared below with the measured hit rates of Q3. The next step is the same read on the six seeds (secondary read, not run now). The per-context lack table is also printed.")
    w("")
    w("| seed | context | p_max - acc HIER | LON lack (p_max_lon - acc_lon) | LAT lack (p_max_lat - acc_lat) | carried by |")
    w("|---|---|---|---|---|---|")
    clab = dict(CTX)
    for r in F["lack_rows"]:
        w(f"| {r['seed']} | {clab[r['ctx']]} | {r['lack_joint']:+.3f} | {r['lack_lon']:+.3f} | {r['lack_lat']:+.3f} | {r['carried_by']} |")
    w("")
    w("Bound against measured hit rates, tol 2.0 m, all windows of the context, pair 1:")
    w("")
    w("| seed | context | bound: share of windows RIGHT and within tol | measured hit HIER | measured hit FLAT | best single plan |")
    w("|---|---|---|---|---|---|")
    for r in F["bound_rows"]:
        w(f"| {r['seed']} | {clab[r['ctx']]} | {r['bound_right_and_hit']:.3f} | {r['hit_H_tol2']:.3f} | {r['hit_F_tol2']:.3f} | {r['best_plan']:.3f} |")
    w("")
    w("## Limits (card section 6, copied)")
    w("")
    w("* Contexts are speed bins, a coarse stand-in for what a flat plan conditions on; the theorem compares with plans constant on a context, which the flat arm is not. Q3 therefore reads the flat arm directly as well.")
    w("* RIGHT is defined on the executed rollout, so it folds the decision and its execution into one indicator; the tactical head's own top-1 is not read here.")
    w("* Hit-or-miss at one endpoint is one acceptance region among many; ADE stays the registered primary metric of the programme's comparison (addendum 11).")
    w("* Added: one training run per arm per seed, so a separated interval is necessary and not sufficient; the replicate read on the six seeds of HIERREP is the secondary read and is not run here.")
    w("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--nboot", type=int, default=NB)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    sha = hashlib.sha256(open(CARD, "rb").read()).hexdigest()
    assert sha == CARD_SHA, f"card hash differs: {sha}"
    me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    print("card sha256 ok", sha, "domprem.py sha256", me, flush=True)
    data = {}
    for pair in PAIRS:
        for s in SEEDS:
            h, f = load_pair(pair, s, a.smoke)
            data[(pair, s)] = {"W": world(h, f)}
            print(f"loaded {pair} s{s}: {len(h['eid'])} windows, {len(np.unique(h['eid']))} episodes", flush=True)
    print("CONTROLS", flush=True)
    controls = run_controls(data, a.nboot, a.smoke)
    cok = all(c["PASS"] for c in controls)
    print(f"controls: {'ALL PASS' if cok else 'FAILED'} ({time.time() - t0:.0f}s)", flush=True)
    R = {}
    for pair in PAIRS:
        R[pair] = {}
        for s in SEEDS:
            print(f"READ {pair} s{s}", flush=True)
            R[pair][f"s{s}"] = read_world(data[(pair, s)]["W"], a.nboot)
            for cname, _ in CTX:
                e = R[pair][f"s{s}"][cname]
                if e["TOO-SMALL"]:
                    print(f"   {cname}: TOO-SMALL n={e['n_windows']} ep={e['n_episodes']}")
                else:
                    print(f"   {cname}: n={e['n_windows']} p_max={e['pmax']['point']:.4f} accH={e['H']['acc']['point']:.4f} mH={e['H']['m']['point']:+.4f} [{e['H']['m']['lo']:+.4f},{e['H']['m']['hi']:+.4f}] "
                          f"accF={e['F']['acc']['point']:.4f} hit2 H/F={e['hit']['tol2.0']['all']['H']['point']:.4f}/{e['hit']['tol2.0']['all']['F']['point']:.4f}", flush=True)
    O = outcomes(R)
    Fw = follows(R, O)
    print("OUTCOMES", json.dumps({p: {"Q1": O[p]["Q1"], "Q3": O[p]["Q3"]} for p in O}), flush=True)
    res = {"card": CARD, "card_sha256": sha, "domprem_sha256": me, "evidence": "MEASURED; tier T1-kin (kinematic rollout on the model's own actions, logged frames); zero training; never driving performance",
           "STRATEGIC": "PAUSED (PI item 25)", "nboot": a.nboot, "bootstrap_seed": H.BSEED, "smoke": a.smoke, "tolerances": {"primary_tol_m": TOL, "sensitivity_tol_m": [1.0, 4.0]},
           "v0_edges_m_s": list(EDGES), "min_windows": MIN_WIN, "min_episodes": MIN_EP,
           "dump_folders": {pair: {s: {arm: PAIRS[pair][arm].format(s=s) for arm in ("H", "F")} for s in SEEDS} for pair in PAIRS},
           "n_windows_episodes": {f"{p}|s{s}": [int(len(data[(p, s)]["W"]["jg"])), int(len(np.unique(data[(p, s)]["W"]["eid"])))] for p in PAIRS for s in SEEDS},
           "controls": controls, "controls_all_pass": cok, "results": R, "outcomes": {p: {k: (v if k != "Q2" else v) for k, v in O[p].items()} for p in O}, "follows": Fw,
           "exit_code": 0 if cok else 2}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(a.out_dir + "/RESULT_DOMPREM.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    write_verdict(res, a.out_dir + "/VERDICT_DOMPREM.md")
    print(f"done {time.time() - t0:.0f}s", flush=True)
    return 0 if cok else 2


if __name__ == "__main__":
    sys.exit(main())
