"""controls5.py -- CONTROLS_V5: the controls of chapter 8 (DAgger) of the evidence video v5 (+ the title / closing cards).   CPU only, ONE process.

  python controls5.py --main     render every chapter-8 scene (+ closing + title card) in THIS process with the figure artists captured, and check (writes CONTROLS_V5_main.json)
  python controls5.py --swap     DELIBERATE REGRESSION: V5_SWAP_ARMS=WORST swaps the DAG and DAG-0 arrays of ONE scene at the data-access layer; the SAME track / number checks must FAIL
  python controls5.py --post     after the full render: video integrity, text and array equality with the control render, chapters 1-7 unchanged vs v4 (writes CONTROLS_V5_post.json, merges)
  python controls5.py --merge    CONTROLS_V5.json = main + swap (+ post if present)

  T1  every plotted line of the chapter-8 window frames (GT, executed path, S checkpoints, S GT, both arms; the artists of the figure, read back) = the raw arrays from an INDEPENDENT loader
      (own np.load by episode file / own (window, tick) row index), <= 1e-6 m
  A1  every per-window number on screen (ADE6, delta, path length, T decisions and labels, T right counts, S error) = recomputed from the raw arrays, string-equal and in the frame's drawn text
  L1  the label chain: for ALL 2,399 windows x 3 ticks x 4 runs: pred = argmax of the tick file's logits; true = the join's class at the route-matched row (own formula) with the on-route cut
  F1  the macro-F1 of the extracted cells (own implementation) = the pre-registered read's own-cell F1 (8 numbers, 1e-9) and its matched-cell F1 (R-TLAT / R-TLON f1_a / f1_b)
  S1  S error = |S checkpoint - S GT| from the tick file and the cells (independent), and the selection re-derived from the raw dumps (own code) = data/selection_v5.json, written before any render
  H1  histogram counts / shares / means recomputed from the dumps; each mean = DAG_READ R-ADE delta (4 decimals)
  J1  every DAG_READ.json pointer on screen re-read by own code; value in the frame's drawn text
  B1  R-ADE and R-T0 intervals re-derived from the float64 dumps / T0 arrays with the repo's estimator (B = 2,000, seed 0) = the DAG_READ cells
  N4  every numeric token of every chapter-8 drawn string is registered, declared, or in the selection
  W   wording: no 'drives' / 'driving' / 'autonomous' / 'SOTA'; 'closed loop' only inside 'NOT closed loop'; T1-kin on every chapter-8 frame; estimator + seed on the cards
  X   regression arm: with DAG and DAG-0 swapped for one scene, T1 and A1 must FAIL
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

TREE = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a"
ART = "E:/Projects/TanitAD-artifacts/v8f-dagger-t1"
JOIN = f"{TREE}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz"
DUMP = {(a, s): f"{ART}/dump_{a}_s{s}_m10_causal_heads" for a in ("DAG", "DAG0") for s in (0, 1)}
TICKF = {(a, s): f"{ART}/ticks/ticks_{a}_s{s}_m10_causal_heads.npz" for a in ("DAG", "DAG0") for s in (0, 1)}
CELLF = {(a, s): f"{HERE}/data/dag_cells_{a}_s{s}.npz" for a in ("DAG", "DAG0") for s in (0, 1)}
SEL = json.load(open(f"{HERE}/data/selection_v5.json", encoding="utf-8"))
DRJ = json.load(open(f"{HERE}/data/DAG_READ.json", encoding="utf-8"))
LAT5 = ("LANE_KEEP", "TURN_L", "TURN_R", "LANE_CHG_L", "LANE_CHG_R")
LON7 = ("HOLD", "CREEP", "STOP", "FOLLOW", "DECEL", "ACCEL", "KEEP")
MINUS = "\u2212"
TOL = 1e-6
NUM = re.compile(r"(?<![A-Za-z0-9_.])[\u2212+\-]?\d[\d,]*(?:\.\d+)?")


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def ntok(t):
    return t.lstrip("\u2212+-").replace(",", "")


def mm(x, spec):
    return spec.format(x).replace("-", MINUS)


def gp(obj, ptr):                                              # own pointer walker (not v3_lib's)
    for k in ptr.split("/"):
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    return obj


# ------------------------------------------------------------------------------------------------ independent raw access
_RAW: dict = {}


def raw_all(arm, seed):
    if (arm, seed) not in _RAW:
        acc = {k: [] for k in ("g", "cl", "ws", "clip_index")}
        for f in sorted(glob.glob(f"{DUMP[(arm, seed)]}/ep*.npz")):
            z = np.load(f)
            for k in acc:
                acc[k].append(z[k])
        _RAW[(arm, seed)] = {k: np.concatenate(v) for k, v in acc.items()}
    return _RAW[(arm, seed)]


def raw_row(arm, seed, clip_index, ws):
    z = np.load(f"{DUMP[(arm, seed)]}/ep{clip_index:04d}.npz")
    k = np.where(z["ws"] == ws)[0]
    assert len(k) == 1 and int(z["clip_index"][k[0]]) == clip_index
    return {n: np.asarray(z[n][k[0]], np.float64) for n in ("g", "cl")}


_TK: dict = {}


def raw_ticks(arm, seed):
    """own (window, tick) row index over the tick file."""
    if (arm, seed) not in _TK:
        z = np.load(TICKF[(arm, seed)], allow_pickle=False)
        win, tick = z["win"], z["tick"]
        idx = {(int(w), int(t)): r for r, (w, t) in enumerate(zip(win, tick))}
        _TK[(arm, seed)] = (z, idx)
    return _TK[(arm, seed)]


def tick_row(arm, seed, w, k):
    z, idx = raw_ticks(arm, seed)
    r = idx[(w, k)]
    return {"s_ckpt_w0": np.asarray(z["s_ckpt_w0"][r], np.float64), "lat": np.asarray(z["t_lat_logits"][r], np.float64), "lon": np.asarray(z["t_lon_logits"][r], np.float64)}


def cells_raw(arm, seed):
    return np.load(CELLF[(arm, seed)])


def ade_ind(p, g):
    return float(np.sqrt(((np.asarray(p) - np.asarray(g)) ** 2).sum(-1)).mean())


def plen_ind(t):
    p = np.vstack([np.zeros((1, 2)), np.asarray(t)[:, :2]])
    return float(np.sqrt((np.diff(p, axis=0) ** 2).sum(-1)).sum())


# ------------------------------------------------------------------------------------------------ the render of every chapter-8 scene
def render_all(frames_only=False, only_tag=None, fs=None):
    import v3_lib as L
    import v3_draw as DR
    import v3_scenes as V3
    import v4_scenes as V7
    import v5_scenes  # noqa: F401  (registers the chapter-8 scenes)
    import v5_lib as V
    out = []

    def call(fn, args):
        key = f"{fn}{tuple(args)}"
        L.CUR["scene"] = key
        n0 = len(L.REG)
        V3.SCENES[fn](*args)
        out.append({"key": key, "fn": fn, "args": list(args), "texts": list(DR.LAST_TEXTS), "reg": [dict(r) for r in L.REG[n0:]], "lines": [(g, x.copy(), y.copy()) for g, x, y in V7.LAST_LINES]})

    if not frames_only:
        call("d5_intro", ())
        call("d5_rule", ())
    for i, s in enumerate(V.SCN):
        if only_tag and s["tag"] != only_tag:
            continue
        for f in (fs if fs is not None else range(0, 31)):
            call("d5_frame", (i, f))
    if not frames_only:
        call("d5_hist", ())
        call("d5_tcard", ())
        call("d5_gcard", ())
        call("closing_card", ())
        call("title_card", ())
    return out


# ------------------------------------------------------------------------------------------------ T1, A1 (shared by --main and --swap)
def exp_lines(i, f, scn):
    s = scn[i]
    w = s["grid_row"]
    exp = []
    for arm in ("DAG", "DAG0"):
        r = raw_row(arm, 0, s["clip_index"], s["ws"])
        g = raw_row("DAG", 0, s["clip_index"], s["ws"])["g"]
        k = min(f // 10, 2)
        ck = tick_row(arm, 0, w, k)["s_ckpt_w0"]
        c = cells_raw(arm, 0)
        gck, ok = c["gt_w0"][w, k].astype(np.float64), c["gt_ok"][w, k]
        exp += [(f"{arm}:GT", -g[:30, 1], g[:30, 0]), (f"{arm}:O", -r["cl"][:f, 1], r["cl"][:f, 0]), (f"{arm}:S", -ck[:, 1], ck[:, 0]), (f"{arm}:SGT", -gck[ok][:, 1], gck[ok][:, 0])]
    return exp


def want_value(r, scn):
    """the string the registered per-window item must have, from the raw arrays."""
    kind = r["kind"]
    s = scn[[c["grid_row"] for c in scn].index(r["grid_row"])] if "grid_row" in r else None
    if kind in ("ade6", "ade6d", "plen"):
        row = {a: {sd: raw_row(a, sd, s["clip_index"], s["ws"]) for sd in (0, 1)} for a in ("DAG", "DAG0")}
        g = row["DAG"][0]["g"]
        if kind == "ade6":
            return r["spec"].format(ade_ind(row[r["arm"]][r["seed"]]["cl"], g))
        if kind == "ade6d":
            sd = r["seed"]
            return mm(ade_ind(row["DAG"][sd]["cl"], g) - ade_ind(row["DAG0"][sd]["cl"], g), r["spec"])
        return "{:.1f}".format(plen_ind(g if r["who"] == "GT" else row[r["who"]][r["seed"]]["cl"]))
    w = r["grid_row"]
    if kind in ("tcell", "tcount"):
        c = cells_raw(r["arm"], r["seed"])

        def one(k, head):
            lg = tick_row(r["arm"], r["seed"], w, k)[head]
            p, t = int(np.argmax(lg)), int(c[f"{head}_true"][w, k])
            return p, t

        if kind == "tcount":
            ok = n = 0
            for k in (1, 2):
                p, t = one(k, r["head"])
                if t >= 0:
                    n += 1
                    ok += int(p == t)
            return f"{ok}/{n}"
        names = LAT5 if r["head"] == "lat" else LON7
        p, t = one(r["tick"], r["head"])
        st = "no label" if t < 0 else ("match" if p == t else "MISS")
        return f"{names[p]}|{names[t] if t >= 0 else '-'}|{st}"
    if kind == "serr":
        c = cells_raw(r["arm"], r["seed"])
        ck = tick_row(r["arm"], r["seed"], w, r["tick"])["s_ckpt_w0"]
        gk, ok = c["gt_w0"][w, r["tick"]].astype(np.float64), c["gt_ok"][w, r["tick"]]
        e = np.sqrt(((ck[:, :2] - gk[:, :2]) ** 2).sum(-1))
        return "/".join("{:.1f}".format(e[q]) if ok[q] else "-" for q in range(3))
    raise KeyError(kind)


def check_tracks_and_numbers(R, scn):
    res = {"T1": {"n_frames": 0, "n_lines": 0, "max_abs_m": 0.0, "worst": None, "failures": []}, "A1": {"n_numbers": 0, "by_kind": {}, "failures": []}}
    for e in R:
        if e["fn"] != "d5_frame":
            continue
        i, f = e["args"]
        s = scn[i]
        exp = exp_lines(i, f, scn)
        got = e["lines"]
        res["T1"]["n_frames"] += 1
        if len(got) != len(exp):
            res["T1"]["failures"].append({"scene": s["tag"], "f": f, "why": f"{len(got)} artist lines, expected {len(exp)}"})
            continue
        for (gid, gx, gy), (eid, ex, ey) in zip(got, exp):
            res["T1"]["n_lines"] += 1
            if gid != eid or gx.shape != ex.shape:
                res["T1"]["failures"].append({"scene": s["tag"], "f": f, "why": f"line {gid} vs {eid}, shape {gx.shape} vs {ex.shape}"})
                continue
            d = float(max(np.abs(gx - ex).max() if len(ex) else 0.0, np.abs(gy - ey).max() if len(ey) else 0.0))
            if d > res["T1"]["max_abs_m"]:
                res["T1"]["max_abs_m"], res["T1"]["worst"] = d, f"{s['tag']} f={f} {gid}"
            if d > TOL:
                res["T1"]["failures"].append({"scene": s["tag"], "f": f, "line": gid, "max_abs_m": d})
        txt = norm(" ".join(e["texts"]))
        for r in e["reg"]:
            if r.get("kind") not in ("ade6", "ade6d", "plen", "tcell", "tcount", "serr"):
                continue
            if r["tag"] != s["tag"]:
                res["A1"]["failures"].append({"scene": s["tag"], "why": "tag mismatch"})
                continue
            want = want_value(r, scn)
            res["A1"]["n_numbers"] += 1
            res["A1"]["by_kind"][r["kind"]] = res["A1"]["by_kind"].get(r["kind"], 0) + 1
            if r["kind"] == "tcell":                                           # the shown decision names must be in the text; the label / status are in the text next to them
                p, t, st = want.split("|")
                intext = norm(f"T {r['head']}: {p}") in txt and ((norm(f"label {t}  {'OK' if st == 'match' else st}") in txt) or (t == "-" and "no label at this tick" in txt))
            else:
                intext = r["shown"] in txt
            if want != r["shown"] or not intext:
                res["A1"]["failures"].append({"scene": s["tag"], "kind": r["kind"], "arm": r.get("arm"), "seed": r.get("seed"), "shown": r["shown"], "recomputed": want, "in_text": intext})
    for k in ("T1", "A1"):
        res[k]["PASS"] = len(res[k]["failures"]) == 0 and (res[k].get("n_frames", 1) > 0) and (res[k].get("n_numbers", 1) > 0 or k == "T1")
    return res


# ------------------------------------------------------------------------------------------------ f1 helper (own)
def f1_own(true, pred, n):
    true, pred = np.asarray(true).ravel(), np.asarray(pred).ravel()
    m = (true >= 0) & (true < n)
    cm = np.zeros((n, n), np.int64)
    np.add.at(cm, (true[m], pred[m]), 1)
    f = np.full(n, np.nan)
    for c in range(n):
        tp = cm[c, c]
        fp, fn = cm[:, c].sum() - tp, cm[c, :].sum() - tp
        if cm[c].sum() > 0:
            f[c] = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return float(np.nanmean(f))


def main_checks():
    t0 = time.time()
    R = render_all()
    scn = SEL["scenes"]
    out = {"controls": {}, "n_scenes_rendered": len(R), "render_s": round(time.time() - t0, 1)}
    c = check_tracks_and_numbers(R, scn)
    out["controls"]["T1_plotted_lines_equal_raw_arrays"] = {**c["T1"], "tolerance_m": TOL}
    out["controls"]["A1_per_window_numbers_recomputed"] = c["A1"]

    # ---- L1 label chain over all windows, F1 reproduction
    J = np.load(JOIN, allow_pickle=False)
    l1 = {"runs": {}, "failures": []}
    f1c = {"rows": [], "failures": []}
    for (arm, sd) in DUMP:
        c_ = cells_raw(arm, sd)
        z, idx = raw_ticks(arm, sd)
        order = np.zeros((2399, 3), np.int64)
        for (w, k), r in idx.items():
            order[w, k] = r
        ci = np.asarray(raw_all(arm, sd)["clip_index"])
        for hd, n in (("lat", 5), ("lon", 7)):
            pred = z[f"t_{hd}_logits"][order].argmax(-1)
            jT = J["off"][ci][:, None] + c_["rT"]
            hasT = J["has"][jT]
            cls = J[f"{hd}_cls"][jT].astype(np.int64)
            true = np.where(hasT & (cls >= 0) & (c_["dT"] <= 3.0), cls, -1)
            ok = np.array_equal(pred, c_[f"{hd}_pred"]) and np.array_equal(true, c_[f"{hd}_true"])
            l1["runs"][f"{arm}_s{sd}_{hd}"] = {"pred_equal": bool(np.array_equal(pred, c_[f"{hd}_pred"])), "true_equal": bool(np.array_equal(true, c_[f"{hd}_true"])), "n_scored": int((true >= 0).sum())}
            if not ok:
                l1["failures"].append(f"{arm}_s{sd}_{hd}")
            own = f1_own(c_[f"{hd}_true"][:, 1:], c_[f"{hd}_pred"][:, 1:], n)
            key = f"rows/s{sd}/R-T{hd.upper()}_own_cells_beside/" + ("f1_a" if arm == "DAG" else "f1_b")
            ref = gp(DRJ, key)
            f1c["rows"].append({"run": f"{arm}_s{sd}_{hd}", "own_cells_f1": own, "read": ref, "abs_diff": abs(own - ref)})
            if abs(own - ref) > 1e-9:
                f1c["failures"].append(f"{arm}_s{sd}_{hd}")
    for sd in (0, 1):                                                              # matched cells: scored in both runs
        for hd, n in (("lat", 5), ("lon", 7)):
            a, b = cells_raw("DAG", sd), cells_raw("DAG0", sd)
            ta, tb = a[f"{hd}_true"][:, 1:], b[f"{hd}_true"][:, 1:]
            m = (ta >= 0) & (tb >= 0)
            fa = f1_own(np.where(m, ta, -1), a[f"{hd}_pred"][:, 1:], n)
            fb = f1_own(np.where(m, tb, -1), b[f"{hd}_pred"][:, 1:], n)
            ra, rb = gp(DRJ, f"rows/s{sd}/R-T{hd.upper()}/f1_a"), gp(DRJ, f"rows/s{sd}/R-T{hd.upper()}/f1_b")
            f1c["rows"].append({"run": f"matched_s{sd}_{hd}", "own": [fa, fb], "read": [ra, rb], "abs_diff": max(abs(fa - ra), abs(fb - rb))})
            if max(abs(fa - ra), abs(fb - rb)) > 1e-9:
                f1c["failures"].append(f"matched_s{sd}_{hd}")
    l1["PASS"] = not l1["failures"]
    f1c["PASS"] = not f1c["failures"] and len(f1c["rows"]) == 12
    out["controls"]["L1_label_chain_all_windows"] = l1
    out["controls"]["F1_macro_f1_reproduces_the_read"] = f1c

    # ---- S1: S error = |S ckpt - S GT| for ALL windows; selection re-derived
    s1 = {"failures": [], "max_abs_diff_m": 0.0}
    for (arm, sd) in DUMP:
        c_ = cells_raw(arm, sd)
        z, idx = raw_ticks(arm, sd)
        order = np.zeros((2399, 3), np.int64)
        for (w, k), r in idx.items():
            order[w, k] = r
        ck = z["s_ckpt_w0"][order].astype(np.float64)
        e = np.sqrt(((ck[..., :2] - c_["gt_w0"][..., :2]) ** 2).sum(-1))
        ok = c_["gt_ok"]
        d = float(np.abs(np.where(ok, e - c_["s_pos_err"], 0.0)).max())
        s1["max_abs_diff_m"] = max(s1["max_abs_diff_m"], d)
        if d > 1e-6 or not np.array_equal(np.isfinite(c_["s_pos_err"]), ok):
            s1["failures"].append(f"{arm}_s{sd}")
    g, cl, ws, ci = (raw_all("DAG", 0)[k] for k in ("g", "cl", "ws", "clip_index"))
    cl0 = raw_all("DAG0", 0)["cl"]
    strata = np.load(f"{HERE}/inputs/window_strata_eval141.npz", allow_pickle=True)["stratum"].astype(int)
    delta = np.sqrt(((cl - g) ** 2).sum(-1)).mean(-1) - np.sqrt(((cl0 - g) ** 2).sum(-1)).mean(-1)
    rows = []
    for k in range(3):
        ix = np.nonzero(strata == k)[0]
        o = sorted(ix, key=lambda i: (abs(float(delta[i])), int(i)))
        rows.append(int(o[(len(o) - 1) // 2]))
    rows.append(int(np.argmax(delta)))
    seg = np.diff(np.concatenate([np.zeros((len(g), 1, 2)), g], axis=1), axis=1)
    sp = np.sqrt((seg ** 2).sum(-1)).mean(-1) / 0.2
    cand = [j for j in range(len(sp)) if sp[j] >= 5.0 and j not in rows]
    pick = np.random.default_rng(20261027).choice(len(cand), size=3, replace=False)
    want = rows + [cand[int(p)] for p in pick]
    got = [c_["grid_row"] for c_ in scn]
    src = open(f"{HERE}/select_scenes5.py", encoding="utf-8").read()
    keys = set(re.findall(r"z\[\"(\w+)\"\]", src))
    s1["rows_rederived"], s1["rows_in_selection_v5"], s1["keys_read_by_selection_script"] = want, got, sorted(keys)
    s1["note_keys"] = "dump keys read: g, cl, ws, clip_index, eid; the others (stratum, t: the strata file; clip: the join) are other files; never fl / ol / ha0 / v0"
    s1["rule_sha256_matches_text"] = hashlib.sha256(SEL["rule_text"].encode("utf-8")).hexdigest() == SEL["rule_sha256"]
    mt = {p: os.path.getmtime(f"{HERE}/{p}") for p in ("data/selection_v5.json", "SELECTION_V5.md", "v5_lib.py", "v5_scenes.py")}
    qas = [os.path.getmtime(p) for p in glob.glob(f"{HERE}/qa_ch8_*.png")]
    s1["selection_files_older_than_every_chapter8_artifact"] = bool(max(mt["data/selection_v5.json"], mt["SELECTION_V5.md"]) < min([mt["v5_lib.py"], mt["v5_scenes.py"]] + qas))
    s1["PASS"] = (not s1["failures"] and want == got and not (keys & {"fl", "ol", "ha0", "v0"}) and s1["rule_sha256_matches_text"] and s1["selection_files_older_than_every_chapter8_artifact"])
    out["controls"]["S1_S_error_and_selection_rederived"] = s1

    # ---- H1, J1
    h1 = {"n": 0, "failures": []}
    j1 = {"n_items": 0, "failures": [], "by_kind": {}}
    ade_all = {sd: np.sqrt(((raw_all("DAG", sd)["cl"] - raw_all("DAG", sd)["g"]) ** 2).sum(-1)).mean(-1) - np.sqrt(((raw_all("DAG0", sd)["cl"] - raw_all("DAG0", sd)["g"]) ** 2).sum(-1)).mean(-1) for sd in (0, 1)}
    reg_all = []
    for e in R:
        txt = norm(" ".join(e["texts"]))
        for r in e["reg"]:
            reg_all.append((e["key"], r))
            k = r.get("kind")
            if k == "hist5":
                d = ade_all[r["seed"]]
                want_ = {"n_dag": int((d < 0).sum()), "n_dag0": int((d > 0).sum()), "n_tie": int((d == 0).sum()), "mean": mm(float(d.mean()), "{:.4f}"), "n": len(d)}
                h1["n"] += 1
                for kk, v in want_.items():
                    if r[kk] != v:
                        h1["failures"].append({"seed": r["seed"], "field": kk, "shown": r[kk], "recomputed": v})
                if want_["mean"] not in txt:
                    h1["failures"].append({"seed": r["seed"], "why": "mean not in drawn text"})
            elif k == "ref5":
                cell = mm(gp(DRJ, r["ptr"]), "{:.4f}")
                if cell != r["mean"]:
                    h1["failures"].append({"seed": r["seed"], "why": "histogram mean != DAG_READ R-ADE delta", "mean": r["mean"], "cell": cell})
            elif k in ("dagj", "dagb", "dagt", "dagw", "argv"):
                j1["n_items"] += 1
                j1["by_kind"][k] = j1["by_kind"].get(k, 0) + 1
                if k == "dagj":
                    v = gp(DRJ, r["ptr"])
                    v = v * r["scale"] if r["scale"] != 1.0 else v
                    want_ = r["spec"].format(v).replace("-", MINUS)
                elif k == "dagb":
                    want_ = str(gp(DRJ, r["ptr"]))
                elif k == "dagt":
                    want_ = gp(DRJ, r["ptr"])
                    if r.get("upto"):
                        want_ = want_.split(r["upto"])[0]
                elif k == "dagw":
                    sepr, dl = gp(DRJ, r["ptr"] + "/separated"), gp(DRJ, r["ptr"] + "/delta")
                    want_ = "n.s." if not sepr else ("better" if dl < 0 else "WORSE")
                else:
                    a = json.load(open(f"{HERE}/data/argv_M2b_s0_{'dag' if r['arm'] == 'DAG' else 'dag0'}.json", encoding="utf-8"))["argv"]
                    v = a[a.index(r["flag"]) + 1]
                    want_ = r["spec"].format(float(v)) if r.get("spec") else v
                shown_in = norm(want_) in txt or (k == "dagb")                      # booleans are used in conditions and in words ('yes'), not printed verbatim
                if want_ != r["shown"] or not shown_in:
                    j1["failures"].append({"scene": e["key"][:40], "kind": k, "ptr": r.get("ptr", r.get("flag")), "shown": r["shown"], "recomputed": want_, "in_text": shown_in})
    h1["PASS"] = h1["n"] == 2 and not h1["failures"]
    j1["PASS"] = j1["n_items"] > 0 and not j1["failures"]
    out["controls"]["H1_histogram_recomputed"] = h1
    out["controls"]["J1_dag_read_pointers_equal_the_file"] = j1

    # ---- B1: R-ADE and R-T0 intervals re-derived with the repo's estimator
    os.environ["TANITEVAL_STACK_OVERRIDE"] = f"{TREE}/stack"
    sys.path.insert(0, f"{TREE}/stack")
    sys.path.insert(0, f"{TREE}/taniteval")
    from taniteval.ci import paired_episode_cluster_bootstrap as pcb
    b1 = {"rows": [], "failures": []}
    runs = json.load(open(f"{ART}/runs.json", encoding="utf-8"))
    for sd in (0, 1):
        a, b = raw_all("DAG", sd), raw_all("DAG0", sd)
        am, bm = np.sqrt(((a["cl"] - a["g"]) ** 2).sum(-1)).mean(-1), np.sqrt(((b["cl"] - b["g"]) ** 2).sum(-1)).mean(-1)
        r = pcb(am, bm, a["clip_index"], n_boot=2000, seed=0)
        cell = gp(DRJ, f"rows/s{sd}/R-ADE")
        mine = [round(float(r["delta"]), 4), round(float(r["lo"]), 4), round(float(r["hi"]), 4)]
        b1["rows"].append({"row": f"R-ADE s{sd}", "rederived": mine, "read": [cell["delta"], cell["lo"], cell["hi"]], "equal": mine == [cell["delta"], cell["lo"], cell["hi"]]})
        ta = np.load(runs["arms"]["DAG"][str(sd)]["t0"]["npz"])
        tb = np.load(runs["arms"]["DAG0"][str(sd)]["t0"]["npz"])
        assert np.array_equal(ta["eid"], tb["eid"])
        r0 = pcb(np.asarray(ta["ade6"], np.float64), np.asarray(tb["ade6"], np.float64), ta["eid"], n_boot=2000, seed=0)
        cell0 = gp(DRJ, f"rows/s{sd}/R-T0")
        mine0 = [round(float(r0["delta"]), 4), round(float(r0["lo"]), 4), round(float(r0["hi"]), 4)]
        b1["rows"].append({"row": f"R-T0 s{sd}", "rederived": mine0, "read": [cell0["delta"], cell0["lo"], cell0["hi"]], "equal": mine0 == [cell0["delta"], cell0["lo"], cell0["hi"]]})
    b1["failures"] = [x for x in b1["rows"] if not x["equal"]]
    b1["PASS"] = not b1["failures"] and len(b1["rows"]) == 4
    out["controls"]["B1_intervals_reproduced_with_the_repo_estimator"] = b1

    # ---- N4 existence scan
    allowed = set()
    for _, r in reg_all:
        if "shown" in r:
            allowed |= {ntok(t) for t in NUM.findall(str(r["shown"]))}
        if r.get("kind") == "hist5":
            allowed |= {f"{100 * r['n_dag'] / r['n']:.1f}", f"{100 * r['n_dag0'] / r['n']:.1f}", str(r["n_dag"]), str(r["n_dag0"]), str(r["n_tie"])} | {ntok(t) for t in NUM.findall(str(r["mean"]))}
        if r.get("kind") == "cnt7":
            allowed.add(r["shown"])
    allowed |= {ntok(t) for t in NUM.findall(json.dumps(SEL))}
    allowed |= {ntok(t) for t in NUM.findall(" ".join(s["clip"] for s in SEL["scenes"]))}
    allowed |= {ntok(t) for t in NUM.findall(hashlib.sha256(open(f"{HERE}/data/DAG_READ.json", "rb").read()).hexdigest()[:8])}
    for s_ in SEL["scenes"]:
        allowed |= {f"{0.2 * s_['ws']:.1f}", f"{s_['gt_mean_speed']:.1f}"}
    allowed |= {f"{0.2 * k:.1f}" for k in range(0, 31)} | {f"{0.2 * k:.0f}" for k in range(0, 31, 5)}
    v4t = json.load(open(f"{HERE}/v4_ref/v4_onscreen_text.json", encoding="utf-8"))
    allowed |= {ntok(t) for t in NUM.findall(" ".join(v4t[-1]["texts"]))}                          # the numbers the v4 closing card already carried (controlled in v4)
    DECL = {"constants": ["0", "1", "2", "3", "4", "5", "6", "7", "10", "30", "141", "2399", "2000", "0.2", "6.0", "5.2", "20261027", "25", "50", "75", "80", "100", "150", "200", "250", "300", "350", "400", "120", "5.0", "2.0", "4.0", "0.0"],
            "why": "layout counters (window k of 7, 'of 2' replans, ticks 0-2, triangles 1-7, rule numbers), declared run constants (m = 10 steps of 0.2 s, 6 s window, 30 knots, 141 episodes, 2,399 windows, B = 2,000, "
                   "the 5 m/s speed bar, the rng seed), the 30 / 50 / 80 m checkpoint arcs, range-ring radii, the sha256 head of DAG_READ.json"}
    allowed |= set(DECL["constants"])
    n4 = {"n_tokens": 0, "unmatched": {}, "declared": DECL}
    for e in R:
        if e["fn"] == "title_card":
            continue
        for t in e["texts"]:
            for tok in NUM.findall(t):
                n4["n_tokens"] += 1
                if ntok(tok) not in allowed:
                    n4["unmatched"].setdefault(ntok(tok), []).append(e["key"][:40])
    n4["n_unmatched"] = len(n4["unmatched"])
    n4["unmatched"] = {k: v[:3] for k, v in list(n4["unmatched"].items())[:60]}
    n4["PASS"] = n4["n_unmatched"] == 0
    out["controls"]["N4_existence_scan"] = n4

    # ---- W wording
    w = {"failures": [], "n_frames": 0}
    bad = re.compile(r"\bdrives?\b|\bdriving\b|autonom|\bSOTA\b", re.I)
    for e in R:
        txt = norm(" ".join(e["texts"]))
        w["n_frames"] += 1
        if bad.search(txt):
            w["failures"].append({"scene": e["key"], "why": "forbidden word", "m": bad.search(txt).group(0)})
        for m in re.finditer(r"closed loop", txt, re.I):
            if txt[max(0, m.start() - 4):m.start()] != "NOT ":
                w["failures"].append({"scene": e["key"], "why": "'closed loop' not inside 'NOT closed loop'", "ctx": txt[max(0, m.start() - 20):m.end() + 5]})
        if e["fn"].startswith("d5") and "T1-kin" not in txt:
            w["failures"].append({"scene": e["key"], "why": "no T1-kin tier stamp"})
        if e["fn"] in ("d5_tcard", "d5_gcard", "d5_intro", "d5_rule", "d5_hist") and "episode-cluster bootstrap" not in txt:
            w["failures"].append({"scene": e["key"], "why": "no estimator"})
        if e["fn"] in ("d5_tcard", "d5_gcard", "d5_hist") and not re.search(r"seed 1|seed 0|s1\b", txt):
            w["failures"].append({"scene": e["key"], "why": "no seed"})
        if e["fn"] == "d5_frame" and e["args"][1] == 30 and ("s0:" not in txt or "s1:" not in txt):
            w["failures"].append({"scene": e["key"], "why": "end frame without both seeds"})
    w["PASS"] = not w["failures"]
    out["controls"]["W_wording_tier_estimator_seed"] = w

    # ---- G1: every window ends inside its clip
    g1 = {"failures": []}
    for s_ in SEL["scenes"]:
        if not (s_["ws"] + 30 < int(J["n"][s_["clip_index"]]) and str(J["clip"][s_["clip_index"]]) == s_["clip"]):
            g1["failures"].append(s_["tag"])
    g1["PASS"] = not g1["failures"]
    out["controls"]["G1_windows_inside_their_clips"] = g1

    # ---- the control render's per-frame digest + texts (the --post check compares the full render with them)
    dg = {}
    for e in R:
        if e["fn"] == "d5_frame":
            hh = hashlib.sha1()
            for gid, x_, y_ in e["lines"]:
                hh.update(gid.encode() + np.ascontiguousarray(x_, np.float64).tobytes() + np.ascontiguousarray(y_, np.float64).tobytes())
            dg[e["key"]] = {"sha1": hh.hexdigest(), "n_lines": len(e["lines"])}
    json.dump(dg, open(f"{HERE}/data/ch8_lines_digest_control.json", "w"))
    json.dump([{"scene": e["key"], "texts": e["texts"]} for e in R if e["fn"].startswith("d5") or e["fn"] == "closing_card"], open(f"{HERE}/data/onscreen_text_ch8_control.json", "w"))
    out["ALL_PASS"] = all(v.get("PASS") for v in out["controls"].values())
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{HERE}/CONTROLS_V5_main.json", "w"), indent=1, default=str)
    print(json.dumps({k: v.get("PASS") for k, v in out["controls"].items()}), "ALL_PASS", out["ALL_PASS"], out["elapsed_s"], "s", flush=True)
    for k, v in out["controls"].items():
        if not v.get("PASS"):
            print("FAIL", k, json.dumps(v, default=str)[:1800])


def swap_checks():
    """the deliberate regression: DAG and DAG-0 arrays of ONE scene swapped at the data-access layer (V5_SWAP_ARMS in the environment); the same T1 / A1 checks must FAIL for that scene."""
    assert os.environ.get("V5_SWAP_ARMS"), "run with V5_SWAP_ARMS=<tag>"
    tag = os.environ["V5_SWAP_ARMS"]
    R = render_all(frames_only=True, only_tag=tag, fs=(0, 10, 20, 30))
    c = check_tracks_and_numbers(R, SEL["scenes"])
    from PIL import Image
    import v3_scenes as V3
    import v3_lib as L
    i = [s["tag"] for s in SEL["scenes"]].index(tag)
    L.CUR["scene"] = "swap-still"
    Image.fromarray(V3.SCENES["d5_frame"](i, 30)).save(f"{HERE}/regression_swap_{tag}_end.png")
    det = {"scene_swapped": tag, "T1_detected": not c["T1"]["PASS"], "A1_detected": not c["A1"]["PASS"], "n_T1_failures": len(c["T1"]["failures"]), "n_A1_failures": len(c["A1"]["failures"]),
           "first_T1_failure": c["T1"]["failures"][:2], "first_A1_failure": c["A1"]["failures"][:2], "T1_max_abs_m": c["T1"]["max_abs_m"]}
    det["PASS"] = det["T1_detected"] and det["A1_detected"]                  # the regression arm PASSES when both checks FAIL on the swapped scene
    json.dump(det, open(f"{HERE}/CONTROLS_V5_swap.json", "w"), indent=1, default=str)
    print("regression arm", json.dumps({k: det[k] for k in ("T1_detected", "A1_detected", "n_T1_failures", "n_A1_failures", "PASS")}), flush=True)


def ffprobe(path):
    o = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration,pix_fmt", "-of", "json", path],
                       capture_output=True, text=True).stdout
    return json.loads(o)["streams"][0]


def post_checks():
    out = {}
    mp4 = f"{HERE}/v8f_evidence_v5.mp4"
    ch = json.load(open(f"{HERE}/video_chapters.json"))
    pr = ffprobe(mp4)
    out["V1_video"] = {"probe": pr, "expected_frames": ch["n_frames"], "expected_duration_s": ch["duration_s"], "bytes": os.path.getsize(mp4),
                       "PASS": pr["codec_name"] == "h264" and pr["width"] == 1920 and pr["height"] == 1080 and int(pr["nb_read_frames"]) == ch["n_frames"]}
    full = json.load(open(f"{HERE}/data/onscreen_text.json"))
    ctl = {e["scene"]: e["texts"] for e in json.load(open(f"{HERE}/data/onscreen_text_ch8_control.json"))}
    bad = [e["scene"] for e in full if e["chap"].startswith(("8", "9")) and e["scene"] in ctl and e["texts"] != ctl[e["scene"]]]
    miss = [k for k in ctl if k not in {e["scene"] for e in full}]
    out["P1_full_render_text_equals_control_render"] = {"n_checked": len(ctl), "differences": bad[:10], "missing_in_full_render": miss[:10], "PASS": not bad and not miss}
    dg_full, dg_ctl = json.load(open(f"{HERE}/data/ch78_lines_digest.json")), json.load(open(f"{HERE}/data/ch8_lines_digest_control.json"))
    dbad = [k for k in dg_ctl if dg_full.get(k) != dg_ctl[k]]
    out["P2_plotted_arrays_of_the_video_equal_the_controlled_ones"] = {"n_frames": len(dg_ctl), "differences": dbad[:10], "PASS": not dbad and all(k in dg_full for k in dg_ctl)}
    v4t = json.load(open(f"{HERE}/v4_ref/v4_onscreen_text.json"))
    HDR4 = "evidence video v4 (HACC + T1-kin)  |  numbers read from banked artifacts, paths in VIDEO_NOTES_V4.md"
    HDR5 = "evidence video v5 (HACC + T1-kin + DAgger)  |  numbers read from banked artifacts, paths in VIDEO_NOTES_V5.md"
    v4main = [e for e in v4t if not e["chap"].startswith("8")]                              # v4's chapters 1-7 (its chapter 8 is the closing card)
    v5main = [e for e in full if not e["chap"].startswith(("8", "9"))]
    assert len(v4main) == len(v5main), (len(v4main), len(v5main))
    n_same, diffs = 0, []
    for a, b in zip(v4main, v5main):
        ta = [t.replace(HDR4, HDR5) for t in a["texts"]]
        if a["scene"] == b["scene"] and ta == b["texts"]:
            n_same += 1
        else:
            diffs.append(a["scene"])
    out["U1_chapters_1_to_7_unchanged_vs_v4"] = {"n_frames": len(v4main), "identical_after_header_swap": n_same, "differing_frames": diffs, "expected_differing": ["title_card()"], "PASS": diffs == ["title_card()"]}
    nr4 = json.load(open(f"{HERE}/v4_ref/v4_numreg.json"))
    nr5 = json.load(open(f"{HERE}/data/numreg.json"))
    keep = lambda rr: [r for r in rr if not r["scene"].startswith(("closing_card", "d5_", "title_card"))]
    k4, k5 = keep(nr4), keep(nr5)
    out["U2_v4_registered_numbers_unchanged"] = {"n_v4": len(k4), "n_v5": len(k5), "PASS": k4 == k5}
    json.dump(out, open(f"{HERE}/CONTROLS_V5_post.json", "w"), indent=1, default=str)
    print(json.dumps({k: v["PASS"] for k, v in out.items()}), flush=True)
    merge()


def merge():
    f = {"main": f"{HERE}/CONTROLS_V5_main.json", "swap": f"{HERE}/CONTROLS_V5_swap.json", "post": f"{HERE}/CONTROLS_V5_post.json"}
    J = {k: json.load(open(p)) for k, p in f.items() if os.path.exists(p)}
    out = {"title": "CONTROLS_V5: chapter 8 (DAgger) of the evidence video v5", "main": J.get("main"), "X_regression_arm_swap_dag_dag0": J.get("swap"), "post_render": J.get("post")}
    ok = [J["main"]["ALL_PASS"], J["swap"]["PASS"]] if "main" in J and "swap" in J else [False]
    if "post" in J:
        ok.append(all(v["PASS"] for v in J["post"].values()))
    out["post_render_done"] = "post" in J
    out["ALL_PASS_before_render"] = bool(all(ok[:2]))
    out["ALL_PASS"] = bool(all(ok)) and "post" in J
    json.dump(out, open(f"{HERE}/CONTROLS_V5.json", "w"), indent=1, default=str)
    print("CONTROLS_V5.json  ALL_PASS_before_render", out["ALL_PASS_before_render"], " post_render_done", out["post_render_done"], " ALL_PASS", out["ALL_PASS"], flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--main"
    {"--main": main_checks, "--swap": swap_checks, "--post": post_checks, "--merge": merge}[mode]()
