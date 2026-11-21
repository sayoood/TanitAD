"""controls4.py -- CONTROLS_V4: the controls of chapter 7 (T1-kin) of the evidence video v4.   CPU only, ONE process.

  python controls4.py --main      render every chapter-7 scene (+ title / closing card) in THIS process, with the figure artists captured, and check (writes CONTROLS_V4_main.json)
  python controls4.py --swap      DELIBERATE REGRESSION: V4_SWAP_CLFL=JUNCTION swaps cl and fl of ONE scene at the data-access layer; the SAME track / ADE checks must FAIL (CONTROLS_V4_swap.json)
  python controls4.py --post      after the full render: video integrity, digest + text equality with the control render, chapters 1-6 unchanged vs v3 (updates CONTROLS_V4.json)
  python controls4.py --merge     CONTROLS_V4.json = main + swap (+ post if present); ALL_PASS only if main passes AND the regression arm was detected AND (if present) post passes

Controls (each with a tolerance; the numbers are in the JSON):
  T1  every plotted track (the artists of the figure, read back) = the raw dump arrays, <= 1e-6 m: 217 scene frames x 6 lines.  The reference is an INDEPENDENT loader (own np.load by
      episode file, row found by (clip_index, ws)), not v4_lib.
  A1  per-window ADE6 / delta shown = recomputed from the raw dumps (own formula), string-equal and found in the frame's drawn text.
  H1  histogram: counts, shares and means recomputed from the raw dumps of all 141 episodes; each mean = the RESULT file's point estimate string.
  R1  every RESULT cell / snippet registered by a scene = the cell of the file as RE-PARSED here (own code, own file read); its value token is in the scene's drawn text.
  N4  existence scan: every numeric token of every chapter-7 drawn string is in the RESULT files, in a registered per-window / histogram number, or in a declared constant list.
  W   wording: no 'drives' / 'driving' / 'autonomous' / 'SOTA'; 'closed loop' only inside 'NOT closed loop'; T1-kin on every frame; estimator + seed on every card.
  S1  selection re-derived from the raw GT (own code) = data/selection_v4.json; select_scenes.py never reads cl / fl / ol; the four v3 windows are in the grid with grid row = v3 window id.
  G1  frame accessor: the chapter's frames at +0.0 s = the v3 accessor's (pixel-identical); the dump's GT = v3's cf_windows GT of the same window.
  X   regression arm: with cl and fl of one scene swapped, T1 and A1 must FAIL on that scene.
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
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, "C:/Users/Admin/t1k/stack")
sys.path.insert(0, "C:/Users/Admin/t1k/taniteval")
MINUS = "\u2212"
DUMPS = {("causal", 0): "C:/Users/Admin/t1kc_run/dump_HACC_s0_m10_causal", ("causal", 1): "C:/Users/Admin/t1kc_run/dump_HACC_s1_m10_causal",
         ("strict", 0): "C:/Users/Admin/t1ks_run/dump_HACC_s0_m10_strict", ("strict", 1): "C:/Users/Admin/t1ks_run/dump_HACC_s1_m10_strict"}
RDIR = f"{HERE}/data/results"
SEL = json.load(open(f"{HERE}/data/selection_v4.json"))
TOL = 1e-6


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------------------------------------ independent raw access
def raw_row(run, seed, clip_index, ws):
    z = np.load(f"{DUMPS[(run, seed)]}/ep{clip_index:04d}.npz")
    k = np.where(z["ws"] == ws)[0]
    assert len(k) == 1 and int(z["clip_index"][k[0]]) == clip_index, (run, seed, clip_index, ws)
    return {n: np.asarray(z[n][k[0]], np.float64) for n in ("g", "cl", "fl")}


def raw_all(run, seed):
    acc = {"g": [], "cl": [], "fl": []}
    for f in sorted(glob.glob(f"{DUMPS[(run, seed)]}/ep*.npz")):
        z = np.load(f)
        for n in acc:
            acc[n].append(z[n])
    return {n: np.concatenate(v).astype(np.float64) for n, v in acc.items()}


def ade_ind(p, g):
    return float(np.sqrt(((np.asarray(p) - np.asarray(g)) ** 2).sum(-1)).mean())


def mm(x, spec):
    return spec.format(x).replace("-", MINUS)


def rfile(name):
    return open(f"{RDIR}/{name}.md", "rb").read().decode("utf-8").replace("\r\n", "\n")


def cell_ind(name, row_key, col, nth=0):
    hits = []
    for line in rfile(name).split("\n"):
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if row_key in cells[0]:
                hits.append(cells)
    assert len(hits) > nth, (name, row_key, len(hits))
    return hits[nth][col].replace("**", "").replace("\u2705", "").strip()


# ------------------------------------------------------------------------------------------------ the render of every chapter-7 scene
def render_all(frames_only=False, only_tag=None, fs=None):
    import v3_lib as L
    import v3_draw as DR
    import v3_scenes as V3
    import v4_scenes as V7
    import v4_lib as V
    out = []

    def call(fn, args):
        key = f"{fn}{tuple(args)}"
        L.CUR["scene"] = key
        n0 = len(L.REG)
        V3.SCENES[fn](*args)
        out.append({"key": key, "fn": fn, "args": list(args), "texts": list(DR.LAST_TEXTS), "reg": [dict(r) for r in L.REG[n0:]],
                    "lines": [(g, x.copy(), y.copy()) for g, x, y in V7.LAST_LINES]})

    if not frames_only:
        call("t1k_intro", ())
        call("t1k_rule", ())
    for i, s in enumerate(V.SCN):
        if only_tag and s["tag"] != only_tag:
            continue
        for f in (fs if fs is not None else range(0, 31)):
            call("t1k_frame", (i, f))
    if not frames_only:
        call("t1k_hist", ())
        call("t1k_card", ("causal",))
        call("t1k_card", ("strict",))
        call("t1k_lat", ())
        call("closing_card", ())
        call("title_card", ())
    return out


# ------------------------------------------------------------------------------------------------ T1, A1 (shared by --main and --swap)
def check_tracks_and_ade(R, scn):
    res = {"T1": {"n_frames": 0, "n_lines": 0, "max_abs_m": 0.0, "worst": None, "failures": []}, "A1": {"n_numbers": 0, "failures": []}}
    for e in R:
        if e["fn"] != "t1k_frame":
            continue
        i, f = e["args"]
        s = scn[i]
        rc = raw_row("causal", 0, s["clip_index"], s["ws"])
        rs = raw_row("strict", 0, s["clip_index"], s["ws"])
        exp = []
        for r in (rc, rs):
            exp += [("GT", -r["g"][:30, 1], r["g"][:30, 0]), ("FLOOR", -r["fl"][:f, 1], r["fl"][:f, 0]), ("MODEL", -r["cl"][:f, 1], r["cl"][:f, 0])]
        res["T1"]["n_frames"] += 1
        got = e["lines"]
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
        if f == 30:                                                  # A1: the numbers of the end frame
            txt = norm(" ".join(e["texts"]))
            for r in e["reg"]:
                if r.get("kind") not in ("ade6", "ade6d"):
                    continue
                row = {"causal": rc, "strict": rs}[r["run"]] if r["tag"] == s["tag"] else None
                if row is None:
                    res["A1"]["failures"].append({"scene": s["tag"], "why": "tag mismatch"})
                    continue
                # the registered seed may be 1: take that seed's rows
                rr = raw_row(r["run"], r["seed"], s["clip_index"], s["ws"])
                am, af = ade_ind(rr["cl"], rr["g"]), ade_ind(rr["fl"], rr["g"])
                want = {"ade6": lambda: ("{:.2f}".format(am if r["who"] == "model" else af)), "ade6d": lambda: mm(am - af, "{:+.2f}")}[r.get("kind")]()
                res["A1"]["n_numbers"] += 1
                if want != r["shown"] or r["shown"] not in txt:
                    res["A1"]["failures"].append({"scene": s["tag"], "run": r["run"], "seed": r["seed"], "kind": r.get("kind"), "who": r.get("who"), "shown": r["shown"], "recomputed": want,
                                                  "in_text": r["shown"] in txt})
    for k in ("T1", "A1"):
        res[k]["PASS"] = (len(res[k]["failures"]) == 0 and (res[k].get("n_frames", 1) > 0) and (res[k].get("n_numbers", 1) > 0 or k == "T1"))
    return res


# ------------------------------------------------------------------------------------------------ main
NUM = re.compile(r"(?<![A-Za-z0-9_.])[\u2212+\-]?\d[\d,]*(?:\.\d+)?")


def ntok(t):
    return t.lstrip("\u2212+-").replace(",", "")


def main_checks():
    t0 = time.time()
    import v4_lib as V
    R = render_all()
    scn = SEL["scenes"]
    out = {"controls": {}, "n_scenes_rendered": len(R), "render_s": round(time.time() - t0, 1)}
    c = check_tracks_and_ade(R, scn)
    out["controls"]["T1_tracks_equal_dumps"] = {**c["T1"], "tolerance_m": TOL}
    out["controls"]["A1_window_ade6_recomputed"] = c["A1"]

    # H1 + R1 registered numbers
    allrows = {}
    for run in ("causal", "strict"):
        for sd in (0, 1):
            a = raw_all(run, sd)
            allrows[(run, sd)] = ade_ind_all = np.sqrt(((a["cl"] - a["g"]) ** 2).sum(-1)).mean(-1) - np.sqrt(((a["fl"] - a["g"]) ** 2).sum(-1)).mean(-1)
    h1 = {"n": 0, "failures": []}
    r1 = {"n_cells": 0, "n_snippets": 0, "failures": [], "full_cell_on_screen": 0, "val_on_screen": 0}
    reg_all = []
    for e in R:
        txt = norm(" ".join(e["texts"]))
        for r in e["reg"]:
            reg_all.append((e["key"], r))
            if r.get("kind") == "hist":
                d = allrows[(r["run"], r["seed"])]
                want = {"n_model": int((d < 0).sum()), "n_floor": int((d > 0).sum()), "n_tie": int((d == 0).sum()), "mean": mm(float(d.mean()), "{:.4f}"), "n": len(d)}
                h1["n"] += 1
                for k, v in want.items():
                    if r[k] != v:
                        h1["failures"].append({"run": r["run"], "seed": r["seed"], "field": k, "shown": r[k], "recomputed": v})
                if mm(float(d.mean()), "{:.4f}") not in txt:
                    h1["failures"].append({"run": r["run"], "seed": r["seed"], "why": "mean not in drawn text"})
            elif r.get("kind") == "ref":
                cell = cell_ind(r["res_file"], r["row"], r["col"])
                if not cell.startswith(r["mean"]):
                    h1["failures"].append({"run": r["run"], "seed": r["seed"], "why": "histogram mean != RESULT point estimate", "mean": r["mean"], "cell": cell})
            elif r.get("kind") == "res":
                r1["n_cells"] += 1
                cell = cell_ind(r["file"], r["row"], r["col"], r.get("nth", 0))
                val = re.match(r"^([\u2212+\-]?\d+(?:\.\d+)?)", cell)
                if cell != r["shown"] or not val:
                    r1["failures"].append({"scene": e["key"], "row": r["row"], "col": r["col"], "shown": r["shown"], "file_cell": cell})
                elif val.group(1) not in txt:
                    r1["failures"].append({"scene": e["key"], "row": r["row"], "col": r["col"], "why": "value token not in the drawn text", "val": val.group(1)})
                else:
                    r1["val_on_screen"] += 1
                    r1["full_cell_on_screen"] += int(norm(cell) in txt)
            elif r.get("kind") == "txt":
                r1["n_snippets"] += 1
                ok_file = norm(r["shown"]) in norm(rfile(r["file"]))
                ok_txt = norm(r["shown"]) in txt or all(norm(p) in txt for p in re.split(r"\s{2,}", r["shown"]) if p)
                if not (ok_file and ok_txt):
                    r1["failures"].append({"scene": e["key"], "snippet": r["shown"][:90], "in_file": ok_file, "in_text": ok_txt})
    h1["PASS"] = h1["n"] == 4 and not h1["failures"]
    r1["PASS"] = r1["n_cells"] > 0 and not r1["failures"]
    out["controls"]["H1_histogram_recomputed"] = h1
    out["controls"]["R1_result_cells_equal_files"] = r1

    # N4 existence scan over the drawn text of chapter 7 (+ the closing card)
    allowed = set()
    for name in ("RESULT_V8F_T1KIN_CAUSALBUCKET", "RESULT_V8F_T1KIN_STRICT", "RESULT_V8F_T1_LATCOND", "RESULT_V8F_HACC_LAT"):
        allowed |= {ntok(t) for t in NUM.findall(rfile(name))}
    allowed |= {ntok(t) for t in NUM.findall(json.dumps(SEL))}
    allowed |= {ntok(t) for t in NUM.findall(" ".join(s["clip"] for s in SEL["scenes"]))}
    for _, r in reg_all:
        if r.get("kind") in ("ade6", "ade6d", "hist"):
            allowed |= {ntok(t) for t in NUM.findall(" ".join(str(v) for v in r.values()))}
    # the per-window numbers formatted for the table ('{:.2f}') of every scene are registered; the histogram percentages are derived from registered counts:
    for _, r in reg_all:
        if r.get("kind") == "hist":
            allowed.add(f"{100 * r['n_model'] / r['n']:.1f}")
            allowed.add(f"{100 * r['n_floor'] / r['n']:.1f}")
    DECL = {"constants": ["0", "1", "2", "3", "4", "5", "6", "7", "10", "30", "141", "2399", "2000", "0.2", "6.0", "5.2", "20261026", "25", "50", "75", "100", "150", "200", "250", "300", "350", "400"],
            "why": "layout counters (window k of 7, 'of 4 cells', replans 'of 2', rule / chapter numbers, triangles 1-7), declared run constants (m = 10 steps of 0.2 s, 6 s window, 30 knots, 141 episodes, 2,399 windows, "
                   "B = 2,000, the 5.2 m/s threshold, 5 m/s speed bar, the rng seed), range-ring radii, axis ticks (matplotlib tick labels are not in the drawn text; ring labels are)"}
    allowed |= set(DECL["constants"])
    for s_ in SEL["scenes"]:                                               # the rounded forms of the selection numbers shown in the rule card / sub-title: t = 0.2 ws (1 decimal), GT mean speed (1 decimal)
        allowed |= {f"{0.2 * s_['ws']:.1f}", f"{s_['gt_mean_speed']:.1f}"}
    for _, r in reg_all:                                                   # every registered, source-bound number (v3's N() items on the closing card, RESULT cells, per-window numbers)
        if "shown" in r:
            allowed |= {ntok(t) for t in NUM.findall(str(r["shown"]))}
    v3t_ = json.load(open(f"{HERE}/data/v3_onscreen_text.json"))
    allowed |= {ntok(t) for t in NUM.findall(" ".join(v3t_[-1]["texts"]))}   # the numbers the v3 closing card already carried (controlled in v3)
    # window times (+0.0 .. +6.0 s) and the frame time string
    allowed |= {f"{0.2 * k:.1f}" for k in range(0, 31)} | {f"{0.2 * k:.0f}" for k in range(0, 31, 5)}
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

    # W wording
    w = {"failures": [], "n_frames": 0}
    bad = re.compile(r"\bdrives?\b|\bdriving\b|autonom|\bSOTA\b", re.I)
    for e in R:
        txt = norm(" ".join(e["texts"]))
        if e["fn"] == "title_card":
            tit = txt
        w["n_frames"] += 1
        if bad.search(txt):
            w["failures"].append({"scene": e["key"], "why": "forbidden word", "m": bad.search(txt).group(0)})
        for m in re.finditer(r"closed loop", txt, re.I):
            if txt[max(0, m.start() - 4):m.start()] != "NOT ":
                w["failures"].append({"scene": e["key"], "why": "'closed loop' not inside 'NOT closed loop'", "ctx": txt[max(0, m.start() - 20):m.end() + 5]})
        if e["fn"].startswith("t1k") and "T1-kin" not in txt:
            w["failures"].append({"scene": e["key"], "why": "no T1-kin tier stamp"})
        if e["fn"] in ("t1k_card", "t1k_lat", "t1k_intro", "t1k_rule", "t1k_hist") and "episode-cluster bootstrap" not in txt:
            w["failures"].append({"scene": e["key"], "why": "no estimator"})
        if e["fn"] in ("t1k_card", "t1k_lat", "t1k_hist") and not re.search(r"seed 1|seed 0|s1\b", txt):
            w["failures"].append({"scene": e["key"], "why": "no seed"})
        if e["fn"] == "t1k_frame" and e["args"][1] == 30 and "seed 0" not in txt:
            w["failures"].append({"scene": e["key"], "why": "end frame without the seed"})
    w["PASS"] = not w["failures"]
    out["controls"]["W_wording_tier_estimator_seed"] = w

    # S1 selection
    s1 = {"failures": []}
    src = open(f"{HERE}/select_scenes.py", encoding="utf-8").read()
    s1["script_reads_only_g_ws_clip_index"] = not re.search(r"""\[['"](cl|fl|ol|ha0)['"]\]""", src) and 'z["g"]' in src
    from tanitad.data.v8f_mini_data import load_join  # noqa: E402
    J = load_join("C:/Users/Admin/t1k/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz")
    clips = [str(c_) for c_ in J["clip"]]
    G, WS, CI = [], [], []
    for f in sorted(glob.glob(f"{DUMPS[('causal', 0)]}/ep*.npz")):
        z = np.load(f)
        G.append(z["g"]); WS.append(z["ws"]); CI.append(z["clip_index"])
    G, WS, CI = np.concatenate(G), np.concatenate(WS), np.concatenate(CI)
    seg = np.diff(np.concatenate([np.zeros((len(G), 1, 2)), G], axis=1), axis=1)
    sp = np.sqrt((seg ** 2).sum(-1)).mean(-1) / 0.2
    v3 = json.load(open(f"{HERE.rsplit('/', 1)[0]}/viz/out/selection.json"))
    rows = []
    for nm in ("JUNCTION", "CURVE", "NO-TURN", "FAILURE(worst)"):
        k = np.where((CI == clips.index(v3[nm]["clip"])) & (WS == v3[nm]["t"]))[0]
        s1.setdefault("v3_in_grid", {})[nm] = {"found": len(k) == 1, "row_equals_v3_window_id": bool(len(k) == 1 and int(k[0]) == v3[nm]["window"])}
        rows.append(int(k[0]))
    cand = [j for j in range(len(sp)) if sp[j] >= 5.0 and j not in rows]
    pick = np.random.default_rng(20261026).choice(len(cand), size=3, replace=False)
    want_rows = rows + [cand[int(p)] for p in pick]
    got_rows = [s["grid_row"] for s in SEL["scenes"]]
    s1["rows_rederived"], s1["rows_in_selection_v4"] = want_rows, got_rows
    s1["n_candidates"] = len(cand)
    st = {"selection_v4.json": os.path.getmtime(f"{HERE}/data/selection_v4.json"), "v4_lib.py": os.path.getmtime(f"{HERE}/v4_lib.py"), "SELECTION_V4.md": os.path.getmtime(f"{HERE}/SELECTION_V4.md")}
    s1["mtimes_selection_before_data_layer"] = st["selection_v4.json"] < st["v4_lib.py"]
    s1["PASS"] = (want_rows == got_rows and s1["script_reads_only_g_ws_clip_index"] and all(v["found"] and v["row_equals_v3_window_id"] for v in s1["v3_in_grid"].values()) and s1["mtimes_selection_before_data_layer"])
    out["controls"]["S1_selection_rederived"] = s1

    # G1 frames and GT alignment with v3
    import v3_lib as L
    g1 = {"frame_pixel_equal": {}, "gt_vs_v3_cf_max_abs_m": {}}
    CF = np.load(f"{HERE}/data/cf_windows.npz")
    import v4_lib as V4
    for i, s in enumerate(SEL["scenes"]):
        if s["source"] != "v3":
            continue
        sel = v3[s["tag"]]
        a = L.v2ep_frame(sel["clip"], int(sel["t"]))
        b = V4.frame_for(i, "causal", 0)
        c_ = V4.frame_for(i, "strict", 7)
        g1["frame_pixel_equal"][s["tag"]] = bool((a == b).all() and (a == c_).all())
        r = raw_row("causal", 0, s["clip_index"], s["ws"])
        d = float(np.abs(CF[f"M2b_s0/{s['tag']}/gt"][:, :2] - r["g"]).max())
        g1["gt_vs_v3_cf_max_abs_m"][s["tag"]] = d
    g1["PASS"] = all(g1["frame_pixel_equal"].values()) and all(v < 1e-3 for v in g1["gt_vs_v3_cf_max_abs_m"].values())
    out["controls"]["G1_frames_and_gt_alignment_with_v3"] = g1

    # floor identical across seeds and runs (used by the histogram note)
    fl = {k: raw_all(*k)["fl"] for k in DUMPS}
    ref = fl[("causal", 0)]
    out["controls"]["F1_floor_identical_all_four_dump_sets"] = {"max_abs_m": {f"{k[0]}_s{k[1]}": float(np.abs(v - ref).max()) for k, v in fl.items()}, "PASS": all(float(np.abs(v - ref).max()) == 0.0 for v in fl.values())}

    # B1: the four primary intervals re-derived from the float64 dumps with the repo's estimator (taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000, seed 0) = the RESULT cells
    from taniteval.ci import paired_episode_cluster_bootstrap as pcb
    b1 = {"rows": [], "failures": []}
    for run, name in (("causal", "RESULT_V8F_T1KIN_CAUSALBUCKET"), ("strict", "RESULT_V8F_T1KIN_STRICT")):
        for sd in (0, 1):
            a = raw_all(run, sd)
            am, af = np.sqrt(((a["cl"] - a["g"]) ** 2).sum(-1)).mean(-1), np.sqrt(((a["fl"] - a["g"]) ** 2).sum(-1)).mean(-1)
            ci = np.concatenate([np.load(f)["clip_index"] for f in sorted(glob.glob(DUMPS[(run, sd)] + "/ep*.npz"))])
            r = pcb(am, af, ci, n_boot=2000, seed=0)
            mine = f"{mm(r['delta'], '{:.4f}')} [{mm(r['lo'], '{:.4f}')}, {mm(r['hi'], '{:.4f}')}]"
            cell = cell_ind(name, "ADE6, PRIMARY", sd + 1)
            b1["rows"].append({"run": run, "seed": sd, "rederived": mine, "result_file": cell, "equal": mine == cell, "n_windows": r["n_windows"], "n_episodes": r["n_episodes"]})
            if mine != cell:
                b1["failures"].append(b1["rows"][-1])
    b1["PASS"] = not b1["failures"] and len(b1["rows"]) == 4
    out["controls"]["B1_primary_intervals_reproduced_from_dumps"] = b1

    # write the control render's per-frame digest + texts (the --post check compares the full render with them)
    dg = {}
    for e in R:
        if e["fn"].startswith("t1k_"):
            hh = hashlib.sha1()
            for gid, x_, y_ in e["lines"]:
                hh.update(gid.encode() + np.ascontiguousarray(x_, np.float64).tobytes() + np.ascontiguousarray(y_, np.float64).tobytes())
            dg[e["key"]] = {"sha1": hh.hexdigest(), "n_lines": len(e["lines"])}
    json.dump(dg, open(f"{HERE}/data/ch7_lines_digest_control.json", "w"))
    json.dump([{"scene": e["key"], "texts": e["texts"]} for e in R if e["fn"].startswith("t1k_") or e["fn"] == "closing_card"], open(f"{HERE}/data/onscreen_text_ch7_control.json", "w"))
    out["ALL_PASS"] = all(v.get("PASS") for v in out["controls"].values())
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{HERE}/CONTROLS_V4_main.json", "w"), indent=1, default=str)
    print(json.dumps({k: v.get("PASS") for k, v in out["controls"].items()}), "ALL_PASS", out["ALL_PASS"], out["elapsed_s"], "s", flush=True)
    for k, v in out["controls"].items():
        if not v.get("PASS"):
            print("FAIL", k, json.dumps(v, default=str)[:1500])


def swap_checks():
    """the deliberate regression: cl and fl of ONE scene swapped at the data-access layer (V4_SWAP_CLFL in the environment); the same T1 / A1 checks must FAIL for that scene."""
    assert os.environ.get("V4_SWAP_CLFL"), "run with V4_SWAP_CLFL=<tag>"
    tag = os.environ["V4_SWAP_CLFL"]
    R = render_all(frames_only=True, only_tag=tag, fs=(0, 10, 20, 30))
    c = check_tracks_and_ade(R, SEL["scenes"])
    from PIL import Image
    import v3_scenes as V3
    import v4_scenes  # noqa: F401
    import v3_lib as L
    i = [s["tag"] for s in SEL["scenes"]].index(tag)
    L.CUR["scene"] = "swap-still"
    Image.fromarray(V3.SCENES["t1k_frame"](i, 30)).save(f"{HERE}/regression_swap_{tag}_end.png")
    det = {"scene_swapped": tag, "T1_detected": not c["T1"]["PASS"], "A1_detected": not c["A1"]["PASS"], "n_T1_failures": len(c["T1"]["failures"]), "n_A1_failures": len(c["A1"]["failures"]),
           "first_T1_failure": c["T1"]["failures"][:2], "first_A1_failure": c["A1"]["failures"][:2], "T1_max_abs_m": c["T1"]["max_abs_m"]}
    det["PASS"] = det["T1_detected"] and det["A1_detected"]                  # the regression arm PASSES when both checks FAIL on the swapped scene
    json.dump(det, open(f"{HERE}/CONTROLS_V4_swap.json", "w"), indent=1, default=str)
    print("regression arm", json.dumps({k: det[k] for k in ("T1_detected", "A1_detected", "n_T1_failures", "n_A1_failures", "PASS")}), flush=True)


def ffprobe(path):
    o = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration,pix_fmt", "-of", "json", path],
                       capture_output=True, text=True).stdout
    return json.loads(o)["streams"][0]


def post_checks():
    out = {}
    mp4 = f"{HERE}/v8f_evidence_v4.mp4"
    ch = json.load(open(f"{HERE}/video_chapters.json"))
    pr = ffprobe(mp4)
    out["V1_video"] = {"probe": pr, "expected_frames": ch["n_frames"], "expected_duration_s": ch["duration_s"], "bytes": os.path.getsize(mp4),
                       "PASS": pr["codec_name"] == "h264" and pr["width"] == 1920 and pr["height"] == 1080 and int(pr["nb_read_frames"]) == ch["n_frames"]}
    full = json.load(open(f"{HERE}/data/onscreen_text.json"))
    ctl = {e["scene"]: e["texts"] for e in json.load(open(f"{HERE}/data/onscreen_text_ch7_control.json"))}
    bad = [e["scene"] for e in full if e["chap"].startswith(("7", "8")) and e["scene"] in ctl and e["texts"] != ctl[e["scene"]]]
    n7 = sum(1 for e in full if e["chap"].startswith("7") or e["chap"].startswith("8"))
    miss = [k for k in ctl if k not in {e["scene"] for e in full}]
    out["P1_full_render_text_equals_control_render"] = {"n_checked": len(ctl), "differences": bad[:10], "missing_in_full_render": miss[:10], "PASS": not bad and not miss}
    dg_full, dg_ctl = json.load(open(f"{HERE}/data/ch7_lines_digest.json")), json.load(open(f"{HERE}/data/ch7_lines_digest_control.json"))
    dbad = [k for k in dg_ctl if dg_full.get(k) != dg_ctl[k]]
    out["P2_plotted_arrays_of_the_video_equal_the_controlled_ones"] = {"n_frames": len(dg_ctl), "differences": dbad[:10], "PASS": not dbad and len(dg_full) == len(dg_ctl)}
    v3t = json.load(open(f"{HERE}/data/v3_onscreen_text.json"))
    HDR3 = "evidence video v3 (HACC)  |  numbers read from banked artifacts, paths in VIDEO_NOTES_V3.md"
    HDR4 = "evidence video v4 (HACC + T1-kin)  |  numbers read from banked artifacts, paths in VIDEO_NOTES_V4.md"
    n_same, diffs = 0, []
    v3main = v3t[:-1]                                                         # the closing card is the last v3 frame
    v4main = [e for e in full if not e["chap"].startswith(("7", "8"))]
    assert len(v3main) == len(v4main), (len(v3main), len(v4main))
    for a, b in zip(v3main, v4main):
        ta = [t.replace(HDR3, HDR4) for t in a["texts"]]
        if a["scene"] == b["scene"] and ta == b["texts"]:
            n_same += 1
        else:
            diffs.append(a["scene"])
    out["U1_chapters_1_to_6_unchanged_vs_v3"] = {"n_frames": len(v3main), "identical_after_header_swap": n_same, "differing_frames": diffs, "expected_differing": ["title_card()"], "PASS": diffs == ["title_card()"]}
    nr3 = json.load(open(f"{HERE}/data/v3_numreg.json"))
    nr4 = json.load(open(f"{HERE}/data/numreg.json"))
    k3 = [(r["scene"], r["src"], r["ptr"], r["shown"]) for r in nr3 if not r["scene"].startswith("closing_card")]
    k4 = [(r["scene"], r["src"], r["ptr"], r["shown"]) for r in nr4 if "src" in r and not r["scene"].startswith("closing_card")]
    out["U2_v3_registered_numbers_unchanged"] = {"n_v3": len(k3), "n_v4": len(k4), "PASS": k3 == k4}
    json.dump(out, open(f"{HERE}/CONTROLS_V4_post.json", "w"), indent=1, default=str)
    print(json.dumps({k: v["PASS"] for k, v in out.items()}), flush=True)
    merge()


def merge():
    f = {"main": f"{HERE}/CONTROLS_V4_main.json", "swap": f"{HERE}/CONTROLS_V4_swap.json", "post": f"{HERE}/CONTROLS_V4_post.json"}
    J = {k: json.load(open(p)) for k, p in f.items() if os.path.exists(p)}
    out = {"title": "CONTROLS_V4: chapter 7 (T1-kin) of the evidence video v4", "main": J.get("main"), "X_regression_arm_swap_cl_fl": J.get("swap"), "post_render": J.get("post")}
    ok = [J["main"]["ALL_PASS"], J["swap"]["PASS"]] if "main" in J and "swap" in J else [False]
    if "post" in J:
        ok.append(all(v["PASS"] for v in J["post"].values()))
    out["post_render_done"] = "post" in J
    out["ALL_PASS_before_render"] = bool(all(ok[:2]))
    out["ALL_PASS"] = bool(all(ok)) and "post" in J
    json.dump(out, open(f"{HERE}/CONTROLS_V4.json", "w"), indent=1, default=str)
    print("CONTROLS_V4.json  ALL_PASS_before_render", out["ALL_PASS_before_render"], " post_render_done", out["post_render_done"], " ALL_PASS", out["ALL_PASS"], flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--main"
    {"--main": main_checks, "--swap": swap_checks, "--post": post_checks, "--merge": merge}[mode]()
