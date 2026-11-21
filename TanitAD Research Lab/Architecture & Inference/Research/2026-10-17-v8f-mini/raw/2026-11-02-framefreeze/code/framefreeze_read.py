#!/usr/bin/env python3
"""framefreeze_read.py -- the control read of the FRAME-FREEZE (``--scene hold_now``) rollouts (RECOVER-FT card section 7; DRAFT_RECOVERY_CARD_v2.md section 8.3), on the base model DAG-OACC, s0 / s1.
Tier T1-kin; evidence class MEASURED (ours); estimator of every interval: paired episode-cluster bootstrap, B = 2000, seed 0 (taniteval.ci draw stream via gapdecomp.decompose), answers another draw of EPISODES only
(X and Y share the checkpoint, so no training-seed variance is inside X - Y).

Arms per seed:  Y = KAPPA-T1's SREP_DAGOACC_s (logged frames, reused);  FZ-Y = the same checkpoint, ``--smooth sg72 --scene hold_now``;  FZ-HLVL = the same, ``--histswap cv_logged --scene hold_now``
(the known-answer positive control: its level enters through the history input, not the frames).
Controls computed before any row (the frozen card REFUSES the frame-freeze contrast when KFZ3 or KFZ4 fails; the read here WRITES them, it does not crash on them):
  KFZ1  every item the network was handed had feats rows == the tick-j frame (max abs deviation exactly 0.0, no unequal element), over EVERY item of every production run (the run header's receipt) + the unit test;
  KFZ2  the unit tests (apply line deleted / frozen to the first row / one row left unfrozen turn KFZ1 RED; ``--scene logged`` and flag-absent bit-identical to the source tool): the pytest log's verdict;
  KFZ3  the freeze reached the network: FZ-Y's plan_raw differs from the logged-frame Y's by > 1e-3 m at ticks >= 1 (matched by (window, tick));
  KFZ4  FZ-HLVL keeps its closure against FZ-Y: M1 X - Y separated-better at BOTH seeds and the mean share of (FZ-Y - P10)'s M1 excess closed >= 0.5 (HISTSWAP-LF's m1_cell; an undefined share counts 0);
  plus KY (the reused Y is the pinned one), KC1 (flags reached the runs), KC3 (the plan is committed to the executed state), KF (windows and P10 floor equal Y's: the floor reads no frames), KB1 (the P10 side of every row is Y's).
Reported: the freeze's own effect FZ-Y - Y on all 19 rows (and FZ-Y - P10, Y - P10, FZ-HLVL - FZ-Y, FZ-HLVL - P10). A failed KFZ4 is a FINDING (written to FZ_CONTROLS.json, ``recover_ft_contrast_usable`` false), not a crash.
``--dry-run`` exercises the whole heavy path on EXISTING runs (FZ-Y := KAPPA-T1's SREP_*, i.e. NO freeze, so KFZ3 must read 0; FZ-HLVL := HISTSWAP-LF's HLV_DAGOACC_*, so KFZ4 must reproduce LF's HLVL cells) and writes to --raw only.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
LF_PK = f"{ART}/dagland/histswap_lf"
LF_READ = f"{LF_PK}/code/histswap_lf_read.py"
PIN_LF_READ = "4e3a75db6023d7eb07465bb826fe56455dac01aa1aba42fda0b197626cbe119f"      # histswap_lf FROZEN_SHA.txt
FT_PK = f"{ART}/dagland/recover_ft"
OUT_Y = "F:/TanitAD-scratch/kappa_t1/out"                 # KAPPA-T1's fresh S runs (SREP_*), reused as Y
OUT_X = "F:/TanitAD-scratch/framefreeze/out"
OUT_LF = "F:/TanitAD-scratch/histswap_lf/out"             # dry run only
WORK = "F:/TanitAD-scratch/framefreeze/work/rig"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SEEDS = {"s0": ("DAGOACC_s0", "CTRL", 0), "s1": ("DAGOACC_s1", "CTRL", 1)}
M1 = "M1 crawl share"
KFZ3_TOL_M = 1e-3
SHARE_BAR = 0.5
TOL = 1e-4
FL_GROSS_M = 1.0
REPLAN_M = 10


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ pure layer (unit-checkable, no file access)
def match_rows(tx, ty):
    """Row indices (ix, iy) that pair the two ticks files by (win, tick); raises when the row sets differ (the same windows and ticks are required)."""
    kx = {(int(w), int(t)): i for i, (w, t) in enumerate(zip(np.asarray(tx["win"]), np.asarray(tx["tick"])))}
    ky = {(int(w), int(t)): i for i, (w, t) in enumerate(zip(np.asarray(ty["win"]), np.asarray(ty["tick"])))}
    if kx.keys() != ky.keys():
        raise ValueError(f"the ticks files do not cover the same (window, tick) rows: {len(kx)} vs {len(ky)}, symmetric difference {len(kx.keys() ^ ky.keys())}")
    keys = sorted(kx)
    ix = np.array([kx[k] for k in keys])
    iy = np.array([ky[k] for k in keys])
    for k in ("ci", "t0"):
        if not np.array_equal(np.asarray(tx[k])[ix], np.asarray(ty[k])[iy]):
            raise ValueError(f"the ticks files disagree on '{k}' for the same (window, tick)")
    return ix, iy


def kfz3(tx, ty, tol=KFZ3_TOL_M):
    """KFZ3: the freeze reached the network. ``tx`` = FZ-Y's ticks, ``ty`` = the logged-frame Y's. PASS iff the max abs plan_raw deviation (m) over rows at ticks >= 1 exceeds ``tol``.
    Also reported (not gating): the tick-0 deviation (the two runs share the executed state there, so it is the freeze alone), the share of rows over ``tol`` and the median deviation."""
    ix, iy = match_rows(tx, ty)
    dev = np.abs(np.asarray(tx["plan_raw"], np.float64)[ix] - np.asarray(ty["plan_raw"], np.float64)[iy]).reshape(len(ix), -1).max(1)
    tick = np.asarray(tx["tick"])[ix]
    ge1, t0 = tick >= 1, tick == 0
    return {"n_rows": int(len(ix)), "n_rows_ticks_ge1": int(ge1.sum()), "max_dev_ticks_ge1_m": float(dev[ge1].max()), "median_dev_ticks_ge1_m": float(np.median(dev[ge1])),
            "frac_rows_ticks_ge1_over_tol": float((dev[ge1] > tol).mean()), "max_dev_tick0_m": float(dev[t0].max()) if t0.any() else None,
            "frac_rows_tick0_over_tol": float((dev[t0] > tol).mean()) if t0.any() else None, "tol_m": tol, "PASS": bool(ge1.any() and dev[ge1].max() > tol)}


def kfz4(cells, bar=SHARE_BAR):
    """KFZ4: FZ-HLVL keeps its closure against FZ-Y. ``cells`` = {"s0": m1_cell, "s1": m1_cell} (HISTSWAP-LF's ``m1_cell`` of X = FZ-HLVL, Y = FZ-Y). PASS iff X - Y is separated-better on M1 at BOTH seeds
    AND the mean share closed (an undefined share, i.e. FZ-Y - P10 has no M1 excess, counts 0) is >= ``bar``. ``each_seed_share_ge_bar`` is reported beside it (the stricter per-seed reading)."""
    if len(cells) != 2:
        raise ValueError("KFZ4 is defined on exactly two cells (s0, s1)")
    n_better = sum(bool(c["better"]) for c in cells.values())
    shares = {k: (0.0 if c["share_closed"] is None else float(c["share_closed"])) for k, c in cells.items()}
    mean = float(np.mean(list(shares.values())))
    return {"n_seeds_separated_better": int(n_better), "share_closed": {k: c["share_closed"] for k, c in cells.items()}, "mean_share_closed": mean, "share_bar": bar,
            "each_seed_share_ge_bar": bool(all(v >= bar for v in shares.values())), "undefined_share_seeds": [k for k, c in cells.items() if c["share_closed"] is None],
            "PASS": bool(n_better == 2 and mean >= bar)}


def freeze_receipt(hdr, n_tick_rows):
    """KFZ1 on a production run: the run header's receipt says every item built by ``build_item`` had every feats row equal to the tick-j frame (max row deviation exactly 0.0, no unequal element). The number
    of items is a positive multiple of the ticks-file rows (the model's rollout and the P10 floor's rollout both build items through ``build_item``)."""
    ff = hdr.get("frame_freeze") or {}
    n = ff.get("items_frozen")
    ok = bool(ff and n and n_tick_rows and n % n_tick_rows == 0 and n >= n_tick_rows and ff.get("max_row_dev") == 0.0 and ff.get("n_unequal_elements") == 0
              and hdr.get("scene") == "hold_now" and hdr.get("camera_motion") is False and hdr.get("strict", {}).get("camera") is False)
    return {"items_frozen": n, "ticks_rows": int(n_tick_rows), "items_per_tick_row": (n / n_tick_rows) if n and n_tick_rows else None, "max_row_dev": ff.get("max_row_dev"),
            "n_unequal_elements": ff.get("n_unequal_elements"), "scene": hdr.get("scene"), "camera_motion": hdr.get("camera_motion"), "strict_camera": hdr.get("strict", {}).get("camera"),
            "card_ref": ff.get("card_ref"), "PASS": ok}


def pytest_verdict(path):
    """KFZ1 / KFZ2 unit-test verdict: the last non-empty line of the pytest log must say ``N passed`` and no ``failed`` / ``error``."""
    if not os.path.exists(path):
        return {"log": path, "PASS": False, "summary_line": "UNVERIFIED: log absent"}
    lines = [ln.strip() for ln in open(path, encoding="utf-8", errors="replace") if ln.strip()]
    last = lines[-1] if lines else ""
    return {"log": path, "sha256": sha256_file(path), "summary_line": last, "PASS": bool(" passed" in last and " failed" not in last and " error" not in last)}


def kc1_cell(tfy, tfh, dry_run=False):
    """KC1 for one seed: the flags reached the runs. FZ-HLVL: ``histswap == cv_logged``, ``v0_in == v0`` at tick 0 and different after (the oracle level reached the network); outside a dry run both runs
    carry ``scene == hold_now`` and FZ-Y carries no histswap key."""
    h0, hl = tfh["tick"] == 0, tfh["tick"] >= 1
    d_in = np.abs(tfh["v0_in"].astype(np.float64) - tfh["v0"].astype(np.float64))
    c = {"FZ-HLVL_histswap": str(tfh["histswap"]) if "histswap" in tfh else None, "FZ-HLVL_scene": str(tfh["scene"]) if "scene" in tfh else None, "FZ-Y_scene": str(tfy["scene"]) if "scene" in tfy else None,
         "FZ-Y_has_histswap_key": "histswap" in tfy, "FZ-HLVL_v0_in_minus_v0_max_abs_tick0": float(d_in[h0].max()), "FZ-HLVL_frac_later_ticks_v0_in_ne_v0_gt_0p05": float((d_in[hl] > 0.05).mean())}
    c["PASS"] = bool(c["FZ-HLVL_histswap"] == "cv_logged" and d_in[h0].max() <= TOL and (d_in[hl] > 0.05).mean() > 0.05
                     and (dry_run or (c["FZ-HLVL_scene"] == "hold_now" and c["FZ-Y_scene"] == "hold_now" and not c["FZ-Y_has_histswap_key"])))
    return c


# ------------------------------------------------------------------------------------------------ heavy layer
LR = None
SR = None
LOG_DIR = f"{PK}/logs"


def heavy_init():
    """HISTSWAP-LF's reader, loaded by file path (pinned sha256): its pure layer (m1_cell, exec_chain_dev, tickload, _dev, y_pins) and SMOOTH-T1's reader (``SR``) are reused, never re-spelled."""
    global LR, SR
    got = sha256_file(LF_READ)
    if got != PIN_LF_READ:
        raise SystemExit(f"REFUSED: {LF_READ} is {got}, not the pinned HISTSWAP-LF reader {PIN_LF_READ}")
    LR = _load("histswap_lf_read", LF_READ)
    SR = LR.SR


def extract(tag, arm, seed, out_dir, arr_dir):
    dst = f"{arr_dir}/{tag}.npz"
    if os.path.exists(dst):
        return dst
    os.makedirs(arr_dir, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    cmd = [PY, "-I", f"{LR.SMT}/code/extract_smooth.py", "--arm", arm, "--seed", str(seed), "--out", dst, "--dump-dir", f"{out_dir}/{tag}/dump_{tag}",
           "--ticks", f"{out_dir}/{tag}/ticks/ticks_{tag}.npz", "--work-dir", WORK, "--floor-guard-m", str(FL_GROSS_M)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    open(f"{LOG_DIR}/extract_{tag}.log", "w", encoding="utf-8", newline="\n").write(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(f"extract {tag} failed rc={r.returncode}: {r.stderr[-400:]}")
    return dst


def run_header(out_dir, tag):
    return json.load(open(f"{out_dir}/{tag}/t1v8f_{tag}.json", encoding="utf-8"))["t1v8f"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_X)
    ap.add_argument("--raw", default=f"{PK}/raw")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true", help="exercise the heavy path on existing runs (see the module docstring); writes only to --raw, never FZ.done")
    ap.add_argument("--pytest-log", default=f"{PK}/logs/pytest_framefreeze.log")
    ap.add_argument("--log-dir", default=f"{PK}/logs", help="where the extractor logs go")
    a = ap.parse_args(argv)
    global LOG_DIR
    LOG_DIR = a.log_dir
    heavy_init()
    # the frozen RECOVER-FT card: the freeze's reading thresholds are frozen there; refuse when the card does not hash to its own FROZEN_SHA.txt
    first = open(f"{FT_PK}/FROZEN_SHA.txt", encoding="utf-8").readline().strip()
    toks = first.split()
    if len(toks) != 3 or toks[1] != "sha256":
        raise SystemExit(f"REFUSED: the first line of {FT_PK}/FROZEN_SHA.txt is not '<file> sha256 <hash>': {first!r}")
    card_got = sha256_file(f"{FT_PK}/{toks[0]}")
    if card_got != toks[2]:
        print(f"REFUSED: {toks[0]} {card_got} != frozen {toks[2]}", file=sys.stderr)
        return 2
    GD = SR.load_gd()
    res = {"what": "FRAMEFREEZE controls (--scene hold_now) on DAG-OACC s0/s1; RECOVER-FT card section 7 / DRAFT_RECOVERY_CARD_v2 section 8.3", "tier": "T1-kin", "evidence_class": "MEASURED (ours; artifacts in MANIFEST.txt)",
           "dry_run": bool(a.dry_run), "recover_ft_card": {"first_line": first, "card_sha256": card_got, "frozen_sha_file_sha256": sha256_file(f"{FT_PK}/FROZEN_SHA.txt")},
           "estimator": f"paired episode-cluster bootstrap, B={a.n_boot}, seed 0 (taniteval.ci draw stream via gapdecomp.decompose); answers another draw of EPISODES only; X and Y share the checkpoint",
           "run_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "controls": {}, "seeds": {}}
    tool = f"{PK}/code/v8f_t1_eval_framefreeze.py"
    res["tool_sha256"] = sha256_file(tool)
    arr_dir = f"{a.raw}/arrays"
    os.makedirs(arr_dir, exist_ok=True)
    pins, pins_ok = LR.y_pins()
    res["controls"]["KY_reused_Y_equals_frozen_pins"] = {"files": pins, "PASS": bool(pins_ok)}
    res["controls"]["KFZ1_KFZ2_unit_tests"] = pytest_verdict(a.pytest_log)

    def arms_for(suf):
        """(FZ-Y tag, dir, FZ-HLVL tag, dir). Dry run: FZ-Y := Y itself (no freeze), FZ-HLVL := HISTSWAP-LF's HLV_*."""
        if a.dry_run:
            return f"SREP_{suf}", OUT_Y, f"HLV_{suf}", OUT_LF
        return f"FZY_{suf}", a.out_dir, f"FZH_{suf}", a.out_dir

    kc1, kc3, kf, k3, k1p = {}, {}, {}, {}, {}
    for k, (suf, arm, seed) in SEEDS.items():
        yt = f"SREP_{suf}"
        fyt, fyd, fht, fhd = arms_for(suf)
        _, dy = SR.load_dump(f"{OUT_Y}/{yt}/dump_{yt}")
        ty = LR.tickload(f"{OUT_Y}/{yt}/ticks/ticks_{yt}.npz")
        _, dfy = SR.load_dump(f"{fyd}/{fyt}/dump_{fyt}")
        tfy = LR.tickload(f"{fyd}/{fyt}/ticks/ticks_{fyt}.npz")
        _, dfh = SR.load_dump(f"{fhd}/{fht}/dump_{fht}")
        tfh = LR.tickload(f"{fhd}/{fht}/ticks/ticks_{fht}.npz")
        k3[k] = kfz3(tfy, ty)
        if not a.dry_run:
            k1p[f"{k}:FZ-Y"] = freeze_receipt(run_header(fyd, fyt), len(tfy["tick"]))
            k1p[f"{k}:FZ-HLVL"] = freeze_receipt(run_header(fhd, fht), len(tfh["tick"]))
        wk = ("clip_index", "ws", "eid", "g", "v0")
        kc1[k] = kc1_cell(tfy, tfh, a.dry_run)
        for nm, t in (("FZ-Y", tfy), ("FZ-HLVL", tfh)):
            ec = LR.exec_chain_dev(t)
            kc3[f"{k}:{nm}"] = {"n_windows": int(len(ec)), "max_dev_m": float(ec.max()) if len(ec) else None, "frac_within_tol": float((ec <= LR.EXEC_TOL_M).mean()) if len(ec) else None, "tol_m": LR.EXEC_TOL_M,
                                "PASS": bool(len(ec) > 0 and (ec <= LR.EXEC_TOL_M).mean() >= 0.999)}
        kf[k] = {}
        for nm, d in (("FZ-Y", dfy), ("FZ-HLVL", dfh)):
            kf[k][nm] = {"windows_equal_Y": bool(all(np.array_equal(d[q], dy[q]) for q in wk)), "floor_max_abs_dev_vs_Y_m": float(np.abs(d["fl"] - dy["fl"]).max())}
            kf[k][nm]["PASS"] = bool(kf[k][nm]["windows_equal_Y"] and kf[k][nm]["floor_max_abs_dev_vs_Y_m"] <= TOL)
        kf[k]["PASS"] = all(v["PASS"] for v in kf[k].values())
    res["controls"]["KFZ1_production_receipt_every_item"] = {"runs": k1p, "PASS": bool(a.dry_run or all(v["PASS"] for v in k1p.values())), "note": "dry run: not applicable" if a.dry_run else "run headers' frame_freeze block"}
    res["controls"]["KFZ3_the_freeze_reached_the_network"] = {"seeds": k3, "tol_m": KFZ3_TOL_M, "PASS": all(v["PASS"] for v in k3.values())}
    res["controls"]["KC1_flags_reached_the_runs"] = {"seeds": kc1, "PASS": all(v["PASS"] for v in kc1.values())}
    res["controls"]["KC3_plan_committed_to_the_executed_state"] = {"runs": kc3, "PASS": all(v["PASS"] for v in kc3.values())}
    res["controls"]["KF_windows_and_P10_floor_equal_Y"] = {"seeds": kf, "PASS": all(v["PASS"] for v in kf.values())}
    # ---- arrays and rows
    kb1, cells, rows_all = {}, {}, {}
    for k, (suf, arm, seed) in SEEDS.items():
        fyt, fyd, fht, fhd = arms_for(suf)
        zy = np.load(extract(f"SREP_{suf}", arm, seed, OUT_Y, f"{LR.KAP_PK}/raw/arrays"), allow_pickle=True)
        zf = np.load(extract(fyt, arm, seed, fyd, arr_dir), allow_pickle=True)
        zh = np.load(extract(fht, arm, seed, fhd, arr_dir), allow_pickle=True)
        labels = tuple(str(q) for q in zy["labels"])
        assert labels == GD.LABELS, (labels, GD.LABELS)
        for nm, z in (("FZ-Y", zf), ("FZ-HLVL", zh)):
            lab_ok = tuple(str(q) for q in z["labels"]) == labels
            win = bool(all(np.array_equal(z[q], zy[q]) for q in ("eid", "v0", "g", "ws", "clip_index")))
            eq = {l: bool(np.array_equal(z[f"b_{i}"], zy[f"b_{i}"], equal_nan=True)) for i, l in enumerate(labels) if not l.startswith("S-POS")}
            kb1[f"{k}:{nm}"] = {"labels_equal": lab_ok, "windows_equal_Y": win, "n_rows_b_equal": sum(eq.values()), "n_rows": len(eq), "failed": [q for q, v in eq.items() if not v], "PASS": bool(lab_ok and win and all(eq.values()))}
        cls, _ = GD.classify(zf["v0"], zf["g"])
        rows = {nm: {} for nm in ("FZY_minus_Y", "FZY_minus_P10", "Y_minus_P10", "FZH_minus_FZY", "FZH_minus_P10")}
        full_m1 = {}
        for i, lab in enumerate(labels):
            for nm, (u, v) in (("FZY_minus_Y", (zf[f"a_{i}"], zy[f"a_{i}"])), ("FZY_minus_P10", (zf[f"a_{i}"], zf[f"b_{i}"])), ("Y_minus_P10", (zy[f"a_{i}"], zy[f"b_{i}"])),
                               ("FZH_minus_FZY", (zh[f"a_{i}"], zf[f"a_{i}"])), ("FZH_minus_P10", (zh[f"a_{i}"], zh[f"b_{i}"]))):
                r = GD.decompose(u, v, zf["eid"], cls, GD.POOL[lab], a.n_boot)
                rows[nm][lab] = {q: r[q] for q in ("delta", "lo", "hi", "separated")}
                if lab == M1 and nm in ("FZY_minus_Y", "FZH_minus_FZY"):
                    full_m1[nm] = r
        cells[k] = LR.m1_cell(rows["FZH_minus_FZY"][M1], rows["FZY_minus_P10"][M1])
        rows_all[k] = rows
        res["seeds"][k] = {"arm_seed": suf, "tags": {"Y": f"SREP_{suf}", "FZ-Y": fyt, "FZ-HLVL": fht}, "rows": rows, "m1_cell_FZHLVL_vs_FZY": cells[k], "M1_per_class_split": full_m1,
                           "read_a_FZY_vs_P10": SR.read_a(GD, rows["FZY_minus_P10"]), "read_xy_FZHLVL_minus_FZY": SR.read_xy(GD, rows["FZH_minus_FZY"]), "read_xy_FZY_minus_Y": SR.read_xy(GD, rows["FZY_minus_Y"])}
        print(k, "freeze effect M1 FZ-Y - Y %+.4f [%+.4f, %+.4f] sep %s | KFZ4 cell M1 FZH-FZY %+.4f sep %s share closed %s | ADE6 FZ-Y - Y %+.4f" % (
            rows["FZY_minus_Y"][M1]["delta"], rows["FZY_minus_Y"][M1]["lo"], rows["FZY_minus_Y"][M1]["hi"], rows["FZY_minus_Y"][M1]["separated"], rows["FZH_minus_FZY"][M1]["delta"],
            cells[k]["better"], cells[k]["share_closed"], rows["FZY_minus_Y"]["ADE6"]["delta"]), flush=True)
    res["controls"]["KB1_P10_side_equals_Y"] = {"runs": kb1, "PASS": all(v["PASS"] for v in kb1.values())}
    res["kfz4"] = kfz4(cells)
    res["controls"]["KFZ4_FZ-HLVL_keeps_its_closure_against_FZ-Y"] = dict(res["kfz4"])
    gating = [q for q in res["controls"] if not q.startswith("KFZ4")]
    res["controls_all_pass_except_kfz4"] = bool(all(res["controls"][q]["PASS"] for q in gating))
    res["recover_ft_contrast_usable"] = bool(res["controls_all_pass_except_kfz4"] and res["kfz4"]["PASS"])
    res["failed_controls"] = [q for q, v in res["controls"].items() if not v["PASS"]]
    res["freeze_effect_FZY_minus_Y"] = {k: {lab: v for lab, v in res["seeds"][k]["rows"]["FZY_minus_Y"].items()} for k in SEEDS}
    res["note"] = ("A failed KFZ3 / KFZ4 is a FINDING: the RECOVER-FT read REFUSES the frame-freeze contrast. Disclosed limits (card 8.5): the current frame still depicts the human's position at tick j and may carry blur; "
                   "four identical frames are off the training distribution (FZ-Y - Y measures the freeze's own effect); T2 re-perception is not provisioned. S-POS rows are outside the reading (UNVERIFIED).")
    out = f"{a.raw}/FZ_CONTROLS.json"
    json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    print("CONTROLS", {q: v["PASS"] for q, v in res["controls"].items()})
    print("KFZ4", {q: res["kfz4"][q] for q in ("PASS", "n_seeds_separated_better", "mean_share_closed", "each_seed_share_ge_bar")})
    print("recover_ft_contrast_usable", res["recover_ft_contrast_usable"], "failed", res["failed_controls"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
