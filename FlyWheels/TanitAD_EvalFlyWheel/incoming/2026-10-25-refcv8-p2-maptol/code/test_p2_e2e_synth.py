"""test_p2_e2e_synth.py -- END-TO-END plumbing test of the P2 pass' per-window/banking code and of the analysis, on SYNTHETIC logits.

No model, no GPU, no dataset.  Two synthetic "arms" (A = good localisation with a lateral jitter, B = noisier) over 8 episodes x 5 windows on a
[1000, 200] grid: `window_job` -> `bank_rows` -> `p2_tol_read.py` (with --no-bank; C6 bootstrap-stream control ON, n_boot 40) -> `P2_TOL.json` + `RESULT.md`.
Known-answer checks: the verdict rule fires as written on constructed cases; pooled counts equal a direct recount; GT census equals the construction.

    PYTHONPATH=<tree>/taniteval:<tree>/stack python -X utf8 test_p2_e2e_synth.py <tmpdir>
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2_tol_jobs as J  # noqa: E402
import p2_tol_lib as T  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if (detail and not cond) else ""))
    if not cond:
        FAILS.append(name)


def make_gt(rng, H=1000, W=200):
    g = np.full((H, W), 1, np.uint8)                       # drivable everywhere
    g[:, :30] = 7
    g[:, 170:] = 7                                          # sidewalks
    for x0 in (60, 100, 140):                               # three 2-cell-wide lane lines, long
        g[:, x0:x0 + 2] = 2
    for r0 in range(20, 900, 150):                          # crosswalk stripes: 3 rows x 40 cols
        g[r0:r0 + 3, 70:110] = 3
    g[:5] = 255
    return g


def logits_from_codes(codes, rng, strength=6.0):
    H, W = codes.shape
    z = rng.normal(0, 0.3, (8, H, W)).astype(np.float32)
    np.put_along_axis(z, codes[None].astype(np.int64), strength, axis=0)
    return z


def predict(gt, rng, jitter, drop_lane):
    p = np.where(gt == 255, 0, gt).astype(np.uint8)
    if jitter:
        p = np.roll(p, jitter, axis=1)                      # lateral shift of EVERYTHING by `jitter` cells
    if drop_lane:
        p[p == 2] = 1                                       # lane absent
    return p


def run_arm(arm, eids, gts, jitter, drop_lane, out_dir, rng):
    rows = []
    for i, (e, gt) in enumerate(zip(eids, gts)):
        pred = predict(gt, rng, jitter, drop_lane)
        z = logits_from_codes(pred, rng)
        rows.append(J.window_job(e, i, gt, z, "prior_corrected", np.ones(8), i < 3))
    tpath, xpath = J.bank_rows(rows, arm, out_dir, gts[0].shape[0])
    ctl = [r["ctl"] for r in rows if r["ctl"] is not None]
    rec = {"arm": arm, "ckpt_md5": "x" * 32, "step": 1, "windows_json_md5": "y" * 32, "decision_rule": "prior_corrected",
           "controls": {"law_frame": {"pass": True},
                        "C4_cheb_k1_vs_EDT_0p2m": {"windows": len(rows), "mismatching_cells": int(sum(r["edt_bad"] for r in rows))},
                        "C0_counts_vs_MH_window_stats": {"mismatching_arrays": int(sum(r["base_bad"] for r in rows))},
                        "C1p_gt_as_pred": {"windows": len(ctl), "all_pass": all(c["gt_as_pred"] for c in ctl)},
                        "C2p_all_one_class": {"windows": len(ctl), "all_pass": all(c["one_class"] for c in ctl)}}}
    json.dump(rec, open(Path(out_dir) / f"p2_{arm}.json", "w"), indent=1)
    return rows


def main():
    tmp = Path(sys.argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    eids = [f"ep{e:02d}" for e in range(8) for _ in range(5)]
    gts = [make_gt(rng) for _ in eids]
    (tmp / "a").mkdir(exist_ok=True)
    (tmp / "b").mkdir(exist_ok=True)
    run_arm("armA", eids, gts, jitter=3, drop_lane=False, out_dir=tmp / "a", rng=rng)
    run_arm("armB", eids, gts, jitter=0, drop_lane=True, out_dir=tmp / "b", rng=rng)
    # --- known-answer checks on the banked counts ----------------------------------------------------------------- #
    za = np.load(tmp / "a" / "p2_armA_tol.npz")
    zb = np.load(tmp / "b" / "p2_armB_tol.npz")
    n = len(eids)
    # construction arithmetic (NOT read back from the code): rows 0-4 unseen -> band 0 has 195 seen rows, bands 1-4 have 200;
    # lines at cols 60 and 140 are 2 wide and never overlapped; the line at col 100 loses the rows of every 3-row crosswalk stripe:
    # stripes start at rows 20,170,320,470,620,770 -> per band [2,1,1,2,0] stripes
    seen_rows_band = [195, 200, 200, 200, 200]
    stripes_band = [2, 1, 1, 2, 0]
    check("synthetic lane GT cells per band == construction arithmetic",
          all(int(za["gt"][:, 2, b].sum()) == (4 * seen_rows_band[b] + 2 * (seen_rows_band[b] - 3 * stripes_band[b])) * n for b in range(5)))
    check("synthetic crosswalk GT cells per band == 120 x stripes",
          all(int(za["gt"][:, 3, b].sum()) == 120 * stripes_band[b] * n for b in range(5)))
    # arm A shifts everything by 3 cells: a 2-cell line shifted by 3 has no exact overlap (d=3 >= w=2) -> inter 0; k=2: partial; k=5: full
    check("jittered lane: exact inter = 0 (shift 3 >= width 2)", int(za["inter"][:, 2, 0].sum()) == 0)
    check("jittered lane: tpp5 == pred (every predicted lane cell within 5 cells of GT)", int(za["tpp5"][:, 2, 0].sum()) == int(za["pred"][:, 2, 0].sum()))
    check("jittered lane: tpp1 == 0 (shift 3 > 1 cell: nearest GT cell is 2 away)", int(za["tpp1"][:, 2, 0].sum()) == 0)
    check("arm B lane absent: pred 0 and tpg10 == 0", int(zb["pred"][:, 2, :].sum()) == 0 and int(zb["tpg10"][:, 2, :].sum()) == 0)
    check("GT identical across arms (the pairing identity)", np.array_equal(za["gt"], zb["gt"]) and all(np.array_equal(za[f"census_{k}"], zb[f"census_{k}"]) for k in ("thick", "lat", "lon")))
    th = za["census_thick"]                                        # [5 classes, 5 bands, 65]
    check("census: lane thickness is 2 cells for every lane cell", int(th[0, :, 2].sum()) == int(th[0].sum()) > 0)
    check("census: crosswalk stripe thickness is 3 cells", int(th[1, :, 3].sum()) == int(th[1].sum()) > 0)
    # --- the analysis, end to end ------------------------------------------------------------------------------- #
    out = tmp / "read"
    cmd = [sys.executable, "-X", "utf8", str(HERE / "p2_tol_read.py"), "--a-dir", str(tmp / "a"), "--a-arm", "armA", "--b-dir", str(tmp / "b"), "--b-arm", "armB",
           "--out", str(out), "--n-boot", "40", "--no-bank", "--min-gt-cells", "1000", "--min-gt-episodes", "5", "--prereg", str(HERE.parent / "PREREG_P2.md")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-600:], r.stderr[-1500:])
    check("p2_tol_read.py exits 0", r.returncode == 0)
    if r.returncode == 0:
        res = json.loads((out / "P2_TOL.json").read_text(encoding="utf-8"))
        v = res["verdicts"]["lane / road line|0-20m"]
        check("verdict armA lane 0-20: shift 3 -> R_5 = 1 (all GT within 5), P_5 = 1, IoU_5/IoU_0 huge -> MISPLACED", v["armA"]["verdict"] == "MISPLACED", str(v["armA"]))
        check("verdict armB lane 0-20: lane absent -> ABSENT", v["armB"]["verdict"] == "ABSENT", str(v["armB"]))
        c6 = res["controls"].get("C6_bootstrap_stream", {})
        check("C6 bootstrap-stream control equal on every cell", bool(c6.get("pass")), json.dumps(c6)[:400])
        check("RESULT.md written", (out / "RESULT.md").exists() and (out / "RESULT.md").stat().st_size > 2000)
    print(f"ZZP2E2E fails={len(FAILS)} ZZ")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
