#!/usr/bin/env python3
"""trk_verify.py -- G0-TRK, G0-TRK-DET, C-TRK-LIVE for PREREG_VDAC_P6_A2 sec.3. Instrument identities ONLY: no pick, no PDMS, no gain, no veto.

Run in the NAVSIM venv (needs navsim for the corner rebuild):
  PYTHONPATH=F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896 \
  C:/Users/Admin/navsim-crun/venv/Scripts/python.exe trk_verify.py --full DIR --det DIR2 --out OUTDIR [--partial]
--full : dir with trk_*of*_part*.npz from trk_dump.py (all shards)    --det : independent dump of the first 100 sorted tokens
        (trk_dump.py --limit 100 --shard 0/1 --out DIR2)             --partial : smoke mode, judge only the tokens present
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trk_dump as T  # noqa: E402

NDET = 100
DISP_THR = 0.01  # metres


def load_parts(d):
    parts = sorted(Path(d).glob("trk_*of*_part*.npz"))
    if not parts:
        sys.exit(f"FATAL no parts in {d}")
    acc = {k: [] for k in ("idx", "token", "origin", "vp", "raw_xyh", "trk_xyh", "d0", "d1", "corners_chk_global")}
    for p in parts:
        z = np.load(p)
        for k in acc:
            acc[k].append(z[k])
    return {k: np.concatenate(v) for k, v in acc.items()}, [p.name for p in parts]


def corners_ego(xyh, origin, vp):
    return T.rebuild_corners(xyh, vp, origin=origin)  # [64,41,4,2] float64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tab = np.load(T.TABLE, allow_pickle=True)
    table_sha = T.sha_file(T.TABLE)
    assert table_sha.startswith(T.TABLE_SHA_PREFIX), f"table sha {table_sha}"
    tok = [str(t) for t in tab["token"]]
    N = len(tok)
    sub = tab["sub"]
    D, parts = load_parts(a.full)
    idx = D["idx"]
    res = {"table_sha256": table_sha, "n_parts": len(parts), "partial_mode": bool(a.partial)}

    # ---- coverage
    assert len(set(idx.tolist())) == len(idx), "duplicate idx across parts"
    covered = len(idx)
    res["tokens_covered"], res["tokens_expected"] = int(covered), N
    if not a.partial and covered != N:
        res["verdict"] = "INVALID: coverage"
        (out / "trk_verify.json").write_text(json.dumps(res, indent=1))
        sys.exit(f"FATAL coverage {covered}/{N}")
    assert [tok[i] for i in idx] == [str(t) for t in D["token"]], "token/idx mismatch"
    order = np.argsort(idx)  # table order
    idx = idx[order]
    for k in ("token", "origin", "vp", "raw_xyh", "trk_xyh", "d0", "d1"):
        D[k] = D[k][order]

    # ---- G0-TRK: d0 == table DAC on every proposal
    tdac = sub[idx][:, :, T.DAC_COL]
    eq = D["d0"] == tdac
    mism = np.argwhere(~eq)
    res["G0_TRK"] = {
        "n_proposals": int(eq.size), "n_equal": int(eq.sum()), "n_mismatch": int((~eq).sum()),
        "expected_n_if_full": N * 64,
        "mismatches": [{"token": str(D["token"][i]), "k": int(k), "d0": float(D["d0"][i, k]), "table_dac": float(tdac[i, k])} for i, k in mism[:2000]],
        "pass": bool(eq.all() and (eq.size == N * 64 or a.partial)),
    }

    # ---- corner rebuild vs stored NAVSIM corners (global frame), subset
    chk_max, nchk = 0.0, 0
    cg = np.load(sorted(Path(a.full).glob("trk_*of*_part*.npz"))[0])["corners_chk_global"]
    z0 = np.load(sorted(Path(a.full).glob("trk_*of*_part*.npz"))[0])
    for j in range(cg.shape[0]):
        rb = T.rebuild_corners(z0["trk_xyh"][j], z0["vp"][j])  # [64,41,4,2]
        # stored corners hold 5 points incl. centre; drop centre via BBCoordsIndex
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex
        keep = [i for i in range(len(BBCoordsIndex)) if i != int(BBCoordsIndex.CENTER)]
        chk_max = max(chk_max, float(np.abs(rb - cg[j][..., keep, :]).max()))
        nchk += 1
    res["corner_rebuild_check"] = {"n_tokens": nchk, "max_abs_diff_m": chk_max, "pass": chk_max == 0.0}

    # ---- C-TRK-LIVE + full tracked-corner hash (ego frame, float64, table order)
    h_c = hashlib.sha256()
    n_disp = 0
    n_d_diff = int((D["d0"] != D["d1"]).sum())
    maxdisp_all = []
    for i in range(len(idx)):
        ct = corners_ego(D["trk_xyh"][i], D["origin"][i], D["vp"][i])
        cr = corners_ego(D["raw_xyh"][i], D["origin"][i], D["vp"][i])
        h_c.update(np.ascontiguousarray(ct).tobytes())
        md = np.linalg.norm(ct - cr, axis=-1).max(axis=(1, 2))  # [64] max over 41 steps x 4 corners
        maxdisp_all.append(md)
    md = np.stack(maxdisp_all)
    res["tracked_corner_array_sha256"] = {"sha256": h_c.hexdigest(), "dtype": "float64", "shape": [int(len(idx)), 64, 41, 4, 2], "frame": "t0 rear-axle (ego), table token order"}
    res["tracked_xyh_array_sha256"] = hashlib.sha256(np.ascontiguousarray(D["trk_xyh"]).tobytes()).hexdigest()
    res["C_TRK_LIVE"] = {
        "n_proposals": int(md.size), "disp_threshold_m": DISP_THR,
        "share_max_point_disp_gt_thr": float((md > DISP_THR).mean()), "n_gt_thr": int((md > DISP_THR).sum()),
        "max_disp_m": float(md.max()), "median_disp_m": float(np.median(md)),
        "share_d0_ne_d1": float(n_d_diff / D["d0"].size), "n_d0_ne_d1": n_d_diff,
        "note": "d1 = NAVSIM DAC rule on RAW states, d0 on TRACKED states, same polygons; the veto-verdict share under the TRUE field is this d0!=d1 share. Pose-only (AUXVETO constants) comparison is the reader's.",
        "pass": bool((md > DISP_THR).sum() > 0),
    }

    # ---- G0-TRK-DET: independent re-dump of the first 100 sorted tokens, bit for bit
    E, _ = load_parts(a.det)
    first = sorted(range(N), key=lambda i: tok[i])[:NDET]
    pos_full = {int(i): r for r, i in enumerate(idx)}
    pos_det = {int(i): r for r, i in enumerate(E["idx"])}
    need = [i for i in first if i in pos_full] if a.partial else first
    miss = [i for i in need if i not in pos_det or i not in pos_full]
    bit_xyh = bit_cor = bit_d0 = True
    n_cmp = 0
    h_det, h_ref = hashlib.sha256(), hashlib.sha256()
    for i in need:
        if i in miss:
            continue
        r, s = pos_full[i], pos_det[i]
        n_cmp += 1
        bit_xyh &= np.array_equal(D["trk_xyh"][r], E["trk_xyh"][s]) and np.array_equal(D["raw_xyh"][r], E["raw_xyh"][s])
        bit_d0 &= np.array_equal(D["d0"][r], E["d0"][s])
        c1 = corners_ego(D["trk_xyh"][r], D["origin"][r], D["vp"][r])
        c2 = corners_ego(E["trk_xyh"][s], E["origin"][s], E["vp"][s])
        h_ref.update(c1.tobytes())
        h_det.update(c2.tobytes())
    res["G0_TRK_DET"] = {
        "n_tokens_required": len(need), "n_compared": n_cmp, "missing_in_det_or_full": [tok[i] for i in miss],
        "states_bit_identical": bool(bit_xyh), "d0_bit_identical": bool(bit_d0),
        "corner_sha256_full_subset": h_ref.hexdigest(), "corner_sha256_det": h_det.hexdigest(),
        "pass": bool(bit_xyh and bit_d0 and not miss and h_ref.hexdigest() == h_det.hexdigest() and n_cmp == len(need) and n_cmp > 0),
    }
    res["verdict"] = "PASS" if all(res[k]["pass"] for k in ("G0_TRK", "G0_TRK_DET", "C_TRK_LIVE", "corner_rebuild_check")) else "FAIL (INVALID until explained)"
    if a.partial:
        res["verdict"] += " [PARTIAL/SMOKE MODE: not the real gate]"
    (out / "trk_verify.json").write_text(json.dumps(res, indent=1))
    g, d, c = res["G0_TRK"], res["G0_TRK_DET"], res["C_TRK_LIVE"]
    md_txt = f"""# trk_verify (G0-TRK instrument; no selection, no PDMS)
verdict: **{res['verdict']}**
- table sha256 {table_sha}; tokens {covered}/{N}; parts {len(parts)}
- **G0-TRK** d0 == table DAC: {g['n_equal']}/{g['n_proposals']} (expected {g['expected_n_if_full']}), mismatches {g['n_mismatch']}, pass={g['pass']}
- corner rebuild vs NAVSIM stored corners: {nchk} tokens, max abs diff {chk_max} m, pass={res['corner_rebuild_check']['pass']}
- **G0-TRK-DET** first {NDET} sorted tokens, compared {d['n_compared']}/{d['n_tokens_required']}: states bit-identical={d['states_bit_identical']}, d0={d['d0_bit_identical']}, corner sha equal={d['corner_sha256_full_subset'] == d['corner_sha256_det']}, pass={d['pass']}
- **C-TRK-LIVE** share of proposals with max point displacement tracked vs raw > {DISP_THR} m: {c['share_max_point_disp_gt_thr']:.4f} ({c['n_gt_thr']}/{c['n_proposals']}); max {c['max_disp_m']:.3f} m; share d0 != d1: {c['share_d0_ne_d1']:.4f} ({c['n_d0_ne_d1']}); pass={c['pass']}
- tracked-corner array sha256 (float64 [{len(idx)},64,41,4,2], ego frame, table order): `{res['tracked_corner_array_sha256']['sha256']}`
- tracked xyh array sha256: `{res['tracked_xyh_array_sha256']}`
"""
    (out / "trk_verify.md").write_text(md_txt, encoding="utf-8")
    print(md_txt)


if __name__ == "__main__":
    main()
