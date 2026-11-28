"""mk_window_strata.py -- PREREG_V8F_KIND Amendment 1: the per-WINDOW stratum labels of the eval-141 grid (CPU only, read-only on every input).

  python mk_window_strata.py --release <v9_labels_eval139.npz> --out window_strata_eval141.npz

WHERE THE LABEL COLUMNS REACH THE EVAL WINDOWS (measured here, not assumed):
  * `curve` and `turn_is_turn` are columns of the v9 label RELEASE (`row__curve`, `row__turn_is_turn`; stack/scripts/build_v9_labels.py:828-848 writes them). They are NOT columns of
    the shipped eval join (`raw/join_eval141.npz` carries the 26 constraint cells, the nav block and the classes, not these two) and NOT of any eval npz.
  * The join takes a release row for cache step j of a clip by `rows = r0 + clip(2j - k0, 0, nrows - 1)`, inside = (k0 <= 2j < k0 + nrows)  (v8f_mini_data.build_label_join).
    This script applies the SAME mapping to the two columns, and PROVES it is the join's own mapping by re-deriving three columns the join does carry (`lat_cls`, the 26 `cons` cells,
    `has`) from the same release rows and requiring them equal to the shipped join on all rows.
  * A window (clip c, tick t) is scored against the label row the TACTICAL layer READS: the T-tick row  rowT = off[c] + t - a_T,  a_T = min(i % 3, max(t - 3, 0), 2) with i the global
    window index (v8f_mini_eval.py:290-297; the audit raw/2026-10-25-t-attrib/s1_build_table.py). The banked audit's table.npz holds the same (ci, tt, a_t, rowT, has): required equal.

STRATA (PREREG_V8F_KIND.md Amendment 1, verbatim):  JUNCTION = a dominant turn segment in the label band with turn_is_turn = 1;  CURVE = curve = 1;  NO-TURN = every other window
(including a window with no v9 row: the oracle falls back to T's own code there, its gain is exactly 0; they are counted separately in the reader's output).
The builder writes curve = int(big and not is_junc), so curve = 1 and turn_is_turn = 1 never co-occur by construction; the count of co-occurrences is checked and written (must be 0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/")
PKG = HERE.rsplit("/raw/", 1)[0]
REPO = PKG.split("/TanitAD Research Lab/")[0]
RELEASE_MD5 = "6b5c7f207cffc3b7eb3cd527fd433599"          # v9 eval139 (v8f_mini_data.ACCEPTED_EVAL_MD5)
STRATA = ("JUNCTION", "CURVE", "NO-TURN")                 # stratum code 0 / 1 / 2 in the output file
CODE = {n: i for i, n in enumerate(STRATA)}
WINDOW_START, MAX_AGE_T = 3, 2                            # hist 4 -> window_start = hist - 1; cfg.max_age_t default 2


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stratum_of(curve, turn_is_turn, has):
    """0 JUNCTION / 1 CURVE / 2 NO-TURN per window. JUNCTION wins a tie (never happens: the count of co-occurrences is written)."""
    s = np.full(curve.shape, CODE["NO-TURN"], np.int8)
    s[(curve == 1) & has] = CODE["CURVE"]
    s[(turn_is_turn == 1) & has] = CODE["JUNCTION"]
    return s


def window_rows(grid, J):
    ci, tt = [], []
    for k, c in enumerate(grid["order"]):
        for t in grid["windows"][c]:
            ci.append(k)
            tt.append(int(t))
    ci, tt = np.array(ci, np.int64), np.array(tt, np.int64)
    i = np.arange(len(ci))
    a_t = np.minimum(np.minimum(i % 3, np.maximum(tt - WINDOW_START, 0)), MAX_AGE_T)
    return ci, tt, a_t, J["off"][ci] + tt - a_t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, help="v9_labels_eval139.npz (md5 6b5c7f20...)")
    ap.add_argument("--join", default=f"{PKG}/raw/join_eval141.npz")
    ap.add_argument("--grid", default=f"{PKG}/raw/eval_grid_windows.json")
    ap.add_argument("--audit-table", default=f"{PKG}/raw/2026-10-25-t-attrib/table.npz")
    ap.add_argument("--out", default=f"{HERE}/window_strata_eval141.npz")
    a = ap.parse_args(argv)
    sys.path.insert(0, f"{REPO}/stack")
    from tanitad.data import v8f_mini_data as D
    from tanitad.data.v9_labels import load_v9_release
    rel = load_v9_release(a.release, expect_md5=RELEASE_MD5)
    J = D.load_join(a.join)
    grid = json.load(open(a.grid, encoding="utf-8"))
    assert J["clip"].tolist() == list(grid["order"]), "join clip order != grid order"
    R = rel.rows
    N = int(J["has"].shape[0])
    curve_j, tit_j, has_j = np.zeros(N, np.int8), np.zeros(N, np.int8), np.zeros(N, bool)
    lat_j = np.full(N, D.IGNORE, np.int16)
    cons_j = np.full((N, len(D.CONS_FIELDS)), np.nan, np.float32)
    for c, cid in enumerate(J["clip"]):
        meta = rel.clips.get(D.stable_episode_id(str(cid)))
        if meta is None:
            continue
        r0, nrows, k0 = meta
        nf = int(J["n"][c])
        o = int(J["off"][c])
        k = 2 * np.arange(nf)
        inside = (k >= k0) & (k < k0 + nrows)
        rows = r0 + np.clip(k - k0, 0, nrows - 1)
        sl = slice(o, o + nf)
        has_j[sl] = inside
        curve_j[sl] = np.where(inside, np.asarray(R["curve"])[rows], 0)
        tit_j[sl] = np.where(inside, np.asarray(R["turn_is_turn"])[rows], 0)
        lat_j[sl] = np.where(inside, np.asarray(R["lat_cls_a"])[rows], D.IGNORE)
        for q, f in enumerate(D.CONS_FIELDS):
            cons_j[sl, q] = np.where(inside, np.asarray(R[f], np.float64)[rows], np.nan)
    # (1) the mapping IS the shipped join's: three columns the join carries, re-derived from the same rows, equal on every row
    ok_has = bool(np.array_equal(has_j, J["has"]))
    ok_lat = bool(np.array_equal(lat_j, J["lat_cls"]))
    ok_cons = bool(np.array_equal(np.isnan(cons_j), np.isnan(J["cons"])) and np.array_equal(np.nan_to_num(cons_j), np.nan_to_num(J["cons"])))
    print(f"join rows {N}; has {int(has_j.sum())}; re-derived vs the shipped join: has equal {ok_has}, lat_cls equal {ok_lat}, 26 cons cells equal {ok_cons}")
    if not (ok_has and ok_lat and ok_cons):
        print("MAPPING MISMATCH: the release rows do not reproduce the shipped join", file=sys.stderr)
        return 2
    # (2) the windows and their T-tick rows
    ci, tt, a_t, rowT = window_rows(grid, J)
    has = J["has"][rowT]
    if os.path.exists(a.audit_table):
        T = np.load(a.audit_table, allow_pickle=True)
        same = all(np.array_equal(x, T[k]) for x, k in ((ci, "ci"), (tt, "tt"), (a_t, "a_t"), (rowT, "rowT"), (has, "has")))
        print(f"windows {len(ci)} over {len(np.unique(ci))} episodes; (ci, tt, a_t, rowT, has) equal to the banked audit table.npz: {same}")
        if not same:
            print("WINDOW PAIRING MISMATCH vs the banked audit table", file=sys.stderr)
            return 2
    curve, tit = curve_j[rowT], tit_j[rowT]
    both = int(((curve == 1) & (tit == 1)).sum())
    s = stratum_of(curve, tit, has)
    n_no_row = int((~has).sum())
    print(f"curve=1 & turn_is_turn=1 on the same window: {both} (must be 0 by construction)")
    for nm in STRATA:
        m = s == CODE[nm]
        print(f"  {nm:9s} windows {int(m.sum()):5d}  episodes {len(np.unique(ci[m])):3d}")
    print(f"  no v9 row (inside NO-TURN) {n_no_row}")
    prov = {"schema": "v8f-kind-window-strata-v1", "strata": list(STRATA), "codes": CODE, "release_md5": RELEASE_MD5, "release_file": os.path.basename(a.release),
            "join_sha256": sha256(a.join), "grid_sha256": sha256(a.grid), "rowT": f"off[ci] + t - a_T, a_T = min(i % 3, max(t - {WINDOW_START}, 0), {MAX_AGE_T})",
            "definition": "JUNCTION: turn_is_turn == 1 at rowT; CURVE: curve == 1 at rowT; NO-TURN: every other window (incl. no v9 row)",
            "n_windows": int(len(ci)), "n_both_curve_and_turn": both, "n_no_v9_row": n_no_row,
            "counts": {nm: int((s == CODE[nm]).sum()) for nm in STRATA}, "episodes": {nm: int(len(np.unique(ci[s == CODE[nm]]))) for nm in STRATA}}
    np.savez_compressed(a.out, stratum=s, curve=curve, turn_is_turn=tit, has=has, eid=ci, t=tt, a_t=a_t, rowT=rowT, provenance=np.array(json.dumps(prov)))
    print(f"wrote {a.out}  stratum sha256 {hashlib.sha256(s.tobytes()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
