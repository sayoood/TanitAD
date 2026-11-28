"""Nav-text join for the STRATEGIC cells: (clip, cache step) -> a row of navtok_table.npz. numpy only (the offset check also reads native_tracks.pt through torch).

Table: F:/TanitAD-scratch/datascale/navtok/navtok_table.npz, sha256 prefix 2836369e (MEASURED, navtok_table_report.json). One row per v8f-mini EVAL cache step: row = eval_off[clip] + step.
Flags: 0 ok | 1 step before the clip's first nav row | 2 clip has no nav rows | 3 step has no label-clock time. A row with flag != 0, a clip not in the table and a step outside the clip are NaN in every
STRATEGIC cell, never 0: ``attach`` returns text_ok False there and the cells read text_ok.

NOW_OFFSET: the table step that belongs to a window's NOW. The audit planned ``now + 3`` and marked it UNVERIFIED. ``verify_offset`` compares the speed at the join row's own time (label clock
now_s of join_eval141.npz, native_tracks speed column) with the window's own v0 over the 994 ACTSENS windows; the result (STRATEGIC_TEXT_OFFSET_CHECK.json) selects offset 0, so NOW_OFFSET = 0.
"""
from __future__ import annotations

import hashlib

import numpy as np

import strategic_cells as S

TABLE_SHA_PREFIX = "2836369e"
NOW_OFFSET = 0
T0_S = -0.195                      # label-clock to track-clock shift of the table (navtok build_table.py T0_US, ESTIMATED -195 ms)
RAW_DT_S = 0.1                     # one raw track row


def load_table(path, sha_prefix=TABLE_SHA_PREFIX):
    """Load the table, refusing a file whose sha256 does not start with ``sha_prefix`` (pass None to skip, for synthetic tables in tests)."""
    if sha_prefix is not None:
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if not h.startswith(sha_prefix):
            raise ValueError(f"navtok table sha256 {h[:12]} does not start with {sha_prefix}: {path}")
    z = np.load(path)
    tab = {k: z[k] for k in z.files}
    tab["clip_index"] = {str(c): i for i, c in enumerate(tab["eval_clip"].tolist())}
    tab["side"], tab["theta"] = S.side_table(list(tab["vocab"]))
    return tab


def rows_for(tab, eids, steps):
    """Row index into the eval arrays for each (clip, step); -1 where the clip is absent or the step is outside the clip."""
    eids = np.asarray(eids).astype(str)
    steps = np.asarray(steps).astype(np.int64)
    row = np.full(len(eids), -1, np.int64)
    for n, (e, j) in enumerate(zip(eids, steps)):
        i = tab["clip_index"].get(str(e))
        if i is not None and 0 <= j < int(tab["eval_n"][i]):
            row[n] = int(tab["eval_off"][i]) + int(j)
    return row


def attach(tab, eids, nows, offset=NOW_OFFSET):
    """The text inputs of S.cells for every window: dict(side, theta_deg, d_text, text_ok, row, flag). text_ok is False (and d_text NaN, side 0) on every row with flag != 0 and on an unmapped window."""
    row = rows_for(tab, eids, np.asarray(nows, np.int64) + int(offset))
    mapped = row >= 0
    r = np.where(mapped, row, 0)
    flag = np.where(mapped, tab["eval_flag"][r], 255).astype(np.int64)
    ok = flag == 0
    cls = np.where(ok, tab["eval_cls_id"][r], 0)
    d = np.where(ok, tab["eval_dist_m"][r].astype(np.float64), np.nan)
    return dict(side=np.where(ok, tab["side"][cls], 0).astype(np.int8), theta_deg=np.where(ok, tab["theta"][cls], np.nan), d_text=d, text_ok=ok, row=row, flag=flag)


def verify_offset(tab, join_path, tracks_path, eids, nows, v0, offsets=range(0, 7), shift_s=T0_S):
    """For each candidate table-step offset d: the track speed at the label time of join row (clip, now + d), against the window's own v0.
    Agreement per d: n compared, median and p90 |dv| (m/s), fraction within 0.05 m/s. Beside it, at d = 0, a scan of the clock shift (s) to show where the fit peaks.
    The check reads the join's label clock (join_eval141.npz now_s) and the tracks; it does not read the window's own step index, so it is not the identity v0_of(eid, now) = tracks[2 now]."""
    import torch
    J = np.load(join_path, allow_pickle=True)
    TR = torch.load(tracks_path, weights_only=False)["tracks"]
    eids = np.asarray(eids).astype(str)
    nows = np.asarray(nows, np.int64)
    v0 = np.asarray(v0, np.float64)
    cols = {}

    def v_join(d, shift):
        out = np.full(len(eids), np.nan)
        rows = rows_for(tab, eids, nows + d)
        for n, (e, r) in enumerate(zip(eids, rows)):
            if r < 0:
                continue
            t = J["now_s"][r]
            if not np.isfinite(t):
                continue
            if e not in cols:
                x = np.asarray(TR[e], np.float64)[:, 3]
                k = np.arange(len(x))
                m = np.isfinite(x)
                cols[e] = (k[m], x[m])
            k, x = cols[e]
            f = (t + shift) / RAW_DT_S
            if len(k) > 2 and k[0] <= f <= k[-1]:
                out[n] = np.interp(f, k, x)
        return out

    def agree(v):
        ok = np.isfinite(v) & np.isfinite(v0)
        dv = np.abs(v[ok] - v0[ok])
        if not ok.any():
            return dict(n=0)
        return dict(n=int(ok.sum()), median_abs_dv=float(np.median(dv)), p90_abs_dv=float(np.percentile(dv, 90)), frac_within_0p05=float((dv < 0.05).mean()))

    res = {"clock_shift_s": shift_s, "by_offset": {int(d): agree(v_join(d, shift_s)) for d in offsets}}
    res["shift_scan_at_offset0"] = {f"{s:+.3f}": agree(v_join(0, s)) for s in np.arange(-0.40, 0.1001, 0.05)}
    best = max(res["by_offset"], key=lambda d: res["by_offset"][d].get("frac_within_0p05", -1))
    res["best_offset"] = int(best)
    return res
