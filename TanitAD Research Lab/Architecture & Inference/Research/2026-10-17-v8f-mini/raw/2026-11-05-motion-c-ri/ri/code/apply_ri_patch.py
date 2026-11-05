"""Apply the REINDEXED patch (--scene reindexed) to the FRAMEFREEZE tool (sha256 0721fc01...). One-shot; each replacement must match exactly once.
Usage: python -I apply_ri_patch.py <framefreeze v8f_t1_eval_framefreeze.py> <out v8f_t1_eval_ri.py>
``--scene logged``, ``--scene frozen``, ``--scene hold_now`` (and every flag the FRAMEFREEZE tool has) are the source tool's behaviour bit for bit: the new code runs only under ``--scene reindexed``.
``--scene reindexed``: in ``build_item``, right after the item is built, the history rows whose time index exceeds the window's t0 take the logged cached frame at the along-track index of the model's OWN executed
position (``min(i, tau*)``, non-decreasing over the item's rows); rows with index <= t0 stay the logged frames bit for bit. The run header gets a ``scene_reindex`` receipt."""
import hashlib
import sys

SRC_SHA = "0721fc014089bea6b494e941a7d2a3d385e5c2ae89329848419fd843fbd9a0a9"
src, dst = sys.argv[1], sys.argv[2]
b = open(src, "rb").read()
assert hashlib.sha256(b).hexdigest() == SRC_SHA, "REFUSED: the source is not the FRAMEFREEZE tool 0721fc01"
s = b.decode("utf-8")


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


# 1. docstring
rep('''
Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)
"""''', '''
``--scene reindexed`` (REINDEXED, a longitudinal-only approximate re-perception; this file is ``v8f_t1_eval_ri.py``; ``logged``, ``frozen`` and ``hold_now`` are the FRAMEFREEZE tool's behaviour bit for bit).
WHY: T1-kin integrates the model's own actions but feeds the LOGGED camera frames on the human's clock. When the model lags the human (own speed far below the logged speed) the frames show positions the
car has not reached and advance at the human's speed, so a model that reads motion from vision copies the human's speed from the camera (a leak of the evaluation target). ``reindexed`` makes the frames follow
the model's OWN progress along the logged path.
RULE (``build_item``, after the item is built; ``reindex_item`` / ``reindex_rows`` / ``project_taus``): the item's ``feats`` / ``frame_ok`` / ``hmax`` rows end at the tick's cache index j; row r has time index
i = j - hmax + 1 + r. Rows with i <= t0 (the window's measured past) stay the LOGGED rows, bit for bit (checked on every item, not asserted by construction: the receipt compares the bytes). For a row with
i > t0: p_own(i) = the model's executed (x, y) at cache step i; it is projected onto the logged polyline ``pose_grid[:, :2]`` of the clip (nearest segment, the lateral offset is ignored and reported); tau* = the
logged cache index whose along-track arc length is nearest to the projected point (ties and a dwell of less than RI_DWELL_M go to the LATEST index); tau(i) = min(i, tau*); then tau is made non-decreasing over the
item's rows (running maximum; the past rows enter with their own index). The row takes the cached frame at tau(i) (``feats`` of the logged clip) and the ``frame_ok`` the item builder gives that index
(``tau >= 0`` and a finite logged ``p10f`` row, or ``tau < 1``); a row with tau(i) == i is left exactly as the item builder built it. A frame from AFTER the row's own time is never used (tau <= i: the receipt counts
``future_frame_rows``); a model AHEAD of the human sees the logged frame at i (counted ``rows_clamped``). The static scene geometry is consistent with the own position; dynamic agents are shown at the human's time i;
the lateral offset is ignored. This is an approximation of re-perception, NOT T2.
EXECUTED TRACK vs CACHE TIME (confirmed in the source tool v8f_t1_eval_framefreeze.py sha256 0721fc01..., line numbers are that file's): the clip's native 10 Hz track has 2 rows per 0.2 s cache step, row 2*i is
cache step i (``check_track_matches_grid``, v8f_mini_data.py:309: the even rows equal the cache ``pose_grid``). ``WindowState.__init__`` copies the rows <= 2*t0 of the logged track (line 264) and the rows <= t0 of the
ego arrays; ``commit_plan`` writes the 10 Hz rows 2*j+1 .. 2*(j+m) (line 294; the even row of step j+s is the plan's knot s, ``hermite_frames``) and ``_finish`` sets ``d["pose"][j] = track[2*j]`` for every executed
step (line 316) and ``self.j = j_new`` (line 325). ``rollout`` runs ticks n = 0 .. H//m - 1 (line 633), builds the items at ``st.j = t0 + n*m`` (line 637) BEFORE committing that tick's plan (line 649), so every row
i <= j of the item has an executed pose in ``st.d["pose"][i]`` (rows <= t0: the logged pose). p_own(i) = ``st.d["pose"][i][:2]``, clip-local metres, the frame of ``pose_grid``.
RECEIPT (``header["scene_reindex"]``): items, rows (= items x hmax), rows_logged (i <= t0), rows_reindexed (tau < i), rows_on_time (tau* == i), rows_clamped (tau* > i), rows_monotone_raised, future_frame_rows (must be
0), the lag i - tau distribution in cache steps (mean, p50, p90, max) over all rows with i > t0 and over the reindexed rows, the lateral offset of the projection in metres, for the MODEL's rollout (top level) and for
the P10 floor's rollout (``floor``; it reads no frames), and ``past_rows_bit_identical`` (true iff the compared past rows are > 0 and none changed). The ticks file carries ``scene``.

Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)
"""''')

# 2. scene modes + constants + stats
rep('SCENE_MODES = ("logged", "frozen", "hold_now")\n', '''SCENE_MODES = ("logged", "frozen", "hold_now", "reindexed")
#: REINDEXED: a stretch of the logged path that moves less than this (metres of arc length) between two cache steps is a dwell (a stop with sensor jitter); tau* then goes to the LATEST index of the dwell.
RI_DWELL_M = 0.05
#: REINDEXED: per-run receipt of the reindexed apply line (a fresh process per run): one block per rollout role (the model's ticks / the P10 floor's ticks, told apart by ``stats is not None`` in ``build_item``)
#: plus the past-row byte comparison over both. ``lag_hist`` = {lag in cache steps: row count} over the rows with i > t0.
RI_STATS = {}


def ri_reset():
    RI_STATS.clear()
    for role in ("model", "floor"):
        RI_STATS[role] = {"items": 0, "rows": 0, "rows_logged": 0, "rows_reindexed": 0, "rows_on_time": 0, "rows_clamped": 0, "rows_monotone_raised": 0, "future_frame_rows": 0, "lag_hist": {}, "offset_m": []}
    RI_STATS["past"] = {"rows_checked": 0, "rows_changed": 0}


ri_reset()
''')

# 3. helpers
rep('def build_item(ds, proxy, st: WindowState, tick: int, a_t: int, a_s: int, scene: str, limit: str, stats: dict | None = None) -> dict:\n', '''def logged_arc(xy_log) -> np.ndarray:
    """REINDEXED: the cumulative arc length S[k] (metres, S[0] = 0, non-decreasing) of the logged polyline ``xy_log`` [T, 2] at its cache indices k. Refuses a polyline with a non-finite point."""
    xy = np.asarray(xy_log, np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError("REFUSED: --scene reindexed needs a finite logged (x, y) polyline of at least two points")
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


def project_taus(xy_log, own_xy, dwell_m: float = RI_DWELL_M):
    """REINDEXED: tau* for each own point. ``own_xy`` [R, 2] -> (``tau_star`` [R] int, ``offset_m`` [R] float = the distance from the point to its projection on the polyline).
    The point is projected on the NEAREST segment of ``xy_log`` (the latest segment among exact ties), its arc length s = S[k] + u * |segment| is read, and tau* is the logged index whose arc length S is nearest
    to s (the latest among exact ties), moved to the latest index whose arc length is within ``dwell_m`` of it (a dwell: a stop with jitter)."""
    xy = np.asarray(xy_log, np.float64)
    S = logged_arc(xy)
    own = np.asarray(own_xy, np.float64).reshape(-1, 2)
    if not np.isfinite(own).all():
        raise ValueError("REFUSED: --scene reindexed got a non-finite own (executed) position")
    a, ab = xy[:-1], xy[1:] - xy[:-1]
    l2 = (ab ** 2).sum(1)
    ln = np.sqrt(l2)
    safe = np.where(l2 > 0, l2, 1.0)
    tau = np.empty(len(own), np.int64)
    off = np.empty(len(own), np.float64)
    for r, p in enumerate(own):
        u = np.where(l2 > 0, np.clip(((p - a) * ab).sum(1) / safe, 0.0, 1.0), 0.0)
        d = np.linalg.norm(p - (a + u[:, None] * ab), axis=1)
        k = int(np.flatnonzero(d <= d.min() + 1e-9)[-1])
        sk = S[k] + u[k] * ln[k]
        ds_ = np.abs(S - sk)
        k0 = int(np.flatnonzero(ds_ <= ds_.min() + 1e-9)[-1])
        tau[r] = max(k0, int(np.searchsorted(S, S[k0] + dwell_m, side="right")) - 1)
        off[r] = d[k]
    return tau, off


def reindex_rows(i_rows, t0: int, own_xy, xy_log) -> dict:
    """REINDEXED: the frame index for each row of an item. ``i_rows`` [hm] = the rows' time indices (may start below 0: padded rows), ``own_xy`` [hm, 2] = the executed (x, y) at each row's time (only the rows with
    i > t0 are read), ``xy_log`` [T, 2] = the clip's logged polyline. Returns ``tau`` [hm] int (the index of the frame each row takes; = the clipped own index on the rows with i <= t0), ``tau_star`` [hm] (-1 on those rows),
    ``past`` [hm] bool, ``offset_m`` [hm] (0 on the past rows), ``raised`` [hm] bool (the running maximum moved the row), ``idc`` [hm] (the clipped own index)."""
    i_rows = np.asarray(i_rows, np.int64)
    hm = len(i_rows)
    idc = np.clip(i_rows, 0, len(xy_log) - 1)
    past = i_rows <= t0
    tau_star = np.full(hm, -1, np.int64)
    off = np.zeros(hm, np.float64)
    if (~past).any():
        tau_star[~past], off[~past] = project_taus(xy_log, np.asarray(own_xy, np.float64)[~past])
    raw = np.where(past, idc, np.minimum(i_rows, tau_star))
    tau = np.maximum.accumulate(raw)
    return {"tau": tau, "tau_star": tau_star, "past": past, "offset_m": off, "raised": tau != raw, "idc": idc}


def reindex_item(it: dict, st: WindowState, is_model: bool = True) -> tuple:
    """REINDEXED: ``(feats, frame_ok)`` of the item with the rows i > t0 taken from the logged frame at tau(i) (see the module docstring); updates ``RI_STATS``. Refuses an item without cached ``feats``."""
    f = it.get("feats")
    if f is None:
        raise RuntimeError("REFUSED: --scene reindexed needs the cached per-frame feats (absent under --encoder lora): a flag that silently reindexes nothing is worse than none")
    hm, j, t0 = f.shape[0], int(st.j), int(st.t0)
    i_rows = np.arange(j - hm + 1, j + 1)
    base = st.base_log
    idc = np.clip(i_rows, 0, st.T - 1)
    own = np.asarray(st.d["pose"], np.float64)[idc][:, :2]
    rr = reindex_rows(i_rows, t0, own, np.asarray(base["pose"], np.float64)[:, :2])
    tau = rr["tau"]
    feats, ok = np.array(f, copy=True), np.array(it["frame_ok"], copy=True)
    ch = tau != rr["idc"]
    if ch.any():
        feats[ch] = base["feats"][tau[ch]]
        ok[ch] = (tau[ch] >= 0) & (np.isfinite(np.asarray(st.d_in["p10f"])[tau[ch]]).all(1) | (tau[ch] < 1))
    pst = RI_STATS["past"]                                              # the check, not an assertion: the rows handed on for i <= t0 are byte-equal to the rows the item builder built;
    chk = i_rows <= t0                                                  # its mask is spelled here on its own, so a wrong mask inside reindex_rows cannot hide from it
    npast = int(chk.sum())
    if npast:
        a_b, b_b = feats[chk].view(np.uint8).reshape(npast, -1), np.asarray(f)[chk].view(np.uint8).reshape(npast, -1)
        pst["rows_checked"] += npast
        pst["rows_changed"] += int((((a_b != b_b).any(1)) | (ok[chk] != np.asarray(it["frame_ok"])[chk])).sum())
    r = RI_STATS["model" if is_model else "floor"]
    fut = ~chk
    r["items"] += 1
    r["rows"] += hm
    r["rows_logged"] += npast
    r["rows_reindexed"] += int((fut & (tau < i_rows)).sum())
    r["rows_on_time"] += int((fut & (rr["tau_star"] == i_rows)).sum())
    r["rows_clamped"] += int((fut & (rr["tau_star"] > i_rows)).sum())
    r["rows_monotone_raised"] += int(rr["raised"].sum())
    r["future_frame_rows"] += int((tau > rr["idc"]).sum())
    for lg in (i_rows - tau)[fut].tolist():
        r["lag_hist"][lg] = r["lag_hist"].get(lg, 0) + 1
    r["offset_m"].append(rr["offset_m"][fut])
    return feats, ok


def ri_summary(r: dict) -> dict:
    """REINDEXED: one role's receipt block (JSON-safe): counts, the lag distribution over the rows with i > t0 and over the reindexed rows (lag > 0), the lateral offset of the projection."""
    def dist(vals):
        v = np.asarray(vals, np.float64)
        return {"n": int(v.size), "mean": float(v.mean()), "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)), "max": float(v.max())} if v.size else {"n": 0}
    lags = np.repeat(np.array(sorted(r["lag_hist"]), np.float64), [r["lag_hist"][k] for k in sorted(r["lag_hist"])]) if r["lag_hist"] else np.zeros(0)
    off = np.concatenate(r["offset_m"]) if r["offset_m"] else np.zeros(0)
    keys = ("items", "rows", "rows_logged", "rows_reindexed", "rows_on_time", "rows_clamped", "rows_monotone_raised", "future_frame_rows")
    return {**{k: int(r[k]) for k in keys}, "lag_cache_steps_all_rows_after_t0": dist(lags), "lag_cache_steps_reindexed_rows": dist(lags[lags > 0]), "lateral_offset_m_rows_after_t0": dist(off)}


def build_item(ds, proxy, st: WindowState, tick: int, a_t: int, a_s: int, scene: str, limit: str, stats: dict | None = None) -> dict:
''')

# 4. the apply line (right after the hold_now block, before the limit handling)
rep('''        FREEZE_STATS["n_unequal_elements"] += nu
    if limit == "off":''', '''        FREEZE_STATS["n_unequal_elements"] += nu
    if scene == "reindexed":                                             # REINDEXED: the frames follow the model's OWN progress along the logged path (rows i <= t0 stay logged)
        it["feats"], it["frame_ok"] = reindex_item(it, st, is_model=stats is not None)
    if limit == "off":''')

# 5. parser help
rep('''hold_now (FRAMEFREEZE) = every history row of the feats is the CURRENT frame's row, repeated (no frame motion)")''',
    '''hold_now (FRAMEFREEZE) = every history row of the feats is the CURRENT frame's row, repeated (no frame motion); reindexed (REINDEXED) = the history rows after the window's t0 take the logged frame at the along-track index of the model's OWN executed position, min(i, tau*), rows up to t0 stay logged")''')

# 6. per-run stats reset
rep('''    FREEZE_STATS.update({"items": 0, "max_row_dev": 0.0, "n_unequal_elements": 0})
''', '''    FREEZE_STATS.update({"items": 0, "max_row_dev": 0.0, "n_unequal_elements": 0})
    ri_reset()
''')

# 7. header
rep('''"prereg": "dagland/histswap/DRAFT_RECOVERY_CARD_v2.md section 8.3 (frame-freeze control); reading thresholds frozen with dagland/recover_ft"}
''', '''"prereg": "dagland/histswap/DRAFT_RECOVERY_CARD_v2.md section 8.3 (frame-freeze control); reading thresholds frozen with dagland/recover_ft"}
    if a.scene == "reindexed":                                           # only then: every other --scene value => the header (and the result JSON) is byte-identical to the FRAMEFREEZE tool's
        header["open_channels"].append("camera stream: REINDEXED, an approximate longitudinal re-perception: the history rows after t0 are the logged frames at the along-track index of the model's own executed position, "
                                       "min(i, tau*); static scene geometry consistent with the own position, dynamic agents shown at the human's time, lateral offset ignored; NOT T2")
        pr = RI_STATS["past"]
        header["scene_reindex"] = {"scene": "reindexed", **ri_summary(RI_STATS["model"]), "floor": ri_summary(RI_STATS["floor"]),
                                   "past_rows_checked": pr["rows_checked"], "past_rows_changed": pr["rows_changed"], "past_rows_bit_identical": bool(pr["rows_checked"] > 0 and pr["rows_changed"] == 0),
                                   "rule": "rows i <= t0 stay logged (byte-compared); rows i > t0: tau(i) = min(i, tau*), tau* = the logged cache index nearest in arc length to the projection of the model's executed (x, y) at i on the logged polyline "
                                           "(ties and a dwell under dwell_m -> the latest index), tau non-decreasing over the item's rows; the row takes the logged feats[tau] and the frame_ok of that index",
                                   "dwell_m": RI_DWELL_M, "role_note": "top level = the MODEL's rollout (the ticks the network reads); 'floor' = the P10 floor's rollout, which reads no frames",
                                   "limits": "longitudinal only: the lateral offset of the own position from the logged path is ignored (reported as lateral_offset_m); dynamic agents are shown at the human's time; the current frame still depicts a logged "
                                             "position (the along-track one of the own position) and T2 is not provisioned"}
''')

# 8. ticks file carries the scene
rep('''**({"scene": a.scene} if a.scene == "hold_now" else {})})''', '''**({"scene": a.scene} if a.scene in ("hold_now", "reindexed") else {})})''')

open(dst, "wb").write(s.encode("utf-8"))
print("wrote", dst, hashlib.sha256(s.encode("utf-8")).hexdigest())
