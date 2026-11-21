"""Edits of stack/tanitad/data/v8f_mini_data.py for LEADSUP (run once on tree_patch; argv[1] = tree root). Each edit must match exactly once."""
import sys

root = sys.argv[1]
p = f"{root}/stack/tanitad/data/v8f_mini_data.py"
s = open(p, encoding="utf-8", newline="").read()
assert "\r\n" not in s


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---- label function (after lead_label_arrays: before the kinematic floors block)
rep('''# ------------------------------------------------------------------------------------------------ kinematic floors
KIN_MODES = ("steer", "yaw")''', '''#: LEADSUP label rule (the rule of spatial/objslot ``hist_run`` / ``ls_slope`` and of S0E E3): gap labels are the join's ``lead_gap`` where ``lead_valid`` is 1 and 0 <= gap <= 60 m
LEAD_SUP_GAP_MAX_M = 60.0
LEAD_SUP_STEP_MAX_M = 3.0          # a 0.2 s step of the gap above this breaks the run (another lead)
LEAD_SUP_RUN_MAX = 5               # at most this many trailing frames enter the slope
LEAD_SUP_RUN_MIN = 3               # fewer labelled frames: no closing label
LEAD_SUP_DT = 0.2                  # s between two cache steps


def lead_sup_label_arrays(J: dict) -> dict:
    """LEADSUP: per-join-row labels, ROW-ALIGNED to ``J`` (row ``off[c] + j`` = cache step ``j`` of clip ``c``, 0.2 s apart). Computed ONCE from the join arrays (no extra file): privileged, training only.

      ``pres`` [N] float32  1.0 where ``lead_valid`` is 1, else 0.0 (every row has a presence label)
      ``gap``  [N] float32  ``lead_gap`` [m] where ``lead_valid`` == 1, the gap is finite and 0 <= gap <= 60 m; else NaN
      ``clos`` [N] float32  closing speed [m/s] = least-squares slope of the gap over the trailing run of labelled frames ending at this row (0.2 s apart), at most 5 frames, a step above 3.0 m breaks the run
                            (the run never crosses a clip), at least 3 frames; else NaN. Positive = the gap opens.
    plus ``sha256`` of the three arrays (bytes, in this order) and the counts ``n_rows``, ``n_pres``, ``n_gap``, ``n_clos``."""
    import hashlib
    off, n = np.asarray(J["off"], np.int64), np.asarray(J["n"], np.int64)
    N = int(n.sum())
    lv = np.asarray(J["lead_valid"])[:N] == 1
    g = np.asarray(J["lead_gap"], np.float64)[:N]
    ok = lv & np.isfinite(g) & (g >= 0.0) & (g <= LEAD_SUP_GAP_MAX_M)
    gap = np.where(ok, g, np.nan)
    first = np.zeros(N, bool)                                       # first row of each clip: its predecessor belongs to another clip
    first[off[n > 0]] = True
    s = np.zeros(N, bool)                                           # s[r]: row r links to row r - 1 (both labelled, same clip, a step within the bound)
    s[1:] = ok[1:] & ok[:-1] & ~first[1:] & (np.abs(g[1:] - g[:-1]) <= LEAD_SUP_STEP_MAX_M)
    alive = ok.copy()
    run = ok.astype(np.int64)
    for k in range(1, LEAD_SUP_RUN_MAX):                            # frame r - k is in the run iff s[r], s[r-1], ..., s[r-k+1] all hold
        sh = np.zeros(N, bool)
        sh[k - 1:] = s[:N - (k - 1)]
        alive = alive & sh
        run += alive
    clos = np.full(N, np.nan, np.float64)
    r_all = np.arange(N)
    for m in range(LEAD_SUP_RUN_MIN, LEAD_SUP_RUN_MAX + 1):
        r = r_all[run == m]
        if len(r) == 0:
            continue
        G = np.stack([g[r - (m - 1) + j] for j in range(m)], 1)    # [n, m], oldest first
        tt = np.arange(m, dtype=np.float64) * LEAD_SUP_DT
        tb = tt.mean()
        clos[r] = ((tt - tb)[None, :] * (G - G.mean(1, keepdims=True))).sum(1) / ((tt - tb) ** 2).sum()
    pres = lv.astype(np.float32)
    gap32, clos32 = gap.astype(np.float32), clos.astype(np.float32)
    h = hashlib.sha256()
    for a_ in (pres, gap32, clos32):
        h.update(np.ascontiguousarray(a_).tobytes())
    return {"pres": pres, "gap": gap32, "clos": clos32, "off": off, "sha256": h.hexdigest(), "n_rows": N, "n_pres": int(lv.sum()),
            "n_gap": int(np.isfinite(gap32).sum()), "n_clos": int(np.isfinite(clos32).sum())}


# ------------------------------------------------------------------------------------------------ kinematic floors
KIN_MODES = ("steer", "yaw")''')

# ---- dataset: flag + arrays
rep('''    def __init__(self, store: ClipStore, J: dict, windows, *, hist: int = 4, hmax: int = 10, max_age_t: int = 2,
                 max_age_s: int = 6, window_start=None):
        self.store, self.J, self.windows = store, J, list(windows)''', '''    def __init__(self, store: ClipStore, J: dict, windows, *, hist: int = 4, hmax: int = 10, max_age_t: int = 2,
                 max_age_s: int = 6, window_start=None, lead_sup: bool = False):
        self.store, self.J, self.windows = store, J, list(windows)
        self.lead_sup = bool(lead_sup)                              # LEADSUP: True adds the per-frame labels lead_pres / lead_gapm / lead_clos to every item (False = today's item, key for key)
        self.lead_sup_labels = lead_sup_label_arrays(J) if self.lead_sup else None''')

rep('''        # geometry between the tick frames and NOW (measured odometry): used to re-express held outputs in the NOW frame
        it["pose_now"]''', '''        if self.lead_sup:                                        # LEADSUP: the labels of the F frames of the window (join rows off + idc; the clamped frames before the clip start are masked by frame_ok)
            LS = self.lead_sup_labels
            rr = int(self.J["off"][ci]) + idc
            it["lead_pres"], it["lead_gapm"], it["lead_clos"] = LS["pres"][rr], LS["gap"][rr], LS["clos"][rr]
        # geometry between the tick frames and NOW (measured odometry): used to re-express held outputs in the NOW frame
        it["pose_now"]''')

# ---- collate
rep('''    out["p10_t"] = torch.as_tensor(np.stack([np.nan_to_num(it["p10_t"]) for it in items]), dtype=torch.float32)''', '''    for k in ("lead_pres", "lead_gapm", "lead_clos"):         # LEADSUP: present only when the dataset was made with lead_sup=True (NaN = no label)
        if k in items[0]:
            out[k] = stack(k, torch.float32)
    out["p10_t"] = torch.as_tensor(np.stack([np.nan_to_num(it["p10_t"]) for it in items]), dtype=torch.float32)''')

open(p, "w", encoding="utf-8", newline="").write(s)
print("OK", p)
