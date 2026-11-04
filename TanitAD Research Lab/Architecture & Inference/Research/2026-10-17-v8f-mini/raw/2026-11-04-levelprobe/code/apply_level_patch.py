"""Apply the LEVELPROBE patch to the HISTSWAP-LF tool (sha256 6c66f0a5...): two NEW values of ``--histswap``, ``cv_t0`` and ``cv_pred``, and one new option ``--level-file``.  One-shot; each replacement must match exactly once.
Usage: python -I apply_level_patch.py <histswap_lf v8f_t1_eval_histswap_lf.py> <out v8f_t1_eval_levelprobe.py>
The four existing values (``off`` / ``logged`` / ``cv_exec`` / ``cv_logged``) and the default path are the shipped tool's behaviour bit for bit (``tests/test_levelprobe_tool.py`` pins it on a real window).
  cv_t0    (HLVL-T0,   ORACLE)   v = the LOGGED measured speed at the window's t0 (``base_log["p10f"][t0][0]``), held constant at EVERY tick; omega = the EXECUTED yaw rate at the tick (as HCV / HLVL).
  cv_pred  (HLVL-PRED, vision)   v = the per-window level read from ``--level-file`` (an npz with int arrays ``clip_index``, ``ws`` and a float array ``level``; one row per window of the grid), held constant at EVERY tick.
                                 A window with no row, a non-finite or negative level, or a missing file is a REFUSAL (no fallback value exists)."""
import hashlib
import sys

SRC_SHA = "6c66f0a527b3b08c640550dd4d065c45f43186d1af5220830f38bd933245719f"
src, dst = sys.argv[1], sys.argv[2]
b = open(src, "rb").read()
assert hashlib.sha256(b).hexdigest() == SRC_SHA, "REFUSED: the source is not the HISTSWAP-LF tool 6c66f0a5"
s = b.decode("utf-8")


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)


rep('''import argparse
''', '''import argparse
import hashlib
''')
rep('''HISTSWAP_MODES = ("off", "logged", "cv_exec", "cv_logged")
''', '''HISTSWAP_MODES = ("off", "logged", "cv_exec", "cv_logged", "cv_t0", "cv_pred")
#: LEVELPROBE (PREREG_LEVELPROBE.md): a per-window CONSTANT level held at every tick (the speed is not re-read from the executed / logged track at the tick).
LEVEL_MODES = ("cv_t0", "cv_pred")
''')
rep('''CV_MODES = ("cv_exec", "cv_logged")
''', '''CV_MODES = ("cv_exec", "cv_logged", "cv_t0", "cv_pred")
''')
rep('''        self.hist_swap = self.hist_mode != "off"
''', '''        self.hist_swap = self.hist_mode != "off"
        self.level = None                                                 # LEVELPROBE: the per-window constant level of cv_pred (set by make_states); cv_t0 reads the logged clip at t0
''')
rep('''        src = d if self.hist_mode == "cv_exec" else self.base_log
        v = float(src["p10f"][j][0])
''', '''        if self.hist_mode == "cv_t0":
            v = float(self.base_log["p10f"][self.t0][0])                  # the LOGGED measured speed at t0, held constant
        elif self.hist_mode == "cv_pred":
            if self.level is None or not (math.isfinite(float(self.level)) and float(self.level) >= 0.0):
                raise ValueError(f"cv_pred: window (clip_index {self.ci}, ws {self.t0}) has no valid level ({self.level!r})")
            v = float(self.level)
        else:
            src = d if self.hist_mode == "cv_exec" else self.base_log
            v = float(src["p10f"][j][0])
''')
rep('''def make_states(store, tracks, J, wins, nav_mode, hist_swap=False):
    out = []
    for ci, t0, _, _ in wins:
        clip = str(J["clip"][ci])
        base = store.get(clip)
        o, n = int(J["off"][ci]), int(J["n"][ci])
        out.append(WindowState(base, tracks[clip], J["nav"][o:o + n], ci, t0, nav_mode, hist_swap))
    return out
''', '''def make_states(store, tracks, J, wins, nav_mode, hist_swap=False, levels=None):
    out = []
    for ci, t0, _, _ in wins:
        clip = str(J["clip"][ci])
        base = store.get(clip)
        o, n = int(J["off"][ci]), int(J["n"][ci])
        st = WindowState(base, tracks[clip], J["nav"][o:o + n], ci, t0, nav_mode, hist_swap)
        if st.hist_mode == "cv_pred":
            if levels is None or (int(ci), int(t0)) not in levels:
                raise ValueError(f"REFUSED: cv_pred has no level for window (clip_index {ci}, ws {t0})")
            st.level = float(levels[(int(ci), int(t0))])
        out.append(st)
    return out


def load_levels(path):
    """``--level-file`` npz -> {(clip_index, ws): level}.  Refuses duplicate windows and non-finite / negative levels."""
    z = np.load(path, allow_pickle=False)
    ci, ws, lv = np.asarray(z["clip_index"], np.int64), np.asarray(z["ws"], np.int64), np.asarray(z["level"], np.float64)
    if not (len(ci) == len(ws) == len(lv)) or not np.isfinite(lv).all() or (lv < 0).any():
        raise SystemExit(f"REFUSED: {path}: lengths differ, or a level is non-finite or negative")
    out = {(int(a), int(b)): float(c) for a, b, c in zip(ci, ws, lv)}
    if len(out) != len(lv):
        raise SystemExit(f"REFUSED: {path}: duplicate (clip_index, ws) rows")
    return out
''')
rep('''    p.add_argument("--max-windows", type=int, default=0, help="smoke: evenly-spaced subset of the grid (0 = all)")
''', '''    p.add_argument("--max-windows", type=int, default=0, help="smoke: evenly-spaced subset of the grid (0 = all)")
    p.add_argument("--level-file", default="", help="LEVELPROBE: npz (clip_index, ws, level) with one row per window; required by, and only allowed with, --histswap cv_pred")
''')
rep('''        raise SystemExit(f"REFUSED: --histswap {a.histswap} is defined with --limit causal only (got {a.limit})")
''', '''        raise SystemExit(f"REFUSED: --histswap {a.histswap} is defined with --limit causal only (got {a.limit})")
    if (a.histswap == "cv_pred") != bool(a.level_file):
        raise SystemExit("REFUSED: --histswap cv_pred needs --level-file, and --level-file is allowed with cv_pred only")
    LEVELS = load_levels(a.level_file) if a.level_file else None
''')
rep('''        st_m = make_states(store, tracks, J, chunk, a.nav, hist_swap=a.histswap)''', '''        st_m = make_states(store, tracks, J, chunk, a.nav, hist_swap=a.histswap, levels=LEVELS)''')
rep('''    if a.dump_ticks:                                                     # only then: flags absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        tick_path''', '''    if a.histswap in LEVEL_MODES:                                        # LEVELPROBE: the header says where the constant level comes from
        header["histswap"].update({"speed_source": ("the LOGGED measured speed at the window's t0, held constant at every tick (ORACLE, level at t0 only)" if a.histswap == "cv_t0"
                                                    else "a per-window level read from --level-file (vision-derived), held constant at every tick"),
                                   "oracle": "yes: the logged speed at t0 is the human's; NOT deployable" if a.histswap == "cv_t0" else "no: the level file is the vision probe's prediction from the t0 frame",
                                   "prereg": "dagland/levelprobe/PREREG_LEVELPROBE.md (H-LEVELPROBE-1)"})
        if a.level_file:
            header["histswap"]["level_file"] = {"path": a.level_file, "sha256": hashlib.sha256(open(a.level_file, "rb").read()).hexdigest(), "n_windows": len(LEVELS)}
    if a.dump_ticks:                                                     # only then: flags absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        tick_path''')
open(dst, "wb").write(s.encode("utf-8"))
print("wrote", dst, hashlib.sha256(s.encode("utf-8")).hexdigest())
