"""Apply the FRAMEFREEZE patch (--scene hold_now) to the HISTSWAP-LF tool (sha256 6c66f0a5...). One-shot; each replacement must match exactly once.
Usage: python -I apply_fz_patch.py <histswap_lf v8f_t1_eval_histswap_lf.py> <out v8f_t1_eval_framefreeze.py>
``--scene logged`` and ``--scene frozen`` (and every flag the LF tool has) are the source tool's behaviour bit for bit: the new code runs only under ``--scene hold_now``.
``--scene hold_now`` (dagland/histswap/DRAFT_RECOVERY_CARD_v2.md section 8.3, the frame-freeze control): in ``build_item``, right after the item is built, every row of ``it["feats"]`` is replaced by the CURRENT
frame's row (``feats[k] := feats[-1]`` for all ``hmax`` rows), so the ``hist`` frames O reads (and the ones T and S read) are the tick-j frame repeated and no frame MOTION reaches the network. ``frame_ok`` is
unchanged (same token count and padding pattern); the run header gets ``camera_motion: false`` (``strict.camera`` stays False: the camera is still logged)."""
import hashlib
import sys

SRC_SHA = "6c66f0a527b3b08c640550dd4d065c45f43186d1af5220830f38bd933245719f"
src, dst = sys.argv[1], sys.argv[2]
b = open(src, "rb").read()
assert hashlib.sha256(b).hexdigest() == SRC_SHA, "REFUSED: the source is not the HISTSWAP-LF tool 6c66f0a5"
s = b.decode("utf-8")


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


# 1. docstring
rep('''
Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)
"""''', '''
``--scene hold_now`` (FRAMEFREEZE, the frame-freeze control of DRAFT_RECOVERY_CARD_v2.md section 8.3; this file is ``v8f_t1_eval_framefreeze.py``; ``logged`` and ``frozen`` are the HISTSWAP-LF tool's behaviour bit for
bit): in ``build_item``, right after the item is built, every row of ``it["feats"]`` is replaced by the CURRENT frame's row, repeated (``feats[k] := feats[-1]`` for all ``hmax`` rows), so the ``hist`` frames O reads
(and the frames T and S read) are the tick-j frame repeated and no frame MOTION is available to the network. ``frame_ok`` is unchanged. The camera is still logged (``strict.camera`` stays False) and the run header
gets ``camera_motion: false``, an open-channel note (the current frame still depicts the human's position at the tick) and a ``frame_freeze`` receipt (items frozen, max row deviation, unequal elements: the
every-batch check). ``--freeze-card-ref`` carries the RECOVER-FT card's frozen sha line into that header. Needs the cached per-frame ``feats`` (refuses otherwise).

Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)
"""''')

# 2. scene modes
rep('SCENE_MODES = ("logged", "frozen")\n', '''SCENE_MODES = ("logged", "frozen", "hold_now")
#: FRAMEFREEZE: per-run receipt of the hold_now apply line (a fresh process per run): items frozen, max |row - tick-j frame| over finite elements, count of elements that differ from the tick-j frame as built (NaN == NaN counts as equal).
FREEZE_STATS = {"items": 0, "max_row_dev": 0.0, "n_unequal_elements": 0}
''')

# 3. helpers + the apply line
rep('def build_item(ds, proxy, st: WindowState, tick: int, a_t: int, a_s: int, scene: str, limit: str, stats: dict | None = None) -> dict:\n', '''def hold_now_feats(it: dict) -> np.ndarray:
    """FRAMEFREEZE: the item's per-frame feature rows with EVERY row replaced by the current frame's row (the last one): ``[hmax, ...]`` in, same shape and dtype out. Refuses an item without cached ``feats``."""
    f = it.get("feats")
    if f is None:
        raise RuntimeError("REFUSED: --scene hold_now needs the cached per-frame feats (absent under --encoder lora): a flag that silently freezes nothing is worse than none")
    return np.repeat(f[-1:], f.shape[0], axis=0)


def freeze_dev(f, cur) -> tuple:
    """(max |row - cur| over finite differences, number of elements that differ from ``cur``; NaN equals NaN): ``f`` = a ``[hmax, ...]`` feature array after the freeze, ``cur`` = ``[1, ...]``, the tick-j frame as the item
    builder built it (BEFORE the freeze). Both are 0 iff every row IS the current frame (a freeze to a stale row reads > 0, not 0)."""
    f, cur = np.asarray(f), np.asarray(cur)
    eq = (f == cur) | (np.isnan(f) & np.isnan(cur))
    d = np.abs(f.astype(np.float64) - cur.astype(np.float64))
    d = d[np.isfinite(d)]
    return (float(d.max()) if d.size else 0.0), int((~eq).sum())


def build_item(ds, proxy, st: WindowState, tick: int, a_t: int, a_s: int, scene: str, limit: str, stats: dict | None = None) -> dict:
''')
rep('''        it["feats"], it["frame_ok"] = st.scene0
    if limit == "off":''', '''        it["feats"], it["frame_ok"] = st.scene0
    if scene == "hold_now":                                              # FRAMEFREEZE: no frame motion; frame_ok untouched
        cur = None if it.get("feats") is None else np.array(it["feats"][-1:])      # the tick-j frame as built, before the freeze
        it["feats"] = hold_now_feats(it)
        dv, nu = freeze_dev(it["feats"], cur)
        FREEZE_STATS["items"] += 1
        FREEZE_STATS["max_row_dev"] = max(FREEZE_STATS["max_row_dev"], dv)
        FREEZE_STATS["n_unequal_elements"] += nu
    if limit == "off":''')

# 4. parser
rep('''    p.add_argument("--scene", choices=SCENE_MODES, default="logged")\n''', '''    p.add_argument("--scene", choices=SCENE_MODES, default="logged", help="logged = the logged frames (default); frozen = the TICK-0 feats for the whole rollout; hold_now (FRAMEFREEZE) = every history row of the feats is the CURRENT frame's row, repeated (no frame motion)")
    p.add_argument("--freeze-card-ref", default="", help="FRAMEFREEZE: free text recorded in the run header (the RECOVER-FT card's frozen sha line); only with --scene hold_now")
''')

# 5. refusal + per-run stats reset
rep('''def cmd_run(a):
    import torch
''', '''def cmd_run(a):
    import torch
    if a.freeze_card_ref and a.scene != "hold_now":
        raise SystemExit("REFUSED: --freeze-card-ref is recorded only under --scene hold_now (a flag that silently writes nothing is worse than none)")
    FREEZE_STATS.update({"items": 0, "max_row_dev": 0.0, "n_unequal_elements": 0})
''')

# 6. header
rep('''    header["open_channels"] = [c for c in header["open_channels"] if c]
''', '''    header["open_channels"] = [c for c in header["open_channels"] if c]
    if a.scene == "hold_now":                                            # only then: every other --scene value => the header (and the result JSON) is byte-identical to the HISTSWAP-LF tool's
        header["camera_motion"] = False
        header["open_channels"].append("camera stream: the CURRENT frame is logged (it depicts the human's position at the tick; a position or blur cue survives); the hist frames are that frame repeated, so no frame motion reaches the network")
        header["frame_freeze"] = {"scene": "hold_now", "applied": "every row of the item's feats := the current frame's row (feats[k] := feats[-1]), right after the item is built, at every tick of every window; frame_ok unchanged",
                                  "items_frozen": FREEZE_STATS["items"], "max_row_dev": FREEZE_STATS["max_row_dev"], "n_unequal_elements": FREEZE_STATS["n_unequal_elements"],
                                  "card_ref": a.freeze_card_ref or "UNVERIFIED: no --freeze-card-ref was passed",
                                  "prereg": "dagland/histswap/DRAFT_RECOVERY_CARD_v2.md section 8.3 (frame-freeze control); reading thresholds frozen with dagland/recover_ft"}
''')

# 7. ticks file carries the scene
rep('''**({"histswap": a.histswap} if a.histswap != "off" else {})})''', '''**({"histswap": a.histswap} if a.histswap != "off" else {}),
                             **({"scene": a.scene} if a.scene == "hold_now" else {})})''')

open(dst, "wb").write(s.encode("utf-8"))
print("wrote", dst, hashlib.sha256(s.encode("utf-8")).hexdigest())
