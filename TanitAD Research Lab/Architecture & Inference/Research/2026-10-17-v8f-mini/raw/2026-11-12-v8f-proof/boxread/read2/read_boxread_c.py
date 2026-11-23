"""read_boxread_c.py -- card PREREG_V8F_PROOF_ADD128 (BOXREAD-T0-C): the reader of card ADD127 (read_boxread.py, imported unchanged, identity checked by its sha256)
with the changed controls of card ADD128, section 2. Tier T0: recorded frames, no rollout; a diagnostic of a perception read and not driving performance.

What changes against the imported reader (and nothing else: arms, measures, bars, names and the estimator are the imported ones):
  c0  = b0, and the detector listings are those of the controls file of card ADD127's run (--prior-controls)
  c1  = the tool test of b1; the share of label directions inside the wide window is stored beside and decides nothing
  c2  = (i) the pick share of b2; (ii) under the label direction displaced left, right (half the label width + 4 deg) and up (half the label height + 4 deg)
        the same detection is picked on at most 0.10 of the frames, for each of the three; the mirrored share is stored beside and decides nothing
  c4  = b4; the given sha256 of the locator file is taken as a beginning of at least 16 characters
The card's c<k> is stored and printed under the imported reader's key b<k>. CPU only. Usage: as read_boxread.py, plus --prior-controls <BOXREAD_CONTROLS.json of card ADD127>."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
import re
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD128"
PRIOR_CARD = "PREREG_V8F_PROOF_ADD127"
RB_PREFIX = "aaac070b11e827c0"
C2_SHIFT_DEG = 4.0
C2_SAME_MAX = 0.10
C2_MIN_FRAMES, C2_MIN_CLIPS = 1000, 100
C4_MIN_HEX = 16
HERE = os.path.dirname(os.path.abspath(__file__))
LIMIT_MORE = (" The set of controls is that of card ADD128: it was changed after the controls run of card ADD127 had failed and its control values were known.")
INSIDE_LINE = "the label direction lies inside the wide window"
MIRROR_LINE = "the share with a pick under mirrored directions"


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_reader():
    m = sys.modules.get("read_boxread")
    if m is not None and getattr(m, "__file__", None) and os.path.isfile(m.__file__):
        if not _sha(m.__file__).startswith(RB_PREFIX):
            raise SystemExit("REFUSED: the loaded read_boxread.py is not the reader of card ADD127 (%s)" % RB_PREFIX)
        return m
    cands = [os.environ.get("BOXREAD_READER"), "/workspace/v8proof/boxread/kit/read/read_boxread.py", "F:/TanitAD-scratch/v8proof/boxread/read/read_boxread.py"]
    for p in cands:
        if p and os.path.isfile(p):
            hx = _sha(p)
            if not hx.startswith(RB_PREFIX):
                raise SystemExit("REFUSED: %s has sha256 %s..., expected %s" % (p, hx[:16], RB_PREFIX))
            d = os.path.dirname(os.path.abspath(p))
            if d not in sys.path:
                sys.path.insert(0, d)
            spec = importlib.util.spec_from_file_location("read_boxread", p)
            m = importlib.util.module_from_spec(spec)
            sys.modules["read_boxread"] = m
            spec.loader.exec_module(m)
            return m
    raise SystemExit("REFUSED: read_boxread.py not found (set BOXREAD_READER)")


RB = load_reader()
ORIG = {k: getattr(RB, k) for k in ("CARD", "LIMITS", "ctl_b0", "ctl_b1", "ctl_b2", "ctl_b4", "input_hashes", "build_parser")}


def _again(res, bad, **kw):
    out = dict(res)
    out["failures"] = list(bad)
    out["pass"] = not bad
    out.update(kw)
    return out


# ============================================================================================ c0
def ctl_c0(a, ctx):
    res = ORIG["ctl_b0"](a, ctx)
    bad = list(res["failures"])
    pj = RB.BS.try_json(a.prior_controls) if getattr(a, "prior_controls", "") else None
    same = {}
    if not isinstance(pj, dict) or pj.get("card") != PRIOR_CARD or not isinstance(pj.get("inputs"), dict):
        bad.append("no controls file of card ADD127 given")
    else:
        for cam in ("wide", "tele"):
            now = RB.dir_listing_sha(os.path.join(a.det_dir, cam), ".npz")
            was = pj["inputs"].get("det_%s_listing" % cam)
            same[cam] = bool(now is not None and now == was)
            if not same[cam]:
                bad.append("the %s detector files are not the ones the run of card ADD127 read" % cam)
    return _again(res, bad, detector_listing_as_card_add127=same)


# ============================================================================================ c1
def ctl_c1(a, ctx):
    res = ORIG["ctl_b1"](a, ctx)
    bad = [f for f in res["failures"] if not f.startswith(INSIDE_LINE)]
    return _again(res, bad, inside_share_decides=False)


# ============================================================================================ c2
SHIFTS = (("left", 1.0, 0.0), ("right", -1.0, 0.0), ("up", 0.0, 1.0))


def displaced_same(a):
    """clause (ii): train frames with a lead within 20 m, the detector run in the wide read, a pick by the label direction. -> counts"""
    store = RB.DetStore(a.det_dir)
    tclips = RB.BS.ids_of(RB.BS.parse_kv(a.list_file)["train_list"])
    n_f = n_c = 0
    same = {k: 0 for k, _, _ in SHIFTS}
    for c in tclips:
        p = os.path.join(a.box_dir, str(c) + ".npz")
        if not os.path.isfile(p):
            continue
        d = store.get("wide", str(c))
        if d is None:
            continue
        z = np.load(p)
        has = z["has_lead"].astype(bool)
        n = len(has)
        gap = np.asarray(z["gap_re"], np.float64)
        az0, el0 = np.asarray(z["ctr_az"], np.float64), np.asarray(z["ctr_el"], np.float64)
        w, h = np.asarray(z["ang_w"], np.float64), np.asarray(z["ang_h"], np.float64)
        use = has & np.isfinite(az0) & np.isfinite(el0) & np.isfinite(w) & np.isfinite(h) & (w > 0) & (h > 0) & np.isfinite(gap) & (gap <= RB.B2_GAP_MAX) & RB.run_of(d, n)
        pk = RB.pick_clip(d, np.where(use, az0, np.nan), np.where(use, el0, np.nan), n)
        m = use & (pk >= 0)
        if not m.any():
            continue
        n_c += 1
        n_f += int(m.sum())
        for k, sa, se in SHIFTS:
            az = np.where(m, az0 + sa * (w / 2.0 + C2_SHIFT_DEG), np.nan)
            el = np.where(m, el0 + se * (h / 2.0 + C2_SHIFT_DEG), np.nan)
            pk2 = RB.pick_clip(d, az, el, n)
            same[k] += int((m & (pk2 == pk)).sum())
    return n_f, n_c, same


def ctl_c2(a, ctx):
    res = ORIG["ctl_b2"](a, ctx)
    bad = [f for f in res["failures"] if not f.startswith(MIRROR_LINE)]
    n_f, n_c, same = displaced_same(a)
    shares = {k: (v / n_f if n_f else None) for k, v in same.items()}
    if n_f < C2_MIN_FRAMES or n_c < C2_MIN_CLIPS:
        bad.append("the displacement clause has %d frames of %d clips (it needs %d of %d)" % (n_f, n_c, C2_MIN_FRAMES, C2_MIN_CLIPS))
    else:
        for k, _, _ in SHIFTS:
            if shares[k] is None or shares[k] > C2_SAME_MAX:
                bad.append("with the direction displaced %s the same detection is picked on %.4f of the frames (bar %.2f)" % (k, shares[k], C2_SAME_MAX))
    return _again(res, bad, mirrored_share_decides=False, displaced_same_share=shares, n_frames_displaced=n_f, n_clips_displaced=n_c, shift_deg=C2_SHIFT_DEG)


# ============================================================================================ c4
def ctl_c4(a, ctx):
    a2 = copy.copy(a)
    a2.locator_sha256 = ""
    res = ORIG["ctl_b4"](a2, ctx)
    bad = list(res["failures"])
    given = (a.locator_sha256 or "").strip().lower()
    hs = RB.sha_of(RB.locator_path(a))
    if not given:
        bad.append("no sha256 of the locator file given")
    elif len(given) < C4_MIN_HEX or not re.fullmatch(r"[0-9a-f]+", given) or hs is None or not hs.startswith(given):
        bad.append("the sha256 of the locator file does not begin with the given characters (at least %d)" % C4_MIN_HEX)
    return _again(res, bad)


# ============================================================================================ inputs, parser, main
def input_hashes_c(a):
    out = ORIG["input_hashes"](a)
    out["read_boxread_c.py"] = _sha(os.path.abspath(__file__))
    pc = getattr(a, "prior_controls", "")
    out["prior_controls"] = _sha(pc) if pc and os.path.isfile(pc) else None
    return out


def build_parser_c():
    ap = ORIG["build_parser"]()
    ap.add_argument("--prior-controls", default="", help="BOXREAD_CONTROLS.json of the controls run of card ADD127 (the detector listings are compared with it)")
    return ap


PATCH = {"CARD": CARD, "LIMITS": ORIG["LIMITS"] + LIMIT_MORE, "ctl_b0": ctl_c0, "ctl_b1": ctl_c1, "ctl_b2": ctl_c2, "ctl_b4": ctl_c4, "input_hashes": input_hashes_c,
         "build_parser": build_parser_c}


def main(argv=None):
    for k, v in PATCH.items():
        setattr(RB, k, v)
    try:
        return RB.main(argv)
    finally:
        for k, v in ORIG.items():
            setattr(RB, k, v)


if __name__ == "__main__":
    sys.exit(main())
