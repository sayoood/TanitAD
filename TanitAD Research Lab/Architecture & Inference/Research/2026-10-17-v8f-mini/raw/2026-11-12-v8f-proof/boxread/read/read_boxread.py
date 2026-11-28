#!/usr/bin/env python
"""read_boxread.py -- read of card PREREG_V8F_PROOF_ADD127 (BOXREAD-T0). CPU, numpy; imports the reader of card ADD124 (read_boxsize.py, sha256 pinned) unchanged.

  python read_boxread.py controls [inputs, see BOXREAD_READ_DEFS.md]
  python read_boxread.py names    [the same inputs]

Tier T0: recorded frames, no rollout; a diagnostic of a perception read and not driving performance.
Mode controls: runs b0 to b7 of card section 5 and prints only `ctl b<k> PASS|FAIL <short reason>` lines (plus the share of b5), then `ZZ BOXREAD CONTROLS ALL-PASS|FAILED`.
  No error, no ratio and no share by bin of an evaluation frame is computed or printed there. b1, b2, b3 run on train clips only; b0 stores pass flags only.
Mode names: refuses (rc 3) unless BOXREAD_CONTROLS.json in the out dir is all-pass with b0..b7 and the inputs hash as they did at the controls; then computes every value of
  card sections 3 and 4, writes BOXREAD_RESULT.json and BOXREAD_VERDICT.md (never over an existing file: a second call writes _2) and prints `ZZ BOXREAD NAME <name>` per name.
Exit codes: 0 all pass / names written, 2 a control failed, 3 refused."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import os
import re
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import box_geom as BG  # noqa: E402

CARD = "PREREG_V8F_PROOF_ADD127"
PRE = "BOXREAD-T0"
V8 = os.environ.get("V8PROOF_ROOT", "/workspace/v8proof")
BS_PREFIX = "b88aecf06dd504c4"
CONTROLS = tuple("b%d" % i for i in range(8))
VEHICLE_LABELS = (3, 4, 6, 8)
SCORE_MIN = 0.5
EDGE_TOL = 1.5
WIDE_WIN = (32.0, 768.0, 0.0, -6.0)
TELE_WIN = (16.0, 1024.0, 0.0, 0.0)
N_BOOT, BOOT_SEED = 2000, 127
PERM_SEED = 127
N_FOLDS = 5
ROW191 = {"D_0": 3.960, "D_1": 3.913, "D_2": 4.112}
TOL_ROW191 = 1e-3
B1_ENC_TOL_DEG = 0.1
B1_INSIDE_MIN = 0.90
B2_PICK_MIN, B2_GAP_MAX, B2_MIRROR_FRAC = 0.80, 20.0, 0.5
B3_MAP_MAX, B3_PERM_MIN = 0.90, 0.95
B5_PICK_MIN = 0.50
BAR_R = 0.90
FAR_HALF = 0.5
ARM_D = ("D_0", "D_1", "D_2")
GAP_BINS = ((0, 10), (10, 20), (20, 40), (40, 60))
LABEL_KEYS = frozenset(["has_lead", "gap_re", "ctr_az", "ctr_el", "bot_az", "bot_el", "ang_w", "ang_h", "rng", "lab_gap", "lab_present", "lead_gap", "lead_valid", "ctr_ego"])
LOCATOR_KEYS = ("clip_index", "frame", "az_pred", "el_pred", "clips")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TIER = "T0 (recorded frames, no rollout; a diagnostic of a perception read and not driving performance)"
FAMILY_LINES = (
    "LONGITUDINAL (distance keeping): the family this read serves; at T0 it is measured as the gap error only.",
    "LATERAL: not measured.",
    "TACTICAL: not measured.",
    "STRATEGIC: not computed, n = 0: a gap read on recorded frames reads no strategic quantity.")
LIMITS = ("One detector, one set of weights, no training on our data; 141 evaluation clips; the map is fitted on 500 train clips; the tele read exists only where the lead is inside the "
          "tele window; frames with no pick take arm D's gap, so the arms are mixtures and the share of picked frames is part of every reading. Sheets with camera frames and boxes "
          "are private and stay off the repository. The interval answers what another draw of evaluation clips would say with the fits held fixed, not what another fit would say.")


# ============================================================================================ the imported reader of card ADD124
def load_boxsize_reader():
    cands = [os.environ.get("BOXSIZE_READER"), V8 + "/boxsize/kit/read/read_boxsize.py", "F:/TanitAD-scratch/v8proof/boxsize/read/read_boxsize.py"]
    for p in cands:
        if p and os.path.isfile(p):
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 20), b""):
                    h.update(b)
            if not h.hexdigest().startswith(BS_PREFIX):
                raise SystemExit("REFUSED: %s has sha256 %s..., expected %s" % (p, h.hexdigest()[:16], BS_PREFIX))
            spec = importlib.util.spec_from_file_location("read_boxsize", p)
            m = importlib.util.module_from_spec(spec)
            sys.modules["read_boxsize"] = m
            spec.loader.exec_module(m)
            m.__boxread_path__ = p
            return m
    raise SystemExit("REFUSED: read_boxsize.py not found (set BOXSIZE_READER)")


BS = load_boxsize_reader()


# ============================================================================================ small helpers
def fresh(path):
    """a path that does not exist yet: path, then path_2, path_3 ..."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 2
    while os.path.exists("%s_%d%s" % (base, i, ext)):
        i += 1
    return "%s_%d%s" % (base, i, ext)


def short(c):
    return str(c)[:8]


def sha_of(p):
    return BS.sha_of(p)


def parse_ref(s):
    return {k: float(v) for k, _, v in (t.partition("=") for t in (s or "").split(",") if t)}


def _res(bad, **kw):
    d = {"pass": not bad, "failures": list(bad)}
    d.update(kw)
    return d


def dir_listing_sha(d, suffix=None):
    try:
        names = sorted(n for n in os.listdir(d) if suffix is None or n.endswith(suffix))
        return hashlib.sha256("".join("%s:%d\n" % (n, os.path.getsize(os.path.join(d, n))) for n in names).encode("utf-8")).hexdigest()
    except OSError:
        return None


# ============================================================================================ detector files
class DetStore:
    """reads <det_dir>/<camera>/<clip>.npz; remembers what it saw (weights hashes, windows, missing clips, frame-count facts)."""

    def __init__(self, det_dir):
        self.root = det_dir
        self.weights = {"wide": set(), "tele": set()}
        self.wins = {"wide": set(), "tele": set()}
        self.missing = {"wide": 0, "tele": 0}
        self.n_read = {"wide": 0, "tele": 0}
        self.no_frame_run = {"wide": 0, "tele": 0}

    def path(self, cam, clip):
        return os.path.join(self.root, cam, str(clip) + ".npz")

    def get(self, cam, clip):
        p = self.path(cam, clip)
        if not os.path.isfile(p):
            self.missing[cam] += 1
            return None
        z = np.load(p, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        d["win"] = np.asarray(d["win"], np.float64).reshape(-1)
        for k in ("camera", "weights_sha256"):
            d[k] = str(np.asarray(d[k]).reshape(-1)[0]) if k in d else ""
        d["n_frames"] = int(np.asarray(d["n_frames"]).reshape(-1)[0])
        self.weights[cam].add(d["weights_sha256"])
        self.wins[cam].add(tuple(float(x) for x in d["win"]))
        self.n_read[cam] += 1
        if "frame_run" in d:
            d["frame_run"] = np.asarray(d["frame_run"], bool).reshape(-1)
        else:
            d["frame_run"] = np.ones(d["n_frames"], bool)
            self.no_frame_run[cam] += 1
        return d


def vehicle_dets(d, n):
    """indices of the detections that count: a vehicle label, score at least 0.5, a frame in 0..n-1 that is ok, a box of positive size."""
    fr = np.asarray(d["det_frame"], np.int64)
    sc = np.asarray(d["det_score"], np.float64)
    lb = np.asarray(d["det_label"], np.int64)
    bx = np.asarray(d["det_box"], np.float64).reshape(-1, 4)
    fok = np.asarray(d["frame_ok"], bool)
    frun = np.asarray(d["frame_run"], bool) if "frame_run" in d else np.ones(len(fok), bool)
    inr = (fr >= 0) & (fr < min(n, len(fok), len(frun)))
    ok = inr & (sc >= SCORE_MIN) & np.isin(lb, VEHICLE_LABELS) & (bx[:, 2] > bx[:, 0]) & (bx[:, 3] > bx[:, 1])
    ok[inr] &= fok[fr[inr]] & frun[fr[inr]]
    return np.flatnonzero(ok)


def run_of(d, n):
    """bool [n]: the detector was run on that cache frame (frame_run and frame_ok); all False when there is no detector file."""
    out = np.zeros(n, bool)
    if d is not None:
        r = np.asarray(d["frame_run"], bool) & np.asarray(d["frame_ok"], bool)
        m = min(n, len(r))
        out[:m] = r[:m]
    return out


def pick_clip(d, az, el, n, edge_tol=EDGE_TOL):
    """card section 3 Pick for every frame of a clip: az, el [n] in the camera of the detector file (nan = no direction). -> det index per frame, -1 for no pick."""
    az, el = np.asarray(az, np.float64), np.asarray(el, np.float64)
    pick = np.full(n, -1, np.int64)
    idx = vehicle_dets(d, n)
    if idx.size == 0:
        return pick
    fr = np.asarray(d["det_frame"], np.int64)[idx]
    order = np.argsort(fr, kind="stable")
    idx, fr = idx[order], fr[order]
    boxes = np.asarray(d["det_box"], np.float64).reshape(-1, 4)
    win = d["win"]
    u, st = np.unique(fr, return_index=True)
    en = list(st[1:]) + [len(fr)]
    for j, s, e in zip(u, st, en):
        if not (np.isfinite(az[j]) and np.isfinite(el[j])):
            continue
        ii = idx[s:e]
        loc = BG.pick_one(boxes[ii], az[j], el[j], win, edge_tol)
        if loc >= 0:
            pick[j] = ii[loc]
    return pick


def sizes_of(d, pick):
    """angular width and height (degrees) of the picked boxes, nan where no pick. -> w [n], h [n]"""
    w = np.full(len(pick), np.nan)
    h = np.full(len(pick), np.nan)
    m = pick >= 0
    if m.any():
        _, _, ww, hh = BG.box_angles_many(np.asarray(d["det_box"], np.float64).reshape(-1, 4)[pick[m]], d["win"])
        w[m], h[m] = ww, hh
    return w, h


# ============================================================================================ the locator's files (vision only)
class LabelFileInLocatorPlace(ValueError):
    pass


def load_locator(path):
    """the banked prediction file of the locator of card ADD107 (az_pred, el_pred in degrees, azimuth left and elevation up, wide camera frame). Refuses a file that holds a label."""
    z = BS.load_npz(path)
    lab = sorted(set(z) & LABEL_KEYS)
    if lab:
        raise LabelFileInLocatorPlace("a label file in the place of the locator's file (keys %s)" % lab[:3])
    miss = [k for k in LOCATOR_KEYS if k not in z]
    if miss:
        raise LabelFileInLocatorPlace("not a locator prediction file (missing keys %s)" % miss)
    return z


def locator_path(a):
    return os.path.join(a.locator_dir, a.locator_fit + ".npz")


def eval_pick_locator(J, loc_az, loc_el, det_for_clip, rel_for_clip, want_tele):
    """BW (and BT) picks on the join rows from the locator's direction ONLY. The signature carries no label argument: no label is read here.
    det_for_clip(cam, clip) -> det dict or None; rel_for_clip(clip) -> (R_wide, R_tele) or None. -> {'wide': {pick, w, h}, 'tele': {pick, w, h}} on the join rows"""
    n = np.asarray(J["n"], np.int64)
    off = np.asarray(J["off"], np.int64)
    nrow = int(n.sum())
    out = {c: {"pick": np.zeros(nrow, bool), "run": np.zeros(nrow, bool), "w": np.full(nrow, np.nan), "h": np.full(nrow, np.nan)} for c in ("wide", "tele")}
    for i, c in enumerate(J["clip"]):
        sl = slice(int(off[i]), int(off[i] + n[i]))
        az, el = loc_az[sl], loc_el[sl]
        dw = det_for_clip("wide", str(c))
        if dw is not None:
            pk = pick_clip(dw, az, el, int(n[i]))
            w, h = sizes_of(dw, pk)
            out["wide"]["pick"][sl], out["wide"]["w"][sl], out["wide"]["h"][sl] = pk >= 0, w, h
            out["wide"]["run"][sl] = run_of(dw, int(n[i]))
        if want_tele:
            dt = det_for_clip("tele", str(c))
            rel = rel_for_clip(str(c)) if dt is not None else None
            if dt is not None and rel is not None:
                ta, te = BG.point_in_other_cam(rel[0], rel[1], az, el)
                pk = pick_clip(dt, ta, te, int(n[i]))
                w, h = sizes_of(dt, pk)
                out["tele"]["pick"][sl], out["tele"]["w"][sl], out["tele"]["h"][sl] = pk >= 0, w, h
                out["tele"]["run"][sl] = run_of(dt, int(n[i]))
    return out


def eval_pick_label(J, box_dir, det_for_clip):
    """BL (the bound): the pick by the LABEL direction in the wide read, on the join rows. Not a candidate form."""
    n = np.asarray(J["n"], np.int64)
    off = np.asarray(J["off"], np.int64)
    nrow = int(n.sum())
    out = {"pick": np.zeros(nrow, bool), "w": np.full(nrow, np.nan), "h": np.full(nrow, np.nan)}
    for i, c in enumerate(J["clip"]):
        p = os.path.join(box_dir, str(c) + ".npz")
        dw = det_for_clip("wide", str(c))
        if dw is None or not os.path.isfile(p):
            continue
        z = np.load(p)
        m = min(len(z["has_lead"]), int(n[i]))
        use = z["has_lead"].astype(bool)[:m] & np.isfinite(z["ctr_az"][:m]) & np.isfinite(z["ctr_el"][:m])
        az = np.full(int(n[i]), np.nan)
        el = np.full(int(n[i]), np.nan)
        az[:m] = np.where(use, np.asarray(z["ctr_az"], np.float64)[:m], np.nan)
        el[:m] = np.where(use, np.asarray(z["ctr_el"], np.float64)[:m], np.nan)
        pk = pick_clip(dw, az, el, int(n[i]))
        w, h = sizes_of(dw, pk)
        sl = slice(int(off[i]), int(off[i] + n[i]))
        out["pick"][sl], out["w"][sl], out["h"][sl] = pk >= 0, w, h
    return out


# ============================================================================================ camera rotations (wide -> tele)
class Rotations:
    def __init__(self, wide_csv, tele_csv):
        self.w = self._read(wide_csv)
        self.t = self._read(tele_csv)

    @staticmethod
    def _read(p):
        out = {}
        if not p or not os.path.isfile(p):
            return out
        with io.open(p, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    out[str(r["clip_id"])] = BG.rot_cam_to_vehicle(float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"]))
                except (KeyError, ValueError):
                    continue
        return out

    def rel(self, clip):
        return (self.w[clip], self.t[clip]) if clip in self.w and clip in self.t else None


# ============================================================================================ train reads by the label direction
def train_clip_read(box_dir, store, clip, cam, rel=None, mirror=False):
    """one train clip: the pick by the LABEL direction (labels are allowed in fitting). -> dict or None (no box file / no detector file)"""
    p = os.path.join(box_dir, clip + ".npz")
    if not os.path.isfile(p):
        return None
    z = np.load(p)
    has = z["has_lead"].astype(bool)
    n = len(has)
    gap = np.asarray(z["gap_re"], np.float64)
    az0, el0 = np.asarray(z["ctr_az"], np.float64), np.asarray(z["ctr_el"], np.float64)
    use = has & np.isfinite(az0) & np.isfinite(el0)
    az = np.where(use, -az0 if mirror else az0, np.nan)
    el = np.where(use, el0, np.nan)
    d = store.get(cam, clip)
    out = {"n": n, "lead": has, "use": use, "gap": gap, "det_ok": d is not None, "n_frames_match": None, "run": run_of(d, n)}
    win = np.asarray(WIDE_WIN if cam == "wide" else TELE_WIN)
    if cam == "tele":
        if d is None or rel is None:
            out["det_ok"] = False
            return out
        az, el = BG.point_in_other_cam(rel[0], rel[1], az, el)
    out["inside"] = BG.label_inside(az, el, d["win"] if d is not None else win) & use
    if d is None:
        return out
    out["n_frames_match"] = d["n_frames"] == n
    pk = pick_clip(d, az, el, n)
    w, h = sizes_of(d, pk)
    out.update({"pick": pk >= 0, "w": w, "h": h})
    return out


def train_reads(box_dir, store, clips, cam, rotations=None, with_mirror=False):
    res = []
    for c in clips:
        rel = rotations.rel(c) if (rotations is not None and cam == "tele") else None
        r = train_clip_read(box_dir, store, c, cam, rel)
        if r is not None and with_mirror and r.get("pick") is not None:
            m = train_clip_read(box_dir, store, c, cam, rel, mirror=True)
            r["pick_mirror"] = m["pick"]
        res.append(r)
    return res


def map_arrays(reads):
    """label-picked train frames: ln h, ln w, gap, clip index (frames with a lead, a finite gap, a pick and a finite positive size)."""
    lnh, lnw, gp, ci = [], [], [], []
    for i, r in enumerate(reads):
        if r is None or r.get("pick") is None:
            continue
        m = r["lead"] & r["run"] & np.isfinite(r["gap"]) & r["pick"] & np.isfinite(r["w"]) & np.isfinite(r["h"]) & (r["w"] > 0) & (r["h"] > 0)
        lnh.append(np.log(r["h"][m]))
        lnw.append(np.log(r["w"][m]))
        gp.append(r["gap"][m])
        ci.append(np.full(int(m.sum()), i, np.int64))
    if not gp:
        z = np.zeros(0)
        return z, z, z, z.astype(np.int64)
    return np.concatenate(lnh), np.concatenate(lnw), np.concatenate(gp), np.concatenate(ci)


def build_map(reads):
    lnh, lnw, gp, _ = map_arrays(reads)
    return BS.MapG(lnh, lnw, gp) if len(gp) else None


# ============================================================================================ controls
def ctl_b0(a, ctx):
    bad = []
    wr = BS.try_json(a.weights_record) if a.weights_record else None
    sha = None
    if not isinstance(wr, dict):
        bad.append("the weights record is missing or unreadable")
    else:
        sha, tag = str(wr.get("sha256", "")).lower(), str(wr.get("name_tag", "")).lower()
        if not HEX64.match(sha):
            bad.append("the weights record has no sha256")
        if len(tag) != 8 or not sha.startswith(tag):
            bad.append("the weights sha256 does not begin with the 8 characters of the file name")
        if wr.get("sha256_begins_with_name_tag") is not True:
            bad.append("the record does not say sha256_begins_with_name_tag true")
        if a.weights_file:
            hs = sha_of(a.weights_file)
            if hs is None or hs != sha:
                bad.append("the weights file does not hash to the record")
    for cam in ("wide", "tele"):
        rp = os.path.join(a.det_dir, cam, "BOXDET_RECORD.json")
        rec = BS.try_json(rp)
        if rec is None:
            if cam == "wide":
                bad.append("no BOXDET_RECORD.json for the wide read")
            continue
        if sha and str(rec.get("weights_sha256", "")).lower() != sha:
            bad.append("%s BOXDET_RECORD weights differ from the weights record" % cam)
        seen = ctx["store"].weights[cam]
        if sha and seen and seen != {sha}:
            bad.append("%s detector files carry %d other weights hash(es)" % (cam, len(seen - {sha}) + (0 if sha in seen else 1)))
    for cam in ("wide", "tele"):
        if ctx["store"].no_frame_run[cam]:
            bad.append("%s detector files without a frame_run key" % cam)
    for cam, want in (("wide", WIDE_WIN), ("tele", TELE_WIN)):
        seen = ctx["store"].wins[cam]
        if seen and seen != {tuple(want)}:
            bad.append("%s detector files carry a window other than the card's" % cam)
    ent = BS._entries(BS.try_json(a.list_record) if a.list_record else None)
    files = {k: sha_of(v) for k, v in BS.parse_kv(a.list_file).items()}
    for need in ("train_list", "eval_list"):
        if need not in files:
            bad.append("no %s passed" % need)
        elif files[need] is None or ent.get(need) != files[need]:
            bad.append("%s differs from the list record of the crop study" % need)
    flags = {}
    ref = parse_ref(a.row191_values)
    L = ctx["L"]
    t3 = BS.t3_rows(L)
    res = None
    if a.boxsize_result:
        rj = BS.try_json(a.boxsize_result)
        try:
            res = {k: float(rj["values"]["E"][k]) for k in ARM_D}
        except (TypeError, KeyError, ValueError):
            res = None
            bad.append("the boxsize result file does not hold the errors of arm D")
    for k in ARM_D:
        e = BS.mean_abs_err(L[k]["gap"], L["D_0"]["lab_gap"], t3)
        ok = e is not None and k in ref and abs(e - ref[k]) <= TOL_ROW191
        if ok and res is not None:
            ok = abs(e - res[k]) <= 1e-6
        flags[k] = bool(ok)
        if not ok:
            bad.append("%s: its error is not the banked value of row 191 within %g" % (k, TOL_ROW191))
    return _res(bad, arm_d_within_tolerance=flags)


def ctl_b1(a, ctx):
    bad = []
    errs = []
    try:
        enc = BG.load_enc_common(a.enc_common or None)
        for win, az, el, w, h in BG.RENDER_CASES:
            r = BG.render_roundtrip(enc, win, az, el, w, h)
            if r is None:
                errs.append(None)
                continue
            errs.append(float(max(abs(r[0] - az), abs(r[1] - el), abs(r[2] - w), abs(r[3] - h))))
        if any(e is None or e > B1_ENC_TOL_DEG for e in errs):
            bad.append("the rendered rectangle does not come back within %.2f degrees" % B1_ENC_TOL_DEG)
    except Exception as ex:  # noqa: BLE001 - a tool test that cannot run is a failed control, with its reason
        bad.append("the render tool test could not run: %s: %s" % (type(ex).__name__, str(ex)[:60]))
    n_lead = n_in = n_notrun = 0
    for r in ctx["train_wide"]:
        if r is None:
            continue
        n_lead += int((r["use"] & r["run"]).sum())
        n_notrun += int((r["use"] & ~r["run"]).sum())
        n_in += int((r["inside"] & r["use"] & r["run"]).sum())
    share = n_in / n_lead if n_lead else None
    if share is None or share < B1_INSIDE_MIN:
        bad.append("the label direction lies inside the wide window on %s of the lead frames (bar %.2f)" % ("none" if share is None else "%.4f" % share, B1_INSIDE_MIN))
    return _res(bad, render_max_err_deg=errs, inside_share=share, n_lead_frames=n_lead, n_lead_frames_not_run_left_out=n_notrun)


def ctl_b2(a, ctx):
    bad = []
    n_t = n_p = n_m = n_notrun = 0
    miss = mism = 0
    for r in ctx["train_wide"]:
        if r is None or r.get("pick") is None:
            miss += 1
            continue
        if r["n_frames_match"] is False:
            mism += 1
        m = r["use"] & r["run"] & np.isfinite(r["gap"]) & (r["gap"] <= B2_GAP_MAX)
        n_notrun += int((r["use"] & ~r["run"] & np.isfinite(r["gap"]) & (r["gap"] <= B2_GAP_MAX)).sum())
        n_t += int(m.sum())
        n_p += int((m & r["pick"]).sum())
        n_m += int((m & r["pick_mirror"]).sum())
    if miss:
        bad.append("%d train clips without a box file or a wide detector file" % miss)
    if mism:
        bad.append("%d train clips whose detector frame count differs from the box file" % mism)
    sp = n_p / n_t if n_t else None
    sm = n_m / n_t if n_t else None
    if sp is None or sp < B2_PICK_MIN:
        bad.append("the pick by the label direction exists on %s of the frames with a lead within 20 m (bar %.2f)" % ("none" if sp is None else "%.4f" % sp, B2_PICK_MIN))
    elif sm > B2_MIRROR_FRAC * sp:
        bad.append("the share with a pick under mirrored directions %.4f is above half of %.4f" % (sm, sp))
    return _res(bad, n_frames=n_t, share=sp, share_mirrored=sm, n_clips_missing=miss, n_clips_frame_count_differs=mism, n_frames_not_run_left_out=n_notrun)


def ctl_b3(a, ctx):
    bad = []
    reads = ctx["train_wide"]
    nclip = len(reads)
    sm = sc = sp = 0.0
    n = 0
    for f in range(N_FOLDS):
        fit = [r for i, r in enumerate(reads) if i % N_FOLDS != f]
        held = [r for i, r in enumerate(reads) if i % N_FOLDS == f]
        lnh, lnw, gp, _ = map_arrays(fit)
        hh, hw, hg, _ = map_arrays(held)
        if len(gp) < 2 or len(hg) == 0:
            bad.append("fold %d has no frames to fit or to read" % f)
            continue
        const = float(np.median(gp))
        G = BS.MapG(lnh, lnw, gp)
        pred = G.predict(np.exp(hw), np.exp(hh))
        perm = np.random.RandomState(PERM_SEED + f).permutation(len(gp))
        Gp = BS.MapG(lnh, lnw, gp[perm])
        predp = Gp.predict(np.exp(hw), np.exp(hh))
        sm += float(np.abs(pred - hg).sum())
        sp += float(np.abs(predp - hg).sum())
        sc += float(np.abs(const - hg).sum())
        n += len(hg)
    em, ep, ec = (sm / n, sp / n, sc / n) if n else (None, None, None)
    if em is None:
        bad.append("no picked train frames for the map")
    else:
        if not em <= B3_MAP_MAX * ec:
            bad.append("the wide map's error is %.4f of the constant's (bar at most %.2f)" % (em / ec, B3_MAP_MAX))
        if not ep >= B3_PERM_MIN * ec:
            bad.append("the map fitted on permuted gaps is %.4f of the constant's (bar at least %.2f)" % (ep / ec, B3_PERM_MIN))
    return _res(bad, n_train_clips=nclip, n_frames=n, ratio_map_to_const=(em / ec if em is not None and ec else None), ratio_perm_to_const=(ep / ec if ep is not None and ec else None))


def ctl_b4(a, ctx):
    bad = []
    lp = locator_path(a)
    try:
        if os.path.realpath(a.locator_dir) == os.path.realpath(a.box_dir) or os.path.realpath(lp).startswith(os.path.realpath(a.box_dir) + os.sep):
            bad.append("the locator folder is the label box folder")
        load_locator(lp)
    except LabelFileInLocatorPlace as ex:
        bad.append(str(ex)[:90])
    except (OSError, ValueError, KeyError) as ex:
        bad.append("the locator file cannot be read: %s" % type(ex).__name__)
    # tool test: the loader refuses a label file; the arms' pick function takes no label argument
    try:
        lab = os.path.join(a.out_dir, ".b4_probe_label.npz")
        np.savez(lab, has_lead=np.zeros(3, bool), ctr_az=np.zeros(3), ctr_el=np.zeros(3), az_pred=np.zeros(3), el_pred=np.zeros(3), clip_index=np.zeros(3), frame=np.zeros(3), clips=np.array(["x"]))
        try:
            load_locator(lab)
            bad.append("the locator loader accepted a label file")
        except LabelFileInLocatorPlace:
            pass
        finally:
            ctx["scratch"].append(lab)
    except OSError as ex:
        bad.append("the loader test could not write its probe file: %s" % type(ex).__name__)
    params = set(inspect.signature(eval_pick_locator).parameters)
    if params & {"box_dir", "lab", "labels", "box", "label"}:
        bad.append("the arms' pick function takes a label argument")
    meta = BS.try_json(os.path.splitext(lp)[0] + "_meta.json")
    hs = sha_of(lp)
    if not isinstance(meta, dict) or hs is None or str(meta.get("pred_sha256", "")).lower() != hs:
        bad.append("the locator file does not hash to the banked meta record of card ADD107")
    if a.locator_sha256 and (hs is None or hs != a.locator_sha256.lower()):
        bad.append("the locator file does not hash to the given sha256")
    return _res(bad, locator_file=os.path.basename(lp), locator_sha256_prefix=(hs or "")[:16])


def ctl_b5(a, ctx):
    bad = []
    t3 = BS.t3_rows(ctx["L"])
    ev = ctx["eval_pick_wide"]
    share = float(ev[t3].mean()) if t3.any() else None
    if ctx["store"].missing["wide"] and ctx.get("eval_missing"):
        bad.append("%d evaluation clips without a wide detector file" % ctx["eval_missing"])
    if share is None or share < B5_PICK_MIN:
        bad.append("share of evaluation frames with a pick in the wide read is %s (bar %.2f)" % ("none" if share is None else "%.4f" % share, B5_PICK_MIN))
    return _res(bad, share=share, n_rows=int(t3.sum()), n_pick=int(ev[t3].sum()))


def ctl_b6(a, ctx):
    bad = []
    tests = ctx["tests"]
    for need in ("det", "read"):
        if need not in tests:
            bad.append("no test output named %s" % need)
    for k, t in tests.items():
        if t.get("text") is None:
            bad.append("%s: no test output" % k)
            continue
        ln = [x for x in t["text"].splitlines() if x.strip()]
        last = ln[-1] if ln else ""
        if not re.search(r"\b\d+ passed\b", last) or re.search(r"failed|error", last, re.I):
            bad.append("%s: the last line is not a passed line" % k)
        if t.get("before") is None or t.get("mtime") is None or not t["mtime"] < t["before"]:
            bad.append("%s: the test output is not older than the read" % k)
    return _res(bad)


def ctl_b7(a, ctx):
    bad = []
    J, L = ctx["J"], ctx["L"]
    t3 = BS.t3_rows(L)
    nclip = len(J["clip"])
    if int(t3.sum()) != a.expect_rows:
        bad.append("%d evaluation rows, expected %d" % (int(t3.sum()), a.expect_rows))
    if nclip != a.expect_clips:
        bad.append("%d evaluation clips, expected %d" % (nclip, a.expect_clips))
    ev_ids = set(BS.ids_of(BS.parse_kv(a.list_file)["eval_list"])) if "eval_list" in BS.parse_kv(a.list_file) else set()
    if ev_ids != set(str(c) for c in J["clip"]):
        bad.append("the evaluation list and the join name different clips")
    for k in ARM_D[1:]:
        for key in ("lab_gap", "lab_present"):
            x, y = L[k][key][t3].astype(np.float64), L["D_0"][key][t3].astype(np.float64)
            if not np.allclose(x, y, rtol=1e-5, atol=1e-5, equal_nan=True):
                bad.append("the label fields of %s differ from D_0 on the rows" % k)
                break
    n = np.asarray(J["n"], np.int64)
    loc = ctx["locator"]
    if loc is not None and not set(str(c) for c in J["clip"]) <= set(str(c) for c in np.asarray(loc["clips"])):
        bad.append("the locator's file does not cover every evaluation clip")
    if ctx["eval_missing"]:
        bad.append("%d evaluation clips without a wide detector file" % ctx["eval_missing"])
    n_notrun = int((t3 & ~ctx["eval_run_wide"]).sum())
    if n_notrun:
        bad.append("%d evaluation rows whose frame has frame_run false in the wide read (a missing input, not a no-pick)" % n_notrun)
    if ctx["eval_short"]:
        bad.append("%d evaluation clips whose detector file has fewer frames than the join" % ctx["eval_short"])
    return _res(bad, n_rows=int(t3.sum()), n_clips=nclip, n_rows_wide_frame_not_run=n_notrun)


# ============================================================================================ context and runs
def dump_paths(a):
    return BS.dump_paths(types.SimpleNamespace(dump=a.d_pred))


def build_context(a, with_eval=True):
    J = BS.load_npz(a.eval_join)
    L = {k: BS.lay_dump(J, BS.load_npz(p)) for k, p in dump_paths(a).items()}
    store = DetStore(a.det_dir)
    tclips = BS.ids_of(BS.parse_kv(a.list_file)["train_list"])
    ctx = {"J": J, "L": L, "store": store, "scratch": []}
    ctx["train_wide"] = train_reads(a.box_dir, store, tclips, "wide", with_mirror=True)
    try:
        ctx["locator"] = load_locator(locator_path(a))
    except (OSError, ValueError, KeyError):
        ctx["locator"] = None
    ctx["eval_missing"] = 0
    ctx["eval_short"] = 0
    nrow = int(np.asarray(J["n"], np.int64).sum())
    ctx["eval_pick_wide"] = np.zeros(nrow, bool)
    ctx["eval_run_wide"] = np.zeros(nrow, bool)
    if with_eval and ctx["locator"] is not None:
        P = BS.lay_pred(J, ctx["locator"])
        cache = {}

        def det_for(cam, c):
            if cam != "wide":
                return None
            if c not in cache:
                cache[c] = store.get("wide", c)
            return cache[c]
        out = eval_pick_locator(J, P["az"], P["el"], det_for, lambda c: None, False)
        ctx["eval_pick_wide"] = out["wide"]["pick"]
        ctx["eval_run_wide"] = out["wide"]["run"]
        n = np.asarray(J["n"], np.int64)
        for i, c in enumerate(J["clip"]):
            d = cache.get(str(c))
            if d is None:
                ctx["eval_missing"] += 1
            elif d["n_frames"] < int(n[i]):
                ctx["eval_short"] += 1
    before = BS.parse_kv(a.test_before, float)
    ctx["tests"] = {}
    for k, p in BS.parse_kv(a.test_out).items():
        ctx["tests"][k] = {"text": BS.read_text(p), "mtime": os.path.getmtime(p) if os.path.isfile(p) else None, "before": before.get(k)}
    return ctx


def input_hashes(a):
    paths = {"join": a.eval_join, "list_record": a.list_record, "weights_record": a.weights_record, "locator": locator_path(a),
             "locator_meta": os.path.splitext(locator_path(a))[0] + "_meta.json", "boxsize_result": a.boxsize_result, "wide_extr": a.wide_extr, "tele_extr": a.tele_extr,
             "boxdet_record_wide": os.path.join(a.det_dir, "wide", "BOXDET_RECORD.json"), "boxdet_record_tele": os.path.join(a.det_dir, "tele", "BOXDET_RECORD.json")}
    for k, p in BS.parse_kv(a.list_file).items():
        paths["list_" + k] = p
    for k, p in dump_paths(a).items():
        paths["dump_" + k] = p
    for k, p in BS.parse_kv(a.test_out).items():
        paths["tests_" + k] = p
    out = {k: (sha_of(p) if p else None) for k, p in paths.items()}
    out["box_dir_listing"] = dir_listing_sha(a.box_dir, ".npz")
    out["det_wide_listing"] = dir_listing_sha(os.path.join(a.det_dir, "wide"), ".npz")
    out["det_tele_listing"] = dir_listing_sha(os.path.join(a.det_dir, "tele"), ".npz")
    out["read_boxread.py"] = BS.sha_file(os.path.abspath(__file__))
    out["box_geom.py"] = BS.sha_file(os.path.join(HERE, "box_geom.py"))
    out["read_boxsize.py"] = BS.sha_file(BS.__boxread_path__)
    return out


def run_controls(a):
    ctx = build_context(a)
    ctl = {}
    ctl["b0"] = ctl_b0(a, ctx)
    ctl["b1"] = ctl_b1(a, ctx)
    ctl["b2"] = ctl_b2(a, ctx)
    ctl["b3"] = ctl_b3(a, ctx)
    ctl["b4"] = ctl_b4(a, ctx)
    ctl["b5"] = ctl_b5(a, ctx)
    ctl["b6"] = ctl_b6(a, ctx)
    ctl["b7"] = ctl_b7(a, ctx)
    allp = sorted(ctl) == sorted(CONTROLS) and all(v["pass"] for v in ctl.values())
    return {"card": CARD, "tier": TIER, "controls": ctl, "all_pass": bool(allp)}, ctx


def controls_main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    try:
        res, ctx = run_controls(a)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as ex:
        why = "the controls could not be computed: %s: %s" % (type(ex).__name__, str(ex)[:60])
        res = {"card": CARD, "tier": TIER, "controls": {k: _res([why]) for k in CONTROLS}, "all_pass": False}
        ctx = None
    try:
        res["inputs"] = input_hashes(a)
    except (OSError, ValueError, TypeError) as ex:
        res["inputs"] = {"error": type(ex).__name__}
    path = fresh(os.path.join(a.out_dir, "BOXREAD_CONTROLS.json"))
    BS.wt(path, json.dumps(res, indent=1, default=BS._jdefault))
    if ctx:
        for p in ctx["scratch"]:
            pass  # the probe file of b4 is left in place (no deletes by this kit); it holds only zeros
    for k in CONTROLS:
        c = res["controls"][k]
        print("ctl %s PASS" % k if c["pass"] else "ctl %s FAIL %s" % (k, (c["failures"][0] if c["failures"] else "")[:100]))
    b5 = res["controls"]["b5"]
    if b5.get("share") is not None:
        print("b5 share %.4f" % b5["share"])
    print("ZZ BOXREAD CONTROLS ALL-PASS" if res["all_pass"] else "ZZ BOXREAD CONTROLS FAILED")
    return 0 if res["all_pass"] else 2


# ============================================================================================ names: the values
def pick_gap(gmap, flag, base):
    return np.where(flag, gmap, base)


def boot_ratio(num, den, n_boot, seed):
    return BS.boot_ratio(num, den, n_boot, seed)


def fv(x, d=3):
    return "None" if x is None else ("%." + str(d) + "f") % x


def fiv(iv):
    return "[%.4f, %.4f]" % (iv[0], iv[1]) if iv else "None"


def name_ratio(tag, iv, R):
    if iv is None or R is None:
        return None
    if iv[1] < 1.0:
        return PRE + "-" + tag + ("-BETTER" if R <= BAR_R else "-SMALL-GAIN")
    if iv[0] > 1.0:
        return PRE + "-" + tag + "-WORSE"
    return PRE + "-" + tag + "-NOT-SEPARATED"


def name_far(sBW, sD, iv):
    if sBW is not None and sD is not None and iv is not None and iv[0] > 0.0 and abs(sBW) <= FAR_HALF * abs(sD):
        return PRE + "-FAR-LEADS-LESS-SHORT"
    return PRE + "-FAR-LEADS-STAY-SHORT"


def compute_values(J, L, reads, Gw, Gt, n_boot=N_BOOT, seed=BOOT_SEED):
    """every value of sections 3 and 4. reads = {'wide': {pick, w, h}, 'tele': {...}, 'BL': {...}} on the join rows."""
    n = np.asarray(J["n"], np.int64)
    nc = len(n)
    ci_all = np.repeat(np.arange(nc), n)
    t3 = BS.t3_rows(L)
    rows = np.flatnonzero(t3)
    ci, labg = ci_all[rows], L["D_0"]["lab_gap"][rows]
    gD = {k: L[k]["gap"][rows] for k in ARM_D}

    def gmap(G, rd):
        p = rd["pick"][rows]
        g = np.full(len(rows), np.nan)
        if G is not None and p.any():
            g[p] = G.predict(rd["w"][rows][p], rd["h"][rows][p])
        return g, p & np.isfinite(g)
    gw, pw = gmap(Gw, reads["wide"])
    gt, pt = gmap(Gt, reads["tele"])
    gl, pl = gmap(Gw, reads["BL"])
    gaps = {}
    for i, k in enumerate(ARM_D):
        gaps[k] = gD[k]
        gaps["BW_%d" % i] = pick_gap(gw, pw, gD[k])
        gaps["BT_%d" % i] = pick_gap(gt, pt, gaps["BW_%d" % i])
        gaps["BL_%d" % i] = pick_gap(gl, pl, gD[k])
    for k, g in gaps.items():
        if len(g) != len(rows) or not np.isfinite(g).all():
            raise ValueError("arm %s does not cover the same finite rows" % k)
    absk = {k: np.abs(g - labg) for k, g in gaps.items()}
    sgn = {k: g - labg for k, g in gaps.items()}
    sums = {k: np.bincount(ci, weights=absk[k], minlength=nc) for k in gaps}

    def fits(X):
        return ["%s_%d" % (X, i) for i in range(3)] if X != "D" else list(ARM_D)

    def msum(X):
        return sum(sums[k] for k in fits(X)) / 3.0

    P = {"n_clips": int(nc), "n_rows": int(len(rows)), "n_boot": n_boot, "seed": seed, "tier": TIER,
         "interval_level": "two-sided 95 % percentile, cluster bootstrap over the evaluation clips, fits held fixed (it answers what another draw of clips would say, not what another fit would say)"}
    P["E"] = {k: BS.fnum(absk[k].mean()) for k in gaps}
    P["mean_E"] = {X: BS.fnum(np.mean([P["E"][k] for k in fits(X)])) for X in ("D", "BW", "BT", "BL")}
    P["R"], P["iv"] = {}, {}
    for X in ("BW", "BT", "BL"):
        P["R"][X] = BS.fnum(P["mean_E"][X] / P["mean_E"]["D"]) if P["mean_E"]["D"] else None
        P["iv"][X] = boot_ratio(msum(X), msum("D"), n_boot, seed)
    P["R"]["BT_over_BW"] = BS.fnum(P["mean_E"]["BT"] / P["mean_E"]["BW"]) if P["mean_E"]["BW"] else None
    P["iv"]["BT_over_BW"] = boot_ratio(msum("BT"), msum("BW"), n_boot, seed)
    far_m = BS.gap_bin_mask(labg, 40, 60)
    far_n = np.bincount(ci[far_m], minlength=nc).astype(np.float64)
    far_s = {k: np.bincount(ci[far_m], weights=sgn[k][far_m], minlength=nc) for k in gaps}
    nfar = float(far_n.sum())
    sf = {X: (float(np.mean([far_s[k].sum() / nfar for k in fits(X)])) if nfar > 0 else None) for X in ("D", "BW", "BT", "BL")}
    P["far"] = {"n": int(nfar), "n_clips": int((far_n > 0).sum()), "signed": sf,
                "signed_single": {k: (BS.fnum(far_s[k].sum() / nfar) if nfar > 0 else None) for k in gaps},
                "iv_diff_BW_minus_D": BS.boot_diff_over(sum(far_s[k] for k in fits("BW")) / 3.0, sum(far_s[k] for k in fits("D")) / 3.0, far_n, n_boot, seed) if nfar > 0 else None}
    # the per-clip sums behind the intervals
    P["sums"] = {"abs": {k: sums[k].tolist() for k in gaps}, "far_n": far_n.tolist()}
    # beside, naming nothing
    bs = {}
    bs["BL"] = {"R": P["R"]["BL"], "iv": P["iv"]["BL"], "E_single": {k: P["E"][k] for k in fits("BL")}}
    flags = {"BW": pw, "BT": pt, "BL": pl}
    bs["share_picked"] = {}
    for X, f in flags.items():
        row = {"all": {"n": int(len(f)), "picked": int(f.sum()), "share": BS.fnum(f.mean()) if len(f) else None}, "by_gap_bin": []}
        for lo, hi in GAP_BINS:
            m = BS.gap_bin_mask(labg, lo, hi)
            row["by_gap_bin"].append({"bin": "%d-%d m" % (lo, hi), "n": int(m.sum()), "picked": int((f & m).sum()), "share": BS.fnum((f & m).mean() if False else ((f & m).sum() / m.sum()) if m.any() else None)})
        bs["share_picked"][X] = row
    bs["picked_frames_alone"] = {}
    for X in ("BW", "BT", "BL"):
        f = flags[X]
        if f.any():
            bs["picked_frames_alone"][X] = {"n": int(f.sum()), "E_arm": BS.fnum(np.mean([absk["%s_%d" % (X, i)][f].mean() for i in range(3)])),
                                            "E_D_same_frames": BS.fnum(np.mean([absk[k][f].mean() for k in ARM_D]))}
        else:
            bs["picked_frames_alone"][X] = {"n": 0, "E_arm": None, "E_D_same_frames": None}
    bs["size_error_pct"] = {}
    # label sizes on the evaluation rows
    lw, lh = reads["lab_w"][rows], reads["lab_h"][rows]
    for X, rd, f in (("BL", reads["BL"], pl), ("BW", reads["wide"], pw), ("BT", reads["tele"], pt)):
        dw, dh = rd["w"][rows], rd["h"][rows]
        m_all = f & np.isfinite(lw) & np.isfinite(lh) & (lw > 0) & (lh > 0)
        out = []
        for lo, hi in GAP_BINS + ((None, None),):
            m = m_all if lo is None else (m_all & BS.gap_bin_mask(labg, lo, hi))
            nm = "all" if lo is None else "%d-%d m" % (lo, hi)
            if not m.any():
                out.append({"bin": nm, "n": 0, "w": None, "h": None})
                continue
            out.append({"bin": nm, "n": int(m.sum()), "w": BS.fnum(np.abs(np.log(dw[m]) - np.log(lw[m])).mean() * 100.0), "h": BS.fnum(np.abs(np.log(dh[m]) - np.log(lh[m])).mean() * 100.0)})
        bs["size_error_pct"][X] = out
    P["beside"] = bs
    P["bins"] = {}
    for X in ("D", "BW", "BT", "BL"):
        out = []
        for lo, hi in GAP_BINS:
            m = BS.gap_bin_mask(labg, lo, hi)
            if not m.any():
                out.append({"bin": "%d-%d m" % (lo, hi), "n": 0, "E": None, "signed": None})
                continue
            out.append({"bin": "%d-%d m" % (lo, hi), "n": int(m.sum()), "E": BS.fnum(np.mean([absk[k][m].mean() for k in fits(X)])), "signed": BS.fnum(np.mean([sgn[k][m].mean() for k in fits(X)]))})
        P["bins"][X] = out
    return P


def derive_names(P):
    names = []
    for tag, X in (("WIDE", "BW"), ("TELE", "BT")):
        names.append(name_ratio(tag, P["iv"][X], P["R"][X]) or PRE + "-" + tag + "-NOT-SEPARATED")
    iv = P["iv"]["BT_over_BW"]
    names.append(PRE + ("-TELE-ADDS" if iv is not None and iv[1] < 1.0 else "-TELE-DOES-NOT-ADD"))
    names.append(name_far(P["far"]["signed"]["BW"], P["far"]["signed"]["D"], P["far"]["iv_diff_BW_minus_D"]))
    return names


def verdict_lines(P, names, counts):
    L = ["# BOXREAD-T0 verdict (card %s)" % CARD, "", "Tier: " + TIER, ""]
    L += ["## Families", ""] + ["* " + x for x in FAMILY_LINES] + ["", "## Names", ""] + ["* `%s`" % n for n in names]
    L += ["", "## Measure: mean absolute gap error on %d evaluation frames of %d clips (m)" % (P["n_rows"], P["n_clips"]), ""]
    L.append("* arm D, each fit singly: " + "; ".join("%s %s" % (k, fv(P["E"][k])) for k in ARM_D) + "; mean %s" % fv(P["mean_E"]["D"], 4))
    for X in ("BW", "BT", "BL"):
        L.append("* arm %s, formed against each fit: " % X + "; ".join("%s_%d %s" % (X, i, fv(P["E"]["%s_%d" % (X, i)])) for i in range(3)) + "; mean %s" % fv(P["mean_E"][X], 4)
                 + "; R = %s, interval %s" % (fv(P["R"][X], 4), fiv(P["iv"][X])))
    L.append("* BT divided by BW: %s, interval %s" % (fv(P["R"]["BT_over_BW"], 4), fiv(P["iv"]["BT_over_BW"])))
    L.append("* interval: %d draws, RandomState(%d); %s" % (P["n_boot"], P["seed"], P["interval_level"]))
    fr = P["far"]
    L += ["", "## Far leads (label gap 40 to 60 m): n %d in %d clips" % (fr["n"], fr["n_clips"]), "",
          "* mean signed error (read minus label), mean of three fits: D %s, BW %s, BT %s, BL %s" % tuple(fv(fr["signed"][x]) for x in ("D", "BW", "BT", "BL")),
          "* paired difference BW minus D, interval %s" % fiv(fr["iv_diff_BW_minus_D"])]
    L += ["", "## Beside, naming nothing", ""]
    bs = P["beside"]
    L.append("* bound arm BL: R %s, interval %s" % (fv(bs["BL"]["R"], 4), fiv(bs["BL"]["iv"])))
    for X in ("BW", "BT", "BL"):
        s = bs["share_picked"][X]
        L.append("* share of evaluation frames with a pick, %s: all %s (%d of %d); by label gap: " % (X, fv(s["all"]["share"], 4), s["all"]["picked"], s["all"]["n"])
                 + "; ".join("%s %s (n %d)" % (b["bin"], fv(b["share"], 4), b["n"]) for b in s["by_gap_bin"]))
    for X in ("BW", "BT", "BL"):
        s = bs["picked_frames_alone"][X]
        L.append("* error on the %s-picked frames alone (n %d): arm %s %s, arm D on the same frames %s" % (X, s["n"], X, fv(s["E_arm"]), fv(s["E_D_same_frames"])))
    for X in ("BL", "BW", "BT"):
        L.append("* detector size against label size, %s-picked frames, mean absolute error of ln width and ln height in per cent: " % X
                 + "; ".join("%s n %d width %s height %s" % (b["bin"], b["n"], fv(b["w"], 1), fv(b["h"], 1)) for b in bs["size_error_pct"][X]))
    for X in ("D", "BW", "BT", "BL"):
        L.append("* by label gap, arm %s (mean of three fits): " % X + "; ".join("%s n %d E %s signed %s" % (b["bin"], b["n"], fv(b["E"]), fv(b["signed"])) for b in P["bins"][X]))
    L += ["", "## Counts", "", "* " + json.dumps(counts, default=BS._jdefault, sort_keys=True)]
    L += ["", "## Limits of the card", "", LIMITS, ""]
    return L


def check_inputs(a, cj):
    now = input_hashes(a)
    was = cj.get("inputs") or {}
    return sorted(k for k in set(now) | set(was) if now.get(k) != was.get(k))


def names_main(a):
    cpath = os.path.join(a.out_dir, "BOXREAD_CONTROLS.json")
    cj = BS.try_json(cpath) or {}
    cs = cj.get("controls", {}) if isinstance(cj.get("controls"), dict) else {}
    if cj.get("card") != CARD or cj.get("all_pass") is not True or sorted(cs) != sorted(CONTROLS) or any(v.get("pass") is not True for v in cs.values()):
        print("REFUSED: the controls file in the out dir is not all-pass with b0 to b7 present: no name and no value is opened")
        return 3
    diff = check_inputs(a, cj)
    if diff:
        print("REFUSED: inputs differ from those of the controls: %s" % diff[:6])
        return 3
    J = BS.load_npz(a.eval_join)
    L = {k: BS.lay_dump(J, BS.load_npz(p)) for k, p in dump_paths(a).items()}
    store = DetStore(a.det_dir)
    rot = Rotations(a.wide_extr, a.tele_extr)
    tclips = BS.ids_of(BS.parse_kv(a.list_file)["train_list"])
    tw = train_reads(a.box_dir, store, tclips, "wide")
    tt = train_reads(a.box_dir, store, tclips, "tele", rot)
    Gw, Gt = build_map(tw), build_map(tt)
    if Gw is None:
        print("REFUSED: no wide map could be fitted")
        return 3
    if Gt is None:
        Gt = None  # no tele train frames: BT falls back to BW everywhere
    loc = load_locator(locator_path(a))
    Pz = BS.lay_pred(J, loc)
    cache = {}

    def det_for(cam, c):
        if (cam, c) not in cache:
            cache[(cam, c)] = store.get(cam, c)
        return cache[(cam, c)]
    rd = eval_pick_locator(J, Pz["az"], Pz["el"], det_for, rot.rel, True)
    reads = {"wide": rd["wide"], "tele": rd["tele"], "BL": eval_pick_label(J, a.box_dir, det_for)}
    lab = BS.box_rows(J, a.box_dir)
    reads["lab_w"], reads["lab_h"] = lab["w"], lab["h"]
    P = compute_values(J, L, reads, Gw, Gt, a.B)
    names = derive_names(P)
    t3 = BS.t3_rows(L)
    counts = {"n_rows": P["n_rows"], "n_clips": P["n_clips"], "n_train_clips": len(tclips),
              "n_map_frames_wide": int(Gw.n), "n_map_frames_tele": int(Gt.n) if Gt is not None else 0,
              "n_eval_clips_with_tele_file": int(sum(1 for c in J["clip"] if os.path.isfile(store.path("tele", str(c))))),
              "n_eval_rows_with_tele_pick": int(reads["tele"]["pick"][t3].sum()), "n_eval_rows_with_wide_pick": int(reads["wide"]["pick"][t3].sum()),
              "n_eval_rows_with_label_pick": int(reads["BL"]["pick"][t3].sum())}
    res = {"card": CARD, "tier": TIER, "names": names, "measures": {k: v for k, v in P.items() if k not in ("beside", "sums")}, "beside": P["beside"], "counts": counts,
           "input_hashes": input_hashes(a), "controls_sha256": BS.sha_file(cpath), "sums": P["sums"]}
    rp, vp = fresh(os.path.join(a.out_dir, "BOXREAD_RESULT.json")), fresh(os.path.join(a.out_dir, "BOXREAD_VERDICT.md"))
    BS.wt(rp, json.dumps(res, indent=1, default=BS._jdefault))
    BS.wt(vp, "\n".join(verdict_lines(P, names, counts)))
    for n in names:
        print("ZZ BOXREAD NAME %s" % n)
    return 0


# ============================================================================================ command line
def build_parser():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("mode", choices=("controls", "names"))
    ap.add_argument("--eval-join", default=V8 + "/hierrep/pins/join_eval141.npz")
    ap.add_argument("--box-dir", default=V8 + "/cropread/enc_out/box")
    ap.add_argument("--list-record", default="")
    ap.add_argument("--list-file", action="append", default=[], help="NAME=PATH; train_list and eval_list are required")
    ap.add_argument("--d-pred", "--dump", dest="d_pred", action="append", default=[], help="NAME=PATH of one banked fit of arm D (D_0, D_1, D_2); the model reader's defaults are the banked paths")
    ap.add_argument("--det-dir", default=V8 + "/boxread/det_out", help="<det-dir>/<wide|tele>/<clip>.npz and BOXDET_RECORD.json")
    ap.add_argument("--locator-dir", default=V8 + "/lookat/out/pred", help="folder of the banked prediction files of card ADD107")
    ap.add_argument("--locator-fit", default="LOC_s0", help="file stem of the locator's prediction file")
    ap.add_argument("--locator-sha256", default="", help="expected sha256 of that file (optional; its _meta.json record is always checked)")
    ap.add_argument("--weights-record", default="", help="WEIGHTS_RECORD.json: sha256, name_tag, sha256_begins_with_name_tag")
    ap.add_argument("--weights-file", default="", help="optional: the weights file itself, hashed against the record")
    ap.add_argument("--boxsize-result", default="", help="BOXSIZE_RESULT.json of card ADD124; only values.E of arm D is read")
    ap.add_argument("--row191-values", default=",".join("%s=%s" % kv for kv in ROW191.items()))
    ap.add_argument("--wide-extr", default="", help="front_wide_extrinsics.csv (clip_id, qx, qy, qz, qw): carries a wide direction into the tele camera")
    ap.add_argument("--tele-extr", default="", help="front_tele_extrinsics.csv")
    ap.add_argument("--enc-common", default="", help="enc_common.py of card ADD99 (env BOXREAD_ENC_COMMON also)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-out", action="append", default=[], help="NAME=PATH; names det and read")
    ap.add_argument("--test-before", action="append", default=[], help="NAME=EPOCH")
    ap.add_argument("--B", type=int, default=N_BOOT)
    ap.add_argument("--expect-rows", type=int, default=3841)
    ap.add_argument("--expect-clips", type=int, default=141)
    return ap


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    a = build_parser().parse_args(argv)
    return controls_main(a) if a.mode == "controls" else names_main(a)


if __name__ == "__main__":
    sys.exit(main())
