#!/usr/bin/env python
"""boxsize_fit.py -- size head of card PREREG_V8F_PROOF_ADD124 (BOXSIZE-T0), section 2. One fit per call. Tier T0.

  python boxsize_fit.py [--locator P] <the arguments of locator_fit.py>

Imports locator_fit.py (card ADD107) unchanged from --locator and sets ONE value of the imported module for the time of the run: load_box.
The replacement reads the same box file (<box-dir>/<clip>.npz: has_lead, ctr_az, ctr_el, ang_w, ang_h) and returns (has, az, el) in the shape and order of the original, with
  has = has_lead and finite ctr_az, ctr_el and finite positive ang_w, ang_h;   first value = 16 u_w;   second value = -2 + 8 u_h;   u = ln(angle in degrees) - 1.6.
The targets of the imported tool are then u_w and u_h limited to -2..+2 and its read-out returns the same encoding. Everything else is the imported main, called with the remaining argv.
Afterwards BOXSIZE_FIT_RECORD.json is written in the fit folder (identity run: BOXSIZE_IDENTITY_RECORD.json in the out root).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

CARD, TIER = "PREREG_V8F_PROOF_ADD124", "T0"
DEFAULT_LOCATOR = "/workspace/v8proof/lookat/kit/locator_fit.py"
U_OFF = 1.6
U_LIM = 2.0
LIMITS = (float(np.exp(U_OFF - U_LIM)), float(np.exp(U_OFF + U_LIM)))      # 0.67 .. 36.6 degrees
RECORD_KEYS = ("card", "tier", "module_values_changed", "module_values_restored", "locator_fit_sha256", "tool_sha256", "argv", "seed", "permute_targets", "finished_utc")
_CACHE = {}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def encode(ang_w, ang_h):
    """angles in degrees -> (a, b) = (16 u_w, -2 + 8 u_h), u = ln(angle) - 1.6."""
    uw = np.log(np.asarray(ang_w, np.float64)) - U_OFF
    uh = np.log(np.asarray(ang_h, np.float64)) - U_OFF
    return 16.0 * uw, -2.0 + 8.0 * uh


def decode(a, b):
    """inverse of encode, no clipping -> (ang_w, ang_h) in degrees."""
    uw = np.asarray(a, np.float64) / 16.0
    uh = (np.asarray(b, np.float64) + 2.0) / 8.0
    return np.exp(uw + U_OFF), np.exp(uh + U_OFF)


def box_size_loader(box_dir, clip, n):
    """replacement of locator_fit.load_box: -> (has bool[n], first[n], second[n]); refuses a missing file and a frame count other than n."""
    p = os.path.join(box_dir, clip + ".npz")
    if not os.path.exists(p):
        raise SystemExit("REFUSED: no box file for %s in %s" % (clip, box_dir))
    z = np.load(p)
    w, h = z["ang_w"].astype(np.float64), z["ang_h"].astype(np.float64)
    has = z["has_lead"].astype(bool) & np.isfinite(z["ctr_az"]) & np.isfinite(z["ctr_el"]) & np.isfinite(w) & np.isfinite(h)
    has = has & (np.where(np.isfinite(w), w, -1.0) > 0) & (np.where(np.isfinite(h), h, -1.0) > 0)
    if len(has) != n:
        raise SystemExit("REFUSED: box file of %s has %d frames, the token file %d" % (clip, len(has), n))
    a, b = encode(np.where(has, w, 1.0), np.where(has, h, 1.0))
    return has, np.where(has, a, 0.0).astype(np.float64), np.where(has, b, 0.0).astype(np.float64)


def import_locator(path):
    """imports locator_fit.py from path (once per path) and returns the module."""
    path = os.path.abspath(path)
    if path in _CACHE:
        return _CACHE[path]
    if not os.path.isfile(path):
        raise SystemExit("REFUSED: no locator_fit.py at %s" % path)
    spec = importlib.util.spec_from_file_location("locator_fit_add107", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHE[path] = mod
    return mod


def run(argv):
    """the whole tool; returns the return code of the imported main. The module value is restored in a finally block."""
    argv = list(argv)
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--locator", default=DEFAULT_LOCATOR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default="")
    ap.add_argument("--permute-targets", action="store_true")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--predict", action="store_true")
    a, _ = ap.parse_known_args(argv)
    # the remaining argv of the imported main: everything but --locator and its value (the other options are read by it too)
    rest, skip = [], False
    for t in argv:
        if skip:
            skip = False
            continue
        if t == "--locator":
            skip = True
            continue
        if t.startswith("--locator="):
            continue
        rest.append(t)
    LF = import_locator(a.locator)
    original = LF.load_box
    LF.load_box = box_size_loader
    try:
        rc = LF.main(rest)
    finally:
        LF.load_box = original
    restored = LF.load_box is original
    if not a.out_root:
        return rc
    if a.identity:
        rec_path = os.path.join(a.out_root, "BOXSIZE_IDENTITY_RECORD.json")
    else:
        rec_path = os.path.join(a.out_root, "fits", LF.fit_name(a.seed, a.permute_targets), "BOXSIZE_FIT_RECORD.json")
    rec = {"card": CARD, "tier": TIER, "module_values_changed": [{"name": "load_box"}], "module_values_restored": bool(restored),
           "locator_fit_sha256": sha256_file(os.path.abspath(a.locator)), "tool_sha256": sha256_file(os.path.abspath(__file__)), "argv": list(argv),
           "seed": None if a.identity else int(a.seed), "permute_targets": bool(a.permute_targets), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if os.path.isdir(os.path.dirname(rec_path)) and not os.path.exists(rec_path):      # never overwrites; a predict-only call keeps the record of the fit
        with open(rec_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rec, fh, indent=1)
    return rc


def main(argv=None):
    return run(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    sys.exit(main())
