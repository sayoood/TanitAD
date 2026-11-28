#!/usr/bin/env python
"""tele_fit.py -- fit of card PREREG_V8F_PROOF_ADD122 (TELEGAP-T0), note ADD122_N1. Tier T0.

  python tele_fit.py --arm M|F|D2 --head H5 --seed K [--permute-labels]
         --wide-tok-dir DIR --tele-tok-dir DIR --match-dir DIR [--out-root DIR] [--lists-dir DIR] [--join-train NPZ]
         [--kit-fit DIR] [--device cuda]

The fit tool of card ADD99 (crop_fit.py) is IMPORTED unchanged (never edited, never copied). This tool sets a few module values of it for the arm, runs crop_fit's own run_fit with
crop_fit's own hyper-parameters, and restores the module values. Arms (windows in token order):
  D2  wide W32, W16            (identity arm: the two windows of the banked wide tokens, read through this code path; nothing dropped)
  M   wide W16, tele T8        (window index 1 of the wide file, window index 0 of the tele file)
  F   wide W32, W16, tele T8, T4
The combined arrays are never written to disk: a lazy view reads both token folders with memory maps. A sample whose run of frames (H5: f-4..f, H1: f) contains a frame with
mask false (match/<clip>.npz key "mask") is dropped, for M and F, in the fit and in the dump.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD122"
HERE = os.path.dirname(os.path.abspath(__file__))
REFUSED = 2
DEFAULT_KIT_FIT = "/workspace/v8proof/cropread/kit/fit"
DEFAULT_OUT_ROOT = "/workspace/v8proof/telegap/fit_out"
DEFAULT_LISTS = "/workspace/v8proof/cropread/lists"
TELE_ARMS = ("M", "F", "D2")
# (source, window index in that source's token file); order = token order
ARM_WINDOWS = {
    "D2": (("wide", 0), ("wide", 1)),
    "M": (("wide", 1), ("tele", 0)),
    "F": (("wide", 0), ("wide", 1), ("tele", 0), ("tele", 1)),
}
ARM_WINDOW_NAMES = {"D2": ["W32", "W16"], "M": ["W16", "T8"], "F": ["W32", "W16", "T8", "T4"]}


def refuse(msg):
    print("REFUSED: " + msg, file=sys.stderr)
    raise SystemExit(REFUSED)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ============================================================================================ the unchanged tools
def load_kit(kit_fit=None):
    """-> (kit dir, crop_fit module, crop_dump module), both imported from the same folder. kit_fit None: DEFAULT_KIT_FIT, else ../../cropread/fit beside this file."""
    if kit_fit:
        cands = [kit_fit]
    else:
        cands = [DEFAULT_KIT_FIT, os.path.normpath(os.path.join(HERE, "..", "..", "cropread", "fit"))]
    kit = None
    for c in cands:
        if os.path.isfile(os.path.join(c, "crop_fit.py")) and os.path.isfile(os.path.join(c, "crop_dump.py")):
            kit = os.path.abspath(c)
            break
    if kit is None:
        refuse("no crop_fit.py and crop_dump.py in " + " or ".join(cands))
    if kit in sys.path:
        sys.path.remove(kit)
    sys.path.insert(0, kit)
    import crop_fit as CFT          # noqa: E402
    import crop_dump as CDP         # noqa: E402
    for m in (CFT, CDP):
        if os.path.dirname(os.path.abspath(m.__file__)) != kit:
            refuse("%s was imported from %s, not from %s" % (m.__name__, m.__file__, kit))
    if CDP.CFT is not CFT:
        refuse("crop_dump imported another crop_fit than this tool")
    return kit, CFT, CDP


# ============================================================================================ the combined token view
class CombinedTokens:
    """read-only lazy view [n, n_win, N_TOK_WIN, WIDTH] float16 over memory maps; parts = [(array [n, 2, 65, 1024], window index)] in token order.
    Slicing materialises only the slice (so a clip is never held whole)."""

    def __init__(self, parts, tail):
        self.parts = list(parts)
        self.shape = (int(self.parts[0][0].shape[0]), len(self.parts)) + tuple(tail)
        self.ndim = 4
        self.dtype = np.dtype(np.float16)

    def __len__(self):
        return self.shape[0]

    def _block(self, start, stop):
        out = np.empty((stop - start,) + self.shape[1:], np.float16)
        for j, (arr, w) in enumerate(self.parts):
            out[:, j] = arr[start:stop, w]
        return out

    def __getitem__(self, key):
        n = self.shape[0]
        if isinstance(key, (int, np.integer)):
            k = int(key) + (n if int(key) < 0 else 0)
            if not 0 <= k < n:
                raise IndexError("frame %d out of range for %d frames" % (int(key), n))
            return self._block(k, k + 1)[0]
        if isinstance(key, slice):
            start, stop, step = key.indices(n)
            if step == 1:
                return self._block(start, max(start, stop))
        rows = np.arange(n)[key]
        out = np.empty((len(rows),) + self.shape[1:], np.float16)
        for i, r in enumerate(rows):
            out[i] = self._block(int(r), int(r) + 1)[0]
        return out


def drop_frames_of(mask, head, hist):
    """bool [n]: frame f is dropped when its run (H5: f-hist+1..f clipped at 0, H1: f) contains a frame with mask false."""
    bad = ~np.asarray(mask, bool)
    nf = hist if head == "H5" else 1
    c = np.concatenate([[0], np.cumsum(bad.astype(np.int64))])
    f = np.arange(len(bad))
    lo = np.maximum(0, f - nf + 1)
    return (c[f + 1] - c[lo]) > 0


class TeleStore:
    """token store of one arm. Same interface as crop_fit.TokenStore for what crop_fit and crop_dump use: get(clip) -> array-like [n, n_win, 65, 1024] float16."""

    def __init__(self, CFT, arm, wide_dir, tele_dir=None, match_dir=None):
        if arm not in TELE_ARMS:
            refuse("arm %s is not one of %s" % (arm, TELE_ARMS))
        self.CFT, self.arm = CFT, arm
        self.wide_dir, self.tele_dir, self.match_dir = wide_dir, tele_dir, match_dir
        self.uses_tele = any(s == "tele" for s, _ in ARM_WINDOWS[arm])
        self.maps, self.masks, self.views = {}, {}, {}
        if not os.path.isdir(wide_dir):
            refuse("no wide token folder " + str(wide_dir))
        if self.uses_tele:
            for d, what in ((tele_dir, "tele token folder"), (match_dir, "match folder")):
                if not d or not os.path.isdir(d):
                    refuse("no %s %s" % (what, d))

    def _open(self, d, clip):
        key = (d, clip)
        a = self.maps.get(key)
        if a is None:
            p = os.path.join(d, clip + ".npy")
            if not os.path.isfile(p):
                refuse("no token file " + p)
            a = np.load(p, mmap_mode="r")
            C = self.CFT
            if a.ndim != 4 or a.shape[1:] != (2, C.N_TOK_WIN, C.WIDTH) or a.dtype != np.float16:
                refuse("tokens %s have shape %s dtype %s, expected [n, 2, %d, %d] float16" % (p, list(a.shape), a.dtype, C.N_TOK_WIN, C.WIDTH))
            self.maps[key] = a
        return a

    def mask(self, clip):
        """bool [n] (true = a tele frame exists) or None for an arm that reads no tele window."""
        if not self.uses_tele:
            return None
        m = self.masks.get(clip)
        if m is None:
            p = os.path.join(self.match_dir, clip + ".npz")
            if not os.path.isfile(p):
                refuse("no match file " + p)
            z = np.load(p, allow_pickle=False)
            if "mask" not in z.files:
                refuse("match file %s has no key mask" % p)
            m = np.asarray(z["mask"])
            n = self._open(self.wide_dir, clip).shape[0]
            if m.dtype != np.bool_ or m.ndim != 1 or len(m) != n:
                refuse("mask of %s has dtype %s shape %s, expected bool [%d]" % (clip, m.dtype, list(m.shape), n))
            self.masks[clip] = m
        return m

    def get(self, clip):
        v = self.views.get(clip)
        if v is None:
            wide = self._open(self.wide_dir, clip)
            parts = []
            for src, w in ARM_WINDOWS[self.arm]:
                if src == "wide":
                    parts.append((wide, w))
                else:
                    t = self._open(self.tele_dir, clip)
                    if t.shape[0] != wide.shape[0]:
                        refuse("clip %s has %d wide frames and %d tele frames" % (clip, wide.shape[0], t.shape[0]))
                    parts.append((t, w))
            v = CombinedTokens(parts, (self.CFT.N_TOK_WIN, self.CFT.WIDTH))
            self.views[clip] = v
        return v

    def drop_frames(self, clip, head):
        m = self.mask(clip)
        if m is None:
            return np.zeros(self.get(clip).shape[0], bool)
        return drop_frames_of(m, head, self.CFT.HIST)


# ============================================================================================ module values: set, then restore
@contextlib.contextmanager
def patched(CFT, values):
    """sets module values of crop_fit (dict name -> value); yields the list of those that changed; restores every one on exit, also after an exception."""
    saved = {k: getattr(CFT, k) for k in values}
    changed = []
    try:
        for k, v in values.items():
            if getattr(CFT, k) is not v and getattr(CFT, k) != v:
                changed.append({"name": k, "before": _short(saved[k]), "after": _short(v)})
            setattr(CFT, k, v)
        yield changed
    finally:
        for k, v in saved.items():
            setattr(CFT, k, v)


def _short(v):
    if callable(v):
        return getattr(v, "__qualname__", repr(v))
    return repr(v) if not isinstance(v, tuple) else list(v)


def arm_values(CFT, arm, store, wrapped_builder):
    n_win = len(ARM_WINDOWS[arm])
    arms = tuple(CFT.ARMS) + ((arm,) if arm not in CFT.ARMS else ())

    class _Store:                     # crop_fit calls TokenStore(root, arm)
        def __new__(cls, root, arm_):
            return store
    return {"N_WIN": n_win, "N_TOK": n_win * CFT.N_TOK_WIN, "ARMS": arms, "TokenStore": _Store, "build_samples": wrapped_builder}


def make_wrapped_builder(CFT, original, counters):
    """build_samples of crop_fit, then the samples whose run holds a frame with mask false are dropped. counters gets one dict per call (fit first, then sel)."""
    def build_samples(clips, labels, store, head):
        sc, sf, sy = original(clips, labels, store, head)
        keep = np.ones(len(sf), bool)
        if hasattr(store, "drop_frames"):
            for k, c in enumerate(clips):
                a, b = np.searchsorted(sc, k, "left"), np.searchsorted(sc, k, "right")
                if b > a:
                    keep[a:b] = ~store.drop_frames(c, head)[sf[a:b]]
        counters.append({"total": int(len(sf)), "kept": int(keep.sum()), "dropped": int((~keep).sum())})
        return sc[keep], sf[keep], sy[keep]
    return build_samples


def ensure_lists(out_root, lists_dir):
    """crop_fit reads <root>/lists/{train_final,eval141}.txt: the two files are copied once into the new root (compared by sha256 when present). Writes under out_root only."""
    dst = os.path.join(out_root, "lists")
    if os.path.abspath(dst) == os.path.abspath(lists_dir):
        return
    os.makedirs(dst, exist_ok=True)
    for f in ("train_final.txt", "eval141.txt"):
        s, d = os.path.join(lists_dir, f), os.path.join(dst, f)
        if not os.path.isfile(s):
            refuse("no list " + s)
        if os.path.exists(d):
            if sha256_file(d) != sha256_file(s):
                refuse("%s differs from %s" % (d, s))
        else:
            shutil.copyfile(s, d)


# ============================================================================================ the fit
def run_tele_fit(a, log=print):
    t0 = time.time()
    kit, CFT, CDP = load_kit(a.kit_fit)
    arm, head, seed = a.arm, a.head, int(a.seed)
    if arm not in TELE_ARMS:
        refuse("arm %s" % arm)
    out_root = os.path.abspath(a.out_root)
    ensure_lists(out_root, a.lists_dir)
    store = TeleStore(CFT, arm, a.wide_tok_dir, a.tele_tok_dir, a.match_dir)
    counters = []
    originals = {k: getattr(CFT, k) for k in ("N_WIN", "N_TOK", "ARMS", "TokenStore", "build_samples")}
    ns = argparse.Namespace(arm=arm, head=head, seed=seed, permute_labels=bool(a.permute_labels), root=out_root, join_train=a.join_train, device=a.device,
                            epochs=CFT.CARD_HPARAMS["epochs"] if a.epochs is None else a.epochs, batch=CFT.CARD_HPARAMS["batch"] if a.batch is None else a.batch,
                            lr=CFT.CARD_HPARAMS["lr"], weight_decay=CFT.CARD_HPARAMS["weight_decay"], warmup_epochs=CFT.CARD_HPARAMS["warmup_epochs"])
    vals = arm_values(CFT, arm, store, make_wrapped_builder(CFT, CFT.build_samples, counters))
    with patched(CFT, vals) as changed:
        n_win, n_tok = CFT.N_WIN, CFT.N_TOK
        CFT.run_fit(ns, log=log)
    restored = all(getattr(CFT, k) is v or getattr(CFT, k) == v for k, v in originals.items())
    name = CFT.fit_name(arm, head, seed, a.permute_labels)
    fdir = os.path.join(out_root, "fits", name)
    roles = ("fit", "sel")
    rec = {"card": CARD, "tool": "tele_fit.py", "tier": "T0", "arm": arm, "head": head, "seed": seed, "permute_labels": bool(a.permute_labels),
           "n_win": int(n_win), "n_tok": int(n_tok), "windows": ARM_WINDOW_NAMES[arm],
           "n_samples": {roles[i]: counters[i] for i in range(min(2, len(counters)))},
           "crop_fit_path": os.path.abspath(CFT.__file__), "crop_fit_sha256": sha256_file(CFT.__file__), "crop_dump_sha256": sha256_file(CDP.__file__),
           "module_values_changed": changed, "module_values_restored": bool(restored),
           "wide_tok_dir": a.wide_tok_dir, "tele_tok_dir": a.tele_tok_dir if store.uses_tele else None, "match_dir": a.match_dir if store.uses_tele else None,
           "out_root": out_root, "argv": list(sys.argv), "seconds": float(time.time() - t0), "finished_utc": utc_now()}
    with open(os.path.join(fdir, "TELE_FIT_RECORD.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    log("ZZ TELE-FIT %s n_win %d n_tok %d kept %s dropped %s" % (name, n_win, n_tok, [c["kept"] for c in counters], [c["dropped"] for c in counters]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--arm", required=True, choices=TELE_ARMS)
    ap.add_argument("--head", default="H5")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--permute-labels", action="store_true")
    ap.add_argument("--wide-tok-dir", required=True)
    ap.add_argument("--tele-tok-dir", default=None)
    ap.add_argument("--match-dir", default=None)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--lists-dir", default=DEFAULT_LISTS)
    ap.add_argument("--join-train", default="/workspace/v8proof/data/inputs/join_train.npz")
    ap.add_argument("--kit-fit", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=None, help="default: crop_fit's own value; the queue never sets it")
    ap.add_argument("--batch", type=int, default=None, help="default: crop_fit's own value; the queue never sets it")
    a = ap.parse_args(argv)
    if a.head not in ("H1", "H5"):
        refuse("head " + a.head)
    return run_tele_fit(a)


if __name__ == "__main__":
    sys.exit(main())
