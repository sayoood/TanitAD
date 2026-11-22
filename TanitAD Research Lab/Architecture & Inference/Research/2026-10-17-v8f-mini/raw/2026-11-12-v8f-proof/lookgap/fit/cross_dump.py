#!/usr/bin/env python
"""cross_dump.py -- importing tool of card PREREG_V8F_PROOF_ADD108 (LOOKGAP-T0), stage C. Tier T0.

  python cross_dump.py --kit-fit DIR --fit-dir DIR --tok-dir DIR --present FILE --seed-index K --out FILE
                       [--compare FILE --ident-out JSON --tol 1e-3] [--device cuda]
  python cross_dump.py --selftest --kit-fit DIR

Reads the head of a finished fit (<fit-dir>/head.pt, head H5 only) on the token files of ANY token folder (--tok-dir) and writes a dump in the form of the present reader's dump of
seed index K: the same code as crop_dump.py (head_outputs, dump_from_present), imported unchanged from --kit-fit (put first on sys.path). crop_fit.py and crop_dump.py are not edited.
Beside the dump it writes reader_s<K>_meta.json. With --compare it compares the gap and valid arrays with those of another dump (max absolute difference over the rows where both
gaps are finite, equal NaN pattern, equal valid) and writes --ident-out; it prints one line and still exits 0 (a printed fact, not a stop). Exit 2 only on a refusal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

REFUSED = 2


def refuse(msg):
    print("REFUSED: " + msg, file=sys.stderr)
    raise SystemExit(REFUSED)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def compare_dumps(a, b, tol=1e-3):
    """a, b: mappings with 'gap' and 'valid' arrays -> record (dict). max_abs_diff_m is None when it cannot be computed."""
    ga, gb = np.asarray(a["gap"], np.float64), np.asarray(b["gap"], np.float64)
    va, vb = np.asarray(a["valid"], bool), np.asarray(b["valid"], bool)
    rec = {"n_rows": int(len(ga)), "n_rows_equal": bool(len(ga) == len(gb))}
    if not rec["n_rows_equal"]:
        rec.update({"max_abs_diff_m": None, "nan_pattern_equal": False, "valid_equal": False, "tol_m": float(tol), "pass": False})
        return rec
    fa, fb = np.isfinite(ga), np.isfinite(gb)
    both = fa & fb
    mx = float(np.max(np.abs(ga[both] - gb[both]))) if both.any() else None
    rec["max_abs_diff_m"] = mx
    rec["nan_pattern_equal"] = bool(np.array_equal(fa, fb))
    rec["valid_equal"] = bool(np.array_equal(va, vb))
    rec["tol_m"] = float(tol)
    rec["pass"] = bool(mx is not None and mx <= tol and rec["nan_pattern_equal"] and rec["valid_equal"])
    return rec


def _load_tools(kit_fit):
    kit_fit = os.path.abspath(kit_fit)
    if not os.path.isfile(os.path.join(kit_fit, "crop_fit.py")) or not os.path.isfile(os.path.join(kit_fit, "crop_dump.py")):
        refuse("no crop_fit.py and crop_dump.py in " + kit_fit)
    if kit_fit in sys.path:
        sys.path.remove(kit_fit)
    sys.path.insert(0, kit_fit)
    import crop_fit as CFT          # noqa: E402
    import crop_dump as CDP         # noqa: E402
    for m in (CFT, CDP):
        if os.path.dirname(os.path.abspath(m.__file__)) != kit_fit:
            refuse("%s was imported from %s, not from %s" % (m.__name__, m.__file__, kit_fit))
    return kit_fit, CFT, CDP


def run_cross(kit_fit, fit_dir, tok_dir, present, seed_index, out, compare=None, ident_out=None, tol=1e-3, device="cuda", log=print):
    """-> (meta dict, ident record or None). Raises SystemExit(2) on a refusal."""
    out = os.path.abspath(out)
    odir = os.path.dirname(out)
    meta_path = os.path.join(odir, "reader_s%d_meta.json" % seed_index)
    if os.path.exists(out) or os.path.exists(out + ".part"):
        refuse("%s exists" % out)
    if os.path.exists(meta_path):
        refuse("%s exists" % meta_path)
    if ident_out is not None and os.path.exists(ident_out):
        refuse("%s exists" % ident_out)
    if (compare is None) != (ident_out is None):
        refuse("--compare and --ident-out go together")
    for p, what in ((fit_dir, "fit dir"), (tok_dir, "token dir")):
        if not os.path.isdir(p):
            refuse("no %s %s" % (what, p))
    for p in (os.path.join(fit_dir, "head.pt"), present) + ((compare,) if compare else ()):
        if not os.path.isfile(p):
            refuse("no file " + p)
    kit_fit, CFT, CDP = _load_tools(kit_fit)
    import torch
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    head_pt = os.path.join(fit_dir, "head.pt")
    ck = torch.load(head_pt, map_location="cpu")
    if ck.get("head") != "H5":
        refuse("head.pt in %s is for head %s, expected H5" % (fit_dir, ck.get("head")))
    model = CFT.make_model("H5")
    model.load_state_dict(ck["state_dict"])
    model.to(dev)
    z = np.load(present, allow_pickle=False)
    Dp = {k: np.asarray(z[k]) for k in z.files}
    store = CFT.TokenStore("", "unused")
    store.dir = os.path.abspath(tok_dir)
    per = {}
    for c in [str(c) for c in Dp["clips"]]:
        per[c] = CDP.head_outputs(model, store.get(c), dev)
    D = CDP.dump_from_present(Dp, per, seed_index)
    os.makedirs(odir, exist_ok=True)
    part = out + ".part"
    with open(part, "wb") as fh:
        np.savez(fh, **D)
    os.replace(part, out)
    ident = None
    if compare:
        zc = np.load(compare, allow_pickle=False)
        ident = compare_dumps(D, {"gap": zc["gap"], "valid": zc["valid"]}, tol)
        ident["compare_file"] = compare
        with open(ident_out, "x", encoding="utf-8", newline="\n") as fh:
            json.dump(ident, fh, indent=1)
    meta = {"fit_dir": fit_dir, "tok_dir": tok_dir, "head_pt_sha256": sha256_file(head_pt), "present_dump": present, "present_dump_sha256": sha256_file(present),
            "dump_sha256": sha256_file(out), "n_rows": int(len(D["frame"])), "n_valid": int(D["valid"].sum()),
            "crop_fit_sha256": sha256_file(os.path.join(kit_fit, "crop_fit.py")), "crop_dump_sha256": sha256_file(os.path.join(kit_fit, "crop_dump.py")), "seed_index": int(seed_index)}
    with open(meta_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=1)
    log("ZZ CROSS-DUMP %s rows %d valid %d" % (out, meta["n_rows"], meta["n_valid"]))
    if ident is not None:
        mx = ident["max_abs_diff_m"]
        log("ZZ CROSS-IDENT pass %s max_abs_diff_m %s" % (ident["pass"], "nan" if mx is None else repr(mx)))
    return meta, ident


# ============================================================================================ selftest
def make_world(root, kit_fit, n_clips=3, n_frames=8, seed=0):
    """tiny synthetic world under root: tok/<clip>.npy, fit/head.pt (random H5), present.npz. Returns a dict of paths."""
    import torch
    kit_fit, CFT, CDP = _load_tools(kit_fit)
    rng = np.random.default_rng(seed)
    root = os.path.abspath(root)
    os.makedirs(os.path.join(root, "tok"), exist_ok=True)
    os.makedirs(os.path.join(root, "fit"), exist_ok=True)
    clips = ["clip%d" % i for i in range(n_clips)]
    for c in clips:
        np.save(os.path.join(root, "tok", c + ".npy"), (rng.standard_normal((n_frames, CFT.N_WIN, CFT.N_TOK_WIN, CFT.WIDTH)) * 0.5).astype(np.float16))
    torch.manual_seed(seed)
    model = CFT.make_model("H5")
    torch.save({"head": "H5", "arm": "B", "state_dict": model.state_dict()}, os.path.join(root, "fit", "head.pt"))
    cid = np.repeat(np.arange(n_clips), n_frames).astype(np.int64)
    frame = np.tile(np.arange(n_frames), n_clips).astype(np.int64)
    n = len(cid)
    present = {"cid": cid, "clip_index": cid.copy(), "frame": frame, "lab_present": np.ones(n), "lab_gap": rng.uniform(5, 50, n), "lab_closing": rng.standard_normal(n),
               "lab_run": np.zeros(n), "clips": np.array(clips), "dt_s": np.float64(0.2), "valid": np.ones(n, bool)}
    missing = [k for k in CDP.COPIED if k not in present]
    if missing:
        raise SystemExit("selftest world lacks fields " + ",".join(missing))
    with open(os.path.join(root, "present.npz"), "wb") as fh:
        np.savez(fh, **present)
    return {"tok": os.path.join(root, "tok"), "fit": os.path.join(root, "fit"), "present": os.path.join(root, "present.npz"), "root": root, "clips": clips}


def selftest(kit_fit, workdir=None):
    """-> True when the selftest passes. Makes a folder under workdir (default: the current directory); leaves it in place."""
    import torch
    base = os.path.abspath(workdir or os.getcwd())
    root = os.path.join(base, "tmp_xdump_selftest_%d" % os.getpid())
    ok = True
    try:
        w = make_world(root, kit_fit)
        outs = os.path.join(root, "out")
        m1, _ = run_cross(kit_fit, w["fit"], w["tok"], w["present"], 0, os.path.join(outs, "a", "reader_s0_score.npz"), device="cpu", log=lambda s: None)
        _, i2 = run_cross(kit_fit, w["fit"], w["tok"], w["present"], 0, os.path.join(outs, "b", "reader_s0_score.npz"),
                          compare=os.path.join(outs, "a", "reader_s0_score.npz"), ident_out=os.path.join(root, "ident_same.json"), device="cpu", log=lambda s: None)
        if not (i2["pass"] and i2["max_abs_diff_m"] == 0.0 and i2["nan_pattern_equal"] and i2["valid_equal"]):
            print("FAIL identity on equal inputs: %s" % i2)
            ok = False
        if m1["n_rows"] != 24 or m1["n_valid"] != 3 * 4:
            print("FAIL row counts: %s" % m1)
            ok = False
        tok2 = os.path.join(root, "tok_changed")
        os.makedirs(tok2, exist_ok=True)
        for c in w["clips"]:
            a = np.load(os.path.join(w["tok"], c + ".npy"))
            if c == w["clips"][1]:
                a = (np.random.default_rng(99).standard_normal(a.shape) * 2.0).astype(np.float16)
            np.save(os.path.join(tok2, c + ".npy"), a)
        _, i3 = run_cross(kit_fit, w["fit"], tok2, w["present"], 0, os.path.join(outs, "c", "reader_s0_score.npz"),
                          compare=os.path.join(outs, "a", "reader_s0_score.npz"), ident_out=os.path.join(root, "ident_changed.json"), device="cpu", log=lambda s: None)
        if i3["pass"] or not i3["max_abs_diff_m"] or i3["max_abs_diff_m"] <= 1e-3:
            print("FAIL planted change did not flip pass: %s" % i3)
            ok = False
    except SystemExit as ex:
        print("FAIL refusal in selftest: %s" % (ex,))
        ok = False
    print("ZZ CROSS-DUMP SELFTEST " + ("OK" if ok else "FAIL"))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--kit-fit", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fit-dir")
    ap.add_argument("--tok-dir")
    ap.add_argument("--present")
    ap.add_argument("--seed-index", type=int)
    ap.add_argument("--out")
    ap.add_argument("--compare")
    ap.add_argument("--ident-out")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.selftest:
        return 0 if selftest(a.kit_fit) else 1
    for k in ("fit_dir", "tok_dir", "present", "seed_index", "out"):
        if getattr(a, k) is None:
            refuse("--%s is required" % k.replace("_", "-"))
    try:
        run_cross(a.kit_fit, a.fit_dir, a.tok_dir, a.present, a.seed_index, a.out, a.compare, a.ident_out, a.tol, a.device)
    except SystemExit as ex:
        if isinstance(ex.code, str):         # a refusal raised inside the pilot's tools
            print(ex.code, file=sys.stderr)
            return REFUSED
        return ex.code if isinstance(ex.code, int) else REFUSED
    return 0


if __name__ == "__main__":
    sys.exit(main())
