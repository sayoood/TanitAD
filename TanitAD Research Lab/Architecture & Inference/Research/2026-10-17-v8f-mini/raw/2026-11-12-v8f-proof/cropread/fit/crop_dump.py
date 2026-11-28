#!/usr/bin/env python
"""crop_dump.py -- dump of one fit of card PREREG_V8F_PROOF_ADD99 (CROPREAD), section 6, in the form of the present reader's dump of seed index K. Tier T0.

  python crop_dump.py --arm D|B --head H1|H5 --seed K [--permute-labels] [--root /workspace/v8proof/cropread] [--present-dump DIR] [--device cuda]

Applies <root>/fits/<arm>_<head>_s<K>[_perm]/head.pt to <root>/tok/<arm>/<clip>.npy of the clips of the present reader's score dump of seed index K
(<present-dump>/reader_s<K>_score.npz) and writes <root>/dump/<arm>_<head>[_perm]/reader_s<K>_score.npz. The folder file must not exist.
Fields (format of closeread/closeread_dump.py:dump_split):
  copied from the present dump: cid, clip_index, frame, lab_present, lab_gap, lab_closing, lab_run, clips, dt_s
  gap      the head's output in metres at (clip, frame); NaN where the head has no output (H5: frames 0 to 3)
  p        1.0 (the pilot has no presence head)          clos   0.0 (the pilot has no closing head)
  valid    the present dump's valid AND the gap is finite  (so the scored rows of card ADD81 stay the same rows)
  seed     K
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import crop_fit as CFT  # noqa: E402

COPIED = ("cid", "clip_index", "frame", "lab_present", "lab_gap", "lab_closing", "lab_run", "clips", "dt_s")
DEFAULT_PRESENT = "/workspace/v8proof/closeread/dump/both"


def head_outputs(model, tokens, dev, chunk=64):
    """tokens [n, 2, 65, 1024] float16 (memmap ok) -> gap [n] float64: H1 every frame, H5 NaN for frames 0 to 3."""
    import torch
    n = tokens.shape[0]
    model.eval()
    out = np.full(n, np.nan)
    with torch.no_grad():
        Z = []
        for i in range(0, n, chunk):
            x = torch.from_numpy(np.array(tokens[i:i + chunk]).reshape(-1, CFT.N_TOK, CFT.WIDTH)).to(dev).float()
            Z.append(model.embed(x))
        Z = torch.cat(Z)
        if model.head == "H1":
            out[:] = model.from_z(Z).double().cpu().numpy()
        elif n >= CFT.HIST:
            idx = torch.arange(CFT.HIST - 1, n, device=dev)
            win = torch.stack([Z[idx - (CFT.HIST - 1 - k)] for k in range(CFT.HIST)], 1).reshape(len(idx), -1)
            out[CFT.HIST - 1:] = model.from_z(win).double().cpu().numpy()
    return out


def dump_from_present(Dp, per_clip_gap, seed):
    """the dump of the form: Dp the present dump (dict), per_clip_gap {clip: gap [n]} -> dict of arrays."""
    cid, fr = np.asarray(Dp["cid"], np.int64), np.asarray(Dp["frame"], np.int64)
    clips = [str(c) for c in Dp["clips"]]
    gap = np.empty(len(cid))
    for i, c in enumerate(clips):
        m = cid == i
        g = per_clip_gap[c]
        if m.any() and fr[m].max() >= len(g):
            raise SystemExit("REFUSED: clip %s has %d token frames, the present dump has frame %d" % (c, len(g), int(fr[m].max())))
        gap[m] = g[fr[m]]
    D = {k: np.asarray(Dp[k]) for k in COPIED}
    D["gap"] = gap
    D["p"] = np.ones(len(cid))
    D["clos"] = np.zeros(len(cid))
    D["valid"] = np.asarray(Dp["valid"], bool) & np.isfinite(gap)
    D["seed"] = np.int64(seed)
    return D


def run_dump(a, log=print):
    import torch
    root = a.root
    name = CFT.dump_name(a.arm, a.head, a.permute_labels)
    fdir = os.path.join(root, "fits", CFT.fit_name(a.arm, a.head, a.seed, a.permute_labels))
    odir = os.path.join(root, "dump", name)
    opath = os.path.join(odir, "reader_s%d_score.npz" % a.seed)
    if os.path.exists(opath):
        raise SystemExit("REFUSED: %s exists" % opath)
    if not os.path.isfile(os.path.join(fdir, "FIT_RECORD.json")):
        raise SystemExit("REFUSED: no finished fit in " + fdir)
    ppath = os.path.join(a.present_dump, "reader_s%d_score.npz" % a.seed)
    z = np.load(ppath, allow_pickle=False)
    Dp = {k: np.asarray(z[k]) for k in z.files}
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    ck = torch.load(os.path.join(fdir, "head.pt"), map_location="cpu")
    if ck["head"] != a.head or ck["arm"] != a.arm:
        raise SystemExit("REFUSED: head.pt is for arm %s head %s" % (ck["arm"], ck["head"]))
    model = CFT.make_model(a.head)
    model.load_state_dict(ck["state_dict"])
    model.to(dev)
    store = CFT.TokenStore(root, a.arm)
    per = {}
    for c in [str(c) for c in Dp["clips"]]:
        per[c] = head_outputs(model, store.get(c), dev)
    D = dump_from_present(Dp, per, a.seed)
    os.makedirs(odir, exist_ok=True)
    with open(opath, "wb") as fh:
        np.savez(fh, **D)
    meta = {"card": CFT.CARD, "arm": a.arm, "head": a.head, "seed": a.seed, "permute_labels": bool(a.permute_labels), "head_pt_sha256": CFT.sha256_file(os.path.join(fdir, "head.pt")),
            "present_dump": ppath, "present_dump_sha256": CFT.sha256_file(ppath), "dump_sha256": CFT.sha256_file(opath), "n_rows": int(len(D["frame"])), "n_valid": int(D["valid"].sum())}
    with open(os.path.join(odir, "reader_s%d_meta.json" % a.seed), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=1)
    log("ZZ CROP-DUMP %s s%d rows %d valid %d" % (name, a.seed, meta["n_rows"], meta["n_valid"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--permute-labels", action="store_true")
    ap.add_argument("--root", default=CFT.DEFAULT_ROOT)
    ap.add_argument("--present-dump", default=DEFAULT_PRESENT)
    ap.add_argument("--device", default="cuda")
    return run_dump(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
