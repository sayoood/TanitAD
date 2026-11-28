#!/usr/bin/env python
"""pres_dump.py -- card PREREG_V8F_PROOF_ADD118 (PRESHEAD-T0), section 2: the dump of one presence fit on the evaluation clips. Tier T0.

  python pres_dump.py --seed K [--permute-labels] --root TOKENROOT --out-root ROOT --eval-join FILE [--device cuda]

Applies <out-root>/fits/D_H5_s<K>[_perm]/head.pt (a fit of pres_fit.py) to <root>/tok/D/<clip>.npy of every clip of the evaluation join and writes
<out-root>/dump/D_H5_s<K>[_perm]/pres.npz with one row per row of the join (row = off[clip] + frame):
  p        sigmoid of the head's logit; not a number for frames 0 to 3 (the head reads frames f-4 to f)
  has_out  p is finite
  clips, off, n   copied from the join (clip names as text)        seed   K
and pres_meta.json beside it. A file that exists is refused; a clip whose token file has another number of frames than the join is refused.
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
import pres_fit as PF  # noqa: E402

ARM, HEAD = "D", "H5"


def head_logits(model, tokens, dev, chunk=64):
    """tokens [n, 2, 65, 1024] float16 (memmap ok) -> logit [n] float64, not a number for frames 0 to 3"""
    import torch
    n = tokens.shape[0]
    model.eval()
    out = np.full(n, np.nan)
    with torch.no_grad():
        Z = []
        for i in range(0, n, chunk):
            x = torch.from_numpy(np.array(tokens[i:i + chunk]).reshape(-1, PF.N_TOK, PF.WIDTH)).to(dev).float()
            Z.append(model.embed(x))
        if n >= PF.HIST:
            Z = torch.cat(Z)
            idx = torch.arange(PF.HIST - 1, n, device=dev)
            win = torch.stack([Z[idx - (PF.HIST - 1 - k)] for k in range(PF.HIST)], 1).reshape(len(idx), -1)
            out[PF.HIST - 1:] = model.from_z(win).double().cpu().numpy()
    return out


def sigmoid(x):
    x = np.asarray(x, np.float64)
    return np.where(np.isfinite(x), 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))), np.nan)


def run_dump(a, log=print):
    import torch
    name = PF.fit_name(ARM, HEAD, a.seed, a.permute_labels)
    fdir = os.path.join(a.out_root, "fits", name)
    odir = os.path.join(a.out_root, "dump", name)
    opath, mpath = os.path.join(odir, "pres.npz"), os.path.join(odir, "pres_meta.json")
    for p in (opath, mpath):
        if os.path.exists(p):
            raise SystemExit("REFUSED: %s exists" % p)
    if not os.path.isfile(os.path.join(fdir, "FIT_RECORD.json")):
        raise SystemExit("REFUSED: no finished fit in " + fdir)
    with open(os.path.join(fdir, "FIT_RECORD.json"), encoding="utf-8") as fh:
        frec = json.load(fh)
    if frec.get("label") != "presence":
        raise SystemExit("REFUSED: the fit record of %s does not name the label presence" % fdir)
    J = np.load(a.eval_join, allow_pickle=True)
    clips = [str(c) for c in J["clip"]]
    off, n = np.asarray(J["off"], np.int64), np.asarray(J["n"], np.int64)
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    ck = torch.load(os.path.join(fdir, "head.pt"), map_location="cpu")
    if ck["head"] != HEAD or ck["arm"] != ARM or ck.get("card") != PF.CARD:
        raise SystemExit("REFUSED: head.pt is for arm %s head %s card %s" % (ck["arm"], ck["head"], ck.get("card")))
    model = PF.make_model(HEAD)
    model.load_state_dict(ck["state_dict"])
    model.to(dev)
    store = PF.TokenStore(a.root, ARM)
    p = np.full(int((off + n).max()) if len(n) else 0, np.nan)
    for i, c in enumerate(clips):
        tok = store.get(c)
        if tok.shape[0] != int(n[i]):
            raise SystemExit("REFUSED: clip %s has %d token frames, the join has %d" % (c[:8], tok.shape[0], int(n[i])))
        p[off[i]:off[i] + n[i]] = sigmoid(head_logits(model, tok, dev))
    os.makedirs(odir, exist_ok=True)
    with open(opath, "wb") as fh:
        np.savez(fh, p=p, has_out=np.isfinite(p), clips=np.array(clips), off=off, n=n, seed=np.int64(a.seed))
    meta = {"card": PF.CARD, "tier": "T0", "arm": ARM, "head": HEAD, "seed": int(a.seed), "permute_labels": bool(a.permute_labels),
            "head_pt_sha256": PF.sha256_file(os.path.join(fdir, "head.pt")), "head_pt_sha256_of_the_fit_record": frec.get("head_pt_sha256"),
            "eval_join": a.eval_join, "eval_join_sha256": PF.sha256_file(a.eval_join), "dump_sha256": PF.sha256_file(opath),
            "tool_sha256": PF.sha256_file(os.path.abspath(__file__)), "fit_tool_sha256": PF.sha256_file(os.path.abspath(PF.__file__)),
            "n_clips": len(clips), "n_rows": int(len(p)), "n_rows_of_the_join": int(n.sum()), "n_finite": int(np.isfinite(p).sum())}
    with open(mpath, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=1)
    log("ZZ PRES-DUMP %s rows %d finite %d" % (name, meta["n_rows"], meta["n_finite"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--permute-labels", action="store_true")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--eval-join", required=True)
    ap.add_argument("--device", default="cuda")
    return run_dump(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
