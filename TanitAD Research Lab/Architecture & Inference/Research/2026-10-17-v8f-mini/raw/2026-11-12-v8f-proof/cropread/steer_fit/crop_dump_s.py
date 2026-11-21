#!/usr/bin/env python
"""crop_dump_s.py -- dump of one fit on arm S of card PREREG_V8F_PROOF_ADD100 (STEERREAD), in the form of crop_dump.py (card ADD99 section 6). Tier T0.

  python crop_dump_s.py --head H5|H5d --seed K [--permute-labels] [--s-root S] [--present-dump DIR] [--device cuda]

Applies <s-root>/fits/S_<head>_s<K>[_perm]/head.pt to <s-root>/tok/S/<clip>.npy (and, for H5d, <s-root>/steer/<clip>.npy) of the clips of the present reader's score dump of seed
index K and writes <s-root>/dump/S_<head>[_perm]/reader_s<K>_score.npz with the fields of crop_dump.dump_from_present (every field not made by the head copied from the present dump;
gap = the head's output, NaN on frames 0 to 3; p 1.0; clos 0.0; valid = present valid and finite gap) and reader_s<K>_meta.json. The dump file must not exist.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import steer_common as SC  # noqa: E402
import crop_fit_s as CS  # noqa: E402
import crop_dump as CD  # noqa: E402  (card ADD99 kit)

CFT = SC.CFT
DEFAULT_PRESENT = CD.DEFAULT_PRESENT


def head_outputs_s(model, tokens, steer, dev, chunk=64):
    """tokens [n, 2, 65, 1024] float16, steer float32 [n, 2] -> gap [n] float64, NaN on frames 0 to 3. H5: the dump routine of card ADD99 unchanged; H5d: the same with the direction appended."""
    import torch
    if model.head == "H5":
        return CD.head_outputs(model, tokens, dev, chunk)
    n = tokens.shape[0]
    if steer.shape != (n, 2):
        raise SystemExit("REFUSED: steering has shape %s for %d token frames" % (list(steer.shape), n))
    model.eval()
    out = np.full(n, np.nan)
    with torch.no_grad():
        Z = []
        for i in range(0, n, chunk):
            x = torch.from_numpy(np.array(tokens[i:i + chunk]).reshape(-1, CFT.N_TOK, CFT.WIDTH)).to(dev).float()
            Z.append(model.embed(x))
        Z = torch.cat(Z)
        zd = torch.cat([Z, torch.from_numpy(np.asarray(steer, np.float32)).to(dev) / SC.DIR_SCALE], 1)          # [n, 258]
        if n >= CFT.HIST:
            idx = torch.arange(CFT.HIST - 1, n, device=dev)
            win = torch.stack([zd[idx - (CFT.HIST - 1 - k)] for k in range(CFT.HIST)], 1).reshape(len(idx), -1)
            out[CFT.HIST - 1:] = model.from_zd(win).double().cpu().numpy()
    return out


def run_dump_s(a, log=print):
    import torch
    name = CFT.dump_name("S", a.head, a.permute_labels)
    fdir = os.path.join(a.s_root, "fits", CFT.fit_name("S", a.head, a.seed, a.permute_labels))
    odir = os.path.join(a.s_root, "dump", name)
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
    if ck["head"] != a.head or ck["arm"] != "S":
        raise SystemExit("REFUSED: head.pt is for arm %s head %s" % (ck["arm"], ck["head"]))
    model = CS.make_model_s(a.head)
    model.load_state_dict(ck["state_dict"])
    model.to(dev)
    store, steer = CFT.TokenStore(a.s_root, "S"), SC.SteerStore(a.s_root)
    per, shas = {}, {}
    for c in [str(c) for c in Dp["clips"]]:
        tok = store.get(c)
        per[c] = head_outputs_s(model, tok, steer.get(c, tok.shape[0]), dev)
        shas[c] = SC.sha256_file(steer.path(c))
    D = CD.dump_from_present(Dp, per, a.seed)
    os.makedirs(odir, exist_ok=True)
    with open(opath, "wb") as fh:
        np.savez(fh, **D)
    meta = {"card": SC.CARD, "arm": "S", "head": a.head, "seed": a.seed, "permute_labels": bool(a.permute_labels), "head_pt_sha256": SC.sha256_file(os.path.join(fdir, "head.pt")),
            "present_dump": ppath, "present_dump_sha256": SC.sha256_file(ppath), "dump_sha256": SC.sha256_file(opath), "n_rows": int(len(D["frame"])), "n_valid": int(D["valid"].sum()),
            "steer_sha256": shas}
    SC.write_json(os.path.join(odir, "reader_s%d_meta.json" % a.seed), meta)
    log("ZZ CROP-DUMP-S %s s%d rows %d valid %d" % (name, a.seed, meta["n_rows"], meta["n_valid"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--head", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--permute-labels", action="store_true")
    ap.add_argument("--s-root", default=SC.S_ROOT)
    ap.add_argument("--present-dump", default=DEFAULT_PRESENT)
    ap.add_argument("--device", default="cuda")
    return run_dump_s(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
