#!/usr/bin/env python3
"""INPUT-PHASE-ONLY check of `srttc_dev.py` against the REAL landed dev inputs.  It spends nothing: it never calls `read_outcomes` / `run_dev`, so no `pdms`, `sub` or `valid` VALUE is opened, no SR-TTC
row is applied (there is none yet: S0 is still running) and no READ_STARTED marker is written.  On the real files it proves what the dev gate needs BEFORE any outcome is read:

  * the three dev files hash to the landed sha256 (A2.10 condition 4) and load under the pinned shape (1,705 tokens / 24 logs);
  * the outcome members the gate will read LATER exist with the expected shapes -- read from the zip directory and the .npy HEADERS only (no array data);
  * integrity: P2's real head reproduces the stored logits from the cached pre-head features within 2e-3, and a next-token control reads far above it;
  * C0 (identity): with P2's own TTC row the logits are bit-identical and the pick equals P2's banked pick on every token;
  * C4 (fan frozen): the dump the features came from holds exactly the table's proposals (max abs diff 0.0 m);
  * A2.10 (amendment2.md:245-248): the four admissibility conditions, evaluated on the landed package attestations (P2HO_DONE, recompute_check_333.json, MANIFEST.sha256.json).

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I srttc_dev_inputs_check.py --out srttc_dev_inputs_check.json [--p2ho-dir DIR]

The attestation files are read from --p2ho-dir (default: the package directory inside the repo checkout).  Where that directory is absent on the machine (the worktree is behind the tip), pass a directory
holding the three files exactly as `git show <tip>:<path>` prints them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import srttc_dev as D   # noqa: E402

OUTCOME_MEMBERS = ("pdms", "sub", "valid", "sub_names")


def _io_npz_headers(guard: "D.DevGuard", path: str, members=OUTCOME_MEMBERS) -> dict:
    """-> {member: {"dtype", "shape"}} for the named members, from the zip directory and the first bytes of each .npy only.  No array VALUE is read."""
    guard.check_read(path)
    out = {}
    with zipfile.ZipFile(path) as z:
        names = {n[:-4]: n for n in z.namelist() if n.endswith(".npy")}
        for m in members:
            if m not in names:
                out[m] = None
                continue
            with z.open(names[m]) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(f)
                out[m] = {"dtype": str(dtype), "shape": list(shape)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-dir", default=D.DEV_DIR)
    ap.add_argument("--p2-ckpt", default=D.DEFAULT_P2_CKPT)
    ap.add_argument("--p2ho-dir", default=D.P2HO_DIR)
    a = ap.parse_args(argv)
    t0 = time.time()
    out_dir = os.path.dirname(os.path.abspath(a.out))
    opts = D.DevOptions(run_dir=out_dir, registered_result_sha256="n/a (inputs-only check)", dev_dir=a.dev_dir, p2_ckpt=a.p2_ckpt, out_dir=out_dir, p2ho_dir=a.p2ho_dir)
    guard = opts.guard()
    W6, b6 = D.load_p2_head(guard, opts.p2_ckpt)
    inp = D.load_dev_inputs(guard, opts)
    hdr = _io_npz_headers(guard, os.path.join(a.dev_dir, "table_p2_heldout.npz"))
    want = {"pdms": [inp.n, D.N_PROPOSALS], "sub": [inp.n, D.N_PROPOSALS, 6], "valid": [inp.n, D.N_PROPOSALS], "sub_names": [6]}
    hdr_ok = all(hdr.get(m) is not None and hdr[m].get("shape") == want[m] for m in OUTCOME_MEMBERS)
    rec = {"what": "srttc_dev.py input phase on the REAL landed dev inputs; no outcome VALUE opened; no row applied; no marker written",
           "dev": inp.provenance,
           "outcome_member_headers": {"headers": hdr, "expected_shapes": want, "pass": bool(hdr_ok)},
           "integrity_featcheck": D.integrity_featcheck(inp, W6, b6, opts.featcheck_tol, opts.chunk),
           "C0_pick_identity": D.c0_identity(inp, W6, b6, opts.chunk),
           "C4_fan_frozen": D.c4_fan_frozen(inp),
           "outcomes_opened": False}
    try:
        rec["A210_admissibility"] = {"pass": True, **D.check_a210(guard, opts, inp.provenance["sha256"])}
    except D.DevRefusal as ex:
        rec["A210_admissibility"] = {"pass": False, "refusal": str(ex)[:1200]}
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["all_pass"] = bool(all(rec[k]["pass"] for k in ("outcome_member_headers", "integrity_featcheck", "C0_pick_identity", "C4_fan_frozen", "A210_admissibility")))
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(D._jsonable(rec), f, indent=1)
    print(json.dumps(D._jsonable({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("sha256", "headers")}) for k, v in rec.items()}), indent=1))
    return 0 if rec["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
