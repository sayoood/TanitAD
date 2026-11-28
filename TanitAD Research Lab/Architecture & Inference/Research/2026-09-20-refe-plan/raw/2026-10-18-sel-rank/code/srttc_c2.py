#!/usr/bin/env python3
"""SR-TTC C2 negative-control FIT (PREREG_SELRANK_ARM.md:71): the SAME fit at c* with the TTC labels SHUFFLED ACROSS THE PROPOSALS WITHIN EACH SET.  CPU only.

WHY THIS FILE EXISTS.  The dev gate's bar includes C2 (PREREG_SELRANK_ARM.md:93 'C0-C2, C4 pass') and `srttc_dev.py` refuses to read without the shuffled-label rows.  `srttc_r1.py` has NO `c2`
stage (EP-R1's `epr1.py c2` has one), so nothing could have produced those rows: C2 would have been a registered control that no caller reaches.  This module is that stage, written as a NEW file
(srttc_r1.py is landed and frozen) on top of srttc_r1's own loader and fit path (`_setup`, `fit_row`), so the shuffled fit differs from the real fit in ONE thing: the labels.

WHAT IT DOES
  reads r1_result.json of the same fraction/bank (c* is NOT re-chosen), permutes `y3` (the TTC label) across the 64 proposals of every set independently, seeded (C2_SHUFFLE_SEED), refits c* on ALL fit
  sets of S_f for each seed {0,1} with srttc_r1.fit_row (the contender membership `top8` stays the FROZEN P2 membership), and writes
      ttc_row_c2shuffle_seed{0,1}.npy   (float32[257], srttc_r1.row_bytes layout)      r1_c2_result.json   (schema srttc-r1-v1, stage C2)
  Refuses to overwrite either (exclusive create).  Reads NOTHING that srttc_r1 does not read (it uses srttc_r1's PathGuard: the dev-gate / navtest tables are unreachable).

SHUFFLE SEED: 20261018 is NOT registered for SR-TTC (it is EP-R1's PREREG_EPR1 s6 value).  A null fit has the same expected gain whatever the seed; recorded as GAP G1 in srttc_dev.py.

NO LAUNCH-GATE STAGE.  A C2 fit is a head fit, i.e. a training run, and CLAUDE.md's launch-gate rule has no size exemption; `srttc_r1_gate.py` mints stages `run` and `lc` only and is landed/frozen.
The CLI therefore refuses to start without `--no-gate-stage-ack`, and the hand-back escalates it: the Master Mind either extends the gate with a `c2` stage or rules that R1's token covers it.

CLI
    srttc_c2.py --fraction 10 [--out-dir D] [--threads 8] --no-gate-stage-ack      (same --bank-dir/--manifest/--ranked-logs/--p2-ckpt/--s0-marker/--replacements-file flags as srttc_r1.py)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import srttc_r1 as R   # noqa: E402

C2_SHUFFLE_SEED = 20261018                  # UNREGISTERED for SR-TTC (GAP G1); EP-R1's PREREG_EPR1 s6 value
RESULT_NAME = "r1_c2_result.json"
ROW_FMT = "ttc_row_c2shuffle_seed{sd}.npy"


class C2Error(RuntimeError):
    """The C2 fit cannot be produced or would test nothing."""


def shuffle_labels_within_set(y3: torch.Tensor, seed: int) -> torch.Tensor:
    """PREREG_SELRANK_ARM.md:71: the TTC labels permuted ACROSS the 64 proposals of each set (every set its own permutation; each set's label multiset is unchanged).  Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    yn = y3.numpy().copy()
    n_prop = yn.shape[1]
    for i in range(len(yn)):
        yn[i] = yn[i, rng.permutation(n_prop)]
    return torch.from_numpy(yn)


def fit_c2_rows(data: "R.FitData", star: "R.Cfg", seeds, W6, b6, shuffle_seed: int = C2_SHUFFLE_SEED, fit=None) -> tuple:
    """-> ({seed: (w, b)}, y_shuffled).  The fit is `srttc_r1.fit_row` on a copy of `data` whose ONLY difference is `y3`; `fit` is injectable so a test can prove the shuffled labels are what it trains on."""
    fit = fit or R.fit_row
    y_shuf = shuffle_labels_within_set(data.y3, shuffle_seed)
    tr = dataclasses.replace(data, y3=y_shuf)                                      # MUTATION-TARGET-C2FIT-LABELS: the fit sees the SHUFFLED labels
    rows = {}
    for sd in seeds:
        rows[str(sd)] = fit(tr, np.arange(tr.n_sets), star, sd, W6, b6)
    return rows, y_shuf


def run_c2(opts: "R.Options") -> dict:
    t0 = time.time()
    guard, out, manifest, man_sha, ranked_sha, selected, repl_rep, per_fold_logs, W6, b6, data = R._setup(opts)
    path_r1 = os.path.join(out, "r1_result.json")
    if not R._io_exists(guard, path_r1):
        raise C2Error(f"{path_r1} does not exist: C2 is run at the c* that R1 wrote; it never chooses one")
    r1 = json.loads(R._io_read_bytes(guard, path_r1).decode("utf-8"))
    if r1["inputs"]["bank_manifest_sha256"] != man_sha:
        raise C2Error("the bank manifest changed since r1_result.json was written: C2 must use the SAME bank")
    if r1["fraction_pct"] != opts.fraction:
        raise C2Error(f"r1_result.json is for f = {r1['fraction_pct']} %, not {opts.fraction} %")
    cs = r1["c_star"]
    star = R.Cfg(lr=cs["lr"], anchor=cs["anchor"], epochs=r1["params"]["epochs"], batch_sets=r1["params"]["batch_sets"])
    if star.sha256() != cs["config_sha256"]:
        raise C2Error("c* from r1_result.json does not reproduce its recorded config sha256")
    for p in [os.path.join(out, RESULT_NAME)] + [os.path.join(out, ROW_FMT.format(sd=sd)) for sd in opts.seeds]:
        if R._io_exists(guard, p):
            raise C2Error(f"{p} exists: the C2 rows are written once")
    rows, y_shuf = fit_c2_rows(data, star, opts.seeds, W6, b6, C2_SHUFFLE_SEED)
    moved = float((y_shuf.numpy() != data.y3.numpy()).mean())
    if not moved > 0.0:                                                              # MUTATION-TARGET-C2FIT-MOVED
        raise C2Error("the shuffle moved no label: C2 would test nothing")
    blobs = {}
    for sd in opts.seeds:                                                            # decide BEFORE anything is written: a refused C2 leaves no row behind
        w, b = rows[str(sd)]
        blobs[str(sd)] = R.row_bytes(w, b)
        real = r1["final_rows"].get(str(sd))
        if real is not None and real["sha256"] == R.sha256_bytes(blobs[str(sd)]):    # MUTATION-TARGET-C2FIT-DIFFERS
            raise C2Error(f"seed {sd}: the shuffled-label row equals the real-label row: the shuffle changed nothing the fit can see")
    final = {}
    for sd in opts.seeds:
        w, b = rows[str(sd)]
        real = r1["final_rows"].get(str(sd))
        name = ROW_FMT.format(sd=sd)
        sha = R._io_write_bytes(guard, os.path.join(out, name), blobs[str(sd)], exclusive=True)
        final[str(sd)] = {"file": name, "sha256": sha, "bytes": len(blobs[str(sd)]), "layout": "float32[257] = weight[3,:256] then bias[3], .npy",
                          "max_abs_change_vs_p2": float(max(np.abs(w - np.asarray(W6)[R.TTC_COL]).max(), np.abs(b - np.asarray(b6)[R.TTC_COL]).max())),
                          "differs_from_real_label_row": bool(real is None or real["sha256"] != sha)}
    result = {"schema": R.SCHEMA, "stage": "C2", "fraction_pct": opts.fraction, "c_star": cs, "bank_manifest_sha256": man_sha, "shuffle_seed": C2_SHUFFLE_SEED,
              "shuffle": "TTC labels permuted across the 64 proposals of each set (PREREG_SELRANK_ARM.md:71); trained on shuffled labels at c*, all fit sets of S_f, both seeds; frozen top-8 membership",
              "shuffle_seed_registered": False, "fraction_of_labels_moved": moved, "n_sets": data.n_sets, "n_logs": data.n_logs, "final_rows": final,
              "srttc_r1_code_sha256": R.sha256_bytes(R._io_read_bytes(guard, os.path.abspath(R.__file__))), "r1_result_code_sha256": r1.get("code_sha256"),
              "run_meta": {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": round(time.time() - t0, 1), "torch": torch.__version__, "numpy": np.__version__}}
    R._io_write_bytes(guard, os.path.join(out, RESULT_NAME), R.canon_json(result), exclusive=True)
    R._log(f"C2 rows written for seeds {list(opts.seeds)}: {[final[str(s)]['sha256'][:12] for s in opts.seeds]}")
    return result


def main(argv=None) -> int:
    ap = R.build_parser()
    ap.description = "SR-TTC C2: the shuffled-label negative-control fit at c* (writes ttc_row_c2shuffle_seed{0,1}.npy and r1_c2_result.json). Never reads the dev-gate or navtest tables."
    ap.add_argument("--no-gate-stage-ack", action="store_true", help="REQUIRED: acknowledges that srttc_r1_gate.py has no c2 stage (see the module docstring)")
    # `cmd` of srttc_r1's parser is positional; C2 has a single command
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("run", "lc"):
        argv = ["run"] + argv
    a = ap.parse_args(argv)
    if not a.no_gate_stage_ack:
        print("REFUSED: pass --no-gate-stage-ack. A C2 fit is a training run and srttc_r1_gate.py has no c2 stage; the Master Mind rules on that before this runs.", flush=True)
        return 2
    run_c2(R.options_from_args(a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
