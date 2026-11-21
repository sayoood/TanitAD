"""Scheduling helper (NO logic change): the registered `c2` stage of headrow_probe.py for ONE row and an explicit LIST OF REPLICATES in the order given (so a second worker can start from the replicate the first
worker reaches last).  It calls exactly the functions `stage_c2` calls, with the same arguments, seeds and output format; cells are independent and deterministic; an existing cell is skipped, and a cell that
the other worker finished first makes the exclusive write fail (the table on disk stays the first one, bit-identical by determinism).
usage: python run_c2_cells.py <row> <rep,rep,...>"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import headrow_probe as H   # noqa: E402

E = H.E
row = sys.argv[1]
reps = [int(x) for x in sys.argv[2].split(",")]
assert row in H.ROWS and all(1 <= r <= H.C2_REPS for r in reps)
a = H.build_parser().parse_args(["c2"])
opts, guard, out, man_sha, ranked_sha, W6, b6, pd, tok = H._setup_stage(a)
cfg = H.ProbeCfg()
if not E._io_exists(guard, os.path.join(out, "cid.json")):
    raise H.ProbeError("REFUSED: cid.json missing")
odir = os.path.join(out, "c2")
E._io_makedirs(guard, odir)
col = H.ROW_COL[row]
for rep in reps:
    y_shuf = H.shuffle_valid(pd.y[col], pd.valid[col], H.c2_seed(rep, row))
    for arm in H.ARMS:
        p = os.path.join(odir, f"c2_{row}_{arm}_rep{rep}.npz")
        if E._io_exists(guard, p):
            H._log(f"skip (exists) {p}")
            continue
        import time
        t0 = time.time()
        res = H.oof_logits(pd, cfg, arm, row, H.C2_SEED, W6, b6, y_train=y_shuf, cold=True, ram_guard=True, log=H._log)
        E._io_write_bytes(guard, p, H.npz_bytes(Z=res["Z"], train_bce_by_fold=H.np.array(res["train_bce_by_fold"]), fit_s_by_fold=H.np.array(res["fit_s_by_fold"])), exclusive=True)
        H._log(f"c2 {row}/{arm}/rep{rep} done in {time.time() - t0:.0f}s (worker B)")
