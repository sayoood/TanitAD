"""C2x (Amendment 2; POST-HOC, UNREGISTERED, DESCRIPTIVE): cold-start probes trained on labels of a random PERMUTATION OF THE SETS (feature-label relation destroyed at both levels), scored OOF against the real
labels.  Decides nothing.  NEW FILE: headrow_probe.py (gate-bound) is imported unchanged.
  python c2x.py fit <row> [<row> ...]     -> run/f10/c2x/c2x_<row>_<arm>.npz  (written once; an existing cell is skipped)
  python c2x.py report                    -> run/f10/c2x.json
"""
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import headrow_probe as H   # noqa: E402

E, np, torch = H.E, H.np, H.torch


def perm_labels(pd, row):
    col = H.ROW_COL[row]
    pi = np.random.default_rng(20261019 + 77 + col).permutation(pd.n_sets)
    return pd.y[col][torch.from_numpy(pi)].clone()


def main():
    cmd = sys.argv[1]
    a = H.build_parser().parse_args(["c2"])
    opts, guard, out, man_sha, ranked_sha, W6, b6, pd, tok = H._setup_stage(a)
    odir = os.path.join(out, "c2x")
    E._io_makedirs(guard, odir)
    if cmd == "fit":
        cfg = H.ProbeCfg()
        for row in sys.argv[2:]:
            assert row in H.ROWS
            y_tr = perm_labels(pd, row)
            for arm in H.ARMS:
                p = os.path.join(odir, f"c2x_{row}_{arm}.npz")
                if E._io_exists(guard, p):
                    H._log(f"skip (exists) {p}")
                    continue
                t0 = time.time()
                res = H.oof_logits(pd, cfg, arm, row, H.C2_SEED, W6, b6, y_train=y_tr, cold=True, ram_guard=True, log=H._log)
                E._io_write_bytes(guard, p, H.npz_bytes(Z=res["Z"], train_bce_by_fold=np.array(res["train_bce_by_fold"])), exclusive=True)
                H._log(f"c2x {row}/{arm} done in {time.time() - t0:.0f}s")
    else:
        draws = E.boot_draws(pd.n_logs)
        res = {"schema": "headrow-c2x-v1", "status": "POST-HOC, UNREGISTERED, DESCRIPTIVE (Amendment 2); decides nothing", "cells": {}, "rows_M_minus_L": {}}
        n_red = 0
        for row in H.ROWS:
            col = H.ROW_COL[row]
            lg = {}
            for arm in H.ARMS:
                z = np.load(io.BytesIO(E._io_read_bytes(guard, os.path.join(odir, f"c2x_{row}_{arm}.npz"))))["Z"]
                n, d = H.conc_per_set(z, pd.y[col], pd.valid[col])
                lg[arm] = (H.per_log(pd, n), H.per_log(pd, d))
                ci = H.ratio_ci(*lg[arm], draws)
                red = bool(ci["lo"] - 0.5 > 0)
                n_red += int(red)
                res["cells"][f"{row}/{arm}"] = {"conc": ci, "conc_minus_half": ci["point"] - 0.5, "lo_minus_half": ci["lo"] - 0.5, "red": red}
            res["rows_M_minus_L"][row] = H.diff_ci(lg["M"][0], lg["L"][0], lg["M"][1], draws)
        pts_ok = all(abs(c["conc_minus_half"]) <= 0.01 for c in res["cells"].values())
        res["n_red"] = n_red
        res["all_points_within_0.01"] = bool(pts_ok)
        res["reading_a_supported"] = bool(n_red <= 1 and pts_ok)
        E._io_write_bytes(guard, os.path.join(out, "c2x.json"), E.canon_json(res), exclusive=True)
        print(json.dumps({"n_red": n_red, "pts_ok": pts_ok, "reading_a_supported": res["reading_a_supported"]}), flush=True)


if __name__ == "__main__":
    main()
