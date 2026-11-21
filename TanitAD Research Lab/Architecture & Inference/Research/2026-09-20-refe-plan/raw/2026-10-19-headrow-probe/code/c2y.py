"""C2y + index diagnostic (Amendment 3; POST-HOC, UNREGISTERED, DESCRIPTIVE): cold-start probes trained on an ENTRY-LEVEL permutation of the labels, scored OOF against the real
labels.  Decides nothing.  NEW FILE: headrow_probe.py (gate-bound) is imported unchanged.
  python c2y.py fit <row> [<row> ...]     -> run/f10/c2y/c2y_<row>_<arm>.npz  (written once; an existing cell is skipped)
  python c2y.py report                    -> run/f10/c2y.json
  python c2y.py index                     -> run/f10/index_diag.json   (no training: concordance of z = proposal index vs the real labels)
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
    """ENTRY-level permutation: the flattened [sets x 64] labels of the row, randomly permuted (independent of set, position and features)."""
    col = H.ROW_COL[row]
    flat = pd.y[col].reshape(-1)
    pi = np.random.default_rng(20261019 + 177 + col).permutation(flat.numel())
    return flat[torch.from_numpy(pi)].reshape(pd.y[col].shape).clone()


def main():
    cmd = sys.argv[1]
    a = H.build_parser().parse_args(["c2"])
    opts, guard, out, man_sha, ranked_sha, W6, b6, pd, tok = H._setup_stage(a)
    odir = os.path.join(out, "c2y")
    E._io_makedirs(guard, odir)
    if cmd == "fit":
        cfg = H.ProbeCfg()
        for row in sys.argv[2:]:
            assert row in H.ROWS
            y_tr = perm_labels(pd, row)
            for arm in H.ARMS:
                p = os.path.join(odir, f"c2y_{row}_{arm}.npz")
                if E._io_exists(guard, p):
                    H._log(f"skip (exists) {p}")
                    continue
                t0 = time.time()
                res = H.oof_logits(pd, cfg, arm, row, H.C2_SEED, W6, b6, y_train=y_tr, cold=True, ram_guard=True, log=H._log)
                E._io_write_bytes(guard, p, H.npz_bytes(Z=res["Z"], train_bce_by_fold=np.array(res["train_bce_by_fold"])), exclusive=True)
                H._log(f"c2y {row}/{arm} done in {time.time() - t0:.0f}s")
    elif cmd == "index":
        draws = E.boot_draws(pd.n_logs)
        zi = np.tile(np.arange(64, dtype=np.float32), (pd.n_sets, 1))
        res = {"schema": "headrow-index-diag-v1", "status": "POST-HOC, UNREGISTERED, DESCRIPTIVE (Amendment 3); no training", "rows": {}}
        for row in H.ROWS:
            col = H.ROW_COL[row]
            n, d = H.conc_per_set(zi, pd.y[col], pd.valid[col])
            res["rows"][row] = H.ratio_ci(H.per_log(pd, n), H.per_log(pd, d), draws)
        E._io_write_bytes(guard, os.path.join(out, "index_diag.json"), E.canon_json(res), exclusive=True)
        print(json.dumps({r: round(v["point"], 4) for r, v in res["rows"].items()}), flush=True)
    else:
        draws = E.boot_draws(pd.n_logs)
        res = {"schema": "headrow-c2y-v1", "status": "POST-HOC, UNREGISTERED, DESCRIPTIVE (Amendment 3); decides nothing", "cells": {}, "rows_M_minus_L": {}}
        n_red = 0
        for row in H.ROWS:
            col = H.ROW_COL[row]
            lg = {}
            for arm in H.ARMS:
                z = np.load(io.BytesIO(E._io_read_bytes(guard, os.path.join(odir, f"c2y_{row}_{arm}.npz"))))["Z"]
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
        res["reading_c_supported"] = bool(n_red <= 1 and pts_ok)
        E._io_write_bytes(guard, os.path.join(out, "c2y.json"), E.canon_json(res), exclusive=True)
        print(json.dumps({"n_red": n_red, "pts_ok": pts_ok, "reading_c_supported": res["reading_c_supported"]}), flush=True)


if __name__ == "__main__":
    main()
