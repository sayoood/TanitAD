"""C2z (Amendment 4; POST-HOC, UNREGISTERED, DESCRIPTIVE): the registered C2 rule's false-RED rate when the probe is TRAINED on noise labels (entry-level permutation) -- L arm, cold start, 8 replicates per row.
Decides nothing.  NEW FILE: headrow_probe.py (gate-bound) is imported unchanged.
  python c2z.py fit <row> [<row> ...]    -> run/f10/c2z/c2z_<row>_rep<k>.npz
  python c2z.py report                   -> run/f10/c2z.json
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
REPS = 8


def perm_labels(pd, row, rep):
    col = H.ROW_COL[row]
    flat = pd.y[col].reshape(-1)
    pi = np.random.default_rng(20261019 + 1000 * rep + 10 * col + 5).permutation(flat.numel())
    return flat[torch.from_numpy(pi)].reshape(pd.y[col].shape).clone()


def main():
    cmd = sys.argv[1]
    a = H.build_parser().parse_args(["c2"])
    opts, guard, out, man_sha, ranked_sha, W6, b6, pd, tok = H._setup_stage(a)
    odir = os.path.join(out, "c2z")
    E._io_makedirs(guard, odir)
    if cmd == "fit":
        cfg = H.ProbeCfg()
        for row in sys.argv[2:]:
            for rep in range(1, REPS + 1):
                p = os.path.join(odir, f"c2z_{row}_rep{rep}.npz")
                if E._io_exists(guard, p):
                    continue
                t0 = time.time()
                res = H.oof_logits(pd, cfg, "L", row, H.C2_SEED, W6, b6, y_train=perm_labels(pd, row, rep), cold=True, ram_guard=True, log=H._log)
                E._io_write_bytes(guard, p, H.npz_bytes(Z=res["Z"], train_bce_by_fold=np.array(res["train_bce_by_fold"])), exclusive=True)
                H._log(f"c2z {row}/rep{rep} done in {time.time() - t0:.0f}s")
    else:
        draws = E.boot_draws(pd.n_logs)
        res = {"schema": "headrow-c2z-v1", "status": "POST-HOC, UNREGISTERED, DESCRIPTIVE (Amendment 4); decides nothing", "arm": "L", "reps": REPS, "rows": {}}
        for row in H.ROWS:
            col = H.ROW_COL[row]
            cells = []
            for rep in range(1, REPS + 1):
                z = np.load(io.BytesIO(E._io_read_bytes(guard, os.path.join(odir, f"c2z_{row}_rep{rep}.npz"))))["Z"]
                n, d = H.conc_per_set(z, pd.y[col], pd.valid[col])
                ci = H.ratio_ci(H.per_log(pd, n), H.per_log(pd, d), draws)
                cells.append({"rep": rep, "conc_minus_half": ci["point"] - 0.5, "lo_minus_half": ci["lo"] - 0.5, "hi_minus_half": ci["hi"] - 0.5})
            k_red = sum(c["lo_minus_half"] > 0 for c in cells)
            k_below = sum(c["hi_minus_half"] < 0 for c in cells)
            pts = [c["conc_minus_half"] for c in cells]
            res["rows"][row] = {"cells": cells, "n_red": k_red, "rate_red": k_red / REPS, "wilson95_red": H.wilson(k_red, REPS), "n_significantly_below": k_below, "min": min(pts), "max": max(pts)}
        res["max_rate_red"] = max(r["rate_red"] for r in res["rows"].values())
        res["c2_rule_miscalibrated_rate_gt_10pct"] = bool(res["max_rate_red"] > 0.10)
        E._io_write_bytes(guard, os.path.join(out, "c2z.json"), E.canon_json(res), exclusive=True)
        print(json.dumps({r: (v["n_red"], v["n_significantly_below"], round(v["min"], 3), round(v["max"], 3)) for r, v in res["rows"].items()}), flush=True)


if __name__ == "__main__":
    main()
