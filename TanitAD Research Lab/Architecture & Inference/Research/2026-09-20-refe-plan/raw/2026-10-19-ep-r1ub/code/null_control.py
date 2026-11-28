"""POST-HOC, DESCRIPTIVE, UNREGISTERED control for reading outcome N of PREREG_EPR1UB.  It trains NOTHING and gates NOTHING.

Question: is "the unsafe-pick rate of ANY pick-changing EP row is above P2's" a property of the comparator (P2's pick is its own in-sample argmax, so every deviation from it costs safety on these sets)
rather than of the learned rows?  Test: replace P2's EP row by P2's row + Gaussian noise (no learning), score the pick against P2's pick on the SAME sets with the SAME estimator
(landed `epr1.score_sets`, per-log sums, `epr1.paired_delta`, the 10,000 / seed 20260927 draws).  Same bank, same PathGuard, f = 10 %.
Read it as: if noise rows that change as many picks as the fitted rows do also read dUnsafe > 0, the constraint 'dUnsafe upper CI <= 0 vs P2' is close to unsatisfiable for any row that moves the pick.
"""
import json
import sys
import time

sys.path.insert(0, "C:/Users/Admin/epr1ub/code")
import numpy as np  # noqa: E402
import epr1ub as U  # noqa: E402

E = U.E
opts = U.options_from_args(U.build_parser().parse_args(["run", "--fraction", "10", "--out-dir", "C:/Users/Admin/epr1ub/run/f10_null"]))
guard, out, manifest, man_sha, ranked_sha, selected, repl_rep, per_fold_logs, W6, b6, data = E._setup(opts)
E._io_makedirs(guard, out)
p2_w, p2_b = np.asarray(W6, np.float32)[E.EP_COL], np.asarray(b6, np.float32)[E.EP_COL:E.EP_COL + 1]
idx = np.arange(data.n_sets)
draws = E.boot_draws(data.n_logs)
rows = []
t0 = time.time()
for sigma in (0.001, 0.003, 0.01, 0.03, 0.1):
    for ns in (0, 1):
        rng = np.random.default_rng(1000 + ns)
        w = (p2_w + rng.normal(0.0, sigma, p2_w.shape)).astype(np.float32)
        b = (p2_b + rng.normal(0.0, sigma, p2_b.shape)).astype(np.float32)
        st = E.score_sets(data, idx, w, b, p2_w, p2_b)
        n_log = E._bin(data, idx, np.ones(len(idx)))
        du = E.paired_delta(E._bin(data, idx, st["unsafe_new"]), E._bin(data, idx, st["unsafe_p2"]), n_log, draws)
        dp = E.paired_delta(E._bin(data, idx, st["pd_new"]), E._bin(data, idx, st["pd_p2"]), n_log, draws)
        ch = float((st["picks"] != st["pick_p2"]).mean())
        rec = {"sigma": sigma, "noise_seed": ns, "row_max_abs_change": float(max(np.abs(w - p2_w).max(), np.abs(b - p2_b).max())), "changed_pick_fraction": ch,
               "unsafe_rate_new": float(st["unsafe_new"].mean()), "unsafe_rate_p2": float(st["unsafe_p2"].mean()),
               "d_unsafe": {k: dp_ for k, dp_ in du.items()}, "d_pdms_x100": {k: (100.0 * v if k in ("point", "lo", "hi") else v) for k, v in dp.items()}}
        rows.append(rec)
        print(f"[{time.time() - t0:5.0f}s] sigma {sigma:<6g} seed {ns} changed {ch:.3f}  dUnsafe {du['point']:+.4f} [{du['lo']:+.4f}, {du['hi']:+.4f}]  dPDMS x100 {100 * dp['point']:+.3f}", flush=True)
doc = {"schema": "epr1ub-null-control-v1", "status": "POST-HOC, DESCRIPTIVE, UNREGISTERED; trains nothing; gates nothing", "bank_manifest_sha256": man_sha, "n_sets": data.n_sets, "n_logs": data.n_logs,
       "estimator": "paired log-cluster bootstrap, 10,000 resamples, seed 20260927 (landed epr1.paired_delta); comparator P2's pick is IN-SAMPLE on these sets", "rows": rows,
       "code_sha256": {"epr1.py": E.sha256_bytes(E._io_read_bytes(guard, E.__file__)), "epr1ub.py": E.sha256_bytes(E._io_read_bytes(guard, U.__file__))}}
E._io_write_bytes(guard, out + "/null_control.json", E.canon_json(doc), exclusive=True)
print("wrote", out + "/null_control.json")
