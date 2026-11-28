"""pod_smoke_check_q.py <imag log> : plumbing facts of a cand-cvq imag log (columns, shapes, finite counts, meta). No outcome value is read."""
import sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=False)
print("mode", z["mode"].item(), "rows", z["g_head"].shape, "qhead_sha256", str(z["qhead_sha256"].item())[:16], "const", float(z["qhead_const"]), "model_sha", str(z["qhead_model_sha256"].item())[:12])
act = z["active"]
print("active rows", int(act.sum()), "q_head finite", int(np.isfinite(z["q_head"]).sum()), "g_used<=g_head all", bool((z["g_used"][np.isfinite(z["q_head"])] <= z["g_head"][np.isfinite(z["q_head"])] + 1e-6).all()),
      "g_src==g_used", bool(np.array_equal(z["g_src"], z["g_used"], equal_nan=True)))
