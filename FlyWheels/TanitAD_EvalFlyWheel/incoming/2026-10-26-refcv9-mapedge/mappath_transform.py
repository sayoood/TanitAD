"""mappath_transform.py -- PREREG_MAPPATH section 1: rewrite the MAPSEL trace with a PROGRESS-MATCHED fan (each candidate's path sampled at the native
emitted plan's per-instant arc length); every other field byte-identical.  Writes the K-ID / K-PROG control record beside it.  CPU, 0 GPU."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l1/code")
import l1_fit_score as F  # noqa: E402
AL, PH = F.AL, F.PH
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
z = np.load(src, allow_pickle=False)
d = {k: z[k] for k in z.files}
fan, sel, plan8 = d["fan"].astype(np.float64), d["sel"].astype(np.int64), d["plan8"].astype(np.float64)
n, N, S, _ = fan.shape
nat = fan[np.arange(n), sel]
s_nat = AL.arc(nat)                                            # [n, 8] per-instant cumulative arc length of the native plan
flat = fan.reshape(n * N, S, 2)
s_rep = np.repeat(s_nat, N, axis=0)
out = PH.at_progress(flat, s_rep).reshape(n, N, S, 2)
kid = bool(np.array_equal(out[np.arange(n), sel].astype(z["fan"].dtype), z["fan"][np.arange(n), sel]) and np.array_equal(z["fan"][np.arange(n), sel], z["plan8"]))
dprog = np.abs(AL.arc(out.reshape(n * N, S, 2)).reshape(n, N, S) - s_nat[:, None, :]).max(-1)      # [n, N]
deg = (AL.arc(flat)[:, -1] <= 1e-6).reshape(n, N)
meta = json.loads(str(z["meta_json"]))
meta["mappath_transform"] = {"src_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "rule": "fan'[c] = at_progress(fan[c], arc(fan[sel]))"}
d["fan"] = out.astype(z["fan"].dtype)
d["meta_json"] = np.array(json.dumps(meta))
np.savez(dst, **d)
rec = {"K-ID_native_unchanged_bit_equal": kid, "K-PROG": {"max_abs_progress_diff_le_1e-4_share_of_candidates": float((dprog <= 1e-4).mean()),
       "windows_all_candidates_matched_share": float((dprog <= 1e-4).all(1).mean()), "degenerate_candidate_paths": int(deg.sum()),
       "p99_max_abs_progress_diff_m": float(np.quantile(dprog, 0.99))}, "n_windows": int(n), "n_candidates": int(N), "dst": str(dst)}
Path(sys.argv[3]).write_text(json.dumps(rec, indent=1))
print(json.dumps(rec))
