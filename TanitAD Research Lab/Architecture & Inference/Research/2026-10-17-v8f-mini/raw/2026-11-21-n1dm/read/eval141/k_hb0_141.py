"""k_hb0_141.py -- control K-HB0-141 of the N1DH eval-141 chain.  The smoke dump of N1D_s0 made through the hb wrapper with --kappa-max none (smoke_khb0/dump_N1DH-N1D_s0_m10_causal_heads) must equal the BANKED
old-executor N1D_s0 eval-141 dump (banked_n1d/x/dumps141/dump_NAVRBD-N1D_s0_m10_causal_heads, from navrbd_read_outputs.tar sha256 143f3b2a) array for array: same episode files, same keys, same shapes and dtypes,
np.array_equal with NaN equal to NaN.  Beside, the ticks file is compared the same way.  Writes K-HB0-141.json (pass only when every array is equal; the file is the gate the chain reads).
Usage: python k_hb0_141.py [NEW_DUMP_DIR BANKED_DUMP_DIR OUT.json]"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = sys.argv[1] if len(sys.argv) > 1 else HERE + "/smoke_khb0/dump_N1DH-N1D_s0_m10_causal_heads"
OLD = sys.argv[2] if len(sys.argv) > 2 else HERE + "/banked_n1d/x/dumps141/dump_NAVRBD-N1D_s0_m10_causal_heads"
OUT = sys.argv[3] if len(sys.argv) > 3 else HERE + "/K-HB0-141.json"


def cmp_npz(a, b, skip=()):
    za, zb = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    ka, kb = sorted(za.files), sorted(zb.files)
    rep = {"keys_equal": ka == kb, "arrays": 0, "unequal": []}
    for k in sorted(set(ka) & set(kb)):
        if k in skip:
            continue
        x, y = za[k], zb[k]
        rep["arrays"] += 1
        if x.shape != y.shape or x.dtype != y.dtype:
            rep["unequal"].append(f"{k}: shape/dtype {x.shape}{x.dtype} vs {y.shape}{y.dtype}")
            continue
        same = np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y)
        if not same:
            mx = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.dtype.kind in "fiu" and x.size else None
            rep["unequal"].append(f"{k}: max abs diff {mx}")
    rep["only_in_new"], rep["only_in_banked"] = sorted(set(ka) - set(kb)), sorted(set(kb) - set(ka))
    return rep


fa = sorted(f for f in os.listdir(NEW) if f.endswith(".npz")) if os.path.isdir(NEW) else []
fb = sorted(f for f in os.listdir(OLD) if f.endswith(".npz")) if os.path.isdir(OLD) else []
res = {"new": NEW, "banked": OLD, "files_new": len(fa), "files_banked": len(fb), "files_equal_set": fa == fb, "n_arrays": 0, "unequal": {}, "keys_mismatch": []}
for f in sorted(set(fa) & set(fb)):
    r = cmp_npz(f"{NEW}/{f}", f"{OLD}/{f}")
    res["n_arrays"] += r["arrays"]
    if r["unequal"]:
        res["unequal"][f] = r["unequal"]
    if not r["keys_equal"]:
        res["keys_mismatch"].append(f)
tk_new = os.path.dirname(NEW) + "/ticks/ticks_" + os.path.basename(NEW.rstrip("/"))[len("dump_"):] + ".npz"      # MM 2026-11-21 (N1DM Amendment 5): from the dump's own tag (was hard-coded to the N1DH tag)
tk_old = os.path.dirname(OLD) + "/ticks/ticks_" + os.path.basename(OLD.rstrip("/"))[len("dump_"):] + ".npz"
if os.path.isfile(tk_new) and os.path.isfile(tk_old):
    r = cmp_npz(tk_new, tk_old, skip=("ckpt", "tag"))        # provenance strings (checkpoint path, tag) differ by construction: the file was written under another tag and path
    res["ticks"] = {"arrays": r["arrays"], "unequal": r["unequal"], "keys_equal": r["keys_equal"], "skipped_provenance_keys": ["ckpt", "tag"]}
else:
    res["ticks"] = "missing"
res["n_files"] = len(fa)
tick_ok = isinstance(res["ticks"], dict) and not res["ticks"]["unequal"] and res["ticks"]["keys_equal"]
res["pass"] = bool(fa and fa == fb and len(fa) == 141 and res["n_arrays"] > 0 and not res["unequal"] and not res["keys_mismatch"] and tick_ok)
res["rule"] = "pass iff 141 episode files equal in name, keys, shapes, dtypes and values (NaN equal), and the ticks file equal"
res["executor"] = "v8f_t1_eval_hb.py --kappa-max none (tree bcf3ba3f, N1D_s0 checkpoint sha256 b51d6901) vs the banked old-executor dump (tree 0c3b374f)"
json.dump(res, open(OUT, "w"), indent=1)
print("K-HB0-141", "PASS" if res["pass"] else "FAIL", "files", len(fa), "arrays", res["n_arrays"], "unequal files", len(res["unequal"]), "ticks", "ok" if tick_ok else res["ticks"], tk_new, tk_old)
sys.exit(0 if res["pass"] else 1)
