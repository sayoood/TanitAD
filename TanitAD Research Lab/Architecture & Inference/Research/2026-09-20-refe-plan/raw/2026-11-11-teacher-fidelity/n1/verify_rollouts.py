"""Verify the banked P4 navtest teacher rollouts for N1: bytes (raw vs repo .xz), token coverage vs the T seam and the
12,110 primary, row shape, and the seam identity to_navsim(traj20) -> float32 == T seam poses. No scorer, no metric."""
import glob, hashlib, json, lzma, subprocess, sys
import numpy as np
sys.path.insert(0, "C:/Users/Admin/n1_scratch/w/pk/refe")
import navsim_dac as ND
RAW = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-05-teacher-navtest/full"
SEAM = "C:/Users/Admin/n1_scratch/w/pk/raw/2026-10-05-teacher-navtest/refe_p4_teacher_navtest.npz"
FC = "C:/Users/Admin/n1_scratch/w/fc36.json"
def sha(b): return hashlib.sha256(b).hexdigest()
out = {"files": {}}
z = np.load(SEAM, allow_pickle=False); order = [str(t) for t in z["token"]]; idx = {t: i for i, t in enumerate(order)}
known = set(json.load(open(FC))["over_bar"])
rows = {}
for i in range(4):
    raw = open(f"{RAW}/rows_s{i}of4.jsonl", "rb").read()
    xz_local = open(f"{RAW}/rows_s{i}of4.jsonl.xz", "rb").read()
    out["files"][f"rows_s{i}of4.jsonl"] = {"bytes": len(raw), "sha256": sha(raw), "xz_bytes": len(xz_local), "xz_sha256": sha(xz_local),
        "xz_decompressed_equals_raw": lzma.decompress(xz_local) == raw}
    for ln in raw.decode("utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln); t = r["token"]
            if t in rows: out.setdefault("dups", []).append(t)
            p20 = np.asarray(r["traj20"], np.float64)
            rows[t] = (r["log"], p20.shape == (20, 3) and bool(np.isfinite(p20).all()), r.get("dt"),
                       bool(np.array_equal(ND.to_navsim(p20).astype(np.float32), z["poses"][idx[t]])) if t in idx else None,
                       float(np.abs(ND.to_navsim(p20) - np.asarray(r["plan8"])).max()))
toks = set(rows)
out["n_rows"] = len(rows); out["n_seam"] = len(order)
out["token_sets_equal_seam"] = toks == set(order)
out["primary_12110_covered"] = len(set(order) - known - toks) == 0 and len(set(order) - known) == 12110
out["n_known36_in_rows"] = len(known & toks)
out["all_traj20_finite_20x3"] = all(v[1] for v in rows.values())
out["dt_values"] = sorted({v[2] for v in rows.values()})
out["n_logs"] = len({v[0] for v in rows.values()})
out["n_logs_primary"] = len({rows[t][0] for t in rows if t not in known})
out["c_seam_float32_equal"] = sum(bool(v[3]) for v in rows.values())
out["plan8_vs_to_navsim_max_abs"] = max(v[4] for v in rows.values())
json.dump(out, open("C:/Users/Admin/n1_scratch/rollout_verify.json", "w"), indent=1)
print(json.dumps(out, indent=1))
