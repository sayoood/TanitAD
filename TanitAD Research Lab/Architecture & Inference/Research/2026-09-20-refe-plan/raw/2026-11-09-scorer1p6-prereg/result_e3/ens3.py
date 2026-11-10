"""ENS3, read after the verdict (PREREG_SCORER1P6 section 5): emit ENS3's unscored picks, score them in chunks, re-run the read with --ens3-csv into verdict_ens3/."""
import csv, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, "F:/tanitad_scratch/p6arms/read")
import chain_read as C, chain_read2 as R2
OUT = C.OUT; em = OUT / "emit_ens3"
armargs = sum([["--arm", f"{a}={C.PASS / a}"] for a in C.ARMS], []) + ["--arm", "ENS3"]
rc = subprocess.call([C.S2.PY_D, "read_scorer1p6.py", "emit-changed", *C.COMMON, *armargs, "--out", str(em)], cwd=C.E, stdout=open(OUT / "emit_ens3.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
print("emit rc", rc, sorted(p.name for p in em.glob("*")))
j = json.load(open(em / "tokens_ENS3.json", encoding="utf-8")); toks = j["tokens"]; print("ENS3 tokens", len(toks))
R2.OUT = OUT
rows, header = [], None
for k in range(0, len(toks), 1000):
    p = R2.score_chunk("ENS3", em / "seam_ENS3.npz", toks[k:k + 1000], j["token_log"], k // 1000)
    with open(p, encoding="utf-8", newline="") as f:
        r = csv.reader(f); h = next(r); header = header or h; rows += [x for x in r if x[1] not in ("average", "")]
assert {x[1] for x in rows} == set(toks), "coverage"
dst = OUT / "score" / "merged_ENS3.csv"
with open(dst, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(header); [w.writerow([i] + x[1:]) for i, x in enumerate(rows)]
cmd = [C.S2.PY_D, "read_scorer1p6.py", "read", *C.COMMON, "--p6-csv", "E:/Projects/TanitAD/data/refe_navtest/score/refe_p6final/refe_p6final.csv", *armargs[:-2],
       "--c3-tokens", "E:/Projects/TanitAD-artifacts/refe_ego_probe_20261106/tokens_all.json", "--out", str(OUT / "verdict_ens3"), "--ens3-csv", str(dst)]
print("read rc", subprocess.call(cmd, cwd=C.E, stdout=open(OUT / "read_ens3.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
