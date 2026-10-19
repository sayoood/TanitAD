"""Writes PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.md (a DRAFT) from the live bytes.  Reads file hashes only; opens no label, no array, no scalar store, writes no .sha256 and no seal."""
import hashlib
import json
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
SEALED = json.load(open(f"{ROOT}/SEALED_AT.txt", encoding="utf-8"))["code_sha256"]
FILES = ("code/pgc_read_chain.py", "code/pgc_seal.py", "code/PINS.json")
repin = {n: {"from": SEALED[n], "to": sha(f"{ROOT}/{n}")} for n in FILES}
unchanged = [n for n in SEALED if n not in FILES]
assert all(sha(f"{ROOT}/{n}") == SEALED[n] for n in unchanged), "an unlisted pinned file differs from the seal"
assert all(v["from"] != v["to"] for v in repin.values())
man = {k: sha(f"{ROOT}/ops1/{k}") for k in ("scalars_npz_sha256.txt", "scalars_timing_sha256.txt", "worker_done_sha256.txt")}
unp = {k: sha(f"{ROOT}/code/{k}") for k in ("test_ops1.py", "fake_pg_extract_ops1.py")}
dif = {k: sha(f"{ROOT}/ops1/{k}") for k in ("pgc_read_chain.py.diff", "pgc_seal.py.diff", "PINS.json.diff")}
block = json.dumps({"scope": "operational-only", "repin": repin}, indent=1)
rows = "\n".join(f"| `{n.split('/', 1)[1]}` | `{repin[n]['from']}` | `{repin[n]['to']}` |" for n in FILES)
T = open(f"{ROOT}/ops1/AMENDMENT_TEMPLATE.md", encoding="utf-8").read()
out = (T.replace("@@REPIN_BLOCK@@", block).replace("@@PINROWS@@", rows).replace("@@N_UNCHANGED@@", str(len(unchanged))).replace("@@PEX@@", SEALED["code/pg_extract.py"])
       .replace("@@MAN_NPZ@@", man["scalars_npz_sha256.txt"]).replace("@@MAN_TIM@@", man["scalars_timing_sha256.txt"]).replace("@@MAN_WD@@", man["worker_done_sha256.txt"])
       .replace("@@TESTOPS@@", unp["test_ops1.py"]).replace("@@FAKE@@", unp["fake_pg_extract_ops1.py"])
       .replace("@@D_CHAIN@@", dif["pgc_read_chain.py.diff"]).replace("@@D_SEAL@@", dif["pgc_seal.py.diff"]).replace("@@D_PINS@@", dif["PINS.json.diff"]))
assert "@@" not in out
open(f"{ROOT}/PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.md", "w", encoding="utf-8", newline="\n").write(out)
print("wrote", len(out), "bytes;", {n: v["to"][:12] for n, v in repin.items()})
