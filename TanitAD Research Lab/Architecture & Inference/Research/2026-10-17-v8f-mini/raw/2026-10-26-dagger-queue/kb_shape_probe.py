#!/usr/bin/env python3
"""Do the K-B4 / K-B5 records the CURRENT bank generator writes pass the chain judge's K-B4 / K-B5 rules?

Imports tanitad.data.v8f_dagger from the integrated tree (di/tree, READ-ONLY: PYTHONDONTWRITEBYTECODE, nothing is written there), builds the records exactly as the generator's merge does
(kb4_combine([kb4_record()]), kb5_combine([Kb5Accumulator().record()])), and hands them to box2_ship/v8f_dag_chain_checks.py's kb_record_problem / kb4_problem / kb5_problem.
It checks KEY NAMES and TYPES (PASS boolean, the six case names, the two error keys), not the numbers a real corpus would produce: the K-B5 accumulator here is EMPTY (zero errors, zero records).

    CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 python kb_shape_probe.py [DI_STACK_DIR]
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
STACK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("E:/Projects/TanitAD-artifacts/di/tree/stack")
sys.path.insert(0, str(HERE / "box2_ship"))
sys.path.insert(0, str(STACK))
import v8f_dag_chain_checks as CK  # noqa: E402
from tanitad.data import v8f_dagger as G  # noqa: E402

out = []


def say(x):
    print(x, flush=True)
    out.append(x)


bank = STACK / "scripts" / "v8f_dagger_bank.py"
say(f"generator {bank} sha256 {hashlib.sha256(bank.read_bytes()).hexdigest()[:16]}...; v8f_dagger.py sha256 {hashlib.sha256(Path(G.__file__).read_bytes()).hexdigest()[:16]}...")
k4 = G.kb4_combine([G.kb4_record()])
k5 = G.kb5_combine([G.Kb5Accumulator().record()])
say("K-B4 as the merge writes it: " + json.dumps(k4, sort_keys=True)[:400])
say("K-B5 as the merge writes it (empty accumulator): " + json.dumps({k: k5[k] for k in ("PASS", "max_pose_now_err_m", "max_rel_transform_err_m") if k in k5}, sort_keys=True) + f" ... keys {sorted(k5)}")
res = {
    "K-B4 record: PASS exactly True": k4.get("PASS") is True,
    "K-B4: kb_record_problem is None": CK.kb_record_problem(k4, "K-B4") is None,
    "K-B4: kb4_problem is None (all six cases True)": CK.kb4_problem(k4) is None,
    "K-B4: the generator's case names == the judge's": tuple(k4["cases"]) == CK.KB4_CASES or set(k4["cases"]) == set(CK.KB4_CASES),
    "K-B5: both error keys present and finite numbers": all(isinstance(k5.get(k), (int, float)) and not isinstance(k5.get(k), bool) and math.isfinite(k5[k]) for k in ("max_pose_now_err_m", "max_rel_transform_err_m")),
    "K-B5: kb5_problem is None": CK.kb5_problem(k5) is None,
    "K-B5: tolerance == the judge's": k5.get("tol_m") == CK.KB5_TOL_M,
}
# a record with the numbers over tolerance must be refused by BOTH the generator's PASS and the judge
bad5 = dict(k5, max_pose_now_err_m=2e-4)
res["K-B5 at 2e-4: the generator's kb5_pass says False AND the judge refuses"] = G.kb5_pass(bad5) is False and CK.kb5_problem(bad5) is not None
for k, v in res.items():
    say(("PASS  " if v else "FAIL  ") + k)
n_bad = sum(1 for v in res.values() if not v)
say(f"KB-SHAPE-PROBE: {len(res) - n_bad}/{len(res)} pass")
(HERE / "kb_shape_probe.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
sys.exit(1 if n_bad else 0)
