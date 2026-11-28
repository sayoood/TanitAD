#!/usr/bin/env python3
"""rehearse_reader.py -- lead's rehearsal of the reader of card ADD110 on the pod BEFORE the rollouts of the card exist: the four arm names are pointed at banked rollouts of card ADD106
(G2, G3, G4, E3c) so that every engine call of the controls stage runs once on real files. It opens no value of card ADD110 (none exists). Expected: a1, a3, a5 pass; a2 fails (the
stand-in rollouts used other gap files); a4 as given. Prints the pass flags and the failure texts of the controls only; the payload of the rehearsal is not printed.
Usage (pod, in the working directory and environment of podread_fallattr10k.sh): rehearse_reader.py <out json>"""
import importlib.util
import json
import os
import sys
import time

V = "/workspace/v8proof"
W, P, FA, CS = V + "/integ10k", V + "/podread", V + "/fallattr", V + "/cropscale"
RD, CODE = P + "/root/integ10k", P + "/code"
K, KG, GF, CHK = FA + "/kit_read", FA + "/kit_gap", FA + "/gapfiles", V + "/visrule10k/check_v"
out = sys.argv[1]
spec = importlib.util.spec_from_file_location("read_fallattr10k", K + "/read_fallattr10k.py")
RF = importlib.util.module_from_spec(spec)
sys.modules["read_fallattr10k"] = RF
spec.loader.exec_module(RF)
RF.RUN = {"LA": "RQ50KF5P30Hb20gG2", "LN": "RQ50KF5P30Hb20gG3", "LM": "RQ50KF5P30Hb20gG4", "LF": "RQ50KF5P30Hb20gE3c"}
rc = RF.main(["--stage", "controls", "--model-reader", CS + "/kit_read/read_cropscale10k.py", "--pull", RD + "/pull", "--code", CODE, "--grid-dir", W + "/longroll", "--w", W, "--tree", W + "/tree",
              "--wrapper-file", W + "/build/v8f_t1_eval_reader_rule_rzgfhqetjv.py", "--fa", FA, "--gap-dir", GF, "--gap-record", GF + "/FALLATTR_GAPFILES_RECORD.json",
              "--eval-join", V + "/hierrep/pins/join_eval141.npz", "--wrapper-check", CHK + "/CHECK_WRAPPER_V_DONE.txt", "--wrapper-hashes", CHK + "/CHECK_WRAPPER_V.hashes",
              "--tests-gap", KG + "/lead_pytest.out", "--tests-read", K + "/lead_pytest.out", "--tests-mtime-gap", str(int(os.path.getmtime(KG + "/lead_pytest.out"))),
              "--tests-mtime-read", str(int(os.path.getmtime(K + "/lead_pytest.out"))), "--queue-start", str(int(time.time())), "--out", out])
print("REHEARSAL rc", rc)
d = json.load(open(out, encoding="utf-8"))
for k in sorted(d.get("controls", {})):
    c = d["controls"][k]
    print("REHEARSAL", k, "pass", c.get("pass"), "| failures:", " ;; ".join(str(x)[:160] for x in (c.get("failures") or [])[:3]))
print("REHEARSAL keys of the value part:", sorted((d.get("payload") or d.get("withheld") or {}).keys()))
