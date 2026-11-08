#!/usr/bin/env python3
"""Write CODE_PIN.json: sha256 of every code/data file the confirmation's verdict depends on. Run ONCE before the prereg is frozen."""
import hashlib, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["selc_rule.py", "selc_analyze.py", "selc_score.py", "selc_dump.py", "selc_gates.py", "selc_chain.py", "make_confirm_tokens.py",
         "tokens_confirm.json", "tokens_smoke.json", "fc_masked_tokens.json"]
pin = {"what": "sha256 of the files the H-SEL-1 confirmation depends on; selc_score.check_pins refuses any drift",
       "files": {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest() for f in FILES}}
with open(os.path.join(HERE, "CODE_PIN.json"), "wb") as f:
    f.write(json.dumps(pin, indent=1).encode("utf-8"))
print(json.dumps(pin, indent=1))
print("CODE_PIN.json sha256", hashlib.sha256(open(os.path.join(HERE, "CODE_PIN.json"), "rb").read()).hexdigest())
