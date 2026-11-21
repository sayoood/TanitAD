import hashlib
import json
import os

H = os.path.dirname(os.path.abspath(__file__))
F = ("selc6_analyze.py", "selc6_dump.py", "selc6_score.py", "p6_read.py", "p6_control.py", "u400_reference.pinned.json")
json.dump({"what": "sha256 of the P6-read code, frozen BEFORE any P6 value exists (prereg section 3 'CODE_PIN6.json'). selc6_analyze.g6_pin refuses any drift.",
           "files": {f: hashlib.sha256(open(os.path.join(H, f), "rb").read()).hexdigest() for f in F}}, open(os.path.join(H, "CODE_PIN.json"), "w"), indent=1)
