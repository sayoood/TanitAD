#!/usr/bin/env python3
"""CODE_SHA256.json: sha256 of every code file of the package (what ran, what was added after the launch)."""
import glob
import hashlib
import json
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace(chr(92), "/")
out = {"note": "p2ho_audit.py gained the EXTRA_DB_DIR lookup AFTER the main launch (the audit had already run; not executed again unless the job is relaunched). p2ho_featcheck.py, "
                "make_paths.py, p2ho_finalize.py, p2ho_hashcode.py, status_p2ho.py and test_p2ho.py are post-launch helpers; the dump/score/fidelity/dgdump/preflight/driver files are as launched.",
       "files": {}}
for p in sorted(glob.glob(root + "/code/*.py") + [f for f in glob.glob(root + "/landed/*") if os.path.isfile(f)] + [root + "/launch_p2ho.ps1"]):
    out["files"][os.path.relpath(p, root).replace(chr(92), "/")] = hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump(out, open(root + "/CODE_SHA256.json", "w"), indent=1)
print(len(out["files"]), "files hashed")
