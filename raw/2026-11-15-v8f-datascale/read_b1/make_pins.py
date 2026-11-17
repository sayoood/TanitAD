"""make_pins.py -- READ_PINS.json: sha256 of the read tree's key files from the code-of-record commit's own blobs (git cat-file, raw bytes), and a check that the export equals them.  Usage: python make_pins.py"""
import hashlib, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ds_lib as L
MIRROR = "C:/Users/Admin/tanitad-push"
pins = {}
for f in L.TREE_FILES:
    b = subprocess.run(["git", "-C", MIRROR, "cat-file", "blob", f"{L.COMMIT}:{f}"], capture_output=True, check=True).stdout
    pins[f] = hashlib.sha256(b).hexdigest()
    assert L.sha256_file(f"{L.TREE}/{f}") == pins[f], (f, "export differs from the commit's blob")
json.dump({"commit": L.COMMIT, "tree_sha256": pins, "inputs_sha256": L.PINS}, open(HERE + "/READ_PINS.json", "w"), indent=1)
print("READ_PINS.json written", len(pins))
