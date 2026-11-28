"""make_pins.py -- READ_PINS.json: sha256 of the read tree's key files, taken from the commit's own blobs (git cat-file, raw bytes), and of the inputs.  Usage: python make_pins.py"""
import hashlib, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import strat1_lib as L
MIRROR = "C:/Users/Admin/tanitad-push"
FILES = ["stack/scripts/v8f_t1_eval.py", "stack/scripts/v8f_mini_eval.py", "stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py",
         "taniteval/taniteval/ci.py", "taniteval/taniteval/v8f_consistency.py", "taniteval/taniteval/four_families.py"]
pins = {}
for f in FILES:
    b = subprocess.run(["git", "-C", MIRROR, "cat-file", "blob", f"{L.COMMIT}:{f}"], capture_output=True, check=True).stdout
    pins[f] = hashlib.sha256(b).hexdigest()
    got = L.sha256_file(f"{L.TREE}/{f}")
    assert got == pins[f], (f, "export differs from the commit's blob")
json.dump({"commit": L.COMMIT, "tree_sha256": pins, "inputs_sha256": {k: v for k, v in L.PINS.items()},
           "run_dumps_sha256": L.sha256_file(HERE + "/theory/otband/read/run_dumps.py"), "read_otband_sha256": L.sha256_file(HERE + "/theory/otband/read/read_otband.py")},
          open(HERE + "/READ_PINS.json", "w"), indent=1)
print("READ_PINS.json written", len(pins))
