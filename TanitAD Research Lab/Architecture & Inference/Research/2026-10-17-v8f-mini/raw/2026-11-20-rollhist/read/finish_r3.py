"""finish_r3.py: after out/refs_r3.log ends 'ZZ R3 REFS DONE ZZ' -> REFS_READ_UEA.json (reader output paths + P2 reference D1/D2/ACC on UE.A), then dry_check.py and the manifest."""
import json, os, subprocess, sys, hashlib
H = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
O = H + "/out"
assert open(O + "/refs_r3.log").read().rstrip().endswith("ZZ R3 REFS DONE ZZ"), "refs readers not done"
out = {"evidence_class": "MEASURED (ours, UE.A under the form of record; a reference that decides nothing)", "log": O + "/refs_r3.log", "hw": O + "/read/refs_hw", "q3": O + "/read/refs_q3",
       "files": sorted(os.listdir(O + "/read/refs_hw") + os.listdir(O + "/read/refs_q3")), "strat_floor": O + "/read/STRAT_FLOOR_UEA.json"}
json.dump(out, open(H + "/REFS_READ_UEA.json", "w"), indent=1)
r = subprocess.run([sys.executable, H + "/dry_check.py"], capture_output=True, text=True)
print(r.stdout[-300:])
man = json.load(open(H + "/tmp/dry_sha.json"))
open(H + "/MANIFEST_SHA256.txt", "w", newline="\n").write("".join(f"{v}  {k}\n" for k, v in sorted(man.items())))
print("manifest", len(man))
