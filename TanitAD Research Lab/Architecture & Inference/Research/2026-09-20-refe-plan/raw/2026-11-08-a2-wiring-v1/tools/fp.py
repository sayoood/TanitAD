import sys, json, hashlib, importlib
sys.path.insert(0, sys.argv[1])
import launch_gate as LG
json.dump({n: hashlib.sha256(repr(sorted(LG.PROFILES[n].items())).encode()).hexdigest() for n in sorted(LG.PROFILES)}, open(sys.argv[2], "w"), indent=0)
print(len(LG.PROFILES))
