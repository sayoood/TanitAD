"""DIAGNOSTIC what-if: a COPY of the 100-step measurement smoke's config.json with (a) the nav-tau seam values set to
the A2 tau the rebuilt model carries and (b) argv := the provisional A2 argv, so G-EVAL's later stages can run.
NOT admissible for a binding G-EVAL (that needs the trainer's own config.json from THIS argv's smoke)."""
import json, sys
W = "/home/nvidia/pregate_d014ff2"
c = json.load(open(f"{W}/work/stamps/config_refcv8_smoke_measure.json"))
patch = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
a = json.load(open(f"{W}/work/argv_PROVISIONAL.json"))["argv"]
c["argv"] = a
seam = c["seams"]["nav_compliance_tau_file"]
seam["sha256"] = "9fa7ca38b529bacfe79209c4bb9ccad039c7af284e5080f1c4f856ea46ebb5a3"
seam["tau"] = 0.1806411302089691
json.dump(c, open(f"{W}/work/stamps/config_DIAG_PATCHED.json", "w"), indent=1)
print("seam", json.dumps(seam)[:300])
