"""PRE-GATE step 2 (NON-BINDING): the gate's in-process cheap rules on a PROVISIONAL refcv8 A2 argv. CPU only."""
import json, sys, os, time
from pathlib import Path
T = "/home/nvidia/refcv6_run/d014ff2fd6"
W = "/home/nvidia/pregate_d014ff2"
sys.path[:0] = [f"{T}/stack/scripts", f"{T}/stack", f"{T}/taniteval"]
import launch_gate as G
print("launch_gate from", G.__file__)
tree = Path(T)
prof = G.PROFILES["refcv8"]
argv = G.load_argv_file(f"{W}/work/argv_PROVISIONAL.json")
dflt = G._resolve_repo_rel(tree, prof["nav_tau_record_default"])
print("nav_tau_record_default", prof["nav_tau_record_default"], "->", dflt, "is_file", dflt.is_file())
options = {"nav_tau_record": str(dflt) if dflt.is_file() else None}
clock_ref = [str(G._resolve_repo_rel(tree, r)) for r in prof["clock_reference"]]
print("clock_reference", clock_ref, [Path(c).is_file() for c in clock_ref])
pmap = []
out = {"argv_sha256": G.argv_sha256(argv), "n_tokens": len(argv), "options": options, "clock_reference": clock_ref}
# --- rules
refusals, pi_pending = G.profile_argv_rules(prof, argv, options, pmap)
out["profile_argv_rules"] = {"refusals": refusals, "pi_pending": pi_pending}
# individual sub-rules, to attribute each reason
out["sub"] = {
  "required_flag_reasons": G.required_flag_reasons(prof, argv),
  "forbidden_value_reasons": G.forbidden_value_reasons(prof, argv),
  "forbidden_lever_reasons": G.forbidden_lever_reasons(prof, argv)[0],
  "nav_tau_reasons": G.nav_tau_reasons(prof, argv, options["nav_tau_record"], pmap)[0],
  "required_on_reasons": G.required_on_reasons(prof, argv)[0],
  "pi_pending_reasons": G.pi_pending_reasons(prof, argv),
  "open_item_reasons": G.open_item_reasons(prof),
}
nt = G.nav_tau_reasons(prof, argv, options["nav_tau_record"], pmap)
out["nav_tau_detail"] = nt[1]
out["required_checks"] = G.required_checks_of(prof, argv)
# --- manifest
gin = {f"gate:clock-reference[{i}]": r for i, r in enumerate(clock_ref)}
if options["nav_tau_record"]:
    gin["gate:nav-tau-record"] = options["nav_tau_record"]
t0 = time.time()
man = G.data_manifest(argv, prof, pmap, gin)
out["data_manifest_s"] = round(time.time() - t0, 1)
out["data_manifest"] = man
out["data_manifest_problems"] = G.data_manifest_problems(man)
# flag -> kind table
out["kinds"] = {k: (v.get("kind"), v.get("path")) for k, v in man.items()}
json.dump(out, open(f"{W}/raw/step2_rules_manifest.json", "w"), indent=1, default=str)
print("DONE_STEP2")
print("refusals:", len(refusals)); [print("  R:", r) for r in refusals]
print("pi_pending:", len(pi_pending)); [print("  P:", r) for r in pi_pending]
print("manifest problems:", len(out["data_manifest_problems"])); [print("  M:", r) for r in out["data_manifest_problems"]]
for k, (kind, p) in out["kinds"].items():
    print(f"  {k:36s} {kind:8s} {p}")
