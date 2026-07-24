"""Gate secondaries for flagship-v3enc-10k, computed by the ORIGINAL
/root/diag_v2mech.py verbatim (no re-implementation), so the numbers are
directly comparable to the v2@6k / v1@30k references in
`2026-07-19-flagshipv2-6k-diagnostic.md:196-199`.

  encoder_speed_probe_r2   = diag `probe_speed_r2`
      ridge z_t -> v0 on the same 40 val eps, per-EPISODE held-out split
      (8 held-out eps), lambda chosen from (1e-2,1e-1,1,10) by best held-out
      R^2. v2@6k = 0.300, v1@30k = 0.861. GATE >= 0.55.
  highspeed_long_overshoot_m = diag `op_long2s_high`
      mean SIGNED along-track error at the 2 s waypoint, GT-track frame,
      restricted to the HIGH-speed tercile (v0 >= 2/3 quantile of the same
      881 windows). Positive = drives too far. v2@6k = +23.7. GATE <= 8.0.

The wrapper only APPENDS a registry entry in memory; nothing on disk is edited.
"""
import importlib.util, json, os, sys

os.environ["TANITEVAL_STACK_OVERRIDE"] = "/root/models/assess-20260719/stack-v2"
spec = importlib.util.spec_from_file_location("diag", "/root/diag_v2mech.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)

sys.path.insert(0, "/root/taniteval")
from taniteval.registry import MODELS as MAIN            # noqa: E402

KEY = "flagship-v3enc-10k"
entry = [m for m in MAIN if m["key"] == KEY][0]
diag.v2registry.MODELS.append(entry)                     # in-memory only
print(f"[wrap] injected {KEY} -> {entry['ckpt']}", flush=True)

res = diag.run_model(KEY)
print(json.dumps(res, indent=1), flush=True)

# merge into the existing summary next to the v2/v1 references
SUM = "/root/taniteval/results/diagv2_summary.json"
allr = json.load(open(SUM))
allr[KEY] = res
json.dump(allr, open(SUM, "w"), indent=1)

print("\n===== GATE SECONDARIES =====")
print(f"encoder_speed_probe_r2      = {res['probe_speed_r2']}  "
      f"(lambda={res['probe_lam']})   GATE >= 0.55")
print(f"highspeed_long_overshoot_m  = {res['op_long2s_high']}   GATE <= 8.0")
print("\n===== REFERENCES (same tool, same val set) =====")
for k in ("flagship-v2-6k", "flagship-30k", "flagship-speed"):
    r = allr.get(k, {})
    if "probe_speed_r2" in r:
        print(f"  {k:16s} probe_r2={r['probe_speed_r2']:<8} "
              f"long2s_high={r['op_long2s_high']:<8} "
              f"ade2s_op={r.get('ade2s_op')}")
