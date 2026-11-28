"""DIAGNOSTIC WRAPPER (NOT the gate): run the gate's own `check` child with the refcv8 profile's `model_input_flags`
EXTENDED IN MEMORY by PREGATE_EXTRA_MODEL_INPUTS (comma list). Purpose: see whether the first model-job refusal
(a sentinel path for a flag the build really reads) MASKS further blockers. Optional PREGATE_TRACE_DIGESTS=<json>
records WHICH forward outputs differ between the gate's repeated calls. Nothing in the tree is edited."""
import os, sys, json, atexit
T = "/home/nvidia/refcv6_run/d014ff2fd6"
sys.path[:0] = [f"{T}/stack/scripts", f"{T}/stack", f"{T}/taniteval"]
import launch_gate as G
extra = tuple(x for x in os.environ.get("PREGATE_EXTRA_MODEL_INPUTS", "").split(",") if x)
p = G.PROFILES["refcv8"]
before = tuple(p["model_input_flags"])
p["model_input_flags"] = before + extra
print("[diag] model_input_flags", before, "+", extra, flush=True)
if os.environ.get("PREGATE_TRACE_DIGESTS"):
    _calls = []
    _orig = G.output_digests
    _depth = [0]
    def _wrap(o, *a, **k):
        _depth[0] += 1
        try:
            r = _orig(o, *a, **k)
        finally:
            _depth[0] -= 1
        if _depth[0] == 0:
            _calls.append(dict(r))
        return r
    def _unused(o, *a, **k):
        r = _orig(o, *a, **k)
        _calls.append(dict(r))
        return r
    G.output_digests = _wrap
    def _dump():
        c = _calls
        res = {"n_calls": len(c)}
        if len(c) >= 2:
            res["call1_vs_call2_trainer_repeat_differ"] = sorted(k for k in c[0] if c[1].get(k) != c[0][k])
        if len(c) >= 3:
            res["trainer_vs_loader_differ"] = sorted(k for k in set(c[0]) | set(c[2]) if c[0].get(k) != c[2].get(k))
        json.dump(res, open(os.environ["PREGATE_TRACE_DIGESTS"], "w"), indent=1)
    atexit.register(_dump)
raise SystemExit(G.main(sys.argv[1:]))
