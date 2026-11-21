"""The pristine-vs-modified pin digest differs ONLY where round 3 intends it to: the ONE added `--r9-train-only`
forbidden-lever entry of the five refcv8-family profiles, and the NEW inert `r9_schedule_reasons` (absent at aeb5446,
([], {'on': False}) on profiles without the rule). Every other key must be identical; the expectation is a literal."""
import json, sys
from pathlib import Path
D = Path("F:/refcv9_code3/evidence/digests")
r = json.loads((D / "CMP_pin_pristine_vs_modified.json").read_text(encoding="utf-8"))
a = json.loads((D / "pin_pristine.json").read_text(encoding="utf-8"))["arms"]
b = json.loads((D / "pin_tree.json").read_text(encoding="utf-8"))["arms"]
ENTRY = "[[['--r9-train-only'], 'the refcv9 warm-start freeze (PI 2026-10-27) -- launched under profile refcv9-warm only']]"
WANT = {("profiles", f"{p}.forbidden_levers_r9_entries"): ("[]", ENTRY)
        for p in ("refcv8", "refcv8-nt", "refcv8-mapft", "v8f", "v8f_steer")}
WANT.update({("gate_argv", f"{p}.r9_schedule"): ("absent", "([], {'on': False})") for p in ("refcv8", "refcv8-mapft")})
got = {(arm, k): (a[arm]["out"][k], b[arm]["out"][k]) for arm, v in r.items() if isinstance(v, dict)
       for k in v["out_differ"]}
extra = {arm: v["out_extra_in_b"] for arm, v in r.items() if isinstance(v, dict) and v["out_extra_in_b"]}
n_same = sum(v["n_out"] for v in r.values() if isinstance(v, dict)) - len(got)
ok = got == WANT and not extra
out = {"verdict": "ONLY_INTENDED_DIFFERENCES" if ok else "UNEXPECTED_DIFFERENCE", "n_keys_identical": n_same,
       "n_keys_differing": len(got), "differing": {f"{k[0]}:{k[1]}": list(v) for k, v in sorted(got.items())},
       "unexpected": {f"{k[0]}:{k[1]}": list(v) for k, v in got.items() if WANT.get(k) != v}, "extra": extra}
(D / "CLASSIFY_pin_pristine_vs_modified.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(out["verdict"], out["n_keys_identical"], out["n_keys_differing"])
