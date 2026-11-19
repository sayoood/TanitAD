#!/bin/bash
# pod_selfcheck_eq.sh -- runs ON THE POD via ssh stdin, reads only. Control of note ADD37_N4 section 4: the selfcheck of RL_s2 and RL_s3 equals that of RL_s0 in
# every lever. Prints the differing lever keys per base (expected: none besides the seed). Starts nothing, deletes nothing.
/workspace/venv/bin/python - <<'PYEOF'
import glob, json
W = "/workspace/v8proof/runs/"
def sc(chain, arm):
    for p in (W + "%s/%s/selfcheck.json" % (chain, arm), W + "%s/%s/out/selfcheck.json" % (chain, arm)):
        try:
            return json.load(open(p)), p
        except OSError:
            pass
    g = glob.glob(W + "%s/%s/**/selfcheck.json" % (chain, arm), recursive=True)
    return (json.load(open(g[0])), g[0]) if g else (None, None)
ref, pr = sc("imag_a", "RL_s0")
print("REF", pr, "levers", len((ref or {}).get("levers", {})))
for chain, arm in (("imag_b2", "RL_s2"), ("imag_b3", "RL_s3"), ("imag_a", "RL_s1")):
    s, p = sc(chain, arm)
    if s is None or ref is None:
        print("EQ", arm, "NO-FILE"); continue
    a, b = ref.get("levers", {}), s.get("levers", {})
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    print("EQ", arm, "dvb", s.get("declared_vs_built"), "live", (s.get("live") or {}).get("pass"), "steps_total", s.get("steps_total"), "n_levers", len(b), "differing_keys", diff,
          [(k, a.get(k), b.get(k)) for k in diff][:6])
PYEOF
