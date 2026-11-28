# pod_leadcond_q3.sh : runs ON THE POD (bash -s). Control q3 of card PREREG_V8F_PROOF_ADD35 and its floor: the live record of the dry gate of LC_s0 against
# LFT_s0, and LFT_s0 against a second dry gate of the same argv (is the live record of a selfcheck repeatable at all). Selfchecks only, trains nothing.
set -u
W=/workspace/v8proof; cd $W/chain2 || exit 1
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "${n:-0}" -lt 3 ] || { echo "Q3-NOT-RUN: $n GPU jobs running"; exit 1; }
echo "Q3-RAN $(date -u +%Y-%m-%dT%H:%M:%SZ) with $n GPU jobs before"
cp $W/runs/motion_dry/LFT_s0/out/selfcheck.json $W/chains/leadcond_q3_LFT_s0_runA.json
cp $W/runs/motion_dry/LC_s0/out/selfcheck.json $W/chains/leadcond_q3_LC_s0_runA.json
bash dry_gate.sh arms_leadcond_a.txt LFT_s0 PREREG_V8F_PROOF_ADD35 > $W/chains/leadcond_dry_LFT_s0_runB.out 2>&1; echo "second dry gate LFT_s0 rc $?"; tail -n 1 $W/chains/leadcond_dry_LFT_s0_runB.out | cut -c1-200
cp $W/runs/motion_dry/LFT_s0/out/selfcheck.json $W/chains/leadcond_q3_LFT_s0_runB.json
$W/../venv/bin/python - <<'PYEOF'
import json
C = "/workspace/v8proof/chains/leadcond_q3_%s.json"
A, B, L = (json.load(open(C % n)) for n in ("LFT_s0_runA", "LFT_s0_runB", "LC_s0_runA"))
def flat(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, pre + k + "."))
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out[pre + k] = float(v)
    return out
fa, fb, fl = flat(A.get("live") or {}), flat(B.get("live") or {}), flat(L.get("live") or {})
print("live keys (non-term):", [k for k in fa if not k.startswith("terms.")][:20])
def cmp(x, y, name):
    ks = sorted(set(x) & set(y))
    rel = sorted(((abs(x[k] - y[k]) / max(abs(x[k]), 1e-12), k) for k in ks), reverse=True)
    n0 = sum(1 for r, _k in rel if r == 0.0)
    print("%s: fields %d, exactly equal %d, largest relative differences: %s" % (name, len(ks), n0, ", ".join("%s %.2e" % (k, r) for r, k in rel[:5])))
    for k in ks:
        if k.endswith("total") or k.endswith(".ade") or k.endswith("lead_sup"):
            print("    %-22s %.9g vs %.9g (rel %.2e)" % (k, x[k], y[k], abs(x[k] - y[k]) / max(abs(x[k]), 1e-12)))
cmp(fa, fb, "LFT_s0 run A vs LFT_s0 run B (same argv)")
cmp(fa, fl, "LFT_s0 run A vs LC_s0")
cmp(fb, fl, "LFT_s0 run B vs LC_s0")
print("live description fields:", {k: v for k, v in (A.get("live") or {}).items() if not isinstance(v, dict)})
PYEOF
