"""C3 cross-surface comparison. ASCII-only prints."""
import io
import json

d = json.load(io.open(r"C:\Users\Admin\_roll\c3\refcv4b_c3_devbox.json", encoding="utf-8"))
arms = d["arms"]


def ade(a):
    x = arms[a]["four_families"]
    for k in ("ade_0_2s", "ade"):
        if k in arms[a]:
            return float(arms[a][k])
    return None


# find the headline ADE key
print("ARM KEYS SAMPLE:", [k for k in arms["os"] if "ade" in k.lower()][:8])
for a in ("os", "ha", "ha0", "ha0_ext", "os_navshuf", "os_navzero"):
    if a in arms:
        v = arms[a]
        cand = {k: v[k] for k in v if "ade" in k.lower() and isinstance(v[k], (int, float))}
        print(f"{a:12s} {cand}")

BANKED = {
    "A40 (landing dump)":  {"os": 0.2975, "ha": 0.2996, "ha0": 0.6723,
                            "ha0_ext": 0.2874, "os_navshuf": 0.3013,
                            "os_navzero": 0.3928},
    "Thor (navpred dump)": {"os": 0.2965, "os_navshuf": 0.3006,
                            "os_navzero": 0.3926},
}
print()
print("=== C3-style reproduction control, tolerance 0.001 ===")
for surf, ref in BANKED.items():
    print(f"-- vs {surf}")
    for a, want in ref.items():
        v = arms.get(a, {})
        got = None
        for k in v:
            if "ade" in k.lower() and isinstance(v[k], (int, float)):
                got = float(v[k]); break
        if got is None:
            print(f"   {a:12s} INCONCLUSIVE (no ade key)"); continue
        diff = abs(got - want)
        print(f"   {a:12s} measured {got:.6f}  banked {want:.4f}  "
              f"abs_diff {diff:.6f}  {'PASS' if diff <= 0.001 else 'FAIL'}")
