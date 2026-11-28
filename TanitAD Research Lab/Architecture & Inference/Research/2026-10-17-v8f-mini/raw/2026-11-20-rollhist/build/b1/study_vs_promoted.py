"""One-off control (N1): the promoted functions vs the study file's OWN functions (exec'd from the study source text), bit for bit, on seeded plans. Prints the golden digest the repo test pins."""
import hashlib, re, sys, numpy as np
sys.path.insert(0, sys.argv[2])  # scripts dir of the tree
import v8f_t1_eval as T1
import v8f_t1_readout as RO
src = open(sys.argv[1], encoding="utf-8").read()
ns = {"np": np, "DT_S": 0.2}
for name in ("backward_frames", "causal_frames", "causal_g_frames"):
    i = src.index(f"def {name}("); j = src.index("\n\n\ndef ", i)
    exec(src[i:j], ns)
def cases():
    rng = np.random.default_rng(20261120)
    for c in range(40):
        m = int(rng.choice([5, 10]))
        th = np.cumsum(rng.normal(0, 0.05, m + 3)); sp = 0.2 * np.abs(8 + rng.normal(0, 2.0, m + 3)) * (0.0 if c % 7 == 0 else 1.0)
        pts = np.cumsum(np.stack([sp * np.cos(th), sp * np.sin(th)], -1), 0)
        yield pts[0], pts[1:], m, float(rng.uniform(-3, 3))
h = hashlib.sha256(); n = 0; ok = True
for kp, kn, m, p0 in cases():
    ref_b = ns["backward_frames"](T1.hermite_frames, kp, kn, m, p0); new_b = RO.backward_frames(T1.hermite_frames, kp, kn, m, p0)
    outs = [(ref_b, new_b)]
    for acc in ("step", "zero", "window"):
        outs.append((ns["causal_frames"](T1.hermite_frames, kp, kn, m, p0, acc), RO.causal_frames(T1.hermite_frames, kp, kn, m, p0, acc)))
    for g in (-0.25, 0.0, 0.5, 1.0, 1.7):
        for px in (False, True):
            for fg in (False, True):
                outs.append((ns["causal_g_frames"](T1.hermite_frames, kp, kn, m, p0, g, px, fg), RO.causal_g_frames(T1.hermite_frames, kp, kn, m, p0, g, px, fg)))
    for a, b in outs:
        for u, w in zip(a, b):
            ok &= np.array_equal(np.asarray(u), np.asarray(w)); h.update(np.ascontiguousarray(w, np.float64).tobytes()); n += 1
print("arrays compared", n, "bit-identical", ok, "golden_sha256", h.hexdigest())
