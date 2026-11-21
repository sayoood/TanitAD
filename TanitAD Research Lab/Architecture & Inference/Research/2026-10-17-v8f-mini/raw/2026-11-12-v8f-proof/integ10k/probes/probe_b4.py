# probe_b4.py -- controls-only probe of control b4 of card ADD71 part B (read stopped; no part value is printed or kept).
# Prints, per cell, how far POSE + STATE and TOTAL are from the banked total, and the counts of finite windows per arm.
import sys

sys.path.insert(0, "/workspace/v8proof/podread/root/integ10k")
import read_agerule_decomp as M  # noqa: E402

np = M.np
_rp, _b4 = M.read_parts, M.ctl_b4


def rp(E, V, ref, n_boot, extra):
    for s in M.SEEDS:
        for k in sorted(V[("F", s)]):
            if k.startswith("_"):
                continue
            f, d, a = (np.asarray(V[(t, s)][k], float) for t in ("F", "D", "A"))
            nf, nd, na = np.isfinite(f), np.isfinite(d), np.isfinite(a)
            if not (nf.all() and nd.all() and na.all()):
                print("NAN seed%d %-22s finite F %d D %d A %d | F&A %d | F&D&A %d | F!=D mask %d | n %d" % (
                    s, k, nf.sum(), nd.sum(), na.sum(), (nf & na).sum(), (nf & nd & na).sum(), (nf != nd).sum(), len(f)))
    return _rp(E, V, ref, n_boot, extra)


def b4(parts, banked):
    for c, p in parts.items():
        b = (banked or {}).get(c) or {}
        if "D" in b:
            print("B4 %-34s dev_sum %.3e dev_tot %.3e" % (c, abs(p["POSE"]["D"] + p["STATE"]["D"] - b["D"]), abs(p["TOTAL"]["D"] - b["D"])))
    return _b4(parts, banked)


M.read_parts, M.ctl_b4 = rp, b4
if __name__ == "__main__":
    sys.exit(M.main(sys.argv[1:]))
