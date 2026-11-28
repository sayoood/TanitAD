"""CAPLADDER sets (IMPL_CHOICES sections 1 and 2): rows of set A (finite lead_gap), D (A + a 1:1 balanced sample of lead_present = 0), folds, beside labels. CPU, labels only; no feature, no probe.
Writes sets_capladder.npz + sets_capladder.report.json."""
import hashlib, json, os, sys
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import capladder_lib as CL

LAB = os.path.join(HERE, "..", "imag1", "labels_imag1_eval141.npz")
LAB_SHA = "bd0a265bde15b2ac9f5de680dd6d5c1e9249ae0d00a42ffebdbf0d00199c93ef"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    assert sha(LAB) == LAB_SHA, "labels file changed"
    L = np.load(LAB)
    beid, bg = L["beid"].astype("U36"), L["bg"].astype(int)
    lg, lp = L["lead_gap"], L["lead_present"]
    A = np.flatnonzero(np.isfinite(lg))
    assert (lp[A] == 1).all() and ((lp == 1) & np.isfinite(lp)).sum() == len(A), "lead_present = 1 must be exactly set A"
    neg_pool = np.flatnonzero(np.isfinite(lp) & (lp == 0))
    negs = np.sort(np.random.default_rng(0).choice(neg_pool, size=len(A), replace=False))
    rowsD = np.concatenate([A, negs])
    uniq = np.unique(beid)
    epD = beid[rowsD]
    nA = len(A)
    foldD, innerD = CL.outer_fold(epD), CL.inner_fold(epD)
    posD = np.searchsorted(uniq, epD)
    gap = lg[A]
    clos, ttc = L["lead_closing"][A], L["ttc"][A]
    B = np.flatnonzero(np.isfinite(clos) & np.isfinite(ttc))
    C = np.flatnonzero(gap <= 30.0)
    out = dict(beid=beid, bg=bg, rowsA=A, rowsD=rowsD, nA=nA, nD=len(rowsD), uniq=uniq, epD=epD, gD=bg[rowsD], foldD=foldD, innerD=innerD, posD=posD,
               gap=gap, loggap=np.log(np.maximum(gap, 0.1)), clos=clos, ttc=ttc, idxB=B, idxC=C, present=lp[rowsD])
    np.savez("sets_capladder.npz", **out)
    rep = {"labels_sha256": LAB_SHA, "nA": int(nA), "nD": int(len(rowsD)), "n_bank": int(len(beid)), "n_episodes": int(len(uniq)), "A_episodes": int(len(np.unique(epD[:nA]))),
           "nB": int(len(B)), "nC": int(len(C)), "neg_pool": int(len(neg_pool)),
           "A_frames_by_fold": np.bincount(foldD[:nA], minlength=5).tolist(), "A_episodes_by_fold": [int(len(np.unique(epD[:nA][foldD[:nA] == k]))) for k in range(5)],
           "D_frames_by_fold": np.bincount(foldD, minlength=5).tolist()}
    json.dump(rep, open("sets_capladder.report.json", "w"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
