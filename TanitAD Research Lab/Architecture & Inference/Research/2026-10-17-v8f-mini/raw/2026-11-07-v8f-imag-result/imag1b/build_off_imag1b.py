"""IMAG1B: bank indices of the same-clip offset latents (decorrelation curve, beside). No model, no probe, no statistic: integers only.
For every window i, level l, horizon k and offset d in {1, 4, 8} frames (+0.2 / +0.8 / +1.6 s after the future frame f_k = k x stride), the bank index of (episode of window i, cache step G[i, f_k] + d),
or -1 when that frame is not in the bank. Uses the same frame-bank call as IMAG1's build_ctx_imag1.py. Asserts (a) the window episodes equal the cached ctx's, (b) the d = 1 indices reproduce IMAG1's cached
OFF latents (the +0.2 s control latent) bit for bit, which proves the mapping is the one IMAG1 used. Usage: python build_off_imag1b.py eval141|train  (WMACT_CODE_ROOT set).
Writes F:/TanitAD-scratch/imag1b_off_<ds>.pkl (skipped when it exists)."""
import json, os, pickle, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "wmact")); sys.path.insert(0, os.path.join(HERE, "..", "latsem2b")); sys.path.insert(0, os.path.join(HERE, "..", "latsem2"))
os.chdir(HERE)
import numpy as np
import imag1b_model  # noqa: F401  (puts the tip's code on the path through wmact_lib)
import wmact_lib as W  # noqa: F401
import labels_latsem2b as LBb  # noqa: E402
import imag1b_lib as IL

LATENTS = {"eval141": os.path.join(HERE, "..", "wmact", "latents.pt"), "train": os.path.join(HERE, "..", "wmact", "latents_train.pt")}
SCR = "F:/TanitAD-scratch"


def main(ds):
    out = f"{SCR}/imag1b_off_{ds}.pkl"
    if os.path.exists(out):
        print("exists, skipped:", out)
        return
    B = LBb.build_bank_train(LATENTS[ds])
    eid, ok, G, key2b = B["eid"], B["ok"], B["G"], B["key2b"]
    sel = B["wkeep"][ok[B["wkeep"]].all(1)]
    with open(f"{SCR}/imag1_ctx_{ds}.pkl", "rb") as fh:
        c = pickle.load(fh)
    assert (eid[sel].astype("U36") == c["qe"]).all(), "window episodes differ from the cached ctx"
    idx, miss = {}, {}
    for l in IL.LEVELS:
        for k in IL.KS:
            f = k * IL.STRIDE[l]
            for d in IL.OFFD:
                a = np.array([key2b.get((eid[i], int(G[i, f]) + d), -1) for i in sel], dtype=np.int64)
                idx[(l, k, d)] = a
                miss[f"L{l}k{k}d{d}"] = int((a < 0).sum())
    for tag in c["tags"]:
        for l in IL.LEVELS:
            for k in IL.KS:
                a = idx[(l, k, 1)]
                assert (a >= 0).all(), "a +0.2 s frame is missing, IMAG1's OFF could not have been built"
                assert np.array_equal(c["BZ"][tag][l][a], c["OFF"][tag][(l, k)]), f"d=1 index does not reproduce IMAG1's OFF latents ({tag} L{l} k{k})"
    pickle.dump({"idx": idx, "n_windows": len(sel), "missing": miss}, open(out, "wb"), protocol=4)
    print(json.dumps({"dataset": ds, "n_windows": len(sel), "bytes": os.path.getsize(out), "missing_by_cell": {k: v for k, v in miss.items() if v}}), flush=True)


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        main(ds)
