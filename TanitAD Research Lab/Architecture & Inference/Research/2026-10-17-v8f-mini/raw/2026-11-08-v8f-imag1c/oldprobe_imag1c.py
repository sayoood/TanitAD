"""IMAG1C D8: IMAG1B's probe (ridge lambda 10, standardised on the target's valid fit rows, 2 folds sha256(episode)[:8] % 2) on the old window subset and on all frames, `lead_gap`,
real labels only (no reference, no bar). The old-window cell is the reproduction check (iii) against ../imag1b/RESULT_IMAG1B.json. Writes OLDPROBE_IMAG1C.json."""
import json, os, pickle, sys, types
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "imag1b"))
import imag1c_lib as ML
import imag1b_probes as PB
IL = ML.IL


def main():
    d = pickle.load(open("F:/TanitAD-scratch/imag1c_ctx.pkl", "rb"))
    c = ML.finish_ctx(types.SimpleNamespace(**d))
    bf2, qf2 = np.array([ML._h(str(e)) % 2 for e in c.beid]), np.array([ML._h(str(e)) % 2 for e in c.qe])
    old_json = json.load(open(os.path.join(HERE, "..", "imag1b", "RESULT_IMAG1B.json")))["datasets"]["eval141"]["ia2"]
    out, maxdiff = {}, 0.0
    val = np.isfinite(c.Y["lead_gap"])
    for tag in c.tags:
        for l in ML.LEVELS:
            X = np.asarray(c.BZ[tag][l], np.float64)
            PR = [PB.fit_lin("cont", X[val & (bf2 != k)], c.Y["lead_gap"][val & (bf2 != k)]) for k in (0, 1)]
            def dec(Z):
                o = np.full(len(Z), np.nan)
                for k in (0, 1):
                    m = qf2 == k
                    o[m] = PB.predict_lin(PR[k], np.asarray(Z, np.float64)[m])
                return o
            decs = {"copy": dec(c.Z0[tag][l])}
            for k in ML.KS:
                decs[("pred", k)], decs[("oracle", k)] = dec(c.PRED[tag][(l, k)]), dec(c.ORA[tag][(l, k)])
                y = c.Yw["lead_gap"][:, k * ML.STRIDE[l]]
                for sub in ML.SUBS:
                    valid = np.isfinite(y) & ML.sub_mask(c, sub)
                    ym = np.array([y[valid & (qf2 == f)].mean() if (valid & (qf2 == f)).any() else 0.0 for f in (0, 1)])
                    for qt in ("copy", "pred", "oracle"):
                        p = decs["copy"] if qt == "copy" else decs[(qt, k)]
                        m = IL.mom_cont(y, p, valid, c.qpos, c.n_ep, ym[qf2])
                        r = m.row(c.W)
                        out[f"{sub}|{tag}|{l}|{k}|{qt}"] = {"nmse": 1 - r["point"], "lo": 1 - r["hi"], "hi": 1 - r["lo"], "n": r["n"], "nep": r["nep"]}
                        if sub == "old":
                            ref = old_json[f"ridge|{tag}|{l}|{k}|lead_gap"]["nmse"][qt]["point"]
                            maxdiff = max(maxdiff, abs(ref - (1 - r["point"])))
    res = {"rows": out, "repro_max_abs_diff_nmse_vs_RESULT_IMAG1B": maxdiff, "repro_pass": bool(maxdiff < 1e-4)}
    json.dump(res, open(os.path.join(HERE, "OLDPROBE_IMAG1C.json"), "w"), indent=1)
    print("reproduction max abs diff", maxdiff, "pass", res["repro_pass"])


if __name__ == "__main__":
    main()
