"""SH5 C-REACH diagnosis, probe 2 (CPU, read-only): the BEST AUC an attention reader of the plant can reach at a given positional RESOLUTION and TEMPERATURE.
For each query point (set, proposal, key time) of a fold-0 token, the reader returns the softmax(beta * kernel)-weighted mean of the token's planted signed clearances, where
kernel(dx, dy) = mean over octaves j of [cos(w_j dx) + cos(w_j dy)] / 2  (the dot product of two Fourier codes = what a dot-product attention over the plant's code can implement),
w_j = 2^j pi / 64 (the branch's octaves; K=6 -> finest wavelength 4 m = the query features of sh5_branch.point_features; K=8 -> 1 m = the plant's code).
Score = min over key times of the read clearance; within-set AUC (FL.pair_stats, all strata / F).  No distractor slots: an UPPER bound on what the branch can read at that resolution."""
import json, sys, time, numpy as np
sys.path.insert(0, "C:/Users/Admin/par18/sh5/code")
import sh5_lib as L, sh5_train_eval as TE
DD = "D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19"
allt = np.load(f"{DD}/data/sh5_inputs.npz", allow_pickle=True)["token"]
keep, _ = TE.ctx_keep(allt, f"{DD}/ctx_local/ctx_manifest.json")
D = TE.load_data(f"{DD}/data/sh5_inputs.npz", f"{DD}/data/sh5_labels.npz", keep)
FL, _ = L.refs()
lab = D["lab"][..., L.DAC_COL]; code = D["code"]
sc_all = np.clip(D["sc_key"].astype(np.float64), -10, 10)
f0_tokens = [t for t in range(D["T"]) if D["tok_fold"][t] == 0]
f0_sets = np.concatenate([D["tok_sets"][t] for t in f0_tokens])
def auc(z, sets):
    st = FL.pair_stats(z, lab[sets] == 0.0, lab[sets] == 1.0, code[sets], names=("ALLQ", "Q4"))
    return {k: round(float(n.sum() / d.sum()), 4) for k, (n, d) in st.items()}
def read(K, beta, sigma_noise=0.0):
    out = np.zeros((len(f0_sets), 64, 9))
    pos = 0
    for t in f0_tokens:
        s = D["tok_sets"][t]
        xy = D["pts"][s][..., :2].reshape(-1, 2).astype(np.float64)
        c = sc_all[s].reshape(-1)
        ws = (2.0 ** np.arange(K)) * np.pi / 64.0
        dx = xy[:, None, 0] - xy[None, :, 0]; dy = xy[:, None, 1] - xy[None, :, 1]
        ker = np.zeros_like(dx)
        for w in ws:
            ker += np.cos(w * dx) + np.cos(w * dy)
        ker /= (2 * K)
        lg = beta * ker; lg -= lg.max(1, keepdims=True)
        a = np.exp(lg); a /= a.sum(1, keepdims=True)
        est = a @ c
        out[pos:pos + len(s)] = est.reshape(len(s), 64, 9)
        pos += len(s)
    return out
R = {"n_f0_tokens": len(f0_tokens), "n_f0_sets": int(len(f0_sets)), "oracle_exact": auc(sc_all[f0_sets].min(-1), f0_sets), "rows": []}
for K in (6, 8, 10, 12):
    for beta in (4, 16, 64, 256, 1024, 4096):
        t0 = time.time()
        est = read(K, beta)
        a = auc(est.min(-1), f0_sets)
        # regression R^2 of |est| vs the 9 |clearance| targets (the reg head's job)
        tg = D["tgt"][f0_sets].astype(np.float64); r = np.minimum(np.abs(est), 10)
        r2 = 1 - ((r - tg) ** 2).sum() / ((tg - tg.mean((0, 1))) ** 2).sum()
        R["rows"].append({"K": K, "finest_wavelength_m": 128 / 2 ** (K - 1), "beta": beta, "auc": a, "r2_abs": round(float(r2), 4)})
        print(R["rows"][-1], f"{time.time() - t0:.0f}s", flush=True)
json.dump(R, open("C:/Users/Admin/par18/sh5diag/raw/probe2_ideal_reader.json", "w"), indent=1)
print("oracle", R["oracle_exact"])
