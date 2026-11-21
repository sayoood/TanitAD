import json, numpy as np
z = np.load("F:/TanitAD_p6read_work/tactical/table_p6final.npz", allow_pickle=False)
P, S, V, pick, L, log = z["pdms"], z["sub"], z["valid"], z["pick"], z["logits"].astype(np.float64), z["log"]
N = len(pick); ix = np.arange(N)
sg = 1 / (1 + np.exp(-L))
rule = sg[..., 0] * sg[..., 1] * (5 * sg[..., 2] + 5 * sg[..., 3] + 2 * sg[..., 4]) / 12
rpick = rule.argmax(1)
out = {"n_tokens": int(N), "rule_pick_equals_stored_pick": int((rpick == pick).sum())}
best = np.nanmax(P, 1); pk = P[ix, pick]; reg = best - pk
out["mean_pick_x100"], out["mean_best_x100"], out["regret_x100"] = 100 * pk.mean(), 100 * best.mean(), 100 * reg.mean()
zero = pk == 0; prop = zero & (best == 0); sel = zero & (best > 0)
ttc0 = (~zero) & (S[ix, pick, 3] == 0)
rest = (~zero) & (~ttc0)
nc0 = zero & (S[ix, pick, 0] == 0); dac0 = zero & (S[ix, pick, 1] < 1) & (S[ix, pick, 0] > 0)
cls = {"zero_pick_selection (a non-zero proposal existed)": sel, "zero_pick_proposal (all 64 score 0)": prop, "nonzero_pick_TTC0": ttc0, "rest (EP, comfort)": rest}
ul, inv = np.unique(log, return_inverse=True)
rng = np.random.default_rng(20260927); W = rng.multinomial(len(ul), np.ones(len(ul)) / len(ul), size=10000).astype(np.float64)
cnt = np.bincount(inv, minlength=len(ul)).astype(np.float64)
def boot(x):
    s = np.bincount(inv, weights=x, minlength=len(ul)); r = (W @ s) / (W @ cnt)
    return [float(100 * np.percentile(r, 2.5)), float(100 * np.percentile(r, 97.5))]
out["components_x100"] = {k: {"n_tokens": int(m.sum()), "share_of_tokens": float(m.mean()), "regret_contribution": float(100 * (reg * m).mean()), "ci95": boot(reg * m)} for k, m in cls.items()}
out["zero_pick_by_gate"] = {"NC 0 at pick": int(nc0.sum()), "DAC below 1 at pick (NC above 0)": int(dac0.sum()), "other": int(zero.sum() - nc0.sum() - dac0.sum())}
out["sum_of_components_x100"] = float(sum(v["regret_contribution"] for v in out["components_x100"].values()))
order = np.argsort(-rule, 1, kind="stable")
out["topk_oracle_x100"] = {int(k): float(100 * np.nanmax(np.take_along_axis(P, order[:, :k], 1), 1).mean()) for k in (1, 2, 3, 4, 5, 8, 16, 32, 64)}
out["n_best_strictly_above_pick"] = int((best > pk).sum())
out["note"] = "MEASURED on the full navtest truth table (selection_quality table_p6final.npz, 12,146 tokens x 64 proposals, harness PDMS v1). Descriptive; token-unweighted means. CI: log-cluster bootstrap over 136 logs, 10,000 draws, seed 20260927 (another draw of logs only). Tier T1-class open loop."
json.dump(out, open("F:/tanitad_scratch/selq2read/p6_e3_regret_split.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
