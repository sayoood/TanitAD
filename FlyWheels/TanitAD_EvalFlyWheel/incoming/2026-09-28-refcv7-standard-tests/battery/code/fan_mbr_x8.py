"""X8 lever test for D-REFCV7-FANCEIL: can an INFERENCE-ONLY rule recover part of the reachable headroom?

Rule = minimum-Bayes-risk pick inside the model own top-K: i* = argmin_{i in topK} sum_j p_j d(i,j),
p = softmax(score/T) over the top-K, d = mean-over-4-waypoints L2 between two fan members.
No GT, no ego state, no extra network: admissible at inference (vision-only rule untouched).
DISCIPLINE: 36-cell grid (3 scores x K{4,8,16,32} x T{0.5,1,2}). The cell is CHOSEN ON EPISODE-HALF A
(even rank of sorted episode ids) and REPORTED ON HALF B only. Post-hoc exploratory: not a registered claim.
Tier T1 open loop for the pick (no GT used by the rule); GT enters only the scoring.
CI answers ONLY: would another draw of EPISODES say this.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fan_control_x8 as X  # noqa: E402

SCORES = ("anchor_logits", "refined_logits", "sel_score")
KS = (4, 8, 16, 32)
TS = (0.5, 1.0, 2.0)


def mbr_pick(fan, score, k, t):
    B = fan.shape[0]
    idx = np.argsort(-score, axis=1, kind="stable")[:, :k]
    out = np.empty(B, np.int64)
    for s in range(0, B, 400):
        e = min(B, s + 400)
        ii = idx[s:e]
        c = np.take_along_axis(fan[s:e].astype(np.float32), ii[:, :, None, None], axis=1)
        d = np.linalg.norm(c[:, :, None] - c[:, None, :], axis=-1).mean(-1)
        lg = np.take_along_axis(score[s:e].astype(np.float64), ii, axis=1) / t
        lg = lg - lg.max(1, keepdims=True)
        p = np.exp(lg)
        p /= p.sum(1, keepdims=True)
        risk = (d * p[:, None, :]).sum(-1)
        out[s:e] = ii[np.arange(e - s), risk.argmin(1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan-npz", required=True)
    ap.add_argument("--gt-dump", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--fixed-cell", default=None,
                    help="score,K,T : use this cell INSTEAD of choosing on half A (replicate of a cell fixed on another roll)")
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.fan_npz, allow_pickle=True)
    ci = np.asarray(z["clip_index"]).astype(np.int64)
    ws = np.asarray(z["ws"]).astype(np.int64)
    fan = z["anchor_traj"]
    sel = np.asarray(z["sel_idx"]).astype(np.int64).ravel()
    B, N = fan.shape[0], fan.shape[1]
    tab, _ = X.load_gt_dump(a.gt_dump)
    keys = [(int(c), int(w)) for c, w in zip(ci, ws)]
    assert len(set(keys)) == B and all(k in tab for k in keys)
    gt = np.stack([tab[k][0] for k in keys])
    eid = np.array([tab[k][1] for k in keys])
    de = X._de_all(fan, gt)
    selv = de[np.arange(B), sel]
    ue = sorted(set(eid.tolist()))
    half_a = set(ue[0::2])
    ma = np.array([e in half_a for e in eid.tolist()])
    mb = ~ma
    rows = []
    for nm in SCORES:
        sc = np.asarray(z[nm])
        for k in KS:
            for t in TS:
                p = mbr_pick(fan, sc, k, t)
                v = de[np.arange(B), p]
                da = float((selv[ma] - v[ma]).mean())
                rows.append({"score": nm, "K": k, "T": t, "gain_on_A": da, "_v": v})
    best = max(rows, key=lambda r: r["gain_on_A"])
    cell_source = "chosen_on_A"
    if a.fixed_cell:
        fs, fk, ft = a.fixed_cell.split(",")
        hit = [r for r in rows if r["score"] == fs and r["K"] == int(fk) and r["T"] == float(ft)]
        assert len(hit) == 1, "fixed cell not in grid: " + a.fixed_cell
        best = hit[0]
        cell_source = "FIXED_BY_CLI " + a.fixed_cell
    out = {"n_windows": B, "n_episodes": len(ue), "n_half_A_windows": int(ma.sum()), "n_half_B_windows": int(mb.sum()),
           "grid_gain_on_A_only": [{k: r[k] for k in ("score", "K", "T", "gain_on_A")} for r in rows],
           "chosen_on_A": {k: best[k] for k in ("score", "K", "T", "gain_on_A")}, "cell_source": cell_source}
    v = best["_v"]
    for tag, m in (("B_heldout", mb), ("A_selection_half", ma), ("all_for_reference", np.ones(B, bool))):
        ee = eid[m].tolist()
        out[tag] = {"selected": X._boot(selv[m], ee, "selected_" + tag),
                    "mbr": X._boot(v[m], ee, "mbr_" + tag),
                    "selected_minus_mbr": X._paired(selv[m], v[m], ee, "selected - mbr " + tag)}
    am = np.asarray(z[best["score"]]).argmax(1)
    va = de[np.arange(B), am]
    for tag, m in (("B_heldout", mb), ("all_for_reference", np.ones(B, bool))):
        ee = eid[m].tolist()
        out[tag]["argmax_same_score"] = X._boot(va[m], ee, "argmax_" + best["score"] + "_" + tag)
        out[tag]["argmax_minus_mbr"] = X._paired(va[m], v[m], ee, "argmax - mbr " + tag)
    out["stamp"] = {"tier": "T1 open-loop pick (rule uses no GT); GT only in scoring; post-hoc exploratory, not a registered claim",
                    "estimator": "FULL-SET mean; episode_cluster_bootstrap / paired n_boot=2000 seed=0",
                    "ci_answers": "ONLY: another draw of EPISODES. NOT training seed, NOT inference seed",
                    "fan_npz_sha256": X._sha256(a.fan_npz), "wall_s": round(time.time() - t0, 1)}
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("WROTE", a.out_json, out["stamp"]["wall_s"])


if __name__ == "__main__":
    main()
