"""X8: inference-seed replicate of the refcv7 50,400 fan (seed 0 vs seed 1, SAME checkpoint, SAME windows).

Question answered: would another INFERENCE run (stochastic refinement) say this?  (not episodes, not training).
Everything here is a per-window paired comparison s0 vs s1 with the episode-cluster bootstrap, so the CI
reads the episode draw OF THE REPLICATE DIFFERENCE; the magnitude of the difference is the inference noise.
Usage: python fan_seedcmp_x8.py --fan-s0 F0 --fan-s1 F1 --gt-dump D --out-json O
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fan_control_x8 as X  # noqa: E402


def load(fan_npz, tab):
    z = np.load(fan_npz, allow_pickle=True)
    ci, ws = np.asarray(z["clip_index"]).astype(np.int64), np.asarray(z["ws"]).astype(np.int64)
    keys = [(int(c), int(w)) for c, w in zip(ci, ws)]
    assert len(set(keys)) == len(keys) and all(k in tab for k in keys)
    return z, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan-s0", required=True)
    ap.add_argument("--fan-s1", required=True)
    ap.add_argument("--gt-dump", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    t0 = time.time()
    tab, _ = X.load_gt_dump(a.gt_dump)
    z0, k0 = load(a.fan_s0, tab)
    z1, k1 = load(a.fan_s1, tab)
    assert k0 == k1, "window order differs between the two rolls"
    B = len(k0)
    gt = np.stack([tab[k][0] for k in k0])
    eid = [tab[k][1] for k in k0]
    out = {"n_windows": B, "n_episodes": len(set(eid))}
    de = {}
    sel = {}
    for tag, z in (("s0", z0), ("s1", z1)):
        d = X._de_all(z["anchor_traj"], gt)
        s = np.asarray(z["sel_idx"]).astype(np.int64).ravel()
        de[tag], sel[tag] = d, s
    sv = {t: de[t][np.arange(B), sel[t]] for t in de}
    ov = {t: de[t].min(1) for t in de}
    rv = {t: de[t].mean(1) for t in de}
    for nm, v in (("selected", sv), ("oracle_in_fan", ov), ("random_expectation", rv)):
        out[nm] = {"s0": X._boot(v["s0"], eid, nm + "_s0"), "s1": X._boot(v["s1"], eid, nm + "_s1"),
                   "s0_minus_s1": X._paired(v["s0"], v["s1"], eid, nm + " s0 vs s1 (paired a=s0,b=s1)")}
    gap = {t: sv[t] - ov[t] for t in sv}
    out["sel_gap"] = {"s0": X._boot(gap["s0"], eid, "sel_gap_s0"), "s1": X._boot(gap["s1"], eid, "sel_gap_s1"),
                      "s0_minus_s1": X._paired(gap["s0"], gap["s1"], eid, "sel_gap s0 vs s1")}
    out["skill_fraction"] = {t: round(float((rv[t] - sv[t]).mean() / (rv[t] - ov[t]).mean()), 4) for t in sv}
    out["sel_idx_agreement_frac"] = round(float((sel["s0"] == sel["s1"]).mean()), 4)
    # selected PLAN agreement in metres (the model own 2 s plan): how far apart are the two picks
    f0 = z0["anchor_traj"][np.arange(B), sel["s0"]].astype(np.float64)
    f1 = z1["anchor_traj"][np.arange(B), sel["s1"]].astype(np.float64)
    pd = np.linalg.norm(f0 - f1, axis=-1).mean(-1)
    out["selected_plan_l2_s0_vs_s1_m"] = X._boot(pd, eid, "mean 4wp L2 between the two selected plans")
    # candidate-set identity: are the 117 proposals the same across the two rolls (anchors) or refined differently
    a0 = z0["anchor_traj"].astype(np.float64); a1 = z1["anchor_traj"].astype(np.float64)
    out["fan_candidate_l2_s0_vs_s1_m_mean"] = round(float(np.linalg.norm(a0 - a1, axis=-1).mean()), 6)
    out["fan_candidate_bitwise_equal_frac"] = round(float((a0 == a1).all(axis=(2, 3)).mean()), 6)
    for nm in ("anchor_logits", "refined_logits", "sel_score"):
        if nm in z0.files and nm in z1.files:
            s0_, s1_ = np.asarray(z0[nm]).astype(np.float64), np.asarray(z1[nm]).astype(np.float64)
            out["score_maxabs_diff_" + nm] = round(float(np.abs(s0_ - s1_).max()), 6)
            out["argmax_agree_" + nm] = round(float((s0_.argmax(1) == s1_.argmax(1)).mean()), 4)
    out["stamp"] = {"tier": "T0 diagnostic (oracle uses GT); selected alone is T1 open-loop",
                    "ci_answers": "paired episode-cluster bootstrap OF THE REPLICATE DIFFERENCE; the size of s0-s1 is the INFERENCE-seed noise for this checkpoint; training-seed floor still unmeasured",
                    "fan_s0_sha256": X._sha256(a.fan_s0), "fan_s1_sha256": X._sha256(a.fan_s1),
                    "wall_s": round(time.time() - t0, 1), "script": os.path.abspath(__file__)}
    json.dump(out, open(a.out_json, "w", encoding="utf-8"), indent=1)
    print("WROTE", a.out_json)


if __name__ == "__main__":
    main()
