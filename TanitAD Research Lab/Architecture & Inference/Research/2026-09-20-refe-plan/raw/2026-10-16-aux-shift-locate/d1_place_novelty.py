#!/usr/bin/env python3
"""H-b (scene composition, PLACE axis): how far is each evaluated frame from the head's training frames, in the map's own coordinates?  CPU, read-only.

nuPlan logs revisit the same streets, and splits are by LOG, so a log-disjoint evaluation frame can still sit on a street the head trained on (or not).
The frame poses (x, y, heading in the map frame) are stored in every label-store shard header (`p`).  Distance = Euclidean in (x, y, 20 cos h, 20 sin h):
a frame on the same street heading the other way is 40 m "away".  Nearest neighbour is taken within the SAME MAP over the head's training frames:
  S1 head (navtrain-trained): all train-store frames of the 1,168 non-held-out logs      (101,431 frames)
For the in-domain navtest probe heads (V, trained on the other 4 folds of the 1,058 C3 tokens) the reference is those folds' tokens (see d2).
Then the H1-s0 IoU is stratified by that distance in both domains and the navtest read is re-weighted to the held-out distance mix (direct standardisation)."""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shf_common as C
import shf_arrays as SA
import aux_label_store as ALS
from pathlib import Path

ARM = os.environ.get("SHF_ARM", "H1-s0")
SHORT = {"us-nv-las-vegas-strip": "LV", "us-ma-boston": "BOS", "us-pa-pittsburgh-hazelwood": "PIT", "sg-one-north": "SG"}


def poses(store_dir):
    out = {}
    for f in sorted(os.listdir(f"{store_dir}/shards")):
        head, _ = ALS.read_shard(Path(f"{store_dir}/shards/{f}"), with_blobs=False)
        for e in head["entries"]:
            out[(head["log"], e["t"])] = (SHORT[head["map"]], e["p"][0], e["p"][1], e["p"][2])
    return out


def feat(p):
    m, x, y, h = p
    return m, np.array([x, y, 20 * np.cos(h), 20 * np.sin(h)])


def nn_dist(queries, ref_by_map):
    out = np.zeros(len(queries))
    pos = np.zeros(len(queries))
    trees = {m: (cKDTree(v[:, :4]), cKDTree(v[:, :2])) for m, v in ref_by_map.items()}
    for i, (m, f) in enumerate(queries):
        out[i] = trees[m][0].query(f)[0]
        pos[i] = trees[m][1].query(f[:2])[0]
    return out, pos


def main():
    held = set(C.heldout_logs())
    P_tr = poses(f"{C.AUXDIR}/label_store")
    P_nt = poses(f"{C.AUXDIR}/label_store_navtest")
    ref = {}
    for k, p in P_tr.items():
        if k[0] in held:
            continue
        m, f = feat(p)
        ref.setdefault(m, []).append(f)
    ref = {m: np.array(v) for m, v in ref.items()}
    n_ref = {m: len(v) for m, v in ref.items()}
    Dh, Dn = SA.domain("heldout", (ARM,)), SA.domain("navtest", (ARM,))
    q_h = [feat(P_tr[k]) for k in Dh["keys"]]
    q_n = [feat(P_nt[k]) for k in Dn["keys"]]
    dh, ph = nn_dist(q_h, ref)
    dn, pn = nn_dist(q_n, ref)
    np.savez_compressed(f"{C.SCRATCH}/place_dist.npz", dh=dh, ph=ph, dn=dn, pn=pn)
    res = {"n_ref_frames_by_map": n_ref, "metric": "NN distance in (x, y, 20cos h, 20sin h), same map, reference = head-training frames (non-held-out train store)"}

    def summ(d):
        return dict(median=float(np.median(d)), p25=float(np.percentile(d, 25)), p75=float(np.percentile(d, 75)), p90=float(np.percentile(d, 90)),
                    **{f"share_le_{t}m": float((d <= t).mean()) for t in (2, 5, 10, 25, 50)})
    res["dist_summary"] = {"heldout": summ(dh), "navtest_c3": summ(dn)}
    res["pos_only_summary"] = {"heldout": summ(ph), "navtest_c3": summ(pn)}
    res["by_map"] = {}
    for m in ("LV", "BOS", "PIT", "SG"):
        res["by_map"][m] = {"heldout": summ(dh[Dh["maps"] == m]) if (Dh["maps"] == m).any() else None, "navtest_c3": summ(dn[Dn["maps"] == m])}
    # IoU stratified by place distance
    edges = [0, 2, 5, 10, 25, 1e9]
    names = ["<2", "2-5", "5-10", "10-25", ">=25"]
    ih, uh, _ = SA.frame_iou_parts(Dh, ARM)
    inn, un, _ = SA.frame_iou_parts(Dn, ARM)
    bh, bn = np.digitize(dh, edges[1:-1]), np.digitize(dn, edges[1:-1])
    tab = {}
    for k, nm in enumerate(names):
        mh, mn = bh == k, bn == k
        tab[nm] = dict(n_heldout=int(mh.sum()), share_heldout=float(mh.mean()), iou_heldout=(float(ih[mh].sum() / uh[mh].sum()) if mh.any() else None),
                       n_navtest=int(mn.sum()), share_navtest=float(mn.mean()), iou_navtest=(float(inn[mn].sum() / un[mn].sum()) if mn.any() else None))
    res["iou_by_place_distance"] = tab
    # standardise navtest to the held-out distance mix, and held-out to the navtest mix
    def rew(i, u, b_from, b_to):
        w = np.ones(len(i))
        for k in range(len(names)):
            m = b_from == k
            if m.any():
                w[m] = float((b_to == k).mean()) / float(m.mean())
        return float((w * i).sum() / (w * u).sum())
    res["navtest_to_heldout_place_mix"] = rew(inn, un, bn, bh)
    res["heldout_to_navtest_place_mix"] = rew(ih, uh, bh, bn)
    # joint with map
    def rew_joint(i, u, D_from, b_from, D_to, b_to):
        kf = np.array([f"{m}|{b}" for m, b in zip(D_from["maps"], b_from)])
        kt = np.array([f"{m}|{b}" for m, b in zip(D_to["maps"], b_to)])
        w = np.zeros(len(i))
        miss = 0
        for kk in set(kf):
            m = kf == kk
            w[m] = float((kt == kk).mean()) / float(m.mean())
            miss += int(not (kt == kk).any())
        return dict(iou=float((w * i).sum() / (w * u).sum()), ess=float(w.sum() ** 2 / (w ** 2).sum()), n_cells_absent_in_target=miss, frames_with_zero_weight=int((w == 0).sum()))
    res["navtest_to_heldout_map_x_place_mix"] = rew_joint(inn, un, Dn, bn, Dh, bh)
    res["heldout_to_navtest_map_x_place_mix"] = rew_joint(ih, uh, Dh, bh, Dn, bn)
    # correlation of IoU with distance at frame level (rank), within each domain
    from scipy.stats import spearmanr
    fh, fn = ih / np.maximum(uh, 1), inn / np.maximum(un, 1)
    res["spearman_frame_iou_vs_distance"] = {"heldout": float(spearmanr(dh, fh)[0]), "navtest_c3": float(spearmanr(dn, fn)[0])}
    C.jdump(res, os.path.join(C.HERE, "d1_place_novelty.json"))
    print(json.dumps(res, indent=1, sort_keys=True)[:6000])


if __name__ == "__main__":
    main()
