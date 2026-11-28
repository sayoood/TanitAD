"""P3 -- shard disjointness, selection completeness, corpus-key reproduction."""
import hashlib, json, os, sys
import numpy as np, pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
SEL = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Hub"
       r"\Data Engineering\Implementation\incoming\2026-07-24-v2-corpus-50h-balanced"
       r"\r0_selection_v2.parquet")
TMAN = [0.45, 0.14, 0.14, 0.13, 0.14]
TSPD = [0.10, 0.52, 0.38]

sel = pd.read_parquet(SEL)
sel_ids = set(sel["clip_id"].astype(str))
p1 = {l.strip() for l in open(os.path.join(SP, "ids_pod1.txt")) if l.strip()}
p3 = {l.strip() for l in open(os.path.join(SP, "ids_pod3.txt")) if l.strip()}
union = p1 | p3

out = {
    "selection_rows": int(len(sel)),
    "selection_unique_clip_ids": len(sel_ids),
    "pod1_built": len(p1), "pod3_built": len(p3),
    "union_built": len(union),
    "overlap_pod1_pod3": len(p1 & p3),
    "overlap_clip_ids": sorted(p1 & p3)[:50],
    "selected_missing_from_both": sorted(sel_ids - union),
    "built_not_in_selection": sorted(union - sel_ids),
}
out["n_selected_missing"] = len(out["selected_missing_from_both"])
out["n_built_not_selected"] = len(out["built_not_in_selection"])

# corpus key reproduced from the BUILT clip-id set (select_v2_corpus.py recipe)
def key_of(ids, k=9000):
    return hashlib.sha1(json.dumps(
        {"ids": sorted(ids), "target": TMAN + TSPD, "k": k},
        sort_keys=True).encode()).hexdigest()[:12]

out["corpus_key_from_selection"] = key_of([str(x) for x in sel_ids])
out["corpus_key_from_built_union"] = key_of(list(union))
out["corpus_key_expected"] = "4b7eeeac222d"

# selection-time (projected) aggregates recomputed from the parquet itself
S = sel
man = S[["lk", "tl", "tr", "ac", "bs"]].to_numpy(float)
out["selection_projected_maneuver"] = dict(zip(
    ["lane_keep", "turn_left", "turn_right", "accelerate", "brake_stop"],
    (man.sum(0) / man.sum()).round(6).tolist()))
out["selection_projected_speed_clipmean"] = {
    k: round(float(S[k].mean()), 6) for k in ("stopped", "city", "hw")}
nl = S["nlab"].to_numpy(float)
out["selection_projected_speed_stepweighted"] = {
    k: round(float((S[k].to_numpy(float) * nl).sum() / nl.sum()), 6)
    for k in ("stopped", "city", "hw")}
out["selection_projected_presence"] = {
    k: round(float(S[k].mean()), 6) for k in ("junction", "has_stop",
                                              "has_brake", "has_turn")}
out["selection_projected_net_head_gt45"] = round(float((S.net_head > 45).mean()), 6)
out["selection_projected_net_head_gt90"] = round(float((S.net_head > 90).mean()), 6)
out["selection_projected_total_label_steps"] = int(nl.sum())
out["selection_projected_T_mean"] = float((nl + 20).mean())
out["selection_chunks"] = int(S["chunk"].nunique())

# which pod got which chunk (the build sharded by chunk index)
S2 = S.copy(); S2["clip_id"] = S2["clip_id"].astype(str)
S2["pod"] = np.where(S2.clip_id.isin(p1), "pod1",
                     np.where(S2.clip_id.isin(p3), "pod3", "MISSING"))
out["pod_by_chunk_split_chunks"] = int(
    (S2.groupby("chunk")["pod"].nunique() > 1).sum())
out["pod_counts"] = S2["pod"].value_counts().to_dict()

print(json.dumps(out, indent=2)[:6000])
json.dump(out, open(os.path.join(SP, "p3_disjoint.json"), "w"), indent=2)
