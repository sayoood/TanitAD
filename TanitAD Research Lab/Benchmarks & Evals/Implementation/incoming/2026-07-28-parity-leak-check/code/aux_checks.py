"""Auxiliary checks around the parity leak verdict.

1. Is the 40-episode DEPLOYED val cache on `tanitad-eval` a content-subset of the
   600-episode registered split `physicalai-val-0c5f7dac3b11` (whose poses-only
   VIEW survives on pod3)?  If yes, the 600-ep poses check covers the deployed 40.
2. Is the poses-only VIEW of the train cache byte-faithful to the real cache?
   (the view is a derived artefact; if it is not faithful, its result is void)
3. `maneuvers_sha256` is the ONE family that disagreed.  Quantify its entropy so
   the disagreement is explained rather than waved away.
4. episode_id uniqueness inside the train corpus.
"""
import json, sys
from collections import Counter
from pathlib import Path


def load(p):
    rows = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[0], {r["file"].replace(".pt", ""): r for r in rows[1:]}


base = sys.argv[1] if len(sys.argv) > 1 else "."
h40, e40 = load(f"{base}/fp_val_0c5f_deployed40_full.jsonl")
h600, e600 = load(f"{base}/fp_val_0c5f_view600_poses.jsonl")
htrf, etrf = load(f"{base}/fp_train_e4387_full.jsonl")
htrv, etrv = load(f"{base}/fp_train_e4387_view_poses.jsonl")

out = {
    "what": "auxiliary checks around the PhysicalAI parity leak verdict",
    "date": "2026-07-28",
    "evidence_class": "MEASURED (ours; sha256 of raw bytes)",
}

# --- 1. deployed-40 subset of registered-600 ------------------------------- #
s40 = {v["poses_sha256"] for v in e40.values()}
s600 = {v["poses_sha256"] for v in e600.values()}
matched = sorted(k for k, v in e40.items() if v["poses_sha256"] in s600)
# do the tag names line up too?
by_hash600 = {v["poses_sha256"]: k for k, v in e600.items()}
same_tag = sum(1 for k, v in e40.items()
               if by_hash600.get(v["poses_sha256"]) == k)
out["deployed40_is_subset_of_registered600"] = {
    "n_deployed": len(e40), "n_registered_view": len(e600),
    "n_of_40_found_in_600_by_poses_sha256": len(matched),
    "is_subset": len(matched) == len(e40),
    "n_where_the_TAG_also_matches": same_tag,
    "deployed_host": h40["host"], "deployed_path": h40["cache"],
    "view_host": h600["host"], "view_path": h600["cache"],
    "why": "the eval pod carries 40 of the split's 600 episodes; if the 40 are a "
           "content-subset of the 600, then the cheap 600-ep poses check covers "
           "the deployed substrate too.",
}

# --- 2. is the train VIEW byte-faithful to the real train cache? ----------- #
agree = sum(1 for k in etrf if k in etrv and etrf[k]["poses_sha256"] == etrv[k]["poses_sha256"])
out["train_view_is_byte_faithful"] = {
    "n_full": len(etrf), "n_view": len(etrv),
    "n_tags_with_identical_poses_sha256": agree,
    "faithful": agree == len(etrf) == len(etrv),
    "why": "the poses-only VIEW is a DERIVED artefact. If it were not "
           "byte-identical to the source poses, every result computed on it "
           "would be void. This is the check that licenses using it.",
}

# --- 3. maneuvers_sha256 entropy — the family that disagreed --------------- #
def ent(eps, fam):
    c = Counter(v[fam] for v in eps.values() if v.get(fam))
    return {"n_episodes": len(eps), "n_distinct": len(c),
            "modal_group_size": max(c.values()) if c else 0,
            "frac_in_modal_group": (max(c.values()) / len(eps)) if c else 0.0}

out["maneuvers_sha256_is_NOT_an_identifying_hash"] = {
    "train": ent(etrf, "maneuvers_sha256"),
    "val40": ent(e40, "maneuvers_sha256"),
    "train_poses_for_contrast": ent(etrf, "poses_sha256"),
    "train_frames_for_contrast": ent(etrf, "frames_sha256"),
    "why": "`maneuvers` is a [T] int64 CATEGORICAL LABEL sequence, not content. "
           "Many distinct episodes carry the identical sequence (e.g. all-follow). "
           "Its 'matches' are label collisions, NOT shared content — which is why "
           "it is the one family that disagrees with the other six.",
}

# --- 3b. prove it: the val episodes that 'match' on maneuvers share NOTHING - #
mtr = {}
for k, v in etrf.items():
    if v.get("maneuvers_sha256"):
        mtr.setdefault(v["maneuvers_sha256"], []).append(k)
rows = []
for k, v in e40.items():
    if v.get("maneuvers_sha256") in mtr:
        tt = mtr[v["maneuvers_sha256"]]
        rows.append({
            "val_tag": k,
            "n_train_episodes_with_the_same_maneuver_sequence": len(tt),
            "any_identifying_family_also_matches": any(
                etrf[t][f] == v[f] for t in tt
                for f in ("poses_sha256", "frames_sha256", "actions_sha256")),
        })
out["maneuvers_collisions_carry_no_content_match"] = {
    "n_val_episodes_colliding": len(rows),
    "n_with_ANY_identifying_family_also_matching": sum(
        1 for r in rows if r["any_identifying_family_also_matches"]),
    "rows": rows,
}

# --- 4. episode_id uniqueness inside train --------------------------------- #
cid = Counter(v["episode_id"] for v in etrf.values())
dupids = {k: v for k, v in cid.items() if v > 1}
out["episode_id_is_not_unique_inside_train"] = {
    "n_train_episodes": len(etrf), "n_distinct_episode_ids": len(cid),
    "n_ids_used_more_than_once": len(dupids),
    "n_episodes_sharing_an_id": sum(dupids.values()),
    "max_multiplicity": max(dupids.values()) if dupids else 1,
    "why": "an `episode_id`-based disjointness claim is weaker than it looks "
           "when the id is not even a key inside one cache.",
}

# --- 5. within-train near-duplicate found by the primary scan --------------- #
pairs = {}
for fam in ("poses_xy_sha256", "poses_v_sha256", "poses_sha256", "frames_sha256"):
    seen = {}
    for k, v in etrf.items():
        seen.setdefault(v[fam], []).append(k)
    pairs[fam] = [sorted(g) for g in seen.values() if len(g) > 1]
out["within_train_near_duplicates"] = {
    fam: {"n_groups": len(g), "groups": g[:10]} for fam, g in pairs.items()
    if fam != "maneuvers_sha256"
}

print(json.dumps(out, indent=1))
Path(f"{base}/aux_checks.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
