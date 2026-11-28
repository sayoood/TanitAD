"""The 2,400-vs-2,376 reconciliation, as a POSITIVE assertion."""
import hashlib, json
from pathlib import Path
from tanitad.data import parity
from tanitad.data.v2_dataset import stable_episode_id
import tanitad.data.v7_labels as v7l

A = Path(r"C:/Users/Admin/tanitad-review-20260906/art")
clips = json.loads((A/"parity_clip_ids.json").read_text())
if isinstance(clips, dict):
    for k in ("clip_ids","clips","ids"):
        if k in clips: clips = clips[k]; break
clips = [str(c) for c in clips]

raw = parity.manifest_entry("physicalai-train-e438721ae894")
v2  = parity.manifest_entry("physicalai-train-e438721ae894-w120-256x640cyl")
print("RAW  key=%s  episode_count=%s  n_clips=%s  skip_count=%s"
      % (raw["corpus_key"], raw["episode_count"], raw["clip_membership"]["n_clips"], raw["skip_count"]))
print("V2   key=%s  episode_count=%s  skip_count=%s  uid_kind=%s"
      % (v2["corpus_key"], v2["episode_count"], v2["skip_count"], v2.get("uid_kind")))
print("CLAUDE.md invariant PARITY_TRAIN_EPISODES =", parity.PARITY_TRAIN_EPISODES)
print()
print("ARITHMETIC: discovered %s - val %s = %s train CLIPS; %s raw-build failures -> %s raw EPISODES"
      % (raw["clip_membership"]["discovered_clips"], raw["clip_membership"]["val_clips"],
         raw["clip_membership"]["n_clips"], raw["skip_count"], raw["episode_count"]))
print("            the V2 cache built ALL %s clips (skip_count %s) -> %s v2 EPISODES"
      % (v2["provenance"]["membership_proof"]["clips_built"], v2["skip_count"], v2["episode_count"]))
print()
# the 24 clips present in v2 and ABSENT from the raw epcache
si = sorted(int(i) for i in raw["skip_indices"])
ordered_is_sorted = raw["clip_membership"]["clip_id_sha256_ordered"] == raw["clip_membership"]["clip_id_sha256_sorted"]
print("ordered==sorted digest:", ordered_is_sorted, "(so skip index i == sorted position i)")
cs = sorted(clips)
skipped = [cs[i] for i in si]
print("the 24 RAW-SKIPPED clips (present in the v2 cache, absent from the 2,376 raw epcache):")
for c in skipped[:5]: print("   ", c)
print("    ... (%d total)" % len(skipped))
# are they in the parity-v7geom label blob the refcv5 arm will train on?
labels,_ = v7l.load_v7_labels(str(A/"s2_labels_parity-v7geom-1_train.jsonl.gz"), allow_oracle_nav=True)
lab_sid = {stable_episode_id(l.clip_id) for l in labels}
inlab = sum(1 for c in skipped if stable_episode_id(c) in lab_sid)
print("of those 24, LABELLED by the new parity blob: %d/24" % inlab)
print()
print("=> refcv5 (v2 cache) trains on %d episodes; every raw-epcache arm trained on %d."
      % (v2["episode_count"], raw["episode_count"]))
print("=> delta = %d episodes = %.2f %% of the corpus, DISJOINT between the two arms' training sets."
      % (v2["episode_count"]-raw["episode_count"], 100*(v2["episode_count"]-raw["episode_count"])/v2["episode_count"]))
