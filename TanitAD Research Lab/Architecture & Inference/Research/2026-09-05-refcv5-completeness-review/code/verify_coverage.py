"""INDEPENDENT re-measurement of the refcv5 label-coverage claim.

Reproduces refc_v3_train.py:2627-2628 arithmetic EXACTLY:
    by_sid = {stable_episode_id(l.clip_id): l for l in labels}
    hit    = sum(1 for e in eps if int(e.episode_id) in by_sid)
    frac   = hit / max(len(eps), 1)
`eps` episode ids are stable_episode_id(clip_id) (build_v2_providers stable_ids=True).
"""
import gzip, hashlib, json, sys
from pathlib import Path

from tanitad.data.v2_dataset import stable_episode_id
from tanitad.data import parity
from tanitad.data import v7_labels as v7l

PKG = Path(r"C:/Users/Admin/tanitad-review-20260906/art")
ARCH = Path(r"C:/Users/Admin/tanitad-review-20260906/art")

out = {}

# ---- 1. the parity clip set, and its digest, asserted not assumed -----------
clips = json.loads((ARCH / "parity_clip_ids.json").read_text())
if isinstance(clips, dict):
    out["parity_clip_ids_json_keys"] = list(clips.keys())
    for k in ("clip_ids", "clips", "ids"):
        if k in clips:
            clips = clips[k]; break
clips = [str(c) for c in clips]
dig = hashlib.sha256("\n".join(sorted(clips)).encode()).hexdigest()
man = parity.manifest_entry("physicalai-train-e438721ae894")
exp = man["clip_membership"]["clip_id_sha256_sorted"]
out["n_parity_clips"] = len(clips)
out["parity_digest_recomputed"] = dig
out["parity_digest_manifest"] = exp
out["parity_digest_MATCH"] = (dig == exp)
out["manifest_episode_count"] = man["episode_count"]
out["manifest_n_clips"] = man["clip_membership"]["n_clips"]
out["manifest_skip_count"] = man["skip_count"]

# ---- 2. the trainer's `eps` ids -------------------------------------------
eps_ids = [stable_episode_id(c) for c in clips]
out["n_distinct_stable_ids"] = len(set(eps_ids))

# ---- 3. AFTER blob: the trainer's own arithmetic ---------------------------
def cov(blob):
    labels, manifest = v7l.load_v7_labels(str(blob), allow_oracle_nav=True)
    by_sid = {stable_episode_id(l.clip_id): l for l in labels}
    hit = sum(1 for e in eps_ids if int(e) in by_sid)
    return len(labels), hit, hit / max(len(eps_ids), 1), manifest.to_dict()

n, hit, frac, mani = cov(PKG / "s2_labels_parity-v7geom-1_train.jsonl.gz")
out["AFTER"] = {"records": n, "hit": hit, "n_eps": len(eps_ids),
                "frac": frac, "frac_pct": round(100*frac, 3),
                "trainer_floor_0.50": "PASS" if frac >= 0.5 else "REFUSED",
                "manifest": mani}
print(json.dumps(out, indent=1, default=str))
