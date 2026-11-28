"""PREREG_V8F_WMACT TRAIN read, step 0: the sample (fixed in the card log 2026-11-07 09:12 before any TRAIN data). Writes train_stream/clips141.json."""
import hashlib, json, os
M = "train_stream/box2_meta/manifest.jsonl"
ents = {}
for ln in open(M, encoding="utf-8"):
    if ln.strip():
        d = json.loads(ln); ents[d["clip"]] = d
clips_all = list(ents)
sha = lambda c: hashlib.sha256(c.encode("utf-8")).hexdigest()
sha12 = lambda c: sha(c)[:12]
assert all(ents[c]["sha12"] == sha12(c) for c in clips_all), "manifest sha12 field disagrees with sha256(clip)[:12]"
order = sorted(clips_all, key=sha12)                       # == v8f_mini_train.py:2386 sorted(clips_all, key=D.sha12)
excluded = order[540:600]                                  # --exclude 540:600 (v8f_mini_train.py:2383-2388)
ex = set(excluded)
trained = [c for c in clips_all if c not in ex]            # the 4,512
pick = sorted(trained, key=sha)[:141]                      # smallest sha256(clip id) among the 4,512
pick12 = sorted(trained, key=sha12)[:141]
out = {"n_manifest": len(clips_all), "n_excluded": len(excluded), "n_trained": len(trained),
       "excluded_sha256": hashlib.sha256("\n".join(excluded).encode()).hexdigest(),
       "train_sha256_trainer_form": hashlib.sha256("\n".join(sorted(trained, key=sha12)).encode()).hexdigest(),
       "pick_equals_sha12_order": pick == pick12, "pick_in_excluded": len(set(pick) & ex),
       "pick_positions_in_full_sha12_order": [order.index(c) for c in pick][:3] + ["..."] + [order.index(c) for c in pick][-3:],
       "clips": pick, "n_frames": {c: ents[c]["n_frames"] for c in pick}, "bytes": {c: ents[c]["bytes"] for c in pick}, "md5": {c: ents[c]["md5"] for c in pick}}
out["clips_sha256"] = hashlib.sha256("\n".join(pick).encode()).hexdigest()
out["total_bytes"] = sum(out["bytes"].values())
json.dump(out, open("train_stream/clips141.json", "w"), indent=1)
open("train_stream/clips141.txt", "w", newline="\n").write("\n".join(pick) + "\n")
print({k: v for k, v in out.items() if k not in ("clips", "n_frames", "bytes", "md5")})
print("clips.txt sha256", hashlib.sha256(open("train_stream/clips141.txt", "rb").read()).hexdigest())
