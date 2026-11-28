#!/usr/bin/env python3
# Resolve the balanced scenario suite (8/category) 8-char prefixes -> full clip UUIDs,
# write suite_clips.txt (full uuids) + suite_labels.json (clip8 -> category) + the wizard scene_ids.
import json
SEL = {
  "roundabout":     ["3cc29c99","471f2484","6dcd2117","bc843fa0","fd3a49fa","adb72a39","c3d4065e","d3267951"],
  "traffic_light":  ["03376794","0580c069","0fb04790","2387cbf7","2a47ea01","44c3b4d5","6c0a56c4","738c453d"],
  "highway":        ["0e05cb0c","15dd433e","1c9774e9","3dce1d04","4fdce89c","51249fb2","ca481583","d0d8a2b6"],
  "intersection":   ["00169207","0810968d","41c06176","59cb0598","69efe005","780ece49","7bb6216b","8b04d54e"],
  "straight_other": ["00eb506e","0ce02937","1e0f1cb2","27068a85","3416c63f","48a50c2a","8ca50753","bfb44da0"],
}
pool = [l.strip() for l in open("/workspace/pool_2604.txt") if l.strip()]
labels, full, missing, dup = {}, [], [], []
for cat, prefs in SEL.items():
    for p in prefs:
        matches = [c for c in pool if c[:8] == p]
        if not matches:
            missing.append(p); continue
        if len(matches) > 1:
            dup.append((p, matches))
        full.append(matches[0]); labels[p] = {"clip": matches[0], "category": cat}
with open("/workspace/suite_clips.txt", "w") as f:
    f.write("\n".join(full) + "\n")
json.dump(labels, open("/workspace/suite_labels.json", "w"), indent=1)
scene_ids = "[" + ",".join("clipgt-" + c for c in full) + "]"
print("resolved:", len(full), "missing:", missing, "dups:", dup)
print("SCENE_IDS=" + scene_ids)
