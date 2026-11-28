"""Build the `clip_index.json` our label release does not ship.

⛔ WHY IT IS REQUIRED. The epcache carries NO labels — its episodes hold only
pixels, actions and poses, keyed by an int `episode_id`. The labels key on the
clip UUID. The index is the ONLY bridge, and the loader says so when it refuses:
*"without it the labels are unjoinable and the term silently never fires.
Refusing rather than silently never firing."* MEASURED: running the real loader
against our v7.2 release raised exactly that.

⭐ THE IDS ARE COMPUTED BY CALLING `tanitad.data.v2_dataset.stable_episode_id`,
never by reproducing its logic. The loader CROSS-CHECKS every recorded
`episode_id_stable` against that function and refuses on drift — so a
reimplementation that agrees today is a second source of truth that can diverge
tomorrow, and the cross-check would then refuse the whole join.

⚠️ `episode_id_legacy` (the 16-bit id actually baked into the v2ep payloads) is
recorded too, because the index is the only place the two ids can be reconciled;
it collides (6.8 % of clips corpus-wide) and MUST NOT be used for the join.
"""
import argparse
import glob
import gzip
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True, help="a v7 labels .jsonl(.gz)")
ap.add_argument("--epcache", required=True, help="dir of *.v2ep.pt")
ap.add_argument("--corpus", required=True, help="corpus name recorded per clip")
ap.add_argument("--label-split", default="v7")
ap.add_argument("--out", required=True)
ap.add_argument("--t0-s", type=float, default=8.0)
a = ap.parse_args()

op = gzip.open if a.labels.endswith(".gz") else open
with op(a.labels, "rt", encoding="utf-8") as f:
    clips = [json.loads(x)["clip_id"] for x in f if x.strip()]
print(f"labels: {len(clips)} clips")

have = {Path(p).name.split(".")[0]: Path(p).name
        for p in glob.glob(os.path.join(a.epcache, "*.v2ep.pt"))}
print(f"epcache: {len(have)} episodes")

entries, missing = {}, []
for cid in sorted(clips):
    st = stable_episode_id(cid)                      # ⭐ CALLED, not reproduced
    leg = int.from_bytes(cid.encode()[:4], "big")    # the id baked into payloads
    f = have.get(cid)
    if f is None:
        missing.append(cid)
    entries[cid] = {
        "label_split": a.label_split,
        "corpus": a.corpus,
        "v2ep_file": f,                              # None when no episode exists
        "episode_id_legacy": leg,
        "episode_id_stable": st,
        # ⛔ `excluded` means "the loader must REFUSE a label record for this
        # clip". A clip with no episode is NOT marked excluded here — that would
        # silently redefine the release. It is reported instead, so the decision
        # is made by a person rather than by this script.
        "excluded": False,
    }

# collision census on the legacy id — the reason the stable id exists at all
by_leg: dict[int, list[str]] = {}
for cid, e in entries.items():
    by_leg.setdefault(e["episode_id_legacy"], []).append(cid)
coll = {k: v for k, v in by_leg.items() if len(v) > 1}
assert len({e["episode_id_stable"] for e in entries.values()}) == len(entries), \
    "stable-id collision — refusing to write an index that cannot join"

idx = {
    "_doc": ("clip UUID -> episode ids for the v7 label join. "
             "`episode_id_stable` is the ONLY admissible join key and is computed "
             "by calling tanitad.data.v2_dataset.stable_episode_id. "
             "`episode_id_legacy` is the 16-bit id BAKED INTO the v2ep payloads "
             "and COLLIDES — recorded for reconciliation, never for joining."),
    "_t0_s": a.t0_s,
    "_valid_window_s": [-2.0, 2.0],
    "_built_from": {"labels": os.path.basename(a.labels),
                    "epcache": a.epcache, "n_labels": len(clips),
                    "n_episodes": len(have)},
    "_legacy_collisions": {a.corpus: {"n_clips": len(entries),
                                      "n_legacy_ids": len(by_leg),
                                      "n_clips_in_colliding_ids":
                                          sum(len(v) for v in coll.values())}},
    "_clips_without_episode": missing,
    "clips": entries,
}
Path(a.out).write_text(json.dumps(idx, indent=1), encoding="utf-8")
print(f"wrote {a.out}: {len(entries)} clips")
print(f"  legacy-id collisions: {sum(len(v) for v in coll.values())} clips "
      f"in {len(coll)} colliding ids  (stable ids: 0 collisions, asserted)")
if missing:
    print(f"  ⚠️ {len(missing)} clips have NO epcache episode: "
          f"{[c[:8] for c in missing[:8]]}")
    print("     They are indexed but unjoinable to pixels. NOT marked excluded — "
          "that is a release decision, not a script's.")
