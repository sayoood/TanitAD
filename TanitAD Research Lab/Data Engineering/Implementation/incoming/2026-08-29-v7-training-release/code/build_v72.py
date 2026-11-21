"""v7.2 — the corpus SHIPPED AS TRAIN AND EVAL DATA, not as a manifest to apply.

PI: *"add the 6 alpamayo labeled clips to our val set … push it as new data v7.2
containing the train and eval splits"*.

WHAT CHANGES FROM v7.1
  * The split stops being a manifest a trainer has to apply correctly and becomes
    the DATA LAYOUT: two label files, one train, one eval. A trainer that reads
    the train file cannot accidentally train on eval.
  * The **6 val40-overlap clips join the EVAL side**. They were previously
    excluded from `eligible` entirely — held out of training for leakage and then
    used for nothing. They carry FULL v7.1 labels already (verified per clip:
    a_tac, g_tac, g_str, alpamayo and strata all present), so **nothing needed
    generating for them** — they were simply unused.
  * ⇒ every one of the 4,719 clips is now ASSIGNED. Nothing is discarded.

⚠️ THE CONSEQUENCE OF ADDING THEM, STATED RATHER THAN DISCOVERED: our eval set now
OVERLAPS the deployed val40 by 6 clips (6 of 40 = 15 % of val40; 6 of 147 = 4 % of
our eval). Neither set is trained on, so this is NOT a leak — but the two numbers
are no longer fully independent, and a comparison between our val curve and THE
published open-loop statistic must say so. That is why the overlap is recorded
explicitly in the manifest instead of being implicit in the ids.
"""
import gzip
import hashlib
import json
from pathlib import Path

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release")
V71 = REL / "v71"
OUT = REL / "v72"
OUT.mkdir(exist_ok=True)

rows = [json.loads(x) for x in gzip.open(V71 / "s2_labels_v7.1.jsonl.gz",
                                         "rt", encoding="utf-8") if x.strip()]
by_id = {r["clip_id"]: r for r in rows}
v40 = set(json.load(open(REL / "val40_digests.json"))["clip_id_digests"])


def dig(c):
    return hashlib.sha256(c.encode("utf-8")).hexdigest()


overlap6 = sorted(c for c in by_id if dig(c) in v40)
assert len(overlap6) == 6, len(overlap6)

split3 = json.load(open(V71 / "eval_split_v3.json"))
eval_ids = set(split3["eval_clip_ids"]) | set(overlap6)
train_ids = set(split3["train_clip_ids"])

# --- ASSERTIONS: the whole point of shipping data rather than a manifest ------
assert not (eval_ids & train_ids), "⛔ a clip is in BOTH train and eval"
assert eval_ids | train_ids == set(by_id), "some clip is assigned to neither side"
assert not (train_ids & {c for c in by_id if dig(c) in v40}), \
    "⛔ TRAIN contains a deployed-val40 clip — that is the leak we exclude for"
assert set(overlap6) <= eval_ids, "the 6 did not land in eval"
print(f"train {len(train_ids)} | eval {len(eval_ids)} | total {len(train_ids)+len(eval_ids)} "
      f"of {len(by_id)}")
print(f"  eval includes the 6 val40-overlap clips: {[c[:8] for c in overlap6]}")


def write(ids, name):
    p = OUT / name
    with gzip.open(p, "wt", encoding="utf-8", compresslevel=9) as f:
        for cid in sorted(ids):
            r = dict(by_id[cid])
            # ⛔ schema_version identifies the FORMAT CONTRACT, never the release.
            # MEASURED: bumping it to "s2_labels_v7.2" made the blob UNLOADABLE
            # by tanitad.data.v7_labels (EXPECTED_SCHEMA = "s2-geom-v7") — every
            # consumer refused a file whose format had not actually changed.
            # The additions here are backward-compatible FIELDS, so the contract
            # is unchanged; the release rides in its own key.
            r["schema_version"] = "s2-geom-v7"
            r["release"] = "v7.2"
            r["split"] = "eval" if cid in eval_ids else "train"
            f.write(json.dumps(r, sort_keys=True) + "\n")
    b = p.read_bytes()
    return p, hashlib.md5(b).hexdigest(), hashlib.sha256(b).hexdigest(), len(b)


tp, tmd5, tsha, tsz = write(train_ids, "s2_labels_v7.2_train.jsonl.gz")
ep, emd5, esha, esz = write(eval_ids, "s2_labels_v7.2_eval.jsonl.gz")
print(f"\nwrote {tp.name} {tsz/1e6:.2f} MB md5 {tmd5}")
print(f"wrote {ep.name} {esz/1e6:.2f} MB md5 {emd5}")

# --- read the shipped files BACK and re-assert on THEM ------------------------
tr2 = {json.loads(x)["clip_id"] for x in gzip.open(tp, "rt", encoding="utf-8") if x.strip()}
ev2 = {json.loads(x)["clip_id"] for x in gzip.open(ep, "rt", encoding="utf-8") if x.strip()}
assert tr2 == train_ids and ev2 == eval_ids, "written files do not match the intended split"
assert not (tr2 & ev2), "written files overlap"
assert not (tr2 & {c for c in by_id if dig(c) in v40}), "written TRAIN contains val40"
print("✅ re-asserted ON THE WRITTEN FILES: disjoint · complete · train is val40-clean")

import pandas as pd  # noqa: E402
ev_rows = [by_id[c] for c in eval_ids]
tr_rows = [by_id[c] for c in train_ids]


def comp(rs, key):
    s = pd.Series([(r.get("strata") or {}).get(key) for r in rs])
    return (s.value_counts(normalize=True) * 100).round(1).to_dict()


man = {
    "schema": "s2_labels_v7.2", "based_on": {"v7.1_md5": hashlib.md5(
        (V71 / "s2_labels_v7.1.jsonl.gz").read_bytes()).hexdigest()},
    "layout": "TRAIN AND EVAL SHIPPED AS SEPARATE FILES — the split is the data "
              "layout, not a manifest a trainer must apply",
    "files": {"train": {"name": tp.name, "n": len(train_ids), "md5": tmd5, "sha256": tsha},
              "eval": {"name": ep.name, "n": len(eval_ids), "md5": emd5, "sha256": esha}},
    "eval_pct_of_corpus": round(len(eval_ids) / len(by_id) * 100, 2),
    "changes_from_v7.1": [
        "corpus split into train/eval FILES",
        "the 6 deployed-val40 clips ADDED to eval (they already carried full "
        "v7.1 labels; they were held out of training and previously unused)",
        "every clip now assigned — nothing discarded",
    ],
    "deployed_val40_overlap": {
        "n": len(overlap6), "clip_ids": overlap6,
        "pct_of_val40": round(len(overlap6) / 40 * 100, 1),
        "pct_of_our_eval": round(len(overlap6) / len(eval_ids) * 100, 1),
        "consequence": ("NOT a leak — neither set is trained on. But our eval "
                        "number and THE published val40 open-loop statistic are no "
                        "longer fully independent, and any comparison between them "
                        "must say so."),
    },
    "composition": {
        "eval": {"road_class": comp(ev_rows, "road_class"),
                 "daynight_clock": comp(ev_rows, "daynight_clock")},
        "train": {"road_class": comp(tr_rows, "road_class"),
                  "daynight_clock": comp(tr_rows, "daynight_clock")}},
    "interpretation": {
        "admissible_use": "aggregate val curve at any cadence; checkpoint selection.",
        "inadmissible_use": ("⛔ PER-COMPETENCE numbers — at n=147 the eval side "
                             "holds only ~13 stop-launch and ~24 highway clips. "
                             "Also not a substitute for the deployed val40, which "
                             "it now partially overlaps."),
    },
    "assertions_passed": ["train ∩ eval = empty", "train ∪ eval = the whole corpus",
                          "train contains NO deployed-val40 clip",
                          "re-verified by reading the written files back"],
}
json.dump(man, open(OUT / "V72_MANIFEST.json", "w"), indent=1)
print(f"\neval {len(eval_ids)} = {len(eval_ids)/len(by_id)*100:.2f}% of the corpus")
print("wrote V72_MANIFEST.json")
