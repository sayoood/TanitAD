"""Re-derive the S2 strategic labels under the §LC ruling -> `review/labels_v2/`.

⛔ NON-DESTRUCTIVE BY DESIGN. The v1 files stay exactly as they are: they are
the artifact the PI adjudicated, and `PI_VERDICTS_2026-08-16.json` indexes
them. Overwriting them would destroy the primary source of the review that
caused this change.

⛔ AND THE v2 SET GETS ITS OWN DIRECTORY, NOT A `_v2` SUFFIX BESIDE v1.
MEASURED THE HARD WAY: `s2_labels.load_s2_labels()` takes a DIRECTORY and
**globs `s2_labels_*.jsonl`**, so `s2_labels_aug120_v2.jsonl` dropped into
`labels/` was picked up ALONGSIDE v1 and the loader refused the whole set on
a duplicate `clip_id` — correctly, and loudly. (It cost one full-suite red:
`test_v6_s2_loss.py::test_the_REAL_797_record_artifact_loads_with_the_
published_census`. The guard did its job; my file placement was the defect.)
⇒ v2 lives in `review/labels_v2/` with the CANONICAL filenames plus a copy of
`clip_index.json`, so the whole switch is ONE path change and the directory
loads cleanly on its own.

⚠️ ESCALATION, not a silent swap: whoever consumes these (the S2 loss label
loader, `--s2-labels`) must be REPOINTED at `review/labels_v2/`. That is a
decision, not a detail — see LANE_CHANGE_DEEP_REVIEW.md §7.

Provenance note: the v1 records do not persist the raw VLM symbols, but they
persist EXACTLY the two fields `s2_derive` reads (`corroboration.
vlm_goal_kind` and `corroboration.vlm_verbs`), so they are reconstructed from
each record's own corroboration block. Passing {} instead would silently
disable the ROUTE_TO gate and manufacture a delta the ruling did not cause —
MEASURED: it moved NONE_ABSTAIN by 12 records on the first attempt.
"""
import json
import os
import shutil
import sys
from collections import Counter

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
PKG = os.path.join(REPO, "TanitAD Research Hub", "Data Engineering",
                   "Implementation", "incoming", "2026-08-16-s2-v1-labels")
LABELS = os.path.join(PKG, "labels")
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "stack", "scripts"))
sys.path.insert(0, os.path.join(REPO, "colab"))
import s2_derive                                             # noqa: E402
import s2_schema                                             # noqa: E402

assert s2_derive.check_vocab_drift() == "checked"
assert s2_schema.check_v6_drift() == "checked"

RULING = ("PI 2026-08-16 §LC: geometric gate emits neither LANE_TARGET nor "
          "PREPARE_LANE_CHANGE; PREPARE_LANE_CHANGE is route-serving and "
          "context-derived. Lane context UNAVAILABLE -> requirement UNKNOWN.")

#: ⛔ NOT `labels/` — see the module docstring. Canonical names, own directory.
OUT_DIR = os.path.join(PKG, "review", "labels_v2")
os.makedirs(OUT_DIR, exist_ok=True)
PAIRS = (("s2_labels_aug120.jsonl", "engine_a_aug120.jsonl",
          "s2_labels_aug120.jsonl"),
         ("s2_labels_w120val.jsonl", "engine_a_w120val.jsonl",
          "s2_labels_w120val.jsonl"))

total, changed_all = 0, 0
for lab_f, ea_f, out_f in PAIRS:
    banked = {}
    for line in open(os.path.join(LABELS, lab_f), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            banked[r["clip_id"]] = r
    ea_by = {}
    for line in open(os.path.join(LABELS, ea_f), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            ea_by[r["clip_id"]] = r["engine_a"]

    out, changed, tok = [], 0, Counter()
    for cid, b in banked.items():
        ea = ea_by.get(cid)
        gc = (b.get("g_str") or {}).get("corroboration") or {}
        ac = (b.get("a_str") or {}).get("corroboration") or {}
        sym = {"goal_kind": gc.get("vlm_goal_kind"),
               "actions": [{"verb": v.get("verb"),
                            "direction": v.get("direction")}
                           for v in (ac.get("vlm_verbs") or [])]}
        g = s2_derive.derive_g_str(ea, sym)
        a = s2_derive.derive_a_str(ea, sym)
        rec = dict(b)
        if (b.get("g_str") or {}).get("token") != g["token"] or \
                (b.get("a_str") or {}).get("token") != a["token"]:
            changed += 1
        rec["g_str"], rec["a_str"] = g, a
        prov = dict(rec.get("_provenance") or {})
        prov["lane_change_ruling"] = RULING
        prov["supersedes"] = f"{lab_f} (v1, PI-adjudicated 2026-08-16)"
        rec["_provenance"] = prov
        s2_schema.validate(rec)                 # refuses rather than ships
        out.append(rec)
        tok[g["token"]] += 1

    out.sort(key=lambda r: r["clip_id"])
    dst = os.path.join(OUT_DIR, out_f)
    with open(dst, "w", encoding="utf-8", newline="\n") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    total += len(out)
    changed_all += changed
    print(f"{out_f}: {len(out)} records, {changed} changed vs v1")
    print(f"   g_str: {dict(sorted(tok.items()))}")
    assert tok.get("LANE_TARGET", 0) == 0, "LANE_TARGET must not be emitted"

#: the loader needs `clip_index.json` BESIDE the labels (s2_labels.py:267) —
#: without it `load_s2_labels` refuses rather than silently never firing.
shutil.copyfile(os.path.join(LABELS, "clip_index.json"),
                os.path.join(OUT_DIR, "clip_index.json"))

# ---- PROVE it loads through the REAL production loader, not just on disk ---
sys.path.insert(0, os.path.join(REPO, "stack", "scripts"))
from s2_labels import load_s2_labels                          # noqa: E402

ls = load_s2_labels(OUT_DIR)
cen = ls.token_census()
assert "LANE_TARGET" not in cen["g_str"], cen
assert "PREPARE_LANE_CHANGE" not in cen["a_str"], cen
assert len(ls) == total, (len(ls), total)
print(f"\nTOTAL {total} records, {changed_all} changed "
      f"({100*changed_all/total:.2f}%); all schema-validated; "
      f"v1 files untouched.")
print(f"load_s2_labels({os.path.basename(OUT_DIR)}) -> {len(ls)} records")
print(f"   g_str: {cen['g_str']}")
print(f"   a_str: {cen['a_str']}")
