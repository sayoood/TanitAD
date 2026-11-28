"""Which manoeuvres does the band structure claim, and which does it not?

MM-E15 asked whether manoeuvres starting past 30 s are deliberately untracked.
Answer: yes — see `emit_one`'s `max_s = RAW_T0_S + LOOKAHEAD_S + 5.0`.

⛔⛔ THIS SCRIPT NOW REPORTS PER SPLIT, AND THAT IS THE WHOLE POINT OF v2.
My first census reported train+eval (3,580 manoeuvres, 54 unassigned) while
MM-E15 reported train-only (3,464, 53). Neither was wrong and both were right
about what they measured — but the note I wrote from these numbers carries the
standing instruction *"any count must name which convention it used"* and did
not name its own CORPUS SCOPE. An instrument that breaks its own rule is the
defect class this investigation has been about all day, one level down.

⭐ IT ALSO EMITS `band_census.json`, WHICH THE MANIFEST PATCH CONSUMES. Numbers
typed by hand into a document are how the scope got lost; numbers carried from
the measurement cannot lose it.

⚠️ WHAT THIS CANNOT REPRODUCE: the emitter assigns a manoeuvre to a band only if
the CLIPPED |dyaw| over that band reaches MIN_TURN_DEG (20 deg), and that needs
the pose track, which is not in the label. So this measures BAND OVERLAP — the
necessary condition, and the one a consumer would apply. Overlap is an UPPER
bound on what the emitter claims; a manoeuvre with no overlap is claimed by
nothing, definitively.
"""
import gzip
import json
from collections import Counter
from pathlib import Path

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release")
BANDS = {"operative": (0.0, 2.0), "tactical": (2.0, 6.0),
         "gap": (6.0, 8.0), "strategic": (8.0, 30.0)}
ASSIGNABLE = ("tactical", "gap", "strategic")   # split_by_band() returns only these
SPLITS = {"train": "s2_labels_v7.2_train.jsonl.gz",
          "eval": "s2_labels_v7.2_eval.jsonl.gz"}


def overlaps(m, band):
    return min(m["t_end_s"], band[1]) > max(m["t_start_s"], band[0])


def start_band(t):
    return ("[0,2) operative" if t < 2 else "[2,6) tactical" if t < 6 else
            "[6,8) gap" if t < 8 else "[8,30] strategic" if t <= 30 else
            ">30 out of horizon")


def census(rows):
    starts, overlap = Counter(), Counter()
    n_man = n_unassigned = n_op_only = n_edge30 = 0
    for r in rows:
        seq = r.get("manoeuvre_sequence") or []
        n_man += len(seq)
        n_unassigned += len((r.get("bands") or {}).get("unassigned_manoeuvres") or [])
        for m in seq:
            starts[start_band(m["t_start_s"])] += 1
            if m["t_start_s"] == 30.0:
                n_edge30 += 1
            hits = [n for n, b in BANDS.items() if overlaps(m, b)]
            assign = [h for h in hits if h in ASSIGNABLE]
            if assign:
                overlap["+".join(sorted(assign, key=list(BANDS).index))] += 1
            else:
                overlap["no assignable band"] += 1
                if hits == ["operative"]:
                    n_op_only += 1
    return {"n_records": len(rows), "n_manoeuvres": n_man,
            "unassigned_manoeuvres_entries": n_unassigned,
            "starts_by_band": dict(starts), "overlap_by_band": dict(overlap),
            "overlap_operative_only": n_op_only, "t_start_exactly_30_0": n_edge30}


per = {}
allrows = []
for split, name in SPLITS.items():
    rows = [json.loads(x) for x in
            gzip.open(REL / "v72" / name, "rt", encoding="utf-8") if x.strip()]
    per[split] = census(rows)
    allrows += rows
per["train+eval"] = census(allrows)

# ⭐ the split must ACCOUNT for the combined figure, or one of them is wrong
for k in ("n_records", "n_manoeuvres", "unassigned_manoeuvres_entries",
          "overlap_operative_only", "t_start_exactly_30_0"):
    assert per["train"][k] + per["eval"][k] == per["train+eval"][k], \
        "train + eval != combined on %s — the scopes do not reconcile" % k

out = REL / "band_census.json"
out.write_text(json.dumps({"_scope_warning": (
    "EVERY number here is stamped with its corpus scope. train-only and "
    "train+eval differ and BOTH are correct; a count that does not name its "
    "scope is unreadable."), "by_scope": per}, indent=1), encoding="utf-8")

for scope in ("train", "eval", "train+eval"):
    c = per[scope]
    print("=== %s ===  %d records | %d manoeuvres | unassigned %d"
          % (scope.upper(), c["n_records"], c["n_manoeuvres"],
             c["unassigned_manoeuvres_entries"]))
    o30 = c["starts_by_band"].get(">30 out of horizon", 0)
    print("    starts>30s %d (%.1f%%) | overlap-none %d | operative-only %d | t=30.0 %d"
          % (o30, 100 * o30 / c["n_manoeuvres"],
             c["overlap_by_band"].get("no assignable band", 0),
             c["overlap_operative_only"], c["t_start_exactly_30_0"]))

t, e = per["train"], per["eval"]
print("\n⚠️ THE EVAL SIDE IS PROPORTIONALLY MORE OVER-HORIZON:")
print("    train %d/%d = %.1f%%   eval %d/%d = %.1f%%"
      % (t["starts_by_band"].get(">30 out of horizon", 0), t["n_manoeuvres"],
         100 * t["starts_by_band"].get(">30 out of horizon", 0) / t["n_manoeuvres"],
         e["starts_by_band"].get(">30 out of horizon", 0), e["n_manoeuvres"],
         100 * e["starts_by_band"].get(">30 out of horizon", 0) / e["n_manoeuvres"]))
print("\nwrote %s  (consumed by patch_v72_manifest_bands.py)" % out.name)
