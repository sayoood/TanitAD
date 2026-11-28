"""S2 labels v3 — DECLINE the 80 a_str rows that v2 silently REASSIGNED.

THE DEFECT (documented by its finder, verified by content here)
---------------------------------------------------------------
`labels/SUPERSEDED.json` records it precisely:

    "a_str has NO abstain token (STRATEGIC_ACTION_TOKENS is six positive
     manoeuvres), so the 80 removed PREPARE_LANE_CHANGE rows were REASSIGNED
     to 71 HOLD_CORRIDOR + 9 REDUCE_TO rather than declined. ... until it does,
     those 80 a_str targets are UNVERIFIED, not corrected."

MEASURED here, by joining v1 and v2 on ``clip_id`` (not by trusting the note):
aug120 `PREPARE_LANE_CHANGE` 19 -> 0 with `HOLD_CORRIDOR` +18 / `REDUCE_TO` +1;
`a_str` abstain flags in v2: **0 of 797**. The remedy the finder named has
never been emitted.

WHY THESE ROWS AND NOT ALL OF THEM — the question that decides correctness
--------------------------------------------------------------------------
`lane_change_requirement()` returns None (UNKNOWN) for 801/801 clips, so one
could argue every row is underdetermined. It is not:

* On the other 717 rows the geometry showed **no lateral-displacement event at
  all**, and v1 and v2 AGREE. `HOLD_CORRIDOR` there is positively supported.
* On these 80 the refuted gate **fired** — there IS lateral displacement, and
  the PI's adjudication is that lateral displacement cannot distinguish a lane
  change from road curvature *in either direction*. Labelling them
  `HOLD_CORRIDOR` asserts "the ego did NOT change lanes", which is exactly the
  claim the refuted gate cannot settle. That is a manufactured confident claim,
  and it is the one an abstain exists for.

⛔ This does NOT re-derive, re-select, or re-order anything. g_str is untouched.
Only the a_str family of exactly the joined rows becomes an abstain, and the
identity of those rows is read off the v1/v2 diff, never re-inferred.

⚠️ INTEGRATION CONSEQUENCE, stated because it is not local: once ANY record
abstains, `S2WindowSupervision.emits_family_masks` flips True
(`s2_labels.py:752-754`) and the loader begins emitting family masks. A
consumer that ignored masks was previously safe by accident. Flagged to the PI;
v2 is left in place and nothing is repointed here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V1 = Path("C:/Users/Admin/tanitad-wt/_s2labels")
V2 = Path("C:/Users/Admin/tanitad-wt/_s2review/labels_v2")
V3 = Path("C:/Users/Admin/tanitad-wt/_s2build/labels_v3")
SPLITS = ("aug120", "w120val")
GATE_TOKEN = "PREPARE_LANE_CHANGE"


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def main() -> int:
    V3.mkdir(parents=True, exist_ok=True)
    report: dict = {"probe": "S2-ABSTAIN-V3", "evidence_class": "MEASURED",
                    "gate_token": GATE_TOKEN, "splits": {}}
    total_abstained = 0

    for split in SPLITS:
        v1 = {r["clip_id"]: r for r in read(V1 / f"s2_labels_{split}.jsonl")}
        v2_rows = read(V2 / f"s2_labels_{split}.jsonl")

        # the joined set: v1's gate fired, v2 assigned something else
        targets, reassigned_to = [], {}
        for r in v2_rows:
            cid = r["clip_id"]
            old = v1.get(cid, {}).get("a_str", {}).get("token")
            new = r.get("a_str", {}).get("token")
            if old == GATE_TOKEN and new != GATE_TOKEN:
                targets.append(cid)
                reassigned_to[new] = reassigned_to.get(new, 0) + 1

        tset = set(targets)
        out = []
        for r in v2_rows:
            if r["clip_id"] in tset:
                r = dict(r)
                prior = r["a_str"]
                # an abstaining block carries NO token, NO args, NO arg_mask —
                # s2_labels._check_block REFUSES any of them beside abstain.
                r["a_str"] = {
                    "abstain": True,
                    "provenance": prior.get("provenance", "path"),
                    "sources": prior.get("sources", []),
                    "reason": (
                        "REFUTED_LANE_CHANGE_GATE: v1 derived "
                        f"{GATE_TOKEN} here from the lateral-displacement gate "
                        "the PI ruled out 2026-08-16 (~78% wrong). v2 removed "
                        f"the gate and defaulted this row to "
                        f"{prior.get('token')!r}, which asserts the ego did NOT "
                        "change lanes — a claim the same refuted geometry "
                        "cannot settle in either direction. Declined, not "
                        "guessed."),
                    "superseded_token_v1": GATE_TOKEN,
                    "superseded_token_v2": prior.get("token"),
                }
                out.append(r)
            else:
                out.append(r)

        (V3 / f"s2_labels_{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
            encoding="utf-8")

        report["splits"][split] = {
            "n_records": len(v2_rows),
            "n_abstained": len(targets),
            "v2_reassigned_to": reassigned_to,
            "clip_ids": sorted(targets),
        }
        total_abstained += len(targets)
        print(f"[s2-v3] {split}: {len(targets)} of {len(v2_rows)} declined "
              f"(v2 had put them in {reassigned_to})")

    # carry the clip index across unchanged
    src_idx = V2 / "clip_index.json"
    if src_idx.exists():
        (V3 / "clip_index.json").write_text(
            src_idx.read_text(encoding="utf-8"), encoding="utf-8")

    report["total_abstained"] = total_abstained
    report["SUPERSEDED_json_claimed"] = 80
    report["matches_documented_count"] = (total_abstained == 80)
    (V3 / "_abstain_provenance.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[s2-v3] total declined: {total_abstained} "
          f"(SUPERSEDED.json documented 80 -> "
          f"{'MATCH' if total_abstained == 80 else 'MISMATCH'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
