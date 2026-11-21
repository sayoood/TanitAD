"""The §4 re-tiering pass — ONE pass, after production completes. DRY-RUN BY DEFAULT.

⛔ THE RULING IT IMPLEMENTS. PI, 2026-09-20, verbatim: *"a) for the term, (c) for the reporting"*.
⇒ a cell that is **seen but carries no class** is **COMPLIANT** (it moves into the numerator), and
the **withheld / unlabelled share is published beside every DAC / EP / PDMS number**.

⛔ WHY IT IS ONE PASS AFTERWARDS, NOT A LIVE RULE CHANGE. The rule is a numerator change and is
re-derivable from banked data with NO re-inference; changing it mid-run would split the corpus into
two populations judged by two rules. So `--apply` REFUSES until the DONE marker exists.

⭐ PREMISE, ALREADY VERIFIED ON LIVE DATA (`b513a26`, 77/77 at 95.19 % of the run) AND RE-ASSERTED
HERE PER CLIP: every clip carrying the flag *"near path unlabelled (seen, no class), no path cell
on a non-drivable class"* has **zero** NON_DRIVABLE path cells, so its compliant share under (a) is
**exactly 1.0** and it clears the 0.9 threshold outright, not marginally. ⛔ A clip that fails that
assertion is REFUSED, never re-tiered — the flag's own definition would have been violated.

⛔ THE LAYOUT OPTION IS THE PI'S (queue item 19), because re-tiering moves a clip between
`semantic_maps/gt_flagged/` and `semantic_maps/gt/` — a change to a PUBLISHED artifact's layout:

  A  manifest-only; bytes stay in gt_flagged/    0 MB, and `gt/` under-delivers vs the manifest
  B  add to gt/, KEEP the flagged copy           ~150 MB, nothing irreversible
  C  add to gt/, DELETE from gt_flagged/         0 net, the only self-consistent option

⛔ NONE of them is the default — see the correction below.

⛔⛔ CORRECTED 2026-09-22, BEFORE ANY APPLY: "B BY DEFAULT IF THE PI IS SILENT" WAS MY OWN
STATEMENT, AND MY OWN STATEMENT IS NOT HIS APPROVAL. A/B/C was put to him as queue item 19; acting
on my own default would be TAKING that decision, not covering for its absence. And B is not
actually free -- it commits the repo to an untidy layout that only C (his call) resolves.

⭐ SO THE DEFAULT IS NARROWED TO WHAT THE RULING ITSELF AUTHORISES: the JUDGEMENT, recorded in
the manifest, with NO bytes moved. `--layout` still defaults to `pending`, and every re-tiered entry
then carries `layout_open: true` naming queue item 19 and the path its bytes are ACTUALLY at.
⚠️ That is NOT option A. A's hazard was SILENCE -- a file under `gt_flagged/` that the record
calls validated, with nothing saying so. A documented interim state is not a silent inconsistency:
a reader in isolation is told exactly what is true, which is the whole point of the rule that
artifact hazard came from.
⛔ `--layout B` and `--layout C` move bytes and BOTH require `--i-have-the-pi-ruling <X>`.

⭐ PROVENANCE IS NOT OPTIONAL: a re-tiered clip keeps its `flag`, `failed_checks` and
`path_classes` and GAINS a `retier` block. A clip judged by rule (a) must stay distinguishable from
one that passed outright, or the corpus silently loses which rule judged it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CORPUS = pathlib.Path("/home/nvidia/sam3map/corpus")
MANIFEST = CORPUS / "publish" / "SEMANTIC_MAPS_MANIFEST.json"
OUT = CORPUS / "out"
DONE = CORPUS / "DONE"
FLAG = "near path unlabelled (seen, no class), no path cell on a non-drivable class"
ROAD, NON_DRIVABLE, UNLABELLED = {1, 2, 3, 4, 6}, {5, 7}, 0
RULING = "PI 2026-09-20: 'a) for the term, (c) for the reporting'"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="execute; without it this prints the plan and touches nothing")
    ap.add_argument("--layout", choices=["pending", "A", "B", "C"], default="pending",
                    help="what happens to the BYTES. `pending` (default) moves none and records "
                         "the judgement only; A/B/C are the PI's queue item 19 and each needs "
                         "--i-have-the-pi-ruling")
    ap.add_argument("--i-have-the-pi-ruling", default="",
                    help="must equal --layout for any byte-moving option; C additionally DELETES "
                         "published files")
    ap.add_argument("--out", default="/home/nvidia/sam3map/corpus/publish/s4_retier_plan.json")
    a = ap.parse_args()

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cf = man["clips_flagged"]
    items = list(cf.items()) if isinstance(cf, dict) else [(e.get("sha12"), e) for e in cf]

    sel, refused = [], []
    for key, e in items:
        if e.get("flag") != FLAG:
            continue
        pc = e.get("path_classes") or {}
        tot = sum(pc.values())
        nd = sum(v for k, v in pc.items() if int(k) in NON_DRIVABLE)
        unl = sum(v for k, v in pc.items() if int(k) == UNLABELLED)
        road = sum(v for k, v in pc.items() if int(k) in ROAD)
        # ⛔ THE PREMISE, RE-ASSERTED PER CLIP. `nd == 0` is what makes the compliant share
        # exactly 1.0; a clip that fails it carries the flag wrongly and must NOT be re-tiered.
        if not tot or nd != 0 or (tot - nd - road - unl) != 0:
            refused.append({"sha12": e.get("sha12"), "tot": tot, "nd": nd,
                            "unaccounted": tot - nd - road - unl,
                            "why": ("no seen path cells" if not tot else
                                    "a NON_DRIVABLE path cell exists -- the flag's own definition "
                                    "is violated" if nd else
                                    "a path cell carries a code outside {0} u ROAD u NON_DRIVABLE")})
            continue
        sel.append({"key": key, "entry": e, "tot": tot,
                    "withheld_unlabelled_share": round(unl / tot, 6)})

    shares = sorted(s["withheld_unlabelled_share"] for s in sel)
    plan = {
        "_what": "the S4 re-tiering pass", "_ruling": RULING, "_layout": a.layout,
        "_layout_open": a.layout == "pending",
        "_applied": False,
        "production_complete": DONE.exists(),
        "n_flagged_total": len(items),
        "n_selected": len(sel), "n_refused": len(refused), "refused": refused,
        "compliant_share_under_a": 1.0,
        "_compliant_share_is_exact": ("every selected clip has ZERO NON_DRIVABLE path cells, so "
                                      "(road + unlabelled) / tot == 1.0 identically -- not an "
                                      "estimate"),
        "c_withheld_unlabelled_share": ({
            "min": shares[0], "max": shares[-1],
            "mean": round(sum(shares) / len(shares), 6),
            "median": shares[len(shares) // 2],
            "n_over_0.5": sum(1 for s in shares if s > 0.5),
            "_ruling": "(c) requires this be published BESIDE every DAC/EP/PDMS number"}
            if shares else None),
        "clips": [{"sha12": s["entry"].get("sha12"),
                   "withheld_unlabelled_share": s["withheld_unlabelled_share"]} for s in sel],
    }

    if a.layout != "pending" and a.i_have_the_pi_ruling != a.layout:
        print(json.dumps(plan, indent=1, ensure_ascii=False))
        print("\n⛔ REFUSED: --layout %s MOVES BYTES in a published dataset repo%s. The §4 ruling "
              "authorises a RE-TIER (a judgement), not a layout change; A/B/C is PI queue item 19. "
              "Pass --i-have-the-pi-ruling %s only with the PI's word."
              % (a.layout, " and DELETES published files" if a.layout == "C" else "", a.layout))
        return 4

    if not a.apply:
        pathlib.Path(a.out).write_text(json.dumps(plan, indent=1, ensure_ascii=False),
                                       encoding="utf-8")
        print(json.dumps(plan, indent=1, ensure_ascii=False))
        print(f"\n⭐ DRY RUN — nothing written to Thor or HF. Plan banked at {a.out}. "
              f"{len(sel)} clips would re-tier under layout {a.layout}; {len(refused)} refused.")
        return 0

    # ---- apply ----------------------------------------------------------- #
    if not DONE.exists():
        print("\n⛔ REFUSED: no DONE marker. The prereg applies this rule as ONE pass AFTER "
              "production completes, so every clip is judged by the same rule. Applying now "
              "would split the corpus into two populations judged by two rules.")
        return 5
    print("\n⛔ apply is not wired in this revision: the HF commit half lands with the PI's "
          "option (queue item 19) so the operations are written once, for the chosen layout.")
    return 6


if __name__ == "__main__":
    sys.exit(main())
