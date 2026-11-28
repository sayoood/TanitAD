#!/usr/bin/env python3
"""W2 (E3): CRITERIA_REGISTRY.json 2.10.0 -> 2.10.1 — the modality correction W4 flagged,
VERIFIED HERE against the banked primary, plus the qualifiers three published numbers need.

⛔ WHAT WAS WRONG. `GATE_modality_label.why` said *"Drive-JEPA's 93.7 PDMS headline uses
1024x256 stacking FRONT+LEFT+RIGHT, while its perception-free row uses the front camera
only at 512x256"*. The primary says the opposite, and I read it myself rather than taking
the correction on trust — `TanitAD Research Lab/Library/papers/2601.22032_…pdf`, sha256
`d88c053a67aca11b7bbd4c5f9f000cbf95c3f5e94656b6bad89cc9a717ab3672` (== `library.json`):

* **Table 8, p.14** ("Input image resolution"): Transfuser **1024×256** · HydraMDP++
  1024×256 · DriveSuprim 1024×256 · GoalFlow 1024×256 · iPad **4×768×432** · **Ours
  2×512×256**; the surrounding text: *"HydraMDP++, DriveSuprim, and GoalFlow follow
  Transfuser in using an input resolution of 1024×256, formed by stacking the front, left,
  and right camera images … Our setting uses only the front camera at 512×256. We include
  both I_t and I_t−1, resulting in an input tensor of 2×512×256."*
* **§4.2, p.7**: *"We use only the front-view camera, resized to 512×256."*
⇒ the three-camera 1024×256 stack is **Transfuser's** (and of the methods that follow it);
Drive-JEPA is front-camera-only in EVERY row, headline included.

And the ladder key's own name was wrong for the same reason in the other direction —
**Table 2, p.8** prints an `Inputs` column: LAW **C & L**, World4Drive **C & L**, Epona
**Camera**, Ours **Camera**. So "perception-free" is not "front-camera-only": two of the
four rungs take LiDAR. Epona also reads **86.1** in Table 1 (p.4) and **86.2** in Table 2.

Re-runnable ONLY on 2.10.0; preserves the file's exact serialisation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
REG = REPO / "products" / "P7-TanitEval" / "CRITERIA_REGISTRY.json"
W4 = ("FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-leaderboard-currency/RESULT.md")
PDF = ("TanitAD Research Lab/Library/papers/"
       "2601.22032_Drive-JEPA-Video-JEPA-Meets-Multimodal-Trajectory-Distillati.pdf")
SHA = "d88c053a67aca11b7bbd4c5f9f000cbf95c3f5e94656b6bad89cc9a717ab3672"


def dump(d) -> bytes:
    return json.dumps(d, indent=1, ensure_ascii=False).replace("\n", "\r\n").encode("utf-8")


def main() -> int:
    raw = REG.read_bytes()
    d = json.loads(raw.decode("utf-8"))
    if dump(d) != raw:
        print("REFUSED: the registry does not round-trip byte-exactly")
        return 2
    if d.get("version") != "2.10.0":
        print(f"REFUSED: expected 2.10.0, found {d.get('version')!r}")
        return 2
    nv = d["benchmarks"]["navsim"]
    g = nv["GATE_modality_label"]

    g["why"] = (
        "The leaderboard server records NO modality — only TEAM_NAME, AUTHORS, EMAIL, INSTITUTION, "
        "COUNTRY — so every 'camera-only' claim in the field is the authors', not the server's, and "
        "the INPUTS behind two rows can differ by a factor of four. MEASURED 2026-09-20 from the "
        f"banked primary ({PDF}, sha256 {SHA[:12]}…, == library.json), Drive-JEPA Table 8 p.14 "
        "'Input image resolution': Transfuser, HydraMDP++, DriveSuprim and GoalFlow run at "
        "1024x256 'formed by stacking the front, left, and right camera images'; iPad at 4x768x432 "
        "(four views incl. back); Drive-JEPA at 2x512x256 — the FRONT CAMERA ONLY, two consecutive "
        "frames (also §4.2 p.7: 'We use only the front-view camera, resized to 512x256'). ⛔ "
        "CORRECTED 2026-09-20 (flagged by W4, re-read from the PDF by W2): until registry 2.10.1 "
        "this text attributed the three-camera 1024x256 stack to DRIVE-JEPA's headline — it is "
        "TRANSFUSER's, and Drive-JEPA is front-camera-only in every row, headline included. "
        "Comparing our single 256x640 front-camera arm against a three- or four-camera row is a "
        "category error in whichever direction it flatters; the label is what makes it visible.")
    g["why_correction_note"] = (
        "⚠️ The superseded sentence may still be quoted in EvalFlyWheel packages written before "
        "2026-09-20 (MEASURED: it is in "
        "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-leaderboard-currency/raw/"
        "external_field_sources.json, and it was written into the W1/E2 briefs). The registry is "
        "the source of truth; those copies are stale.")

    lad = g.pop("comparable_ladder_perception_free_front_only")
    lad["_source"] = ("PUB-PAPER Drive-JEPA (arXiv 2601.22032) **Table 2, p.8** (the first block is "
                      "the perception-free setting), re-read from the banked PDF by W2 2026-09-20; "
                      "sha256 " + SHA[:12] + "…")
    lad["_inputs_column"] = ("⛔ THE LADDER IS NOT 'FRONT-CAMERA-ONLY' — Table 2 prints an `Inputs` "
                             "column: LAW 'C & L' and World4Drive 'C & L' (camera AND LiDAR), Epona "
                             "'Camera', Drive-JEPA 'Camera'. Only the top two rungs are camera-only. "
                             "Key renamed from comparable_ladder_perception_free_front_only in 2.10.1.")
    lad["LAW"].update({"inputs": "C & L", "backbone": "resnet34"})
    lad["World4Drive"].update({"inputs": "C & L", "backbone": "resnet34"})
    lad["Epona"].update({"inputs": "Camera", "backbone": "ViT/G",
                         "PDMS_note": "86.2 in Table 2 p.8; the SAME paper prints 86.1 in Table 1 "
                                      "p.4 — an internal inconsistency, quote the table"})
    lad["Drive-JEPA"].update({"inputs": "Camera (2x512x256, front only)", "backbone": "ViT/L"})
    lad["_note"] = ("TanitAD's comparable class is PERCEPTION-FREE (supervised solely by human "
                    "trajectories, no perception annotations) at our scale band (entry rung 21M, top "
                    "rung 307M). ⚠️ It is NOT a modality-matched class: LAW and World4Drive take "
                    "LiDAR. The only camera-only rungs are Epona (ViT/G, 1.1B) and Drive-JEPA.")
    g["comparable_ladder_perception_free"] = lad
    g["comparable_ladder_perception_free_front_only"] = (
        "RENAMED 2026-09-20 -> `comparable_ladder_perception_free`. The old name asserted a "
        "modality the primary contradicts (LAW and World4Drive print 'C & L' in Drive-JEPA Table 2). "
        "A consumer that indexes this key now fails loudly instead of quoting a wrong class.")

    pr = nv["published_reference_numbers"]
    pr["drive_jepa_perception_free_front_only_PDMS"] = 89.0
    pr["drive_jepa_perception_free_PDMS_NOTE"] = (
        "89.0 is Drive-JEPA ViT/L, Camera-only, in Table 2's perception-free block (p.8). The block's "
        "other rungs are NOT camera-only (LAW / World4Drive: 'C & L'). The key name "
        "'…_front_only' is kept for continuity; read this note with it.")
    pr["_qualifiers_that_travel_with_each_number"] = {
        "_evidence": ("INHERITED from W4's currency audit (" + W4 + " §2 findings 1-4), which "
                      "verified each against a banked primary. ⛔ Not re-read by W2 — W4's "
                      "products/P7-TanitEval/benchmarks/published_results.json is the primary "
                      "record; a number quoted from here must carry its qualifier."),
        "PDM-Closed navhard 51.3 vs 56.6": "51.3 is PRE-#151 (the human-filter fix); 56.6 is post-fix. "
                                           "The two are not one number.",
        "DrivoR navhard 56.3": "= DrivoR + 134k SimScale SYNTHETIC training samples + TOAD test-time "
                               "search. Navtrain-only DrivoR is 48.3.",
        "Drive-JEPA 93.3 vs 93.7": "93.7 is the abstract and §1; 93.3 appears only in the paper's own "
                                   "NeurIPS checklist, misquoting its abstract.",
        "navtest ladder backbones": "Transfuser 76.7 and Hydra-MDP++ 81.4 are ResNet34 rows; "
                                    "DriveSuprim 87.1 and Drive-JEPA 87.8 are ViT/L; DriveSuprim at "
                                    "ResNet34 reads 83.1.",
    }
    oc = nv["OFFICIAL_COLUMN"]
    oc["consequence_for_our_positioning"] += (
        " ⚠️ QUALIFIED 2026-09-20 (INHERITED from W4's audit): 'DrivoR 56.3' is DrivoR + 134k "
        "synthetic + TOAD test-time search (navtrain-only DrivoR 48.3), and the 56.6 privileged "
        "ceiling is PDM-Closed POST-#151 (its pre-fix value is 51.3). Quote the qualifier or the "
        "number is three things at once.")

    d["protocol_tags"]["external_only_tags"] = (
        "Tags that exist only to label CITED external rows (e.g. `Bench2Drive_closed_loop` in W4's "
        "published_results.json) are NOT protocol tags for our artifacts and are deliberately OUTSIDE "
        "this union: nothing we produce may carry one.")

    d["version"] = "2.10.1"
    d["changelog"].append({
        "version": "2.10.1", "date": "2026-09-20", "by": "EvalFlyWheel W2 (E3 resumed)",
        "change": ("GATE_modality_label.why CORRECTED — the three-camera 1024x256 stack is "
                   "TRANSFUSER's, not Drive-JEPA's (which is front-camera-only at 2x512x256 in every "
                   "row); `comparable_ladder_perception_free_front_only` renamed to "
                   "`comparable_ladder_perception_free` with each rung's printed Inputs (LAW and "
                   "World4Drive are 'C & L'), Epona's 86.1/86.2 self-inconsistency recorded; "
                   "published_reference_numbers gains the qualifiers PDM-Closed 51.3 (pre-#151) / "
                   "DrivoR 56.3 (+134k synthetic + TOAD) / the navtest backbone mix; protocol_tags "
                   "records that external-only cited tags stay outside the union."),
        "why": ("W4 flagged the modality sentence; W2 owns the registry and re-read the banked "
                "primary rather than taking a correction on trust (Drive-JEPA 2601.22032, sha256 "
                + SHA[:12] + "…: Table 8 p.14 and §4.2 p.7 for the resolution, Table 2 p.8 for the "
                "Inputs column). A wrong modality claim in a GATE about modality labels is the "
                "worst place for one."),
        "ships_with": ("tools/tests/test_criteria_check.py — the ladder/modality pins rewritten to "
                       "the verified literals, with a regression arm that fails if the Drive-JEPA "
                       "attribution returns.")})
    out = dump(d)
    REG.write_bytes(out)
    print(f"wrote {REG} version {d['version']} ({len(raw)} -> {len(out)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
