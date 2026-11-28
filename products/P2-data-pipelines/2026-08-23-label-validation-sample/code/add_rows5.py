"""Insert the iteration-5 claim rows into GOALS_AND_CLAIMS.md."""
from pathlib import Path

P = Path("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/"
         "Project Steering/GOALS_AND_CLAIMS.md")
ANCHOR = "\n## Standing rules with their origin"

ROWS = [
    "| C140 | ⛔⛔ RETRACTED: the label↔frame↔pose join via `episode_id_legacy` "
    "— **WRONG ON 20.5 % OF CLIPS** | **RETRACTED — structurally fixed** | 8/39 "
    "joins resolved to the WRONG EPISODE (r = −0.96…+0.87, rmse 2.6–18.4 m/s vs "
    "r > 0.999 for the 31 correct). The mismatched set included every clip the "
    "previous frame-by-frame validation had 'confirmed' ⇒ that validation "
    "inspected other clips' frames. Fix: the pipeline keys on CLIP UUID "
    "everywhere (`egomotion_source.py`). `RETRACTION_LOG.md` C140 |",

    "| C141 | ⚠️ RETRACTED: \"81 % of the strategic horizon is past the end of "
    "the clip\" — escalated TWICE as a programme blocker | **RETRACTED — the "
    "problem does not exist** | That measured the 20 s EPISODE CACHE, not the "
    "corpus. Provider egomotion runs 20–140 s: median horizon **37.0 s**, "
    "strategic band **median 22.0/22.0 s = 100 %**, **757/801 = 94.5 %** carry "
    "the FULL band. No definition to narrow, no data to source. "
    "`RETRACTION_LOG.md` C141 |",

    "| D-DATA-EMIT | ⭐ The label pipeline is REBUILT and the corpus RE-EMITTED "
    "from ego geometry, keyed by clip UUID | **MEASURED — 801/801, 0 failures** "
    "| `stack/scripts/s2_geom_emit.py`. Self-consistency against `label_guard`: "
    "**REFUSE 61 (7.7 %) → 0 (0.00 %)**, CLEAN 89.5 % → **98.1 %**; "
    "`G1-fallback-absorbs-turn` now ZERO. 154/797 labels changed vs the shipped "
    "set (75 FOLLOW_MAIN_ROAD hiding a manoeuvre, 36 bends miscalled turns, all "
    "14 NONE_ABSTAIN and all 80 action-abstains resolved). "
    "`…/raw/labels_geom/s2_labels_geom.jsonl` |",

    "| D-DATA-EVADE | ⛔ `EVADE_IN_CORRIDOR` is NOT derivable from ego geometry "
    "on this corpus | **MEASURED — now emits 0** | An evasion is an out-and-back "
    "returning to the original heading; **0 of 40 candidate emissions showed any "
    "return (100 % monotonic lane shifts)**, and 79/132 were below 1.0 m (min "
    "0.01 m). With the return + net-yaw gates the token emits **zero** — the "
    "honest answer. It needs perception, not geometry |",

    "| D-DATA-VALID5 | Validation of the re-emitted labels on JOIN-FREE frames "
    "(mp4 filename = UUID) | **MEASURED** | 8 clips inspected with frames that "
    "cannot be mis-joined — `0e56dae2` TURN_LEFT ✓, `2cf5d4c8` TURN_RIGHT ✓, "
    "`416601c0` TURN_LEFT ✓, `1ad7bf7b` + `e850f1fb` FOLLOW_MAIN_ROAD ✓, "
    "`3a0165bd` composite (defensible), plus `82b8780b`/`e084c7c3` on "
    "verified-correct joins. **Zero label errors found.** ⚠️ 8 of 801 is a "
    "coverage statement, not a clean bill of health |",
]


def main() -> None:
    s = P.read_text(encoding="utf-8")
    assert ANCHOR in s, "anchor section missing"
    s = s.replace(ANCHOR, "\n".join(ROWS) + "\n" + ANCHOR, 1)
    P.write_text(s, encoding="utf-8")
    print(f"inserted {len(ROWS)} register rows")


if __name__ == "__main__":
    main()
