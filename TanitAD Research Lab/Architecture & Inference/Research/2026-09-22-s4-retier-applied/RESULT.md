
> ### ✅ RULED BY THE PI, 2026-09-22: **C** — verbatim *"follow C"*. ITEM 19 IS CLOSED AND APPLIED.
>
> Applied the same day, **one atomic commit**, 166 operations (83 adds + 83 deletes), manifest
> updated and uploaded. VERIFIED FROM THE HUB, not from the tool's own success message:
>
> | | before | after | expected |
> |---|---|---|---|
> | `semantic_maps/gt/` | 4,569 | **4,652** | 4,569 + 83 ✅ |
> | `semantic_maps/gt_flagged/` | 108 | **25** | 108 − 83 ✅ |
> | `semantic_maps/worldmap/` | 4,677 | 4,677 | unchanged ✅ |
> | **total repo files** | 14,589 | **14,589** | **unchanged — it was a MOVE** ✅ |
>
> ⭐ The unchanged total is the load-bearing check: a copy would read 14,672, a botched delete
> fewer. The manifest agrees independently (`clips` 4,652 / `clips_flagged` 25), and
> `flagged_by_reason` now contains **only** *"path untestable (parked / stopped ego)": 25* — the
> *"near path unlabelled"* reason is gone entirely, which is the ruling taking effect.
>
> **(c) IS DISCHARGED.** `counts.s4_retier` publishes the withheld / unlabelled share beside the
> corpus numbers — min **0.100077** · median **0.137173** · mean **0.175081** · max **0.485261** ·
> **0** above 0.5 — and **all 83** entries carry a per-clip `retier` block with their own share.
>
> ⭐ **PROVENANCE SURVIVED, WHICH WAS THE POINT.** Each re-tiered clip KEEPS its `flag`,
> `failed_checks` and `path_classes` and gains `retier` naming the rule, your ruling, the layout,
> its compliant share (1.0) and its previous path. **A clip judged by rule (a) is still
> distinguishable from one that passed outright** — the corpus did not silently lose which rule
> judged it.
>
> ⚠️ Nothing was destroyed: the same bytes are in `gt/` from the same commit, the local sources
> remain in `corpus/out/`, each clip's `worldmap/` copy is untouched, and the pre-pass manifest is
> snapshotted at `publish/SEMANTIC_MAPS_MANIFEST.pre-s4.json` (md5 `705afde1fcb57f608be71ffe22440c86`).
