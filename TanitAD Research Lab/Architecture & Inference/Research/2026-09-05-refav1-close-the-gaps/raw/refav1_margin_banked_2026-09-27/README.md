# refav1 p4-panel planner arms — banked from the dev box on 2026-09-27

**What this is.** The raw outputs of refav1's zero-training planner arms run on the 40-window /
8-episode p4 panel on 2026-09-05/06 (cost-geometry, longitudinal and close-the-gaps queues): per-arm
records (`p4out/rec_<arm>.json`), run logs (`p4out/<arm>.log`), per-episode window dumps
(`p4out/dump_<arm>/`) and the panel analysis (`analysis/`).

**Why it is here.** Until today these existed ONLY at `C:/Users/Admin/refav1_margin/` on the dev box.
278 of the 599 source files were already in the repo (matched by md5 against tip `b3f7ea6f`, in four
sibling packages: `2026-09-05-refav1-cost-geometry/raw/arms/`, `2026-09-05-refav1-longitudinal/raw/arms/`,
`2026-09-05-refav1-goal-margin/raw/` and `2026-09-05-refav1-lonshift-t1/raw/devbox/`; the exact match for
each file is its `at` field in `manifest.json`) and are NOT duplicated here. The other **305** are banked here, including the 17 arm records that had
never been read back into the programme (among them the pre-registered `kammshift`, `loncomb3`,
`lonshift_s1`, `seambase`/`seamon`, `gkappa`, `wk1`/`wk3`/`wk7`, `wk15_ladder` and the
turn-asymmetry `ta_*` seed pairs).

**Two transformations, both recorded in `manifest.json`:**
- ⛔ **Clip ids are sha12.** Every full UUID in a text file was rewritten to `sha256(uuid)[:12]`
  (556 occurrences), including the file paths inside `manifest.json`. `md5_original` (source bytes)
  and `md5_banked` (these bytes) are both recorded per file. To join against a sibling file that still
  carries raw ids, hash its ids the same way.
- **16 input files were deliberately NOT banked:** `p4/eps/*.v2ep.pt` and `p4/fp8/*.pt`, the panel's
  per-episode frame cache and DINOv3 fp8 features. They are corpus slices named by clip id,
  regenerable from the v2ep and `refav1-fp8-*` caches, and do not belong in the repo.

**Status of the numbers.** NOT yet read into any RESULT or register row. Reading them — paired
four-family deltas against `ha`, `ha0`, `ha0_ext` with the episode-cluster bootstrap, each against its
own inference-seed floor, and `kammshift` against its pre-registered bar in
`../../SPEC_CLOSE_THE_GAPS.md` — is the next step, not this one.

Source md5s: `manifest.json`. Banked by the TrainingFlyWheel, staged only (never committed or pushed).
