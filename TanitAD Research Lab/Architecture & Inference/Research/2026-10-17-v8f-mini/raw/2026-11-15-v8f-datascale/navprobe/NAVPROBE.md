# NAVPROBE: the PhysicalAI-AV repo ships turn-by-turn route text for 99.7 % of clips (Master Mind, 2026-11-16)

**Layer: published repo FILES.** `navigation/navigation.parquet` is a repo-level file of the NVIDIA PhysicalAI-AV dataset repo. It is not one of the per-clip features listed in `features.csv`, which is why five probes of the feature set never met it.

## What it holds (MEASURED, `navprobe.json`, `ds_navprobe.py`, CPU on the pod, read-only)

| quantity | value |
|---|---|
| file | 67,084,481 bytes; columns `clip_id` (str), `timestamp` (int64, µs, clip time), `nav_text` (str) |
| rows | 16,311,883 |
| clips | 305,372 of the 306,151 in `clip_index.parquet` (99.75 %); 0 nav clips outside the index |
| distinct `nav_text` | 9,864; 0 null, 0 empty |
| leading words (rows) | turn right 4.29 M, turn left 3.98 M, continue in 2.14 M, at roundabout 1.37 M, take the 1.14 M, merge slight 0.79 M, keep slight 0.61 M, continue straight 0.53 M, turn slight 0.39 M, make a 0.30 M, exit roundabout 0.27 M, exit rotary 0.09 M, turn sharp 0.07 M, enter roundabout 0.02 M |
| form | `<manoeuvre> in <d>m`, counting down per metre; e.g. "Turn right in 249m", "Exit rotary in 214m"; plain "Continue straight", "Turn right" |
| texts per clip | median 6, p95 204, max 679; 148,513 clips carry one text |

**Coverage of our lists:**

| list | covered |
|---|---|
| train4572 (v7.x label corpus) | 4,564 / 4,572 |
| eval141 | 140 / 141 |
| the 10k selection | 9,980 / 10,000 |
| eligible pool | 133,703 / 133,929 |
| NVIDIA default train | 153,333 / 153,625 |
| NVIDIA default val | 90,653 / 90,927 |
| NVIDIA default test | 61,386 / 61,599 |

## Second check: against the ego's own motion (MEASURED, `navalign.json`, `ds_navalign.py`)

The check used 3,000 of the 10k run's new clips, with the egomotion kept in `keep10k/*/ego_members`.

* **Countdown.** In 1,745 runs of one `<manoeuvre> in <d>m` text, `d` was regressed on the ego's arc length at each row's timestamp.
  * The slope has median **−0.9993** (IQR −1.0029 to −0.9962).
  * R² has median **0.99997**.
  * **93.1 %** of runs have a slope in [−1.1, −0.9] with R² ≥ 0.99.
  * ⇒ the distance is measured along the path the ego drove, on the egomotion time base.
* **Direction.** This covers 830 "Turn left / Turn right" runs whose manoeuvre point lies inside the clip. The ego's heading change was taken from 15 m before that point to 30 m after it, with a 30° threshold.
  * **804 agree** (right→right 401, left→left 403).
  * 26 show no turn above 30° (16 right, 10 left).
  * **0 are opposite.**
* **Look-ahead.** A manoeuvre is first announced at a median **127 m** ahead (p5 18 m, p95 250 m). That is the strategic band (6-30 s) at urban speeds.

## What this changes, and what it does not

* ⛔ **The repo-wide statement "no roundabout label and no route/goal signal in PhysicalAI-AV" is false at the published-repo layer.**
  * It holds for the per-clip feature set in `features.csv`.
  * It does not hold for the repo's files.
  * The rest of that paragraph is untouched: there is still no map, lane graph or traffic-light feature, and still no lat/lon (the countdown is a distance, not a position).
* **Admissibility (CLAUDE.md goal-input rule):**
  * **Route side.** The text is computed along the driven route, so as a route input it is optimistic by construction. That is the same sense as our existing ego-future route, but in the form a navigation system gives at inference: the next manoeuvre and its distance, with no geometry.
  * **Situation classifier.** The text is not built from the situation classifier.
* ⚠️ **Tactical family.** A text such as "Turn left in 3m" names the manoeuvre the tactical family scores. Any arm fed this text reports manoeuvre-decision quality split by announced manoeuvre versus none, the nav-echo check.
* **Next lever (0-GPU, then a PI go to train):**
  1. Parse the text into (manoeuvre class, distance) per corpus frame for train4572 + eval141.
  2. Measure agreement with the nav command the v8f-mini trainer feeds today.
  3. Pre-register a nav-text input arm against the current nav.
  * The STRATEGIC family stays PAUSED (PI item 25). This lever acts on the route input the tactical and operational layers already take.
