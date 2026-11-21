# CORPUS / CACHE TOKEN GLOSSARY — what each token in a name actually encodes

**Backlog P-19. Written 2026-09-02 by the Master Mind, adjudicated against source.**

⛔ **WHY THIS EXISTS.** On 2026-09-02 the token **`w120`** in
`physicalai-train-e438721ae894-w120-256x640cyl` was read as **"120 frames per
episode"**. It is the **120-degree horizontal field of view**. The misreading did
not stay in prose: it produced a windows-per-episode table (`114 − 20K`), a
conclusion (*"the corpus is exhausted at 12 s"*, *"K ≥ 6 yields no windows at
all"*), a **live launch guard** that refused `--s1-multi-k ≥ 6`, a **test that
defended the guard**, a `--help` string, and two comments — **six sites from one
token**, and it had retired the 8–30 s strategic band, which is the programme's
own thesis. Episodes are ~199 frames; K = 4…9 are all reachable.

⭐ **THE RULE THIS ENCODES.** A token in a filename or config key is an
**identifier, not a measurement**. Before any arithmetic rests on what a token
"obviously" means, find the line that **defines** it. This file is that lookup.

⚠️ **Scope, stated so it is not over-trusted:** every row below cites a source
line I read. Rows marked ⚠️ are inferred from usage rather than a definition, and
should be confirmed before anything load-bearing rests on them.

---

## The tokens

| token | what it ACTUALLY encodes | ⛔ what it is NOT | defined at |
|---|---|---|---|
| **`w120`** | the **120-DEGREE horizontal field of view**. It is the short form of the `wide120` directory level in the cache path. | ⛔ **not a frame count**, not 120 episodes, not a window length | `stack/tanitad/data/parity.py:222` — the path is spelled `…/physicalai-train-e438721ae894/**wide120**/physicalai-train-w120-<hex>/`; the rig is `camera_front_wide_120fov` |
| **`wide120`** | the same thing, unabbreviated — the directory level that groups the 120° crops | ⛔ not a resolution | `parity.py:222` |
| **`256x640cyl`** | frame geometry: **256 px high × 640 px wide, CYLINDRICAL projection** | ⛔ **not pinhole** — the pinhole FOV formula gives 92.6° here and is wrong; the column is linear in azimuth | `stack/tests/test_v2_parity.py:516`; geometry contract in `stack/tanitad/geometry.py` |
| **`e438721ae894`** | the **parity TRAIN corpus key** — a content hash naming exactly the 2,376-episode canonical selection | ⛔ not a date, not a version number, not a geometry | `parity.py:78` — `PARITY_TRAIN_KEY = "physicalai-train-e438721ae894"` |
| **`0c5f7dac3b11`** | the **parity VAL corpus key** | ⛔ not related to the train key beyond both being corpus hashes | `parity.py:79` — `PARITY_VAL_KEY` |
| **`f09e44db`** | the **skip-hash**: a marker for the **24-corrupt-clip skipset** | ⛔ **not a corpus key** and not interchangeable with one — it identifies which clips were EXCLUDED | `parity.py:80` — comment reads *"24-corrupt-clip skipset marker"* |
| **`v2ep`** | the **v2 episode file format** (`<clip_id>.v2ep.pt`), holding `frames_u8`, `actions`, `poses`, `episode_id` | ⛔ not "version 2 epoch"; ⛔ **`actions` stores `(κ, a)` — curvature FIRST**, the reverse of `RefAV1Config`'s `(a, κ)` | `stack/tanitad/data/v2_dataset.py`; written by `scripts/v2_compressed.py::build_compressed` |
| **`e4m3`** | **`float8_e4m3fn`** quantisation — 4 exponent bits, 3 mantissa bits | ⛔ not a model name | `stack/scripts/fp8_l2_gate.py:27` |
| **`-b14`** (in `dinov2-b14`) | a **MODEL** spec: ViT-**B**ase, patch **14** | ⛔ **not a corpus token at all** — the number is a patch size, not a count. Our DINOv3 cache is ViT-**L**/**16** | `stack/tests/test_refa.py:28` (usage); geometry pinned in `refa_v1.DINOV3_GEOMETRY` |
| ⚠️ **`EVAL6`** (in `physicalai-b1-EVAL6-w120-256x640cyl`) | ⚠️ **inferred, not defined in code**: the 6-clip eval subset — the 6 v7.2 eval clips the parity gate drops (147 eval labels − 141 present in B1 = 6) | ⛔ not "evaluation at 6 seconds" | no source definition found in `stack/` or `taniteval/` — **confirm before relying on it** |
| **`b1`** | the **B1 corpus**: 4,713 clips, the v7.2-labelled superset used by refcv3 and refav1 | ⛔ **NOT the parity corpus, and NOT a superset of it** — MEASURED: parity∩v7.2 = 190/2,400 (7.9 %), B1∩v7.2 = 4,572/4,713 (97.0 %), and *parity is not a subset of B1* | `GOALS_AND_CLAIMS.md` D-REFCV3-CORPUS |

---

## The two shapes that keep biting

**1. A SUFFIXED NUMBER IS RARELY A COUNT.** `w120` = degrees · `-b14` = patch size ·
`256x640` = pixels · `e4m3` = exponent/mantissa bits. **None of them is "how many".**
The one number that *is* a count — 4,713, 2,376 — never appears as a name suffix;
it appears in an index or a log line.

**2. A HASH NAMES A SELECTION, NOT A PROPERTY.** `e438721ae894` and `f09e44db`
look alike and mean opposite things: one names the clips that are **in**, the
other names the clips that are **out**. Neither encodes geometry, date or size.

⇒ **When a number in a name is about to enter arithmetic, look it up here first.**
If the token is not in this table, find its defining line and **add it** — the
cost of the lookup is a minute, and the cost of the assumption has twice been a
guard that refused real work.
