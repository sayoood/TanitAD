<title>navhard start-speed precondition</title>

# navhard's starts are NOT faster than warmup's — so navhard cannot test `H-NAVHARD-STOP-1`, and no arm should be run there

`Benchmarks & Evals · 2026-09-20 · Master Mind · MEASURED, 0 GPU, no scoring — read streaming from the downloaded archive`
`Instrument: code/startspeed.py + code/peek.py → raw/navhard_startspeed.json. Source: navsim_v2.2_navhard_two_stage_scene_pickles.tar.gz (sha256-verified on download, receipt landed 3b02a41).`

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **1** | ⛔ **The precondition fails.** navhard: median start speed **3.89 m/s**, **18.7 %** of scenes below 1 m/s (n = **5,462**, 0 unreadable). Warmup's scored stage-2 subset: median **4.14 m/s**, **17.6 %** below 1 m/s (n = 204). navhard is marginally **slower**, not faster. | MEASURED |
| **2** | ⇒ **`H-NAVHARD-STOP-1` cannot be tested on navhard.** Its antecedent is *"a split whose starts are NOT slow"*. navhard does not satisfy it, so running the five arms there would compare two samples of the **same regime** and discriminate nothing. ⛔ The hypothesis is **not amended** — no arm has run and no result has been seen; this is a precondition measurement, and what it retires is the *venue*, not the claim. | ruling |
| **3** | ⭐ **The fix is to stratify by START SPEED, not by split.** navhard's p75 is **6.66 m/s** and p90 **9.41 m/s**, so fast starts are plentiful *within* it. A fast-start stratum is the split the hypothesis actually names. | MEASURED + design |
| **4** | ⚠️ **A reporting error of mine, caught before it was quoted.** The first run reported **2,731 "unreadable"** scenes. They were not unreadable: the archive holds **two** `.pkl` directories — `synthetic_scene_pickles` (5,462, the scenes) and `openscene_meta_datas` (2,731, a different object with no `ego_status`) — and my filter matched both while `except Exception` swallowed the `KeyError`. | MEASURED |

## 1 · The numbers

| split | n | median \|v0\| | p25 | p75 | p90 | below 1 m/s |
|---|---|---|---|---|---|---|
| **navhard** (all synthetic scenes) | 5,462 | **3.89 m/s** | 1.58 | 6.66 | 9.41 | **18.7 %** |
| **warmup** (the 204 scored stage-2 scenes) | 204 | **4.14 m/s** | — | — | — | **17.6 %** |

⚠️ **Scope, stated before the comparison and not after:** warmup's figures are over the **204 stage-2 scenes the scorer actually ran**; this reads **every** synthetic scene in the navhard archive. The populations are not identical, so the comparison is **indicative of the regime**, not a matched contrast. It is nonetheless decisive for the question asked, because the hypothesis needs navhard's starts to be *materially* faster and they are not faster at all.

**Cross-check that the count is right:** 5,462 scenes read with **0 unreadable**, and `synthetic_scenes_attributes.csv` carries exactly **5,462 rows**. Two independent routes to the same n.

## 2 · ⚠️ The "2,731 unreadable", and why it mattered more than the fix

A third of the archive appeared to fail. Had that reached the write-up beside the distribution, it would have read as a **data-quality problem in a PI-authorised download** — and the honest-looking move (quote the distribution, flag the losses) would have been wrong in both halves: the distribution was fine and the losses did not exist.

⛔ The cause was mine: a filter on `.pkl` alone, plus a bare `except Exception` that cannot tell *"this file is broken"* from *"this is a different kind of file"*. ⭐ The discriminating check was two lines — **count the entries by directory** — and it turned a phantom defect into a one-line filter. Same family as the rule that *"0 hits" is a claim about the SEARCH, not the content*: here **"2,731 unreadable" was a claim about my filter, not about the archive.**

## 3 · What should happen instead

1. ⛔ **Do not run the five arms on navhard as a whole.** It costs scoring time and cannot answer the question.
2. **Pre-register a FAST-START stratum before scoring** — e.g. scenes with \|v0\| above a threshold fixed in advance, from navhard, warmup, or both pooled. State the threshold and the resulting n **before** any arm runs, exactly as the original SPEC did.
3. **Report the ≤ 5 m EP clause's firing fraction per stratum.** On warmup it fires on **37/204 = 18.1 %** (`d86dccb`), and that fraction is the mechanism made visible; on a fast-start stratum it should collapse toward zero, and if it does not, the clause is not what makes stopping win.
4. ⚠️ Carry forward what the EP census already showed: the clause explains **18.1 %** of warmup scenes and the remaining **82 %** is **unexplained** — a zero-displacement plan still earns EP median 0.195 there. A fast-start stratum tests the clause, **not** that second mechanism.

## 4 · Reproducing this on Windows, because it is not obvious

Two independent obstacles, both recorded so the next reader does not rediscover them:

* The pickles carry **`pathlib.PosixPath`**, which CPython refuses to instantiate on Windows (`UnsupportedOperation`). The data is portable; the path **class** is not. `code/peek.py` maps it to `PurePosixPath` at unpickle time — same string, instantiable everywhere, nothing rewritten.
* They also need **`nuplan`**, which is not in the project venv. Use `C:/Users/Admin/navsim-crun/venv/Scripts/python.exe`.

⭐ And the archive is read **streaming from the tar**: 8,194 members, none extracted to disk. The scene pickles are downloaded and sha256-verified but were **never unpacked** — which was the real blocker on this work, not the assignment.
