# v2corpus vs v1 — long overlay videos for visual assessment

**Rendered 2026-08-02 on `tanitad-eval`**, PI request: *"generate some eval video from v2 corpus,
make them a little bit longer and improve the quality of the visualization, put in a dedicated
folder."*

| | |
|---|---|
| clips | **12** (6 episodes × 2 arms), 26 MB |
| length | **~171 frames ≈ 17 s each** — the FULL episode |
| quality | **CRF 16 + `-preset slow`** (was CRF 21) |
| overlay | camera projection + metric BEV inset + text HUD (decoded manoeuvre, route/goal, ADE) |

⚠️ **"Longer" is capped by the data, not the setting.** `--max-frames` was raised to 600 but the
canonical val episodes are only ~171 frames. These are now the *whole* episode rather than an
excerpt; 17 s is the ceiling on this corpus.

---

## ⛔ READ THE FILENAME BEFORE READING THE VIDEO

**`LEAKFREE` vs `INTRAIN` is not decoration.** C64 found that **21 of the 40** canonical val
episodes sit inside `physicalai-v2bal`, v2corpus's TRAINING corpus. On an `INTRAIN` clip, v2corpus
is being shown data it trained on — **good behaviour there is not generalisation**. The tag is in
every filename so a clip cannot be read out of context.

* `LEAKFREE-ep{00,03,08,31}` — genuinely held out for **both** arms
* `INTRAIN-ep{01,11}` — inside v2corpus's training set (still held out for v1)

---

## ⭐ The per-clip ADE, and the finding it produces

**MEASURED** (clip-mean ADE, printed by the renderer):

| episode | leak status | **v1** | **v2corpus** | |
|---|---|---|---|---|
| ep00 | LEAKFREE | **0.178** | 0.324 | v1 |
| ep03 | LEAKFREE | **0.362** | 0.813 | v1 |
| ep08 | LEAKFREE | 0.444 | **0.300** | ⭐ v2corpus |
| ep31 | LEAKFREE | **0.977** | 1.200 | v1 |
| ep01 | **INTRAIN** | **0.187** | 0.314 | v1 |
| ep11 | **INTRAIN** | **0.623** | 1.143 | v1 |

⭐⭐ **v2corpus is worse even on the episodes it TRAINED ON** (ep01 0.187 → 0.314; ep11 0.623 →
1.143). That matters for diagnosis: if v2corpus were merely *over-fitted* to its bigger corpus we
would expect it to look strong on `INTRAIN` clips and weak on `LEAKFREE` ones. **It does not.** It
is worse on data it has memorised, which points at the arm being genuinely weaker — consistent with
the four-family result (**speed bias −1.260 m/s, lateral ~2× worse**) rather than with a
train/test-distribution story.

⚠️ **n = 6 clips.** This is a *visual-assessment* set, not an estimator. The decision-grade number
is C64 option A: v1 **0.393** [0.307, 0.493] vs v2corpus **0.575** [0.429, 0.752], paired
episode-cluster bootstrap B=2000, ΔCI **[−0.221, −0.145]**, n=418 windows over the 19 leak-free
episodes. ⛔ Do not quote the table above as a result.

⭐ **ep08 is the one clip v2corpus wins**, and it is worth watching precisely because it is the
counter-example — a single-clip win inside a separated aggregate loss.

---

## Watching them

Compare the **same episode across the two arms** (`v1_…ep03` beside `v2corpus_…ep03`), not
different episodes. The HUD carries the decoded manoeuvre and route/goal, so a divergence can be
read as *wrong decision* vs *right decision badly executed* — which the scalar ADE cannot
distinguish.

**Known from the instruments, and visible here:** v1 tends to run **ahead** of the human
(speed bias **+1.465 m/s**), v2corpus **behind** (**−1.260 m/s**). Expect v1's predicted path to
overshoot into the distance and v2corpus's to lag the ground truth.

⚠️ Files are `.mp4` and therefore **gitignored** (`.gitignore:24`) — the videos live locally, this
index is what is committed.

## Evidence class

| claim | class |
|---|---|
| per-clip ADEs above | **MEASURED (ours)** — renderer output 2026-08-02, `/workspace/v2vid.log` |
| C64 option A headline + CI | **MEASURED (ours)** — `taniteval` episode-cluster bootstrap B=2000 |
| speed-bias figures | **MEASURED (ours)** — `four_families.py`, `fourfam_*.json` |
| leak status per episode | **MEASURED (ours)** — `v2bal_leakfree_val19.json` |
| "worse on memorised data ⇒ genuinely weaker, not over-fitting" | **HYPOTHESIS** — 6 clips; consistent with the four families but not established |
