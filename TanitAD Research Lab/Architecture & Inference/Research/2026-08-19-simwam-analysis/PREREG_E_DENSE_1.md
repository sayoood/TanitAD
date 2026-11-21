# PRE-REGISTRATION — E-DENSE-1: is the missing patch-level pressure the cause, and does adding it fix decodability?

`PRE-REGISTERED 2026-08-21, BEFORE ANY ARM WAS TRAINED.`
**T0-DIAGNOSTIC.** Dev-box only — **Thor untouched.**

---

## 1. The claim under test, and why it is now worth GPU

E-DETECT-1 established, against a **validated** instrument (`oracle` AP 0.3673 /
AUC 0.9098), that v6's 640-token field is statistically indistinguishable from
the raw patches it consumes — `v6_tokens` 0.0923 [0.0815, 0.1034] vs `pixel`
0.0912 [0.0814, 0.1014] on the grid, and **0.0401 vs 0.0401** on boxes — while
DINOv3 on the same head scores **0.1884**.

⭐ **A MECHANISM WAS THEN FOUND IN OUR OWN SOURCE.** v6's only spatial objective
is `o3_masked_cell_loss` → `MaskedCellPredictor`, which predicts *"MASKED
readout-grid cells"* with shape `[B, C, d_r]`, **C = 16, d_r = 128** — the 4×4
pooled cells. **v6's objective never touches the 640 patch tokens.** It also
scores masked cells only, with the stated rationale: *"Scoring visible cells too
would let the model win by copying, which is exactly how a masking objective
silently becomes an autoencoder."*

⭐ **AND THE LITERATURE SAYS THIS IS THE FAMILY'S KNOWN FAILURE, WITH A FIX.**
`V-JEPA 2.1` (arXiv 2603.14482, FAIR, 2026-06; **banked**) reports JEPA-family
models with strong global and broken dense features, DINOv3 as the reference
that beats them, and two remedies: **(i) a Dense Predictive Loss in which "all
tokens — visible context and masked tokens alike — contribute to the training
loss"** — precisely what v6's comment forbids — and **(ii) Deep Self-Supervision
at intermediate encoder layers.** Measured: ADE20K **24.4 → 47.9 (+96 %)**,
DAVIS tracking **52.5 → 69.0 (+31 %)**, NYUv2 depth RMSE **0.642 → 0.307
(+52 %)**, global tasks flat.

⇒ **FIRST HYPOTHESIS (ours): v6's latent is undecodable for spatial content
because its objective applies no pressure at patch granularity. Adding
token-level dense pressure should make it decodable.**

## 1b. ⛔⛔ THAT HYPOTHESIS'S STRONG FORM IS ALREADY REFUTED — by our own control, before any arm ran

The granularity story makes a sharp prediction: **v6 should beat raw pixels AT
CELL GRANULARITY**, because that is precisely where `O3` applies pressure. The
`pixel_pooled` arm — raw patches through v6's OWN 4×4 pool — tests it:

| granularity | v6 | raw pixels | delta |
|---|---|---|---|
| **cell (16)** — O3 operates here | `v6_cells` **0.0888** [0.0788, 0.0990] | `pixel_pooled` **0.0853** [0.0767, 0.0956] | **+0.0035**, CIs OVERLAP |
| **token (640)** — no pressure | `v6_tokens` **0.0923** [0.0815, 0.1034] | `pixel` **0.0912** [0.0814, 0.1014] | +0.0011 |

⛔ **v6 is indistinguishable from raw pixels at BOTH granularities, INCLUDING ITS
OWN.** Granularity alone therefore does NOT explain the null: the objective is
not merely applying its pressure in the wrong place, it is producing nothing
decodable *even where it applies*.

⚠️ **This is the C130 pattern caught one step earlier.** The mechanism —
`MaskedCellPredictor` operating on 16 cells and never on 640 tokens — is a FACT
about our source, and the arithmetic around it is right. But a correct mechanism
is not thereby the operative one, and the control that could refute it was cheap
(2 minutes, 16-token arm). It ran BEFORE the GPU was committed, which is the only
reason this pre-registration is not built on a refuted premise.

⇒ **REVISED HYPOTHESIS, and what the arms below actually test:** the defect is
not the RESOLUTION of the pressure but its NATURE. V-JEPA 2.1's Dense Predictive
Loss is not merely "the same loss at finer granularity" — it makes **every token,
visible and masked alike, carry training signal**, which changes what the
representation is asked to be, not just where. Arm **B** therefore carries the
weight of the experiment, and arm **D** carries the weight of interpreting a
negative.

## 2. Arms — identical platform, ONE named change each

Platform: the **E-V6SHAPE harness** (`e_v6shape.py`), which already reproduces
v6's signature (`v6shape` b/w 5.00 vs v6's 4.56; `cls192` 2.11). Same 130 clips,
same 5,617 keys, same episode-disjoint folds.

| arm | change | what it isolates |
|---|---|---|
| **A `cells-masked`** ⛔ CONTROL | masked prediction on **16 pooled cells, masked-only** — v6's design as shipped | the incumbent. **Must reproduce v6's null.** |
| **B `tokens-dense`** | masked prediction on **all 640 patch tokens**, **every** token contributes (visible + masked) | ⭐ the granularity + all-tokens change, together |
| **C `tokens-dense+deep`** | B, plus the objective at **intermediate encoder layers** | whether Deep Self-Supervision adds anything here |
| **D `distill-dino`** ⭐ POSITIVE CONTROL | same encoder, trained to **regress DINOv3 patch tokens** | **whether this architecture at this scale CAN carry patch content at all** |

⚠️ **B changes two things at once** (granularity *and* visible-token inclusion).
That is deliberate: they are V-JEPA 2.1's single "Dense Predictive Loss"
ingredient, and splitting them doubles the cost of a test whose first job is to
find out whether the direction works at all. **If B wins, the split becomes the
next experiment and B is NOT quotable as "granularity did it".**

## 3. ⛔⛔ Why arm D is non-negotiable

Without a positive control, a negative result here is **uninterpretable** — it is
equally consistent with *"the fix does not work"* and with *"6.4 M params on 130
clips cannot carry patch-level content whatever the objective"*. V-JEPA 2.1's
gains were obtained at **ViT-G scale on web-scale video**; we are four orders of
magnitude away. D settles which world we are in, and it is the same role
`oracle` played in E-DETECT-1 — the arm that made every other row readable.

## 4. Read-out

Primary: **E-DETECT-1 grid AP**, identical head, folds and episode-cluster
bootstrap, reported against the established ladder:

| reference | AP |
|---|---|
| `oracle` (instrument ceiling) | 0.3673 |
| `dino_tokens` | 0.1884 |
| `prior` (no features) | 0.1242 |
| `pixel` (raw patches) | 0.0912 |
| `pixel_pooled` (raw patches, v6's 4×4 pool) | 0.0853 |
| `v6_cells` (the deployed latent) | 0.0888 |
| `v6_tokens` (what we are trying to beat) | 0.0923 |

Secondary, for continuity with the nine earlier nulls: the **E-TRUNK-2 scalar
targets** (`lead_gap_m`, `left/right_occupied`, `ego_speed`) and **SIGReg** and
**b/w** so a fix cannot silently trade conditioning for content.

## 5. ⛔ Committed decision rule — written before any arm was trained

| observation | conclusion, committed now |
|---|---|
| **A ≈ `pixel`** | ✅ the harness is faithful; every other row is readable |
| ⛔ **A clearly > `pixel`** | the harness does NOT reproduce v6's defect — **the experiment is void** and must be rebuilt before anything is read |
| **B and/or C clearly > A and > `pixel`** | ⭐ **the missing patch-level pressure IS the cause, and the published fix transfers.** Pre-register a full-scale v6.5 arm |
| **C > B** | Deep Self-Supervision earns its complexity |
| **C ≈ B** | adopt the dense loss ONLY; do not pay for deep supervision |
| **B ≈ C ≈ A, and D SUCCEEDS** | ⛔ the objective change does not transfer at this scale, but the architecture is capable — **a scale question, not an architecture question** |
| **B ≈ C ≈ A, and D ALSO FAILS** | ⚠️ **nothing about the objective is licensed.** 6.4 M / 130 clips cannot carry patch content; the whole line needs a bigger platform before any claim |

⚠️ **THE READ IS ASYMMETRIC AND THAT IS DECLARED UP FRONT: a POSITIVE is
decisive; a NEGATIVE is scale-confounded and does NOT refute V-JEPA 2.1.**

## 6. Falsifiers built in

1. **Arm A must reproduce v6's null** (§5 row 2). A control that does not
   reproduce the defect cannot test its cure.
2. **Arm D**, §3.
3. **Params matched across A/B/C** and reported per arm; only the loss changes.
4. **The E-DETECT-1 floors carry over unchanged** — `prior`, `pixel`,
   `oracle`, `oracle_pooled` are not re-fit.
5. **Report SIGReg and b/w** alongside AP: a latent that gains decodability
   while collapsing has not been fixed.

## 7. What this CANNOT settle

* ⛔ **Not driving.** T0. Decodability is not planning — `C129` retracted exactly
  that step, and `RETRACTION_LOG` records DINOv3 decoding far better than v6
  while driving **2.62 m worse**.
* ⛔ **Not v6 at 336 M on 2,376 episodes.**
* ⚠️ **Not the `visible-tokens` question alone** — see §2.
* ⚠️ **No `D-` row licenses adopting this**, and no arm in the programme has
  tested it. This experiment exists to create that evidence, not to assume it.

## 8. Cost and priority

Dev-box RTX 4060 only; **Thor's `v6F-SW-30k` is untouched**. ⚠️ Note that run has
**~1.07 % of its learning budget remaining** (`E_V6MOVE`), so it is not a
competitor for this question — it cannot answer it either way.

## 9. Manifest

| artifact | where |
|---|---|
| this pre-registration | `…/simwam-analysis/PREREG_E_DENSE_1.md` |
| harness | `…/simwam-analysis/code/e_v6shape.py` (extended) |
| read-out | `…/simwam-analysis/code/e_detect.py` (unchanged) |
| result (to be written) | `…/simwam-analysis/E_DENSE_1_RESULT.md` + `raw/e_dense.json` |
