# SPEC — v7f instrument repair (P1 non-collapse, P2 drift)

**ArchInf FlyWheel · 2026-09-05 · 0 GPU on both hosts, by construction**
(`p2_leak_rescore.guard_and_stack` REFUSES to run if `torch.cuda.is_available()`;
`CUDA_VISIBLE_DEVICES=-1` is set before torch is imported. Thor was saturated by three
sibling `refav1_lon` arms and the dev-box 4060 was at 100 % / 7341 MiB throughout.)

Predecessor: `…/2026-09-05-v7f-six-problems/STATUS_BOARD.md` (commit `7207d52`).
⛔ The PI's directive stands: **v7f was not trained and no training was launched.**

---

## 1. What the board asked for

Two of the six rows are `UNVERIFIED-CLAIM`, and the board's own key insight is that both
fail the same way — **the criterion's own control does not behave** — so neither needs a
training arm. Both are instrument repairs and both are zero-GPU.

* **P1 NON-COLLAPSE.** Re-derive the frozen-DINOv3 participation floor from source and
  state the pooling behind each of 5.756 / 8.56 / 20.23 / 40.77. Say which single value is
  the admissible bar and why. Re-read every arm against it. Measure the arms that were
  never measured (`emao14_30k`, `emao14_30k_tauramp`, `o14fut30k`, `postrain30k_freeze`).
* **P2 DRIFT.** The endpoint-shuffled control reads **higher** than the signal. Subtract it
  properly, report the corrected effect with its estimator and CI, and rule the row.

---

## 2. Pre-registered decision rules — committed BEFORE the numbers were read

### P1

1. The statistic is **`participation_ratio` (p ∝ σ²)**, never `effective_rank` (p ∝ σ).
   `effective_rank` is emitted beside it **only** to exhibit the C132 divergence and is
   never a criterion.
2. A participation number is quotable only with **corpus sample + n + ambient dimension d**.
   The artifact records the **clip ids**, because the entire four-value confusion exists
   precisely because no artifact named its corpus SAMPLE.
3. ⭐ **The panel is INADMISSIBLE unless `rdw8p30k` reproduces `gateb_panel.json`'s
   `participation_val` 25.583 to within 2 %.** If it does not, the local cache is not the
   cache `full_panel.py` read and no comparison in the artifact may be quoted.
4. **Rank is NECESSARY, NOT SUFFICIENT (C131).** No arm is passed on rank alone, and the
   row must say so.
5. An arm whose checkpoint is not available is reported **missing**, never imputed.

### P2

6. The corrected drift effect is **`r_true − r_endpoint_shuffled`**, per direction, on the
   identical code path (`p2_leak_rescore.py:cmd_drift`, unmodified).
7. **Committed in advance, both outcomes:**
   * if the freeze arm's control **stays near 0.677** while its signal falls to ~0.39, the
     drift improvement is REAL and the row becomes **SOLVED**;
   * if the freeze arm's control **falls with the signal**, the improvement is arithmetic
     and the row is **OPEN with a null effect** — reported as such, not softened.
8. The interval is reported with its estimator named, and **which of the three variance
   questions it answers** (episodes / training runs / inference runs) is stated explicitly.
9. ⛔ Per `MM-E4-L1`, no drift number is quoted without its paired prediction read.

---

## 3. Method

* **Corpus (P1):** `physicalai-val-w120-256x640cyl`, `sorted(glob('*.v2ep.pt'))[:12]`,
  120 frames/clip ⇒ n 1440 — bit-identical to `full_panel.py:88,103`, the selection every
  banked arm number came from. Clip ids are banked in the artifact.
* **Corpus (P2):** `physicalai-val130-heldout`, 80 clips / 7,680 rows, K 4, band [0,8) —
  the identical selection as the banked `drift.json`.
* **Representation:** `z_op` via `world.encode_window` (n_stack 3), d 2048.
* **Instrument:** `tanitad.models.v6.spectrum_report → participation_ratio` (centred
  covariance) — the same function the O6 monitor and the offline analysis use.
* **Checkpoints:** scp'd from `thor:/home/nvidia/v7tiny/<arm>/ckpt.pt`, **md5-verified
  against the remote in the same command** (hashes in `tools/p2_leak_rescore.py:59-73`).
  Network only; no GPU load was added to Thor.
* **Tool changes:** the only edit to `p2_leak_rescore.py` is **five path registrations** in
  `CKPTS` — `cmd_l3`/`cmd_drift` call `load_arm(arm)` with no override (only `cmd_actdiv`
  honours `--ckpt`, `:288`), so an arm must be registered to be scoreable. **No estimator,
  corpus, or protocol was touched.** New file: `tools/l1_participation.py`.

---

## 4. Traps this SPEC is written against

* ⛔ **`grep -F` literal + same-breath control before believing any zero** (`M50`). The
  absence claim in §P1 ("the 8.56 clips exist in no cache on this box") is paired with a
  control that must read non-zero, and it did (1, 1, 3 hits vs 0, 0, 0).
* ⛔ **A separated CI answers only "would another draw of EPISODES say this?"** — named
  explicitly wherever an interval appears.
* ⚠️ **One config namespace's silence is not absence** (`M49`) — the kill-gate criterion was
  read out of the *code that applies it*, not out of the prose that describes it, and the
  two turned out to differ.
* ⛔ **Content assertion, never existence**: every latent bank is checked finite and
  non-zero before it is scored (an all-zero bank scores like a valid one).
* ⚠️ **cp1252 box** — no non-ASCII in `print()`.
