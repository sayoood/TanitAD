# Ladder at v6F-SW-30k@16000 — the four-point decline was noise, and the ladder cannot see trunk progress at this resolution

`MEASURED` · **T0-DIAGNOSTIC — a frozen-trunk linear readout is a world-model
diagnostic, NEVER driving performance** · route A (`unpen`) · seeds 0/1/2 ·
130-clip lead-enriched probe pool, **70 episode clusters**, ⚠️ **not the
40-episode val set** · `taniteval.ci.paired_episode_cluster_bootstrap`,
`n_boot 2000` · ⛔ `overlapping_holdout_se` forbidden.

## 0. The question, and why it needed care

The PI asked whether the v6 trunk shows **progress versus the early test**. The
banked 3-seed ladder (`…/2026-08-18-ladder-3seed`) had four checkpoints whose
`n_agents_all` margin declined monotonically — **+0.262 → +0.243 → +0.217 →
+0.211** — which reads as the latent slowly closing on the trivial control.

⛔ **The withdrawn D1 probe could not be re-used to answer this.** It failed its
own positive control, and its headline (*"~1.8 m better than the random null,
r +0.159"*) was an **ego-speed proxy** — at the eval-optimal alpha the true
margin was 0.02–0.07 m, a 25–90× overstatement. The comparison therefore had to
run through the **repaired** ladder with the trivial-proxy control on every row.

**Sign convention, pinned from `LADDER_3SEED.md:237`:** K1B is
negative-is-better, so `margin = arm_K1B − C-V0_K1B` and **POSITIVE MEANS THE
ONE-NUMBER EGO-SPEED SCALAR WINS**. ⚠️ `K1B` is the arm's own statistic, **not**
the margin; reading it as the margin inverts the conclusion, and it nearly did
here — `n_agents_all` K1B alone falls −1.11 → −1.48 across the series, which
looks like the latent improving when the margin says the opposite.

## 1. Pre-registration (committed BEFORE the fit ran)

| outcome | threshold | reading committed in advance |
|---|---|---|
| continues | margin ≈ **+0.13 … +0.19** | a real slow trend; 30k worth measuring properly |
| **flat or up** | margin **≥ ≈ +0.21** | **the decline was noise on a seed-unstable statistic; the D1 negative stands at a second checkpoint** |
| crosses zero, all 3 seeds | < 0 ×3 | first genuine positive on this ladder — provisional until re-run |

## 2. Result — the "flat or up" branch fired

| target | s02000 | s09000 | s09250 | s10000 | s11250 | **s16000** |
|---|---|---|---|---|---|---|
| **n_agents_all** | +0.5816 | +0.2621 | +0.2428 | +0.2173 | **+0.2105** | ⛔ **+0.5102** |
| n_agents_grid | +0.3115 | +0.5693 | +0.2846 | +0.1583 | +0.7324 | +0.2943 |
| nearest_any | +0.1618 | +0.2013 | +0.1381 | +0.2076 | +0.1500 | +0.1547 |
| lead_gap | +1.8125 | +1.8784 | +1.7430 | +1.7776 | +1.7408 | +1.7364 |
| lead_present | +0.0011 | −0.0001 | −0.0001 | +0.0004 | −0.0000 | −0.0002 |
| ego_v0 | +4.0636 | +4.3754 | +4.2889 | +4.1627 | +4.2254 | +4.4413 |

⇒ **+0.5102 — not flat but back to roughly the step-2000 level (+0.5816).** The
monotone decline over four checkpoints did not continue; it reverted.

⇒ ⛔ **NO TARGET CROSSES ZERO AT ANY CHECKPOINT.** The 2 048-dimension operative
latent is out-read by one scalar of ego speed on every rung, at every step
measured, including the latest. `ego_v0` is at its series maximum (+4.4413).

## 3. Two facts that make this a finding about the INSTRUMENT

**3a. The seed spread swamps the mean.**

| checkpoint | per-seed margins | spread | mean |
|---|---|---|---|
| s11250 | −0.541 / +0.310 / +0.863 | 1.404 | +0.2107 |
| **s16000** | −0.077 / +0.411 / +1.197 | **1.274** | **+0.5103** |

The spread is **2.5× the mean** at 16000. A quantity whose seed spread is
larger than its central value cannot support a trend read off four means. The
sign pattern is `-++` at **every** checkpoint from 9000 on — seed 0 is the only
seed ever favouring the latent, and it never separates.

**3b. The two agent-count targets move in OPPOSITE directions.** From 11250 to
16000, `n_agents_all` rose +0.211 → +0.510 while `n_agents_grid` fell
+0.732 → +0.294. They are anti-correlated across the whole series. Two views of
a genuine scene representation would improve together; this is what noise on a
shared ridge looks like.

⇒ ⭐ **THE OPERATIONAL CONCLUSION: this ladder cannot detect trunk progress at
this resolution.** Separating a real ~0.05/checkpoint drift from a ±1.3 seed
spread needs more seeds or a larger eval pool. Worth knowing **before** the 30k
gate spends GPU on a measurement that cannot answer the question put to it.

⚠️ **Not claimed:** that the trunk is not improving. Claimed: **this instrument
cannot tell**, and the one apparent signal it produced reverted at the next
checkpoint.

## 4. Provenance and gates

* **Reproduction gate PASS** — the analyser re-derives all four banked §6a
  margins from JSON on disk to 4 dp (`+0.2621 / +0.2428 / +0.2173 / +0.2105`
  vs banked `+0.262 / +0.243 / +0.217 / +0.211`) **before** reporting any new
  point. An analyser that cannot re-derive the known answer may not be trusted
  with the unknown one.
* **Pool identity** — the s16000 dump is byte-comparable to the banked pool:
  `n_frames 5617`, `n_episodes 130`, `max_in_grid_agents_seen 92`,
  `cuda_max_mem_gb 1.457058816`, `latents.pt` 32,630,727 B — all identical to
  `cache_s11250`. Same split, same join file, same 70 clusters.
* **Staleness gate** — `sp1_cache_latents.py` md5 `0b4f4a50…` matches both repo
  copies; `sp2_probe.py` `aabbee36…` matches the parity run's stated md5;
  `pc6_linear_readout.py` and `ll1_ladder.py` verified against the repo AND
  import-probed for the repaired `intercept_col` / `centred` signatures.
* **Checkpoint provenance** — `v6F_sw_step016000.fp16.pt`, md5
  `fccafd8a132dfefb14d4ca8611e5f0b5`, 673,312,891 B, verified identical on Thor
  and the dev box. ⚠️ The **fp16 snapshot**, not the live `ckpt.pt`: the trainer
  rewrites the latter periodically and reading it mid-write risks a torn file.
* **No load was added to Thor.** Dump and fits ran on the dev-box RTX 4060
  (1.46 GB peak, 12 min/dump at 7.9 fr/s) while S-W training continued
  undisturbed at ~26.45 s/step.

## 5. Deliverable manifest

| artifact | where |
|---|---|
| `ll3_s16000.json` (3-seed fit, 11 targets) | `…/2026-08-19-ladder-s16000/raw/` |
| `margin_series.py` (analyser + reproduction gate) | `…/2026-08-19-ladder-s16000/code/` |
| this report | `…/2026-08-19-ladder-s16000/LADDER_S16000.md` |
| s16000 latents (32.6 MB) | scratchpad `sp2/cache_s16000/` ⚠️ **not banked — regenerable from the checkpoint in 12 min** |

⏳ **In flight:** dumps + fits at **12000** and **14000**, to establish whether
the 11250 dip was a single excursion or the series wanders on this scale
throughout. Either way it measures the instrument's noise floor, not the trunk.
