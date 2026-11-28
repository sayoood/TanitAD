# Brief: DRAFT the next decoder-side card, N1DH: an auxiliary route-decision head on O's plan feature (paper only; no build, no pod, no GPU)

You are a TanitAD design agent working for the Master Mind (MM). Read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip first:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.

Rules:
* Stage nothing, commit nothing, push nothing.
* No pod, no GPU, no training, no pytest.
* Read code with `git -C C:/Users/Admin/tanitad-push show <sha>:<path>` (the E: tree lags the tip).
* No secrets.

## Why this card exists (MEASURED; read the sources, do not take this summary on trust)
The deciding cell is D1-HO kinematic roundabout TAC-LAT, T1-kin/hb0.2, at 5,736 windows / 497 held-out episodes. N1D (the reference arm) decodes about 0.63 there, while its own plan-head INPUT carries the decision at about 0.70 (linear probe, T0). The decision is lost when the plan head decodes the 6 s track, not over the rollout. Two decoder-side levers are now read:
* **N1DW**, a loss weight on roundabout windows: its gain depends on the executor, and the PI parked it.
* **N1DC**, a direct linear skip from T's code into O's plan head: no gain. The skip is barely read (Z-USE about 0), and the guard carries mm-scale costs.

Both committed the same next lever: an auxiliary route-decision head on O's plan feature (kinematic label, cross-entropy) that conditions the decoder, under its own card.

Sources, all at the mirror tip, under `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/`:
* `2026-11-19-n1dw/probe/MM_READING_N1DWPROBE.md`
* `2026-11-19-n1dw/t0t1/MM_READING_N1D_T0T1.md`
* `2026-11-19-n1dw/read/verdict/MM_READING_N1DW_VERDICT.md`
* `2026-11-20-n1dc/PREREG_V8F_N1DC.md` (THE TEMPLATE: copy its structure, controls and read)
* `2026-11-20-n1dc/read/MM_READING_N1DC_VERDICT.md`

Also on the dev box:
* N1DC build records: `F:/TanitAD-scratch/n1dc/build/`
* N1DC code: branch `agent/v8f-n1dc-20261120` at `bcf3ba3f`. Find how `--o-code-skip` was wired, gated (G-DVB / REACH / Z0 / mutation) and minted. N1DH's flag follows the same path.

## What the draft must pin down
1. **The lever, from source.** Name O's plan feature exactly: module, tensor, dim, `file:line` at bcf3ba3f. Then specify:
   * a head `Linear(dim -> K)`, with K = the classes of the deciding TAC-LAT label. Find where that kinematic label is built and its class set (`obd_geom.labels` and the trainer's label path); quote `file:line`.
   * the CE weight flag;
   * the conditioning: the posterior embedded by a zero-init `Linear(K -> d)` and added to the plan-head input, so step 0 is bit-identical to N1D (Z0).
   * Say whether the posterior is detached into the conditioning, and why. Recommend one option and name the other.
2. **One lever or two.** CE + conditioning is the named lever, and CE alone is a representation-only variant. The programme forbids non-attributable multi-lever arms. Recommend whether the card trains CE + conditioning only (2 arms, s0/s1), or both variants (4 arms), with the cost of each.
3. **Admissibility (CLAUDE.md BINDING rules), from source:**
   * the head's inputs at inference contain nothing its label is derived from;
   * the decoder conditioning is the planner's own decision, not a goal input;
   * O's plan feature does not carry the situation classifier's output. If a shared trunk feeds both, say so and justify it.
4. **Read, bar, guard, estimator, outcomes and controls** as in N1DC:
   * the deciding cell N1DH - N1D per seed, with bar max(0.0383, F_HO, both seed draws);
   * the eval-141 four-family guard, STRATEGIC in full, executor of record T1-kin/hb0.2.
   * Write the guard's cost rule explicitly per CELL or per SEED. N1DC's wording "guard cost at both seeds" left that open; state it unambiguously.
   * Add the N1DC guard cells (s0 TAC-LON; s1 operative 0-2 s ADE / along / headway) as named risks.
   * Add Z-USE as the posterior zeroed at inference.
   * Add a deliberate-regression arm or a mutation that must go RED.
5. **Cost.** LoRA pod-h, ESTIMATED from N1DC's measured wall times (`/root/strat1/n1dc/runs/*/run_*.log` are not yours to read; use `F:/TanitAD-scratch/n1dc/build/` records and the N1DC verdict).
6. **Questions for the PI,** each with the MM's recommended answer.

## Deliverable
`F:/TanitAD-scratch/n1dc/next/PREREG_V8F_N1DH_DRAFT.md`, about 2.5k words at most, in the same plain style as the N1DC card. Every fact carries its `file:line` or artifact path and its evidence class.

Hand back in at most 200 words: the lever as specified, the one-or-two-lever recommendation with costs, any admissibility concern, and the PI questions.
