# Where the scorer's pick loses PDMS against its own best proposal (P7 field fans F and C, navtest)

MM, 2026-11-22. **MEASURED** (ours): `pickloss.py` over the banked per-proposal tables
`E:/Projects/TanitAD-artifacts/field_read/propscore/full_{F,C}_s{0,1}of2.jsonl` (12,146 navtest scenes x 64 proposals, tracked = LQR, NAVSIM v1.1 PDMS).
Tier: the pick is **T1** (tracked). `best_of_64` is a **label-side bound (T0-class)**: it uses the true sub-scores to choose, so it is never a driving result.

**Controls.** (1) The NAVSIM v1.1 formula `nc*dac*(5*ep + 5*ttc + 2*c)/12` reproduces every stored proposal PDMS exactly (max abs error 0.0).
Adding `ddc` to the multiplier gives error 1.0, so this harness's PDMS does not use ddc. (2) Pick and best-of-64 reproduce `field_read.json`: F 91.15 / 97.66, C 91.24 / 97.78.

**Attribution (exact; the terms sum to the gap).** If the pick's multiplier `nc*dac` is below the best proposal's, the whole scene gap counts as a **gate** loss, split between the failing terms.
Otherwise the multipliers are equal, and the gap splits exactly into `mult*w_k*(best_k - pick_k)/12` for k in ep, ttc, c.

| arm | pick | best of 64 | gap (points) | gate: collision (nc) | gate: drivable area (dac) | progress (ep) | TTC | comfort |
|---|---|---|---|---|---|---|---|---|
| F | 91.15 | 97.66 | 6.51 | 1.24 (19 %) | 1.56 (24 %) | 2.28 (35 %) | 1.44 (22 %) | 0.00 |
| C | 91.24 | 97.78 | 6.54 | 1.07 (16 %) | 1.75 (27 %) | 2.29 (35 %) | 1.43 (22 %) | 0.00 |

Scene counts (C): the pick is already the best in 5,178 of 12,146 scenes. The pick is worse in 6,968 scenes:
- 356 scenes (2.9 % of all) are gate scenes and hold 2.82 points (43 % of the gap);
- 6,612 scenes are weighted-term scenes, where the median progress deficit is 0.061.

F is the same within 0.2 points on every term.

**Reading (diagnostic, not a lever result).**
- Two failure modes carry the gap almost equally.
- **(a) Rare hard misses:** about 3 % of scenes, where the pick collides or leaves the drivable area although a safe proposal existed. They hold about 2.8 to 2.9 points.
- **(b) A broad mild-conservatism / TTC misrank:** in about 54 % of scenes the pick makes less progress or has a worse TTC than the best proposal. It holds about 3.7 points.
- The 94.55 target needs +3.3 points over C1 (92.55, `MODEL_REGISTRY.md` 14.6). This decomposition is on the P7 field fans, not on C1: C1's banked read keeps only the picked proposal.
- A lever that fixes (a) alone, or (b) alone, recovers at most about 2.8 or 3.7 points on these fans.

**Gate scenes split by plan vs tracked execution** (same tables; `tp` = tracker-transparent, i.e. the plan scored as drawn).
In a gate scene, the picked plan may be safe as drawn and fail only once the LQR tracks it, or it may already fail as a plan:

| arm | gate scenes | plan safe, tracked fails | plan already fails |
|---|---|---|---|
| F | 360 | 165 scenes, 1.28 points | 195 scenes, 1.52 points |
| C | 356 | 130 scenes, 1.03 points | 226 scenes, 1.79 points |

The pick's multiplier is the same tracked and transparent in 96.2 % (F) and 97.3 % (C) of scenes.

- About 1.0 to 1.3 points of the hard misses are **tracker-induced**. F and C train on P7's own bank (`train_grow_cmd_3c_p7`, `targets_sha256 92af8d66`; recipe line of `eval/PREREG_AUX_FIELD_LOSS_P7.md`), not on P5-C's tracker-aware Tc20s labels.
- That is the size of C1's measured gain over C0: +1.17 [+0.67, +1.65] tracked (`MODEL_REGISTRY.md` 14.6). C1 is P5-C, trained on tracker-aware Tc20s labels.
- That match in size is a HYPOTHESIS, not an attribution. C1 has no 64-proposal table, so its own gate split cannot be read.
- The remaining ~1.5 to 1.8 points are plan-level misranks, which is the scorer's job. SCORER-FT runs on the C1 base.

Input for the card that follows the SCORER-FT read. It does not change SCORER-FT's pre-registered read.

## Addendum (MM, 2026-11-22 ~10:50Z): what LQR tracking costs, pick vs plan-level oracle

**MEASURED**, `trackcost.py` -> `trackcost.json`, same tables.
- `tr` is tracked (the T1 harness) and `tp` is the plan scored as drawn.
- Descriptive means x100, with no CI. If any of these is quoted as a lever effect, it needs the paired log-cluster bootstrap first.
- Control: the v1.1 formula reproduces `tr.pdms` exactly (max abs error 0.0).
- Caution: the harness field `tp20_pick` does **not** equal `tp.pdms[pick]` (max abs error 1.0), so it is not used here.

| arm | pick, tracked | pick, as drawn | tracking cost to the pick | best of 64, tracked | best of 64, as drawn | best-as-drawn plan, then tracked |
|---|---|---|---|---|---|---|
| C | 91.24 | 91.53 | +0.29 | 97.78 | 98.28 | **89.15** |
| F | 91.15 | 90.87 | -0.28 | 97.66 | 98.22 | **89.52** |

**Reading.**
- **No systematic tracking cost on the pick.** Tracking moves the pick by under 0.3 points net, in opposite directions on C and F.
- **But it churns individual scenes.** For C, tracking makes the pick worse in 4,297 scenes and better in 3,326. For F, the counts are 4,241 and 3,439.
- **A tracker-blind oracle is worse than the learned scorer.** Take the proposal that scores best *as drawn* and track it: it scores 89.15 (C) and 89.52 (F), below the scorer's own tracked pick (91.24 and 91.15).
- **Plan-level optimality does not survive the tracker.** The tracked best of 64 is 97.78, so 8.6 points separate the best plan as drawn from the best plan as executed.
- **Consequence for the next card.** Any selection target, label or rank loss must be defined on the **tracked** outcome, never the transparent one. This is consistent with C1's tracker-aware Tc20s labels being the one measured gain (+1.17, `MODEL_REGISTRY.md` 14.6). SCORER-FT already meets this. Its pod argv `ARGS_SFTC` differs from C1's real argv only in `--init-from`, `--steps`, `--out` and `--rank-w`, so it trains on C1's own a33 tracker-aware label table (`--traj-label-table /workspace/data/p5c_label_a33/P5C_label_table_p5r-rj5_rank0.json.gz`, sha256 `3b198314`).
