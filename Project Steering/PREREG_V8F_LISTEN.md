# PREREG_V8F_LISTEN: which input does each layer listen to? A causal input-dependence grid for S, T and O (written 2026-10-25, before any LISTEN data)

**Status: pre-registered.** Zero training. Eval-only on the BANKED M2b s0 / s1 checkpoints, on the dev box's CPU, reading the eval cache that `PREREG_V8F_SFRESH` fetches. It runs under `PI-V8F-2026-10-25b`. The PI asked for it on 2026-10-25 (*"Which visualization or finding show that the different layers are focussing on different abstractions? how can visualize this in asimple understable manner?"*, then *"Yes, build the who-listens-to-what grid"*).

## Why

The report's evidence for "different abstractions" is of two kinds (`D-V8F-ATTN-1`, `raw/2026-10-25-viz/`):
* **Attention share** (MEASURED, seed-robust): S and T put 97–100 % of their cross-attention on scene tokens; O puts 76 % / 93 % on its conditioning tokens. ⚠️ **Attention is not dependence.**
* **Three causal cells** (MEASURED):
  * shuffled nav costs S +4.25 m at 50 m;
  * it drops T's lateral macro-F1 from 0.84 to 0.30;
  * a donor T code costs O +4.4 to +6.4 m ADE6.

  Those shuffles replaced the input for EVERY layer at once (`v8f_mini_eval.py --nav-shuffle` swaps `it["nav"]` in the batch item), so they do not say which layer the dependence lives in.
* The **spatial attention maps are NOT seed-robust**, so they cannot carry the claim.

This read fills the grid with one causal number per (layer, input), injected at that layer only. It also answers the PI's question *"Is the tactical layer reacting to the nav commands and is then the trajectory conditioned to follow it?"*: O reads nav directly as well as through T, so the route nav → T → O has never been separated from nav → O.

## The inputs each layer reads (source, `stack/tanitad/models/v8f_mini.py` at `583cfc02`)

| layer | `run_*` | conditioning tokens | scene |
|---|---|---|---|
| S | `run_s` | ego (`p10f` at S's tick), nav, limit | 4 frames × 40 tokens at S's tick |
| T | `run_t` | ego, nav, limit, S checkpoints (`ck_for_t`) | at T's tick |
| O | `run_o` | ego, nav, limit, S checkpoints (`ck_for_o`), T code, tick ages | at "now" |

⚠️ **O's kinematic prior also reads `p10f`** (`self.prior(self._at(b["p10f"], i), b["base"])`). The O × ego cell swaps **only the ego conditioning token**: the prior keeps the window's own input. Otherwise the cell would measure the prior, not O.

## The intervention (ONE input of ONE layer per cell)

* **Donor:** `tanitad.models.hjepa.donor_indices(N, episode_ids=eid, seed=k)`, the eval's own: a derangement over EPISODES, so every donor window comes from a different episode. The donor item is built at the SAME ages, as `--tac shuffle` does.
* **Cell (L, X):** layer L reads input X from the donor window; every other input of L, and every other layer, reads the window's own values. **Upstream layers are computed clean.** Each row's metric is read at L's own output, so downstream propagation does not enter the cell.
* What "the donor's X" means:
  * scene = the donor's adapted tokens and frame mask;
  * ego / nav / limit = the donor's token input;
  * S checkpoints = the donor's own S output in its own frames (`ck_for_t` for T, `ck_for_o` for O);
  * T code = the donor's own code (as `--tac shuffle`);
  * tick ages = the age token of the donor's cycle position, with O's held checkpoints and code unchanged.
* **Row reference (L, all):** every input of L from the donor, the prior excepted (as above). The colour scale is $n(L,X)=\Delta(L,X)/\Delta(L,\mathrm{all})$.
* **Ages:** `--ages cycle`, the banked eval's schedule.

## The row metrics (the eval's own functions, `stack/scripts/v8f_mini_eval.py`)

| row | metric | Δ |
|---|---|---|
| S | checkpoint position error at 50 m (`checkpoint_metrics`, `pos_err_m@50`) | donor − clean (m) |
| T-lat | lateral macro-F1 (`tactical_metrics`) | clean − donor (F1 drop) |
| T-lon | longitudinal macro-F1 | clean − donor |
| O | ADE6 (`traj_metrics`); ADE2 beside | donor − clean (m) |

Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes (`taniteval/ci.py`; macro-F1 through the eval's own `paired` reducer). Tier: **T1-class open loop, DIAGNOSTIC.** Donor inputs are interventions, not driving.

## Controls (each must read its known value, or the grid is not reported)

1. **Identity:** donor = self (identity permutation) ⇒ every Δ is **exactly 0.0**, and the outputs are bit-identical to `forward_infer`.
2. **Reproduction:** the clean pass reproduces the banked own-code eval (S @50, T lat/lon F1, ADE6) of the same checkpoint on CPU. State the tolerance observed; a mismatch above 1e-3 stops the read.
3. **Structural zero (deliberate regression):** a copy of the model whose O nav token is replaced by a constant must read the O × nav cell as **exactly 0.0**, and a copy whose T scene path is fed a constant must read T × scene as exactly 0.0. This is what proves that a cell measures dependence and not the probe.
4. **Shuffle noise:** on s0, the whole grid at donor seeds 0 and 1. A cell counts as "listens" only if $|\Delta|$ also exceeds the two donor seeds' difference.
5. **Consistency with the banked shuffles (descriptive):** O × T code against `--tac shuffle` (+4.4 to +6.4 m), in sign and order of size.

## The reading (both seeds; committed before data)

A cell **listens** iff Δ > 0 and separated at BOTH seeds, and $|\Delta|$ exceeds the donor-seed difference (control 4).

| read | hypothesis (from the attention result, written before data) |
|---|---|
| **P-UPPER** | S's and T-lat's largest $n$ is **scene or nav** |
| **P-LOWER** | O's largest $n$ is an input **from above**, T code or S checkpoints, not the scene |

| outcome | condition | next (committed) |
|---|---|---|
| **DIVISION OF LABOUR SHOWN** | P-UPPER and P-LOWER at both seeds | the grid becomes the report's primary evidence for "different abstractions"; the attention video is kept as illustration only |
| **PARTIAL** | exactly one of them holds at both seeds | the report's claim is narrowed to the cells that hold, stated per cell. If O's largest cell is the scene, O re-derives the plan from the image instead of following the hierarchy: that is FLAT's question, and `PREREG_V8F_FLAT` decides it |
| **NOT SHOWN** | neither holds | the report states that the levels' input dependence does not differ as the attention suggested, and the abstraction claim rests on the oracle and donor reads alone |
| **SEEDS DISAGREE** | a hypothesis holds at one seed only | reported as such; seeds come before windows (`H-ESTIM-SEED-1`) |

### Q-NAV: does nav reach the trajectory through T, or around it?

Read O's ADE6 with the donor nav injected at **T only** and at **O only** (plus S only and all three, beside). The decisive quantity is the paired $D = \mathrm{ADE6}(\text{nav at T only}) - \mathrm{ADE6}(\text{nav at O only})$.

| outcome | condition | next (committed) |
|---|---|---|
| **THROUGH T** | $D > 0$, separated, both seeds | the answer to the PI's question is yes: the trajectory follows nav mainly through T's decision |
| **AROUND T** | $D < 0$, separated, both seeds | O follows nav directly, and T's nav reaction is not what conditions the trajectory. Next: pre-register an arm with nav REMOVED from O's inputs. That is an architecture change, so it needs the PI's go |
| **BOTH / NOT ESTABLISHED** | anything else | reported with both cells and their intervals |

⛔ No donor seed, metric, threshold or hypothesis changes after any LISTEN number is seen.

## Visual (committed form)

* **Heatmap:** rows S, T-lat, T-lon, O; columns scene, ego, nav, limit, S checkpoints, T code, tick ages.
  * Colour = $n(L,X)$, clipped to [0, 1].
  * Cell text = raw Δ with its unit, marked where it "listens".
  * Non-inputs are hatched grey ("not an input").
  * One panel per seed.
* **Beside:** four bars for Q-NAV.
