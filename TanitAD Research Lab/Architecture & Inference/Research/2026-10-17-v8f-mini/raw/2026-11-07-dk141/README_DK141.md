# dk141: distance keeping (headway / time gap / TTC / DK violation rate) for any eval-141 run

Tip read from: `3a8d4761e81b8ab731ffc51157eb5331bb865b83` (agent/arch-inf-20260803). Nothing under `stack/` or `taniteval/` was edited. Nothing staged, committed or pushed.

## 1. What already existed (quoted from the tip)
| item | where |
|---|---|
| headway = min over horizon of gap (rig origin to lead REAR face, `gap = along - size_x/2`), corridor |lat| < 2 m, lead must be ahead | `taniteval/taniteval/lead_metrics.py:106-128` (`per_step_gap`, gap line :126), `:131-222` (`distance_keeping`, `nanmin` :176) |
| time gap = headway / v0, NaN below 0.5 m/s | `lead_metrics.py:177-178`, `MIN_SPEED_MPS` :52 |
| min-TTC from the plan's own closing rate, capped 30 s (censored), single admissible step = censored | `lead_metrics.py:179-191`, `TTC_CAP_S` :50 |
| LEAD-VIOL (plan tg < 1.0 s while human tg >= 1.0 s) + progress control (>= 0.90, clipped) | `lead_metrics.py:433,436,474-484,610-747` |
| TTC violation threshold 1.5 s | `stack/tanitad/rl/rewards.py:563`; `stack/scripts/r9_e2_distkeep.py:54` (rows `LON_dk_*` :23-29) |
| the family entry point accepts a `lead=` dict and computes distance_keeping | `taniteval/taniteval/four_families.py:448-483`; `v8f_mini_eval.families()` (`stack/scripts/v8f_mini_eval.py:75-95`) does NOT pass it |
| **a banked reader for exactly this grid** | `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/v8f_dk_read.py` (613 lines) + `RESULT_V8F_DK.md` + `raw/2026-10-22-dk/dk_read.json` (2026-10-22; M1/M2b x s0/s1 + P10) |
| DK-VIOL / DK-HW / DK-TG rows of the T1-kin reads | `raw/2026-10-26-t1-families/t1kf_read.py:7-9,49-51` (DK-VIOL = `lead_metrics.lead_viol` at-risk; DK-HW/DK-TG = abs error to the human) |

Label layer of the lead agent: the lead TRACK is the published PhysicalAI-AV `obstacle.offline` feature, built into the per-frame B1 block by `taniteval/tools/build_lead_block_b1.py` (147 v7.2 EVAL clips; K=10 banked block sha c0525943..., K=30 rebuild sha 7a879baa...). It is NOT the augmented v7.x label release, and NOT the v9 scalar `lead_valid / lead_gap` of `lead_eval139.npz` (`--lead-eval`, `v8f_mini_eval.py:615`), which only feeds the eval's `viol_LEAD` label-constraint check.

Premise check: `v8f_mini_eval.py run` itself does not compute distance keeping and the banked `eval_*.json` carry no such block (true). The READ of the saved npz did exist, but it hard-codes the arms M1/M2b x s0/s1 + P10, so B', the LoRA arm L or any other tag could not be scored without editing it.

## 2. What dk141.py adds
`dk141.py read --arm NAME=eval_<tag>.npz ... [--pair A:B ...]`: any arm name and any pair, on the 2,399-window grid, 2 s (K=10 block) and 6 s (K=30 block). It calls the sha-pinned `v8f_dk_read.read_dk` (no new metric) and adds: TTC-violation rate (min TTC < 1.5 s), plan tight-gap rate (tg < 1.0 s, any human), per-window DK flag file, paired deltas of the rates, per-window states with reasons (n per state; lost-lead counts). Plans ARE in the npz (`pred` [2399,30,2]), so **no stack patch is needed**.

## 3. Files
- `dk141.py` (reader + `controls`), `mutate_dk141.py` (14 mutants), `build_gt141.py` (one-time GT rebuild from tip code), `gt141.npz` (sha da38ce86...).
- `ref/` pinned inputs (sha256 in `ref_SHA256.txt`); `tip_export/` = `git archive` of `stack/tanitad` and `taniteval/taniteval` at the tip.
- `raw/DK141_READ.json`, `raw/DK141_TABLE.md`, `raw/dk141_perwindow.npz`, `raw/DK141_CONTROLS.json`, `raw/DK141_MUTATION.json`, `*.out` logs.
- Run for a new arm: `python -I dk141.py read --arm L=<path>/eval_L.npz --arm Bp=<path>/eval_Bp.npz --pair L:P10 --pair L:Bp --out raw/DK141_READ_L.json --table-md raw/DK141_TABLE_L.md`.

## 4. Caveats (all stated, none decided)
Tier T1-class open loop. Interval = another draw of EPISODES only (H-ESTIM-SEED-1); seed floors need two training seeds. 31.0 % of windows are LEAD (743 of 2,399, 77 episodes); NOT_STRAIGHT (937 / 931) is never scored. No distance-keeping bar is registered.
Instrument edge found by control C8: a plan with ONE admissible corridor step reads TTC censored at 30 s even when it rams the lead (headway ~0), `lead_metrics.py:190-191`; time gap and LEAD-VIOL still flag it.
