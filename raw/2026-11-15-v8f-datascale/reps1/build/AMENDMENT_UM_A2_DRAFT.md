# AMENDMENT UM-A2 (DRAFT, not frozen): the seed-1 replicate pair UE_s1 + UM_s1

Status: DRAFT written by the builder agent, 2026-11-19. The MM freezes and hashes it. Nothing here has been run on the GPU.
Authority: the PI's go of 2026-11-19 ("Both, N1D8 first"; this pair is the second item, about 8.2 LoRA pod-h, ESTIMATED).
Parent: `CARD_DSCALE_MIX.md` (sha256 0bc241f5), `MM_RULINGS_DSCALE_MIX.md`, Amendment UM-A1 (sha256 b2b2c4f0). Read plan source: `MM_READING_DSCALE_MIX.md` section 7 item 1 and card section 5.6 ("UM_s1 and UE_s1 (seed 1), about 2 x 4.1 pod-h (E)").

## 1. The one change

* Seed 1 for both arms. `--seed 0` becomes `--seed 1`, and `--out` becomes `runs/UE_s1/A` / `runs/UM_s1/A`. Nothing else in either argv moves.
* Proof (MEASURED on the pod, `blockreps1/PAIRING_PROOF.json`, written by `gen_argv_reps1.py`): `UE_s1` vs `UE_s0` and `UM_s1` vs `UM_s0` differ in exactly `--seed` and `--out`; `UM_s1` vs `UE_s1` differ in exactly `--out` plus the three `--tac-src-*` flags. Eight deliberate regressions (a second lever, a changed seed, a changed tac flag, a wrong out path, a removed flag) are each refused by the same function.
* The gate checker is a copy, `blockreps1/study/dscale10k_check.py`, with arms `UE_s1` and `UM_s1` added to `ARMS` and the `UM_s0` branches generalised to `MIX_ARMS`. No bar, pin or list is changed (`patch_checker_reps1.py` holds the 12 asserted substitutions; the seed-0 arms behave as before).

## 2. Init source of each arm (from source)

* Neither arm loads a checkpoint. The argvs carry no `--init-ckpt` (`argv/UE_s0.A.json`, `argv/UM_s0.A.json`). Both start from a seeded random init, as U_s0 and B_s0 do.
* The trainer seeds, then builds the data, then builds the model: `v8f_mini_train.py:3248-3249` (`torch.manual_seed(a.seed)`, `np.random.seed(a.seed)`) and `:3290` (`M.V8FMini(cfg, prior, mu, sd)`) in `$D/tree` (cdbff971, UE_s0's tree); `:3515-3516` and `:3557` in `$D/tree_um` (648d4ca3, UM_s0's tree).
* So UE_s0 is NOT initialised from U_s0. Its step-0 state equals U_s0's by construction (same seed, same code path), checked as C-INIT in `gate/init_b3.json`; UM_s0's equals UE_s0's and U_s0's in `gate/init_um.json`.
* s1 mirrors this: seed 1, no checkpoint. The pod records `gate/init_ue_s1.json` and `gate/init_um_s1.json` compare UE_s1's step-0 state with U_s1's (`gate/init_s1.json`) and UM_s1's with UE_s1's and U_s1's. The pair therefore shares its init and differs from the s0 pair by the seed alone.
* Trees: each arm runs on the tree its seed-0 twin used (UE_s1 on cdbff971, UM_s1 on 648d4ca3), as the s0 pair did. The s0 pair itself is on two commits; 648d4ca3 is a child of cdbff971 with the tac-src lever off by default. Putting UE_s1 on 648d4ca3 as well is the alternative; it is the MM's call and needs a different token set.

## 3. Unchanged

* Steps 61,568 (4 x S_U of `dry/U_s0.A.dry.json`, asserted by the chain and by C-STEPS), batch 64, data, `--fit-join`, grid, flags, the 4,512-id list (sha256 37f4f9ee), `--tac-src-w 0.0`.
* Stage A only, no stage B. Same private HF repo and `<arm>/A/` layout as UE_s0 and UM_s0 (the banker is arm-agnostic).
* K-TACSRC bars as written, including `tac_src_empty_batches` <= 5 (card line 266). The chain's `um_checks post` still FAILS on a miss. Amendment UM-A1 is a proposal to the PI and is not applied here.
* MEASURED before launch (CPU replay of the trainer's own sampler, `blockreps1/replay/um_empty_replay_s1.json`, same method as UM-A1): UM_s1 replays **7** empty batches (sizes 1, 2, 2, 1, 1, 2, 3; 1,191 partial batches; rows listed 1,223,681, other 2,708,912). 7 > 5, so as written UM_s1 is BLOCKED (reach) at its post check, as UM_s0 was (10). The slot is open for the MM to rule: adopt UM-A1's exact-replay test for both seeds, or keep the bound as written. The chain banks UM_s1's checkpoint before the post check, so a FAIL strands nothing.

## 4. How the s0 and s1 readings combine

* The card has no combination rule. It says only: "ONE seed. No label of this card is quotable as a lever effect until the replicate in 5.6 exists" (section 5.2) and lists "UM_s1 and UE_s1 (seed 1)" as the replicate for UM - UE (section 5.6). The outcome table's next steps name "the replicate pair first" (DILUTION-PARTIAL, DILUTION-CONFIRMED rows). Nothing says whether the seeds are pooled, whether both must separate, or which floor applies.
* SLOT FOR THE MM: [rule to be written before any UM_s1 or UE_s1 number is read].
* Facts for that ruling: the existing F_cell comes from the 19,164-step B pair (B_rep - B_s0) and is applied at 61,568 steps; the card says no 61,568-step union floor exists. UE_s1 - UE_s0 is a replicate pair on the union at 61,568 steps and would give one. Card 5.6 assigns that role to UM_rep (seed 2).

## 5. Cost and blockers

* About 2 x 4.1 pod-h ESTIMATED (UE_s0 and UM_s0 loop times are in their `summary.json`). Launch needs the MM, the gate tokens minted by `blockreps1_prep.sh` (done by the builder if the prep reads DONE), and the ruling in section 3 on the empty-batch bar.
