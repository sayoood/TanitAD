# SCORER-FT trainer half: implementation report (2026-11-20)

Base: git tip `36b92a2b` (mirror `C:/Users/Admin/tanitad-push`, read-only, exported with `archive`, raw blob bytes). Nothing committed, pushed or staged. Deliverables are in `land/` at their repo paths; the repin list is `land/REPIN_TABLE.tsv`.
PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

## Files (build list 1, 2, 3, 6a)
| repo path | base blob @36b92a2b | new blob |
|---|---|---|
| `PK/refe/rank_loss.py` (new) | absent | `4dc2caab65060e0e86ec6ff5a86d7f663a214786` |
| `PK/refe/train.py` (edit) | `5f12576790bc7fb852d05f22abe3a256aca13c78` | `29ee68e5acbe6c2dbb2e3fc821be27527357565a` |
| `PK/refe/weight_rule_rank.py` (new, from `weight_rule.py`) | absent | `945bd4e83291e8953f3adf0e9d6e8b58db54f12e` |
| `PK/refe/test_rank_loss.py` (new) | absent | `0275ba587d4d0118ecc841c7bf52acb9cb952523` |

## Reachability trace: argv -> `.backward()` (file:line in the landed `train.py`; `grep -n` the quoted token to re-find)
| hop | where |
|---|---|
| argparse: `--rank-w` (default 0.0), `--rank-heads` (""), `--rank-ep-gap` (0.02), `--rank-gate-topk` (8) | `ap.add_argument("--rank-w" ...)` block, just before `--field-labels` |
| refusals before any build | `bad_rank = rank_refusals(a, cfg.n_proposals)` -> `return 4`, just before the P7.1 aux refusals; the function `rank_refusals` sits above `warm_start` |
| resume identity | `IDENTITY_ARGS` += 4 keys, `IDENTITY_DEFAULTS` += `0.0 / "" / 0.02 / 8` (old checkpoints read as rank-off) |
| object built (None when `--rank-w` is 0) | `rank = RK.RankLoss(a.rank_heads, ep_gap=a.rank_ep_gap, gate_topk=a.rank_gate_topk) if a.rank_w > 0.0 else None`, after the labeller is built |
| `rank` start event (built facts) | `{"event": "rank", ..., **rank.facts()}` after the aux event |
| forward (only `traj_leak` arm adds a second forward) | `score_rk` before `n_cand = ...`; stored in `cur["score_rk"]` |
| the term | inside `for P in done:` (online branch only): `l_rank = rank.term(score=..., labels=lab_t, msk=msk, batch=a.batch)` |
| added to the loss | `tot_ = tot_ + a.rank_w * l_rank` (skipped only under `REFE_RANK_MUTATE=drop`) |
| backward | the existing `(tot_ / a.accum).backward()`, unchanged |
| log fields | `rank_L, rank_w, rank_pairs_<head>, rank_sets_<head>` on the step row; `pick_v1, random_v1, best_v1, skill_v1` on the `onpolicy` event, all only when the term is on |

Proof it runs (not just exists), `test_rank_loss.py::trainer_reach`, real `train.run` on a tiny synthetic bank with the BCE zeroed and the trajectory term's gradient removed, so every gradient seen is L_rank's: non-zero on `score_head`, `score_dec`, `score_q_mlp`, `backbone.lora`, `scene_proj`; EXACTLY 0.0 on `traj_head` and `dec`; the same run without `--rank-w` reads EXACTLY 0.0 on all of them; `--rank-w 1.0` gives 2.0x the gradient of 0.5. Model-level (`model_reach`): the real forward, L_rank alone: `traj_head`, `dec`, `queries` EXACTLY 0.0, score_head rows C and DDC (and TTC unless named) EXACTLY 0.0, rows NC/DAC/EP non-zero, x0 control reads 0.0 everywhere.

## Results
`test_rank_loss.py` full (15 checks + 31 mutations): see `logs/test_rank_loss_full_patched.log` -> `ZZRANK_PASS 144 (15 checks green + 31 RED mutations: 25 source lines, 6 REFE_RANK_MUTATE arms)`.
Checks: hand-computed analytic case (L_ep = 0.8070888/2, an unlabelled set contributing exactly 0), loop-reference pair-set + value audit for every head, topk=64 / gap 1.5 / gap 0.02 / topk=1 identities, accumulation (B2 x accum2 vs B4 x accum1; gradient bit-identical in fp64, the VALUE is within 1 ulp 1.1e-16: summation order, so the spec's "bit for bit" holds for the gradient only), boundary semantics, gradient rows, model reach, 22 refusal cases, identity args, trainer reach, trainer env arms, trainer row fields, OFF-path identity, weight-rule decide, weight-rule smoke.
Mutations that went RED (31): delete the add line; constant weight; term never built; heads / top-k / gap not passed; detached logits (trainer and unit); refusals not called; identity keys dropped; OFF path building the term; EP safe condition removed; gate top-k removed; gate weight 1; own-count denominator; mask dropped; EP gap exclusive; safe uses >=; gate reads the EP logit; C joins the EP logit; sum not mean; 4 weight-rule mutations; and the env arms `drop`, `unsafe_ep`, `shuffle`, `head_swap`, `traj_leak` and a typo (`dorp`, which must fail the trainer build).

### Existing refe tests, pristine export vs patched (tokens and counts)
| test | pristine | patched alone | patched + the gate-side overlay below |
|---|---|---|---|
| `test_p6_levers.py --fast` | ZZP6_PASS 169 | RED (G-DVB census) | ZZP6_PASS 169 |
| `test_p7_head_wiring.py --static` | ZZP7HEAD_PASS 182 | ZZP7HEAD_FAIL 2 | see below |
| `test_traj_label_table.py --fast` | ZZTRAJLABEL_PASS 166 | RED (census) | ZZTRAJLABEL_PASS 166 |
| `test_ego_yaw.py --fast` | ZZEGOYAW_PASS 160 | RED (census) | ZZEGOYAW_PASS 160 |
| `test_p5c_live.py --fast` | ZZP5CLIVE_PASS 49 | ZZP5CLIVE_PASS 49 | same |
| `test_p7_integration.py` | ZZP7INT_PASS 225 | ZZP7INT_PASS 225 | same |

The three REDs are the designed tripwire, not a regression: `gate_checks.DVB_KINDS` must equal the parser's dests ("a new trainer flag without a G-DVB entry is refused", the PI rule), and four tests hardcode `len(dests) == 70`. That is the GATE half's work (build list 4, `gate_checks.py`: add `rank_w`, `rank_heads`, `rank_ep_gap`, `rank_gate_topk` to `DVB_KINDS` plus `rank_dvb_rows`). To prove 0 other regressions I overlaid, in a SCRATCH tree only (not delivered), the four DVB_KINDS entries and `74` for the count in `test_p7_head_wiring.py` / `test_ego_yaw.py`: all six tests then match pristine, except `test_p7_head_wiring` S6, a hunk census of `train.py` against a frozen pre-P7 fixture whose allow-regex lists the levers by name (P5-C, WP YAW precedent). My hunks mention `rank` / `RK.` / `SCORER-FT`; adding `|rank|RK\.|SCORER-FT` to that regex is the precedent-following fix (one hunk, a variable I had named `opv1`, did not match; renamed `rank_v1`). The full `test_p7_head_wiring.py` was NOT re-run after that rename (the dev-box mutex queue did not clear for 90 min); I re-ran its S6 computation alone (difflib of the final `train.py` against `fixtures/train_pre_p7_head.py.txt`, with the overlaid regex): 50 hunks, 0 stray. Its other 181 checks passed on the overlay tree before the rename.

### OFF path (`--rank-w` 0 or absent) against the pre-change trainer
`test_rank_loss.py --check off_path_identity --pristine <pristine refe>`: tiny CPU run, 3 steps x accum 2, subprocess `train.py` from each tree: every logged loss row (traj_L1, score, pos_L1, yaw_L1, six bce) and the final-weights sha256 identical. First run (before the cosmetic rename): `7eed2f6d6bcaa69a == 7eed2f6d6bcaa69a`, 3 rows identical. NOT re-run against pristine after the later identifier rename `opv1` -> `rank_v1` (a variable used only when the term is on; the mutex queue did not clear); the in-repo OFF check (explicit 0.0 == omitted) was green on the final `train.py` in the full 144-assertion run. Also: explicit `--rank-w 0.0` == flags omitted (same check), and a run without `--rank-w` writes no `rank` event, no `rank_*` field, no `pick_v1`.

### Weight rule (`weight_rule_rank.py`, item 3)
Implemented per spec 2.5 (8 micro-batches, seed 20261012, real labels from the arm's own captured `OnlineLabeller`, S and T groups, pooled gradients, 9 candidates 1/64..4, K2 / K2b, per-head pair counts and `dead_heads`, exit 0/5/4, `NOT_A_DECISION` stamping). The spec allowed a dev-box CPU smoke only with the synthetic labeller, so that is what ran (`weight_rule_smoke`: through the real `train.run`, 8 sets labelled, pairs > 0 for both heads, r and rho linear in w, status computed); it is NOT a decision and says so in its JSON. The real run needs the pod: C1's final, the P7 bank, the D9 metric cache, the labeller venv.

## Where the spec is wrong, thin or I had to decide (nothing redesigned)
1. **"Named without ep/gate => refused" cannot be exact.** argparse cannot tell `--rank-ep-gap 0.02` from the default. The refusal fires when the value differs from the default and the head is absent; a default value named without its head passes silently (harmless: unused). The gate half's template check (`scorerft_refusals`, argv set equality) is what catches it.
2. **`*_v1` monitor fields appear only when the term is on.** Spec 1.5 says "the build adds" them; emitting them on the OFF path would change today's logs, which the OFF-path rule forbids. Consequence: the control arm C has no `pick_v1/skill_v1`; its harness numbers come from the reads, as the card says. If the MM wants them on C too, it is a 6-line change that makes the OFF-path logs (not losses) differ.
3. **Accumulation "bit for bit"**: gradient yes (fp64), value 1 ulp (see above).
4. `--rank-w 0` with a non-default gap/top-k and no heads is refused too ("configured and never trained"), an extension of the spec's rule for heads; the weight rule substitutes `--rank-w 1.0` for its BUILD only when the argv has 0/absent.
5. The traj_leak arm needs a second model forward (the term cannot re-forward itself); it is inside `train.py` behind `rank.mutate == "traj_leak"` only.
6. `P["score"]`'s constant denominator is `--batch` (`a.batch`), not the micro-batch's own B, per spec; a short tail batch therefore contributes less, like the BCE term.
7. Pinned-file side effects the MM must re-pin: the `train.py` blob moved (any `PRE_..._TRAIN_SHA256`-style pin of the CURRENT trainer, tree manifests, launch-gate profile fingerprints: memory "pin edit moves profile fingerprints": run all launch_gate tests after landing); `gate_checks.DVB_KINDS` (+4) and the `== 70` counts in `test_p7_head_wiring.py` (S3), `test_ego_yaw.py:441`.

## Environment note
E: was at 0 bytes free when I started (exFAT 1 MB clusters: a 3.8 GB export of a ~100 MB tree filled it). I deleted my own failed export and worked in the session scratchpad on C: (NTFS). C: is at 13 GB free and E: at 7.8 GB free; neither is mine to sweep beyond the log below.

## Deletions log (literal paths)
Deleted (mine, literal paths, all re-creatable from git `36b92a2b`): the first failed export `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/impl_trainer/tree` (3.8 GB allocated on exFAT, the cause of E: reading 0 free), and under `C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/e07efd49-d158-4484-85f3-d8888196bc46/scratchpad/`: `tmp_pristine`, `tmp_patched`, `tmp_patched_g` (empty after the runs), `patch_train.py`, `run_existing.sh`.
NOT deleted (a queued job of mine still uses them; my attempt to stop it was refused by the permission classifier): bash PID 8068 = `scratchpad/final.sh`, waiting on `scorer_ft/pytest.lockdir` to run `test_p7_head_wiring.py --static` (patched_g), the pristine-vs-patched OFF-path check and the weight-rule smoke, writing `impl_trainer/logs/final_summary.txt` and two logs. When it has finished (or if you stop PID 8068), delete `scratchpad/pristine` (459 MB), `scratchpad/patched` (458 MB), `scratchpad/patched_g` (458 MB, includes the scratch gate overlay, NOT a deliverable), `scratchpad/tmp`, `run.sh`, `final.sh`. Its results, if they arrive, are informative only; the numbers above do not depend on them.

## Deliverable manifest
| artifact | lives in |
|---|---|
| `rank_loss.py`, `train.py`, `weight_rule_rank.py`, `test_rank_loss.py` | `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/impl_trainer/land/PK/refe/` only (ONE place; not in the repo, not staged: the brief forbids touching the tree; the MM lands them) |
| `land/REPIN_TABLE.tsv`, this report, `logs/` | same hold only |
