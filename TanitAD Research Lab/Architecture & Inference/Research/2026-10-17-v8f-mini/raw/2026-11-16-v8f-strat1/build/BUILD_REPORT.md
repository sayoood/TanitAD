# BUILD_REPORT: STRAT1 (base cdbff971), 2026-11-16

Staged only: nothing committed or pushed. Working tree `F:/TanitAD-scratch/strat1/build/tree` (stack + taniteval code of cdbff971; file blob hashes checked against `git ls-tree`).

## Deviations and decisions (escalated)
1. **Existing-suite before/after counts are NOT delivered.** My parallel pytest runs exhausted dev-box memory (coordinator stop order); the in-flight results were lost and the follow-up single-process run was refused by the auto-mode classifier. The MM runs G-SUITE (cdbff971 vs tip, same environment). What was run: `test_v8f_strat1.py` 33 tests and `test_v8f_strat1_navbuilder.py` 4 tests, all passed (last full run of the first file: 31/31 before the 2 T-EVAL tests were added, which passed alone; the file was not re-run whole afterwards). The T-SNAP check (level / lead / otband snapshot tests at the new defaults) is UNVERIFIED.
2. **Peak memory** of my largest run: not measured (the instrumented run did not start). The parallel runs that caused the incident: 12 processes.
3. `tests/run_mutation_strat1.py` NOT built: each mutation arm is a test inside `test_v8f_strat1.py` (rewrites one source line, re-runs the same check, requires RED): off-equals-tip, fork_rng, hygiene, total line, detach, vision-only, isolation, nav overwrite, ctx attach, optimiser decay group, T-INIT prefix. Not mutated: T-CKPT (cfg drop is covered by the G-EVAL regression arm in `strat1_check.geval`), T-EVAL `real reads donor` (pinned by an equality test), builder `+ travelled` (G-CLOCK-NAV regression arm, run on real data).
4. `strat1_check.py` implements argv rendering, gate (G-DVB/G-LIVE read), post moved-identity, geval, hyg. NOT built: pilot timing, chain runner, token minting, pod preflight. G-SUITE and real-data 8-window G-EVAL parity are the MM's.
5. Coverage guard (LABEL_SPEC 4): valid share is the table validity BEFORE the twilight mask (twilight counts as valid, reported separately), threshold 0.98 over trained windows.
6. Two defects found by real data and fixed: (a) the builder's arc cumsum was poisoned by the NaN first two rows of `native_tracks.pt` (every countdown read 0); (b) `ctx_head_probe` needed a seeded global RNG (two unseeded loss calls differ by 1.4e-3). Both have tests / evidence.

## Built (repo path -> package path, sha256)
| repo path | package | sha256 |
|---|---|---|
| `stack/scripts/build_navtext_nav.py` | `F:/TanitAD-scratch/strat1/package/stack/scripts/build_navtext_nav.py` | `fdbe4d23683484852bc858200a34e59e753069de20701a3bea530c79af085a2d` |
| `stack/scripts/strat1_check.py` | `F:/TanitAD-scratch/strat1/package/stack/scripts/strat1_check.py` | `83bb986f655bea88645fd2b2b1c73b5dad1090da35a0e74e846872003e9f878a` |
| `stack/scripts/v8f_mini_eval.py` | `F:/TanitAD-scratch/strat1/package/stack/scripts/v8f_mini_eval.py` | `008f48b79fd11ca553d118d8e29ece6aea29d1dd013653ac23b1dc7fe2dad558` |
| `stack/scripts/v8f_mini_train.py` | `F:/TanitAD-scratch/strat1/package/stack/scripts/v8f_mini_train.py` | `d27893d3857da7d03789a25207bf372bb36dfefa756e6ab2ef2ca120543b97d3` |
| `stack/scripts/v8f_t1_eval.py` | `F:/TanitAD-scratch/strat1/package/stack/scripts/v8f_t1_eval.py` | `180044eb28246ba94a18366ed337f061ec659d6667496c0f88a4a325a0d65c93` |
| `stack/tanitad/data/v8f_mini_data.py` | `F:/TanitAD-scratch/strat1/package/stack/tanitad/data/v8f_mini_data.py` | `fc34869702848f525d618dfc8db60f1235646d8922c7ea8af95e70263e0ff744` |
| `stack/tanitad/models/v8f_mini.py` | `F:/TanitAD-scratch/strat1/package/stack/tanitad/models/v8f_mini.py` | `776643b52b0a3286ef60d12c7c9088960ea1ab7f15030b5d4c8722d17bd217eb` |
| `stack/tests/test_v8f_strat1.py` | `F:/TanitAD-scratch/strat1/package/stack/tests/test_v8f_strat1.py` | `38c5c435591e18ecf8501a93a924ea587a2f1694a07ceda67b69c20315a137b2` |
| `stack/tests/test_v8f_strat1_navbuilder.py` | `F:/TanitAD-scratch/strat1/package/stack/tests/test_v8f_strat1_navbuilder.py` | `d57257d8fe65583b8beb012cbe4670c252715cd7028ca2fd72b99c214f314a2b` |
| `stack/tests/v8f_mini_tip_snapshot_strat1.txt` | `F:/TanitAD-scratch/strat1/package/stack/tests/v8f_mini_tip_snapshot_strat1.txt` | `388f453a752d9c0cf20f04838d6c63d03ee7a9c0a7511401433d205bd91c9dd3` |

Edited files: v8f_mini.py (+ctx head, loss, groups, levers, 3 guards), v8f_mini_data.py (ctx_label_arrays, nav_text_arrays), v8f_mini_train.py (9 flags + FLAG_CHECKS, bad_ctx, optimiser groups ctx/ctx_nd, ctx_stage before the ego stages, ctx_head_probe, nav_source_probe, G-LIVE terms, checkpoint/summary), v8f_mini_eval.py and v8f_t1_eval.py (refusals, text nav, --nav-withheld, --nav-variant, --dump-nav, ctx_logits). New: strat1_check.py, build_navtext_nav.py, two test files, the tip snapshot. Single copies: all of the above exist only in this package/tree (not in git).
Artifacts (package/artifacts): `navtext_nav.npz` sha 03bd75bc3a3de17928681bedb8e46a48a891304de45d395c49cc876acf6e6160 (train + eval, from navtok_table 2836369e, pod native_tracks.pt, join_train md5 9ebc7a48, join_eval141 md5 532fcd64), report json, `strat_platform_by_clip.npz` sha b18705db6a2665e324a4ce4d1e7364c57305c15a34a770a5a6274ef1e2d148ee (train 1925/2647, eval 67/74, 0 unmapped; also in labels/). Single copy of the eval join source: E: research lab raw.

## G-CLOCK-NAV (real data, both outcomes pre-committed)
Train: countdown median |err| 6.481 m vs raw 6.963 m (n 38,909): PASS (margin 7 %, thin). Regression arm (+travelled) 7.323 vs 6.936: FAILS as required. Eval: 5.347 vs 5.716 (PASS); regression 6.212 vs 5.687 (FAILS). Rowclamp train 6,774 steps.

## Reachability table
| mechanism | flag/default | proof |
|---|---|---|
| ctx head built | --ctx-heads off | OFF==TIP bit-equal to cdbff971 snapshot (state dict, terms, grads, forward_infer); mutation unconditional head -> RED |
| head init isolation | fork_rng | non-ctx init + global RNG equal off vs on; mutation RED |
| ctx term in total | --w-ctx 0 | total moves by w x term (hand CE match); w=0 head grad exactly 0.0; delete total line -> RED; real data: 0.89728546 vs 0.89728504 |
| adapter coupling | --ctx-detach off | adapter grad 273.06 (off), exactly 0.0 (on control); drop detach -> RED |
| vision only / isolation | n/a | key-guard + source + perturb; p10f mutation RED; logits->S mutation RED |
| label staging | --ctx-labels "" | ctx_stage before comp/ego/dag/rcv (source order test), probe equals an independent table re-read on real data; attach-delete mutation -> ValueError |
| optimiser | --ctx-lr-mult 1 | groups ctx/ctx_nd hold exactly the head; head in decay -> refused |
| nav overwrite | --nav-source v9 | nav_source_probe: digest, n_text_ne_v9 276,392 (real), e_nav grad 6.23, zero-nav dckpt 1.42; delete overwrite line -> gate refusal |
| every argv names all nine flags | | `strat1_check.gate_problems` refuses otherwise; bad_ctx refuses each misdeclaration (tests) |
G-LIVE on real data: S1 groups_grad CTX 229.5; ctx terms country 3.19, weather 1.37, surface 1.14, illumination 1.34, period 1.61.

## Real-data CPU smoke (4 real train clips from the pod cache, real join/p10/tracks/labels, d=256, batch 8, 3 steps, no init ckpt, no DAgger bank, no --exclude)
C0, N1, S1: rc 0, declared_vs_built PASS, live PASS, finite terms (evidence/smoke). Not covered by the smoke: the M2b init checkpoint, DAgger bank, eval windows on real eval clips (2 eval clips pulled; manifests and md5 match, eval join/grid path not exercised).

## Pod readiness (read-only)
- `/root/data/cache_pod`: train 4,572 clips + eval 141 + native_tracks.pt (4,713 tracks incl. 141/141 eval). Format = ClipStore (clips/<id>.pt fp8, manifest.jsonl with sha12/md5); 6 pulled clips md5 equal the manifest and load in the trainer. train_sha256 from the pod manifest after --exclude 540:600 = 37f4f9ee (matches), excluded f6a1dee0 (matches).
- Pod join_train.npz md5 9ebc7a48 (canonical); p10_prior.pt present; no join_eval141 on the pod.
- Pod `/root/tad` is not a git repo; its v8f_mini_train.py md5 df3354ac and v8f_mini.py md5 098331d7 differ from cdbff971 (847f784a, 94f8a2e4): the pod stack is NOT cdbff971, ship the tree.
- Not touched: pod processes.

## Argv per arm (frozen base = OBD-CTL argv of the seed, box-2 paths mapped to the pod root)
`python stack/scripts/strat1_check.py argv {C0|N1|S1} {s0|s1|rep} --root <pod root> --out <dir> [--nav-npz P --ctx-npz P]` (rep = init 0, bank s0, dagger-seed 0, seed 2).
C0 = base + `--nav-source v9 --nav-text-npz "" --nav-text-sha256 "" --ctx-heads off --w-ctx 0 --ctx-labels "" --ctx-labels-sha256 "" --ctx-lr-mult 1 --ctx-detach off`.
N1 = base + text nav group (npz, its sha 03bd75bc... for the file as built) + ctx group of C0.
S1 = N1's nav group + `--ctx-heads on --w-ctx 0.5 --ctx-labels <strat_labels_table.npz> --ctx-labels-sha256 8055a321df6db1725ea17abde2999d48bcb68d43f1a480a69ed42dd0cf184eff --ctx-lr-mult 10 --ctx-detach off`.
The base argv file is read in place from the dagland frozen folder (sha not pinned: UNVERIFIED).

## Temp deletions
Logged in the hand-back message.
