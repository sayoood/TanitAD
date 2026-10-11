# P6 planner command mode: runner hop wired and proved end to end (2026-10-11, dev box, CPU, 0 pod)

**Finding (MEASURED):** the hop "runner sets `planner.driving_command`" did NOT exist for a command-mode checkpoint. `eval/refe_navtest_seam.py` set it only under `--sanitize-goal`, so `eval_checkpoint.py -> seam -> REFePlanner.infer` crashed at token 1 of any P6 checkpoint (`planner.py:670`, reproduced on the pre-edit seam with a tiny command ckpt). **Fixed:** `set_driving_command` (seam:78-105), used at seam:249. Supersedes `P6_IMPLEMENTATION.md` section 8 last bullet and `P6_IMPL_HANDOFF.md` item 9.
Evidence class: MEASURED (this work). Random-init ViT-S, 4 cameras: a **wiring** test; no claim about driving or quality follows.

## Hop table (all paths under the package)
| # | hop | file:line |
|---|---|---|
| 1 | eval launcher runs the seam (default ViT-L; the test passes `--backbone vits16`) | `eval/eval_checkpoint.py:96` |
| 2 | trainer writes `meta["nav_mode"]` only when not `goal` (full AND partial snapshots share `meta`) | `refe/train.py:1384-1385` |
| 3 | planner built from the checkpoint; `config_for_checkpoint` copies the cfg and sets `nav_mode` | `eval/refe_navtest_seam.py:196` -> `refe/planner.py:352` -> `refe/ckpt_io.py:132-134` |
| 4 | model builds `cmd_embed` and no `goal_freqs` | `refe/model.py:529,538` |
| 5 | source: NAVSIM agent input `ego_statuses[-1]["driving_command"]` (navsim v1.1 @ 3e8291b, int64 4-vector, 12,146/12,146 tokens length 4) read from the W3 export | `eval/refe_navtest_seam.py:249` |
| 6 | **NEW** runner sets `planner.driving_command`; refuses None / non-4-vector / non-finite; no-op for a goal-mode planner unless Amendment 8 | `eval/refe_navtest_seam.py:102` (helper 78-105) |
| 7 | `infer` branches on `cfg.nav_mode`; `_command_for` refuses None, returns the one-hot | `refe/planner.py:467-468`, `:664-671` (raise `:670`) |
| 8 | the one-hot rides the `goal` slot into forward; `cmd_embed` feeds the ego token | `refe/planner.py:474` -> `refe/model.py:763-771` |

Never derived from the logged future, the goal fix or any classifier (PI ruling P6-N). Report JSON gains `nav_mode` and `driving_command_set`; the seam npz is unchanged.

## Test: `refe/test_p6_planner_command.py` (run with the **driverl-eval** venv), `ZZP6PLAN_PASS 172`
6 checks = 156 assertions + 16 for the mutation harness. Tokens (log `2021.05.25.14.16.10_veh-35_01100_01664`): LEFT `025b657634505df3`, STRAIGHT `054743e97abd5b9c`, RIGHT `026e7b1e0e335625`; real frame zips, real nuPlan scenarios, the pairing asserted from the export itself. about 3.5 min in total including the mutations (MEASURED, one run).
| ask | result (MEASURED) |
|---|---|
| (a) command mode built | `cmd_embed` present, `goal_freqs` absent, `cfg.nav_mode=command`, base cfg stays `goal`; goal ckpt the reverse |
| (b) command reaching the model = NAVSIM's | forward pre-hook sees `[1,0,0,0]`, `[0,1,0,0]`, `[0,0,1,0]` = the export's one-hots; the seam subprocess records the same; seam poses vs in-process NAVSIM-command poses max 3.6e-7 m, vs any other command 0.085-0.212 m |
| (c) flip changes output | proposals move 0.085-0.224 m between any two commands; **zero control** exactly 0.0 for the same command spelled as list / int list / int64 array / float32 array and for a repeat (bitwise); all-zero == UNKNOWN one-hot |
| (d) missing refuses | fresh planner, reset-to-None, runner given no key / None / length 3 / NaN: all `RuntimeError`, planner command untouched |
| (e) goal ckpt unchanged | `driving_command` poisoned (None / LEFT / RIGHT / `[9,9,9,9]`): output diff exactly 0.0; helper returns False. **Pre-edit vs post-edit seam** (remote-tip source vs edited) on the goal ckpt: npz poses bytewise identical, `--record-inputs` identical, with and without `--sanitize-goal`; reports differ only in the two added keys |

## Mutations (each applied to a copy; the named check must exit nonzero) all RED
| mutation | check | RED message |
|---|---|---|
| delete `planner.driving_command = cmd` (runner) | `runner_hop` | RUNNER DID NOT SET planner.driving_command |
| same | `seam_subprocess` | seam rc!=0: RuntimeError ... set planner.driving_command |
| `_command_for` returns constant one-hot | `runner_hop` | model received [0,1,0,0] but NAVSIM said [1,0,0,0] |
| same | `flip_changes_output` | LEFT vs STRAIGHT moved the proposals by only 0.00e+00 m |
| drop `config_for_checkpoint` nav_mode hook | `built_in_command_mode` | config_for_checkpoint must return a COPY with the checkpoint's nav_mode (the planner load itself then raises: Missing key "goal_freqs") |
| `_command_for` None guard off | `missing_raises` | TypeError instead of RuntimeError |
| runner None guard off | `missing_raises` | Amendment-8 path did not raise |
| runner vector validation off | `missing_raises` | wrong length did not raise |
A first run had the runner None-guard mutation SURVIVE (redundant in command mode: the 4-vector check subsumes it); the guard only matters on the Amendment-8 path, so a check was added for that path and the mutation now goes RED.

## UNVERIFIED / limits
* **No real P6 checkpoint exists.** Not run: ViT-L, GPU numerics, the PARTIAL `snap_epochNNN.pt` format in command mode (its `meta` carries `nav_mode` by code reading, train.py:1385 -> 1515, not executed).
* `planner._command_for` maps a wrong-length vector to UNKNOWN silently (`nav_command.from_driving_command`); only the runner helper refuses it. `planner.py` was left unedited. (The "other agent's unstaged hunks" in P6_IMPLEMENTATION section 8 were D:'s stale index: D:'s planner.py equals the tip blob, Master Mind check 2026-10-11.)
* Command-mode checkpoints through other planner callers refuse loudly but are not made capable: the nuPlan closed-loop `compute_planner_trajectory`, `eval/proxy_eval_cache.py:90`, `eval/fan_speed_swap_probe.py:183` (both read `goal_freqs`).
* The training-time command source (`annotate_commands.py`, HANDOFF item 6) is a separate open item. The test needs the dev-box data plus nuPlan, so it is NOT in `launch_gate.SUITE` (`launch_gate.py` untouched): adding it to the pod suite is a Master Mind decision.
