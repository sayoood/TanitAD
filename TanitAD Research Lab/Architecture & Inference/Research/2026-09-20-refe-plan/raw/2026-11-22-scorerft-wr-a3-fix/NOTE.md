# The SCORER-FT weight rule crashed again (A2), and the chain could not have read a successful result (MM, 2026-11-22)

## 1. The crash (package A2, `code_a9e7a004`)

**What happened.**
- After Cs posted (21:38:30Z), `after_f3.sh` ran `refe/weight_rule_rank.py` on T's argv (`ARGS_WR.txt`, with `--compile`). It exited 1 at 21:39:51Z (`weight_rule_rank_CRASH_pod_20261122T2139Z.log`, pod sha256 `10450ed94ba90f11`).
- The failing line was `measure()` line 194, the **second** `torch.autograd.grad` call (L_rank), with:
  `one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [16, 1925, 1024]] is at version 18; expected version 0`.
- The error is raised from `torch/_functorch/_aot_autograd/runtime_wrappers.py` (the compiled backward's `ctx.saved_tensors`).
- The first gradient, l_score, had succeeded. So the first backward through the compiled backbone changed a saved activation in place. Donated buffers were already off (the A2 fix), so this is a second mechanism with the same trigger: a retained `--compile` graph cannot be backpropagated twice.
- This is a CODE defect. No K2 / K2b outcome was produced, and none may be read from it.

**Fix (A3, this commit).**
- `measure()` now takes **one forward and one backward per gradient**, with no retained graph. That is the trainer's own pattern, which has run every training step on this pod.
- This is the lever the A2 note committed for this case ("three separate forward passes ... at three times the cost").
- The three forwards of one micro-batch replay the same input and the same RNG state (`fork_rng` + `manual_seed(REPLAY_SEED + k)`). The labels still come from the first forward's detached proposals.
- `replay_max_abs_diff` (score and traj) is reported per micro-batch and pooled. It is a control, not part of the rule.
- The rule itself is unchanged: card section 2.5 (8 micro-batches, seed 20261012, pooled norms, candidates 1/64 to 4, r(w), rho(w), K2, K2b). So are the five existing mutation targets. The result JSON gains `grad_passes`, `retained_graph`, `replay_seed` and `replay_max_abs_diff`.

**Tests** (`test_rank_loss.py`, on an export of tip `2005c212` with the fix):
- `--fast`: 147 checks pass (`rank_fast.log`). That is A2's 145 plus two new assertions in `weight_rule_smoke`:
  1. Every `torch.autograd.grad` the tool takes is recorded by a wrapper. There must be exactly `3 x 8` calls, none with `retain_graph` or `create_graph`.
  2. The three passes give `replay_max_abs_diff == 0.0` on CPU.
- Full run (`rank_full.log`): two new RED mutation arms.
  - "a retained graph again": l_score's `retain_graph=False` becomes `True`.
  - "pass 2 is not a replay of pass 1": pass 2 forwards `img * 0.5`.
- **What the CPU cannot show:** the compiled-backward mechanism. Inductor needs the pod. The decisive test is again the pod run. The design does not depend on it, because no gradient is taken twice from one graph.

## 2. A caller defect found while restarting: the chain could not read a SUCCESSFUL rule (built, tested, unreachable)

**The JSON parse.** `after_f3.sh` `sft_stage1` parsed the result as `d['status'] == 'OK'` and `d['rank_w']`. The tool writes neither key at top level: the decision is `d['rule'] = {status: SELECT | K2 | K2b | ERROR, rank_w, why}`.
- MEASURED on the tool's REAL output: `wr_json_probe.py` writes `wr_probe.json` (tiny CPU bank, synthetic labels, NOT A DECISION; top-level keys include `rule` and NOT `status` / `rank_w`).
- v3's parse raises `KeyError: 'status'` on a correctly shaped SELECT result.
- So had A2 run cleanly, the chain would have logged "JSON unreadable" and **skipped SCORER-FT anyway**.

**The banking check.** v3's `WR_BANKED` check counted `chain.log` lines AFTER the rule and grepped only the lines added after that, with no egress in between. It could never match: `WR_BANK_FAIL` is unconditional, and the JSON was never banked.

**Fix.** `after_f4.sh` (pod `/workspace/data/refe_p5c/after_f4.sh`) sources v3's helpers unchanged (`AF3_NO_MAIN=1`; it refuses unless v3's sha256 is `e5e9751bf055d697`). It replaces only the stage function, with `sft_stage1_v4`:
- It reads `d['rule']`, requires `SELECT`, and refuses a synthetic-label result.
- It logs the JSON's sha256 (the MM banks the JSON in the repo).
- It refuses a package A whose weight rule is not A3 (`GRAD_PASSES = 3`).

Parse checks:

| input | v4 | v3 |
|---|---|---|
| the real K2 JSON | refused (rc 1) | |
| a SELECT copy with real labels | `0.25` (rc 0) | `KeyError` |
| a SELECT copy marked synthetic | refused (rc 1) | |

Order: weight rule (A3) -> SFTC -> package B (6 h window) -> SFTT -> Fs -> handoff (C1r).

**Class.** "Built, tested, unreachable from its caller" (`CLAUDE.md`): v3's stub test exercised the chain against a hand-written JSON shape, never the tool's output. **Rule applied here: test a consumer against the producer's real output.**

## 3. What the MM did on the pod (2026-11-22, ~21:48Z)

- Fs's evidence and egress gate had passed, and its launch gate was running.
- The MM killed **its own** `after_f3.sh` (pid 210085) first, so that no fall-through could hand off to C1r. Then it killed the gate (222757, 222682).
- **Fs never launched.** The GPU was free from ~21:49Z until A3 shipped.
- Note: `/workspace/data/refe_p5c/MM_NOTES.txt`.
