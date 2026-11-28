# Brief: build N1DH phase B1 (dev box, CPU): the code, its tests and its controls. No pod, no GPU, no training

You are a TanitAD build agent working for the Master Mind (MM). Read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip first:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.

Rules:
* Stage nothing, commit nothing, push nothing. Your deliverable is a PATCH against `bcf3ba3f` plus its records; the MM commits it to a new branch.
* No edits to protected files (launch_gate.py, stack/tanitad/models/__init__.py, anything under E:/Projects/TanitAD/stack, the sha-pinned v8f_t1_eval.py).
* No secrets.
* One pytest at a time. Gate each on dev-box free commit >= 5.5 GB (`F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60`). Check `(Get-PSDrive F).Free` before each run and stop below 4 GB.
* Keep TMP/TEMP and every pytest `--basetemp` under your folder; the REFe tests leak GBs into C: %TEMP% otherwise.
* Never wait more than about 4 minutes in one call; run long suites in the background.

## The spec (FROZEN; do not edit it)
`F:/TanitAD-scratch/n1dc/next/PREREG_V8F_N1DH.md`, sha256 f1df05bc (check it first).
* Read the FREEZE RECORD and section 1 (the lever, `file:line` at bcf3ba3f), 3 (admissibility), 5 (controls) and the closing MM review.
* The PI approved 2 arms (`cond`, s0/s1). Build `--aux-route off|cond` with `off` as the default, bit-identical to bcf3ba3f.
  * The `ce` mode in section 2 is NOT built. Refuse it in argparse, with a message naming this card.
  * The CE weight flag is required with `cond`.
* If a code fact contradicts the card (a `file:line`, the label's class set, where `use_t` is drawn), STOP that item. Write it in your handoff as an amendment for the MM, and do not decide it yourself.

## The template: how N1DC was built
* Brief: `F:/TanitAD-scratch/n1dc/BRIEF_BUILD_N1DC_B1.md`.
* Records: `F:/TanitAD-scratch/n1dc/build/b1/`: `HANDOFF_B1.md`, `REPORT_B1.json`, `mkpatch.py`, `n1dc.patch`, the mutation tables, `run_mutants.py`, `envrun.sh`, `run_chain*.sh`.
* Code: `git -C C:/Users/Admin/tanitad-push show bcf3ba3f` shows the commit that added `--o-code-skip`: the trainer flag, its FLAG_CHECKS / G-DVB entry, the model wiring, the study checker `strat1_check.py` entries and the tests.

Follow the same shape: export `bcf3ba3f` (`git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive bcf3ba3f stack taniteval`), patch it, test it, and emit the patch with a per-file sha256 list.

## Build (in this order; bank each step's output as you go)
1. **Lever.**
   * `route_head = Linear(1024 -> K)` on `h` (`v8f_mini.py:1077`).
   * The kinematic label stage, built as section 1 says (`maneuver_kinematics`, `factor_from_kinematics`), as a training label only.
   * The CE term in the loss, on own-code rows only: `_forward` emits its `use_t` draw as one bool output key, with no new RNG draw.
   * The conditioning: `h + route_emb(softmax(logits).detach())`, with `route_emb = Linear(K -> 1024)` zero-initialised.
2. **G-DVB:** a FLAG_CHECKS entry for each new flag (argv lever == built model value), plus the `strat1_check.py` entries, as `--o-code-skip` got them.
3. **Controls, each a named test or script with the value it must read:**
   * **Z0:** with `--aux-route off`, the state dict and every loss term are bit-identical to bcf3ba3f.
   * **Z1:** with `cond` at step 0, the plan-head output is bit-identical to `off`, because of the zero-init.
   * **REACH:** a non-zero gradient on `route_head` and `route_emb` after one step, and an exact 0.0 control on the CE gradient on a batch with no own-code rows.
   * **MUTATIONS that must go RED:**
     * delete the CE term;
     * delete the conditioning line;
     * remove the own-code mask.
   * **K-LABEL:** the label's class counts on a fixed real-data sample equal the counts from the label source called directly.
   * **K-LIM:** the script the card describes, built and run on CPU on a small real batch. Report only.
   * **R2:** CPU fp32 with one thread is deterministic across two runs.
4. **Suite:** the full `stack/` pytest on the patched tree against the bcf3ba3f base: 0 regressions, junit diff.

## Deliverables in `F:/TanitAD-scratch/n1dc/build/n1dh_b1/`
* `n1dh.patch`, with a per-file sha256 list;
* the tests and their logs;
* the mutation table;
* `REPORT_B1.json`;
* `HANDOFF_B1.md` (at most 1k tokens), listing any card amendment needed;
* `MANIFEST_SHA256.txt`.

Delete your own temp at the end, by literal absolute path, logged in `F:/TanitAD-scratch/n1dc/DELETIONS.log`.

Hand back in at most 250 words: the patch's files, every control's value, any amendment, and the suite result.
