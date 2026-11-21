# BUILD_S1D: arm kind S1d in strat1_check.py (base a4357e79)

S1d = S1 exactly, except the trainer flag `--ctx-detach on` (S1 keeps `off`). Context heads train; their gradient does not reach the shared adapter. Seeds s0, s1, s10, s11 as the checker already allows. Trainer, launch_gate.py and models/__init__.py are untouched.

## Diff per site (stack/scripts/strat1_check.py, line numbers in the edited file; LF endings kept, 0 CR bytes before and after)
* 35-37: `KINDS` gains "S1d"; new `NAV_KINDS = (N1, S1, S1d)` and `CTX_KINDS = (S1, S1d)` replace the scattered `("N1","S1")` / `== "S1"` tests.
* 71-72 render: nav args use `NAV_KINDS`; ctx args use `CTX_KINDS` and `--ctx-detach` = "on" for S1d, "off" otherwise. S1 argv is byte-identical (sha256 pinned, see tests).
* 101-103 gate_problems (new branch): argv `--ctx-detach` and built lever `ctx_detach` must both equal the arm's value (on for S1d, off for every other kind). This is also a declared-vs-built check for the detach lever, which S1 did not have.
* 113-115 gate_problems (new branch, ctx-on arms): probe `adapter_grad_ctx` must be exactly 0.0 for S1d and > 0.0 for S1.
* 94/99 gate: nav source text and ctx-on use the new tuples.
* 133-135 moved_identity: S1d needs ctx_head keys that moved from their seeded init, as S1.
* nav_report_problems, mint (binding nav_report, G-CLOCK note), run_arm nav report path: `NAV_KINDS`.
* argparse choices come from `KINDS`, so `argv`, `gate`, `post` and the mint/run subcommands accept S1d.

## Per-check decision for S1d
* argv render: changed (detach on), everything else equal to S1.
* nav source / G-CLOCK-NAV report: unchanged from S1 (text nav).
* ctx_heads on, w_ctx, ctx_dim 41, vision-only, label sha, CTX group gradient > 0: unchanged (heads still train).
* `control_adapter_grad_detach_on == 0.0`: holds unchanged; it is a control that the trainer measures with detach forced on, for both arms.
* adapter_grad_ctx: adapted (0.0 for S1d, > 0 for S1; the trainer probe itself asserts this at lines 2315-2320).
* post moved-identity and geval (levers starting with ctx are compared; regression arm drops ctx_heads): unchanged.
* binding / token: the kind is bound; an S1 token is refused under kind S1d.

## Tests (run from tree/stack, CUDA off, tanitad resolved inside the export)
* New file stack/tests/test_strat1_s1d.py: 19 tests (4 real-checker checks render/gate/post/nav, 6 argv sha pins for C0/N1/S1 at s0 and s1 against the a4357e79 value, 1 kinds/seeds, 1 binding+token+run_arm, 7 mutations).
* Combined run with test_strat1_token.py and test_v8f_strat1.py: 73 passed.

## Mutations (each copy lives under the test's tmp_path, off the import path; each must turn its check RED, and did)
* kinds_drop_s1d -> render RED
* render_detach_off -> render RED
* ctx_kinds_drop_s1d -> render RED
* nav_kinds_drop_s1d -> nav RED
* gate_drop_detach_check -> gate RED
* gate_drop_adapter_grad_check -> gate RED
* post_drop_s1d -> post RED

## Manifest (all on F:, staged in the export, nothing committed)
* F:/TanitAD-scratch/strat1/s1d/tree/stack/scripts/strat1_check.py (edited)
* F:/TanitAD-scratch/strat1/s1d/tree/stack/tests/test_strat1_s1d.py (new)
* F:/TanitAD-scratch/strat1/s1d/s1d.patch (diff against a4357e79)
* F:/TanitAD-scratch/strat1/s1d/BUILD_S1D.md (this file)
* F:/TanitAD-scratch/strat1/s1d/orig/strat1_check.py (the a4357e79 copy, scratch)
