# R0c build: the `--epo-ref {base,armA}` lever, proved at zero A40

Research Lab agent (Sonnet), 2026-10-15. Built against the remote tip `d257647` (extracted to a short scratch path; the E: working tree is stale and was not edited). **Nothing is launched, staged or committed.** The Master Mind lands the patches below. R0c's A40 arm still waits for a PI decision (~3.5 A40-h, beyond RL-S1b's authorised budget, `PREREG_R0C_DRAFT.md` section 5).

Evidence classes: every number below is **MEASURED (ours, CPU stubs, logs in `logs/`)** unless a line says otherwise. Nothing was run on a pod or a GPU.

## 1. Headline

* **Built and proved:** the flag, the gate profile `refe_rls1c`, the G-DVB entry, and `test_rl_r0c.py` (`ZZTEST_PASS 26` = 1 unmutated suite of 81 checks + 25 RED mutations).
* **Integration needed (escalation):** 7 patches + 1 new file must be landed together (section 9). A **pod chain wiring for R0c is NOT built** (section 8).
* **One finding that changes how the draft reads** (section 7.1): the literal mutation `sxA.detach()` -> `sxA` does **not** give arm A an EPO gradient, because `epo_loss` already computes pi_SFT under `torch.no_grad()`. There are two independent guards. The mutation still goes RED (through the selector, the built-reference probe, the data event and the runtime reachability check), and removing **both** guards gives arm A a gradient and turns the A-zero check RED.

## 2. Counts (patched vs unpatched `d257647`; torch 2.11 unless stated)

| suite | unpatched | patched | note |
|---|---|---|---|
| `test_rl_r0c.py` (NEW) | - | **26** | also 26 under torch 2.7.1 |
| `test_rl_s1b.py` | 18 | **18** | also 18 under 2.7.1; two R0 pins now compared with the R0c edit reversed (section 6) |
| `test_rl_epo.py` | 42 | **42** | fixture only (section 6) |
| `test_rls1_chain.py` | 392 | **392** | needs `MODEL_REGISTRY.md` fixture |
| `test_launch_gate.py --static` | 658 | **659** | +1 = the existing loop that pins every registered EPO value to the parser default now covers `epo_ref`; one hard-coded G-DVB row count 14 -> 15 |
| `test_aux_s1_wiring.py` | 20 | **20** | one normalisation line (section 6) |
| `test_aux_g12_bf16.py` torch 2.11 | 7 | **7** | |
| `test_aux_g12_bf16.py` torch 2.7.1, `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported | 7 | **7** | re-run with the variable exported in the calling shell: `logs/patched_g12_t271_explicit_export.log` |

`test_rls1_queue.py` (not on the required list) cannot run on this Windows box: it fails identically on the unpatched tree (`logs/baseline_rls1_queue_NOT_RUNNABLE_on_windows.log`). It is therefore UNVERIFIED for this patch (section 8).

## 3. What changed

| file | change | patch |
|---|---|---|
| `refe/scorer_finetune.py` | `EPO_REFS`, `EPO_REF_DEFAULT = "base"`, `"epo_ref"` in `EPO_DESTS`; new `epo_reference(ref, sx0, sxA)` and `epo_reference_built(ref)`; the `--epo-ref` parser flag (`choices`, default `base`); value check in `epo_refusals`; **one changed line in `train_losses`**; `info["epo"]["ref"]`; docstring | `PATCH_1` (+34 / -3) |
| `refe/launch_gate.py` | `epo_ref: base` added to `RLS1_EPO_REGISTERED` (so R0, R0b, R1, R1b, R2 refuse `armA`); `RLS1C_*` constants; profile `refe_rls1c` (arm R0c, seed 0, `epo_ref armA` registered, `must_name = ("--epo-ref",)`, suite = s1b suite + `test_rl_r0c.py`); the `must_name` rule in `rls1_refusals`; `--arm-name` choice; `--epo-ref` in `AUXS1_EPO_FLAGS` | `PATCH_2` (+23 / -5) |
| `refe/gate_checks.py` | `"epo_ref": "built"` in `DVB_KINDS_SCORER`; `"ref"` in `SCORER_EPO_KEYS` | `PATCH_3` (+2 / -1) |
| `refe/test_rl_epo.py` | fixture `mk()` namespace gains `epo_ref="base"`; `smoke()` gains `extra=()` and `hook=None` (both default to nothing) | `PATCH_4` |
| `refe/test_rl_s1b.py` | R0 pins kept, two compared with the R0c edit reversed; one mutation snippet follows the changed line | `PATCH_5` |
| `refe/test_launch_gate.py` | `observed_rows == 14` -> `15` | `PATCH_6` |
| `refe/test_aux_s1_wiring.py` | `norm_row` drops `epo.ref` from the data event | `PATCH_7` |
| `refe/test_rl_r0c.py` | NEW, 631 lines | the file itself |

R0c's argv is in `R0c_ARGV.txt`: R0's banked token argv (reconstructed from `rls1_pod.sh arm_args`; **it hashes to the banked `24451b62...`**) with `/arms/R0` -> `/arms/R0c` and `--epo-ref armA` appended. R0c differs from R0 by exactly those tokens (test `A_R0c_differs_from_R0_by_the_arm_paths_and_--epo-ref_armA_only`).

## 4. Design

* **Where the reference is chosen:** inside `train_losses`, the single loss assembly that the training loop, the speed probe and the G11 probe all call. It passes `epo_reference(a.epo_ref, sx0, sxA)` to `epo_term` where it used to pass `sx0`. `epo_term`, `epo_loss`, `losses` and the reward code are byte-identical to R0's.
* **Default path:** `epo_reference("base", sx0, sxA)` returns `sx0` itself (the same object), so `--epo-ref base` is bit-identical to the pre-patch trainer by construction, and the test proves it end to end (section 5, E).
* **armA:** returns `sxA.detach()`, arm A's logits of the same micro-batch. Guard 1 is that detach; guard 2 is `epo_loss`'s existing `with torch.no_grad(): logpi0 = epo_policy(sx_ref, ...)`.
* **G-DVB reads the BUILT reference:** `epo_reference_built(ref)` calls the selector on two sentinel tensors and names what it returns (`base`, `armA`, or `other` for an attached, un-detached armA). `main()` writes that into the data event (`epo.ref`). The gate compares the argv's `--epo-ref` with it. A selector that returns the wrong thing or leaks the graph is therefore visible to the gate, not just to the unit tests.
* **A flag without a G-DVB entry is refused by the gate itself:** this was already true of `check_dvb_scorer` (`unknown_flags`). The test demonstrates it on the real flag: with `epo_ref` popped from `DVB_KINDS_SCORER` the verdict is FAIL ("trainer flags with NO declared-vs-built entry ... epo_ref"). The converse also holds: a `built` entry nothing observes (`"ref"` missing from `SCORER_EPO_KEYS`) is FAIL.
* **Profile semantics:** `epo_ref` is registered by *effective value*. `armA` on R0/R1/R2 (`refe_rls1`) or R0b/R1b (`refe_rls1b`) resolves to a value other than `base` and is refused; on a control it is refused twice (`must NOT carry`, and the trainer's own "levers named without --epo" rule). R0c requires `armA` by effective value **and NAMED** (`must_name`), because the parser default `base` would otherwise reproduce R0 and pass unnoticed (the draft's section 3 item 5).
* **"No arm A in the run":** not a reachable state of this trainer. Arm A is the model's own scorer, always built and always trained (`optA`, `scorer_finetune.py:1361`). The reachable unsafe states are refused instead: `--epo-ref armA` without `--epo` (tested at the trainer, at the gate and through the real `main()`, which returns 4 before touching the disk), a value outside `{base, armA}` (argparse `choices` AND `epo_refusals`, each tested alone), and arm B whose non-EPO loss is not plain BCE (the existing "plain BCE" refusal, re-tested with `armA`), because only then does A differ from B by the EPO term alone.

## 5. Proofs (all in `test_rl_r0c.py`)

**Reachability, `file:line` at each hop** (patched `scorer_finetune.py`):

| hop | where |
|---|---|
| argparse flag | `:575` `--epo-ref` (choices `EPO_REFS`, default `EPO_REF_DEFAULT` `:179-180`) |
| refusals before any disk write | `:591` `bad_cli = epo_refusals(a, ap)`; the value check `:340` |
| namespace -> built fact | `:706` `"ref": epo_reference_built(a.epo_ref)` (the data event G-DVB reads) |
| arm A's logits | `:728` `sxA = model(...)` in `fwd` (arm B `:737`, base `:739`) |
| flag -> selector, inside the loss assembly | `:798` `epo_term(epo_reference(a.epo_ref, sx0, sxA), sxB, ...)` in `train_losses` (`:792`); selector `:194` |
| selector -> `epo_loss` | `:790` `epo_loss(sxB, sx0, ...)` in `epo_term` (`:780`); its parameter is still *named* `sx0` and receives the selected reference |
| reference -> policy, under guard 2 | `:225-226` `with torch.no_grad(): logpi0 = epo_policy(sx_ref, ...)` |
| KL -> loss | `:228` `kl = (logpi0.exp() * (logpi0 - logpi)).sum(1)`; `:231` `return lam_pg * (-J + beta * kl).mean()` |
| loss -> B only | `:799` `lB = lB + ep`; the loop `:1379-1380` `train_losses(...)` then `((lA + lB) / a.accum).backward()`; `optB` `:1362` |

The static hops are AST checks (`R_hop_*`). **The runtime hop is checked in the real `main()`**: a hook wraps `epo_reference` and `epo_loss` and records every call. In the `armA` run every selection is not `sx0`, equals arm A's logits bit for bit, carries no graph, and the object `epo_loss` receives as `sx_ref` **is** the selected object (identity), and at least one selection differs from the base once A has moved. In the `base` run the selection is `sx0` itself.

**Identity at step 0** (A = B = base; the real `train_losses` exec'd from `main()`'s source; 6 columns and 7 with the lane head): `base` and `armA` give bit-identical `lA, lB, listnet, EPO term`, a KL of **exactly 0.0**, and a bit-identical, non-zero gradient on B; arm A has no gradient. The zero-weight control (an explicit `lam_pg = 0.0` is never short-circuited) reads exactly 0.0 on B under `armA`.

**Non-zero where it must be:**
* A and B moved apart (A by one AdamW step on its BCE, B by one step on a different micro-batch; A != base, B != A): the KL-term gradient w.r.t. B is non-zero for both anchors, **differs** between them, and the KL values differ.
* The literal "one B-only update" scenario of the draft (section 3 item 3): A has not moved, so **A is the base**, and the two anchors give a **bit-identical** gradient. They differ only once A has moved.
* Through `train_losses`: `torch.autograd.grad(ep, A.parameters())` is `None` for every parameter (no path), `module_grad_abs(A)` reads **exactly 0.0** on all three scorer modules from the EPO term, B is live on all three, and the control `lA.backward()` reads non-zero on A (the probe can read non-zero).

**End to end through the real `main()`** (CPU stubs for the GPU parts only; 7-wide native head like u400; 25 updates):

| check | result |
|---|---|
| preflight G1-G11 PASS for `base` (default), `base` named, `armA`, `armA` at weight 0, `base` at weight 0, and `armA` with 6 columns | all PASS; G11 `controls_exactly_zero` true, `kl_step0 == 0.0` under `armA` |
| data event `epo.ref` | `armA` / `base` as declared (and in the preflight's data event) |
| control arm A, `base` vs `armA` | all weights **bit-identical**; `bceA` rows equal; `epo_kl` rows differ |
| arm B, `base` vs `armA` | differs; and B has left A |
| `armA` at `--epo-lam 0` | B == A exactly (K1 holds under `armA`) |
| default vs explicit `--epo-ref base` | weights and every logged row bit-identical |
| **`--epo-ref base` vs the PRE-PATCH trainer** | arm A and arm B weights **bit-identical**, every row of `sft.jsonl` and `preflight.jsonl` identical (minus wall-clock, temp paths, and the new `epo.ref`), `PREFLIGHT_PASS` gates identical |
| `--epo-ref armA` without `--epo` | rc 4, `REFUSED`, no output directory |

**Pre-patch trainer without a copy of it:** the test embeds the 8 edits of the patch as (after, before) text pairs (generated by `difflib`). Reversing them on the patched `scorer_finetune.py` gives the **`d257647` blob `d34d5cf6`** exactly (`P_reversal_is_the_pre_patch_trainer`), and that reconstruction is the "pre-patch trainer" the end-to-end run compares with. The only functions whose text differs are `epo_refusals`, `build_parser`, `main`, `main.train_losses` plus the two new ones (`P_only_the_named_functions_differ_from_the_pre_patch_trainer`); every other flag of the parser is unchanged and `--epo-ref` is the only new one.

## 6. Pins: what moved, and why

`test_rl_s1b.py` pins 13 code segments to the bytes R0 trained with. **Two had to change**, because the lever is, by its definition, an edit of them:
* `epo_refusals`: a 2-line tail (the value check).
* `main.train_losses`: one line (`epo_term(sx0, ...)` -> `epo_term(epo_reference(a.epo_ref, sx0, sxA), ...)`).

**The R0 hashes in `R0_SEGMENTS` were not edited.** `P_pinned_*` now hashes the segment with the R0c edit **reversed** (`R0C_REVERSALS`, `undo_r0c`), so any other change to those two functions still breaks the pin. The other 11 pins are untouched and compared as before. `base` is bit-identical to R0's behaviour by the end-to-end comparison above and by `P_R0_pin_*` in `test_rl_r0c.py`.

Other test edits, each additive or a fixture update, none lowers a count:
* `test_rl_s1b.py` mutation `sf_dose_reaches_control`: its snippet is the `train_losses` line, which changed; the mutation is the same (arm A gets its own EPO term), now anchored on the new text.
* `test_rl_epo.py`: `mk()` builds a namespace for the exec'd closures, which now read `a.epo_ref`; `smoke()` gets two optional arguments used by the new test.
* `test_launch_gate.py`: the scorer G-DVB observes one more row (`epo.ref`): 14 -> 15.
* `test_aux_s1_wiring.py`: it compares a pre-aux trainer fixture with the edited trainer row by row; the data event of an `--epo` run gained `epo.ref`, a deliberate difference, so `norm_row` drops that one key. Everything else is still compared bit for bit.

## 7. Mutations (`test_rl_r0c.py` prints the full table; each snippet must occur exactly once)

25 mutations, all RED. The pins fire on any edit of the patched text, so the table lists the behavioural catchers; no mutation relies on a pin alone.

| mutation | what catches it |
|---|---|
| `sxA.detach()` -> `sxA` (**guard 1 alone**) | selector check, `built` probe reads `other`, data event, runtime reachability. **Arm A's gradient check stays GREEN** (guard 2 holds on its own), as does the bit-identical control A |
| `torch.no_grad()` on the reference removed (**guard 2 alone**) | only the static hop check. **Behaviourally equivalent by design**: arm A's gradient check, the identity and the control all stay GREEN (guard 1 holds on its own) |
| **both guards removed** | arm A **receives an EPO gradient**: the A-zero check, both identity checks and the zero-weight control go RED |
| `sxA.detach()` -> `sx0` | the step-0 identity checks stay GREEN (A = B = base), the "A and B apart" checks go RED, as the draft predicted |
| default changed to `armA` | refusal, data-event, base-reachability and base-vs-armA checks; at the gate every `refe_rls1*` arm |
| `epo_ref` removed from `EPO_DESTS` | `armA` without `--epo` is silently accepted: the refusal checks and the real-`main()` refusal go RED |
| `train_losses` bypasses the selector (the old line) | **the "built, tested, unreachable" case:** the AST hop, the A-and-B-apart gradient, and the runtime reachability go RED. The gate's G-DVB would still read `armA` here (it observes the selector, not the call), so this one is caught by the suite, which runs inside the token as G-SUITE |
| `train_losses` reads a constant instead of the flag | same checks |
| data event echoes `a.epo_ref` **and** the selector returns `sx0` | the data-event check stays green (an echo hides the break from G-DVB) and the runtime reachability check catches it. This is why the event records the built reference, not the namespace |
| data event hard-codes `base` / drops `ref` | static and runtime data-event checks |
| value check in `epo_refusals` removed / argparse `choices` removed | each caught on its own |
| `built` probe ignores the graph **and** the detach is removed | the selector and runtime reachability checks (the probe alone is an equivalent mutant: it only matters when the detach is also missing) |
| 11 gate mutations (`epo_ref` not registered on the old arms; R0c registers `base`; `must_name` emptied or its rule removed; R0c profile without its registry or as a control; suite without the test; aux profile missing the flag; `--arm-name` choice missing; G-DVB entry removed; G-DVB ref unobserved) | the `G_*` checks, named in the log |

## 8. UNVERIFIED, and what remains

* **No pod, no GPU, no bf16.** Everything above ran on CPU with the stubs of `test_rl_epo.smoke` (the real `main()`, stub model and bank). The claim that sxA == sxB bit for bit at step 0 on the A40 (so G11's `kl_tol 0.0` holds under `armA`) is INHERITED: R0's banked preflight reads `max_B_vs_A 0.0` and `kl_step0 0.0` (`raw/2026-10-14-rls1-r0-kill/preflight.log`), which implies it, but R0c's own `--preflight` on the A40 is the measurement. It is a pod step of the launch, not something this build can show.
* **The full `launch_gate.py run` (mint a token) for `refe_rls1c` was not exercised** (it needs a pod preflight log and the pod key). Exercised instead: `profile_refusals` for every arm, the CLI `refusals` command, `check_dvb_scorer` on the profile, and the suite membership.
* **The pod chain is NOT wired for R0c.** Missing: `rls1_pod.sh` (`arm_args` case `R0c`, `arm_profile` -> `refe_rls1c`, the control rule: R0c is read against its own arm A and the banked C0), an `eval/rls1c_arms.json` (the argv in `R0c_ARGV.txt` is the content), and the matching cases in `test_rls1_chain.py` / `test_rls1_queue.py`. `test_rls1_queue.py` cannot run on Windows even unpatched. This is the next job before launch; none of it needs the A40.
* `SHIP_MANIFEST.json` must be regenerated at landing (`launch_gate.py ship-manifest`): 7 files change and `test_rl_r0c.py` is new and is in `refe_rls1c`'s G-SUITE.
* Torch 2.7.1 was run for `test_rl_r0c.py`, `test_rl_s1b.py` and `test_aux_g12_bf16.py`; the other four suites ran under 2.11 only.
* **The scientific question is untouched.** This build proves the lever is wired, not that it helps. The draft's own caveat stands: no CPU test discriminates the reference choice (over 50 simulated updates A has barely left the base), so only the A40 arm does.
* If the trainer moves before landing, `R0C_FILE_REVERSALS` (hence `P_reversal_is_the_pre_patch_trainer`) must be regenerated from the new base blob; the recipe is the header comment of that table. The patches apply to `d257647`.

## 9. Findings for the Master Mind

1. **Two guards, not one** (section 7): the draft's mutation 4 needs both guards removed to give arm A a gradient.
2. **The draft's section 3 item 3 needs one word changed:** "after one B-only update" leaves A at the base, so the two anchors are identical; the gradients differ once A and B have both moved. The test covers both scenarios.
3. **`epo_kl` changes meaning under `armA`:** it logs KL(pi_A || pi_B), not KL(pi_base || pi_B). The draft already says so (M2); `rls1_checks.py` only checks it is finite, no kill rule reads its value.
4. **Naming debt, left on purpose:** `epo_term`'s first parameter is still called `sx0` and its docstring says "the base as pi_SFT". Changing it would move a third pinned segment for no behaviour. The new flag's help, `epo_reference`'s docstring and the module docstring say what it now receives.
5. For the extended (non-native) 7-column head, `armA` is *more* exact at step 0 than `base` (A and B share the 7-row GEMM, so KL is 0 where the base-anchored KL needs `kl_tol 1e-2`). Not relevant for u400, which is native.

## 10. Deliverable manifest

All files: `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-15-r0c-build/` (working tree of the E: checkout, **unstaged**; also reproducible from the scratch tree `C:/Users/Admin/AppData/Local/Temp/r0c/...`, which holds the patched copy, and `.../r0c_fresh/...`, a fresh `d257647` extract with the patches applied for real). The patch set was proved on that fresh extract: `patch -p1 --dry-run` clean for all 7, applied for real, every patched file **byte-identical** to the scratch copy, and `test_rl_r0c.py` re-run there (26). Apply from the package root: `for p in raw/2026-10-15-r0c-build/PATCH_*.diff; do patch -p1 < "$p"; done; cp raw/2026-10-15-r0c-build/test_rl_r0c.py refe/`.

| artifact | where |
|---|---|
| `NOTES.md` (this file) | repo working tree, unstaged |
| `PATCH_1_scorer_finetune.diff` ... `PATCH_7_test_aux_s1_wiring.diff` | repo working tree, unstaged |
| `test_rl_r0c.py` | repo working tree, unstaged (to be copied to `refe/`) |
| `R0c_ARGV.txt` | repo working tree, unstaged |
| `logs/` (baseline and patched run of every suite, patch dry-run and apply logs) | repo working tree, unstaged |
| the patched copy (full) | scratch only: `C:/Users/Admin/AppData/Local/Temp/r0c/` (**one place**; the patches reproduce it from `d257647`) |
