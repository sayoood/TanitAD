# v7F merge — the COMPLETE v7F change set: R1 + R4 (`v7f_r1r4`) + R3 (`v7f_r3`) + the THREE R2 nav fixes + R6 `--strategic-off`

**2026-09-27, TrainingFlyWheel (P4). Built on the R1/R4 stream's copy of origin tip `c36b6ddd`; CPU only; STAGED,
never committed.** ⭐ **This package SUPERSEDES every `code/fix/` file of `v7f_r1r4/` and `v7f_r3/`** — land v7F from
here only (`LANDING_READY.txt` lists the stream copies as superseded, never mapped). The two streams wrote DIFFERENT
new files at the SAME path (`stack/tanitad/train/declared_vs_built_v6.py`: same registry name, incompatible reader
signatures — a naive union raises `TypeError` both ways); the file here is the UNION (§0).

## 0. Integration of R3 (2026-09-27 ~15:50)
- `train_v6_staged.py`: 3-way `git merge-file` in LF space (base = the tip blob, ours = R1/R4 + R2 + R6, theirs = R3's
  fix). **3 conflicts, all of one shape** — both sides adding an independent block at the same anchor (a dry-run report
  block, a `dry_run.json` key, a `config.json` record) — resolved as BOTH blocks (`raw/resolve_r3.py`, which refuses
  unless exactly 3 conflicts, no surviving marker, the file parses, and 10 per-stream fingerprints are present).
- `declared_vs_built_v6.py`: ONE registry, ONE reader signature `(stack, args, weights_in_force)` (the R1/R4 readers
  wrapped); `check`/`refuse_on_mismatch` (R1/R4 API, build time) run the levers that need no loss weights;
  `check_v6`/`refuse_on_mismatch_v6` (R3 API) run ALL levers + the parser coverage of all 7 new dests.
- ⛔ **EOL follows each tip blob:** `tests/test_v6_effective_weights.py` is stored CRLF in the repo (602 CRLF / 0 LF);
  it is banked CRLF. ⚠️ R3's own `code/fix/stack/scripts/train_v6_staged.py` is CRLF while the tip blob is LF — landing
  that copy would have flipped every line of an 11k-line file; the merged file here is LF.
- **Tests on the fully merged code:** every stream's suite together — `test_tactical_label_reach_v6.py`,
  `test_v6_effective_weights.py`, `test_v7f_r1r4.py` (bit-identity baseline on), `test_v7f_r2_nav_fixes.py`,
  `test_v7f_r6_strategic_off.py`, `test_nav_v6stack.py`, `test_v6_stage_init_introduction.py`: **178 passed, 0 failed**;
  the 8 skips were R3's bit-identity tests (need `TANITAD_R3_REF_TRAINER`) — run with the tip trainer as reference:
  **39/39** ⇒ with EVERY new flag off, the merged trainer equals the tip trainer on the loss, every term and the RNG
  stream. Diffs: **10/10** `git apply` onto the tip blobs and reproduce the banked files byte-for-byte.
- ✅ **Full 107-file v6 suite on the COMPLETE merge** (+ the R2, R6 and R3 test files; bit-identity baselines on):
  **2,655 passed, 44 skipped, 8 failed, 24 errors** (`raw/pytest_v6_related_full_merge_tail.txt`) — the failure set is
  IDENTICAL to the R1/R4 stream's baseline, test id for test id (`raw/failset_full_merge.txt`: 32 = 32, none new, none
  missing), and the pass count reconciles exactly: 2,606 (§5) + 10 (R6) + 39 (R3) = 2,655.

## 0b. The gate's registry, adopted (2026-09-27 ~16:45)
`declared_vs_built_v6.py` here is the `v7f` gate stream's version: my union + its 232 gate-only entries + `check_all`,
verified byte-equal to (my union + its `diffs_vs_v7f_merge` diff). The trainer still runs only the 7 new-lever entries
(`check` / `check_v6`); the gate runs `check_all`. The gate batch's own copy is listed SUPERSEDED in `LANDING_READY.txt`.
With the gate files present, the merge's own tests all pass (223 passed in the combined run); the gate's
`test_launch_gate_v7f.py` had 10 failures, ALL downstream of the nav fix (§1b) — its fixtures pinned the defect and
built the S-W rehearsal predecessor without nav — and is being updated by the gate stream.

## 1. Why R2 needed fixing: a real `--nav-cond` v7F run could not START at the tip
PI R2: the navigation command MUST reach the tactical and operative layers at training and inference. The model has
the conditioner (`--nav-cond`), but three defects — each MEASURED — made every nav-conditioned build or run die:

| # | defect | where (tip) | found by |
|---|---|---|---|
| 1 | `V6Stack.synthetic_batch` carries no nav keys; `assert_isolation` (run at EVERY trainer build) forwards it and raises `NavTokenMissing` | `v6.py` `synthetic_batch` / `forward` `:6038` | R1/R4 stream (probe) |
| 2 | `train()`'s batch-dict literal splats the NavEmitter output and THEN writes `"nav_token": b.get("nav_token")` — the later key wins, so every emitted token becomes `None` at step 1 (pinned by `test_nav_v6stack.py:195`, which asserts the line exists but not its order) | `train_v6_staged.py:7314` vs `:7336` | R1/R4 stream (static) |
| 3 | ⭐ **found here, after fixing 1–2:** the NavConditioner's 12 parameters belong to NO group in `V6Stack._GROUP_PREFIXES`, so `group_of` raises on every nav build — the existing nav tests construct and forward a nav stack but never run the grouping path a real build runs | `v6.py` `_GROUP_PREFIXES` / `group_of` | this merge (test RED) |

### ⛔⛔ 1b. A FOURTH defect — found by the `v7f` gate's G-DVB, and it invalidated my "R2 fixed" (2026-09-27 ~16:35)
`build_stack_from_args` NEVER mapped `--nav-cond` into `V6Config`: the flag was REQUIRED by preflight for every v7-line
run and then silently dropped, so every such launch built a stack with NO nav conditioner — a declared lever the built
model did not have (the D-REFCV6-CONFIG-BUILD class). ⛔ **My fixes 1–3 were tested on configs built DIRECTLY and never
on the real launch path** — the same blind spot I had just criticised in the existing nav tests. With 1–3 alone, R2 was
still met by NO v7F launch. **Fix 4:** `nav_cond=bool(getattr(a, "nav_cond", False))` in `build_stack_from_args`
(default False = byte-identical). **Pinned through the REAL path:** `test_REAL_LAUNCH_nav_cond_builds_the_conditioner`
(argv → `build_parser` → `build_stack_from_args` at production geometry, which also runs `assert_isolation` on
`synthetic_batch`, so fixes 1 and 3 are exercised on this path too) + its OFF control; mutant "mapping removed" → 1 FAIL,
file restored bit-identically.

### ⛔ 1c. F7 + F8 — the gate's next finds once nav was really built (2026-09-27 ~17:20), both FIXED here
- **F7 — the operative layer never learned from nav.** The S-W world-model losses (O1 via `stage_a_losses`, O2/O3/O5
  via `rollout_transitions`) roll `predictor_op` through SHARED helpers that never forward nav, so the 6 operative nav
  tensors got NO gradient in S-W, and S-T freezes their group. **Fix:** `_NavBoundPredictor` binds
  `stack.nav(..., "operative")` to the predictor for exactly those 3 call sites (the shared helpers stay untouched;
  nav off = the same object). ⚠️ FiLM (`film.to_scale_shift`) and `nav.layer_proj.operative` are ZERO-init by design,
  so NO conditioning term — actions included — gets gradient at step 0; the unit test wakes both and then requires all
  6 tensors reached, and its regression arm (nav not forwarded) requires NONE reached even when woken.
  **Independent confirmation:** the gate's real-trainer S-W rehearsal flipped G-LIVE FAIL → PASS with the unreached-leaf
  list now empty (`raw/pytest_merge_plus_gate2.txt`: its 2 F7-pinning tests fail BECAUSE of the fix; 240 others pass).
- **F8 — `--dry-run` died with `NavTokenMissing` on every nav launch** (`synthetic_train_batch` had no nav keys).
  **Fix:** nav keys drawn LAST when the stack has nav; every pre-existing key unchanged (tested).
- Tests: `test_v7f_r2_nav_fixes.py` §5 (12 total in the file, all pass); source mutants — call sites reverted → F7 test
  FAILS; dry-run nav keys removed → F8 test FAILS; file restored bit-identically.

### ✅ 1d. The gate's flipped tests, integrated (2026-09-27 ~17:40)
The `v7f_gate` stream turned its two F7-pinning tests into a POSITIVE (`S-W G-LIVE PASSES and the operative nav leaves
TRAIN`, no `--allow-unreached`), plus two regression arms: `v6_nav_unbound_wm` swaps `_NavBoundPredictor` for the bare
predictor, and a source mutation reverts the 3 WM-loss call sites. Both must FAIL naming the operative nav leaves, and
both do. Its `launch_gate.py` (sha256 `f1534182a874e082…`) and `test_launch_gate_v7f.py` (`428a647b5340a0bf…`) were copied
into this merge copy **for testing only**; the gate package owns and banks them (`../v7f_gate/code/fix/`). Its
`declared_vs_built_v6.py` is byte-identical to this package's (`0df4bf43d28ecfd1…`).
- **Combined run: 256 passed, 0 failed, 0 skipped** (`raw/pytest_merge_plus_gate3.txt`). Scope: `test_launch_gate_v7f.py`
  plus the 6 merge-side files (`test_declared_vs_built`, `test_tactical_label_reach_v6`, `test_v6_effective_weights`,
  `test_v7f_r1r4`, `test_v7f_r2_nav_fixes`, `test_v7f_r6_strategic_off`). Settings: CPU only, with both bit-identity
  baselines set from the `c36b6ddd` blobs, so nothing skips. The previous run was 240 passed + 2 failed (`raw/pytest_merge_plus_gate2.txt`).
- refcv7 `test_launch_gate.py` on this copy: **125 passed, 1 failed** (`raw/pytest_refcv7_gate.txt`). The failure is
  `test_G_SUITE_the_pinned_env_list_is_content_bound_by_the_profile`, and it is **EOL-only, MEASURED**: this copy's checkout
  of `stack/ops/launch_gate_thor_env_failures.json` is CRLF (sha256 `9b6aa814e5711459`). Its LF form hashes to
  `fc1214e5978383c6`, which is exactly the tip blob and the profile's pin. On Thor (LF checkout) it passes. For the
  Master Mind: the pin hashes worktree bytes with no EOL normalisation, so a Windows autocrlf checkout reads a false
  "changed list".

## 2. The fixes (all bit-identical when `--nav-cond` is off)
1. `synthetic_batch` draws `nav_token` (int64 [B] over the 3-token vocabulary) and `nav_args` ([B, 2]) LAST, and only
   when the stack has a conditioner — every pre-existing key keeps its bytes and RNG stream (tested nav on vs off).
2. The NavEmitter splat now sits AFTER the `.get` forwarding, so an emitted token wins and the `.get` path still serves
   a dataset that carries nav in the item. The pin at `test_nav_v6stack.py:195` stays green (the line still exists).
3. Groups (TrainingFlyWheel design call — `SPEC_NAV_CONDITIONING_ALL_LAYERS.md` names this stream as owner and fixes no
   stage grouping): the SHARED `nav.embed` + `nav.arg_proj` and the operative projection + gate → `predictor_op`
   (S-W, the first stage that consumes nav); the tactical projection + gate → `layer_tac` (S-T); the strategic
   projection + gate → `layer_str` (S-S, frozen while R6 keeps the strategic layer off). ⚠️ Consequence to know: in
   S-T the shared embedding is FROZEN (trained in S-W); only the tactical projection/gate adapt. That is the stage
   rule working as designed, and the alternative (shared → `layer_tac`) would leave the operative path reading an
   untrained embedding throughout S-W.

## 3. Verification
- `tests/test_v7f_r2_nav_fixes.py`: **7 passed** (literal expectations; a regression arm per fix).
- **Mutation check** (`raw/mutate_r2.py` → `raw/mutate_r2.out.json`): unmutated 7/7 GREEN; M1 synthetic_batch drops
  the nav keys → 3 FAIL; M2 nav groups removed → 2 FAIL; M3 the tip's order restored → 1 FAIL; M4 a duplicate `.get`
  line re-added AFTER the splat → 1 FAIL. Files restored bit-identically (sha256) after every mutant.
- `test_nav_v6stack.py` + `test_v7f_r1r4.py` (with the tip `v6.py` as the bit-identity baseline — without it 5 of its
  tests SKIP, which is not a pass) + `test_v7f_r2_nav_fixes.py` + `test_v6_stage_init_introduction.py`: **97 passed**
  (`raw/pytest_core.txt`).
- The full 107-file v6-related suite on this merge: see §5 (compared failure-set-for-failure-set with the R1/R4 stream's).
- Diffs vs `c36b6ddd` (`code/diffs_vs_c36b6ddd/`): each `git apply`s onto the tip blob and reproduces the banked file
  byte-for-byte (`make_diffs.py`, 3/3; `core.autocrlf=false` — with it on, git on this box writes CRLF and a byte
  check reads a false mismatch).

## 4. Not done / next
*(Updated 17:40: the R3 merge is DONE (§0), and the v7f gate profile is DONE and integrated (§0b, §1d).)*
- ~~**R3 merge**~~ → §0.
- **R6 literally off:** BUILT as `--strategic-off` (§6). Whether the first compliant run uses it is the PI's D3; my
  recommendation is yes — R6 says the strategic layer is off, and a fixed random embedding feeding the tactical head is
  not off.
- ~~**G-DVB / the v7f gate profile**~~ → DONE in `v7f_gate/` (§1d). The rehearsal still FAILS on G-HYG (open config
  classes, so `@strict_fields` is needed) and G-CKPT (resume replays the RNG stream, so the checkpoint must carry RNG state
  and data position). Both are trainer changes owed before a PASS token can exist; they are listed in the HANDOFF.
- The per-arm eval tools must pass `plan_v_max_ms` (R1 cap) — the R1/R4 stream's §5.

## 5. Full v6-related suite on this merge (R1+R4+R2, BEFORE R6 was added)
The R1/R4 stream's 107-file v6-related list + `test_v7f_r2_nav_fixes.py`, with `taniteval/` beside `stack/` and the tip
`v6.py` as the bit-identity baseline: **2,606 passed, 44 skipped, 8 failed, 24 errors** (`raw/pytest_v6_related_merge_tail.txt`).
The failure set is **IDENTICAL, test id for test id, to the R1/R4 stream's** (`raw/failset_merge.txt` vs
`raw/failset_agent.txt`: 32 = 32, none new, none missing) — the environmental set of a bare `stack/` copy (repo documents
absent, one CRLF-vs-LF sha pin). (Superseded by §0's complete-merge run.) The full suite had NOT yet been re-run with R6 applied; R6 is covered by its own
10 tests + 5/5 mutants and by the 105-test targeted re-run below, and is bit-identical when off. The post-R6 full run is
owed (queued behind the refav1 A2 probe, which needs the CPU).

## 6. R6 — `--strategic-off` (added after §5)
PI R6: strategic layer OFF in the first experiments. MEASURED (R1/R4 stream, `AUDIT_R4.md` §5): with S-S never trained,
the tactical goal heads still read `cond=e_g_str` from the UNTRAINED strategic head, and the plan loss reaches 13
`layer_str` tensors through it. `V6Config.strategic_off` / `--strategic-off` (default OFF, byte-identical) hands the
mixed AND factored tactical goal heads a ZERO conditioning (`zeros_like`, no gradient path) and REFUSES `tac_goal_cond`
(the g_str → P_T dynamics port, the only other strategic → tactical route; ⚠️ the v7 chain's S-T command turns it ON,
so an R6 run must drop it). The real `e_g_str` stays in the outputs for logging.
- `tests/test_v7f_r6_strategic_off.py`: **10 passed** — ON: perturbing every `layer_str` parameter leaves `g_tac`
  (mixed and factored) bit-identical and no gradient reaches `layer_str`; OFF (regression arms): both move / both reach,
  i.e. the tip defect reproduced; the refusal; OFF bit-identical to the explicit default; the CLI flag and its mapping.
- Mutants (`raw/mutate_r6.py` → `raw/mutate_r6.out.json`): unmutated 10/10 GREEN; M5 flag ignored → 3 FAIL; M6 refusal
  removed → 1 FAIL; M7 CLI mapping removed → 1 FAIL; M8 mixed head reads the real embedding → 2 FAIL; M9 factored heads
  read it → 1 FAIL. Files restored bit-identically.
- Targeted re-run with R6 applied: R6 + R2 + R1/R4 (bit-identity baseline on) + `test_nav_v6stack.py` +
  `test_v6_stage_init_introduction.py` = **105 passed** (before the 2 factored tests were added; they pass, 10/10).

## 7. F2 — G-HYG: the v6 config tree now REFUSES undeclared attributes (2026-09-27 ~18:00)
**What the gate measured.** The v7f dev-box rehearsal FAILED G-HYG naming exactly the four open config classes:
`tanitad.config.EncoderConfig`, `PredictorConfig`, `ReadoutConfig` and `tanitad.models.v6.V6Config`. With them open,
the D-REFCV6-EQUALIZE-DROPPED mechanism was still possible on the v6 family: a lever set as an ad-hoc attribute stays
in argv and `config.json` while the first `dataclasses.replace` drops it.
**The fix.** `@strict_fields` from `tanitad/train/config_hygiene.py` (stdlib-only; refc/refcv7 already use it) on
those four classes, applied by `raw/apply_f2_strict.py` (anchor-based, EOL-preserving).
**Scope.** Only the four classes in the v6 tree. The other `tanitad.config` dataclasses (`SigRegConfig`, `LossConfig`,
`TacticalConfig`, …, `StackConfig`) are the same defect class for v1–v5 but are outside the v7F launch. That work is
listed for the Master Mind, not done here.
- **Static probe** (`raw/scan_undeclared.py` → `raw/scan_undeclared.out.json`): 0 assignments of an undeclared name
  onto these four classes anywhere under `stack/`. All 28 hits are false positives by construction of the name
  heuristic: refc's own `cfg.core.encoder`, a different class that is already strict, and module attributes.
- **Unit test** `tests/test_v6_config_strict.py`: **12 passed**. It covers:
  - literal class names;
  - refusal at assignment;
  - `replace`, `asdict` and `deepcopy` still working;
  - a built stack's tree being strict and clean;
  - the REAL launch path (`build_parser` → `build_stack_from_args`, with nav on);
  - a regression arm that makes EncoderConfig open again and requires the walker to name it.
  On a copy with F2 reverted: **11 failed, 1 passed**. The survivor is the replace/asdict behaviour test, which must
  hold either way.
- **Through the gate.** `test_launch_gate_v7f.py`'s pin `test_positive_G_HYG_names_the_open_config_classes…` is
  flipped to `test_positive_G_HYG_PASSES_every_config_class_in_the_v6_tree_is_strict`. It asserts G-HYG PASS,
  `not_strict == []`, and that the census really walked the four literal classes. It comes with a new gate arm,
  `test_ARM_F2_encoder_config_unstrict_FAILS_G_HYG_on_census_AND_probe`: EncoderConfig made genuinely open must FAIL on
  BOTH the census and the active setattr probe. G-HYG subset: **3 passed** (the positive, the F2 arm, and the existing
  `v6_undeclared_attribute` arm, which injects through `object.__setattr__` and so stays valid under strict classes).
  **Mutation:** with F2 reverted in a copy, the positive FAILS, and the gate's own reason names all four classes:
  `4 config dataclass(es) do NOT refuse undeclared attributes …` and `setting an undeclared attribute SUCCEEDED on 4
  config object(s) … ['cfg (V6Config)', 'cfg.encoder (EncoderConfig)', 'cfg.predictor (PredictorConfig)',
  'cfg.readout (ReadoutConfig)']`.
  The gate test file belongs to `../v7f_gate/`; its F2 flip is applied by `raw/apply_f2_gate_test.py` and banked with
  the gate files once F3 (G-CKPT, a separate stream) has also landed in that file.
- **Full stack suite with F2:** see §7b (running under a commit-headroom watchdog, `raw/run_guarded_suite.py`).
