# P7-WIRE-INTEGRATE-2 -- the surviving mutant M2 is a REAL untested behaviour (now tested), and `refe_p7` gets a CONTENT pin for its bank

Agent P7-WIRE-INTEGRATE-2, 2026-10-22. 0 GPU, no network, no pod, no git tree touched, **nothing staged / committed / pushed** (local override: the Master Mind lands this together with `p7int/`).
Base: mirror tip `a8f2c89af000a7497187ff52177603901733353a`; `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Package `C:/Users/Admin/par18/p7int2/`: `code/` = the COMPLETE integrated set at repo paths
(p7int's files + this package's edits) + `P7_WIRE_INTEGRATE_2.diff` (vs p7int) + `P7_WIRE_INTEGRATE_vs_tip.diff` (vs the tip, what the lander applies) + `MANIFEST.txt`; `raw/` = probes, runs, verbatim outputs.
Evidence classes: MEASURED (ours, artifact named) / INHERITED / PENDING.

## HEADLINE -- read before landing

1. **LAND `p7int/` AND THIS PACKAGE AS ONE UNIT: `code/` here supersedes `p7int/code/` file for file** (it contains all of p7int's files; the 8 that differ are listed in section 6). Same lander rules as before (LF, `base=<blob>` per file, regenerate `SHIP_MANIFEST.json` after). `train.py`, `model.py`, `session_reserve.py`, `p6_pod.sh`, `p6_prep.py` are NOT touched (sha256 in `raw/probes/probe_p6_identity_*.out`).
2. **GAP 1 closed, option (a): M2 is NOT equivalent.** It removes the early `return` after an omitted audit input. The verdict is FAIL either way (the omission reason is appended BEFORE the return), which is why every verdict-only arm let it live. MEASURED
   (`raw/probes/probe_m2_original.out` vs `probe_m2_mutated_M2.out`): without the return the REAL audit RUNS on the partial input set (0x -> 1x; at the real scale it walks the 92,896-scene bank, the metric cache and a 413,184-file image listing), the verdict
   carries a SECOND reason (the partial audit's by-product) and the evidence records an `audit` block for an audit that proved less than SPEC_P7 2 asks. 23 assertions pin those three observables with controls that must read the known value. **M2 is now RED (15 failing).**
3. **GAP 2 closed, with one finding you must read: P6 does not pin a bank's CONTENT at a fresh start; its `8775e73e...` is the trainer's RESUME identity.** (section 2). So "exactly as P6" cannot be copied; what P7 gets instead is STRONGER than P6's launch check:
   `P7_TARGETS_SHA256` registered per bank, refused by `p7_refusals` before any check, FAILED by G-CLOCK (the last data check before the PASS token) when the bank's bytes differ, printed by `p7_pins.py`, and re-compared with the trainer's OWN recorded identity after the first start by `p7_pins.py --check-run`.
   **The trainer itself still does not refuse a wrong bank at a fresh start** (that needs a `train.py` edit + G-DVB entry + bit-identity proof; not done, as instructed). The residual window: the PASS token fingerprints a directory by its LISTING, so a same-size in-place edit between G-CLOCK and the trainer's start is invisible to `verify`; `--check-run` detects it afterwards from the trainer's own number.
4. **DECISION for the MM: wire `p7_pins.py --check-run <run dir>` into the P7 chain (`p7_pod.sh`, still OPEN) right after the first checkpoint.** It is the only part of Gap 2 that needs the chain; without that call the window above is open.
5. **`refe_p7` STILL REFUSES, by design.** Three pins are PENDING the rebuilt 3-C bank (`P7_BANK_TUPLES`, `P7_TARGETS_SHA256`, `P7_AUX_STORE_PINS`); `PROCEDURE_P7_PINS.md` is updated (three lines pasted instead of two, Step G added) and its literal commands were re-run on the synthetic tree (`raw/rehearsal/rehearse.out`).
6. **The head owner's `test_p7_head_wiring.py` is edited again** (S2: `registered()` carries a digest, the unknown-pins clause and the `UN` copy include it, "nothing else is refused" 2 -> 3, one new red arm): static `ZZP7HEAD_PASS` 181 -> 182. Merge on the tip, not over it.

## 1. Gap 1 -- M2

`MUTATIONS[1]` replaces `if missing:\n        return verdict("G-RESERVE", reasons, det, ...)` by `if False:` in `gate_checks.check_reserve`. Probe (`raw/probes/probe_m2.py`, driver `run_m2_probe.py`; five omissions + a control):

| argv | status | reasons | audit runs | evidence `audit` block |
|---|---|---|---|---|
| all five named (control) | PASS | 0 | 1x | yes |
| any ONE omitted, original | FAIL | **1** (the omission) | **0x** | **no** |
| any ONE omitted, M2-mutated | FAIL | **2** (omission + "audit_inputs raised ReserveError" / "the audit examined [...]") | **1x** | **yes** |

So the verdict is identical and the behaviour is not: **(a) real and untested**, not an equivalent mutant (the branch is reachable: every omitted-input arm takes it). New arms (`test_p7_integration.py` section I, 23 assertions): a recording audit is called 0x for each of the five omissions and exactly 1x on the full argv;
the omission is the ONLY reason; the evidence carries no `audit` / `audit_counts`; and the same with the REAL `audit_inputs` wrapped in a counting shim (control: counts 1 on the full argv). M2 goes RED with 15 failing.
Side change: `main()` now runs each section inside a try/except that records the raise as a named failure (a section that crashes no longer hides the others; M20 below used to go RED only through a traceback).

## 2. Gap 2 -- how P6 gets and enforces `targets_sha256 8775e73e...` (quoted, tip a8f2c89) and what P7 does instead

| fact | where (MEASURED by grep/read) |
|---|---|
| the digest is the trainer's `_digest_files(sorted(Path(a.targets).glob("targets_rank*.jsonl")))`, recorded in the run IDENTITY | `refe/train.py:1067-1074` (`_digest_files`), `:1077-1100` (`run_identity`, `:1088` `"targets_sha256": None if grow else _digest_files(tfiles)`) |
| it is ENFORCED only on `--resume`: `_identity_diff` compares it with the checkpoint's recorded identity; a mismatch is "REFUSING TO RESUME", rc 4 | `train.py:1102-1125` (compared keys `:1117`), call `:1525`, `return 4` `:1531` |
| it is RECORDED at the first start | `config.json` `"identity": ident` `train.py:1575-1588`; `ckpt_last.pt` `:1651` |
| `8775e73e...` appears in NO code path | only `eval/PREREG_P6.md:191` ("The trainer's own preflight then requires `targets_sha256 8775e73e...`", the resume from the HF checkpoint, whose recorded identity is the pin) and the fixture `test_p6_chain.py:719,745` |
| the chain's guard for the same fact | `refe/p6_prep.py:260-290` (`resume-bank` -> `ZZRESUME_BANK_SAME|MISMATCH`), `p6_pod.sh:254` |
| P6's gate pins the COUNT only | `launch_gate.py:252-259` (`P6_BANK_TUPLES`), `gate_checks.py:2144-2153` (G-CLOCK compares the count of the trainer's own `TargetBank`, captured at `:192`) |

**Consequence:** there is nothing at P6's fresh start to copy, and P7's first start is fresh. The mechanism that exists is (i) the gate's registered per-bank count and (ii) the trainer's resume identity. P7 now has both for content:

* **Registration shape -- separate dict `P7_TARGETS_SHA256` (profile `expect_schedule["targets_sha256_by_bank"]`), not `P7_BANK_TUPLES` widened to `{dir: {"tuples", "targets_sha256"}}`.** Why: (1) `n_tuples_by_bank` values are ints in `refe_p6` and in `clock_bank_tuples`; widening them changes shared code on P6's default path and breaks the anchor of `test_p6_levers.py:634`
  (a mutant of `return reg[name], ...`); a separate key is additive and P6 takes no new line (proof: section 4); (2) a wrong count and wrong bytes are different failures and get different refusal strings and mutants; (3) keyed by the same directory name, with a refusal if the two dicts name different banks, so skew is impossible.
* **Refusals** (`launch_gate.p7_refusals`, before any check): bank has no registered digest; digest malformed (not 64 lowercase hex); count and digest register different banks; **the profile has no `targets_sha256_by_bank` key at all** (a key that can be absent is a default that disables the check).
* **G-CLOCK** (`gate_checks.check_clock` + `clock_bank_sha256` + `bank_targets_sha256`): recomputes the digest of the `--targets` files with the trainer's own definition and FAILS on any other bank, an unregistered bank, no rank file, or (for a `p7_rules` profile) a missing key; `refe_p6` / `refe_p2` have no key, so they never read the bank.
  The digest function is the trainer's text: its AST equals `train._digest_files` (test), and MEASURED under the torch venv the gate's value equals `train.run_identity(...)["targets_sha256"]` and `train._digest_files(...)` and the reserve module's (`raw/probes/probe_digest_vs_trainer.out`, ALL EQUAL; a same-size edit changes all of them and the trainer's own `_identity_diff` reports it).
* **`p7_pins.py`** prints `P7_TARGETS_SHA256: dict = {...}` beside the other two lines and `targets_sha256=<16>` in `ZZP7PINS_OK`; it refuses unless the GATE's definition and the reserve module's agree on that bank (two copies of one definition, compared where a drift costs nothing).
* **`p7_pins.py --check-run <run dir>`** (after the first start): `config.json`'s `identity.targets_sha256` and `identity.n_tuples` -- the trainer's own numbers on the bank it loaded -- must equal the registration of the bank named in `args.targets`; null digest (`--grow`; forbidden in `refe_p7`), unregistered bank, missing config, wrong digest, wrong count all rc 1 `ZZP7RUN_REFUSED`. With no argument the tool still exits 2.

**Reachability from the real caller** (the class this repo paid for four times): `raw/probes/probe_real_build_clock.out` runs the REAL trainer's BUILD (`instrumented_run`, tiny rehearsal fixture, torch venv, CPU) through `launch_gate.run_checks(only={"G-CLOCK"})`: registered bank PASS; two rows of the bank SWAPPED (same row count, same size, the trainer still loads it) -> **FAIL for the content**;
the count-only profile PASSes the same tampered bank (so the defect was real). The suite runs the same through `run_checks` with a fake build capture, because the real build needs torch.

## 3. Tests

`test_p7_integration.py` 139 -> **220** assertions (~30 s). New: section 0 (digest definition: independent recompute == reserve == gate == trainer by AST), section I (+23, Gap 1), section III (3 pins: nothing / count only / digest only / key missing / 6 malformed digests / different banks), section V (the content pin, ~50 assertions):
registered bank PASS; a SAME-COUNT, DIFFERENT-BYTES bank (one byte flipped in place; two rows swapped) FAILS G-CLOCK directly and through `run_checks` for the content and for nothing else, while the count-only gate PASSes it (red arm); unregistered bank, no rank file, key-less `p7_rules` profile FAIL; a P6-like profile never reads the bank
(a booby-trapped digest function is not called); `p7_pins` refuses on a definition drift; `--check-run` on the registered identity and on each defect, in-process and as a subprocess.
`--mutations`: **25 mutants** (p7int's M1-M15 + M16-M25: content comparison removed, unregistered bank accepted, malformed digest accepted, missing key skipped, digest definition drifted, wrong digest registered, definition cross-check removed, `--check-run` digest / count comparison removed, count and digest allowed to name different banks); M2 RED. Verdict in section 4.

## 4. VERIFICATION (generated by `finalize2.py` from the raw outputs; do not hand-edit between the markers)

<!-- VERIFICATION:BEGIN -->
**Verification verdict (generated by finalize2.py): GREEN: every suite rc 0 before and after, the four refe_p6 suites' tokens IDENTICAL, the integration test green, 25 mutants RED with the control PASS (ZZP7INT_MUTATIONS_OK), train.py / model.py / session_reserve.py / p6_pod.sh / p6_prep.py untouched**

### the suite set of P7-WIRE-INTEGRATE, CPU (`CUDA_VISIBLE_DEVICES=-1`), BEFORE = pristine tip export, AFTER = the integrated overlay of BOTH packages

| suite | BEFORE (tip a8f2c89) rc | BEFORE token | AFTER (overlay) rc | AFTER token | secs before/after | same |
|---|---|---|---|---|---|---|
| test_launch_gate | 0 | ZZLG_STATIC_PASS 1168 | 0 | ZZLG_STATIC_PASS 1168 | 48/47 | yes (identical token) |
| test_p7_head_wiring_full | 0 | ZZP7HEAD_PASS 207 | 0 | ZZP7HEAD_PASS 210 | 408/409 | rc equal, count differs (see note) |
| test_session_reserve | 0 | ZZSESSRES_PASS 43  (optional checks run: ['aux_label_store.finalize', 'filter over the real rank-0 shape', 'sandbox chai | 0 | ZZSESSRES_PASS 43  (optional checks run: ['aux_label_store.finalize', 'filter over the real rank-0 shape', 'sandbox chai | 109/108 | yes (identical token) |
| test_p6_chain | 0 | ZZP6CHAIN_PASS 432 | 0 | ZZP6CHAIN_PASS 432 | 735/734 | yes (identical token) |
| test_p6_levers | 0 | ZZP6_PASS 165 | 0 | ZZP6_PASS 165 | 189/189 | yes (identical token) |
| test_p7_integration | - | - | 0 | ZZP7INT_PASS 220 | -/7 | NEW (p7int) , grown here |

The head test's `--full` count differs by construction: the tip's 207, P7-WIRE-INTEGRATE's 209 (+2 assertions in its S2 / S4 blocks) and this package's +1 (S2: the unregistered bank CONTENT refuses). Every refe_p6 suite (`test_launch_gate`, `test_p6_levers`, `test_p6_chain`, `test_session_reserve`) has an identical token before and after.

### every file that differs from the tip (tree comparison of the two run trees, not a hand list)

| file (under `PK`) | vs tip |
|---|---|
| `.gitattributes` | MODIFIED |
| `raw/2026-10-22-p7-wire-integrate/PROCEDURE_P7_PINS.md` | NEW |
| `raw/2026-10-22-p7-wire-integrate/rehearsal/build_synth.py` | NEW |
| `raw/2026-10-22-p7-wire-integrate/rehearsal/rehearse.out` | NEW |
| `raw/2026-10-22-p7-wire-integrate/rehearsal/rehearse.sh` | NEW |
| `raw/2026-10-22-p7-wire-integrate/RESULT.md` | NEW |
| `refe/gate_checks.py` | MODIFIED |
| `refe/launch_gate.py` | MODIFIED |
| `refe/p7_pins.py` | NEW |
| `refe/test_p7_head_wiring.py` | MODIFIED |
| `refe/test_p7_integration.py` | NEW |
| `SPEC_P6_PAPER_FAITHFUL.md` | MODIFIED |
| `SPEC_P7.md` | MODIFIED |

### `test_p7_integration.py --mutations` (verbatim)

```
CONTROL (unmutated copy): PASS ZZP7INT_PASS 220
  RED    M1 REVERT THE FIX: hand the audit the parsed JSON again: 40 failing; first: I a CLEAN tree PASSes G-RESERVE (real load_reserve + real audit_inputs): ["audit_inputs raised ReserveError: 'attempt_accepted' is not a s
  RED    M2 an input the argv does not name is no longer refused: 15 failing; first: I the argv omits --targets: the audit does NOT RUN on the partial input set (a recording audit, called 1x): [['calib_rows', 'image_index',
  RED    M3 an audit with ok False no longer fails: 34 failing; first: I FAIL: a planted LEAKED RANK-1 TWIN: []
  RED    M4 the result-side 'examined all five' check is gone: 1 failing; first: I FAIL: an audit that did not examine the calibration table, whatever its ok: expected a reason containing 'not all of'; got ''
  RED    M5 the gate stops requiring the registered reserve sha256: 1 failing; first: I FAIL (closed): a CRLF checkout of the same draw fails the gate's sha pin, and ONLY that: []
  RED    M6 the gate audits another glob than the trainer loads: 8 failing; first: I ... bank_files is a list of existing rank files, the other four are path strings: {'bank_files': ['C:\\Users\\Admin\\AppData\\Local\\Tem
  RED    M7 unregistered store pins no longer refuse: 2 failing; first: III nothing registered REFUSES: the label store's pins: expected a reason containing "label store's pins"; got "refe_p7: P7_AUX_STORE_PINS
  RED    M8 an unregistered bank no longer refuses: 5 failing; first: II ... and the bank's tuple count: expected a reason containing 'no registered tuple count'; got "profile refusal: refe_p7: the P7 label s
  RED    M9 a malformed store registration is accepted: 5 failing; first: III a MALFORMED store registration is refused: a short sha: expected a reason containing 'not a registration'; got ''
  RED    M10 a malformed bank registration is accepted: 5 failing; first: III a MALFORMED bank registration is refused: 0: expected a reason containing 'not a positive int'; got ''
  RED    M11 the reserve's own suite is dropped from refe_p7's G-SUITE: 1 failing; first: IV refe_p7's suite = P6's + the head's static block + the reserve's own suite + this test: [{'script': 'test_p7_head_wiring.py', 'args': [
  RED    M12 p7_pins no longer checks the filter report's digest: 1 failing; first: III p7_pins REFUSES a bank edited after the filter (same row count, different bytes -> digest): expected 'targets_sha256'; got 'the bank h
  RED    M13 p7_pins no longer audits the bank + store: 2 failing; first: III p7_pins REFUSES a LEAKED twin whose report was patched to match (the audit still sees it): it printed a registration
  RED    M14 p7_pins reads the key count from the manifest instead of the shard headers: 1 failing; first: III p7_pins REFUSES a manifest whose stored_keys is not the shard headers' key count: it printed a registration
  RED    M15 p7_pins pins the wrong store field (the first shard's sha, not the store's): 2 failing; first: III p7_pins' store pins == an independent recompute (shard bytes -> files map -> sha256; header entries -> keys): 59b9552c2f73925a 6 vs 34
  RED    M16 G-CLOCK stops comparing the bank's content digest with the registered one: 8 failing; first: V [ONE BYTE changed in place (same size, same row count)] G-CLOCK REFUSES the bank at the point where training would start: []
  RED    M17 an unregistered bank no longer refuses for its content digest: 4 failing; first: III ... and its content digest too: expected a reason containing 'no registered targets_sha256'; got "refe_p7: --targets '/w/train_grow_cm
  RED    M18 a malformed content digest is accepted: 6 failing; first: III a MALFORMED content digest is refused: a short digest: expected a reason containing 'not a sha256'; got ''
  RED    M19 a refe_p7 profile without the content key is skipped instead of failing G-CLOCK: 1 failing; first: V a refe_p7 profile WITHOUT the key FAILS G-CLOCK (never a skip): expected a reason containing 'registers no targets_sha256_by_bank'; got 
  RED    M20 the gate's digest drifts from the trainer's (hashes the full path, not the basename): 3 failing; first: 0 the gate's bank_targets_sha256 == an independent recompute == the reserve module's digest == the gate's _digest_files: effe3b76390fa486e
  RED    M21 p7_pins registers a digest that is not the bank's: 5 failing; first: III CLI: the OK line carries the content digest's prefix: grow_cmd_3c_p7": "c66bc98e6a46f720b377c84b8737acb763ab487212ce9caa167e1264f71663
  RED    M22 p7_pins stops cross-checking the gate's digest definition: 1 failing; first: V p7_pins REFUSES when the gate's digest definition disagrees with the reserve module's: it printed a registration
  RED    M23 check_run stops comparing the trainer's recorded digest: 2 failing; first: V check_run REFUSES a recorded digest that is not the registered one (the trainer trained on another bank): expected a reason containing '
  RED    M24 check_run stops comparing the trainer's recorded tuple count: 1 failing; first: V check_run REFUSES a recorded tuple count that is not the registered one: expected a reason containing 'the trainer built 10 tuples'; got
  RED    M25 the count and the digest may name different banks: 1 failing; first: III the two registrations naming DIFFERENT banks are refused: expected a reason containing 'register DIFFERENT banks'; got ''
ZZP7INT_MUTATIONS_OK
```
<!-- VERIFICATION:END -->

**P6's default path is byte-identical (MEASURED, `raw/probes/probe_p6_identity_{tip,overlay}.out`, diff empty):** `refe_p6` profile sha256 `80fa4d7ec757...` and its `expect_schedule` sha256 before == after; `check_clock` on the P6 profile (clean / wrong count / unregistered bank) gives the same JSON sha256 and the same evidence keys before and after
(no `bank_content` block); `train.py`, `model.py`, `session_reserve.py`, `p6_pod.sh`, `p6_prep.py` have equal sha256. `train.py` was not touched, so the head package's determinism control (which proves a trainer EDIT bit-identical) does not apply.

## 5. Not covered / caveats (plainly)

* **No real data.** The three pins are PENDING the rebuilt 3-C bank; every number here is from a synthetic tree. The digest of the real bank is whatever `p7_pins.py` prints on it.
* **The trainer does not itself refuse a wrong bank at a fresh start**; the gate does (G-CLOCK), and the trainer's identity machinery guards every resume. `--check-run` is UNWIRED until `p7_pod.sh` exists. The PASS token's directory fingerprint (listing) is unchanged: making it content-based (`content_flags` + a sealed `content_manifest.json` on the bank) is the alternative that closes the window preventively and was NOT built (it re-hashes the whole bank on every `verify` and adds a file to the bank directory).
* G-CLOCK now reads the whole bank once more at launch (the trainer already reads it twice: rows + identity digest). Cost not measured on the real ~1-2 GB bank (UNVERIFIED).
* `test_p7_integration.py` still imports `test_p7_head_wiring` and `test_session_reserve` (their fixtures). Timing numbers were measured with other agents' jobs sharing the 24-core box.
* Single seed / intervals: not applicable (no training, no estimate is quoted).

## 6. Deliverable manifest

| artifact | where | single location? |
|---|---|---|
| the complete integrated set at repo paths (p7int's files + edits): `refe/gate_checks.py`, `refe/launch_gate.py`, `refe/p7_pins.py`, `refe/test_p7_integration.py`, `refe/test_p7_head_wiring.py`, `SPEC_P7.md`, `raw/2026-10-22-p7-wire-integrate/{PROCEDURE_P7_PINS.md, RESULT.md (addendum), rehearsal/{rehearse.sh, rehearse.out}}` edited here; `SPEC_P6_PAPER_FAITHFUL.md`, `.gitattributes`, `rehearsal/build_synth.py` carried from p7int unchanged | `C:/Users/Admin/par18/p7int2/code/<PK path>` (sha256 + CR bytes in `code/MANIFEST.txt`) | **YES: only here** (not in any repo) |
| `raw/2026-10-22-p7-wire-integrate-2/{RESULT.md, VERIFICATION.md, mutations.out, probes/*}` | `C:/Users/Admin/par18/p7int2/code/<PK path>/raw/2026-10-22-p7-wire-integrate-2/` (also under `p7int2/raw/`) | **YES** |
| `P7_WIRE_INTEGRATE_2.diff` (vs p7int), `P7_WIRE_INTEGRATE_vs_tip.diff` (vs the tip) | `C:/Users/Admin/par18/p7int2/code/` | **YES** |
| before / after suite outputs, `raw/mutations.out`, `raw/mutations_try1.out` (the trial run), `raw/probes/*`, `raw/rehearsal/rehearse.out` | `C:/Users/Admin/par18/p7int2/raw/` | **YES** |
| scratch, reproducible: `work/` (edit tree), `base_full/`, `before_full/`, `after_full/`, `code_p7int/`, `patch_*.py`, `run_suites2.py`, `make_jobs.py`, `finalize2.py`, `stage_code2.py` | `C:/Users/Admin/par18/p7int2/` | scratch |
