# P7-WIRE-INTEGRATE -- the two landed halves of P7's wiring connected: `refe_p7`'s G-RESERVE now runs the REAL session reserve

Agent P7-WIRE-INTEGRATE, 2026-10-22. 0 GPU, no network, no pod, no git tree touched, **nothing staged / committed / pushed** (local override of the staging rule: the Master Mind lands).
Base: mirror tip `a8f2c89af000a7497187ff52177603901733353a`, exported with `core.autocrlf=false` (LF). `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
Package: `C:/Users/Admin/par18/p7int/` (`code/` = the deliverable at repo paths + `P7_WIRE_INTEGRATE.diff` + `MANIFEST.txt`; `raw/` = probes, runs, rehearsal). Evidence classes: MEASURED (ours, artifact named) / INHERITED / PENDING.

## HEADLINE -- read before landing

1. **A real defect, MEASURED (`raw/probes/probe_before.out`).** At the tip, `G-RESERVE` **FAILED ON A CLEAN synthetic P7 tree** against the real module: `audit_inputs raised ReserveError: 'closure' is not a session name`. The head half passed `audit_inputs`
   the PARSED JSON DOCUMENT; the reserve half takes the SESSION SET (`load_reserve`). Every earlier test used a fake audit that accepted anything, so a check that could never PASS shipped green. Fixed in `gate_checks.check_reserve`
   (one `loader` seam, default `session_reserve.load_reserve`), with the old adapter replayed as a RED arm. This is the "built, tested, unreachable from its caller" class again, found by running the caller.
2. **DECISION for the MM / PI -- mismatch 2: I chose (a), and it departs from the literal text of SPEC_P7 section 2.** No trainer-side `--reserve-manifest`, no `filter_reserve.py`: the gate audits the chain's OUTPUTS (the five inputs the launch argv names) with the real
   `audit_inputs`. SPEC_P7 s.2 asks for the flag, a G-DVB `reserve_manifest` row and "the refusal string in `train.py`". **That text is now stale; a dated INTEGRATION NOTE is in `code/.../SPEC_P7.md` (table left as written).** If you want (b), the design is in section 3 and costs a
   `train.py` edit + G-DVB entry + a bit-identity proof; say so and it is a short job.
3. **LAND (scoped lander, `base=<blob>` per file; export/land with LF):** `refe/gate_checks.py`, `refe/launch_gate.py`, `refe/test_p7_head_wiring.py` (S2 + S4 edited, 179 -> 181 assertions: it is the head owner's file, merge on the tip), **new** `refe/test_p7_integration.py`, **new** `refe/p7_pins.py`,
   `SPEC_P6_PAPER_FAITHFUL.md` (bank figure + dated correction note), `SPEC_P7.md` (integration note), `.gitattributes` (+1 line: keep the sha-pinned reserve fixture LF), and the package dir `raw/2026-10-22-p7-wire-integrate/`. **Then regenerate `SHIP_MANIFEST.json`.** `train.py`, `model.py`, `session_reserve.py`, `p6_pod.sh`, `p6_prep.py` are NOT touched.
4. **`refe_p7` STILL REFUSES, by design.** The two pins are PENDING the rebuilt 3-C bank; the exact procedure is `PROCEDURE_P7_PINS.md` and its literal commands were rehearsed on a synthetic tree (`raw/rehearsal/rehearse.out`). The pin step also needs `test_p7_head_wiring.py` S2 to tolerate registration: **already done here**, so registering does not turn the G-SUITE red.
5. **OPEN (not in this brief):** `p7_pod.sh` does not exist. A P7 chain is a COPY of `p6_pod.sh` (its own `$P6` / `$RUN`), with `P7_RESERVE` + `P7_LABEL_STORE`, and the gate called as `--profile refe_p7 --gate-reserve-json "$P7_RESERVE"` (today `p6_pod.sh` names `refe_p6` at its `run` and `verify` calls, and its file-presence list does not name `session_reserve.py`, `p7_pins.py`, `fixtures/p7_reserve_sessions.json`). Until it exists nothing connects the chain's block 5a to the P7 gate except the data on disk.

## 1. What was built

| file (repo path under `PK/`) | change |
|---|---|
| `refe/gate_checks.py` | `check_reserve` rewritten against the real module: loads with `session_reserve.load_reserve` (the `loader` seam), refuses an argv that omits any of the five inputs (`RESERVE_INPUT_FLAGS`: `audit_inputs` treats an input it is not given as NOT AUDITED, which reads as clean), requires the real result to report all five in `counts.inputs_audited`, names the leaks in its reason, records the audit's counts in the evidence |
| `refe/launch_gate.py` | `import re`; P7 comments corrected (they named `filter_reserve.py` and `--reserve-manifest`, neither built); `P7_SUITE_EXTRA` += `test_session_reserve.py` (`ZZSESSRES_PASS`) and `test_p7_integration.py` (`ZZP7INT_PASS`); `p7_refusals` also refuses a MALFORMED pin (sha256 not 64 lowercase hex, keys / tuple count not a positive `int`) |
| `refe/test_p7_head_wiring.py` | S4: the fake audits are given the loader's frozenset (the real module's type) via `loader=`; two new red arms (a lazily imported audit that does not say what it examined); S2: the live-profile pin assertions now accept "unregistered OR well-formed" and the refusal arms run on an explicitly unregistered copy |
| **NEW** `refe/test_p7_integration.py` | 139 assertions, ~25 s: sections I-IV below, `--mutations` (15 one-line mutations in a COPY, each must go RED) |
| **NEW** `refe/p7_pins.py` | READS the two pins off the bank directory + re-finalised store; refuses unless every artifact check passes (section 5) |
| `SPEC_P6_PAPER_FAITHFUL.md`, `SPEC_P7.md`, `.gitattributes` | dated notes / one line (section 6) |
| **NEW** `raw/2026-10-22-p7-wire-integrate/` | `PROCEDURE_P7_PINS.md`, `rehearsal/{build_synth.py, rehearse.sh, rehearse.out}`, this `RESULT.md` |

## 2. Mismatch 1 -- G-RESERVE vs the real module (`test_p7_integration.py` section I-II)

Run on a synthetic P7 tree built with the reserve suite's own fixtures (`test_session_reserve.make_inputs`: bank rank-0 + twins, D9 cache, finalised aux label store, image root, calibration table) and the **real** reserve file (`fixtures/p7_reserve_sessions.json`, sha256 `53f3bcd4...`, 16 sessions). The training logs are asserted NOT in the reserve; the planted logs ARE.

| arm | outcome (real `load_reserve` + real `audit_inputs`) |
|---|---|
| clean tree | **PASS**; audit examined all five inputs, no leak, 16 sessions |
| planted leaked **rank-1 twin** | **FAIL**, reason names `<reserved log>/tw\|rank1\|lane_rank=1` in `'bank'` |
| planted **D9 cache key** | **FAIL**, reason names `<reserved log>/planted.pkl.xz` in `metric_cache` |
| also, each FAIL for its own reason | a leaked rank-0 row; a lane-rank twin part in a reserved cache dir; a label-store shard of a reserved log + its manifest entry; an image zip; a calibration row; a LOGLESS bank row; an EMPTY cache directory (a mis-wired probe is not a clean input) |
| the argv omits any one of `--targets --metric-cache --aux-drv-labels --images --calib`, or `--targets` has no rank files | **FAIL** (the real module would have skipped that input and read it as clean) |
| a changed reserve file / a CRLF copy | changed byte: FAIL on both pins; CRLF: FAIL on the gate's sha pin ONLY (closed) while `load_reserve` accepts it (documented asymmetry; `.gitattributes` line) |
| **the OLD adapter replayed** (parsed JSON as the reserve) | `audit_inputs` raises `ReserveError`; the clean tree FAILS: the deliberate-regression arm of the fix |
| argument types vs `inspect.signature(SR.audit_inputs)` | the gate's kwargs == the real keyword-only parameters; `bank_files` a list of existing rank files, the rest path strings |
| through the gate machinery, `launch_gate.run_checks(only=("G-RESERVE",))` | clean PASS, planted twin FAIL, planted cache key FAIL, evidence file records the reserve sha256 and the audit's counts; the live profile with unregistered pins FAILS G-RESERVE on a clean tree before the audit runs |
| the gate audits the files the trainer loads | both glob `targets_rank*.jsonl` (source-pinned); the trainer refuses any other `targets_*.jsonl` |

## 3. Mismatch 2 -- the head's assumptions the reserve half did not build: DECISION (a)

The head side's comments expected `filter_reserve.py` and a trainer flag `--reserve-manifest` in `data_flags`. The reserve half built neither: it filters inside the data chain (`p6_pod.sh` block 5a under `P7_RESERVE`, `p6_prep.py reserve-*`) and offers `session_reserve.assert_rows_clear`.
**(a) is the smaller correct design** -- drop both expectations; the gate audits the chain's outputs. Why, each point checked in source:
* the bank the gate audits IS the bank the trainer loads: the same `sorted(Path(target_dir).glob("targets_rank*.jsonl"))`, and the trainer refuses a stray `targets_*.jsonl` (`train.py` TargetBank); both source-pinned in the test;
* the other four inputs are read only by a bank row's key (metric cache by `(log, token)`, images by the row's `image_ref` / log, calibration by the row's log, label store by `(log, token)`; `FrameStore.index()` is a lookup map, not a read), so a clean bank bounds what is trained on;
* the bank is already content-audited three times (block 5a, the final audit before `ARGS`, G-RESERVE); a fourth, in the trainer, is the only thing (b) adds;
* (b) adds a flag whose DEFAULT disables it (the class this repo paid for four times in a day: needs a G-DVB entry, `required_flags`, P6 bit-identity, and it collides with the head's "every changed hunk mentions the head" source census) in the trainer P6 resumes from.

**What (a) gives up, stated:** the token fingerprints a directory by its LISTING (relpath + size, contents not hashed: `gate_token.fingerprint`), so a same-size in-place edit of the bank between G-RESERVE's content audit and the trainer's start would not be seen. A trainer-side
check would close that window for the bank. It needs a deliberate same-size edit to matter. **SPEC_P7 s.2 asks for the flag and the trainer-side refusal string**; (a) meets "the refusal string must exist in the implementation, not only in its test" at the gate (`gate_checks.py`, `session_reserve.py`) but not in `train.py`.
**Design of (b), if you want it:** `--reserve-manifest <reserve_sessions.json>` (default None = no-op, P6 bit-identical, proved by the head's F1 default-off identity), `TargetBank.__init__` calls `session_reserve.assert_rows_clear(self.rows, session_reserve.load_reserve(path))` after the rows load, `DVB_KINDS["reserve_manifest"] = "built"` with a
G-DVB row comparing the namespace with what `TargetBank` received, `--reserve-manifest` in `refe_p7`'s `required_flags` and `data_flags`, and a flag-omitted arm that must FAIL. `test_p7_integration.py` IV pins today's decision (no such flag) so adding it is a conscious update, not drift.

## 4. Mismatch 3 -- the suite

`refe_p7`'s G-SUITE = P6's + `test_p7_head_wiring.py --static` (`ZZP7HEAD_PASS`) + **`test_session_reserve.py` (`ZZSESSRES_PASS`, the real token; the head's comment said `ZZRESERVE_PASS 17`)** + `test_p7_integration.py` (`ZZP7INT_PASS`). Each script exists and prints its own token (source-pinned). `refe_p6`'s profile digest is unchanged (`80fa4d7e...`).
Cost: +~5.5 min on the pod (`test_session_reserve.py` ~290 s MEASURED here, `test_p7_integration.py` ~25 s). On a pod the reserve suite's real-data checks print themselves ABSENT (never a skip); G-SUITE only needs rc 0 and the token.

## 5. Mismatch 4 -- the pins that refuse today: the procedure, and the tests

`PROCEDURE_P7_PINS.md`: (A) `measure_rank1_3c.sh` (unchanged, the reserve package's) filters the rebuilt 3-C bank -> `train_grow_cmd_3c_p7/` + `reserve_filter_report.json`; (B) on a COPY of the full aux store: `p6_pod.sh` block 5a's own commands (`reserve-labelstore`, `aux_label_store.py finalize` twice); (C) optional filtered calibration;
(D) `p7_pins.py` READS the two numbers and prints the paste-ready lines; (E) paste into `launch_gate.py`, land, regenerate `SHIP_MANIFEST`; (F) the pod's own block 5a output must reproduce the same two lines (`diff`).
`p7_pins.py` refuses (rc 1, no registration) unless: the reserve loads; the bank has no stray; its filter report's `n_out` and `targets_sha256` equal the rows / digest recomputed here; **no reserve key or unresolved log in the bank + store (the real audit)**; the scene count equals `launch_gate.P7_SCENES` (92,896); the store opens with the trainer's own `AuxLabelSource`, every shard re-hashes, and the distinct shard-header keys equal `counts.stored_keys`; every bank log has a calibration row.
Tests (section III): the gate REFUSES with nothing registered, only one pin, and each malformed pin (short / uppercase sha, keys 0 / string / bool, tuple count 0 / string / float / bool / negative) -- via `p7_refusals`, and independently via G-CLOCK and G-DVB. With the pins set to the values `p7_pins.py` read off a synthetic tree whose reserve stage was replayed with the module's own functions
(filter bank, filter calibration, move cache + images, split + re-finalise the store): `p7_refusals` and `profile_refusals` are EMPTY; G-CLOCK's registered count == an INDEPENDENT recount of the rank files; the store pins == an independent recompute (shard bytes -> files map -> sha256; headers -> keys); G-DVB's aux rows are ALL ok on the `aux` event the trainer's own reader reports;
G-RESERVE PASSes through `run_checks`; on the stage's raw inputs (unfiltered bank + FULL store) it FAILS; the FULL store's real numbers are not the registered ones; a second replay reproduces the registration. Each `p7_pins.py` refusal has its planted-defect arm (no report, tampered `n_out`, tampered bytes, a leaked twin with a patched report, a stray file, a wrong scene count, a missing calibration row, the full store, a split-but-not-re-finalised store, a tampered `stored_keys`).
**The real numbers are PENDING:** the rank-1 exclusion, the tuple count and the store pins need the rebuilt 3-C bank (not fetched here). Nothing was invented; the estimate `~150,834` is labelled ESTIMATED everywhere and refused by nothing but the consumers (G-CLOCK / G-DVB compare the registered number with what the trainer built).

## 6. Mismatch 5 -- docs

`SPEC_P6_PAPER_FAITHFUL.md`: `SPEC_P6_bank_figure.patch` applied UNCHANGED (4 cells, 189,888 -> 167,300, each now carrying both figures with their classes; `patch --dry-run` and the real apply clean), plus a dated **CORRECTION 2026-10-22** note under the header. `SPEC_P7.md`: a dated INTEGRATION NOTE after section 2's table (what was built instead of three of its rows; the table is untouched).
`.gitattributes`: `refe/fixtures/p7_reserve_sessions.json text eol=lf` (the gate's sha pin is on the LF bytes; MEASURED on this box that `git archive` with `autocrlf=true` exports the non-`.py` fixture as CRLF, sha `4f5b1f1d...`).

## 7. VERIFICATION (generated by `finalize_int.py` from the raw outputs; do not hand-edit between the markers)

<!-- VERIFICATION:BEGIN -->
**Verification verdict (generated by finalize_int.py): NOT GREEN: mutation run not OK**

### the five suites the brief names (+ the head's `--full` and the new test), CPU (`CUDA_VISIBLE_DEVICES=-1`), BEFORE = pristine tip export, AFTER = the integrated overlay

| suite | BEFORE (tip a8f2c89) rc | BEFORE token | AFTER (overlay) rc | AFTER token | secs before/after | same |
|---|---|---|---|---|---|---|
| test_launch_gate | 0 | ZZLG_STATIC_PASS 1168 | 0 | ZZLG_STATIC_PASS 1168 | 90/127 | yes |
| test_p7_head_wiring_full | 0 | ZZP7HEAD_PASS 207 | 0 | ZZP7HEAD_PASS 209 | 1176/1063 | rc equal, count differs (see note) |
| test_session_reserve | 0 | ZZSESSRES_PASS 43  (optional checks run: ['aux_label_store.finalize', 'filter over the real rank-0 shape', 'sandbox chain (P6 default path)', 'sandbox | 0 | ZZSESSRES_PASS 43  (optional checks run: ['aux_label_store.finalize', 'filter over the real rank-0 shape', 'sandbox chain (P6 default path)', 'sandbox | 309/379 | yes |
| test_p6_chain | 0 | ZZP6CHAIN_PASS 432 | 0 | ZZP6CHAIN_PASS 432 | 1389/1439 | yes |
| test_p6_levers | 0 | ZZP6_PASS 165 | 0 | ZZP6_PASS 165 | 269/223 | yes |
| test_p7_integration | - | - | 0 | ZZP7INT_PASS 139 | -/21 | NEW |

The head test's token count is expected to differ by exactly the assertions added to its S2 / S4 blocks (181 vs 179 on the static block; `--full` adds the same 2 on top of its own 207). Every other suite's token is identical before and after.

### `test_p7_integration.py --mutations` (verbatim)

```
CONTROL (unmutated copy): PASS ZZP7INT_PASS 139
  RED    M1 REVERT THE FIX: hand the audit the parsed JSON again: 39 failing; first: I a CLEAN tree PASSes G-RESERVE (real load_reserve + real audit_inputs): ["audit_inputs raised ReserveError: 'rule' is not a session name 
  NOT RED M2 an input the argv does not name is no longer refused: 0 failing
  RED    M3 an audit with ok False no longer fails: 34 failing; first: I FAIL: a planted LEAKED RANK-1 TWIN: []
  RED    M4 the result-side 'examined all five' check is gone: 1 failing; first: I FAIL: an audit that did not examine the calibration table, whatever its ok: expected a reason containing 'not all of'; got ''
  RED    M5 the gate stops requiring the registered reserve sha256: 1 failing; first: I FAIL (closed): a CRLF checkout of the same draw fails the gate's sha pin, and ONLY that: []
  RED    M6 the gate audits another glob than the trainer loads: 8 failing; first: I ... bank_files is a list of existing rank files, the other four are path strings: {'bank_files': ['C:\\Users\\Admin\\AppData\\Local\\Tem
  RED    M7 unregistered store pins no longer refuse: 2 failing; first: III nothing registered REFUSES: the label store's pins: expected a reason containing "label store's pins"; got "refe_p7: P7_AUX_STORE_PINS
  RED    M8 an unregistered bank no longer refuses: 4 failing; first: II ... and the bank's tuple count: expected a reason containing 'no registered tuple count'; got "profile refusal: refe_p7: the P7 label s
  RED    M9 a malformed store registration is accepted: 5 failing; first: III a MALFORMED store registration is refused: a short sha: expected a reason containing 'not a registration'; got ''
  RED    M10 a malformed bank registration is accepted: 5 failing; first: III a MALFORMED bank registration is refused: 0: expected a reason containing 'not a positive int'; got ''
  RED    M11 the reserve's own suite is dropped from refe_p7's G-SUITE: 1 failing; first: IV refe_p7's suite = P6's + the head's static block + the reserve's own suite + this test: [{'script': 'test_p7_head_wiring.py', 'args': [
  RED    M12 p7_pins no longer checks the filter report's digest: 1 failing; first: III p7_pins REFUSES a bank edited after the filter (same row count, different bytes -> digest): expected 'targets_sha256'; got 'the bank h
  RED    M13 p7_pins no longer audits the bank + store: 2 failing; first: III p7_pins REFUSES a LEAKED twin whose report was patched to match (the audit still sees it): it printed a registration
  RED    M14 p7_pins reads the key count from the manifest instead of the shard headers: 1 failing; first: III p7_pins REFUSES a manifest whose stored_keys is not the shard headers' key count: it printed a registration
  RED    M15 p7_pins pins the wrong store field (the first shard's sha, not the store's): 2 failing; first: III p7_pins' store pins == an independent recompute (shard bytes -> files map -> sha256; header entries -> keys): 59b9552c2f73925a 6 vs 34
ZZP7INT_MUTATIONS_NOT_OK 1
```
<!-- VERIFICATION:END -->

Also run: `test_p7_head_wiring.py --static` -> `ZZP7HEAD_PASS 181` (179 before + the 2 new S4 arms); `test_p7_integration.py` -> `ZZP7INT_PASS 139`; `raw/rehearsal/rehearse.out` (literal CLI commands of the procedure, three refusals rc 1); `raw/probes/probe_before.out` (the defect, before the fix).
NOT run by me: the head's own `--mutations` mode (its mutations target `train.py` source and the argv, which I did not touch); the pod.

## 8. Not covered / caveats (plainly)

* **No real data.** The synthetic tree proves wiring, not the real bank's numbers; the real rank-1 exclusion, tuple count and store pins are PENDING.
* **P7 profile still refuses on purpose** and the P7 chain (`p7_pod.sh`) is not built (headline 5).
* `test_p7_integration.py` imports `test_p7_head_wiring` and `test_session_reserve` (their fixtures); a rename there is red here, by design.
* `check_reserve`'s result-side "examined all five" check trusts the module's own `counts.inputs_audited` names (`bank, metric_cache, label_store, images, calibration`); the real-module arm patches the module to drop one and requires FAIL.
* The gate's sha pin is on the LF bytes; on a CRLF checkout the gate fails closed while `load_reserve` accepts the file (`.gitattributes` line fixes the dev box; pods receive blobs).
* Timing numbers were measured with other agents' jobs running on the same 24-core box.
* Single seed / intervals: not applicable (no training, no estimate is quoted).

## 9. Deliverable manifest

| artifact | where | single location? |
|---|---|---|
| `refe/gate_checks.py`, `refe/launch_gate.py`, `refe/test_p7_head_wiring.py`, **new** `refe/test_p7_integration.py`, **new** `refe/p7_pins.py`, `SPEC_P6_PAPER_FAITHFUL.md`, `SPEC_P7.md`, `.gitattributes`, `raw/2026-10-22-p7-wire-integrate/{PROCEDURE_P7_PINS.md, RESULT.md, rehearsal/*}` | `C:/Users/Admin/par18/p7int/code/<PK path>` (sha256 + CR-lines in `code/MANIFEST.txt`) | **YES: only here** (not in any repo) |
| `code/P7_WIRE_INTEGRATE.diff` (unified, against the tip `a8f2c89`) | `C:/Users/Admin/par18/p7int/code/` | **YES** |
| `raw/probes/probe_before.{py,out}`, `raw/probes/probe_clean.py` | `C:/Users/Admin/par18/p7int/raw/probes/` | **YES** |
| `raw/before/*`, `raw/after/*` (verbatim suite output, rc, seconds), `raw/mutations.out`, `raw/VERIFICATION.md`, `raw/rehearsal/*` | `C:/Users/Admin/par18/p7int/raw/` | **YES** |
| `raw/SPEC_P6_bank_figure.patch.orig`, `raw/measure_rank1_3c.sh.orig` | copies of the reserve package's files at `PK/raw/2026-10-22-p7-wire-reserve/raw/` | in the repo (originals) |
| `run_suites_int.sh`, `finalize_int.py`, `stage_code.py`, `patches/*`, `base/`, `base_full/`, `before_full/`, `after_full/`, `work/`, `docroot/` | `C:/Users/Admin/par18/p7int/` | scratch; reproducible from the tip + `code/` |
