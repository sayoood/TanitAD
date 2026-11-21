# PROCEDURE_P7_PINS -- how `P7_BANK_TUPLES`, `P7_TARGETS_SHA256` and `P7_AUX_STORE_PINS` are produced (nothing is typed from an estimate)

Written 2026-10-22 by P7-WIRE-INTEGRATE; THIRD PIN (the bank's content digest) and Step G added 2026-10-22 by P7-WIRE-INTEGRATE-2. `refe_p7` REFUSES today on purpose; this is the exact way it stops refusing. 0 GPU, CPU only, ~minutes.
Evidence classes below: MEASURED (this package, artifact named) / INHERITED (another doc, not re-verified) / PENDING (cannot be run yet).

## 0. What refuses today, and what each pin is

| pin (`refe/launch_gate.py`, P7 section) | what it is | refuses where |
|---|---|---|
| `P7_BANK_TUPLES = {}` | the tuple count (= rows of `targets_rank*.jsonl`) of the bank directory `train_grow_cmd_3c_p7`, keyed by the DIRECTORY NAME the launch argv's `--targets` names | `p7_refusals` (before any check) and, independently, G-CLOCK (`clock_bank_tuples` -> "has no registered tuple count") |
| `P7_TARGETS_SHA256 = {}` | the bank's CONTENT: the trainer's own `targets_sha256` (sha256 over basename + bytes of each `targets_rank*.jsonl`, sorted) of that SAME directory, keyed by the same directory name. A count cannot tell two same-size banks apart; this can. It is the value `train.run_identity` records in `config.json` / `ckpt_last.pt` and `train._identity_diff` enforces on every `--resume` (train.py:1077-1125, 1525) | `p7_refusals` (before any check; also refuses a profile that LACKS the key, a malformed digest, and a count and a digest naming different banks) and, independently, G-CLOCK (`clock_bank_sha256`: the gate hashes the `--targets` files at launch and FAILS on any other bank; `refe_p7` without the key FAILS too, never a skip) |
| `P7_AUX_STORE_PINS = {"sha256": None, "keys": None}` | the manifest `store_sha256` and `counts.stored_keys` of the TRAINING label store AFTER the reserve's shards were moved out and the store was re-finalised | `p7_refusals` and, independently, G-DVB's `aux_drv_labels` rows (the trainer's `aux` event is compared with the registered pins; there is no fallback to the full store, which holds the reserve's keys) |

`P7_RESERVE_SHA256` is already registered (`53f3bcd4...`, == `session_reserve.RESERVE_FILE_SHA256`; `test_p7_integration.py` pins the equality).
A pin is also refused if it is **malformed** (sha256 / digest not 64 lowercase hex, keys / tuple count not a positive `int`): a typed value that is not the tool's output is refused, but a well-formed WRONG number is only
caught by the consumers (G-CLOCK compares it with the bank the trainer actually built; G-DVB with the store the trainer actually opened). So: **paste what `p7_pins.py` printed, never a number from a document.**
`~150,834` (RESULT of P7-WIRE-RESERVE) is ESTIMATED (rank-1 twins assumed to fall uniformly over scenes); it must never be registered.

## 1. Preconditions (each one a fact to check, not assume)

1. The rebuilt 3-C bank is on HF: `bank3c_hf.sh` printed `ZZB3_OK` (dataset `Sayood/tanitad-refe-navtrain-banks`, file `refe_train_grow_cmd_3c.tar.zst`). PENDING at the time of writing (the pod was still rebuilding it).
2. The full aux label store is on the dev box: `D:/Projects/TanitAD-artifacts/aux_drivable/label_store` (a finalised store, `manifest.json` present). It holds 103,139 frames (INHERITED from SESSION_RESERVE.md).
   It is READ-ONLY here: step B works on a COPY.
3. The reserve file is the LF form, sha256 `53f3bcd4684513f0ee765fbcfbaf583b587f58c97e9c8915f4e4df3ae3e32f29` (`refe/fixtures/p7_reserve_sessions.json`; MEASURED by `test_p7_integration.py`). A CRLF checkout (`core.autocrlf=true`)
   is the same text: `load_reserve` accepts it, but the GATE's sha pin does not (fail closed). Use the repo's LF bytes.
4. `PKG` = a tree holding `refe/` and `raw/2026-10-22-p7-session-reserve/` (for the token table). Python with numpy (the `driverl-eval` venv on this box). Scratch space OUTSIDE the repo and not at the top level of `D:` (exFAT, 1 MiB clusters).

## 2. Step A -- the bank (the filter, rank-0 AND the twins)

```
HF_TOKEN_FILE=<path to Keys.txt> PKG=<tree> W=<scratch dir> bash "$PKG/raw/2026-10-22-p7-wire-reserve/raw/measure_rank1_3c.sh"
```
(The script is the reserve package's `raw/measure_rank1_3c.sh`; it is unchanged. It reads the token from the file in place and never prints it.) It downloads and verifies the archive, extracts `train_grow_cmd_3c`, **STOPS if the rank files
are not the expected ones** (sha256 prefixes `daa23b96` / `2a63acc2`, 167,300 rows: INHERITED expectations from PREREG_3C.addendum6), then runs
`session_reserve.py filter --reserve <fixture> --bank <3c> --out $W/train_grow_cmd_3c_p7 --token-table <token table>` and writes `reserve_filter_report.json` beside the output.
Success reads `n_in 167300`, `by_rank_in {"0": 103037, "1": 64263}`, `by_rank_excluded {"0": 10141, "1": <THE RANK-1 NUMBER>}`, `token_log_checked true`.
The output directory name **`train_grow_cmd_3c_p7`** is the key the gate registers the count under: do not rename it.

## 3. Step B -- the training label store (split the reserve's shards out, RE-FINALISE) -- on a copy

These are block 5a's own commands (`p6_pod.sh`, the `P7_LABEL_STORE` branch), so the dev-box numbers and the pod's come from the same code:
```
cp -r D:/Projects/TanitAD-artifacts/aux_drivable/label_store  <scratch>/label_store_p7
python refe/p6_prep.py reserve-labelstore --reserve refe/fixtures/p7_reserve_sessions.json --store <scratch>/label_store_p7 --reserve-store <scratch>/label_store_p7_reserve     # ZZRESERVE_LABELSTORE_OK moved=<n>
python refe/aux_label_store.py finalize --out <scratch>/label_store_p7
python refe/aux_label_store.py finalize --out <scratch>/label_store_p7_reserve
```
`store_sha256` hashes the `shards/<log>.aux1` files map (names + file sha256), not paths or timestamps, so a copy gives the pod's number.

## 4. Step C (optional cross-check) -- the calibration table without the reserve's logs

```
python refe/p6_prep.py reserve-calib --reserve refe/fixtures/p7_reserve_sessions.json --calib <the P6 calib_table.json ($BANK/calib_table.json)> --out <scratch>/calib_p7.json
```

## 5. Step D -- READ the three pins off the outputs

```
python refe/p7_pins.py --reserve refe/fixtures/p7_reserve_sessions.json --bank $W/train_grow_cmd_3c_p7 --store <scratch>/label_store_p7 --calib <scratch>/calib_p7.json
```
It prints a JSON report and three paste-ready lines, then `ZZP7PINS_OK tuples=<n> scenes=<s> store_keys=<k> store_sha256=<16> targets_sha256=<16>`:
```
P7_BANK_TUPLES: dict = {"train_grow_cmd_3c_p7": <n>}
P7_TARGETS_SHA256: dict = {"train_grow_cmd_3c_p7": "<64 hex>"}
P7_AUX_STORE_PINS = {"sha256": "<64 hex>", "keys": <k>}
```
It exits **1 and prints no registration** (`ZZP7PINS_REFUSED <why>`) unless: the reserve loads; the bank holds `targets_rank*.jsonl` and no stray `targets_*.jsonl`; `reserve_filter_report.json` is present with `n_out` == the rows
counted here and `targets_sha256` == the digest recomputed here **and equal to the GATE's own definition (`gate_checks.bank_targets_sha256`: two copies of one definition are compared, so a drift is a refusal now rather than a G-CLOCK FAIL on the pod)**; no scene holds a rank twice; **`session_reserve.audit_inputs` finds no reserve key and no unresolved log in the bank and the store**; the bank holds
exactly `launch_gate.P7_SCENES` = 92,896 scenes (`--expect-scenes` overrides ONLY for a synthetic tree); the store opens with the trainer's own reader (`AuxLabelSource`, pinned to its manifest), every shard re-hashes, and the shard
headers' distinct key count equals `counts.stored_keys`; every bank log has a calibration row (the trainer DROPS a tuple without one, and then its count differs from the file's).
If the scene count is not 92,896 the pin is NOT the problem: the schedule (25 x ceil(S/256) = 9,075 steps) was sized on 92,896 and is a PI-level re-registration. Do not pin around it.
`coverage_of_bank_rows` is reported (G-DVB enforces >= `AUX_MIN_COVERAGE` on the trainer's own count).

## 6. Step E -- register

Paste the three lines into `refe/launch_gate.py` (replacing `P7_BANK_TUPLES: dict = {}`, `P7_TARGETS_SHA256: dict = {}` and the `P7_AUX_STORE_PINS` line), write the same numbers into the P7 pre-registration (SPEC_P7 2: "registered before step 0"),
land with the scoped lander, and regenerate `SHIP_MANIFEST.json` (`launch_gate.py ship-manifest`). The static tests that guard these pins (`test_p7_head_wiring.py` S2, `test_p7_integration.py` IV) accept
"unregistered OR well-formed", so registering does not turn the G-SUITE red. Nothing else needs editing.

## 7. Step F -- the pod must reproduce the dev-box numbers

The chain's block 5a (`P7_RESERVE` + `P7_LABEL_STORE`, `P6_STOP_AFTER=reserve` for a dry run) produces `$CBANK` (`..._p7`) and the re-finalised store on the pod. Run `p7_pins.py` on THOSE and `diff` its three lines against step D's:
identical, or STOP (the filter is deterministic, the bank digest is a hash of the filtered files' bytes and the store pin depends only on shard bytes: a difference means different inputs). The gate then re-checks the numbers against what the trainer actually built (G-CLOCK: count AND content, G-DVB).

## 7b. Step G -- AFTER the first start: the trainer's OWN recorded identity must be the registration

```
python refe/p7_pins.py --check-run <the run's --out directory>        # rc 0 ZZP7RUN_OK | rc 1 ZZP7RUN_REFUSED <why>
```
The trainer writes `config.json` (with `identity.targets_sha256` and `identity.n_tuples`, computed by `train.run_identity` on the bank it actually loaded) before step 0, and enforces the same identity on every `--resume`
(P6's content pin `8775e73e...` is exactly this resume identity, `eval/PREREG_P6.md:191`; a FRESH start has nothing recorded to compare with, which is why the profile registers the digest and G-CLOCK refuses at launch).
The PASS token fingerprints a directory by its LISTING (relpath + size), so a same-size in-place edit of the bank between G-CLOCK and the trainer's start is not visible to `verify`; this step closes that window with the
trainer's own number. It belongs in the P7 chain right after the first checkpoint's identity exists (`p7_pod.sh`, OPEN). A failure means STOP: the run trained on a bank the profile never vouched for.

## 8. Proof that this procedure and its refusals work (MEASURED, this package)

* `raw/rehearsal/rehearse.sh` runs steps A-D's LITERAL commands (the filter, `reserve-labelstore`, both `finalize`, `reserve-calib`, `p7_pins.py`) on a synthetic tree with the real reserve file: `raw/rehearsal/rehearse.out`.
  The same script then shows three refusals (unfiltered bank, the FULL store, a scene count the schedule is not sized on), each rc 1.
* `refe/test_p7_integration.py` section V (P7-WIRE-INTEGRATE-2): a SAME-COUNT, DIFFERENT-BYTES bank (one byte flipped in place; two rows swapped) is refused by G-CLOCK, directly and through `run_checks`, while the count-only gate
  PASSes it (the red arm); the gate's digest function has the AST of the trainer's `_digest_files`; `--check-run` is exercised on the registered identity and on each defect.
* `refe/test_p7_integration.py` section III: the gate REFUSES with nothing / only one pin / a malformed pin registered; with the pins set to a synthetic tree's `p7_pins.py` output the profile is clean,
  G-CLOCK's registered count equals an independent recount of the rank files, G-DVB's store rows are all ok on the `aux` event the trainer's own reader reports, and G-RESERVE PASSes through `run_checks`;
  the FULL store is not the registered one (real numbers); a second replay reproduces the same registration. The tool's own refusals each have a planted-defect arm; `--mutations` deletes the guarding lines and requires RED.

## 9. Not covered

The real numbers: the 3-C bank was not available to this package (no network, no pod). The rank-1 exclusion, the tuple count and the store pins are PENDING until steps A-D run. Downstream arms (RL, scorer fine-tunes) are not wired.
