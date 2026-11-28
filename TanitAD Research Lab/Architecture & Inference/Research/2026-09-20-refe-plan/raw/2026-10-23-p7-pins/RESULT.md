# P7-PINS -- the three P7 launch pins and the rank-1 exclusion, MEASURED on the real rebuilt 3-C bank

Run 2026-10-22/23 on the dev box (CPU only, pod untouched, no git tree touched, nothing committed or pushed). Procedure followed: `raw/2026-10-22-p7-wire-integrate/PROCEDURE_P7_PINS.md` at mirror tip `760cd83708ade90e7c8bf90b2fce77f8d1719337`.
Evidence classes: **MEASURED** (ours, artifact named) / **INHERITED** (another doc, not re-verified) / **ESTIMATED**. Every registered value is copied from the tool's output (`raw/stepD_p7_pins.out`), none typed.

## 1. Headline

| item | value | class | produced by |
|---|---|---|---|
| **rank-1 exclusion** | **6,556 of 64,263 excluded (10.20 %)**, 57,707 remain | MEASURED | `session_reserve.py filter` via `measure_rank1_3c.sh`: `raw/rank1_filter.json`, `raw/reserve_filter_report.json` |
| rank-0 exclusion (for the pair) | 10,141 of 103,037 excluded (9.84 %), 92,896 remain | MEASURED | same report (matches the registered 10,141 / 92,896) |
| total | 16,697 of 167,300 excluded; **n_out = 150,603** = 92,896 + 57,707; `n_unresolved 0`, `token_log_checked true` | MEASURED | same report |
| vs the old estimate | ESTIMATED ~150,834 (6,325 rank-1 excluded) was **231 tuples high**; the twins fall slightly more often into reserved sessions than uniform | ESTIMATED vs MEASURED | `raw/2026-10-22-p7-wire-reserve/RESULT.md` row 85 |
| **P7_BANK_TUPLES** | `{"train_grow_cmd_3c_p7": 150603}` | MEASURED | `p7_pins.py` |
| **P7_TARGETS_SHA256** | `{"train_grow_cmd_3c_p7": "92af8d663ecdd04a1fe7a2ed5ec0995499ca87d48a01052da4868503a92d385f"}` | MEASURED | `p7_pins.py` (== the filter report's, == the gate's `bank_targets_sha256`, == the trainer's `_digest_files` definition) |
| **P7_AUX_STORE_PINS** | `{"sha256": "df2ddd1ba2aa16a27ecea4251178038ffadaa65693be4adbe1796d74cf78e997", "keys": 92993}` | MEASURED | `p7_pins.py` on the re-finalised training store (1,076 logs) |

The tool line, verbatim (`raw/stepD_p7_pins.out`, rc 0):
```
P7_BANK_TUPLES: dict = {"train_grow_cmd_3c_p7": 150603}
P7_TARGETS_SHA256: dict = {"train_grow_cmd_3c_p7": "92af8d663ecdd04a1fe7a2ed5ec0995499ca87d48a01052da4868503a92d385f"}
P7_AUX_STORE_PINS = {"sha256": "df2ddd1ba2aa16a27ecea4251178038ffadaa65693be4adbe1796d74cf78e997", "keys": 92993}
ZZP7PINS_OK tuples=150603 scenes=92896 store_keys=92993 store_sha256=df2ddd1ba2aa16a2 targets_sha256=92af8d663ecdd04a
```
`scenes=92896` == `launch_gate.P7_SCENES`, so the 9,075-step schedule needs no re-registration. The three registration lines are identical with and without `--calib` (`raw/stepD_p7_pins_nocalib.out`).

## 2. Inputs verified (nothing used on trust)

| input | check | result | class |
|---|---|---|---|
| `refe_train_grow_cmd_3c.tar.zst` | size; full LFS sha256 against the remote listing (inside the script) and again locally | 217,062,290 B; `8b965193c23a5afe78b97a125a07e162838f5479ba4dccab227f4450ace7074d` (task said 217,062,290 / `8b965193c23a5afe`) | MEASURED `raw/stepA_measure.log` |
| `targets_rank0.jsonl` | sha256 / rows | `daa23b962e95...` / 103,037 | MEASURED |
| `targets_rank1.jsonl` | sha256 / rows | `2a63acc2f634...` / 64,263 | MEASURED |
| source bank's trainer digest | `reserve_filter_report.json` `source_targets_sha256` | `8775e73ee948...` == P6's content pin (PROCEDURE 7b) | MEASURED |
| reserve file | sha256 | `53f3bcd4684513f0...` == `P7_RESERVE_SHA256` (16 sessions, 116 logs) | MEASURED |
| full aux label store (P7.1) | `D:/Projects/TanitAD-artifacts/aux_drivable/label_store`, manifest `store_sha256` | `4a17db673b17...`, 103,139 keys, 1,192 shards (matches SESSION_RESERVE.md's 103,139) | MEASURED `raw/manifest_full_store_103139.json` |

## 3. The store step (Step B/C, on a COPY on C:)

* `p6_prep.py reserve-labelstore` -> `ZZRESERVE_LABELSTORE_OK moved=116` (`raw/stepB_reserve_labelstore.log`).
* `aux_label_store.py finalize`: training store `{"stored_keys": 92993, "logs": 1076, "bank": 92896, "db": 97}` store_sha256 `df2ddd1b...`; reserve store `{"stored_keys": 10146, "logs": 116, "bank": 10141, "db": 5}` `3d172d53...` (`raw/stepB_finalize.log`). 92,896 and 10,141 reproduce the rank-0 numbers exactly.
* `p6_prep.py reserve-calib` -> `ZZRESERVE_CALIB_OK logs_in=1192 logs_out=1076 excluded=116`. The table used is the dev-box `_rlfan_work/bank_full/calib_table.json` (sha256 `772c00abfd50...`, 1,192 logs, nuPlan `camera` table), **not the pod's `$BANK/calib_table.json`** (not reachable without the pod). It decides none of the three pins; it only fed `p7_pins.py`'s "every bank log has a calibration row" cross-check, which passed (1,076 logs).
* `p7_pins.py` audit: `session_reserve.audit_inputs` examined bank 150,603 rows, calibration 1,076, label store 2,152 and found **0 reserve keys and 0 unresolved logs**; label coverage of the bank's rows 150,603 / 150,603 = 1.0; 1,076 shards re-hashed.

## 4. The patch (`code/`) and its proof

`code/P7_PINS_ALL.diff` (= `P7_PINS_launch_gate.diff` + `P7_PINS_test_p7_integration.diff`, repo paths, LF, `git apply --check` clean against the tip export and byte-equal to the overlay after applying; apply with `-c core.autocrlf=false`, this box's autocrlf=true writes CRLF otherwise). Patched files at repo paths under `code/TanitAD Research Lab/.../refe/`. Scripts that make the edits and read the values off the tool output: `code/apply_p7_pins_patch.py`, `code/apply_p7_pins_test_patch.py`.

1. `refe/launch_gate.py`: the three UNREGISTERED lines replaced by the tool's three lines + comments (no logic touched).
2. **A finding, not a typo: registering the pins turns `test_p7_integration.py` RED.** One assertion (section V, `check_run on the LIVE profile (pins unregistered) refuses everything`) reads the live profile and expects the unregistered wording. Measured on the patched overlay before the fix: `ZZP7INT_FAIL 1 failing, 219 passing` (`raw/tests/overlay_test_p7_integration.out`). PROCEDURE 6 says the static tests tolerate registration; this one did not. Because `test_p7_integration.py` is in `refe_p7`'s G-SUITE, registering without this fix would have made the gate fail at the first launch. The fix asserts the same claim on the test copy `unregistered(P7)` and adds a live-profile twin that holds in both states.

### Proof: the pin refusals clear, every other refusal holds (`code/prove_pin_refusals.py`, outputs `raw/proof_base.json` / `raw/proof_overlay.json`)
Computed by the gate's own functions on the REAL bank and REAL training store:

| | BEFORE (tip `760cd83`) | AFTER (patched) |
|---|---|---|
| `p7_refusals(refe_p7, argv naming the real bank)` | 3 refusals: store pins unregistered, no registered tuple count, no registered targets_sha256 | **[]** |
| `profile_refusals(refe_p7, same argv)` | (refuses) | **[]** |
| G-CLOCK on the real bank (build capture faked at 150,603 / 92,896 / 9,075) | FAIL (no number, no digest) | **PASS**, evidence carries the recomputed digest `92af8d66...` |
| G-DVB aux rows on the trainer's own reader (`AuxLabelSource`) over the real training store | -- | all ok; event `rows 150603 rows_labelled 150603 store_keys 92993`, sha `df2ddd1b...` |
| `refe_p6` profile sha256 | `80fa4d7ec757...` | `80fa4d7ec757...` (identical; also == `test_p7_head_wiring.P6_PROFILE_SHA256`); its own argv refusals `[]` both |

Every other refusal still holds on the patched profile (all REFUSED, each naming the right reason):
count off by one (G-CLOCK FAIL, "150604 tuples != registered 150603"); the unfiltered `train_grow_cmd_3c` bank (name unregistered: "no registered tuple count" + "no registered targets_sha256", G-CLOCK FAIL); a bank of the **registered name** with ONE byte changed (same size, same rows) -> G-CLOCK FAIL "the bank's CONTENT is not the registered one ... the same tuple count is not the same bank"; the FULL 103,139-key store (G-DVB: sha and key count not the registered store's); store sha malformed / keys None / pins removed; tuple count 0; digest malformed; digest key absent; count and digest naming different banks; reserve sha removed; schedule steps 10,075; `--grow` ("FORBIDS"); `--aux-drv` dropped ("requires --aux-drv"). 11/11 checks ALL_OK on the patched tree; the unpatched tree shows only the expected "unregistered refuses" (2/2).

### Test suites (CPU, `CUDA_VISIBLE_DEVICES=-1`, venv `driverl-eval`; BEFORE = pristine tip export, AFTER = patched overlay; `raw/tests/`)
| suite | BEFORE (pristine tip) | AFTER (patched overlay) | note |
|---|---|---|---|
| `test_launch_gate.py` | rc 0, `ZZLG_FULL_PASS 1213`, 375 s | rc 0, `ZZLG_FULL_PASS 1213`, 375 s | the two 162-line outputs are identical after masking tree names and hashes |
| `test_p7_integration.py` | rc 0, `ZZP7INT_PASS 220` | **registered pins, UNPATCHED test: rc 1, `ZZP7INT_FAIL 1 failing, 219 passing`**; PATCHED test: rc 0, `ZZP7INT_PASS 221` | +1 = the new live-profile twin |
| `test_p7_integration.py --mutations` | `ZZP7INT_MUTATIONS_OK`, 25 RED / 0 NOT RED, control `ZZP7INT_PASS 220` | `ZZP7INT_MUTATIONS_OK`, 25 RED / 0 NOT RED, control `ZZP7INT_PASS 221` | the guarding lines still turn the suite red with the pins registered |
| `test_p7_head_wiring.py --static` (in refe_p7's G-SUITE) | rc 0, `ZZP7HEAD_PASS 182` | rc 0, `ZZP7HEAD_PASS 182` | tolerant of registration, as PROCEDURE 6 says |
| `test_session_reserve.py` (in refe_p7's G-SUITE) | rc 0, `ZZSESSRES_PASS 43` | rc 0, `ZZSESSRES_PASS 43` | 96 s each |
| `test_p6_levers.py` (P6 suite) | rc 0, `ZZP6_PASS 199`, 328 s | rc 0, `ZZP6_PASS 199`, 325 s | `refe_p6` unchanged |

`refe_p6` unchanged: profile sha256 `80fa4d7ec757...` before == after, its own argv refuses nothing before or after, and the only code edit is the three P7 constants (+ comments). NOT RUN (long, not reading any P7 constant: only the files grep-listed above do): `test_p6_chain.py` (~1,400 s in the previous package), `test_p7_head_wiring.py --full` (~1,100 s).

## 5. Durability: the filtered P7 bank is on HF

`Sayood/tanitad-refe-navtrain-banks` (private) `refe_train_grow_cmd_3c_p7.tar.zst`: **192,135,900 B, LFS sha256 `e95f39878945ea72a66cf97ac8fff7e433c6ed6b2b7f6531624f97d12ca69035`**, remote size and LFS sha256 == the local file (`ZZP7BANK_HF_OK`, re-listed independently in `raw/hf_relist.txt`; the 3-C file unchanged). Packed exactly like `bank3c_hf.sh` (`tar -cf - train_grow_cmd_3c_p7 | zstd -T2 -10`), top directory `train_grow_cmd_3c_p7/` (`targets_rank0.jsonl`, `targets_rank1.jsonl`, `reserve_filter_report.json`). Round trip verified: extracted bytes identical to the filter output and `bank_targets_sha256` of the extract == the pin. `HF_HUB_DISABLE_XET=1`; token read in place, never printed.

## 6. What is NOT done (flagged, not narrowed)

* **Registration is not landed.** The Master Mind lands `code/P7_PINS_ALL.diff` with the scoped lander, **regenerates `SHIP_MANIFEST.json`** (`launch_gate.py ship-manifest --commit <landed>`; the shipped one hashes the old `launch_gate.py` and `test_p7_integration.py`) and writes the numbers into SPEC_P7 section 2 (line 46 still reads "~150.8k ESTIMATED"; lines 58-59 still say UNREGISTERED). Not done here by instruction.
* **Step F (the pod must reproduce these three lines) and Step G (`--check-run`) are not done**: they need the pod (P6 resuming; untouched). Step F: run `p7_pins.py` on the pod's `$CBANK`/store from block 5a and `diff` against `raw/pins_paste_ready.txt`.
* G-RESERVE's five-input audit was not run end to end: the D9 metric cache and image root are pod-side. `p7_pins.py` audited bank + calibration + store (3 of 5), 0 leaks; the other two are the chain's (G-RESERVE on the pod).
* The calibration table is the dev-box copy, not the pod's (section 3); it affects no pin.
* The aux label store (full, 103,139 keys, 351 MB) was found on the dev box (`D:/Projects/TanitAD-artifacts/aux_drivable/label_store`) and is **not** in `Sayood/tanitad-refe-navtrain-banks` (15 files listed). I did not probe other Sayood repos for it. It is an INPUT whose only confirmed copy is that one disk.
* Environment deviations (no effect on values): no `zstd` binary on this box -> a stdlib shim (Python 3.14 `compression.zstd`, libzstd 1.5.7, `work/shim/zstd_shim.py`) ran the script's literal `zstd -d -c | tar -x`; the venv's `huggingface_hub` lacked `httpx/httpcore/anyio`, installed with `pip --target` into an isolated dir (venv untouched, torch untouched).

## 7. Reproduce
```
export PATH=<venv>/Scripts:<shim>:$PATH PYTHONPATH=<pydeps> HF_HUB_DISABLE_XET=1
HF_TOKEN_FILE=E:/Projects/TanitAD/Keys.txt PKG=<tree> W=<scratch> bash $PKG/raw/2026-10-22-p7-wire-reserve/raw/measure_rank1_3c.sh     # ZZR1_DOWNLOAD_OK 217062290 8b965193c23a5afe ; rank1 6556
cp -r <D:/...>/aux_drivable/label_store <scratch>/label_store_p7
python $PKG/refe/p6_prep.py reserve-labelstore --reserve $PKG/refe/fixtures/p7_reserve_sessions.json --store <scratch>/label_store_p7 --reserve-store <scratch>/label_store_p7_reserve
python $PKG/refe/aux_label_store.py finalize --out <scratch>/label_store_p7 ; ... finalize --out <scratch>/label_store_p7_reserve
python $PKG/refe/p6_prep.py reserve-calib --reserve ... --calib <calib_table.json> --out <scratch>/calib_p7.json
python $PKG/refe/p7_pins.py --reserve ... --bank $W/train_grow_cmd_3c_p7 --store <scratch>/label_store_p7 --calib <scratch>/calib_p7.json
python code/apply_p7_pins_patch.py <refe/launch_gate.py> ; python code/apply_p7_pins_test_patch.py <refe/test_p7_integration.py>
```

## 8. Deliverable manifest

Local override of rule 1: nothing staged, committed or pushed; the package is outside every git tree. "devbox" = this machine, `C:/Users/Admin/par18/p7pins/`. **ONLY-ONE-PLACE rows are marked.**

| artifact | where | note |
|---|---|---|
| `RESULT.md` (this file) | devbox `p7pins/RESULT.md` | **only one place** (the Master Mind lands the package) |
| patch at repo paths: `code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/{launch_gate.py,test_p7_integration.py}` | devbox `p7pins/code/` | **only one place**; launch_gate.py sha256 `ce67292c3c0d7bbe...`; the diff below is the review form |
| `code/P7_PINS_ALL.diff` (+ `P7_PINS_launch_gate.diff`, `P7_PINS_test_p7_integration.diff`) | devbox `p7pins/code/` | **only one place**; `git -c core.autocrlf=false apply` clean on the 760cd83 export |
| `code/apply_p7_pins_patch.py`, `code/apply_p7_pins_test_patch.py`, `code/prove_pin_refusals.py`, `code/upload_p7_bank.py` | devbox `p7pins/code/` | **only one place** |
| raw measurements: `raw/stepA_measure.log`, `stepB_*.log`, `stepC_*.log`, `stepD_p7_pins.out`, `stepD_p7_pins_nocalib.out`, `pins_paste_ready.txt`, `rank1_filter.json`, `reserve_filter_report.json`, `manifest_{training_store_p7,reserve_store,full_store_103139}.json`, `calib_in_sha256.txt`, `p7_bank_tar_sha256.txt`, `hf_relist.txt`, `stepUpload*.{log,txt}` | devbox `p7pins/raw/` | **only one place** |
| proof outputs `raw/proof_base.json`, `raw/proof_overlay.json`; test outputs `raw/tests/*.{out,rc}` | devbox `p7pins/raw/` | **only one place** |
| **filtered P7 bank** `refe_train_grow_cmd_3c_p7.tar.zst` (192,135,900 B, sha256 `e95f3987...9035`) | **HF `Sayood/tanitad-refe-navtrain-banks` (private)** + devbox `p7pins/work/upload/` | two places |
| filtered bank directory `train_grow_cmd_3c_p7/` (150,603 rows) | devbox `p7pins/work/rank1_measure/` | regenerable from HF in ~1 min; also inside the HF tar |
| source 3-C archive and extract | HF (same dataset) + devbox `p7pins/work/rank1_measure/` | two places |
| re-finalised TRAINING store `label_store_p7` (305 MB) and RESERVE store `label_store_p7_reserve` (34 MB) | devbox `p7pins/work/` | **only one place, regenerable** deterministically from the full store with section 7's commands (the pod's block 5a must reproduce the pin) |
| FULL aux label store (103,139 keys, input) | devbox `D:/Projects/TanitAD-artifacts/aux_drivable/label_store` | **only one confirmed place** (not on `Sayood/tanitad-refe-navtrain-banks`; other repos not probed) |
| scratch: `work/{base,overlay,tipfull,src,applycheck,pydeps,shim}` | devbox | throwaway; `work/shim/zstd_shim.py` and `run_tests*.sh` document how the runs were made |
