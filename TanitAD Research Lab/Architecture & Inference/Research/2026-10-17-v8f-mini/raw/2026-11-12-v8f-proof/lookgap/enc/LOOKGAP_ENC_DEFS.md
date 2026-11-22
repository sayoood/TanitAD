# LOOKGAP ENCODE kit (card ADD108, stage B and control m1): definitions as coded, open points

Kit for `PREREG_V8F_PROOF_ADD108.md` (frozen, not edited). Dev-box folder `F:/TanitAD-scratch/v8proof/lookgap/enc/`; pod copy `/workspace/v8proof/lookgap/kit_enc/` (pushed by `push_lookgap_enc.sh`, sha256 compared per file, never run by this kit's author). Derived from the pattern of `cropscale/enc/` (card ADD106) and from the pilot's encode kit of card ADD99 (`cropread/enc/`, read only, never edited; on the pod it stays in `/workspace/v8proof/cropread/kit/enc/`). Nothing of this kit was run on the pod; no result, token or image file was opened. Evidence class of every estimate below: ESTIMATED from the card's and the pilot's own figures.

## 1. The variant of the encode tool (`gen_window_encode_p.py` -> `window_encode_p.py`)

Source asserted by sha256 of the LF-normalised text (`ee3919f8...`, the same pilot file the generator of card ADD106 asserts). 9 counted edits, each old text exactly the stated number of times, result parsed by `ast`, leftovers of the two-arm form asserted absent, the generated file written only if absent or byte-identical.

| # | edit (old text count) |
|---|---|
| 1 | module docstring: names the variant (1) |
| 2 | arm D grids are built only `if gridsD is None and not has.all()` (1) |
| 3 | arm D sampling `out[js, i] = ...` only `if not has.all()` (1) |
| 4 | `encode_clip_tokens`: returns only tB; when every frame has a direction ONE encoder call on all 2n window images, otherwise the pilot's code path as it is (D tokens of all frames, then the direction frames written over them) (1) |
| 5 | `--arm` argument, default `P`; `--box-dir` stays required and now names the direction folder (1) |
| 6 | assert after argument parsing: arm name alphanumeric/underscore, and not `D` or `B` (the pilot's own folders) (1) |
| 7 | the six `("D", "B")` arm loops (folders, ledgers, `done()`, manifests) become `(a.arm,)` (6) |
| 8 | `tD, tB = encode_clip_tokens(...)` becomes `tB = ...` (1) |
| 9 | the write loop `(("D", tD), ("B", tB))` becomes `((a.arm, tB),)` (1) |

Kept as the pilot has it: `load_box_dirs` (has = `has_lead` and finite `ctr_az`, `ctr_el`; frame count must equal the plan), the grids `window_grid(intr, h, w, frames[i], az, el)` on the same batches of 8 frames, `window_sample`, `Encoder.window_tokens` on the same call, the float16 cast, the file layout `tok/<arm>/<clip>.npy` float16 `[n, 2, 65, 1024]` (W32 then W16), ledger and manifest.

**What the pilot's arm B does on a frame with has_lead False (kept as it is):** that frame's windows are arm D's windows (`az, el` fall back to `ARM_D_WINDOWS`: W32 at (0, -6), W16 at (0, -2)); its B tokens are arm D's tokens, taken from the D encode of all frames of the clip (`tB = tD.clone()`, then only the lead frames are overwritten). The direction files have has_lead True on every frame, so this path does not run in the 641-clip encode. It runs in the identity run (the pilot's own box folder), and the variant keeps it there (the D encode is done only in that case).

## 2. Control m1 (`ctl_m1.py`, CPU, 8 reader threads)

Record keys in lower case: `clauses.i_identity_three_clips.pass`, `clauses.ii_arm_p_files.pass`, `clauses.iii_pilot_arms_unchanged.pass`, and `pass_m1` at the top (`null` when only `identity` or `tokens` was run). `--selftest`: 16 synthetic cases, a clean world passes and every planted fault flips only its own clause.

* **i** The three clips are the first three of the pilot's evaluation list `eval141.txt` (column 1). The identity run writes arm name `B_ident` into `check_tok/tok/B_ident/`; each file's sha256 and byte size must equal the pilot's manifest of arm B (`run1/tok/B/MANIFEST.json`). Byte for byte, no tolerance (the card's wording; the kit of ADD106 used 1e-3).
* **ii** The clip list is the 500 ids of `run1/lists/train_final_ids.txt` followed by the 141 of `eval141.txt`; the count must be 641 (distinct). Every file: float16, tail `[2, 65, 1024]`, frame count equal to the frame count in the pilot's arm B manifest for that clip; its sha256 and size are in `tok/P/MANIFEST.json`; no `.npy` outside the 641. The manifest is the encode tool's own, so the hash clause tests the files against the tool's record (a file changed after the encode), not against an independent value.
* **iii** Every entry of `run1/tok/D/MANIFEST.json` and `run1/tok/B/MANIFEST.json` (size, then sha256 re-computed now). About 2 x 17.2 GB of reading.

### Clauses that could fail by construction, or are coded as the closest reading

1. **i depends on the encoder being batch-composition and run independent in float16.** With a partly lead clip the pilot's B bytes at lead frames come from an encoder call on the lead frames only, and at no-lead frames from the call on all windows of the clip. The variant repeats exactly those two calls in the identity run, so the same calls on the same inputs are made. What cannot be shown without the pod: that two runs of the pilot's encoder on the same inputs give the same bytes (GPU kernel selection, `torch` threads, `window_tokens` internal batching of `--enc-bs 8`). If the pilot's own encoder is not bit-stable, clause i fails by construction, whatever the variant does; the queue then stops before the 641 clips.
2. **i for the 641-clip run is not tested by bytes.** The variant with direction files makes ONE encoder call on all 2n windows of a clip; the pilot, on a clip with partial lead, would use a call on the lead frames only. The identity test on three clips covers the code path of the pilot's partial case plus the all-lead case only if one of the three clips is all-lead. The unit tests show the all-lead and the partial case equal the pilot's B output with a fake encoder; the real encoder is untested on the dev box.
3. **ii "its hash is in the manifest of this card"** is read as the manifest `tok/P/MANIFEST.json` that the encode tool writes (see above).
4. **iii** reads `run1/tok/{D,B}`, as the kit of ADD106 read `run1/tok/D`. If `run1/tok` is not a complete copy of the pilot's output the clause fails on a missing file.
5. **"Stops nothing"**: on a stop rule the queue ends its OWN encode child with TERM (the stop rule would mean nothing otherwise) and writes `STOP_ENC.txt`. It never touches another process.

## 3. The queue (`pod_lookgap_enc_queue.sh`)

Refuses (exit 4, nothing created) unless `/workspace/v8proof/lookgap/GO_STAGE_B.txt` and `DONE_DIRS.txt` exist. Order: lock, markers, inputs -> lists (`lists/enc641.txt` = 500 + 141 ids, `lists/identity3.txt`; written only when absent), direction file for every clip in `dirs/`, every clip staged in the pilot's stage root `cropread/stage` (checked with the pilot's `Layout`) -> variant generated on the pod from `cropread/kit/enc/window_encode.py` -> slot wait -> identity run (3 clips, pilot box folder `cropread/enc_out/box`, arm `B_ident`, `--out check_tok`) -> `ctl_m1.py identity`, STOP on failure -> slot wait -> encode of 641 clips (`--box-dir dirs --arm P --out /workspace/v8proof/lookgap`, tokens to `tok/P`) -> `ctl_m1.py all` -> `DONE_ENC.txt`.

* **Slot**: starts an encode job when fewer than 3 GPU compute processes run and more than 28 GB of the container memory are free (`memory.max` minus `anon`); polls every 60 s, STOP after `SLOT_WAIT_MAX_S` (21,600 s).
* **Stop rules, measured by the queue**: encode jobs' wall seconds (identity run + encode, summed in `ENC_GPU_SECONDS.txt`) above 7,200 s; `du -sb` of `tok/` above 20 GB; free disk under `MIN_FREE_GB` (25, the pilot tool's own floor). Checked before each step and every 30 s during a job. The identity folder `check_tok/` is outside `tok/` and not counted in the 20 GB. "GPU-hours" is read as the wall time of the encode jobs (the queue has no per-process GPU time).
* `RESUME=1` continues after a stop or after DONE; the tool's ledger skips finished clips. `STOP_ENC.txt` is not removed (the queue deletes nothing); a resumed run that stops again overwrites it. The GPU seconds carry over, so after a time stop a resume needs the lead to raise `GPU_LIMIT_S`.
* `DRY=1` prints every command and creates nothing (no folder, lock or marker).
* Environment: the pilot queue's (as in ADD106's queue).

## 4. Disk and time expected for 641 clips (ESTIMATED from the pilot's figures)

Tokens: 26.9 MB per clip and arm (MEASURED by the pilot, 26,890,368 bytes, `cropread/enc/ENC_DEFS.md`) x 641 = about 17.2 GB in `tok/P` (the card: about 17 GB), under the 20 GB rule by about 2.8 GB. Time: the card's estimate 0.35 to 1.46 GPU-hours for the encode, under the 2-hour rule; the identity run (3 clips) adds a few minutes. The pilot's per-clip wall time is not in the files read for this kit, so no figure of its own is quoted here. Control m1 hashes about 17.2 GB (ii) and 34.4 GB (iii).

## 5. Open points

* The pilot's stage root must still hold all 641 clips (mp4 and timestamps); the queue checks and stops otherwise.
* Whether any of the three identity clips has a clip with all frames lead: not known from the files read (the box files were not opened).
* Direction files are made by stage A; their m0 control is outside this kit.
* Never done by this kit: staging, committing, pushing, pod access, deleting. Dev-box scratch left by the tests: `lookgap/enc/tmp_gen/`, `lookgap/enc/tmp_pt/`.

## 6. Commands (dev box, one Python at a time)

```
cd /f/TanitAD-scratch/v8proof
PYCAP_MB=3000 bash ops/pycap.sh lookgap/enc/ctl_m1.py --selftest
PYCAP_MB=3000 bash ops/pycap.sh -m pytest lookgap/enc/test_lookgap_enc.py -q -p no:cacheprovider --basetemp lookgap/enc/tmp_pt/<new name>
```
Pod, after the PI's go: the lead writes `GO_STAGE_B.txt`, then `nohup setsid env WORKERS=5 bash /workspace/v8proof/lookgap/kit_enc/pod_lookgap_enc_queue.sh > /dev/null 2>&1 < /dev/null &`.
