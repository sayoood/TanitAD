# CROPSCALE ENCODE kit (card ADD106, sections 2 and 5 g0, g1): definitions, assumptions, commands, estimates

Kit for `PREREG_V8F_PROOF_ADD106.md` (frozen, not edited). Dev-box folder `F:/TanitAD-scratch/v8proof/cropscale/enc/`; pod copy `/workspace/v8proof/cropscale/kit_enc/` (pushed by `push_cropscale_enc.sh`, sha256 compared per file). Derived from the encode kit of card ADD99 (`cropread/enc/`, read only; it stays in its pod folder `/workspace/v8proof/cropread/kit/enc/` and is called from there). Evidence class of every number: ESTIMATED from the pilot's files unless it says MEASURED; nothing of this kit was run on the pod, and no result file, token or rollout was opened.

## 0. FIRST: which encode case applies, and the clauses that could fail by construction or that the code cannot do as written

**Encode case: the generator case.** `window_encode.py` of the pilot cannot write arm D alone: it requires `--box-dir`, builds the four windows per frame (D32, D16, B32, B16), and its `done()`, ledger, files and manifest are for both arms (`window_encode.py:148,160,164,167,179,205`). Called unchanged it would also write arm B (26.9 MB per clip): 1,179 new clips x 2 x 26.9 MB = 63.4 GB of tokens, above the card's 60 GB limit before a single video is counted. So `gen_window_encode_d.py` makes `window_encode_d.py` by 12 counted edits (each old text exactly the stated number of times, source sha256 `ee3919f8107065379ec96c29d4c80f7533a14b716b2e48926008c2d4b2aae7e9` asserted on the LF-normalised text, the result parsed by `ast`, B names asserted absent). The D path is untouched: same two grids built once per call from the first batch's frame size, the same `window_sample`, the same `Encoder.window_tokens` on the same batches (n x 2 windows per clip), the same float16 cast. Changed only: no box read, `out` array has 2 windows instead of 4, no B sampling, no B encode, ledger/manifest/files for D only, `--box-dir` removed, `n_lead_frames` field and log text dropped. MEASURED on the dev box: the generator ran on the dev copy of the pilot file and the diff is exactly those edits.

Clauses that could fail by construction, or that are coded as the closest reading:

1. **g1 identity (three clips) needs the pilot's staged inputs.** The three clips are staged in the pilot's stage root `/workspace/v8proof/cropread/stage` (never swept as far as the files say). The queue reads them from there (read only); if the mp4, timestamps, egomotion or the calibration csv of any of the three is missing there, the queue stages those three pilot clips (not new clips) into the new stage root first. The comparison is `max |pilot - check|` over float32 values of the float16 tokens, tolerance 1e-3, exactly three clips. The check files are at `cropscale/check_tok/tok/D/<clip>.npy` (the encode tool's own layout under `--out`), which is what "into `check_tok`" is read as.
2. **"Shape and type of every new token file"**: the card names shape and type; the code also requires `n_frames` equal to the frame count `n` of that clip in `/workspace/v8proof/data/inputs/join_train.npz` (as c7 of the pilot did), because the fit tool refuses a clip whose token frames differ from the join (`crop_fit.py` build_samples). Any `.npy` in `enc_out/tok/D` that is not in the large list also fails the clause (a stray file).
3. **"The token files of the pilot are unchanged against the pilot's manifest"**: read as every entry of `run1/tok/D/MANIFEST.json` (641 clips: 500 train + 141 evaluation; bytes and sha256 re-computed now). That is about 17 GB of reading. The manifest of `enc_out/tok/D` (the pilot's encode output) is not used; if `run1/tok/D` is not a complete copy of it the clause fails by construction on a missing file, which is what it is for.
3b. **Pilot-skipped clips are tried again.** The pilot's replacement rule skipped some clips among its first 520 positions; they are not among the pilot's 500, so they are in the new list and will fail to stage again for the same reason (for example "no obstacle source for this clip"). The unchanged stage tool refuses a train clip with no obstacle rows although arm D does not use them; this card keeps the tool unchanged, so such clips are left out and listed (g0 asks only that they are listed with a reason).
4. **g0 "selection set is the pilot's 50 clips"** requires exactly 50 `sel` lines in the pilot's `train_final.txt` and in the large list. If the pilot's list has another count (it should be 450 fit + 50 sel, `make_final.py:11`), the clause fails by construction.
5. **Two added clauses** are consistency checks of the code, not of the card: `g0_no_duplicates` (no clip twice, new lines role `fit` and in the order of `train_eligible.txt`) and `g0_record_matches_file` (record hash and counts equal the file). `g0` passes only if all seven pass; the five of the card are named first in `ENC_CONTROLS.json`.
6. **Disk.** The card's 60 GB limit is close to the estimate in section 4 (about 54 GB for this kit alone). The limit is a `du` of the whole `/workspace/v8proof/cropscale`, so the lead's probe folder and the later fit files count too; the queue stops with `STOP_ENC.txt` above 60 GB and does not start again without `RESUME_AFTER_STOP=1` (a lead decision: the card says the card goes back to the PI). The free-disk read is `df`, which on a pod volume may not be the quota; `du` is exact.
7. **`DONE_ENC.txt` is written even when a control fails** (it carries `controls_rc`, the pilot queue's pattern); the record `ENC_CONTROLS.json` says which clause. With one control failed nothing of the card is opened.
8. The card's order "memory check -> lists -> identity first -> stage -> encode -> final list -> controls" is the queue's order. The identity compare cannot fail the later steps silently: its failure ends the queue before any new clip is staged.

## 1. Files (all in `cropscale/enc/`)

| file | job |
|---|---|
| `make_large_list.py` | modes `new-order` (eligible minus the pilot's 500 -> `lists/train_new_order.txt`), `staged` (content-addressed `train_new_staged_<hash>.txt`), `final` (`lists/train_final.txt` + `LISTS_LARGE_RECORD.json`) |
| `gen_window_encode_d.py` | makes `window_encode_d.py` from the pilot's tool (counted edits, hash asserted; never overwrites a different file) |
| `ctl_g0g1.py` | controls g0, g1 -> `ENC_CONTROLS.json`; `identity` -> `ENC_IDENTITY.json`; `--selftest` (21 cases, one planted fault per clause) |
| `pod_cropscale_enc_queue.sh` | the queue (markers, stop rules, DRY, lock, resumable) |
| `push_cropscale_enc.sh` | push to `/workspace/v8proof/cropscale/kit_enc`, noclobber, sha256 compare; refuses without `lead_pytest.out` |
| `test_cropscale_enc.py` | 17 pytest tests, synthetic only |
| `CROPSCALE_ENC_DEFS.md` | this file |

Unchanged tools used from the pilot's pod folder: `stage_clips.py`, `window_encode.py` (only as the source of the variant), `enc_common.py` (imported through `PYTHONPATH`).

## 2. Definitions with sources

* **Train list** (card section 2): `train_eligible.txt` = `/workspace/v8proof/cropread/enc_out/lists/train_eligible.txt` (the pilot queue's `$RUN/lists`; `pod_cropread_enc_queue.sh:28,62`, written by `make_lists.py:91`), 1,679 clips. New clips = eligible minus the clips of the pilot's `train_final.txt` (`/workspace/v8proof/cropread/run1/lists/train_final.txt`, 500 lines `<clip> <role>`), in the order of `train_eligible.txt`: 1,179 clips. Large list = the pilot's 500 lines unchanged, then every new clip with a token file, role `fit`. Format as `crop_fit.py:61-72` reads it (`<clip> fit|sel`, no repeated clip).
* **Left out** (`make_large_list.build_final`): a new clip with no token file in `tok/D` (manifest entry and file size equal) is left out with `did not stage: <ledger reason or "no ledger line">` when the stage ledger has no `ok` row, else `did not encode: <message of ENCODE_FAILS*.json or "no token file in the manifest">`. The queue keeps a copy of `ENCODE_FAILS.json` per encode call (the tool rewrites it each call).
* **Staging**: `stage_clips.py --kind train --list lists/train_new_order.txt --root /workspace/v8proof/cropscale/stage --hf-dir /workspace/v8proof/cropscale/hfdl --first N`, in chunks of 150 (`CHUNK`), so that the stop rules are read between chunks; the tool skips a staged clip (`stage_clips.py:194-195`) and reads its token in place (`stage_clips.py:37-41`; the queue never reads or prints it). Ledger: `stage/ledger.jsonl`.
* **Encode of arm D**: `window_encode_d.py --list <staged ok clips> --root stage --out /workspace/v8proof/cropscale/enc_out --workers $WORKERS --min-free-gb 100` -> `enc_out/tok/D/<clip>.npy` float16 `[n_frames, 2, 65, 1024]`, `enc_out/tok/D/ledger.jsonl`, `enc_out/tok/D/MANIFEST.json` (`n_frames`, `sha256`, `bytes` per clip; the pilot's format, `window_encode.py:136-141`), `enc_out/tok/ENCODE_FAILS.json`. The windows are the pilot's: W32 at (0, -6 degrees) and W16 at (0, -2) (`enc_common.py:30`).
* **Environment**: the pilot queue's lines 40 to 46, word for word (a test asserts each), plus the pilot's kit directory on `PYTHONPATH` so that `enc_common` imports.
* **Stop rules** (card section 1 budget): `diskcheck` = `du -sb /workspace/v8proof/cropscale` above 60 GB or `df` free under 100 GB; read before every step, after every stage chunk, and every 30 s during the encode (a stop in the encode ends only the encode child this queue started, with TERM; the encode tool's own `.part` rule makes it resumable). Memory: `memory.max` minus the `anon` line of `memory.stat`, 28 GB kept free, before every step and chunk (the pilot's `memcheck`, `pod_cropread_enc_queue.sh:48-53`). A GPU process on the pod stops the queue before the identity encode and the encode (the pilot's `gpucheck`).
* **Controls** (`ctl_g0g1.py`): clauses listed in its header. g0 reads the three list files and the record; g1 re-hashes every new token file (4 threads) and every file of the pilot's manifest, checks shape, type and frame count by memory map, and compares the three identity clips in float32.

## 3. Time stamps and markers

`ENC_TIMES.txt` gets a line at start, after the identity compare, after every stage chunk, after the encode and after the controls. Markers in the root: `START_ENC.txt`, `DONE_ENC.txt`, `STOP_ENC.txt` (one line why). One log `enc_queue.log`; lock `enc_queue.lock` (a second start is refused and logged). `DRY=1` prints every command and creates nothing (no folder, file, lock or marker; tested in a scratch folder).

## 4. Estimates of run time and disk (ESTIMATED from the pilot's files, `cropread/enc/ENC_DEFS.md` sections 4 and 5)

| item | basis | estimate |
|---|---|---|
| stage 1,179 clips | 0.6-0.9 s per sha-verified mp4 download, 4 workers; one-off tars (egomotion 2.0 GB about 8 s, timestamps 2 s), up to 48 obstacle shards of about 52 MB | 10 to 15 min |
| window cuts + decode | 10 s per clip at one decode thread, `WORKERS=5` | 1,179 x 10 / 5 = about 39 min (decode bound) |
| GPU encode, arm D only | pilot 2.0-2.6 s per clip with both arms; D is the 202 windows of one clip | about 1.2 to 1.5 s per clip = 24 to 30 min, overlapped with the decode |
| identity encode + compare | lead's probe: 23 s for 3 clips both arms | under 1 min plus start-up |
| controls g0, g1 | re-hash about 49 GB (32 GB new + 17 GB pilot), 4 threads | 3 to 10 min |
| **whole queue** | | **about 1.1 to 1.5 h** (the card's own estimate: about 2 h) |
| tokens | 26,890,368 bytes per clip (MEASURED in the pilot) x 1,179 | 31.7 GB |
| staged mp4 | about 15 MB per clip (pilot estimate; smoke 5 to 27 MB) x 1,179 | about 17.7 GB |
| shared pulls `hfdl` | pilot 2.1 GB, up to 4.5 GB with all shards | 2 to 4.5 GB |
| ego, timestamps, obstacle rows, check tokens (3 x 27 MB) | small | under 1 GB |
| **files of this kit** | | **about 52 to 55 GB** against the card's 60 GB limit |

## 5. The lead's commands, in order

```
# 1. tests on the dev box (one Python at a time; create tmp_pt first, new basetemp name each time)
cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest cropscale/enc/test_cropscale_enc.py -q -p no:cacheprovider --basetemp /f/TanitAD-scratch/v8proof/cropscale/enc/tmp_pt/<new name> 2>&1 | tee cropscale/enc/lead_pytest.out
PYCAP_MB=3000 bash ops/pycap.sh cropscale/enc/ctl_g0g1.py --selftest        # 21 cases, last line ZZ SELFTEST PASS
# 2. push (refuses without lead_pytest.out)
bash /f/TanitAD-scratch/v8proof/cropscale/enc/push_cropscale_enc.sh
# 3. on the pod: read memory.max and anon (keep 28 GB free), no other GPU process; then the dry run
DRY=1 bash /workspace/v8proof/cropscale/kit_enc/pod_cropscale_enc_queue.sh
# 4. start, detached
nohup setsid env WORKERS=5 STAGE_WORKERS=4 bash /workspace/v8proof/cropscale/kit_enc/pod_cropscale_enc_queue.sh > /dev/null 2>&1 < /dev/null &
# 5. wait (one pssh call, remote loop, at most 540 s per call)
until [ -e /workspace/v8proof/cropscale/DONE_ENC.txt ] || [ -e /workspace/v8proof/cropscale/STOP_ENC.txt ]; do sleep 20; done
# 6. read: ENC_IDENTITY.json (written first), ENC_TIMES.txt, ENC_CONTROLS.json (pass_g0, pass_g1, pass_all), lists/LISTS_LARGE_RECORD.json
```
For the fit kit: the large list is `/workspace/v8proof/cropscale/lists/train_final.txt`; the new tokens are `/workspace/v8proof/cropscale/enc_out/tok/D/<clip>.npy` and the pilot's are in `/workspace/v8proof/cropread/run1/tok/D/` (the fit tool reads `<root>/tok/D/<clip>.npy` and `<root>/lists/eval141.txt`; assembling that root is not part of this kit).

## 6. Temporary and kept files (no delete was issued)

* dev box: `cropscale/enc/tmp_pt/` (pytest basetemp folders), `cropscale/enc/tmp_gen/window_encode_d.py` (the generator's test output from the dev copy; the pod makes its own).
* pod, after the run: `stage/` (mp4 about 17.7 GB, re-creatable from HF), `hfdl/` (shared pulls, re-creatable), `check_tok/` (3 tokens), `enc_out/` (tokens, ledger, manifest), `lists/`, markers, log, time stamps. Nothing of it is deleted by the queue; the mp4 folder and `hfdl` are the first to sweep when the card is finished.
