# CROPSCALE_FIT_DEFS -- fit, gap-file and rollout-queue kit of card PREREG_V8F_PROOF_ADD106 (CROPSCALE10K)

Tier T0 for the fits and the gap error, T1-kin/hb0.2 for the rollouts, pre-roll of 30 steps (6 s), UE_s0 and UE_s1. The card (frozen), the kits of cards ADD104 (`visseed/`) and ADD105 (`visens/`) and the fit tools of card ADD99 (`cropread/fit/`, used unchanged from their pod folder `/workspace/v8proof/cropread/kit/fit/`) were read, not edited. Everything new is in `F:/TanitAD-scratch/v8proof/cropscale/fit/`. No result file, dump, token or rollout of real data was opened; all runs are on synthetic data. The pod, the engine and the encode kit were not touched.

## 0. FIRST: clauses that could fail by construction, and what the code cannot do as written

1. **Does the unchanged fit tool accept the large list as it is?** Yes, by reading the source: `crop_fit.py:61` takes lines `<clip> fit|sel`, refuses a repeated clip and a role other than fit or sel; the tool has no cap on the number of clips; it refuses an evaluation clip in the list (`refuse_overlap`); it reads `<root>/lists/train_final.txt`, `<root>/lists/eval141.txt` and `<root>/tok/D/<clip>.npy`. **It also needs every large-list clip in the train join** (`load_join_labels`, `crop_fit.py:86`: a clip that is not in `join_train.npz` is a REFUSED stop). The card names `join_train.npz` of the pilot's data folder; I could not look whether it holds all 1,679 eligible clips. If it holds only the 500, every fit stops at its start with `clip <id> is not in the train join` (cheap: the stop comes before any epoch).
2. **What the fit record holds as its fit clips.** `FIT_RECORD.json` holds `clips_read = {"fit": [...], "sel": [...]}` in list order (`crop_fit.py:300-302`), so g2 compares the two lists to the role-fit and role-sel clips of the large list, in order. The number of fit clips is not fixed by the card (at most 450 + 1,179); g2 does not hard-code it; the record of the gap files carries it per fit (`n_fit_clips`) and in `fit_facts_for_reader`.
3. **g2 clause "the pilot's 50 selection clips".** Read as: the record's sel clips equal the pilot list's 50 role-sel clips and the large list's role-sel clips (the large list has no other sel clip). It holds only if the large list starts with the pilot's 500 lines. I added that as clause (i) of g2 (an addition: it makes (d) and (e) checkable) and the fit queue stops early on it (it compares the first lines of the large list, as many as the pilot list has, with the pilot list). If the encode helper writes the pilot lines differently (spaces, order), the fit queue stops at its start; no GPU time is lost.
4. **g2 "the settings of card ADD99".** Checked as in card ADD104: `hparams_equal_card` true, `hparams` equal to the constants copied from `crop_fit.py:32`, equal to seed 0's record. The tool hash is recorded, not part of the pass.
5. **g2 "different from the five on record".** The five are the fits of seeds 0 and 1 (`cropread/run1/fits`) and the three 500-clip fits of seeds 2, 3, 4 (`visseed10k/fitroot/fits`). The two sets are read from their `FIT_RECORD.json` files; one unreadable is a g2 failure, not a skip. I did not know whether those records exist unchanged at those pod paths (the lead gave them).
6. **g3 "each dump's meta names the head file of its fit".** `reader_s<S>_meta.json` (`crop_dump.py:103-106`) must carry `head_pt_sha256` equal to the sha256 of `fits/D_H5_s<S>/head.pt` read from disk at build time, seed S, arm D, head H5, `permute_labels` false, the ADD99 card name and `dump_sha256` equal to the dump file read.
7. **g3 "dumps on the frames of the evaluation join"** is control v0 of card ADD104 unchanged (label gap of the dump equals the join's lead gap within 1e-4 m on every label frame, a dump row for every join frame, the same clip order). It needs the three dumps to be made from the present reader's dump of seed 0 as row template, which the fit queue does as in card ADD104 (seeds 2, 3, 4 each linked to `reader_s0_score.npz`).
8. **g3 "an output exactly where all three have one"** is read as the finiteness of the three dumped gaps on that join row (a dump with no row on a join row counts as no output). The mean is the arithmetic mean of the three dumped gaps, not mapped, not clipped (the dumps hold outputs above 60 m, kept as dumped). Both mean files have seed -1, as in card ADD105.
9. **VGAPSE3 and its binding.** The card builds it from "the three 500-clip fits of seeds 2, 3, 4". It is built from their dumps on the pod (`visseed10k/fitroot/dump/D_H5`) and bound by sha256 to the dumps that the record of card ADD104 names (`files.VGAPF_s<S>.dump_sha256`), as card ADD105's e0 does; that record must read x0 and x1 PASS. A 500-clip dump whose meta does not name the head file of its fit on disk is also a g3 failure (my addition).
10. **The order of the ten rollouts.** The card says: per model seed k the five arms in this order. The queue keeps the loop of card ADD104 (`for k in ARMS` around one arm, the rollout function byte for byte): with `GR_SEED=0` or `1` (one process per model seed, as in cards ADD104 and ADD105) each process runs the five arms in the order G2, G3, G4, GEc, E3c; with `GR_SEED=ALL` the order is arm by arm over both model seeds. The tests check the per-seed order.
11. **Admission of banked runs.** RR and RLc as dump folders (as before); V0 and VD5c (card ADD102), F2, F3, F4 (card ADD104) and E5c (card ADD105) as `T1_DONE` in `integ10k/t1/<name>_s<k>`; the run VD5cx of card ADD103 is not read by this queue (the card does not name it). It was not possible to check on the pod that those folders carry exactly these names; the names are those of the queues that made them.
12. **PERM stage and g10.** The permuted-label fit writes `fits/D_H5_s2_perm` and `dump/D_H5_perm/reader_s2_{score.npz,meta.json}` (the tools' own names, `crop_fit.py:48-53`), so it cannot collide with the real fit of seed 2 (`fits/D_H5_s2`, `dump/D_H5`); its markers and job folder are separate (`START_PERM.txt`, `perm_q/`). The gap-file record does not read the permuted fit (g2 and g3 are about the three real fits); the reader of g10 reads `dump/D_H5_perm` itself. It reads the same present-dump template (`present_tpl/reader_s2_score.npz`), which the stage links if it is not there. If the lead runs the PERM stage while the FITS stage still runs, both make the same links; a link made by the other process is accepted (same target).
13. **Links in the fit root.** `tok/D/` is a real folder with one link per clip of the large list and of the evaluation list. A clip's file is taken from the pilot's folder when it exists there, else from the new folder, else STOP. A clip in both folders gets the pilot's file (the pilot's tokens are the reference of g1). **DRY mode cannot check that the token files exist** (it makes nothing and the pod paths are not there): it prints one template line for the links, not one line per clip.
14. **The T0 numbers in the record are a diagnostic, not the card's read.** The record has `t0_diagnostic` (the mean absolute gap error of the six dumps on the rows with a label lead and an output of all six, and the ratio of the large mean to the 500-clip mean). The card's statistic (rows with a lead in the label and an output of every head compared, bootstrap over the 141 clips, the five-fit measure of g9) is the reader's job.
15. **Not done by code (lead):** freezing the hash file, logging g2 and g3 in `TIMES_NOTE.md` before any name, running the pytest on the dev box (the push script refuses without `lead_pytest.out`), running the fits and the rollouts.

## 1. Files and how they were derived

| file | role | derivation |
|---|---|---|
| `pod_cropscale_fit_queue.sh` | admission, fit root, stage FITS (three fits, three dumps), stage PERM (one fit, one dump) | by hand from `visseed/pod_visseed_fit_queue.sh` (memory check, markers, resume, DRY kept); new: the admission by `ENC_CONTROLS.json`, the per-clip link folder, stage PERM |
| `make_scale_gapfiles.py` | five gap files, `CROPSCALE_GAPFILES_RECORD.json`, g2 and g3, `--selftest` | by hand; the single-file build (`frame_table`, `build_arrays`, `second_routine`, `measure_file`, v0, build check, meta clause) copied from `visseed/make_seed_gapfile.py`, the mean build (`on_join_rows`, mean, second routine) from `visens/make_ens_gapfile.py` with three dumps; a test compares both builds with the originals on synthetic data |
| `make_cropscale_queue.py` -> `pod_cropscale10k_queue.sh` | ten rollouts | generated from `visseed/pod_visseed10k_queue.sh` (sha256[:16] ba9c4bf35430e42f asserted) by 13 counted edits; the rollout function is asserted equal to those of the ADD102, ADD104 and ADD105 queues; the VD5c call of the ADD102 queue is asserted as the base of the arguments; the test regenerates the file and compares bytes |
| `test_cropscale_fit.py` | pytest, synthetic | new |
| `push_cropscale_fit.sh` | dev box to pod, folder `/workspace/v8proof/cropscale/kit_fit` | pattern of `visseed/push_visseed.sh` |

## 2. Definitions

* **Fit root** (`/workspace/v8proof/cropscale/fitroot`, built by the fit queue): `tok/D/<clip>.npy` links, `lists/train_final.txt` -> the large list, `lists/eval141.txt` -> the pilot's list, real `fits/` and `dump/`. Nothing is written into the pilot's run folder or the encode output.
* **Fits** (stage FITS): `crop_fit.py --arm D --head H5 --seed S --root $FR --join-train <join_train.npz> --device cuda` for S = 2, 3, 4, at most three at a time; then `crop_dump.py --arm D --head H5 --seed S --root $FR --present-dump <present_tpl> --device cuda`. Output `fits/D_H5_s<S>/{head.pt,FIT_RECORD.json}`, `dump/D_H5/reader_s<S>_{score.npz,meta.json}`.
* **Gap files** (folder `/workspace/v8proof/cropscale/gapfiles`): one row per frame of the evaluation join in the order of its clips; arrays `clip_index` int64, `frame` int64, `gap` float64 (metres, NaN where no output), `has_out` bool, `clips` (join clip names), `seed` int64 (fit seed for `VGAPL_s<S>`, -1 for the two mean files). Same layout as cards ADD104 and ADD105, so the unchanged wrapper variant reads them.
* **g2** (`g2_check`), clauses by letter, each per large fit: (a) record names seed S, arm D, head H5; (b) settings of card ADD99; (c) no permuted labels (flag and argv, argv names arm D, head H5, seed S); (d) fit clips equal the role-fit clips of the large list in order, list hash in the record equals the large list on disk, the pilot's fit clips are among them; (e) sel clips are the pilot's 50; (f) no evaluation clip, no clip read twice, evaluation list hash as on disk and as in seed 0's record; (g) the record's head hash equals the file on disk; (h) the eight head files (three large, three 500-clip, two of card ADD99) are all different; (i) the large list starts with the pilot's 500 lines (the pilot has 50 sel clips).
* **g3** (`g3_check`), parts: `v0`, `meta`, `build` (single files against a second routine within 1e-9 m on every row; flag = finiteness of the dumped gap; rows cover the join once in order), `mean_large` and `mean_mid` (mean files against a second routine within 1e-6 m; output exactly where all three have one; the three dumps on the same rows, each once), `mid_binding` (section 0 item 9). `g3.PASS` needs all six.

## 3. Exact format of `CROPSCALE_GAPFILES_RECORD.json` (for the reader's helper)

```
{
 "card": "PREREG_V8F_PROOF_ADD106",
 "eval_join_sha256": "<64 hex>",                       # the queue's admit() reads this
 "inputs": {"large_list"|"pilot_list"|"eval_list"|"rec104"|"eval_join": {"file": path, "sha256": hex}},
 "files": {                                            # admit() reads files.<name>.file_sha256 for all five names
   "VGAPL_s2"|"VGAPL_s3"|"VGAPL_s4": {"file", "file_sha256", "dump_file", "dump_dir_name": "D_H5", "dump_sha256", "fit_seed",
        "n_rows", "n_join_rows", "n_with_output", "n_without_output", "max_abs_dev_saved_minus_second_routine",
        "n_rows_without_dump_row", "n_output_flag_mismatch", "keys_sorted_unique", "rows_cover_the_join_in_order", ...},
   "VGAPLE"|"VGAPSE3": {"file", "file_sha256", "n_rows", "n_with_output", "n_without_output", "dump_set", "dump_sha256": {"fit2","fit3","fit4"}} },
 "dumps": {"VGAPL_s<S>": {"dump_file", "dump_dir_name", "dump_meta": <reader_s<S>_meta.json or null>, "e0": {v0 facts}}},
 "means": {"VGAPLE"|"VGAPSE3": {"facts": {"n_dumps": 3, "same_rows": [..], "saved": {"max_abs_dev_saved_minus_second_routine_m", "n_output_flag_mismatch", "n_with_output", "n_with_output_second_routine", ...}}}},
 "fits": {"large"|"mid"|"old": {"s<S>": {fit facts: seed, card, arm, head, permute_labels, argv, hparams, hparams_equal_card, train_list_sha256, eval_list_sha256,
              n_samples, head_pt_sha256_record, head_disk_sha256, kept_epoch, kept_sel_loss, record_sha256, n_fit_clips, n_sel_clips, fit_clips_sha256, sel_clips_sha256}},
          "lists": {"large_sha256", "pilot_sha256", "eval_list_sha256", "n_large", "n_large_fit", "n_large_sel", "n_pilot", "n_eval"},
          "large_dumps"|"mid_dumps": {"s<S>": {"dump_file", "dump_dir_name", "dump_sha256", "dump_meta"}}, "rec104": {"x0_PASS", "x1_PASS", "dump_sha256": {"s2","s3","s4"}, "found"}},
 "v0": {"PASS", "failures"}, "build": {...}, "meta": {...},
 "g2": {"PASS": bool, "failures": [str], "clauses": {"a".."i": bool}},
 "g3": {"PASS": bool, "failures": [str], "parts": {"v0","meta","build","mean_large","mean_mid","mid_binding": bool}},
 "fit_facts_for_reader": {"L2","L3","L4" (large), "M2","M3","M4" (500-clip), "O0","O1" (card ADD99): {"fit_seed","kept_epoch","kept_sel_loss","n_fit_clips","n_sel_clips","n_samples","head_pt_sha256"}},
 "t0_diagnostic": {...}                                # only when g3 passes; not the card's read (section 0 item 14)
}
```
The clip lists themselves are not stored (counts and sha256 only); the lists are the files named in `inputs`. Left out clips (not staged or not encoded) are the reader's difference between `lists/train_eligible.txt` and the large list, listed by the encode helper's record.

## 4. Commands for the lead, in order

Dev box (`tmp_pt` exists):
1. `cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest cropscale/fit/test_cropscale_fit.py -q -p no:cacheprovider --basetemp /f/TanitAD-scratch/v8proof/cropscale/fit/tmp_pt/<new name> > cropscale/fit/lead_pytest.out 2>&1` (whole file).
2. `PYCAP_MB=3000 bash ops/pycap.sh cropscale/fit/make_scale_gapfiles.py --selftest` (last line `ZZ MAKE-SCALE-GAPFILES-SELFTEST PASS`).
3. To rebuild the queue: `cd cropscale/fit && VQ_OVERWRITE_OWN=1 /c/Users/Admin/venvs/tanitad/Scripts/python.exe make_cropscale_queue.py`.
4. `bash /f/TanitAD-scratch/v8proof/cropscale/fit/push_cropscale_fit.sh` (last line `ZZ PUSH-CROPSCALE-FIT bad=0`; refuses without `lead_pytest.out`; never overwrites; makes `cropscale/kit_fit`).

Pod, after the encode kit has passed g0 and g1 (`ENC_CONTROLS.json`):
5. `cd /workspace/v8proof/cropscale/kit_fit && DRY=1 bash pod_cropscale_fit_queue.sh > ../dry_fits.out` (3 fit and 3 dump commands, link lines), then `bash pod_cropscale_fit_queue.sh > ../fit_queue.out 2>&1` (last line `ZZ CROPSCALE-FITS DONE`; a stop needs `RESUME=1`; the first lines say `ADMIT-OK g0 g1` and the number of token links).
6. `/workspace/venv/bin/python make_scale_gapfiles.py --eval-join /workspace/v8proof/hierrep/pins/join_eval141.npz --out-dir /workspace/v8proof/cropscale/gapfiles > ../make_gapfiles.out 2>&1` (last line `ZZ SCALE-GAPFILES g2 PASS g3 PASS (parts ...)`; on FAIL read section 0 first; all other paths are the defaults of this card).
7. `DRY=1 bash pod_cropscale10k_queue.sh > ../dry_all.out` (ten `CMD:` lines), then `GR_SEED=0 bash pod_cropscale10k_queue.sh` and `GR_SEED=1 bash pod_cropscale10k_queue.sh` (one process per model seed).
8. Beside the rollouts (GPU: the rollout queue waits for fewer than 3 compute processes; the PERM fit is one more): `PERM=1 bash pod_cropscale_fit_queue.sh > ../perm_queue.out 2>&1` (last line `ZZ CROPSCALE-PERM DONE`).

## 5. Temp files left in place (no deletes)

`cropscale/fit/tmp_pt/` (pytest basetemp folders), `cropscale/fit/tmp_selftest_scale_*/` (selftest output, small). `__pycache__` is not written.
