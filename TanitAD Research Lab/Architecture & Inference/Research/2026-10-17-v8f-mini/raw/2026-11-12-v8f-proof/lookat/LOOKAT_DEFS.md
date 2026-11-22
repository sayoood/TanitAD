# LOOKAT_DEFS -- kit of card PREREG_V8F_PROOF_ADD107 (LOOKAT-T0), as coded

Tier T0. Built on the dev box with tiny synthetic data; nothing of it has run on the pod. The card is frozen and was not edited.

## 0. Clauses that can fail or be empty by construction, or that the code cannot do as written (first duty)

1. **k3 and the names `DOES-NOT-LOCATE` / `NOT-SEPARATED` are partly empty.** k3 needs the upper end of `S4_perm - S4_seed0` below 0, i.e. seed 0 separated better than a head fit on permuted targets (which learns the marginal of the direction, about the fixed window).
   * If seed 0 is weak (an interval of `S4_perm - S4_seed0` that includes 0) the card fails at k3 with no name, so a weak locator cannot be named `NOT-SEPARATED` through seed 0.
   * If seed 0 is clearly below the fixed window it is also below the permuted fit: k3 fails, and `DOES-NOT-LOCATE` through seed 0 cannot be named. The name `DOES-NOT-LOCATE` is reachable only through seed 1 or seed 2 (seed 0 good, another seed upper end below 0), and `NOT-SEPARATED` only when seed 0 is separated better than the permuted fit and some other clause misses. Coded as written; the reader reports k3 as FAIL.
2. **k5 within 1e-4 as written.** The recorded `kept_sel_loss` is about 5.17 (metres, beta 1.0): 1e-4 is 2e-5 relative. The coded control recomputes it from the loaded head file in eval mode, with the same sample order, batches of 256 and a sum / n reduction, so on the same card and the same torch it is expected to agree to about 1e-6; it is NOT guaranteed on a different GPU, torch build or on the CPU (kernel choice changes the float32 sums). Run it on the pod's A40 in the venv of the fit (the queue does). PASS also needs the same sample count as the record, the head file's sha256 equal to the record's, kept epoch equal to the epoch in the head file, and the imported `crop_fit.py` sha256 beginning `61bf2f4e8e7ec61f`. If the identity run does not pass, the queue stops before any fit (k5 would fail the card anyway; it saves the GPU time).
3. **`S8_out` when a lead is more than 16 degrees off axis.** It can be computed (it is arithmetic on label and prediction), but a lead beyond 16 degrees is outside both windows (W32 reaches +-16, W16 +-8): the head can only infer it from context. The target is clipped at +-32 degrees (`t_az = +-2`), so a row with `|az| > 40` cannot reach `d_az <= 8` unless the unbounded output extrapolates beyond every training target; likewise `d_el <= 8` for `el` outside -26 to 22. The names log prints these three counts (`|az| > 16`, `|az| > 40`, el outside) as no-name label facts. No change to the rule.
4. **k4 prints a count.** The card says the count of rows with `|el + 2| > 8` is printed when it is not 0; the lead's rule says the controls stage prints only `k<n> PASS` lines. Coded: when the count is above 0 one extra line `k4 note: N rows have |el + 2| > 8 ...` is printed after the k lines (a label count, not a value of the read). k4 also requires, as my reading, a finite output of all four fits on every row of the set (the 3,841 rows come from H5 gap fits with a five-frame history, so this should hold).
5. **k0 / k1 and the pod layout.**
   * k1 needs the manifest of arm D (`MANIFEST.json` with `clips.<id>.{n_frames,bytes,sha256}`). The reader tries `--manifest`, then `<root>/tok/D/MANIFEST.json`, then `/workspace/v8proof/cropread/enc_out/tok/D/MANIFEST.json`; if none exists k1 FAILS (it cannot be shown). It hashes all 641 token files (tens of GB, 4 threads, minutes of CPU); `--no-sha` exists but then k1 fails by rule, so there is no short cut.
   * k0 reads the token file's frame count from the `.npy` header (memory map) and the box file's `has_lead` length; it needs a box file for each of the 641 clips (500 train + 141 evaluation) in `enc_out/box`. The "list hash as in the fit records of card ADD99" is read as: sha256 of `lists/train_final.txt` and `lists/eval141.txt` equal `train_list_sha256` / `eval_list_sha256` in `fits/D_H5_s0/FIT_RECORD.json`.
6. **k6 cannot be run by the reader.** It reads the lead's output file `/workspace/v8proof/lookat/kit/lead_pytest.out` (pushed with the kit; `push_lookat.sh` refuses without it): PASS = a line `N passed`, N > 0, no `failed`, no `error`. The reader cannot see that the output belongs to the pushed sources; `READER_HASHES.txt` (written by `podread_lookat.sh controls`) stores the sha256 of the kit files and the output file side by side.
7. Settings of k2: "the settings of section 2" = the settings of card ADD99 with smooth-L1 beta 0.25 (`hparams` equal to lr 1e-3, weight decay 0.05, batch 256, 40 epochs, warm-up 1, beta_m 0.25, token drop 0.1, d_model 256, 4 heads; 40 history entries). The last layer's initialisation is PyTorch's default (the card names none).

## 1. Definitions as coded

* **Samples** (`locator_fit.build_samples`, mode box): per clip, frames with `has_lead` and finite `ctr_az`, `ctr_el` (as `window_encode.load_box_dirs`) and `f >= 4` (head H5 reads frames f-4..f, as `crop_fit.build_samples` does with `lo = HIST - 1`). A box file whose frame count differs from the token file stops the fit.
* **Targets** `t_az = clip(az, -32, 32) / 16`, `t_el = clip(el + 2, -16, 16) / 8`. **Read-out** `az = 16 a`, `el = -2 + 8 e`. **Loss**: smooth L1, beta 0.25, mean over both values and the batch (the selection loss is the mean over both values and the selection samples, kept epoch = lowest, first on a tie).
* **Head**: `CropHead` of `crop_fit.make_model("H5")` with `mlp[2]` replaced by `Linear(256, 2)` and `from_z` returning the raw output (no sigmoid). The token reader, `TokenStore`, `gather`, `prefetch`, `lr_at` are imported from `--kit-fit`, not copied. Fit loop: that of `crop_fit.run_fit` (AdamW, 40 epochs, warm-up 1 epoch then cosine, token dropout 0.1, `torch.manual_seed(seed)`, `default_rng(seed)`).
* **Permuted fit** (`--permute-targets`, seed 0): the `(t_az, t_el)` pairs are permuted jointly across the fit frames (`default_rng(9900 + seed)`) and across the selection frames (`default_rng(10900 + seed)`), as `crop_fit.py --permute-labels` does for the gap.
* **Prediction file** `pred/LOC_s<K>[_perm].npz`: `clip_index` (into `clips` = the evaluation list), `frame`, `az_pred`, `el_pred` (degrees; NaN for frames 0 to 3 of a clip), `has_out`, `clips`; every frame of the 141 clips; `pred/LOC_s<K>[_perm]_meta.json` holds the sha256 of head and file.
* **Row set**: join rows (`join_eval141.npz`) with `lead_valid == 1`, finite gap in 0 to 60 m, and `has_out` with a finite gap in all five gap files. `d_az = |az_pred - az|`, `d_el = |el_pred - el|` (az, el = the box centre). `S4`, `S8`: share of rows with both `<= 4` / `<= 8`. Fixed window: `|az|`, `|el + 2|`. Outside rows: `|az| > 8`; `S8_out` = `S8` on them.
* **Interval**: `(sum h_fit - sum h_fixed) / sum rows` per resampled clip set; `numpy.random.RandomState(107).randint(0, 141, (2000, 141))` (the 141 clips of the join, clips with no row count as zero), the same draws for every interval, two-sided 95 % percentile. k3 uses `S4_perm - S4_seed0` with the same draws.
* **Names**: card section 3, in the card's order (LOCATES, CENTRES-ONLY, DOES-NOT-LOCATE, else NOT-SEPARATED). A lower end of exactly 0 is not above 0; `S8_out` exactly 0.50 holds. A second derivation (`name_second`, plain loops and hand-made percentiles, from the per-clip sums in the payload) must agree in name and intervals (1e-9), else the names stage stops (rc 4).
* **Bins (no-name tables)**: gap 0-10, 10-20, 20-40 as `lo <= g < hi`, 40-60 as `40 <= g <= 60`; `|az|` as `<= 4`, `(4, 8]`, `(8, 16]`, `> 16`.
* **Controls file** `read/LOOKAT_CONTROLS.json`: `controls` (k0 to k6, each with PASS and failures), `controls_all_pass`, and the values under `payload` only when all pass, else under `withheld`. The names stage refuses unless all seven are present and PASS.
* Families (card section 4): the log states STRATEGIC, LONGITUDINAL, LATERAL and TACTICAL as not read, n = 0, with the card's reason.

## 2. File formats

* Box file `enc_out/box/<clip>.npz`: `has_lead`, `ctr_az`, `ctr_el` per frame. Gap file: `clip_index`, `frame`, `gap`, `has_out`, `clips`. Evaluation join: `clip`, `n`, `off`, `lead_gap`, `lead_valid`.
* `fits/LOC_s<K>[_perm]/FIT_RECORD.json`: card, tier, seed, permute_targets, argv, tool_sha256, crop_fit_sha256, train_list_sha256, eval_list_sha256, clips_read, n_samples, hparams, hparams_equal_card, history, kept_epoch, kept_sel_loss, device, torch, head_pt_sha256, seconds, finished_utc. `IDENTITY_K5.json`: sel_loss_recomputed, kept_sel_loss_record, abs_diff, counts, shas, PASS.

## 3. The lead's commands, in order

Dev box (from `F:/TanitAD-scratch/v8proof`):
1. `PYCAP_MB=3000 bash ops/pycap.sh lookat/read_lookat.py --selftest`
2. `PYCAP_MB=3000 bash ops/pycap.sh -m pytest lookat/test_lookat.py -q -p no:cacheprovider --basetemp lookat/tmp_pt/<new name> > lookat/lead_pytest.out`  (writes the file k6 reads; the folder `lookat/tmp_pt` must exist)
3. `bash lookat/push_lookat.sh`

Pod (`K=/workspace/v8proof/lookat/kit`):
4. `DRY=1 bash $K/pod_lookat_queue.sh > /workspace/v8proof/lookat/dry_queue.out`
5. `nohup bash $K/pod_lookat_queue.sh > /workspace/v8proof/lookat/queue_lookat.out 2>&1 &`   (identity k5 first, then seeds 0, 1, 2 and the permuted fit; `RESUME=1` continues after a stop; markers `START_LOOKAT.txt`, `DONE_LOOKAT.txt`, `STOP_LOOKAT.txt`)
6. when `DONE_LOOKAT.txt` exists: `bash $K/podread_lookat.sh controls`  (prints k0 to k6 lines only)
7. only if all seven read PASS: `bash $K/podread_lookat.sh names`
8. dev box: `bash lookat/pull_lookat.sh`  (small text results into `lookat/pod_add107`; no head file, no prediction file)
