# BOXREAD_READ_DEFS -- reader of card PREREG_V8F_PROOF_ADD127 (BOXREAD-T0), as coded

Tier T0: recorded frames, no rollout; a diagnostic of a perception read and not driving performance. Files: `box_geom.py` (geometry, numpy), `read_boxread.py` (controls and names), `podread_boxread.sh`, `read_args.txt`, `test_read_boxread.py`. The reader imports `read_boxsize.py` of card ADD124 unchanged (path env `BOXSIZE_READER`; the import is refused unless its sha256 begins `b88aecf06dd504c4`) and uses its `MapG`, `lay_dump`, `lay_pred`, `box_rows`, `t3_rows`, `boot_ratio`, `boot_diff_over`, `gap_bin_mask`, `mean_abs_err`, `ids_of`, `dump_paths`. `enc_common.py` of card ADD99 is imported only for the render tool test of b1 (env `BOXREAD_ENC_COMMON`).

## 0. Open points (read first)

1. **Which locator fit.** The card says "its banked predictions" of card ADD107 and names no seed. The reader takes `<--locator-dir>/<--locator-fit>.npz`, default `LOC_s0` (seed 0). The prediction files hold NaN for frames 0 to 3 of every clip (the head reads five frames), so those frames have no direction and no pick; the evaluation rows come from five-frame-history gap fits, so this should cost nothing, but b5 counts such rows if they exist.
2. **Tele direction.** The locator's direction is in the wide camera's frame. It is carried into the tele camera by the extrinsics of both cameras of the clip (`--wide-extr`, `--tele-extr`: the csv files `front_wide_extrinsics.csv` and `front_tele_extrinsics.csv`, columns clip_id, qx, qy, qz, qw), by the routine of `tele_encode.point_in_other_cam` (here vectorised, the full rotation applied). A clip with a missing row has no tele read. These two arguments are new inputs the pod must supply.
3. **frame_run.** The detector runs only on the frames with a labelled lead (the lead's addendum). An evaluation row whose wide frame has `frame_run` false (or `frame_ok` false) is a missing input: b7 fails and names the count. A tele row with `frame_run` false has no tele pick and falls back to BW. Train frames with `frame_run` false are left out of the map, b1, b2 and b3 and are counted (`n_lead_frames_not_run_left_out` in b1, `n_frames_not_run_left_out` in b2). A detector file with no `frame_run` key fails b0.
4. **Denominators.** b1: train frames with a lead (a finite label direction) and `frame_run` true. b2: those frames with a finite label gap of at most 20 m. A clip with no box file or no wide detector file fails b2 with the count; a detector file whose `n_frames` differs from the box file's frame count fails b2.
5. **b3 constant.** "The median train gap" is the median of the gaps of the fitting folds' picked frames; the permuted map uses the same constant. Folds: clip i of the train list belongs to fold i mod 5. The permutation of each fold uses `RandomState(127 + fold)`.
6. **b0 and row 191.** The three errors of arm D are computed here (unrounded) from the three banked dumps on the t3 rows and compared with 3.960, 3.913, 4.112 within 1e-3; when `--boxsize-result` is given they must also equal `values.E` of that file within 1e-6. Only pass flags are stored, no error is printed or stored.
7. **Tele resolution.** torchvision's detector resizes its input to a shorter side of 800 px, so the tele window rendered at 1,024 px is read at about 50 pixels per degree. Sizes come from the window pixels and `win` (the file's own `win` key).
8. **Pod paths in `read_args.txt`** other than the ones copied from the ADD124 file (`det-dir`, `locator-dir`, `weights-record`, `wide-extr`, `tele-extr`, test outputs and their epochs) are guesses and are the lead's to set.
9. **The bootstrap seed** is 127 (the card number); the card names the estimator of ADD124 (which used 124) and no seed.

## 1. Geometry (`box_geom.py`)

The window is `win = (size_deg, px, az_left, el_up)`, square, cylindrical, exactly `enc_common.window_frame / rotation_az_el / window_grid`: `f = (px/2) / radians(size/2)`; the ray of pixel index (i, j) is `p = (sin u, v, cos u)`, `u = (i - (px-1)/2) / f`, `v = (j - (px-1)/2) / f`; the camera ray is `Q = R p`, `R = Ry(-az0) Rx(el0)`; `az_left = atan2(-Qx, Qz)`, `el_up = atan2(-Qy, hypot(Qx, Qz))`. Box coordinates are continuous window pixels (pixel i covers [i, i+1), centre of the window at (px/2, px/2)); the detector's x1 y1 x2 y2 are in this convention.

* `pix_to_dir(col, row, win)` and `dir_to_pix(az, el, win)` are inverses (`ahead` false behind the window axis).
* `box_angles(box, win)` -> (az_c, el_c, width, height) in degrees: the centre is the direction of the box-centre pixel; width and height are the range of azimuth and of elevation over the four corners and four edge midpoints, which is the label's own definition (range over the corners of the 3D box, `lead_box.py`).
* `contains(box, az, el, win)`; `edge_distance_deg` = the angle between the direction and the nearest point of the box (0 inside, inf behind the axis).
* `pick_one`: the box that contains the direction (smallest pixel area when several do; ties to the lower index), else the box with the smallest edge distance if it is at most 1.5 degrees, else none.
* `point_in_other_cam(R_src, R_dst, az, el)`: see open point 2; `rot_cam_to_vehicle(qx, qy, qz, qw)` is the stack's quaternion formula.
* `render_roundtrip`: a rectangle of angular size (w, h) at (az, el) is drawn into a synthetic f-theta frame (r = f theta, 1600 x 1600, f = 1200; an anti-aliased membership image), rendered through `enc_common.window_grid` and `window_sample`, the lit region's box is read and `box_angles` must return (az, el, w, h) within 0.1 degree. Cases: `RENDER_CASES` (two wide windows, one tele window).

## 2. Detector files and the pick

`<det-dir>/<wide|tele>/<clip>.npz` as in the lead's brief plus `frame_run`. A detection counts when: label in {3, 4, 6, 8}, score at least 0.5, frame in range with `frame_ok` and `frame_run` true, a box of positive size. Pick per frame as in section 1. Size of the pick = `box_angles` width and height.

## 3. Arms (card section 3)

* Map G: `read_boxsize.MapG` (25 nearest in ln height and ln width scaled by their mean and standard deviation, median of their gaps), fitted on the train clips on the detector boxes picked by the LABEL direction (`ctr_az`, `ctr_el` of the box files; for tele carried through the extrinsics) with `gap_re` as the gap, on frames with a lead, a finite gap, `frame_run` true and a pick. Separately for wide and tele.
* `BW`: pick by the locator's direction in the wide read; gap = G_wide(size); no pick: arm D's gap of that fit. `BT`: pick in the tele read where the frame has one (G_tele); else BW. `BL`: BW with the label direction. Each is formed against each of the three fits of arm D (`BW_0..2`, ...).
* Row set: the t3 rows of the ADD124 reader (a label lead and a finite gap in D_0 to D_2): 3,841 rows of 141 clips.

## 4. Measures and names (card section 4)

E(X_k) = mean absolute gap error on the rows. R(X) = mean_k E(X_k) / mean_k E(D_k). Interval: `read_boxsize.boot_ratio` on the per-clip sums (mean over the three fits), `RandomState(127).randint(0, 141, (B, 141))`, two-sided 95 percent percentile, fits held fixed.

* `BOXREAD-T0-WIDE-BETTER` (upper end below 1 and R(BW) at most 0.90), `-WIDE-SMALL-GAIN` (upper end below 1, R above 0.90), `-WIDE-WORSE` (lower end above 1), `-WIDE-NOT-SEPARATED`; the same four with `TELE` for R(BT).
* `BOXREAD-T0-TELE-ADDS` when the upper end of the interval of E(BT)/E(BW) is below 1, else `-TELE-DOES-NOT-ADD`.
* `BOXREAD-T0-FAR-LEADS-LESS-SHORT` when the paired difference (BW minus D, signed error, label gap 40 to 60 m, per-clip sums over the number of far rows) has its lower end above 0 and |signed BW| is at most half of |signed D| (means over the three fits); else `-FAR-LEADS-STAY-SHORT`.
* No interval for a ratio: the WIDE or TELE name is `-NOT-SEPARATED`.
* Beside (naming nothing): `BL` (R, interval, errors singly); share of rows with a pick for BW, BT, BL in all and by label gap bin (0-10, 10-20, 20-40, 40-60 m); the error of each arm on its own picked rows against arm D on the same rows; the error of the detector's sizes against the label sizes (mean absolute error of ln width and ln height in per cent, by bin) on the BL-, BW- and BT-picked rows; errors and signed errors by bin for D, BW, BT, BL.

## 5. Controls (card section 5)

Printed only as `ctl b<k> PASS` or `ctl b<k> FAIL <short reason>`, then `b5 share <x>` and `ZZ BOXREAD CONTROLS ALL-PASS|FAILED`. All values go to `BOXREAD_CONTROLS.json`; no error, ratio or share by bin of an evaluation frame is computed in controls mode (b5 computes the pick share on the t3 rows, which the card asks for).

* b0 inputs: weights record (sha256, 8-character name tag the hash begins with, `sha256_begins_with_name_tag` true; the weights file hashed too when `--weights-file` is given); both `BOXDET_RECORD.json` carry the same hash and so does every detector file that was read; every detector file's `win` is the card's (wide 32 deg, 768 px, centre 0 / -6; tele 16 deg, 1024 px, centre 0 / 0); every detector file has `frame_run`; the two lists equal the entries of the crop study's list record; the three errors of arm D (open point 6).
* b1 geometry: the render tool test (section 1) within 0.1 degree; the label direction inside the wide window on at least 0.90 of the train frames with a lead.
* b2 live detector: the pick by the label direction exists on at least 0.80 of the train frames with a lead and a label gap of at most 20 m; with the label azimuth mirrored (az -> -az) the share is at most half of that.
* b3 live map: five folds by clip; the wide map's gap error on held-out picked train frames at most 0.90 of the constant's; the map fitted on permuted gaps at least 0.95 of the constant's.
* b4 vision only: the locator file loads (a file with any label key, or without the locator's keys, is refused: `LabelFileInLocatorPlace`); the loader refuses a planted label-like file; the locator folder is not the box folder; `eval_pick_locator` (the only function that forms BW and BT) has no label argument; the file hashes to the `pred_sha256` of its `_meta.json` (card ADD107) and to `--locator-sha256` when given.
* b5 instrument: at least 0.50 of the t3 rows have a pick in the wide read (the share is printed).
* b6 tests: `--test-out det=...` and `read=...` end with a line `<n> passed` (no failed, no error) and are older than the epoch of `--test-before`.
* b7 one evaluation set: the t3 rows number `--expect-rows` (3,841), the join has `--expect-clips` (141) clips equal to the evaluation list, the label fields of D_1 and D_2 equal those of D_0, the locator file covers every evaluation clip, every evaluation clip has a wide detector file with at least as many frames as the join, and no t3 row has a wide frame with `frame_run` false.

The controls file also stores the sha256 of every input (`inputs`); names mode refuses when one differs.

## 6. Output files and exit codes

`BOXREAD_CONTROLS.json` (`controls` -> b0..b7 -> `pass`, `failures`, values; `all_pass`; `inputs`), `BOXREAD_RESULT.json` (`names`, `measures`, `beside`, `counts`, `input_hashes`, `sums`), `BOXREAD_VERDICT.md` (tier line; the four family lines of card section 8; names; per arm the three values and their mean, R with its interval; far leads; beside values; the limits of the card). An existing file is never overwritten: a second call writes `..._2`, `..._3`. Names mode reads only `BOXREAD_CONTROLS.json` (the first one) in the out dir. Exit codes: 0 all pass (controls) or names written, 2 a control failed, 3 refused. Names print `ZZ BOXREAD NAME <name>` per name (four lines: wide, tele, tele adds, far leads).

## 7. Commands

Dev box: `PYCAP_MB=3000 bash ops/pycap.sh -m pytest boxread/read/test_read_boxread.py -q -p no:cacheprovider --basetemp <tmp> > boxread/read/helper_pytest.out`. Pod: `bash $K/podread_boxread.sh controls`, then only if all pass `bash $K/podread_boxread.sh names`.
