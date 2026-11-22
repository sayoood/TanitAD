# PRESWRAP definitions (study "preswrap")

Generator `make_wrapper_p.py` writes `v8f_t1_eval_reader_rule_rzgfhqetjvp.py` and `WRAPPER_P_GEN_RECORD.json` from `../visrule/v8f_t1_eval_reader_rule_rzgfhqetjv.py`
(LF text, sha256 prefix 28bc31f21ecb4ccb; any other text is refused; an existing output or record is refused). Check: `check_wrapper_p.py`. Tests: `test_preswrap.py`.

## Substitutions (each old text matches exactly once in the source)
- P1 header docstring: names the variant, the two flags and the logging additions.
- P2 flags: `--vis-pres-file` (default empty) and `--vis-pres-mode` (choices head, or; default empty) after `--vis-closing`.
- P3 refusals right after the argument parse (before the pinned tool is loaded): either flag outside form RR; the file without the mode; the mode without the file.
- P4 `ReaderPlanner.__call__` binds `STATE["vis_states"]` when the gap file OR the presence file is on (the tick look needs clip index and frame).
- P5 `head_values_reader` (tick look): `p_rd = p` (the reader's own), then `p = pres_tick(p)` before `feed(...)`, so presence, `--vis-gap-file`, `--vis-closing` and the returned p (read by the mix planner, hence the tick log's p_src) all see p2.
- P6 `head_values_reader` log columns: column `p` keeps the reader's own p; with the file a column `p_used` (= p2) is appended.
- P7 later look (`_reader_at` path in the plan function): `p_rd = p`, `p = pres_look(st, q, p)` before `feed`, `vis_look` and `hp = p >= HEAD_P_MIN`; `p_src` of the mid log = p2.
- P8 later-look side file (`M["rd"]`): `p` = reader's own, `p_used` = p2 only with the file.
- P9 `vis_look` closing run: the four earlier frames read again by `_reader_at` get p2 too (`pres_look(..., False)`: not counted again).
- P10 load: `STATE["pres"] = load_pres(file, mode, T.D.load_join(a.eval_join))` (None without the file); with the file prints `pres run-line fields: vis-pres-file ... vis pres sha256 ... vis-pres-mode ...`.
- P11 run record (rule log meta): `**pres_meta(...)` adds the four keys below only with the file.
- P12 helper block (`PRES_MODES`, `STATE["pres"]`, `STATE["pres_counts"]`, `load_pres`, `pres_values`, `pres_replace`, `pres_look`, `pres_tick`, `pres_fields`, `pres_meta`) before `build_rule_label`.

## Exact names added
- Flags: `--vis-pres-file`, `--vis-pres-mode` (head|or).
- Run record / rule log meta keys (only with the file): `vis_pres_file`, `vis_pres_file_sha256`, `vis_pres_mode`, `vis_pres_counts` (JSON text, sorted keys: `rows`, `with_file`, `no_file`, `raised`, `lowered`).
- Side-file column (only with the file): `p_used` in `<rule log>_rd.npz` and in `ticks/rule_log_mid_<run>_rd.npz`; the column `p` there is the reader's own, unchanged. The `p_src` column of the rule log and the mid log carries p2.
- First line fields (only with the file; text returned by `pres_fields`): ` vis-pres-file <path> vis pres sha256 <hash> vis-pres-mode <mode>`.

## Semantics
mode head: p2 = file value. mode or: p2 = max(p, file value) with `np.fmax` (a reader p that is not a number takes the file value). A frame with no usable file value (the file's p not finite, `has_out` false, a frame outside 0..n-1 of its clip, a clip index outside the join) has p2 = p and is counted as `no_file`.
`ok`, the gap, the closing input and the rule are as in the source. Counts: `rows` = looks seen, `raised` = p < 0.5 <= p2, `lowered` = p >= 0.5 > p2 (a reader p that is not a number is neither).

## Decisions this brief left open
1. The first line of the run log is written by the queue script (`echo ... > run_<arm>.txt`), not by the wrapper. The wrapper cannot own that line: it prints the fields at load (`pres run-line fields: ...`) and `pres_fields` gives the exact text; the queue has to append it (`${PRES:+...}`) as it does for the gap file.
2. Meta keys appear only with the file (the v variant's keys are always present; these are not, as the brief asks that the additions appear only with the flag).
3. Required arrays: p, has_out, clips, off, n (seed is not read). The file must have exactly `max(off+n)` rows, and `has_out` may not mark an output where p is not finite. A usable value is `has_out` and finite p. Finite p outside [0, 1] refuses (0 and 1 are allowed).
4. The join is reachable at load (`T.D.load_join(a.eval_join)`), so the join check is at load, not at first use. Compared: `off`, `n`, clip names (as text).
5. The frame looked up is `s.j + e_off`, the same frame as `--vis-gap-file` uses; the clip index is `s.ci`; row = `off[ci] + frame`. Pre-roll looks go through the same functions, so they are counted in `vis_pres_counts` too, with the frame index the window carries there.
6. `vis_pres_counts` counts the tick look and the later looks, not the earlier frames re-read for the closing slope.
7. With the file the returned p is float64 (p2); without it the source's float32.
8. `--vis-pres-mode` has argparse choices, so a bad value exits with the argparse error (code 2) rather than a REFUSED line; `load_pres` also refuses it.
9. The check/tests create small temp files under `F:/TanitAD-scratch/v8proof/preswrap_tmp/` and leave them (no deletes).
