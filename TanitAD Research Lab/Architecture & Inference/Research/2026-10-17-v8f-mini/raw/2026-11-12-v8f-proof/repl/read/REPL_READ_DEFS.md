# REPL read definitions (card PREREG_V8F_PROOF_ADD121, kit `repl/read`)

`make_repl_reader.py` writes `read_repl10k.py` and `podread_repl10k.sh` from `gapmargin/read/read_gapmargin10k.py` (LF text, sha256 prefix 0aab17c67eb60756) and
`gapmargin/read/podread_gapmargin10k.sh` (b44d8181d3894ae9); any other text is refused, an existing output is refused (`--print reader|driver` writes to stdout). Tests: `test_repl_read.py`. Pull: `pull_repl.sh`
(pod root `/workspace/v8proof/repl`, local folder `repl/pod_add121`, transfer files `repl/tmp_pull_add121.txt|.tar`).
Kept from the source: the engine calls, the two-level bootstrap (2,000 draws, `--n-boot`), the gap-file record control and the wrapper control (parts of e1), the identity control (e3), the tests control (e4), the engine
control (e5) and the loader of rule logs. The generated reader refuses `--selftest`. Controls print as `e0` to `e7` (= r0 to r7 of the card); the names stage refuses unless all eight are present and pass.

## Seeds and estimates
Model seeds 2 and 3 are new (the reader seed of model seed m is m - 2); 0 and 1 are banked. Every arm of every setting (CONV, K0, K30; RLc, RR, GEc, A90) is read at the four seeds. For each setting the engine's grid
read runs three times with the engine's module global `SEEDS` set for the call and restored after it (`SeedsCtx`): `new` = (2, 3), `first` = (0, 1), `four` = (0, 1, 2, 3). Pairs A90-VS-GEc, GEc-VS-RLc, A90-VS-RLc,
GEc-VS-RR, measures U, C, F, estimator as the source (paired two-level bootstrap over seeds and episodes). Levels: per model seed for all four seeds (taken from the `four` read), pooled per pair of seeds (`new`, `first`) and
over four seeds. Families and the strategic read: on the new pair only. Printed with no name, per arm and setting: pooled level of the new pair minus that of the first pair.

## Controls (exact definitions)
- **e0 models (r0).** `--models-record` (pod `$GE/models/REPL_MODELS_RECORD.json`), keys `{"models": {"UE_s2": {...}, "UE_s3": {...}}, "note_sha256": <64 hex>}`, each model with `ckpt_sha256` (64 hex), `gate_pass`
  (the JSON value true), `recipe_same_but_seed` (true), `record` (non-empty text). Passes when for m in 2, 3: `<w>/ckpt/UE_s<m>/A/ckpt_last.pt` (`--w`, pod `$W` = integ10k) has that sha256; both flags are true; and the first line
  of EVERY new run of model seed m (the 24 arm runs and the two spread runs R0Hb20, 26 in all) names `ckpt sha256 <hex>` whose value and the record's hash agree by prefix of at least 16 characters; `note_sha256` is 64 hex.
- **e1 wrapper and gap files (r1).** The source's gap-file record control for the scaled file A90 only (A80 is not used), together with the source's wrapper control (file hash 28bc31f21ecb4ccb, banked check files, pod check
  last line, hashes, and the wrapper hash in the run text of the 24 arm runs). Both parts have to pass.
- **e2 arguments (r2).** First line of each of the 26 new runs against that of the banked run of the same name at model seed m - 2 (`d2_repl_failures`): every field both carry is equal but model seed and ckpt; the new line carries
  `model seed` (= m) and `reader seed` (= m - 2); a banked line that carries a model seed or a reader seed names m - 2; a compared field (form, replan, every, readout, kappa, preroll, exec, nav, grid, reader seed) that the banked
  line carries and the new line lacks is a failure; where the banked line carries no preroll, grid or nav the new line reads the default of the setting (preroll 0 or 30, `eval_grid_windows.json` or `grid_p30.json`, nav deadreckon).
- **e3 identity (r3).** Table `EXPECTED` of the card (first pair pooled levels C, U, F for GEc, A90, RLc, RR at CONV, K0, K30, tolerance 1e-4), computed from the `first` read, same code path as the new cells.
- **e4 tests, e5 engine.** As the source. Lead test outputs `$GE/lead_tests/lead_pytest_queue.out` and `lead_pytest_read.out`; times as the source. The engine control requires the grid controls of all three reads of a setting and
  the stages A and B of the strategic grid.
- **e6 counts (r6).** Every one of the 26 runs has `T1_DONE` and 141 `ep*.npz` dump files (`dump_I10<run>_s<m>_m10_causal_heads`); the done marker's last line reads `ZZ REPL10K DONE <time> all steps done`; per setting the number of
  windows with a lead of each new model seed equals 642 (CONV), 642 (K0), 325 (K30).
- **e7 rule logs (r7).** For each of the 12 new runs of GEc and A90 (3 settings, 2 seeds): the model reader's `w2_one` against the run's own gap file (GEc: `--gap-file`, the base VGAPLE; A90: `--margin-dir`/VGAPLE_A90.npz), the
  rule log meta naming that file's hash and the closing switch 1; reference rows from the RR run of the same setting and the SAME model seed.

## Names (card section 4)
Seven claims, each with the side separated on the first pair: M-K0 (A90-VS-GEc U, K0, below 0), M-K30 (same, K30, below), P-K0 (A90-VS-GEc F, K0, above 0), D-CONV (GEc-VS-RLc U, CONV, above), D-K0 (same, K0, above), C-K0 and C-CONV
(GEc-VS-RR C, K0 and CONV, below). From the interval on the new pair: `REPL10K-<X>-REPLICATED` (on the claimed side: upper end below 0 for a side below, lower end above 0 for a side above), `REPL10K-<X>-OPPOSITE`
(wholly on the other side), `REPL10K-<X>-NOT-SEPARATED-ON-THE-NEW-PAIR` (the interval includes 0, an end at 0 included). C-K0 and C-CONV carry the value `not a claim` whenever the recomputed interval on the first pair is not on the
claimed side. `REPL10K-MARGIN-REPLICATED` / `-AT-K0-ONLY` / `-AT-K30-ONLY` / `-NOT-REPLICATED` from M-K0 and M-K30 replicated. Two routines (`names_of` from the interval ends, `derive_all` from signs) must agree.
The bound `U(A90) <= U(RLc) + 0.0045` at K0 on the new pair (pooled levels of the new pair) is printed with no name; the four-seed estimates carry no name.

## Open decisions taken (the brief left them open)
1. **Required fields.** The source required `every`, model seed and reader seed in both lines. Banked lines of older queue forms may lack them (CONV runs have no rule every). Here only the new line must carry model seed and
   reader seed; for every other compared field the banked line decides (carried there means required in the new line).
2. **Defaults of a banked line without preroll, grid or nav** are those of the setting, not those of the source's K0 world: at K30 the default is `preroll 30` and `grid_p30.json`.
3. **Share of rule steps with a lead.** The source's e2 also required a share above 0; it is not part of r2 of this card, so it is dropped from e2 and still printed per rule run (`rollout <key>: {...}` lines).
4. **Strategic grid at the new seeds needs R0.** The engine's strategic grid compares every run to the run `R0` at the same model seed and reads `R0` at seed 0 for its control s3. The queue makes `R0Hb20` at seeds 2 and 3 and no run
   called `R0`. The strategic grid is therefore run once over all four seeds (so that `R0_s0` is present) with the tick file of `R0` at a model seed of 2 or more taken from `R0Hb20` (`R0Alias`); the pair reads then use
   the new pair only. `R0` enters only the controls s3 and s5, never a pair.
5. **Windows with a lead per new model seed (r6)** is the engine's count `n_lead_windows` of the new-pair read; the engine's own control that every dump has the same windows makes it equal for both seeds.
6. **Spread runs** `R0Hb20` at seeds 2 and 3 count among the "new runs" in e0, e2 and e6 (26 rollouts); they are not in e1's wrapper part or in e7.
7. **Rule log arms.** e7 covers GEc and A90 only (the arms with a gap file); RLc and RR have none.
8. **The read of the new pair alone (lead, after the rehearsal on banked runs).** The engine's grid reader takes the spread run at its seed label 0 as the reference of the windows, so it cannot run with the seed set (2, 3).
   The `new` read therefore runs with the engine's labels (0, 1) and with the engine's run name (`key_of`) of the card's own runs shifted by 2 (`KeyShift`): label 0 is model seed 2, label 1 is model seed 3; the report
   keys of the engine name the model seed. The `first` and `four` reads are unshifted. The bootstrap draws of the `new` and the `first` read are then the same streams.
9. **r2, the checkpoint hash (lead, note N1 of the card).** A new line that names the checkpoint hash of the banked run of model seed m - 2 is a failure.

## Clauses that could fail by construction on real data
- e2: a banked first line without a field (preroll, grid, nav, reader seed, kappa) is handled by 1 and 2; a new line that lacks a field the banked line carries is a failure by design. The spread run R0 has no rule and its first
  line may carry no `rule every` or `kappa`; then nothing is compared for those fields.
- e0: the spread runs' first line must carry `ckpt sha256` (the queue writes it for every run of its wrapper; if the R0 line does not, e0 fails).
- e7: rows with a lead in a GEc or A90 log where the gap file has no output and the RR run logged no lead row at that step cannot be compared (the source's w2_one reads them from RR); they would fail.
- Strategic stage A/B (e5) at the new seeds depends on decision 4: the ticks of `R0Hb20` must exist at seeds 2 and 3 and their first ticks must equal those of the other runs (s5).
- e6 counts: windows with a lead do not depend on the model by construction (label windows), so the number reads the same at every seed; dump file counts depend on the run finishing all 141 clips.
