# Build note: the TanitAD-4B paper, version 4 (2026-11-13)

Source: raw/2026-11-10-paper-pdf-v3/tanitad_4b_paper.tex (sha256 1b783e91..., asserted by the patch). Edits: patch_tex_v4.py, six edits, each asserted to match exactly once. Argument `STAGE1`: `training` if the STOPPERC stage-1 chain had started at build time, else `queued` (one phrase in the paragraph, one roadmap cell). This build: STAGE1 = `queued` (built 2026-11-12 ~23:56; the SLOWWM chain was banking its last run and the waiter had not launched stage 1).
Engine: Tectonic 0.17.0, `tectonic --keep-logs --outdir build tanitad_4b_paper.tex`. Result: 9 pages (v3: 9), 0 overfull boxes, 0 undefined references. The underfull boxes in the log sit in lines v4 did not touch.
Pages 1, 6, 7, 8 and 9 were rendered to PNG with the Windows PDF API and read. A first build with four new headline rows pushed Table 1 past the float limit, so it set after the references; the build of record folds them into one row and Table 1 sets on page 7 beside the Results heading. The roadmap sets on page 9 beside the references, as in v3.

## What changed (all MEASURED; tier T1-kin first tick, zero training, oracle reads, unless marked)
* Abstract: one sentence. 87-89 % of the remaining tactical headroom is in the continuous constraints, chiefly target speed. The operative level executes a correct stop (1.5 m on stop-point windows). Loss re-weighting moves the stop decision along one ROC curve, so the next lever is stop perception.
* Table 1: one row (tactical headroom split; forced-stop gain; re-weighting moves recall only along one ROC curve).
* Results: new paragraph "Where the tactical headroom sits" (two paragraphs of text). It covers the selector share, the constraint/decision split, the speed concentration on stop-point windows, the decision miss and its AUROC, the forced stop (STOPORC), hardening (STOPHARD, STOPTAU), the two re-weighting doses (STOPBAL), the stop ceiling (STOPPERC S0-C), the frozen-feature probes (S0-P) and the closed band loss (OBD).
* Roadmap, Table 2:
  * world-model planner row: EGOWM NO-EFFECT (T0); a slow-feature basis is being read.
  * band retraining: four seeds, closed as a recipe lever.
  * tactical selection: oracle read, no selector built.
  * new row: stop perception.
* Outlook bullet: "motion and stop cues".
* No number of v3 changed.

## Sources of the new numbers (folder raw/ of the v8f-mini programme; every number counted in its file on tip 13e30630 before writing)
| claim | file | values (s0 / s1) | rounding |
|---|---|---|---|
| selector share | 2026-11-10-v8f-consorc/VERDICT_TOPK_PAIR_CONSORC.md | 0.239 / 0.228 | 24 / 23 % |
| GT code vs own | same | -1.625 / -1.773 m | 1.63 / 1.77 |
| constraint / decision share (leave-one-out) | same | 0.872 / 0.892; 0.111 / 0.094 | 87 / 89 %; 11 / 9 % |
| stop-point speed mass | 2026-11-10-v8f-speedsrc/VERDICT_SPEEDSRC.md | 0.200 / 0.197 on 0.050 of windows | 20 % on 5 % |
| goal-head hit, 117 windows | 2026-11-10-v8f-stopdec/VERDICT_STOPDEC.md | 0.103 / 0.128 | 10 / 13 % |
| stop AUROC, 131 / 2,365 | 2026-11-10-v8f-stopcue/VERDICT_STOPCUE.md | 0.836 / 0.857 | 0.84 / 0.86 |
| forced stop | 2026-11-10-v8f-stopbal/oracle/VERDICT_STOPORC.md | -1.539 (5.02 -> 3.48) / -1.477 (4.94 -> 3.47) m | 1.54 / 1.48; 30-31 % |
| hardening | .../oracle/VERDICT_STOPHARD.md, VERDICT_STOPTAU.md | 3-6 % of the ceiling; TAU-TRADES / TAU-NULL | as written |
| re-weighting | .../read/VERDICT_STOPBAL.md, .../dose2/read/VERDICT_DOSE2.md | recall 0.060-0.145 -> 0.453-0.615; false stops 0.005-0.013 -> 0.120-0.145; P floor 0.703 | 0.06-0.15 -> 0.45-0.62; 0.5-1.3 % -> 12-14.5 %; 0.70 |
| stop ceiling, c 0.7 (82 of 117) | 2026-11-11-v8f-stopperc/VERDICT_S0.md | -0.957 / -0.855 m; yaw rate +2.166 / +2.178 deg/s | 0.96 / 0.86; +2.2 |
| frozen-feature probes | 2026-11-11-v8f-stopperc/VERDICT_S0P.md | 0.8387-0.8653 over all input sets, pool8x20 included | 0.84-0.87 |
| detach / band loss | 2026-11-10-v8f-obd/VERDICT_OBD.md, read_seed3/VERDICT_OBD_SEED3.md | DETACH-KEEPS; -0.0019 / -0.0014 / -0.0020 m; BAND-UNSETTLED | 1-2 mm |
| measured-speed predictor (T0) | 2026-11-09-v8f-plantest3/decinfo/egowm/VERDICT_EGOWM.md | NO-EFFECT; floor 0.0124 | as written |

## Not re-verified
* As v3: the count "2^7 corners" and the PUBLISHED statements of the cited papers.
* The stage-1 status phrase reflects the chain state at build time only.

| file | sha256 |
|---|---|
| tanitad_4b_paper.tex | e51ef9c21ca13c1143b931f2533676d125e69aeddc90840c92807a856faf53e6 |
| tanitad_4b_paper.pdf | 8545f51635dd4d79baf1aea5ade5e66183621ce8674246d7ed8597df18253a68 |
| patch_tex_v4.py | 8cbd467746756c8806e485e298b0d2e0076491e2fa86c1b414945e1939d79c98 |
