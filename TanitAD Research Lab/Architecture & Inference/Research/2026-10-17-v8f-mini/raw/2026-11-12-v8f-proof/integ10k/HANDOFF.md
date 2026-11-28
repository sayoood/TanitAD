# HANDOFF integ10k (card ADD46 INTEG10K-PRE + note N1). Build agent, 2026-11-19. Nothing real was launched.

## State
All 7 deliverables exist and are tested. Dev box (numpy, pytest) 20/20 green; pod CPU (minipytest, torch 2.8) 20/20 green (incl. DRY run on pod bash).
Mutations: MUTATIONS_INTEG10K.json, 16 of 16 red (the 4 of the card: min_dropped, reader_fed_future_frame, reader_of_the_other_seed, heldout_split_ignored), unmutated green before and after.
GPU smoke (UE_s0, planted constant reader p 0.88 / gap 25 m, 20 windows, forms R0 and RRid, pod folder build/smoke3): rc 0 both; 20 dump files each, 0 differing (RRid = R0 bit for bit);
rule log 60 ticks, 59 present, 0 active (g_src 1e9); the reader saw 40-token batches. It found two bugs (see below), both fixed and pinned by tests.

## Launch (lead)
`bash /f/TanitAD-scratch/v8proof/integ10k/chain_i10k.sh > /f/TanitAD-scratch/v8proof/integ10k/chain_i10k.out 2>&1 &`   (pushes with sha256 compare, builds the overlay tree, starts the pod queue, polls, pulls; `--no-start` = push and pull only)
Then `bash integ10k/read_i10k.sh` (headway + family readers, then read_i10k.py; controls first). DRY: `DRY=1 bash integ10k/pod_integ10k_queue.sh` = 20 commands (train, check-log, 2 rows, 2 maps, 2 reader-control runs, 12 rollouts), 13 slot waits.

## Files (dev box integ10k/ ; pod /workspace/v8proof/integ10k/build/ for the code files)
v8f_t1_eval_reader_rule.py (forms R0 RLb RR RRid RR0; imports the distrule wrapper unedited; copies of dist_rule.py, v8f_t1_eval_rule.py, v8f_t1_eval_rule_mix.py are pushed next to it)
tok_reader.py (reader, held-out rule, extract) | make_tok_probe_lr.py -> tok_probe_lr.py (4 asserted edits of tok_probe_gr.py) | qmap_fit.py + qcal_map.py (copy, same sha) | reader_controls.py (k1 k6 k7, pod CPU)
pod_integ10k_queue.sh | chain_i10k.sh | read_i10k.py + read_i10k.sh | test_integ10k.py | run_mutations_i10k.py | run_pod_tests.py | smoke_i10k.sh, smoke_check.py, make_planted_reader.py | taniteval_cdbff971.tar (5.9 MB, git archive of taniteval/taniteval + taniteval/tools at cdbff971)
Pod: build/tree_ov (stack = link to ../tree/stack, taniteval from the tar), build/smoke, smoke2 (failed run), smoke3 (good), build/tmp, tmp_final, tmp_t1.

## REVIEW items
1. BLOCKER FOUND AND WORKED AROUND: the staged tree has no `taniteval` package; the pinned tool imports `<tree>/taniteval` (v8f_t1_eval.py:43-49). The queue uses the overlay tree build/tree_ov (taniteval of cdbff971; t1_eval.py sha equals this fork's tree_leadsup copy 1f558f3c). Please confirm the overlay is acceptable.
2. The cdbff971 tree has no LeadSupHead and no LEAD_SUP constants: the reader is therefore a numpy re-implementation (tok_reader.py), not the probe class; torch GELU vs numpy fallback differ 1e-6. Reader training and extract use the fork's tree_leadsup (tok_probe_lr.py needs it).
3. Reader arm names: LR16_s0/s1 are trained as arms TK16_s0/s1 (original names kept so the format, extract and p5 check stay ADD43's) into $W/lr16; steps forced to 19,164 with --steps/--expect-steps (the held-out split changes the window count, so the trainer's own step formula is not used).
4. Rule form: floor `cap-only`, closing speed 0, as the ADD33/ADD42 arms (card says "banked form without the closing speed"; DISTRULE_DEFS E3b says cap-only needs an amendment: ADD33 used it). RLb = label presence and gap, closing 0.
5. Map coverage band (k9) is checked on raw Q(g_reader) (label below the mapped value); the share against g_used is stored beside. Fit/check rows: label present, finite, reader history complete.
6. k1 AUROC is over ALL labelled evaluation rows with a complete history (positive: lead gap <= 40 m, negative: the rest); the card does not say which rows. MAE is clipped at 40 m at the read, scored at cache step g0 (note of the coordinator).
7. k5: tolerance and values from K5_BANKED.json; UE_s1 null -> k5 NOT EVALUABLE -> no name and no outcome difference printed. The ckpt sha is compared to the card's 16-hex prefixes. R0 read from `intervals.metrics` (absent in a --max-windows run).
8. k10 / seed spread: the headway and family readers are called with `--spread-arm I10R0 --spread-seeds 0,1` (they were built for 6-seed spread arms); the crossed arm is I10RRx (model s0 + reader s1, model s1 + reader s0), pairs RRx:R0 per model seed.
9. The three reader controls (k1, k6, k7) run on the pod CPU (reader_controls.py) before the rollouts; k7 checks 40 reference windows.

## Not verified
Reader training on real data, the extract on the 4,572-clip cache (host memory, time), the map fit on real rows, reader_controls.py on real eval clips (only unit-tested), the RR/RR0/RLb forms on the real model, chain_i10k.sh end to end (bash -n only; the pull of tars through pcat is the pattern of the banked chains), headway/family reader pair names with the I10 arms, read_i10k.py on real pulled data. Pod RR with the real reader passes batch["feats"] (40 tokens) as the smoke showed, but only a constant reader was run.
Held-out filter: the trainer rebuild `D.train_windows(store, J, clips, ...)` was not run (needs the pod trainer); only the text edit is tested.

## Bugs found by the smoke (fixed)
tree has no taniteval (item 1); ReaderPlanner recursed because main() re-points W.RulePlanner (fixed with a kept original class, test added).

## Strays / deletions
Local temp: integ10k/work/mut and work/bt cleared after each run (sizes in work_clear.log). integ10k/work/ itself is empty. Pod: build/smoke, smoke2, smoke3, tmp, tmp_final, tmp_t1 are small scratch under build (not deleted; the pod build folder is the only place I wrote). No git, no dagland, nothing under E:/Projects touched.
