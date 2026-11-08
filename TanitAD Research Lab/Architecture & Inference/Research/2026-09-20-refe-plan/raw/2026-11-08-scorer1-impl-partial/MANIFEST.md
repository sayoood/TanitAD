# MANIFEST SCORER1-IMPL (everything under E:/Projects/TanitAD-artifacts/scorer1_impl_20261108/; nothing staged in a repo; the Master Mind lands)
blobs.tsv = repo path, base blob sha at the pinned tip b626d941..., file, sha256[:16]. scorer1_impl.patch = unified diff of the 4 modified files. Changed files in files/.
| repo file (refe/ dir) | change |
|---|---|
| scorer_finetune.py | PairLoss, chord_speed, pair_refusals; flags --pair-w/--pair-dv/--pair-heads/--pair-calibrate; pair_term in train_losses (arm B only); `pair` data-event block; preflight gate G16; --pair-calibrate mode. The diff removes 2 lines only. |
| gate_checks.py | DVB_KINDS_SCORER pair_*, scorer_dvb_rows `pair` rows, SCORER_GATES G16, scorer_required_gates, judge_g16 |
| launch_gate.py | consume_scorer_preflight re-judges G16 |
| test_launch_gate.py | G16 exempt like G14/G15 for r0; unknown-gate fixture G16 -> G17 |
| test_pair_loss.py (NEW) | 7 CPU test groups, ZZTEST_PASS; OFF path bit-identical vs base (loss + AdamW step) |
Evidence: test_pair_loss ZZTEST_PASS; NOBALANCE mutation unbalances F/S on 3/3 heads (T2); w=0 control gradient exactly 0.0 (T3).
NOT run: GPU preflight, G16 end to end, mutation arms through the gate, the full launch_gate suite.
Deletion: work tree F:/TanitAD_scorer1_impl_20261108 (re-creatable from the tip, ~25 MB) NOT deleted: the safety check blocked the rm; the user should delete it.
