# PREREG_V8F_N1DH, Amendment 4 (FROZEN 2026-11-21 09:53Z by the MM, after N1DH_s0 trained and before any N1DH dump, read or number)

The card `PREREG_V8F_N1DH.md` (sha256 f1df05bc) and Amendments 1 to 3 (25d6b42a, 0f78a055, cd6a326c) stand. This amendment repairs one character in the arm script's zero-grad gate line. The gate's form, its controls and every bar are unchanged.

## What happened (MEASURED, pod)
* The chain relaunched at 09:28:38Z. N1DH_s0 trained all 3,000 steps (`summary.json`: `done` true, `steps` 3000), and its post check read **PASS**.
* The next line, the zero-grad gate, exited with an argparse error: `unrecognized arguments: /root/strat1/n1dh/runs/N1DH_s0/zerograd_n1dh.json`. The chain logged `ZZ N1DH FAIL ZZ arm N1DH_s0`.
* The cause: `n1dh_arm.sh` line 27 read `--control 'route_head\.weight'--json-out <path>`, with no space. The shell joined the quoted regex and `--json-out` into one argument, so the path was left over. The gate never ran; it did not read the checkpoint.
  * The N1DC arm script (`/root/strat1/n1dc/n1dc_arm.sh`) does not carry this line, so the typo came in when the N1DH controls were added.
  * The chain's syntax check (`bash -n`) cannot see it: the line is valid shell.

## The repair (`F:/TanitAD-scratch/n1dc/build/n1dh_b6zg/fix_zg_n1dh.sh`, `ZZ ZG FIX OK ZZ`)
* One space is inserted before `--json-out`. Exactly one line changes; the copy is kept as `n1dh_arm.sh.bzg`.
* N1DH_s0's three remaining arm lines then ran by hand, as the fixed script writes them, under the GPU queue lock:
  * zero-grad gate: **VERDICT PASS**, both controls (`route_emb.weight`, `route_head.weight`) trained, untrained numel 0;
  * K-CLIP: `kclip_N1DH_s0.json` written (60 logged steps; share clipped 1.0 at `--clip-grad 1.0`; grad-norm median 14.69, max 305.18; 0 non-finite rows). The card reads K-CLIP as in N1DC (line 99); this amendment sets no new bar on it;
  * `ARM_DONE` written.
* The failed chain log becomes `n1dh_chain_1f217bf0_zgtypo.log`, and the failed gate log becomes `runs/zerograd_N1DH_s0_typo.log`. Nothing is deleted.
* The chain relaunched at 09:51:58Z. It skips N1DH_s0 (ARM_DONE plus checkpoint) and runs N1DH_s1 with the fixed line, then the D1-HO dumps.
* Training, argv, checkpoint, dumps, reads and bars are unchanged. This is an instrument repair made before any value is read, so the gate's first real reading is the one above.

## Lesson
A shell line can be valid and still pass the wrong arguments. For a new gate line in an arm script, run that exact line once on a real checkpoint (any finished arm) before the launch. `bash -n` and a dry mint cannot reach a line that runs only after training.
