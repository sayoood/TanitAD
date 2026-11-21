# PREREG L2 Step B, AMENDMENT 1 (REPORTED-ONLY addition; Master Mind, 2026-10-19 ~20:20)

**Sealed while the Step B chain is RUNNING and BEFORE `ZZSTEPB_DONE`. No Step B output, label or verdict has been read.** It changes no gate, no bar, no arm and no verdict rule of `PREREG_L2_STEPB.md` (sha256 `c661588c…`). The running chain is untouched. Its analysis and verdict line stand exactly as sealed.

## Why (the consequence committed BEFORE READ 1 was read)
`MM_RULING_L2_FRESH.md`, sealed into `PREREG_L2_FRESH.md` (sha256 `5006ed84…`), says: *"if paired ΔA3's upper CI is below 0 on both seeds, the read records an EP-RELIAB-class risk. The Step B amendment must then report the aggregate concordance of the search outputs beside dPDMS."*

READ 1 (`FRESH_RESULT.md`, carve B, 2026-10-19 20:12) MEASURED paired ΔA3 (L2 − P2) of −0.048 [−0.067, −0.029] (s0) and −0.061 [−0.086, −0.036] (s1). Both upper ends are below 0, so the condition fires, and this amendment carries it out.

## What is added (REPORTED ONLY; never gated; it never changes Step B's verdict)
For each arm (L2-s0, L2-s1, P2HEAD) and each search seed, form the pair (anchor = P2's pick, candidate = the search output) per token. Compute, with `stepA.admission_terms` UNCHANGED (L1's code, sha-pinned):
- **A3**: the arm's own scorer aggregate (`refe/planner.py` aggregate, V1_W = (5, 5, 2)) against the TRUE PDMS. A pair counts only if the true PDMS differs by ≥ 0.05; a scorer tie counts 0.5.
- **A4**: among pairs where the arm's aggregate rises by ≥ 0.02, the share where the true PDMS falls by ≥ the Step A threshold (the exploit rate the search actually realised).
- **A5**: among those same pairs, the share where true NC or DAC goes from 1 to 0.

Pool over the 3 search seeds within an arm. Estimator: the paired log-cluster bootstrap (24 logs, 10,000 resamples, seed 20260927), plus the paired difference of each L2 arm against P2HEAD. Each per-seed value is reported beside the pooled one (the search-seed spread).

**Inputs.** Step B's banked outputs and labels. If the arm's scorer logits at the search output are not banked, they are re-scored on CPU with the SAME `score_fn` (head installed by `l2_scorer.install_ep_head`, sha-checked). The re-score must first reproduce the banked logits at P2's pick bit for bit (identity). If it does not, the addition is reported as NOT COMPUTABLE, with the reason.

**Where.** It is reported in a separate section of the Step B result, `AMENDMENT 1: aggregate concordance of the search outputs (reported only)`, next to dPDMS and the four families. It is computed by a separate script after `ZZSTEPB_DONE`.
