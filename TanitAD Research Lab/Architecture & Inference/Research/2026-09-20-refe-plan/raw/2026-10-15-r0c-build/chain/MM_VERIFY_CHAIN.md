# Master Mind re-verification of the R0c chain wiring (2026-10-15)

**Stack.** `PATCH_1..7` (the build, `../`) and `PATCH_8..12` (this folder) were applied with **`git -c core.autocrlf=false apply`** to a fresh
`git archive` of the tip `7f82f3e` (refe/ is unchanged since `066ca5a`), and `test_rl_r0c.py` was added.
- Every patch's `--check` passed.
- **0 CRLF files** in `refe/*.py`, `refe/*.sh` and `eval/*.md`. The agent's warning holds: plain `git apply` on this box writes CRLF, which breaks bash on the pod.
- `bash -n refe/rls1_pod.sh` is OK.

| suite | result | agent's claim |
|---|---|---|
| `test_rl_r0c.py` (torch 2.11 / 2.7.1) | **26 / 26** | 26 |
| `test_rls1_chain.py` | **512** | 512 (tip: 392) |
| `test_rls1_queue.py` | **134** | 134 (tip: 101) |
| `test_rl_s1b.py` | **18** | 18 |
| `test_rl_epo.py` | **42** | 42 |
| `test_launch_gate.py --static` | **659** | 659 |
| `test_aux_s1_wiring.py` | **20** | 20 |
| `test_s1aux_chain.py` | **385** | 385 (unchanged) |
| `test_p2rep_chain.py` | **353** | 353 (unchanged) |
| `test_p6_chain.py` | ZZP6CHAIN_PASS **341** | 341 (unchanged) |

**An environment trap, not a patch defect.** With `TMP` on E: (exFAT), three suites failed, each with an exFAT signature:
- `test_launch_gate --static`: "a dirty file shipped under a commit" (its temp git repo on exFAT);
- `test_p2rep_chain`: "the helper checks left the shared tree changed" (exFAT records no file modes);
- `test_rls1_queue`: `WinError 145` on `rmtree`.

With `TMP` on C: all three pass. Only `test_p6_chain` needs `TMP` on E:, because its 2 GB write does not fit on C:.
⇒ **Run the git- and mode-sensitive suites with `TMP` on an NTFS volume.**

**Master Mind decision (the agent's open question).** R0c is read against **its own arm A**, as `PREREG_R0C_DRAFT.md` §2 says. C0's banked dumps are the
**second** control. If they are missing, the chain logs `ZZRLS1_R0C_NO_SECOND_CONTROL` and continues; it does NOT refuse.
- The governing read never depends on C0.
- A refusal there would idle the A40 for a secondary read: the 2026-10-14 stall class.

**Status:**
- **Patches NOT applied to `refe/`:** running R0c is a PI decision (about 3.5 A40-h), and `PREREG_R0C_DRAFT.md` must be finalised before launch.
- **When applied:** regenerate `SHIP_MANIFEST.json` at that commit. At the same re-ship, fix `eval/rl_val_eval.load_scores` at its root (one read per npz column), because it ships with RL-S1 and runs at `rls1_pod.sh:559`.
- **UNVERIFIED:** any pod run; a real `launch_gate.py run` (mint) for `refe_rls1c`.

## Rebase of PATCH_10 onto `5a81e8d` (2026-10-15, after the `rl_val_eval.load_scores` root fix)

**Why.** `5a81e8d` changed `refe/test_rls1_queue.py` in two places, so the original `PATCH_10` no longer applied (`refe/test_rls1_queue.py:682`):
- its pin of the old loader defect became "both loaders stay light";
- the mutation "the original loader is back" was replaced by "the per-row decode is back in the light loader".

**How it was rebuilt.** By replay, not by hand:
1. The original `PATCH_10` was applied to the `94248ec` file it was built on.
2. The two `5a81e8d` edit scripts were re-run on the result.
3. The diff was taken against `5a81e8d`'s file. It is byte-identical to applying the rebased patch on `5a81e8d`.

**Files.**
- `PATCH_10_test_rls1_queue.diff` is now the rebased patch.
- The original is kept as `PATCH_10_test_rls1_queue.on_94248ec.diff`.

**Re-verified** on a fresh `git archive` of `5a81e8d`, with PATCH_1..12 applied (`git -c core.autocrlf=false apply`, the rebased PATCH_10, 0 CR bytes):

| suite | result |
|---|---|
| `test_rls1_queue` | **134** |
| `test_rls1_chain` | **512** |
| `test_rl_r0c` | **26** |
| `test_rl_val_eval` | **32** |
