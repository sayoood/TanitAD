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
