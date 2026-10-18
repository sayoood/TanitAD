# P6-HN-S screen: Master Mind decisions at registration (2026-10-18)

Registration: `PREREG_P6HN_SCREEN.md`, sha256 `9eb7cb2fc5e0e27927fc69e83926d2bed7878d866be9bd6f16f24511e7625679`,
landed unchanged, before any screen data exists. The questions are `SCREEN_NOTES.md` Q-S1 to Q-S6.

| Q | owner | decision |
|---|---|---|
| Q-S1 | PI | **Asked, not presumed.** Until the PI rules, a SCREEN-GO puts the full P6-HN arm in front of the PI *with the screen's evidence*; it does not book the 6.8-7.65 A40-days. This matches the prereg's own reading. |
| Q-S2 | MM | **Adopted.** The full-arm draft takes the screen's masked two-call set scoring (fan call unchanged; a second call over `[fan ; hn]` keeps only the hard-negative rows), by `PREREG_P6HN.draft.amendmentA.md`. The draft stays a DRAFT. |
| Q-S3 | MM | **Dev box**, behind `devbox_gpu.lock`, in units of 30 min or less. Box 2 is not assigned: it is the PI's machine and the 2026-10-17 ruling gave it to v8F experiments. |
| Q-S4 | MM -> PI | **BLOCKER for stages S0-S3.** The plan needs ~105 GiB (69.6 GiB of it the bf16 context cache) and ~35 GiB staged off the pod. MEASURED 2026-10-18: C: ~34 GB free, D: ~4 GB free. Asked of the PI (free space, a drive, or a smaller cache). Implementation that needs no disk proceeds. |
| Q-S5 | MM | **Accepted.** u400's possible exposure to up to ~26 % of RL-val tokens (ESTIMATED) shifts both arms equally; the endpoint is B - A. RL-S1 accepted the same property. |
| Q-S6 | MM | **Deferred** to the conditional re-screen. If P6's data manifest does not exclude the 24 SFT held-out logs, that re-screen's amendment must say so before it runs. |

Implementation order: checklist items 1, 2, 5, 6, 7 (new files) start now; items 3, 4 and 8 (`scorer_finetune.py`,
`gate_checks.py`, `launch_gate.py`, the chain) start after B8's carve-feed lands, because B8 edits the same files.
