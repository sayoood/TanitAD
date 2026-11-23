# HANDOFF_MM_KI (MM, 2026-11-23 ~13:46Z): K-INPUT trained; the dev-box read runs

## State
* **PI go:** "Yes do it" (2026-11-23, ~13:05Z), for KI_s0 / KI_s1 at about 0.55 pod-h, return line 0.9.
* **Chain:** LoRA pod, `rh_ki_chain.sh`, lock 13:09:14Z to 13:40:21Z, about 0.52 pod-h. Last line `ZZ ROLLHIST-KI DONE ZZ`.
* **Arm records:** in `readki/pod_ki_run/`, 29 files, every one equal to the pod's sha256 (`SHA256_POD.txt`, `SHA256_LOCAL.txt`).

| check | KI_s0 | KI_s1 |
|---|---|---|
| argv sha256 | baf33625 | 2ad1ccae |
| gate record | MINTED, verify OK | MINTED, verify OK |
| `levers.dagger.dagger_base` | logged | logged |
| declared_vs_built | PASS | PASS |
| dagger_probe | pass, `rows_changed.base` 0, 46 selected | pass, `rows_changed.base` 0, 51 selected |
| lever acts (`dagger_rows_sum`) | 191,664 | 191,687 |
| zerograd | PASS, 252/252 | PASS, 252/252 |
| post | 252/252 moved | 252/252 moved |
| checkpoint | step 3000, sha256 d9e93f63 | step 3000, sha256 c12c989f |

* **Read:** `rh_read_ki_retry.cmd`, launched via WMI at 13:45Z.
  * Driver: `F:/TanitAD-scratch/rollhist/read/out/rh_read_ki_retry.log`. Chain log: `out/rh_read_ki_chain.log`. Outputs: `out/read_ki/`.
  * Dev-box commit headroom was 2.1 GB at launch (other sessions' python jobs). The retry wrapper waits through up to 8 headroom stops, 10 min apart.
* **Waiter:** `readki/mm_wait_read_ki.sh` → `mm_wait_read_ki.out`.
  * Exit 0: DONE.
  * Exit 1: STOP.
  * Exit 4: the driver is gone without a final line.
  * Exit 3: 8 h cap.

## Next
1. **On DONE:** read `out/read_ki/RESULT_ROLLHIST_KI.json`, the G-T0 output and `MECH4_KINPUT.json`. Write `readki/MM_VERDICT_KINPUT.md` by the card's section 5 precedence (BLOCKED, then HARMS, then SEED-DISAGREE, then table order), with all four families and STRATEGIC in full.
2. Land the verdict, `pod_ki_run/` and `read_ki/` under `.../2026-11-23-kinlab/kinput/read/`, with the register row H-V8F-KINPUT-1.
3. Then the next lever the card names:
   * REPAIRS: a consistent-base card.
   * HARMS or PARTIAL with MECH4-YES: a dose card.
   * MECH4-NO or SEED-DISAGREE: the third seed pair. It goes to the PI with its cost and needs its own go.
4. **On STOP or exit 4:** read the chain log. On a headroom stop, relaunch the `.cmd` via WMI; the chain resumes.

## Not deleted
`/root/rollhist/kit_ki/ki_records.tar` (81,920 B) and `rh_ki_chain.log.copy` on the pod. The safety check refused an `rm` inside the ssh command; it is left for the PI. Both are copies, and the records are local and sha-verified.
