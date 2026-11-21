# PREREG_V8F_LORA AR-2 handoff (written 2026-10-25 ~06:15) -- ARMED AND RUNNING

* PI go (2026-10-25 ~06:10, verbatim "Go"), the answer to: "run box 2's next jobs in order: AR-2 now, then KIND once its instruments land and pass the MM's checks".
* Chain: `box2_ship/chain_v8f_lora_ar2.sh` (== tip 59c27d92's `code/box2_ops/chain_v8f_lora_ar2.sh`, landed at 3b4ba0c1), wrapper `run_chain_v8f_lora_ar2.cmd`, log `logs/chain_v8f_lora_ar2.out`, launched via WMI 06:14, driver bash pid 53556. Dry-run: `logs/chain_v8f_lora_ar2_dryrun.out`.
* Guard: `box2_ship/v8f_guard.sh` replaced at 06:13 by the tip's widened version (kills v8f_lora_smoke / v8f_lora_identity_dev too); the old one is `v8f_guard.sh.pre_ar2_20261025`. Guard log `logs/v8floraar2_guard.out`.
* Waker: `box2_ship/wake_v8floraar2.sh 53556` (background; exits on DONE/STOP, driver gone, guard stop).
* Steps: dev-side production frame build (4 train + 2 eval clips, CPU) -> compare to banked frames -> ship + sha256 -> preflight -> identitylora (same-device identity + mutant, tau_device 1e-3) -> smokelora (200 steps, bf16, --amp OFF) -> `v8f_lora_ar2_read.py` (venue rule: <= 36 h per seed => box 2, else the A40, which needs its own PI go).
* On DONE / STOP:
  1. Report the identity verdict and the smoke read (step time, peak memory, projected hours per seed, venue) as committed; INSTRUMENT-only, no capability claim.
  2. Bank `raw/2026-10-25-lora-ar2/run/` (chain log, guard, dry-run, identity/*, smoke/*, reads, wrapper, this handoff closed). Register H-V8F-LORA-1.
  3. Arm KIND next if its instruments have landed (they are being built by an agent in scratchpad/kind/; verify, land, dry-run, then launch with AFTER_OUT = this chain's log).
  4. If the identity FAILS: Amendment 2's fallback (the arm is two levers until it passes); diagnose from identity_read.json, do not launch the full LoRA run.

## Run 1 STOPPED 06:25, run 2 ARMED 06:55
* Run 1: identity PASS bit for bit (rel L2 0.0, 6 clips, RTX 4060; mutant 0.028 caught), then the smoke crashed: CPU torch.Generator in live_check_lora on a cuda model. Fixed + test pinned + run-1 evidence banked at 771a4eac (raw/2026-10-25-lora-ar2/run1_genbug/). Run-1 files renamed *.run1_genbug* on both machines.
* Run 2: same chain, launched via WMI 06:55, driver pid 47068, box 2 refreshed from 771a4eac. Waker `wake_v8floraar2.sh 47068` (background).
* KIND is queued behind it: `chain_v8f_kind.sh` (tip 7205d4a6) with AFTER_OUT = logs/chain_v8f_lora_ar2.out, wrapper `run_chain_v8f_kind.cmd`, log `logs/chain_v8f_kind.out`, driver pid 10080, waker `wake_v8fkind.sh 10080`. If AR-2 STOPs, KIND stops too ("predecessor stopped"): diagnose AR-2, clear BOX2_BUSY.txt by rename, relaunch KIND with AFTER_OUT pointing at a DONE log.
* On KIND DONE: report the committed outcome (ONE KIND CARRIES / DIFFUSE / INTERACTIVE / MIXED), the SPEED side-read, Amendment 1's strata, the four families; bank RESULT_V8F_KIND.md + raw/2026-10-25-kind/run/; register H-V8F-KIND-1; then the branch: DIFFUSE/INTERACTIVE/MIXED -> PREREG_V8F_MIXO instruments; ONE KIND -> its T-side PREREG.
