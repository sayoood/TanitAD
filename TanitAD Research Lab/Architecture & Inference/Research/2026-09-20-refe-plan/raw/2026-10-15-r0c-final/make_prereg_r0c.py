"""Build eval/PREREG_R0C.md (IN FORCE) = a dated header + the draft's sections 1-5 VERBATIM (read from git, never retyped)."""
import subprocess, sys

tip, out, counts = sys.argv[1], sys.argv[2], sys.argv[3]
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
draft = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "show", f"{tip}:{PK}/raw/2026-10-15-rls1-adam-dose/PREREG_R0C_DRAFT.md"],
                       capture_output=True, check=True).stdout.decode("utf-8")
assert draft.startswith("# PREREG DRAFT: R0c"), draft[:60]
body = draft.split("\n## 1. Question", 1)
assert len(body) == 2
sections = "## 1. Question" + body[1]

header = f"""# PREREG: R0c -- anchor the EPO term to arm A's policy instead of the frozen base (IN FORCE 2026-10-15)

> ⭐ **IN FORCE. Finalised 2026-10-15 ~22:45 Berlin, BEFORE the first R0c step.**
> - **Source.** `raw/2026-10-15-rls1-adam-dose/PREREG_R0C_DRAFT.md`. **Sections 1-5 below are that draft, VERBATIM.** They were copied
>   from git by `make_prereg_r0c.py`, not retyped. Nothing in them changed after the decision; the draft's "DRAFT / NOT implemented"
>   line is historical, and this header records what changed since.
> - **PI authorisation, verbatim (2026-10-15):** *"run R0c, I dont have currently asecon pod or gpu aprt of my local computer with the
>   8gb gpu"*. This is the PI decision that section 5 asks for: about 3.5 A40-h, beyond RL-S1b's authorised budget.
> - **Slot: after the P2 seed replicate (P2-s1), BEFORE P6's GPU stages.** No code change is needed for it.
>   - The RL-S1 chain's queue (`refe/rls1_pod.sh` header and `queue_wait`) waits behind the replicate holding only `chain.queue`.
>   - It takes `chain.lock` within `RLS1_QUEUE_POLL_S` (5 s) once the replicate's chain exits. P6 is not "ahead" until it logs
>     `ZZP6_WAIT_OK`.
>   - P6's `wait_clear` (`refe/p6_pod.sh`) does not pass while `$DATA/rls1/chain.lock` is held, and it needs 3 idle polls of 10 s.
>   - ⇒ P6 starts after R0c's chain ends. Cost to P6: about 3.5-4 h of delay (ESTIMATED).
> - **Implementation (section 2's "trainer edit (NOT done)" is now DONE).**
>   - Code: `raw/2026-10-15-r0c-build/PATCH_1..7` + `test_rl_r0c.py`, and the chain wiring `raw/2026-10-15-r0c-build/chain/PATCH_8..12`.
>     They are applied in the commit that lands this file. The superseded `PATCH_10_*.on_94248ec.diff` is NOT applied.
>   - Argv: `raw/2026-10-15-r0c-build/R0c_ARGV.txt` = `eval/rls1c_arms.json` (profile `refe_rls1c`). The launch gate's PASS token binds
>     it with the commit and the data manifests.
> - **Section 3's checks.**
>   - `test_rl_r0c.py`: identity at step 0, a non-zero B gradient, an EXACT 0.0 on A, the mutations, a named default. Each item is mapped
>     in `raw/2026-10-15-r0c-build/MM_VERIFY.md`.
>   - Re-run on this exact tree: **{counts}** (logs: `raw/2026-10-15-r0c-final/`).
> - **Second control.** C0's banked score dumps were restored to the pod from `Sayood/tanitad-refe-rls1-c0` (`final/scores_*.npz`,
>   15 files, each sha256-verified against HF), so the `rlval_vs_C0.json` read runs.
>   - Recorded before launch: had C0 been missing, the run would NOT refuse. The chain logs `ZZRLS1_R0C_NO_SECOND_CONTROL`, and every
>     registered delta is against the arm's own A (section 2, guards).
> - **Tier and estimator, as section 4.** RL-val selection tier: one training seed, paired log-cluster bootstrap, 10,000 resamples,
>   seed 20260927. **A separated interval is NECESSARY, not sufficient** (`H-ESTIM-SEED-1`). An O1 is a candidate until its
>   replicate runs.
> - **From here on, nothing below may change without a dated amendment written before the read.**

"""
open(out, "w", encoding="utf-8", newline="\n").write(header + sections)
print("written", out, len(header + sections))
