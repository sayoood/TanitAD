# Brief: finish, test and build the stage-2 overlay (AGENT I; CPU only; do NOT launch)

**Rules.** `BRIEF_T2V8F_STAGE2_G.md` binds in full: the preamble, CPU only, no ssh, never commit, stage or delete, LF, banned words,
≤ 150k context with `HANDOFF_I.md` at about 130k, and the frozen card is never edited. You own `t2v8f/`; never write in
`motion_c/exec_standstill/`.

**Your plan is `HANDOFF_G.md`.** Its code is written, compiles, and has NEVER RUN. Expect first-run errors.

## The Master Mind's rulings
* **The deciding pass is the stage-2 job.** It runs C0_s0/_s1, P10, ZERO and BLIND under T2 and RR, with `--executor base`, as the
  frozen card says.
* **The beside pass (DAGOACC) stays in the driver** behind its 10 GiB free-space refusal. If it refuses, that is reported, not worked
  around. Freeing the 525 MiB stage-1 tar on box 2 is a PI decision, and the MM will ask.
* **`hold` exists for a later dated amendment only.** It is never run by the stage-2 job unless the MM changes the job file.

## Items, in order
1. **Tests and mutations** (HANDOFF_G open 1). For the hold executor:
   * G-ID: at ε = 1e-3, hold equals base bit for bit;
   * G-MOV;
   * G-PHYS;
   * each with its RED arm.

   Then the stub rehearsal of the stage-2 driver: a planted missing scene file must make the preflight refuse. Add a resume test: DONE
   cells are skipped.
2. **The scorer** (open 2). Lateral and heading rows reported per start class (launch / moving). A launch start means v0 < 0.5 m/s at
   f0, measured from the logged track. Test it on the stage-1 pull: 00040136 is moving, 7c72937c is a launch.
3. **Build the real overlay** (open 3), detached. Report its sha256, its size, and the exact box-2 footprint per pass.
4. **`STAGE2_LAUNCH.md`** (open 4):
   * the commands: busy marker, a resumable scene copy for the 41 scenes, overlay put, certutil, extract, `launch.ps1`, a background
     waiter capped at 6 h, pull, dev-box score;
   * the marker meanings and the ESTIMATED wall time.
5. **`PINS_G.txt`.** Run the full suite once at the end (`pytest tests -q`, CUDA hidden, about 7 min). Every test stays green.

## Hand-back (≤ 300 words)
Tests (n passed) and mutations, the rehearsal, the overlay sha256 and size, the footprint per pass, the launch steps, open items,
blockers, and a deliverable manifest.
