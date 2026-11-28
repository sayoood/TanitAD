# R8-WISEFT + R8-CKPT-TRAJ -- no-training levers for refcv8's lost NavSim transfer (registered 2026-10-21 13:37 Berlin)

**Why.** refcv8-r101-s0 is a fine-tune of refcv7-r101-s0 (step 50,400, md5 d5f104ee). On navtest refcv7's learned stage adds
+11.54 PDMS over the shared causal prior; refcv8's adds -0.82 (R8_LEGAL 57.84, PRIOR 58.66, refcv7 70.20). The command feed is
not the cause (R8-NAVFEED-DIAG, +0.75 [-1.60, +3.04]). Before the training-side lever (PI + compute), two no-training levers:

* **WiSE-FT** -- theta_lam = (1-lam)*theta7 + lam*theta8 on the shared keys, lam in {0, 0.25, 0.5, 0.75} (`code/build_wiseft.py`,
  controls in `build/WISEFT_BUILD.json`: lam=1 == theta8 and lam=0 == theta7 bit-exact, read-back bit-exact).
* **Checkpoint trajectory** -- EMA-5,000 / 15,000 / 30,000 beside the banked 40k / 48,852 (diagnostic only).

Reading rule: `PREREG_R8_WISEFT.json` (and the GOALS_AND_CLAIMS row of the same name), fixed before any of these arms is scored.
Rolls: Thor CUDA (`code/r8ns_wiseft_LAUNCH.sh`, behind the GPU flock); scoring: Thor CPU (`code/r8ns_wiseft_score_loop.sh`), which
refuses to score until `PREREG_ACK.txt` (the landing commit of this registration) exists on Thor.

Where things live: interpolated checkpoints (4 x ~403 MB) stay on Thor at `/home/nvidia/r8ns/navtrain_sel/wiseft/ckpt/` (md5s in
the build report); results will be banked here as `result/`.
