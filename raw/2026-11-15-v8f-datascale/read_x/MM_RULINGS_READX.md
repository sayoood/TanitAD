# MM rulings on the X_s0 read build (2026-11-18, before any X_s0 number has been read)

1. **K-VLM-ZERO's "block-1 floor" is |VLM macro-F1(B_rep) − VLM macro-F1(B_s0)| at stage A (0.0274), as the builder read it.**
   * The control passes iff X_s0's stage-A VLM macro-F1 is at most B_s0's minus 0.0274 (0.0984), OR at most 0.05.
   * This is the reading intended in `blockx/MM_RULINGS_X_READ.md` R4. The bar is unchanged.
2. **C-JOIN-EQ is quoted from the banked gate record only** (`X_s0.<stage>/gate/checks.json`, sha256-verified against the banker ledger).
   * The chain log carries no C-JOIN-EQ line, so it adds nothing.
   * The stage-B mint record on the pod read C-JOIN-EQ True (MM, 10:19Z refusal record, and again at the 10:22Z re-mint).
3. **Launch.** The MM launches after `ZZ DSCALE10K BLOCKX DONE ZZ` in `/root/dscale10k/blockx_r2.log` (the relaunched chain; `blockx.log` holds the first run's stage-B mint refusal, explained in LOOP_STATE 2026-11-18).
   * Use the WMI create of `job_dscale10k_x.cmd` (CreateFlags 0x08000200, stdin NUL). `powershell -File` is refused in this environment.
