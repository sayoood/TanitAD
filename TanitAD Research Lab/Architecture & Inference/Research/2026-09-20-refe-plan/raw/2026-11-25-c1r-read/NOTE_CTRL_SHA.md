# C1r read: the first run's CTRL-SHA FAIL was a tool fault, and how it was fixed

MM, 2026-11-25.

## What happened
- **The first run** (`VERDICT_c1r.run1_chainlog.txt`) stamped the read INVALID on CTRL-SHA alone. The other six controls passed, and the numbers matched the final run.
- **The cause.** C1's `model_final.pt` was no longer on the dev box, so `run_post.sh` passed `C1=chainlog:<p5c_read>/CHAIN.log`. That made `c1r_read.py` look for C1's sha in the chain log instead of hashing the file. The lookup took the chain log's FIRST `RUN s2_c1:` line (CHAIN.log:21), which was an attempt that exited rc 1. The successful run is CHAIN.log:54-55: rc 0, `--expect-ckpt-sha256 f09949e2…`. So C1's sha printed `NOT-VERIFIED`.
- **The chain-log route was never the card's control anyway.** Card §5 says: "both model files equal their sha256 values above". That needs the file itself.

## The fix (tool mechanics only; the card is unchanged)
- C1 was re-pulled from HF `Sayood/tanitad-refe-p5c` `p5c/c1/model_final.pt` with the P5-C read's own `hf_pull.py`. It is 1,288,515,000 B, and its sha256 is `f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4`, equal to card §2 and MODEL_REGISTRY 14.6.
- `run_post.sh` now passes `C1=<that file>`, and only the `read` step was re-run.
- The result is **CTRL-SHA PASS**, with all seven controls PASS and the read VALID.
- The numbers are identical to the first run's, because CTRL-SHA reads no data. This decision was made from the control's wording, not from the numbers: the d_seed, CI and family values printed by run 1 are the same values the VALID run prints.
- The re-pulled copy is temporary. It is deleted after this land; HF keeps the banked copy.
