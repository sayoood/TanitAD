# MM ruling: HNSCREEN-FP-COST (Master Mind, 2026-10-19 ~22:55)

**Decision: option (b), strengthened to (b′). The content check is kept in full; only its REPETITION is bounded.**

## Why not (a) or (c)
- **(a) is impossible on this box.** MEASURED 2026-10-19 22:50: C: has 7.3 GB free and D: has 24.1 GB free (Get-PSDrive). The cache is about 69.6 GiB (INHERITED, `PREREG_P6HN_SCREEN.md` §3).
- **(c) is not refused on principle.** It costs about 17 × 27 min ≈ 7.7 h of D: reads per 3-seed screen (COST.md: 44 MB/s cold, MEASURED). That is GPU-idle time on the box's one GPU, and the GPU is the binding resource.

## (b′), the rule to implement
1. **Full content fingerprints, unchanged**, at:
   - the cache seal (`ctx_cache` finalize/seal, 1 read);
   - the fill (`hn_fingerprints.py`, 1 read);
   - **once at CHAIN START** (1 read), recorded as the chain's content anchor.
2. **Intermediate gate calls** (`launch_gate.py run` / `finalize` / `verify` per seed) bind to that anchor's manifest by `(relative path, size, mtime_ns)` for EVERY file. Any mismatch, missing file or extra file is a REFUSAL, exactly as a content mismatch is today.
   - Each such call records which mode it used (`content` or `anchor-bound`) in its token or verify record.
   - An anchor older than the chain, or minted from a different cache directory, is refused.
3. **A full content fingerprint again at CHAIN END** (1 read), before any read of the screen's outputs.
   - A mismatch with the start anchor marks the chain **VOID**. Its outputs are never read, and the reader refuses a chain without an `END_VERIFY_OK` record bound to the same anchor.
   - This closes the residual (a same-size, same-mtime edit mid-chain): it is DETECTED after the fact and voids the run, instead of passing silently.
4. **Total:** 4 full reads per screen instead of 17, about 1.8 h on D: instead of 7.7 h (ESTIMATED from COST.md's MEASURED 44 MB/s).

## Required RED arms (each must make the gate, the chain or the reader refuse)
- R1: a same-size, same-mtime_ns byte edit made mid-chain ⇒ the end verify FAILS ⇒ VOID ⇒ the reader refuses.
- R2: a size change mid-chain ⇒ the next intermediate call refuses.
- R3: an mtime_ns change with identical bytes ⇒ refuses (conservative by design).
- R4: an extra file and a missing file ⇒ refuses.
- R5: the end verify skipped or deleted ⇒ the reader refuses.
- R6: an anchor from another cache directory, or minted before the chain started ⇒ refuses.
- R7: forcing `content` mode everywhere reproduces today's behaviour bit for bit (the identity arm).

## Scope
- This applies ONLY to the HN screen's ctx cache path.
- Every other `launch_gate` data-manifest check keeps full content reads at every call.
- The G-* gate semantics of the PI's 2026-09-26 rule are unchanged: a token is still bound to the commit, the argv sha256 and the data manifest. Only the manifest's re-verification cost changes, inside one chain.
