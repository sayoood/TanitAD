# SR-TTC Amendment 2: the PI question (DRAFT, 2026-10-18)

## The question

This must be answered before any SR-TTC fit data is read, and before Amendment 2 is registered.

**Do you approve (a) or (b)?**

**(a) The full fit set.**
- S0 is ≈ **39.8 GPU-h** at 1.44 s/token: 99,495 tokens × 1.44 / 3,600.
- It could be ≈ 84 h of wall-clock on the contended dev-box GPU, or ≈ 70 GPU-h if the forward turns out to be per set rather than per token.
- It needs nuPlan DBs for 927 of the 1,167 fit logs, which must be staged: ≈ 135 GB.

**(b) The nested log subsample in `PREREG_SELRANK_ARM.amendment2.draft.md`.**
- **Start at 10 % of the fit logs:** 117 logs, 8,922 tokens, 15,808 sets. S0 ≈ **3.6 GPU-h**, 7.6 h contended, and 92 DBs ≈ 13 GB to stage.
- The logs are picked by a salted sha256 of the log name, fixed before any data is read.
- **Grow 10 → 25 → 50 → 100 %, forwarding only the increment, and only when a fit-set learning-curve rule says the head is still data-limited.**
- **Your cap is the envelope f_max.**

**Separately, where does S0 run?** The dev box has both the DB and four camera zips for only **22 of the 117 logs** at 10 %. Frame completeness inside those zips is unverified. So S0 needs the host where P2 trained, or staging, at any size.

## Recommendation: (b), with f₀ = 10 % and f_max = 100 %

**Why it is better than (a):**
- **It costs no more than (a) in the worst case.** The subsamples are nested, so scaling to 100 % costs exactly (a)'s 39.8 GPU-h and nothing is wasted.
- **The best case is 3.6 GPU-h.**
- **The dev gate is read once in every branch,** because scale-up is decided on the fit set before the gate is read.
- **A FAIL can be interpreted.**
  - A plateaued learning curve means "the head or features are limiting, not the data".
  - Any other learning-curve result triggers more data before the gate is read, so "too little data" is never mislabelled as "head cannot learn".

**What (b) costs over (a):** calendar time, up to four S0 rounds and four staging rounds.

**If you set a lower cap:**
- Example: 25 % ≈ 9.6 GPU-h cumulative.
- The cap-hit branch is pre-committed as **DATA-LIMITED, no verdict**.
- It allows one disclosed second gate read after a later scale-up.

## Corrections found while drafting

1. **The ~0.17 MDE belongs to navtest, not the dev gate.**
   - The brief called ~0.17 the "dev-gate MDE". It is the **navtest** confirm MDE (registered §6).
   - The dev gate has no MDE. Its +0.30 bar comes from the training-seed range 0.33–0.44.
   - Neither number depends on the fit size.
2. **Amendment 2 is incomplete without Amendment 1 item 3.**
   - Item 3 is the dev-gate coverage question: 603 of 3,137 sets are covered today.
   - This draft does **not** decide it. Amendment 2 cannot be registered until item 3 is added.
3. **The per-token cost is not re-verified.**
   - Amendment 1 says "one pass per token suffices". That is INHERITED, not re-verified.
   - The fit sets are on-policy fans from training checkpoints, at 1.76 sets per token.
   - Time one log both ways before S0.

## Register

No claim was asserted, supported or refuted, so `GOALS_AND_CLAIMS.md` needs no update from this draft.
