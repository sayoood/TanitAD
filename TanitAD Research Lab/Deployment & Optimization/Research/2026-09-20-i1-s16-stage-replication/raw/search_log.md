# search log — E-DEP-VGEO-6 / s16 stage replication (2026-09-20)

`Every probe, including the empty ones.`

| # | probe | hits | note |
|---|---|---|---|
| 1 | bank: `tokens_s16_fp16.npy` present and shaped for the panel | (27664, 352, 16, 40) fp16, 12.46 GB | ✅ the s16 stage exists in the SAME bank as s32 — this run needed no new extraction |
| 2 | bank meta `index_s16_meta.json` | `ckpt_step` 40284, `encoder_params` 90,458,632, `missing: []`, `unexpected: []`, `tokens_finite: true` | ✅ same checkpoint as VGEO-5 ⇒ the stage really is the only variable |
| 3 | ⭐ **V-1 / LI19-1 re-find grep** — has this panel been run on s16 before? `grep -rl "s16" Deployment & Optimization/Research/*/` | prior mentions are **cache-geometry** notes only; no prior VGEO panel on s16 | named, rather than asserting "first" |
| 4 | ⭐ **V-1 re-find on `2601.00844`** before citing it as support | **10 files** — `LAB-RUN-004`, `2026-09-10-i1-value-guided-latent`, `2026-09-13-i1-value-geometry-screen`, `LEDGER_A2_jepa`, Deploy `KNOWLEDGE_BASE`, three Frontier dailies | ⛔ **NOT a new find.** It is this line's *founding* citation and is cited as such below, not presented as today's discovery |
| 5 | primary re-read of `2601.00844` from the banked PDF (local `pypdf`) | 7 pages, 24,958 chars | ⭐⭐ verbatim: *"we **learn representations such that** the Euclidean distance (or a quasi-distance) between embedded states approximates the negative goal-conditioned value function"*. **The encoder is SHAPED for the property.** Evaluated on toy **wall (200 instances) and maze (80)** environments. ⛔ the string *"frozen"* appears **0 times** in the paper |
| 6 | web: *"arxiv 2026 frozen visual features place recognition versus metric distance estimation which layer stride"* | 10 | ⛔ **EMPTY for the question asked** — VPR papers (KappaPlace `2605.19435`, VDNA-PR, Riemannian VPR `2602.00841`) optimise **place retrieval** and none contrasts it against **metric** distance recovery from the *same frozen* features, nor across encoder **stages**. ⇒ the s32-vs-s16 × place-vs-geometry contrast has **no located published analogue**; single-probe absence, not folklore |
| 7 | web: *"arxiv 2026 world model value function learned terminal value latent planning driving beats sampling"* | 9 | `2604.14732` **World-Value-Action Model** (trajectory value function in latent space) and `2609.03294` **LEAP** (already on the backlog as FS19-4). Neither is a **frozen**-representation result. Noted for the A1/A2 ledgers; not deep-read here |

## ⭐ What probe 5 changes about today's verdict

The programme's founding citation for I-1 **shapes the representation so latent distance is value**.
This run measured that our **unshaped, frozen** trunk does *not* carry metric path geometry (a powered
TIE on s32; pixels ahead on s16) but *does* carry place identity, twice, on two stages.
⇒ The literature and the measurement **agree**: a metric-distance value head needs a *trained-for-it*
representation. What our frozen trunk offers for free is **place**, not **distance** — which is precisely
what LR15-2's re-target proposes. ⚠️ `2601.00844` evaluates on **toy wall/maze** environments with no
driving benchmark, so it supports the *mechanism*, never a number of ours.
