# P-battery on v6, first run — and the headline number is an ECHO

**MEASURED 2026-08-14, pod4** (idle; pod5 was training and was never touched).
Checkpoint `v6F-SW-30k/ckpt.pt` @ **step 2500 of 30000 (8.3 %)**, n = **881 windows** on the
canonical grid (episodes < 40, stride 8), parity VERIFIED (600 clips, clip sha256
`0b176d2e5cb4…`). Trunk: `v6 (V6Stack)`, `state_dim 2048` (= `cfg.d_op`, the geometry
firewall), md5 `e98e45a5647d`, frozen (md5 unchanged across the run).

**TIER: T0-diagnostic.** Nothing here is a driving-performance number.

---

## 1. The result, and the control that overturns half of it

`r2_enc` = probe on the ENCODED latent z_{t+k} · `r2_pred` = on the PREDICTED latent ẑ_{t+k}.

| target | k=5 | k=10 | k=15 | k=20 |
|---|---|---|---|---|
| **speed** | −1.438 / **+0.995** | −2.301 / **+0.980** | −1.939 / **+0.960** | −1.985 / **+0.938** |
| **speed — v0 SHUFFLED** | −1.438 / **−0.721** | −2.301 / **−1.067** | −1.939 / **−1.114** | −1.985 / **−1.246** |
| yaw_rate | −2.978 / −3.288 | −3.218 / −1.869 | −3.829 / −3.637 | −2.750 / −2.921 |
| curvature | −2.211 / −2.114 | −3.607 / −1.304 | −3.539 / −2.457 | −2.430 / −2.205 |
| lead_gap | n = 0 — no `--join-file`; reported not-computable, never faked |

⛔ **`R²(ẑ, speed) = 0.995` WAS NOT THE WORLD MODEL. It collapses to −0.721 when v0 is
shuffled.** `lift_actions3` appends `v0/SPEED_SCALE` as the predictor's **3rd action channel**,
constant along the horizon, and the predictor is FiLM-conditioned on it — so ẑ carries v0 **by
construction**. Future speed is smooth in v0, which is also exactly why the "signal" decayed so
gently with k (0.995 → 0.938): that is the autocorrelation of speed, not a learned dynamic.

⭐ **The control is self-validating: `r2_enc` is BIT-IDENTICAL across both runs** (−1.438,
−2.301, −1.939, −1.985). The intervention touched only the predictor's conditioning path, which
is precisely what it claims to isolate. A control that also moved the encoder numbers would
have been measuring something else.

**Method:** `--speed-echo-control` permutes v0 across the batch immediately before the action
lift — same frames, same recorded actions, wrong speed scalar. Implemented at the `collect_grid`
call site so `p8_latents` stays byte-identical for every other caller.

---

## 2. What this means

1. **The P1 speed row is INADMISSIBLE as evidence that the v6 world model carries speed.**
   Any future report quoting it must quote the control beside it, or not quote it.
2. **This is the third appearance of the same defect family**, and the reason the programme
   keeps a rule about it: the v1 route head was an exact bijection of its own nav input and
   scored **1.0000**; T1 found open-loop lateral skill was an action echo (97.9 % open-loop
   vs ~5 % closed-loop). **The generalised test — "does an input at inference contain something
   the thing being measured also produces?" — caught this one too.**
3. ⚠️ **The instrument found it in hours, not at final eval.** That is the entire argument for
   getting the P-battery running at step 2500 instead of step 30000. Had this gone unchecked,
   "the v6 WM decodes speed at R² 0.995" would have entered a report as a success.

## 3. What is NOT claimed

- ⚠️ **Every encoded R² is negative** — at step 2500 the encoder does not linearly carry speed,
  yaw rate or curvature. At **8.3 % of training** that is a **BASELINE, not a verdict.** A
  barely-trained encoder failing linear probes is unremarkable; what matters is the trajectory,
  and this is the first point on it.
- The gate printed `[P1 GATE] NOT COMPUTABLE` (R²(enc) ≤ 0 at the gate horizon, so the
  retention ratio has no denominator) and `[P2 GATE] FAIL`. **Neither is yet meaningful at this
  step count** and neither should be quoted as a v6 verdict.
- yaw_rate and curvature are negative under both conditions, so the control says nothing about
  them either way.

## 4. Next

- **Re-run at the next milestones** (5 k, 10 k) — the value is the curve, not this point.
- **Always run the control alongside**, now that it is one flag.
- ⚠️ **The same audit is owed by every other probe that consumes the lifted action channel.**
  P3/P6 roll counterfactual actions through the same 3-channel lift, and their speed channel is
  held at the OBSERVED v0 under every counterfactual — `stage_a_probes.py` states that as a
  limitation in its own docstring. Whether that limits P3's gain estimates is unmeasured.
- Supply a `--join-file` so `lead_gap` (the perception row) stops being n = 0.
