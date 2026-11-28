# RESULT — P3: the Alpamayo meta-action census on the parity corpora

**Measured** 2026-08-27 · MEASURED (ours; local parquet + Thor cache basenames) ·
raw `raw/alpamayo_census.json` · source `Sayood/tanitad-alpamayo2-augmentation`
(`meta_action` task, 4,729 rows, labeller `nvidia/Alpamayo2-Super`,
⚠️ quantisation stamped `NF4-backbone-4bit-UNVALIDATED` in the data).

## 1. ⭐ The label space is FACTORED — exactly what the v7r tactical head needs

**93.6 %** of rows parse into `Longitudinal × Lateral`:

| LON (7) | rate | | LAT (7) | rate |
|---|---|---|---|---|
| Gentle Deceleration | 33.7 % | | Go Straight | **52.9 %** |
| Maintain Speed | 25.9 % | | Steer Right | 21.6 % |
| Gentle Acceleration | 24.3 % | | Steer Left | 13.7 % |
| Stop | 6.4 % | | Sharp Steer R/L | 2.9/2.4 % |
| Strong Decel / Accel | 5.6/3.8 % | | Reverse R/L | ~0 % |
| Reverse | 0.1 % | | | |

Base rates are workable (majority: LON 33.7 %, LAT 52.9 %) — **not** C136's
degenerate axis. Chance baselines for any classifier claim: those majorities.
⇒ This is the LAT/LON-separated vocabulary that replaces the 5-way mixed softmax
(the longitudinal-blindness root cause), as adopted in `V7R_DESIGN_PROPOSAL` §1.2.

## 2. ⛔ THE BLOCKER: parity coverage is ~8–9 %

| corpus | covered | of |
|---|---|---|
| parity **train** | **201** | 2,402 names (8.4 %) |
| parity **val** | **56** | 602 names (9.3 %) |

*(Name lists from Thor cache basenames — the manifest withholds raw clip ids by
design; denominators carry ~1 % non-clip filename noise, stated.)*

The 4,729 labelled clips came from the Alpamayo SELECTION pool, not the parity
corpus. ⇒ **The v7r tactical head cannot be trained on parity clips from this
label set as-is.** Two routes, a PI/Stage-B decision:

1. **Label the parity corpus** with the same battery — measured cost basis
   (registry): meta_action ≈ 11.7 s/clip ⇒ ~7.8 h of Alpamayo2-Super inference
   for the 2,376 train clips (compute provisioning = PI).
   ⚠️ The existing pass is stamped **NF4-4bit-UNVALIDATED** — a re-pass should
   settle the quantisation validation question first or inherit the caveat.
2. **Train tactical on the labelled 4,729** (labels may use anything; the WM
   stays on parity) and keep parity val for eval via its 56 covered clips —
   n=56 eval clips is thin and must be stated wherever quoted.

## 3. What this feeds

`V7R_DESIGN_PROPOSAL` §1.2 (tactical label space: adopted, with this coverage
caveat now attached) and the Stage-B commissioning decision (the PI's pending go).
