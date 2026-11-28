# PRE-REGISTRATION — E1e: is the closed-loop / open-loop conflict a WEIGHTING artifact?

**Written and committed BEFORE the run produced any number.** pod3 (idle; pod1 and pod2 are
unreachable and untouched). Standing directive **D-A** — *"loop until significant closed-loop
performance … when it lands, read the frontier and decide the next step."*

## 1. Where the chain stands, and why this is the next rung

- **E1c** (CL-SFT): corridor departure **0.5877 → 0.147** (Δ −0.4407 at step 2750), junction
  **0.8414 → ~0.40**; P1∧P2 CI-separated at **15/17** checkpoints. **BOUND** — guardrail Ga
  (open-loop ADE@2s not separated-higher) held at **0/17**.
- **Training longer is ruled out**: open-loop cost plateaus from step 2250 (eight points inside
  ±0.02, no trend).
- **E1d** (WiSE-FT α-interpolation): **BOUND**, and stronger than a null — `dep_overall` is
  **separated-WORSE at five consecutive interior α** (0.20–0.60). The base→FT segment crosses a
  barrier; the two solutions are **not linearly mode-connected** for this metric. Logged **C52**.

E1d moved along a segment between two *already-trained* endpoints. It could not reach a solution
that is not on that segment. **E1e changes the objective instead**, which is exactly where E1c's
§4.2 and E2a both pointed ("the lever is the objective, not the encoder or the denoise steps").

## 2. The lever — and it is one existing flag, untouched by design

`e1c_clsft.py:420` — `loss = lam_cl * cl_loss + lam_replay * replay_loss`.
E1c ran `lam_cl = lam_replay = 1.0`, and its own header states **"lam_replay is deliberately NOT a
lever here"**. So the anti-forgetting term exists, was never varied, and open-loop *still* regressed
41 %. **Raising it is the cheapest discriminating experiment available** — one flag, no new code.

**Arms** (everything else byte-identical to `run_e1c.sh`: same base ckpt, same mined buffer
`a32cfe9bfea4b1b5c196d3bb7f71fa5f`, same parity dir, same held-out dir, `--steps 4000 --lr 2e-5
--warmup 100 --cl-batch 16 --replay-batch 16 --lam-cl 1.0 --freeze-encoder 1 --seed 0`):

| arm | `lam_replay` | status |
|---|---|---|
| E1c (incumbent) | 1.0 | MEASURED — dep −0.4274, ade +0.1947 @4000 |
| **E1e-A** | **3.0** | to run |
| **E1e-B** | **8.0** | to run, **only if A is informative** (see §4) |

**Priority order** — A first. If `lam_replay = 3.0` already destroys the closed-loop gain, B is
pointless and is **not run**; that decision is recorded here in advance so skipping it is not
post-hoc.

Evaluated by **`e1c_eval.py`, unmodified**, at steps 1000/2000/3000/4000 — same estimator (**paired
episode-cluster bootstrap**, `taniteval/ci.py`, B=2000, 44 held-out episodes), same
`evaluate_point`/`render_verdict`, same P1/P2/Ga/Gb1/Gb2/Gc conditions as E1c and E1d.
⚠️ **The held-out split is content-verified clean at the sensor level** against *this base arm's own
training corpus* (0/44 episodes, 0 shared frames — `…/2026-07-28-e1-heldout-frames-leakcheck/`), so a
result here is not a leak artifact.

## 3. Both outcomes, committed in advance

**OUTCOME A — a SUCCESS POINT exists.** Some (arm, step) satisfies **P1 ∧ P2 ∧ Ga ∧ Gb1 ∧ Gb2 ∧ Gc**
on held-out data.
⇒ **The conflict was a WEIGHTING artifact**: the anti-forgetting term was simply under-weighted, and
the closed-loop win is available without an open-loop regression. That checkpoint becomes the **D-A
deliverable**, is registered with its `lam_replay`, and the next step is a from-scratch confirmation
rather than another knob.

**OUTCOME B — BOUND again at every arm.**
⇒ **A real result, and it closes a whole class:** the conflict is **not** a scalar-weighting artifact.
Between E1c (training time), E1d (weight space) and E1e (loss weighting), three independent
one-dimensional levers will have failed. ⇒ The remaining hypothesis is that the **TARGET is wrong,
not its weight** — the buffer supervises *all* recoverable pre-failure states, while E1d measured
that **junction recovery is cheap and monotone whereas overall-corridor recovery is expensive and
barrier-crossing**. The next experiment would then be a **junction-restricted buffer**, i.e. a change
to *what* is supervised. **We do NOT run a finer `lam_replay` grid** — pre-committed as inadmissible,
for the same reason a finer α grid was.

**Either way it is reported**, with the estimator named, per corpus, no pooling.

## 4. Stop rules, fixed now

- **A is uninformative** iff it neither improves Ga (open-loop delta not smaller than E1c's +0.1947
  at the matched step 4000) **nor** retains P1. In that case B still runs — the direction would be
  unresolved.
- **B is skipped** iff A shows the closed-loop gain already destroyed (P1 false at every step) *and*
  Ga still fails — more replay can only push further along the same direction.
- **No arm is re-run with a different seed** inside this experiment; seed sensitivity is a separate
  question and would need its own pre-registration.

## 5. Honest bounds

- n = 43 episode clusters at K=185, **6 junction windows** — junction CIs are wide and must not be
  quoted as precise effect sizes.
- `lam_replay` and `lam_cl` are not independent: only their ratio matters up to the LR schedule, so
  "raise replay" and "lower CL" are the same axis. Stated so the arm list is not read as two levers.
- This is a **fine-tune of one base checkpoint**, not a from-scratch result; any winner owes a
  from-scratch confirmation before entering a headline.
