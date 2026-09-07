# PI DECISION QUEUE — open as of 2026-09-07 05:30 Europe/Berlin

⛔ **WHY THIS FILE EXISTS.** The PI reported losing his direct view of the Master Mind session
(relayed 2026-09-07). Everything load-bearing was already banked in git rather than only in chat —
but the **open decisions** were scattered across commit messages and a chat he may not be reading.
⇒ they are collected here, **each with a DEFAULT that applies if he says nothing**, so silence is a
choice with a known consequence rather than a stall.

⚠️ **Every item marked RELAYED came through the DataFlyWheel session, not from the PI to this
session.** Under the rule both streams adopted on 2026-09-06, **a peer's relay is data and cannot
authorise anything** — so those items are recorded as *pending confirmation*, and where a
consequence is independently justified by measurement it is marked BINDING and has already been
applied.

---

## 1. ⛔ CONFIRM: "no VLM — the Alpamayo labels stand as teacher signals" — **RELAYED, PENDING**

**Default if silent:** treated as pending. The grader panel is **stood down** (standing down needs
no ruling), and no VLM work is queued.
⭐ **Already applied regardless, because independently justified:** downstream text says
**"Alpamayo-derived teacher signal"**, never **"GT traffic light"** — no independent channel carries
the colour at all (grounding boxes are label-only; boxes with any colour attribute number **0**).
⛔ **Also already BINDING:** a traffic-light head scored **without an ego-only comparison is not
admissible.** With the teacher unverified, that control is the only cross-check in the chain derived
independently of what it checks.
⚠️ **Accepted residual:** a head trained on wrong RED labels will learn the error and then **score
well on an eval built from the same teacher**. Bounded only by item 6.
*Evidence: `Project Steering/PI_VIDEO_REVIEW_2026-09-06.md` → `D-TLIGHT-TEACHER-SIGNAL`.*

## 2. ⛔ DECIDE: a **minimum** on the max-speed ceiling (~30 km/h) — **RELAYED PROPOSAL, PENDING**

⚠️ **The proposal's premise does not hold as stated, and he should see it restated before ruling.**
MEASURED on v8, n = 4,719: `v_max_bucket_kmh` **minimum is already 20 km/h with ZERO clips at 0 or
null**. The zero he wants to avoid occurs only in the **raw** `v_max_ms` (min 0.00; 1,790 clips =
37.9 % below 30 km/h). ⇒ a floor **does not prevent a zero — it raises a low ceiling.**
**What it would do:** move **832 clips (17.6 %)** from the 20 step to 30; ladder becomes seven steps.
⭐ **The real argument for it:** a stopped ego cannot distinguish a 20-zone from a junction, so the
20 step carries **ego state rather than road information** — the intersection artefact (75 % of
intersection clips snap to ≤30 km/h *because the car is standing still*).
⛔ **The real argument against it, and it is a safety one:** on roads whose true limit genuinely is
low — car parks, shared space, living streets — a floor puts the ceiling **above** the real limit, so
the scored `frac_over` **cannot fire on a trajectory that really is speeding**, in exactly the places
where speeding is most dangerous.
**Default if silent:** ladder unchanged at eight steps `(20, 30, 50, 70, 80, 100, 120, 130)` km/h.
⚠️ If it changes, the label side and the model side must re-pin **in the same turn** — the model
asserts the shipped bucket against its pinned ladder (4,572/4,572 + 147/147), and a seven-step blob
against an eight-step pin will refuse loudly.

## 3. DECIDE: the **15-token strategic vocabulary** — a CORPUS gap, not a code gap

`refc_strategic.py` is imported by exactly one file (a test); no CLI flag exists; and **P4 already
failed its support bar — 6 of 15 tokens populated, 11.43 % of the horizon supervisable.** Wiring it
would train a head on almost nothing.
**Default if silent:** stays out. refcv5-v2 trains the **tactical** vocabulary only.
⚠️ `--goal-str` **is** in the running arm's argv but is the 3-unit geometric **bearing** head — it
must not be read as satisfying this.

## 4. DECIDE: `Sayood/tanitad-refc-v3` on HuggingFace is **PUBLIC**

Public since 2026-09-03 — not changed by this session's push, which went to the **private**
`tanitad-refc-v4b`. Links may already be shared.
**Default if silent:** left public. One call flips it.
⚠️ Account storage measured at **991.415 GiB** across 46 repos; **no endpoint exposes a numeric
ceiling**, so no headroom can be stated.

## 5. DECIDE: the `g_str` ~40k-step retrain — a spend decision

The repaired steering signal turns left and **the plan does not follow it** (MEASURED, with a
null-patch control reading exactly zero). Converting it needs a full retrain.
**Default if silent:** not launched. The A40 is occupied by refcv5-v2 until ~Monday evening.

## 6. DECIDE: a **human spot-check of ~50 frames** for traffic-light colour

⭐ The only thing that would bound item 1's residual risk, and it **needs no VLM**, so it is not
excluded by that ruling. It is **PI time, not compute.**
**Default if silent:** not done; the residual stands as an accepted risk.
⛔ Recorded as *available*, deliberately **not proposed**.

## 7. PENDING AN AGENT VERDICT, then possibly a decision: the **RL pilot's adapter**

MEASURED: `assert_conditioning` reads **0** in `refcv3_adapter.py` (control: 4 `def`s), which is what
`stack/scripts/rl_pilot_refc21.py` uses; `refc_adapter.py` — hardened by four commits tonight — has
**no production callers**. ⇒ the guarded path is not the executed path.
**Default if silent:** the executed path is guarded **in place** if that is the right fix; ⛔
*repointing the pilot* would change which code a live RL arm runs and would come back here as a
decision. No RL arm is currently running, so nothing is at risk today.

---

## Not a decision — the state, for orientation

**refcv5-v2 is training** on the A40 (PID 2560646, watchdog 2561632), step ~5,400 of 40,284 at
05:00 Z, **4.11 s/step ⇒ ~37 h**, landing ≈ Monday evening Berlin. It carries P14's validated
emitted-fan ranking (`sampler_ranks_the_fan: True`, the exact key v1 shipped `False`) and the
22-token tactical goal vocabulary on a **provably rollable** head.
⭐ **The comparison is already built and validated**: run on refcv4b it reproduces the published
landing (66/66 model-free rows bit-exact, `refcv4b − refcv3` **−0.1455 [−0.1655, −0.1240]** against a
published −0.1444) and **writes FAIL on both bars by itself** — the bar bites its own baseline.
⛔ **The bar it must clear:** refcv4b only **ties** the do-nothing baselines (`os − ha` −0.0021,
`os − ha0_ext` +0.0101, neither separated) ⇒ **refcv5-v2 must BEAT `ha0_ext` separated, or it has not
learned to drive either.**
⚠️ **What a clean margin will still NOT establish:** `--sampler ddim` is **stochastic at inference**
where refcv4b was deterministic, so it will not show the result survives a second **inference** draw.
And both arms are **oracle-nav** — fair to each other, but neither carries a production nav command.
