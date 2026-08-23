# Hierarchy panel re-run, gate-swept — the blast radius is smaller than feared

**MEASURED 2026-08-06** · `flagship-v1arch-v2bal-30k` @ step **29999** · **880 windows** /
40 PhysicalAI OOD-val q90 episodes · stride 8 · 141.1 s on an idle A40 ·
`results/hier_v1arch_gateswept.json.xz` · evidence class **MEASURED (ours)**.

Run because `RETRACTION_LOG.md` R-2026-08-06-yawgate left an open item: `DIR_YAW_RAD = 0.15`
decides every published manoeuvre-coherence κ in the programme, and no banked panel could be
re-read at another gate because they stored only the **thresholded** direction classes. The
raw net yaw is now banked, so this is the last panel that will ever need a GPU to answer
the question.

## The sweep

| gate (rad) | man~traj κ | traj~gt κ | frac of GT windows turning |
|---|---|---|---|
| **0.15** *(published)* | **0.5787** | 0.8260 | 13.41 % |
| 0.10 | 0.5715 | 0.7781 | 18.75 % |
| 0.06 | 0.4796 | 0.7245 | 26.36 % |
| 0.04 | 0.4075 | 0.6922 | 32.50 % |
| 0.02 | 0.3065 | 0.6582 | 46.59 % |
| 0.01 | 0.2038 | 0.5853 | 60.23 % |

Human net yaw over 2 s: median **0.0171 rad**, p90 **0.2095**, **13.41 %** above the
published gate. (Independently consistent with the 39-clip Alpamayo read: 0.023 / 0.185 /
17.9 %.)

## ⇒ Verdict: STABLE. The panel's conclusion survives; its NUMBER does not travel.

`kappa_range = [0.2038, 0.5787]`, `verdict_stable = **true**` — κ stays at or above the
panel's own 0.2 coherence threshold at **every** swept gate. **The published coherence
call for this arm was not an artifact and is not retracted.**

⚠️ **But the magnitude spans 2.8×.** "κ = 0.5787" is only true at 0.15; at 0.01 the same
model on the same windows scores 0.2038. ⇒ **The verdict is quotable, the number is not —
never without its gate.** That distinction is exactly what the instrument was built to
draw, and this is the first panel where it could be drawn at all.

⚠️ **`kappa_turn_subset` at the published gate is 0.2005** — the subset where at least one
side signals a turn sits *on* the 0.2 threshold, not above it. The comfortable 0.5787 is
carried by the straight-dominated majority. On the windows where a direction decision
actually exists, coherence is marginal.

## What this does and does not change

| claim | status |
|---|---|
| R-2026-08-06-yawgate's retraction of the **Alpamayo TACTICAL ranking** | **STANDS** — that ranking genuinely reversed between 0.15 and 0.10 (0.4968/0.3333 → 0.7263/0.7292) |
| "every published manoeuvre-coherence κ in the programme must be re-read" | ⭐ **PARTIALLY CLOSED, in the panel's favour.** For the arm that matters — the deployed flagship — the verdict holds across the sweep. |
| the flagship's declared-vs-driven κ of **0.3432** from the 39-clip comparison | ⚠️ **superseded by power, not by error.** This panel is **880 windows** against 39 single windows and reads **0.5787** at the same gate. Quote the panel. |
| our declaration degrades at fine scales while Alpamayo's improves | **CONFIRMED, and now well-powered.** Ours: 0.5787 → 0.2038 as the gate tightens. Alpamayo's: 0.1961 → 0.4660. Our manoeuvre head is coarse-scale coherent and carries no fine lateral information. |

⇒ The architectural reading in `V1_DEFECT_TRIAGE.md` §3 is **strengthened**: the problem with
our tactical head is not that it is incoherent, it is that it is **coarse** — it has no
severity axis and no lateral/longitudinal factorisation, so everything below ~0.1 rad is
invisible to it. That is a vocabulary problem, and it is the cheap fix.

## Remaining

Panels for other arms (REF-B, REF-C, v2corpus) still carry unswept κ and now say so in their
own output (`gate_sensitivity.status = UNAVAILABLE` with the reason). Re-running them is
cheap but not urgent: the deployed arm's verdict held, so no decision is currently resting on
an unswept number.
