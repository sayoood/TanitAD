<title>LEDGER A2 — JEPA family, joint-embedding and predictive architectures</title>

# LEDGER A2 — JEPA · joint-embedding · predictive architectures

⛔ **APPEND-ONLY.**

`⭐ CREATED 2026-09-02. TRACKS.md had linked this file since 2026-08-31 without it existing (backlog FS-6,`
`root-cause class: a link written in the same turn as the intention to write the file, never verified).`
`The 2026-08-31 A2 DEEP is recorded below from that pass's RESULT.md as INHERITED, and is marked as such.`

## Our position in this track

**TanitAD's objective family is JEPA.** The v7 predictor learns action-conditioned dynamics in a latent
space rather than in pixels, which is what makes the programme's sub-300M ambition arithmetically possible.

⛔ **And the programme's gating defect lives exactly here.** `V7_LAUNCH_GATE` P1: **no v7 arm has ever
beaten its own hold-action control** — all three arms `LOSES_TO_HOLDV0` on every distance metric, heading
error ~95 deg (chance). P2: **the predictor does not use its actions**; h=1 action/scene ratio **0.004**.
The one positive is that `copy_detector` is **CLEAN** (echo 0.0000) — fake-skill-by-echo was traded for an
honest absence of skill.

⇒ **This track's job is to answer whether that is our bug or the field's.** As of 2026-09-02 the evidence
says: **the field's** — and the field has published fixes.

---

## Entry 2026-08-31-01 — anti-collapse landscape (INHERITED from the 2026-08-31 pass)

⚠️ `INHERITED — recorded here for continuity when this ledger was created on 2026-09-02; NOT re-verified.`

- `2605.09701` **LatentAlign** (full text, that pass): sigmoid-anneal of the prediction target from
  grounded/teacher-forced toward self-predicted. Produced backlog row **P-1**.
- `2602.03604` **EB-JEPA** (abstract-only): energy-based JEPA library, published 97 % Two-Rooms planning
  success. Produced backlog row **P-4** (use it as an external anti-collapse control).

---

## Entry 2026-09-02-01 — ⭐⭐⭐ Delta-JEPA: our P-1/P-2 defect is a documented field failure, with a fix

`PUBLISHED lib 2606.31232 · FULL TEXT READ 2026-09-02 · Zhang et al. (UCAS, IIE-CAS, HIT, IA-CAS) · arXiv 2026-06-30`
`⚠️ Was ALREADY BANKED and unread before this pass — see FS-5.`

### The field-level finding that reframes our gate

⭐⭐⭐ **`LeWorldModel` (`2603.19312`) — a strong, current end-to-end JEPA world model — is measured
ACTION-INSENSITIVE on the same diagnostic TanitAD built for v7.** Delta-JEPA perturbs the action and
measures the predictor's displacement from its own **zero-action** prediction:

> *"Delta-JEPA produces well-separated action-wise mean responses, with larger action magnitudes generally
> inducing larger predicted shifts. In contrast, **LeWM's action-wise means remain concentrated near the
> origin and substantially overlap**, indicating that changing the action does not induce a stable
> directional change in its prediction."*

⇒ **That is our `actdiv` / hold-action control, built independently by another group, and a well-regarded
published model fails it.** TanitAD is not uniquely broken. *(Same reframing as B13's geometry ceiling and
Waymo's long-horizon concession — three in one pass-series.)*

### The mechanism they diagnose (verbatim)

> *"when trained end-to-end with only latent prediction objectives, JEPA-based world models can easily
> collapse to trivial constant representations... the model achieves deceptively low prediction loss while
> destroying the representation structure needed for planning."*

⛔⛔ **The leak argument — this lands on INSTRUMENTS, not architecture:**

> *"its inverse dynamics module decodes actions from concatenated adjacent latent states [z_t, z_t+1].
> Because the forward predictor is itself conditioned on the executed action, end-to-end optimization can
> make the next-state representation z_t+1 absorb action-correlated cues that are easy for the inverse
> decoder to exploit, **without requiring the model to represent the actual transition between the two
> states**."*

⇒ **An action-decodability probe reading concatenated endpoints can pass on leaked endpoint cues.** This is
the same family as the programme's own leak rules (nav-echo bijection scoring 1.0000; label-provenance in
`situations.py`; the action echo). **Any TanitAD probe of this shape is confounded by construction.**

### The method — LDAD

Decode the executed action from the latent **displacement** `Dz_t = z_t+1 - z_t`. Two loss terms total:
`L = L_pred + L_action`. No pixel reconstruction, no distribution-matching regulariser, no frozen encoder,
no stop-gradient branches.

### Numbers (planning success %, 4 visual continuous-control envs, 3 seeds, 50 epochs, lr 5e-5)

| result | value |
|---|---|
| OGB-Cube vs strongest baseline | **+15.14 pp** |
| Two-Room vs PLDM | **+6.27 pp** |
| Push-T vs LeWM | **+4.54 pp** |
| ⭐ Ablation `Dz` vs `concat[z_t,z_t+1]` (Table 2) | **`Dz` wins on all four** — Push-T **+12.60**, Two-Room **+4.07** |
| ⭐ Control: lambda=0 (LDAD removed) | *"nearly collapses, yielding only a negligible planning success rate"* |
| Decoding-target ablation (Table 3) | raw action best; D-joint-position comparable; D-finger-position much worse |
| lambda sensitivity | poor at 0 and 0.1; stable over a broad range; best at **50.0** |

⭐ **The lambda=0 row is a control that must read the no-information value, and does.** That is why this
ablation is admissible under our own probe-panel rule rather than merely suggestive.

### Baselines and their anti-collapse mechanisms (useful map for backlog row 9)

| model | mechanism | Delta-JEPA's critique |
|---|---|---|
| **LeWM** `2603.19312` | SIGReg Gaussian latent regulariser | *"does not explicitly constrain the latent space to be sensitive to executed actions"* |
| **PLDM** | VICReg + inverse dynamics on `[z_t, z_t+1]` | multi-loss, hyperparameter-sensitive; **endpoint-cue leak** |
| **DINO-WM** | frozen DINOv2 features | *"limits task-specific adaptation of the representation"* |
| **Sub-JEPA** `2605.09241` | subspace Gaussian regularisation | strong on Reacher |

### ⚠️ Why this may NOT transfer to us — the risk that decides the line

⛔ **Our action channel IS realised motion (r 0.9988 with realised motion).** Decoding realised motion from
`Dz` may be **near-tautological** in any competent encoder — LDAD could sit near zero loss from
initialisation and buy nothing. **That is gate-problem P-2 branch (b) biting the fix rather than the
model**, and it must be tested before an arm is designed.

Secondary limits: four continuous-control benchmarks, **no driving**; encoder trained **from scratch**
where ours is pretrained/frozen; planning by CEM in a reward-free offline setting.

### Consequences for live rows

| row | movement |
|---|---|
| **P-2** (predictor does not use actions) | branch (c) *representation* gains a **named published mechanism** (endpoint leak), and F4's FiLM entanglement gives it a second. |
| **backlog row 13** (H-RANK-17, `z_t + m_t ~ z_t+1`) | ⭐ same latent-difference family — **now has an external primary with an ablation table.** |
| **backlog row 9** (VICReg placement / anti-collapse three-way) | the baseline map above is directly reusable; PLDM's endpoint decoder is a **design to avoid**, not to copy. |
| **instrument audit (new)** | ⛔ every action-decodability probe must be checked for the concat leak. |

`Next in this track: (1) the 0-GPU LDAD tautology check on banked v7 latents with constant-only and`
`shuffled-action controls; (2) LeWorldModel 2603.19312 and Sub-JEPA 2605.09241 are UNBANKED and unread —`
`both are current anti-collapse primaries and both are owed.`
