# TanitAD — Program Report · 2026-07-15 18:25 UTC

**Headline:** The **speed/scale reset is complete and gated.** Root cause found and fixed — none of the arms received the **current ego‑speed as an input**, so they were trying to infer absolute speed from vision, which a frozen encoder cannot do. Feeding `v0` as a 3rd action channel **halved REF‑A's driving error (3.73 → 2.14 m held‑out)**. REF‑A 4‑brain is **DONE at 30k** but has **plateaued at the frozen‑DINO ceiling** (2.14 m, still above the constant‑velocity bar). The **flagship (trained encoder) is at 21%** and already makes **sharper per‑step predictions than REF‑A's finished model**, on a rising curve. **Neither beats CV yet** (0.83 m — brutally strong on highway); the fair verdict is the **flagship at 30k (~4 days)**.

> Scope note: the hub `PROJECT_STATE.md` (07‑11, W31) still describes the pre‑reset era. This report covers the **speed/scale reset** (all three arms restarted from scratch 2026‑07‑14) and its decision‑grade gates — ahead of the hub. The reset's code (4‑brain REF‑A trainer, speed‑aware eval harnesses, overlay renderer) currently lives **on the pods only** (`/workspace/tmp`) and is **not yet in the repo** — durability item, §7.

---

## 1. Training state — the 3‑arm bake‑off (fresh, 18:25Z)

| Arm | Pod | Step | % | Pace | In‑training signal | GPU |
|---|---|---|---|---|---|---|
| **Flagship** 4‑brain (trained ViT) | pod2 | **6,350 / 30k** | 21% | 13.7 s/step | op fwd‑ADE **0.114 m**, man_acc 0.63, erank 23.3, no collapse; tac/str fwd 0.40/0.54 | 87% |
| **REF‑A** 4‑brain (frozen DINO) | pod3 | **30,000 DONE** | 100% | — | final in‑train fwd‑ADE 0.71 m; process idle | 0% (free) |
| **REF‑B** from‑scratch ViT (BC) | pod1 | **3,600 / 30k** | 12% | 23.4 s/step (data‑bound) | man_acc 0.64, route_acc 1.0, wp 0.64, ood_score 1.85 (frozen) | data‑gap |

All three carry the full reset recipe: **speed‑input + jerk + aux‑accel**; flagship & REF‑A also carry the **full 4 brains** (strategic route + tactical maneuver/goal + tactical‑predictor + intent‑conditioned operative). REF‑A's frozen‑DINO adapter vs the flagship's trained ViT is now the **only model‑axis difference** — the clean encoder isolation the bake‑off needs.

**ETAs:** flagship 15k re‑gate ~1.4 d · 30k ~3.75 d · REF‑B 30k ~7 d.

---

## 2. Decision‑grade gates (held‑out, 40 eps, 8 splits, vs CV = 0.83 m)

```
                     overall   de@1s  de@2s  |  1s by stratum: gentle/sharp/straight
REF-A 4-brain  30k   2.14 m    1.77   3.27   |   1.85 / 1.41 / 1.13   plateaued, DONE
Flagship        5k   2.34 m    1.43   4.60   |   0.63 / 0.98 / 1.01   21%, rising
constant-vel    --   0.83 m    0.47   1.71   |   0.38 / 0.58 / 0.15
```

- **Neither clears CV.** Highway constant‑velocity is nearly unbeatable on straights (CV@1s 0.15 m).
- **Encoder signal is real:** at the 1‑second horizon the flagship at **5k already beats REF‑A at 30k on every stratum** — sharper near‑term at ⅐ the training.
- **Flagship's aggregate is dragged by its 2 s endpoint** (de@2s 4.60): long rollout not yet stable at 5k, and it trains rollout‑k=4 while the gate rolls 20 (REF‑A trained k=12). Maturity + possibly recipe.
- **REF‑A plateaued** (14k 2.05 → 30k 2.14): the frozen‑DINO ceiling, final.

---

## 3. The speed/scale reset — what happened and why

**The bug (conceptual, not a crash):** actions are *derivatives* `[steering, acceleration]`. To produce an absolute metric displacement (`disp ≈ v0·dt + ½·accel·dt²`) the model must know the **current speed v0**. None of the arms fed it — they inferred speed from the image, and a **frozen DINO encoder can't** (speed‑probe R² ceilinged at 0.61). This is why REF‑A failed the driving gate at 3.73 m.

**The fix:** append `v0 = pose_last[:,3] / 10` (last observed frame — leakage‑safe) as a **3rd action channel**. Validated in isolation: REF‑A operative went **3.73 → 0.83 m** in‑training, speed‑decodability **0.61 → 0.965 R²**.

**Actions taken (2026‑07‑14 → 15):**
- All three arms **restarted from scratch** with speed + jerk (0.02) + aux‑accel; valid checkpoints archived.
- **REF‑A given the full 4 brains by hand** (self‑contained port of the flagship joint‑loss into its frozen‑DINO trainer + a genuine 25 s nav‑label dataset), CPU‑smoke‑validated that all four brain losses carry gradient.
- Two **speed‑aware eval harnesses** built + validated (REF‑A 4‑brain, flagship), since the old evals couldn't load a speed/4‑brain checkpoint.
- Both arms gated at their checkpoints (§2); **BEV trajectory overlays** rendered for flagship + REF‑A on 8 shared held‑out scenes (published as a lightweight vector gallery + ASCII plots).

---

## 4. Program position — Master Plan & the four edges

**Phase 0** (foundation & first edge proofs), ~day **11/42**; final eval P7 2026‑10‑05. The core Phase‑0 question — *does the 4‑brain trained‑encoder model beat the flat/frozen references on identical data?* — is **in progress, trending toward the trained encoder** but **not yet proven vs CV**.

| Edge | Grade | Evidence |
|---|---|---|
| **Planning / hierarchy** (4‑brain) | 🔶 in progress | Operative strong (flagship in‑train 0.11 m); tactical/strategic maturing (0.40/0.54). REF‑A now has the same 4 brains → encoder is the only diff. Verdict at flagship@30k. |
| **Efficiency** (CNCE, 261 M) | ✅ on track | 261 M vs 15–32 B competitors; ~13.7 s/step A40, no regression this cycle. |
| **Safety / self‑knowledge** (OOD) | 🔶 partial | REF‑B carries `ood_score` (1.85, baseline frozen). Real clean‑vs‑degraded **D8 separation still blocked** (no degraded cache on pods). |
| **Data efficiency** (H15 imagination) | 🔶 partial | Flagship H15 loss active (stochastic mask gate); not yet ablated for a driving‑gain number. |

---

## 5. Agent & hub updates (knowledge transfer)

**Repo advanced minimally since the last report** — the reset work was executed on the pods and is not yet committed.

| Source | Output | State | Note |
|---|---|---|---|
| Reset (this session) | 4‑brain REF‑A trainer, 2 speed‑aware eval harnesses, overlay renderer, gate JSONs | 🔶 on pods only | **Sync to repo** (§7 durability) |
| Hub research | `2026-07-16-benchmark-ecosystem-and-metric-suite.md`, backlog3 synthetic‑corpora first‑pass | 🔶 queued | Uncommitted; needs orchestrator sweep |
| Implementation intake | `cosmos-robustness`, `lal-v2-anticipation`, `eval-metric-suite` | 🔶 queued for triage | In `Implementation/incoming/` |
| Hub state files | `PROJECT_STATE.md`, `BACKLOG.md`, `Research/STATE.md` modified | 🔶 uncommitted | Pre‑reset era — needs refresh |

Only commit since 07‑14 13:22 is the last report itself (`50eb2d6`). Branch: `agent/phase0-eval-harness`.

---

## 6. Next steps (ordered)

1. **Flagship 15k re‑gate** (~1.4 d) — does `de@2s` tighten (under‑training) or stay ~4 (rollout‑k=4 recipe)? This answers the open recipe question for free.
2. **Flagship → 30k** (~3.75 d) — the **fair verdict**: can the trained encoder push under CV where the frozen one plateaued at 2.1 m?
3. **REF‑B → ~5k** (~1.5 d) — first meaningful checkpoint → overlay + held‑out gate → fold into the comparison gallery (deferred until it's a fair showing).
4. **Sync the reset code to the repo** — the 4‑brain trainer + eval harnesses + renderer are pod‑only; commit them for durability.
5. **Re‑arm the 3‑arm monitor** (stopped at the session boundary).
6. **Refresh hub `PROJECT_STATE.md`** to the reset reality (orchestrator).

---

## 7. Decisions required from Sayed (with defaults)

| # | Decision | Default if quiet |
|---|---|---|
| A | **Flagship rollout‑k** — the 2 s gap may be k=4 (REF‑A used k=12) | Leave k=4, re‑gate at 15k; spin k=12 probe only if de@2s stays ~4 |
| B | **Free pod3** (REF‑A done, GPU idle) | Leave idle; or run the k=12 flagship probe there (needs image cache staged) |
| C | **REF‑B pace** (~7 d to 30k, data‑bound) | Leave as the reference arm |
| D | **Commit reset code to repo** (pod‑only durability risk) | **Sync it** — I'll stage the key scripts on the agent branch unless you object |

---

## 8. Incidents (honest)

- **The reset itself** was a major mid‑course correction — a real design gap (no speed input) that capped every arm, not a tuning miss. Caught by probing why REF‑A's error localized to speed/scale.
- **Self‑kill trap ×2** — `pkill -f <pattern>` matched the ssh command's own cmdline when the literal run‑name appeared in an echo/ls string; killed my own session (exit 255) but the target died too. Now use `[x]`‑bracket patterns + no literal run‑names near pkill.
- **REF‑B weight relay corrupted** (718 MB truncated over the ssh pipe) → REF‑B overlay deferred (it's only 12% trained anyway, not yet a fair showing).
- **Image delivery to phone** — file attachments, the artifact link, and the inline widget all failed to render in the user's mobile app; root cause was a **680 KB image‑heavy artifact** too large for mobile. Fixed with a **24 KB lightweight vector gallery** (same URL) + **ASCII trajectory plots** that render as text regardless.
- **Monitor stopped** at the session boundary (no completion marker) — re‑arm pending (§6.5).

---
*Fresh pod tails + gate JSONs read 2026‑07‑15 18:15–18:25 UTC. Grounded gate numbers are held‑out 8‑split means. Next report on the D‑025 cadence or on the flagship 15k re‑gate, whichever lands first.*
