# TanitAD program report — 2026-08-15 ~21:00 CEST — POST-HANDOVER CONSOLIDATION

*Special edition: the 2026-08-05..15 cloud campaign (259 commits) merged into the local line,
plus the v6-on-Thor resumption in progress. Times Europe/Berlin. Every number carries its
evidence class; registry rows are cited by section, per the source-of-truth rule.*

## 0. The three sentences that matter

1. **The handover is merged, verified, and pushed** — `f3e0206` on `agent/arch-inf-20260803`
   unifies the campaign's 259 commits with the local line's banked partial work; the Google
   Drive folder now materialises the full state including the v1arch OOD videos.
2. **The programme has moved a generation: v6 (config E, 336.5 M, staged S-W→S-T→S-S→S-J) is
   the successor architecture**, trained to step 6,250 of 30k on the A40 before the pod stop;
   its resumption **on the Thor** is in progress tonight, gated only by a slow WAN.
3. **The compute fleet is now Thor + this dev box** — all four rented pods are gone
   (verified: connection refused/timeout on every one). Everything of value was banked to
   HF + GitHub before the stop, and the stop runbook's far-side verifications all held.

---

## 1. Fresh measurements (this evening, MEASURED unless marked)

| what | state |
|---|---|
| **Fleet** | Thor alive (`thor6`; **WiFi ssh — inbound ethernet ssh resets**, though the eth link itself is up at 1000 Mb/s and carries Thor's outbound traffic). pod4/pod5(`tanitad-new`)/pod/pod3 **all gone**. |
| **Thor readiness** | both venvs present (`tanitad-train`/`tanitad-edge`), GPU idle 0 %, 463 GB free; parity 256px corpus intact (2,376 train / 40 val); **stack checkout was stale at `4954544`** — sync to merged HEAD initiated; **conv2d-on-CUDA venv check pending** (box went briefly unreachable, see incidents) |
| **v6 resume assets** | `Sayood/tanitad-v6/v6F-SW-30k/`: ckpt.pt (step 6,250, md5 `01a0c5e8`) + config + logs pulled to Thor `~/experiments/v6F-SW-30k/` ✅; **w120 train cache (85 GB) + val (21.2 GB) downloading** — measured ~2–2.6 MB/s aggregate ⇒ honest ETA **~9–12 h** unless the line frees |
| **WAN diagnosis** | Thor did 23 MB/s on Aug 4; tonight both its interfaces measure 1.2–1.9 MB/s ⇒ the cap is the household line, not the NICs. Prime suspect: the dev-box Drive client uploading the freshly-merged tree. *(HYPOTHESIS on the suspect; the rates are MEASURED)* |
| **Dev-box suite** | merged tree: **2,799 passed / 12 failed** on Windows — **all 12 are environment artifacts, zero logic defects**: 11× cp1252 vs `⛔` in subprocess I/O, 1× POSIX-separator assertion. Campaign certified the same tree **2,804 / 0 on Linux** at stop. Fixed locally (conftest `PYTHONIOENCODING`/`PYTHONUTF8`, one path-normalised assertion, one honest platform skip for a BLAS ULP-identity test); full re-certification running. |
| **A40 baseline for the speed comparison** | **17.37 s/step throughout v6F S-W** (runbook §1, MEASURED by the campaign) — the number Thor will be measured against. |

## 2. What the cloud campaign delivered (phase map, integration status)

*Source: `HANDOVER_TO_LOCAL_2026-08-15.md` (pointer-first) + `MODEL_REGISTRY.md` rows; ✅ =
merged & verified in this consolidation, 🔶 = merged, follow-up owed.*

| phase | deliverable | status |
|---|---|---|
| **v5f 30k COMPLETE** (registry §1.8) | T0 final: **ade@2s 0.4011 / oracle 0.1975 / sel_gap 0.2036** on the 881-window w120 val — the selector still leaves ~half | ✅ |
| **v1arch COMPLETE** (§1.9) | **the programme's first COMPLETE four-family block** (`_complete: true`, OOD-val 290 clips): over-speed is a **prior not a tail** (+0.484 m/s bias, 71.95 % of windows ahead at 2 s); lateral tight (0.055 m); tactical κ 0.6033 honest; **strategic = a constant predictor; the seam FALSIFIED at this ckpt** | ✅ |
| **EVAL_DOCTRINE (T0/T1/T2 tiers)** | born from the measured **T1 action echo** — open-loop lateral skill that vanished closed-loop; every quotable claim now carries its tier stamp | ✅ binding |
| **v5.8f wedge ladder + Stage-A** (§1.13–1.14) | ⭐ **Stage-A post-training REPAIRS the action interface — ALL GATES PASS** (lateral gain 0.27→0.97, longitudinal sign →1.0, P6 subspace stays 3-dim, no-harm passed) at head-scale cost. v5.8f assembly: +0.08 ADE traded for **16× kinematic improvement** (accel MAE 0.515 vs 8.10) over a fan whose oracle nearly halves (0.1077); the deficit is SELECTION, sitting over a *feasible* fan for the first time | ✅ |
| **v6 design + S-W run** | `HIERARCHICAL_WM_REDESIGN` → config E: ViT-5 768×12 encoder (registers, RoPE), ModernCausal 1024×12 predictor, 256×640 cyl 120°, o5_k=60 (6 s rollout contract), 336.5 M params. S-W trained to **6,250/30k**, two gradient-spike episodes both self-recovered, stopped cleanly | ✅ · resumption in progress |
| **P-battery on v6** | ⛔ **third member of the echo family found**: P1 speed row R² 0.995 **collapses to −0.72 under the v0-shuffle control** (FiLM v0 channel) — `--speed-echo-control` now mandatory. ⭐ encoded-latent curve **moving** (−2.30 → −0.74, steps 2.5k→5k). P3/P6 verdict pending → armed at the 10k milestone | ✅ instrument · 🔶 10k watcher must be re-armed on Thor |
| **PH0 (VLM+SAM3) at scale** | 600 w120-val clips fully labelled; **201/201 runnable Alpamayo clips at 120° processed & pushed**; B3 VLM grounding demoted to diagnostic (2/23); ⛔ 4,472 clips lack w120 caches (chunk-index build scoped, not started) | ✅ |
| **PH1 fusion** | jurisdiction-not-averaging fuser (`ph1_fuse.py`, 12 tests): 600 val clips fused — 175 corroborations / 41 conflicts / 56 with the Alpamayo layer; ego layer labels-only | ✅ · 🔶 aug120 batches not yet fused |
| **G1 sign-OCR gate** | **CLOSED: 0/31 verifiable** at pipeline fidelity, and **~⅔ of SAM3's best "traffic sign" crops contained no sign** (scores to 0.94) — sign class needs a threshold study; `pending_g1_gate` enforced in every fused record | ✅ closed honestly |
| **Alpamayo-2 augmentation set** | `records.parquet` **23,644 rows = 4,729 clips × 5 tasks** + quantisation arms, on HF | ✅ |
| **Orbis 2 analysis** | nearest published analogue; **we are 139× under it in hours-per-M-param** — data, not parameters, is the binding constraint (`V6_DATA_REQUIREMENT.md`) | ✅ frames the roadmap |
| **Ops with teeth** | HF silent-push-failure class fixed (far-side verify every cycle, public+gated-manual policy); polling-monitor self-match trap (3× measured) added to CLAUDE.md; ckpt-layout compat (`ckpt_compat`) | ✅ |

## 3. What the local line contributed to the merge

Banked immediately before the merge so the lineages stay separable (commit before `f3e0206`):
the stranded 2026-08-04 partials — **target-speed** (VTARGET leak-guarded label stream),
**λ-findability** (E-EXP-2 code), **budget-composition** (prereg pin + per-arm raw results —
it banked incrementally, which is why anything survived its OAuth death), two refc scripts,
and the v1arch video metadata. Plus, this evening: the Windows suite-parity fixes above.

The local line's late results already inside the campaign's foundations (chronicled 08-03/04):
the four-instrument convergence on **selection over generation**, the **off-fan ceiling**
(E-EXP-1: 53.7–56.5 % of the oracle gap, ~10× along-path vs cross-path), the **anchor
reachability filter** (2.78× decode cut, bit-exact 881/881), the **nav known-bit** and
**distance-keeping** instruments, and the echo-test doctrine the campaign generalised.

## 4. Program position vs the Master Plan — the four edges

| edge | position | grade |
|---|---|---|
| **Planning** | The story converged: fans are good, **selection/action-interface was the defect**, and Stage-A **repaired** the action interface at head-scale cost (all gates pass). v6's staged trainer bakes the repair's lessons in (X3/X5 isolation, o1 control probes in-loop). Next proof point: v6 S-W → S-T stage gate. | **MEASURED, strong** |
| **Efficiency** | Sub-300M thesis holds into v6 (336.5 M vs the 350 M budget, PI-raised); Thor endpoints: 60.3 ms tick p50 (6.17×), REF-C 1.03–1.25× A40 training. Tonight adds the v6 Thor-vs-A40 number. | **MEASURED** |
| **Safety / self-knowledge** | The echo family (nav-echo → action echo → **v0-FiLM speed echo**) is now a *doctrine with controls*, not anecdotes; T-tier stamps binding; G1 gate closed rather than shipped. | **MEASURED, doctrine-grade** |
| **Data efficiency** | Orbis 2 comparison quantifies the gap (**139× under** on task-matched hours/M-param) — the binding constraint is data. Pipeline exists end-to-end (PH0→PH1); the **4,472-clip build is the single biggest lever** and is scoped. | **MEASURED gap, scoped lever** |

## 5. Next steps (ordered)

1. **v6 S-W resume on Thor** — pull completing (WAN-bound), then: strict resume from step
   6,250 with the banked exact flags (paths swapped to `~/data`, `~/experiments`), inside
   `~/venvs/tanitad-train`, logs to `/tmp`; **measure marginal s/step over ≥3 logged points
   and report vs the A40's 17.37 s/step**. Auto-launch watcher armed on pull completion.
2. **Re-arm the safety loops** (runbook §3.6): `pbattery_watcher.py TARGET_STEP=10000`
   (first P3/P6 verdict, always with `--speed-echo-control`) + `hf_push_loop.py` with a
   long cycle (the uplink is precious).
3. **Fuse the aug120 batches** (`ph1_fuse.py` loop — labels exist, one run).
4. **The 4,472-clip chunk-index build** — the biggest data job; also unlocks G1 native-res.
5. Windows-parity commit + this report; then the deferred queue (BEV probe port, SAM3
   threshold study, E-ENC ViT-5-form, DataLoader measurement — 42.8 % util says headroom).

## 6. Decisions required from Sayed (with defaults)

1. **Thor restart?** — *Default: only if it stays unreachable past ~22:00 tonight.* It went
   dark mid-download (saturation-shaped, not crash-shaped); everything running on it is
   detached and resumable, so a restart is safe if needed.
2. **Pause Google Drive sync on the dev box overnight?** — *Default: yes, recommended.* It
   is the prime suspect for Thor's 2 MB/s (vs 23 MB/s on Aug 4); pausing likely turns the
   9–12 h pull into ~1–2 h.
3. **Keys hygiene** — the campaign's own README flags rotating the pod JupyterLab token
   **and the four `Keys.txt` credentials** (exposed during the campaign's transfers; also
   flagged 08-03 locally). Pods are dead, so the pod token is moot; *default: rotate the
   `Keys.txt` four at your convenience; the HF token stays needed for the pull.*
4. **PR #2** (the campaign's draft PR of the handoff branch) — *default: close it as
   superseded; its content is fully merged into `agent/arch-inf-20260803` by `f3e0206`.*

## 7. Incidents, honestly

- **Thor went unreachable at ~20:50** on both addresses, minutes after I started its two
  concurrent downloads on a ~2 MB/s line — saturation-shaped (timeout, not reset; the box
  was healthy 90 s prior). Detached processes survive either way; a reachability monitor
  is polling. *If it does not return, the corpus pull resumes from where it stopped.*
- **Thor's first git sync silently failed** — a backgrounded ssh whose output was swallowed
  reported exit 0 while the checkout still sat at `4954544`. Caught by re-verification
  (the perishable-drift rule: a real `import`, never `git log`). Re-sync was in flight when
  the box went quiet; it will be re-verified before any launch.
- **12 Windows test failures on the merged tree** — all environment (cp1252 × `⛔`, POSIX
  separators, BLAS ULP identity); fixed/skipped honestly; zero logic defects. The lesson
  is now in `conftest.py` where it executes, not in prose.
- **The ethernet mystery, stated as it is**: link up at 1000 Mb/s and carrying Thor's
  outbound traffic, while *inbound* ssh to the .194 address resets. Unexplained; WiFi
  address works; not blocking anything tonight.

*Suites at close: campaign Linux certification 2,804/0 (stop-state); dev box after parity
fixes: full re-run in progress, targeted files 141 passed / 2 honest skips. This report:
committed + pushed alongside the Windows-parity fixes.*
