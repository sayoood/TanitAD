# PRE-REGISTRATION — `DIR_YAW_RAD` 0.15 → 0.10, 2026-08-18 pass

**Written 2026-08-18, BEFORE any envelope or bootstrap number in this pass was computed.**
Operating-standard rule 5: both outcomes committed in advance. Estimators and tiers declared here.

---

## 0. What this pass is, given three prior passes exist

The brief's premise — *"nobody has measured what 0.10 changes"* — is **stale**. MEASURED (grep +
read of the deliverables, this clone, tip `65211ecc`):

| pass | where | what it answered |
|---|---|---|
| 2026-08-15 | `…/Benchmarks & Eval/…/2026-08-15-dir-yaw-gate-reread/` | verdict census over 16 banked panels; 1 recomputable; one verdict moves (already-retracted 39-clip A2 comparison) |
| 2026-08-16 | `…/Architecture & Inference/…/2026-08-16-dir-yaw-reread/` | exact envelope for the two `-lf19` paper panels; "(v1, weak)" removed from the paper |
| 2026-08-17 | `…/Architecture & Inference/…/2026-08-17-diryaw-reread/` | provenance (author-chosen, `35adfab`); the gate ALIASES the training-label constant (`hierarchy.py:169` → `refb_labels.YAW_TURN_RAD`, `refb_labels.py:57`); the asymmetric-sweep ruling; band mass 3.97 % CI95 [1.93, 6.82] with the mandated paired episode-cluster bootstrap; **verdict: do NOT change the threshold** |

**This pass therefore does not re-litigate.** It (a) verifies rather than inherits the load-bearing
numbers, (b) closes the ONE measurable item all three passes left open — **`kappa_turn_subset` is
gate-dependent and has never been swept** (flagged 2026-08-15 §2b, 2026-08-16 §5.2, 2026-08-17
§5 work-item 2, and `MANEUVER_LABEL_MISMATCH.md:280` lists it as still OPEN) — and (c) records
why the full GPU re-run remains non-executable on this box.

## 1. Substrate finding (MEASURED, recorded before computing)

The full re-run (per-window `man_pred` + `traj_net_yaw` under the tip instrument, which now banks
them — `hierarchy.py` `PER_WINDOW_KEYS:790`) is **credential-blocked, not compute-blocked**:

* deployed-arm ckpt **IS local and verified**: `C:/Users/Admin/tanitad-data/eval/v1_speedjerk_ckpt.pt`,
  md5 `b5f07d9e3dd2ca643949bc86832e6585` = `flagship4b-speedjerk-30k_ckpt.pt` in
  `_pod_backup/pod2-2026-08-03/ckpts/BACKUP_LOG.txt` (MEASURED, this session);
* RTX 4060 CUDA works (real conv2d probe, MEASURED);
* canonical val40 FRAMES are not local: local `physicalai-val-bb543bdf7836` cache shares **0 / 40**
  episode ids with the canonical set (MEASURED via `ep40_clip_map.json` join);
  `val40-poses-20260818` is a poses-only view (frames_u8 `[199,9,1,1]` stubs, MEASURED);
* the frames exist at
  `hf:datasets/Sayood/tanitad-physicalai-w120-256x640cyl/epcache-256px-phase0/physicalai-val-0c5f7dac3b11`
  (≈ 40 × 117 MB ≈ 4.7 GB) but anonymous access is **HTTP 401** (MEASURED) and no HF token exists
  on this box (Keys.txt is on the dead G:; probed 4 locations + env + worktrees, all absent);
* Thor holds the cache but is untouchable (live 30k run).

⇒ The measured part of this pass is the **exact admissible-set envelope** over banked panel
summaries (0 GPU, deterministic), plus an **independent recomputation** of the band-mass bootstrap
from the banked window dumps.

## 2. At-risk published claims, and the change-thresholds — declared before computing

The published κ-verdict ladder is `four_families.KAPPA_VERDICT_LADDER`: **< 0.1 DECORATIVE,
< 0.4 WEAK, ≥ 0.4 SUBSTANTIAL**. The old κ ≥ 0.2 predicate was retired 2026-08-16 (`5daa3d7`);
0.2 is NOT a published boundary today.

| # | claim | where published | at-risk reading | change-threshold |
|---|---|---|---|---|
| 1 | `kappa_turn_subset = 0.2005` (v1arch, OOD-val q90, 880 win / 40 ep) with the clause *"on the windows where a direction decision actually exists, coherence is marginal"* | `RETRACTION_LOG.md:3276-3278`; `…/2026-08-06-v1-defect-triage/results/GATE_RERUN_RESULT.md:39`, `TEMPORAL_STABILITY_RESULT.md:59`, `V1_DEFECT_TRIAGE.md:211`; `…/2026-08-16-verdict-stable-kappa/VERDICT_STABLE_KAPPA.md:250-251` | "marginal" = WEAK band on the published ladder | the κ_ts@0.10 **envelope** (anchored by this panel's banked κ@0.10 = 0.5715) crossing **0.1** (admits DECORATIVE) or **0.4** (admits SUBSTANTIAL) at admissible m |
| 2 | `kappa_turn_subset = 0.2074` (flagship-30k, canonical val, 881/40 — the deployed v1 FINAL's panel) | panel JSON only (`stack/experiments/pod-rescue-20260802/pod3/root/taniteval/results/hier_flagship-30k.json`); **not quoted in any prose doc** (MEASURED: grep over Project Steering, Paper, Research Hub) | nothing published is at risk; the number gates the v6F TACTICAL family's history | same ladder boundaries, reported as envelope-vs-m with the measured GT band mass as the plausibility reference |
| 3 | `MODEL_REGISTRY.md` | — | **no registry row quotes any `turn_subset`** (MEASURED: grep) | pre-committed: **no registry edit in any outcome** |
| 4 | the 2026-08-17 band mass 3.97 % CI95 [1.93, 6.82], m = 35 (canonical val, 881/40) | `…/2026-08-17-diryaw-reread/` §3.2 | INHERITED until re-derived | my independent recomputation must reproduce it within rounding; **if it does not, STOP and publish the discrepancy, not the envelope** |

## 3. Committed outcomes

* **(a)** If the κ_ts@0.10 envelope for claim-1 stays inside **[0.1, 0.4)** over the admissible m
  set → the "marginal / WEAK on turn-active windows" reading **survives** 0.10; no published verdict
  moves; work-item 2 is closed by measurement-bound; the backlog item stays CLOSED with the
  2026-08-17 do-not-change ruling intact.
* **(b)** If the envelope admits crossing either boundary at admissible m → the clause is
  **not established at 0.10**; it must be quoted with its gate everywhere it appears (edit list in
  the report; RETRACTION_LOG entry proposed, not self-applied to closed entries); the do-not-change
  ruling STILL stands (a wider envelope is not evidence for moving the constant — the 2026-08-17
  asymmetry argument is unaffected).
* In BOTH outcomes: ⛔ no constant changes in code; no `MODEL_REGISTRY.md` edit; the stale
  `V6F_PLANNER_DESIGN.md:722/:827` "re-read still owed / re-read at 0.10 not 0.15" lines get a
  correction citing the four passes (bookkeeping, outcome-independent — those lines contradict the
  2026-08-17 ruling and the fact the re-read exists).

## 4. Method + estimator declarations

* **Envelope**: extend the verified 2026-08-16 machinery. At 0.15, enumerate every 3×3 confusion
  matrix consistent with (man marginal, traj marginal, agreement T, **n_turn_active** — pins the
  (S,S) cell exactly, **κ@0.15** and **κ_ts@0.15** at their published 4-dp rounding). For the gate
  move, enumerate crossing allocations (traj-side S→L/S→R only — monotone, per the 08-16 proof)
  and, for claim 1, keep only allocations reproducing the panel's **banked κ@0.10 = 0.5715**.
  Report min/max κ_ts@0.10 per m and the admissible m set itself.
  **Self-test**: brute-force generator over random synthetic per-window data; the true κ_ts@0.10
  must lie inside the envelope computed from the summary alone, and the true matrix must be in the
  admissible set. ≥ 200 random cases before any real number is quoted.
* **Band mass**: `taniteval/ci.py` `episode_cluster_bootstrap` / `paired_episode_cluster_bootstrap`,
  B = 2000, seed 0, resampling unit = episode, on `windows_flagship-30k.pt` (`head_deg` → rad;
  identity `head_deg·π/180 = |gt_net|` re-verified against the panel's own `gt_dir` integer counts
  before use). Duplicate dumps excluded per `taniteval/results/dump_exclusions.json` (C126).
* **Tier**: every κ here is an open-loop, model-internal coherence **diagnostic** (T0-family) —
  never "driving performance" (EVAL_DOCTRINE).
* Envelope bounds are **deterministic admissible-set bounds, not confidence intervals**, and are
  labelled as such wherever quoted.

Evidence classes: substrate facts above MEASURED (this session); the three passes' findings
INHERITED until §2-row-4 verification passes; envelope outputs MEASURED (deterministic function of
banked artifacts); any statement about which m is realistic ESTIMATED (band masses measured on
other corpora / GT side).
