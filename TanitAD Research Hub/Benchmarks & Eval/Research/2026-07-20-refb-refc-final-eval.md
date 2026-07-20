# REF-B v2 (30k FINAL) and REF-C-XL (28k, provisional) — full eval + videos

**Date:** 2026-07-20 · **Pod:** `tanitad-eval` (A40 46 GB) · **Corpus:** canonical val
`physicalai-val-0c5f7dac3b11`, 40 episodes, **n = 881 windows**

**Headline.** REF-C-XL beats REF-B v2 decisively and lands within **0.018 m** of flagship v1
on ADE@2s — while **beating flagship outright in the high-speed stratum** (0.330 vs 0.551 m).
Both arms clear the constant-velocity floor at every horizon.

> **PROVISIONAL — READ THIS FIRST.** REF-C-XL is evaluated at **step 28000 of 30000**. It was
> *still training* on `tanitad-pod3` when the checkpoint was pulled. This is **not** a 30k
> number and must not be quoted as one. REF-B v2 **is** final (step 29999).
> See [Provisional status](#provisional-status-of-the-ref-c-number) for why the gap is expected
> to be small, and the exact refresh command.

---

## 1. Checkpoints under test

| Arm | Path (eval pod) | Step | Load | Notes |
|---|---|---|---|---|
| REF-B v2 | `/root/models/refb-v2-30k/ckpt.pt` | **29999 (FINAL)** | strict ✔ | `--arch-v2 --refbpatch`; needs `TANITEVAL_STACK_OVERRIDE=/root/models/assess-20260719/stack-v2b` |
| REF-C-XL | `/root/models/refc-xl-live/ckpt.pt` | **28000 (PROVISIONAL)** | strict ✔ | ~252 M anchored-diffusion; `refc1=False` → horizons **are** the 0.5/1/1.5/2 s time waypoints |

Registered as `refc-xl-live` in `taniteval/registry.py` (the pre-existing `refc-xl` entry is the
stale **step-16000** snapshot and should not be used).

### Checkpoint transfer (pod3 → eval)

The direct path was **broken** at session start and had to be re-established:

- The eval pod's `~/.ssh/config` had pod3 on a **stale port** (22087; the live port is **22079** —
  RunPod re-maps on restart).
- pod3's `authorized_keys` does **not** contain the eval pod's public key, and pod3 holds **no**
  outbound private key, so neither direction authenticates on its own. Only TCP 22 is exposed
  (`RUNPOD_TCP_PORT_22=22079`), so no HTTP side-channel either.
- **Resolution: SSH agent forwarding.** `ssh -A` from the dev box to the eval pod, which then uses
  the *forwarded* agent (holding the dev box key, which pod3 *does* authorise) to scp straight from
  pod3. Authentication hops through the dev box; **the 3 GB of payload flows pod3 → eval directly.**
  No key material was copied and no host config was modified.

Measured **18.2 MB/s** (3 024 021 445 B in 2 m 45 s) — matching the expected direct-transfer rate
and ~100× the dev-box relay. Integrity: `md5 = 531fd19cc13f411cd0bf2ef49c72ec26`, **identical** to
the source, and the source mtime was unchanged across the copy (07:22:35 UTC, next save not due
until ~08:15) → **untorn read**.

---

## 2. Protocol

Identical harness, windows, GT and baselines as every other arm, so the rows are directly
comparable:

- window 8, stride 8, horizons **WP_STEPS 5/10/15/20** = 0.5/1/1.5/2 s, ego frame of the last
  window pose, `nav_cmd=None` (the `follow` command).
- 8 splits, `val_frac` 0.2; CI95 across splits. Claim strength: **open-loop / weak**
  (arXiv:2605.00066) — these are *not* closed-loop numbers.
- **Pairing verified**, not assumed: `torch.allclose(gt)` holds across `flagship-30k`,
  `refb-v2-30k` and `refc-xl-live` (881 × 4 × 2), so every A/B below is strictly paired.

Both arms are **direct-trajectory-head** arms — REF-B via tactical waypoint heads, REF-C via its
anchored-diffusion decoder. Neither has a grounded operative rollout (`step_readout=None`); the
decode *mechanism* differs from the world-model arms, the *measurement* does not.

---

## 3. Full metric table (heldout mean ± CI95, metres)

| Metric | CV floor | **REF-B v2** (30k final) | **REF-C-XL** (28k prov.) | flagship v1 (30k) |
|---|---|---|---|---|
| ADE@0.5 s | 0.1292 | 0.1033 ± 0.0120 | **0.0720 ± 0.0069** | 0.0762 ± 0.0046 |
| ADE@1 s | 0.2972 | 0.2173 ± 0.0260 | 0.1624 ± 0.0185 | **0.1584 ± 0.0149** |
| ADE@1.5 s | 0.5304 | 0.3793 ± 0.0450 | 0.2952 ± 0.0350 | **0.2883 ± 0.0227** |
| **ADE@2 s** | **0.8248** | **0.5921 ± 0.0685** | **0.4703 ± 0.0574** | **0.4522 ± 0.0312** |
| FDE@2 s | 1.7081 | 1.2305 ± 0.1401 | 0.9955 ± 0.1247 | **0.9437 ± 0.0630** |
| RMSE | 1.5407 | 1.0599 ± 0.1168 | 0.8234 ± 0.0868 | **0.6865 ± 0.0376** |
| miss@2 m | 0.3131 | 0.2025 ± 0.0356 | 0.1544 ± 0.0419 | **0.0602 ± 0.0121** |
| TMS open-loop | 0.9999 | 0.3044 ± 0.0195 | 0.1996 ± 0.0143 | **0.1070 ± 0.0229** |

Both arms **beat the CV floor at every horizon**. REF-C-XL edges flagship at the shortest horizon
(ADE@0.5 s, 0.0720 vs 0.0762) but the split CIs overlap — treat that one as a tie, not a win.
flagship keeps the long horizons and, by a wide margin, the tail metrics (miss@2 m, RMSE).

### By speed (ADE@2 s, model / CV)

| Stratum | n | REF-B v2 | REF-C-XL | flagship v1 | CV |
|---|---|---|---|---|---|
| low | 294 | 0.8171 | 0.6075 | **0.3594** | 0.9322 |
| med | 293 | 0.5244 | 0.4987 | **0.3704** | 0.9345 |
| **high** | 294 | 0.4321 | **0.3301** | 0.5513 | 0.6468 |

**This is the load-bearing stratum result.** In the high-speed bucket REF-C-XL is the *best arm on
the board* — 40 % better than flagship (0.330 vs 0.551) — and it is the **only** stratum where
flagship is beaten. It is also the only stratum where flagship fails to beat its own low/med
numbers, consistent with the known flagship longitudinal weakness at speed.

### By curvature (ADE@2 s)

| Stratum | n | REF-B v2 | REF-C-XL | flagship v1 | CV |
|---|---|---|---|---|---|
| straight | 634 | 0.4782 | 0.3931 | 0.3931 | 0.4393 |
| gentle | 125 | 0.8649 | 0.6672 | **0.5158** | 1.3566 |
| sharp | 122 | 0.8984 | 0.7305 | **0.5128** | 2.3764 |

**The straight-stratum tie is real, and it is a coincidence — verified, not assumed.** REF-C and
flagship agree to 4 dp (0.3931) on n=634. Recomputed independently from the raw window tensors:
**0.39314601** (flagship) vs **0.39314798** (REF-C) — they diverge at the 6th decimal, and
`allclose(pred_flagship, pred_refc)` is `False` (pred sums 58267.8 vs 57568.7). Not a
copy/alias bug.

The *distributions* behind that tie are very different: on straights REF-C's **median** error is
**0.219 m** vs flagship's **0.347 m**. REF-C is markedly better on the typical straight window and
pays it all back in a heavier tail.

---

## 4. Head-to-head (paired, 881 windows, 10k-resample bootstrap)

Positive Δ ⇒ the **B** arm is better. "Significant" = CI95 excludes 0.

| A vs B | ADE A | ADE B | B win-rate | mean Δ (m) | CI95 | Verdict |
|---|---|---|---|---|---|---|
| **REF-B v2 vs REF-C-XL** | 0.5913 | 0.4788 | **59.6 %** | +0.1125 | [0.0814, 0.1445] | **REF-C-XL** ✔ sig |
| **REF-C-XL vs flagship v1** | 0.4788 | 0.4271 | 45.4 % | +0.0516 | [0.0186, 0.0859] | flagship ✔ sig |
| REF-B v2 vs flagship v1 | 0.5913 | 0.4271 | 51.8 % | +0.1655 | [0.1253, 0.2057] | flagship ✔ sig |
| REF-C-XL 16k vs 28k | 0.6050 | 0.4788 | 70+ % (73.8) | +0.1262 | [0.1047, 0.1475] | 28k ✔ sig |
| REF-B v2 20k vs 30k | 0.6430 | 0.5913 | 70.5 % | +0.0517 | [0.0262, 0.0768] | 30k ✔ sig |

*(A/B means are plain per-window means over all 881 windows and so differ slightly from the
split-heldout means in §3 — e.g. flagship 0.427 here vs 0.452 there. Both are correct; §3 is the
leaderboard number.)*

### REF-B v2 vs REF-C-XL, by curvature

| Stratum | REF-C win-rate | mean Δ (m) | n |
|---|---|---|---|
| straight | 58.2 % | +0.0851 | 634 |
| gentle | 62.4 % | +0.1977 | 125 |
| sharp | 63.9 % | +0.1679 | 122 |

**REF-C-XL beats REF-B v2 in every stratum** — this is not a tail artefact, it is a uniform win.

### REF-C-XL vs flagship v1, by curvature — where the gap actually is

| Stratum | flagship win-rate | mean Δ (m) | n |
|---|---|---|---|
| straight | 39.6 % | **+0.0000** | 634 |
| gentle | 57.6 % | +0.1515 | 125 |
| sharp | 63.1 % | +0.2178 | 122 |

**REF-C's entire deficit vs flagship is in curves.** On straights (634/881 = 72 % of windows) the
mean delta is zero to 4 dp (−0.000002 m — the §3 coincidence) and REF-C wins **60.4 %** of
individual windows.

Note the shape of the overall flagship win: flagship takes the **mean** (Δ +0.0516, significant)
while winning only **45.4 %** of windows — REF-C wins **54.6 %**. flagship's advantage is
**tail-driven**, which the miss@2 m column states directly (0.060 vs 0.154). REF-C is better on the
typical window; flagship is better at not blowing up.

---

## 5. Videos — the deliverable

Rendered to the standing TanitEval visualization standard: **camera projection + metric BEV inset
together**, plus a text HUD carrying the model's decoded **tactical maneuver**, **strategic
route/goal**, per-frame **ADE**, v0 and clip-mean ADE. Every clip is labelled with **arm + step**.

**Directory (eval pod):** `/root/taniteval/results/videos/`
171 frames each (stride 1, full episode) @ 10 fps ≈ 17 s per clip.

Filenames follow `<arm>_step<step>_physicalai_ep<NN>_<regime>.mp4`. Clip-mean ADE in metres
(stride 1, all frames); ✅ marks the better arm on that clip.

| Clip | Regime | REF-B v2 (step 29999) | REF-C-XL (step 28000) |
|---|---|---|---|
| **ep31** | **high-speed straight**, 36.3 m/s | 0.388 | **0.142** ✅ |
| **ep03** | **sharp curve**, 68° net heading | 1.012 | **0.905** ✅ |
| **ep11** | **failure case** — worst episode for *both* arms | 1.430 | **1.145** ✅ |
| ep28 | high-speed curve, 19.7 m/s | 0.931 | **0.680** ✅ |
| ep17 | straight cruise, 17.9 m/s | 0.213 | **0.185** ✅ |

**REF-C wins every regime clip**, including both required failure/curve cases — consistent with
the uniform stratum win in §4.

**Copied into the repo** (`TanitAD Research Hub/Benchmarks & Eval/Research/videos-2026-07-20/`) —
the 3 required regimes × both arms, same clips, so they can be watched side by side:
`{refb-v2-30k_step29999,refc-xl-live_step28000}_physicalai_ep{31_highspeed-straight,03_sharpturn,11_failure-worstwindow}.mp4`

Clips were chosen from the per-episode strata of `results/windows_refb-v2-30k.pt` (mean speed, max
net-heading change, mean/worst ADE) so each covers a distinct regime. ep11 is the worst episode for
**both** arms, so the failure clip is a genuine paired failure rather than a cherry-pick.

*Clip-mean ADE is computed at **stride 1** (every frame), whereas the §3 leaderboard numbers are
stride 8 — so clip means will not equal the per-episode numbers in the window tensors.*

**What the videos show.** On the ep03 sharp left, REF-C decodes `tactical: turn left` /
`strategic: route left` correctly and the BEV inset shows it **under-turning** relative to GT —
the curve deficit of §4 made visible. On ep11 the same under-turn is larger (frame ADE 1.30 m).
On ep31 at 36 m/s REF-C tracks GT almost exactly (clip ADE 0.142) — the high-speed stratum win,
visible.

### Tooling note

`corpus_overlay.py` — THE standard — renders only **grounded-rollout** arms; it asserts a
`step_readout` exists, which REF-B and REF-C do not have. Added
**`taniteval/direct_overlay.py`**: the direct-trajectory-head branch, which **imports
corpus_overlay's own drawing primitives** (`draw_frame`, `FlatProjector`, `clip_extent`,
`pretty_man`, `pretty_route`) so the visual contract is identical rather than merely similar, and
calls each arm exactly as its scoring collector does — so the ADE burned into the HUD is the same
quantity the leaderboard reports.

**Honest rendering note.** Both arms emit **4 time waypoints** (0.5/1/1.5/2 s), not a dense
20-step path. For drawing, those 4 waypoints are placed at their **true** slots of a 20-point path
(indices 4/9/14/19 = the standard's `WP_IDX`, so every ring marker is a real model output) and the
points between them are straight-line interpolation. **No curvature is invented between
waypoints**, and every scored quantity comes from the 4 waypoints alone. The HUD states the surface
on every frame (`pred = 4 wp @ 0.5/1/1.5/2 s · <decoder>`).

A `_fit()` ellipsis guard was added after the first render silently truncated the HUD **tail** —
which is exactly where the model/step/ADE labels live.

---

## 6. Provisional status of the REF-C number

REF-C-XL is **step 28000 of 30000**; the run was live on pod3 throughout this eval
(`refc-diffusion-xl-30k`). Last observed at **step 28950, 08:32 UTC**, averaging **2.86 s/step**
over the 949 steps since the 07:46 resume → **30k ETA ≈ 09:20–09:30 UTC 2026-07-20**.

**Why the remaining 2000 steps are expected to move it very little:** the cosine schedule is
essentially annealed out. LR at step 28600 was **6.16e-7**, i.e. **~0.6 % of the ~1e-4 peak**. The
16k → 28k delta was large (0.605 → 0.479) because that span carried real LR; 28k → 30k does not.

This is an expectation, **not** a measurement. Treat 0.470 as a step-28000 number until refreshed:

```bash
# after pod3 reaches 30000 (ETA ~09:20-09:30 UTC) — check: ssh tanitad-pod3 "tail -1 /tmp/refc.log"
eval $(ssh-agent -s) && ssh-add ~/.ssh/tanitad_pod
ssh -A tanitad-eval "scp -P 22079 root@69.30.85.16:/workspace/experiments/refc-diffusion-xl-30k/ckpt.pt /root/models/refc-xl-live/ckpt.pt"
ssh tanitad-eval "cd /root/taniteval && PYTHONPATH=/root/taniteval:/root/TanitAD/stack \
  python3 -m taniteval.runner run --model refc-xl-live --episodes 40 && \
  python3 -m taniteval.runner ab --a refb-v2-30k --b refc-xl-live"
# videos: python3 -m taniteval.direct_overlay --models refc-xl-live
```

Note pod3's SSH port (22079) changes on pod restart — re-check before reusing.

---

## 7. Read-outs

1. **REF-C-XL is the stronger reference arm.** It beats REF-B v2 significantly and *uniformly*
   (every curvature stratum, 59.6 % of windows, Δ +0.11 m) — while still 2000 steps short.
2. **REF-C-XL closes to within 0.018 m of flagship v1 on the ADE@2 s heldout mean** (0.470 prov.
   vs 0.452) — but do not call that parity: the *paired* test (more powerful than the overlapping
   split CIs) still favours flagship significantly, Δ +0.052 m, CI [0.019, 0.086]. Where REF-C
   does win outright is the **high-speed stratum**: 0.330 vs 0.551.
3. **flagship's remaining advantage is the tail, not the typical case.** It wins the mean while
   losing the per-window majority (45.4 %); miss@2 m 0.060 vs 0.154. For REF-C the lever is
   tail/robustness — specifically **curves** — not average accuracy.
4. **REF-C's whole deficit vs flagship is curvature.** Straights: mean delta exactly 0.0000, REF-C
   wins 60.4 % of windows and has a *lower median* (0.219 vs 0.347). Gentle/sharp: −0.15/−0.22 m.
5. **Both arms are well clear of the CV floor** at every horizon (ADE@2 s 0.825 → 0.592 / 0.470).
6. REF-B v2 20k → 30k was a real but modest gain (0.646 → 0.592, significant); the arch-v2
   `--refbpatch` line is converged and is now the weakest of the three.

All numbers above are **open-loop / weak** claims. Per the closed-loop gap already on record
(open-loop 0.45 m → closed-loop 1.69 m for flagship), none of this predicts closed-loop behaviour.

---

## 8. Artefacts

**Eval pod** — raw JSON is the source of truth:
- `/root/taniteval/results/refc-xl-live.json`, `/root/taniteval/results/refb-v2-30k.json`
- `/root/taniteval/results/windows_refc-xl-live.pt`, `windows_refb-v2-30k.pt`
- `/root/taniteval/results/ab_refb-v2-30k_vs_refc-xl-live.json`,
  `ab_refc-xl-live_vs_flagship-30k.json`, `ab_refc-xl_vs_refc-xl-live.json`
- `/root/taniteval/results/videos/*step*.mp4` (10 clips), run log
  `/root/taniteval/results/videos_run.log`

**Repo:**
- `taniteval/taniteval/direct_overlay.py` — **new**, direct-head video renderer
- `taniteval/taniteval/registry.py` — **modified**, adds the `refc-xl-live` entry
- `TanitAD Research Hub/Benchmarks & Eval/Research/videos-2026-07-20/` — the 6 headline clips
- this report

Both code files are byte-identical to what ran on the eval pod (md5-verified both ways).

**Two working-tree caveats for whoever picks this up next:**

1. **The two `taniteval/` files ended up COMMITTED — by another agent, not by this one, and under
   an unrelated message.** This agent ran no `git add` and no `git commit`. Mid-session the files
   were found already staged (`A`/`M`), and the index was deliberately left untouched; a
   concurrent commit then swept them in:

   ```
   60265d3 fix(ops): boot hook would have DOUBLE-LAUNCHED a trainer onto a live GPU
     stack/scripts/pod_boot_hook.sh        |  21 ++-
     stack/scripts/supervise_run.sh        |  36 ++++-
     taniteval/taniteval/direct_overlay.py | 261 ++++++++++++++++++++++++++++++++++   <- this work
     taniteval/taniteval/registry.py       |  17 +++                                  <- this work
   ```

   So the eval tooling is **in history under a boot-hook commit message** and will not be found by
   anyone searching the log for it. No history was rewritten to fix this — flagging it for a
   deliberate decision.
2. **git cannot see `videos-2026-07-20/` at all** — not via `status`, `ls-files --others`, or
   `add -n`, with `core.fsmonitor=false` and `core.untrackedCache=false`. This is the known
   new-files-in-new-dirs blind spot in this working copy; the parent-dir rename workaround did
   **not** clear it. **The 6 .mp4 files are physically present and play normally** — they are
   simply invisible to git, so no sweep will commit them, and they must be handled deliberately if
   they are wanted in history (~7 MB of binaries; probably they are not).
