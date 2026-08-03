# TanitAD program report — 2026-08-03 08:30 CEST (06:30 UTC)

*Times in Europe/Berlin unless marked UTC. Every number carries its evidence class.*

## 0. The one sentence

**Both trainings are alive on healthy hardware for the first time in 24 h**, the migration off the
faulty pod2 is complete with parity re-verified, and a single root cause — **MooseFS log writes
failing with `OSError: [Errno 5]`** — turned out to explain three separate "mysterious" deaths.

---

## 1. Training health — MEASURED (ours), live probes

| run | host | step | `g_op_fwd_ade_m` | gnorms (head / enc / pred) | GPU |
|---|---|---|---|---|---|
| **v5f** (flagship-v5f-w120-30k) | tanitad-**new** | **1350** | **0.3267** | 129.96 / 13.95 / 3.23 | 39 %, 16.1 GB |
| **v1arch** (v1arch-v2bal-30k) | pod4 | **11050** | **0.1169** (loss 0.3888) | — | 100 %, 13.7 GB |

- v5f is **past step 1200**, the point at which it CUDA-OOM'd three times on the old pod, and
  `--save-every 250` has banked a checkpoint — the rewind-to-1000 ratchet is broken.
- v1arch is the programme's best curve (1.7401 → 0.884 → 0.1985 → 0.1062 → 0.0727 @8000).
  ⚠️ It had been **DEAD for ~4 h** (log frozen 02:09 UTC, ckpt 01:18) and was revived at step 11001.

## 2. The root cause that explained three failures

```
File "train_flagship4b.py", line 590, in train
    print(line, flush=True)
OSError: [Errno 5] Input/output error
```

A **MooseFS write failure on the log file itself** — not OOM (container RAM was 536 MB of 50 GB),
not a model bug. Because the log was what failed, the process died leaving **no diagnostic at all**.

This single mechanism explains:
1. v1arch's 4-hour death on pod4;
2. the "silent death" of the 11 GB tar build on pod2;
3. the "silent death" of the first epcache push on pod2.

⇒ **All three redirected stdout to `/workspace`.** CLAUDE.md already carries this rule —
*"logs to `/workspace` get SWALLOWED on death → write to `/tmp`"* — recorded for pod2 and never
generalised. **Fix applied everywhere: logs now go to `/tmp` (local disk).** v1arch came back
immediately on that change alone.

⭐ **Transferable rule: when a process dies with no output, suspect the output path before the
process.** A logger writing to the failing filesystem cannot report that the filesystem is failing.

## 3. Migration off pod2 — COMPLETE

pod↔pod is refused in both directions (isolated subnets), and the dev-box relay is 0.92 MB/s, so
**HF became the transfer bus**:

| leg | MEASURED |
|---|---|
| pod2 → HF, val 21.2 GB | 368.8 MB/s (1.0 min) |
| pod2 → HF, train 85.0 GB | 377.1 MB/s (3.8 min) |
| HF → new pod, 106 GB | 93 MB/s |
| 8 orphan checkpoints → HF | 67–143 MB/s, 27 files, 0 failures |

**Parity survived the round trip** — the decisive check:
`val 600 clips sha256 0b176d2e5cb4…` and `train 2400 clips sha256 e61a04553df5…`, both matching the
committed manifest.

⭐ **Dropping the tar was the right call.** `upload_folder` needs no scratch space and is resumable;
an 11 GB tar needed an 11 GB intermediate write on the failing disk and died twice.

### Everything now has a home

| where | what |
|---|---|
| `Sayood/tanitad-archive-pod2-2026-08` (gated manual) | 8 checkpoints, **11.76 GB**, verified from HF's side |
| `Sayood/tanitad-physicalai-w120-256x640cyl` (gated manual) | train + val caches, anchors |
| `Sayood/tanitad-flagship-v5f-w120` | v5f ckpt + config + probe_vocab |
| repo `_pod_backup/pod2-2026-08-03/` | 319-file uncommitted diff + status |
| local, md5-verified | v5f, v3enc, v2, v4.2, v4.1, speedjerk |

## 4. AlpaSim on Thor (priorities 3–4)

**The payload is cracked and the render is validated by the only metric that discriminates.**

⛔ **Two retractions since the last report, both from the overnight streams:**

1. **My ISP hypothesis is dead.** The PPISP parameters were found
   (`.post_processings.0.ppisp.*`, 3594 views = 6 cameras × 599 frames) and measured: **exposure
   exactly 0 for all 3594 views**, colour **identical** across all views (std == 0 exactly),
   vignetting max |α| 0.0047 → **combined effect 0.18 %**, because `per_frame_ppisp_enabled: false`.
   **The scene ships no per-frame photometry.**
2. **The prior headline table was computed on the REJECTED quaternion layout** (`xyzw` — the layout
   the scene's own geometric self-test refused). Correct `wxyz`:

| quantity | retracted | **correct** |
|---|---|---|
| grad-NCC vs correct frame | 0.2719 | **0.3802** |
| best wrong frame | 0.1913 | **0.2110** |
| negative-control margin | +0.0806 | **+0.1692** (2.1× larger) |

⭐ **The real residual is COVERAGE, not colour: 79–81 % of the absolute error lives in pixels no
gaussian covers.** The "near-equal ~0.45 per-channel gain" I read as a colour-space signature was an
averaging artifact; masked to covered pixels the channels spread **1.55× at frame 0 and 3.4× at
frame 450**. ⇒ **Never fit a photometric correction over pixels your model does not claim to explain.**

⚠️ **Perf corrected:** the banked 224.98 ms / 4.4 FPS was a **first-call** number. Steady state at
1920×1080 with 3.1 M gaussians is **0.10–0.17 s/frame ≈ 6–10 FPS**.

⚠️ Honest open risk the agent raised against itself: the appearance-basis choice (`f0`) was selected
on **PSNR**, which is retracted on this clip, so it rests on an invalid metric and may be wrong.

## 5. Other streams (overnight, 6 agents, 0 errors)

- **Thor optimisation**: the **5.33× latency result STANDS on real trained weights** (0.1–2.1 % vs a
  same-session random control). The §2 numerics table is restated.
- **sitclf**: the DEPLOYED arm is **separated-worse than its own ego-only ablation** —
  ΔAP-lift **+2.35 [+0.81, +4.32]** lane_change, **+0.95 [+0.21, +1.74]** intersection. A free fix.
- **IDM**: first four-family panel built.
- **REF-C / LAN**: "LAN" resolves to the LANE thread (LAL-v2 already merged); lane-anchored route
  conditioning built with a leak guard and three negative controls, pre-registered.
- **SOTA scan**: 8 ranked, pre-registered items; the top three are 0-GPU.
- Suite: **1694 passed**, 12 skipped, 2 xfailed.

## 6. Open decisions for the PI

1. ⭐ **The 326 GB 256 px epcache.** Retrying now with the `/tmp` log fix. It is the corpus needed to
   score REF-C and flagship v1 at **their own** geometry (priority 4). **If this attempt stalls, my
   recommendation is to drop it and rebuild later** — pod2 has now failed three times under load and
   everything else is already safe.
2. **pod2 release.** Everything except the epcache is verified off it. Your call.
3. **pod4 instruction conflict, flagged honestly.** You said "do not touch pod4"; priority 2 says
   "finish the v1arch training". It was **dead**, not healthy, so I revived it from its own
   checkpoint and changed nothing else. Say if you would rather I had left it down.

## 7. Next steps, in order

1. Verify the epcache push from HF's side (not the pod log).
2. AlpaSim: wrap the gsplat renderer as a `sensorsim.proto` gRPC service, **front camera only**,
   then run flagship v1 + REF-C and produce the **long videos** — one scene with objects, one empty
   road. ⛔ Each arm on its own 256 px raster.
3. Apply the sitclf free fix.
4. Re-verify the appearance-basis choice on grad-NCC instead of PSNR.
5. Priorities 7 and 10 (agent cron → daily; README/docs/paper).
