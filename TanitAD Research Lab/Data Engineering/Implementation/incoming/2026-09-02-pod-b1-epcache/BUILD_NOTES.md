# B1 epcache on the A40 pod — the corpus that was needed, built

**Stream:** Data Engineering FlyWheel · **Wall-clock:** 2026-09-01 (directory
carries the coordinator's `2026-09-02` label so their cross-references resolve).
**Pod:** A40 46 GB, `69.30.85.211:22001`.

Two phases. Phase 1 pulled the published **parity** cache (2,400+600) and proved
it was the wrong corpus for this launch. Phase 2 built the **B1** cache the
launch actually needs. Both are on the pod, in separate directories.

---

## 1. Headline

| | MEASURED |
|---|---|
| **B1 corpus size** | **4,713 clips — NOT 4,719** (§3) |
| B1 build throughput | **~980 eps/h** steady state, 48 workers (§6a) |
| B1 ETA | **~4.7 h**, finishing ≈12:30 local (started 07:44) |
| Contract test | **`jpeg_buf` BYTE-IDENTICAL** to the verified parity payload (§4) |
| parity cache (fallback) | train 2,400/2,400 + val 600/600 CONTENT-VERIFIED, BAD 0 |
| pod CPU ceiling | **7.65 cores, not 96** (§6) |

---

## 2. Why B1 and not parity — the label measurement

The coordinator measured v7.2 label coverage per corpus, and it is decisive:

| corpus | clips with a v7.2 label |
|---|---|
| parity 2,400 | 190 = **7.9 %** |
| **B1 4,713** | **4,572 = 97.0 %** |
| val 600 | 47 = 7.8 % |

Training the tactical/strategic heads on parity would supervise them on under
8 % of windows. Independently reproduced here: `|v7.2 train labels ∩ B1| =
4,572`, and B1 partitions **exactly** into 4,572 train + 147 eval labels with
**zero unlabelled and zero train/eval intersection**.

## 3. ⛔ THE CORPUS IS 4,713, NOT 4,719 — and I nearly built the wrong number

The brief, the DATACARD and the camera directory all say **4,719**. The parity
instrument, run on those 4,719 ids, says:

```
201 in physicalai-train-e438721ae894
6 of 40 deployed-val episodes          (15 % of that episode set)
DROPPED 6 -- "The corpus is 4713 clips; quote THAT number, never 4719."
```

Dropped: `01acc9de 026ef99a 030011f7 07f7b41f 09759d8c 0c99c815`.
Those 6 sit inside the **40-episode val deployment behind every published
open-loop number**, so a 4,719-clip build puts val episodes into training.

**My first launch built all 4,719.** It was caught by a background repo search
that surfaced an earlier Thor build of the same corpus at **4,713** — the
discrepancy is what sent me to the instrument. ⇒ The gate is now **run inside
the driver** (`pod_build_b1_epcache.py`), not assumed, and the dropped ids are
written into `_build_manifest.json`. A driver that calls `build_compressed`
directly bypasses `v2_compressed.build`'s gate entirely; that bypass is silent,
and this is the second time in this programme a gate mattered only because
something else contradicted the headline number.

⚠️ **A wider overlap remains and is deliberately NOT dropped:** `|B1 ∩ val600|
= 56`, `|B1 ∩ parity2400| = 201`. Only the val40 set disqualifies a training
corpus. But **47 of the 600 val episodes are inside B1's training set**, so a
B1-trained arm's numbers on val600 are contaminated for those 47. State that
before comparing a B1 arm to any val600 figure.

## 4. The contract test — byte-identical, not merely "same schema"

Exactly **one** clip (`13141fac-…`) is in both B1 and the 24 verified parity
reference payloads. Building it through the new driver and diffing every field
against the known-good payload:

```
keys equal: True
actions (201,2) equal=True      poses (201,4) equal=True
jpeg_buf (38544640,) equal=True jpeg_len (201,) equal=True
frame {'height':256,'width':640,'f_ref':305.5774907364391,
       'projection':'cylindrical'} equal=True
codec 'png'  projection_mode 'cylindrical'  n_stack 3  quality 90
image_h 256  image_w 640  image_size 256  episode_id 825438516   ALL equal

jpeg_buf md5 parity: bef95bf18bf6e92d4226f06ddb55adc3
jpeg_buf md5 B1    : bef95bf18bf6e92d4226f06ddb55adc3
PIXELS BYTE-IDENTICAL: True
```

This is stronger than a schema check: it proves the **v7 provider egomotion
yields the same poses** as PhysicalAI's per-chunk egomotion zips, and that the
whole decode → cylindrical-rectify → PNG path reproduces the published cache
bit-for-bit. Re-run and still identical after the intrinsics table was trimmed.

## 5. What made the build possible — three input adapters, each content-checked

The v7 corpus ships a **flat per-clip layout**, not PhysicalAI's per-chunk zips.
Staged at `/workspace/v7build/`:

* **camera** — the mp4s **are** shipped in the v7 repo (4,719, 57.4 GB), pulled
  in **3.2 min at 307.7 MB/s**, 0 failures. ⚠️ `DATACARD.md` says *"Not shipped
  (~47 GB)"* and documents an HTTP-range retrieval contract against nvidia's
  zips — **that is stale**; the bytes were pushed on 2026-08-30, after the card
  was written. Reading the card alone would have cost a needless range-read path.
* **timestamps** — `timestamps.tar` → 4,719 `<clip>.timestamps.parquet`.
* **egomotion** — the loader `physicalai.load_egomotion` opens a **ZIP** and
  wants `<clip_id>.egomotion.parquet`; the corpus ships a **TAR** of
  `<clip_id>.parquet`. Repacked once into a single STORED zip with the expected
  member names, so the loader runs **unmodified**. Measured cost 0.30 s/call —
  checked explicitly because it was my prime suspect for the slow start, and it
  was **not** the cause.
* **intrinsics** — pulled 1,411 `camera_intrinsics` chunk parquets from nvidia
  (39 s, 0 errors) → per-clip CSV covering **4,719/4,719**, rig clusters
  A median 541.9 / B median 754.2, matching the documented A≈542/B≈753 split.
  ⛔ Load-bearing: without a per-clip table the crop silently reverts to
  geometric-centre, ~215 px wrong for rig B — **the majority of this corpus**.

## 6. ⛔ THE POD HAS 7.65 CPUs, NOT 96 — the brief's premise was wrong

The brief said *"You have 96 CPUs — PARALLELISE the encode"*. Every standard
probe agrees with that and every one of them is answering the wrong question:

```
nproc                      96
os.sched_getaffinity(0)    96
cpuset.cpus.effective    0-95
cpu.cfs_quota_us      765000   <-- the real ceiling
cpu.cfs_period_us     100000       765000/100000 = 7.65 CPUs
```

The signature that exposed it: all 48 workers sat at **exactly 15.8 % CPU**;
48 × 15.8 = 758 % ≈ the 765 % quota. `top` confirms **78.3 % idle** system-wide
while 45 processes are in `R`.

⇒ **More workers cannot help — throughput is quota-bound.** This is the `df` /
Thor `free` / cgroup `usage_in_bytes` family exactly: a probe reporting the
wrong scope, read as an answer — and here it came from the brief, not the box.

### 6a. ⚠️ A RATE MEASURED IN ONE WINDOW IS NOT A RATE — I quoted 1,560 eps/h and it was wrong

My first throughput reading was **39 episodes in 90 s = 1,560 eps/h**, and I
reported an ETA of 3.0 h from it. That window was a **startup burst**: all 48
workers began simultaneously, so their *first* episodes completed within
seconds of one another and the window happened to sit on that cohort boundary.

Three samples tell the real story:

| sample | rate |
|---|---|
| 90 s window over the first cohort | 1,560 eps/h *(artifact)* |
| 180 s window at steady state | **980 eps/h** |
| build's own cumulative counter | 559 → 721 eps/h, still amortizing startup |

⇒ **~980 eps/h, ETA ~4.7 h.** Single-process was 173 eps/h, so 48 workers buy a
**5.7×** speed-up on a 7.65-core quota. ⇒ **Sample a periodic process at least
twice, on windows that do not align with its period, before quoting a rate** —
and prefer the build's own cumulative counter once it has amortized. This is
the "verify before alarming / take multiple samples" rule with the sign
flipped: the single sample was not a false alarm, it was false *reassurance*,
which nobody goes looking for.

## 7. Two self-inflicted errors, logged because they are the cheap lessons

1. **A join on a mis-sliced stem produced a confident false negative.** I
   computed corpus stems with `basename[:-4]` against `.parquet` (8 chars),
   yielding `<uuid>.par`, and reported *"v7.2 train labels inside B1: 0/4572"* —
   flatly contradicting the coordinator's 97 %. The identifiers were fine; my
   slice was not. Corrected by removing the **real** suffix length and
   cross-checking three independent stem sources (camera == egomotion ==
   timestamps, all True). *An identifier is only as good as the operator that
   produced it.*
2. **A missing output directory burned 10 minutes and produced nothing.** The
   `torch.save` is the **last** step of a ~20 s decode+encode, so a probe
   pointed at a non-existent dir paid full price per episode and failed at the
   final line — looking exactly like a slow build. Same family as the
   analysis-time import that dies after the rollout. `build_one` now creates the
   directory before doing any work.

## 8. Verification contract (unchanged from phase 1, applied per episode)

Each payload is verified **inside the build**, and a failure deletes the file so
it is rebuilt on resume: `torch.load` succeeds · `jpeg_len>0` ·
`poses==actions==jpeg_len` · `jpeg_buf.numel()==sum(jpeg_len)` ·
`image_h/w==256/640` · `projection_mode==cylindrical` · `codec==png` ·
`n_stack==3` · **a real `decode_png` of frame 0 returning `(3,256,640)` with
`max()>0`** (the all-zero-bank trap). Resume is **by content**: on restart every
payload on disk is re-verified in parallel and any that fails is discarded, so
"already built" can never mean "present but broken".

## 9. Eval-split measurement (requested — measured, NOT built)

All **147** v7.2 eval clips are inside B1. B1 = 4,572 train + 147 eval exactly.

| where the 147 eval clips live | n |
|---|---|
| inside B1 (4,719 as shipped) | **147** |
| inside val600 | 9 |
| inside parity2400 | 11 |
| in none of the three | **0** |

⇒ **The eval split should be built from the 147 clips, which are already inside
B1.** Two consequences for whoever builds it:

* **2 of the 147 are among the 6 val40 clips the parity gate drops**, so they
  will not exist in the built train cache — the eval split must be built as its
  own directory, from the full 4,719, not by selecting out of the 4,713.
* 9 of the 147 are also in val600; they are *not* trained on, so they are safe
  as eval — but they are not independent of legacy val600 numbers.

Clip ids banked at pod `/workspace/v7src/split_measure.json`.

## 10. ⚠️ ESCALATION — a 4,713-episode B1 cache ALREADY EXISTS ON THOR

A background repo search found `/home/nvidia/data/physicalai-b1-w120-256x640cyl`
on **Thor** — **4,713 episodes / 177,998,547,213 B (~178 GB)**, `_geometry.json`
sha `f0d34b84` (`Project Steering/SPEC_V7_LABEL_TRAINER_WIRING.md:219-221`;
`…/2026-09-01-refcv3-training-readiness/READINESS.md:58-62`). It was **never
pushed to HF**, which is why the phase-1 HF sweep could not see it — that sweep
covered the HF namespace and the pod, **not Thor**, so its "exists nowhere" was
true of where it looked and false of the programme.

**This does not make the pod build redundant** — REF-C v3 trains on the pod, and
moving 178 GB off Thor over wifi is slower than the 3 h rebuild. It matters
because (a) it independently corroborates **4,713**, and (b) **no script in the
repo has ever pushed a `.v2ep.pt` cache**; both B1 builds are single-disk
artifacts. That is the "finish before you start" rule unmet twice over.

## 11. ⛔ BANK THIS — a tree listing near 1,000 entries is a TRUNCATION

The HF tree API caps at 1,000 entries per page. A single-page read of
`Sayood/tanitad-physicalai-w120-256x640cyl` returns **993 files** — a number
that looks like a complete small repo, and whose first page is entirely legacy
raw `ep_NNNNN.pt`. The repo actually holds **6,061 files / 424 GB over 7 pages**,
with every `.v2ep.pt` on later pages. **That truncation would have concluded
"the cache does not exist" and triggered a needless 4 h rebuild.** It also
understated `tanitad-ph0-aug120` as 909 files (actually 7,241).

⇒ **Follow `Link: rel="next"` until absent; treat any count at or near the page
limit as unproven.** Guarded in `pod_pull_b1_epcache.list_files`'s docstring.

## 12. Deliverable manifest

| artifact | location |
|---|---|
| B1 build driver | repo `stack/scripts/pod_build_b1_epcache.py` (staged) |
| puller + content-verifier (generalised) | repo `stack/scripts/pod_pull_b1_epcache.py` (staged) |
| these notes | repo `…/incoming/2026-09-02-pod-b1-epcache/BUILD_NOTES.md` (staged) |
| **B1 cache (building, 4,713)** | pod `/workspace/TanitAD/data/b1-epcache-4719/` + `_build_manifest.json` |
| B1 build log | pod `/workspace/b1build4719.log` |
| parity train cache (fallback) | pod `/workspace/TanitAD/data/b1-epcache/` (2,400, 80 GB) |
| parity val cache | pod `/workspace/TanitAD/data/b1-epcache-val/` (600, 20 GB) |
| staged v7 inputs | pod `/workspace/v7build/` (camera, timestamps, egomotion zip, intrinsics CSV) |
| parity gate record | pod `/workspace/v7src/parity_gate.json`, `b1_kept.txt` |
| eval-split measurement | pod `/workspace/v7src/split_measure.json` |

⚠️ The output directory is named `b1-epcache-4719` (as instructed) but holds
**4,713** episodes. The count in the directory name is wrong by design of the
instruction; `_build_manifest.json` carries the true number and the dropped ids.

## 13. Escalations

1. **Neither B1 cache has ever been pushed** (§10). Both are single-disk.
2. **`|B1 ∩ val600| = 56`** (§3) — 47 in the training set. Any B1-vs-val600
   comparison needs this stated.
3. **The v7 `DATACARD.md` camera section is stale** (§5) — it says the mp4s are
   not shipped; they are.
4. **The pod is a 7.65-core box** (§6). Any future "we have 96 CPUs" plan for
   this pod is wrong.
5. **The eval split must be built from the full 4,719**, not the 4,713 (§9).
