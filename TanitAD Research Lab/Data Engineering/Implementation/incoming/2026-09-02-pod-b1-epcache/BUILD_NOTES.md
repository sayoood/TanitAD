# B1 epcache on the A40 pod — it already existed; PULLED, not rebuilt

**Stream:** Data Engineering FlyWheel · **Wall-clock date:** 2026-09-01 (the
directory carries the coordinator's `2026-09-02` label so their cross-references
resolve; the measurements below are 2026-09-01).
**Pod:** A40 46 GB / 96 CPU / 503 GB RAM, `69.30.85.211:22001`.
**Outcome:** the cache REF-C v3 needs is **on the pod, content-verified, ready**.
No build was run.

---

## 1. Headline

The epcache was **already built and pushed** — the PI's recollection was
correct. It is published at

> **`Sayood/tanitad-physicalai-w120-256x640cyl`**
> `physicalai-train-e438721ae894-w120-256x640cyl/` — **2,400 `.v2ep.pt`, 79.15 GB**
> `physicalai-val-0c5f7dac3b11-w120-256x640cyl/` — **600 `.v2ep.pt`, 19.75 GB**

Pulling it took **2.1 min at 622.8 MB/s** against a rebuild costed at
**3.4–4.6 h**. Every payload was opened and checked.

| | MEASURED |
|---|---|
| train CONTENT-VERIFIED | **2,400 / 2,400 · BAD 0** |
| val CONTENT-VERIFIED | **600 / 600 · BAD 0** |
| reference-sample identity | **24 / 24 md5-identical** |
| downlink | **622.8 MB/s** (train), 589.9 MB/s (val) |

---

## 2. ⚠️ A NAMING CORRECTION THE READER MUST NOT SKIP

**What is on the pod is the canonical PARITY corpus (2,400 clips), not a
4,719-clip "B1".** These are two different corpora and the brief used one name
for both:

* `v2_compressed._ensure_ego`'s docstring records *"MEASURED 2026-08-30 on B1:
  **4,719 clips** spread over 1,411 chunks"*, and
  `Sayood/tanitad-v7-training-corpus` (the source the brief named) holds
  **4,719** `.mp4`. On that reading **B1 = the 4,719-clip v7 corpus, and no
  epcache for it exists anywhere** (§3).
* What was published, and what the pod's own 24 reference payloads are drawn
  from, is the **2,400-clip canonical train corpus, parity key
  `e438721ae894`, skip-hash `f09e44db`** — the corpus `CLAUDE.md` calls sacred.

⇒ **If REF-C v3 is meant to train on the canonical parity corpus, it is
unblocked right now.** If it was genuinely meant to train on all 4,719 v7 clips,
that cache does not exist and is a **4 h build** — a PI/coordinator decision,
not one I should take. Flagged, not assumed. *(Class: an identifier that omits
its scope is not an identifier.)*

---

## 3. How I established absence — two independent probes

*Absence found at one location is not absence.*

1. **HF API, full namespace, paginated.** `whoami-v2` → `Sayood`. Enumerated
   **18 datasets + 26 models**, then walked each candidate's
   `tree/main?recursive=true`. Exactly **two** repos contain any `.v2ep.pt`:
   the w120 repo above (3,000) and `Sayood/tanitad-transfer-2026-08` (20 — a
   duplicate val subset). `tanitad-v7-training-corpus` was walked to
   completion: **4,744 files / 59.30 GB / 5 pages / ZERO `.v2ep.pt`** and no
   epcache tar or zip shard — only `timestamps/timestamps.tar` (53 MB) and
   `egomotion/`, in **subdirectories**, as the coordinator asked.
2. **The repos' own manifests.** The w120 `README.md` and both
   `_geometry.json` files declare the geometry and the parity key
   independently of the file listing; the v7 `DATACARD.md` describes a
   **camera-retrieval contract** (pull mp4s by HTTP range read), i.e. it
   documents itself as a *source* corpus, not a built cache.

⛔ **PAGINATION WAS LOAD-BEARING AND NEARLY COST THE ANSWER.** My first pass
read one page and reported the w120 repo as *"993 files, 990 `.pt`, no
`.v2ep.pt`"* — which would have concluded **"the cache does not exist"** and
sent me into a 4 h rebuild. The repo actually holds **6,061 files / 424 GB**
across 7 pages; the first page was all legacy raw `ep_NNNNN.pt` (117,381,403 B
each — the 117.4 MB raw-uint8 figure from `CLAUDE.md`), and every `.v2ep.pt`
lived on later pages. The same truncation understated
`tanitad-ph0-aug120` as 909 files (actually 7,241).
⇒ **A tree listing at exactly-or-near the page limit is a truncation until the
`Link rel="next"` header says otherwise.** Same family as the `df` / Thor
`free` / cgroup `usage_in_bytes` traps: a probe reporting the wrong scope,
read as an answer. Guarded in the tool's `list_files` docstring.

---

## 4. The schema I verified against

From the repo's `_geometry.json`, matching the pod's 24 reference payloads at
`/workspace/TanitAD/data/eps/` field-for-field:

```
frame            {"height":256,"width":640,"f_ref":305.5774907364391,
                  "projection":"cylindrical"}
frame_tag        256x640f305.5775cyl
projection_mode  cylindrical
codec            png          <-- LOSSLESS, load-bearing
n_stack          3
geometry_check   requested_hfov 120.0 -> achieved 120.0, f_eff 305.5775,
                 observed_frac 0.9348642592 (POPULATION of 600, not one clip)
keys             jpeg_buf jpeg_len actions poses n_stack image_size episode_id
                 clip_id quality image_h image_w frame projection_mode codec
```

The `codec: png` is **not a speed knob** — `v2_dataset.py:325` and
`slice_v2_cache.py` refuse to sub-frame a lossy cache, so only a PNG cache can
be sliced to the `176x624` sub-frame the README names without a rebuild. The
published cache is PNG, so that route stays open.

`observed_frac 0.9348` is a real property, not a defect: the front-wide camera
is **two rigs** whose principal points differ ~211 px vertically, so a
120°×45.456° request over-runs rig B's sensor and the rectifier masks ~8.9 % of
every rig-B frame. The manifest carries a dated `corrections` block recording
that this figure was restamped from a false `1.0` that came from probing **one**
clip. Consumers should read it.

## 5. Content verification — what "verified" means here

⛔ *Verify by content, never by presence.* A pre-allocated or aborted file sits
at full size and reads as zeros. Each of the 3,000 payloads was opened and
required to satisfy **all** of:

* `torch.load` succeeds
* `jpeg_len.shape[0] > 0`
* `poses.shape[0] == actions.shape[0] == jpeg_len.shape[0]`
* `jpeg_buf.numel() == jpeg_len.sum()` (no truncated tail)
* `image_h/image_w == 256/640`, `projection_mode == cylindrical`,
  `codec == png`, `n_stack == 3`
* **a real `decode_png` of frame 0 returns `(3,256,640)` with `max() > 0`**

Result: **train 2,400/2,400 in 50 s; val 600/600 in 20 s; BAD = 0; zero stray
`.tmp`.** Reported as a *content-verified count*, never a file count.

**Independent confirmation of identity:** all **24/24** payloads that already
sat on the pod are **md5-identical** to the ones pulled — so this is the same
artifact those samples came from, not a lookalike at the same geometry.

## 6. Throughput MEASURED

| phase | n | bytes | rate | wall-clock |
|---|---|---|---|---|
| probe (16 workers) | 24 | 0.80 GB | 307.8 MB/s | 4 s |
| train (24 workers) | 2,376 | 78.35 GB | **622.8 MB/s** | **2.1 min** |
| val (24 workers) | 600 | 19.75 GB | 589.9 MB/s | 0.6 min |
| verify train (24 proc) | 2,400 | 80 GB read | — | 50 s |
| verify val (24 proc) | 600 | 20 GB read | — | 20 s |

Pod disk after: **100 GB** used by the two caches; a real `dd` write still
returns **480 MB/s** (never judged by `df`).

Verification is fanned out across **processes** with `OMP_NUM_THREADS=1` and
`torch.set_num_threads(1)` — torch spawns ~113 threads per process and an
unpinned pool stalls in a way that looks exactly like a hang.

## 7. Failures

**None.** 0 download failures, 0 content failures, 0 excluded episodes across
both splits.

## 8. Deliverables

| artifact | location |
|---|---|
| puller + content-verifier | repo `stack/scripts/pod_pull_b1_epcache.py` (staged) |
| these notes | repo `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-02-pod-b1-epcache/BUILD_NOTES.md` (staged) |
| train cache, 2,400 verified | pod `/workspace/TanitAD/data/b1-epcache/` (80 GB) + `_geometry.json` |
| val cache, 600 verified | pod `/workspace/TanitAD/data/b1-epcache-val/` (20 GB) + `_geometry.json` |
| pull log | pod `/workspace/b1build.log` |
| verify log | pod `/workspace/b1verify.log` |
| script on pod | pod `/workspace/pod_pull_b1_epcache.py` (md5 `5318f7888afe2ce2fb99b89ce6bf3ee4`, identical to the repo copy) |

Train and val are in **separate directories on purpose** — merging them would
put val episodes inside a train cache, which is exactly what the parity ingest
gate in `v2_compressed.build` exists to refuse.

## 9. Escalations

1. **Decide which corpus REF-C v3 trains on** (§2): the canonical 2,400-clip
   parity cache (ready now) or a 4,719-clip v7 build (~4 h, does not exist).
   This is the only thing between here and a launch.
2. **The pod holds no source data** — no mp4, no timestamps parquet, no
   PhysicalAI root. A v7 build would first pull ~47 GB of camera from
   `tanitad-v7-training-corpus`; budget that on top of the build.
3. **`Sayood/tanitad-transfer-2026-08`** carries 20 val `.v2ep.pt` duplicating
   the w120 val dir — a redundant copy worth retiring once someone confirms
   nothing points at it.
