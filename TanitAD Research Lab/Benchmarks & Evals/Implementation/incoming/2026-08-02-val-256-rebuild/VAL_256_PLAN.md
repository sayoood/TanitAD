# VAL @ 256px SQUARE — re-derivation plan, and why no rebuild is needed

**Stream C** · 2026-08-02 · agent deliverable · every number labelled with its evidence class.

> **Headline (MEASURED).** The plain-256px val corpus is **NOT lost with pod3.** The
> **raw epcache** `physicalai-val-0c5f7dac3b11` — 600 episodes of
> `frames_u8 [T, 9, 256, 256] uint8` — is **alive on pod2** at
> `/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11`
> (65.56 GiB, `DONE = {"episodes": 600, "skipped": 0}`). pod2 also holds the
> flagship-v1 (speedjerk) and REF-C base/xl checkpoints **and its A40 is idle**.
> ⇒ **REF-C and flagship v1 are scoreable at their own geometry today, with ZERO
> rebuild and ZERO data movement.** Cost of the "rebuild" is **0 GB and 0 wall-clock.**

This is a textbook **"absence found at ONE location is not absence"** (CLAUDE.md
operating standard rule 2). The brief's premise — *"the plain 256px val cache lived on
pod3, which is TERMINATED"* — is true of the **v2 compressed** form and of pod3, and
false of the fleet: the **raw epcache** form survives on pod2. One extra probe path
(`_epcache/` under `physicalai_phase0/` rather than a top-level `physicalai-val-*` dir)
was the whole difference.

---

## 0. What was asked, answered

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | What SOURCE does a v2 cache build from? | The **original PhysicalAI clips** — `mp4` + `timestamps.parquet` + `egomotion` zip, fetched per chunk from HF `nvidia/PhysicalAI-Autonomous-Vehicles`, then the mp4s+zip are **deleted**. | MEASURED (code read) |
| 1b | Is the w120 cache re-derivable to 256px? | **NO — REFUSED by the code.** Double geometry mismatch. It is a **RESAMPLE**, i.e. a rebuild from source video. | MEASURED (ran it) |
| 2 | What source data still exists? | pod2: raw val epcache (600 @256px), raw train epcache (2376 + 24 skips), 600 source mp4s, 197 egomotion zips. Thor: w120 only, + 5 relayed 256px eps. HF: source repo **reachable**. | MEASURED |
| 3 | Parity implication + how identity is verified | Four independent checks, three of them content-level. **The manifest's uid digest is NOT one of them** — see §4.3. | MEASURED |
| 4 | 3–5 episodes at 256px on Thor, load-verified | **5 relayed, md5-identical, strict-loaded, real forward run on Thor's Blackwell.** | MEASURED |
| 5 | Cost for all 40 | **4.375 GiB, ~89 min** to Thor. **0 GiB / 0 min** if scored on pod2. | MEASURED |

---

## 1. What a v2 cache builds from — and why that closes the question

`stack/scripts/v2_compressed.py` · `build_compressed(clip, …)` → `_resampled(clip, …)`
→ `_decode_cropped_selected(clip["mp4"], …)`.

The `clip` dict carries `mp4`, `timestamps`, `ego_zip`, `clip_id`. The `build` mode
fetches `camera/camera_front_wide_120fov/…chunk_NNNN.zip` from
`nvidia/PhysicalAI-Autonomous-Vehicles`, extracts the selected clips, builds, and then
**deletes the mp4s + zip** (docstring: *"per-chunk fetch camera -> extract selected ->
build compressed -> delete mp4s+zip"*).

**A v2 cache is therefore a terminal derived artifact.** It is built from raw video by a
`grid_sample` resampling (`cylindrical_rectify` or `ftheta_crop_resize`, selected by
`projection_mode`). Nothing in the pipeline can go *backwards* from one resampled raster
to a differently-resampled one.

### 1b. The slice route is REFUSED — measured, not argued

The only geometry-changing operation the loader supports is a **centred sub-frame slice**
(`V2CompressedCache(frame=…)` → `calib.subframe_slice`), and it *"[r]efuses any pair that
is not a centred sub-rectangle with the SAME focal and projection"*.

MEASURED on pod2 against a **real w120 payload**, with `tanitad/data/calib.py`
**md5-verified identical to repo HEAD** (`041c600e7264685e38545994c599e2a4`):

```
stored (w120)   : {height 256, width 640, f_ref 305.5774907364391, projection 'cylindrical'}
                  tag 256x640f305.5775cyl   codec 'png' (LOSSLESS)
target          : CANONICAL_256 = {256, 256, f_ref 266.0, projection 'pinhole'}
                  tag 256x256f266pin

subframe_slice(w120 -> 256sq)  ->  REFUSED  ValueError:
  "256x256f266pin is not a slice of 256x640f305.5775cyl: a slice preserves f_ref and
   projection exactly (got f_ref 266.0 vs 305.5774907364391, projection pinhole vs
   cylindrical). Changing either is a RESAMPLE, i.e. a rebuild from the source video."
```

**Both** invariants are violated at once — focal (266.0 vs 305.577) *and* projection
(pinhole vs cylindrical). It is not a near-miss.

**Positive control** (proves the refusal is about geometry, not a broken code path) —
slicing w120 to a 256×256 window at the *same* f_ref and projection **succeeds**:

```
subframe_slice(w120 -> 256x256 @ f305.5775 cyl)  ->  SLICE OK, rows [0,256] cols [192,448]
```

⇒ **The w120 cache cannot be turned into the 256px square by any supported operation.**
Re-deriving 256px from w120 would require the original PhysicalAI clips.

Artifact: `raw/pod2_probe.json` → `P1_slice_refusal`.

---

## 2. Source-data probe — what still exists, where

### pod2 (`ssh tanitad-pod2`) — probed READ-ONLY

| what | path | measured |
|---|---|---|
| **raw val epcache @256px SQUARE** | `/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11` | **600** `ep_*.pt`, 0 `skip_*`, `DONE {"episodes": 600, "skipped": 0}`, **70,392,852,474 B = 65.56 GiB** |
| raw train epcache | `…/_epcache/physicalai-train-e438721ae894` | **2376** `ep_*.pt` + **24** `skip_*`, `DONE {"episodes": 2376, "skipped": 24}`, 259.64 GiB |
| source video (val clips) | `/workspace/data/physicalai_phase0/r0/camera_front_wide/` | **600** `*.mp4` + **600** `*.timestamps.parquet` (+3000 blurred_boxes), 6.6 GB |
| egomotion labels | `/workspace/data/physicalai_phase0/labels/egomotion/` | **197** chunk zips, 7.7 GB |
| calibration | `/workspace/data/physicalai_phase0/calibration/sensor_extrinsics/` | per-chunk parquet |
| w120 val v2 | `/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl` | 600 `*.v2ep.pt`, 20 GB, codec `png` |
| **flagship v1 (speedjerk)** | `/workspace/v4gate30k/v1_speedjerk_ckpt.pt` | step **29999**, `encoder.pos [1,256,768]`, `action_dim 3` |
| **REF-C base / xl** | `/workspace/models/refc-base-30k/ckpt.pt`, `…/refc-xl-30k/ckpt.pt` | step **29999** both; `config.json` declares `encoder.image_size 256`, `in_channels 9`, `window 8` |
| GPU | — | **A40 46 GB, 0 MiB used, 0 % util** — pod2 is **NOT training** (only `jupyter-lab` alive) |

> The train epcache composition **independently re-confirms parity**: 2376 episodes + 24
> decode-failure skips = the 2400 train clips of the canonical
> `physicalai-train-e438721ae894` / skip-hash `f09e44db` split.

### Thor (`ssh tanitad-thor`)

- `~/valdata/physicalai-val-0c5f7dac3b11-w120-256x640cyl` (the w120 cache), **and now**
  `~/valdata/physicalai-val-0c5f7dac3b11/` with the relayed 256px episodes (§5).
- `~/models/flagship-v1-speedjerk/ckpt.pt` (md5 `b5f07d9e3dd2ca643949bc86832e6585`),
  `~/models/refc-base`, `~/models/refc-xl`, `~/models/flagship-v4.2b`, `~/models/v5f`.
- **835 GB free** on `/` — the full 600-episode val (65.56 GiB) would fit ~12× over.
- Edge venv: torch **2.13.0+cu130**, torchvision 0.28, numpy 2.5.1, CUDA available,
  GPU `NVIDIA Thor`, **sm 11.0**.

### HuggingFace

`nvidia/PhysicalAI-Autonomous-Vehicles` — **reachable** from Thor with the Keys.txt
token: `gated:"auto"`, `private:false`, **70,775 files**. MEASURED 2026-08-02.
⇒ A genuine from-source rebuild remains possible even if pod2 were lost. (The CLAUDE.md
note that *HF has been 403-storage-full* concerns **push** quota on our own repos; **pull**
of NVIDIA's source dataset is unaffected.)

---

## 3. Parity implication

Canonical corpus `physicalai-train-e438721ae894` (2376 eps, skip-hash `f09e44db`);
canonical clean val `physicalai-val-0c5f7dac3b11` (600 built).

**Nothing in this plan re-selects episodes.** That is the whole point of preferring the
surviving epcache over a rebuild: a rebuild re-runs `discover_r0_clips` →
`split_clips(val_frac=0.2, seed=0)` and any drift in clip discovery silently produces a
*different* val set with the *same* directory name — the exact failure
`tanitad/data/parity.py` was written to catch.

`parity.py` registers **two admissible val deployments** and nothing else:

- **600** — full build (the epcache split dir on the training pods)
- **40** — *"canonical TanitEval deployment → 881 stride-8 windows (THE published
  open-loop statistic)"*

and `check_uids(mode='subset')` requires the deployed set to be a **sorted PREFIX** of the
full build. So the admissible 40-episode deployment is exactly
`ep_00000.pt … ep_00039.pt`.

---

## 4. How episode identity is verified — exactly

Four checks. **Three are content-level**; the fourth (the manifest digest) is not, and
§4.3 shows why.

### 4.1 Deployment guard — `parity.assert_val_cache` (MEASURED, run on pod2)

| cache | result |
|---|---|
| the full 600-ep epcache | **PASSED** — `episodes_loaded 600`, deployment *"full build"*, `decision_grade True` |
| a **40-episode prefix** (symlinks in `/tmp`) | **PASSED** — deployment *"canonical TanitEval deployment → 881 stride-8 windows (THE published open-loop statistic)"*, `decision_grade True` |
| **negative control: 39 episodes** | **REFUSED** — `PARITY VIOLATION … 39 present — NOT a registered val deployment; registered: [40, 600]` |

The negative control matters: it proves the guard is a real discriminator at n=40, not a
rubber stamp. Artifacts: `raw/pod2_probe.json` → `P2_parity_guard`, `raw/pod2_parity_guard.log`.

### 4.2 The 881-window reproduction — the sharp check

`taniteval/bench.py`: `WINDOW = 8`, `STRIDE = 8`, `K_MAX = 20`,
`starts = range(0, T - window - K_MAX, stride)`.

Applying that rule to the **measured `T` of the 40-episode prefix** (`[199×10, 205, 198,
199×6, 198, 199×21]`, from pod2):

```
TOTAL WINDOWS = 881      <-- the published canonical statistic, EXACT
n = 39 episodes -> 859   <-- so the count is a sharp discriminator, not a coincidence
```

⇒ The 40-episode prefix of pod2's raw epcache **reproduces the published
40-episode/881-window deployment exactly**. Artifact: `raw/window_count_check.json`.

### 4.3 ⚠️ The manifest's uid digest is NOT an identity check for val — do not rely on it

`parity_manifest.json` records `episode_uid_sha256: null` for
`physicalai-val-0c5f7dac3b11` (`uid_source: "count-only-unrecorded"`), and its
`provenance.todo` says to *"run `--record --split val` on a pod"* to fix that.

**That todo would add ZERO discriminating power, and it should not be presented as
closing the gap.** `parity.episode_uid` is the **`ep_%05d.pt` basename**, and
`uid_digest` is `sha256` over the newline-joined sorted basenames. Val has
`skip_count = 0`, so its uid set is **contiguous** — and a contiguous uid set is a pure
function of `n`. Demonstrated by deriving the digests with **no cache access at all**:

```
n=  600  sha256(sorted basenames) = 75a4d429be8cef8ea47a319e2033d792ee9eecbff033fad27dbb624b5634df20
n=   40  sha256(sorted basenames) = c36c2168e0ea6eaa0668c10003b8d97355d24d9f7363ab64f1b582328d40828a
n= 2376  sha256(sorted basenames) = 92d49accf9ec054e80b55d19d7ca2b58aaf6910ae89d4874676ac02796e4007c
```

Self-check that this reasoning is sound rather than a coincidence: the **committed train**
digest is `9877bef64da35f384b380b23ab0e760f3ef5396c6f3e849d5de81c7243ac7386`, which
**differs** from the contiguous-2376 digest above — precisely because train has 24 skip
positions and is therefore **non**-contiguous, so *its* digest does carry information.
Val's cannot.

⇒ Recording the val uid digest is harmless bookkeeping, but the real val identity
instrument is §4.4.

### 4.4 ⭐ Bit-exact cross-cache pose match — the content-level identity proof

The raw epcache stores **positions** (`ep_%05d.pt`) and no clip id. The w120 v2 cache
stores **clip ids** and no positions. `poses`/`actions` are **geometry-independent**
(both caches call the same `signals_at(ego, t_query)`), so they are a shared key that
links the two uid spaces.

MEASURED on pod2 (index over all 600 w120 clips, built in 12.3 s):

```
n_w120_clips          600
n_matched              40 / 40
n_unmatched             0
n_distinct_clip_ids    40
grades                 {'BIT_EXACT': 40}      max_abs_pose_diff == 0.0 for all 40
clip_id_sha256_sorted  be2eedcec5e966fb7d8d160c24fd0ae506e27ce8918720d2fec3ed5d7c39f997
```

and the recovered map shows **`ep_%05d` order == sorted-clip_id order**
(`ep_00000` ↔ the alphabetically-first val clip, `ep_00001` ↔ the second, …).

This is strong: it proves (a) the two independently-built caches hold the **same
episodes**, (b) the 40-prefix of one **is** the 40-prefix of the other, and (c) it
recovers the `ep_index → clip_id` map that neither cache carries alone — which is what
makes an epcache-scored arm comparable with a v2-scored arm at the episode level (and
what the **episode-cluster bootstrap** needs to cluster correctly).

> 🔒 **Confidentiality.** The clip ids themselves are gated PhysicalAI-AV content
> (`parity_manifest.json` → `clip_membership.provenance.confidentiality`: *only these
> digests are quotable in the repo*). `raw/pod2_probe.json` therefore ships the map
> **REDACTED** — per-episode 8-hex `clip_id_sha256` prefixes plus the sorted-set digest
> above. The un-redacted map exists only on pod2 and can be regenerated in 12 s with
> `code/pod2_probe.py`.

### 4.5 Transfer integrity

`md5sum` on **both** ends, per episode. MEASURED for the 5 relayed episodes — all
identical (e.g. `ep_00000.pt 47ee0e683ad2ad78f2368245d641d334` on pod2 **and** Thor).

---

## 5. The Thor proof — 5 episodes, relayed and LOAD-VERIFIED

**No rebuild was performed, because none is needed.** The correct operation is a
**transfer of the surviving artifact**; §1b already established a rebuild would have to
start from the mp4s.

Relay path: `pod2 → dev-box → Thor`, streamed through a pipe (no local staging), source
`ssh` run with `-n` so the nested `ssh` cannot eat the script's stdin (CLAUDE.md trap).
Script: `code/relay_val256.sh` — resumable (skips any file already present at the right
size).

MEASURED per-episode: 99 s, 149 s, 151 s, 132 s, 139 s → **670 s for 586,916,280 B =
0.835 MB/s aggregate**, all five `src == dst` bytes.

**Load-verify on Thor** (`code/thor_loadverify.py`, output `raw/thor_loadverify.txt`) —
this is a real load and a real forward, not an exit code:

```
host thor6 · torch 2.13.0+cu130 · GPU "NVIDIA Thor" · sm [11, 0]
5 relayed episodes, md5 verified, each frames [199, 9, 256, 256] uint8

flagship v1 (speedjerk) strict-loaded:
  step 29999 · encoder.pos [1, 256, 768] · action_dim 3 · state_dim 2048 · 263.44 M params
  cfg frame = {256, 256, f_ref 266.0, pinhole}

FlagshipWindowDataset(5 eps): 855 windows, 110 at stride 8, window 8, max_horizon 20

FORWARD: input [4, 8, 9, 256, 256] -> encoder_out [32, 256, 768]
         finite True · mean 0.0036001901607960463 · std 0.6800031065940857 · 0.692 s
```

**Cross-host agreement.** The identical batch on pod2's A40 gave
`mean 0.0036001908592879772`, `std 0.6800031065940857` — std identical to every printed
digit, mean agreeing to **7 significant figures** (Δ ≈ 7e-10, float32 reduction-order
noise across two different GPUs). The relayed corpus is not merely loadable on Thor; it
reproduces pod2's numbers.

**Negative control on Thor:** `assert_val_cache` on the 5-episode dir **REFUSED**
(*"5 present — NOT a registered val deployment; registered: [40, 600]"*), which is
correct behaviour and confirms the guard travels with the data.

### Two hazards found while doing this

1. ⚠️ **A relay in flight is invisible to the parity guard.** The first Thor run globbed a
   **partially-written** `ep_00003.pt`; `assert_val_cache` printed *"count OK (4/600)"*
   and did **not** refuse, because the val content check is COUNT-ONLY (§4.3). It then
   died inside `torch.load` with *"PytorchStreamReader failed … failed finding central
   directory"*. Loud, so not dangerous here — but **never launch an eval against a
   directory a transfer is still writing into**; wait for the relay's completion marker
   and md5 every file.
2. ⚠️ **Thor's `stack/` checkout is DRIFTED from repo HEAD** — MEASURED md5, on exactly
   the four modules that decide eval geometry and parity:

   | module | repo HEAD **and** pod2 | **Thor** |
   |---|---|---|
   | `tanitad/data/calib.py` | `041c600e7264685e38545994c599e2a4` | `2f042d633e0d70596f6cabaae8011c95` |
   | `tanitad/models/fourbrain.py` | `aac64b00da260a11e6e22da86f5adaee` | `0199033585a92654fe1c97da97e9a591` |
   | `tanitad/config.py` | `e31afdd9baa82b629c24087319d740cc` | `cc6ad949a2abbb576a1e10b8ccc493ef` |
   | `tanitad/data/parity.py` | `7b42991893b27182c2384e79349d4acf` | `eb8c2dad86df51be7646a1df0674617b` |

   Thor is at `4954544`; the repo is at `63ae826`. The forward above still agreed
   numerically, so the drift does not touch the encoder path — but per CLAUDE.md
   (*"a launch from a drifted checkout resurrects fixed bugs"*) **Thor MUST be synced and
   re-verified by a real `import` before any decision-grade eval is run there.**
   pod2's checkout is at `0f93b98` overall, but the four modules above are **md5-identical
   to HEAD**, which is why the pod2 results in this document are quotable.

---

## 6. Cost estimate

### Option A — score on pod2 (RECOMMENDED)

| item | cost |
|---|---|
| data to move | **0 GB** |
| data prep wall-clock | **0 min** |
| GPU | A40 already idle, 46 GB free |
| val available | full **600**, or the registered **40**-prefix |

pod2 has the corpus, both checkpoints and a free A40. Nothing needs to be built.

```bash
# 40-episode canonical deployment, straight off the surviving epcache
E=/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11
mkdir -p /workspace/val40/physicalai-val-0c5f7dac3b11
for i in $(seq 0 39); do f=$(printf ep_%05d.pt $i); ln -sf $E/$f /workspace/val40/physicalai-val-0c5f7dac3b11/$f; done

OMP_NUM_THREADS=6 PYTHONPATH=/workspace/TanitAD/stack \
python3 /workspace/TanitAD/stack/scripts/eval_flagship_v4.py \
  --ckpt /workspace/v4gate30k/v1_speedjerk_ckpt.pt \
  --val-cache /workspace/val40/physicalai-val-0c5f7dac3b11 \
  --episodes 40 --stride 8 --batch 16 --device cuda
```

⛔ **Before running it: sync pod2's `stack/` to HEAD and re-verify with a real `import`.**
The four modules this eval reads are md5-clean, but the checkout as a whole is at
`0f93b98`, and the drift rule is a runbook step, not a nicety.

### Option B — ship the canonical 40 to Thor (IN PROGRESS)

| item | measured |
|---|---|
| bytes, 40-episode prefix | **4,697,690,000 B = 4.375 GiB** |
| relay throughput (streamed, pod2→dev→Thor) | **0.835 MB/s** aggregate |
| **wall-clock, 40 episodes** | **≈ 5,363 s ≈ 89 min** (5 already done ⇒ **~78 min remaining**) |
| Thor disk after | 4.4 GiB of 835 GB free — negligible |

Leg rates for reference: pod2→dev-box **2.31 MB/s** (67.1 MB / 29.1 s); dev-box→Thor
**2.69 MB/s** (67.1 MB / 25.0 s). The streamed end-to-end rate is bounded by the serial
chain, ≈ `1/(1/2.31 + 1/2.69) = 1.24 MB/s` theoretical, **0.835 MB/s observed**.

**A launched relay of all 40 is running** (`code/relay_val256.sh 40`, resumable).

> **This retracts the inherited "~1 MB/s dev-box relay" as a single number**: the
> single-hop legs are **2.3–2.7 MB/s** and only the two-hop stream is ~0.8 MB/s.

### Option C — direct Thor ← pod2 (FASTEST, but GATED)

The C56 recipe would cut the dev-box out entirely (CLAUDE.md measured **42 MB/s**
pod↔pod cross-datacenter; ⇒ 4.375 GiB in **≈ 2 minutes**). pod2's direct endpoint is
`69.30.85.123:22091` (`RUNPOD_PUBLIC_IP` / `RUNPOD_TCP_PORT_22`, read from
`/proc/1/environ`), and Thor already has a keypair
(`~/.ssh/id_ed25519.pub` = `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIL/mukI29h1x/Pemd32f/9P/BKum4XGY1ieizgaB8Ev nvidia@thor6`).

⛔ **NOT DONE — needs an explicit decision.** It requires appending Thor's **public** key
to pod2's `~/.ssh/authorized_keys`, which is a **configuration change on a host the brief
designated READ-ONLY**. That is outside what an agent should do unilaterally. The exact
one-liner, for whoever authorises it:

```bash
ssh tanitad-pod2 'echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIL/mukI29h1x/Pemd32f/9P/BKum4XGY1ieizgaB8Ev nvidia@thor6" >> ~/.ssh/authorized_keys'
# then, ON THOR:
scp -P 22091 root@69.30.85.123:'/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11/ep_000{0,1,2,3}?.pt' \
    ~/valdata/physicalai-val-0c5f7dac3b11/
```
(⛔ never copy a private key; the DIRECT port, never the `ssh.runpod.io` proxy.)

### For reference — the full corpus, if it is ever wanted

| corpus | bytes | at 0.835 MB/s | at 42 MB/s (Option C) |
|---|---|---|---|
| val 40 (canonical) | 4.375 GiB | ~89 min | ~2 min |
| val 600 (full build) | 65.56 GiB | ~22 h | ~28 min |
| train 2376 | 259.64 GiB | ~87 h | ~1.8 h |

### Cost of an actual from-source rebuild (only if pod2 is lost)

Not required today, and **not** costed by measurement here — it would mean re-pulling
camera chunks from HF and re-running `build_compressed`/`build_episode` per clip.
Flagged as **ESTIMATED, not measured**: the dominant term is per-clip mp4 decode +
`grid_sample`, and the program's own build logs are the right source for a real number.
⚠️ A from-source rebuild is also **the one path that can break parity** (§3) and would
have to reproduce `e438721ae894` / `0c5f7dac3b11` before anything built from it is
quotable.

---

## 7. Recommendation

1. **Score REF-C and flagship v1 on pod2, now.** Zero rebuild, zero transfer, idle A40,
   registered 40-episode deployment, 881 windows reproduced exactly. Sync pod2's `stack/`
   to HEAD and re-verify by real `import` first.
2. **Finish the 40-episode relay to Thor** (running) so the 256px deployment stops living
   on a single RunPod disk — the *"finish before you start"* rule. Then md5-verify all 40
   and re-run `code/thor_loadverify.py`, which should report `n_windows` consistent with
   §4.2 and `assert_val_cache` should **PASS** at exactly 40.
3. **Sync Thor's `stack/` to HEAD before any decision-grade number is produced there**
   (§5 hazard 2).
4. **Do not "fix" the val parity gap by recording the uid digest** (§4.3). If a real
   content-level val guard is wanted, register the §4.4 **pose-match digest**
   (`be2eedce…` for the 40-prefix) as the instrument.
5. ⛔ **Every eval that comes out of this must report the FOUR METRIC FAMILIES**
   (longitudinal / lateral / tactical / strategic), not an ADE horizon sweep — CLAUDE.md,
   binding 2026-08-02. This document restores the *corpus*; it does not license an
   ADE-only result.

---

## 8. Deliverable manifest

**Repo** — `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-02-val-256-rebuild/`

| file | what |
|---|---|
| `VAL_256_PLAN.md` | this document |
| `raw/pod2_probe.json` | P1 slice refusal + P2 parity guard (incl. the n=39 negative control) + P3 identity match (**clip ids REDACTED**) |
| `raw/pod2_parity_guard.log` | the guard's own stdout for the 600 / 40 / 39 cases |
| `raw/window_count_check.json` | the 881-window reproduction from the measured `T` values |
| `raw/pod2_loadverify.txt` | pod2 strict-load + forward of flagship v1 at 256px |
| `raw/thor_loadverify.txt` | Thor strict-load + forward, md5s, sm 11.0, cross-host agreement |
| `code/pod2_probe.py` | the three probes (re-runnable, read-only) |
| `code/pod2_loadverify.py` | pod2 load-verify |
| `code/thor_loadverify.py` | Thor load-verify |
| `code/relay_val256.sh` | resumable pod2→dev→Thor relay (`RELAY_LOG=… bash relay_val256.sh 40`) |

**thor:** `~/valdata/physicalai-val-0c5f7dac3b11/ep_000*.pt` — relayed 256px-square val
episodes, md5-verified against pod2 (5 complete at time of writing; a 40-episode relay is
running).

**pod2 (source of truth, unmodified):**
`/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11/` (600 eps),
`/workspace/v4gate30k/v1_speedjerk_ckpt.pt`, `/workspace/models/refc-{base,xl}-30k/`.
Only `/tmp/val40_probe` and `/tmp/val39_probe` (symlinks) were created there; no project
data, config or account setting was changed.

**Escalations (not silently parked in this doc):**
- **Option C needs an authorisation** to append Thor's public key to pod2's
  `authorized_keys` (§6C). Until then the relay runs at 0.835 MB/s instead of ~42 MB/s.
- **Thor's `stack/` drift** (§5 hazard 2) blocks decision-grade evals on Thor.
- **pod2 is idle** — an A40 sitting at 0 % with the corpus and both checkpoints on it.
