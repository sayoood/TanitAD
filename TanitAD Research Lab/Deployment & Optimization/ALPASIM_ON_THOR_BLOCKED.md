# ⚠️ CORRECTED 2026-08-02 — the title below OVERREACHED

**The narrow claim survives: the NRE renderer is a closed x86_64 binary, so it cannot run on Thor
(`aarch64`).** ⛔ **But the framing "AlpaSim is blocked" was WRONG, and the probe I used tested a
path this programme does not use.**

- ✅ **AlpaSim RUNS, and we have run it — bare on one A40.** `ALPASIM_STATE.md` §1: *"Upstream
  deploys it with Docker Compose; we run it bare on one A40, which is the program's own
  contribution and the reason it works on a RunPod container at all."*
- ✅ The renderer was acquired **without Docker** — a bearer-token layer-fetch into
  `/workspace/nre/rootfs` (38 GB extracted). My `docker pull` manifest probe was therefore
  **irrelevant to how we actually obtain and run it**.
- ✅ **Closed-loop videos for REF-C (base/xl/small) and flagship v1 ALREADY EXIST** and are now
  collected at `TanitAD Research Hub/Evaluation/Videos/alpasim-closedloop/`.
- ⇒ The real constraint is **not architecture, it is that `tanitad-eval` is stopped**
  (MEASURED: `Connection refused`). Restarting it restores the whole capability.

**Root-cause class: I probed ONE acquisition path, found it closed, and generalised to the whole
capability — the "absence found at ONE location is not absence" rule, in a doc I wrote minutes
after applying that same rule to someone else.** The second probe I should have run first was
`grep -ril alpasim` over our own hub, which would have found `ALPASIM_STATE.md` immediately.

The original text follows, with its narrow aarch64 finding intact.

---

# ⛔ AlpaSim cannot run on the Jetson Thor — the renderer is amd64-only

**MEASURED 2026-08-02, four independent probes.** This is not a configuration problem and no amount
of environment work fixes it.

## The evidence

| # | probe | result |
|---|---|---|
| 1 | `docker manifest inspect nvcr.io/nvidia/nre/nre-ga:26.04` | platforms: **`amd64`** + `unknown` (attestation). **No arm64.** |
| 2 | same for tag `26.02` | no arm64 either |
| 3 | `grep -raiE 'aarch64\|arm64\|jetson\|sbsa' docs/ README.md` | **no hits** — arm64 is not a documented target |
| 4 | **`docker pull nvcr.io/nvidia/nre/nre-ga:26.04`** | ⛔ **`no matching manifest for linux/arm64/v8 in the manifest list entries`** |

`uname -m` on Thor = **`aarch64`**. The AlpaSim renderer image (NVIDIA NRE, the component that
generates the camera images the driver consumes) has **no aarch64 build**.

⭐ The image is **PUBLIC** — `docker manifest inspect` succeeded with no NGC login. Authentication
is NOT the blocker; the CPU architecture is.

## What this means

- ⛔ **No AlpaSim closed-loop run, and therefore no AlpaSim closed-loop video, can be produced on
  Thor.** Not for REF-C, not for flagship v1, not for any driver.
- ✅ Everything else we do on Thor is unaffected: TensorRT optimisation, latency profiling,
  four-family scoring on cached val, REF-C evaluation. Thor remains a first-class **evaluation and
  deployment** node. It is not a **simulation** node.
- The TanitAD driver adapter itself is **finished and proven on the contract path** — it is not
  the blocker. It will run unchanged on an amd64 host.

## What is required to unblock

**An amd64 GPU machine** (a RunPod pod) with docker, to host the NRE renderer + the wizard. The
sim and the driver can share one GPU (`topology=1gpu`).

⛔ **It cannot be pod2** — pod2 is training v5f, and the standing rule is never to add GPU/RAM load
to a training pod and never to eval on one. pod1/pod3 are terminated.

⇒ **This needs the PI to provision one amd64 GPU pod.** That is a spend decision, so it is not
mine to make.

## Second geometry finding, independent of the above

Even with a renderer, **the val data currently reachable does not match these two models**:

| model | trained raster | tokens |
|---|---|---|
| flagship v1 (`speedjerk`) | **256x256** | 256 (verified: `encoder.pos (1,256,768)`) |
| REF-C base / xl | **256x256** (`grid_shape (8,8)`) | 256 |
| v5f (in training) | 176x624 sub-frame of 256x640 cyl | **429** (verified: `encoder.pos (1,429,768)`) |

Thor and pod2 both hold ONLY `physicalai-val-0c5f7dac3b11-**w120-256x640cyl**`. The plain 256 px
`physicalai-val-0c5f7dac3b11` cache lived on **pod3, which is terminated**.

⇒ Scoring or rendering REF-C / flagship v1 against the w120 cache would repeat **exactly** the
defect retracted earlier today (REF-C's Thor numbers). Do not do it. Either rebuild the 256 px val
cache, or evaluate those arms only where their own cache exists.

## Assets that ARE now in place (so the unblock is cheap)

| asset | location | verified |
|---|---|---|
| flagship v1 `speedjerk` ckpt | `thor:~/models/flagship-v1-speedjerk/ckpt.pt` 3.31 GB | LOAD-VERIFIED |
| REF-C base / xl | `thor:~/models/refc-base`, `refc-xl` | present |
| v5f ckpt (step 1000) | `thor:~/models/v5f/ckpt.pt` 3.25 GB | LOAD-VERIFIED, `encoder.pos (1,429,768)` |
| TanitAD AlpaSim driver | `stack/experiments/alpasim-driver/` + installed on Thor | contract smoke PASSED |
| AlpaSim source + venv | `thor:~/alpasim` | imports clean |
| Thor <- pod2 direct scp | key installed | 12 MB/s measured, 3.25 GB byte-exact |

## Evidence class

| claim | class |
|---|---|
| renderer has no arm64 manifest | **MEASURED (ours)** — 4 probes incl. a real `docker pull` |
| the image is public | **MEASURED** — manifest inspect with no auth |
| flagship v1 / REF-C are 256 px; v5f is 429-token | **MEASURED** — `encoder.pos` read from each ckpt |
| only the w120 val cache exists on Thor + pod2 | **MEASURED** — `ls`/`du` on both |
| "an amd64 pod would unblock it" | ⚠️ **ESTIMATED** — follows from the manifest, not yet run |
