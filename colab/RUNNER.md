# Colab S2 label lab — operator guide

Three artifacts, all in this directory, all thin wrappers over `s2_lab_lib.py`
with the vocabulary in `s2_schema.py` (**PROVISIONAL** — see §7):

| artifact | what it does | GPU |
|---|---|---|
| `SAM3_BACKFILL_115.ipynb` | re-runs the SAM3 leg for the **115** aug120 clips whose fused records carry `perception.absent` (root cause: `AUG120_FUSION_RESULT.md` §3) | ~30 min T4 |
| `STRATEGIC_LABEL_LAB.ipynb` | VLM (unsloth 4-bit Qwen3.5-9B) + SAM3 + ego legs → fused `g_str`/`a_str` with per-token provenance → **review sheet** | ~2–4 min/clip (ESTIMATED, unmeasured on T4) |
| `smoke_run.py` | executes either notebook's cells on CPU with GPU stubs — the plumbing test that already ran green on the dev box | none |

Run the **backfill first**: it is small and known-shape, and it validates the
whole loop (auth → pull → GPU → bank → far-side verify → resume) before the
lab spends GPU time on open-ended review.

> ⛔ **SUPERSEDED FOR THE SAM3 LEG, 2026-08-16 — `SAM3_BACKFILL_115.ipynb` IS v1.**
> The SAM3 extraction is now **schema v2** (contours + oriented extents + a
> separate scene-concept channel + a derived ego lane) at
> `confidence_threshold=0.25`, and it banks to **`sam3_backfill_v2/`**, a
> different prefix, so the two detection floors are never mixed in one corpus.
> **The v2 path is SCRIPT-driven, not notebook-driven** — a headless CLI session
> has no Drive mount, so the notebook's first cell never applies there anyway:
> `…/incoming/2026-08-16-sam3-extraction-v2/code/{ship,bootstrap_v2,p1…p4}.py`,
> operated exactly as §8 of `SAM3_EXTRACTION_V2.md` describes.
> This notebook is left at v1 **deliberately and is NOT the way to (re)produce the
> corpus** — rewriting it to v2 without re-running its own smoke would have made
> two divergent entry points, and the scripts are the ones that were MEASURED.
> Two facts from that work bind anything run through this file:
> * ⚠️ **`PYTHONUTF8=1` on every `colab` invocation.** colab-cli 0.6.0 opens the
>   script with the locale codec (cp1252 on the dev box) and any file carrying a
>   ⛔/⚠️ dies with `UnicodeDecodeError … byte 0x8f` before a line reaches the VM.
> * ⚠️ **A re-ship is a NO-OP without a `sys.modules` purge.** The kernel persists
>   across `colab exec`, so extracting a newer file over an already-imported
>   module changes nothing — `import ph0_sam3` returns the cached one and any
>   verify-gate passes on the OLD code. Pod-checkout drift with a Colab accent.

## 1. Opening the notebooks (the repo IS on the PI's Drive)

This repo lives at `My Drive/SayBouBase/raw/Projects/TanitAD`, so the
notebooks are **already in Colab's reach with no sync tooling**:

* drive.google.com → navigate to `…/TanitAD/colab/` → double-click the
  `.ipynb` → *Open with → Google Colaboratory*; or
* colab.research.google.com → *File → Open notebook → Google Drive* → search
  the filename.

Edits made to `colab/*.py` or `stack/scripts/*.py` from the dev box reach the
mounted Colab runtime after the usual Drive sync delay (seconds–minutes);
`importlib.reload(...)` picks them up without a runtime restart.

> **SUPERSEDED 2026-08-16** — the paragraph below is retained for the record
> (retraction discipline; root-cause class: *stale absence-claim about external
> tooling — re-probe vendors before repeating*). Google has since shipped an
> **official Colab CLI** (`google-colab-cli`: headless `colab exec -f nb.ipynb`
> writes `<basename>_output.ipynb`; `colab run` = provision→exec→teardown; T4
> selectable) and an **official open-source Colab MCP server**
> (`googlecolab/colab-mcp`: agent drives the notebook in the PI's signed-in
> browser tab). Both are installed on the dev box and measured to the auth
> boundary. Findings + the PI's 2-minute auth sequence:
> `colab/COLAB_CLI_MCP.md`; ready Claude Code config: `colab/MCP_SETUP.md`.
> Still true and unchanged: **nothing executes until the PI authenticates**
> (CLI OAuth paste-code flow, or being signed in to Colab in the browser for
> MCP pairing), and the CLI is unsupported by Google on native Windows
> (WSL or shim — see the findings doc).

⚠️ **Honest limitation (pre-2026-08 state, superseded above):** `colab-cli`-style
GitHub tools can sync and open
notebooks but do **not** execute them headlessly — free Colab has no official
headless-execution API. Execution is driven from the Colab UI by the PI, or
by the orchestrator's browser pane **after the PI has authenticated** in that
browser profile. Plan reviews accordingly; nothing here assumes unattended
Colab runs.

## 2. One-time setup per Colab account

1. **Runtime → Change runtime type → T4 GPU.**
2. **HF token via Colab Secrets** (key icon in the left sidebar): add a
   secret named exactly `HF_TOKEN`, paste the program's `Sayood/`-write
   token, and enable *Notebook access*. The code path is
   `google.colab.userdata.get("HF_TOKEN")` — the token is **never**
   hardcoded, printed, or written to a cell output. Use the account whose HF
   token has (a) write on `Sayood/` and (b) the granted `facebook/sam3`
   access (MEASURED working 2026-08-16: `config.json` downloads, 25 843 B).
3. Run all cells. The first cell of each notebook prints its parameters; the
   Drive-mount cell will ask for the usual Drive authorisation.

Dependency installs run inside the notebook (Colab-only, skipped in smoke):
`unsloth`, `lm-format-enforcer`, `qwen-vl-utils`, `sam3` (**`--no-deps`** —
its dependency closure includes torch, and an unpinned install can silently
replace Colab's torch with a wheel the driver cannot run; same trap class as
the pod incident in CLAUDE.md), `open_clip_torch` (**`--no-deps`**, ships the
CLIP BPE vocab SAM3 needs), `imageio(-ffmpeg)`. After installs the notebook
verifies torch with a **real conv2d on CUDA**, not `import torch`.

## 3. Session death and resume — the design assumption

**Free T4 sessions die without warning.** Both notebooks are built for it:

* every clip is banked to HF **immediately** after it is produced, and every
  push is **far-side verified by byte round-trip** (`force_download=True`,
  exact byte compare) — the push log is never trusted;
* every run **starts** by listing the far side and skipping what is already
  banked (`done_set`), with a sampled round-trip proving filename == the
  record's own `clip_id`;
* therefore the recovery procedure after ANY disconnect is exactly:
  **Runtime → Run all.** Never restart-from-zero, never re-derive by hand.

A run manifest is banked under `…/_runs/<ts>-<tag>.json` at the end of every
(re)start, so partial sessions stay accountable.

### 3b. ⛔ THE COMPLETION CRITERION IS THE CENSUS, NOT THE FILE COUNT (C77, 2026-08-16)

The first production run banked **115 well-formed records containing ZERO
detections** — SAM3 raised `RuntimeError: mat1 and mat2 must have the same
dtype, but got BFloat16 and Float` on every concept of every frame and the
pipeline faithfully recorded it. Record count, zero-byte scan, and a 3-clip
`clip_id == filename` round-trip **all passed**. Root cause + fix:
`…/incoming/2026-08-16-sam3-dtype-fix/SAM3_DTYPE_FIX.md`.

Two things changed and both are now in the notebook, not in an operator's head:

* **`L.content_census(api, repo, prefix, want=…)`** reads every banked RECORD
  and returns `n_det_total · per_concept_totals · error_census ·
  clips_with_zero_det · liveness_live/dead · pass_`. Cell 9 refuses to print
  `BACKFILL_DONE` on a real run unless `pass_` is True; the **resume** in cell 5
  uses the same criterion, because a stem-based resume would have skipped all
  115 empty records as "done".
* **The liveness positive control.** Every AGENT concept (`car`, `pedestrian`,
  …) may legitimately be zero on a frame, which is why 115 empty clips looked
  plausible. `road` and `sky` cannot, so `ph0_sam3` runs them **once per clip**
  and banks `liveness.live`; a zero there is an ALARM
  (`SAM3_LIVENESS_ALARM`, exit 1), and `--no-liveness` is the only way off.
  ⇒ **The fix for a perception outage is proven by a NON-ZERO detection on the
  control, never by the absence of a traceback.**

## 4. Where things bank

| mode | repo | prefix |
|---|---|---|
| backfill, real | `Sayood/tanitad-ph0-aug120` | `sam3_backfill/<clip>.json` |
| lab, real | `Sayood/tanitad-s2-lab` | `lab_v0/<clip>.s2.json` + `lab_v0/_sheets/` |
| any, smoke | `Sayood/tanitad-s2-lab` | `smoke/…` (never the production label repo) |

## 5. T4 memory budget (16 GB) — sequential legs, never together

| leg | expected peak | basis |
|---|---|---|
| Qwen3.5-9B, unsloth `load_in_4bit` | ~7–8 GB | ESTIMATED from 4-bit weight math (9B × 0.5 B/param + activations); **unmeasured on T4 until the first real run prints it** |
| SAM3 image model, **weights loaded** | **3.58 GB** | ⭐ **MEASURED** on the T4, 2026-08-16 fixed re-run (`gpu_mem_report('sam3 load')` = `torch.cuda.max_memory_allocated`) |
| SAM3 image model, **inference peak (weights + activations)** | **4.01 GB** | ⭐ **MEASURED** after `reset_peak_memory_stats()`, one clip, 6 frames × 7 concepts (`raw/encode_once_equivalence.json`) |
| SAM3 image leg, **wall-clock** | **~21 s / 6-frame clip** | ⭐ MEASURED; it was 97–98 s before the encode-once fix (`raw/eq3_whole_clip.json`) |
| SAM3 image leg, **schema v2**, wall-clock | **29.8 s / clip** (v1 at the same conf: 24.0 s; v1 at conf 0.5: **23.0 s**) | ⭐ MEASURED 3-arm pilot, one session, five clips — `…/2026-08-16-sam3-extraction-v2/raw/p2_pilot_size.json` |
| SAM3 image leg, **schema v2**, peak | **4.241 GB** (v1: 4.231 GB) | ⭐ MEASURED after `reset_peak_memory_stats()` per arm — the schema costs **+10 MB**, memory is not the constraint |
| SAM3 **record size**, schema v2 | **120 005 B/clip compact** (317 109 B at `indent=1`; v1 as banked 49 257 B) | ⭐ MEASURED — **64 % of an `indent=1` v2 record is whitespace**, so `bank_json(..., indent=None)` is what the v2 run uses |
| the **bridge** (12-clip batch: shard pull + mp4 re-encode) | **~165 s** (first batch ~890 s: cold caches + `records.parquet`) | ⭐ MEASURED from far-side commit timestamps — the bridge, not the GPU, is now the backfill's bottleneck |
| ego leg | 0 (CPU) | by construction |

⚠️ **The pre-2026-08-16 peaks in any report are contaminated and must not be
quoted.** `max_memory_allocated` is a **process-global** counter: the diagnosis
session built several processors in one kernel, so its "14.2 GB peak for one
`set_image`" was the sum of everything resident, not the leg's cost. It is also
how the first production attempt OOM'd at model load with 14.5 / 14.6 GiB in
use. ⇒ **one processor per kernel, `reset_peak_memory_stats()` before the leg
you are measuring, and a `free_leg()` between legs.**

The notebooks enforce the discipline rather than hoping: each leg ends with
`gpu_mem_report(tag)` printing `torch.cuda.max_memory_allocated()` (the only
admissible in-process probe — CLAUDE.md, Thor trap) and `free_leg(...)`
(`del` + `gc` + `empty_cache`) before the next leg loads. **Read the printed
numbers on the first real run and update this table** — it is deliberately
labelled ESTIMATED until then. If the VLM leg OOMs: reduce frames per clip
(`sample_clip_frames` px/fps) before touching anything else.

Other Colab budgets: disk pulls are bounded to one batch of w120 shards
(`BATCH × ~36 MB`, deleted after bridging); `records.parquet` is 26 MB once
per session; the token-enforcer vocab table build is a one-time full-vocab
sweep (a few minutes of CPU at VLM load).

## 6. Parameters (env vars or edit the first cell)

| var | default | meaning |
|---|---|---|
| `S2_SMOKE` | `0` | `1` = CPU plumbing test: GPU legs stubbed, banking to `smoke/` |
| `S2_N` | backfill: whole gap · lab: 4 | clips this run |
| `S2_BATCH` | 12 | backfill: shards pulled per batch |
| `S2_CLIPS` | *(auto)* | lab: comma-separated clip ids (default: SAM3-covered pool) |
| `S2_VLM_MODEL` | *(resolver)* | override the VLM id outright |
| `S2_ALLOW_FALLBACK` | `0` | `1` = permit the older Qwen3-VL-8B generation, loudly |

VLM resolution order (MEASURED existing on HF 2026-08-16):
`unsloth/Qwen3.5-9B` → `Qwen/Qwen3.5-9B` (both
`Qwen3_5ForConditionalGeneration` — the production ph0 arm, i.e. the PI's
"qwen3.5 9B VL"; no pre-quantised 4-bit exists, so 4-bit is applied at
load). What loaded, and via which loader (unsloth, or the
transformers+bitsandbytes fallback if unsloth rejects the arch), is printed
and recorded in the run manifest — a silent substitute is impossible.

## 7. The optimise loop, and the schema swap

* **Prompts:** the VLM's B1–B4 prompts are `P_B1…P_B4` in
  `stack/scripts/ph0_v2.py`. Edit on Drive, then in the notebook:
  `import importlib, ph0_v2; importlib.reload(ph0_v2)` and re-run the VLM
  leg + fusion + sheet cells. Judge by the review sheet.
* **Vocabulary/schema:** `colab/s2_schema.py` is PROVISIONAL and is the
  **only** file that changes when the S2-gap agent's authoritative
  `…/2026-08-16-s2-strategic-gap/S2_STRATEGIC_GAP.md` lands. Both notebooks
  import every token, argument enum, mapping and validation from it. Lossy
  v6→S2 mapping edges are flagged per record (`_mapping_lossy`) and on the
  sheet.
* **Scaling** happens only after the PI judges the sheet right — then the
  same lib runs with bigger `S2_N` (or moves to a pod), the banking and
  resume semantics unchanged.

## 8. What is already validated vs what the first T4 run must confirm

**Validated on the dev box (MEASURED, 2026-08-16, smoke green end-to-end):**
auth via token → real HF pulls → gap derivation from the 201 fused records
(115/86/201, triple cross-checked) → v2 assembly (353 records, matching the
fusion run's own count) → real ego leg on a real npz → ph1_fuse fusion →
schema validation + disjointness assert → per-clip banking with far-side
byte verification → resume across two consecutive runs (run 2 skipped run
1's clip and advanced).

**T4-only, unconfirmed until the first real run:** the unsloth load path for
`Qwen3_5ForConditionalGeneration` (the transformers+bnb fallback is wired in
and printed if unsloth rejects it), the sam3 wheel install, real per-leg
peak memory and wall-clock. The first backfill run is the cheapest way to
confirm all of it.

## 9. ⛔ HEADLESS-CLI OPERATING TRAPS — MEASURED 2026-08-19, four in one session

The VLM tactical/strategic smoke hit five distinct failures before producing a
single usable generation. **Two of them were already documented above** (§SAM3
note: `PYTHONUTF8=1`; the `colab run` teardown at the head of this file) and I
hit them anyway by operating before reading. That is the first lesson and it is
procedural, not technical: **read this file before driving the CLI.**

The other three were new. All five, in the order they bite:

| # | trap | what it looks like | the fix |
|---|---|---|---|
| 1 | `colab exec -f` reads the script with the **locale codec** (cp1252) | `UnicodeDecodeError: 'charmap' codec can't decode byte 0x90` — any file with `⛔`/`→` in a comment | `PYTHONUTF8=1` on every invocation (already §SAM3) |
| 2 | **`colab run` = provision → exec → RELEASE THE VM** | your results vanish with the VM; using it to *inspect* a live session destroys that session | `colab exec` for a persistent session, or `colab run --keep`. **`colab ls` is the non-destructive lister** |
| 3 | ⭐ **MSYS mangles the REMOTE path** | `colab ls -s x /content` → `Error: File or directory not found: C:/Program Files/Git/content`. `//content` is passed through literally and the server rejects that too | **drive the CLI from PowerShell, never the Git-Bash tool.** Remote paths are POSIX and MSYS must not see them |
| 4 | ⭐ **write-once output + client timeout = TOTAL LOSS** | `TimeoutError: Timeout waiting for output` client-side; the kernel is interrupted before the single end-of-run `json.dump`; the VM's output file still holds the PREVIOUS run byte-for-byte | **persist after EVERY unit of work**, tmp+`os.replace`. A run that produced N answers must never be able to yield 0 |
| 5 | ⭐ **`restart-kernel` does not reap the old kernel** | T4 shows **8,741 MiB used / 6,172 free with nothing running**; the next 4-bit load dies with *"Some modules are dispatched on the CPU or the disk"*, which names the MODEL and reads as "too big" | `colab exec -f colab/reap_gpu.py`. Probe with `nvidia-smi --query-compute-apps`, kill by **explicit PID** |

⚠️ **Traps 3 and 4 compound into a false diagnosis.** Trap 3 produced a
"file not found", which I read as *"the script never wrote its output"* — an
absence found at ONE path form, treated as absence. The file was there. Only
after re-adopting the session and listing from PowerShell did it appear, and its
**md5 was identical to the previous run's**, which is what actually proved trap 4.
⇒ **Never conclude "the remote artifact does not exist" from one path form**, and
never take a destructive step on that conclusion.

### 9b. Recovering an ORPHANED assignment

`colab run` wipes local session state but the **server-side assignment can
survive**. It then shows in `colab sessions` as `[?] <endpoint>`, is
unaddressable by name (`stop`/`ls`/`download` all resolve through local state),
and blocks `colab new` with `TooManyAssignmentsError: Precondition Failed`.

It is recoverable — `ListedAssignment.runtime_proxy_info` carries the `url` and
`token` that `SessionState` needs, so the orphan can be re-adopted through the
CLI's own API rather than waited out:

```python
from colab_cli.common import state
from colab_cli.state import SessionState
_, assigns = state.sync_sessions()
for a in assigns:                      # re-adopt under a name you choose
    rp = a.runtime_proxy_info
    state.store.add(SessionState(name="recovered", token=rp.token, url=rp.url,
                                 endpoint=a.endpoint, variant=str(a.variant),
                                 accelerator=str(a.accelerator)))
```

Then `colab ls -s recovered /content` (**from PowerShell**) and download. This
recovered a VM that had been written off as destroyed. ⚠️ `token_expires_in` was
**3600 s** — re-adopt and pull promptly.

### 9c. ⭐ THE THINKING-MODE TOKEN CAP IS A SILENT CORRECTNESS BUG

MEASURED on unsloth 4-bit Qwen3.5-9B / T4: **~9.8 tok/s** (derived from the only
two generations that terminated early, i.e. actually emitted EOS — n=2, thin).
At `max_new_tokens=1200`, every generation that ran **130–131 s** produced
**~1,280 tokens against a 1,200 budget** — cut off mid-`<think>`, with **no
verdict line at all**. Only the two that finished at 96 s / 101 s were complete.

⛔ **A truncated thinking trace parses as ABSTAIN.** A strict parser finds no
verdict and abstains, correctly — and the resulting "the model abstained on 90 %
of clips" reads as a statement about the MODEL when it is a statement about the
CAP. That is a wrong conclusion about capability drawn from an operator setting,
and it is exactly the class this programme keeps retracting.

⇒ **Every generation records `n_new_tokens` and `hit_cap`.** Truncation must be
visible in the artifact, never inferred from wall-clock afterwards.

⇒ **Corpus-scale consequence, and it needs a PI decision.** At cap 2600
(265 s/generation, UPPER bound — many calls EOS earlier, and the running smoke
measures the real distribution):

| corpus | 3 calls (lon+lane+sign) | 2 calls (drop sign/ROUTE_TO) |
|---|---|---|
| 201 (aug120, video local now) | 1.9 T4-days | 1.2 T4-days |
| **4,522 (w120 extraction target)** | **41.7 T4-days** | 27.8 T4-days |

**A single T4 cannot deliver the 4,522-clip target.** Batch generation is the
untested lever with the largest expected win — the smoke runs `batch=1` — and
measuring it is the cheapest discriminating experiment before any scale plan.

### 9d. ⚠️ "Session lost (404/401)" USUALLY MEANS AN EXPIRED TOKEN, NOT A DEAD VM

MEASURED 2026-08-19, and I got this wrong in a report before checking. The
runtime-proxy token in `SessionState` has `token_expires_in = 3600 s`. When it
lapses, `ls`/`download`/`upload` fail with **`File or directory not found:
/content`** and `exec` prints **`Session 'x' appears to be lost (404/401).
Cleaning up.`** — and the CLI then PRUNES the local entry, which makes the loss
look confirmed. The VM and the running kernel are usually **fine**.

⇒ **Distinguish them before reacting, because the correct responses are
opposite:**

| observation | meaning | action |
|---|---|---|
| `colab sessions` lists an assignment | token expired, VM alive | re-adopt (§9b) — mints a fresh token, `/content` reappears intact |
| `colab sessions` shows **no active sessions on server** | genuinely reclaimed | rebuild resume state from the streamed frames and relaunch |

Both happened in one session: a "lost" VM was re-adopted with `/content` and its
result file intact, and a later one was genuinely gone. Reading the first as the
second wastes a rebuild; reading the second as the first wastes a wait.

### 9e. THE RUN MUST CARRY ITS OWN DEPENDENCIES — a warm VM hides this

MEASURED 2026-08-19: the extractor ran on a VM where an earlier smoke script had
`pip install`ed unsloth, so it looked self-sufficient. On the next FRESH VM it
died instantly with `no loader succeeded: ModuleNotFoundError: No module named
'unsloth' | ... requires bitsandbytes`. **A dependency satisfied by a previous
run's side effect is not a dependency you have.**

⛔ The fix is OPT-IN (`--install-deps`), never automatic, and the reason is the
torch trap in CLAUDE.md: `pip install <anything>` can resolve torch from the
default index and replace a working build with one the driver cannot run
(MEASURED twice on pod4 — `accelerate` and `compressed-tensors` each landed
torch 2.13.0+cu130 on a CUDA-12.8 driver). Ephemeral VMs may take that risk;
Thor and pods must arrive with a correct env and never pass the flag. The
install uses `--no-deps` and is followed by a **real CUDA `conv2d`**, because
cuBLAS can work while cuDNN is broken and `import torch` proves neither.
