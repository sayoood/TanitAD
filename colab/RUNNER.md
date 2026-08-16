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

⚠️ **Honest limitation:** `colab-cli`-style GitHub tools can sync and open
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
| SAM3 image model | ~3 GB | ESTIMATED (task brief); unmeasured on T4 |
| ego leg | 0 (CPU) | by construction |

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
