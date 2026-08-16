# SAM3 115-clip backfill — first headless Colab CLI production run

**Status: ✅ COMPLETE — far-side verified 2026-08-16 by the orchestrator after the
run agent hit a session limit mid-write. §3b/§4/§5 below are MEASURED from the
far side and the live CLI, not from the agent's report.**

| | |
|---|---|
| what | re-run the SAM3 leg for the 115 aug120 clips whose fused records carry `perception.absent = AUG120_SAM3_STAGE_GAP` |
| how | `colab/SAM3_BACKFILL_115.ipynb` executed **headlessly** on a free-Colab T4 via the official `google-colab-cli` (native Windows + termios shim), session `tanitad-backfill` |
| bank | `Sayood/tanitad-ph0-aug120` under `sam3_backfill/<clip>.json`, per-clip far-side byte-verified |
| root cause of the gap | `aug120_pipeline.py` never passed `--n` to `ph0_sam3.py` (default 4) — `AUG120_FUSION_RESULT.md` §3 |
| gap fixture | `colab/fixtures/sam3_backfill_expected.json` (115 clips, derived 2026-08-16T12:36:39Z) |

## 1. Baseline (MEASURED, dev box, 2026-08-16 before launch)

`hf_count.py` (this package `raw/`) against `Sayood/tanitad-ph0-aug120`:

```
far-side records: 0 | zero-byte: 0 | run-manifests: 0
fixture coverage: 0/115 | missing: 115 | extra-not-in-fixture: 0
```

The gap was fully open; nothing under `sam3_backfill/` existed.

## 2. Headless bring-up — what it actually took (MEASURED)

The notebook was written for the Colab UI (Drive mount + Colab Secrets). A
headless CLI session has neither; the run needed four pieces of session
bring-up, none of which changed the repo:

1. **No Drive mount** (`colab drivemount` is interactive, PI-only) ⇒ the
   16-file import closure was shipped as one tar
   (`colab/{s2_lab_lib,s2_schema}.py`, the fixture,
   `stack/scripts/{ph0_v2,ph0_sam3,ph0_pilot,v2_to_pilot}.py`,
   `stack/tanitad/__init__.py`,
   `stack/tanitad/data/{__init__,_contract,toy_driving,metadrive_env,comma2k19,stats,calib,v2_dataset}.py`
   — 119 476 B) → `/content/repo`, with `S2_REPO_ROOT=/content/repo` and the
   three import paths inserted in the persistent kernel. The notebook's
   own `import s2_lab_lib` then succeeds and its Drive-mount fallback is
   never reached.
2. **Token channel** (PI-sanctioned): token read in-process from `Keys.txt`,
   shipped as a file to `/content/hf_tok.txt`, moved into the kernel's
   `HF_TOKEN` env by the bootstrap, **file deleted on both sides in the same
   step** (local lifetime: seconds; VM copy deleted before any workload ran;
   verified `token_file_deleted=True`). The token never appeared in argv,
   in any output, in kernel code history, or in any repo file.
   `s2_lab_lib.get_hf_token()` reads Secrets → env → Keys.txt, so env wins
   headlessly (read order verified at `colab/s2_lab_lib.py:179-201`).
3. **Colab Secrets time out headlessly** (task-brief MEASURED:
   `TimeoutException`) and `get_hf_token()` retries Secrets FIRST on every
   one of its hundreds of calls (`hf_download` calls it per file) ⇒ the
   bootstrap stubs `google.colab.userdata` to raise instantly, making the
   lib's designed fallback zero-latency. Kernel-state-only; no repo change.
4. **`ph0_sam3.find_bpe()` cannot find the CLIP BPE vocab on Colab**: its
   absolute roots (`/workspace/a2venv`, `/usr/lib/python3`, `/root`) do not
   cover Colab's `/usr/local/lib/python3.12/dist-packages`, and its `**/`
   globs are cwd-relative. MEASURED: the vocab was found only under
   `/usr/local/.../dist-packages/open_clip/` ⇒ the bootstrap copies it to
   `/content/bpe_simple_vocab_16e6.txt.gz`, where glob pattern 3
   (`**/bpe_simple_vocab_16e6.txt.gz` from cwd=`/content`) hits it.
   *Left as session bring-up, not a repo patch — a repo fix belongs to
   `ph0_sam3.py`'s owner (suggested: add `site.getsitepackages()` roots).*

Preflight probe (MEASURED, before the notebook exec): `repo root
/content/repo · in_colab True · fused_records_listed 201 · cuda True ·
Tesla T4` — auth, imports, and GPU all proven before any GPU time was spent.

MSYS quirk for the record: `colab upload <local> /content/...` from Git Bash
mangles the remote path (`/content/…` → `C:/Program Files/Git/content/…`,
500 from the VM's contents API). `MSYS_NO_PATHCONV=1` + Windows-form local
path + POSIX remote path is the working combination.

## 3. Run 1 (MEASURED): failed in 114 s on the one thing smoke cannot see — and the fix is staged

Launch 2026-08-16T14:12:07Z → exited 14:14:01Z. Cells 1–7 all green on the
T4: installs rc=0 with **CUDA conv2d OK after them** (the uv-pip trap check),
gap re-derived on the VM **115/201 with cross-check + fixture check OK**,
resume listing `far side already holds 0 -> this run: 115`, v2 records
**115/115** from 25 far-side files (353 records — records counted, not
files), w120 shards located for all 115.

Cell 8 then died in ~1 s: **`ModuleNotFoundError: No module named 'iopath'`**
inside `ph0_sam3.build_processor` → `import sam3.model_builder`. Root cause:
`pip_install_colab` installs the sam3 wheel `--no-deps` (deliberate torch
protection) but never supplied sam3's **torch-free runtime deps**. Exactly
the risk RUNNER.md §8 pre-registered ("the sam3 wheel install" is T4-only,
unconfirmed by CPU smoke — smoke skips installs and stubs the GPU leg).

Measured dependency surface (VM probe, `importlib.metadata.requires('sam3')`
+ import trial): missing were `iopath` (+its pure-python closure) and
`ftfy==6.1.1`/`wcwidth`; `timm/tqdm/regex/typing_extensions/huggingface_hub`
already present. Fix applied to the live kernel and PROVEN
(`SAM3_IMPORT OK`, conv2d still OK, torch 2.11.0+cu128 untouched), then
mirrored into `colab/s2_lab_lib.py::pip_install_colab` (+6 lines), validated
by `colab/smoke_run.py` (**SMOKE PASS, 12 s**), **staged** (index content
verified to carry the fix).

Two operational learnings, both now load-bearing for any future CLI run:

1. ⚠️ **`colab exec -f nb.ipynb` executes ALL remaining cells after a cell
   error and exits rc=0.** The exit code is NOT evidence of success (run 1's
   cell 9 NameError cascade + `rc=0`). Judge only by content markers
   (`BACKFILL_DONE`, `BANKED n`) and the far side.
2. The failure cost 2 minutes and nothing else: banking had not started,
   the kernel survived, and the designed resume made the re-exec idempotent.

## 3b. Run 2 — SUCCESS (MEASURED far-side by the orchestrator)

The run agent was terminated by a session limit with the words *"the main exec
has completed — reading the completion markers"*. Rather than trust that, the
completion criterion was re-measured independently from the dev box.

## 4. Far-side verification — THE GAP IS CLOSED (MEASURED 2026-08-16)

`Sayood/tanitad-ph0-aug120`, prefix `sam3_backfill/`, **records counted, not
files** (C18):

```
far-side files      116   (115 clips + 1 run manifest)
zero-byte            0
banked ∩ fixture   115 / 115
missing              0
extra                1  ->  sam3_backfill/_runs/20260816-145005-sam3-backfill.json
                            (the run manifest — identified, not assumed)
total bytes    730,019   (mean 6,293 B/clip)
```

**Round-trip on 3 randomly sampled clips** (seed 0): `clip_id == filename` on
3/3, real payload keys `frames · per_concept_hits · n_frames_run · n_det_total ·
vlm_cross_check`, with **5–7 frames per clip**.

⭐ That frame count is the defect's own signature, inverted: the gap existed
because `aug120_pipeline.py` never passed `--n` to `ph0_sam3.py`, leaving the
default **4**. The backfilled records carry **5–7**, i.e. the re-run did not
merely re-fill the hole, it filled it with the coverage the original stage was
supposed to produce.

## 5. Session hygiene (MEASURED)

`colab sessions` → `[colab] No active sessions found on server.` The T4 was
released; no session leaked past the run.

## 5b. What this run PROVES beyond its own payload

**The first production workload ever executed on Colab from this box, headlessly,
end to end.** The chain — authenticate → rent a T4 → ship an import closure →
inject a token with seconds-long lifetime → run a GPU pipeline → bank per clip →
verify far-side → release the VM — is now MEASURED, not hypothesised. The
label-lab iterations the PI asked for run on exactly this path.

## 6. Escalation (standing, fires when §4 confirms)

The 115 fused records in `fused_aug120/` still carry `perception.absent` —
the backfill banks the SAM3 legs but does NOT re-emit fusion. **Owner:
aug120-fusion package** (`…/2026-08-15-aug120-fusion/AUG120_FUSION_RESULT.md`
§9 items 1-2; the fuser resumes per clip). Escalated in the run report of
this agent, not buried here.

## Deliverable manifest

| artifact | where |
|---|---|
| this report | `TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-16-sam3-backfill-run/SAM3_BACKFILL_RUN.md` |
| exec log (full stdout/stderr of the headless run) | `…/2026-08-16-sam3-backfill-run/raw/exec_run1.log` |
| executed notebook with outputs | `…/2026-08-16-sam3-backfill-run/raw/SAM3_BACKFILL_115_output.ipynb` |
| far-side verifier | `…/2026-08-16-sam3-backfill-run/raw/hf_count.py` |
| kernel bootstrap (headless bring-up, no secrets) | `…/2026-08-16-sam3-backfill-run/raw/bootstrap.py` |
| banked records | HF `Sayood/tanitad-ph0-aug120` `sam3_backfill/*.json` + `sam3_backfill/_runs/<ts>-sam3-backfill.json` |
