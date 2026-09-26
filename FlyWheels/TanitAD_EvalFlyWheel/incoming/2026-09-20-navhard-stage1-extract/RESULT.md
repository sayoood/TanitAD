# navhard STAGE-1 frames are already on this box — the official two-stage EPDMS needs an EXTRACTION, not a download

**Package** `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-20-navhard-stage1-extract/`
**Owner** EvalFlyWheel orchestrator · **Date** 2026-09-20 (Europe/Berlin)
**Status (current — read the dated section at the END for detail)** ⏳ full 32-shard extraction
**RUNNING since 2026-09-21 12:39Z** (direct launch). History: control passed 09-20 → waiter armed on W3's marker →
**disarmed** 19:51Z (it was minutes from firing into the RAM contention that killed W7's scoring) → re-gated on W7
SUCCESS → timed out 09-21 07:55Z because W7's retry had reported a FALSE success → W7 then succeeded on content
(run `859e25`, 4/4 OK) → launched directly. ⚠️ Lines below that say the waiter "starts on W3's completion marker"
describe the reasoning at the time they were written, not the current trigger.

## ⭐ The control, and why it is a control rather than a demonstration

Shard 6 was chosen because it is the **one shard whose answer was already known from an independent
mechanism**: W7 probed the archive's member listing and measured *2 of the 76 navhard logs, 120/120
of their stage-1 files, stage-2 control 0/1,185*.

**My extractor, run blind against that number: `extracted=120, already=0, rejected=0`, 2 logs**
(`2021.05.25.14.24.08_veh-25_03764_04034` → 15 jpgs, `2021.09.16.19.49.00_veh-42_00990_01609` → 105;
CAM_F0/L0/R0 each), `wall_s=958.8`, archive 6,213,453,844 B — `raw/receipts/shard_6.json`.

⭐ **The two routes are genuinely different on the archive side**: W7 read the tar *listing*, I read
the *bytes* and asserted JPEG magic on each. ⚠️ **They do share one premise** — both derive the
"needed files" set from the same `navsim_agent_inputs.json` — so this agreement confirms *what the
archive contains*, and does **not** independently confirm *what the scorer asks for*. That
distinction is exactly the sharpening W7 logged today (**two independent probes sharing one
unexamined premise are one probe**), and it is recorded here rather than glossed.

**MEASURED cost:** 958.8 s for one 6.2 GB shard while W3 streams the same disk ⇒ **~8.5 h contended**
for 32, against W7's ~1.5–2 h uncontended. The waiter therefore starts on W3's completion marker.

## The finding

W7 measured that refcv4b's **official two-stage EPDMS on `navhard_two_stage` is UNDEFINED**: stage 1
has **0 / 5,400** camera jpgs unpacked, against a same-breath stage-2 control of **65,544 / 65,544**.
That is correct about the unpacked disk state. It is **not** correct that the data is off this box.

| # | claim | evidence class |
|---|---|---|
| 1 | navhard stage 1 needs **450 tokens · 76 logs · 3,375 DISTINCT camera files** (= 5,400 token→file *references*; history frames are shared between consecutive tokens, so 3,375 distinct files satisfy all 5,400 references) | **MEASURED**, `code/extract_navhard_stage1.py --need` output |
| 2 | **76 / 76** of those logs lie inside NAVSIM v1 navtest's **136**-log universe; zero outside | **MEASURED** |
| 3 | **3,198 / 3,375 (94.76 %)** of the exact (cam, basename) pairs are also referenced by navtest's own scorer index; 41 logs partially covered | **MEASURED** |
| 4 | the remaining **177** are missing from navtest's *index*, not from the data — the archives store WHOLE logs | **MEASURED** on log `2021.05.25.14.24.08_veh-25_03764_04034` (shard 6): 3/3 index-missing files present. W7 independently strengthened this: shard 6 holds **2 of the 76 navhard logs and 120/120 of their stage-1 files, complete per log**, stage-2 control **0/1,185** |
| 5 | the 32 archives are **127,882,665,618 B, sha256 == HF `X-Linked-ETag`, 32/32 verified** | **MEASURED**, `…/2026-09-19-navhard-download/raw/receipt_navtest_camera.json` |

⇒ **The unlock is a targeted extraction of already-verified local data.** No download, no spend, no GPU.

## Why it is a FULL pass and not a targeted subset

navhard's 76 logs are spread across essentially every shard. Over the 19 shards W3 has banked so
far, **only shard 4 contains none**, and those 19 shards cover just **45 / 76** navhard logs.

⇒ ~31 of 32 archives must be streamed. Targeting by shard saves nothing.

## Cost, MEASURED under contention rather than estimated

One shard (6.2 GB gz) is taking **> 15 min** while W3's chain streams the same disk — ≈ 7 MB/s on an
external exFAT drive serving two sequential readers. ⇒ **~10 h contended vs W7's ~1.5–2 h
uncontended.** ⭐ **Waiting for W3 to finish is therefore FASTER in wall-clock than racing it**, which
is why `code/extract_when_disk_free.sh` triggers on W3's completion marker instead of starting now.
Output is **< 1 GB** (3,375 jpgs); the read is the whole cost.

## What the tool refuses to do

`code/extract_navhard_stage1.py` streams (`r|gz`, single pass, flat RSS), extracts **only** the needed
members, and asserts **CONTENT** on every file (JPEG magic + non-zero size) — not existence, and not
size alone. `--verify` re-checks the full 3,375-name set and prints exactly one of:

* `COMPLETE` — all present and content-valid
* `INCOMPLETE` — names missing, listed
* ⭐ `INCONCLUSIVE` — the probe itself could not read

⛔ **The third verdict exists because a count of 0 from an unreadable tree is indistinguishable from
a genuine absence** (`CLAUDE.md`). A silent partial extraction here would score stage 1 on fewer
frames than the scorer asks for — a plausible, quiet, wrong number, which is the poisoned-bank class
that has already cost this programme a false floor arm. The tool is resumable, so an interrupted
pass costs nothing.

## ✅ The placement step is SOLVED — a root override, not a copy (MEASURED)

I flagged this as the open, unverified gap. It is now closed by reading the loader rather than
guessing it.

`export_agent_inputs.py:60` resolves stage-1 frames against
`ORIG_SENSORS = os.environ.get("E2_ORIG_SENSORS", DATA/"sensor_blobs"/"test")` — a **first-class
environment override**, whose default that same line annotates as *"ABSENT on this box"*. Stage-1
`cams` entries are **relative** `data_path`s of the form `<log>/CAM_X/<hash>.jpg`, joined to that
root (`:141`); stage 2 uses `SYN_SENSORS` (`:166`) and is untouched.

**Test against the files actually extracted** (shard 6's two logs, every stage-1 reference for them):
`E2_ORIG_SENSORS=C:/Users/Admin/tanitad-caches/navhard-stage1-frames-20260920` ⇒ **192 / 192
references resolve**, verdict `ROOT_OVERRIDE_WORKS`. Sample `data_path`:
`2021.09.16.19.49.00_veh-42_00990_01609/CAM_L0/58689113c5d25c5f.jpg`.

⇒ ⭐ **No copy, no symlink** (D: is exFAT — symlinks are unavailable, which would have blocked the
obvious approach), and ⛔ **no mutation of the live `navhard_two_stage/sensor_blobs` tree while an
experiment is running on it.** The extraction root is pointed at, not merged in.

## ⭐ What the official two-stage number actually costs from here

Only the **450 stage-1 tokens** need model inference — A1's **5,462 stage-2 rows are already
computed and reusable** (3.12 h CPU, banked). At the run's measured 1.97 s/scene that is
**≈ 15 minutes of CPU**, not another 3 hours.

⇒ once extraction completes, the remaining path is: re-export agent inputs under the override (so
`cam_files_exist` flips true for stage 1) → run A1's stage 1 → the **official two-stage EPDMS**
becomes DEFINED, replacing the 450 declared CV stand-ins that currently force
`headline = UNAVAILABLE, n = 450`.

⚠️ Still to be asserted at that point, not assumed: that the re-export reports **5,400/5,400**
stage-1 references present, and that the resulting row is labelled a real two-stage EPDMS rather
than the `official_combined_row_HYBRID` it is today.

## Manifest

| artifact | where |
|---|---|
| `code/extract_navhard_stage1.py` (streaming, content-asserting, resumable, 3-verdict verify) | repo, staged |
| `code/extract_when_disk_free.sh` (agent-free waiter, **re-gated on W7 SUCCESS + sustained RAM**; exited 07:55Z, superseded by the direct launch) | repo, staged |
| `code/run_extract_now.sh` (the direct launch, 2026-09-21 12:39Z) | repo, staged |
| `raw/receipts/shard_NN.json`, `raw/extract_run.log`, `raw/VERIFY.txt` | produced by the run |
| staging root, **off-repo** | `C:/Users/Admin/tanitad-caches/navhard-stage1-frames-20260920` |

## ⭐ 2026-09-21 — the extraction is now the DISCRIMINATING experiment, not a completeness step. LAUNCHED 12:39Z.

W7's navhard run `…/20260921T122714Z-…-859e25` is **4/4 arms OK** on content. MEASURED from its `summary.json`,
`statistics.S2_EPDMS_u.value`, all four arms on the **identical 450 groups** (same `group_sizes`):

| arm | S2-EPDMS-u | official two-stage EPDMS (`score`) |
|---|---|---|
| **refcv4b A1** | **0.3841** | **UNDEFINED** (`n = 450` CV stand-ins in stage 1) |
| STOP | **0.4828** | 0.2985 |
| ECHO | 0.3413 | 0.1429 |
| CV | 0.3389 | **0.11481608441648 = official exactly** |

BAR-W7-1 **FAIL, sub-case FAIL-A** (beats CV and ECHO, loses to STOP: −0.0987, separated on logs) — the outcome
PRE-REGISTERED as expected. ⚠️ Beware the sibling key `stage_two_scene_mean` (A1 0.3742, STOP 0.4702): a
different statistic. I read it first by grabbing the first key matching "stage two"; the ordering and sign agree,
but only `S2_EPDMS_u` is the pre-registered one.

**Why this pass now decides something.** W7 localised the loss to ONE term: with perfect drivable-area compliance
A1 would reach **0.4875 > STOP**, and no other single term does it. Its failing plans turn **~28–30° in 4 s vs ~3°**
for passing ones at 4–8 m/s, while picking the right turn DIRECTION 94 % of the time (179/190). ⛔ **But stage 2 is
SYNTHETIC (3DGS re-rendered), so "turn geometry is wrong" cannot yet be separated from "the model is confused by
rendered frames"** — camera transfer. **Stage 1 is REAL OpenScene camera frames.** ⇒ scoring A1 on stage 1:

* if A1's drivable-area failure **persists on real frames** → shape/geometry error in the planner (a training lever);
* if it **largely vanishes on real frames** → camera transfer to synthetic scenes (an evaluation-domain effect, and
  a warning about reading stage-2 numbers as driving skill).

Both outcomes are written here BEFORE any stage-1 score exists. It also produces the first **official two-stage
EPDMS** for a camera arm, which is the number comparable to the published leaderboard.

**Launched** `code/run_extract_now.sh` (direct: the waiter's two gates — W7 success on content, RAM — were both met
by measurement at launch: FINISH_REPORT.md present, 12,204 MB free, 0 scoring processes). Resumable; shard 6's 120
files are skipped as already valid. Verdict lands in `raw/VERIFY.txt` as COMPLETE / INCOMPLETE / INCONCLUSIVE.

**Next, on `COMPLETE` only:** re-export navhard agent inputs with `E2_ORIG_SENSORS` →
`C:/Users/Admin/tanitad-caches/navhard-stage1-frames-20260920` (assert **5,400 / 5,400** stage-1 references present),
infer A1's **450 stage-1 tokens** (~15 min CPU at the measured 1.97 s/scene), reuse its 5,462 stage-2 rows and the
banked floors, score. Owner: W7 (has the seam/floor-reuse machinery).
