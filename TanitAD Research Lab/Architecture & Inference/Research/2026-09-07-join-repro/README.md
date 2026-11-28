# ⭐ VERDICT 1 — BYTE-IDENTICAL. The 317 MB B1 TRAIN agent join is reproducible by RE-EXECUTION, not merely by construction

**2026-09-07 · Architecture & Inference · zero GPU · zero HuggingFace · no pod writes · the live artifact was never opened for writing.**
Machine-readable: [`raw/repro_result_20260907.json`](raw/repro_result_20260907.json) ·
[`raw/preflight_20260907.json`](raw/preflight_20260907.json) ·
[`raw/scanA_original.json`](raw/scanA_original.json) · [`raw/scanB_repro.json`](raw/scanB_repro.json) ·
[`raw/compare_joins.py`](raw/compare_joins.py)

⛔ **This is a REPRODUCIBILITY audit. It carries no eval tier and no four-family table, deliberately.**
Nothing here predicts a trajectory. The longitudinal/lateral/tactical/strategic families bind the
refcv5-v2 *run*; stamping them on a digest comparison would be a category error.

---

## 1. The headline

The compose receipt left this open as **"reproducible by construction, unverified by
re-execution"**, and named the fix: re-run the recorded argv and compare. That was done.
**31 minutes, zero GPU.**

| | original (2026-09-06) | re-build (2026-09-07) | |
|---|---|---|---|
| ⭐ **decompressed stream sha256** | `0283df30…37fc` | `0283df30…37fc` | ⭐ **IDENTICAL** |
| ⭐ **decompressed stream md5** | `8a9277a77b132e2631b133f9bea0758e` | `8a9277a77b132e2631b133f9bea0758e` | ⭐ **IDENTICAL** |
| decompressed bytes | 3,031,865,540 | 3,031,865,540 | identical |
| **`.xz` container md5** | `1c985e6d6ad34e605c4ebd30cb353558` | `1c985e6d6ad34e605c4ebd30cb353558` | ⭐ **IDENTICAL** |
| `.xz` bytes | 317,028,572 | 317,028,572 | identical |
| rows | 849,263 | 849,263 | identical |
| clips joined / offered | 4,427 / 4,572 | 4,427 / 4,572 | identical |
| agent boxes | 28,053,187 | 28,053,187 | identical |
| clip-set sha256 | `300ad2d3…6a0c` | `300ad2d3…6a0c` | identical |
| `(clip_id, frame)` order sha256 | `27aca065…e4b0` | `27aca065…e4b0` | identical |
| ⭐ **sorted-record content sha256** | `a8267ccf…3620` | `a8267ccf…3620` | ⭐ **IDENTICAL** |

**Ten axes, ten matches.** ⭐ **This is form (1), and it does not need the qualification form (2)
would have needed** — the decompressed stream matches *and* so does the container, so there is no
"same data, different bytes" caveat to attach.

⭐ **The `.xz` container digest was read with GNU `md5sum`, not taken from the builder's own
`hashlib` print.** The builder also reported `1c985e6d…`, but a program certifying itself is not
evidence; the two independent readings agreeing is.

### 1.1 ⭐ The decompressed digest ties to something that IS in git

The committed sidecar's `digest_scope.note` records that when the scope was decided, **both**
artifacts of the file were hashed, and names the rejected one: *"the other candidate was
`8a9277a77b132e2631b133f9bea0758e`"*. That is the **decompressed** md5 — and it is what **both**
the original and the re-build hash to, measured here.

⇒ The decompressed axis is pinned to a digest **already inside version control**, not merely to a
local file that could drift. The container digest and the decompressed digest are now *both*
anchored in git, from two independent directions.

### 1.2 The answer to the actual question

> **Is it safe for this artifact to stay out of git?**

⭐ **Yes.** The premise of the risk was that two unversioned copies were the entire safety margin.
They are not: the **recipe** is the margin, and the recipe is now MEASURED to work. Everything
needed to regenerate the file bit-for-bit is committed — the argv, the builder and its four
modules, the clip list, and even the script that re-pulls the one non-trivial input directory.
A 317 MB binary in git history would be permanent and unnecessary.

⛔ **The residual risk is not this file.** It is (a) the *derived* file the launch actually reads,
which had no sidecar and no recorded digest until this document, and (b) the missing `.gitignore`
guard that currently makes committing 317 MB a one-`cp`-plus-one-`git add` accident. Both are in
§7.

## 2. Method — what was actually done, and what was deliberately not

**Done:** the argv recorded in `pod_ship_20260906.json -> b1_train_agent_join.build_argv` was
replayed **verbatim**, with exactly one change: `--out` was redirected to a new path.

```
python <private tree>/stack/scripts/build_b1_agent_join.py \
  --pose-source   reconstruct \
  --ts-dir        C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov \
  --clips         b1_train_clips.json \
  --n-stack       3 \
  --ego-dir       C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo \
  --obstacle-dir  C:/Users/Admin/tanitad-data/physicalai/labels/obstacle_offline_b1train \
  --out           C:/Users/Admin/tanitad-caches/join-repro-20260907/out/b1train_agents.jsonl.xz
```

⛔ **The live artifact was never opened for writing.** It is a launch input and a sibling may
read it at any moment; the re-build wrote to a directory that did not exist before this session.

⛔ **Zero GPU, zero HuggingFace, no pod writes.** `CUDA_VISIBLE_DEVICES=''` was set even though
`torch` is imported only inside `read_raw_poses()` — the *v2ep* path, which `reconstruct` mode
never enters. The only pod contact was three read-only `ssh -n` calls (`ls`, `md5sum`).

### 2.1 ⭐ The build was run from a PRIVATE tree, not the shared mirror

`git -c core.autocrlf=false -c core.eol=lf archive HEAD stack taniteval` was extracted to
`C:\Users\Admin\tanitad-caches\join-repro-20260907\tree\`, and every module was confirmed at
runtime to resolve **inside** it.

⚠️ **Not a nicety.** The shared mirror `C:\Users\Admin\tanitad-wt` is re-synced *from* the repo
by siblings mid-run and silently drops edits and deletes repo-absent files — it killed a 23-minute
suite once already. A 36-minute build must not read its code from a surface another agent rewrites.
The `-c core.autocrlf=false` pin is the second half of the same discipline: without it `git archive`
packs CRLF on this box and the shipped bytes stop matching the blob.

### 2.2 The five modules that decide the result, pinned by digest

| module | HEAD blob md5 | private tree md5 |
|---|---|---|
| `stack/scripts/build_b1_agent_join.py` | `2a56c451da775face871b36c2c2bccb1` | same |
| `stack/scripts/build_obstacle_join.py` | `11777e245b329517fca8bdf7b47e7d31` | same |
| `stack/tanitad/data/bev_raster.py` | `4f6edc5ce6253be85ed7fa002f8ee130` | same |
| `stack/tanitad/data/join_meta.py` | `2ddef3e46d965c9c1712d34aa30a3687` | same |
| `taniteval/taniteval/lead_source.py` | `c15c35b0c91aaac2965b5bd20b7c556a` | same |

**Commits touching any of those five since the build commit `1c46b23`: 0.**
Same-breath control: **34** commits landed on the branch in that window, so the log was really read.

⚠️ **Three of the five have a DIFFERENT raw md5 in the worktree while `git status` is clean.**
That is `core.autocrlf=true`, not drift: stripping CR makes all three match their HEAD blob
exactly, and the CR count equals the line count in each case (899/899, 541/541, 480/480).
⛔ A line-ending delta must never be read as a content delta. Comparator control: two genuinely
different files hashed unequal, so the comparison is not stuck-on-equal.

### 2.3 Why byte identity was *expected* — the determinism audit

Read off the builder before the run, so the result could be predicted and then tested rather than
rationalised afterwards:

* `lzma.open(p, "wt", encoding="utf-8", preset=6)` — **the preset is pinned in code**, Python's
  `lzma` is single-threaded, and the `.xz` container has no timestamp field. The encoder cannot
  drift between runs the way a `xz -T0` command line can.
* `want = set(...)` → **`cids = sorted(want)`** — the corpus set is sorted before use, so
  `PYTHONHASHSEED` cannot reorder the clips. `frames = sorted(fb)` does the same per clip.
* `json.dumps(rec, separators=(",", ":"))` — fixed separators, insertion-ordered dicts.
* No `Pool`, no threads, no `concurrent.futures` anywhere in the builder.
* `time.time()` feeds only the sidecar's `wall_s`; **no clock value reaches the jsonl**.

⭐ **Container fingerprint of the original, decoded from its first 32 bytes:** single stream,
single block, filter `0x21` (LZMA2), dictionary **8 MiB**, check **CRC64**. That is precisely
`preset=6`, single-threaded, and it is what makes "the compressor changed" a testable claim rather
than an escape hatch.

### 2.4 What was compared, and why the decompressed stream is the headline

⚠️ `.xz` compression is not deterministic across preset/threads, so a container digest alone can
neither prove nor disprove anything about the data. Both files were therefore streamed and
compared on five axes:

| axis | what it can detect |
|---|---|
| `xz_md5` | container identity — the digest the receipts quote |
| `raw_sha256` | **the headline**: the decompressed byte stream, order-sensitive |
| `key_seq_sha256` | `(clip_id, frame)` in file order — isolates a pure **re-ordering** |
| `multiset_sha256` | sha256 over the **sorted** per-line digests — content identity **independent of row order** |
| `clipset_sha256`, `n_lines`, `n_agent_boxes` | the counts a consumer would notice |

⚠️ Every digest was asserted **full length** (32 hex for md5, 64 for sha256) and checked against
the empty-input digests before any comparison was allowed to return "match". A short or empty
digest is **INCONCLUSIVE**, never a pass.

## 3. ⭐ The build LOG reproduced too — 4,602 lines of per-clip evidence

A final digest matching is one bit of information. The builder emits one line per clip, and those
lines are an independent, per-clip check that **the join itself** reproduced rather than some
compensating pair of errors landing on the same hash.

**`diff` of the two 4,600-line logs returns exactly 3 differing lines**, and every one is
explained:

| difference | why |
|---|---|
| the `--out` / `meta` paths in `B1_JOIN_DONE` | intended — `--out` was redirected so the live file was never touched |
| `wall_s` 1806.1 → 1868.9 | a clock, not data |
| the re-build prints **one extra line**: `[b1join] digest_scope declared: md5(compressed of …) = 1c985e6d…` | ⚠️ see §4 — the original build **predates** that print |

Everything else is byte-identical, including:

* every per-clip `n_raw / labelled / boxes / vis / dt / mis` statistic across all 4,572 clips;
* the `registration impossible (RegistrationError(…)) -> time base from camera timestamp grid`
  recovery on `feae16f1`, which reproduced exactly — including the fallback path;
* `ALIGNMENT GATE frames (KEPT clips) true_max=0.1872 · misjoin+1 min=0.8193 · sep=4.4x -> PASS`;
* `per-clip gate EXCLUDED 7 of 4434 probed clips (0.1579%, bound 0.50%)` — ⭐ **the same 7 clips,
  with the same margins to four decimals**: `065482b3` (true 1.0506 / misjoin+1 0.0017),
  `27173eb8` (0.7404 / 0.9749), `4d09c5c1` (0.2506 / 0.9993), `66a2f11a` (0.5765 / 0.9799),
  `c6b6eb9b` (0.4964 / 0.9932), `e3639a77` (0.9381 / 0.9708), `e8047d08` (0.7125 / 0.9581).
* `SKIP no obstacle parquet` — **132** occurrences on both sides.

Same-breath controls on that grep: a marker that must be present read 7 and 132; a deliberately
fake marker read 0. ⇒ The census instrument was working, so the agreements are real.

⇒ The gate did not merely pass twice; it passed **identically**, on the same clips, with the same
margins.

## 4. ⚠️ The ARTIFACT is byte-identical. The SIDECAR is not — and saying so is the point

⛔ **Reporting "everything reproduced" would be an overclaim.** The `.meta.json` differs, in three
places and only three:

| key | original | re-build | is it data? |
|---|---|---|---|
| `summary.wall_s` | 1806.1 | 1868.9 | ⛔ no — a clock |
| `args.out`, `args.argv[-1]` | the 2026-09-06 cache path | my new path | ⛔ no — I redirected it |
| `digest_scope.declared_by` | `backfill-MEASURED` | `build_b1_agent_join` | ⛔ no — a **provenance stamp**; the digest itself is identical |

⭐ **All nine other keys are identical**, including the ~4.4 MB `per_clip` block, `alignment_proof`,
`corpus`, and — the one that matters — `content_assertion`:

```
n_lines 849263 · n_clips 4427 · n_agent_boxes 28053187 · n_visible_boxes 11270675
visible_frac 0.4018 · mean_abs_cx_m 49.7568 · mean_abs_cy_m 22.1761 · unique_keys 849263
```

**Why `digest_scope` differs, and why it is good news.** The original build ran *before* the
builder learned to emit `digest_scope` itself; that field was written afterwards by
`backfill_digest_scope.py` (hence `declared_by: backfill-MEASURED` and the note naming the
rejected candidate). The committed builder now emits it natively — which is also the extra log
line in §3. ⇒ The re-build is running a **slightly newer builder that produces the identical
artifact**, which is a stronger reproducibility result than a frozen-binary match, not a weaker one.

### 4.1 ⚠️ The sidecar's on-disk size differs by 190,442 B, and that is NOT content

`ls` reports 4,482,188 B vs 4,672,630 B. ⛔ **Do not read that as drift.** `Path.write_text()`
translates `\n` to `\r\n` on Windows, so the builder writes **CRLF**; the original file was
rewritten **LF** by `backfill_digest_scope.py`. Re-serialised identically at `indent=1` the two are
**4,482,188** vs **4,482,109** UTF-8 bytes — a **79-byte** difference, which is entirely the
`digest_scope` note above.

⚠️ 190,442 is one CR per line. This is the same `core.autocrlf` trap that made 174 pod files look
like content drift in the compose receipt, and it caught me here in a completely different file
type. **A size delta of one CR per line is never a content delta.**

## 5. The recipe's inputs — what "reproducible" actually depends on

⭐ **A build is only as reproducible as its inputs, and this is where the answer is decided.**

| input | where | size | versioned? | recoverable? |
|---|---|---|---|---|
| **the argv** | `2026-09-07-refcv5-v2-compose/raw/pod_ship_20260906.json` | — | ⭐ **IN GIT** | yes |
| **the builder + 4 modules** | `stack/scripts`, `stack/tanitad/data`, `taniteval/` | — | ⭐ **IN GIT** (`1c46b23`) | yes |
| **the corpus definition** `b1_train_clips.json` | `…/2026-09-06-b1-train-join/raw/` | 182,880 B | ⭐ **IN GIT**, md5 `533d5353ffd41ed011878d615781e9f4` | yes |
| **camera timestamp parquets** | `devbox:…/camera/camera_front_wide_120fov/*.timestamps.parquet` | ⭐ **56 MB** (4,719 files) | ⛔ no | PhysicalAI, re-pullable |
| **egomotion parquets** | `devbox:…/labels/egomotion_alpamayo/` | 1,912 MB (4,800) | ⛔ no | PhysicalAI, re-pullable |
| **obstacle.offline parquets** | `devbox:…/labels/obstacle_offline_b1train/` | 2,215 MB (4,440) | ⛔ no | ⭐ re-pullable by `pull_b1_train_obstacle.py`, **which IS in git** |

⚠️ **The 58.8 GB in the camera directory is mp4 the builder never opens.** The actual timestamp
input is **56 MB**. Quoting the directory size as the input cost would overstate it ~1,000×.

⇒ The whole recipe is versioned except ~4.1 GB of upstream PhysicalAI parquets, and the script
that re-pulls the one non-public-shaped part of that (`obstacle_offline_b1train`) is itself
committed. **No step of the recipe depends on an artifact that exists nowhere but a cache.**

## 6. ⭐ The artifact is in FOUR places, not two — measured

The compose receipt says "exactly two unversioned places". That was an undercount, because the
combined launch file embeds the TRAIN join **verbatim**:

| # | location | evidence |
|---|---|---|
| 1 | `devbox:C:\Users\Admin\tanitad-caches\b1-train-join-20260906\b1train_agents.jsonl.xz` | md5 `1c985e6d…3558`, 317,028,572 B — re-verified this session |
| 2 | `tanitad-a40:/workspace/TanitAD/data/joins/b1train_agents.jsonl.xz` | md5 `1c985e6d…3558` — re-verified this session by `ssh -n md5sum` |
| 3 | **embedded** in `tanitad-a40:/workspace/TanitAD/data/joins/b1_train_plus_eval_agents.jsonl.xz` | bytes `[0, 317028572)` |
| 4 | **embedded** in `devbox:…\scratchpad\b1_train_plus_eval_agents.jsonl.xz` | md5 of the first 317,028,572 B = `1c985e6d6ad34e605c4ebd30cb353558` — **measured** |

The combined file is exactly `cat train eval`: streaming the two local parts in order gives md5
`0c31a3a63d7205e9fa50e8ac4204ef46` at **327,041,136 B**, which equals the pod's file digit for
digit. ⇒ Its reproducibility reduces entirely to its two parts, and slicing the first
317,028,572 bytes off it recovers the TRAIN join exactly.

⚠️ **But copy 4 lives in a session scratchpad** — a temp directory that is cleaned without notice.
Counting it toward durability would be a mistake.

## 7. ⛔ Findings a launcher should act on

1. ⛔ **`b1_train_plus_eval_agents.jsonl.xz` — the file `--agents head` actually reads — has NO
   sidecar and its digest was recorded nowhere.** It is written here for the first time:
   **`0c31a3a63d7205e9fa50e8ac4204ef46`, 327,041,136 B, 875,657 rows.** The TRAIN join was
   carefully documented; the derived file the launch consumes was not.
2. ⚠️ **`CONTROL_b1eval_recon.jsonl.xz` (10,012,564 B, md5 `3ddb42ec…2aed`) is the thin part of
   the chain**, not the 317 MB one: one loose dev-box copy, plus its bytes embedded at offset
   317,028,572 of the two combined files. Its *own* reproducibility is the MEASURED one
   (`reconstruct_byte_identity.json`), so this is a durability point, not a provenance one.
3. ⚠️ **Nothing in `.gitignore` guards the 317 MB artifact.** `git check-ignore` matches no rule
   for `…/2026-09-06-b1-train-join/raw/b1train_agents.jsonl.xz`. It stays out of git today only
   because nobody has copied it into the repo tree. One `cp` into a `raw/` directory followed by a
   routine `git add` would put 317 MB into history irreversibly. **A `*.jsonl.xz` ignore rule with
   a `!*.meta.json` exception costs one line and removes the whole failure mode.**
4. ⚠️ **The shared git index holds a stale blob for `stack/scripts/build_b1_agent_join.py`.**
   `git status` reads `MM`; `git diff --cached HEAD` reports 45 insertions / 464 deletions, i.e.
   the index blob `ddacde6a` is *older* than HEAD's. The **worktree file equals the HEAD blob
   exactly** (`2a56c451da775face871b36c2c2bccb1`), so the code is fine — this is the known
   `mm_commit.py` private-scratch-index side effect. ⛔ **Reported, not unstaged**: on a shared
   agent branch, silently rewriting a sibling's index is how phantom deletions get committed.

## 8. ⚠️ Two mount incidents, logged because they change how a reader should treat this box

* **G: gave a self-contradictory answer.** `cp` refused with `File exists` while `stat` on the same
  path in the same breath said `No such file or directory`, and the file was genuinely absent.
  A single retry landed it, verified by `git hash-object`. ⇒ **A failed write on this mount is not
  evidence of anything; verify by content and retry.** Every file banked here was hash-verified
  after copying, and one of them needed the retry.
* All banked files were authored on local disk and copied to G: in one operation — never edited in
  place — because a mid-write flap truncated a 17 KB file to 0 bytes earlier tonight.

## 9. Housekeeping

The re-built artifact and the private code tree were **deleted** after the comparison
(`C:\Users\Admin\tanitad-caches\join-repro-20260907\`). The digests survive in `raw/`; the 317 MB
copy does not need to. ⛔ The live artifact, the pod, and the pod's joins directory were not
modified.
