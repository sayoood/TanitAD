# The train-corpus obstacle join exists — 2,308 episodes, 12.1 M agent boxes, verified by its own consumer

**The blocker it clears:** the agent-slot decoder (F-18) could not be trained because
`build_obstacle_join.py`'s documented invocation builds **val40 only**. That was never a code
limitation — it is the `--episodes` default of 40. The train corpus needed the flag and the inputs.

## The result (MEASURED)

```
n_episodes          2308  of 2400 requested
n_frames         433,040
n_agent_boxes 12,122,129
visible_frac      0.4106          (inside the 120° front-camera frustum — the P4 split)
wall_s          10,618.3          (~3 h, download-bound)
md5     24cbdca8c3b23aafc2fb17e6bf99cf76
skipped   no_obstacle 79 · registration_failed 10 · bad_clip 3
```

⭐ **2308 + 92 skipped = 2400 exactly.** Every requested episode is accounted for by name; nothing
was silently dropped. The 79 `no_obstacle` are consistent with `obstacle.offline` covering ~97.4 %
of the corpus.

**Verified through the REAL consumer, not by file size** (`train_p8_occupancy.JoinFileReader`):
`{n_records: 433040, n_clips: 2308, has_occlusion_flags: True}` — matching the built counts exactly.
This is the C77 discipline: a container census would have passed on an empty file.

## Where it lives — and why it is not in git

| artifact | location |
|---|---|
| the join (136 MB) | **HF `Sayood/tanitad-ph0-aug120` → `joins/train2400_agents.jsonl.xz`** |
| its meta sidecar | same, `joins/…meta.json`, **and** `raw/train2400_agents.meta.json` here |

⚠️ **It was built into a SESSION-LOCAL scratchpad**, i.e. three hours of work one session-end away
from being lost — the stranding failure the operating standard exists to prevent. Pushed to HF and
**verified by round-trip: local md5 == far-side md5 == the md5 the join script recorded itself**,
three independent sources agreeing.

## How it was run — the part worth reusing

⛔ `build_obstacle_join.py`'s own usage says **"never on a training pod"**, and Thor is mid-30k. It
was therefore run **entirely on the dev box**, and the trick that makes that cheap is:

⭐ **The join needs POSES, not video.** `corpus_first_clips()` takes a manifest fast-path: if
`_v2manifest.pt` is present and matches the file list, it reads `man["poses"]` and **never opens a
single `.v2ep.pt`**. That manifest is **13 MB** against an **80 GB** corpus. So:

1. pull `_v2manifest.pt` from Thor (13 MB, md5-verified against source);
2. build a **metadata-only corpus mirror** — the manifest plus zero-byte placeholder files whose
   names match `man["files"]` exactly, which satisfies the glob and the equality check while the
   manifest branch returns before anything is read (verified: `glob == manifest files`);
3. get `clip_id -> chunk` from the dataset's own `clip_index.parquet` (306,152 rows, gated — use
   `tanitad.keys`), giving **198 chunks ≈ 9.9 GB** instead of the full **158 GB** of
   `obstacle.offline`;
4. run with `--selection`, which turns chunk fetching into on-demand.

⚠️ **`PYTHONUTF8=1` is required** — without it the script dies in `UnicodeEncodeError` printing its
own ⚠️ character, which is C82's shell-vs-code confusion appearing in a production script rather
than a test.

⚠️ **The background log capture was 0 BYTES for the entire 3-hour run** while the job was perfectly
healthy. Progress had to be read by counting records inside the growing `.xz`. ⇒ **For this job the
artifact is the progress indicator; the log is not.** Same family as every scope-trap in
`RETRACTION_LOG.md`: a probe whose silence was mistaken for the subject's silence.

## What it enables, and one caveat

It is the **target side** for the F-18 agent-slot decoder: per labelled frame,
`{clip_id, frame_idx, t_s, agents:[{cx, cy, yaw, l, w, occ, track_id, cls}]}` in that frame's ego
frame (+x forward, +y left), with `occ` the P4 visibility flag (0 = inside the 120° frustum).

⚠️ **The slot probe has already returned NEGATIVE (D1) at step 9000 on a 61-clip non-parity
subset** — the WM latent did not support agent extraction, and neither did raw encoder tokens. This
join does **not** change that result; what it changes is that the re-read at 30 k can run on the
**full parity train corpus** instead of a subset, which is one of the limits that reading carried.

⚠️ An ABSENT `(clip, frame)` line is **NO_LABEL** (skip and count downstream, never "road clear"); an
EMPTY `agents` list **is** a label meaning labelled-clear. That distinction is the same one whose
absence produced the `"no agents"` defect in `ph1_fuse.py`, and here it is correct by construction.

**Evidence class:** MEASURED (ours; artifacts = the jsonl on HF + the meta sidecar in `raw/`).
