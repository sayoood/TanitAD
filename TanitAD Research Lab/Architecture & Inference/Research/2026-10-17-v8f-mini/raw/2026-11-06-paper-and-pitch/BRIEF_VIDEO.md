# Brief: a high-quality motion-graphics pitch video for TanitAD v8F (investor audience)

**The ask (PI, Dr.-Ing. Sayed Bouzouraa, in chat, 2026-11-06):** a high-quality motion-graphics video that shows the approach, its
results and its background, usable as a pitch for investors. Include the ideas not yet implemented, as outlook. Do not dwell on negative
results. Motivate why more research and engineering work is needed.

## Rules
* Read `E:/Projects/TanitAD/Project Steering/AGENT_OPERATING_STANDARD.md` (preamble).
* Never commit or push. Never delete: move aside. LF line endings for scripts.
* CPU only (`CUDA_VISIBLE_DEVICES=-1`). No downloads, no network, no new packages.
  * Tools on this box: Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I` with matplotlib 3.11, numpy, PIL and imageio
    (+ imageio_ffmpeg); system ffmpeg 8.1 is on PATH (WinGet Gyan.FFmpeg).
* Do not touch other sessions' processes or files.
* Context ≤ 150k tokens.
* **Content:** every number and claim comes from `paper_v8f/FACTS_PITCH.md` and follows its wording rules. Never "closed loop", never
  "dominant", never state of the art. Every result shown carries its tier in small type (for example "open loop" or "T1-kin: own motion,
  logged cameras").
  * Outlook items are visibly labelled "Outlook" or "In preparation".
  * No real company logos, no third-party branding, no music or footage you do not have.
  * The programme's own clips may be used.

## Assets you may reuse (read-only; copy segments, never edit the originals)
* `git -C C:/Users/Admin/tanitad-push show 24b97ba9:<path>` under
  `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/report/2026-10-25/`:
  * `v8f_hierarchy_attention.mp4`: per-level attention, the "focus on different abstractions in the different hierarchy levels" view the
    PI asked for earlier;
  * the `fig_*.png` figures.
* `.../raw/2026-10-27-video-v5/v8f_evidence_v5.mp4`.
* Export them to `F:/TanitAD-scratch/pitch/assets/`.
* Inspect a clip's frames before using it (ffprobe and a few extracted frames). Use only segments whose captions agree with
  FACTS_PITCH.md, or crop the captions out.

## The film (about 100–140 s, 1920×1080, 30 fps, H.264 yuv420p, CRF ≤ 18, silent, with on-screen text)
A consistent, calm, premium visual identity: a dark navy background and one accent per level (S magenta `#b04a86`, T blue `#1f6fb2`,
O amber `#c98500`), a clean sans-serif, smooth eased motion (ease-in-out cubic), and text on screen long enough to read. Suggested
scenes:
1. **Title (6 s):** "TanitAD", the subtitle "A hierarchical world model that drives at every level of abstraction", and
   "Dr.-Ing. Sayed Bouzouraa · TanitAD Research".
2. **The problem (10 s):** end-to-end driving models are data-hungry, opaque, unaware of their own limits, and expensive at inference.
3. **The 4B idea (18 s):** four brains stacking in with their clocks and horizons (operative 10–20 Hz; tactical 1–2 Hz; strategic
   ≤ 0.5 Hz; fallback continuous). Goals flow down, latents flow up through a stop-gradient barrier.
4. **v8F-mini, the reference implementation (18 s):** an animated architecture build. Frozen DINOv3 features feed S (route
   checkpoints), then T (the typed 60-d tactical code), then O (the trajectory as a residual on a kinematic prior; zero residual = the
   prior exactly). 7.2 M trainable parameters, 4 h on one 8 GB GPU.
5. **Seeing in layers (12 s):** a segment of `v8f_hierarchy_attention.mp4`, captioned "each level attends to its own abstraction".
6. **Results (30 s):** animated bars and counters, with tiers. Show:
   * K-1: −0.31 / −0.43 m 6 s ADE against the learned kinematic floor (open loop);
   * every row of the pre-registered set better than the floor (open loop, HACC);
   * own-motion rollout: −1.26 / −1.19 m 6 s ADE against the floor (T1-kin, DAG-OACC base), with four of four metric families improved
     somewhere;
   * route checkpoints ahead of a kinematic arc by up to 8.8 m at 80 m (T1-kin);
   * a typed interface worth about 1.1 m of headroom (diagnostic).
7. **Rigour as a moat (10 s):**
   * pre-registered, with both outcomes committed;
   * launch gates bound to commit, argv and data hashes;
   * deliberate-regression arms;
   * episode-cluster bootstrap and seed replicates;
   * four metric families, never one score.
8. **The frontier and the next step (14 s), positive framing.**
   * One behaviour remains: assertive launch from standstill.
   * Logged cameras cannot show a car its own consequences, so we are building re-rendered simulation on neural reconstructions:
     62 scenes staged, harness reproducing the plans bit for bit. Label it "In preparation".
9. **Outlook (14 s), labelled "Outlook":** perception fine-tuning for motion cues, scaling data and size toward the 300 M 4B budget,
   strategic routing over a latent map, the fallback brain for minimal-risk manoeuvres, and Jetson Thor deployment within 100 ms on
   4 cameras.
10. **Close (8 s):** "Why now: more data, more compute, and re-rendered simulation turn a validated architecture into a driver." Then
    "TanitAD" and "Dr.-Ing. Sayed Bouzouraa".

## How
* Render frames with matplotlib (Agg) or PIL into a raw pipe to ffmpeg, scene by scene, then concatenate. Re-encode reused clip segments
  to the same format.
* Write all code under `paper_v8f/video/code/`, the output to `paper_v8f/video/out/tanitad_pitch.mp4`, and a 20-frame contact sheet
  `paper_v8f/video/out/contact_sheet.png` for review.
* Keep the render under about 30 minutes of CPU time. Launch long renders detached (memory `devbox-drive-letter-and-detached-jobs.md`),
  and do not wait alive more than about 4 minutes on one render: write a handoff and end if needed.
* Self-check: ffprobe duration, resolution and codec, and read every on-screen number against FACTS_PITCH.md. Write
  `video/NUMBERS_CHECK.md` (one line per on-screen number → its FACTS line).

## Hand-back (≤ 250 words)
The output path, duration, size, scene list, the numbers check, anything you could not do, and a deliverable manifest.
