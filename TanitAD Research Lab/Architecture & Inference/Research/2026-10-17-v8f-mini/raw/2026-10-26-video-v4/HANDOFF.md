# HANDOFF: evidence video v4 (v3 + chapter 7 "T1-kin: the model on its own motion")

All paths under `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/viz4/` (cwd for every command). CPU only (`CUDA_VISIBLE_DEVICES=-1`), one python process at a time, python = `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`. No git writes were made.

## State
* Chapter 7 code, scene selection (`SELECTION_V4.md`, fixed before any per-window T1 number), QA stills (`qa_ch7_*.png`) and the **pre-render controls** are done: `CONTROLS_V4.json` ALL_PASS_before_render = True (T1 tracks = dumps max abs 0.0 m over 217 frames x 6 lines; A1, H1, R1, B1, N4, W, S1, G1, F1 pass; deliberate-regression arm X passes: cl/fl swap of JUNCTION made T1 and A1 fail).
* The **full render** (v3 chapters 1-6 re-rendered with the v4 header + new chapter 7 + closing, 400 unique frames, 381.4 s, 3,814 frames) runs DETACHED via `run_chain.cmd`: build -> `controls4.py --post` -> `notes_gen4.py` -> `mk_files4.py`.
* Markers: `chain_started.marker`, then `chain_done.marker` (ok) or `chain_failed.marker`. Logs: `build_video4.log` (ends `ZZVIDEO4-DONE`, a refusal prints `ZZVIDEO4-STOP <why>`), `controls_post.log`, `notes_gen4.log`, `mk_files4.log`.

## Exact next step (a fresh agent, no waiting inside a model turn)
1. If `chain_done.marker` exists: read `controls_post.log`; `CONTROLS_V4.json` must show `post_render_done` true and `ALL_PASS` true (V1 video integrity: h264 1920x1080, 3,814 frames; P1/P2: the video's text and plotted arrays equal the controlled ones; U1/U2: chapters 1-6 text-identical to v3 except the title card). Then verify one still from the mp4 (`ffmpeg -ss 300 -i v8f_evidence_v4.mp4 -frames:v 1 x.png` to a path under viz4/, then delete it).
2. If `chain_failed.marker` or no marker after ~15 min with no python process: read `build_video4.log`, fix, re-run `run_chain.cmd` (detached: Win32_Process Create, CreateFlags 0x08000200, `cmd /c run_chain.cmd < NUL`).
3. Report to the Master Mind: the mp4 path, `CONTROLS_V4.json` ALL_PASS, `VIDEO_NOTES_V4.md`, `FILES.txt`.

## Not done / caveats
* Nothing is staged, committed or pushed; registers (`GOALS_AND_CLAIMS.md`) are not updated (MM's step).
* C: has ~4 GB free; viz4 is ~12 MB before the mp4 (~10 MB). Never write frame sequences to C: (frames are piped to ffmpeg). `E:/Projects/TanitAD-artifacts/scratch/viz4/` was created empty and removed.
