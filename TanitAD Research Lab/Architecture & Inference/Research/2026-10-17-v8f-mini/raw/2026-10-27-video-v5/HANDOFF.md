# HANDOFF: evidence video v5 (v4 + chapter 8 "DAgger: trained on its own states"; closing = chapter 9)

Work dir `E:/Projects/TanitAD-artifacts/video5/` (cwd for every command). CPU only, ONE python process at a time, python = `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`, env `CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1`. No git writes were made. E: has ~6 GB free; the work dir is ~60 MB + the mp4 (~20 MB); no frame sequence is ever written to disk (frames are piped to ffmpeg).

## State
* DONE: label cells (`extract_dag_cells.py` -> `data/dag_cells_*.npz`, the programme's own reader; its macro-F1 reproduces `DAG_READ.json` exactly), selection (`SELECTION_V5.md`, rule sha256 `ee759319...`, written before any chapter-8 render), scenes (`v5_scenes.py`), data layer (`v5_lib.py`), controls (`controls5.py`), notes generator (`notes_gen5.py`), builder (`build_video5.py`, 622 unique frames, 536.2 s, chapter 8 at 372.4 s, closing at 526.2 s).
* Pre-render controls: `CONTROLS_V5_main.json` (all must PASS) and `CONTROLS_V5_swap.json` (regression arm PASSED: T1 and A1 both failed on the swapped WORST scene).
* The FULL RENDER + post controls + notes + FILES.txt run DETACHED via `run_chain5.cmd` (build -> `controls5.py --post` -> `notes_gen5.py` -> `mk_files5.py`). Markers: `chain_started.marker`, then `chain_done.marker` (ok) or `chain_failed.marker`. Logs in `logs/` (`build_video5.log` ends `ZZVIDEO5-DONE`; a refusal prints `ZZVIDEO5-STOP <why>`).

## Exact next step (a fresh agent; do not wait inside a model turn)
1. If `chain_done.marker` exists: read `logs/controls_post.log`; `CONTROLS_V5.json` must show `post_render_done` true and `ALL_PASS` true (V1 h264 1920x1080 5,362 frames; P1/P2 video text and plotted arrays = the controlled ones; U1/U2 chapters 1-7 text-identical to v4 except the title card). Then look at 3 stills: `ffmpeg -ss 380 -i v8f_evidence_v5.mp4 -frames:v 1 qa_check_380.png` (also -ss 450 and -ss 530), read them, delete them.
2. If `chain_failed.marker` or no marker after ~20 min with no python process: read `logs/build_video5.log` / `logs/controls_post.log`, fix, relaunch the chain detached (Win32_Process Create, CreateFlags 0x08000200, `cmd /c run_chain5.cmd < NUL`).
3. QA still to eyeball (never opened after the last layout fix): `qa_ch8_08c*_end.png` (7 windows), `qa_ch8_08d_hist.png`, `qa_ch8_08e_card_theads.png`, `qa_ch8_08f_card_guards.png`, `qa_09_closing.png`, `qa_01_title.png`; the closing / title text must not overflow.
4. Report to the Master Mind: mp4 path, chapter times, `CONTROLS_V5.json` ALL_PASS, `VIDEO_NOTES_V5.md`, `FILES.txt`; name where DAG looks worse.

## Not done / caveats
* Nothing is staged, committed or pushed; `GOALS_AND_CLAIMS.md` is not updated (MM's step).
* After the mp4 passes QA, delete only own intermediates (`__pycache__`, `*.tmp.mp4`, `qa_check_*.png`); keep `qa_*.png`, logs, data.
* Chapters 1-6 inputs still come from the old scratchpad dirs (`viz/`, `viz2/`, `hacc/`, `hblend/`, `hspec/`, `listen/`); `sfresh_*.json` were copied from tip `bdf8d561` into `inputs/sfresh/` (U2 proves the registered numbers are unchanged).

## Update 07:35 (relaunched; READ THIS FIRST)
* Title-card overlap fixed and eyeballed (`qa_01_title.png` clean). Pre-render controls re-run on the FINAL code: `CONTROLS_V5_main.json` ALL_PASS True (113.8 s); regression arm (`V5_SWAP_ARMS=WORST`) PASSED (T1 14 failures, A1 42 failures).
* First chain launch (07:29) FAILED at frame ~125: the scratchpad `hacc/` dir that v3_lib read for chapter 4 no longer exists. FIX: the same 7 JSONs were copied from tip `bdf8d561` (`.../raw/2026-10-25-hacc/run/`) into `inputs/hacc/` (sha256 in `inputs/hacc/SHA256SUMS.txt`, `PROVENANCE.txt`), and `v3_lib.py` line 18 now has `HACC = f"{OUT}/inputs/hacc"`. All `v3_lib.SOURCES` paths verified to exist. NOTE for the report: chapters 1-7 numbers must still equal v4's (post control U2 proves it); if U2 fails on a hacc number, the banked copy differs from the old scratchpad file.
* Chain RELAUNCHED detached 07:33 (pid 78792, `cmd /c run_chain5.cmd`); at 07:35 it was at frame 270/622 and past the earlier failure. Expect build ~5-8 min, post controls + notes ~3 min. Markers: `chain_done.marker` (ok) / `chain_failed.marker`.
* Where DAG looks worse (name in the hand-back): WORST #1795 NO-TURN (+8.86 m s0, +2.05 m s1; DAG executes 73.4 m vs human 38.9 m); also JUNCTION #496, NO-TURN #59, DRAW3 #526 at s0 (4 of 7 shown windows at s0). DAG better on 64.7 % (s0) / 66.1 % (s1) of 2,399 windows.
* Hand-back numbers: duration 536.2 s; chapters 1..9 at 0 / 8.0 / 37.0 / 94.5 / 134.3 / 188.3 / 206.6 / 372.4 / 526.2 s.
