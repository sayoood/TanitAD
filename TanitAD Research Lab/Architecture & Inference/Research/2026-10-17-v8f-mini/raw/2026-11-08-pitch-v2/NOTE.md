# Pitch film, version 2 (2026-11-08)

Corrected to the PI's rulings of 2026-11-08 (REQ_HORIZONS.md). Changed from `raw/2026-11-06-paper-and-pitch/video/code/`: `scenes.py` (level roles and bands, every level reads the current ego state, the tactical output as the action for the operative level, one or more trajectories up to 6 s, budget at most 800 M, T2 first reads on 48 scenes, strategic outlook) and `build.py` (output path only).

Re-rendered with `render_v2.cmd`: scenes 2, 3, 4, 5, 8, 9 (one-based); scenes 1, 6, 7, 10 are the 2026-11-06 renders. 132 s, 1920x1080, 30 fps, H.264 yuv420p. Seven frames across the changed scenes were read.

The film file is not in this repository, because scene 5 shows camera frames and the repository is public. It was delivered to the PI directly, sha256 `a339cfb42bb9e4d3f2ac19bc999ce3f8d849f99f2de0b1d953ff1d02aee2a2d5`, and it is re-creatable from this code and the assets of `BRIEF_VIDEO.md`.
