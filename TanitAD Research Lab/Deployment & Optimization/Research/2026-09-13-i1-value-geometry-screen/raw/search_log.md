# search log — 2026-09-13 i1-value-geometry-screen

| # | probe | route | result |
|---|---|---|---|
| 1 | banked v7-trunk latent frame bank | `find D:/Projects/TanitAD -iname "*latent*.pt" / "*tokens*.pt" / "windows_*.pt"` | REF-C `latents_refc-*-30k(-ep).pt` (881 windows, 8-frame sequences — no dense per-clip ordering) and `taniteval/results/windows_*.pt`; **no v7 bank** |
| 2 | same, second location | `find C:/Users/Admin/tanitad-caches, tanitad-v7fbase, tanitad-v7fbudget -iname "*latent*" / "*zbank*"` | only scripts (`refc_dump_latents.py`, `v6_dump_sw_latents.py`, `latent_screen.py`) — **no v7 bank at a second location** ⇒ EMPTY stated at two probes |
| 3 | current frozen-trunk frame bank | `C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens/index.npz` | ✅ 27,664 rows, 139 clips, refcv5-v2 ckpt 40284, s32 + pixel floor — used |
| 4 | frame clock | A&I `2026-09-13-bev-lidar-corpus-and-head/RESULT.md` §3.1 | v2ep grid `int(span·10)` ⇒ 10 Hz (INHERITED) |
| 5 | GPU/RAM state before running | `nvidia-smi`, `Win32_Process`, `Win32_OperatingSystem` | 4060 at 95 % held by `p4_bev_head.py --arm main --extra-train --seed 1 --tag rb` (PIDs 54248/55984, 13.5 GB RSS); ~2.07 GB free RAM ⇒ CPU-only, below-normal priority |

No web literature in this package; the literature context (D10-1's origin) is `Frontier Scan/Daily/2026-09-10/RESULT.md` F3.
