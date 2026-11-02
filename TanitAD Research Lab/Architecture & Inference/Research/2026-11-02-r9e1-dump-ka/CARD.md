# r9e1 dump known-answer read (KA-REF / KA-STEP0), 2026-11-02 — MEASURED, Thor

Tool: `stack/scripts/r9_e1_dump.py` (landed 7efa26e) + `r9dump_ka_read.py` (agent E1-DUMP's chain, `/home/nvidia/r8ns/r9dump/`).
Bars declared before the numbers (in `raw/KA_RESULT.json` → `declared_before_numbers`): |dAP@2 m| <= 0.002; rasters: `code`, `gt`
exact, `pdrv` / `pedge` / `psw` within 2 counts of 255. T0 label level, eval139 (139 clips, 4,532 windows), refcv8 EMA-48,852.

| check | result | verdict |
|---|---|---|
| KA-REF boxes vs PERC8's banked A_nosync packs | 4,532 / 4,532 windows bit-identical (max abs 0.0 on logit, xy, cls, every GT key); AP@2 m 0.32819 = banked 0.32819 (quoted 0.328, d 0.00019); AP@1 m 0.19870; n_pos 14,586 | **PASS** |
| KA-STEP0: the rig built from the same EMA, step 0, vs KA-REF | packs 4,532 / 4,532 bit-identical, d AP@2 m 0.0; rasters 300 / 300 within the rule | **SAME** — the rig's newest-row K = 3 forward IS refcv8's perception forward |
| KA-REF rasters vs MAPSEL's banked `rast_s0` (60-window sample) | `gt` exact (0 cells); `pdrv` max 2, `pedge` max 1, `psw` **max 3** counts of 255; argmax `code` differs on 12,698 / 36,000,000 cells (0.035 %) | **FAIL as declared** (psw 3 > 2, code not exact) |

**Reading.** The raster FAIL is a numeric mismatch (near-tie argmax flips, <= 0.012 in probability), not a logic one: GT is exact and
the rig-vs-ref rasters, through ONE code path, pass. The MAPSEL producer's numeric settings (batch composition / TF32 / AMP) were
not reproduced exactly: UNVERIFIED which. **Rule that follows:** E1 map numbers are compared only among dump-produced rasters,
never against MAPSEL's banked rasters or the numbers derived from them. Next lever (cheap, not blocking): identify the numeric setting
that differs and re-read the 60 windows.
