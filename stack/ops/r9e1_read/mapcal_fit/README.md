# r9e1_read/mapcal_fit — the Thor-side map read scripts, AS RUN (MM, banked 2026-11-07)

These scripts produced the E1 map arms' decisive reads (FIT-threshold protocols `psw`, `pswpedge`, `pswpedgeF`, and the D3 beside rows). They ran
from `/home/nvidia/r8ns/r9e1/mapcal_fit/` on Thor and are banked here byte-for-byte (md5 below = the Thor copies on 2026-11-07 ~02:15Z) so the
read protocol is not held on one disk. Paths are Thor's and hard-coded: they are a record of what ran, and the runnable copy is the Thor one.
Kits they name (`kit_*`) are `git -c core.autocrlf=false archive` exports of the commit in their name.

| script | md5 | what it did | read it produced |
|---|---|---|---|
| `mapcal_q.sh` | 6dc807d6 | one GPU-lock hold for the C0 / C1 / M1 FIT dumps (1,112 TRAIN-DIAG windows each), idle-GPU check | inputs of every FIT read |
| `mapthr_read.sh` | 65836b2c | `PREREG_M1_FITTHR` steps 5-6: FIT floors + per-variant arm reads (psw DECISIVE, pswpedge secondary) | M1 vs C0 psw / pswpedge |
| `mapthr_arm_psw.sh` | aade9f4b | re-run of the psw arm read (the first kit was CRLF; bash refused it) | M1 vs C0 psw |
| `mapthr_arm_pswpedge.sh` | 28147f44 | re-run of the pswpedge arm read (ALREADYRUNNING on the first try) | M1 vs C0 pswpedge |
| `mapcal_m2a.sh` | 7e86b473 | `PREREG_M2A_FITTHR`: M2a's FIT dump, its fits, its reads vs M1 | M2a vs M1 psw / pswpedge |
| `pedgefit_run.sh` | 05838f5e | `PREREG_M1_PEDGEFIT` + A1: `pswpedgeF` fits for C0 C1 M1 M2a, its floor, the arm reads | M1 vs C0, M2a vs M1 pswpedgeF (DECISIVE) |
| `pf_reverdict.sh` | c070db27 | verdict-only re-run at kit ac90ce1e (edge F@0.3 m beside rows = PREREG_M1_PEDGEFIT's selector) | `reverdict_ac90ce1e/` |
| `fitread_arm.sh` | 7ae0af45 | generic map-arm pswpedgeF read (wait for the chain reader, FIT dump under the lock, fits + A1 check, reads per REF) | staged for M1N20 (`C0 M1`), M3 / M3N20 |
| `d3_reverdict.sh` | 0b85c38a | verdict-only re-run at kit 4e49ebb5 with `--map-beside-d3` (lane tolerance-IoU, boundary recall rows) | `reverdict_d3_4e49ebb5/` |
| `d3_cmp.py` | e6143a9f | the comparison `d3_reverdict.sh` uses: only the formerly NOT-COMPUTABLE `<ARM>.beside.lane/.edge` reads may move | — |

Banked verdicts: `TanitAD-artifacts/dagland/e1/verdicts/{M1,M2a}/fitthr_*/`. Pre-registrations: `TanitAD-artifacts/dagland/e1/m1_diag/`,
`e1/m3_diag/`. GOALS rows: `H-R9-E1-M1`, `H-R9-E1-M2a`, `H-R9-E1-M1N20`.
