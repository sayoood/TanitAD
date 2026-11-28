# PREREG_MAPEDGE -- AMENDMENT 1 (sealed separately; supersedes ONLY the named parts of PREREG_MAPEDGE.md, sha256 `31a53010…`)

Written 2026-10-26 ~00:50 CET, **before any statistic exists**: the first real run (00:42) died on its FIRST raster read with
`KeyError: 'gt is not a file in the archive'` -- no window was scored, nothing was printed but the traceback.

## Why
MAPSEL banks a `gt` plane only for windows that HAVE a SAM3 GT map (its `has_gt`); a window without one carries the four OWN planes
only. The sealed K5 ("every raster has the five planes") was written without knowing that, so it would VOID the run on a property
of the bank, not a decode failure. This is a population clarification, not a change of question, metric, estimator or bar.

## A1.1 Population (replaces the population sentence of sec. 1)
A window enters the scored population only if its raster carries a `gt` plane (a window without a GT map has no GT edge cells and
could never meet `|E_gt| >= 20`). A window WITHOUT `gt` may still serve as a DERANGED DONOR (the donor contributes `pdrv` only).
The number of rasters without `gt`, and the number of episodes they touch, is reported.

## A1.2 K5 (replaces sec. 2 K5)
**K5 decode**: every raster has `pdrv`, `pedge`, `psw`, `code` at [1000, 600] uint8; every raster that has `gt` has it at
[1000, 600] uint8; at least one window is scored; `pdrv` is non-constant on >= 95 % of SCORED windows.

Everything else in PREREG_MAPEDGE.md (definitions, K1-K4, B1-B3, the INSTRUMENT-PASS rule, the committed next steps) is unchanged.
PREREG_MAPEDGE_Q2.md is unchanged (it reads MAPSEL's own S6c / has_gt populations).
