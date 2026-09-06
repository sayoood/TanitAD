# RUNBOOK — reproducing the selection-quality package

Everything here runs on the **dev-box RTX 4060** and on **local disk**.
⛔ The A40 (refcv5 to 2026-09-08 07:33 UTC) is not touched at any step.

## Inputs, each verified by content before use

| input | path | verification |
|---|---|---|
| refcv4b checkpoint | `C:/Users/Admin/refcv4b_final/ckpt_40284_FINAL.pt` | md5 `99b573e8277d94a5e3bfbf630cb4d751` — equals the pod's own `pod_md5.txt` **and** `LANDING_RESULT.md`'s published md5 |
| its config | `C:/Users/Admin/refcv4b_final/config.json` | `argv` carries `--anchor-control-units alat` |
| v2 episode cache | `C:/Users/Admin/refav1_eval_full/eps` | 141 `*.v2ep.pt`, the eval set |
| v7.2 eval labels | `C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz` | md5 `aa12c948f062181c3297265b51526ec5` — the blob the published refcv4b T1 record used |
| v7.2 train labels | `C:/Users/Admin/rlgate/s2_labels_v7.2_train.jsonl.gz` | md5 `0ff902130ce76886b8a925eceed9e3a5` |
| banked eval lead block | `C:/Users/Admin/dkhead_run/inputs/b1_eval_lead_block.npz` | 29,556 frame rows / 147 clips, `LEAD` on 8,341 (28.22 %) |

## The code tree

The venv's editable `tanitad` install points at the G: mount, which flaps
(`Invalid request code` / `Errno 22`) while metadata still resolves. The stack was therefore
copied to **local disk** at `C:/Users/Admin/tanitad-selq-20260906` and verified **file by file**:
**498 of 499** `.py` files md5-identical to the repo; the single miss
(`stack/scripts/pod_pull_b1_epcache.py`) was a mount flap on a file nothing here imports.
`raw/_env.py` removes the editable finder from `sys.meta_path` and puts that tree first, and
asserts the resolved `__file__` of both packages.

## Steps

```
# 0  provenance: prove the local "refcv4b" dump is actually refcv3
python raw/p0_identity.py

# 1  labels, zero GPU
python raw/p1a_label_prevalence.py          # prevalence, every vocabulary IN FULL
python raw/p1b_nudge_absorbs_lc.py          # -> INCONCLUSIVE, see D-SELQ-LATPEAK-5
python raw/p1c_anchor_expressibility.py     # the vocabulary's supply, model-free
python raw/p1d_head_dead_classes.py         # trained head + Adam second moments
python raw/p3b_road_signal.py               # is there any lane geometry? (3 probes)

# 2  the GPU step: re-roll the refcv4b arm at STRIDE 1 (~62 min on the 4060)
bash run_dump2.sh 0 out/refcv4b_t1_s1.json out/refcv4b_t1_s1_dump 1

# 3  everything that needs the roll
bash after_roll.sh                          # p1e -> p1g -> p2p3, in priority order
```

`run_dump2.sh` invokes `taniteval/tools/refcv3_arm.py` **unmodified**
(md5 `5ee114199b00104acebe8ecaa1828044`, repo HEAD) with `--no-navshuf --no-navzero
--no-lead-block`, which drops two of the three forward passes and takes the cost from
0.353 s/window to **0.167 s/window**. Those two arms are not needed for a selection question; the
tool prints its own warning that the record is then inadmissible for a nav-conditioned claim, and
no nav claim is made here.

⚠️ **Why stride 1 and not the published stride 5.** The PI's evidence for observation (6) is
clip `73e750eb`, frames **032 and 034** — 0.2 s apart. The published grid samples every 0.5 s and
structurally cannot see that. The published 4,823-window grid is exactly the `ws % 5 == 2` subset
of this roll, and `p2p3_run.py` scores that subset separately as a cross-check.

## Controls that must read known values

| id | control | required value |
|---|---|---|
| C1 | straight candidate (`a_lat = 0`) | `max\|y(6 s)\|` and `max\|dyaw(6 s)\|` EXACTLY 0 |
| C2 | v0 round-trip | `dump v0 == poses[ws + provider_to_raw_frame_offset, 3]` to 1e-5 |
| C3 | GT self-consistency | the GT plan at t vs t+1 in a common frame agrees to ~0 — validates the rigid transform AND is the waypoint-stability ceiling |
| C4 | shuffled-selection floor | strictly above the arm's switch rate |
| C5 | index mapping | `ADE(F[sel_idx])` must be far better than `ADE(F[random])`, or `sel_idx` does not index `anchor_controls` |
| C-closedform | integrator | `a_lat = −3.0` at `v0 = 10 m/s` gives `\|dyaw(6 s)\| = 103.132°` in closed form and integrated |
| read-controls | every label table | printed IN FULL; the non-zero rows prove the field was read |

## Traps hit and how they were closed

* **A 0 that was a failed read.** The first prevalence probe counted `g_tac.tokens`, a key that
  does not exist, and produced an all-zero goal table that looked exactly like a finding. Every
  vocabulary is now printed in full so the non-zero rows are the same-breath control.
* **`poses[ws]` is the wrong row.** `v2_dataset` stores `poses[n_stack-1:]`, so the provider index
  needs `+ provider_to_raw_frame_offset` (2). Read from the run's own manifest, never assumed; C2
  fails loudly otherwise.
* **The dump-file → clip mapping is not directory order.** `refcv3_arm.py` filters the corpus, so
  `clip_index` indexes the FILTERED list. The manifest's `episodes[]` is the only honest source.
* **A quantity persisted outside its scope.** `lat_peak_m` — see D-SELQ-LATPEAK-5.
* **A ceiling that scores worse than the arm.** `a_star` binds against `model.core.decoder.anchors`
  (the fixed `ref_speed_ms` bank) — see D-SELQ-ASTAR.
