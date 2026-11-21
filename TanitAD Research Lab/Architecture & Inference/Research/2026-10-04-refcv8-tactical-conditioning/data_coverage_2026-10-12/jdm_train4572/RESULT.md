# refcv8 join-defect masks for the 4,572-clip train view (2026-10-12, 0 GPU, dev box)

Layer key: VIEW = train symlink view, JOIN = agent join (3-D, `b1_train_plus_eval_agents_3d.jsonl.xz`), CFG = `tanitad.join_label_defects/1` mask file. Evidence MEASURED (ours) unless tagged; every figure below is in `raw/controls.json` unless another artifact is named.

## Headline
* **Control: PASS** (tool + inputs reproduce the OLD file; 0 of 4,508 old clips differ; 0 eval-only ids changed; loader accepts the new file).
* **NEW config (the path the launch must use):** `stack/tanitad/configs/refcv8_join_label_defects_train4572.json` -- md5 `b2591cb2ec4f4089831124e4738a961c`, sha256 `dde7a4cb862d11acf0a86535c12ca7039e27cfa6e036454a16c50790993a6149`, 151,824 bytes, LF. The old `refcv8_join_label_defects.json` is untouched and stays valid for the 4,369 view.
* **The 58 newly joined ADD clips:** 3 carry ego-as-agent boxes (**375 frames**: 52 + 127 + 196) and 3 carry track-id-switch events (**5 events**); union 5 clips, 53 of the 58 clips need no mask. All 375 ego boxes are the D-2 defect exactly (within 0.5 m of (1.4, 0), ~4.7 x 2.0 m automobile, one track id per clip).
* **"693-box ego mask" -> 1,068-box** (18 -> 21 clips); track events 6,227 -> 6,232 (828 -> 831 clips).
* **Using the OLD file on the 4,572 view is a SILENT no-op for these clips:** the mask is keyed by clip sha12 and `JoinDefectMasks.load` / the `_pin_wpc_fixes` call (`refc_v3_train.py:1783-1792`) check only schema and key shape, never coverage of the train manifest -- so 375 phantom-ego frames and 5 id-switch events would train unmasked with no error. The launch argv must name the new file.

## Method
1. Inputs (md5 in `raw/input_md5.txt`): the join named by the old config's own provenance block -- `join_md5 401c9c3119a50632dafc35c595d9e051`, 409,336,004 B, 875,657 lines -- pulled read-only from Thor `/home/nvidia/data/join3d/b1_train_plus_eval_agents_3d.jsonl.xz` (md5 re-computed on the dev box = the provenance md5); old train manifest Thor `refcv6-b1-416x1024-train/_v2manifest.pt` (md5 `3c9f8bc8...` = provenance); eval139 manifest `3443323229279a18253691ff36ebffd3` (= provenance); new train manifest `77dafb50af0aedd1b940cf12c12518ef`.
2. Tool: `stack/scripts/mine_join_label_defects.py` (unchanged, repo export 58d413f), single process, CPU only, run on the dev box (`raw/run_mine.sh`): `--join <3-D join> --manifest <train> --manifest <eval139> --out ...`.
3. Run A (control): old manifest + eval139 -> `raw/old_repro.json`. Run B (deliverable): new manifest + eval139 -> the new config. Same join, same code, same eval139; only the train manifest differs. The miner skips any join line whose clip is not in the manifests (`mine_join_label_defects.py:77`), which is why the old file held nothing for the 58 ADD clips: they were in the join but not in the old manifest. So the new file is the old file plus the ADD58 entries -- additive by construction, and checked below, not assumed.

## Controls
| # | control | result |
|---|---|---|
| 0 | old inputs -> old file. Repo blob (`git show agent/arch-inf-20260803:stack/tanitad/configs/refcv8_join_label_defects.json`, LF, md5 `3a5b4807...`) vs `raw/old_repro.json` (md5 `fe4e867c...`) | **identical**: `ego_footprint`, `track_jumps`, `schema`, `provenance` all equal; a shell `diff` of the two files prints exactly ONE line, `wall_s` 169.8 vs 106.9. (The dev repo export's copy has CRLF, md5 `f311eb03...`; stripping CR gives `3a5b4807...`.) The re-run also reproduces D3's aggregates in the old header: 693 boxes / 18 clips / 6,410 event frames / 887 clips / 853 pose-glitch-like / 5,822 single-track-like |
| A | every old clip's entries unchanged: 4,369 old train + 139 eval = **4,508 clips compared** | **0 differ** (ego: 0; jumps: 0). Of those, 18 have ego entries and 828 have jump entries (806 train + 22 eval) -- all 18 and all 828 identical in the new file. Loader level: every old clip's ego frozenset, track-event frozenset and the 9,359 derived rate-records are equal in the new object (new has 9,368) |
| B | keys in the new file but not the old | 3 ego + 3 jump keys; **0 outside the 203 ADD clips, 0 in eval139, 0 in the old train** |
| C | eval-only ids changed | **0** (139 eval clips compared inside control A: 0 differ, 22 with jump entries unchanged, 0 ego entries) |
| D | `JoinDefectMasks.load(new)` | succeeds (it refuses a wrong schema, a missing section, a non-sha12 key; `raw_id_leak_check`: 0 non-sha12 keys). Stats below |
| E | manifests | old 4,369 + 203 = new 4,572 unique clip ids; old subset of new; eval139 intersects the new train in **0** ids; no sha12 collision in any of the three sets |
| F | join census, independent of the miner (prefix slice of each line, `raw/join_clip_census.py`) | 875,657 lines / 4,566 clips; all 4,369 old, all 139 eval, **58 of the 203 ADD** and 0 other clips; the 58 = the 58 in the train-3D join's own meta `per_clip` (`census_vs_meta_agree true`); 11,133 lines for the ADD58, all with non-empty `agents`; 864,524 + 11,133 = 875,657, so the new `n_records` covers every join line |
| G | pose input of the jump rule (`raw/pose_sanity.py`) | all 4,369 old clips' poses in the new manifest are `torch.equal` to the old manifest; the 203 ADD clips: 0 non-finite, max step 3.59 m/frame (old max 4.03), 0 clips over 5 m/frame; pose-glitch-like frames unchanged at **853** (a mis-posed ADD clip would likely have added some (inference, not a direct pose check against an independent source)) |
| H | mutation (`stack/tests/test_refcv8_join_defects_train4572.py`) | 4 tests pass on the real files; deleting one old ego clip from a scratch copy of the new file turns 3 of the 4 RED |

## Totals: old -> new (`JoinDefectMasks.load(...).stats()` and file headers)
| quantity | old (`refcv8_join_label_defects.json`) | new (`..._train4572.json`) | delta |
|---|---|---|---|
| ego boxes removed / frames listed (`n_boxes`) | 693 | **1,068** | +375 |
| ego clips | 18 | 21 | +3 |
| class counts | automobile 691, trailer 2 | automobile 1,066, trailer 2 | +375 automobile |
| share within 0.5 m of (1.4, 0) | 0.8427 | 0.8979 | ADD part 375/375 |
| track-jump event frames (all, incl. pose glitch) | 6,410 | 6,415 | +5 |
| jump clips (any event frame) | 887 | 890 | +3 |
| pose-glitch-like frames excluded | 853 | 853 | 0 |
| track events listed (`n_track_events_listed`) | 6,227 | **6,232** | +5 |
| clips with listed events (`n_track_event_clips`) | 828 | 831 | +3 |
| join lines read / `n_records` | 875,657 / 864,524 | 875,657 / 875,657 | +11,133 |

## The 58 newly joined ADD clips (sha12; `raw/controls.json` -> `add58_rows` has the frame lists)
| clip sha12 | join lines | ego-footprint frames | track-jump events |
|---|---|---|---|
| `9dee20474b44` | 199 | **196** (frame range -1..196) | **3** (frames 142, 143, 145; one track) |
| `70928506fcc6` | 200 | **127** (0..126) | 0 |
| `5ed957d47a41` | 199 | **52** (57..108) | 0 |
| `11c906128f92` | 200 | 0 | **1** (frame 66) |
| `4b3773213797` | 200 | 0 | **1** (frame 2) |
| other 53 clips | 32-200 | 0 | 0 |
| **total (58 clips)** | 11,133 | **375** (3 clips) | **5** (3 clips) |

Ego boxes of the three clips (`raw/add_ego_boxes.json`, a read-only pass over the join): 375 boxes, 375 within 0.5 m of (1.4, 0); median size 4.57-4.73 x 1.93-1.99 m; class automobile only; ONE track id per clip; `occ` 0. This is the old D-2 signature at 100 % (the old 18 clips: 84.27 %), and the miner's rule removes only the rows whose centre lies in the footprint, so legitimate boxes in the same frames stay. Rate: 3/58 = 5.2 % of the ADD58 clips carry it vs 18/4,369 = 0.41 % of the old train -- ESTIMATED to be a property of the no-map clip population (the ADD58 are slow/near-stationary: median over clips of the per-clip median step: 0.086 m/frame vs 0.90 for the old set, `raw/pose_sanity.json`), n = 3 clips, not a finding about the corpus.

## Not covered / limits
* The 145 ADD clips WITHOUT a join line (13 have obstacle data and were dropped by the 2026-09-06 builder for cause -- see `landed_hold_add203/RESULT.md` section 4) have no agent rows, so there is nothing to mask; they emit NO_LABEL, not labelled-clear.
* The mask lists defects the two D3 rules find; it does not re-validate the ADD58's join alignment (the builder's frame gate did that when the join was built).
* No GPU, no Thor write. Thor was read only (3 files copied to the dev box).

## Integration (Master Mind) -- the new file must replace the old path at every one of these sites
| site | what to change |
|---|---|
| `stack/scripts/refcv8_assemble_argv.py:145` | `--join-defect-masks` value -> `.../refcv8_join_label_defects_train4572.json` (and the `:256` test literal) |
| `stack/scripts/refcv8_prelaunch.sh:143,149` | the `-f` presence check and the "argv path == code path" check both name the old file |
| `stack/ops/runs.d/refcv8-wpb-smoke.argv.json:163-165,408-409` | smoke argv (smoke tree path) |
| `stack/scripts/launch_gate.py:673` | reason string "the 693-box ego mask" -> "1,068-box" (text only; the gate checks flag presence, not the file) |
| ship to Thor | `/home/nvidia/refcv8_run/tree/stack/tanitad/configs/` (new file, no overwrite) before the prelaunch |
| optional | pin the new file's sha256 in the launch record; `stack/tests/test_refcv8_join_defects_train4572.py` (staged) pins its totals against literals |

## Reproduce
`bash raw/run_mine.sh NEW <train4572_v2manifest.pt> <out.json>` (needs the 3-D join + eval139 manifest at the paths in the script), then `python raw/controls.py`. Wall ~107 s per mining run.
