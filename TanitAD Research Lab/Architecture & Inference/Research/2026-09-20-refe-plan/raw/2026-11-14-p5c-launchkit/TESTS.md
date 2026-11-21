# TESTS.md: serial, CPU (`CUDA_VISIBLE_DEVICES=-1`), per-test TMP/TEMP/TMPDIR under scratch (deleted after each), venv `C:/Users/Admin/venvs/tanitad`. MEASURED; logs in `logs/{ov1,ov2,tip1}/`, drivers in `logs/scripts/`.
Trees: TIP = `git archive 99ffcd3a` (refe, code, eval, report, splits, stack, taniteval, tools), OVERLAY = TIP + `files/`. `rc` / ZZ token per script. Round ov1 = first overlay run; ov2 = overlay re-run after the 7 pin-test edits below; tip1 = unmodified tip.

| script (args) | TIP rc / token | OVERLAY rc / token | verdict |
|---|---|---|---|
| test_p5c_profile.py (full, 14 mutants) | -- (new) | 0 / ZZP5CPROF_PASS 249 | new, green |
| test_traj_label_table.py (full, 21 mutants) | -- (new at tip) | 0 / ZZTRAJLABEL_PASS 208 (`logs/test_traj_label_table_full_overlay.log`); `--fast` 166 | green |
| test_launch_gate.py --static --no-banked | 0 / ZZLG_STATIC_PASS 1180 | 0 / 1180 | same |
| test_launch_gate.py (full) | 1 (no `D:\Projects\TanitAD\data\refe_consistent`, dev data absent) | 1 (same traceback, same line) | environmental, identical |
| test_p7_head_wiring.py (--static and plain) | 0 / ZZP7HEAD_PASS 182 | 0 / 182 | same |
| test_ego_yaw.py --fast | 0 / 160 | 0 / 160 | same |
| test_p7_integration.py | 0 / ZZP7INT_PASS 225 | ov1: 1 (IV AMENDMENT YAW: "no other profile carries --ego-yaw-table"); **ov2: 0 / 225** | fixed by editing the test (the P5-C profiles carry it by design) |
| test_div / dose / rep / s1zsm / scorer1p6 / vdac_gate.py | 0 / ZZTEST_PASS (div, dose, rep, scorer1p6, vdac measured on tip; s1zsm not run on tip) | ov1: 1 (each pins `len(PROFILES)==26` / the exact set of new profiles); **ov2: 0 / ZZTEST_PASS** all six | fixed by editing the six tests (26 -> 29, + the 3 P5-C names) |
| test_s1z_gate.py | 0 | 0 | same |
| test_suite_nesting.py | 0 / 250 | 0 / 277 (ov1) | token count differs: the tip's own number moved (the hold measured 250 on `fa67c63f`; the tip now runs more G-SUITE lines); not run on the same tip twice, so the delta 250 -> 277 is NOT attributed: it may come from the kit (new suite entries in profiles) |
| test_gsuite_tail.py | 0 / 3 | 0 / 3 | same |
| test_gate_gpu_probes / label_dir_refusal / carve_feed_gate / vdac_g12_bound / session_reserve / p7_chain | 0 (14 / 6 / 10 / 15 / 43 / 110) | 0 (same tokens) | same |
| test_aux_g12_bf16.py | 0, 1 skip (no CUDA) | 0, 1 skip | same |
| test_p7_ship_list.py | 1 (`real.json` not found: its derive step needs the full package tree) | 1 (same) | environmental, identical |
| test_s1c.py, test_p6_cpu_probe.py | not run on tip (C: floor, below) | 0 / 220 (2 skips), 0 / 111 | green on overlay; hold measured both green on the tip |

## REGRESSIONS: 0 (after the 7 pin-test edits; the 7 red scripts in ov1 were all "a new profile exists" pins, not behaviour)
Those seven edits (`test_div_gate`, `test_dose_gate`, `test_rep_gate`, `test_s1zsm_gate`, `test_scorer1p6_gate`, `test_vdac_gate`, `test_p7_integration`) are in `files/` and are part of the land: without them the land goes RED on 7 scripts. All are one-line pin edits (the profile-count and name-set pins; `diffs/`).
26 pre-existing profile fingerprints (sha256 of `repr(sorted(profile.items()))`): identical TIP vs OVERLAY (`logs/digest_*.json`).

## NOT RUN (stated, not hidden)
- **C: fell to 6 GB free** (from 24 GB at start; my scratch was <= 1 GB, the rest is other sessions) and the driver stopped itself at its 12 GB floor. Not run: test_s1c / p6_cpu_probe on tip, and the ~32 other launch_gate-reading scripts (`list_more2` in `logs/scripts`; they iterate profiles in at most the way the 7 above do; the likeliest hits are other `len(PROFILES)` pins, but a grep of every `test_*.py` for `len(L.PROFILES)` / `set(L.PROFILES)` found only those 7), test_p6_chain (839 s), test_p6_levers (rc 124 on both trees in the hold), test_gate_suite_prerun (whole G-SUITE). **MM: run them after the land, as the pin-sensitive suite.**
- `stack/tests/*` that mention `launch_gate` read `stack/scripts/launch_gate.py`, a different file; untouched.

# HANDOFF_2 round (2026-11-14): new tests + the ~32 scripts the first round could not run. MEASURED, serial, CPU (`CUDA_VISIBLE_DEVICES=-1`), per-test TMP/TEMP/TMPDIR deleted after each, tanitad venv. Tip at build `e3fe96ac`, moved to `1da03de9` during the work (every file the kit touches has the same blob there). Logs (tails only; the driver is `logs/h2/scripts/runall2.sh`): `logs/h2/`.
TIP = `git archive e3fe96ac` (refe, code, eval, report, splits, stack, taniteval, tools), OVERLAY = TIP + `files/`. Every script was run on the OVERLAY; every script that failed there was run on the TIP.

## New scripts (all green, full mode with the source-mutation harness)
| script | result | mutants / arms that went RED |
|---|---|---|
| `test_p5c_live.py` (G-LIVE (f) + (h)) | rc 0, `ZZP5CLIVE_PASS 71` (`--fast` 48) | 11 source mutants; gate arms `warm_not_loaded`, `warm_perturbed`, `traj_label_not_passed`; wrong checkpoint, unpinned base, stateful optimiser, resume file, ego slot moved by the table, control non-zero |
| `test_p5c_postgate.py` (G-ADAM) | rc 0, `ZZP5CPOST_PASS 56` (`--fast` 32); re-run after adding the `config.json warm_start.sha256` check | 12 source mutants; state deleted, first / second moment zeroed, wrong step, count +/-1, moved trunk, untrained head, identical arms, wrong base |
| `test_p5c_chain.py` (`p5c_pod.sh` in a stub sandbox) | rc 0, `ZZP5CCHAIN_PASS 83`, ~3.5 min | 12 behavioural mutants of the chain text; refusal arms: unfilled / mismatched pins, ckpt_last.pt, FAIL marker, NSHARD > nproc, label / workers unset, tree != manifest, token not verifying (at the gate, at the preflight, at the trainer start), failed post-gate, unverified egress, crashed / killed trainer, table not covering every row, label STOP |
| `test_p5c_profile.py` | rc 0, `ZZP5CPROF_PASS 261` (was 249: the three new suite scripts are asserted per profile arm) | as before |

## Remaining launch_gate-reading scripts, OVERLAY vs TIP
| script | OVERLAY | TIP | verdict |
|---|---|---|---|
| test_launch_gate.py --static --no-banked | 0 / ZZLG_STATIC_PASS 1180 | 0 / 1180 | same |
| test_launch_gate.py (full) | 1 (dev data dir absent, as round 1) | 1 in round 1 (same traceback) | environmental |
| test_traj_label_table.py --fast | 0 / 166 | -- (new at tip) | green |
| test_suite_nesting.py | 0 / 286 | 250 in round 1 | token count moves with the suite lists (3 new P5-C suite entries: 277 -> 286); passes |
| test_gsuite_tail / p7_integration / div / dose / rep / s1zsm / scorer1p6 / vdac / s1z _gate | all rc 0 (3, 225, ZZTEST_PASS...) | rc 0 in round 1 | same |
| test_p7_ship_list.py | 1 (`real.json`: needs the full package tree) | 1 (same) | environmental, identical |
| test_aux_s1_wiring (20), hn_content_fp (81), hn_ctx_anchor (72), p2r_chain / p2rep_chain / p6_overlap / rl_pickcon / rl_r0c / rl_safeobj (61) / rls1e_chain (936) / scorer1_gate / scorer1_mutation / sft4r_gate (153) / vdac_hb (133) / vdac_hb_accum (48) / vdac_hbc (97) / vdac_hbmem (83) | all rc 0 | not run on tip | green on overlay |
| test_carve_feed.py | 1 (`E_id_DEFAULT_vs_the_PRE_B8_trainer...` identity) | 1 (same failure) | environmental / pre-existing, identical |
| test_rls1_chain, s1aux_chain, sft4r_chain, vdac_stage0_forward | 1 (files under `raw/` not in the export) | 1 (same) | environmental, identical |
| test_s1z_preflight, s1zsm_preflight, scorer1p6_preflight | 1 (`preflight.jsonl` compare) | 1 (same) | identical on the tip |
| test_rls1_queue.py | run 1: 1 (3 s, "returned before S1 was over"); **re-run: 0 / ZZTEST_PASS 254 (721 s)** | 0 / 254 (732 s) | a timing flake in run 1, not a regression (the same test passes on both trees) |
| test_p6_chain.py | 0 / ZZP6CHAIN_PASS 431 (357 s) | -- | green (the hold had 839 s) |
| test_p6_levers.py (TMP bound) | 0 / ZZP6_PASS 205 (298 s) | -- | green (the hold timed out at the 1,500 s limit, rc 124, with an unbound TMP) |
| test_gate_suite_prerun.py | 0 / ZZTEST_PASS 144 (275 s) | -- | green |

## REGRESSIONS: 0
Every red on the overlay is also red on the tip with the same cause (dev data, `raw/` absent from the export, the preflight compare, the carve-feed identity) or passed on re-run (rls1_queue). No script that is green on the tip is red on the overlay.
## NOT RUN (stated)
`test_launch_gate.py` full was not re-run on the tip this round (round 1's tip run read the same traceback); the pinned run (`stack/` suite) is the MM's.
