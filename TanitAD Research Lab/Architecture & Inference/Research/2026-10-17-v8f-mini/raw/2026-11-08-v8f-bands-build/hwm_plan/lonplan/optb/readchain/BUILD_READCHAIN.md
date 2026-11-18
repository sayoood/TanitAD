# OPTB read chain: build report (2026-11-18). Tier T0. Staged only; nothing committed or pushed; box 2, the pod, git and every running job untouched.

## Headline
* The chain for the ten FT arms is built and its loading route is proven on a real FT-format checkpoint (4c396aff trainer, CPU). **MM launches one line** (below) once; it polls by script, reads each arm as it banks, and writes `out/manifest_decide.json` for `optb_decide.py`.
* **MM to decide:** (1) land `readchain/` with `read/`; (2) launch the chain (line below); (3) BAND-MET is now computed (ladder read `ia_read.py` + R35's banked `passes`, rule of `verdict_actsens.py:87-91`): confirm it is the card's meaning of "BAND-MET lost at a level"; (4) the decide manifest points `windows` at chassis 0's score windows for both chassis (see UNVERIFIED).
* Cost per checkpoint: planner dominates (ESTIMATED 35.6 + 27.3 min, INHERITED), plus ia_read (INHERITED 3-15 min) and ia3pk (MEASURED 5-6 s + load). Chain total ESTIMATED 10 x ~1.4 h = 14 h, sequential, so it keeps pace with box 2's ~16 h.

## 1 Route (`optb_route.py`)
* Problem (MEASURED, `test_route.py::test_ladder_tree_refuses_an_ft_checkpoint`): `wmact_lib.load_model` under the ladder tree raises `TypeError ... unexpected keyword argument 'wm_gamma_levels'` on an FT checkpoint; and even a tree that knows the fields would try to open `cfg.wm_rsp_probe`, a box-2 path (`load_rsp_probe`, `v8f_mini.py:755`).
* Route: `WMACT_CODE_ROOT = F:/TanitAD-scratch/actrsp/optb/tree` (stack + taniteval, stated blob-equal to 4c396aff: INHERITED from the build brief; `tree_manifest.sha256` pins its 1,865 `.py` sha256 as measured today and `check_tree_manifest()` re-checks it) + `install()`, which wraps `tanitad.models.v8f_mini.load_rsp_probe`: a path that does not exist is remapped to the local copy of the SAME basename (allow-list `readout_K10_R35_s{0,1}.pt`; dirs `amend1/out`, `optb_pkg/frozen/inputs`), and the ORIGINAL function then re-hashes the file against `cfg.wm_rsp_probe_sha256` (never skipped). No landed file edited; the planner (`lonplan_run.py` -> code of record `run_hwm` -> `wmact_lib.load_model`) and `ia3pk_read.py` / `ia_read.py` all run under `python optb_route.py <script> args` (runpy; the planner must be `--workers 1`, enforced: spawned workers would not carry the patch).
* Fixture: `make_ft_ckpt.py` ran the 4c396aff trainer (`--wm-rsp-readout fixed_k`, the REAL R35_s0 probe at a simulated box-2 path that is deleted afterwards; 2 steps). Its cfg names a path that does not exist (test-asserted).
* Tests (`test_route.py`, 9 passed, MEASURED): ladder refuses; route strict-loads (no missing / unexpected key) and records the remap; `ia3pk_read.load_ckpt` works through it; R35_s0 rollout sha256 identical ladder vs route (C1 parity in the read path; 6 real train windows, 10 steps; `1ed2658f...`) with 7,236,315 params both; wrong declared sha, one-byte-tampered local probe, unknown basename, missing local copy each refuse; tree manifest matches.

## 2 Chain (`optb_readchain.sh`, `.cmd`)
Per tag in RUN_ORDER (asserted equal to `optb_check.RUN_ORDER`): wait (sleep poll 120 s, up to 50 h) for `optb_<tag>/BANK.done` + `RUN.done` -> `wait_commit.py 5.5` before each CPU stage -> **evidence** -> **planner** -> **IA-3P-K + beside read** -> **BAND-MET** -> refresh `mech.json` + `manifest_decide.json`. Each stage writes `state/<tag>.<stage>.done` (rerun resumes). CPU env: 2 threads, `CUDA_VISIBLE_DEVICES=-1`.
* Evidence (`readchain_tools.py evidence`, computed from the banked files, nothing typed): sha256 of ckpt / summary / train_log vs `RUN.done` and `BANK_MANIFEST.json` (mismatch rc 3 = STOP); C2 = `optb_check.gamma_log_problems` on `train_log.jsonl` (rsp_w = 1.0 within 1e-4 at every logged step, gamma 100); C3 = `optb_check.c3_problems` on the banked selfcheck (`live.c3`, `live_first`; NULL arms must carry neither); C5 = Adam state on 141 of 141 trainable `wm.*` tensors, `wm.proj` bit-equal to the R35 start ckpt (sha256 vs `INIT_SHA`), non-WM tensors identical (rc 4 = STOP). C2/C3 are DATA (a false `c2_c3_hold` goes to the manifest, the chain continues); C5 is an integrity gate. The checker functions are imported from `C:/Users/Admin/optb_pkg/optb_check.py` unchanged (the box-2 chain's own code).
* Planner: exactly `score_A0_s0.cmd`'s command with `--ckpt <bank>/ckpt_last.pt`, `--ckpt-sha256` = RUN.done's, `--seeds 0 1 --lam=0 --seed-mode hold`, chassis windows (k=0 `actsens/read/out/C7_R35_s0/win`, sha 371c753b; **k=1 `lonplan/score_windows_s1/windows.npz`, sha 8833e292, as `score_chain.py` of record uses; `actsens/.../C7_R35_s1` does not exist**), `hwm_plan/latents_train_R35_s{k}.pt`, fresh `F:/TanitAD-scratch/actrsp/optb/plans/<tag>/`.
* IA-3P-K: `ia3pk_read.py read` (term's 1 sd, 0.4203) into `out/reads/`; beside (MM ruling 3) at IA-3's 1.0 m/s^2 into `out/reads_beside/` with delta computed from the arm's own `wm.scale_actions` (affine check, refuses otherwise).
* BAND-MET: `actsens/read/ia_read.py --name <tag> --run-dir <bank> --chassis-seed k` (code of record `ladder_read`, `LADDER_READ_CODE_ROOT` = optb tree, through the route), then `bandmet`: lost = R35's banked `passes` (`ladder_read/results/arm_R35_s{k}.json`) met a level (beats, healthy, not worse) and the arm does not. `band_met_lost` per chassis = any arm of it lost a level.
* Sentinels, each in its OWN file: `out/ZZ_STOP.txt` (first failure: tag, stage, rc, log path), `out/ZZ_DONE.txt`, `out/ZZ_ENDED_BY_BOX2_STOP.txt` (a box-2 `C:/Users/Admin/optb_state/STOP.txt` or `ARM_REFUSED_<tag>.txt` present before this arm banked: chain ends, rc 0, after the arms already banked). `chain.log` carries no sentinel text. A pre-existing `ZZ_STOP.txt` refuses a restart (move it aside).
* Manifest (`readchain_tools.py manifest`): arms with a complete `plans/<tag>/DONE`, baselines R35 = `lonplan/score/A0_s{k}` (lam 0 hold plans), `mech` = `ia3pk_read.py mech`, `controls.c2_c3_hold` = all evidence rows hold (null if none), `controls.band_met_lost` per chassis (null if none). `controls_detail` per arm is extra.

## 3 Tests (MEASURED): `test_tools.py` 5 passed
Evidence on a real trainer log + selfcheck reads C2 and C3 true; **C2 goes RED when rsp/L1 is scaled 1.5x** (deliberate regression); a one-byte-flipped ckpt is refused rc 3; the synthetic fixture (not an R35 descendant) is refused by C5 rc 4; BAND-MET reads False for R35's own passes (control), True with a planted level-2 loss, level 1 with `worse`; manifest aggregates controls. `test_route.py` 9 passed.

## 4 Dry run and timings
* B (tampered bank `FAKEb_s0`): MEASURED, refused at evidence in 4 s, rc 3, `ZZ_STOP.txt` written, `chain.log` has no sentinel (`tmp/dryB/`).
* A (`FAKEa_s0` genuine R35_s0 copy through every stage, planner `--n 8`, 1 seed, then unbanked `FAKEc_s0` with a STOP.txt): **PENDING at agent end** (launched detached via WMI, pid 1320). Result files: `tmp/dryA/chain.rc`, `tmp/dryA/out/{ZZ_ENDED_BY_BOX2_STOP.txt,state/*.wall,evidence,reads,bandmet,manifest_decide.json}`. Observed so far: evidence stage done, planner stage started (MEASURED).
* One full real planner seed on R35_s0 through the route: **PENDING** (`tmp/timing/timing.txt`, after A). Until then: 35.6 / 27.3 min per seed INHERITED.
* Dry-run deviations: tags are `FAKEa_s0` / `FAKEb_s0` / `FAKEc_s0` (ia_read needs the `_s<k>` suffix), evidence runs `--dry --lever null --init 0 --expect-adam 145` (an R35 copy has no response term and 145 Adam tensors), so its C5 line is expected false and is not enforced.

## 5 MM launches (exact WMI line, once; it waits for banks by itself)
`powershell.exe -NoProfile -ExecutionPolicy Bypass -File F:/TanitAD-scratch/actrsp/optb/readchain/launch_wmi.ps1 -Job optb_readchain.cmd`
(= `Win32_Process.Create("cmd.exe /c F:\TanitAD-scratch\actrsp\optb\readchain\optb_readchain.cmd < NUL", CreateFlags 0x08000200)`; the .cmd runs `C:\Program Files\Git\usr\bin\bash.exe optb_readchain.sh`; stdout `optb_readchain.out`.) Wait on the three sentinel files by name, never `chain.log`. Then `optb_decide.py --manifest out/manifest_decide.json --out ...`. Start it before box 2 banks its first arm or any time: it only polls.

## UNVERIFIED
* No FT bank exists: the planner, `ia3pk read`, `ia_read` and the full evidence (C5 with 141 tensors, `selfcheck.json` of a real gate) have never run on a real FT checkpoint. A `RUN.done` / selfcheck key renamed on box 2 would show as a stage failure with its log.
* Tree blob-equality to 4c396aff is INHERITED (brief); not re-hashed against git (git untouched).
* `optb_decide.py` takes one `windows` npz; chassis 1's plans use `score_windows_s1`. The manifest points at chassis 0's windows for both. Not checked whether the two npz carry identical (eid, now) lists; the MM or the decide author should confirm before the first label.
* `delta` for 1.0 m/s^2 assumes the arm's `scale_actions` is the frozen R35 scale (it is affine-checked, not compared to R35's).
* Planner with other jobs running (AMEND_7, TFAR) may be slower than the INHERITED minutes.

## Manifest (single copy unless noted; all on F: scratch, none banked to git/HF)
| artifact | path under F:/TanitAD-scratch/actrsp/optb/readchain/ |
|---|---|
| route + proofs | optb_route.py, route_probe.py, make_ft_ckpt.py, test_route.py, tree_manifest.sha256, fixture/FT/ (real FT-format ckpt, 2 steps) |
| chain | optb_readchain.sh, optb_readchain.cmd, launch_wmi.ps1, readchain_tools.py, test_tools.py |
| dry run | dry_run.sh, dry_all.cmd, time_plan.sh, mk_fakebank.py, tmp/dryA, tmp/dryB, tmp/timing (temp: delete once read) |
| docs | BUILD_READCHAIN.md, HANDOFF.md |
Outputs of the real chain (created at run time): `out/` here and `F:/TanitAD-scratch/actrsp/optb/plans/<tag>/`.

## Deletions log
Deleted at end of the build: PENDING (the dry-run `tmp/` trees are still in use; they are re-creatable). Already removed: `fixture/box2sim` (the simulated box-2 probe dir, a copy of a banked file), pytest basetemps F:/TanitAD-scratch/actrsp/optb/readchain/tmp/pt (48 MB) and tmp/pt2 (71 MB) (re-creatable).
