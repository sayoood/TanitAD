# HANDOFF: S1c auto-reader (state at 2026-10-16 ~10:30 Berlin; <= 2k tokens)

**State.** Built and dry-run; NOT launched, NOT committed, NOT pushed. No S1c number exists (live listing 08:08Z: `arms/H1u10k-s0/` has 0 files). Package = PK `.../Research/2026-09-20-refe-plan/raw/2026-10-16-s1c-read/` (an archive of tip 8f7fedf, not a worktree: nothing is staged; every file is a SINGLE COPY until the MM lands it; a code+docs backup copy is at `C:/Users/Admin/s1c_read_backup/`).

**Exact launch line** (detached via WMI, resumable; the same line resumes; `-Extra "--force"` re-runs after DONE):
`powershell -NoProfile -File "<repo>\TanitAD Research Lab\Architecture & Inference\Research\2026-09-20-refe-plan\raw\2026-10-16-s1c-read\launch_s1c_autoread.ps1"`
It must be launched from the LANDED package (the driver finds `eval/`, `refe/` as siblings via `PK = package/../..`). Needs: `C:\Users\Admin\venvs\{tanitad,driverl-eval}`, `D:\Projects\TanitAD-artifacts\{aux_navtest_forward,s1_read_cache,_tlane_work}` (if D: flipped to E:, `subst D: E:/`), `Keys.txt` (token read in place, never printed/argv/inherited).

**What it does by itself.** Polls `Sayood/tanitad-refe-s1aux` every 10 min (<= 48 h) for `arms/H1u10k-s0/` complete (13 files; `model_sft_B.pt` listed only) -> fetches to `C:/Users/Admin/s1c_work` (verified) -> evidence + C1 -> C3 forward (GPU under `devbox_gpu.lock` for <= 30 min, never breaking it; else CPU, ~15-40 min) with S1's H1-s0 as comparator in the same call -> harness perception (K2, budget contrast), `nat_p1` (natural population), `s1c_veto` (V-S1C), `s1c_desc_selection` -> `out/RESULT.md` + `RESULT.json` -> marker `out/S1CREAD_DONE` (recorded O1-O4) or `out/S1CREAD_FAILED <phase/reason>` (INVALID included). Progress: `out/autoread.log`, `out/autoread_state.json`.

**Next step for the MM (in order).**
1. Review `QUESTIONS.md` (13 resolutions; veto any by editing the cited line; tests pin the current ones). Land the package into the repo.
2. Launch with the line above (before or after the pod starts; ~16:00Z start, ~4.3 h train).
3. On the marker: read `out/RESULT.md`. The row's committed NEXT lever is quoted verbatim in it (Rule Zero: execute it the same turn; the O2/O3 label-coverage lever is 0-A40). Update `GOALS_AND_CLAIMS.md` the same turn. Stamp every S1c-vs-S1 difference "separated, single seed, not yet a lever claim".
4. FAILED + INVALID: RESULT.md still exists with the INVALID row; find the cause (C1 / run-check / evidence line in the log).

**Open risks.** (a) CUDA forward + its 1e-4 comparator tolerance are UNVERIFIED on a real GPU (lock was held by `refcv8-battery-step5000`); CUDA miss falls back to CPU (auto). (b) The INVALID extension to "evidence contradicts its label" is mine (Q2). (c) STRATEGIC is PAUSED (PI 2026-09-19 item 25), printed as such; never extend it.

**Verify before trusting** (all green at hand-off, 2026-10-16 ~11:00 Berlin): `test_s1c_autoread.py` 186 (tanitad venv), `test_s1c_veto.py` 18 (tanitad), `test_nat_p1.py` 63 (driverl-eval), plain scripts, `ZZTEST_PASS n`; dry run `python -I -X utf8 s1c_autoread.py --dry-run --no-live` -> `ZZS1CAUTOREAD_DRYRUN_PASS`, identities 0.0 (REPORT content is in the hand-back message: the harness blocked writing `REPORT.md`).
