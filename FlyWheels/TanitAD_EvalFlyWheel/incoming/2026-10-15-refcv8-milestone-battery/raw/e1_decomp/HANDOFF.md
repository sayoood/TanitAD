# HANDOFF: e1_decomp (BAR-R8-E1 decomposition, refcv8 EMA-30k, compliant S2), 2026-10-18 ~21:10 Berlin

NOT committed and NOT staged (qland hold, no git writes per the brief). There is a single copy of everything, in PKG = `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery`.

## State

* **UPDATE 21:14: the gated re-run LANDED.** The gate opened at 21:13:43 with 4,656 MB free; `python finished` at 21:14:00 with ALL_PASS.
  * The JSON's `script_sha256` is now `0d721c89…`, which equals the current file.
  * `seeds.0.s2_widen_sets` is present:

    | set | windows / episodes | contribution |
    |---|---|---|
    | S2A | 434 / 37 | +0.0135 |
    | S2B | 424 / 73 | +0.0066 |
    | S2C | 556 / 71 | +0.0189 |

  * The E1 cell is unchanged.
  * The "JSON came from the script version just BEFORE the last patch" and "gated re-run is pending" bullets below are SUPERSEDED.
* **Code:** `PKG/code/e1_decomp_r8.py`, a NEW file. It has two stages: `compute` (about 25 s on CPU) and `render` (JSON → MD only).
* **Results:** `PKG/raw/e1_decomp/E1_DECOMP.json` and `E1_DECOMP.md`.
  * Controls M, R, K, Z, S, P and F: ALL PASS.
  * The analyst verdict lives in `VERDICT_ANALYST.md`. `render` prepends it to the `.md`.
* **The JSON came from the script version just BEFORE the last patch.** That JSON's `script_sha256` is `406d39d4…`. The last patch only ADDED two things:
  * `seeds.0.s2_widen_sets`: the sha12-keyed window lists of S2A, S2B and S2C for the next force roll;
  * the render sections.
  Every number in the verdict is in the current JSON.
* **A gated re-run is pending** (background bash, one poll every 20 s, at most 60 min). It waits for free commit ≥ 4 GB, then runs `compute`, which rewrites the JSON and `.md` with the same numbers plus `s2_widen_sets`. Its log is `PKG/raw/e1_decomp/run.log`. It ends with `python finished`, or with `GATE NEVER OPENED`.
* **Gate deviation:** the third `compute` run started at 3.72 GB free commit, below the brief's 4 GB gate. It ran 22 s with a small footprint (no dump is larger than tens of MB) and caused no failure.

## How to check / continue

1. `tail -3 PKG/raw/e1_decomp/run.log`. Read the ARTIFACT: `python -c "import json;r=json.load(open('E1_DECOMP.json'));print(r['controls']['ALL_PASS'],'s2_widen_sets' in r['seeds']['0'],r['script_sha256'][:16])"`.
2. If `s2_widen_sets` is absent (the gate never opened, or the job died with the session), re-run it under the gate. From `PKG/code`, with `CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code'`:
   `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 e1_decomp_r8.py compute`.
   This needs free commit ≥ 4 GB. The numbers are deterministic, so they will not change.
3. **Next arm (rank 1, BLOCKED).** A NEW S2 force-roll script is needed. `force_lon_reroll_r8.py` is a live-chain file (no edits) and has no HOLD arm.
   * Windows: S2A, S2B and S2C from `s2_widen_sets`.
   * Arms: N, F1 (oracle `lon_v7`), F_slow (HOLD on S2A, BRAKE_TO on S2B, via `set_r8_force(lon6=…)`; LON6 HOLD = 4, BRAKE_TO = 2), Fown and N2.
   * Pre-register the closure bar before rolling. The verdict suggests closing ≥ 50 % of the set's E1 contribution.
   * Run it after the force_lon roll, which is paused at 20/500. `/c/Users/Admin/mmops/relaunch_force_lon.sh` will restart it once the 30k queue is terminal and ≥ 18 GB of commit headroom is free.
4. **Rank 2 (GPU).** Run a K-draw ensemble (K = 4 or 8, two independent seed sets) of `os` on the compliant S2 set. ENS2 (K = 2) moved E1 by −0.005 to −0.006, separated. That read is −0.0138 [−0.0279, +0.0012], not separated.
5. **Rank 3.** When the 40k and 48,852 trees are banked, re-run on them. Two things in the code are specific to 30k:
   * the literals in `BRIEF_E1`;
   * `COMPLIANT_JSON`, which points at the 30k tree.

   So for a new tree:
   * first run `compliant_reread_r8.py read --tree <t>`;
   * put that tree's banked E1 cells into a NEW copy of the script (never edit a file a live chain runs from);
   * then switch `TREE`.

   The mask sha must stay `4ebc6b86…`, and control M asserts it.
