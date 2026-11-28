# BRIEF: prepare the P5-C kit land with its 3 pins (land-ready, NOT landed). Agent: Sonnet. MM, 2026-11-16

Read `Project Steering/AGENT_OPERATING_STANDARD.md` (repo `E:/Projects/TanitAD`) first.
- **Binding:** stage, never commit or push; end with a manifest; escalate integration.
- **Hold:** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_kitland/`. Scratch: `F:/tanitad_scratch/kitland_tmp/` (two levels deep).
- **Every test runs with:** TMP/TEMP/TMPDIR in scratch, `CUDA_VISIBLE_DEVICES=-1`, the tanitad venv `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, SERIALLY (never two test processes at once).

## ⚠️ Dev-box memory (binding)
The P7 navtest read's determinism seam runs on this box until about 22:00Z. It holds ~8.5 GB of commit, and two earlier attempts died from commit exhaustion.
- Before EACH test script, read the free commit: `(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB`, in GB.
- Start a script only if it reads >= 5 GB. Otherwise wait in a script loop, never in model turns.
- If it stays under 5 GB for 60 min, stop and report.

## Facts
**Mirror.** `C:/Users/Admin/tanitad-push`. Run `git fetch -q origin agent/arch-inf-20260803` and use FETCH_HEAD. Export blobs with `git cat-file blob` (`core.autocrlf=true`).

**Kit hold.** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_launchkit/`. Read these first: `LAND_NOTE_A31.md`, `REBASE.md`, `RUNBOOK_P5C.md`, `PROFILE.md`, `TESTS.md`.
- `files/` = 22 files under `PK/refe/`. `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
- The MM's 2026-11-15 check driver is `C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/e07efd49-d158-4484-85f3-d8888196bc46/scratchpad/kitchk.sh` (10 scripts, all rc 0).

**Two A3.1 replacements** (`LAND_NOTE_A31.md` §1). Both come from `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_a31/refe/`:
- `refe/p5c_pod.sh`, blob `11d657d9`;
- `refe/test_p5c_chain.py`, blob `67fda327`.

**The 3 pins**, filled in `refe/launch_gate.py` (lines ~444-447, now `None`). All three are MEASURED and final:
- `P5C_BASE_SHA256 = "2613fc17b9e5d63342c9f42120c5612a6ec7061fda72f83dac46ddd2e70fd3fe"`. This is P7's `model_final.pt`. Base B-P7 under PREREG_P5C §2.1, because P7 is NOT PROVEN vs P6.
- `P5C_LABEL_TABLE_SHA256 = "3b198314721fc7e75e6bd0ca9b569b5b0a56ba25178e4fb63a94b215995637a2"`, from `PK/raw/2026-11-16-p5c-window/label_a33/LABEL_SELECTION.md`.
- `P5C_RANK_POLICY = "rank0"`.

**Land guard** (`REBASE.md`).
- The 13 guarded tip blobs must still read as listed there; they did on tip 68850034 at 12:20Z.
- The 9 new files must still be absent.
- Re-check on the CURRENT tip and report the tip sha.

## Deliverable, in priority order
1. **Land tree.** Export the current tip's `PK/refe/` (and whatever else the tests read) to scratch, overlay the kit's `files/` and the 2 A3.1 files, then fill the 3 pins.
   - Report each file's blob.
   - Report the diff of `launch_gate.py` vs the kit's (it must be only the 3 lines).
2. **Pin-sensitive tests on that tree.**
   - Run the 10 kitchk scripts.
   - Run EVERY test that reads `launch_gate.py`. Find them with `grep -l launch_gate` over `PK/refe/test_*.py` and anything else in the repo that imports it. A pin edit moves profile fingerprints, and earlier ~69 such tests were found.
   - For any failure caused by the pin fill (a fingerprint or pin assertion), update THAT assertion in a copy, in the hold. Record old → new value and why.
   - ⛔ Change nothing else. A failure not caused by the pins is reported, not fixed.
3. **`SHIP_MANIFEST.json` and the ship list**, re-derived for `--profile refe_p5c` as the RUNBOOK says, on the pinned tree.
4. **`LAND_LIST.json`** in the hold. Each item is `[repo_path, abs_src, "raw", "base=<tip blob>|base=<absent>"]`; sources are files in the hold (LF, 0 CR in code); repo paths are ≤ 200 chars. Also write `HANDOFF.md` (≤ 2k tokens) with the tip sha used, the test table (script → rc → token count), the assertion edits, and the ship-list sha.

## ⛔ Do not
- Commit, land, push or ship anything.
- Touch the pod, box 2 or Thor.
- Run a GPU process.
- Edit frozen PREREG files or the kit hold in place (copies go in your hold).
- Delete outside your own scratch (literal paths; log each deletion).
- Exceed about 150k context.

**Report:** at most 300 words, plus a manifest.
