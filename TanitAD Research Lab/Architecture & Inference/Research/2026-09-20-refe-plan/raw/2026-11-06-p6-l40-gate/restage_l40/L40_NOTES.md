# P6 restage on the L40 pod: what differed from the tested driver (2026-11-06)

The repo-side fixes are in `raw/2026-10-25-p6-restage-v2/` at this commit. This folder holds the pod-side record.

| item | what happened | fix |
|---|---|---|
| resume step | the driver's default was 4,321; HF main is at **6,266** | `pod_l40.diff`, one line, `P6_EXPECT_RESUME_STEP` default |
| D9 file count | the driver expected the packer log's **107,860**; the tar holds **103,037** parts (= the rank-0 rows, MEASURED) | `D9_FILES_EXPECT=103037` on the pod; the default is now corrected in the driver |
| no-op rebuild | the prebuild's cache stage printed `built=103037`, but **0 parts were rewritten** (original 10-22 mtimes, 9 s) | `RS_ALLOW_D9_BUILT=1` on the pod; documented in HANDOFF.md |
| watcher | `watch_l40.sh` matched a FAIL line from an EARLIER driver attempt | it now reads only the log after the last `driver start pid` |
| **G-LIVE-LAB** | at `--label-workers 6` the gate REFUSED: GPU stall 5.4 % in one window, against the 5 % bar (`../FAIL_f2f604f5d806.json`) | PREREG_P6 **Amendment (e)**: 13 workers (the chain's `CORES-2`), relaunched by `relaunch_e.sh` (ARGS sha `f74aee47cf0b708f`), watched by `watch_e.sh` |

The two orphan-kill lessons from this restage:
- Kill by an explicit pid from a comm-based list. An args pattern once matched the ssh shell itself.
- A dying orphan can hold `driver.lock` for seconds. Check `/proc/*/fd` before relaunching.

`HANDOFF_L40.md` is the restage agent's handoff, verbatim. Its open risks 3 (the D9 count) and 4 (G-LIVE-LAB at 6 workers) both came true.
