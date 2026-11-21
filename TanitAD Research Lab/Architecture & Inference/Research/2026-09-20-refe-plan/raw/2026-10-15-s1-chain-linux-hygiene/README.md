# S1 chain: two Linux-ship defects found after S1 (Master Mind, 2026-10-15)

Neither defect affected the S1 run. **Both will bite the next arm that reuses `s1aux_pod.sh`** (S1b scene tap / S1c more label logs, whichever the
scene-tap probe selects).

## 1. `test_s1aux_chain.py` shims had no exec bit: FIXED here

- **The defect.** `write_sh()` wrote the `nvidia-smi` shim (`:399`) and the fake-python shim (`:695`) as mode 0644. The chain runs both directly.
  - Git Bash on the dev box ignores the exec bit.
  - Linux does not: the shim returns **rc 126**.
- **Same mechanism as `test_p2rep_chain`**, found by that suite's pod G-SUITE pre-run (`raw/2026-10-15-p2rep-gsuite-pod/`), where it made
  `gpu_busy` read "busy" forever.
- **The fix.** It is identical to `test_p2rep_chain.py:71`: `os.chmod(path, 0o755)`.
- **Verified.**
  - Dev box, torch 2.11 venv, fresh LF archive of `8e106f9`, TMP on NTFS: **`ZZTEST_PASS 385`**, unchanged. Log: `test_s1aux_chain_devbox.log`.
  - **UNVERIFIED on Linux.** This box has no WSL, and the pod is training (no added load). The next pod G-SUITE pre-run of the S1 chain confirms it.

## 2. `s1aux_pod.sh:34` pins the CRLF form of `navtest.json`: OPEN, fix at the S1b/S1c build

MEASURED on `8e106f9`:

| copy of `FlyWheels/.../2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json` | bytes | CR | sha256[:12] |
|---|---|---|---|
| the git blob, and any `core.autocrlf=false` / LF archive (our current ship standard) | 753,185 | 0 | `3ba1c96488f1` |
| a Windows `autocrlf` checkout (what the store build and the S1 ship used) | 765,337 | 12,152 | `0b9c63895d4e` |

- **The pin.** `NAVTEST_SHA=0b9c63895d4e…` is the **CRLF** form. `build_navtest.log` records the same sha.
- **The consequence.** An LF ship of the chain REFUSES at `stage=data` with "absent or not sha256 0b9c63895d4e".
  - Reproduced here: the dev-box suite fails at `sandbox_runs` (`:825`) on an LF archive.
  - It passes with the CRLF fixture copied in.
- **The content is identical:** the same 12,152 lines, and the LF-normalised sha256 of both copies is `3ba1c96488f1`.
- **The fix, for the next build** (not applied here: it changes the chain's data-identity check, so it goes through that build's G-SUITE):
  - hash the CR-normalised bytes and pin `3ba1c96488f1…`, with a RED mutation for a genuinely different list;
  - or ship the CRLF file deliberately and say so.
- ⛔ **Do not re-pin to whichever form happens to be on disk.** That is the silent version of this defect.
