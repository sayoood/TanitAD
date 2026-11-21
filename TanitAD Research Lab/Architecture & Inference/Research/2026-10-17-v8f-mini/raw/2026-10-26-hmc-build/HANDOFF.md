# HANDOFF: PREREG_V8F_HACC_MC (v8F-mini, T1-kin, zero training, CPU). BUILT AND SMOKED, NOT LAUNCHED. 2026-10-26 ~04:50

## State
Bakes done, chain + reader built and run end to end on a 24-window SMOKE (12 runs, tool 81aa3e36, `--limit causal`). No eval row of any arm exists. C-6 run now: PASS.
**Two PREREG controls cannot pass as written; both gate only the reader, no run depends on them. The MM must rule before the full read** (`INTERPRETATIONS.md` A, with the proposed amendment text):
1. **C-1** bar 1e-6 m: |d| is the float32 rounding of `plan`: SMOKE reads 0.50 ulp32 (3.2-3.7e-6 m, plan up to 98 m). Reader: `--c1-bar ulp32 --c1-amendment ID`.
2. **C-4 clause 2** "GT := P gives c = 0": reads **0.3094 / 0.3489** (the fit is pinned by D2-continuity with HACC's held tail; data-only c = 0.084 / 0.072; the real fit 0.3071 / 0.3468). Reader: `--c4-clause2 tail0 --c4-amendment ID`.

## Launch (TOOL_TREE exists: 81aa3e36 stack+taniteval at `E:/Projects/TanitAD-artifacts/v8f-mini-hacc-mc/tool_81aa3e36`)
```
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags=[uint32]0x08000200}
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c C:\Users\Admin\hmc\hmc_chain.cmd E:/Projects/TanitAD-artifacts/v8f-mini-hacc-mc/tool_81aa3e36'; ProcessStartupInformation=$si}
```
Markers `ZZHMC-DONE 12` / `ZZHMC-STOP <why>` in `C:/Users/Admin/hmc/logs/chain.out`. Resume = same command. Run output on E: `.../v8f-mini-hacc-mc/run` (about 2 GB; C: has 0.5 GB free). Expected: ~45-70 min (T1KIN causal measured 5.5 min per m10 run, 2 lanes; +20 s pin check per run; m30 shorter; the suite is unmeasured). The chain refuses a tool without BOTH flags (tested on t1kc, hlat/frozen: REFUSED; stub: pass), pins tool tree + 6 ckpts + report + tables, runs C-7 (suite, one pytest) first, then the matrix MC-H, STEEP, MC-F at m10 then m30 per seed, then the reader's controls-only pass (`C:/Users/Admin/hmc/read/READ_CONTROLS.json`; no row is opened).
Read: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe C:/Users/Admin/hmc/hmc_read.py E:/Projects/TanitAD-artifacts/v8f-mini-hacc-mc/tool_81aa3e36 [amendment switches]` -> `read/HMC_READ.json` (~5-8 min est.). Any control FAIL gives REFUSED and no row (`--force-rows-under-refusal` writes `rows_UNDER_REFUSAL.json` only). Do NOT change `--batch` or OMP threads (C-5 is a same-composition identity).

## What each control must read
P-1 PASS (right ckpt, flags, 141 files). C-1 literal FAILS (see above); regression (HACC table on MC-H ticks) 1.5-1.7 m. C-2 > 1e-2 m (SMOKE 12-31 m). C-3 exactly 0.0 on MC-H/MC-F; HACC/STEEP tables non-zero; HACC TRAIN known value -0.15876 / -0.07263 (bar -0.1588 / -0.0726 +-2e-3, n 1,747) PASS. C-4 clause 1 PASS (0.0021 / 0.0018), clause 2 FAIL; TRAIN bar PASS (MC-H +0.0024 / +0.0022, MC-F +0.0056 / +0.0060, STEEP +0.0009 m ADE6 vs HACC, bar 0.0604). C-5: cl-ol 0.0, floor vs banked P10 <= 1e-4 (SMOKE 1.5e-5), floor ADE6 2.48699212, m10 floor = banked <= 1e-9 (SMOKE 1.1e-3 = 24-window batch artefact; expected 0 on the full grid). C-6 **+0.0210 [+0.0099, +0.0351] / +0.0082 [+0.0026, +0.0148], n 2,106 windows, 135 episodes, abs diff 0.0** PASS. C-6b (M1 0.0536 / 0.0277, M2 +0.0202 / +0.0072, P10 0.0137) PASS.

## Artifacts
- Bakes (E:`.../v8f-mini-hacc-mc/`): `M2b_s{0,1}_hmc_{MCH,MCF,STEEP}.pt` (only `hacc_w` differs from the shipped HACC ckpt; bit-verified). sha256, table shas, C-4, TRAIN qualification (reproduces probe12 to 1e-12): `C:/Users/Admin/hmc/BAKE_REPORT_MC.json`; tables `tables/`. c: MC-H 0.2002 / 0.2198, MC-F 0.3071 / 0.3468.
- Code (`C:/Users/Admin/hmc/`): `hmc_bake.py hmc_lib.py hmc_controls.py hmc_read.py hmc_chain.sh hmc_chain.cmd(CRLF)`; hashes `BUILD_SHA256.txt`; reused sources snapshot `ref/` (= repo 8d4b936d); inputs `inputs/` (HACC float64 bank sha-verified).
- SMOKE evidence (all labelled SMOKE): `smoke/chain/` (12 runs, READ_CONTROLS.json, rows_UNDER_REFUSAL.json), `smoke/tool/SMOKE_TOOL_CONTROLS.json` (tool 81aa3e36 vs hlat tool: P10 floor loop diff 0.0, tick keys equal, causal header present), `smoke/SMOKE_CONTROLS.json`, `smoke/wall_times.txt`, `smoke/guardtest/`.
- Nothing is staged or committed (per brief). The MM lands `hmc/` (code, `BAKE_REPORT_MC.json`, tables, `ref/`) into the repo.

## Interpretations
All quoted in `INTERPRETATIONS.md` B (m30 flags; F/2 0.06035 with 0.0604 / 0.06037 sensitivity; M1 / M2 = probe2's; overlapping rows: first in table order, all listed; C-3 HACC regression; C-5 m10 floor tolerance).
