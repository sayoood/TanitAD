# HANDOFF: SELRULE-DAC, the fit is FROZEN; the confirm waits for P2's full table

PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. This package = `<PK>/raw/2026-10-16-selrule-dac/` (written to the temp archive tree; E:/D: untouched; NOT committed, NOT pushed).

## State
- FIT done (CPU, 3.5 s). **Frozen rule: V(0.8)** (DAC veto, tau 0.8, fall back to B). `FIT.json` sha256 `44d6d2c1ad5dce9388899784e9ba8d8f5bb88b52e348d68341575ded462285a9` (sidecar `FIT.json.sha256`, read-only, write-once).
- Fit scorer is **u400 on the pre-P2 fan, NOT P2** (Amendment 1 A1 in `PREREG_SELRULE_DAC.amended.md`). No P2 RL-val dump exists. The transfer of tau is an assumption.
- `confirm.py` is wired and tested on a synthetic table; smoked on the 509-token pilot for plumbing only (`SMOKE_PLUMBING.json`: shapes and field checks, NO PDMS). The real confirm has NOT run.

## Wait for
`<PK>/raw/2026-10-16-p2-full-table/P2FULL_DONE` (and `table.npz` + `proposals.npz` in the pilot's format: `token, pdms [N,64], sub [N,64,6] (nc,dac,ep,ttc,c,ddc), valid, logits [N,64,6], proposals [N,64,8,3], pick, sub_names`).
Check without a model turn: `ls "<PK>/raw/2026-10-16-p2-full-table/P2FULL_DONE" "<PK>/raw/2026-10-16-p2-full-table/P2FULL_FAILED" 2>/dev/null`.

## The exact command (once DONE exists; run it ONCE; about 2 min of CPU + about 5-10 min for the families6 leg; run it in the background and poll the file)
```
cd "<PK>/raw/2026-10-16-selrule-dac"
C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I confirm.py --expect-fit-sha256 44d6d2c1ad5dce9388899784e9ba8d8f5bb88b52e348d68341575ded462285a9 > confirm_run.log 2>&1
```
Needs the package to sit in its PK tree (it imports `<PK>/eval` and `<PK>/refe`). Output: `confirm_out/CONFIRM.json`, `CONFIRM.md`, `families/`. Exit 6 + `ZZCONFIRM_STOP <reason>` = a gate stopped it, nothing was read (disjointness, token set, valid, finite, B == table pick on >= 99 %, mean PDMS at the table pick == 88.82 +- 0.01, FIT sha). It refuses a second run into the same `--out-dir`.
Last line on success: `ZZCONFIRM_DONE outcome=<ADOPT|NOT PROVEN|REFUTED|NOT ADOPTED (adverse component)|UNRESOLVED (family leg not computed)> dPDMS=...`.

## After it runs
1. Read `CONFIRM.md`; the outcome follows the committed table (section 4 + A7). `NOT ADOPTED (adverse component)` is the one cell the prereg did not write: escalate, do not map it.
2. Update `Project Steering/GOALS_AND_CLAIMS.md` in the same turn (the register is the live claim ledger).
3. If ADOPT: carry V(0.8) into P6's read as a SECOND, separately registered seam beside `navsim_v1`; never replace it.
4. If NOT PROVEN / REFUTED: the zero-tail lever is representational (P6 trunk, P7.1); the oracle controls size the prize (RL-val: a perfect DAC head +2.13, a perfect NC head +0.75, both +2.68 PDMS; the navtest oracle shares are in CONFIRM.json).

## Do NOT
- re-run `selrule.py fit` or touch `FIT.json` (write-once; the sidecar hash is checked);
- run `confirm.py` on the pilot table in full mode (it refuses: no marker); `--smoke` is plumbing only;
- change the family-leg reading after seeing a number (A6: literal reading; `UNRESOLVED` if families6 cannot run).

## Decision waiting for the Master Mind (also the headline of the hand-back; the harness refused a separate REPORT.md, so the findings are in the hand-back text and in FIT.json)
Run the confirm on this transfer fit as is (it costs nothing), or first demand a P2 RL-val dump (a GPU re-dump + 25-85 CPU-h of relabelling, `PREREG_RL_S1` section on variant B).
The fit is flat (V(0.8) = B + 0.021 [-0.001, +0.067] on RL-val, 3 of 3,000 picks changed), so expect NOT PROVEN unless P2's DAC head is much less confident on its picks than u400's; that is a reading of the fit, not a prediction to bank.

## Wiring evidence
`test_selrule.py` 67 checks / 13 tests / 4 deliberate-regression arms RED. `test_confirm.py` 20 checks, 23 with `--families` (family leg run end-to-end through families6.py on SYNTHETIC straight-line poses: wiring only, the values mean nothing). Smoke on the pilot: `SMOKE_PLUMBING.json` (shapes and field checks only). `selrule.py` sha256 `6309222c...` is the code that produced `FIT.json` (recorded inside it; unchanged since).
