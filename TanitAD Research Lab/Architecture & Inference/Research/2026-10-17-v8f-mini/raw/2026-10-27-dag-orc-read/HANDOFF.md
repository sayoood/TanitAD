# HANDOFF: dag_orc_read, DAG-ORC's reader + frozen RF0 (launch preconditions 1 and 2 of PREREG_V8F_DAG_ORC), 2026-10-27

Everything is under `E:/Projects/TanitAD-artifacts/dag_orc_read/` (E: only; nothing staged in git, nothing committed, box 2 not contacted, no DAG-OACC / FLAT-DAG output opened). `MANIFEST.txt` lists every file with its sha256.

## State (FROZEN reader sha256 `a4907f4e4707de8005cc4865544ac56e35a7b626c07a4eb36a6f97625b0eb943`)
* **P-0 DONE.** `P0_RF0.json` (sha256 `1608d70b21b6b6a45136a6d96828f70b5bbabd9c0ceb84db136927593c79ad0f`, also in `P0_RF0.sha256`), produced by `p0_rf0.py` from the banked DAG-0 / ORC-0 causal dumps with Read A's reader unchanged. Re-run reproduces the bytes. Do NOT regenerate or edit.
* **Reader built:** `dag_orc_read.py` composes `ref/dag2_read.py` + `ref/dag_read.py` + `ref/readA_code/` (sha-pinned copies). PREREG pinned: `PREREG_V8F_DAG_ORC.md` = the 3b7b404e blob, sha256 `38bc688b...`.
* **Proof, all on the frozen sha:** `dag_orc_selftest.py` 288/288; `dag_orc_mutation.py` 135/135 mutants killed, control green (`mutation.json`); `RC_DAG2.json` K-RC PASS, 0 differences (bound to the reader's sha); `DRY_RUN.json` ends OK with A / REP / B reported missing; `SMOKE.json` = stand-in run of the non-rc read path (NOT A READ); `--static-check` OK.
* `logs/final_steps.out` shows `mutation rc=1` for the FIRST full run: one mutant's anchor had gone stale after the `prereg_problems(path, pin=None)` edit (harness-only, reader unchanged). Anchor fixed, one mutant added, the full run repeated: `logs/final_mutation.out` = 135/135.

## Next steps (a fresh cheap agent)
1. Do not edit `dag_orc_read.py`. If you must: re-run selftest, mutation, `--rc-dag2` (about 8 min), `--dry-run`, `--smoke-standins`, then `mk_manifest.py`.
2. When A / REP (and B) exist: build `runs.json` (schema in the reader's docstring; `build` = the K-BUILD record) and run `python dag_orc_read.py --manifest runs.json --out DAG_ORC_READ.json`. REFUSED = rc 2 (nothing written), blocked slots = rc 4. Floors RF are written before any A / B contrast is computed.
3. Land (INTEGRATE, single-copy on E:): `dag_orc_read.py`, `p0_rf0.py`, `P0_RF0.json`, `P0_RF0.sha256`, `dag_orc_selftest.py`, `dag_orc_mutation.py`, `ref/`, `PREREG_V8F_DAG_ORC.md`.

## Not built / not proven (said in the reader's output too)
* T0 four-family block (`orc_families.py` pointed at A / CTRL): NOT RUN, no instrument yet for DAG-ORC slots.
* Beside rows M-LAUNCH, the anti-echo block, M2 / L2 on the GT-launch subset: NOT BUILT.
* The heavy path (real dumps for A / REP / B) is proven only by K-RC (DAG2 / DAG2-0) and the stand-in smoke; no DAG-ORC data exists.
* Interpretations to confirm at the first read: S-POS rows for X - Y = difference of the two runs' (S - CTRV, ticks >= 1) contrasts (agrees with the banked point difference at 4 d.p.); B's floors are read from the same RF0 / REP - CTRL; any control failure makes the arm REFUSED globally (fail closed).
