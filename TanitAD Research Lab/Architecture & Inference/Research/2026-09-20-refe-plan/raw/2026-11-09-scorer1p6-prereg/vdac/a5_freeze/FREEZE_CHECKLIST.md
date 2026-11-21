# FREEZE_CHECKLIST: PREREG_VDAC_P6_A5 (HB + HC pair), the MM's ordered steps

Hold: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/vdac_a5_freeze/`. `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. `V` = `$PK/raw/2026-11-09-scorer1p6-prereg/vdac`. Mirror = `C:/Users/Admin/tanitad-push`. All paths with forward slashes. Files LF.
Read the three open decisions in section 0 before step 1; the rest is mechanical.

## 0. Decisions for the MM before freezing (each is in the card; none blocks the freeze)
1. **K ledger: recommended NOT to append.** The card (section 6) and A2 (section 5) key arms at the S2 read only; S1 reads are navtrain and unkeyed; K feeds the navtest Bonferroni interval. `LEDGER_ENTRY.json` holds the exact entry (n = 45) if you overrule. See step 5.
2. **Pod directory:** ship a FRESH tree into `/workspace/refe-plan-vdac` (old one moved aside), because `vdac_pause_p7.sh:62` hard-codes it; re-minting the token is intended. A differently named directory needs a one-line pinned-file edit (card 9.6). `POD_RUN_PLAN.md` section 1.
3. **Optional dry-run pause** for HB (preflight + gate only; prints the unmeasured memory and speed). `POD_RUN_PLAN.md` section 3, recommended before the 10-hour HB-s0 arm.

## 1. Review the card text (2 min)
```
cd E:/Projects/TanitAD-artifacts/scorer_gym_20261107/vdac_a5_freeze
grep -n "<FREEZE_UTC>\|<CARD_SHA256>" PREREG_VDAC_P6_A5.md        # 2 lines carry <FREEZE_UTC> (3 and 254), 1 line carries <CARD_SHA256> (256)
```
Confirm D1 (lora lr 2e-4), D2 (bw 3.0, band 3.0 m), D3, D4 (card section 0), and the pre-registered fallbacks (9.2 batch split, 7.5 init heads, the G21 bf16 bar).

## 2. Fill the freeze time (the ONLY edit to the card)
```
T=$(date -u +%FT%TZ); echo "$T"
sed -i "s/<FREEZE_UTC>/$T/g" PREREG_VDAC_P6_A5.md
grep -c "<FREEZE_UTC>" PREREG_VDAC_P6_A5.md        # must print 0
grep -c "<CARD_SHA256>" PREREG_VDAC_P6_A5.md       # must print 1: it is a pointer and STAYS (the card cannot contain its own sha)
python -c "import sys; b=open('PREREG_VDAC_P6_A5.md','rb').read(); sys.exit(1 if b'\r' in b else 0)" && echo LF_OK
```
`sed -i` strips CRLF to LF even on no-match; the file is already LF, so this is safe; the python line proves it (do not use `grep -c $'\r'`, it reads 0 on CRLF).

## 3. Hash it and write the sidecar (format of A4's `.sha256`: `<sha> *<file>` then a `frozen_utc` line)
```
SHA=$(sha256sum PREREG_VDAC_P6_A5.md | cut -c1-64); echo "$SHA"
printf '%s *PREREG_VDAC_P6_A5.md\nfrozen_utc %s\n' "$SHA" "$T" > PREREG_VDAC_P6_A5.sha256
cat PREREG_VDAC_P6_A5.sha256
```
`<CARD_SHA256>` is filled ONLY in the sidecar and in `LEDGER_ENTRY.json` (if appended), never in the card.

## 4. Land the card and its sidecar (2 NEW files)
Target paths: `$V/PREREG_VDAC_P6_A5.md`, `$V/PREREG_VDAC_P6_A5.sha256` (beside A1 to A4). Use the coordinated lander with your usual hold (new files: the `expect.tsv` base is the literal NEW; land a COPY, because the lander rewrites CRLF sources in place; run the encoding and import guards; **the exact lander command forms are not in this package: UNVERIFIED here, use `qland-hold-hygiene` / `qland_one_c.sh` as you do for any card**). Do NOT touch `refe/`: the landed commit must keep every `refe/` blob equal to `a01be4a4`'s.
Then verify, on the pushed tip:
```
git -C C:/Users/Admin/tanitad-push ls-remote origin <branch>                                  # equals your landed commit <LANDED>
git -C C:/Users/Admin/tanitad-push show <LANDED>:"$V/PREREG_VDAC_P6_A5.md" | sha256sum        # equals $SHA
git -C C:/Users/Admin/tanitad-push diff --stat a01be4a4 <LANDED> -- "$PK/refe"               # EMPTY (blob-identical code)
```
Assert on the remote ref and the blob, not on the push's last line.

## 5. K ledger (only if you overrule the recommendation of section 0)
`ledger_K.json` max `n` = 44 (`TF-N2-A1`) at `a01be4a4`; check the CURRENT tip first (`git show <tip>:"$PK/raw/2026-11-09-scorer1p6-prereg/selectors/ledger_K.json"`, max `n`), edit shared registers from `git show <tip>:<path>`, not from E:. Fill `<CARD_SHA256>` and `<FREEZE_UTC>` in the `entry` object of `LEDGER_ENTRY.json`, set its `n` to max + 1, append ONLY that object to `reads`. UNVERIFIED that `selcore.Ledger` accepts `set: "S1"`. Otherwise skip this step: the keys are written at S2.

## 6. Dev-box tests the build agent could not run (blocks the ship if red)
Where `D:\Projects\TanitAD\data` exists, interpreter `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, `CUDA_VISIBLE_DEVICES=-1`, cwd = `$PK`, `PYTHONPATH` joined with `;` (Windows), scratch tests on F: not E:/C: (disk), `pytest --basetemp` parent must exist:
```
python refe/test_launch_gate.py            # in full: it stopped at test_launch_gate.py:2313 (FileNotFoundError, data absent) in the build agent's environment
python refe/test_vdac_gate.py              # ZZTEST_PASS
python refe/test_vdac_hb.py                # ZZTEST_PASS 133
python refe/test_aux_s1_wiring.py          # ZZTEST_PASS 20
python refe/test_dump_scores.py            # ZZTEST_PASS 16
S1_READ_V1="$V/s1a_read/s1_read.py" python refe/test_vdac_stage0.py && S1_READ_V1="$V/s1a_read/s1_read.py" python refe/test_vdac_stage0_forward.py
```
Then the ~69 launch_gate-reading tests (`pin-edit-moves-profile-fingerprints`). A pre-run of `test_vdac_hb.py` on the pod is part of the pause (the live suite runs `test_vdac_gate` there; `test_vdac_hb` is not a suite member).

## 7. Ship (dev box, then pod): `POD_RUN_PLAN.md` steps A and B
1. Export `refe code splits eval` from `<LANDED>`, run `launch_gate.py ship-manifest` (command in the plan, success `ZZSHIP_OK <n> files tree <hex> commit <hex>`), prune `eval/` to the six named files, tar with LF, `bash -n` the three scripts.
2. Pod: `mv /workspace/refe-plan-vdac /workspace/refe-plan-vdac.s1a_tree`, unpack into a fresh `/workspace/refe-plan-vdac`, run `refe/p7_ship_list.py check --pkg ... --list vdac_ship_list_a01be4a4.json` (`ZZP7SHIP_OK files=146`), place `PREREG_VDAC_P6_A5.md` in `/workspace/data/refe_vdac/` and `sha256sum` it equal to `$SHA`.
No pod access is needed for steps 1 to 6.

## 8. Gate and launch (pod): `POD_RUN_PLAN.md` step D
1. (Recommended) the dry-run pause: `VDAC_STOP_AFTER=gate_hb-s0 ... vdac_pause_p7.sh <p7_pod.sh pid> <train.py pid> A5HB`. Read `hb-s0/preflight.log` for `gpu_mem_gb` and `s_per_microbatch`; send them to the PI. If memory refuses: card 9.2 (before any training).
2. Real pause 1: `... vdac_pause_p7.sh <p7_pod.sh pid> <train.py pid> A5HB` (HB-s0 then HB-SHUF; 36,000 s and 14,400 s timeouts). The token is minted by the chain inside the pause, never by hand.
3. Real pause 2: `... A5HC` (HC-s0 then HC-SHUF; 21,600 s and 7,200 s). Do not wait on the pod: one scripted watcher per pause, wake on `ZZVDAC_CHAIN_DONE <ARM>` or `PAUSE_STOPPED_EARLY` / `P7_RESUMED` in `pause.log`.
4. When a stage ends, **have the next stage's pause armed** (an arm that fails its bar leaves the next arm, not a verdict: Rule Zero). A failed HB-s0 preflight does not hold HC.

## 9. Before the FIRST read (not before the launch)
1. Write the reader's freeze record (`READER_FREEZE`-style): sha256 of `eval/s1_read_v2.py` (`796aea21506e...`), its tests (`eval/test_s1_read_v2.py`), the manifest, the regression identity on the S1a inputs (all four values and CIs digit for digit), the mutation arms RED. Card 7.4.
2. Write the manifest of card 7.5 BEFORE opening the pulled files. Pull, verify by sha256 against the HF LFS sha256, run the read command of `POD_RUN_PLAN.md` step F.

## 10. Register updates (the MM, same turn as a read)
`D-VDAC-S0-1` (pair selected; card A5 frozen), the row of each arm after its read, the claim rows that depend on it. Nothing in this package edits a register, the ledger, `GOALS_AND_CLAIMS.md` or `dagland/`.

## 11. Cleanup (rule, 2026-11-07)
After the reads are banked and verified: delete pulled copies whose source (HF, sha256-verified) still exists, by literal absolute path, logging path and size; never delete `model_sft_A.pt` / `model_sft_B.pt` on the pod (not banked), the old `/workspace/refe-plan-vdac.s1a_tree` is re-creatable from git and may go once nothing needs it.
