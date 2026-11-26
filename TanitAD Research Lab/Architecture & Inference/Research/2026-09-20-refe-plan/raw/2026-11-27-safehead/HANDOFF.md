# HANDOFF: L2 SAFE-HEAD (`H-REFE-SAFEHEAD-1`), built and tested on synthetic data, 2026-11-26 (folder dated 2026-11-27 per the brief)

## State
- **The card is frozen:** `PREREG_SAFEHEAD.md` (sha256 in `PREREG_SAFEHEAD.sha256`), before any SH number.
- **The code is in `CODE.sha256`.**
- **Tests, synthetic, CPU:**
  - `refe/test_safehead.py` (full) **14/14 PASS**, including the 11 gate regression arms and the end-to-end mint → verify → trainer token run;
  - `test_safehead_read.py` **6/6 PASS**;
  - all **40 pre-existing launch-gate profiles unmoved**.
- **The 56-file REFe gate-test regression** (base tip `1df04b58` vs patched) is running WMI-detached, serially, commit-gated at 8,000 MB (`E:/Projects/TanitAD-artifacts/safehead/regression/cmp_chain.sh`, which is resumable). Its results go to `regression/{base,pins7,patched}.tsv` and then `COMPARE.tsv` (`regressions N` on the last line), with the marker `CHAIN.DONE`.
  - ⛔ Land only on `regressions 0`.
  - Two base tests already fail on the unpatched tip (`test_carve_feed.py` rc 1, `test_hn_content_fp.py` ZZTEST_FAIL 1), so they are not caused by this patch.
  - After `CHAIN.DONE`, delete `F:/safehead_tmp` (about 60 MB; the sandboxes can be re-created from git) and log it.
- **Nothing has been trained on real data, and no gate token exists.**

## Preconditions (check each; ⛔ do not start without them)
1. `F:/safehead_cache/markers/chain.DONE` exists. That is the data agent's navtrain cache: 2,272 tokens, 29 train and 6 held-out logs.
2. ⛔ **A NAVTEST `s` cache over all 12,146 tokens.** It is `split = navtest`, made by the data agent's tool (`DATA_CONTRACT.md`). The read cannot run without it.
   - If `chain.DONE` covers only navtrain (plus the 60-token navtest check), this is a **GPU dump of about 3.2 h (ESTIMATED)** under `devbox_gpu.lock`, owned by the data agent.
   - Training does not need it; the read does. Train first, then wait for it.
3. **The MM has landed this folder and `land_at_PK_refe/*`** (5 new files plus `launch_gate.py.patch`) on the branch. `LANDED` = that 40-hex commit.
4. **C1's `model_final.pt`:** use the path in the cache's per-log `meta` (ckpt path + sha256). Verify sha256 = `f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4`. ⛔ Never use the `trkscore/`, `ensscore/` or `scorerft_read/` copies.
5. **Disk:** E: needs ≥ 2 GB free. All runs are CPU, with `CUDA_VISIBLE_DEVICES=-1`, and there is no GPU anywhere in training.

## Exact steps (a fresh Sonnet agent; Git bash; `PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe`)
```bash
PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TIP=F:/TanitAD_safehead_tip; PKG="$TIP/$PKR"; RUN=E:/Projects/TanitAD-artifacts/safehead/run1; mkdir -p "$RUN" "$TIP"
git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive $LANDED "$PKR/refe" "$PKR/splits" "$PKR/code" "$PKR/eval" "$PKR/raw/2026-11-27-safehead" | tar -x -C "$TIP"
cd "$PKG/refe"; export CUDA_VISIBLE_DEVICES=-1
$PY launch_gate.py ship-manifest --pkg "$PKG" --commit $LANDED --git-dir C:/Users/Admin/tanitad-push/.git --repo-prefix "$PKR/" --out "$PKG/SHIP_MANIFEST.json"
$PY test_safehead.py --fast | tail -1                                   # ZZSAFEHEAD_PASS 12
# 1. the contract, the navtest log list, the sealed index (into RUN, never into the cache)
$PY ../raw/2026-11-27-safehead/safehead_read.py prep --table "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next/propscore/full_C1_s*of2.jsonl" --out "$RUN/navtest_logs.txt"   # ZZSH_PREP_OK (136 logs)
$PY safehead_data.py check --index F:/safehead_cache/index.csv --heldout-logs ../splits/sft_heldout_logs.txt --navtest-logs "$RUN/navtest_logs.txt"           # ZZSH_CHECK_OK, leak []
$PY safehead_data.py seal --index F:/safehead_cache/index.csv --out "$RUN/index_sealed.csv"                                                                 # ZZSH_SEAL_OK
C1=<the verified C1 model_final.pt>
A="--cache-index $RUN/index_sealed.csv --init-from $C1 --heldout-logs $PKG/splits/sft_heldout_logs.txt --navtest-logs $RUN/navtest_logs.txt --steps 2000 --lr 0.0003 --batch 32 --gamma 2.0 --rows 0,1,3,5 --save-steps 0,250,500,1000,2000"
declare -A ARGS=([s0]="$A --out $RUN/sh_s0 --seed 0" [s1]="$A --out $RUN/sh_s1 --seed 1" [red]="$A --out $RUN/sh_red --seed 0 --permute-labels-seed 20261127")
# 2. per arm: refusals (instant) -> the gate (a PASS token) -> the trainer under that token (seconds, CPU)
for arm in red s0 s1; do
  $PY launch_gate.py refusals --profile refe_safehead_$arm --args "${ARGS[$arm]}" | tail -1                                                     # ZZGATE-CLEANZZ
  $PY launch_gate.py run --profile refe_safehead_$arm --pkg "$PKG" --args "${ARGS[$arm]}" --out-dir "$RUN/gate_$arm" | tail -2                    # ZZGATE-PASS-<12>ZZ
  TOK=$(ls "$RUN/gate_$arm"/PASS_*.json)
  TANITAD_SH_GATE_TOKEN="$TOK" $PY safehead_train.py ${ARGS[$arm]} | tail -1                                                                    # ZZSH_TRAIN_DONE ... chosen_step N
done
```
- Each gate run takes about 3–6 min (ESTIMATED; G-SUITE runs `test_safehead.py --fast`, about 2 min, plus `test_launch_gate.py --static --no-banked`).
- ⛔ A FAIL token means no training. Read `gate_<arm>/evidence_*.json`; never hand-edit a token.
- **STOP-CHECK after training:**
  - `sh_s0/heldout_select.json` `chosen_step`. **If it is 0, the outcome is NULL-HELDOUT:** do not read navtest for a decision; the next arm is SH-MLP (card 8.1).
  - `sh_red`'s step-2000 safety-miss rate must exceed its step 0 (CTRL-RED). It is checked again in the read.

```bash
# 3. the read (after the navtest s cache exists; seal its index into RUN too)
$PY safehead_data.py seal --index <navtest cache index.csv> --out "$RUN/nt_sealed.csv"
cd "$PKG/raw/2026-11-27-safehead"
$PY safehead_read.py read --table "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next/propscore/full_C1_s*of2.jsonl" \
  --proposals E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p5c_c1.proposals.npz --seam E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p5c_c1.npz \
  --navtest-index "$RUN/nt_sealed.csv" --train-index "$RUN/index_sealed.csv" --heldout-logs "$PKG/splits/sft_heldout_logs.txt" \
  --run-s0 "$RUN/sh_s0" --run-s1 "$RUN/sh_s1" --run-red "$RUN/sh_red" --eval-dir "$PKG/eval" --refe "$PKG/refe" \
  --card PREREG_SAFEHEAD.md --card-sha256 $(cut -c1-64 PREREG_SAFEHEAD.sha256) --out "$RUN/read"          # ZZSH_READ <outcome> (provisional)
# 4. the four families (CPU, navsim-crun; smoke first, then full: launch DETACHED, ~1-2 h, ESTIMATED), then confirm
SMOKE=50 bash run_fam_chain.sh "$RUN/read" "$TIP"; bash run_fam_chain.sh "$RUN/read" "$TIP"     # full run: WMI-detached, marker fam/fam.DONE
$PY safehead_read.py confirm --table "<same table glob>" --read "$RUN/read/READ_SAFEHEAD.json" --pass-rows "$RUN/read/fam/pass/rows_s*.jsonl" \
  --families "$RUN/read/fam/families" --tactical "$RUN/read/fam/tactical.json" --strategic "$RUN/read/fam/strategic" --out "$RUN/read"   # ZZSH_VERDICT <outcome>
```
- ⛔ **Token budget:** do not wait alive on the families chain. Launch it detached (WMI, CreateFlags `0x08000200`), write a ≤ 2k handoff, and end. A fresh agent runs `confirm` on `fam.DONE`.
- **The verdict is VERDICT_SAFEHEAD.json's `outcome`.** The read's "provisional" line is not quotable until `confirm` has passed CTRL-T2 and all four families.
- **In the same turn as the verdict:**
  - update `GOALS_AND_CLAIMS.md` `H-REFE-SAFEHEAD-1`, starting from `gac_row_SAFEHEAD.md`;
  - on ADOPT, add a registry row and run `safehead_data.export_full`;
  - every non-ADOPT launches its committed next arm (card 5 / 8; Rule Zero).
- **Deletions afterwards** (own temp, logged path and size): `$RUN/gate_*/work/`, which holds the smoke runs and probes, about 0.25 GB (ESTIMATED); `F:/TanitAD_safehead_tip` once the verdict is banked. ⛔ Never delete anything under `F:/safehead_cache/` (the data agent's).

## Traps for this run
- `PYTHONPATH` is not needed (the scripts insert their own dir). ⛔ Never run tests with `python -I`.
- The data agent's `index.csv` already carries a sha256 per file. `seal` re-verifies it and writes the copy that the gate binds.
- The cache's `tr_ep` and `tr_pdms` are approximate (a rebuilt metric cache). The card never fits or selects on them. Navtest scores come only from the C1 table.
- `p5c_tactical.py --subset-ok` and the strategic flags mirror `run_srule_families.sh`. Their full-set behaviour is UNVERIFIED by this agent, which is why the SMOKE pass comes first.
