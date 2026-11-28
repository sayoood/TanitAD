**SUPERSEDED 2026-10-19 16:55 by `HANDOFF.md`: Step B is SEALED (`PREREG_L2_STEPB.md`, sha256 `c661588c952d1f8bf65474f9d6346cdb243eb8cf5e882101e040a9cd4f03172b`, Revision 1 = the MM ruling) and its detached chain (`code/stepB_chain.py`) is running. This file is the BUILD-time record (section 2 is still the manual recipe for the O4 arm); its "Needs the Master Mind" line is resolved by the ruling.**

# READY: L2 Step B is built; if the fresh-carve Step A reads O1, Step B is a launch (2026-10-19)

**Headline.** G-REACH-STEPB (a) C-ID, (b) inverted head, (d) logit difference, (e) C8 reach, (f) end to end: **all PASS on a TEST head (the sealed recipe fitted by me, not the L2 head) on the real P2 scorer; all 8 mutation arms read RED** (`raw/gate_testhead/G-REACH-STEPB_PASS_726866f39a7d.json`, about 1 min).
**Projected cost: 37 to 55 CPU-min per arm** (27 search + 10 to 28 label) for 400 tokens x 3 search seeds; three arms 1.8 to 2.7 h sequential (`raw/cost.json`). No GPU, no lock.
**Needs the Master Mind:** (1) one line on E7 (section 4 of `PREREG_L2_STEPB_DRAFT.md`: `launch_gate.py` does not apply to an inference-time search, G-REACH-STEPB does; this narrows E7's literal wording); (2) adopt, amend or strike the draft (ten decisions listed in its section 9).
**Run:** no Step B search, no dPDMS read. The only smoke was 3 tokens, mechanics and timing; its labels were computed and DISCARDED (`--discard-labels`). `pytest test_stepB.py`: **36 passed**.

## 1. What exists (`code/`, all sha256-pinned in `code/MANIFEST.sha256`; L1's four files are imported unchanged and pinned in `stepB.L1_PINS`)
| file | job |
|---|---|
| `l2_scorer.py` | P2 with the EP row of `score_head` replaced by a head file: sha256 REQUIRED and checked, shape / dtype / finite checked, the file's five non-EP rows must equal the checkpoint's bit for bit, install read back. The scoring closure is L1's `real_scorer.make_score_fn`. |
| `stepB.py` | the runner: `search` (shards, resumable), `merge`, `make-config`. C8 by a stated rule (`c8_from_grid`), strict config, stamp check, the in-launch reachability assertion, `run_search_cli` = the whole launch. |
| `g_reach_stepb.py` | G-REACH-STEPB and its PASS stamp (binds head / checkpoint / config / bundle sha256, code sha256, L1 pins). |
| `label_stepB.py` | labels search outputs through `label_perturbations.label_plans` (navsim venv); enforces the PLANTED IDENTITY: composite (0, 0) of the pick must reproduce the pick's banked label (<= 1e-6; MEASURED exactly 0.0 on 3 tokens) and its shifted control must change a label. |
| `analyse_stepB.py` | the draft's endpoint, bar and decision (O1 / TRADE / NOT CLEARED / O4), the four families, the descriptive decomposition and P2HEAD attribution. Built and tested on PLANTED data only. |
| `cost_stepB.py`, `make_test_head.py`, `fake_scorer.py`, `test_stepB.py`, `make_stage.py` | cost projection (timing fields only), the TEST head maker, a tiny stand-in scorer for the tests, the 36 tests, the stager. |

## 2. The launch, in order (CPU only: `CUDA_VISIBLE_DEVICES=-1`; Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, label step `C:/Users/Admin/navsim-crun/venv/Scripts/python.exe`)
Placeholders: `<B>` the fresh-carve bundle, `<CTX>` its ctx.npy, `<H0>`/`<H1>` the final head files, `<S0>`/`<S1>` their sha256 (from `final_heads.json`), `<CK>` = `D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt`, `<CKS>` = `ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51`,
`<RD>` = the refe dir, `<CF>` = `stepB_config.json`, `<CFS>` = `299c6e63dd3016100149f88b6ca498529d79f8a5fbcfc6845753c8121d62c441`.
1. **Pins first.** Write the adopted draft's sha256 beside it. Check the head files' sha256 against `final_heads.json`.
2. **Gate, once per head (about 1 min):** `python g_reach_stepb.py --arm L2 --head <H0> --head-sha256 <S0> --bundle <B> --ctx <CTX> --ckpt <CK> --ckpt-sha256 <CKS> --refe-dir <RD> --config <CF> --config-sha256 <CFS> --out-dir <gate dir>`. Exit 0 and a `G-REACH-STEPB_PASS_<head12>.json`, or STOP.
3. **Search, per arm (about 27 min each):** `python stepB.py search --arm L2 --head <H0> --head-sha256 <S0> --bundle <B> --ctx <CTX> --ckpt <CK> --ckpt-sha256 <CKS> --refe-dir <RD> --config <CF> --config-sha256 <CFS> --gate <PASS json> --out-dir <dir0>`;
   same for `<H1>`; the control: `--arm P2HEAD` (no head flags, any PASS stamp for this checkpoint / config / bundle / code). Detach long runs (WMI, `< NUL`) and leave a HANDOFF; then `python stepB.py merge --out-dir <dir0> --arm L2 --out search_L2s0.npz --n <N>`.
4. **Label, per arm (navsim env as L1's step 5: PYTHONPATH = navsim v1.1 + refe tree, NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION=nuplan-maps-v1.0, REFE_SIM_HZ=10, OMP_NUM_THREADS=1):** `python label_stepB.py --bundle <B> --search search_L2s0.npz --plan <plan json> --out labels_L2s0.npz`. Exit 3 means the planted identity or its control failed: no labels are written, STOP.
5. **Read:** `python analyse_stepB.py --bundle <B> --search L2s0=search_L2s0.npz --search L2s1=... --search P2HEAD=... --labels L2s0=labels_L2s0.npz ... --out RESULT_STEPB.json`. Then EP-ADMISSION (draft section 6) before any navtest look.

## 3. Not built / not done (stated, not hidden)
- **The TRADE-restricted arm** (DAC/NC-safe elite filter): specified in the draft, not built (needs a modified elite selection and its own identity and reach checks). Built only if a TRADE occurs; about 1 h.
- **The EP-ADMISSION bank for a head file:** `build_candidate_bank.py` takes a checkpoint; a head-file input does not exist. `l2_scorer.install_ep_head` is the piece it needs.
- **No real head was gated.** The gate ran on a TEST head; every real head file needs its own PASS stamp (the stamp binds the head's sha256).
- **No real label of any search output has been read or analysed**, and `analyse_stepB.py` has never seen one.
- The labeller was run on 3 tokens of the SEEN Step A bundle for the identity and timing only. `label_stepB.py` has not run on the fresh carve (needs its plan json).
- `launch_gate.py` was not edited and no `stepb` profile exists (E7, draft section 5).

## 4. Traps found while building (each is now pinned by a test or a check)
- A mutation that deletes a line inside a `with` block leaves an empty block: the harness replaces it with `pass` and REFUSES a mutation that edited a different number of lines than expected (a vacuous mutation reads RED or GREEN by accident).
- A launch-level refusal is `SystemExit`, not `Exception`: the gate records it as RED for a mutation arm and as a FAILED check (never a crash) for a real check.
- The in-launch preflight refuses an L2 arm whose EP logits equal P2's (an identity head): it never starts (`test_gate_fails_for_a_head_identical_to_p2`).
- `tts_cem.U_CLIP` / `D_CLIP` are module globals read at call time and not arguments of `cem_search`; C8's clips reach the search only through `stepB.clips()`, which is why check (e) observes the (u, d) the search requests instead of trusting the setting.

## Amendment 1, after ZZSTEPB_DONE (added 2026-10-19; REPORTED ONLY, never gated)

**What.** `PREREG_L2_STEPB_AMENDMENT1.md` (sealed, sha256 `a2e0c2fed1ebc46e9c85b7064070ada5c4fbf6525552010473b12a4126ca48f5`) adds the aggregate concordance (A3 / A4 / A5, L1's `stepA.admission_terms` imported unchanged) of the search outputs beside dPDMS, per arm (L2-s0, L2-s1, P2HEAD), pooled over the 3 search seeds with each seed beside it, plus L2 minus P2HEAD paired differences; paired log-cluster bootstrap (24 logs, B = 10,000, seed 20260927). It changes no gate, bar, arm or verdict. The script is `code/amend1_concordance.py` (NOT in the sealed pin set; the chain never imports it; the seal is untouched).

**Logits: BANKED, no re-score needed.** `stepB.run_search` (the chain's writer) banks the arm's own six-head logits at P2's pick (`pred_logit_pick` [n,6]) and at every search output (`pred_logit_out` [n,S,6], the pick's slot replaced inside the complete 64-set; an unmoved output = the pick's logits). The script uses them directly after: (i) model-free identities (shapes, finiteness, token / pick / label-file alignment; an UNMOVED output's logits equal the pick's bit for bit; the five non-EP columns of the L2 arms' pick logits equal P2HEAD's bit for bit; head / checkpoint / config / carve / bundle shas equal the SEALED pins), and (ii) a MODEL identity on `--identity-tokens N` evenly spaced tokens per arm: the same `score_fn` path (`stepB.build_scorer`, head installed by `l2_scorer.install_ep_head`, sha-checked, CPU, 4 threads like the chain) must reproduce the banked pick logits AND the banked output logits BIT FOR BIT. Any failure = `NOT COMPUTABLE` + the reason (written to `amend1_result.json` / `AMEND1.md`), exit 3. `--logits rescore` (automatic if `pred_logit_out` were absent) re-scores every moved output after the identity held at ALL tokens.

**Exact command** (after `grep -E "^ZZSTEPB_DONE" run/chain_stepB.log` prints; CPU only, no lock, no GPU; about 3 to 6 min, ESTIMATED: three CPU scorer loads plus 16 tokens x up to 4 sets per arm). It reads the run dir and writes ONLY `--out-dir`:
```
cd C:/Users/Admin/par18/l2stepb/code
CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -u amend1_concordance.py --run-dir C:/Users/Admin/par18/l2stepb/run --bundle D:/Projects/TanitAD-artifacts/l2_fresh_stepA/B_token_disjoint/bundle/bundle.npz --out-dir C:/Users/Admin/par18/l2stepb/run/amend1 --identity-tokens 16
```
Defaults (all verified to exist, see `code/stepB_chain.py`): pkg `C:/Users/Admin/par18/l2stepb` (the two sealed texts + their `.sha256`), heads `D:/Projects/TanitAD-artifacts/l2_fresh_stepA/heads`, checkpoint `D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt`, refe `C:/Users/Admin/par18/l1run/tree_hybrid/refe`, ctx = `ctx.npy` beside the bundle, `--expect-logs 24`, chain log `<run-dir>/chain_stepB.log`. Stronger identity: `--identity-tokens 400` (all tokens, about 10 to 15 min ESTIMATED). Without the model identity use `--identity-tokens 0` (structural identities only).
**Exit codes:** 0 computed (prints `ZZAMEND1_DONE`); 3 NOT COMPUTABLE (`ZZAMEND1_NOT_COMPUTABLE <reason>`); 4 no `^ZZSTEPB_DONE` in the chain log (`--allow-no-done` only for a stopped chain); 5 the amendment's (or the main PREREG's) `.sha256` does not match its text, or L1's code is not the landed code: NOTHING is written.
**Outputs:** `run/amend1/amend1_result.json` (strict JSON; an undefined ratio is `null`) and `run/amend1/AMEND1.md` (starts with the section `## AMENDMENT 1: aggregate concordance of the search outputs (reported only)`). Paste that section next to dPDMS and the four families in the Step B result; then `python code/make_stage.py` stages `run/amend1/*` (it globs `run/**` once `RESULT_STEPB.json` exists).
**Tested (planted data only, no real Step B output read):** `code/test_amend1.py` 38 passed (`logs/pytest_test_amend1.log`); 12 mutations of the script each turn it RED (`test_amend1/run_mutations.py`, `logs/mutations_amend1.log`). Run: `CUDA_VISIBLE_DEVICES=-1 python -m pytest test_amend1.py -q -p no:cacheprovider --basetemp=<scratch>` from `code/`.
