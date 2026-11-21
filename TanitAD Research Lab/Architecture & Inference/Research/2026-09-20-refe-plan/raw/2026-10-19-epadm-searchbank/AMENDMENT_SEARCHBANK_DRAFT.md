# AMENDMENT_SEARCHBANK_DRAFT (NOT SEALED; do not seal as written): EP-ADMISSION on a search-output bank, for SEARCH-P2-CONFIRM DQ6 (Research Lab agent, 2026-10-19)

**Status: the DQ6 acceptance bars are MISSED in the re-simulation (below), so by the ruling admission is NOT read on C1's bank; fallback (b) is the Master Mind's call.** This draft records what a reading on C1's bank would need, so that the choice is made with the facts in one place. Nothing in it is licensed until you seal it.

## 1. What is amended
EP-ADMISSION (`DRAFT_PREREG_EP_ADMISSION.md` Revision 1, `MM_RULING_EPADM_R1.md`, the frozen gate `ep_admission.py` sha256 `f0193372085022ab7bfc6241f2ca5638fca710cbd4170a19ce0ae95dde6637c4`, schema `ep-admission-v1`) gains a candidate whose picks are SEARCH OUTPUTS, read through an adapter. The gate's rules, loaders, estimator (paired log-cluster bootstrap, B = 10,000, seed 20260927) and verdict functions are unchanged.

## 2. The adapter (pins)
| item | value |
|---|---|
| `epadm_searchbank.py` sha256 | `c4d6d76f89a5a5bcee68552ec9e55d6db6c87665f5e0462f426e889439fade31` |
| `apply_a2_margin.py` sha256 | `b9c2c80d8453fbc6299e97310f43974a677109243dc636f89840e323dab9690b` |
| `epadm_sim_c1.py` / `fixture_synth.py` / `test_searchbank.py` sha256 | `eca00c86...2960` / `4c5fe787...d42d` / `b4477887...56293b4` (full values in `MANIFEST.md`) |
| search config | `stepB_config.json` sha256 `299c6e63dd3016100149f88b6ca498529d79f8a5fbcfc6845753c8121d62c441` |
| checkpoint | P2-s0 `model_final.pt` sha256 `ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51` |
| token list (C1 = B-REST, 905 tokens, 24 logs) | sorted-token sha256 `3a85cc8124b3dd02297fffa176787b629369a75d82ed52c4440b32aa4cc70c4b` |
| layout | P2 bank: one set per token. Candidate banks `s0 s1 s2` (search seeds 20260927 / 28 / 29): one set per (token, seed). Output in host slot k (k-th lowest-aggregate non-pick slot); the shared label array carries the output's own label there; a moved set carries `pred_logit_pick` at the pick slot and `pred_logit_out` at the host slot; an unmoved set is P2's set verbatim. |
| seed rule | the gate's "on ANY seed" quantifier over the 3 search seeds (R1 applied it over 2 EP seeds). **To ratify** (the alternative, one pooled bank of 2,715 sets, would average the seeds and would not guard inference-seed variance, DQ4). |
| proof on carve B (fixture, NOT a read) | 0 mismatches over 400 P2 sets and 3 x 400 candidate sets against the source arrays (float32 cast = the gate loader's); C-KEYS / C-LABELS / ckpt_step accepted; the frozen pick rule reads the intended pick in 1,200 / 1,200 candidate sets; C-ID on P2 = exactly 0. Bank manifests p2 `7106ae14868e46d9...`, s0 `621319ab47121add...`, s1 `6344c5ee5c70de14...`, s2 `dcc4977f24a6ccde...` (carve B; C1's will differ and must be named below). No admission decision was computed on carve B. |
| tests | 48 passed with RED inputs and mutation arms on every guard (`raw/logs/test_searchbank.log`) |

## 3. The simulation result against the pre-committed bars (`SIM_DESIGN.md` sha256 `e0e326da5de800da5814de54298fe1ced036dcd904e07509e062abb39358a8bb`, hashed before the run)
SIMULATED on the real estimator at 24 logs / 905 tokens x 3 seeds, base = carve B re-drawn to C1's per-log counts, 2,000 replicates per bar condition, Wilson 95 %, new replicate seeds:
| bar | measured | holds |
|---|---|---|
| A1-r1 false refusal, Wilson upper <= 5.0 % | 9.9 % [8.7, 11.3] (198/2000) | **NO** |
| A2 (margin 1.05) false refusal, Wilson upper <= 5.0 % | 13.4 % [11.9, 14.9] (267/2000) | **NO** |
| DAC power x1.25 (margin 1.05), point >= 70 % | 7.4 % [6.3, 8.6] (147/2000) | **NO** |
**Missed: all three. Per DQ6: fallback (b), never (c).** The run is also VOID AS WRITTEN under its own control PC1 (A2 ratio means: NC undefined in 9.35 % of replicates, DAC biased to 1.09-1.10; flag R-PC1BIAS). Mechanism (POST-HOC, descriptive): too few NC- and DAC-unsafe picks among ~900 tokens for A2's ratio to be defined (NC) or to resolve a planted x1.25 (DAC); see `RESULT.md` section 2.

## 4. The exact command that would read admission on C1's bank once it exists (only if you override the miss or after a passing revised design)
Placeholders `<...>` are C1's chain outputs after `ZZSPC_DONE`; run on CPU, `CUDA_VISIBLE_DEVICES=-1`, `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`, from the package `code/` directory.
```
# 1. build the four banks (refuses on any guard); prints the four manifest sha256 values
python epadm_searchbank.py build --bundle <B-REST bundle.npz> --search <merged search_P2HEAD.npz> --labels <labels_P2HEAD.npz> --out-dir <C1 bank dir> --expect-token-sha 3a85cc8124b3dd02297fffa176787b629369a75d82ed52c4440b32aa4cc70c4b
# 2. identity of every unit value to the source arrays (no admission statistic)
python epadm_searchbank.py verify --out-dir <C1 bank dir> --bundle <...> --search <...> --labels <...> --out identity_c1.json
# 3. the amendment, sealed with its .sha256 sidecar, must NAME the checkpoint sha256 above and the four manifest sha256 values from step 1 (the gate's check_prereg refuses otherwise)
# 4. the frozen gate, rules r1:
python ep_admission.py run --rules r1 --p2-bank <C1 bank dir>/p2 --cand-bank <C1 bank dir>/s0 --cand-bank <C1 bank dir>/s1 --cand-bank <C1 bank dir>/s2 --cand-id s0 --cand-id s1 --cand-id s2 --ranked-logs <C1 bank dir>/ranked_logs.tsv --no-ranked-pin --prereg <sealed amendment> --candidate-ckpt-sha256 ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51 --out admission_c1.json
# 5. the registered A2 margin 1.05 (the frozen module's constant is 1.00)
python apply_a2_margin.py admission_c1.json --margin 1.05 --out admission_c1_margin105.json
```

## 5. Open items the seal must settle
1. Ratify the seed rule (3 candidate banks, any seed) and the host-slot layout (section 2).
2. `--no-ranked-pin` is required (the registered pin is S0's table; the adapter writes C1's own 24-log table, all logs selected); the seal must authorise it.
3. The A2 margin 1.05 is applied by `apply_a2_margin.py`, not by the module; the seal must say which verdict is the authority (the script's A2 and recombined label), and whether to edit the module constant instead.
4. Unmeasured: the shuffle null (C-NULL / E-NULL) reads non-pick slots; two or three host slots per bank carry a filler row against another output's label (at most 3 of 64). A C-NULL VOID on C1's bank must be diagnosed against this before it is read as a candidate property.
5. NC's observed / predicted is undefined without an NC-unsafe pick; the module fails closed. The seal must state that policy before any read (about 1 bank in 10 at this size, SIMULATED).
6. All stamps: MEASURED (identity), SIMULATED (rates), SELECTION tier, SEEN-SCENE, one training seed, 3 search seeds, label-only (not T1, not navtest), STRATEGIC PAUSED.
