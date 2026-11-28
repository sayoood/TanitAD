# READ_TOOL_SPEC: the changes `scorerft_read.py`, `test_scorerft_read.py` and `RUNBOOK_SCORERFT_READ.md` need for a P-vs-C read

Written 2026-11-24 for the MM by the SCORER-FT O3 pre-registration agent. **This is a specification; nothing is implemented.**

* **Card it serves:** `o3/card/PREREG_SCORERFT_PICK.md` (draft, `H-SCORERFT-PICK-1`).
* **Files it changes**, under `PK/eval/scorerft/`. Line numbers and sha256 are at mirror tip `5eb74f30`:

  | file | sha256 | size |
  |---|---|---|
  | `scorerft_read.py` | `8a24214c…` | 911 lines |
  | `test_scorerft_read.py` | `fe32e664…` | 573 lines |
  | `RUNBOOK_SCORERFT_READ.md` | `3cd12bcf…` | |

* `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

## 0 · Design rules (each one has a test in §2)
1. **The parent read must not move.** The default card is `scorer_ft`. Run on the Stage 1 inputs, the modified tool must reproduce
   `read_stage1.json`'s `verdict`, `T_minus_C`, `levels`, `goal`, `mechanism`, `families` and `controls[*].pass` **exactly**. Every existing test passes
   unchanged.
2. **One file and one draw matrix.** `p5c_read.py` and `field_read.py` stay IMPORTED, never forked or edited (the tool's own rule, `scorerft_read.py:6`).
   Anything they do not offer is added in `scorerft_read.py`.
3. **Roles stay internal; labels are what the reader sees.**
   * Internally the code keeps the roles `T` (treatment), `C`, `Ts`, `Cs`, so `run_read` changes little.
   * Every **output key and printed string** uses the card's label. Under card `pick`, the role `T` is printed `P` and `Ts` is printed `Ps`.
   * ⛔ A P read must never print `T_minus_C`. That would be the arm-label inversion class (the `flagship4b` row in CLAUDE.md).
4. **A control the card names is never skipped or waived.** A real read refuses (exit 2) and names every missing input. Per MM 2026-11-24, CTRL-S0
   runs in its FULL form only: C's run log is banked, so there is no fallback form.
5. **Pins are filled by the MM at freeze, land or launch.** Until then they are `None`, and a real `pick` read REFUSES. A dry run is allowed and
   stamped.

## 1 · `scorerft_read.py`: exact changes

### 1.1 New constants (after `ALLOWED_SEED`, line 76)
```python
# --- card PICK (PREREG_SCORERFT_PICK.md, H-SCORERFT-PICK-1). Pins: None = REFUSE a real pick read.
PICK_PREREG_SHA256 = None            # MM: sha256 of the FROZEN card file
PICK_TRAIN_SHA256 = None             # MM at land: sha256 of `git show <P land>:PK/refe/train.py` (LF). Expected fd5e5dadf8c8fbd08658ddf0ef055b7f746d3f4b5a91a8db74f009aa3b8e7ed4 if landed as the o3 copy
PICK_CODE_TAG = None                 # MM at launch: "code_<8 hex>" of P's pod code dir (P's argv[0])
C_METRICS_SHA256 = "97a8631cc863d95cf646932ef3833206affbc7a0857d21bff5d5e8b78ff04eed"      # o3/card/evidence/metrics_scorerft_c.jsonl (3,046 lines; MM-banked, framed, = the pod's)
T_METRICS_SHA256 = "09cdaf83c611fba9af7bafef250affc7195af71aa9c40bfcf461888d712b0f6a"      # .../metrics_scorerft_t.jsonl (3,047 lines): the Stage 1 positive control of CTRL-S0 / CTRL-ORDER
S0_KEYS_TRAIN_STAGE1 = ("assign_d", "batch", "bce_c", "bce_dac", "bce_ddc", "bce_ep", "bce_nc", "bce_ttc", "epoch", "lr", "pos", "pos_L1", "samples",
                        "score", "scorer_cov", "step", "traj_L1", "winners", "yaw_L1")      # MEASURED: the 19 keys C's step-0 training row carries minus S0_EXCLUDE
S0_KEYS_ONPOLICY_STAGE1 = ("best", "cov_norm", "drift_m", "event", "lag_steps", "pick", "random", "sets", "skill", "step")   # MEASURED: the 10 keys of C's step-0 onpolicy row
C_TRAIN_SHA256 = "59a737f43947175001202d0114e3d4c83f8a30c984b171c3d284ca1b4684a755"        # C's (and T's) pod source_sha256["train.py"] = git 83411c28
C_CODE_TAG = "code_83411c28"
C_POD_CONFIG_SHA256 = "a66f6750416207cbfefa5a58a51088f5e44068b4d212bc5f5d38b9a8b78c5491"
C_ARGV_SHA256 = "00729d56aa734e370d394385b516d9b8abae94bfe1e3f1df304e323772069e03"          # raw/2026-11-24-scorerft-read/argv_c.json
C_W0_SHA256 = "2dd2dcd25cb582715d1f8846bc352f4c2e767afc061acd36b4a80415c54351ec"            # .../w0_c.json
C_SEAM_SHA256 = "914bdd0aab4459a4ebb61d9fb0d82c537c3b1ec86ebc427882066ddee279b94f"
C_DUMP_SHA256 = "848e6561987b2971bd63f4cdcf9f17c3e4ec76d3aa19da405f7d7d5788492e56"
C_CSV_SHA256 = "2df7f856b46b9dc7f3540c056b15ffd2fc42a7ba39c94e0aad37d4f907f3c81f"
STAGE1_READ_SHA256 = "d301d38c6c358a4220a0cdb4cd7641f86c9c4a01d123b546050305e7fda32d41"
STAGE1_LEDGER_SHA256 = "e627df2af876624588c93e9759210d7c52063e155d4231d87412a7dcf6e5f9b2"
C1_SHA256 = "f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4"
POD_DEVICE, POD_TORCH = "NVIDIA L40", "2.7.1+cu128"
HARNESS_TIP = "751f3ae7"
HARNESS_KEYS = {"select": "best", "rule": "navsim_v1", "repair_last_heading": True, "sanitize_goal": False, "goal_fix": "pdm_route", "nav_mode": "command"}
CREUSE_BAR = 1e-9
ALLOWED_TC_PICK = {"--rank-w", "--rank-heads", "--out"}                                    # card section 3: argv(P) - argv(C)
S0_EXCLUDE = {"elapsed_s", "cuda_max_mem_gb", "at"}                                        # wall-clock / allocator, not model state
S0_RANK_ONLY = re.compile(r"^(rank_L|rank_w|rank_pairs_\w+|rank_sets_\w+|pick_v1|random_v1|best_v1|skill_v1)$")
ORDER_KEYS = ("epoch", "pos", "samples", "batch", "lr")
NEXT_ARM_PICK = {...}               # verbatim from the card's section 9: PL1, PL2, OP1 ... OP7 (OP6 carries both branches, Zup / not Zup)
OUTCOME_VERDICT_PICK = {...}        # verbatim verdict column of the card's section 9
```
`NEXT_ARM_PICK` and `OUTCOME_VERDICT_PICK` carry the card's §9 text **verbatim**. A test asserts that every OP key exists and that each string
contains the card's key phrase (§2, `test_pick_every_outcome_carries_its_committed_next_arm`).

### 1.2 The card registry (new, after the constants)
```python
CARDS = {
  "scorer_ft": dict(prereg=PREREG_SHA256, card="PREREG_SCORER_FT.md (FROZEN 2026-11-20)", key="H-SCORERFT-1",
                    labels={"T": "T", "C": "C", "Ts": "Ts", "Cs": "Cs"}, look=LOOK, look2=LOOK2, allowed_tc=ALLOWED_TC,
                    next_arm=NEXT_ARM, verdict=OUTCOME_VERDICT, fam_labels=("C", "T")),
  "pick":      dict(prereg=PICK_PREREG_SHA256, card="PREREG_SCORERFT_PICK.md", key="H-SCORERFT-PICK-1",
                    labels={"T": "P", "C": "C", "Ts": "Ps", "Cs": "Cs"},
                    look="look 1 of H-SCORERFT-PICK-1 (Stage 1: P, C); C's rows were read at look 1 of H-SCORERFT-1 (shared control)",
                    look2="look 2 of H-SCORERFT-PICK-1 (Stage 2: Ps, Cs)", allowed_tc=ALLOWED_TC_PICK,
                    next_arm=NEXT_ARM_PICK, verdict=OUTCOME_VERDICT_PICK, fam_labels=("C", "P")),
}
```
* **The `--card` flag** takes `{scorer_ft, pick}`, default `scorer_ft`.
* **A real `pick` read refuses** (`Refused`) when any of these holds:
  * `PICK_PREREG_SHA256` or `PICK_TRAIN_SHA256` is `None`;
  * the arm names passed as `--t` / `--c` are not `P` / `C`. This is the pick card's RED-SWAP guard; the parent keeps its current behaviour.
* `res["card"]`, `res["card_sha256"]` and `res["card_key"]` come from `CARDS[a.card]`.

### 1.3 CLI (`main`, lines 860-887)
* **Aliases, same `dest`** (argparse accepts several option strings per argument):
  * `--t/--p`, `--ts/--ps`;
  * `--seam-t/--seam-p`, `--dump-t/--dump-p`, `--csv-t/--csv-p`, `--argv-t/--argv-p`, `--proptable-t/--proptable-p`;
  * the same set for `-ts/-ps`;
  * `--det-seam-t/--det-seam-p` and `--det-full-seam-t/--det-full-seam-p`.
* **New flags** (all `default=None`; read only under `--card pick`, and refused under the parent card with *"flag X belongs to card pick"*):

| flag | content | control |
|---|---|---|
| `--card` | `scorer_ft` \| `pick` | the registry |
| `--w0-digest-p` | file holding P's step-0 digest (P's `config.json` `warm_start.sha256`, framed into `w0_p.json` like `w0_c.json`) | CTRL-W0 (`others={"P": …}`, already supported by `ctrl_w0`) |
| `--metrics-c`, `--metrics-p` | the two runs' `metrics.jsonl`. C's is `o3/card/evidence/metrics_scorerft_c.jsonl`, checked against `C_METRICS_SHA256` | CTRL-S0 (full form, the only form), CTRL-ORDER |
| `--pod-config-c`, `--pod-config-p` | the two runs' `config.json` | CTRL-CODE |
| `--seam-report-c`, `--seam-report-p` | `stage2_scorerft_{c,p}.report.json` | CTRL-HARNESS |
| `--seam-chain-log` | `scorerft_read/CHAIN.log` (the `<ARM> chain start tip=<8 hex>` lines) | CTRL-HARNESS |
| `--weight-rule-p` | the banked `weight_rule_rank.py` JSON on P's argv | CTRL-WRULE |
| `--stage1-read`, `--stage1-ledger` | Stage 1's `read_stage1.json` and `seam_ledger.json` | CTRL-CREUSE |
| (Stage 2) `--pod-config-cs`, `--pod-config-ps`, `--metrics-cs`, `--metrics-ps` | Cs / Ps run files | CTRL-CODE and CTRL-ORDER for Xs vs X (allowed diff `{seed, out}`) |

* **`run_read`'s real-read `need` list** (lines 629-642) gains, under `--card pick`:
  * `--w0-digest-p`, `--metrics-c`, `--metrics-p`, `--pod-config-c`, `--pod-config-p`, `--seam-report-c`, `--seam-report-p`, `--seam-chain-log`,
    `--weight-rule-p`, `--stage1-read`, `--stage1-ledger`.

### 1.4 CTRL-ARGV, pick form (`ctrl_argv`, lines 160-212)
* **Signature:** `ctrl_argv(argv, steps=STEPS_EXPECTED, card="scorer_ft")`. The `scorer_ft` path stays byte-for-byte what it is.
* **Under `pick`**, with roles `T`→P and `Ts`→Ps, every check reports the label, not the role:
  * `_diff_keys(C, P) - ALLOWED_TC_PICK` empty, else FAIL *"P vs C differ in flags outside the card"*.
  * C names `--rank-w 0.0`, and carries none of `--rank-heads`, `--rank-ep-gap`, `--rank-gate-topk`.
  * P: `--rank-w` is a single finite float > 0, and `P["--rank-heads"] in (("pick",),)` exactly. `pick,ep` FAILS, and so does `ep,gate`.
  * P carries neither `--rank-ep-gap` nor `--rank-gate-topk`.
  * Every arm has one `--init-from` and `--steps 1000`; each `--out` is its own and distinct.
  * Xs vs X ⊆ `ALLOWED_SEED`, and `--seed 1`.
  * Ps vs Cs ⊆ `ALLOWED_TC_PICK`. Cs and Ps are supplied together.
* **New helper `argv_script(x)`.** It returns the first token when that token does not start with `--`. `parse_argv` drops it (`cur is None`).
  * `detail["code_tag"] = {label: re.search(r"code_[0-9a-f]{8}", script)}`, recorded for CTRL-CODE.
  * At `5eb74f30` the code tag is invisible to every control: C ran `code_83411c28` and T `code_3609aa2b`, and nothing read that.

### 1.5 New controls (each returns `{"control", "pass", "fails", …}`, as the existing ones do)
**`ctrl_s0(metrics_c, metrics_p, rank_w=None, step=0, check_c_sha=True)` (CTRL-S0, FULL form only).**
* **Inputs.** `sha256(metrics_c) == C_METRICS_SHA256` is required when `check_c_sha` is set (a real read). `_rows(path)` streams the JSONL.
  * The **training row** of a step is the last row with `"step" == s` and no `"event"` key. A resumed run re-logs steps, so the last row is the one that
    counts.
  * `_rows` also returns the `onpolicy` event rows by step, and the count of `resume` events.
* **The rule.**
  * `kc` = the keys of C's training row at `step`, minus `S0_EXCLUDE`. Every key in `kc` must be in P's row and **equal exactly** (`==` on the parsed
    JSON values; both-NaN counts as equal).
  * A key present only in P must match `S0_RANK_ONLY`. Any other P-only or C-only key FAILS.
  * P's `rank_L` is present and finite. P's `rank_w` equals `rank_w` (from argv) exactly.
  * The `onpolicy` rows at `step`: their shared keys minus `S0_RANK_ONLY` are equal, and the row is present in both arms or in neither.
* **In a real read**, the compared key sets must equal `S0_KEYS_TRAIN_STAGE1` and `S0_KEYS_ONPOLICY_STAGE1`. Those are the MEASURED sets on C's log; a
  key set that drifts means P logs differently, and that FAILS.
* **The record** carries `traj_L1_c`, `traj_L1_p`, `keys_compared` (both rows, as lists), `n_keys_compared`, the keys that differ, the P-only keys,
  and the excluded keys present.
* `step` exists **for the tests only**: a real read uses step 0. Step 1 is the non-vacuity arm of §2.

**`ctrl_order(metrics_c, metrics_p, steps=STEPS_EXPECTED)` (CTRL-ORDER, always run).**
* Both arms have a training row for every step 0…`steps−1`, and nothing beyond.
* `ORDER_KEYS` are equal per step, exactly.
* The record carries the first differing step and the resume counts.

**`ctrl_code(cfg_c, cfg_p, label_p="P", allowed_args={"rank_w", "rank_heads", "out"}, script_tags=None)` (CTRL-CODE).** It also serves Xs vs X, with
`allowed_args={"seed", "out"}`. Checks:
* `sha256(cfg_c file) == C_POD_CONFIG_SHA256` (the C read only).
* `cfg_p["device"] == cfg_c["device"] == POD_DEVICE` and `cfg_p["torch"] == cfg_c["torch"] == POD_TORCH`.
* `source_sha256`:
  * the same key set;
  * `model.py`, `load_dinov3.py` and `ckpt_io.py` equal;
  * `train.py`: C == `C_TRAIN_SHA256` and P == `PICK_TRAIN_SHA256`. ⚠️ A P whose `train.py` equals C's ran the **old** code: FAIL.
* `warm_start.sha256` == `C1_SHA256` for both.
* Top-level `args` differ only in `allowed_args`. This is the parsed namespace, defaults included: `rank_ep_gap` and `rank_gate_topk` must equal C's.
* `identity["args"]` differ only in `allowed_args − {"out"}`. Every other `identity` field must be equal: `targets_sha256`, `scorer_sha256`,
  `calib_sha256`, `frozen_sha256`, `traj_label_sha256`, `n_tuples`, `n_scenes`, `per_rank`, and the remaining ones.
* `meta.frozen_sha256`, `meta.traj_label_table_sha256` and `meta.traj_label_coverage` are equal.
* `args.rank_heads == "pick"` and `args.rank_w > 0` for P; `args.rank_w == 0.0` and `args.rank_heads == ""` for C.
* `script_tags`: C's == `C_CODE_TAG`, and P's == `PICK_CODE_TAG` when that pin is set.

**`ctrl_harness(rep_c, rep_p, chain_log)` (CTRL-HARNESS).**
* Each report holds `HARNESS_KEYS` exactly, with `rows == tokens_asked == 12146`, `misses == 0` and `driving_command_set == 12146`.
* In the chain log, the last `SFTC chain start tip=` line and the last `SFTP chain start tip=` line both read `HARNESS_TIP`.
  * Why: C's seam came from the harness export `751f3ae7` (`CHAIN.log` 2026-11-23T13:20:19Z).
  * A P seam built on a later harness would confound P − C with a harness change.

**`ctrl_wrule(j, argv_p)` (CTRL-WRULE).**
* `j["rule"]["status"] == "SELECT"`, `j["heads"] == ["pick"]` and `j["dead_heads"] == []`.
* `j["label_backend"] == "pdm"` and `j["NOT_A_DECISION"] is False`.
* `j["init_from"] == argv_p["--init-from"][0]`.
* `float(argv_p["--rank-w"][0]) == j["rule"]["rank_w"]` **exactly**.
* The record also carries `r_by_w`, `rho_by_w`, `pairs_by_head` and `r1_per_micro_min_median_max`.

**`ctrl_creuse(stage1_read, stage1_ledger, geo_c, csv_c_sha, argv_c_path, w0_c_path, levels_c, sel_levels_c, n_boot)` (CTRL-CREUSE).**
* `sha256(stage1_read) == STAGE1_READ_SHA256` and `sha256(stage1_ledger) == STAGE1_LEDGER_SHA256`.
* The seam and dump sha of `geo_c` equal the ledger's `arms.C`, and equal `C_SEAM_SHA256` / `C_DUMP_SHA256`.
* `csv_c_sha == C_CSV_SHA256`. Take it from this read's CTRL-T record for C.
* `sha256(argv_c_path) == C_ARGV_SHA256` and `sha256(w0_c_path) == C_W0_SHA256`.
* **Levels.** For every metric in `METRICS`, scorer in `{tracked, transparent, TD}` and field in `{point, lo, hi}`:
  `|levels_c − stage1["levels"]["C"]| ≤ CREUSE_BAR`.
* **Selection levels.** For every key of `stage1["families"]["TACTICAL"]["(b)_selection_quality"]["levels"]["C"]` with a `point`, the same
  `≤ CREUSE_BAR` test against `sel_levels_c`.
* **Draw count.** In a real read the control requires `n_boot == 10000 == stage1["estimator"]["draws"]`. The draws are a function of
  (sorted logs, n_boot, seed) only (`p5c_read.Boot`, lines 87-95), so only then can the CIs reproduce.
  * A dry run with another `n_boot` puts CTRL-CREUSE in `controls_not_available`.
* The record carries the max |Δ| per block, the number of fields compared, and the first field that is over the bar.

### 1.6 Mechanism and outcome (pick card only; the parent's `mech_decide` and `decide` are unchanged)
```python
def mech_decide_pick(con):
    m3 = con["regret"]; m4 = con["oracle"]; z = con["zeroed"]; m1 = con["ep_spearman_both_defined"]
    m3_moved, m4_damaged, z_up = bool(m3["hi"] < 0), bool(m4["hi"] < 0), bool(z["lo"] > 0)
    return {"M3_regret": {...oracle_minus_pick, gate, ep, ttc, c, other, top1, "moved": m3_moved, "rule": "CI-hi < 0 of P - C regret"},
            "M2_boundary": {...as today, "Zup": z_up, "rule": "CI-lo > 0 of P - C zeroed-pick rate (selects OP6's branch only)"},
            "M4_generation_guard": {..., "damaged": m4_damaged},
            "M1_ep_spearman": {"delta": m1, "moved_flag": <the parent's rule>, "decisive": False, "rule": "reported, NOT decisive (card section 7)"},
            "mechanism_moved": m3_moved, "m4_damaged": m4_damaged, "m3_moved": m3_moved, "Zup": z_up}

def decide_pick(con0, mech, controls_ok, rep=None):
    # clauses P1 / G1 / P2 / P3 built exactly as decide() lines 475-487
    # INVALID if controls_ok is not True
    # p1 and g1: rep None -> "STAGE-2-TRIGGERED" (next: "run Cs then Ps (conditional Stage 2, ~28.4 L40-h, PI go)"); else OP1 if P2 and P3 else OP2
    # elif p1: OP3 (wording adds the adverse list; next_arm = NEXT_ARM_PICK["OP3"] with "<h,...>" filled by the G1 adverse list, C included)
    # elif mech is None: INCOMPLETE
    # elif mech["m4_damaged"]: OP4
    # elif mech["m3_moved"]: OP5
    # elif d0["hi"] < 0: OP6, next = NEXT_ARM_PICK["OP6_Zup"] if mech["Zup"] else NEXT_ARM_PICK["OP6_noZup"] (= OP7's next)
    # else: OP7
    # every OP3..OP7 on one seed appends ONE_SEED + " (a Stage-1 outcome is accepted on one seed, decided before data)"
    # OP1 appends the GUARD-FAIL caveat, as O1 does
```
* **`run_read` dispatch.** `mech_fn, decide_fn = (mech_decide, decide)` for `scorer_ft` and `(mech_decide_pick, decide_pick)` for `pick`.
* **`ctrl_id`.** It takes `card` and calls the card's pair.
  * Pass rule for `pick`: no non-zero Δ, outcome in `("OP7", None)`, and no P1.
  * A self-read gives Δ₀ CI-hi = 0.0, which is not `< 0`, so it cannot reach OP6. The read itself asserts this.

### 1.7 Labels in the output (pick card; the parent keeps today's keys)
* `lab = CARDS[a.card]["labels"]`. In the result dict:
  * `arms` is keyed by label;
  * `levels`, `goal` and `selection_quality.levels` are keyed by label;
  * `f"{lab['T']}_minus_C"` replaces `T_minus_C`;
  * `mechanism_Ps_minus_Cs` and `replicate.Ps_minus_Cs` are labelled the same way;
  * `families.Ps_minus_Cs_fam_ci` likewise.
* **`relabel_tactical(t, m)` (lines 542-555).**
  * `m` defaults to `{"C1": lab["T"], "C0": "C"}`.
  * The hard-coded `"T-"` and `"paired_T_minus_x"` replacements become `f"{lab['T']}-"` and `f"paired_{lab['T']}_minus_x"`.
  * The `labelling` note names the map it applied.
* **Defaults.** `fam_cells` defaults to `CARDS[card]["fam_labels"]`. `strategic_block` is unchanged; it reads whatever arm names the JSON carries.
* **`verdict_line` and `text_report`.** Every `"T"` / `"T - C"` becomes the label.
  * Under `pick`, the mechanism part reads `M3_moved=… M4_damaged=… Zup=… mechanism_moved=… M1(reported)=…`.
  * The ledger part reads `CARDS[card]["look"]`.
* **`card_questions`.** The pick card prints its own list: Q-P1 … from `CARD_QUESTIONS.md` once they are settled, and the S0 form in use. The
  parent's `Q-S*` list stays with the parent.
* **The module docstring** gains one paragraph: *"CARD pick (PREREG_SCORERFT_PICK.md): arm P (head `pick`), C REUSED from Stage 1, controls +S0
  +ORDER +CODE +HARNESS +WRULE +CREUSE, outcomes PL1/PL2, OP1-OP7."*

### 1.8 What does NOT change
* `wdraws`, `wci` and `wdiff`.
* `mech_arrays`: the per-token instruments are the same for any arm.
* `load_table_arrays`, `load_dump`, `ctrl_t`, `ctrl_det` and `ctrl_w0`.
* `pair_rep`: P3's F is still `|mean(Cs) − mean(C)|`.
* `goal_flag` and `g1_adverse`.
* The polarity rule: the ONE shared `field_read.entry_adverse_pol`.
* The estimator constants.

## 2 · `test_scorerft_read.py`: exact additions (run the WHOLE file; module fixtures; never `-k`)
**The docstring's RED list gains:**
* **RED-SWAP-P:** `--t C --c P` under `pick` is REFUSED.
* **RED-ARGV-P:** one extra flag, `--rank-ep-gap` on P, or heads `pick,ep` ⇒ CTRL-ARGV FAILS.
* **RED-S0:** one step-0 metric one ulp off ⇒ FAIL.
* **RED-ORDER:** two steps' `pos` swapped ⇒ FAIL.
* **RED-CODE:** `model.py` sha, `device` or `args.lr` differs, or P's `train.py` equals C's ⇒ FAIL.
* **RED-HARNESS:** a different tip or `rule` ⇒ FAIL.
* **RED-WRULE:** status K2, heads `ep,gate`, or `rank_w` off by one candidate ⇒ FAIL.
* **RED-CREUSE:** one C row perturbed by 1e-6 or a ledger sha changed ⇒ FAIL.

**New fixtures.**
* `synth_pick`, module scope. It reuses `write_arm` to write arms `P` and `C`, plus rows with arms `P` / `C`.
* `mk_metrics(n_steps, seed, rank)` writes a `metrics.jsonl`. It has training rows with every key C logs (the list in `train.py:2054-2077`), `onpolicy`
  rows, and P-only `rank_*` and `*_v1` keys when `rank` is true.
* `mk_config(arm, **over)` writes a `config.json` with the keys `pod_config_c.json` carries: `argv`, `args`, `identity`, `meta`, `warm_start`,
  `source_sha256`, `torch`, `device`, `started_at`.
* `mk_wrule(**over)`.
* `mk_stage1(synth_pick)` writes a stage-1 read by running `run_read` under `--card scorer_ft` on arms `T`=P-files and `C` (dry, `n_boot` 200). The
  `C` levels it holds are the ones a correct CREUSE must reproduce.

| test | asserts |
|---|---|
| `test_parent_card_is_the_default_and_unchanged` | `ns(synth)` without `card` runs as today. `CARDS["scorer_ft"]` uses the parent's constants. Output keys include `T_minus_C`, with no `P_minus_C`. |
| `test_pick_outcome_order_OP1_to_OP7` | With `fake_con` / `fake_mech_pick(m3, m4, zup)`, each of the seven conditions maps to its OP. The order rules: OP4 wins over OP5 (M4 and M3 both true); OP5 wins over OP6 (M3 true and Δ₀ CI-hi < 0); M1 alone ⇒ OP7, so M1 is not decisive; OP6 with Zup gives the `pick8` (P-top8) arm, without Zup OP7's next. |
| `test_pick_stage1_p1_g1_triggers_stage2_never_adopt` | `STAGE-2-TRIGGERED`, with the text "Cs then Ps". |
| `test_pick_every_outcome_carries_its_committed_next_arm` | The keys `{PL1, PL2, OP1, OP2, OP3, OP4, OP5, OP6_Zup, OP6_noZup, OP7}`, each non-empty. Key phrases: `P-log` (PL1), `--rank-detach-ctx` (PL2, OP4), `--rank-pick-guard` (OP3), `successor card` (OP5), `pick8` (OP6_Zup), `TEACHER-N7` (OP7). No OP string contains `--rank-pick-gate-detach`: it is subsumed by decision 10. |
| `test_pick_op3_names_the_adverse_guard_set` | Adverse `("ttc", "c")` ⇒ the next arm reads `--rank-pick-guard ttc,c`. |
| `test_pick_ctrl_argv_passes_on_exactly_the_cards_difference` | `ARGV_P = BASE + ["--rank-w", "0.25", "--rank-heads", "pick", "--out", "/w/p"]` vs `ARGV_C` passes, and `T_minus_C_flags` is labelled `P_minus_C_flags == {"--rank-w", "--rank-heads", "--out"}`. |
| `test_RED_ARGV_P_variants_fail` | Each of these FAILS: an extra flag in P, C, Ps or Cs; P with `--rank-ep-gap 0.02`; P with `--rank-gate-topk 8`; heads `pick,ep`; heads `ep,gate`; `--rank-w 0`; C with `--rank-heads pick`; `--steps 2000`; a shared `--out`; Ps `--seed 2`. |
| `test_cross_card_argv_is_refused` | `ARGV_T` under `pick` FAILS, and `ARGV_P` under `scorer_ft` FAILS. |
| `test_argv_script_and_code_tag` | `/w/data/refe_p5c/code_83411c28/refe/train.py --steps 1000` ⇒ tag `code_83411c28`. A bare flag list ⇒ `None`. |
| `test_ctrl_s0_full_and_RED` | Equal rows ⇒ PASS, and `elapsed_s` / `cuda_max_mem_gb` differences are ignored. RED arms, each ⇒ FAIL: `traj_L1` moved by `np.nextafter`; a C key missing in P; a P-only key outside `S0_RANK_ONLY`; P without `rank_L`; P `rank_L` NaN; `rank_w` ≠ argv; an `onpolicy` row in one arm only. A resumed run whose step-0 row is re-logged reads the last row. |
| `test_ctrl_s0_on_the_stage1_rows` | **Real-data positive and non-vacuity arms.** The fixture `stage1_s0_rows.json` holds C's and T's step-0 and step-1 training and `onpolicy` rows, copied verbatim from `o3/card/evidence/metrics_scorerft_{c,t}.jsonl` with their sha256s recorded in the fixture. Step 0: CTRL-S0 PASSES, with exactly `S0_KEYS_TRAIN_STAGE1` (19) and `S0_KEYS_ONPOLICY_STAGE1` (10) compared, the T-only keys all inside `S0_RANK_ONLY`, and `traj_L1` 0.2121780663728714 on both sides. Step 1: it FAILS on exactly `{bce_c, bce_dac, bce_ddc, bce_ep, bce_nc, bce_ttc, pos_L1, score, traj_L1, yaw_L1}` and `onpolicy {best, pick, random, skill}`, as MEASURED. `check_c_sha=False` (the fixture is an excerpt). |
| `test_ctrl_order_and_RED` | Equal series ⇒ PASS. Two steps' `pos` swapped ⇒ FAIL, naming the first differing step. A missing step ⇒ FAIL. A duplicate step from a resume is de-duplicated. A real read without `--metrics-c/-p` refuses: there is no waiver. |
| `test_ctrl_code_and_RED` | Equal configs (except `rank_w`, `rank_heads`, `out`; P `train.py` = pin) ⇒ PASS. RED arms, each ⇒ FAIL: `model.py` sha; `device` `NVIDIA A40`; `torch` `2.8.0`; `args.lr 1e-4`; `args.rank_ep_gap 0.05`; `identity.targets_sha256`; P `train.py` == `C_TRAIN_SHA256`; P `warm_start` ≠ C1. The Xs vs X form allows `{seed, out}` only. |
| `test_ctrl_harness_and_RED` | Correct reports and log ⇒ PASS. `rule navsim_v2`, `misses 1`, or an `SFTP chain start tip=0c5edd9e` line ⇒ FAIL. |
| `test_ctrl_wrule_and_RED` | SELECT with `rank_w` == argv ⇒ PASS. Each of these ⇒ FAIL: `K2`; `K2b`; heads `["ep", "gate"]`; `dead_heads ["pick"]`; backend `synthetic`; `NOT_A_DECISION` true; argv `--rank-w` = 2× the rule's value. |
| `test_ctrl_creuse_and_RED` | A pick dry read against `mk_stage1` (same rows, same `n_boot`) ⇒ PASS, and the max |Δ| is reported. RED arms, each ⇒ FAIL: one C tracked PDMS row + 1e-6; a ledger whose `arms.C.seam_sha256` is changed; a stage-1 file edited by one byte (sha pin); a C CSV sha ≠ pin. In a dry run, `n_boot` ≠ the stage-1 draw count ⇒ NOT AVAILABLE. |
| `test_pick_labels_in_every_output` | Under `pick`, the JSON has `P_minus_C`, `levels.P`, `goal.P` and `selection_quality.levels.P`, and **no** key or string `T_minus_C` / `"T"` arm (search the JSON dump). The text report and verdict line say `P - C`, `H-SCORERFT-PICK-1`, and "shared control". |
| `test_relabel_tactical_to_P` | `C1` → `P`, `paired_C1_minus_x` → `paired_P_minus_x`, the strategic stub dropped, `pooled` false. |
| `test_pick_real_read_refuses_without_its_inputs` | Each new flag is named in the refusal: `--w0-digest-p`, `--pod-config-c/-p`, `--seam-report-c/-p`, `--seam-chain-log`, `--weight-rule-p`, `--stage1-read`, `--stage1-ledger`, `--metrics-c/-p`. |
| `test_pick_unfrozen_pins_refuse_a_real_read` | `PICK_PREREG_SHA256` or `PICK_TRAIN_SHA256` set to `None` (monkeypatch) ⇒ `Refused`. A `--metrics-c` whose sha ≠ `C_METRICS_SHA256` ⇒ CTRL-S0 FAIL. Dry runs are stamped and allowed. |
| `test_RED_SWAP_P_names_refused` | `--card pick --t C --c P` ⇒ `Refused`. |
| `test_pick_ctrl_id_and_self_read` | Under `pick`, P against itself gives every Δ exactly 0.0, outcome OP7, and no P1. |
| `test_end_to_end_pick_dry_read` | A dry read on `synth_pick` with every new input synthesised: every control PASSES; the `DRY RUN` stamp is present; STRATEGIC is never PAUSED; TACTICAL (c) is NOT COMPUTABLE with n; `mechanism.M1_ep_spearman.decisive is False`. |
| `test_pick_constants_are_the_cards` | Every C pin equals the value in this spec. `ALLOWED_TC_PICK == {"--rank-w", "--rank-heads", "--out"}`. Once frozen, `PICK_PREREG_SHA256` equals the card file's sha (the MM edits this test at freeze, as `test_prereg_constants_are_the_cards` does for the parent). |

**Expected count:** 40 existing + 24 new = **64**. ⛔ *"No tests ran"* is a FAIL, as is a pass count that does not match.

## 3 · `RUNBOOK_SCORERFT_READ.md`: exact additions
1. **Header.** Rename the existing body to **"Part A: T-vs-C (card PREREG_SCORER_FT.md, Stage 1 DONE 2026-11-24, O3)"**. Add one line under the title:
   *"Part B below reads arm P against the reused C (card PREREG_SCORERFT_PICK.md). `--card` defaults to `scorer_ft`, so Part A's commands are unchanged."*
2. **New Part B,** "P-vs-C read (card `PREREG_SCORERFT_PICK.md`, sha256 `<pin>`)".
   * **Directories.** `OP=E:/Projects/TanitAD-artifacts/scorerft_read/read_p`, a new directory off the Stage-1 dirs. `O1=E:/Projects/TanitAD-artifacts/scorerft_read/read_out`,
     Stage 1's directory, which is **read-only** for Part B.
   * **B0 inputs.** One table: what each input is, where it comes from, and its sha pin.
     * **P:**
       * seam, dump and harness CSV (`sft_seam_chain.sh SFTP scorerft/p scorerft_p`);
       * table `propscore/full_SFTP_s*of2.jsonl`;
       * the CTRL-DET full re-run seam `stage2_scorerft_p_rep.npz`.
     * **Pulled from the pod, framed (GoTTY trap) into `$OP/pod/`:** `config_p.json`, `summary_p.json`, `metrics_p.jsonl`,
       `weight_rule_rank_p.json`; and `w0_p.json` = `{"step0_digest": <config_p warm_start.sha256>, "source": …}`.
     * **C, REUSED:** the Stage 1 seam, dump, CSV and table (their sha pins); `$O1/argv_c.json`; `$O1/w0_c.json`;
       `raw/2026-11-24-scorerft-read/pod_config_c.json`.
     * **C's `metrics.jsonl`:** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/o3/card/evidence/metrics_scorerft_c.jsonl`, banked by the
       MM (sha256 `97a8631c…`). T's `metrics_scorerft_t.jsonl` (sha256 `09cdaf83…`) sits beside it as the positive control.
     * **Stage 1:** `$O1/read_stage1.json` and `$O1/seam_ledger.json`.
   * **B1 Preflight**, all must hold:
     1. The card sha equals the pin.
     2. `sha256sum` of every C and Stage 1 input equals the pins in `scorerft_read.py` §1.1.
     3. The WHOLE test file passes, holding the pytest lock dir; the pass count is 64.
     3b. **The positive control on the real logs.** Call `ctrl_s0` and `ctrl_order` with T's log in P's place: `metrics_scorerft_c.jsonl` vs
        `metrics_scorerft_t.jsonl`, `rank_w` 0.03125. Both must PASS, with 19 + 10 keys at step 0 and 0 of 1,000 steps differing, as the card's §3.1
        MEASURED. Otherwise STOP: the control code does not reproduce its own demonstration.
     4. **Stage 1 reproduction.** Run Part A §5's exact command with the modified tool into `$OP/repro/`. `read_stage1.json`'s `verdict`, `T_minus_C`,
        `levels`, `goal`, `mechanism`, `families` and `controls[*].pass` must equal Stage 1's (`cmp` of the extracted blocks). Otherwise STOP: the tool
        change moved the parent.
     5. `CHAIN.log` holds `SFTP chain start tip=751f3ae7`.
     6. Both seam reports read 12,146 rows and 0 misses.
   * **B2 One transp_score pass, with P, C and base.** C is re-scored from its sha-pinned Stage 1 seam; CTRL-CREUSE proves the re-scored C **is**
     Stage 1's C. This is required because `field_read.load_slim` (lines 68-86) **overwrites** a token's arms when the token occurs in two row files, so
     Stage 1's C rows cannot be read beside a P-only pass.
     ```
     PY_T $EV/p5c/p5c_read_prep.py convert --arm "P=<P seam>,<P dump>" "C=<C seam>,<C dump>" "base=<C1 seam>,<C1 dump>" --outdir $OP/plans
     PY_T $EV/p5c/p5c_read_prep.py score   --plans-dir $OP/plans --out-dir $OP/pass1 --procs 2 --python PY_N
     ```
   * **B3 Families, tactical, strategic:**
     ```
     PY_T $EV/p5c/p5c_families.py --rows "$OP/pass1/rows_s*of2.jsonl" --arms P C base --pairs P:C --out-dir $OP/families --python PY_N --repo $TIP
     PY_T $EV/p5c/p5c_tactical.py --seam C1=<P seam> C0=<C seam> base=<C1 seam> --dump C1=<P dump> C0=<C dump> base=<C1 dump> --out $OP/tactical.json
     PY_T $EV/instruments/strategic_family.py read --arm P=<P seam> --base C=<C seam> --out $OP/strategic_family_P_C.json
     ```
     `famci_P_vs_C.json` must carry `label_a C`, `label_b P`, or the read prints NOT-EVALUATED.
   * **B4 The read:**
     ```
     PY_T $EV/scorerft/scorerft_read.py --card pick --rows "$OP/pass1/rows_s*of2.jsonl" --p P --c C \
       --seam-p <P seam> --dump-p <P dump> --csv-p <P csv> --proptable-p "<propscore>/full_SFTP_s*of2.jsonl" --argv-p $OP/pod/argv_p.json \
       --seam-c <C seam> --dump-c <C dump> --csv-c <C csv> --proptable-c "<propscore>/full_SFTC_s*of2.jsonl" --argv-c $O1/argv_c.json \
       --w0-digest-c $O1/w0_c.json --w0-digest-p $OP/pod/w0_p.json --w0-ref f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4 \
       --det-seam-p <P rep seam> --seam-ledger $OP/seam_ledger.json --convert-report $OP/plans/convert_report.json \
       --metrics-c E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/o3/card/evidence/metrics_scorerft_c.jsonl --metrics-p $OP/pod/metrics_p.jsonl --pod-config-c <PK>/raw/2026-11-24-scorerft-read/pod_config_c.json --pod-config-p $OP/pod/config_p.json \
       --seam-report-c <C report.json> --seam-report-p <P report.json> --seam-chain-log E:/Projects/TanitAD-artifacts/scorerft_read/CHAIN.log \
       --weight-rule-p $OP/pod/weight_rule_rank_p.json --stage1-read $O1/read_stage1.json --stage1-ledger $O1/seam_ledger.json \
       --fam-ci $OP/families/famci_P_vs_C.json --tactical $OP/tactical.json --strategic $OP/strategic_family_P_C.json \
       --out $OP/read_p_stage1.json --text $OP/report_p_stage1.txt --verdict-line $OP/VERDICT_p_stage1.txt
     ```
     * `$OP/seam_ledger.json` is a NEW ledger: it is recorded at first sight and re-checked on a re-run. Stage 1's ledger enters only through
       `--stage1-ledger`.
     * `argv_p.json` is `{"argv": config_p["argv"], "source": "SFTP pod config.json argv (pulled framed, sha256 …)"}`.
   * **B5 Reading the output.**
     * The outcome order is OP1 … OP7, after the pre-launch PL1 / PL2 and the INVALID / INCOMPLETE states. The text names the committed next arm.
     * M1 is reported, not decisive.
     * **Every separated-adverse row is listed in the note** (the PI's every-row rule).
     * STRATEGIC is in full.
     * Update `GOALS_AND_CLAIMS.md` (`H-SCORERFT-PICK-1`) in the same turn as the verdict.
   * **B6 Stage 2** (only on `STAGE-2-TRIGGERED`). Run Cs, then Ps, then repeat B2 and B3 with `Cs=`, `Ps=` and `--arms Cs Ps --pairs Ps:Cs`. Then run
     the read with `--ps Ps --cs Cs`, the Xs inputs, `--pod-config-cs/-ps` and `--metrics-cs/-ps`, and `--fam-ci-s`. It prints OP1 or OP2.
3. **The "Dry runs" section (§7)** gains `dry_pick_*`: the pick card on stand-ins, with every output stamped `DRY RUN`.

## 4 · Dependencies outside these three files (the read cannot run without them; named here, not specified)
* **`raw/2026-11-20-scorer-ft/devbox/sft_seam_chain.sh`** (lines 15 and 81). The arm whitelist `SFTC|SFTT|SFTCs|SFTTs` must gain `SFTP|SFTPs`, and the
  CTRL-DET `rep` step (`case "$A" in SFTT|SFTTs`) must run for `SFTP|SFTPs`. The harness export stays `F:/TanitAD_p5cread_tip` (`751f3ae7`) for
  CTRL-HARNESS.
* **`run_sft_seam.cmd` / `hf_pull.py`** must accept the arm names `SFTP` / `SFTPs` and the HF prefix `scorerft/p` (`scorerft/p_s1`).
* **A chain script for Part B** (as `sft_read_s1.sh` was for Part A), with the B1-4 reproduction step as a hard gate. It is not specified here.
* **The pod side:** P's `config.json`, `summary.json`, `metrics.jsonl` and the weight-rule JSON, banked (HF + sha256) and pulled framed. C's
  `metrics.jsonl` is already banked (card §4.6).
* **MM decision 10 (label-consistent routing) lives in the trainer-side files.** The read cannot check it, so it is named here:
  * **`PK/refe/rank_loss.py`.** The `pick` branch computes s̃ from p̃, with per-factor stop-gradient under the live mask (card §1.1). It also gains the
    env mutation `pick_noroute`, which sets every factor live.
  * **`PK/refe/gate_checks.py`.** `rank_reference` keeps the loss-value audit, which is unchanged by routing. CTRL-ROUTE (b) is added to G-LIVE L3:
    on the smoke's labelled sets, the term-only logit gradient must be exactly 0.0 on every pick head with label 1 and every better-j head with label
    0. The count of such cells must be > 0, at least one live cell must be non-zero, and the RED arm is `pick_noroute`.
  * **`test_rank_loss.py`.** CTRL-ROUTE (a): the analytic exact zeros, the non-zero controls and the forward bit-identity. RED:
    `REFE_RANK_MUTATE=pick_noroute`, and the source-line mutation that deletes the stop-gradient.
  * `launch_gate.py` forbids `REFE_RANK_MUTATE` at launch already, so it needs no new pin.
  * The o3 `rank_loss.py` sha (`3f782697…`) is superseded.
