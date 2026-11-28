#!/usr/bin/env python3
"""Insert the H-VETO-FAN-1 register rows (P1 + P2 + the ctx-rank defect) after H-RL-VETO-1.

Anchored on the `| H-RL-VETO-1 |` row rather than a line number, and it REFUSES if the
anchor is not found exactly once -- a register edit that silently no-ops is worse than one
that fails, because nobody re-reads the file to check.
"""
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
PATH = os.path.join(REPO, "Project Steering", "GOALS_AND_CLAIMS.md")
CR, LF = chr(13), chr(10)
CRLF = CR + LF
ST, NE, W = chr(11088), chr(9940), chr(9888)
ELL = chr(8230)
RHO = chr(961)
MU = chr(956)
DEG = chr(176)
X = chr(215)

ROWS = """| D-RL-REWARD-RANK-1 | {ST}{ST} **THE COMPOSED RL REWARD IS *NOT* DISQUALIFIED AT SOURCE: WITHIN A WINDOW'S OWN FAN IT RANKS ENVELOPE-VIOLATING CANDIDATES *LOWER*, AND THE ONE TERM THAT DOES THE OPPOSITE IS `progress`.** MEASURED 2026-09-05 (`stack/scripts/rl_reward_envelope_rank.py`, one forward on the dev-box 4060, **240 EVAL windows / 121 episodes / 30,720 candidate scores**, per-window Spearman {RHO} over the 128 emitted candidates, episode-cluster bootstrap n_boot 4,000; T0 instrument probe, NON-PARITY not applicable {ELL} the EVAL corpus). A group-relative advantage IS a ranking, so this is the question that decides admissibility. **{RHO}(DEFAULT reward, envelope) = -0.5367 [-0.5581, -0.5138]** (n = 228 windows / 116 episodes; 12 windows UNDEFINED -- the flag had no variance -- reported, not averaged to zero), `kamm_over` **-0.5790**, `peak_g` **-0.4603**, `ttc_below` **-0.3544**, `contact` **-0.6326**, `off_reach` -0.0756 -- every one SEPARATED and NEGATIVE. {ST} **Attribution: `progress` is the only component with a POSITIVE correlation to violation** -- {RHO}(progress, peak_g) **+0.2900** [+0.1811, +0.3966], {RHO}(progress, ttc_below) **+0.4697**, {RHO}(progress, contact) **+0.3286**, all separated -- while {RHO}(comfort, peak_g) is **-0.9695** (an almost perfect inverse ranker of friction load) and {RHO}(feasibility, envelope) -0.5389. {NE} **AND IT PRICES THE OBVIOUS FIX AS WORTHLESS:** quadrupling `feasibility` moves {RHO}(envelope) -0.5367 {ARROW} **-0.5377** (nothing), whereas REMOVING `progress` moves {RHO}(peak_g) -0.4603 {ARROW} **-0.7306** and {RHO}(ttc_below) -0.3544 {ARROW} **-0.5632**. Controls all read their known values: self-correlation **+1.0000** exactly; the constant score **UNDEFINED on 240/240**; the random score **-0.0138** [-0.0250, -0.0022], the probe's own residual bias, below which \\|{RHO}\\| is not interpretable -- the effects above are ~39{X} it. Practical: the reward's own argmax picks an envelope-violating candidate on **5.4 %** of windows against the fan's base rate of **88.8 %** and the MODEL SELECTOR's **10.8 %** {ARROW} the rule reward is a *better* feasibility ranker than refcv3's trained scorer. {W} `peak_g` is the FINITE-DIFFERENCE load (the exact control-rolled `flyability.friction_load` needs `controls` and cannot be applied to a free-waypoint fan); it under-reports LEVELS by 1.21-1.85{X}, but a RANK correlation is invariant to a monotone under-report -- which is why this probe reads ranks | **SUPPORTED (MEASURED 2026-09-05)** -- the `H-RL-VETO-1` disqualification branch does NOT fire | `TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-veto-only-fan-safety/raw/reward_envelope_rank.json` (`panel`, `practical`, `per_window`), `raw/reward_envelope_rank.log`, `raw/fan_bank_base_240w.npz` (the 240{X}128 fan + every component + every flag, banked so ANY future reward design is scorable at **0 GPU**) |
| D-RL-VETO-EXPLICIT-1 | {ST} **THE VETO IS NOW A TYPED, RECORDED FIELD -- A ZERO-WEIGHT CONTROL IS AN ACTUAL NULL AGAIN, AND VETO-ONLY IS A PRODUCT ARM.** `posttrain.rl_objective` keyed the collision veto on `"collision" in spec.weights` (KEY MEMBERSHIP, true at weight 0.0) and the TTC channel on no config at all, while `advantage.py:146` pins vetoed candidates at -1 OUTSIDE the centring {ARROW} a constant reward gave a VETO-ONLY advantage at full strength (`D-RL-FANSAFE-2`; `ctrl_const` measured `veto_rate_mean` **0.0897**, `final_loss` **-1.863**, 14/57 metrics moved). FIXED 2026-09-05: the mask is composed by **`posttrain.veto_mask()`** from `PostTrainConfig.veto_enabled` / `veto_collision` / `veto_ttc`, written into every run's `config.json` by `to_dict()`, and it returns an all-False tensor (never `None`) when off so a `veto_rate` of 0.0 is still LOGGED -- a channel that reports 0.0 is evidence, one that reports nothing is not. `reg_echo` is pinned to `veto_collision=False`, which is what the old key-membership code gave it, so the banked arm still reproduces. New arms `veto200` / `veto2k` (zero reward, veto ON -- the product) and `ctrl_null` (zero reward, veto OFF -- the null `ctrl_const` was supposed to be). Pinned by **`stack/tests/test_rl_veto_explicit.py` (8 tests)**, which asserts BOTH directions and carries a same-breath control proving the constructed scene really contains the violation being vetoed; full RL suite **172 passed** | **SUPPORTED (MEASURED 2026-09-05)** -- closes escalation (2) of `.../2026-09-05-refc-rl-readiness/RESULT.md` §12.7 | `stack/tanitad/rl/{config,posttrain}.py`, `stack/tests/test_rl_veto_explicit.py`, `stack/scripts/rl_refcv3_min.py` (`ARMS`, `make_cfg`), `.../2026-09-05-veto-only-fan-safety/raw/patch_p2{,b}_*.py`, `raw/patch_p3_veto_arms.py` |
| D-RL-READOUT-CTXRANK-1 | {NE} **`readout()`'s R1/R2 WERE A BATCH {X} BATCH OUTER PRODUCT: EVERY WINDOW'S FAN WAS SCORED AGAINST EVERY WINDOW'S LEAD.** MEASURED 2026-09-05, reproduced in three lines. `rl_refcv3_min.reward_ctx()` shapes every scene fact for the TRAINING tensor `[B, N, G, S, 2]` (three leading axes) and `readout()` handed it a `[B, N, S, 2]` FAN (two). Right-alignment then broadcasts `lead_path [B,1,1,S,2]` against `traj [B,N,S,2]` to **`[B, B, N, S, 2]`**, and `v0 [B,1,1]` against `along [B,N]` the same way, so `_collision` / `_headway` / `_progress` returned `[B, B, N]`; the driver indexes `r1[j]` and means over a `[B, N]` slab of the wrong windows' leads. Readout batch = 4. **SCOPED, not blanket-voided: (a)** TRAINING is unaffected -- `make_sample_fn` really does hand a three-leading-axis tensor; **(b)** the PRIMARY fan-safety metrics are unaffected -- `FS.score_paths` is called with correctly-ranked `lead5 [B,1,5,2]` and `v0 [B]`; **(c)** the AFFECTED numbers are exactly `R1` and `R2` in every `readout_*.json` of the 2026-09-05 panel and their paired deltas (incl. *"R1 -0.0058, the stage did not improve its own objective out of sample"*, which is withdrawn as stated). FIXED: `reward_ctx(..., cand_dims=1)` plus a POSITIVE shape assertion in `readout` that raises instead of returning a number. {ST} **Corroborated by the first arm after the fix**: `R1` 0.5876 {ARROW} **0.6071** and `R2` 0.03612 {ARROW} **0.02923** while `R3`, `R_FAN`, `R_REACH`, `R_ORACLE` reproduce the banked base to every printed digit -- exactly the two metrics the defect could touch and nothing else. Root-cause CLASS: the `df` / Thor `free` / `step_s` / units family -- **a correct quantity computed in the wrong SCOPE, returning a number rather than an error** | **SUPPORTED (MEASURED 2026-09-05)** -- drafted for `RETRACTION_LOG.md` | `stack/scripts/rl_refcv3_min.py` (`reward_ctx`, `readout`), `.../2026-09-05-veto-only-fan-safety/raw/patch_p1a_readout_ctx_rank.py`, `SPEC.md` §3 |
| H-VETO-FAN-1 | {ST} **PRE-REGISTERED, both outcomes committed BEFORE any arm ran: the VETO ALONE (zero reward, DDv2's constraint channel only, `w_anchor` 1.0) makes refcv3's emitted fan measurably MORE feasible, and the gain CLEARS THE RIG'S OWN SEED-REPLICATE NOISE FLOOR.** Five arms on the frozen refcv3 @ 40,284 (md5 `b1ed7075{ELL}`), 120 train-split B1 v7.2 fit clips, readout on the SAME fixed 120 EVAL windows (`Random(1234)`), dev-box 4060 only: `ctrl_null` (veto OFF -- **G1: `veto_rate_mean` must read EXACTLY 0.0**), `veto200_s0/_s1`, `veto2k_s0/_s1`. {NE} **A separated CI is NECESSARY, NOT SUFFICIENT (`H-ESTIM-SEED-1`)**: `floor(m) = \\|paired delta(veto_s0, veto_s1)\\|` and a lever is quotable only if BOTH seeds separate, in the SAME sign, with `min(\\|d_s0\\|,\\|d_s1\\|) > floor(m)`. Per-metric separation floors from each metric's own quantum AND UNITS (1e-4 for rates, **1e-3 g** for `fan_peak_g_mean`, 1e-3 m/s for `fan_v_mean_2s_spread`); any metric whose base sits below its own floor is stamped **UNDETECTABLE-DOWNWARD**, never null. **SUCCESS** = `fan_peak_g_mean` negative and quotable AND at least one of `top32_infeasible` / `sel_infeasible` negative and quotable, AND T1 `ade_m` not regressed beyond the same floor. **FAILURE** = the gain does not clear the replicate floor, or ADE regresses past it. SECONDARY (T1, four families, per family, never pooled): committed prediction **NULL** -- a pure constraint with the trust region intact should not move driving quality | **OPEN -- pre-registered 2026-09-05, arms running; result rows follow in the same package** | `TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-veto-only-fan-safety/SPEC.md` §4-§7, `raw/run_veto_arms.sh`, `raw/analyze_veto.py` |
""".replace("{ST}", ST).replace("{NE}", NE).replace("{W}", W).replace("{ELL}", ELL) \
   .replace("{RHO}", RHO).replace("{X}", X).replace("{ARROW}", chr(8594))

raw = open(PATH, "rb").read()
is_crlf = CRLF.encode() in raw
s = raw.decode("utf-8").replace(CRLF, LF)

lines = s.split(LF)
idx = [i for i, l in enumerate(lines) if l.startswith("| H-RL-VETO-1 |")]
if len(idx) != 1:
    raise SystemExit("[register] expected exactly one `| H-RL-VETO-1 |` row, found %d" % len(idx))
i = idx[0]

# update H-RL-VETO-1's status cell in place
old_status = ("**PROPOSED 2026-09-05 (Deploy FlyWheel) " + chr(8212)
              + " both outcomes to be committed before launch; " + NE
              + " launch is the Master Mind's / PI's call**")
new_status = ("**EXECUTED 2026-09-05 (Arch+Inference FlyWheel) as `H-VETO-FAN-1` " + chr(8212)
              + " its 0-GPU precondition test RAN FIRST and the reward is **NOT** disqualified "
              + "(`D-RL-REWARD-RANK-1`: " + RHO + "(reward, envelope) = **-0.5367**, negative and "
              + "separated), so the veto-only arm proceeded with a SEED REPLICATE**")
if old_status in lines[i]:
    lines[i] = lines[i].replace(old_status, new_status)
    print("  [ok]   H-RL-VETO-1 status updated")
else:
    print("  [warn] H-RL-VETO-1 status cell not matched verbatim; row left unchanged")

new_rows = [r for r in ROWS.split(LF) if r.strip()]
if any(r.split("|")[1].strip() in s for r in new_rows if len(r.split("|")) > 1
       and r.split("|")[1].strip().startswith(("D-RL-REWARD-RANK-1", "H-VETO-FAN-1"))
       and ("| " + r.split("|")[1].strip() + " |") in s):
    print("  [skip] rows already present")
else:
    lines[i + 1:i + 1] = new_rows
    print("  [ok]   %d register rows inserted after H-RL-VETO-1" % len(new_rows))

out = LF.join(lines)
open(PATH, "wb").write((out.replace(LF, CRLF) if is_crlf else out).encode("utf-8"))
print("[register] written (%s)" % ("CRLF" if is_crlf else "LF"))
