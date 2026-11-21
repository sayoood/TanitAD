#!/usr/bin/env python3
"""One-shot patcher: adds the refe_sft4r profile to launch_gate.py (SFT-4R readiness, 2026-10-16)."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8", newline="").read()
assert "\r\n" not in s

# ---- 1. header docstring: usage line
old = "    # OPTIONAL, to take G-SUITE's CPU-only minutes OFF the GPU's idle clock"
new = ("    # SFT-4R (refe/sft4r_chain.sh; trained by scorer_finetune.py): the SFT-4 recipe on the deployed model_final.pt over the COMPLETE PDM labels.  The argv is PINNED token for token, the preflight is the trainer's own,\n"
       "    # and the egress gate (hf_egress.py gate) is a required check, so the token covers 'the run can leave the pod'.  See eval/PREREG_SFT4R.md Amendment 3.\n"
       "    python launch_gate.py run --profile refe_sft4r --pkg <pkg> --args \"$ARGS\" --out-dir <dir> --gate-preflight-json <preflight.jsonl> --gate-egr-json <g_egr_result.json>\n\n"
       "    # OPTIONAL, to take G-SUITE's CPU-only minutes OFF the GPU's idle clock")
assert s.count(old) == 1
s = s.replace(old, new)

# ---- 2. constants (after the P2-s1 block's P2REP_ARGV line)
anchor = "P2REP_ARGV = p2rep_argv()\n"
assert s.count(anchor) == 1
block = anchor + '''# SFT-4R (eval/PREREG_SFT4R.md + Amendment 3; chain refe/sft4r_chain.sh; pod script refe/sft4r_pod.sh): the SFT-4 recipe on the DEPLOYED model_final.pt, ONE pass over the COMPLETE PDM labels, scorer only
# (score_q_mlp / score_dec / score_head), arms A and B in one pass.  It is ONE argv, so the profile PINS it token for token (`pinned_argv`, like refe_p2rep), and asserts what a pin cannot:
#   * the EFFECTIVE recipe (`sft4r_refusals`): a parser DEFAULT can move under a pinned argv (--epochs, --warmup, --batch, --accum, --eval-every and --seed are NOT named in it);
#   * from the trainer's OWN `data` event (`data_expect`, G-DVB), that the labels are COMPLETE: SFT-4 launched on 146,185 of 174,912 because "complete" was never a computed fact (PREREG_SFT4 Amendment 4).
#     `train_sets` is the number of served sets that have a PDM label under --require-pdm, so it IS the label coverage as the trainer sees it; the schedule is re-derived from it (2,733 updates).
# The values are MEASURED on the pod: the S1 chain's preflight `data` event for the very same inputs (raw/2026-10-16-sft4r-readiness/evidence/s1_n_s0_data_event.json, written 2026-10-14).
SFT4R_RUN_DIR = "/workspace/data/refe_sft4r"
SFT4R_RECIPE = {"lr": 1e-4, "warmup": 50, "batch": 8, "accum": 8, "epochs": 1.0, "max_updates": 0, "eval_every": 400, "seed": 0,       # PREREG_SFT4 "Settings" (AdamW, 50-update warm-up, cosine to 0, 8 x 8, 1 epoch, seed 0)
                "b_mode": "expreward", "lam": 1.0, "tau_s": 0.1, "lane_weight": 1.0}
SFT4R_UPDATES = 2733                                                                    # ceil(174,912 / 8 / 8) = scorer_finetune.py:1363; PREREG_SFT4R's "2,335" is the lost run's 146,185-label epoch
SFT4R_DATA_EXPECT = {"train_sets": 174912, "train_sets_in_bank": 174912, "heldout_sets": 3137, "heldout_in_bank": 3137, "heldout_logs": 24, "train_rows_in_heldout_logs": 0,
                     "lane_labels.pdm_train": 174912, "lane_labels.pdm_train_stale": 0, "lane_labels.pdm_train_superseded": 0, "lane_labels.pdm_heldout": 3137,
                     "lane_labels.heldout_sets": 3137, "lane_labels.heldout_stale": 0, "lane_labels.teacher_lane_heldout": 3137}
SFT4R_DATA_MIN = {"lane_labels.teacher_lane_train": 2109}                              # PREREG_SFT4R "Changed 2": the lost run's count is the FLOOR (the seeded 15,000-set subset, the rows written by launch)
SFT4R_FORBIDDEN = ("--preflight", "--lane-labels-train", "--b-compw", "--tau-t", "--aux-drv-shuffle-seed", "--max-updates") + AUX_HEAD_FLAGS + AUXS1_EPO_FLAGS
SFT4R_SUITE = [{"script": "test_sft4r_chain.py", "token": "ZZTEST_PASS"}, {"script": "test_sft4r_gate.py", "token": "ZZTEST_PASS"},
               {"script": "test_launch_gate.py", "args": ["--static", "--no-banked"], "token": "ZZLG_STATIC_PASS"}]
# SFT4R_ARGV is the pod script's argv as it EXPANDS on the pod (WS=/workspace): `MODE=argv bash refe/sft4r_pod.sh`; test_sft4r_gate.py diffs the two copies.
SFT4R_ARGV = ("--ckpt /workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt --targets /workspace/data/refe_navtrain/train_grow --images /workspace/data/navtrain_pixels "
              "--calib /workspace/data/refe_navtrain/train_grow/calib_table.json --onpolicy /workspace/data/refe_navtrain/onpolicy/sets --heldout /workspace/data/refe_heldout/sets "
              "--heldout-logs /workspace/data/refe_heldout/heldout_logs.txt --workers 3 --lr 1e-4 --b-mode expreward --lam 1.0 --tau-s 0.1 --lane-head --lane-weight 1.0 "
              "--teacher-lane-train /workspace/data/refe_sft2/train_teacher_lane --teacher-lane-heldout /workspace/data/refe_sft2/heldout_teacher_lane "
              "--lane-labels-heldout /workspace/data/refe_sft2/heldout_lane --pdm-labels-heldout /workspace/data/refe_sft2/heldout_pdm "
              "--pdm-labels-train /workspace/data/refe_sft2/train_pdm --require-pdm --out " + SFT4R_RUN_DIR)
'''
s = s.replace(anchor, block)

# ---- 3. the profile (after refe_p2rep)
anchor2 = 'PROFILES["refe_p2rep"] = {**PROFILES["refe_p2"], "name": "refe_p2rep",'
assert s.count(anchor2) == 1
i = s.index(anchor2)
j = s.index("\n", s.index('"pinned_argv": tuple(P2REP_ARGV)}', i)) + 1
prof = '''# SFT-4R: scorer_finetune.py's runner (static checks + the trainer's own --preflight, like refe_rls1 / refe_aux_s1) with NO arm table (one arm: A and B share a pass), the whole argv PINNED, the registered recipe
# asserted on the EFFECTIVE values (`recipe_rules` = sft4r), the labels' completeness asserted from the trainer's `data` event (`data_expect` / `data_expect_min` / `data_expect_updates`, G-DVB) and the
# egress gate a REQUIRED check (G-EGR, PREREG_SFT4R "Changed 3": the run must be able to leave the pod before its first step; no checkpoint is pushed, so `egr_require_ckpt` is False).
PROFILES["refe_sft4r"] = {**_BASE, "name": "refe_sft4r", "runner": "scorer", "recipe_rules": "sft4r", "trainer": "refe/scorer_finetune.py", "parser_module": "scorer_finetune",
                          "trainer_src": "scorer_finetune.py", "namespace_funcs": ("main", "epo_refusals", "_g_probe", "aux_s1_refusals"), "data_flags": RLS1_DATA,
                          "checks": ("G-HYG", "G-DVB", "G-PREFLIGHT", "G-SUITE", "G-EGR"), "na_ok": ("G-CLOCK",), "suite": SFT4R_SUITE, "live_terms": ("bceA", "lossB"),
                          "clock_na": "SFT-4R has no label clock of its own: its labels are the PDM relabel of an on-policy bank, stamped by the relabeller's ckpt_step; the preflight's G9 bounds the stale "
                                      "share and G-DVB asserts the coverage (train_sets == 174,912, pdm_train_stale == 0) from the trainer's own data event",
                          "forbidden_flags": SFT4R_FORBIDDEN, "output_flags": ("--out",),
                          "required_flags": {"--pdm-labels-train": None, "--pdm-labels-heldout": None, "--require-pdm": None, "--lane-head": None, "--teacher-lane-train": None,
                                             "--teacher-lane-heldout": None, "--lane-labels-heldout": None, "--b-mode": "expreward", "--lam": "1.0", "--tau-s": "0.1", "--lr": "1e-4"},
                          "provisional_flags": (), "pinned_argv": tuple(SFT4R_ARGV.split()), "data_expect": SFT4R_DATA_EXPECT, "data_expect_min": SFT4R_DATA_MIN,
                          "data_expect_updates": SFT4R_UPDATES, "egr_require_ckpt": False}
'''
s = s[:j] + prof + s[j:]

# ---- 4. sft4r_refusals, before the orchestration section
anchor3 = "# ============================================================================================ orchestration\n"
assert s.count(anchor3) == 1
fn = '''# ============================================================================================ SFT-4R recipe rules (refe_sft4r)
def sft4r_refusals(prof: dict, argv: list, mod, parser) -> tuple:
    """(refusals, extra live terms) of the SFT-4R recipe rules.  The argv is PINNED (`pinned_argv`), so this asserts what a pin cannot: the EFFECTIVE recipe (a parser default moving under an argv that
    does not name it), exact flag spelling (argparse accepts a unique prefix; the gate's flag tests are on exact tokens), scorer_finetune's OWN refusals (epo + aux), and that the arm is the
    plain scorer fine-tune: no EPO term, no auxiliary head, the full epoch (no --max-updates), the lane head with both of its label sets."""
    import gate_checks as C
    out = []
    name = prof["name"]
    ns, err = C.parse_scorer_argv(parser, argv)
    if err:
        out.append(f"{name}: the trainer's parser rejects this argv ({err})")
        return out, ()
    known = {o for x in parser._actions for o in x.option_strings}
    for t in argv:
        if t.startswith("--") and t not in known:
            out.append(f"{name}: {t} is not an exact option of the trainer (argparse accepts an abbreviation; the gate's flag tests are on exact tokens)")
    out += [f"scorer_finetune refuses this argv: {m}" for m in mod.epo_refusals(ns, parser) + mod.AUX.aux_refusals(ns) + mod.aux_s1_refusals(ns)]

    def how(flag):
        return "named" if flag in argv else "the parser DEFAULT"
    for dest, want in SFT4R_RECIPE.items():
        f = "--" + dest.replace("_", "-")
        if not C._close(getattr(ns, dest), want):
            out.append(f"{name} requires {f} {want} (PREREG_SFT4 'Settings', the SFT-4R recipe); the argv resolves to {getattr(ns, dest)} ({how(f)})")
    if ns.epo or ns.aux_drv:
        out.append(f"{name}: --epo / --aux-drv are on; SFT-4R is the plain scorer fine-tune (the lever is the COMPLETE labels, nothing else)")
    if ns.lane_head and not (ns.teacher_lane_train and ns.teacher_lane_heldout):
        out.append(f"{name}: --lane-head needs --teacher-lane-train AND --teacher-lane-heldout (the 7th output would have no label)")
    if (ns.teacher_lane_train or ns.teacher_lane_heldout) and not ns.lane_head:
        out.append(f"{name}: --teacher-lane-* is named without --lane-head: the labels would be loaded and never trained on")
    return out, ()


'''
s = s.replace(anchor3, fn + anchor3)

# ---- 5. profile_refusals hook
old5 = '''    elif prof.get("arm_rules") == "auxs1":
        rs, extra = auxs1_refusals(prof, argv, T, parser)
        out += rs
        terms += extra
'''
assert s.count(old5) == 1
s = s.replace(old5, old5 + '''    elif prof.get("recipe_rules") == "sft4r":
        rs, extra = sft4r_refusals(prof, argv, T, parser)
        out += rs
        terms += extra
''')

# ---- 6. run_checks scorer branch: G-EGR
old6 = '''        if "G-SUITE" in want:
            emit(suite_check(prof, pkg, evidence=suite_evidence, key=key, commit=commit, arm=arm, rehearsal=rehearsal))
        if "G-CLOCK" in want:
            emit(C.evidence("G-CLOCK", "NA", [prof.get("clock_na", "no label clock in this profile")], {}, arm=arm, rehearsal=rehearsal))
        return status
'''
assert s.count(old6) == 1
s = s.replace(old6, '''        if "G-SUITE" in want:
            emit(suite_check(prof, pkg, evidence=suite_evidence, key=key, commit=commit, arm=arm, rehearsal=rehearsal))
        if "G-EGR" in want:                                  # refe_sft4r: hf_egress.py gate's own result, bound into the token (`gate:egr_json`); this arm pushes no checkpoint
            emit(C.check_egr(egr_json, arm=arm, rehearsal=rehearsal, require_ckpt=bool(prof.get("egr_require_ckpt", True))))
        if "G-CLOCK" in want:
            emit(C.evidence("G-CLOCK", "NA", [prof.get("clock_na", "no label clock in this profile")], {}, arm=arm, rehearsal=rehearsal))
        return status
''')

# ---- 7. the CLI help for --gate-egr-json
old7 = 'p.add_argument("--gate-egr-json", help="refe_p6: hf_egress.py gate\'s g_egr_result.json (G-EGR)")'
assert s.count(old7) == 1
s = s.replace(old7, 'p.add_argument("--gate-egr-json", help="refe_p6, refe_sft4r: hf_egress.py gate\'s g_egr_result.json (G-EGR)")')
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched", p)
