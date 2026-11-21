"""Mutation harness for tests/test_v7f_r1r4.py -- every guard must FAIL on its mutant.

For each mutant: restore the pristine files from the working copy into the mutation copy,
apply ONE textual mutation, run the targeted tests, and record whether they FAILED (the
expected outcome). A mutant the suite does not kill is a guard that cannot fail.
Usage: python mutation_harness.py <pristine_stack> <mutation_stack> <pre_v6_path>
"""
import json
import os
import shutil
import subprocess
import sys

PRISTINE, MUT, PRE = sys.argv[1], sys.argv[2], sys.argv[3]
FILES = ["tanitad/models/v6.py", "tanitad/models/plan_speed_cap.py",
         "scripts/train_v6_staged.py", "tanitad/train/declared_vs_built_v6.py",
         "tests/test_v7f_r1r4.py", "tests/test_v6_stage_init_introduction.py"]

MUTANTS = [
    ("M1 default build constructs the R4 port (draws RNG, adds keys)",
     "tanitad/models/v6.py",
     'if getattr(cfg, "tac_op_cond", "off") != "off":\n            self.tac_op_port',
     'if True:\n            self.tac_op_port',
     ["test_default_BUILD_is_byte_identical_to_the_pre_change_module",
      "test_the_three_levers_default_OFF_and_build_nothing"]),
    ("M2 the R4 port is built but never added to the plan conditioning",
     "tanitad/models/v6.py",
     "e_plan = e_g_tac + self.tac_op_port(_src).to(e_g_tac.dtype)",
     "e_plan = e_g_tac + 0.0 * self.tac_op_port(_src).to(e_g_tac.dtype)",
     ["test_R4_ON_changing_the_tactical_BEHAVIOUR_moves_the_operative_plan",
      "test_R4_the_REAL_loss_step_makes_the_port_live_from_zero"]),
    ("M3 'detached' mode does not detach",
     "tanitad/models/v6.py",
     '_src = (e_a_tac.detach() if cfg.tac_op_cond == "detached"',
     '_src = (e_a_tac if cfg.tac_op_cond == "detached"',
     ["test_R4_detached_the_operative_loss_trains_the_port_and_NEVER_the_behaviour_heads"]),
    ("M4 the cap tracks speed WITHOUT the v6 floor (refav1 verbatim)",
     "tanitad/models/plan_speed_cap.py",
     "        v = (v + aj * float(dt)).clamp_min(0.0)\n    out = torch.stack(cols, dim=-1)",
     "        v = (v + aj * float(dt))\n    out = torch.stack(cols, dim=-1)",
     ["test_cap_LITERAL_tracks_the_v6_integrators_FLOOR_at_zero"]),
    ("M5 the cap is applied AFTER selection (selector scores the uncapped fan)",
     "tanitad/models/v6.py",
     "for k, v in self.cand_score(wp, g_tac_embed, **kw).items()}",
     "for k, v in self.cand_score(self.emission(feat, v0)[2], g_tac_embed, **kw).items()}",
     ["test_cap_binds_BEFORE_selection"]),
    ("M6 no braking floor: v0 above the limit is not braked at -a_max",
     "tanitad/models/plan_speed_cap.py",
     "lim = ((vmax - v) / float(dt)).clamp(min=-float(a_max))",
     "lim = ((vmax - v) / float(dt)).clamp(min=0.0)",
     ["test_cap_LITERAL_v0_ABOVE_the_limit_brakes_at_a_max_until_under_it"]),
    ("M7 the R1a port output lands on the WM-side z_tac, not the decision view z_tac_p",
     "tanitad/models/v6.py",
     "            z_tac_p = z_tac_p + self.vmax_tac(",
     "            z_tac = z_tac + self.vmax_tac(",
     ["test_R1a_the_limit_moves_the_tactical_DECISIONS_and_not_the_WM_latent"]),
    ("M8 an invalid max-speed row is fed as 30 km/h (validity ignored)",
     "tanitad/models/v6.py",
     "_oh = speed_max_onehot(_bin, _ok, dtype=torch.float32)",
     "_oh = speed_max_onehot(_bin, None, dtype=torch.float32)",
     ["test_R1a_an_INVALID_row_is_all_zero_not_30_kmh"]),
    ("M9 the eval-mode cap refusal is gone",
     "tanitad/models/v6.py",
     "if v_max is None and getattr(cfg, \"plan_vmax_cap\", False) " + chr(92) + "\n                and not self.training:",
     "if False:",
     ["test_the_cap_is_MANDATORY_at_inference_and_absent_in_training"]),
    ("M10 the trainer drops the refcv6 stamp",
     "scripts/train_v6_staged.py",
     'out["speed_max_derivation_v6"] = v6ms.SPEED_MAX_DERIVATION_V6',
     'pass',
     ["test_the_DRY_RUN_exercises_all_three_levers_and_the_cap_never_exceeds"]),
    ("M11 G-DVB reads nothing (always clean)",
     "tanitad/train/declared_vs_built_v6.py",
     "    out: list[Mismatch] = []\n    for lever in REGISTRY_V6.values():",
     "    out: list[Mismatch] = []\n    for lever in []:",
     ["test_G_DVB_v6_CATCHES_a_lever_declared_but_not_built"]),
    ("M12 synthetic batch always carries v_max keys",
     "scripts/train_v6_staged.py",
     'if getattr(stack, "vmax_tac", None) is not None:\n        out["v_max_ms"]',
     'if True:\n        out["v_max_ms"]',
     ["test_synthetic_train_batch_carries_v_max_ONLY_when_the_port_is_built"]),
    ("M13 the preflight forgets the S-W refusal",
     "scripts/train_v6_staged.py",
     '    if a.stage == "S-W":\n        for flag, on, grp in',
     '    if False:\n        for flag, on, grp in',
     ["test_preflight_REFUSES_each_lever_in_S_W"]),
    ("M14 the sidecar join ignores validity",
     "scripts/train_v6_staged.py",
     "if r is not None and float(r[1]) > 0.5:",
     "if r is not None:",
     ["test_the_sidecar_JOIN_is_per_episode_by_stable_id_and_refuses_zero_coverage"]),
]

env = dict(os.environ)
env["PYTHONPATH"] = MUT + ";" + os.path.join(os.path.dirname(MUT), "..", "taniteval")
env["TANITAD_V6_PRE_R1R4"] = PRE
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"
rows = []
for name, rel, old, new, tests in MUTANTS:
    for f in FILES:
        shutil.copyfile(os.path.join(PRISTINE, f), os.path.join(MUT, f))
    p = os.path.join(MUT, rel)
    src = open(p, encoding="utf-8").read()
    n = src.count(old)
    if n != 1:
        rows.append({"mutant": name, "status": f"ANCHOR NOT UNIQUE ({n})"})
        continue
    open(p, "w", encoding="utf-8", newline="").write(src.replace(old, new))
    k = " or ".join(tests)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:randomly",
                        "-p", "no:cacheprovider", "tests/test_v7f_r1r4.py", "-k", k],
                       cwd=MUT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
    rows.append({"mutant": name, "targeted": tests, "pytest_rc": r.returncode,
                 "summary": tail, "KILLED": r.returncode != 0})
for f in FILES:
    shutil.copyfile(os.path.join(PRISTINE, f), os.path.join(MUT, f))
print(json.dumps(rows, indent=1))
print("KILLED", sum(1 for x in rows if x.get("KILLED")), "of", len(rows))
