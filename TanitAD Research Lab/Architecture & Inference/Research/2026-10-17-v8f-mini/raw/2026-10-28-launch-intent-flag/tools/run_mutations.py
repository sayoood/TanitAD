"""Reachability proof by MUTATION (CLAUDE.md 2026-09-10/11): delete / break ONE rule at a time in a COPY of the tree and require stack/tests/test_v8f_launch_intent.py to go RED.
Run: python run_mutations.py <mutation tree> <out file>. Sequential; each pytest starts only when fewer than 2 pytest runs are active (the session cap). The copy is restored (sha256-verified) after every mutant.
Every mutant names the -k selection that must kill it (a narrow selection keeps the proof fast; a CONTROL per distinct selection must be GREEN on the unmutated tree first)."""
import hashlib
import os
import subprocess
import sys
import time

TREE, OUT = sys.argv[1], sys.argv[2]
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
S = "F:/TanitAD-scratch/launch_intent"
TRAIN, MODEL = "stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py"
TEST = "stack/tests/test_v8f_launch_intent.py"
ENV = {"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "4", "TMP": f"{S}/tmp", "TEMP": f"{S}/tmp", "TMPDIR": f"{S}/tmp", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
LINE = 'total = total + cfg.w_traj * cfg.launch_intent_w * T["launch_along"] / 3.0 if cfg.launch_intent_w > 0.0 else total'

# name, rule, [(file, old, new)], -k selection that must kill it
M = [
    ("M01_total_line_deleted", "delete the ONE line that adds the term to the total", [(MODEL, LINE, "total = total")], "total_moves or real_run_reaches or probe_passes"),
    ("M02_ge_to_gt", "flip the registered >= to > in the launch mask", [(MODEL, "(v6 >= v0 + LAUNCH_DV)", "(v6 > v0 + LAUNCH_DV)")], "exact_boundary"),
    ("M03_v0_strict_to_le", "v0 < 6 becomes v0 <= 6", [(MODEL, "(v0 < LAUNCH_V0_MAX)", "(v0 <= LAUNCH_V0_MAX)")], "v0_bound_is_strict or hand_computed"),
    ("M04_chord_on_knots_28_27", "the chord is taken over knots 28 -> 29 instead of 29 -> 30", [(MODEL, "(gt[:, 29] - gt[:, 28])", "(gt[:, 28] - gt[:, 27])")], "last_two_knots or hand_computed or numpy_spelling"),
    ("M05_chord_x_only", "the chord speed reads the along component only", [(MODEL, "torch.linalg.norm((gt[:, 29] - gt[:, 28]).to(torch.float64), dim=-1)", "(gt[:, 29] - gt[:, 28]).to(torch.float64)[:, 0].abs()")], "hand_computed or numpy_spelling"),
    ("M06_knots_to_13", "the up-weighted horizon is knots 1-13", [(MODEL, "LAUNCH_KNOT_LO, LAUNCH_KNOT_HI = 1, 12", "LAUNCH_KNOT_LO, LAUNCH_KNOT_HI = 1, 13")], "gradient_is_nonzero or term_equals or registered_six"),
    ("M07_knots_from_2", "the up-weighted horizon starts at knot 2", [(MODEL, "LAUNCH_KNOT_LO, LAUNCH_KNOT_HI = 1, 12", "LAUNCH_KNOT_LO, LAUNCH_KNOT_HI = 2, 12")], "gradient_is_nonzero or term_equals or registered_six"),
    ("M08_cross_axis", "the term reads the cross axis instead of the along axis", [(MODEL, "dx = traj30[:, lo:hi, 0] - gt[:, lo:hi, 0]", "dx = traj30[:, lo:hi, 1] - gt[:, lo:hi, 1]")], "term_equals or gradient_is_nonzero"),
    ("M09_denominator_launch_rows", "the term is normalised by the launch rows instead of every GT-valid row", [(MODEL, "return (per * g).sum() / gok.sum().clamp_min(1.0)", "return (per * g).sum() / g.sum().clamp_min(1.0)")], "denominator"),
    ("M10_mask_ignored_in_term", "the term ignores the mask (every GT-valid row is up-weighted)", [(MODEL, "g = gok * launch.to(gok.dtype)", "g = gok")], "gradient_is_nonzero or all_false_mask or hand_computed"),
    ("M11_mask_after_the_stages", "the mask is taken AFTER the ego stages (the swapped own state, not the label)", [(TRAIN, "                lmask = launch_stage(b)\n", "                pass\n"), (TRAIN, '                b["launch"] = lmask\n', '                b["launch"] = lmask = launch_stage(b)\n')], "LOGGED_state_not_the_swapped"),
    ("M12_attach_deleted", 'delete `b["launch"] = lmask` in the loop', [(TRAIN, '                b["launch"] = lmask\n', "                pass\n")], "real_run_reaches"),
    ("M13_loop_never_launches", "the loop's launch_on is never true: the lever is declared, built, probed and never applied", [(TRAIN, "    launch_on = cfg.launch_intent_w > 0.0", "    launch_on = False")], "real_run_reaches"),
    ("M14_flag_checks_entry_deleted", "the flag has no G-DVB entry in FLAG_CHECKS", [(TRAIN, '    "launch_intent_w": "cfg:launch_intent_w",', '    "launch_intent_w_x": "cfg:launch_intent_w",')], "gdvb_entry or without_a_gdvb"),
    ("M15_gdvb_readback_deleted", "delete the G-DVB declared == built read-back of the lever", [(TRAIN, '    if built.get("launch_intent_w") != a.launch_intent_w:', "    if False:")], "build_did_not_honour"),
    ("M16_built_lever_constant", "built_levers reports 0.0 whatever was built", [(MODEL, '"launch_intent_w": cfg.launch_intent_w,', '"launch_intent_w": 0.0,')], "gdvb_entry or without_a_gdvb or build_did_not_honour"),
    ("M17_probe_stubbed", "the G-LIVE probe is not run (a stub that always passes)", [(TRAIN, "        lip = launch_intent_probe(model, probe, cfg)", '        lip = {"launch_intent": "off", "pass": True, "launch_n": 0, "launch_eligible": 0}  # M17\n        _unused = (model, probe, cfg)')], "real_run_reaches"),
    ("M18_share_without_the_one", "launch_weight_share gives launch samples W instead of 1 + W", [(TRAIN, "return ((1.0 + w) * n_launch) / ((1.0 + w) * n_launch + (n_eligible - n_launch))", "return (w * n_launch) / (w * n_launch + (n_eligible - n_launch))")], "weight_share_hand_values or dose_rule or real_run_reaches"),
    ("M19_launch_n_not_counted", "the loop does not count the launch rows", [(TRAIN, 'c_["n"] += int(lmask.sum())', 'c_["n"] += 0')], "real_run_reaches"),
    ("M20_checkpoint_loses_totals", "the checkpoint does not carry the launch totals (a resume would lose them)", [(TRAIN, '            blob["launch_totals"] = dict(launch_tot)', "            pass")], "real_run_reaches or resumed"),
    ("M21_default_on", "the parser default is 1.0 (the lever is on unless switched off)", [(TRAIN, 'ap.add_argument("--launch-intent-w", dest="launch_intent_w", type=float, default=0.0,', 'ap.add_argument("--launch-intent-w", dest="launch_intent_w", type=float, default=1.0,')], "gdvb_entry or default_run or flag_off"),
    ("M22_guard_ge", "the loss computes the term at W = 0 as well (guard > becomes >=)", [(MODEL, "        if cfg.launch_intent_w > 0.0:  ", "        if cfg.launch_intent_w >= 0.0:  ")], "off_adds_no_term or default_run"),
    ("M23_scale_third_dropped", "the total adds w_traj x W x term, without ADE's / 3 scale", [(MODEL, LINE, 'total = total + cfg.w_traj * cfg.launch_intent_w * T["launch_along"] if cfg.launch_intent_w > 0.0 else total')], "total_moves or real_run_reaches or probe_passes"),
    ("M24_w_traj_dropped", "the total adds W x term / 3, without the w_traj factor", [(MODEL, LINE, 'total = total + cfg.launch_intent_w * T["launch_along"] / 3.0 if cfg.launch_intent_w > 0.0 else total')], "total_moves"),
    ("M25_negative_weight_allowed", "the config accepts a negative weight", [(MODEL, "if not (math.isfinite(self.launch_intent_w) and self.launch_intent_w >= 0.0):", "if not math.isfinite(self.launch_intent_w):")], "config_refuses"),
    ("M26_forced_mask_empty", "the probe's forced mask admits no row (the wire cannot be seen)", [(TRAIN, "forced[gok.nonzero().flatten()[: max(1, elig // 2)]] = True", "forced[gok.nonzero().flatten()[:0]] = True")], "sees_the_wire or probe_passes or real_run_reaches"),
    ("M27_allfalse_control_deleted", "delete the probe's all-False-mask control", [(TRAIN, '    if term_n != 0.0 or res["control_allfalse_total_diff"] != 0.0 or res["control_allfalse_grad_diff"] != 0.0:', "    if False:")], "ignores_the_mask"),
    ("M28_probe_batch_mask_deleted", "the gate's probe batch carries no mask", [(TRAIN, '            probe["launch"] = launch_stage(probe)', "            pass")], "real_run_reaches"),
    ("M29_wtraj_clause_deleted", "delete the G-DVB w_traj > 0 clause", [(TRAIN, "    if a.launch_intent_w > 0.0 and not model.cfg.w_traj > 0.0:", "    if False:")], "w_traj_zeroes"),
    ("M30_launch_stage_wrong_column", "launch_stage reads kin3's second column (a0) as v0", [(TRAIN, 'return M.launch_mask(batch["kin3"][:, 0], batch["gt"], batch["gt_ok"])', 'return M.launch_mask(batch["kin3"][:, 1], batch["gt"], batch["gt_ok"])')], "LOGGED_state_not_the_swapped or real_run_reaches"),
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def slot():
    import random
    for _ in range(720):
        n = subprocess.run([PY, "C:/Users/Admin/orcread/pytest_runs.py"], capture_output=True, text=True).stdout.strip()
        if n.isdigit() and int(n) < 2:
            time.sleep(random.uniform(3, 12))                                                          # a jittered second look: two pipelines must not both see a free slot at once
            n = subprocess.run([PY, "C:/Users/Admin/orcread/pytest_runs.py"], capture_output=True, text=True).stdout.strip()
            if n.isdigit() and int(n) < 2:
                return
        time.sleep(30)
    raise SystemExit("ZZNO-SLOT")


def run(k=""):
    slot()
    cmd = [PY, "-m", "pytest", TEST, "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={S}/tmp/btm/run", "-W", "ignore"] + (["-k", k] if k else [])
    subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", f"{S}\\tmp\\btm\\run"], capture_output=True)
    os.makedirs(f"{S}/tmp/btm", exist_ok=True)
    r = subprocess.run(cmd, cwd=TREE, capture_output=True, text=True, env={**os.environ, **ENV}, encoding="utf-8", errors="replace")
    tail = r.stdout.strip().splitlines()
    failed = [l for l in tail if l.startswith("FAILED") or l.startswith("ERROR")]
    return r.returncode, (failed[0] if failed else ""), (tail[-1] if tail else "")


def main():
    only = set(sys.argv[3:])                                                                           # optional: run only these mutant names
    base = {f: sha(f"{TREE}/{f}") for f in (TRAIN, MODEL, TEST)}
    out = open(OUT, "a" if only else "w", encoding="utf-8", newline="\n")
    w = lambda s: (out.write(s + "\n"), out.flush())  # noqa: E731
    w(f"MUTATION PROOF  {time.strftime('%Y-%m-%d %H:%M:%S')}  tree={TREE}")
    w(f"tests: {TEST}  sha256 {base[TEST][:16]}  (the unmutated trainer {base[TRAIN][:16]}, model {base[MODEL][:16]})")
    sel = [m for m in M if not only or m[0] in only]
    for k in sorted({m[3] for m in sel}):                                                              # a control per distinct selection: the selection itself must be GREEN unmutated
        rc, failed, last = run(k)
        w(f"CONTROL (no mutation) -k {k!r}: rc={rc}  [{last}]   <- must be 0 (GREEN)" + (f"  FIRST FAILING: {failed}" if rc else ""))
        if rc:
            w("ABORT: a control is not green")
            out.close()
            raise SystemExit(2)
    survivors = []
    for name, rule, edits, k in sel:
        texts = {}
        for f, old, new in edits:
            texts.setdefault(f, open(f"{TREE}/{f}", encoding="utf-8", newline="").read())
        for f, old, new in edits:
            assert texts[f].count(old) == 1, f"{name}: anchor found {texts[f].count(old)} times in {f}: {old[:80]!r}"
            texts[f] = texts[f].replace(old, new)
        for f, t in texts.items():
            open(f"{TREE}/{f}", "w", encoding="utf-8", newline="").write(t)
        try:
            rc, failed, last = run(k)
        finally:                                                                                       # restore from the pristine copies, then verify the bytes
            for f in texts:
                open(f"{TREE}/{f}", "w", encoding="utf-8", newline="").write(open(f"{S}/mutpristine/{f.replace('/', '__')}", encoding="utf-8", newline="").read())
        assert all(sha(f"{TREE}/{f}") == h for f, h in base.items()), "restore failed"
        verdict = "RED (killed)" if rc != 0 else "GREEN  <-- SURVIVED"
        if rc == 0:
            survivors.append(name)
        w(f"{name}: {verdict}  rule: {rule}\n    first failing: {failed or '-'}   [{last}]")
    w(f"SUMMARY: {len(sel) - len(survivors)}/{len(sel)} mutants killed; survivors: {survivors or 'none'}")
    out.close()


main()
