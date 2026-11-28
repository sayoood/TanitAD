"""Reachability proof by MUTATION (CLAUDE.md 2026-09-10/11): delete / break ONE rule at a time in a COPY of the tree and require stack/tests/test_v8f_ego_dagger_compose.py to go RED.
Run: python run_mutations.py <mutation tree> <out file>. Sequential; each pytest starts only when fewer than 2 pytest runs are active (the session cap). The copy is restored (sha256-verified) after every mutant."""
import hashlib
import subprocess
import sys
import time

TREE, OUT = sys.argv[1], sys.argv[2]
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
TRAIN, DATA = "stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py"
TEST = "stack/tests/test_v8f_ego_dagger_compose.py"
ENV = {"CUDA_VISIBLE_DEVICES": "-1", "TMP": "F:/TanitAD-scratch/compose_fix/tmp", "TEMP": "F:/TanitAD-scratch/compose_fix/tmp", "TMPDIR": "F:/TanitAD-scratch/compose_fix/tmp",
       "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}

# name, rule, [(file, old, new)], expected-to-be-caught-by (-k expression; "" = the whole file)
M = [
    ("M01_ego_apply_deleted_on_composed_path", "delete the perturbation's apply on the composed path", [(TRAIN, "out = self.ego.apply(swapped, step, p=ego_p, exclude=sw)", "out = swapped")], "composition_overlap or overlap_guard or glive_a_composed or glive_deliberate or gdvb or ego_exclude or refused or exclusivity"),
    ("M02_swap_deleted_on_composed_path", "delete the swap's apply on the composed path", [(TRAIN, "swapped = self.dag.apply(batch, step, p=dag_p)", "swapped = batch")], "composition_overlap or overlap_guard or glive_a_composed or glive_deliberate or gdvb or ego_exclude or refused or exclusivity"),
    ("M03a_exclusion_not_passed", "the composed call does not pass the swap's mask to the perturbation", [(TRAIN, "out = self.ego.apply(swapped, step, p=ego_p, exclude=sw)", "out = self.ego.apply(swapped, step, p=ego_p)")], "composition_overlap or overlap_guard or glive_a_composed or glive_deliberate or gdvb or ego_exclude or refused or exclusivity"),
    ("M03b_exclusion_ignored_in_perturber", "EgoPerturb.eligible_free ignores the mask (the overlap guard's other half)", [(DATA, "return elig & ~exclude.to(device=elig.device, dtype=torch.bool)", "return elig")], "composition_overlap or overlap_guard or glive_a_composed or glive_deliberate or gdvb or ego_exclude or refused or exclusivity"),
    ("M03c_overlap_guard_deleted", "delete the overlap guard (the raise) -- the exclusion intact", [(TRAIN, "        if self.overlap:\n", "        if False:\n")], "overlap_guard"),
    ("M03d_exclusion_and_guard_both_gone", "the exclusion ignored AND the guard deleted: only the MEASURED overlap can see it", [(DATA, "return elig & ~exclude.to(device=elig.device, dtype=torch.bool)", "return elig"), (TRAIN, "        if self.overlap:\n", "        if False:\n")], "composition_overlap or overlap_guard or glive_a_composed or glive_deliberate or gdvb or ego_exclude or refused or exclusivity"),
    ("M04_loop_call_deleted", "delete `b = comp.apply(b, step)` in the loop", [(TRAIN, "                b = comp.apply(b, step)", "                b = b")], "real_composed_run"),
    ("M05_loop_never_composes", "the loop's `if comp.on:` is never true: a composed argv runs the two original lines", [(TRAIN, "\n            if comp.on:", "\n            if False:")], "real_composed_run"),
    ("M06_compose_probe_not_run", "the composed G-LIVE probe is not run", [(TRAIN, "cmpp = ego_dagger_compose_probe(model, probe, cfg, comp)", 'cmpp = {"pass": True, "overlap": 0, "ego_dagger_compose": "off"}')], "composition_overlap or overlap_guard or glive_a_composed or glive_deliberate or gdvb or ego_exclude or refused or exclusivity"),
    ("M07_gdvb_clause_deleted", "delete the G-DVB read-back (bad_ego_dagger)", [(TRAIN, "    bad += bad_ego_dagger(a, comp, ego, dag) ", "    bad += [] ")], "composition_overlap or overlap_guard or glive_a_composed or glive_deliberate or gdvb or ego_exclude or refused or exclusivity"),
    ("M08_exclusivity_not_lifted", "the exclusivity is not lifted by the flag", [(TRAIN, " and a.ego_perturb_p > 0 and not a.ego_perturb_with_dagger:", " and a.ego_perturb_p > 0:")], "composition_overlap or overlap_guard or glive_a_composed or glive_deliberate or gdvb or ego_exclude or refused or exclusivity"),
    ("M09_exclusivity_always_lifted", "the exclusivity is lifted without the flag", [(TRAIN, " and a.ego_perturb_p > 0 and not a.ego_perturb_with_dagger:", " and a.ego_perturb_p > 0 and a.ego_perturb_with_dagger:")], "composition_overlap or overlap_guard or glive_a_composed or glive_deliberate or gdvb or ego_exclude or refused or exclusivity"),
    ("M10_ego_probe_on_composed_deleted", "the existing ego_perturb_probe is not run on the composed batch", [(TRAIN, 'res["ego_perturb_probe_on_composed"] = ego_perturb_probe(model, swapped, cfg, ego, exclude=sw)', 'res["ego_perturb_probe_on_composed"] = {"pass": True}')], "composition_overlap or overlap_guard or glive_a_composed or glive_deliberate or gdvb or ego_exclude or refused or exclusivity"),
    ("M11_p0_control_is_p1", "the probe's p = 0 control is really p = 1", [(TRAIN, "b_ego0 = comp.apply(batch, step, ego_p=0.0, dag_p=COMPOSE_PROBE_DAG_P)", "b_ego0 = comp.apply(batch, step, ego_p=1.0, dag_p=COMPOSE_PROBE_DAG_P)")], "composition_overlap or overlap_guard or glive_a_composed or glive_deliberate or gdvb or ego_exclude or refused or exclusivity"),
    ("M12_ego_probe_exclusion_dropped", "ego_perturb_probe ignores the exclusion mask (runs on the composed batch as if unmasked)", [(TRAIN, "    pert = ego.apply(batch, ego.PROBE_STEP, p=1.0, **xk)", "    pert = ego.apply(batch, ego.PROBE_STEP, p=1.0)")], "composition_overlap or overlap_guard or glive_a_composed or glive_deliberate or gdvb or ego_exclude or refused or exclusivity"),
    ("M13_off_path_changed", "the flag-off path draws differently (eligible set shrunk when exclude is None)", [(DATA, "        if exclude is None:\n            return elig\n", '        if exclude is None:\n            return elig & (batch["t"] % 7 != 0)\n')], "flag_off_reproduces"),
    ("M14_overlap_not_recorded", "summary.json does not carry the overlap / the flag", [(TRAIN, 'summary["ego_dagger_overlap_n"], summary["ego_perturb_with_dagger"] = ov_tot["n"], True', "pass")], "real_composed_run"),
    ("M15_levers_flag_false", "the selfcheck's levers.ego_perturb_with_dagger is not what was built", [(TRAIN, 'levers["ego_perturb_with_dagger"] = comp.on', 'levers["ego_perturb_with_dagger"] = False')], "glive_a_composed"),
    ("M16_checkpoint_loses_overlap_total", "the composed checkpoint does not carry the overlap total (a resume would lose it)", [(TRAIN, 'blob["ego_perturb_with_dagger"], blob["ego_dagger_overlap"], blob["ego_perturb_stream"] = True, dict(ov_tot), ego.stream', "pass")], "real_composed_run"),
    # ---- 2026-10-28 compose_fix: the perturbation's stream is independent of the swap's (the box-2 defect). BEHAVIOURAL kills (NEWB = the tests that MEASURE the draws) matter most for the first three.
    ("M17a_stream_not_set_by_the_stage", "STREAM SEPARATION REMOVED: the composed stage does not set the perturbation's stream (the perturber stays on the swap's formula)", [(TRAIN, "self.ego.stream = D.EGO_PERTURB_COMPOSED_STREAM", "self.ego.stream = 0")], "equal_seeds_every_sample or selection_uniforms_are_not_one_stream or box2_shape"),
    ("M17b_stream_set_and_recorded_but_not_mixed", "STREAM SEPARATION REMOVED, RECORD INTACT: the stream is set, gated and recorded as 1, but _seed_of ignores it (the 'recorded but dead' form: only the draws can see it)", [(DATA, "        if not self.stream:\n            return base", "        if True:\n            return base")], "equal_seeds_every_sample or selection_uniforms_are_not_one_stream or box2_shape"),
    ("M17c_stream_constant_zeroed", "the composed stream constant is 0: the stage, the gate (needs 0) and the record all agree on 'no separation'", [(DATA, "EGO_PERTURB_COMPOSED_STREAM = 1\n", "EGO_PERTURB_COMPOSED_STREAM = 0\n")], "equal_seeds_every_sample or selection_uniforms_are_not_one_stream or box2_shape or seed_of_stream_zero"),
    ("M17d_stream_ignores_the_step", "the stream's seed ignores the step (every step draws the same uniforms)", [(DATA, 'f"ego_perturb_stream/{self.stream}/{base}"', 'f"ego_perturb_stream/{self.stream}/{self.seed}"')], "seed_of_stream_zero"),
    ("M17e_stream_not_recorded", "levers.ego_perturb_stream is not what was built", [(TRAIN, 'levers["ego_perturb_stream"] = comp.ego.stream', 'levers["ego_perturb_stream"] = 0')], "run_record_names_the_perturbation_stream"),
    ("M17f_gdvb_stream_readback_deleted", "delete the G-DVB read-back of the perturbation's stream", [(TRAIN, "    if comp.ego.stream != want_stream:", "    if False:")], "not_on_the_declared_stream"),
    ("M17g_resume_guard_deleted", "delete the resume guard: a pre-separation composed checkpoint continues on another generator", [(TRAIN, 'if comp.on and int(ck.get("ego_perturb_stream", 0)) != ego.stream:', "if False:")], "not_resumed_on_another_generator"),
    ("M17h_stream_not_in_checkpoint", "the composed checkpoint does not carry the stream (the resume guard then reads 0)", [(TRAIN, 'blob["ego_perturb_with_dagger"], blob["ego_dagger_overlap"], blob["ego_perturb_stream"] = True, dict(ov_tot), ego.stream', 'blob["ego_perturb_with_dagger"], blob["ego_dagger_overlap"], blob["ego_perturb_stream"] = True, dict(ov_tot), 0')], "run_record_names_the_perturbation_stream"),
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def slot():
    import random
    for _ in range(720):
        n = subprocess.run(["C:/Users/Admin/venvs/tanitad/Scripts/python.exe", "C:/Users/Admin/orcread/pytest_runs.py"], capture_output=True, text=True).stdout.strip()
        if n.isdigit() and int(n) < 2:
            time.sleep(random.uniform(3, 12))                                                          # a jittered second look: two pipelines must not both see a free slot at once
            n = subprocess.run(["C:/Users/Admin/venvs/tanitad/Scripts/python.exe", "C:/Users/Admin/orcread/pytest_runs.py"], capture_output=True, text=True).stdout.strip()
            if n.isdigit() and int(n) < 2:
                return
        time.sleep(30)
    raise SystemExit("ZZNO-SLOT")


def run(k=""):
    slot()
    import os
    cmd = [PY, "-m", "pytest", TEST, "-q", "-x", "-p", "no:cacheprovider", "--basetemp=F:/TanitAD-scratch/compose_fix/tmp/btm", "-W", "ignore"] + (["-k", k] if k else [])
    subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", "F:\\TanitAD-scratch\\compose_fix\\tmp\\btm"], capture_output=True)
    os.makedirs("F:/TanitAD-scratch/compose_fix/tmp/btm", exist_ok=True)
    r = subprocess.run(cmd, cwd=TREE, capture_output=True, text=True, env={**os.environ, **ENV}, encoding="utf-8", errors="replace")
    tail = r.stdout.strip().splitlines()
    failed = [l for l in tail if l.startswith("FAILED") or l.startswith("ERROR")]
    return r.returncode, (failed[0] if failed else ""), (tail[-1] if tail else "")


def main():
    base = {f: sha(f"{TREE}/{f}") for f in (TRAIN, DATA, TEST)}
    out = open(OUT, "w", encoding="utf-8", newline="\n")
    w = lambda s: (out.write(s + "\n"), out.flush())  # noqa: E731
    w(f"MUTATION PROOF  {time.strftime('%Y-%m-%d %H:%M:%S')}  tree={TREE}")
    w(f"tests: {TEST}  sha256 {base[TEST][:16]}  (the unmutated trainer {base[TRAIN][:16]}, data lib {base[DATA][:16]})")
    for k in sorted({m[3] for m in M}):                                                                # a control per distinct selection: the selection itself must be GREEN unmutated
        rc, failed, last = run(k)
        w(f"CONTROL (no mutation) -k {k!r}: rc={rc}  [{last}]   <- must be 0 (GREEN)" + (f"  FIRST FAILING: {failed}" if rc else ""))
        if rc:
            w("ABORT: a control is not green")
            out.close()
            raise SystemExit(2)
    survivors = []
    for name, rule, edits, k in M:
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
                open(f"{TREE}/{f}", "w", encoding="utf-8", newline="").write(open(f"F:/TanitAD-scratch/compose_fix/mutpristine/{f.replace('/', '__')}", encoding="utf-8", newline="").read())
        assert all(sha(f"{TREE}/{f}") == h for f, h in base.items()), "restore failed"
        verdict = "RED (killed)" if rc != 0 else "GREEN  <-- SURVIVED"
        if rc == 0:
            survivors.append(name)
        w(f"{name}: {verdict}  rule: {rule}\n    first failing: {failed or '-'}   [{last}]")
    w(f"SUMMARY: {len(M) - len(survivors)}/{len(M)} mutants killed; survivors: {survivors or 'none'}")
    out.close()


main()
