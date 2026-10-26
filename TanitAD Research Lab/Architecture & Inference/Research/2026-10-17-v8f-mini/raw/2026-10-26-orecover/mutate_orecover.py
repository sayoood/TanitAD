"""Mutation proof for O-RECOVER's instruments (PREREG_V8F_HACC_MC section 7): every mutation must turn the named tests of stack/tests/test_v8f_ego_perturb.py RED; the unmutated tree must be GREEN before and after,
and every file is restored byte-identical.

    python mutate_orecover.py <repo root> <python exe> [--list] [--only <substring of a mutant's name>]

Each mutation is one (or, where stated, two) exact substring(s) replaced once in ONE file; a mutation whose anchor is not found exactly once aborts the run. Each mutant carries the selectors (pytest -k) of the
tests that must go RED; every selector is run on its own, and the mutant is RED only if ALL of them fail.
    (b)  the batch    -k "p1_changes_every or rebuild_equals"
    (c)  the loss     -k "perturbed_input_reaches or launch_probe_passes"
    (e)  the caller   -k "real_run_applies"
    (f)  --init-ckpt  -k "init_ckpt_starts"
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.argv[2]
LIST = "--list" in sys.argv
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
DATA = ROOT / "stack" / "tanitad" / "data" / "v8f_mini_data.py"
TRAIN = ROOT / "stack" / "scripts" / "v8f_mini_train.py"
TEST = "stack/tests/test_v8f_ego_perturb.py"
SEL_B = "p1_changes_every or rebuild_equals"
SEL_C = "perturbed_input_reaches or launch_probe_passes"
SEL_E = "real_run_applies"
SEL_F = "init_ckpt_starts"
SEL_NAV = "nav_progress_is_rebuilt or nav_rebuild_on_the_unmodified"
SEL_WARM = "g_dvb_reads_warmup or real_run_builds_the_declared_warmup"

MUT = [
    # ---- THE single applying line (the brief's mutation): (b) and (c) go RED
    (DATA, "the line that applies the perturbation is deleted (data module): `out` is `batch`",
     [("            out.update(self._rebuilt(batch, rows, s[r].to(dev), d[r].to(dev), n[r].to(dev)))         # the ONE line that applies the perturbation: delete it and `out` is `batch`\n",
       "            pass\n")], [SEL_B, SEL_C, SEL_E]),
    # ---- the caller
    (TRAIN, "the loop does not apply the perturbation to the batch the loss receives",
     [("            b = ego.apply(b, step)                                 # O-RECOVER:", "            b = b                                                  # O-RECOVER:")], [SEL_E]),
    # ---- G-DVB / reachability
    (TRAIN, "make_ego_perturb drops --ego-perturb-adrop (the built law is not the declared one)",
     [("adrop=a.ego_perturb_adrop,", "")], ["g_dvb_passes_what_was_built_and_refuses or launch_probe_passes"]),
    (TRAIN, "declared_vs_built does not read the ego perturber back",
     [("    bad += bad_ego_perturb(a, ego) + bad_init_ckpt(a, data_info)", "    bad += bad_init_ckpt(a, data_info)")], ["g_dvb_passes_what_was_built_and_refuses or no_perturber_built or refused_declaration or illegal_or_empty"]),
    (TRAIN, "the flag has no declared-vs-built entry",
     [(' "ego_perturb_seed": "ego:ego_perturb_seed",', '')], ["every_new_flag_has"]),
    (TRAIN, "the probe is faked (never exercises the perturbed batch)",
     [("        egop = ego_perturb_probe(model, probe, cfg, ego)", '        egop = {"pass": True}')], [SEL_E]),
    (DATA, "eligibility is v0 <= vmax (not strictly below)",
     [('(batch["kin3"][:, 0] < self.vmax)', '(batch["kin3"][:, 0] <= self.vmax)')], ["vmax_is_strict"]),
    (DATA, "the acceleration drop is not applied (a - 0)", [("v2, a2, om2 = s_ * v, a - d[:, None], s_ * om + n_", "v2, a2, om2 = s_ * v, a, s_ * om + n_")], [SEL_B]),
    (DATA, "the base is not rebuilt", [('new["base"][rows] = torch.from_numpy(', 'new["base"][rows] = batch["base"][rows] + 0 * torch.from_numpy(')], [SEL_B, SEL_C]),
    (DATA, "the odometry is not rebuilt (pose_t stays true)", [('new["pose_t"][rows] = pose2[k, F - 1 - batch["a_t"][rows]]', 'new["pose_t"][rows] = batch["pose_t"][rows]')], [SEL_B]),
    (DATA, "the draws use the global RNG (a resumed run would draw something else)", [("g = torch.Generator().manual_seed(self._seed_of(step))", "g = torch.default_generator")], ["draw_is_a_function"]),
    # ---- the control counts
    (TRAIN, "the perturbed count is not accumulated", [('                c_["n"] += ego.last["n_selected"]\n', "")], [SEL_E, "log_interval"]),
    (TRAIN, "the summary reports the eligible count as the perturbed one", [('"ego_perturb_n": ego_tot["n"], ', '"ego_perturb_n": ego_tot["eligible"], ')], [SEL_E]),
    # ---- nav: the route progress carries the travelled distance (coordinator fix 1)
    (DATA, "the nav rebuild is dropped: nav[:, 3] keeps the TRUE travelled distance", [('new["nav"][rows, :, 3] = torch.where(nav_ok, nav3.to(nav_r.dtype), nav_r[..., 3])', "pass")], [SEL_NAV, SEL_E]),
    (DATA, "nav is rebuilt across a manoeuvre boundary / validity change too (the token test is dropped)", [("        return vm & same, vm & ~same", "        return vm, vm & ~vm")], [SEL_NAV]),
    (TRAIN, "the nav_kept count is not accumulated", [('c_["nav_kept"] += ego.last["ego_perturb_nav_kept"]', "pass")], [SEL_E]),
    # ---- --warmup read back from the built schedule (coordinator fix 2)
    (TRAIN, "the schedule is built with the warmup ignored", [("LRSchedule(a.lr, a.warmup, total)", "LRSchedule(a.lr, 1, total)")], [SEL_WARM, "planned_finetune_argv"]),
    (TRAIN, "declared_vs_built does not read the schedule back", [(" + bad_schedule(a, sched)", "")], [SEL_WARM]),
    # ---- --init-ckpt (f)
    (TRAIN, "the load line is dropped: the run starts from scratch (f)",
     [("    model.load_state_dict(sd, strict=True)\n    cur = model.state_dict()", "    cur = model.state_dict()")], [SEL_F]),
    (TRAIN, "the config clause is dropped: a config the strict load cannot see is accepted (g)",
     [("    if diff:\n        raise ValueError(\"config mismatch", "    if False:\n        raise ValueError(\"config mismatch")], ["config_mismatch"]),
]

if LIST:
    for i, (p, n, e, sel) in enumerate(MUT):
        print(i, p.name, "|", n, "|", sel)
    sys.exit(0)

orig = {p: p.read_bytes() for p in {m[0] for m in MUT}}
bad_anchor = 0
for path, name, edits, sels in MUT:
    for a, b in edits:
        n = orig[path].count(a.encode("utf-8"))
        if n != 1:
            bad_anchor += 1
            print(f"ANCHOR FOUND {n}x (need exactly 1): [{path.name}] {name}\n    {a!r}", flush=True)
if bad_anchor:
    sys.exit(3)
print(f"{len(MUT)} mutants over {len(orig)} files, each anchor found exactly once", flush=True)

import os  # noqa: E402
BASE = Path(os.environ.get("ORECOVER_MUT_BASETEMP") or tempfile.gettempdir()) / "pt_orecover_mut"          # set it to a roomy drive: the tests write small synthetic worlds
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", **({"TEMP": str(BASE.parent), "TMP": str(BASE.parent)} if os.environ.get("ORECOVER_MUT_BASETEMP") else {})}


def run(sel):
    cmd = [PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp", str(BASE), TEST, "-k", sel]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=ENV, timeout=3600)
    lines = [x for x in r.stdout.strip().splitlines() if x.startswith(("FAILED", "ERROR")) or " passed" in x or " failed" in x or " error" in x]
    return r.returncode, " | ".join(x[:200] for x in (lines[-2:] if lines else [(r.stdout.strip().splitlines() or [r.stderr.strip()[-170:]])[-1]]))


def restore():
    for p, b in orig.items():
        p.write_bytes(b)


all_sels = sorted({s for m in MUT for s in m[3]})
for s in all_sels:
    rc, tail = run(s)
    print(f"baseline [{s}]:", rc, tail, flush=True)
    assert rc == 0, "the unmutated tree must be green"
red, survived, ran = 0, [], 0
try:
    for path, name, edits, sels in MUT:
        if ONLY and ONLY not in name and ONLY not in path.name:
            continue
        ran += 1
        src = orig[path]
        for a, b in edits:
            src = src.replace(a.encode("utf-8"), b.encode("utf-8"), 1)
        path.write_bytes(src)
        try:
            res = [(s, *run(s)) for s in sels]
        finally:
            path.write_bytes(orig[path])
        ok = all(rc != 0 for _, rc, _ in res)
        red += ok
        if not ok:
            survived.append(name)
        print(f"{'RED     ' if ok else 'SURVIVED'} [{path.name}] {name}", flush=True)
        for s, rc, tail in res:
            print(f"      [{s}] rc={rc}: {tail}", flush=True)
finally:
    restore()
for s in all_sels:
    rc, tail = run(s)
    print(f"restored [{s}]:", rc, tail, flush=True)
print("byte-identical:", all(p.read_bytes() == b for p, b in orig.items()))
print(f"{red}/{ran} RED" + (f"  SURVIVED: {survived}" if survived else ""))
sys.exit(0 if red == ran else 1)
