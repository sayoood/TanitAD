"""Mutation check of the GREEN unit test: each mutated COPY of the loader must turn
`test_the_loader_rebuilds_the_trainers_refcv7_model_bit_for_bit` RED; the unmutated copy must pass.
Runs each arm in a fresh subprocess. Never touches the real loader file."""
import json
import subprocess
import sys
from pathlib import Path

TREE = Path("C:/lgt/r7ldr")
SRC = TREE / "stack/tanitad/eval/refcv7_loader.py"
WORK = Path("C:/lgt/r7ldr_scratch/mut")
WORK.mkdir(parents=True, exist_ok=True)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"

MUTATIONS = {
    "control_unmutated": [],
    "M1_query_select_dropped": [(
        '''    _pcfg = tr._dc.replace(_pcfg, **tr._slot_refine_kwargs(args),
                           query_select=str(getattr(args, "slot_query_select", "learned")
                                            or "learned"))''',
        '''    _pcfg = tr._dc.replace(_pcfg, **tr._slot_refine_kwargs(args))''')],
    "M2_hires_lift_bank_without_equalize_rows": [(
        '''    model._lift_bank_hires = _mhr.HiresLiftGeometryBank(
        _htable, frame=_perc.frame_for_model(model), cfg=_hcfg,
        equalize_bottom_rows=int(getattr(args, "equalize_bottom_rows", 0) or 0))''',
        '''    model._lift_bank_hires = _mhr.HiresLiftGeometryBank(
        _htable, frame=_perc.frame_for_model(model), cfg=_hcfg,
        equalize_bottom_rows=0)''')],
    "M3_vis1_not_set": [(
        '''    model._vis1 = bool(getattr(args, "slot_vis1", False))''',
        '''    pass''')],
    "M4_tau_sha_not_stamped": [(
        '''        cfg.nav_compliance_tau_sha256 = _navc["sha256"]''',
        '''        pass''')],
    "M5_decision_rule_raw": [(
        '''        decision_rule=str(getattr(args, "map_hires_decision_rule", _mhr.DECISION_RULES[0])))''',
        '''        decision_rule="raw")''')],
    "M6_trunk_rows_not_as_trained": [(
        '''    enc.trunk_equalize_bottom_rows = int(rows)''',
        '''    enc.trunk_equalize_bottom_rows = 0''')],
    "M7_map_hires_class_weight_uniform": [(
        '''    model._map_hires_class_weight = _hcw.to(device)                     # train():8030''',
        '''    model._map_hires_class_weight = (_hcw * 0 + 1).to(device)''')],
    # the SAME two defects with the loader's own G-DVB self-check neutralised: the GREEN test's
    # forward comparison must see them alone (they change no state_dict tensor)
    "M2b_equalize_rows_0_and_no_dvb": [(
        '''    model._lift_bank_hires = _mhr.HiresLiftGeometryBank(
        _htable, frame=_perc.frame_for_model(model), cfg=_hcfg,
        equalize_bottom_rows=int(getattr(args, "equalize_bottom_rows", 0) or 0))''',
        '''    model._lift_bank_hires = _mhr.HiresLiftGeometryBank(
        _htable, frame=_perc.frame_for_model(model), cfg=_hcfg,
        equalize_bottom_rows=0)'''),
        ("    _bad = list(_dvb.check(model, args, tr.build_parser()))", "    _bad = []")],
    "M6b_trunk_rows_0_and_no_dvb": [(
        '''    enc.trunk_equalize_bottom_rows = int(rows)''',
        '''    enc.trunk_equalize_bottom_rows = 0'''),
        ("    _bad = list(_dvb.check(model, args, tr.build_parser()))", "    _bad = []")],
    "M6c_trunk_rows_0_no_dvb_no_stamp_check": [(
        '''    enc.trunk_equalize_bottom_rows = int(rows)''',
        '''    enc.trunk_equalize_bottom_rows = 0'''),
        ("    _bad = list(_dvb.check(model, args, tr.build_parser()))", "    _bad = []"),
        ('    rec["stamp_checks"] = stamp_checks(model, config, rec, tr)',
         '    rec["stamp_checks"] = {}')],
}

RUNNER = r'''
import sys, types, os
from pathlib import Path
sys.path.insert(0, "C:/lgt/r7ldr/stack"); sys.path.insert(0, "C:/lgt/r7ldr/stack/scripts")
sys.path.insert(0, "C:/lgt/r7ldr/stack/tests")
import pytest, torch
import test_refcv7_eval_loader as TM
LG = TM.LG
mut = Path(sys.argv[1]); d = Path(sys.argv[2]); d.mkdir(parents=True, exist_ok=True)
os.environ["REFCV6_REPO"] = "C:/lgt/r7ldr"
TM._synth_inputs(d); argv = TM._argv(d)
T = LG.load_trainer(types.SimpleNamespace(tree="C:/lgt/r7ldr",
                                          prof={"trainer": "stack/scripts/refc_v3_train.py"}))
cap = LG.run_trainer_until(T, LG.set_flag(argv, "--trunk-compile", None), "model")
ck = d / "ckpt.pt"; torch.save({"model": cap["model"].state_dict(), "step": 0}, ck)
L = TM._load_module("refcv7_loader_mutant", mut)
rig = types.SimpleNamespace(d=d, argv=argv, T=T, model_t=cap["model"], args_t=cap["args"], ck=ck, L=L)
try:
    TM.test_the_loader_rebuilds_the_trainers_refcv7_model_bit_for_bit(rig)
    print("ZZRESULT=GREENZZ")
except BaseException as e:
    import traceback
    tb = [f for f in traceback.extract_tb(e.__traceback__)
          if f.filename.endswith("test_refcv7_eval_loader.py")]
    where = ("test line %d: %s" % (tb[-1].lineno, tb[-1].line)) if tb else ""
    print("ZZRESULT=REDZZ", type(e).__name__, str(e).replace("\n", " ")[:400], "|", where)
'''


def main():
    (WORK / "runner.py").write_text(RUNNER, encoding="utf-8")
    src = SRC.read_text(encoding="utf-8")
    out = {}
    only = sys.argv[1:]
    for name, edits in MUTATIONS.items():
        if only and name not in only:
            continue
        s = src
        for old, new in edits:
            assert s.count(old) == 1, (name, old[:60], s.count(old))
            s = s.replace(old, new)
        mp = WORK / f"{name}.py"
        mp.write_text(s, encoding="utf-8", newline="\n")
        env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4",
                   PYTHONIOENCODING="utf-8",
                   PYTHONPATH="C:/lgt/r7ldr/stack;C:/lgt/r7ldr/stack/scripts;C:/lgt/r7ldr/taniteval")
        r = subprocess.run([PY, str(WORK / "runner.py"), str(mp), str(WORK / f"rig_{name}")],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=900)
        line = [x for x in (r.stdout + r.stderr).splitlines() if "ZZRESULT=" in x]
        out[name] = line[-1] if line else ("NO RESULT LINE rc=%d " % r.returncode
                                          + (r.stdout + r.stderr)[-600:])
        print(name, "->", out[name][:300], flush=True)
    json.dump(out, open(WORK / "mutation_results.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
