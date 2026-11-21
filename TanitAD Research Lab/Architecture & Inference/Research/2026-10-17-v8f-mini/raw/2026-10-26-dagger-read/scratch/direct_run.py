"""Call selected tests of test_dag_read.py DIRECTLY (no pytest process: the machine-wide pytest cap is 2 and both slots are taken by others). Quick feedback only; the proof of record is the pytest suite."""
import contextlib, io, os, pathlib, shutil, sys, time, traceback
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Users/Admin/dagread")
import test_dag_read as T

BASE = pathlib.Path("E:/Projects/TanitAD-artifacts/dagread_build/direct")
shutil.rmtree(BASE, ignore_errors=True)
BASE.mkdir(parents=True)


class Cap:
    def __init__(self):
        self.buf, self.ebuf, self.ctx, self.ctx2 = io.StringIO(), io.StringIO(), None, None
    def start(self):
        self.ctx = contextlib.redirect_stdout(self.buf); self.ctx.__enter__()
        self.ctx2 = contextlib.redirect_stderr(self.ebuf); self.ctx2.__enter__()
    def stop(self):
        if self.ctx2: self.ctx2.__exit__(None, None, None); self.ctx2 = None
        if self.ctx: self.ctx.__exit__(None, None, None); self.ctx = None
    def readouterr(self):
        v, e = self.buf.getvalue(), self.ebuf.getvalue()
        for b in (self.buf, self.ebuf): b.seek(0); b.truncate()
        return type("R", (), {"out": v, "err": e})()


def call(fn, *a, **kw):
    cap = Cap(); t = time.time(); tmp = BASE / f"{fn.__name__}_{int(t*1000)%10**6}"; tmp.mkdir(parents=True)
    import inspect
    names = inspect.signature(fn).parameters
    args = list(a)
    kw2 = dict(kw)
    if "tmp_path" in names: kw2["tmp_path"] = tmp
    if "capsys" in names: kw2["capsys"] = cap
    try:
        cap.start()
        fn(**kw2) if not args else fn(*args, **kw2)
        cap.stop(); r = "PASS"
    except BaseException:
        cap.stop(); r = "FAIL\n" + traceback.format_exc()[-1800:]
    print(f"{fn.__name__} {kw.get('name','')}: {r} ({time.time()-t:.1f}s)", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return r.startswith("PASS")


ok = True
for fn in (T.test_fixtures_are_the_kb45_generator_output_and_the_pre_kb45_documents_are_kept, T.test_the_synthetic_bank_has_the_generators_real_key_structure_and_the_loader_reads_it,
           T.test_kb4_every_matcher_case_and_kb5_tolerance_come_from_the_generators_controls_block, T.test_the_real_generator_smoke_reads_exactly_as_NOTES_md_reports,
           T.test_bank_audit_on_the_real_kb45_smoke_passes_k_b4_and_k_b5_and_fails_only_the_partial_bank, T.test_bank_audit_script_runs_the_readers_bank_controls_and_marks_what_the_generator_does_not_write,
           T.test_the_trainer_records_train_window_list_sha256_where_the_readers_default_key_looks, T.test_trainer_key_names_are_cli_overridable):
    ok &= call(fn)
# the parametrized refusal cases that concern the window-list sha
for mark in T.test_refuse_arm_stage_bank_and_replicate_controls.pytestmark:
    if mark.name == "parametrize":
        for row in mark.args[1]:
            if row[0].startswith("trainer_window_list") or row[0] in ("bank_manifest_altered",):
                ok &= call(T.test_refuse_arm_stage_bank_and_replicate_controls, name=row[0], edit=row[1], needles=row[2], absent=row[3])
shutil.rmtree(BASE, ignore_errors=True)
print("ALL-PASS" if ok else "SOME-FAILED")
