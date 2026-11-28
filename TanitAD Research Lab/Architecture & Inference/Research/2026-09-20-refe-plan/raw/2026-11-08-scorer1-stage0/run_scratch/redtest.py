import subprocess, sys, importlib.util
import common_args as C
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
spec = importlib.util.spec_from_file_location("cs", "chain_stage0.py"); cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)
e = cs.env(); e["CUDA_VISIBLE_DEVICES"] = "-1"
refe = str(cs.Path(cs.PKG) / "refe")
base = C.argv(False)
def run(tag, args):
    p = subprocess.run([PY, "-u", "scorer_finetune.py"] + args, cwd=refe, env=e, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=200, stdin=subprocess.DEVNULL)
    last = [l for l in p.stdout.splitlines() if l.strip()][-1:] or [p.stderr[-200:]]
    print(tag, "rc", p.returncode, last[0][:140])
noflag = [x for x in base if x != "--dump-step0-only"]
run("no_flag_no_preflight_pass(expect REFUSED rc4)", noflag)
badlr = list(base); badlr[badlr.index("--lr") + 1] = "1e-4"
run("flag_with_lr_nonzero(expect REFUSED rc4)", badlr)
