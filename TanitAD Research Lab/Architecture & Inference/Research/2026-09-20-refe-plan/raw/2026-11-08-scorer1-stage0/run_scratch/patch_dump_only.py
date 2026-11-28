"""Scratch-only: an eval-only exit for SCORER1 Stage 0's logit dump. Never lands from here; the diff is kept beside the MM note."""
import difflib
F = "F:/tanitad_scratch/stage0_20261108/code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/scorer_finetune.py"
b = open(F, "rb").read()
orig = b
eol = b"\r\n" if b"\r\n" in b[:4000] else b"\n"
def sub(old, new):
    global b
    assert b.count(old) == 1, (old[:60], b.count(old))
    b = b.replace(old, new)
A = b'    ap.add_argument("--dump-scores", default="",'
sub(A, b'    ap.add_argument("--dump-step0-only", action="store_true", help="eval-only: write the step-0 --dump-scores files and exit before any optimizer step (needs --lr 0)")' + eol + A)
sub(b'    if not a.preflight and not os.path.exists(os.path.join(a.out, "PREFLIGHT_PASS.json")):',
    b'    if a.dump_step0_only and (a.preflight or a.lr != 0 or not a.dump_scores):' + eol +
    b'        print("REFUSED: --dump-step0-only needs --lr 0 and --dump-scores, and not --preflight"); return 4' + eol +
    b'    if not a.preflight and not a.dump_step0_only and not os.path.exists(os.path.join(a.out, "PREFLIGHT_PASS.json")):')
C = b'    run_evaluate("step0")' + eol + b'    upd, micro, t0 = 0, 0, time.time()'
sub(C, b'    run_evaluate("step0")' + eol + b'    if a.dump_step0_only:' + eol + b'        log({"event": "dump_step0_only_exit"}); return 0' + eol + b'    upd, micro, t0 = 0, 0, time.time()')
open(F, "wb").write(b)
d = difflib.unified_diff(orig.decode("utf-8").splitlines(), b.decode("utf-8").splitlines(), "a/scorer_finetune.py", "b/scorer_finetune.py", lineterm="", n=1)
open("E:/Projects/TanitAD-artifacts/scorer1_impl_20261108/stage0_pull/dump_only.diff", "w", encoding="utf-8", newline="\n").write("\n".join(d) + "\n")

P = "F:/tanitad_scratch/stage0_20261108/common_args.py"
s = open(P, "rb").read()
old = b'else: a += ["--dump-scores", DUMP]'
assert s.count(old) == 1
open(P, "wb").write(s.replace(old, b'else: a += ["--dump-scores", DUMP, "--dump-step0-only"]'))

P = "F:/tanitad_scratch/stage0_20261108/chain_stage0.py"
s = open(P, "rb").read()
old = b'need_pre = not (Path(C.OUT) / "PREFLIGHT_PASS.json").exists()'
assert s.count(old) == 1
open(P, "wb").write(s.replace(old, b'need_pre = False  # MM 2026-11-08: eval-only dump, --dump-step0-only; see MM_RULING_STAGE0_INPUTS.md'))
print("ZZPATCH_OK")
