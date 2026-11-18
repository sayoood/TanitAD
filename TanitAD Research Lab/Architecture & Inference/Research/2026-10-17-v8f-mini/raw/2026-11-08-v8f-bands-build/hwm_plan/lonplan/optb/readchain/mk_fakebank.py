"""mk_fakebank.py <bank_dir> <src_ckpt> <fixture_dir_for_logs> [--tamper]: dress a checkpoint as a bank (RUN.done with sha256 evidence, BANK_MANIFEST, BANK.done). Used by the dry run and the tests."""
import hashlib, json, os, shutil, sys
bd, ck, fx = sys.argv[1:4]; tamper = "--tamper" in sys.argv
os.makedirs(bd, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for f in ("summary.json", "train_log.jsonl", "selfcheck.json"):
    shutil.copy(os.path.join(fx, f), os.path.join(bd, f))
shutil.copy(ck, os.path.join(bd, "ckpt_last.pt"))
# MM 2026-11-18: a real OPTB bank carries argv.txt (stages/optb_run.sh:59); the BAND-MET read (lr_lib.load_arm) needs it. The fake bank takes the source run's own argv.txt.
if os.path.isfile(os.path.join(os.path.dirname(ck), "argv.txt")):   # MM 2026-11-18: the unit-test fixture (fixture/FT) has no argv.txt and needs none (no BAND-MET read); an unconditional copy failed 3 tests
    shutil.copy(os.path.join(os.path.dirname(ck), "argv.txt"), os.path.join(bd, "argv.txt"))
ev = {"ckpt_sha256": sha(os.path.join(bd, "ckpt_last.pt")), "summary_sha256": sha(os.path.join(bd, "summary.json")), "train_log_sha256": sha(os.path.join(bd, "train_log.jsonl"))}
json.dump({"tag": os.path.basename(bd), "evidence": ev}, open(os.path.join(bd, "RUN.done"), "w"))
files = {f: {"bytes": os.path.getsize(os.path.join(bd, f)), "sha256": sha(os.path.join(bd, f))} for f in sorted(os.listdir(bd)) if f not in ("BANK_MANIFEST.json", "BANK.done")}
json.dump({"tag": os.path.basename(bd), "files": files}, open(os.path.join(bd, "BANK_MANIFEST.json"), "w"))
open(os.path.join(bd, "BANK.done"), "w").write("fake bank (dry run)\n")
if tamper:   # one flipped byte AFTER the manifest was written: the read chain must refuse
    p = os.path.join(bd, "ckpt_last.pt"); b = bytearray(open(p, "rb").read()); b[100] ^= 1; open(p, "wb").write(bytes(b))
