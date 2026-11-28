"""Dry-run stand-in for optb_hfbank.py: copies the bank dir to $OPTB_HF_STUB_DIR/<repo>/<tag> and compares sha256 file by file. No network."""
import argparse, hashlib, os, shutil, sys
ap = argparse.ArgumentParser(); ap.add_argument("--tag"); ap.add_argument("--bank-dir"); ap.add_argument("--repo"); ap.add_argument("--keys", default="")
a = ap.parse_args()
dst = os.path.join(os.environ["OPTB_HF_STUB_DIR"], a.repo.replace("/", "_"), a.tag)
shutil.copytree(a.bank_dir, dst, dirs_exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
bad = [f for f in os.listdir(a.bank_dir) if sha(os.path.join(a.bank_dir, f)) != sha(os.path.join(dst, f))]
if bad:
    print("VERIFY-FAIL", bad, file=sys.stderr); sys.exit(5)
print("ZZ OPTB-HF-OK " + a.tag)
