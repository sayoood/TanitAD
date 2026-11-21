"""ds_pull.py -- the pull + verify step of the DSCALE10K block-1 read.  Usage: python ds_pull.py [--source auto|hf|pod] [--bank-dir DIR] [--items B_s0.A U_s0.A ...] [--check-only]
For each block-1 checkpoint (B_s0, U_s0, B_rep stages A and B; R_s0 stage A) it
  1. finds the EXPECTED sha256 from the banker's record (never from the file it pulls): `<bank-dir>/<arm>.<stage>.json` {sha256, hf_repo, hf_path[, hf_rev]} (the HF bank, preferred), else the pod's
     `bank_ok/<arm>.<stage>.sha256` which must equal `bank_ready/<arm>.<stage>.txt` (path, sha256) and the pulled file's hash;
  2. pulls ckpt_last.pt (87 MB each) and the stage's summary.json to ckpts/<arm>.<stage>/ from HF (HF_HUB_DISABLE_XET=1; token read in place from the keys file, never printed) or from the pod
     (pscp; reading files only, never the GPU jobs or the lock; no pod address is written anywhere);
  3. hashes the local file and REFUSES (deletes the pulled copy, writes nothing) on a mismatch; a match writes BANK_RECORD.json {arm, stage, sha256, source, commit, steps}.
A finished item (record present and ds_lib.bank_problem is None) is skipped.  F: must hold >= 25 GB free (the card's bar).  Exit 0 only when all requested items verify."""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_lib as L  # noqa: E402

GIT_BASH = "C:/Program Files/Git/usr/bin/bash.exe"      # never the bare name "bash": from Python on this box that is the System32 WSL bash (rc 1, empty stderr)
FLT = re.compile(r"post-quantum|store now|upgraded")
SHA = re.compile(r"^[0-9a-f]{64}$")


def expected_sha(arm, stage, bank_dir, pod_text):
    """-> (sha256, source, info) or (None, None, reason).  pod_text(path) -> text or None (injectable for the tests)."""
    rec = os.path.join(bank_dir or "", f"{arm}.{stage}.json")
    if bank_dir and os.path.isfile(rec):
        r = json.load(open(rec, encoding="utf-8"))
        if SHA.match(str(r.get("sha256", ""))) and r.get("hf_repo") and r.get("hf_path"):
            return r["sha256"], "hf", r
        return None, None, f"the banker record {rec} lacks sha256 / hf_repo / hf_path"
    ok = (pod_text(f"{L.POD_ROOT}/bank_ok/{arm}.{stage}.sha256") or "").strip().split()
    ready = (pod_text(f"{L.POD_ROOT}/bank_ready/{arm}.{stage}.txt") or "").strip().split()
    if not ok or not ready:
        return None, None, f"{arm}.{stage} is not banked (no banker record, no bank_ok / bank_ready on the pod)"
    if not SHA.match(ok[0]) or len(ready) < 2 or ready[1] != ok[0]:
        return None, None, f"bank_ok ({ok[:1]}) and bank_ready ({ready[1:2]}) disagree for {arm}.{stage}"
    return ok[0], "pod", {"pod_path": f"{L.POD_ROOT}/runs/{arm}/{stage}/ckpt_last.pt"}


def verify_and_record(arm, stage, d, want, source, steps):
    """hash the pulled checkpoint; mismatch -> remove it and return the reason; match -> BANK_RECORD.json"""
    p = f"{d}/ckpt_last.pt"
    got = L.sha256_file(p)
    if got != want:
        os.remove(p)
        return f"REFUSED: {arm}.{stage} pulled sha256 {got[:12]} != the banked {want[:12]}; the pulled copy was removed"
    json.dump({"arm": arm, "stage": stage, "sha256": want, "source": source, "commit": L.COMMIT, "steps": steps, "seed": L.ARMS[arm]}, open(f"{d}/BANK_RECORD.json", "w"), indent=1)
    return None


def _pod(cmd, timeout=120):
    r = subprocess.run([GIT_BASH, "-c", f"source C:/Users/Admin/lorapod/podenv.sh; {cmd}"], capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)
    return "\n".join(x for x in (r.stdout or "").splitlines() if not FLT.search(x)), r.returncode


def pod_text(path):
    out, rc = _pod(f'pssh "cat {path}"')
    return out if rc == 0 and out.strip() else None


def pod_get(remote, local):
    d = os.path.dirname(local)
    posix = local.replace("\\", "/")
    out, rc = _pod(f'pscp "$POD_TARGET:{remote}" "{posix}"', 900)
    return rc == 0 and os.path.isfile(local)


def hf_get(r, local_dir):
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        kp = os.environ.get("DS_KEYS", "E:/Projects/TanitAD/Keys.txt")
        m = re.search(r"hf_[A-Za-z0-9]+", open(kp, encoding="utf-8", errors="ignore").read()) if os.path.isfile(kp) else None
        tok = m.group(0) if m else None
    for name, key in (("ckpt_last.pt", "hf_path"), ("summary.json", "hf_summary_path")):
        if r.get(key):
            src = hf_hub_download(repo_id=r["hf_repo"], filename=r[key], revision=r.get("hf_rev"), repo_type=r.get("hf_type", "dataset"), token=tok, local_dir=local_dir + "/_hf")
            os.replace(src, f"{local_dir}/{name}")
    return os.path.isfile(f"{local_dir}/ckpt_last.pt")


def pull_item(arm, stage, bank_dir, source, text=pod_text):
    d = f"{L.CK}/{L.item(arm, stage)}"
    if L.bank_problem(arm, stage) is None:
        return "skip (verified)"
    want, src, info = expected_sha(arm, stage, bank_dir, text)
    if want is None:
        return f"NOT-BANKED: {info}"
    if source in ("hf", "pod") and src != source:
        return f"NOT-BANKED: no {source} record for {arm}.{stage}"
    os.makedirs(d, exist_ok=True)
    if src == "hf":
        ok = hf_get(info, d)
        steps = json.load(open(f"{d}/summary.json"))["steps"] if os.path.isfile(f"{d}/summary.json") else None
    else:
        ok = pod_get(info["pod_path"], f"{d}/ckpt_last.pt") and pod_get(info["pod_path"].replace("ckpt_last.pt", "summary.json"), f"{d}/summary.json")
        steps = json.load(open(f"{d}/summary.json"))["steps"] if os.path.isfile(f"{d}/summary.json") else None
    if not ok:
        return f"FAILED: could not fetch {arm}.{stage} from {src}"
    return verify_and_record(arm, stage, d, want, src, steps) or f"ok ({src})"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto", choices=("auto", "hf", "pod"))
    ap.add_argument("--bank-dir", default=os.environ.get("DS_BANK_DIR", HERE + "/banker"))
    ap.add_argument("--items", nargs="*", default=[L.item(a, s) for a, s in L.DEFAULT_ITEMS])
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--min-free", type=float, default=L.MIN_F_GB, help="F: free-space bar in GB (the card bar, 25, unless the banking step overrides it; logged)")
    a = ap.parse_args(argv)
    why = L.disk_problem(need=a.min_free)
    if why:
        print("REFUSED:", why)
        return 1
    rc = 0
    for it in a.items:
        arm, stage = it.split(".")
        r = "verified" if a.check_only and L.bank_problem(arm, stage) is None else (L.bank_problem(arm, stage) if a.check_only else pull_item(arm, stage, a.bank_dir, a.source))
        print(it, r, flush=True)
        rc |= not str(r).startswith(("ok", "skip", "verified"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
