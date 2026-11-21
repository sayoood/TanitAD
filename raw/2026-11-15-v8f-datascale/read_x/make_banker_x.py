"""make_banker_x.py -- the first step of the X_s0 read: is X_s0 BANKED?  Writes banker/X_s0.A.json and banker/X_s0.B.json (the format ds_pull.py reads) only when everything agrees; otherwise exit 1 and write nothing for that stage.
Reads (read only; no GPU, no process touched; the pod address is never written): the pod's banker ledger (`banker_ledger.jsonl`, last BANKED line per tag) and the chain's sentinel in `blockx.log`, through ds_pull's pssh wrapper (or --ledger FILE / --chain-log FILE).
Checks per stage: ledger status BANKED, repo, revision, a ckpt sha256 equal to the ledger's own file entry, summary.json and gate/checks.json listed; then the PRIVATE HF repo at that revision: private, the ckpt's hub LFS sha256 equals the ledger's, summary.json present.
Evidence: gate/checks.json of each stage is downloaded at the banked revision, its sha256 compared with the ledger's entry, and copied to evidence/X_s0.<stage>.checks.json (the C-JOIN-EQ quote of the reader).
The chain must have ended in `ZZ DSCALE10K BLOCKX DONE ZZ` (a FAIL line, or no sentinel, stops the read).  HF token read in place from the keys file, never printed; HF_HUB_DISABLE_XET=1.
Usage: python make_banker_x.py [--ledger FILE] [--chain-log FILE] [--no-chain-check] [--no-hub]"""
import argparse
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_lib as L  # noqa: E402

REPO = "Sayood/tanitad-v8f-dscale10k-runs"
ARM = "X_s0"
LEDGER_POD = L.POD_ROOT + "/banker_ledger.jsonl"
CHAIN_LOG_POD = L.POD_ROOT + "/blockx.log"
SENTINEL = re.compile(r"^ZZ DSCALE10K BLOCKX (DONE|FAIL)\b")
SHA = re.compile(r"^[0-9a-f]{64}$")
SRC = "pod banker_ledger.jsonl BANKED line (dscale_banker.py); hub LFS sha256 + summary.json presence + gate/checks.json sha256 re-checked from the dev box by make_banker_x.py"


def parse_ledger(text):
    """-> {tag: last record with status BANKED}"""
    out = {}
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("status") == "BANKED" and r.get("tag"):
            out[r["tag"]] = r
    return out


def record_problem(rec, arm, stage):
    """-> reason or None: the ledger line is a complete BANKED record for this arm and stage"""
    tag = f"{arm}.{stage}"
    if not rec:
        return f"{tag} is not banked (no BANKED line in the ledger)"
    if rec.get("tag") != tag or rec.get("status") != "BANKED":
        return f"{tag}: the ledger line is for {rec.get('tag')} / {rec.get('status')}"
    if rec.get("repo") != REPO or not rec.get("revision"):
        return f"{tag}: repo / revision missing or not {REPO}"
    ck, sm, gc = f"{arm}/{stage}/ckpt_last.pt", f"{arm}/{stage}/summary.json", f"{arm}/{stage}/gate/checks.json"
    files = rec.get("files") or {}
    for f in (ck, sm, gc):
        if f not in files:
            return f"{tag}: the ledger lists no {f}"
    if not SHA.match(str(rec.get("sha256", ""))) or files[ck][1] != rec["sha256"]:
        return f"{tag}: the ledger's checkpoint sha256 differs from its own file entry"
    return None


def hub_problem(hub, private, rec, arm, stage):
    """-> reason or None.  hub = {rfilename: lfs sha256 or None} of the repo at the banked revision"""
    ck, sm = f"{arm}/{stage}/ckpt_last.pt", f"{arm}/{stage}/summary.json"
    if not private:
        return "the HF repo is not private"
    if ck not in hub or sm not in hub:
        return f"the hub at {rec['revision'][:8]} lacks {ck if ck not in hub else sm}"
    if hub[ck] != rec["sha256"]:
        return f"the hub's LFS sha256 of {ck} ({str(hub[ck])[:12]}) differs from the ledger's ({rec['sha256'][:12]})"
    return None


def banker_json(rec, arm, stage):
    return {"sha256": rec["sha256"], "hf_repo": rec["repo"], "hf_path": f"{arm}/{stage}/ckpt_last.pt", "hf_rev": rec["revision"], "hf_type": "dataset", "hf_summary_path": f"{arm}/{stage}/summary.json",
            "gate_checks_path": f"{arm}/{stage}/gate/checks.json", "gate_checks_sha256": rec["files"][f"{arm}/{stage}/gate/checks.json"][1], "banked_at": rec.get("t"), "source": SRC}


def chain_done_problem(text):
    """-> reason or None: the LAST sentinel line of the chain log is DONE (a later FAIL, or no sentinel, is a stop)"""
    last = None
    for ln in (text or "").splitlines():
        m = SENTINEL.match(ln.strip())
        if m:
            last = (m.group(1), ln.strip())
    if last is None:
        return "the chain log carries no ZZ DSCALE10K BLOCKX sentinel yet (the chain has not ended)"
    return None if last[0] == "DONE" else f"the chain ended in: {last[1]}"


def _token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    kp = os.environ.get("DS_KEYS", "E:/Projects/TanitAD/Keys.txt")
    m = re.search(r"hf_[A-Za-z0-9]+", open(kp, encoding="utf-8", errors="ignore").read()) if os.path.isfile(kp) else None
    return m.group(0) if m else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--chain-log", default=None)
    ap.add_argument("--no-chain-check", action="store_true")
    ap.add_argument("--no-hub", action="store_true", help="tests only: skip the hub and the evidence download")
    ap.add_argument("--bank-dir", default=HERE + "/banker")
    ap.add_argument("--evidence", default=HERE + "/evidence")
    a = ap.parse_args(argv)
    import ds_pull as P
    rc = 0
    if not a.no_chain_check:
        if a.chain_log:
            log = open(a.chain_log, encoding="utf-8", errors="replace").read()
        else:
            out, r = P._pod(f"pssh \"grep -E '^ZZ DSCALE10K BLOCKX' {CHAIN_LOG_POD} | tail -n 3\"")
            log = out
        why = chain_done_problem(log)
        print("chain:", why or "DONE", flush=True)
        if why:
            return 1
    if a.ledger:
        text = open(a.ledger, encoding="utf-8").read()
    else:
        text, r = P._pod(f"pssh \"grep -E '\\\"tag\\\": \\\"{ARM}\\.' {LEDGER_POD}\"")
    led = parse_ledger(text)
    os.makedirs(a.bank_dir, exist_ok=True)
    os.makedirs(a.evidence, exist_ok=True)
    api = info = None
    if not a.no_hub:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=_token())
        private = api.dataset_info(REPO).private
    for stage in ("A", "B"):
        tag = f"{ARM}.{stage}"
        rec = led.get(tag)
        why = record_problem(rec, ARM, stage)
        if not why and api is not None:
            inf = api.dataset_info(REPO, revision=rec["revision"], files_metadata=True)
            hub = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in inf.siblings}
            why = hub_problem(hub, private, rec, ARM, stage)
            if not why:
                src = hf_hub_download(repo_id=REPO, filename=f"{ARM}/{stage}/gate/checks.json", revision=rec["revision"], repo_type="dataset", token=_token(), local_dir=a.evidence + "/_hf")
                if L.sha256_file(src) != rec["files"][f"{ARM}/{stage}/gate/checks.json"][1]:
                    why = "gate/checks.json differs from the ledger's sha256"
                else:
                    shutil.copyfile(src, f"{a.evidence}/{ARM}.{stage}.checks.json")
        if why:
            print(tag, "NOT-BANKED:", why, flush=True)
            rc = 1
            continue
        json.dump(banker_json(rec, ARM, stage), open(f"{a.bank_dir}/{tag}.json", "w", encoding="utf-8"), indent=1)
        print(tag, "banked, verified", rec["revision"][:8], rec["sha256"][:12], flush=True)
    shutil.rmtree(a.evidence + "/_hf", ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
