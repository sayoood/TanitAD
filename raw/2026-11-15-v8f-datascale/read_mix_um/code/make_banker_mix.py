"""make_banker_mix.py -- the first step of the UM_s0 read: is UM_s0.A BANKED, and does the chain end in the ONE accepted state?  Writes banker/UM_s0.A.json (the format ds_pull.py reads) only when everything agrees.
UM_s0's chain ended FAIL at `um_checks post` on K-TACSRC alone (card_mix/AMENDMENT_UM_A1_KTACSRC.md, PI-accepted 2026-11-18), not DONE, so the DONE sentinel of the other arms' bank check is NOT required here.  The accepted chain state is exactly:
  * the pod chain log /root/dscale10k/blockum.log carries a `ZZ DSCALE10K BLOCKUM FAIL ZZ` line (the LAST sentinel; no DONE after it) whose post-check JSON is {"pass": false, "problems": [<exactly the one string>]},
    the string being `K-TACSRC tac_src_empty_batches 10 > 5`, and its `um_checks pre` line passed;
  * UM_s0.A is BANKED in the pod banker ledger (last BANKED line for the tag), at a revision of the PRIVATE HF repo whose hub LFS sha256 of UM_s0/A/ckpt_last.pt equals the ledger's, summary.json present.
Any other chain state (a DONE, a different or additional problem, no sentinel, a pre-check failure) refuses (exit 1, nothing written).
Reads only (no GPU, no process touched; the pod address is never written), through ds_pull's pssh wrapper, or --ledger FILE / --chain-log FILE.  HF token read in place from the keys file, never printed; HF_HUB_DISABLE_XET=1.
Usage: python make_banker_mix.py [--ledger FILE] [--chain-log FILE] [--no-hub]"""
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
ARM, STAGE = "UM_s0", "A"
LEDGER_POD = L.POD_ROOT + "/banker_ledger.jsonl"
CHAIN_LOG_POD = L.POD_ROOT + "/blockum.log"
ACCEPTED_PROBLEM = "K-TACSRC tac_src_empty_batches 10 > 5"
SENT = re.compile(r"^ZZ DSCALE10K BLOCKUM (DONE|FAIL)\b(.*)$")
SHA = re.compile(r"^[0-9a-f]{64}$")
CKPT_SHA_PREFIX = "39de0b05"
EVIDENCE = {"post.json": "UM_s0.A.post.json", "gate/checks.json": "UM_s0.A.checks.json", "summary_check.json": "UM_s0.A.summary_check.json"}      # banked file -> evidence file
SRC = "pod banker_ledger.jsonl BANKED line (dscale_banker.py); hub LFS sha256 + summary.json presence re-checked from the dev box by make_banker_mix.py; chain state accepted under UM-A1"


def parse_ledger(text):
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


def chain_state_problem(text):
    """-> reason or None.  None only for the one accepted state (module doc): the last sentinel is the BLOCKUM FAIL whose post check lists exactly the K-TACSRC empty-batch refusal; the pre check passed; no DONE."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    sents = [(i, SENT.match(ln)) for i, ln in enumerate(lines) if SENT.match(ln)]
    if not sents:
        return "the chain log carries no ZZ DSCALE10K BLOCKUM sentinel (the chain has not ended)"
    if any(m.group(1) == "DONE" for _, m in sents):
        return "the chain log carries a DONE sentinel: this read accepts only the K-TACSRC-refused state, a DONE chain is a different state (stop and ask the MM)"
    i, m = sents[-1]
    if len(sents) != 1:
        return f"the chain log carries {len(sents)} sentinels, the accepted state has exactly one FAIL line"
    rest = m.group(2)
    j = rest.find("{")
    if j < 0:
        return f"the FAIL line carries no post-check JSON: {rest[:120]!r}"
    try:
        post = json.loads(rest[j:].strip())
    except ValueError:
        return f"the FAIL line's post-check JSON does not parse: {rest[j:j + 160]!r}"
    if post.get("check") != "post" or post.get("pass") is not False or post.get("problems") != [ACCEPTED_PROBLEM]:
        return f"the post check is not exactly the one K-TACSRC refusal ({ACCEPTED_PROBLEM!r}): {post}"
    pre = [ln for ln in lines if "um_checks pre:" in ln]
    if not pre or not all('"pass": true' in ln for ln in pre):
        return "the chain log carries no passing `um_checks pre` line"
    if not any("run UM_s0 stage A" in ln for ln in lines):
        return "the chain log does not show the UM_s0 stage A run"
    return None


def record_problem(rec):
    tag = f"{ARM}.{STAGE}"
    if not rec:
        return f"{tag} is not banked (no BANKED line in the ledger)"
    if rec.get("tag") != tag or rec.get("status") != "BANKED":
        return f"{tag}: the ledger line is for {rec.get('tag')} / {rec.get('status')}"
    if rec.get("repo") != REPO or not rec.get("revision"):
        return f"{tag}: repo / revision missing or not {REPO}"
    ck, sm = f"{ARM}/{STAGE}/ckpt_last.pt", f"{ARM}/{STAGE}/summary.json"
    files = rec.get("files") or {}
    for f in (ck, sm):
        if f not in files:
            return f"{tag}: the ledger lists no {f}"
    if not SHA.match(str(rec.get("sha256", ""))) or files[ck][1] != rec["sha256"]:
        return f"{tag}: the ledger's checkpoint sha256 differs from its own file entry"
    if not rec["sha256"].startswith(CKPT_SHA_PREFIX):
        return f"{tag}: the ledger's checkpoint sha256 {rec['sha256'][:8]} is not the banked {CKPT_SHA_PREFIX}"
    return None


def hub_problem(hub, private, rec):
    ck, sm = f"{ARM}/{STAGE}/ckpt_last.pt", f"{ARM}/{STAGE}/summary.json"
    if not private:
        return "the HF repo is not private"
    if ck not in hub or sm not in hub:
        return f"the hub at {rec['revision'][:8]} lacks {ck if ck not in hub else sm}"
    if hub[ck] != rec["sha256"]:
        return f"the hub's LFS sha256 of {ck} ({str(hub[ck])[:12]}) differs from the ledger's ({rec['sha256'][:12]})"
    return None


def banker_json(rec):
    return {"sha256": rec["sha256"], "hf_repo": rec["repo"], "hf_path": f"{ARM}/{STAGE}/ckpt_last.pt", "hf_rev": rec["revision"], "hf_type": "dataset", "hf_summary_path": f"{ARM}/{STAGE}/summary.json",
            "summary_sha256": rec["files"][f"{ARM}/{STAGE}/summary.json"][1], "banked_at": rec.get("t"), "source": SRC}


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
    ap.add_argument("--no-hub", action="store_true", help="tests only")
    ap.add_argument("--bank-dir", default=HERE + "/banker")
    ap.add_argument("--evidence", default=HERE + "/evidence")
    a = ap.parse_args(argv)
    import ds_pull as P
    if a.chain_log:
        log = open(a.chain_log, encoding="utf-8", errors="replace").read()
    else:
        log, _ = P._pod(f"pssh \"cat {CHAIN_LOG_POD}\"")
    why = chain_state_problem(log)
    print("chain:", why or "the accepted state (FAIL at um_checks post, K-TACSRC empty batches only)", flush=True)
    if why:
        return 1
    if a.ledger:
        text = open(a.ledger, encoding="utf-8").read()
    else:
        text, _ = P._pod(f"pssh \"grep -E '\\\"tag\\\": \\\"{ARM}\\.' {LEDGER_POD}\"")
    rec = parse_ledger(text).get(f"{ARM}.{STAGE}")
    why = record_problem(rec)
    if not why and not a.no_hub:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        from huggingface_hub import HfApi
        api = HfApi(token=_token())
        private = api.dataset_info(REPO).private
        inf = api.dataset_info(REPO, revision=rec["revision"], files_metadata=True)
        why = hub_problem({s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in inf.siblings}, private, rec)
        if not why:      # the banked evidence files of the run (post-training Adam-state check, gate checks, step / sha record), each sha256-checked against the ledger's own entry
            from huggingface_hub import hf_hub_download
            os.makedirs(a.evidence, exist_ok=True)
            for f, nm in EVIDENCE.items():
                src = hf_hub_download(repo_id=REPO, filename=f"{ARM}/{STAGE}/{f}", revision=rec["revision"], repo_type="dataset", token=_token(), local_dir=a.evidence + "/_hf")
                if L.sha256_file(src) != rec["files"][f"{ARM}/{STAGE}/{f}"][1]:
                    why = f"{f} differs from the ledger's sha256"
                    break
                shutil.copyfile(src, f"{a.evidence}/{nm}")
            shutil.rmtree(a.evidence + "/_hf", ignore_errors=True)
    if why:
        print(f"{ARM}.{STAGE} NOT-BANKED:", why, flush=True)
        return 1
    os.makedirs(a.bank_dir, exist_ok=True)
    os.makedirs(a.evidence, exist_ok=True)
    json.dump(banker_json(rec), open(f"{a.bank_dir}/{ARM}.{STAGE}.json", "w", encoding="utf-8"), indent=1)
    json.dump({"ledger_line": rec, "chain_log": log}, open(f"{a.evidence}/{ARM}.{STAGE}.bank_evidence.json", "w", encoding="utf-8"), indent=1)
    print(f"{ARM}.{STAGE} banked, verified", rec["revision"][:8], rec["sha256"][:12], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
