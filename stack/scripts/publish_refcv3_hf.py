#!/usr/bin/env python3
"""Publish the FINAL refcv3 checkpoint to `Sayood/tanitad-refc-v3`.

Step 6 of the 2026-09-03 publication run. Steps 1-5 (quota check, naming, repo
creation, model card, LFS dry-run) are DONE; the repo already carries `README.md`
and `config.json`, and the 428 MB `ckpt_30000.pt` blob was pre-uploaded to LFS
storage as the dry run (uploaded, deliberately NOT committed).

    python publish_refcv3_hf.py --check          # preflight only, uploads nothing
    python publish_refcv3_hf.py --pull           # + pull ckpt.pt, verify md5 both ends
    python publish_refcv3_hf.py --publish        # + commit the weights

⛔ THE TOKEN. Read IN PLACE from `Keys.txt` (git-ignored). Never copied, never
printed, never passed as an argv. Loaded into os.environ at runtime only.

⛔ ON THIS DEV BOX HuggingFace is reachable ONLY through
`truststore.inject_into_ssl()` — plain certifi fails behind the TLS proxy, and
`curl` needs `--ssl-no-revoke` or it reports HTTP 000 and looks like an outage.
The venv carrying `huggingface_hub` is `C:/Users/Admin/venvs/tanitad`.

⛔ WHAT MAY NOT BE PUBLISHED. Never publish a number that is not in the raw eval
JSON (`taniteval/results/refcv3-40284-openloop{,.ARM}.json`) or in
`Project Steering/MODEL_REGISTRY.md`. The preflight makes POSITIVE content
assertions on the card — that it scopes itself to the final step, states the
hold-action loss, declares STRATEGIC unavailable, states non-parity, and marks
the route head a probe — and refuses if any of them has been edited away. The
in-repo card copy (`Project Steering/HF_CARD_tanitad-refc-v3.md`) is what gets
uploaded as `README.md`, and every text artifact is re-downloaded and re-hashed
after the commit, because a create_commit that returned a URL is not evidence.

⛔ NO CLOSED-LOOP CLAIM. Binding PI ruling 2026-09-02: a model consuming its own
planner's output is still OPEN loop. This script greps the card for the guard
sentence and refuses to publish if someone removed it.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import subprocess
import sys
from pathlib import Path

# ⛔ TWO REAL FAILURES, BOTH FIXED ON THIS LINE. (1) Python fully buffers stdout
# when it is not a tty, so run as a background job this script wrote ZERO bytes
# for 20 minutes and there was no way to tell slow from stuck. (2) This dev box's
# console is cp1252: printing the warning emoji below raised UnicodeEncodeError
# and KILLED the preflight — it only survived the first run because that caller
# happened to export PYTHONIOENCODING=utf-8. A script must not depend on its
# caller's environment to be able to speak. Note the exit code was still 0
# through a pipe, so the crash was invisible twice over.
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO = "Sayood/tanitad-refc-v3"
POD = "tanitad-refcv3"                       # = tanitad-a40; TRAINING BOX: scp only
RUN = "/workspace/experiments/refcv3-b1-v72-30k"
TARGET_STEP = 40284
REPO_ROOT = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD")
CARD = REPO_ROOT / "Project Steering" / "HF_CARD_tanitad-refc-v3.md"
KEYS = REPO_ROOT / "Keys.txt"

# ⛔ EVERY CHECKPOINT CARRIES ITS STEP IN ITS FILENAME, AND NOTHING IS CALLED BARE
# `ckpt.pt`. This repo holds TWO evaluated steps (30,000 and 40,284) and an
# ambiguous name is exactly how a reader ends up quoting one checkpoint's numbers
# for the other's weights — the same inversion the `flagship4b-phase0-30k` repo
# name invites and that CLAUDE.md opens by warning about.
#
# (pod filename, path in the HF repo, expected md5 or None)
PUBLISH_FROM_POD = [
    ("ckpt_40284_FINAL.pt", "ckpt_40284_FINAL.pt",
     "fc304b62686ddb9e685d14bdab482404"),          # {model, opt, step} — resumable
    ("ckpt_step40284_frozen.pt", "ckpt_40284.pt",
     "b1ed7075ff730d0993d2eaa3c86f6b56"),          # {model, step} — inference
    ("metrics.jsonl", "metrics.jsonl", None),
    ("summary.json", "summary.json", None),        # the run's own done-marker
    ("supervisor.log", "supervisor.log", None),    # the evidence behind caveat C7
]

# published straight out of the git repo (path in repo, path in the HF repo)
PUBLISH_FROM_REPO = [
    (CARD, "README.md"),
    (REPO_ROOT / "taniteval" / "results" / "refcv3-40284-openloop.json",
     "eval/refcv3-40284-openloop.json"),
    (REPO_ROOT / "taniteval" / "results" / "refcv3-40284-openloop.ARM.json",
     "eval/refcv3-40284-openloop.ARM.json"),
]

GUARD = "This card makes NO closed-loop claim for REF-C v3"
PRO_PUBLIC_TB, PRO_PRIVATE_TB = 10.0, 1.0    # huggingface.co/docs/hub/storage-limits


def load_token() -> None:
    """Read the HF token IN PLACE. Never printed, never written anywhere."""
    m = re.findall(r"hf_[A-Za-z0-9]+", io.open(KEYS, encoding="utf-8",
                                               errors="replace").read())
    if not m:
        raise SystemExit("no hf_ token in Keys.txt")
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = m[0]


def api():
    import truststore
    truststore.inject_into_ssl()
    load_token()
    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"])


def ssh(cmd: str) -> str:
    # -n is mandatory: a nested ssh inside a pipe EATS the rest of stdin.
    r = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25",
                        "-o", "StrictHostKeyChecking=no", POD, cmd],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300)
    if r.returncode != 0:
        raise SystemExit(f"ssh failed ({r.returncode}): {r.stderr[:400]}")
    return r.stdout.strip()


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- preflight
def preflight(a) -> dict:
    out = {}
    print("=" * 68)
    print("PREFLIGHT")
    print("=" * 68)

    # 1. the card must still carry its guards, and match what is published
    card = CARD.read_bytes()
    text = card.decode("utf-8")
    print(f"  card: {len(card):,} B  md5 {hashlib.md5(card).hexdigest()}")
    assert GUARD in text, "REFUSING: the no-closed-loop guard is gone from the card"
    # ⛔ POSITIVE CONTENT ASSERTIONS, not a keyword count. The old check counted the
    # string "NOT MEASURED" anywhere in the file, which also matches the card's own
    # evidence-class LEGEND — so it fired on a fully-measured card. Absence of a
    # keyword is not evidence; presence of the specific claim is.
    for what, needle in [
        ("the FINAL step is the card's scope", f"MEASURED on the FINAL `ckpt_{TARGET_STEP}.pt`"),
        ("the hold-action loss is stated", "THE TRIVIAL CONTROL WON"),
        ("STRATEGIC is declared unavailable", "UNAVAILABLE, n = 0"),
        ("non-parity is stated", "v2_parity.parity false"),
        ("the route head is marked a probe", "reported AS A PROBE"),
    ]:
        assert needle in text, f"REFUSING: the card no longer states {what} ({needle!r})"
        print(f"  card states: {what}")
    print("  no-closed-loop guard: present")

    # 2. has the eval landed? (two differently-bound probes; absence at one
    #    location is not absence)
    # ⛔ BOUND THE SEARCH AND NAME THE BOUNDS. `REPO_ROOT.rglob(...)` walks the
    # whole Google Drive mount and took ~10 MINUTES here — long enough that the
    # preflight read as hung. A probe that never returns is not a probe. Two
    # roots, both named in the output, so the operator can see the scope rather
    # than trust it: absence found at ONE location is not absence, but absence
    # found across two NAMED roots is a statement someone can check.
    res = REPO_ROOT / "taniteval" / "results"
    byname = sorted(p.name for p in res.glob("refcv3*")) if res.is_dir() else []
    roots = [REPO_ROOT / "taniteval",
             REPO_ROOT / "TanitAD Research Lab" / "Benchmarks & Evals"]
    bywalk = []
    for root in roots:
        if root.is_dir():
            bywalk += [str(p.relative_to(REPO_ROOT))
                       for p in root.rglob("refcv3*openloop*.json")]
    print(f"  eval JSON probe A (taniteval/results glob): {byname or 'NONE'}")
    print(f"  eval JSON probe B (recursive under {len(roots)} named roots): "
          f"{sorted(bywalk) or 'NONE'}")
    for r in roots:
        print(f"      root: {r.relative_to(REPO_ROOT)}  (exists: {r.is_dir()})")
    out["eval_present"] = bool(byname or bywalk)
    if not out["eval_present"]:
        print("  ⛔ NO EVAL RESULT. Section 4 of the card must stay NOT MEASURED,")
        print("     and no accuracy number may be added from any other source.")

    # 3. the run
    raw = ssh(f'S=$(ls -l {RUN}/ckpt.pt 2>/dev/null | awk "{{print \\$5}}"); '
              f'P=$(ps -eo args | grep -c "refc_v3_trai[n].py"); '
              f'echo "QQ${{S}}-${{P}}QQ"')
    m = re.search(r"QQ(\d*)-(\d+)QQ", raw)      # marker is disjoint from the grep
    size, procs = (int(m.group(1) or 0), int(m.group(2))) if m else (0, -1)
    print(f"  pod {POD}: ckpt.pt {size:,} B | trainer processes {procs}")
    out.update(pod_size=size, procs=procs)
    if procs > 0:
        print("  ⚠️  TRAINING IS STILL LIVE — never add GPU/RAM load to this pod. scp only.")

    # 4. quota. HF PRO: public 'up to 10TB', private 1TB + PAYG.
    hf = api()
    import httpx
    H = {"Authorization": "Bearer " + os.environ["HF_TOKEN"]}
    # ⚠️ `usedStorage` is NOT a valid `expand` on the LIST endpoints (MEASURED --
    # the API replies with the allowed set), so one call per repo is the only
    # route. Bounded pool + per-request try/except + a progress line, because
    # 44 sequential 60 s timeouts is how this step became unobservable before.
    from concurrent.futures import ThreadPoolExecutor

    repos = ([("model", x.id) for x in hf.list_models(author="Sayood")]
             + [("dataset", x.id) for x in hf.list_datasets(author="Sayood")])
    print(f"  summing usedStorage over {len(repos)} repos ...")

    def one(item):
        kind, rid = item
        try:
            j = httpx.get(f"https://huggingface.co/api/{kind}s/{rid}"
                          "?expand[]=usedStorage&expand[]=private",
                          headers=H, timeout=30, follow_redirects=True).json()
            return rid, j.get("usedStorage"), bool(j.get("private"))
        except Exception as e:
            return rid, None, f"ERR {type(e).__name__}"

    pub = priv = 0
    failed = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (rid, used, private) in enumerate(ex.map(one, repos), 1):
            if used is None:
                failed.append(rid)
            elif private is True:
                priv += used
            else:
                pub += used
            if i % 10 == 0 or i == len(repos):
                print(f"    {i}/{len(repos)} ...")
    if failed:
        # ⛔ A repo whose size we could not read is NOT zero bytes. Say so, and
        # refuse to treat the total as complete.
        raise SystemExit(f"⛔ REFUSING: could not read usedStorage for "
                         f"{len(failed)} repo(s) {failed[:5]} — the total would "
                         f"understate usage. Re-run; do not publish on a partial "
                         f"quota read.")
    print(f"  HF public  {pub/1e9:8.3f} GB of {PRO_PUBLIC_TB*1000:.0f} GB "
          f"({100*pub/1e12/PRO_PUBLIC_TB:.2f} %)")
    print(f"  HF private {priv/1e9:8.3f} GB of {PRO_PRIVATE_TB*1000:.0f} GB "
          f"({100*priv/1e12/PRO_PRIVATE_TB:.2f} %)")
    head = PRO_PUBLIC_TB * 1e12 - pub
    print(f"  headroom (public) {head/1e9:,.1f} GB | this upload ~{size/1e9:.2f} GB")
    if size > head:
        raise SystemExit("⛔ REFUSING: the upload would exceed the PRO public ceiling. "
                         "Report to the PI. Do NOT buy storage.")
    out.update(pub=pub, priv=priv)
    return out


# --------------------------------------------------------------------- pull
def pull(a, pre: dict) -> dict:
    """Fetch every pod-side artifact, md5-verified at BOTH ends.

    Idempotent: a file already on disk whose md5 matches the pod is not re-fetched,
    so a re-run after a failed upload does not repay 1.3 GB of transfer.
    """
    wd = Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    print("=" * 68)
    print(f"PULL {POD}:{RUN}/ -> {wd}")
    got: dict[str, Path] = {}
    for remote_name, repo_path, want_md5 in PUBLISH_FROM_POD:
        dest = wd / remote_name
        remote = ssh(f"md5sum {RUN}/{remote_name}").split()[0]
        if dest.exists() and md5_file(dest) == remote:
            print(f"  {remote_name:28s} already local, md5 matches the pod — skipping")
        else:
            subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25",
                            "-o", "StrictHostKeyChecking=no",
                            f"{POD}:{RUN}/{remote_name}", str(dest)], check=True)
        local = md5_file(dest)
        if remote != local:
            raise SystemExit(f"⛔ md5 MISMATCH on {remote_name} — transfer is corrupt, "
                             f"do not publish (pod {remote} != local {local})")
        # ⛔ A published md5 that nobody re-derived is a claim, not a check.
        if want_md5 and local != want_md5:
            raise SystemExit(f"⛔ {remote_name} md5 {local} != the expected {want_md5}. "
                             f"The artifact changed; STOP and re-verify before publishing.")
        print(f"  {remote_name:28s} {dest.stat().st_size:>13,} B  md5 {local}  ✅ both ends")
        got[repo_path] = dest

    # ⛔ VERIFY THE STEP BY CONTENT. The filename says 40284; only the tensor says so.
    import torch
    ck = torch.load(got["ckpt_40284_FINAL.pt"], map_location="cpu", weights_only=True)
    step, n_t = int(ck["step"]), len(ck["model"])
    n_all = sum(v.numel() for v in ck["model"].values())
    print(f"\n  CONTENT CHECK  step={step} | {n_t} tensors | {n_all:,} elements "
          f"| optimizer state: {'yes' if 'opt' in ck else 'no'}")
    if step != TARGET_STEP:
        raise SystemExit(f"⛔ REFUSING: step is {step}, not the target {TARGET_STEP}.")

    # the model-only file must be the SAME WEIGHTS, or the two artifacts disagree
    mo = torch.load(got["ckpt_40284.pt"], map_location="cpu", weights_only=True)
    if int(mo["step"]) != TARGET_STEP or set(mo["model"]) != set(ck["model"]):
        raise SystemExit("⛔ REFUSING: the model-only checkpoint does not match the final one")
    diff = [k for k in ck["model"] if not torch.equal(ck["model"][k], mo["model"][k])]
    if diff:
        raise SystemExit(f"⛔ REFUSING: {len(diff)} tensors differ between the full and "
                         f"model-only checkpoints, e.g. {diff[:3]}")
    print(f"  CONTENT CHECK  model-only file is BITWISE IDENTICAL to the final "
          f"checkpoint's weights ({n_t}/{n_t} tensors equal)")
    return got


# ------------------------------------------------------------------ publish
def publish(a, got: dict) -> None:
    hf = api()
    from huggingface_hub import CommitOperationAdd, hf_hub_download
    print("=" * 68)
    print(f"PUBLISH -> {REPO}")
    ops, expect = [], {}
    for repo_path, local in got.items():
        ops.append(CommitOperationAdd(repo_path, str(local)))
        expect[repo_path] = md5_file(local)
    for local, repo_path in PUBLISH_FROM_REPO:
        if not local.exists():
            raise SystemExit(f"⛔ REFUSING: {local} is missing — it is named on the card")
        ops.append(CommitOperationAdd(repo_path, str(local)))
        expect[repo_path] = md5_file(local)
    for repo_path in sorted(expect):
        print(f"  + {repo_path:38s} md5 {expect[repo_path]}")
    res = hf.create_commit(
        repo_id=REPO, repo_type="model", operations=ops,
        commit_message=f"FINAL checkpoint (step {TARGET_STEP}) + its four-family eval + card")
    print("  commit:", getattr(res, "commit_url", res))

    # ⛔ VERIFY BY CONTENT, NEVER BY EXIT CODE. Re-download every text artifact and
    # re-hash it; a create_commit that returned a URL is not evidence of what landed.
    info = hf.model_info(REPO, files_metadata=True)
    print(f"\n  gated={getattr(info,'gated',None)} private={getattr(info,'private',None)}")
    for s in sorted(info.siblings, key=lambda x: x.rfilename):
        print(f"    {s.rfilename:38s} {s.size if s.size is not None else '':>13}")
    bad = []
    for repo_path, want in expect.items():
        if repo_path.endswith(".pt"):
            continue                     # multi-GB: verified by md5 at both ends in pull()
        p = hf_hub_download(REPO, repo_path, token=os.environ["HF_TOKEN"],
                            force_download=True)
        got_md5 = md5_file(Path(p))
        ok = got_md5 == want
        print(f"  re-download {repo_path:38s} {'✅ md5 matches' if ok else '⛔ MISMATCH'}")
        if not ok:
            bad.append(repo_path)
    if bad:
        raise SystemExit(f"⛔ published bytes differ for {bad} — fix before reporting this done")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="preflight only")
    ap.add_argument("--pull", action="store_true", help="preflight + pull + verify md5")
    ap.add_argument("--publish", action="store_true", help="preflight + pull + commit")
    ap.add_argument("--workdir", default=os.environ.get("TMP", ".") + "/refcv3-publish")
    a = ap.parse_args()
    if not (a.check or a.pull or a.publish):
        ap.error("pick one of --check / --pull / --publish")

    pre = preflight(a)
    if a.check:
        return 0
    got = pull(a, pre)
    if a.pull:
        return 0
    if pre["procs"] > 0:
        raise SystemExit("⛔ REFUSING: the trainer is still running, so the checkpoints are "
                         "mid-run saves, not the final ones. Never publish a live run's ckpt.")
    publish(a, got)
    return 0


if __name__ == "__main__":
    sys.exit(main())
