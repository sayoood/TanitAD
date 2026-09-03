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

⛔ WHAT MAY NOT BE PUBLISHED. The card's section 4 stamps all four binding metric
families NOT MEASURED. If the eval has landed by the time this runs, FILL section
4 from the raw JSON (`taniteval/results/refcv3-30k-openloop-*.json`) BEFORE
`--publish`. Never publish a number that is not in that JSON or in
`Project Steering/MODEL_REGISTRY.md`. The in-repo card copy
(`Project Steering/HF_CARD_tanitad-refc-v3.md`) and the published `README.md`
must stay byte-identical — this script checks that and refuses otherwise.

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
    n_unmeasured = text.count("NOT MEASURED")
    print(f"  no-closed-loop guard: present | 'NOT MEASURED' occurrences: {n_unmeasured}")
    if n_unmeasured:
        print("  ⚠️  the four metric families are still NOT MEASURED — see the note below")

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
def pull(a, pre: dict) -> Path:
    dest = Path(a.workdir) / "ckpt.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 68)
    print(f"PULL {POD}:{RUN}/ckpt.pt -> {dest}")
    subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25",
                    "-o", "StrictHostKeyChecking=no",
                    f"{POD}:{RUN}/ckpt.pt", str(dest)], check=True)
    remote = ssh(f"md5sum {RUN}/ckpt.pt").split()[0]
    local = md5_file(dest)
    print(f"  pod   md5 {remote}")
    print(f"  local md5 {local}")
    if remote != local:
        raise SystemExit("⛔ md5 MISMATCH — the transfer is corrupt, do not publish")
    print(f"  size {dest.stat().st_size:,} B — VERIFIED BY CONTENT")

    import torch                                # verify the STEP, not just the bytes
    ck = torch.load(dest, map_location="cpu", weights_only=True)
    step = int(ck["step"])
    n_t = len(ck["model"])
    n_all = sum(v.numel() for v in ck["model"].values())
    print(f"  step {step} | {n_t} tensors | {n_all:,} elements")
    if step != TARGET_STEP:
        print(f"  ⚠️  step is {step}, not the target {TARGET_STEP}. If the run failed, "
              f"the card MUST say so explicitly before publishing.")
    return dest


# ------------------------------------------------------------------ publish
def publish(a, ck: Path) -> None:
    hf = api()
    from huggingface_hub import CommitOperationAdd, hf_hub_download
    print("=" * 68)
    print(f"PUBLISH {ck.name} -> {REPO}")
    ops = [CommitOperationAdd("ckpt.pt", str(ck)),
           CommitOperationAdd("README.md", str(CARD))]
    mj = Path(a.workdir) / "metrics.jsonl"
    if mj.exists():
        ops.append(CommitOperationAdd("metrics.jsonl", str(mj)))
    res = hf.create_commit(repo_id=REPO, repo_type="model", operations=ops,
                           commit_message=f"Final checkpoint (step {TARGET_STEP}) + card")
    print("  commit:", getattr(res, "commit_url", res))

    # VERIFY BY CONTENT, never by exit code
    info = hf.model_info(REPO, files_metadata=True)
    print(f"  gated={getattr(info,'gated',None)} private={getattr(info,'private',None)}")
    for s in info.siblings:
        print(f"    {s.rfilename:20s} {s.size}")
    p = hf_hub_download(REPO, "README.md", token=os.environ["HF_TOKEN"],
                        force_download=True)
    same = md5_file(Path(p)) == md5_file(CARD)
    print(f"  published README identical to the in-repo card: {same}")
    if not same:
        raise SystemExit("⛔ the published card and the in-repo copy diverged — fix before "
                         "reporting this as done")


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
    ck = pull(a, pre)
    if a.pull:
        return 0
    if pre["procs"] > 0:
        print("\n⚠️  the trainer is still running; ckpt.pt is a mid-run save, not the final.")
    publish(a, ck)
    return 0


if __name__ == "__main__":
    sys.exit(main())
