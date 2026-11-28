"""Read-only check of the private HF bank for block 3 from the dev box: repo private, ckpt LFS sha256 == the pod's bank_ok marker, summary.json present at the same revision.
Writes banker/<arm>.A.json (the format ds_pull.py reads) only when everything agrees.  Token read in place from Keys.txt, never printed."""
import json, os, re, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi
OUT = os.path.dirname(os.path.abspath(__file__)) + "/banker"
REPO = "Sayood/tanitad-v8f-dscale10k-runs"
EXPECT = {  # from the pod's bank_ok markers (read 2026-11-18 ~05:49Z); arm -> (sha256, revision)
    "UE_s0": ("2fd4b747532f55b24523c578948cb713fb80fa0e698586c329dd23ba6e6df914", "3ba330c66375914214dc01b847010dac5ed62653"),
    "BL_s0": ("0b374ded2fb18c62e576177626f68097a58c3ffba253f30abab94c24d87be94b", "bb35e025bf5c97685f9d72b7f26f80d67d4d79fe"),
}
tok = re.search(r"hf_[A-Za-z0-9]+", open("E:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="ignore").read()).group(0)
api = HfApi(token=tok)
info = api.dataset_info(REPO, revision=None, files_metadata=True)
assert info.private, "repo is not private"
rc = 0
for arm, (sha, rev) in EXPECT.items():
    inf = api.dataset_info(REPO, revision=rev, files_metadata=True)
    hub = {s.rfilename: s for s in inf.siblings}
    ck = hub.get(f"{arm}/A/ckpt_last.pt"); sm = hub.get(f"{arm}/A/summary.json")
    ok = ck is not None and sm is not None and ck.lfs is not None and ck.lfs.sha256 == sha
    print(arm, "private", info.private, "lfs", (ck.lfs.sha256[:12] if ck and ck.lfs else None), "size", ck.size if ck else None, "summary", bool(sm), "files", sum(1 for f in hub if f.startswith(arm + "/")), "OK" if ok else "BAD")
    if ok:
        json.dump({"sha256": sha, "hf_repo": REPO, "hf_path": f"{arm}/A/ckpt_last.pt", "hf_rev": rev, "hf_type": "dataset", "hf_summary_path": f"{arm}/A/summary.json",
                   "source": "pod bank_ok marker written by dscale_banker.py (banker_ledger BANKED 2026-11-18); hub LFS sha256 + summary.json presence re-checked from the dev box by the block-3 read build"},
                  open(f"{OUT}/{arm}.A.json", "w", encoding="utf-8"), indent=1)
    rc |= not ok
sys.exit(rc)
