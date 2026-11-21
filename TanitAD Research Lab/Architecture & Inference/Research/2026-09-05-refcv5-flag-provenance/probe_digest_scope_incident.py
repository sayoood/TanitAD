"""M18 evidence: the two joins' sidecars digest DIFFERENT artifacts.

MEASURED, not copied: the train join is on this box, so both candidate digests
are recomputed here and matched against what the sidecar recorded. val40's file
is not on this box, so its sidecars are reported as READ (their own recorded
fields), and that difference in evidence class is stated per row.
"""
import hashlib
import json
import lzma
import os
import sys

R = r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
TRAIN_JOIN = r"C:/Users/Admin/tanitad-data/joins/joins/train2400_agents.jsonl.xz"
SIDECARS = {
    "train2400 (local box, .xz sidecar)":
        r"C:/Users/Admin/tanitad-data/joins/joins/"
        r"train2400_agents.jsonl.xz.meta.json.bak",
    "train2400 (repo bank)":
        R + r"/TanitAD Research Lab/Data Engineering/Implementation/incoming/"
            r"2026-08-17-train-obstacle-join/raw/train2400_agents.meta.json",
    "val40 (.jsonl sidecar)":
        R + r"/TanitAD Research Lab/Benchmarks & Evals/Implementation/incoming/"
            r"2026-08-18-val40-lead-join/raw/val40_agents.jsonl.meta.json",
    "val40 (.xz sidecar)":
        R + r"/TanitAD Research Lab/Benchmarks & Evals/Implementation/incoming/"
            r"2026-08-18-val40-lead-join/raw/val40_agents.jsonl.xz.meta.json",
}


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def md5_xz_inner(p):
    h = hashlib.md5()
    with lzma.open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


out = {"control": None, "sidecars": {}, "measured": {}, "verdict": {}}
# ⛔ same-breath control: an empty/failed read on this mount is
# indistinguishable from a genuine absence unless a control reads non-zero.
out["control"] = {"CLAUDE.md_bytes": os.path.getsize(R + "/CLAUDE.md")}

for name, p in SIDECARS.items():
    try:
        d = json.load(open(p, encoding="utf-8"))
    except OSError as e:
        out["sidecars"][name] = {"read": False, "error": str(e)}
        continue
    out["sidecars"][name] = {
        "read": True, "path": p,
        "summary.md5": d.get("summary", {}).get("md5"),
        "summary.out": d.get("summary", {}).get("out"),
        "summary.out_basename": os.path.basename(
            str(d.get("summary", {}).get("out", "")).replace("\\", "/")),
        "xz_of_md5": d.get("xz_of_md5"),
        "declares_digest_scope": "digest_scope" in d,
    }

if os.path.exists(TRAIN_JOIN):
    comp = md5_file(TRAIN_JOIN)
    deco = md5_xz_inner(TRAIN_JOIN)
    out["measured"] = {
        "file": TRAIN_JOIN, "bytes": os.path.getsize(TRAIN_JOIN),
        "md5_compressed_xz": comp, "md5_decompressed_jsonl": deco,
    }
    rec = out["sidecars"].get("train2400 (repo bank)", {}).get("summary.md5")
    out["verdict"]["train_summary_md5_covers"] = (
        "compressed" if rec == comp else
        "decompressed" if rec == deco else "NEITHER")
    out["verdict"]["a_val40_style_checker_on_train"] = {
        "hashes": "decompressed .jsonl", "gets": deco,
        "compares_against_recorded": rec,
        "result": "REFUSE (good file rejected)" if rec != deco else "accept",
    }

v_xz = out["sidecars"].get("val40 (.xz sidecar)", {})
out["verdict"]["val40_xz_sidecar_names_the_jsonl_in_summary_out"] = (
    v_xz.get("summary.out_basename"))
out["verdict"]["val40_xz_of_md5_equals_summary_md5"] = (
    v_xz.get("xz_of_md5") == v_xz.get("summary.md5"))
out["verdict"]["any_sidecar_declares_a_scope"] = any(
    s.get("declares_digest_scope") for s in out["sidecars"].values())
json.dump(out, sys.stdout, indent=1)
print()
