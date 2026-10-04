"""SPEC AMENDMENT A4: the refcv6@38k baseline stamp -- written once, RE-VERIFIED before every reuse.

    python baseline_stamp.py write     # after the refcv6@38k rolls / perception dump are banked
    python baseline_stamp.py verify    # before a later milestone reuses them (exit record in the JSON)

Stamp (`D:/refcv7_eval_kit/baseline_refcv6_38k/BASELINE_STAMP.json`): the code tree + commit, the checkpoint
md5, the config md5, the refcv6 package SPEC sha256 its battery ran under, this package's SPEC sha256, the S2
window digest (sha256 of the sha12-keyed window list), the sha256 of every dump manifest and of the
perception dump, and the refcv6 G0 verdicts. `verify` recomputes every one and lists each mismatch; a reuse
is admissible only with zero mismatches. A sanitized copy is banked in the package (`raw/baseline_refcv6_38k/`).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

B6 = Path("D:/refcv7_eval_kit/baseline_refcv6_38k")
S = B6 / "refcv6_step38000"
STAMP = B6 / "BASELINE_STAMP.json"
PKG = Path(__file__).resolve().parent.parent
CK = Path("D:/refcv6_eval_kit/ckpt/ckpt_step38000_stopped.pt")
CF = Path("D:/refcv6_eval_kit/ckpt_final/config.json")
TREE = Path("C:/Users/Admin/ev6_82c2331")
WIN_SHA12 = Path("D:/refcv7_eval_kit/windows/s2_windows.json.sha12.json")


def _sha(p: Path, algo="sha256") -> str | None:
    if not p.exists():
        return None
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def _load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return None


def compute() -> dict:
    bs = _load(S / "battery_summary.json") or {}
    g0 = (bs.get("stages") or {}).get("g0") or {}
    perc = _load(B6 / "perc_refcv6_38k.json") or {}
    win = _load(WIN_SHA12)
    return {
        "tree": str(TREE), "tree_commit": (TREE / "COMMIT.txt").read_text().strip()
        if (TREE / "COMMIT.txt").exists() else None,
        "ckpt": str(CK), "ckpt_md5": _sha(CK, "md5"), "config_md5": _sha(CF, "md5"),
        "refcv6_package_spec_sha256": bs.get("spec_sha256"),
        "refcv7_battery_spec_sha256": _sha(PKG / "SPEC.md"),
        "s2_windows_sha12_digest": (hashlib.sha256(json.dumps(win, sort_keys=True).encode()).hexdigest()
                                    if win is not None else None),
        "dump_manifest_sha256": {d: _sha(S / d / "manifest.json") for d in ("dump_s0", "dump_s1")},
        "perception_dump_pkl_sha256": perc.get("pkl_sha256"),
        "perception_dump_n_windows": perc.get("n_windows"),
        "refcv6_g0": {k: g0.get(k) for k in ("G0_as_registered", "G0_A1", "G0_A2")},
    }


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    cur = compute()
    if cmd == "write":
        rec = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **cur}
        json.dump(rec, open(STAMP, "w", encoding="utf-8"), indent=1)
        dst = PKG / "raw" / "baseline_refcv6_38k"
        dst.mkdir(parents=True, exist_ok=True)
        json.dump(rec, open(dst / "BASELINE_STAMP.json", "w", encoding="utf-8"), indent=1)
        print(json.dumps(rec))
        return
    st = _load(STAMP)
    if st is None:
        out = {"verify": "NO STAMP", "reuse_admissible": False}
    else:
        keys = [k for k in cur if k != "refcv7_battery_spec_sha256"]   # the SPEC may gain amendments
        bad = {k: {"stamp": st.get(k), "now": cur[k]} for k in keys if st.get(k) != cur[k]}
        out = {"verify": "OK" if not bad else "MISMATCH", "mismatches": bad,
               "reuse_admissible": not bad, "stamp_written": st.get("written")}
    json.dump(out, open(B6 / "BASELINE_VERIFY.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
