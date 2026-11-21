"""SPEC A6 step 1: select the 139 TRAIN clips by the registered rule (read-only on Thor).

usage: python a6_select.py
  * lists the run's train cache on Thor (`ssh -n ls`), pulls the run's train label file and train
    max-speed sidecar by read-only scp (md5 checked against `ssh -n md5sum`);
  * candidates = train-cache clips NOT among the 139 eval clips, WITH a train-label record and a
    train-sidecar row; selection = the 139 with the smallest sha12(clip_id);
  * writes raw/a6/train_clips.PRIVATE.json (raw ids: dev box only, never banked) and
    raw/a6/train_clips_sha12.json + raw/a6/select_record.json (sha12 only, bankable).
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcv6_loader as L  # noqa: E402

L.bootstrap()
H = "tanitad-thor-wifi"
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", H]
TRAIN_DIR = "/home/nvidia/data/refcv6-b1-416x1024-train"
FILES = {"labels": "/home/nvidia/data/v8labels/labels/s2_labels_v8_train.jsonl.gz",
         "sidecar": "/home/nvidia/data/refcv6_speed_max_v8_train.jsonl"}
N_SELECT = 139
OUT = HERE.parent / "raw" / "a6"
LOCAL = Path("D:/refcv6_eval_kit/data/a6")


def sha12(c: str) -> str:
    return hashlib.sha256(str(c).encode()).hexdigest()[:12]


def run(cmd, timeout=300) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise SystemExit(f"[a6] {' '.join(cmd[:6])}... failed: {r.stderr[:300]}")
    return r.stdout


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    rec = {"tool": "a6_select.py", "amendment": "A6", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "rule": "139 smallest sha12(clip_id) among train-cache clips not in the eval 139, with a train "
                   "label record and a train max-speed sidecar row", "pulled": {}}
    names = [x.strip() for x in run(SSH + [f"ls {TRAIN_DIR}"]).splitlines() if x.strip().endswith(".v2ep.pt")]
    train_ids = {n[: -len(".v2ep.pt")] for n in names}
    rec["n_train_cache_clips"] = len(train_ids)
    for k, remote in FILES.items():
        local = LOCAL / Path(remote).name
        rm = run(SSH + [f"md5sum {remote}"]).split()[0]
        run(["scp", "-q", "-o", "BatchMode=yes", f"{H}:{remote}", str(local)], timeout=600)
        lm = L.md5_file(local)
        if lm != rm or len(lm) != 32:
            raise SystemExit(f"[a6] md5 mismatch for {remote}: local {lm} remote {rm}")
        rec["pulled"][k] = {"remote": remote, "local": str(local), "md5": lm}
    from tanitad.data import v7_labels as v7l
    from tanitad.data.v2_dataset import load_or_build_manifest
    labs, lman = v7l.load_v7_labels(str(LOCAL / Path(FILES["labels"]).name), allow_oracle_nav=True)
    lab_ids = {str(l.clip_id) for l in labs}
    side_ids = set()
    for line in open(LOCAL / Path(FILES["sidecar"]).name, encoding="utf-8"):
        if line.strip():
            side_ids.add(str(json.loads(line)["clip_id"]))
    eval_ids = {str(c) for c in load_or_build_manifest(str(L.KIT / "data/refcv6-b1-416x1024-eval139"),
                                                       verbose=False)["clip_id"]}
    cand = sorted((train_ids - eval_ids) & lab_ids & side_ids, key=sha12)
    rec.update({"n_label_records": len(lab_ids), "n_sidecar_rows": len(side_ids), "n_eval": len(eval_ids),
                "n_train_in_eval": len(train_ids & eval_ids), "n_candidates": len(cand)})
    if len(cand) < N_SELECT:
        raise SystemExit(f"[a6] only {len(cand)} candidates (< {N_SELECT})")
    sel = cand[:N_SELECT]
    json.dump(sel, open(OUT / "train_clips.PRIVATE.json", "w", encoding="utf-8"), indent=0)
    json.dump([sha12(c) for c in sel], open(OUT / "train_clips_sha12.json", "w", encoding="utf-8"), indent=0)
    rec["selected_sha12_first_last"] = [sha12(sel[0]), sha12(sel[-1])]
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(rec, open(OUT / "select_record.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: rec[k] for k in ("n_train_cache_clips", "n_label_records", "n_sidecar_rows", "n_eval",
                                            "n_train_in_eval", "n_candidates")}, indent=1))


if __name__ == "__main__":
    main()
