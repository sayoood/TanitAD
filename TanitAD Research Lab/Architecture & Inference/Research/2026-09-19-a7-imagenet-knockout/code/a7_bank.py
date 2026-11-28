"""Bank the A7 panel into the repo package -- the Master Mind's list, per arm: config.json
(argv), metrics, the held-out dump; plus what the verdict needs to be re-derived offline.

Per arm -> ``raw/<arm>/``: ``config.json``, ``metrics.jsonl``, ``summary.json``,
``bn_recalib.json``, ``bn_recalib_stats.pt``, ``a7_arm_check.json``, ``train.log``, and
``eval_windows.sha12.jsonl.gz`` -- the dump with every ``episode_id`` replaced by
``sha12(str(episode_id))`` (clip identity only as sha12; the cluster structure the
bootstrap needs is preserved exactly). Panel level: ``panel.log``, ``a7_verdict.json``,
``verdict.out``, the pinned-code md5 lists.

⛔ Refuses to write any TEXT artifact carrying a full UUID (``tools/clipid_scan.UUID_RE``,
the repo's own guard), and never touches git: staging is a separate, conditional step
("gate staging on the scan"). ⛔ ``ckpt.pt`` (~600 MB) is NEVER banked -- it stays on the
dev box, and its path + md5 are recorded instead.

Usage: python a7_bank.py <panel_out_dir> <package_raw_dir>
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

ARMS = ("A7-IN-s0", "A7-RND-s0", "A7-IN-s1", "A7-RND-s1")
ARM_FILES = ("run/config.json", "run/metrics.jsonl", "run/summary.json", "run/bn_recalib.json",
             "run/bn_recalib_stats.pt", "a7_arm_check.json", "train.log")
PANEL_FILES = ("panel.log", "a7_verdict.json", "verdict.out", "code/MD5SUMS.txt",
               "code/RUNTREE_MD5SUMS.txt")


def _uuid_re():
    here = Path(__file__).resolve()
    for up in here.parents:
        if (up / "tools" / "clipid_scan.py").exists():
            sys.path.insert(0, str(up / "tools"))
            import clipid_scan
            return clipid_scan.UUID_RE
    raise SystemExit("⛔ tools/clipid_scan.py not found -- refusing to bank unscanned text")


def sha12(x) -> str:
    return hashlib.sha256(str(x).encode()).hexdigest()[:12]


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bank(panel: Path, raw: Path) -> dict:
    uuid_re = _uuid_re()
    staged, refused, missing = [], [], []

    def put_text(src: Path, dst: Path):
        txt = src.read_text(encoding="utf-8", errors="replace")
        if uuid_re.search(txt):
            refused.append(str(src))
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(txt, encoding="utf-8")
        staged.append(str(dst))

    for arm in ARMS:
        d = panel / arm
        for rel in ARM_FILES:
            s = d / rel
            if not s.exists():
                missing.append(str(s))
                continue
            t = raw / arm / Path(rel).name
            if s.suffix == ".pt":
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, t)
                staged.append(str(t))
            else:
                put_text(s, t)
        dump = d / "eval_windows.jsonl"
        if dump.exists():
            rows = []
            for line in dump.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    r["episode_sha12"] = sha12(r.pop("episode_id"))
                    rows.append(r)
            blob = ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")
            if uuid_re.search(blob.decode("utf-8")):
                refused.append(str(dump))
            else:
                t = raw / arm / "eval_windows.sha12.jsonl.gz"
                t.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(t, "wb") as f:
                    f.write(blob)
                staged.append(str(t))
        else:
            missing.append(str(dump))
        ck = d / "run" / "ckpt.pt"
        if ck.exists():
            (raw / arm).mkdir(parents=True, exist_ok=True)
            (raw / arm / "CKPT_NOT_BANKED.txt").write_text(
                "ckpt.pt stays on the dev box: %s  md5 %s  bytes %d\n"
                % (ck, _md5(ck), ck.stat().st_size), encoding="utf-8")
            staged.append(str(raw / arm / "CKPT_NOT_BANKED.txt"))
    for rel in PANEL_FILES:
        s = panel / rel
        if s.exists():
            put_text(s, raw / Path(rel).name)
        else:
            missing.append(str(s))
    rep = {"banked": staged, "refused_uuid": refused, "missing": missing}
    (raw / "BANK_MANIFEST.json").parent.mkdir(parents=True, exist_ok=True)
    (raw / "BANK_MANIFEST.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 2:
        raise SystemExit(__doc__)
    rep = bank(Path(a[0]), Path(a[1]))
    print("banked %d, refused (UUID) %d, missing %d" % (
        len(rep["banked"]), len(rep["refused_uuid"]), len(rep["missing"])))
    for k in ("refused_uuid", "missing"):
        for p in rep[k]:
            print("  %s: %s" % (k, p))
    return 1 if rep["refused_uuid"] else 0


if __name__ == "__main__":
    sys.exit(main())
