"""Real-data check of the recorded prefix floor, run on an export of the tip.

Green on legacy, red on growth, refused on the wrong list, plus the two mixed-version
cases (old tool + new baseline, new tool + old baseline) so the landing order is known
to be safe. Every scenario runs the CLI in a subprocess exactly as an operator would.

⛔ No clip id or prefix is printed or written. The probes that ADD one choose it in
memory; the report carries paths, counts, verdicts and list digests only, and is
leak-scanned before it is written.

    python verify_floor.py --export <dir> --tool <new clipid_scan.py> --baseline <new>
        --old-tool <tip clipid_scan.py> --old-baseline <tip baseline> --corpus <ids.txt>
        --index <ids.txt> --mirror <camera dir> --universe <clip_universe.json>
        --scratch <dir> --out <report.json>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

_P8 = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])")
_EXC = re.compile(r"^(?:\w+\.)?(LeakGrew|ClipListMismatch|\w+Error): ", re.M)


def cli(tool, root, baseline, clips=None) -> dict:
    argv = [sys.executable, str(tool), "--root", str(root), "--baseline", str(baseline)]
    if clips:
        argv += ["--clips", str(clips)]
    r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if r.returncode == 0:
        rep = json.loads(r.stdout)
        return {"exit": 0, "verdict": rep["verdict"], "files_with_ids": rep["files_with_ids"],
                "total_uuids": rep["total_uuids"], "total_prefixes": rep["total_prefixes"],
                "n_shrank": len(rep["shrank"]), "shrank_head": rep["shrank"][:3]}
    m = _EXC.search(r.stderr)
    msg = r.stderr[m.start():].strip() if m else r.stderr.strip()[-600:]
    return {"exit": r.returncode, "refused_with": m.group(1) if m else "?",
            "message": msg[:900]}


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("export", "tool", "baseline", "old-tool", "old-baseline", "corpus", "index",
              "mirror", "universe", "scratch", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    scratch = Path(a.scratch)
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    corpus = Path(a.corpus).read_text(encoding="utf-8").split()
    index = Path(a.index).read_text(encoding="utf-8").split()
    cset, pre = set(corpus), {i[:8] for i in corpus}
    base = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
    R = {}

    R["S1 default run (UUID half), new tool + new baseline"] = cli(a.tool, a.export, a.baseline)
    R["S2 --clips v7 corpus list, new tool + new baseline"] = cli(a.tool, a.export, a.baseline, a.corpus)

    # S3: the list REBUILT the way the next operator will: from the mirror's file names,
    # reversed, with a BOM and CRLF (a PowerShell-written file).
    rebuilt = [p.name[:36] for p in sorted(Path(a.mirror).glob("*.mp4"))][::-1]
    f3 = scratch / "corpus_rebuilt_bom_crlf.txt"
    f3.write_bytes(b"\xef\xbb\xbf" + ("\r\n".join(rebuilt) + "\r\n").encode("utf-8"))
    R["S3 --clips list rebuilt from mp4 names (reversed, BOM, CRLF)"] = cli(a.tool, a.export, a.baseline, f3)

    R["S4 --clips 306,152-id index (the wrong list)"] = cli(a.tool, a.export, a.baseline, a.index)

    # S5: same size, one id swapped for an index id outside the corpus
    rng = random.Random(20260919)
    outside = sorted(set(index) - cset)
    swapped = sorted(cset)
    swapped[rng.randrange(len(swapped))] = outside[rng.randrange(len(outside))]
    f5 = scratch / "corpus_one_swapped.txt"
    f5.write_text("\n".join(swapped) + "\n", encoding="utf-8")
    R["S5 --clips same size, ONE id swapped"] = cli(a.tool, a.export, a.baseline, f5)

    # S6-S8 on a COPY of the export
    tree = scratch / "tree"
    shutil.copytree(a.export, tree)
    with_pre = sorted(k for k, v in base.items() if not k.startswith("_") and v["prefixes"] > 0)
    p_add = sorted(pre)[rng.randrange(len(pre))]          # chosen in memory, never written out
    f6 = tree / with_pre[0]
    orig6 = f6.read_bytes()
    f6.write_bytes(orig6 + f"\n\nsee clip {p_add} for the bend\n".encode("utf-8"))
    R["S6 growth: +1 prefix in a file that already carries some"] = {"file": with_pre[0], **cli(a.tool, tree, a.baseline, a.corpus)}
    f6.write_bytes(orig6)

    f7 = tree / "Project Steering" / "zz_floor_probe_new_file.md"
    f7.write_text(f"clip {p_add} was inspected\n", encoding="utf-8")
    R["S7 growth: a NEW file carrying one prefix"] = cli(a.tool, tree, a.baseline, a.corpus)
    f7.unlink()

    # S8: a cleanup: remove ONE standalone prefix (not the head of a full UUID)
    cleaned = None
    for rel in with_pre:
        t = (tree / rel).read_text(encoding="utf-8", errors="replace")
        hit = next((m for m in _P8.finditer(t) if m.group(0) in pre
                    and t[m.end():m.end() + 1] != "-"), None)
        if hit:
            cleaned = rel
            orig8 = (tree / rel).read_bytes()
            (tree / rel).write_text(t[:hit.start()] + "xxxxxxxx" + t[hit.end():],
                                    encoding="utf-8", newline="")
            break
    R["S8 cleanup: -1 prefix in one file (allowed, reported)"] = {"file": cleaned, **cli(a.tool, tree, a.baseline, a.corpus)}
    (tree / cleaned).write_bytes(orig8)
    R["S8b the copy restored: back to green"] = cli(a.tool, tree, a.baseline, a.corpus)

    # mixed versions: what each landing order would show
    R["S9a OLD tool + NEW baseline, default"] = cli(a.old_tool, a.export, a.baseline)
    R["S9b OLD tool + NEW baseline, --clips corpus"] = cli(a.old_tool, a.export, a.baseline, a.corpus)
    R["S9c OLD tool + NEW baseline, --clips index"] = cli(a.old_tool, a.export, a.baseline, a.index)
    R["S10a NEW tool + OLD baseline, default"] = cli(a.tool, a.export, a.old_baseline)
    R["S10b NEW tool + OLD baseline, --clips corpus"] = cli(a.tool, a.export, a.old_baseline, a.corpus)

    text = json.dumps(R, indent=1, ensure_ascii=False)
    # leak scan of the REPORT before it is written: ids, 8-hex prefixes, 12-hex tails
    uni = set(json.loads(Path(a.universe).read_text(encoding="utf-8"))["ids"]) | cset | set(index)
    heads, tails = {u[:8] for u in uni}, {u[-12:] for u in uni}
    bad = [i for i in uni if i in text] + [t for t in _P8.findall(text) if t in heads]
    bad += [t for t in re.findall(r"(?<![0-9a-f])[0-9a-f]{12}(?![0-9a-f])", text) if t in tails]
    if bad:
        raise SystemExit(f"⛔ the report would carry {len(bad)} identifier(s); not written")
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    shutil.rmtree(scratch, ignore_errors=True)
    for k, v in R.items():
        brief = v.get("verdict") or f"REFUSED {v.get('refused_with')}"
        extra = (f"uuids {v['total_uuids']} prefixes {v['total_prefixes']} files {v['files_with_ids']} shrank {v['n_shrank']}"
                 if v.get("exit") == 0 else v["message"].splitlines()[0][:150])
        print(f"{k:<62} exit {v['exit']}  {brief:<26} {extra}")
    print(f"report leak scan over {len(uni)} ids: 0 hits; written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
