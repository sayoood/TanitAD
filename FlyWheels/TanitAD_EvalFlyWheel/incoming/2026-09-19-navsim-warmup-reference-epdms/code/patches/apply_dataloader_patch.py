#!/usr/bin/env python3
"""Apply the MetricCacheLoader separator fix to the C: COPY of the devkit (never the D: original).

Needed only when scoring runs in a PROCESS POOL: pool children do not inherit navsim_win.py's
in-process monkeypatch. Same semantics as navsim_win._load_metric_cache_paths_patched (unit-tested
in test_navsim_win_patch.py). Idempotent; records before/after blobs + a unified diff.
"""
import difflib, hashlib, io, json, subprocess, sys
from pathlib import Path
F = Path("C:/Users/Admin/navsim-crun/devkit/navsim/common/dataloader.py")
OLD = '        metric_cache_dict = {cache_path.split("/")[-2]: cache_path for cache_path in cache_paths}\n'
BS = chr(92)   # one backslash, spelled unambiguously (a heredoc ate one escape level in v1 of this file)
NEW = ('        # E1 (2026-09-19) Windows fix: nuPlan writes str(WindowsPath) (backslashes); split on either separator\n'
       '        metric_cache_dict = {__import__("re").split(r"[' + BS + BS + '/]", cache_path.strip())[-2]: '
       'cache_path for cache_path in cache_paths}\n')
assert NEW.count(BS) == 2 and 'r"[' + BS + BS + '/]"' in NEW, "regex must read r\"[<bs><bs>/]\" in the written file"
def blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()
raw = F.read_bytes()
crlf = b"\r\n" in raw
txt = raw.decode("utf-8").replace("\r\n", "\n")
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_suffix(".json")
if NEW in txt:
    print("already applied"); sys.exit(0)
assert txt.count(OLD) == 1, f"original line found {txt.count(OLD)} times - refusing"
new_txt = txt.replace(OLD, NEW)
new_raw = (new_txt.replace("\n", "\r\n") if crlf else new_txt).encode("utf-8")
F.write_bytes(new_raw)
diff = "".join(difflib.unified_diff(txt.splitlines(True), new_txt.splitlines(True), "a/navsim/common/dataloader.py", "b/navsim/common/dataloader.py"))
Path(out).with_suffix(".diff").write_text(diff, encoding="utf-8")
rec = {"file": str(F), "blob_before": blob(raw), "blob_after": blob(new_raw), "crlf": crlf,
       "d_devkit_untouched_blob": subprocess.run(["git", "hash-object", "D:/Archive/devbox-C/navsim/devkit/navsim/common/dataloader.py"], capture_output=True, text=True).stdout.strip()}
Path(out).write_text(json.dumps(rec, indent=1))
print(json.dumps(rec))
