"""Resolve the 3 train_v6_staged.py conflicts of merge_r3.py: every one is two INDEPENDENT additions at the same
anchor (a dry-run report block, a dry_run.json key, a config.json record), so the resolution is BOTH blocks,
merge copy first, then R3. Refuses unless: exactly 3 conflicts, no marker survives, the file parses, and a
fingerprint of every stream is present exactly where expected."""
import ast, sys
from pathlib import Path

M = Path("C:/Users/Admin/v7f_merge")
src = (M / "train_v6_staged.merged_LF.py").read_text(encoding="utf-8")
lines = src.split("\n")
out, i, n = [], 0, 0
while i < len(lines):
    ln = lines[i]
    if ln.startswith("<<<<<<< merge(R1R4+R2+R6)"):
        j = lines.index("=======", i)
        k = next(x for x in range(j, len(lines)) if lines[x].startswith(">>>>>>> R3"))
        out += lines[i + 1:j] + lines[j + 1:k]
        n += 1
        i = k + 1
        continue
    out.append(ln)
    i += 1
merged = "\n".join(out)
assert n == 3, f"expected 3 conflicts, resolved {n}"
for mk in ("<<<<<<<", ">>>>>>>", "\n=======\n"):
    assert mk not in merged, f"marker {mk!r} survived"
ast.parse(merged)
FINGERPRINTS = {
    "R1/R4 sidecar dry report": 'r1_report: dict = {"exercised": False,',
    "R3 dry report": "tac_label_report = None",
    "R1/R4 dry_run.json key": '**({"r1_r4": r1_report',
    "R3 dry_run.json key": '**({"tac_label_all": tac_label_report}',
    "R1 config.json record": 'cfg_json["r1_max_speed_join"] = vmax_join["report"]',
    "R3 config.json record": 'cfg_json["tac_label_all"] = tac_label_cfg',
    "R2 nav fix (emitter LAST)": "# ⭐ NAV (LAST), joined here because",
    "R6 flag": 'ap.add_argument("--strategic-off", action="store_true",',
    "R3 flag": '"--w-tac-label-all"',
    "R4 flag": '"--tac-op-cond"',
}
missing = [k for k, v in FINGERPRINTS.items() if merged.count(v) < 1]
assert not missing, f"fingerprints missing: {missing}"
orig = (M / "stack/scripts/train_v6_staged.py").read_bytes()
eol = "\r\n" if b"\r\n" in orig else "\n"
(M / "stack/scripts/train_v6_staged.py").write_bytes(merged.replace("\n", eol).encode("utf-8"))
print(f"resolved {n} conflicts (both sides kept); parses; {len(FINGERPRINTS)} fingerprints present; "
      f"written with EOL {'CRLF' if eol == chr(13) + chr(10) else 'LF'}; {len(out)} lines")
