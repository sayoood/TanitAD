"""Mechanical extraction (no hand-typing) of the bank generator's `--limit-clips` / `partial` sites from v8f_dagger_bank.py (read-only, tree dagt1_tree_b43c7a9a)."""
import hashlib, pathlib
SRC = pathlib.Path("E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/stack/scripts/v8f_dagger_bank.py")
OUT = pathlib.Path("C:/Users/Admin/dagread/fixtures/generator_a2/LIMIT_SITES.txt")
b = SRC.read_bytes()
lines = b.decode("utf-8").split("\n")
want = []
for i, ln in enumerate(lines, 1):
    if "limit_clips" in ln or "limit-clips" in ln or '"partial"' in ln:
        want.append((i, ln))
hdr = [f"# verbatim lines of {SRC.as_posix()} (read-only) that mention limit_clips / limit-clips / \"partial\"", f"# source file sha256 {hashlib.sha256(b).hexdigest()} ({len(b)} bytes)",
       "# format: <line number>\t<line, byte for byte>"]
OUT.write_bytes(("\n".join(hdr + [f"{i}\t{ln}" for i, ln in want]) + "\n").encode("utf-8"))
print(len(want), "lines;", [i for i, _ in want])
