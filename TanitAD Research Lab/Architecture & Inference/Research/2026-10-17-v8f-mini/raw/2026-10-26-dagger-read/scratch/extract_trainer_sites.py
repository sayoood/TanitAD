"""Mechanical extraction (no hand-typing) of the trainer's write sites of train_window_list_sha256 from the integration tree's v8f_mini_train.py (read-only)."""
import hashlib, pathlib, re, sys
SRC = pathlib.Path("E:/Projects/TanitAD-artifacts/di/tree/stack/scripts/v8f_mini_train.py")
OUT = pathlib.Path("C:/Users/Admin/dagread/fixtures/trainer_kb45/WRITE_SITES.txt")
b = SRC.read_bytes()
lines = b.decode("utf-8").split("\n")
want = []
for i, ln in enumerate(lines, 1):
    if 'info["train_window_list_sha256"] =' in ln or '"train_window_list_sha256": info.get(' in ln or ('"train_sha256": info.get("train_sha256")' in ln and "excluded_sha256" in ln) \
       or ('"steps_total": total' in ln and '"info": info' in ln):
        want.append((i, ln))
hdr = [f"# verbatim lines of {SRC.as_posix()} (read-only, integration tree di/)", f"# source file sha256 {hashlib.sha256(b).hexdigest()} ({len(b)} bytes)",
       "# format: <line number>\t<line, byte for byte>"]
OUT.write_bytes(("\n".join(hdr + [f"{i}\t{ln}" for i, ln in want]) + "\n").encode("utf-8"))
print(len(want), "lines;", [i for i, _ in want])
