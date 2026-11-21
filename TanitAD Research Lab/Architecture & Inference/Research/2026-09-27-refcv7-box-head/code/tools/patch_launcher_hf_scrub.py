"""The package's landed Thor launchers get the HF-token scrub (the Master Mind 2026-09-27: a real bearer token reached 7
gate-env logs through huggingface_hub's offline error text). After the one line that exports HF_HUB_OFFLINE, insert

    export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token_disabled
    unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

Refuses a file whose current bytes are not the TIP's blob (someone else changed it), a file without exactly one such
line, or a file that already scrubs. Keeps each file's line endings. usage: patch_launcher_hf_scrub.py <git dir> <tip>
<repo root> <package rel> <file rel to package> ..."""
import hashlib
import subprocess
import sys
from pathlib import Path

git_dir, tip, root, rel = sys.argv[1:5]
files = sys.argv[5:]
ADD = ["export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token_disabled",
       "unset HF_TOKEN HUGGING_FACE_HUB_TOKEN"]
for f in files:
    p = Path(root) / rel / f
    data = p.read_bytes()
    tip_blob = subprocess.run(["git", f"--git-dir={git_dir}", "rev-parse", f"{tip}:{rel}/{f}"], capture_output=True,
                              text=True).stdout.strip()
    own = hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()
    if len(tip_blob) != 40 or own != tip_blob:
        raise SystemExit(f"{f}: not the tip's blob ({own[:10]} vs {tip_blob[:10] or 'ABSENT'}) -- refusing")
    if b"HF_HUB_DISABLE_IMPLICIT_TOKEN" in data:
        raise SystemExit(f"{f}: already scrubs")
    nl = b"\r\n" if b"\r\n" in data else b"\n"
    lines = data.split(nl)
    hits = [i for i, ln in enumerate(lines) if ln.startswith(b"export ") and b"HF_HUB_OFFLINE=1" in ln]
    if len(hits) != 1:
        raise SystemExit(f"{f}: {len(hits)} HF_HUB_OFFLINE export lines, expected exactly 1")
    i = hits[0]
    lines[i + 1:i + 1] = [a.encode() for a in ADD]
    out = nl.join(lines)
    p.write_bytes(out)
    print(f"{f}: +2 lines after line {i + 1} ({'CRLF' if nl == b'\r\n' else 'LF'})")
