"""MEASURE what the --ctx-cache (directory) fingerprint can and cannot see.  Uses gate_token.fingerprint/fp_key unchanged (the gate's own functions).
usage: probe_dir_fp_limit.py <refe dir>"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(sys.argv[1]))
import gate_token as GT  # noqa: E402

td = Path(tempfile.mkdtemp(prefix="fplimit_"))
cache = td / "cache"
cache.mkdir()
(cache / "manifest.json").write_text('{"tool": "ctx_cache/1"}', encoding="utf-8")
(cache / "shard0.bin").write_bytes(bytes(range(256)) * 4)
fk = lambda p: GT.fp_key(GT.fingerprint(str(p)))  # noqa: E731
base = fk(cache)
rows = []


def variant(name, mutate):
    d = td / name
    shutil.copytree(cache, d)
    mutate(d)
    rows.append((name, fk(d) != base))


def flip_same_size(d):
    b = bytearray((d / "shard0.bin").read_bytes())
    b[10] ^= 1
    (d / "shard0.bin").write_bytes(bytes(b))


def append_byte(d):
    with open(d / "shard0.bin", "ab") as f:
        f.write(b"x")


variant("same_size_byte_flip_in_payload", flip_same_size)
variant("payload_one_byte_longer", append_byte)
variant("file_added", lambda d: (d / "extra.bin").write_bytes(b"x"))
variant("file_removed", lambda d: (d / "shard0.bin").unlink())
variant("file_renamed", lambda d: (d / "shard0.bin").rename(d / "shard1.bin"))
for name, detected in rows:
    print(f"{name:34s} detected_by_dir_fingerprint={detected}")
# the bank (a file): every byte
bank = td / "bank.hnb"
bank.write_bytes(b"HNBANK1\n" + bytes(range(200)))
b0 = fk(bank)
f = bytearray(bank.read_bytes())
f[50] ^= 1
bank.write_bytes(bytes(f))
print(f"{'same_size_byte_flip_in_bank_file':34s} detected_by_file_fingerprint={fk(bank) != b0}")
shutil.rmtree(td, ignore_errors=True)
