"""Open a navhard scene pickle on Windows and find the ego state.

⛔ The pickles were written on Linux and carry `pathlib.PosixPath`, which CPython refuses to
instantiate on Windows (`UnsupportedOperation: cannot instantiate 'PosixPath' on your system`).
The data is fine; the PATH CLASS is not portable. Fix: an Unpickler that maps PosixPath ->
PurePosixPath, which holds the same string and is instantiable everywhere. Nothing else is
touched, and no file is rewritten.
"""
from __future__ import annotations

import io
import pathlib
import pickle
import sys


class WinUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name in ("PosixPath", "WindowsPath"):
            return pathlib.PurePosixPath
        return super().find_class(module, name)


def load(p):
    return WinUnpickler(io.BytesIO(pathlib.Path(p).read_bytes())).load()


def walk(o, prefix="", depth=0, out=None):
    if out is None:
        out = []
    if depth > 3:
        return out
    if hasattr(o, "__dataclass_fields__"):
        for f in o.__dataclass_fields__:
            walk(getattr(o, f, None), f"{prefix}.{f}", depth + 1, out)
    elif isinstance(o, dict):
        for k in list(o)[:40]:
            walk(o[k], f"{prefix}[{k!r}]", depth + 1, out)
    elif isinstance(o, (list, tuple)):
        out.append((prefix, f"{type(o).__name__}[{len(o)}]", ""))
        if o:
            walk(o[0], prefix + "[0]", depth + 1, out)
    else:
        out.append((prefix, type(o).__name__, str(o)[:70]))
    return out


if __name__ == "__main__":
    o = load(sys.argv[1])
    print("top type:", type(o).__name__, "module:", type(o).__module__)
    for name, t, v in walk(o, "scene"):
        print(f"  {name:58s} {t:22s} {v}")
