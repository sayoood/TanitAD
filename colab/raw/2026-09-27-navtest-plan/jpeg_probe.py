"""Cross-OS parity + speed of the seam's image preprocessing -- `planner._image_for` up to the float cast:
cv2.imdecode(IMREAD_COLOR) -> cv2.resize((960, 512)) -> BGR->RGB, on every JPEG of a zip.

    python jpeg_probe.py <zip>      prints ZZJPEG {"hashes": {member: sha256 of the uint8 array}, ...}
Run on the dev box and on a Colab VM with the same zip: equal hashes mean the model sees byte-identical inputs.
"""
import hashlib
import json
import sys
import time
import zipfile

import cv2
import numpy as np


def prep(b):
    im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(cv2.resize(im, (960, 512))[:, :, ::-1])


def main() -> int:
    z = zipfile.ZipFile(sys.argv[1])
    names = sorted(z.namelist())
    data = {n: z.read(n) for n in names}
    out = {n: hashlib.sha256(prep(data[n]).tobytes()).hexdigest() for n in names}
    reps = 20
    t = time.perf_counter()
    for _ in range(reps):
        for n in names:
            prep(data[n])
    dt = (time.perf_counter() - t) / reps
    print("ZZJPEG " + json.dumps({"hashes": out, "decode_resize_all_members_s": round(dt, 4), "n_members": len(names),
                                  "cv2": cv2.__version__, "numpy": np.__version__, "cv2_threads": cv2.getNumThreads()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
