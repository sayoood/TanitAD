"""P0 - FACTS ABOUT THE APPARATUS, established BEFORE the pre-registration's
regression expectations are committed.

⛔ Nothing here scores an arm. It reads the INCUMBENT 117-candidate bank out of
the checkpoint and answers exactly two questions that a committed expectation
depends on:

  1. does the bank contain an (a_lon, a_lat) = (0, 0) candidate?  -> decides
     whether the R4 degenerate arm (t1 = t2 = 0, curvature identically zero)
     is a STRUCTURAL zero or a genuine new shape;
  2. is the a_lat grid symmetric?  -> decides whether R2 (t1 = 0, t2 = horizon,
     i.e. -1 everywhere) lands on candidates already in the bank.

Establishing the apparatus before writing the expectation is what lets both
outcomes be committed. It is not a result about the three-segment family.
"""
import sys

import torch

import _env  # noqa: F401

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CKPT = r"C:\Users\Admin\refcv4b_final\ckpt_40284_FINAL.pt"


def main():
    print("[env]", _env.check())
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    d = sd.get("model", sd)
    C0 = d["core.decoder.anchor_controls"].float()
    A = d["core.decoder.anchors"].float()
    print(f"[bank] controls {tuple(C0.shape)}  anchors {tuple(A.shape)}")

    lon = sorted({round(float(x), 6) for x in C0[:, 0]})
    lat = sorted({round(float(x), 6) for x in C0[:, 1]})
    print(f"[bank] a_lon grid ({len(lon)}): {lon}")
    print(f"[bank] a_lat grid ({len(lat)}): {lat}")

    zero = ((C0[:, 0] == 0.0) & (C0[:, 1] == 0.0))
    print(f"[Q1] rows with (a_lon, a_lat) == (0, 0): {int(zero.sum())}  "
          f"indices {torch.nonzero(zero).reshape(-1).tolist()}")

    lat_t = torch.tensor(lat)
    sym = bool(torch.allclose(lat_t, -lat_t.flip(0)))
    print(f"[Q2] a_lat grid symmetric about 0: {sym}")
    # symmetry must hold PAIRWISE too: for every row, is (a_lon, -a_lat) present?
    rows = {(round(float(a), 6), round(float(b), 6)) for a, b in C0}
    missing = [r for r in sorted(rows) if (r[0], -r[1]) not in rows]
    print(f"[Q2] rows whose (a_lon, -a_lat) mirror is ABSENT: {len(missing)}"
          f"{'' if not missing else '  ' + str(missing[:8])}")
    print(f"[Q2] distinct (a_lon, a_lat) rows: {len(rows)} of "
          f"{C0.shape[0]} candidates")
    print(f"[Q3] a_lat = 0 sub-fan size: "
          f"{int((C0[:, 1] == 0.0).sum())} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
