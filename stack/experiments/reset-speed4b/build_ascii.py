"""Render the trajectory comparisons as ASCII grids (text-only, always visible).
G = ground truth, R = REF-A, F = flagship, X = ego start. Up = ahead."""
import json
from pathlib import Path

D = Path(__file__).resolve().parent
refa = {s["ep"]: s for s in json.loads((D / "refa_coords.json").read_text())}
flag = {s["ep"]: s for s in json.loads((D / "flagship_coords.json").read_text())}

ORDER = [(271, "sharp left turn"), (239, "sharp right turn"), (3, "sharp left turn"),
         (28, "gentle curve, 18 m/s"), (166, "gentle curve, slow"),
         (148, "straight, 18 m/s"), (47, "straight, slow"), (31, "straight, 36 m/s")]
ROWS, COLS = 15, 31


def plot(ep, desc):
    gt = [tuple(p) for p in refa[ep]["gt"]]
    ra = [tuple(p) for p in refa[ep]["pred"]]
    fl = [tuple(p) for p in flag[ep]["pred"]]
    allp = [(0.0, 0.0)] + gt + ra + fl
    fmax = max(p[0] for p in allp)
    lmin = min(p[1] for p in allp); lmax = max(p[1] for p in allp)
    lr = max(lmax - lmin, 1.0); lmin -= lr * 0.12; lmax += lr * 0.12
    grid = [[" "] * COLS for _ in range(ROWS)]

    def put(path, ch):
        for f, l in path:
            r = ROWS - 1 - round(f / (fmax + 1e-9) * (ROWS - 1))
            c = round((l - lmin) / (lmax - lmin + 1e-9) * (COLS - 1))
            r = min(max(r, 0), ROWS - 1); c = min(max(c, 0), COLS - 1)
            grid[r][c] = ch
    put(fl, "F"); put(ra, "R"); put(gt, "G")
    r0 = ROWS - 1 - round(0 / (fmax + 1e-9) * (ROWS - 1))
    c0 = round((0 - lmin) / (lmax - lmin + 1e-9) * (COLS - 1))
    grid[min(max(r0, 0), ROWS - 1)][min(max(c0, 0), COLS - 1)] = "X"
    body = "\n".join("|" + "".join(row) + "|" for row in grid)
    ra_ade, fl_ade = refa[ep]["ade"], flag[ep]["ade"]
    win = "flagship" if fl_ade < ra_ade else "REF-A"
    head = (f"ep {ep:05d}  {desc}   [{fmax:.0f} m ahead x {lmax-lmin:.0f} m wide]\n"
            f"REF-A ADE {ra_ade:.2f} m   flagship ADE {fl_ade:.2f} m   -> {win} tighter")
    top = "+" + "-" * COLS + "+   ^ = ahead"
    bot = "+" + "-" * COLS + "+"
    return head + "\n" + top + "\n" + body + "\n" + bot


out = ["G = ground truth   R = REF-A(21k)   F = flagship(3k)   X = ego (start)"]
for ep, desc in ORDER:
    out.append(plot(ep, desc))
txt = "\n\n".join(out)
(D / "ascii_plots.txt").write_text(txt, encoding="utf-8")
print(txt)
