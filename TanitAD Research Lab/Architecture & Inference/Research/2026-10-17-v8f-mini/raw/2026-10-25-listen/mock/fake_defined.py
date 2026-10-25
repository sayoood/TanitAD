import json, random
from pathlib import Path
H = Path(__file__).resolve().parent
R = json.load(open(H / "READ_mock.json"))
random.seed(1)
for s in ("s0", "s1"):
    for row, ent in R["grid"][s]["rows"].items():
        ent["all_defined"] = True
        ent["cells"]["all"].update(delta=3.0 if row in ("S", "O") else 0.5)
        for col, c in ent["cells"].items():
            if c is None or col == "all": continue
            n = random.random() * 1.2
            m = 3.0 if row in ("S", "O") else 0.5
            c.update(n=n, n_clipped=min(n, 1.0), delta=n * m, lo=n * m - 0.3, hi=n * m + 0.3)
for row in R["analysis"]["listens"]:
    for col in R["analysis"]["listens"][row]:
        R["analysis"]["listens"][row][col] = None if R["grid"]["s0"]["rows"][row]["cells"][col] is None else random.random() > 0.5
json.dump(R, open(H / "READ_fake.json", "w"))
