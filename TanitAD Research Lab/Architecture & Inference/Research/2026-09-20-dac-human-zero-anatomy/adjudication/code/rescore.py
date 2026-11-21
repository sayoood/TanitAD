"""Re-derive the DAC adjudication scoring from the key INDEPENDENTLY of the DataFlyWheel's
own script, and add the clustering read its draw does not account for.

Two things are checked that a per-window count cannot see:
  1. CLIP CLUSTERING -- the draw sampled WINDOWS, so n windows is not n independent
     observations. CLAUDE.md's estimator doctrine is episode-cluster for exactly this.
  2. The BORDERLINE token count in the landed CSV, which the RESULT summarises as 10.
"""
from __future__ import annotations

import collections
import csv
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
KEY = HERE / "key.txt"
LABELS = HERE / "LABELS_MASTERMIND.csv"

key = {}
for line in KEY.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    wid, stratum, fires, sha12, tick = line.split()
    key[wid] = {"stratum": stratum, "fires": "" if fires == "-" else fires,
                "clip": sha12, "tick": tick}

labels, notes = {}, {}
for row in csv.reader(l for l in LABELS.read_text(encoding="utf-8").splitlines()
                      if l and not l.startswith("#") and l.startswith("W")):
    labels[row[0]] = row[1]
    notes[row[0]] = ",".join(row[2:])

assert set(labels) == set(key), f"label/key mismatch: {set(labels) ^ set(key)}"
print(f"windows: {len(key)}  (control: labels and key cover the same ids)")

# --- 1. the three label cells, per stratum -------------------------------- #
cells = collections.defaultdict(collections.Counter)
for w, k in key.items():
    cells[k["stratum"]][labels[w]] += 1
print("\nper stratum  on-surface / over-boundary / cannot-tell")
for s in "ABCD":
    c = cells[s]
    print(f"  {s}  {c['on-surface']:2d} / {c['over-boundary']:2d} / {c['cannot-tell']:2d}")

# --- 2. agreement, re-derived --------------------------------------------- #
print("\ncandidate   fires  false-fire  false-pass  agree  agree%")
for cand, tok in (("V0", "V"), ("P1", "1"), ("P2", "2")):
    fired = [w for w, k in key.items() if tok in k["fires"]]
    human_over = [w for w in key if labels[w] == "over-boundary"]
    ff = [w for w in fired if labels[w] == "on-surface"]
    fp = [w for w in human_over if tok not in key[w]["fires"]]
    agree = len(key) - len(ff) - len(fp)
    print(f"  {cand:3s}       {len(fired):3d}      {len(ff):3d}        {len(fp):3d}"
          f"      {agree:3d}   {100*agree/len(key):5.1f} %")

# --- 3. CLIP CLUSTERING, the read the draw does not account for ----------- #
print("\nCLIP CLUSTERING -- windows are NOT independent observations")
tot_clips = len({k["clip"] for k in key.values()})
print(f"  60 windows come from {tot_clips} distinct clips")
for s in "ABCD":
    ws = [w for w in key if key[w]["stratum"] == s]
    clips = collections.Counter(key[w]["clip"] for w in ws)
    top = clips.most_common(1)[0]
    print(f"  stratum {s}: {len(ws):2d} windows from {len(clips):2d} clips"
          f"  | largest cluster {top[1]} windows ({100*top[1]/len(ws):.0f} %)")
ff_p2 = [w for w in key if "2" in key[w]["fires"]]
c = collections.Counter(key[w]["clip"] for w in ff_p2)
print(f"\n  P2's {len(ff_p2)} false fires are {len(c)} DISTINCT CLIPS: "
      + ", ".join(f"{k}x{v}" for k, v in c.most_common()))

# --- 4. the BORDERLINE token count ---------------------------------------- #
bl = sorted(w for w in notes if "BORDERLINE" in notes[w])
print(f"\nBORDERLINE-flagged rows in the CSV: {len(bl)} -> {', '.join(bl)}")
by_s = collections.Counter(key[w]["stratum"] for w in bl)
print("  by stratum: " + ", ".join(f"{s} {by_s[s]}" for s in "ABCD" if by_s[s]))
