"""Whole-clip A/B: is the 60 -> 64 difference the CODE PATH or the SESSION?

Runs the clip's full run-frame set both ways IN ONE SESSION:
  A  fresh set_image per concept   (vendor README usage; the banked path)
  B  one set_image per frame       (encode-once)
and compares both against each other AND against the record banked from the
PREVIOUS VM. If A == B but both differ from the banked record, the difference
is cross-session nondeterminism, not the optimisation.
"""
import json
import os
import shutil
import sys
import time

os.chdir("/content")
for p in ("/content/repo/colab", "/content/repo/stack",
          "/content/repo/stack/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
import ph0_sam3                                                  # noqa: E402
import ph0_pilot                                                 # noqa: E402
import s2_lab_lib as L                                           # noqa: E402
from PIL import Image                                            # noqa: E402

CID = json.load(open("/content/repo/colab/fixtures/"
                     "sam3_backfill_expected.json"))["clips"][0]
mp4 = "/content/eqchk.mp4"
if not os.path.exists(mp4):
    shutil.copyfile(L.hf_download(
        L.DS_LABELS, f"bridged_w120train_2400/videos/{CID}.mp4"), mp4)
frames = ph0_pilot.sample_clip_frames(mp4, t0_s=8.0)[0]
banked = json.load(open(L.hf_download(
    L.DS_LABELS, f"{L.BACKFILL_PREFIX}{CID}.json", force=True)))
keys = sorted(int(k) for k in banked["frames"])
print("[eq3] banked run frames:", keys,
      "n_det", banked["n_det_total"], banked["per_concept_hits"])
C = ph0_sam3.AGENT_CONCEPTS
if "PROC" not in globals():
    PROC, _m = L.load_sam3()


def run(mode):
    per, tot, t0 = {c: 0 for c in C}, 0, time.time()
    byframe = {}
    for fi in keys:
        img = Image.fromarray(frames[fi])
        if mode == "A":
            d = []
            for c in C:
                d.extend(ph0_sam3.detect(PROC, img, c))
        else:
            d = ph0_sam3.detect_many(PROC, img, C)
        byframe[fi] = sum(1 for r in d if "score" in r)
        for r in d:
            if "score" in r:
                per[r["concept"]] += 1
                tot += 1
    return per, tot, byframe, round(time.time() - t0, 1)


pA, tA, fA, sA = run("A")
pB, tB, fB, sB = run("B")
pA2, tA2, fA2, sA2 = run("A")
print(f"[A ] {sA:5.1f}s tot={tA} {pA}\n    per-frame {fA}")
print(f"[B ] {sB:5.1f}s tot={tB} {pB}\n    per-frame {fB}")
print(f"[A2] {sA2:5.1f}s tot={tA2} {pA2}\n    per-frame {fA2}")
print(f"[banked] tot={banked['n_det_total']} {banked['per_concept_hits']}")
print(f"    per-frame " + str({int(k): v['n_det']
                               for k, v in banked['frames'].items()}))
print("\n[eq3] A == B (encode-once safe)       :", pA == pB and fA == fB)
print("[eq3] A == A2 (deterministic in-session):", pA == pA2 and fA == fA2)
print("[eq3] A == banked (same across sessions):",
      pA == banked["per_concept_hits"] and tA == banked["n_det_total"])
json.dump({"clip_id": CID, "frames": keys,
           "A": pA, "B": pB, "A2": pA2,
           "banked": banked["per_concept_hits"],
           "tot": {"A": tA, "B": tB, "A2": tA2,
                   "banked": banked["n_det_total"]},
           "per_frame": {"A": fA, "B": fB, "A2": fA2,
                         "banked": {int(k): v["n_det"]
                                    for k, v in banked["frames"].items()}},
           "wall_s": {"A": sA, "B": sB, "A2": sA2,
                      "banked": banked.get("wall_s")},
           "A_eq_B": pA == pB and fA == fB,
           "A_eq_A2": pA == pA2 and fA == fA2,
           "A_eq_banked": pA == banked["per_concept_hits"]},
          open("/content/eq3.json", "w"), indent=1)
print("EQ3_DONE")
