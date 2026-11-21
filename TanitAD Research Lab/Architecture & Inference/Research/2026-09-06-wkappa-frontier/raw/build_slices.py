#!/usr/bin/env python3
"""Build EPISODE-DISJOINT slices for the curvature claim's episode replication.

⛔ The binding interval for the curvature claim is the EPISODE one (RESULT.md §4),
so the cheapest experiment that can move it is INDEPENDENT EPISODES, not seeds.

⚠️ These draws are NOT turn-enriched the way `p4` is (p4 was selected at
kappa_thr 0.04 for 17 real turns in 40 windows). That makes them a FAIRER test of
the curvature claim and a WEAKER one of the turn-recall claim, and the realised
n_true per class is printed so the reader can see which.
⛔ Label coverage is asserted BEFORE anything is built: an episode with no label
record silently becomes nav_cmd=0 / nav_valid=False, which would quietly change
what the tactical family measures. ASCII only.
"""
import gzip
import json
import os
import shutil
import sys

SLICE = r"C:\Users\Admin\refav1_eval_slice"
P4J = r"C:\Users\Admin\refav1_margin\p4_episodes.json"
LAB = r"C:\Users\Admin\navcomp\data\s2_labels_v7.2_eval.jsonl.gz"
OUT = r"C:\Users\Admin\wkfront"


def labelled_episodes():
    eps = set()
    with gzip.open(LAB, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            for k in ("episode", "clip_id", "episode_id", "clip", "name"):
                if k in r and isinstance(r[k], str):
                    eps.add(r[k])
    return eps


def main():
    p4 = set(json.load(open(P4J, encoding="utf-8"))["episodes"])
    have = sorted(n[:-len(".v2ep.pt")]
                  for n in os.listdir(os.path.join(SLICE, "eps"))
                  if n.endswith(".v2ep.pt"))
    disj = [e for e in have if e not in p4]
    lab = labelled_episodes()
    print("labels blob distinct episode ids: %d" % len(lab))
    print("p4 episodes covered by labels: %d/8" % len(p4 & lab))
    cov = [e for e in disj if e in lab]
    unc = [e for e in disj if e not in lab]
    print("disjoint available: %d  LABELLED: %d  UNLABELLED: %d"
          % (len(disj), len(cov), len(unc)))
    if len(p4 & lab) != 8:
        print("⛔ CONTROL FAILED: the labels blob does not cover p4's own 8 "
              "episodes, so this id-matching is reading the wrong field. "
              "Nothing built.".replace("\u26d4", "STOP:"))
        return 3
    if len(cov) < 16:
        print("Only %d labelled disjoint episodes -- building what is available."
              % len(cov))
    plan = {"dA": cov[:8], "dB": cov[8:16]}
    for name, eps in plan.items():
        if len(eps) < 8:
            print("SKIP %s: only %d episodes" % (name, len(eps)))
            continue
        for sub, ext in (("eps", ".v2ep.pt"), ("fp8", ".pt")):
            d = os.path.join(OUT, name, sub)
            os.makedirs(d, exist_ok=True)
            for e in eps:
                src = os.path.join(SLICE, sub, e + ext)
                dst = os.path.join(d, e + ext)
                if os.path.exists(dst):
                    continue
                try:
                    os.link(src, dst)          # hardlink: no copy, same volume
                except OSError:
                    shutil.copy2(src, dst)
        print("%s built: %d episodes -> %s" % (name, len(eps), os.path.join(OUT, name)))
        for e in eps:
            print("   %s" % e)
    json.dump(plan, open(os.path.join(OUT, "raw", "disjoint_slices.json"), "w",
                         encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
