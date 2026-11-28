#!/usr/bin/env python3
"""fill_pin.py: write launch_gate.SCORER1P6_DATA_PIN from the pod's labels_manifest.json, on a copy of the tip's launch_gate.py, byte-exact (LF kept).

  python fill_pin.py --manifest <labels_manifest.json> --refe <raw export of refe/ at the tip> --out <new launch_gate.py> [--report <json>]

KEY MAPPING (read from what the chain and the gate actually consume):
  labels_manifest.json data_expect.train_sets   -> SCORER1P6_DATA_PIN["train_sets"]    (chain: NTR, run_check start.train_sets; gate: train_sets, train_sets_in_bank, lane_labels.pdm_train)
  labels_manifest.json data_expect.heldout_sets -> SCORER1P6_DATA_PIN["heldout_sets"]  (chain: NHO; gate: heldout_sets, heldout_in_bank, lane_labels.pdm_heldout)
  CROSS-CHECKS (refuse, because the gate's registered counts would not match the manifest): train_sets_in_bank == pdm_train == train_sets; heldout_in_bank == pdm_heldout == heldout_sets;
  heldout_logs == 24; train_rows_in_heldout_logs == 0.  Not pinned, only reported: pdm_fit_extra, train_skipped_5s_guard, heldout_skipped_5s_guard.  A heldout count other than 3,137 is a WARNING (the label tool refuses it itself).
Exit 0 + ZZFILLPIN_OK on success; exit 1 + ZZFILLPIN_REFUSED <reason> otherwise.  Writes nothing on refusal.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

LINE = re.compile(rb"^SCORER1P6_DATA_PIN = None\n", re.M)
FP_CODE = ("import sys,hashlib,json\n"
           "sys.path[:0]=[p for p in sys.argv[1:] if p]\n"
           "import launch_gate as L\n"
           "print(json.dumps({n:hashlib.sha256(repr(sorted(L.PROFILES[n].items())).encode()).hexdigest() for n in sorted(L.PROFILES)}))\n"
           "print(json.dumps(L.SCORER1P6_DATA_PIN, sort_keys=True))\n")


def refuse(msg):
    print("ZZFILLPIN_REFUSED " + msg)
    sys.exit(1)


def fps(first, refe):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    r = subprocess.run([sys.executable, "-B", "-I", "-c", FP_CODE, first, refe], capture_output=True, env=env, encoding="utf-8")
    if r.returncode != 0:
        refuse("could not import launch_gate from %s: %s" % (first or refe, r.stderr.strip()[-300:]))
    a, b = r.stdout.strip().split("\n")[-2:]
    return json.loads(a), json.loads(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--refe", required=True, help="the tip's refe/ (raw blobs); its launch_gate.py is the base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    a = ap.parse_args()
    with open(a.manifest, encoding="utf-8") as fh:
        man = json.load(fh)
    de = man.get("data_expect")
    if not isinstance(de, dict):
        refuse("%s has no data_expect object" % a.manifest)
    need = ("train_sets", "train_sets_in_bank", "pdm_train", "heldout_sets", "heldout_in_bank", "pdm_heldout", "heldout_logs", "train_rows_in_heldout_logs")
    miss = [k for k in need if k not in de]
    if miss:
        refuse("data_expect lacks %s" % miss)
    if any(isinstance(de[k], bool) or not isinstance(de[k], int) for k in need):
        refuse("data_expect values must be integers: %s" % {k: de[k] for k in need})
    n, h = de["train_sets"], de["heldout_sets"]
    if n <= 0 or h <= 0:
        refuse("non-positive counts train %s heldout %s" % (n, h))
    if not (de["train_sets_in_bank"] == de["pdm_train"] == n):
        refuse("train counts disagree: sets %s in_bank %s pdm_train %s (the gate registers all three equal)" % (n, de["train_sets_in_bank"], de["pdm_train"]))
    if not (de["heldout_in_bank"] == de["pdm_heldout"] == h):
        refuse("heldout counts disagree: sets %s in_bank %s pdm_heldout %s" % (h, de["heldout_in_bank"], de["pdm_heldout"]))
    if de["heldout_logs"] != 24 or de["train_rows_in_heldout_logs"] != 0:
        refuse("heldout_logs %s (want 24) / train_rows_in_heldout_logs %s (want 0)" % (de["heldout_logs"], de["train_rows_in_heldout_logs"]))
    warn = []
    if h != 3137:
        warn.append("heldout_sets %d != 3137 (the label job's own expected count)" % h)
    if man.get("provisional"):
        warn.append("PROVISIONAL manifest: these counts are NOT real; do not ship")
    base_path = os.path.join(a.refe, "launch_gate.py")
    with open(base_path, "rb") as fh:
        raw = fh.read()
    if b"\r" in raw:
        refuse("%s contains CR: it is not the LF blob" % base_path)
    hits = LINE.findall(raw)
    if len(hits) != 1:
        refuse("expected exactly one 'SCORER1P6_DATA_PIN = None' line, found %d (already filled?)" % len(hits))
    new = LINE.sub(('SCORER1P6_DATA_PIN = {"train_sets": %d, "heldout_sets": %d}\n' % (n, h)).encode("ascii"), raw, count=1)
    if os.path.dirname(os.path.abspath(a.out)) == os.path.abspath(a.refe):
        refuse("--out must not be inside --refe (never edit the export in place)")
    d = tempfile.mkdtemp(prefix="fillpin_", dir=os.path.dirname(os.path.abspath(a.out)))
    tmp = os.path.join(d, "launch_gate.py")
    with open(tmp, "wb") as fh:
        fh.write(new)
    try:
        before, pin0 = fps("", a.refe)
        after, pin1 = fps(d, a.refe)
    finally:
        os.remove(tmp)
        os.rmdir(d)
    if pin0 is not None or pin1 != {"heldout_sets": h, "train_sets": n}:
        refuse("pin read back wrongly: before %s after %s" % (pin0, pin1))
    moved = sorted(k for k in before if before[k] != after.get(k))
    with open(a.out, "wb") as fh:
        fh.write(new)
    rep = {"train_sets": n, "heldout_sets": h, "warnings": warn, "out_sha256": hashlib.sha256(new).hexdigest(), "base_sha256": hashlib.sha256(raw).hexdigest(),
           "refe_scorer1p6_fp_before": before["refe_scorer1p6"], "refe_scorer1p6_fp_after": after["refe_scorer1p6"], "profiles": len(before),
           "other_profiles_moved": [k for k in moved if k != "refe_scorer1p6"], "moved": moved, "diff_bytes": len(new) - len(raw)}
    if a.report:
        with open(a.report, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rep, fh, indent=1, sort_keys=True)
    for w in warn:
        print("WARNING " + w)
    print("ZZFILLPIN_OK train %d heldout %d refe_scorer1p6 %s -> %s; %d other profile(s) moved %s; out sha256 %s" % (
        n, h, rep["refe_scorer1p6_fp_before"][:16], rep["refe_scorer1p6_fp_after"][:16], len(rep["other_profiles_moved"]), rep["other_profiles_moved"], rep["out_sha256"][:16]))
    return 0 if not rep["other_profiles_moved"] else 2


if __name__ == "__main__":
    sys.exit(main())
