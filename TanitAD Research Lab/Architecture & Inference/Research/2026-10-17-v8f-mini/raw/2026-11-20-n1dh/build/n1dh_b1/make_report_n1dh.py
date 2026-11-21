"""make_report_n1dh.py -- REPORT_B1.json from the banked records (junit files, mutation table, K-LIM record, the probe record of the K-LIM training run, the patch's file list)."""
import glob
import hashlib
import json
import os
import xml.etree.ElementTree as ET

H = "F:/TanitAD-scratch/n1dc/build/n1dh_b1"


def junit(path):
    r = ET.parse(path).getroot()
    ts = r if r.tag == "testsuite" else r.find("testsuite")
    cases = list(ts.iter("testcase"))
    bad = [c.get("classname", "") + "::" + c.get("name", "") for c in cases if c.find("failure") is not None or c.find("error") is not None]
    skipped = [c.get("name") for c in cases if c.find("skipped") is not None]
    return {"tests": len(cases), "failed": bad, "skipped": skipped}


def summary(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/*.junit.xml")):
        out[os.path.basename(f)[:-10]] = junit(f)
    return out


rep = {"spec": "PREREG_V8F_N1DH sha256 f1df05bc (verified first)", "base": "bcf3ba3f", "phase": "B1 (dev box, CPU, no GPU, nothing staged/committed/pushed)"}
rep["mutations"] = [{"mutant": r["mutant"], "named_tests_red": [t["test"] for t in r["tests"] if t["red"]], "all_named_red": r["named_test_red"], "file_restored": r["file_restored"]} for r in json.load(open(f"{H}/mutation_table_n1dh.json"))]
rep["klim_synthetic"] = json.load(open(f"{H}/records/klim/KLIM_synthetic.json"))
sc = json.load(open(f"{H}/records/klim/world/klim_arm/selfcheck.json"))
rep["probe_record_of_a_60_step_synthetic_cond_arm"] = sc["aux_route_probe"]
rep["summary_record_of_the_same_arm"] = {k: v for k, v in json.load(open(f"{H}/records/klim/world/klim_arm/summary.json")).items() if k.startswith(("route", "aux"))}
rep["regress_new"] = summary(f"{H}/regress_new")
rep["files"] = [l.split() for l in open(f"{H}/TREE_FILES_SHA256.txt", encoding="utf-8").read().splitlines()]
json.dump(rep, open(f"{H}/REPORT_B1.json", "w", encoding="utf-8"), indent=1)
print("report written", len(rep["mutations"]), "mutants", sum(len(v["failed"]) for v in rep["regress_new"].values()), "failures in", len(rep["regress_new"]), "files")
