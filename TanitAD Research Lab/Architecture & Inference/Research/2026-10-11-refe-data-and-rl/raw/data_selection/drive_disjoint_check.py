# Reproduces drive_disjoint_check_2026-10-11.txt. Run from the repo root (D:/Projects/TanitAD) with
# PYTHONIOENCODING=utf-8 and the tanitad venv. Reads only local nuPlan DB directory names and navtrain.yaml.
import os, yaml
base = "data/nuplan/nuplan-v1.1/splits"
test = [f[:-3] for f in os.listdir(base + "/test") if f.endswith(".db")]
val = [f[:-3] for f in os.listdir(base + "/trainval") if f.endswith(".db")]   # holds the nuPlan VAL split (arrange_splits.json)
d = yaml.safe_load(open("TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/splits/navtrain.yaml", encoding="utf-8"))
nt = d["log_names"]
drive = lambda n: n.rsplit("_", 2)[0]
print("navtrain logs", len(nt), "tokens", len(d["tokens"]), "distinct drive ids", len(set(map(drive, nt))))
print("nuPlan test DBs", len(test), "distinct drive ids", len(set(map(drive, test))))
print("navtrain logs present in nuPlan-test DBs  :", len(set(nt) & set(test)))
print("navtrain drive ids shared with test drives:", len(set(map(drive, nt)) & set(map(drive, test))))
print("navtrain logs present in nuPlan-val DBs   :", len(set(nt) & set(val)))
print("val drive ids shared with test drives     :", len(set(map(drive, val)) & set(map(drive, test))))
