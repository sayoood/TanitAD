import sys, json
sys.path.insert(0, "C:/Users/Admin/swp/fampol/w/eval/p5c")
import p5c_read as R
j = json.load(open("E:/Projects/TanitAD-artifacts/p5c_read/out/families/famci_C1_vs_C0.json", encoding="utf-8"))
fr = j["frames"]["executed_tracked"]["families"]
for fam in ("longitudinal", "lateral", "tactical"):
    for k in fr[fam]:
        print(f"{fam}.{k} -> {R.fam_direction(k)}")
