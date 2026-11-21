from pathlib import Path
p = Path("C:/Users/Admin/refcv8_pf/refcv9_r9pf/code/x_attrib_r9pf.py"); s = p.read_text(encoding="utf-8")
a = """            out["cells"][tag]["decision_vs_r_star_S6_compliant"] = {
                f"s{s}": {"confusion_rows_rstar_cols_decision": [[int(((r_star == i) & (dec2[j6] == jj) & comp6).sum()) for jj in range(3)] for i in range(3)]}}"""
b = """            out["cells"][tag].setdefault("decision_vs_r_star_S6_compliant", {})[f"s{s}"] = {
                "confusion_rows_rstar_cols_decision": [[int(((r_star == i) & (dec2[j6] == jj) & comp6).sum()) for jj in range(3)] for i in range(3)]}"""
assert s.count(a) == 1; p.write_text(s.replace(a, b), encoding="utf-8"); print("ok")
