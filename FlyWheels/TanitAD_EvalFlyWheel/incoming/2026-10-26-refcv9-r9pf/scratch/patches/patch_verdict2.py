from pathlib import Path
p = Path("C:/Users/Admin/refcv8_pf/refcv9_r9pf/code/a3_verdict.py"); s = p.read_text(encoding="utf-8")
a = """         f"**committed outcome (prereg s5):** {OUTCOME[word]}", ""]"""
b = """         f"**committed outcome (prereg s5):** {OUTCOME[word]}", "",
         "**every bar, as written (primary R9PF_fs0):** " + ", ".join(f"{k} {'HOLDS' if v else 'FAILS'}" for k, v in b["bars_primary"].items())
         + f", PA-R {'HOLDS' if b['PA-R']['holds'] else 'FAILS'} -> PASS = {b['PASS']}.", ""]"""
assert s.count(a) == 1; p.write_text(s.replace(a, b), encoding="utf-8"); print("ok")
