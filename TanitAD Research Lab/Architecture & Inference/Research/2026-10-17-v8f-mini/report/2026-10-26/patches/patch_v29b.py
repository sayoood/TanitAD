"""v29b: restore five provenance tokens the v27/v28 status rewrites dropped (pre-registration and gate commits), so the repo copy is a superset of v26."""
R = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/report/"
p = R + "V8F_SCIENTIFIC_REPORT.md"
s = open(p, encoding="utf-8").read()
E = [
    ("seed 1 not trained, as committed; `RESULT_V8F_OACC.md` |",
     "seed 1 not trained, as committed; `RESULT_V8F_OACC.md`; pre-registered `PREREG_V8F_OACC` (`17fc9abb`) with Amendment 1 (`ea67b7a0`),"
     " launch gate PASS (`bf43fd58`), training began 2026-10-26 06:13 |"),
    ("(§7.18; `RESULT_V8F_ORECOVER.md`, `8c10066a`; 104 / 104 controls PASS).",
     "(§7.18; `RESULT_V8F_ORECOVER.md`, `8c10066a`; 104 / 104 controls PASS; pre-registered at `28e28c88` with Amendments 1–4)."),
    ("(§7.17; all controls PASS after Amendment 2, which is post-data and control-only); committed next: O-RECOVER |",
     "(§7.17; all controls PASS after Amendment 2, which is post-data and control-only; pre-registered with Amendment 1 at `19d7e069`);"
     " committed next: O-RECOVER |"),
]
for a, b in E:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
t = open(R + "page_template.html", encoding="utf-8").read()
open(R + "v8f_report.html", "w", encoding="utf-8").write(t.replace("%%MD%%", s))
print("v29b ok")
