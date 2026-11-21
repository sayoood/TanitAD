import io, os, sys
sp = os.path.dirname(os.path.abspath(__file__))
t = io.open(os.path.join(sp, "template.html"), encoding="utf-8").read()
d = io.open(os.path.join(sp, "report_data.json"), encoding="utf-8").read().replace("</", "<" + chr(92) + "/")
assert t.count("/*__DATA__*/") == 1
io.open(os.path.join(sp, "refe-training-run.html"), "w", encoding="utf-8", newline="\n").write(t.replace("/*__DATA__*/", d))
print("ok", len(t) + len(d))
