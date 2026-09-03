"""Assemble RESULT.md = head + generated tables + tail, and drop it in the package."""
import os
import subprocess
import sys

SCR = (r"C:\Users\Admin\AppData\Local\Temp\claude"
       r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
       r"\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad")
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    r = subprocess.run([sys.executable, os.path.join(HERE, "make_tables.py")],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print(r.stdout[-4000:]); print(r.stderr[-4000:]); return 1
    tables = open(os.path.join(HERE, "tables.md"), encoding="utf-8").read()
    head = open(os.path.join(SCR, "RESULT_head.md"), encoding="utf-8").read()
    tail = open(os.path.join(SCR, "RESULT_tail.md"), encoding="utf-8").read()
    doc = (head.rstrip() + "\n\n---\n\n## 2-5. The measurement\n\n" +
           tables.strip() + "\n\n---\n\n" + tail.lstrip())
    out = os.path.join(HERE, "RESULT.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print("written", out, len(doc), "chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
