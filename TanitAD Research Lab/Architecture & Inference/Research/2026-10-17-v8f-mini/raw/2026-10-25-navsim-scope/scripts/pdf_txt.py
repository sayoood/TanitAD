import json, subprocess, sys, os, hashlib
import fitz
OUT = sys.argv[1]
keys = sys.argv[2:]
lib = json.load(open(os.path.join(OUT, "library_tip.json"), encoding="utf-8"))["entries"]
for k in keys:
    e = lib.get(k)
    if not e:
        print(k, "NOT IN LIBRARY"); continue
    path = e["path"]
    blob = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*", "show", "f547c11b:" + path],
                          capture_output=True).stdout
    sha = hashlib.sha256(blob).hexdigest()
    ok = (sha == e.get("sha256"))
    tp = os.path.join(OUT, "tmp_pdf", k + ".pdf")
    open(tp, "wb").write(blob)
    doc = fitz.open(tp)
    txt = []
    for i, pg in enumerate(doc, 1):
        txt.append("\n=====PAGE %d=====\n" % i + pg.get_text())
    open(os.path.join(OUT, "papertext", k + ".txt"), "w", encoding="utf-8").write("".join(txt))
    doc.close(); os.remove(tp)
    print(k, "pages", len(txt), "bytes", len(blob), "sha_match", ok, "|", e.get("title", "")[:60])
