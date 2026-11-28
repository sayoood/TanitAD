"""hierwhere2.py -- HIERWHERE-2 (PREREG_V8F_PROOF_ADD6B): the unchanged reader hier/hierwhere.py on a second banked pair.
Pass 1 (control h1a): arms (DAG, FLAT-DAG-0) against the banked difference.  Pass 2 (the read): arms (DAG0, FLAT-DAG-0).
The reader stops when its reproduction check has no banked value to meet, so pass 2 first asks the reader for its own value and hands that back as the reference:
for pass 2 that check is self-consistency only and is NOT counted as a control (h1a is the control)."""
import contextlib, io, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, os.path.dirname(HERE))
import hierwhere as H  # noqa: E402
ART = H.ART
FLAT0 = ART + "/v8f-flatdag-t1/dump_FLATDAG0_s{s}_m10_causal_heads"
# pass 1
H.DIRS = {"DAG": ART + "/v8f-dagger-t1/dump_DAG_s{s}_m10_causal_heads", "FLAT": FLAT0}
H.BANKED = {0: (-0.4973, -0.6075, -0.3874), 1: (-0.2450, -0.3553, -0.1316)}
rc1 = H.main(["run", "--out-dir", HERE + "/h1a"])
a = json.load(open(HERE + "/h1a/RESULT_HIERWHERE.json", encoding="utf-8"))
h1a = {s: a["controls"][f"h1_s{s}"] for s in (0, 1)}
print("H1A", json.dumps({s: (h1a[s]["PASS"], h1a[s]["reader"], h1a[s]["max_abs_dev_m"]) for s in h1a}))
# pass 2
H.DIRS = {"DAG": ART + "/v8f-dagger-t1/dump_DAG0_s{s}_m10_causal_heads", "FLAT": FLAT0}
H.BANKED = {0: (9.0, 9.0, 9.0), 1: (9.0, 9.0, 9.0)}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    try:
        H.main(["run", "--out-dir", HERE + "/read"])
    except SystemExit:
        pass
m = re.search(r"stopping before any read: (\{.*\})", buf.getvalue())
own = json.loads(m.group(1))
H.BANKED = {s: (own[f"h1_s{s}"]["reader"]["delta"], own[f"h1_s{s}"]["reader"]["lo"], own[f"h1_s{s}"]["reader"]["hi"]) for s in (0, 1)}
rc2 = H.main(["run", "--out-dir", HERE + "/read"])
print("RC", rc1, rc2)
