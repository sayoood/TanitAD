"""MUTATION PROOF: can the panel's defect gate actually fail?

refcv3_arm.py:2633 writes the defect at  rec["refcv3"]["_defects"]  (ref is
assigned to rec["refcv3"] at :2445 and :2712).
run_hierarchy_panel.py:71 reads          rec["_defects"]            (top level).
"""
import json, os, sys, tempfile
sys.path.insert(0, r"C:/Users/Admin/tanitad-review-20260906/taniteval")
import importlib.util as _iu
_s=_iu.spec_from_file_location("rhp", r"C:/Users/Admin/tanitad-review-20260906/taniteval/tools/run_hierarchy_panel.py")
_m=_iu.module_from_spec(_s); _s.loader.exec_module(_m); record_ok=_m.record_ok

d = tempfile.mkdtemp()
DEFECT = [{"where": "strategic.nav_compliance", "type": "TypeError",
           "detail": "os.path.exists(<dict>) — the historical STRATEGIC hole"}]

def write(name, rec):
    p = os.path.join(d, name)
    json.dump(rec, open(p, "w", encoding="utf-8"))
    return p

# --- ARM A: the shape refcv3_arm.py ACTUALLY writes -------------------------
a = write("as_written.json", {"arms": {}, "refcv3": {
    "strategic": {"nav_compliance": {"status": "UNAVAILABLE", "defect": True,
                                      "defect_type": "TypeError"}},
    "_defects": DEFECT}})
# --- ARM B (CONTROL): the shape the reader EXPECTS --------------------------
b = write("as_read.json", {"arms": {}, "_defects": DEFECT, "refcv3": {}})
# --- ARM C (CONTROL): genuinely clean --------------------------------------
c = write("clean.json", {"arms": {}, "refcv3": {"strategic": {}}})

for label, p in (("A  record as refcv3_arm WRITES it (defect at rec['refcv3']['_defects'])", a),
                 ("B  CONTROL: defect at rec['_defects'] (what the reader looks for)", b),
                 ("C  CONTROL: genuinely defect-free", c)):
    ok, why = record_ok(p)
    print("%-72s -> ok=%-5s  %s" % (label, ok, why))

ok_a, _ = record_ok(a); ok_b, _ = record_ok(b); ok_c, _ = record_ok(c)
print()
if ok_a and not ok_b and ok_c:
    print("⛔ PROVEN BLIND: a record carrying a real DEFECT passes record_ok(),")
    print("   while the same defect one level up is caught (B) and a clean record passes (C).")
    print("   => the gate CANNOT fire on the defect the writer actually emits.")
elif not ok_a:
    print("✅ the gate fires on the as-written shape — no scope mismatch.")
else:
    print("INCONCLUSIVE: controls did not behave as required (B must fail, C must pass).")
