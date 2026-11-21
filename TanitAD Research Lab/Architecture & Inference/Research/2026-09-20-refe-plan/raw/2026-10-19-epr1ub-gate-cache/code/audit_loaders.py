"""Audit of the module loaders of the three gates for the EPR1UB-GATE-CACHE pattern (a cache keyed on less than the verification covers).
Usage: python -I audit_loaders.py <ep-r1 deploy dir (epr1_gate.py, epr1.py ...)> <sel-rank code dir (srttc_c2_gate.py ...)>  -> prints one JSON document.
Everything is measured on COPIES in a temp dir; nothing in the two input dirs is written."""
import json, os, shutil, sys, tempfile

ub_dir, c2_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
out = {}
tmp = tempfile.mkdtemp(prefix="loader_audit_")


def rb(p):
    with open(p, "rb") as f:
        return f.read()


def wb(p, d):
    with open(p, "wb") as f:
        f.write(d)


def sibling_imports(path, code_dir):
    """Top-level imports of `path` (ast) that name a .py file sitting beside it: what the module would load at exec time besides its own bytes."""
    import ast
    names = set()
    for node in ast.walk(ast.parse(rb(path).decode("utf-8"))):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return sorted(n for n in names if os.path.exists(os.path.join(code_dir, n + ".py")))


# ---------------------------------------------------------------- epr1_gate.load_verified_module (cache: _MODULES, key = "epr1__gate_" + sha256(epr1.py)[:12])
sys.path.insert(0, ub_dir)
import epr1_gate as EG
m1, s1 = EG.load_verified_module(ub_dir)
same = os.path.join(tmp, "e_same")
os.makedirs(same)
shutil.copy(os.path.join(ub_dir, "epr1.py"), same)
m2, s2 = EG.load_verified_module(same)
tam = os.path.join(tmp, "e_tam")
os.makedirs(tam)
wb(os.path.join(tam, "epr1.py"), rb(os.path.join(ub_dir, "epr1.py")) + b"\n#x\n")
m3, s3 = EG.load_verified_module(tam)
try:
    EG.load_verified_module(tam, expect_sha256=s1)
    refused = False
except EG.GateError:
    refused = True
out["epr1_gate.load_verified_module"] = {
    "identical_bytes_in_a_second_dir_returns_the_cached_module": m2 is m1,
    "its___file___is_still_the_first_dir": os.path.dirname(m2.__file__) == os.path.join(ub_dir),
    "tampered_bytes_in_a_second_dir_get_their_own_module": m3 is not m1 and s3 != s1,
    "tampered_bytes_are_refused_when_the_verified_sha_is_passed": refused,
    "sibling_files_epr1_py_imports_at_module_level": sibling_imports(os.path.join(ub_dir, "epr1.py"), ub_dir),
}
sys.path.remove(ub_dir)

# ---------------------------------------------------------------- srttc_c2_gate loaders
sys.path.insert(0, c2_dir)
import srttc_c2_gate as CG
names = ("srttc_r1_gate.py", "srttc_r1.py", "srttc_c2.py")


def mk(name, edits=None):
    d = os.path.join(tmp, name)
    os.makedirs(d)
    for f in names:
        data = rb(os.path.join(c2_dir, f))
        if edits and f in edits:
            data += edits[f]
        wb(os.path.join(d, f), data)
    return d


base = mk("c_base")
r1, c2, _ = CG.load_verified_pair(base, {})
same = mk("c_same")
r1b, c2b, _ = CG.load_verified_pair(same, {})
tam_r1 = mk("c_tam_r1", {"srttc_r1.py": b"\n#x\n"})
r1c, c2c, shas_c = CG.load_verified_pair(tam_r1, {})
tam_c2 = mk("c_tam_c2", {"srttc_c2.py": b"\n#x\n"})
r1d, c2d, shas_d = CG.load_verified_pair(tam_c2, {})
try:
    CG.load_verified_pair(tam_r1, {"srttc_r1.py": CG.sha256_file(os.path.join(base, "srttc_r1.py"))})
    refused_r1 = False
except CG.GateError:
    refused_r1 = True
try:
    CG.load_verified_pair(tam_c2, {"srttc_c2.py": CG.sha256_file(os.path.join(base, "srttc_c2.py"))})
    refused_c2 = False
except CG.GateError:
    refused_c2 = True
out["srttc_c2_gate.load_verified_pair"] = {
    "identical_bytes_in_a_second_dir_return_the_cached_pair": r1b is r1 and c2b is c2,
    "tampered_r1_in_a_second_dir_gets_its_own_r1_AND_c2": r1c is not r1 and c2c is not c2,
    "tampered_c2_in_a_second_dir_gets_its_own_c2_and_the_cached_r1": c2d is not c2 and r1d is r1,
    "tampered_r1_refused_when_the_verified_sha_is_passed": refused_r1,
    "tampered_c2_refused_when_the_verified_sha_is_passed": refused_c2,
    "sibling_files_imported_at_module_level": {f: sibling_imports(os.path.join(c2_dir, f), c2_dir) for f in names},
    "c2_module_name_carries_both_shas": c2.__name__.count("_") >= 3 and shas_c["srttc_r1.py"][:12] in c2c.__name__ and shas_c["srttc_c2.py"][:12] in c2c.__name__,
}


# the sibling defect: a load whose exec RAISED stays registered and the next call gets it back
def poison(fn, d, *a):
    first = second = None
    try:
        fn(d, *a)
    except Exception as e:                     # noqa: BLE001
        first = type(e).__name__
    try:
        got = fn(d, *a)
        second = "RETURNED " + type(got).__name__
    except Exception as e:                     # noqa: BLE001
        second = "RAISED " + type(e).__name__
    return {"first_call": first or "returned", "second_call": second}


b1 = mk("p_r1", {"srttc_r1.py": b"\nraise RuntimeError('boom')\n"})
b2 = mk("p_c2", {"srttc_c2.py": b"\nraise RuntimeError('boom')\n"})
b3 = mk("p_gate", {"srttc_r1_gate.py": b"\nraise RuntimeError('boom')\n"})
out["srttc_c2_gate_failed_load_then_second_call"] = {
    "exec_verified_module(srttc_r1.py)": poison(lambda d: CG.exec_verified_module(d, "srttc_r1.py", "srttc_r1"), b1),
    "load_verified_pair(c2 raises)": poison(lambda d: CG.load_verified_pair(d, {}), b2),
    "_load_frozen_gate(srttc_r1_gate.py raises)": poison(CG._load_frozen_gate, b3),
}
shutil.rmtree(tmp, ignore_errors=True)
print(json.dumps(out, indent=1))
