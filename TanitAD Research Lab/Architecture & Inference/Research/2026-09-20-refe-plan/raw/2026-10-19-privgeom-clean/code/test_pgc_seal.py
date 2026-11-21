"""Tests of the seal (pgc_seal) and of its REACHABILITY from the read's entry point (pgc_run).  CPU only; no label, no array, no scalar store is read.  Every guard has a RED arm (a copy that must be REFUSED) and the
two guards that the prereg leans on hardest also have a MUTATION arm (disable the guard => the bad copy gets through, so the test can fail).

Built in a throw-away directory: a miniature package (code files, a ref file, PINS.json, the MM ruling, a prereg with a pins table and a Revision-1 block), sealed by the REAL `pgc_seal.seal`, then attacked."""
import ast
import builtins
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pgc_seal as S                                                                     # noqa: E402

RULING = "# MM ruling (test)\n\n**DQ1: CENTRED. RATIFIED.**\n- a line with `code`\n\n**DQ2: bars.**\n"
REQUIRED = ("code/a.py", "code/b.py", "ref/c.py")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def quote(text: str) -> str:
    return "\n".join(("> " + ln) if ln else ">" for ln in text.strip("\n").splitlines())


def write(p, text):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def prereg_text(root, status="SEALED", ruling=RULING, table=None):
    rows = table if table is not None else {n: sha(open(f"{root}/{S._rel(n)}", "rb").read()) for n in ("a.py", "b.py", "ref/c.py")}
    tab = "\n".join(f"| `{n}` | new | `{h}` |" for n, h in rows.items())
    return (f"# PREREG (test)\n\n**STATUS: {status}.** the body.\n\n## 13. pins\n\n| file | frozen? | sha256 |\n|---|---|---|\n{tab}\n\n## Revision 1 (MM ruling, pre-seal)\n\n"
            f"{S.BEGIN}\n{quote(ruling)}\n{S.END}\n\nend\n")


def build(td, status="SEALED", seal=True):
    root = f"{td}/pkg"
    write(f"{root}/code/a.py", "A = 1\n")
    write(f"{root}/code/b.py", "B = 2\n")
    write(f"{root}/ref/c.py", "C = 3\n")
    write(f"{root}/{S.RULING_NAME}", RULING)
    write(f"{root}/input.bin", "input-bytes")
    pins = {"a.py": sha(b"A = 1\n"), "b.py": sha(b"B = 2\n"), "ref/c.py": sha(b"C = 3\n")}
    write(f"{root}/code/PINS.json", json.dumps({"frozen_ok": True, "sha256": pins}))
    write(f"{root}/{S.PREREG_NAME}", prereg_text(root, status))
    if seal:
        S.seal(root, sealed_by="test", inputs={"in1": f"{root}/input.bin"}, required=REQUIRED)
    return root


def rewrite_seal(root, text):
    """re-hash a modified prereg so that ONLY the guard under test can refuse it (the .sha256 and SEALED_AT agree with the new bytes)."""
    write(f"{root}/{S.PREREG_NAME}", text)
    h = S.sha_file(f"{root}/{S.PREREG_NAME}")
    write(f"{root}/PREREG_PRIVGEOM_CLEAN.sha256", f"{h}  {S.PREREG_NAME}\n")
    sa = json.load(open(f"{root}/{S.SEALED_AT_NAME}", encoding="utf-8"))
    sa["prereg_sha256"] = h
    write(f"{root}/{S.SEALED_AT_NAME}", json.dumps(sa))


class TestSeal(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="pgc_seal_test_")
        self.root = build(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def check(self, **kw):
        return S.check_seal(self.root, required=REQUIRED, **kw)

    def refused(self, needle):
        with self.assertRaises(S.SealError) as cm:
            self.check()
        self.assertIn(needle, str(cm.exception))

    # ---- the control: the genuine seal is accepted --------------------------------------------------------------------------------------------------------
    def test_genuine_seal_is_accepted(self):
        info = self.check(inputs=True)
        self.assertEqual(info["status"], "SEALED")
        self.assertTrue(info["inputs_checked"])
        self.assertEqual(info["n_code_pins"], 4)                                         # a, b, ref/c, PINS.json

    # ---- RED arms: each must be REFUSED --------------------------------------------------------------------------------------------------------------------
    def test_red_one_byte_edited_prereg(self):
        p = f"{self.root}/{S.PREREG_NAME}"
        b = bytearray(open(p, "rb").read())
        b[b.index(b"body")] ^= 0x01                                                      # ONE byte
        open(p, "wb").write(bytes(b))
        self.refused("does not hash to its recorded sha256")

    def test_red_draft_status_copy_with_a_matching_hash(self):
        rewrite_seal(self.root, prereg_text(self.root, status="DRAFT"))                  # hash, .sha256 and SEALED_AT all agree: ONLY the status guard can refuse
        self.refused("status is 'DRAFT'")

    def test_red_draft_status_copy_with_only_its_own_sha256(self):
        p = f"{self.root}/{S.PREREG_NAME}"
        write(p, prereg_text(self.root, status="DRAFT"))
        write(f"{self.root}/PREREG_PRIVGEOM_CLEAN.sha256", f"{S.sha_file(p)}  x\n")
        with self.assertRaises(S.SealError):
            self.check()

    def test_red_draft_status_copy_with_the_old_sha256(self):
        write(f"{self.root}/{S.PREREG_NAME}", prereg_text(self.root, status="DRAFT"))
        self.refused("does not hash")

    def test_red_missing_status_line(self):
        rewrite_seal(self.root, prereg_text(self.root).replace("**STATUS:", "**State:"))
        self.refused("status is None")

    def test_red_unsealed_package(self):
        for n in ("PREREG_PRIVGEOM_CLEAN.sha256", S.SEALED_AT_NAME):
            td = tempfile.mkdtemp(prefix="pgc_seal_test2_")
            try:
                root = build(td)
                os.remove(f"{root}/{n}")
                with self.assertRaises(S.SealError) as cm:
                    S.check_seal(root, required=REQUIRED)
                self.assertIn("NOT SEALED", str(cm.exception))
            finally:
                shutil.rmtree(td, ignore_errors=True)

    def test_red_pinned_code_file_edited_after_the_seal(self):
        with open(f"{self.root}/code/a.py", "a", newline="\n") as f:
            f.write("#\n")
        self.refused("pinned file(s) differ")

    def test_red_pinned_file_deleted(self):
        os.remove(f"{self.root}/ref/c.py")
        self.refused("pinned file(s) differ")

    def test_red_a_file_the_read_uses_is_not_pinned(self):
        with self.assertRaises(S.SealError) as cm:
            S.check_seal(self.root, required=REQUIRED + ("code/not_pinned.py",))
        self.assertIn("does not pin", str(cm.exception))

    def test_red_pins_table_with_a_wrong_row(self):
        rows = {"a.py": sha(b"A = 1\n"), "b.py": "0" * 64, "ref/c.py": sha(b"C = 3\n")}
        rewrite_seal(self.root, prereg_text(self.root, table=rows))
        self.refused("pins table disagrees")

    def test_red_pins_table_that_omits_a_file(self):
        rows = {"a.py": sha(b"A = 1\n"), "ref/c.py": sha(b"C = 3\n")}
        rewrite_seal(self.root, prereg_text(self.root, table=rows))
        self.refused("absent")

    def test_red_revision_1_is_not_the_ruling_verbatim(self):
        rewrite_seal(self.root, prereg_text(self.root, ruling=RULING.replace("CENTRED", "UNCENTRED")))
        self.refused("not the MM ruling verbatim")

    def test_red_ruling_file_edited(self):
        write(f"{self.root}/{S.RULING_NAME}", RULING + "x\n")
        self.refused("differs from its sealed sha256")

    def test_red_amendment_that_does_not_hash(self):
        write(f"{self.root}/PREREG_PRIVGEOM_CLEAN_AMENDMENT1.md", "amendment text\n")
        write(f"{self.root}/PREREG_PRIVGEOM_CLEAN_AMENDMENT1.sha256", f"{'0' * 64}  x\n")
        self.refused("Amendment 1")
        write(f"{self.root}/PREREG_PRIVGEOM_CLEAN_AMENDMENT1.sha256", f"{S.sha_file(self.root + '/PREREG_PRIVGEOM_CLEAN_AMENDMENT1.md')}  x\n")
        self.assertIn("amend1", self.check()["pins"])                                    # a properly hashed amendment is carried in the pins

    def test_red_input_file_changed(self):
        self.check(inputs=True)
        with open(f"{self.root}/input.bin", "a", newline="\n") as f:
            f.write("x")
        self.check(inputs=False)                                                         # the cheap path does not hash inputs
        with self.assertRaises(S.SealError) as cm:
            self.check(inputs=True)
        self.assertIn("input file(s)", str(cm.exception))

    def test_red_seal_refuses_to_re_seal_and_refuses_a_draft(self):
        with self.assertRaises(S.SealError) as cm:
            S.seal(self.root, sealed_by="x", inputs={}, required=REQUIRED)
        self.assertIn("already sealed", str(cm.exception))
        td = tempfile.mkdtemp(prefix="pgc_seal_test3_")
        try:
            root = build(td, status="DRAFT", seal=False)
            with self.assertRaises(S.SealError) as cm:
                S.seal(root, sealed_by="x", inputs={}, required=REQUIRED)
            self.assertIn("must read SEALED", str(cm.exception))
            self.assertFalse(os.path.exists(f"{root}/PREREG_PRIVGEOM_CLEAN.sha256"))     # a refused seal writes NOTHING
        finally:
            shutil.rmtree(td, ignore_errors=True)

    # ---- MUTATION arms: disable a guard => the bad copy gets through (so the RED arm above is the guard, not an accident) -----------------------------------
    def test_mutation_status_guard_disabled_lets_the_draft_through(self):
        rewrite_seal(self.root, prereg_text(self.root, status="DRAFT"))
        self.refused("status is 'DRAFT'")
        real = S.status_of
        S.status_of = lambda text: "SEALED"
        try:
            self.assertEqual(self.check()["status"], "SEALED")                           # the mutated check ACCEPTS the draft: only the status guard was refusing it
        finally:
            S.status_of = real

    def test_mutation_verbatim_guard_disabled_lets_an_altered_revision_through(self):
        rewrite_seal(self.root, prereg_text(self.root, ruling=RULING.replace("CENTRED", "UNCENTRED")))
        self.refused("not the MM ruling verbatim")
        real = S._verbatim_ok
        S._verbatim_ok = lambda text, path: True
        try:
            self.check()
        finally:
            S._verbatim_ok = real

    # ---- the seal check is LABEL-FREE -------------------------------------------------------------------------------------------------------------------------
    def test_check_opens_only_package_files(self):
        opened, real_open = [], builtins.open

        def spy(file, *a, **k):
            opened.append(os.path.abspath(str(file)))
            return real_open(file, *a, **k)
        builtins.open = spy
        try:
            self.check(inputs=True)
        finally:
            builtins.open = real_open
        root = os.path.abspath(self.root)
        self.assertTrue(opened)
        self.assertTrue(all(p.startswith(root) for p in opened), [p for p in opened if not p.startswith(root)])

    def test_the_seal_module_has_no_route_to_a_label(self):
        src = open(f"{HERE}/pgc_seal.py", encoding="utf-8").read()
        for needle in ("np.load", "numpy", "labels6", "meta.npz", "import torch"):
            self.assertNotIn(needle, src)


# ---- REACHABILITY: the entry point must actually call the seal (a guard that exists only in its own test is not a guard) ------------------------------------
def reach_failures(src: str) -> list:
    """From pgc_run's source: `require_prereg` must call `SL.check_seal`; every stage that reads or writes the read's data must call `require_prereg`; `seal-check` must too."""
    tree = ast.parse(src)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def calls(fn, name):
        return any(isinstance(c, ast.Call) and ((isinstance(c.func, ast.Name) and c.func.id == name) or (isinstance(c.func, ast.Attribute) and c.func.attr == name)) for c in ast.walk(fn))
    bad = []
    if "require_prereg" not in fns or not calls(fns["require_prereg"], "check_seal"):
        bad.append("require_prereg does not call check_seal")
    for st in ("stage_select", "stage_join", "stage_gate", "stage_fit", "stage_analyze", "stage_seal_check"):
        if st not in fns or not calls(fns[st], "require_prereg"):
            bad.append(f"{st} does not call require_prereg")
    if "stage_preflight" not in fns or not calls(fns["stage_preflight"], "stage_seal_check"):
        bad.append("stage_preflight does not call stage_seal_check")
    return bad


class TestReachability(unittest.TestCase):
    def test_entry_point_calls_the_seal_from_every_stage(self):
        src = open(f"{HERE}/pgc_run.py", encoding="utf-8").read()
        self.assertEqual(reach_failures(src), [])

    def test_mutation_removing_the_seal_call_is_detected(self):
        src = open(f"{HERE}/pgc_run.py", encoding="utf-8").read()
        self.assertIn("SL.check_seal(", src)
        self.assertTrue(reach_failures(src.replace("SL.check_seal(", "SL.noop(")))      # the entry point no longer calls the seal => RED
        self.assertTrue(reach_failures(src.replace("def stage_fit(a) -> int:\n    require_prereg()", "def stage_fit(a) -> int:\n    pass")))

    def test_the_seal_covers_everything_the_read_runs(self):
        import pgc_pins
        import pgc_run
        need = {n.split("/", 1)[1] for n in pgc_run.SEAL_REQUIRED if n.startswith("code/")}
        self.assertTrue(need <= set(pgc_pins.CODE) | {"PINS.json"}, need - set(pgc_pins.CODE))
        self.assertTrue(set(pgc_run.CODE_FILES + pgc_run.FROZEN_SUITES) <= need)
        self.assertTrue({"pgc_read_chain.py", "pgc_seal.py", "test_pgc_seal.py", "pgc_run.py", "pg_extract.py"} <= need)
        self.assertTrue(all(n in pgc_pins.CODE for n in ("pgc_read_chain.py", "pgc_seal.py", "test_pgc_seal.py", "pgc_seal_proofs.py")))


if __name__ == "__main__":
    unittest.main(verbosity=1)
