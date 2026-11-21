"""The CUDA K0/KD controls are read from pytest's summary whatever prefix the node id carries.

MEASURED 2026-10-04: K0 PASSED at step 50,400, but the runner's literal match on "PASSED tests/..." read it as
FAILED, because pytest printed an absolute D:/... node id (D: is a `subst` of E:). CUDA was then refused on two
splits. The lines below are copied from the real logs (cuda_controls_navtest.log; the warmup log printed
"PASSED ::test_K0..."). Expectations are literals.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "pytest_control7", os.path.join(os.path.dirname(HERE), "code", "pytest_control7.py"))
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

ABS = ("=========================== short test summary info ===========================\n"
       "PASSED D:\\Projects\\TanitAD\\FlyWheels\\TanitAD_EvalFlyWheel\\incoming\\2026-09-28-refcv7-standard-tests"
       "\\navsim\\tests\\test_model_seam7.py::test_K0_same_seed_is_bit_identical\n"
       "FAILED D:\\Projects\\TanitAD\\FlyWheels\\TanitAD_EvalFlyWheel\\incoming\\2026-09-28-refcv7-standard-tests"
       "\\navsim\\tests\\test_model_seam7.py::test_KD_exact_dedup_equals_native - AssertionError\n")
REL = "PASSED tests/test_model_seam7.py::test_K0_same_seed_is_bit_identical\n"
BARE = "PASSED ::test_K0_same_seed_is_bit_identical\n"


def test_absolute_windows_node_id_reads_PASS_and_its_failed_twin_reads_FAIL():
    assert pc.control_passed(ABS, "test_K0") is True
    assert pc.control_passed(ABS, "test_KD") is False


def test_relative_and_bare_node_ids_read_PASS():
    assert pc.control_passed(REL, "test_K0") is True
    assert pc.control_passed(BARE, "test_K0") is True


def test_a_FAILED_or_ERROR_line_or_a_mention_never_counts():
    assert pc.control_passed("FAILED tests/test_model_seam7.py::test_K0_x - boom\n", "test_K0") is False
    assert pc.control_passed("ERROR tests/test_model_seam7.py::test_K0_x\n", "test_K0") is False
    assert pc.control_passed("    def test_K0_same_seed_is_bit_identical(rig):\n", "test_K0") is False
    assert pc.control_passed("", "test_K0") is False


def test_another_modules_test_does_not_count():
    assert pc.control_passed("PASSED tests/test_other.py::test_K0_same_seed\n", "test_K0") is False


def test_the_OLD_literal_match_was_blind_to_the_real_log():
    """Deliberate-regression arm: the matcher the runner used reads the real log as FAILED."""
    txt = ABS.replace("\\", "/")
    assert ("PASSED tests/test_model_seam7.py::test_K0" in txt) is False
