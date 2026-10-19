"""stage-2 fixes found by the regression batch (pass 1):  keep three PINNED TEXTS of the pre-existing suites intact instead of editing their tests
 (a) launch_gate.py  RLS1_DATA          : test_carve_feed_gate GM7 mutates the exact text  "--aux-drv-labels", "--epo-carve-logs")   -> keep it, append the HN data flags on a following line
 (b) launch_gate.py  CLI --arm-name     : test_rl_r0c mutates the exact text  | set(RLS1C_ARM_CLASS)) + sorted(AUXS1_ARM_CLASS)       -> keep it, append the hn arms at the end
 (c) test_carve_feed.py                 : the B8 pins compare the B8 trainer with the pre-B8 tip; P6-HN-S now sits ON TOP of B8, so the HN layer is peeled first
run on the edit tree AFTER patch_launch_gate.py / patch_tlg*.py  (idempotence is refused: every replacement asserts exactly one occurrence)"""
import sys
from pathlib import Path
R = Path("C:/Users/Admin/hn34/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe")


def sub(path, old, new, label):
    t = (R / path).read_bytes().decode("utf-8")
    assert "\r" not in t
    assert t.count(old) == 1, (label, t.count(old))
    assert new not in t, (label, "already applied")
    (R / path).write_bytes(t.replace(old, new).encode("utf-8"))
    print("ok", label)


# (a)
sub("launch_gate.py",
    '"--aux-drv-labels", "--epo-carve-logs",\n             "--hn-bank", "--ctx-cache")\n',
    '"--aux-drv-labels", "--epo-carve-logs")\nRLS1_DATA = RLS1_DATA + ("--hn-bank", "--ctx-cache")                                  # P6-HN-S: the hard-negative bank and the context cache are DATA inputs (G-DVB hashes them)\n',
    "RLS1_DATA")
# (b)
sub("launch_gate.py",
    "| set(RLS1C_ARM_CLASS) | set(HNSCREEN_ARM_CLASS) | set(HNSCREEN_NEEDS_PI)) + sorted(AUXS1_ARM_CLASS) + sorted(AUXS1C_ARM_CLASS),",
    "| set(RLS1C_ARM_CLASS)) + sorted(AUXS1_ARM_CLASS) + sorted(AUXS1C_ARM_CLASS) + sorted(set(HNSCREEN_ARM_CLASS) | set(HNSCREEN_NEEDS_PI)),",
    "arm-name choices")
# (c)
sub("test_carve_feed.py", "import carvefeed_layer as CL  # noqa: E402\n",
    "import carvefeed_layer as CL  # noqa: E402\nimport hnscreen_layer as HL  # noqa: E402     # P6-HN-S sits ON TOP of the B8 layer: the B8 pins below compare the B8 trainer (HL.peel) with the pre-B8 tip\n", "import HL")
sub("test_carve_feed.py",
    "        tip_src = CL.peel(sf_src)\n        seg, seg_tip = S1B.segments(sf_src), S1B.segments(tip_src)\n    except AssertionError:\n        tip_src = seg = seg_tip = None\n",
    "        sf_b8 = HL.peel(sf_src)                                     # the trainer as B8 left it (a trainer without the HN layer is returned unchanged)\n        tip_src = CL.peel(sf_src)\n        seg, seg_tip = S1B.segments(sf_b8), S1B.segments(tip_src)\n    except AssertionError:\n        sf_b8 = tip_src = seg = seg_tip = None\n",
    "suite peel")
sub("test_carve_feed.py",
    '        nm = {tuple(x.option_strings): x for x in M.build_parser()._actions}\n',
    '        nm = {tuple(x.option_strings): x for x in TE.load(sf_b8, f"sf_cf_b8_{tag}").build_parser()._actions}\n',
    "only_flags")
