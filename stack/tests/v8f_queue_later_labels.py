"""The box-2 queue labels appended AFTER the edits whose suites pin "this edit added exactly its N labels and changed no existing one".

`test_v8f_flat.py` (7 labels) and `test_v8f_oacc.py` (6 labels) each assert that the labels of `v8f_mini_box2_queue.cmd`, minus their own edit's labels, minus the
labels of the edits that came after them, are exactly the labels that existed before their edit. Every LATER edit to the queue has to widen that subtraction, and it
used to do so by editing each suite's private label sets. These two edits did not, and both suites went RED on the tip (MEASURED at 535d981 / b61422e).

Declared ONCE here, imported by both suites, so the next edit widens one place. Nothing else moves: every pre-existing label is still pinned byte for byte by its own
suite, and a label that is not declared here (or there) still fails the set-difference assertion.

Each entry names the edit that appended it, its commit, and the label. A label listed here that is absent from the queue fails the importing test, so this file cannot
quietly drift from the queue.
"""

#: edit (PREREG, commit that appended the label) -> the labels it appended to `v8f_mini_box2_queue.cmd`
LATER_EDIT_LABELS = {
    "PREREG_V8F_DAGDOSE (10a2b7e): dagdose / dagdoserep arms, reached only from :dagarm": {"dagdosearm"},
    "PREREG_V8F_LAUNCH_INTENT (d8c9f7b): launchintent / launchintentrep arms, DAG-OACC's argv plus --launch-intent-w": {"launchintentarm"},
}

LATER_LABELS = set().union(*LATER_EDIT_LABELS.values())
