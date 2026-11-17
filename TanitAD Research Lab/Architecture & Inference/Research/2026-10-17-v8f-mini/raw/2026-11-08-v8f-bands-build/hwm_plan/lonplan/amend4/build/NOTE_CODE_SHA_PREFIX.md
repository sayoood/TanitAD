# NOTE: the hwm_plan.py sha256 prefix in the frozen AMEND_4 and AMEND_5 texts (MM, 2026-11-17)

* The frozen texts `AMEND_HWMPLAN_4.md`, `MM_RULINGS_AMEND4.md`, `AMEND_HWMPLAN_5.md` and `MM_RULINGS_AMEND5.md` quote the sha256 of the code of record `hwm_plan.py` as `5a072272…`.
* MEASURED: the file in the run tree, and LON-PLAN's banked `score/A0_s0/SOLVER.json`, both read `5a0722719f7ce06c94c91213db6ef542b2fe16d346622ba506ce8808bd6d714b`. The quoted prefix is wrong at its 8th hex digit (`2` where the file has `1`). The other three code-of-record prefixes (`935b3422`, `79392075`, `a2d3922e`) match.
* The frozen texts stay byte-equal to their sha256. The builds pin the value from the banked record:
  * AMEND_4's C-IN compares 7 digits, `5a07227` (`amend4_stage.CODE_SHA_PREFIX`). Its smoke read C-IN RED for this reason alone, and the launch re-runs every control.
  * AMEND_5's build was asked to pin the full 64-hex value.
* Class: a hand-copied hash in a frozen text. A control pinned to it would refuse the code it was meant to admit. ⇒ Pin hashes from the banked record, never from prose.
