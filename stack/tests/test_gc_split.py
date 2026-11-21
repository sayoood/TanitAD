"""The encoder's and the rollout's grad-checkpointing are separate decisions.

⛔ WHY. They used to be ONE flag: ``--grad-checkpoint`` set
``EncoderConfig.grad_checkpoint``, and the k-step rollout read
``cfg.encoder.grad_checkpoint``. That coupled two unrelated things —

* the **rollout's** checkpointing is REQUIRED at k=60 (it fixed a MEASURED
  37.97/44 GiB OOM on a 44 GiB A40);
* the **encoder's** is a pure speed/memory trade costing ~2x the ViT forward.

MEASURED 2026-08-14: S-W ran at **42.8 % mean GPU utilisation with 20 GB free**,
i.e. paying encoder recompute it had the memory not to pay. With one flag there
was no way to drop the encoder's without also dropping the rollout's and
restoring the OOM.

⚠️ The compatibility requirement is the point of these tests: ``auto`` must
reproduce the OLD behaviour exactly, so every launch command written before the
split still means what it meant.
"""
from argparse import Namespace

import pytest

from train_v6_staged import resolve_gc


def _ns(master, enc="auto", roll="auto"):
    return Namespace(grad_checkpoint=master, enc_grad_checkpoint=enc,
                     rollout_grad_checkpoint=roll)


@pytest.mark.parametrize("master", [True, False])
def test_auto_reproduces_the_master_flag_for_both_sites(master):
    """Backwards compatibility: unset overrides == the pre-split behaviour."""
    assert resolve_gc(_ns(master), "enc_grad_checkpoint") is master
    assert resolve_gc(_ns(master), "rollout_grad_checkpoint") is master


def test_encoder_can_be_turned_off_while_rollout_stays_on():
    """The acceleration case: keep the k=60 rollout checkpointed (it must be)
    and stop paying ~2x on the ViT forward."""
    a = _ns(True, enc="off", roll="auto")
    assert resolve_gc(a, "enc_grad_checkpoint") is False
    assert resolve_gc(a, "rollout_grad_checkpoint") is True


def test_rollout_can_be_forced_on_under_a_false_master():
    a = _ns(False, enc="auto", roll="on")
    assert resolve_gc(a, "enc_grad_checkpoint") is False
    assert resolve_gc(a, "rollout_grad_checkpoint") is True


def test_missing_attribute_falls_back_to_the_master_flag():
    """A stale Namespace (e.g. a resumed run's banked args from before the
    split) must not crash — it means `auto`."""
    assert resolve_gc(Namespace(grad_checkpoint=True), "enc_grad_checkpoint")
    assert not resolve_gc(Namespace(grad_checkpoint=False),
                          "rollout_grad_checkpoint")


def test_none_is_treated_as_auto():
    assert resolve_gc(_ns(True, enc=None), "enc_grad_checkpoint") is True
