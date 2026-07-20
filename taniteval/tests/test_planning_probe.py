"""Analytic tests for the TanitEval planning panel (planning.py): majority route
base-rate + behavior-decodability probe.

Synthetic inputs with hand-known answers, CPU-only, no checkpoint. pytest is NOT
installed on the eval pod, so these run standalone; they are also plain ``test_*``
functions collectable by pytest if present.

Run:  PYTHONPATH=/root/taniteval:/root/TanitAD/stack python tests/test_planning_probe.py
"""
import sys

import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import planning  # noqa: E402


def _approx(a, b, tol=1e-6):
    assert abs(float(a) - float(b)) <= tol, f"{a} != {b} (tol {tol})"


# --------------------------------------------------------------------------- #
# majority-class base rate                                                      #
# --------------------------------------------------------------------------- #
def test_majority_rate_constant_class():
    assert planning._majority_rate([2, 2, 2, 2]) == 1.0        # all one class
    _approx(planning._majority_rate([0, 0, 0, 1]), 0.75)       # 3/4 majority
    _approx(planning._majority_rate([1, 1, 2, 2, 2]), 0.6)     # 3/5 majority
    assert planning._majority_rate([]) is None


def test_majority_rate_matches_route_intuition():
    # a route set that is 67.5% straight -> base rate 0.675 (the number the
    # flagship's "route-from-vision" must be judged against, not left unjudged)
    labels = [1] * 27 + [0] * 7 + [2] * 6           # 27/40 straight
    _approx(planning._majority_rate(labels), 0.675)


# --------------------------------------------------------------------------- #
# behavior-decodability probe (eval_behavior instrument, balanced accuracy)     #
# --------------------------------------------------------------------------- #
def _separable(n_ep=12, per=24, n_classes=5, f_dim=8, sep=6.0, seed=0):
    """Windows whose latent is a class centroid + small noise — behavior is
    (near) perfectly linearly decodable. Labels are balanced and each episode
    spans all classes (so any episode split keeps every class present)."""
    g = torch.Generator().manual_seed(seed)
    centroids = torch.eye(n_classes, f_dim) * sep if f_dim >= n_classes else \
        torch.randn(n_classes, f_dim, generator=g)
    feats, labels, eid = [], [], []
    for e in range(n_ep):
        for i in range(per):
            c = i % n_classes
            feats.append(centroids[c] + 0.25 * torch.randn(f_dim, generator=g))
            labels.append(c)
            eid.append(e)
    return torch.stack(feats), torch.tensor(labels), eid


def test_behavior_probe_decodes_separable():
    feats, labels, eid = _separable(seed=1)
    r = planning._behavior_probe(feats, labels, eid, 5)
    assert "skipped" not in r, r
    _approx(r["chance_balacc"], 0.2)
    assert r["beats_chance"] is True
    assert r["maneuver_balanced_accuracy"] > 0.8, r["maneuver_balanced_accuracy"]
    assert r["decodability_vs_chance"] > 0.5


def test_behavior_probe_noise_is_chance():
    """Latent independent of the label -> balanced accuracy ~ chance, not above."""
    g = torch.Generator().manual_seed(7)
    n_ep, per, nc = 12, 24, 5
    feats = torch.randn(n_ep * per, 8, generator=g)
    labels = torch.tensor([i % nc for i in range(n_ep * per)])
    eid = [i // per for i in range(n_ep * per)]
    r = planning._behavior_probe(feats, labels, eid, nc)
    assert "skipped" not in r, r
    # no linearly-readable signal: balanced accuracy stays near 1/nc
    assert r["maneuver_balanced_accuracy"] < 0.35, r["maneuver_balanced_accuracy"]
    assert r["decodability_vs_chance"] < planning.DECODE_MARGIN + 0.1


def test_behavior_probe_skips_when_too_few():
    feats = torch.randn(10, 8)  # 10 < 40-window floor -> skipped
    labels = torch.tensor([i % 5 for i in range(10)])
    r = planning._behavior_probe(feats, labels, [0] * 10, 5)
    assert "skipped" in r


def test_behavior_probe_reports_majority_and_raw():
    """Imbalance sanity: raw accuracy and majority baseline are both reported so
    balanced accuracy (the honest metric) can be read against them."""
    feats, labels, eid = _separable(seed=2)
    r = planning._behavior_probe(feats, labels, eid, 5)
    assert 0.0 <= r["maneuver_majority_acc"] <= 1.0
    assert 0.0 <= r["maneuver_accuracy_raw"] <= 1.0
    assert r["n_seeds"] >= 1 and r["n_windows"] == feats.shape[0]


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
