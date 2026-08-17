"""The CEM planner must be reproducible, and the invariant must be PINNED.

WHY THIS EXISTS (2026-08-18, C91 escalation #2). The P2 verdicts were re-decided
against an estimator correction while `cem_plan`'s only stochastic component --
the `eps` draw -- consumed GLOBAL RNG state. So two runs of the SAME arm on the
SAME windows returned different numbers, and "the estimator moved" could not be
separated from "the sampler moved".

That matters right now because `planner_beats_cv` is UNDECIDED and needs a
re-drive: flipping it takes +6.59 % against a measured local upper edge of
+5.877 %. An unbounded sampling component sitting on top of a margin that thin
would make the re-drive unreadable in exactly the region where it has to be read.

⛔ The source-level assertions below are the load-bearing ones. A behavioural
test can only prove the seeding works on the path it exercises; these prove no
UNSEEDED path exists at all -- which is the property that was actually violated,
since the file already seeded all four bootstraps (`seed=0`) and simply never
reached this one site.
"""
import ast
import inspect
import re

import pytest
import torch

from taniteval import planner_p2


SRC = inspect.getsource(planner_p2)


# ----------------------------------------------------------------- behaviour --
def test_same_seed_reproduces_the_draw_and_a_different_seed_does_not():
    a = torch.randn(4, 8, 2, generator=planner_p2.cem_generator("cpu", 0))
    b = torch.randn(4, 8, 2, generator=planner_p2.cem_generator("cpu", 0))
    c = torch.randn(4, 8, 2, generator=planner_p2.cem_generator("cpu", 1))
    assert torch.equal(a, b), "same seed must reproduce the draw bit-exactly"
    assert not torch.equal(a, c), "different seeds must not collide"


def test_one_generator_reused_keeps_draws_INDEPENDENT():
    """The distribution must be unchanged from the unseeded version.

    One generator threaded through a whole collection run gives each chunk a
    FRESH draw. Rebuilding the generator per call would instead hand every chunk
    identical noise -- reproducible, but a different (and worse) sampler. That
    silent distribution change is the failure mode this test exists to block.
    """
    g = planner_p2.cem_generator("cpu", 0)
    first = torch.randn(64, 8, 2, generator=g)
    second = torch.randn(64, 8, 2, generator=g)
    assert not torch.equal(first, second), (
        "consecutive draws from one generator must differ -- if they match, the "
        "generator is being rebuilt per call and the sampler has changed"
    )


def test_generator_is_device_matched():
    """A CUDA `torch.randn` will not accept a CPU generator."""
    assert planner_p2.cem_generator("cpu", 0).device.type == "cpu"
    if torch.cuda.is_available():
        assert planner_p2.cem_generator("cuda", 0).device.type == "cuda"


# --------------------------------------------------------------- source pins --
def test_every_stochastic_draw_in_the_planner_passes_a_generator():
    """No unseeded sampling site may exist anywhere in the module."""
    tree = ast.parse(SRC)
    stochastic = {"randn", "rand", "randint", "randperm", "normal",
                  "multinomial", "bernoulli", "randn_like", "rand_like"}
    unseeded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name not in stochastic:
            continue
        if not any(kw.arg == "generator" for kw in node.keywords):
            unseeded.append(f"{name} at line {node.lineno}")
    assert not unseeded, (
        "unseeded stochastic call(s) in planner_p2 -- every draw must take an "
        "explicit `generator=`, or the run carries an unbounded sampling "
        f"component: {unseeded}"
    )


def test_numpy_global_rng_is_not_used():
    """`np.random.*` has its own global state and would reopen the same hole."""
    hits = re.findall(r"np\.random\.\w+", SRC)
    assert not hits, f"numpy global RNG in planner_p2: {sorted(set(hits))}"


@pytest.mark.parametrize("fn_name", ["collect_openloop", "collect_closedloop"])
def test_each_collector_takes_a_seed_and_builds_exactly_one_generator(fn_name):
    """One generator per RUN -- not per call, and not absent."""
    fn = getattr(planner_p2, fn_name)
    params = inspect.signature(fn).parameters
    assert "cem_seed" in params, f"{fn_name} must expose `cem_seed`"
    assert params["cem_seed"].default == planner_p2.CEM_SEED_DEFAULT

    body = inspect.getsource(fn)
    built = body.count("cem_generator(")
    assert built == 1, (
        f"{fn_name} builds cem_generator {built}x; expected exactly 1 "
        "(0 = unseeded, >1 = a per-chunk rebuild that changes the sampler)"
    )


def test_cem_plan_and_closed_loop_planner_accept_a_generator():
    for fn_name in ("cem_plan", "closed_loop_planner"):
        params = inspect.signature(getattr(planner_p2, fn_name)).parameters
        assert "gen" in params, f"{fn_name} must accept `gen`"


def test_no_cem_plan_call_site_omits_the_generator():
    """A threaded parameter is only as good as its call sites."""
    tree = ast.parse(SRC)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name not in ("cem_plan", "closed_loop_planner"):
            continue
        if not any(kw.arg == "gen" for kw in node.keywords):
            bad.append(f"{name} at line {node.lineno}")
    assert not bad, f"call site(s) omitting `gen=`: {bad}"
