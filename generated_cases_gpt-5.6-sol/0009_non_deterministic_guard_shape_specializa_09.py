# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Mapping Guard Determinism
# sibling    : dictionary key, value, and insertion-order guards

import torch
from torch._dynamo.utils import counters

# Sibling construct under test: dictionary key, value, and insertion-order guards.
def fn(x, options):
    result = x * options["scale"] + options["bias"]
    if options["negate"]:
        result = -result
    return result

x = torch.arange(6.0).reshape(2, 3)
options = [
    {"scale": 2.0, "bias": 1.0, "negate": False},
    {"bias": 1.0, "negate": False, "scale": 2.0},
    {"scale": 3.0, "bias": -2.0, "negate": True},
    {"negate": True, "scale": 3.0, "bias": -2.0},
]
expected = [fn(x, option) for option in options]
signatures = []
baseline = None

for _ in range(3):
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    actual = [compiled(x, option) for option in options]
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)
    if baseline is None:
        baseline = [value.clone() for value in actual]
    else:
        for got, first in zip(actual, baseline):
            torch.testing.assert_close(got, first)
    signatures.append((
        counters["frames"].get("total", 0),
        counters["frames"].get("ok", 0),
        counters["stats"].get("calls_captured", 0),
        counters["stats"].get("unique_graphs", 0),
    ))

assert signatures.count(signatures[0]) == len(signatures), signatures
