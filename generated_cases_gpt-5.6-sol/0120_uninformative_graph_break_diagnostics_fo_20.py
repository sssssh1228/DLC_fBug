# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Distinct disabled call chains
# sibling    : full-call-stack diagnostic identity for repeated calls to one disabled leaf

import torch

# Sibling construct: one disabled leaf reached through two distinct wrapper call chains.
@torch._dynamo.disable(reason="shared leaf boundary")
def disabled_leaf(x, bias):
    return x.mul(2.0).add(bias)


def via_square(x):
    return disabled_leaf(x.square(), 1.0)


def via_shift(x):
    return disabled_leaf(x - 3.0, -2.0)


def fn(x):
    left = via_square(x)
    right = via_shift(x)
    return left - right.tanh()


x = torch.tensor([-2.0, 0.5, 4.0])
expected = fn(x.clone())
compiled = torch.compile(fn, backend="eager")
actual = compiled(x.clone())
torch.testing.assert_close(actual, expected)

report = torch._dynamo.explain(fn)(x.clone())
matching_reasons = [
    str(getattr(item, "reason", item))
    for item in report.break_reasons
    if "shared leaf boundary" in str(getattr(item, "reason", item))
]
assert len(matching_reasons) >= 2, matching_reasons
