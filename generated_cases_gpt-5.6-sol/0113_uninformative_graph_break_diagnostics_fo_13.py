# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Distinct callers of one disabled function
# sibling    : full-call-stack graph-break identity for a shared disabled function

import torch

# Sibling construct: distinct call chains reaching the same disabled function.
@torch.compiler.disable(reason="shared leaf reached from multiple callers")
def disabled_leaf(x, offset):
    return x + offset


def left_path(x):
    return disabled_leaf(x * 2, 1)


def right_path(x):
    return disabled_leaf(x - 3, -2)


def fn(x):
    left, right = left_path(x), right_path(x)
    return torch.stack((left, right), dim=0).sum(dim=0)


x = torch.tensor([2.0, 5.0, -1.0])
eager = fn(x.clone())
compiled_fn = torch.compile(fn, backend="eager")
compiled = compiled_fn(x.clone())
torch.testing.assert_close(compiled, eager)
