# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Non-recursive disabled outer function
# sibling    : torch.compiler.disable with recursive=False and a nested helper call

import torch

# Sibling construct: non-recursive disable surrounding a nested Python helper.
def tensor_helper(x):
    return torch.relu(x) * 4


@torch.compiler.disable(
    recursive=False,
    reason="only the outer fallback frame is disabled",
)
def outer_fallback(x):
    pieces = [tensor_helper(x), x.cos()]
    return pieces[0] + pieces[1]


def fn(x):
    before = x - 0.5
    return outer_fallback(before) / 2


x = torch.tensor([-1.0, 0.25, 2.0])
eager = fn(x.clone())
compiled = torch.compile(fn, backend="eager")(x.clone())
torch.testing.assert_close(compiled, eager)
