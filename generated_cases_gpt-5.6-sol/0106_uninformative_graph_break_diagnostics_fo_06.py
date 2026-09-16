# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Explicit break across distinct callers
# sibling    : torch._dynamo.graph_break with full-call-stack-sensitive call sites

import torch

# Sibling construct: explicit torch._dynamo.graph_break reached through distinct callers.
def break_leaf(x):
    torch._dynamo.graph_break()
    return x.sin()


def left_path(x):
    return break_leaf(x) + 1


def right_path(x):
    return break_leaf(x) - 1


def fn(x, use_left):
    if use_left:
        return left_path(x) * 2
    return right_path(x) * 3


compiled = torch.compile(fn, backend="eager")
for flag in (True, False):
    value = torch.tensor([0.25, -0.5, 1.0])
    eager = fn(value.clone(), flag)
    actual = compiled(value.clone(), flag)
    torch.testing.assert_close(actual, eager)
