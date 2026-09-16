# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Disallowed operator graph break
# sibling    : torch._dynamo.disallow_in_graph on an otherwise traceable operator

import torch

# Sibling construct: disallow_in_graph forces a break around an otherwise traceable operator.
torch._dynamo.disallow_in_graph(torch.sin)


def fn(x):
    before = x * 2
    middle = torch.sin(before)
    return middle.square() + before


value = torch.tensor([-1.0, 0.0, 0.75, 2.0])
eager = fn(value.clone())
compiled = torch.compile(fn, backend="eager")
actual = compiled(value.clone())
torch.testing.assert_close(actual, eager)
