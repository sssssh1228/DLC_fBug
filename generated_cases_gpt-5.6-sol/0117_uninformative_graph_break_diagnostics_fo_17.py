# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Disallowed tensor operation
# sibling    : torch._dynamo.disallow_in_graph on an otherwise traceable operator

import torch

# Sibling construct: torch._dynamo.disallow_in_graph forcing breaks at torch.neg calls.
torch._dynamo.disallow_in_graph(torch.neg)


def fn(x):
    chunks = x.chunk(2)
    negated = [torch.neg(chunk) for chunk in chunks]
    return torch.cat(negated).relu().sum(dim=1)


x = torch.tensor([[-2.0, 1.0], [3.0, -4.0], [0.5, -0.25], [8.0, 2.0]])
expected = fn(x.clone())
compiled = torch.compile(fn, backend="eager")
actual = compiled(x.clone())
torch.testing.assert_close(actual, expected)
