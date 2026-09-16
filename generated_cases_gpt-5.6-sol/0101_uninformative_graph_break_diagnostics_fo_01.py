# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Explicit graph break preserves branch semantics
# sibling    : torch._dynamo.graph_break explicit break

import torch

# Sibling construct: explicit torch._dynamo.graph_break.
def fn(x):
    prefix = x + 2
    torch._dynamo.graph_break()
    if x.shape[0] > 1:
        return prefix * 3
    return prefix - 3

x = torch.tensor([1.0, 2.0])
eager = fn(x.clone())
torch._dynamo.reset()
compiled = torch.compile(fn, backend="eager")(x.clone())
assert torch.equal(compiled, eager), (compiled, eager)
