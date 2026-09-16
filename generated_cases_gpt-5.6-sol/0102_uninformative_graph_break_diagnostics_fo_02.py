# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Disallowed operator resumes correctly
# sibling    : torch._dynamo.disallow_in_graph operator break

import torch

# Sibling construct: torch._dynamo.disallow_in_graph on an otherwise traceable operator.
torch._dynamo.disallow_in_graph(torch.neg)

def fn(x, scale):
    before = x.sin() * scale
    broken = torch.neg(before)
    return broken.cos() + before

x = torch.tensor([-1.0, 0.5, 2.0])
eager = fn(x.clone(), 2.0)
torch._dynamo.reset()
compiled = torch.compile(fn, backend="eager")(x.clone(), 2.0)
assert torch.allclose(compiled, eager), (compiled, eager)
