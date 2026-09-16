# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Graph break after local mutation
# sibling    : graph break with a non-empty local side-effect checkpoint

import torch
from torch._dynamo.exc import Unsupported

# Sibling construct: graph break with a non-empty local side-effect checkpoint.
def fn(x):
    values = [x + 1]
    values.append(x * 2)
    torch._dynamo.graph_break()
    first = values.pop(0)
    return first + values[0]


x = torch.tensor([1.0, 3.0])
eager = fn(x)
torch._dynamo.reset()
compiled = torch.compile(fn, backend="eager")
actual = compiled(x)
torch.testing.assert_close(actual, eager)

torch._dynamo.reset()
strict = torch.compile(fn, backend="eager", fullgraph=True)
try:
    strict(x)
except Unsupported:
    pass
else:
    raise AssertionError("fullgraph=True silently resumed from a non-empty side-effect checkpoint")
