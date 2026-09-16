# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Disabled frame under fullgraph
# sibling    : call into a torch._dynamo.disable-decorated frame

import torch
from torch._dynamo.exc import Unsupported

# Sibling construct: call into a torch._dynamo.disable-decorated frame.
@torch._dynamo.disable
def eager_only(x):
    return torch.tanh(x) + 2


def fn(x):
    y = x * 3
    y = eager_only(y)
    return y - 1


x = torch.tensor([-1.0, 0.5, 2.0])
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
    raise AssertionError("fullgraph=True silently skipped a torch._dynamo.disable frame")
