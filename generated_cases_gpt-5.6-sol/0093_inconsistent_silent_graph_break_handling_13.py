# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Disabled leaf frame skip
# sibling    : torch._dynamo.disable-decorated function called from compiled code

import torch
from torch._dynamo.exc import InternalTorchDynamoError, Unsupported

# Sibling construct: frame skip caused by a torch._dynamo.disable-decorated leaf.
@torch._dynamo.disable
def disabled_leaf(x):
    return torch.sin(x) + 2


def fn(x):
    y = x * 3
    y = disabled_leaf(y)
    return torch.cos(y)


x = torch.tensor([0.5, -1.0])
eager = fn(x)

torch._dynamo.reset()
compiled = torch.compile(fn, backend="eager")(x)
torch.testing.assert_close(compiled, eager)

torch._dynamo.reset()
strict_fn = torch.compile(fn, backend="eager", fullgraph=True)
try:
    strict_fn(x)
except (Unsupported, InternalTorchDynamoError):
    pass
else:
    raise AssertionError("fullgraph=True silently skipped a torch._dynamo.disable frame")
