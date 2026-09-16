# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Nested graph-break policy override
# sibling    : nested error_on_graph_break false and true regions

import torch
from torch._dynamo.exc import InternalTorchDynamoError, Unsupported

# Sibling construct: nested error_on_graph_break regions toggling resume behavior.
def resumable_region(x):
    y = x + 1
    with torch._dynamo.error_on_graph_break(False):
        y = y * 2
        torch._dynamo.graph_break()
        y = y - 3
    return torch.sin(y)


def strict_region(x):
    y = torch.cos(x)
    with torch._dynamo.error_on_graph_break(True):
        torch._dynamo.graph_break()
        y = y + 4
    return y


x = torch.tensor([0.2, -0.4])
eager_resumable = resumable_region(x)

# The inner False region must override the outer strict policy and preserve eager semantics.
torch._dynamo.reset()
with torch._dynamo.error_on_graph_break(True):
    compiled_resumable = torch.compile(resumable_region, backend="eager")(x)
torch.testing.assert_close(compiled_resumable, eager_resumable)

# The inner True region must reject the same class of graph break rather than silently resume.
eager_strict = strict_region(x)
torch.testing.assert_close(eager_strict, torch.cos(x) + 4)
torch._dynamo.reset()
compiled_strict = torch.compile(strict_region, backend="eager")
try:
    compiled_strict(x)
except (Unsupported, InternalTorchDynamoError):
    pass
else:
    raise AssertionError("error_on_graph_break(True) silently resumed after a graph break")
