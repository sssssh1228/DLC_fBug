# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Nested regional graph-break policy
# sibling    : nested error_on_graph_break contexts that locally override resume behavior

import torch
from torch._dynamo.exc import Unsupported

# Sibling construct: nested error_on_graph_break contexts that locally override resume behavior.
def resumable_region(x):
    y = x + 1
    with torch._dynamo.error_on_graph_break(False):
        torch._dynamo.graph_break()
        y = y * 3
    return y


def strict_region(x):
    y = x - 1
    with torch._dynamo.error_on_graph_break(True):
        torch._dynamo.graph_break()
        y = y.square()
    return y


x = torch.tensor([2.0, 4.0])
eager_resumable = resumable_region(x)
torch._dynamo.reset()
with torch._dynamo.error_on_graph_break(True):
    compiled_resumable = torch.compile(resumable_region, backend="eager")
    actual_resumable = compiled_resumable(x)
torch.testing.assert_close(actual_resumable, eager_resumable)

eager_strict = strict_region(x)
torch.testing.assert_close(eager_strict, (x - 1).square())
torch._dynamo.reset()
try:
    with torch._dynamo.error_on_graph_break(False):
        compiled_strict = torch.compile(strict_region, backend="eager")
        compiled_strict(x)
except Unsupported:
    pass
else:
    raise AssertionError("an inner error_on_graph_break(True) region silently resumed")
