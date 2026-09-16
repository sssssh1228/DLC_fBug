# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Nested region permits graph-break resume
# sibling    : nested error_on_graph_break(False) region

import torch


# Sibling construct: nested error_on_graph_break(False) region.
def fn(x):
    y = x + 2
    with torch._dynamo.error_on_graph_break(False):
        torch._dynamo.graph_break()
        y = torch.relu(y)
    return y * 3


x = torch.randn(6)
with torch._dynamo.error_on_graph_break(True):
    eager = fn(x)

torch._dynamo.reset()
compiled = torch.compile(fn, backend="eager")
with torch._dynamo.error_on_graph_break(True):
    actual = compiled(x)

torch.testing.assert_close(actual, eager)
