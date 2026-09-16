# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Error-on-break after accumulated graph
# sibling    : graph break after a non-empty compilation checkpoint

import torch
from torch._dynamo.exc import InternalTorchDynamoError, Unsupported


# Sibling construct: graph break after a non-empty compilation checkpoint.
def fn(x, take_break):
    y = torch.sin(x) + 1
    if take_break:
        torch._dynamo.graph_break()
    else:
        y = -y
    return y + torch.cos(x)


def assert_strict_failure(call):
    try:
        call()
    except (Unsupported, InternalTorchDynamoError):
        return
    raise AssertionError("error_on_graph_break ignored a non-empty checkpoint")


x = torch.randn(7)
eager = fn(x, True)
torch.testing.assert_close(eager, torch.sin(x) + 1 + torch.cos(x))

torch._dynamo.reset()
compiled = torch.compile(fn, backend="eager")
with torch._dynamo.error_on_graph_break(True):
    assert_strict_failure(lambda: compiled(x, True))
