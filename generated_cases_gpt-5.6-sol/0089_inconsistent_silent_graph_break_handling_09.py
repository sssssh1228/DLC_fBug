# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Disabled callee must fail in fullgraph
# sibling    : torch._dynamo.disable-decorated callee frame skip

import torch
from torch._dynamo.exc import InternalTorchDynamoError, Unsupported


# Sibling construct: torch._dynamo.disable-decorated callee frame skip.
@torch._dynamo.disable
def eager_only(x):
    return torch.tanh(x) + 2


def fn(x):
    y = x * 2
    return eager_only(y) - 1


def assert_strict_failure(call):
    try:
        call()
    except (Unsupported, InternalTorchDynamoError):
        return
    raise AssertionError("a disabled callee silently resumed under fullgraph=True")


x = torch.randn(3)
eager = fn(x)
torch.testing.assert_close(eager, torch.tanh(x * 2) + 1)

torch._dynamo.reset()
compiled = torch.compile(fn, backend="eager", fullgraph=True)
assert_strict_failure(lambda: compiled(x))
