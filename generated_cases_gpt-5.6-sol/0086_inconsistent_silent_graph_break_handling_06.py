# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Contextlib manager with strict graph break
# sibling    : generator-based contextlib context manager

import contextlib

import torch
from torch._dynamo.exc import InternalTorchDynamoError, Unsupported


# Sibling construct: generator-based contextlib context manager.
@contextlib.contextmanager
def passthrough():
    yield


def fn(x):
    with passthrough():
        y = torch.sin(x)
        torch._dynamo.graph_break()
        return y + 1


def assert_strict_failure(call):
    try:
        call()
    except (Unsupported, InternalTorchDynamoError):
        return
    raise AssertionError("fullgraph=True silently resumed after a graph break")


x = torch.randn(4)
eager = fn(x)
torch.testing.assert_close(eager, torch.sin(x) + 1)

torch._dynamo.reset()
compiled = torch.compile(fn, backend="eager", fullgraph=True)
assert_strict_failure(lambda: compiled(x))
