# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Dispatch mode must not bypass fullgraph
# sibling    : active TorchDispatchMode around a compiled call

import torch
from torch._dynamo.exc import InternalTorchDynamoError, Unsupported
from torch.utils._python_dispatch import TorchDispatchMode


# Sibling construct: active TorchDispatchMode around a compiled call.
class PassthroughMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **({} if kwargs is None else kwargs))


def fn(x):
    y = torch.cos(x)
    torch._dynamo.graph_break()
    return y.square()


def assert_strict_failure(call):
    try:
        call()
    except (Unsupported, InternalTorchDynamoError):
        return
    raise AssertionError("an active TorchDispatchMode silently bypassed fullgraph=True")


x = torch.randn(5)
with PassthroughMode():
    eager = fn(x)
torch.testing.assert_close(eager, torch.cos(x).square())

torch._dynamo.reset()
compiled = torch.compile(fn, backend="eager", fullgraph=True)
with PassthroughMode():
    assert_strict_failure(lambda: compiled(x))
