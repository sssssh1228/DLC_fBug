# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Dispatch mode graph break
# sibling    : TorchDispatchMode active across a graph break

import torch
from torch._dynamo.exc import InternalTorchDynamoError, Unsupported
from torch.utils._python_dispatch import TorchDispatchMode

# Sibling construct: active TorchDispatchMode spanning a graph break.
class PassthroughMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


def fn(x):
    with PassthroughMode():
        y = torch.sin(x)
        torch._dynamo.graph_break()
        return torch.cos(y)


x = torch.tensor([0.25, -0.75])
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
    raise AssertionError("fullgraph=True silently resumed while a TorchDispatchMode was active")
