# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Active dispatch mode at graph break
# sibling    : TorchDispatchMode active while Dynamo encounters a graph break

import torch
from torch._dynamo.exc import Unsupported
from torch.utils._python_dispatch import TorchDispatchMode

# Sibling construct: TorchDispatchMode active while Dynamo encounters a graph break.
class PassthroughMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


def fn(x):
    y = torch.sin(x)
    torch._dynamo.graph_break()
    return y.cos()


x = torch.tensor([0.25, 0.5])
with PassthroughMode():
    eager = fn(x)

torch._dynamo.reset()
compiled = torch.compile(fn, backend="eager")
with PassthroughMode():
    actual = compiled(x)
torch.testing.assert_close(actual, eager)

torch._dynamo.reset()
strict = torch.compile(fn, backend="eager", fullgraph=True)
try:
    with PassthroughMode():
        strict(x)
except Unsupported:
    pass
else:
    raise AssertionError("fullgraph=True silently resumed with an active TorchDispatchMode")
