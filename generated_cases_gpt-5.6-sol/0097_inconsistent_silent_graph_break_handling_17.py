# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Dispatch Mode Graph Break
# sibling    : active TorchDispatchMode spanning a graph break

import torch
from torch.utils._python_dispatch import TorchDispatchMode

# Sibling construct: active TorchDispatchMode around operations on both sides of a graph break.
class PassthroughMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **({} if kwargs is None else kwargs))


def fn(x):
    with PassthroughMode():
        y = torch.sin(x)
        torch._dynamo.graph_break()
        return torch.cos(y) + 1


def strict_errors():
    return tuple(
        cls
        for cls in (
            getattr(torch._dynamo.exc, 'Unsupported', None),
            getattr(torch._dynamo.exc, 'InternalTorchDynamoError', None),
        )
        if cls is not None
    )


x = torch.linspace(-1.0, 1.0, 5)
eager_result = fn(x.clone())

torch._dynamo.reset()
compiled_fn = torch.compile(fn, backend='eager')
compiled_result = compiled_fn(x.clone())
torch.testing.assert_close(compiled_result, eager_result)

torch._dynamo.reset()
strict_fn = torch.compile(fn, backend='eager', fullgraph=True)
try:
    strict_fn(x.clone())
except strict_errors():
    pass
else:
    raise AssertionError('fullgraph=True silently resumed with an active TorchDispatchMode')
