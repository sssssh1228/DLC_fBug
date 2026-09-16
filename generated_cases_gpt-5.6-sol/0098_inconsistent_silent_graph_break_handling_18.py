# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Disabled Helper Frame
# sibling    : torch._dynamo.disable-decorated helper called from compiled code

import torch

# Sibling construct: torch._dynamo.disable-decorated helper that forces a skipped frame.
@torch._dynamo.disable
def opaque_helper(x):
    return x.square() + 3


def fn(x):
    y = x + 1
    z = opaque_helper(y)
    return z - 2


def strict_errors():
    return tuple(
        cls
        for cls in (
            getattr(torch._dynamo.exc, 'Unsupported', None),
            getattr(torch._dynamo.exc, 'InternalTorchDynamoError', None),
        )
        if cls is not None
    )


x = torch.tensor([-3.0, 0.5, 2.0])
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
    raise AssertionError('fullgraph=True silently resumed through a disabled helper frame')
