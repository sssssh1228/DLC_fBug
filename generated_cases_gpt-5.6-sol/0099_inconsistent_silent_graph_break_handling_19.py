# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Inconsistent/silent graph-break handling under fullgraph or error_on_graph_break contexts
# title      : Scalar Extraction In No Grad
# sibling    : Tensor.item graph break inside a torch.no_grad context

import torch

# Sibling construct: Tensor.item scalar extraction inside a state-restoring torch.no_grad context.
def fn(x):
    with torch.no_grad():
        scale = x.sum().item()
        return x * scale + 1


def strict_errors():
    return tuple(
        cls
        for cls in (
            getattr(torch._dynamo.exc, 'Unsupported', None),
            getattr(torch._dynamo.exc, 'InternalTorchDynamoError', None),
        )
        if cls is not None
    )


old_capture_scalars = torch._dynamo.config.capture_scalar_outputs
try:
    torch._dynamo.config.capture_scalar_outputs = False
    x = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
    eager_result = fn(x.clone())

    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend='eager')
    compiled_result = compiled_fn(x.clone())
    torch.testing.assert_close(compiled_result, eager_result)
    assert compiled_result.requires_grad == eager_result.requires_grad

    torch._dynamo.reset()
    strict_fn = torch.compile(fn, backend='eager', fullgraph=True)
    try:
        strict_fn(x.clone())
    except strict_errors():
        pass
    else:
        raise AssertionError('fullgraph=True silently resumed after Tensor.item inside no_grad')
finally:
    torch._dynamo.config.capture_scalar_outputs = old_capture_scalars
    torch._dynamo.reset()
