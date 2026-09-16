# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Builtin print executes once at the break
# sibling    : print side-effecting builtin graph break

import contextlib
import io
import torch

# Sibling construct: print side-effecting builtin graph break.
def fn(x):
    y = x + 1
    print("elements", x.numel())
    return {"first": y[0], "total": y.sum()}

def run(callable_fn, x):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        result = callable_fn(x)
    return result, stream.getvalue()

x = torch.tensor([4.0, 5.0, 6.0])
eager_result, eager_output = run(fn, x.clone())
torch._dynamo.reset()
compiled_fn = torch.compile(fn, backend="eager")
compiled_result, compiled_output = run(compiled_fn, x.clone())
assert eager_result.keys() == compiled_result.keys()
for key in eager_result:
    assert torch.equal(compiled_result[key], eager_result[key]), key
assert compiled_output == eager_output, (compiled_output, eager_output)
