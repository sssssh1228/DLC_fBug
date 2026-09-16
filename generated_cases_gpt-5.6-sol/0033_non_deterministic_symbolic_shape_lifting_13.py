# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Inlined helper wrap
# sibling    : inlined helper function whose wrap body captures a symbolic size

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling construct: inlined helper function containing a symbol-capturing wrap.
def apply_wrapped(value, factor=2.0):
    length = value.shape[-1]

    def body(tensor):
        return tensor.cos() + length * factor

    return wrap(body, value)


def fn(x):
    pieces = [apply_wrapped(x), x.sin()]
    first, second = pieces
    return first - second


torch.manual_seed(2)
inputs = [
    torch.randn(2, 3),
    torch.randn(2, 6),
    torch.randn(2, 9),
]
expected = [fn(x) for x in inputs]

for _ in range(4):
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    for x, eager_result in zip(inputs, expected):
        compiled_result = compiled(x.clone())
        torch.testing.assert_close(compiled_result, eager_result)
