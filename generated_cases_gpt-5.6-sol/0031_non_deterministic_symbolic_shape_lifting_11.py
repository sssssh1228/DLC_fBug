# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Keyword-only wrap capture
# sibling    : keyword-only arguments with a captured symbolic dimension

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling construct: keyword-only wrap arguments with a captured symbolic dimension.
def fn(x, bias):
    batch = x.shape[0]

    def body(value, *, addend):
        return (value + addend) * (batch + 1)

    return wrap(body, x, addend=bias)


torch.manual_seed(0)
inputs = [
    (torch.randn(2, 3), torch.randn(2, 3)),
    (torch.randn(5, 3), torch.randn(5, 3)),
    (torch.randn(7, 3), torch.randn(7, 3)),
]
expected = [fn(x, bias) for x, bias in inputs]

for _ in range(4):
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    for (x, bias), eager_result in zip(inputs, expected):
        compiled_result = compiled(x.clone(), bias.clone())
        torch.testing.assert_close(compiled_result, eager_result)
