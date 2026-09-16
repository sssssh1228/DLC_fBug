# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Guarded wrap branches
# sibling    : symbolic shape predicate guarding alternative wrap bodies

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling construct: symbolic predicate guards around alternative wrap bodies.
def fn(x):
    rows = x.shape[0]
    columns = x.shape[1]

    if columns % 2 == 0:
        def even_body(value):
            return value[:, ::2] + rows

        return wrap(even_body, x)

    def odd_body(value):
        return value[:, 1::2] - rows

    return wrap(odd_body, x)


torch.manual_seed(4)
inputs = [
    torch.randn(3, 6),
    torch.randn(4, 8),
    torch.randn(5, 5),
    torch.randn(2, 7),
]
expected = [fn(x) for x in inputs]

for _ in range(4):
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    for x, eager_result in zip(inputs, expected):
        compiled_result = compiled(x.clone())
        torch.testing.assert_close(compiled_result, eager_result)
