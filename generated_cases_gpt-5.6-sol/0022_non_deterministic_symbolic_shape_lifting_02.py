# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Pytree symbolic size lifting
# sibling    : Nested pytree arguments containing tensors and symbolic sizes

import torch
from torch._higher_order_ops.wrap import wrap

# Sibling construct: nested pytree arguments containing tensors and symbolic sizes.
def body(tree):
    x = tree["payload"][0]
    rows = tree["metadata"]["rows"]
    return x.reshape(rows, -1).cos()


def fn(x):
    tree = {"payload": (x,), "metadata": {"rows": x.shape[0]}}
    return wrap(body, tree)


shapes = [(3, 4), (5, 4), (2, 7)]
for _ in range(3):
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="eager", dynamic=True)
    for shape in shapes:
        x = torch.randn(shape)
        eager = fn(x)
        actual = compiled(x)
        torch.testing.assert_close(actual, eager)
