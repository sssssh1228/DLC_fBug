# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Equivalent callable instances
# sibling    : user-defined __call__ across changing object identities

import torch

# Sibling construct: user-defined __call__ across distinct object identities.
class AffineOperation:
    def __init__(self, scale, bias):
        self.scale = scale
        self.bias = bias

    def __call__(self, x):
        return x * self.scale + self.bias


def fn(x, operation):
    return operation(x)


x = torch.tensor([1.0, 3.0])
eager_operations = [
    AffineOperation(2, 1),
    AffineOperation(2, 1),
    AffineOperation(3, -1),
    AffineOperation(2, 1),
]
compiled_operations = [
    AffineOperation(2, 1),
    AffineOperation(2, 1),
    AffineOperation(3, -1),
    AffineOperation(2, 1),
]

eager_results = [fn(x, operation) for operation in eager_operations]
compiled_fn = torch.compile(fn, backend="eager")
compiled_results = [compiled_fn(x, operation) for operation in compiled_operations]

assert len(eager_results) == len(compiled_results)
for eager, compiled in zip(eager_results, compiled_results):
    torch.testing.assert_close(compiled, eager)
