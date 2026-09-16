# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Mutable index protocol
# sibling    : user-defined __index__ used for tensor slicing

import operator
import torch

# Sibling construct: user-defined __index__ used for tensor slicing.
class Bound:
    def __init__(self, position):
        self.position = position

    def __index__(self):
        return self.position


def fn(x, bound):
    return x[:operator.index(bound)].sum()


eager_bound = Bound(1)
compiled_bound = Bound(1)
compiled_fn = torch.compile(fn, backend="eager")
x = torch.tensor([1, 2, 3, 4, 5])

for position in [1, 3, 5, 2]:
    eager_bound.position = position
    compiled_bound.position = position
    eager_result = fn(x, eager_bound)
    compiled_result = compiled_fn(x, compiled_bound)
    assert torch.equal(compiled_result, eager_result), (compiled_result, eager_result)
