# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Stateful property value
# sibling    : side-effecting @property getter

import torch

# Sibling construct: side-effecting @property getter.
class Counter:
    def __init__(self):
        self.current = 0

    @property
    def value(self):
        self.current += 1
        return self.current


def fn(x, counter):
    return x * counter.value


eager_counter = Counter()
compiled_counter = Counter()
compiled_fn = torch.compile(fn, backend="eager")
x = torch.tensor([1.0, 2.0])

for _ in range(4):
    eager_result = fn(x, eager_counter)
    compiled_result = compiled_fn(x, compiled_counter)
    assert torch.equal(compiled_result, eager_result), (compiled_result, eager_result)
    assert compiled_counter.current == eager_counter.current
