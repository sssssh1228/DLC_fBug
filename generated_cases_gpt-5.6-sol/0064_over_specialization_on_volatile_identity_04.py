# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Changing index conversion
# sibling    : user-defined __index__ method used for tensor slicing

import torch

# Sibling construct: stateful user-defined __index__ method used for slicing.
class GrowingIndex:
    def __init__(self, index):
        self.index = index

    def __index__(self):
        current = self.index
        self.index += 1
        return current


def fn(x, stop):
    return x[:stop]


def run(callable_fn, stop):
    x = torch.arange(8)
    return [callable_fn(x, stop) for _ in range(4)]


eager_results = run(fn, GrowingIndex(2))
compiled_fn = torch.compile(fn, backend="eager")

stop = GrowingIndex(2)

for i in range(4):
    print(f"Before call {i}: stop.index = {stop.index}")
    result = compiled_fn(torch.arange(8), stop)
    print(f"After call {i}: stop.index = {stop.index}")
    print(result)

assert len(eager_results) == len(compiled_results)
for eager, compiled in zip(eager_results, compiled_results):
    torch.testing.assert_close(compiled, eager)
