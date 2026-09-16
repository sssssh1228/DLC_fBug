# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Stateful truth conversion
# sibling    : user-defined __bool__ method controlling a branch

import torch

# Sibling construct: stateful user-defined __bool__ method controlling a branch.
class AlternatingFlag:
    def __init__(self, value):
        self.value = value

    def __bool__(self):
        current = self.value
        self.value = not self.value
        return current


def fn(x, flag):
    if flag:
        return x + 3
    return x - 5


def run(callable_fn, flag):
    x = torch.tensor([4.0, 8.0])
    return [callable_fn(x, flag) for _ in range(6)]


eager_results = run(fn, AlternatingFlag(True))
compiled_fn = torch.compile(fn, backend="eager")
compiled_results = run(compiled_fn, AlternatingFlag(True))

assert len(eager_results) == len(compiled_results)
for eager, compiled in zip(eager_results, compiled_results):
    torch.testing.assert_close(compiled, eager)
