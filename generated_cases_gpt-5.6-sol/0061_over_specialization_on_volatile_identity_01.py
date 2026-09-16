# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Stateful property getter
# sibling    : user-defined property descriptor returning a changing integer

import torch

# Sibling construct: stateful property descriptor returning a changing integer.
class ScaleSource:
    def __init__(self, value):
        self.value = value

    @property
    def scale(self):
        current = self.value
        self.value += 1
        return current


def fn(x, source):
    return x * source.scale


def run(callable_fn, source):
    x = torch.tensor([1.0, 2.0])
    return [callable_fn(x, source) for _ in range(4)]


eager_results = run(fn, ScaleSource(2))
compiled_fn = torch.compile(fn, backend="eager")
compiled_results = run(compiled_fn, ScaleSource(2))

assert len(eager_results) == len(compiled_results)
for eager, compiled in zip(eager_results, compiled_results):
    torch.testing.assert_close(compiled, eager)
