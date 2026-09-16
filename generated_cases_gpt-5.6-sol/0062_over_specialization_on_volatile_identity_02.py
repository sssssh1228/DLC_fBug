# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Mutable length protocol
# sibling    : user-defined __len__ method with evolving state

import torch

# Sibling construct: user-defined __len__ method with evolving state.
class GrowingLength:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        current = self.length
        self.length += 1
        return current


def fn(x, sized):
    return x + len(sized)


def run(callable_fn, sized):
    x = torch.tensor([10, 20])
    return [callable_fn(x, sized) for _ in range(4)]


eager_results = run(fn, GrowingLength(1))
compiled_fn = torch.compile(fn, backend="eager")
compiled_results = run(compiled_fn, GrowingLength(1))

assert len(eager_results) == len(compiled_results)
for eager, compiled in zip(eager_results, compiled_results):
    torch.testing.assert_close(compiled, eager)
