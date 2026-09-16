# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Stateful iterator advancement
# sibling    : user-defined __next__ with changing internal position

import torch

# Sibling construct: user-defined __next__ with changing internal position.
class NumberStream:
    def __init__(self, values):
        self.values = list(values)
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= len(self.values):
            raise StopIteration
        value = self.values[self.position]
        self.position += 1
        return value


def fn(x, stream):
    return x + next(stream)


eager_stream = NumberStream([1, 4, -2, 8])
compiled_stream = NumberStream([1, 4, -2, 8])
compiled_fn = torch.compile(fn, backend="eager")
x = torch.tensor([3.0, 5.0])

for _ in range(4):
    eager_result = fn(x, eager_stream)
    compiled_result = compiled_fn(x, compiled_stream)
    assert torch.equal(compiled_result, eager_result), (compiled_result, eager_result)
    assert compiled_stream.position == eager_stream.position
