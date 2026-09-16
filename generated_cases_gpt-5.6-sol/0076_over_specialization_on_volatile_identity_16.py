# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Mutable custom length
# sibling    : user-defined __len__ with a changing integer attribute

import torch

# Sibling construct: user-defined __len__ with a changing integer attribute.
class MutableLength:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


def fn(x, obj):
    return x + len(obj)


def run(callable_fn):
    obj = MutableLength(1)
    outputs = [callable_fn(torch.tensor(10), obj)]
    obj.length = 4
    outputs.append(callable_fn(torch.tensor(10), obj))
    obj.length = 2
    outputs.append(callable_fn(torch.tensor(10), obj))
    return outputs


eager = run(fn)
compiled = run(torch.compile(fn, backend="eager"))
assert len(eager) == len(compiled)
for eager_value, compiled_value in zip(eager, compiled):
    assert torch.equal(eager_value, compiled_value), (eager_value, compiled_value)
