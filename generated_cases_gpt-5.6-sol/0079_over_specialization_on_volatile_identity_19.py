# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Mutable slice index
# sibling    : user-defined __index__ used as a tensor slice bound

import torch

# Sibling construct: user-defined __index__ used as a tensor slice bound.
class MutableIndex:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


def fn(x, stop):
    return x[:stop] * 2


def run(callable_fn):
    stop = MutableIndex(2)
    outputs = [callable_fn(torch.arange(6), stop)]
    stop.value = 5
    outputs.append(callable_fn(torch.arange(6), stop))
    stop.value = 1
    outputs.append(callable_fn(torch.arange(6), stop))
    return outputs


eager = run(fn)
compiled = run(torch.compile(fn, backend="eager"))
assert len(eager) == len(compiled)
for eager_value, compiled_value in zip(eager, compiled):
    assert torch.equal(eager_value, compiled_value), (eager_value, compiled_value)
