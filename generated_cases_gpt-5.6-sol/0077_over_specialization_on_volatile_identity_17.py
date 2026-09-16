# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Stateful property reads
# sibling    : property getter whose returned integer changes on every access

import torch

# Sibling construct: property getter whose returned integer changes on every access.
class TickingProperty:
    def __init__(self, value, step):
        self._value = value
        self.step = step

    @property
    def value(self):
        result = self._value
        self._value += self.step
        return result


def fn(x, obj):
    return x * obj.value


def run(callable_fn):
    obj = TickingProperty(2, 3)
    outputs = [callable_fn(torch.tensor(2), obj) for _ in range(3)]
    return outputs, obj._value


eager_outputs, eager_final_value = run(fn)
compiled_outputs, compiled_final_value = run(torch.compile(fn, backend="eager"))
assert eager_final_value == compiled_final_value
assert len(eager_outputs) == len(compiled_outputs)
for eager_value, compiled_value in zip(eager_outputs, compiled_outputs):
    assert torch.equal(eager_value, compiled_value), (eager_value, compiled_value)
