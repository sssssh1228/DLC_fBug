# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Mutable descriptor lookup
# sibling    : descriptor __get__ backed by mutable instance data

import torch

# Sibling construct: descriptor __get__ backed by mutable instance data.
class ComputedValue:
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.raw * 2 + 1


class Config:
    value = ComputedValue()

    def __init__(self, raw):
        self.raw = raw


def fn(x, config):
    return x - config.value


eager_config = Config(1)
compiled_config = Config(1)
compiled_fn = torch.compile(fn, backend="eager")
x = torch.tensor([10.0, 20.0])

for raw in [1, 3, -2, 7]:
    eager_config.raw = raw
    compiled_config.raw = raw
    eager_result = fn(x, eager_config)
    compiled_result = compiled_fn(x, compiled_config)
    assert torch.equal(compiled_result, eager_result), (compiled_result, eager_result)
