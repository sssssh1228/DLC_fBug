# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Volatile descriptor result
# sibling    : stateful descriptor __get__

import torch

# Sibling construct: stateful descriptor __get__.
class CyclingValue:
    def __get__(self, instance, owner):
        if instance is None:
            return self
        result = instance.values[instance.position]
        instance.position = (instance.position + 1) % len(instance.values)
        return result


class Rotor:
    level = CyclingValue()

    def __init__(self):
        self.values = (2, 5, 3)
        self.position = 0


def fn(x, rotor):
    return x * rotor.level


def run(callable_fn):
    rotor = Rotor()
    x = torch.tensor([1.0, 4.0])
    outputs = [callable_fn(x, rotor) for _ in range(7)]
    return outputs, rotor.position


eager_outputs, eager_position = run(fn)
torch._dynamo.reset()
compiled_fn = torch.compile(fn, backend="eager")
compiled_outputs, compiled_position = run(compiled_fn)

assert eager_position == compiled_position
assert len(eager_outputs) == len(compiled_outputs)
for eager, compiled in zip(eager_outputs, compiled_outputs):
    torch.testing.assert_close(compiled, eager)
