# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Toggling truth value
# sibling    : stateful __bool__ protocol

import torch

# Sibling construct: stateful __bool__ protocol.
class Switch:
    def __init__(self):
        self.enabled = True

    def __bool__(self):
        result = self.enabled
        self.enabled = not self.enabled
        return result


def fn(x, switch):
    if switch:
        return x + 10
    return x - 10


def run(callable_fn):
    switch = Switch()
    x = torch.tensor([1.0, 2.0])
    outputs = [callable_fn(x, switch) for _ in range(6)]
    return outputs, switch.enabled


eager_outputs, eager_enabled = run(fn)
torch._dynamo.reset()
compiled_fn = torch.compile(fn, backend="eager")
compiled_outputs, compiled_enabled = run(compiled_fn)

assert eager_enabled == compiled_enabled
assert len(eager_outputs) == len(compiled_outputs)
for eager, compiled in zip(eager_outputs, compiled_outputs):
    torch.testing.assert_close(compiled, eager)
