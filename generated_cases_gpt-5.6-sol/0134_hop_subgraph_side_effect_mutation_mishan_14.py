# -*- pattern-testcase -*-
# root_cause : Side-Effect Tracking Fidelity
# pattern    : HOP subgraph side-effect mutation mishandling
# title      : Checkpoint descriptor assignment
# sibling    : property descriptor setter

import torch
from torch.utils.checkpoint import checkpoint


class Accumulator:
    def __init__(self):
        self._value = 0

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value


def make_case():
    box = Accumulator()

    def body(x):
        # Sibling construct under test: property descriptor setter.
        box.value = box.value + 3
        return x + box.value

    def fn(x):
        return checkpoint(body, x, use_reentrant=False)

    def snapshot():
        return box.value

    return fn, snapshot


eager_fn, eager_snapshot = make_case()
eager_out = eager_fn(torch.tensor(4.0))
eager_state = eager_snapshot()

torch._dynamo.reset()
compiled_fn_source, compiled_snapshot = make_case()
compiled_fn = torch.compile(compiled_fn_source, backend="eager")
compiled_out = compiled_fn(torch.tensor(4.0))
compiled_state = compiled_snapshot()

torch.testing.assert_close(compiled_out, eager_out)
assert compiled_state == eager_state, (compiled_state, eager_state)
