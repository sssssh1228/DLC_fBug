# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Stateful property value
# sibling    : stateful @property access

import torch

# Sibling construct: stateful @property access.
class Counter:
    def __init__(self):
        self.count = 0

    @property
    def value(self):
        result = self.count
        self.count += 1
        return result


def fn(x, state):
    return x * 2 + state.value


def run(callable_fn):
    state = Counter()
    outputs = [callable_fn(torch.tensor(float(i)), state) for i in range(5)]
    return outputs, state.count


eager_outputs, eager_count = run(fn)
torch._dynamo.reset()
compiled_fn = torch.compile(fn, backend="eager")
compiled_outputs, compiled_count = run(compiled_fn)

assert eager_count == compiled_count
assert len(eager_outputs) == len(compiled_outputs)
for eager, compiled in zip(eager_outputs, compiled_outputs):
    torch.testing.assert_close(compiled, eager)
