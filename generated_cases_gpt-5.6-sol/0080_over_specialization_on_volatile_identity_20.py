# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Changing truthiness
# sibling    : stateful user-defined __bool__ controlling a branch

import torch

# Sibling construct: stateful user-defined __bool__ controlling a branch.
class AlternatingTruth:
    def __init__(self, initial):
        self.next_value = initial
        self.calls = 0

    def __bool__(self):
        result = self.next_value
        self.next_value = not self.next_value
        self.calls += 1
        return result


def fn(x, condition):
    if condition:
        return x + 7
    return x - 3


def run(callable_fn):
    condition = AlternatingTruth(True)
    outputs = [callable_fn(torch.tensor(4), condition) for _ in range(4)]
    return outputs, condition.calls, condition.next_value


eager_outputs, eager_calls, eager_next = run(fn)
compiled_outputs, compiled_calls, compiled_next = run(torch.compile(fn, backend="eager"))
assert eager_calls == compiled_calls
assert eager_next == compiled_next
assert len(eager_outputs) == len(compiled_outputs)
for eager_value, compiled_value in zip(eager_outputs, compiled_outputs):
    assert torch.equal(eager_value, compiled_value), (eager_value, compiled_value)
