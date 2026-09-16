# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Stateful descriptor accesses
# sibling    : custom descriptor __get__ backed by mutable instance state

import torch

# Sibling construct: custom descriptor __get__ backed by mutable instance state.
class IncrementingDescriptor:
    def __get__(self, instance, owner):
        if instance is None:
            return self
        instance.reads += 1
        return instance.base + instance.reads


class Holder:
    observed = IncrementingDescriptor()

    def __init__(self, base):
        self.base = base
        self.reads = 0


def fn(x, holder):
    return x + holder.observed + holder.observed


def run(callable_fn):
    holder = Holder(5)
    first = callable_fn(torch.tensor(1), holder)
    holder.base = 10
    second = callable_fn(torch.tensor(1), holder)
    return [first, second], holder.reads


eager_outputs, eager_reads = run(fn)
compiled_outputs, compiled_reads = run(torch.compile(fn, backend="eager"))
assert eager_reads == compiled_reads
assert len(eager_outputs) == len(compiled_outputs)
for eager_value, compiled_value in zip(eager_outputs, compiled_outputs):
    assert torch.equal(eager_value, compiled_value), (eager_value, compiled_value)
