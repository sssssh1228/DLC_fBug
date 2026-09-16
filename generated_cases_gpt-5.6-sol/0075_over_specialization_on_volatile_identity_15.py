# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Changing identity relationship
# sibling    : identity comparison between mutable object attributes

import torch

# Sibling construct: identity comparison between mutable object attributes.
class Holder:
    def __init__(self):
        self.left = object()
        self.right = object()
        self.current = self.left


def fn(x, holder):
    is_left = holder.current is holder.left
    holder.current = holder.right if is_left else holder.left
    if is_left:
        return x.square()
    return -x


def run(callable_fn):
    holder = Holder()
    x = torch.tensor([2.0, 3.0])
    outputs = [callable_fn(x, holder) for _ in range(6)]
    final_is_left = holder.current is holder.left
    return outputs, final_is_left


eager_outputs, eager_final_is_left = run(fn)
torch._dynamo.reset()
compiled_fn = torch.compile(fn, backend="eager")
compiled_outputs, compiled_final_is_left = run(compiled_fn)

assert eager_final_is_left == compiled_final_is_left
assert len(eager_outputs) == len(compiled_outputs)
for eager, compiled in zip(eager_outputs, compiled_outputs):
    torch.testing.assert_close(compiled, eager)
