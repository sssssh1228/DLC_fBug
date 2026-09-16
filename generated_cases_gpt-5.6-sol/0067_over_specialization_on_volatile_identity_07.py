# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Over-specialization on volatile/identity values
# title      : Mutable custom length
# sibling    : user-defined __len__ over mutable state

import torch

# Sibling construct: user-defined __len__ over mutable state.
class Bucket:
    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)


def fn(x, bucket):
    return x + len(bucket)


eager_bucket = Bucket([0])
compiled_bucket = Bucket([0])
compiled_fn = torch.compile(fn, backend="eager")
x = torch.tensor([2, 4, 6])

for value in [None, 10, 20, 30]:
    if value is not None:
        eager_bucket.items.append(value)
        compiled_bucket.items.append(value)
    eager_result = fn(x, eager_bucket)
    compiled_result = compiled_fn(x, compiled_bucket)
    assert torch.equal(compiled_result, eager_result), (compiled_result, eager_result)
