# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Disable wrapper identity reuse
# sibling    : disable decorator wrapper metadata

import gc
import torch

# Sibling construct: disable decorator wrapper metadata.
def make_original():
    def original(x, /):
        return x.sin()
    return original

original = make_original()
disabled = torch._dynamo.disable(original)
disabled_id = id(disabled)
del disabled
del original
gc.collect()

def make_candidate():
    offset = 2.0

    def candidate(x, /):
        return x.cos() + offset

    return candidate

def find_reused_function(target_id):
    for _ in range(100000):
        fn = make_candidate()
        if id(fn) == target_id:
            return fn
        del fn
    return make_candidate()

candidate = find_reused_function(disabled_id)

def model(x):
    return candidate(x) * 3

x = torch.tensor([0.0, 0.25, 1.0])
eager = model(x)
compiled = torch.compile(model, backend="eager", fullgraph=True)
actual = compiled(x)
torch.testing.assert_close(actual, eager)
