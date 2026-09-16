# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Constant-result marker identity reuse
# sibling    : assume_constant_result function marker

import gc
import torch

# Sibling construct: assume_constant_result function marker.
def make_original():
    def original():
        return -100
    return original

original = make_original()
torch._dynamo.assume_constant_result(original)
original_id = id(original)
del original
gc.collect()

state = {"value": 2}

def make_candidate():
    box = state

    def candidate():
        return box["value"]

    return candidate

def find_reused_function(target_id):
    for _ in range(100000):
        fn = make_candidate()
        if id(fn) == target_id:
            return fn
        del fn
    return make_candidate()

candidate = find_reused_function(original_id)

def model(x):
    return x + candidate()

x = torch.tensor([1.0, 4.0])
compiled = torch.compile(model, backend="eager", fullgraph=True)

eager_first = model(x)
actual_first = compiled(x)
torch.testing.assert_close(actual_first, eager_first)

state["value"] = 9
eager_second = model(x)
actual_second = compiled(x)
torch.testing.assert_close(actual_second, eager_second)
