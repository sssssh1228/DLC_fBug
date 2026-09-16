# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Forbidden marker identity reuse
# sibling    : forbid_in_graph function marker

import gc
import torch

# Sibling construct: forbid_in_graph function marker.
def make_original():
    def original(x, *, amount=1):
        return x + amount
    return original

original = make_original()
torch._dynamo.forbid_in_graph(original)
original_id = id(original)
del original
gc.collect()

def make_candidate():
    def candidate(x, *, amount=2):
        return x * amount, x - amount

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
    product, difference = candidate(x, amount=4)
    return {"product": product, "difference": difference}

x = torch.tensor([2.0, 5.0, 8.0])
eager = model(x)
compiled = torch.compile(model, backend="eager", fullgraph=True)
actual = compiled(x)
torch.testing.assert_close(actual["product"], eager["product"])
torch.testing.assert_close(actual["difference"], eager["difference"])
