# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Polyfill registry after function collection
# sibling    : substitute_in_graph polyfill registration

import gc
import torch

# Sibling construct: substitute_in_graph polyfill registration.
def make_original():
    def original(x):
        return x + 100
    return original

original = make_original()
original_id = id(original)

@torch._dynamo.substitute_in_graph(original)
def polyfill(x):
    return x - 1000

del original
gc.collect()

def make_candidate():
    factor = 3

    def candidate(x):
        return x * factor + 1

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
    return candidate(x)

x = torch.tensor([1.0, 2.0, 3.0])
eager = model(x)
compiled = torch.compile(model, backend="eager", fullgraph=True)
actual = compiled(x)
torch.testing.assert_close(actual, eager)
