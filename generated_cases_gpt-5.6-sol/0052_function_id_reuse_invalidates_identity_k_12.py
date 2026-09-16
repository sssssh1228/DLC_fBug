# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Disallowed registry after lambda collection
# sibling    : disallow_in_graph callable registry

import gc
import torch

# Sibling construct: disallow_in_graph callable registration.
original = lambda x, bias=4: x + bias
original_id = id(original)
torch._dynamo.allow_in_graph(original)
torch._dynamo.disallow_in_graph(original)
del original
gc.collect()

def make_candidate():
    return lambda x, scale=2: x * scale

def find_reused_function(target_id):
    for _ in range(100000):
        fn = make_candidate()
        if id(fn) == target_id:
            return fn
        del fn
    return make_candidate()

candidate = find_reused_function(original_id)

def model(x):
    return candidate(x, scale=5) - 2

x = torch.tensor([-2.0, 0.5, 4.0])
eager = model(x)
compiled = torch.compile(model, backend="eager", fullgraph=True)
actual = compiled(x)
torch.testing.assert_close(actual, eager)
