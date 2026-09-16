# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Constant-result marker does not leak to a closure
# sibling    : torch._dynamo.assume_constant_result function marking

import gc
import torch

# Sibling construct: assume_constant_result marking followed by a closure allocation.
def try_reuse(dead_id, factory, attempts=10000):
    for _ in range(attempts):
        candidates = [factory() for _ in range(16)]
        for candidate in candidates:
            if id(candidate) == dead_id:
                return candidate
        del candidates, candidate
    return factory()


def make_constant():
    def constant():
        return 11
    return constant


state = [2]


def make_candidate():
    def candidate():
        return state[0]
    return candidate


constant = torch._dynamo.assume_constant_result(make_constant())
dead_id = id(constant)
del constant
gc.collect()
candidate = try_reuse(dead_id, make_candidate)


def workload(x):
    return x + candidate()


compiled_workload = torch.compile(workload, backend="eager")
x = torch.tensor([1.0, 3.0])
state[0] = 2
eager_first = workload(x)
state[0] = 7
eager_second = workload(x)
state[0] = 2
compiled_first = compiled_workload(x)
state[0] = 7
compiled_second = compiled_workload(x)
torch.testing.assert_close(compiled_first, eager_first)
torch.testing.assert_close(compiled_second, eager_second)
