# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Function id() reuse invalidates identity-keyed registries
# title      : Recycled constant-result identity
# sibling    : torch._dynamo.assume_constant_result

import torch

# Sibling construct: assume_constant_result function marker.
GLOBAL_BIAS = 0


def make_old():
    def old():
        return 1234

    return old


def make_candidate():
    def candidate():
        return GLOBAL_BIAS

    return candidate


def find_reused_id(target_id, factory):
    for _ in range(4096):
        batch = [factory() for _ in range(128)]
        for value in batch:
            if id(value) == target_id:
                return value
    raise RuntimeError("CPython did not reuse the function id")


victim = make_old()
torch._dynamo.assume_constant_result(victim)
stale_id = id(victim)
del victim
candidate = find_reused_id(stale_id, make_candidate)
assert id(candidate) == stale_id


def run(x):
    return x + candidate()


global_input = torch.tensor([1.0, 2.0])
GLOBAL_BIAS = 2
eager_first = run(global_input.clone())
GLOBAL_BIAS = 9
eager_second = run(global_input.clone())

compiled_run = torch.compile(run, backend="eager", fullgraph=True)
GLOBAL_BIAS = 2
compiled_first = compiled_run(global_input.clone())
GLOBAL_BIAS = 9
compiled_second = compiled_run(global_input.clone())

torch.testing.assert_close(compiled_first, eager_first)
torch.testing.assert_close(compiled_second, eager_second)
