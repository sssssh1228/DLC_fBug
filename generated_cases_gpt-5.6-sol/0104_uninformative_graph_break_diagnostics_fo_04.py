# -*- pattern-testcase -*-
# root_cause : Graph Break Correctness
# pattern    : Uninformative graph-break diagnostics for disable/skip functions
# title      : Skipped warning call has matching effects
# sibling    : warnings.warn trace-rules-skipped standard-library call

import warnings
import torch

# Sibling construct: warnings.warn trace-rules-skipped standard-library call.
def fn(x):
    y = x.square()
    warnings.warn("checkpoint reached", UserWarning)
    return y + 1

def run(callable_fn, x):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        result = callable_fn(x)
    effects = [(str(item.message), item.category.__name__) for item in seen]
    return result, effects

x = torch.tensor([2.0, -3.0])
eager_result, eager_effects = run(fn, x.clone())
torch._dynamo.reset()
compiled_fn = torch.compile(fn, backend="eager")
compiled_result, compiled_effects = run(compiled_fn, x.clone())
assert torch.equal(compiled_result, eager_result), (compiled_result, eager_result)
assert compiled_effects == eager_effects, (compiled_effects, eager_effects)
