# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic guard/shape specialization producing flaky compilation results
# title      : Mapping membership guard stability
# sibling    : dictionary membership, optional-key, and insertion-order guards

import torch

# Sibling under test: dictionary membership, optional-key, and insertion-order guards.
def fn(x, options):
    result = x
    if "scale" in options:
        result = result * options["scale"]
    if options.get("negate", False):
        result = -result
    return result + options.get("bias", 0.0)


def make_calls():
    x = torch.arange(4, dtype=torch.float32)
    first = {"scale": 2.0, "bias": 1.0}
    reordered = {"bias": 1.0, "scale": 2.0}
    return [
        (x.clone(), first),
        (x.clone(), {"negate": True}),
        (x.clone(), reordered),
        (x.clone(), {}),
    ]


expected = [fn(x, options) for x, options in make_calls()]
baseline = None
for _ in range(3):
    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="eager", dynamic=True)
    actual = [compiled_fn(x, options) for x, options in make_calls()]
    for eager_result, compiled_result in zip(expected, actual):
        torch.testing.assert_close(compiled_result, eager_result)
    if baseline is not None:
        for previous, current in zip(baseline, actual):
            torch.testing.assert_close(current, previous)
    baseline = [result.clone() for result in actual]
