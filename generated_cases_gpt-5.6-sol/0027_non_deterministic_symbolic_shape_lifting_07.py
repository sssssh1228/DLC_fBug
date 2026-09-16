# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Checkpoint symbolic kwargs
# sibling    : activation checkpoint HOP with symbolic keyword arguments

import torch
from torch.utils.checkpoint import checkpoint

# Sibling construct: activation checkpoint HOP with symbolic keyword arguments.
def fn(x):
    rows, columns = x.shape

    def body(value, *, shift, gain):
        return value * gain + shift

    return checkpoint(
        body,
        x,
        shift=rows,
        gain=columns,
        use_reentrant=False,
    )


def main():
    torch.manual_seed(1)
    inputs = [torch.randn(shape) for shape in ((2, 3), (5, 4), (3, 6))]
    expected = [fn(value.clone()) for value in inputs]
    first_compiled_run = None

    for _ in range(3):
        torch._dynamo.reset()
        compiled_fn = torch.compile(fn, backend="eager", dynamic=True)
        actual = [compiled_fn(value.clone()) for value in inputs]

        for eager_result, compiled_result in zip(expected, actual):
            torch.testing.assert_close(compiled_result, eager_result)

        if first_compiled_run is None:
            first_compiled_run = actual
        else:
            for reference, compiled_result in zip(first_compiled_run, actual):
                torch.testing.assert_close(compiled_result, reference)


if __name__ == "__main__":
    main()
