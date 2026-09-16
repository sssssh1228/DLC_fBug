# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Nested vmap grad capture
# sibling    : nested torch.vmap and torch.func.grad transforms with a captured symbolic size

import torch

# Sibling construct: nested torch.vmap and torch.func.grad with symbolic closure capture.
def fn(x):
    rows = x.shape[0]

    def row_loss(row):
        return (row.sin() * rows).sum()

    row_gradient = torch.func.grad(row_loss)
    return torch.vmap(row_gradient)(x)


def main():
    torch.manual_seed(4)
    inputs = [torch.randn(shape) for shape in ((2, 3), (4, 5), (6, 4))]
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
