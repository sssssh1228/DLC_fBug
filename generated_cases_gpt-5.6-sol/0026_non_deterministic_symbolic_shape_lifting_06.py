# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Cond closure shape lifting
# sibling    : torch.cond with closure-captured symbolic dimensions

import torch

# Sibling construct: torch.cond with closure-captured symbolic dimensions.
def fn(x):
    rows, columns = x.shape
    pred = x[0, 0] > 0

    def true_fn(value):
        return value + rows * 2 + columns

    def false_fn(value):
        return value - rows - columns * 2

    return torch.cond(pred, true_fn, false_fn, (x,))


def main():
    torch.manual_seed(0)
    shapes = [(2, 3), (4, 5), (3, 4)]
    inputs = []
    for index, shape in enumerate(shapes):
        value = torch.randn(shape)
        value[0, 0] = 1.0 if index % 2 == 0 else -1.0
        inputs.append(value)

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
