# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Vmap pytree shape capture
# sibling    : torch.vmap over a dictionary pytree with a captured symbolic size

import torch

# Sibling construct: torch.vmap over a dictionary pytree with a captured symbolic size.
def fn(x):
    columns = x.shape[1]
    tree = {
        "value": x,
        "bias": torch.arange(x.shape[0], dtype=x.dtype, device=x.device),
    }

    def per_item(item):
        return item["value"] * columns + item["bias"]

    return torch.vmap(
        per_item,
        in_dims=({"value": 0, "bias": 0},),
    )(tree)


def main():
    torch.manual_seed(2)
    inputs = [torch.randn(shape) for shape in ((2, 3), (4, 5), (6, 2))]
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
