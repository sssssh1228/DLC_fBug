# -*- pattern-testcase -*-
# root_cause : Guard Correctness
# pattern    : Non-deterministic symbolic-shape lifting into wrap HOP
# title      : Grad through inlined helper
# sibling    : torch.func.grad with an inlined helper and symbolic scale argument

import torch

# Sibling construct: torch.func.grad with an inlined helper and symbolic scale argument.
def scaled_loss(value, scale):
    return (value.cos() * scale).sum()


def invoke_grad(value, scale):
    gradient_fn = torch.func.grad(scaled_loss, argnums=0)
    return gradient_fn(value, scale)


def fn(x):
    symbolic_scale = x.shape[-1]
    return invoke_grad(x, symbolic_scale)


def main():
    torch.manual_seed(3)
    inputs = [torch.randn(shape) for shape in ((2, 3), (5, 4), (3, 7))]
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
