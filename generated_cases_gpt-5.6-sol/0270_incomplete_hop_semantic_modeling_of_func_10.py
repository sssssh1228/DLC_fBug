# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Checkpoint Dynamic Nonzero
# sibling    : dynamic-output-shape torch.nonzero inside a checkpoint subgraph

import torch
from torch.utils.checkpoint import checkpoint

# Sibling construct: dynamic-output-shape torch.nonzero inside a checkpoint subgraph.
def select_positive_sum(x):
    indices = torch.nonzero(x > 0, as_tuple=False).flatten()
    return x.index_select(0, indices).sum()


def fn(x):
    return checkpoint(select_positive_sum, x, use_reentrant=False)


torch._dynamo.config.capture_dynamic_output_shape_ops = True
compiled_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)

for x in (
    torch.tensor([-2.0, 1.0, 3.0, -4.0], requires_grad=True),
    torch.tensor([5.0, -1.0, -2.0, 4.0, 6.0], requires_grad=True),
):
    eager_input = x.detach().clone().requires_grad_(True)
    compiled_input = x.detach().clone().requires_grad_(True)

    eager = fn(eager_input)
    actual = compiled_fn(compiled_input)
    eager.backward()
    actual.backward()

    torch.testing.assert_close(actual, eager)
    torch.testing.assert_close(compiled_input.grad, eager_input.grad)
