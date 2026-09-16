# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Cond Python Bool Negation
# sibling    : operator.not_ on a non-Tensor boolean inside a torch.cond branch

import operator
import torch

# Sibling construct: operator.not_ on a non-Tensor boolean inside a torch.cond branch.
def true_branch(x):
    use_increment = operator.not_(x.requires_grad)
    if use_increment:
        return x + 1
    return x - 1


def false_branch(x):
    is_plain_tensor = operator.not_(x.requires_grad)
    if is_plain_tensor:
        return x * 2
    return x / 2


def fn(x):
    return torch.cond(x.sum() > 0, true_branch, false_branch, (x,))


for x in (torch.tensor([2.0, -0.5]), torch.tensor([-2.0, -0.5])):
    eager = fn(x)
    compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
    actual = compiled_fn(x.clone())
    torch.testing.assert_close(actual, eager)
