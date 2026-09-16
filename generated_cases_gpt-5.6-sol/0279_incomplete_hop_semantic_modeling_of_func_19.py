# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Cond Boolean Aggregation
# sibling    : Python all, not, and chained comparison operations inside torch.cond branches

import torch

# Sibling construct: all, not, and a chained comparison inside cond branches.
def model(x):
    options = (True, False)

    def true_branch(value):
        active = all((options[0], not options[1], 0 < value.ndim <= 3))
        return value + (1.0 if active else 0.0)

    def false_branch(value):
        active = all((options[0], not options[1], 0 < value.ndim <= 3))
        return value - (1.0 if active else 0.0)

    return torch.cond(x.sum() > 0, true_branch, false_branch, (x,))


compiled_model = torch.compile(model, backend="eager", fullgraph=True)
inputs = (
    torch.tensor([1.0, 2.0, 3.0]),
    torch.tensor([-4.0, 1.0, 1.0]),
)
for x in inputs:
    eager = model(x)
    compiled = compiled_model(x)
    torch.testing.assert_close(compiled, eager)
