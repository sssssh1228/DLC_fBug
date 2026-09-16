# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-2
# pattern    : Incomplete HOP semantic modeling of function context/return/argument forms
# title      : Python boolean XOR in cond
# sibling    : operator.xor on non-Tensor bool values inside torch.cond branches

import operator
import torch

# Sibling construct: operator.xor on Python bools inside a cond subgraph.
def true_branch(x):
    enabled = operator.xor(2 < 5, "key" not in {"other": 1})
    if enabled:
        return x.square()
    return x - 3.0


def false_branch(x):
    enabled = operator.xor(False, 7 in (3, 7, 9))
    return x.abs() if enabled else -x


def target(x):
    predicate = x.sum() > 0
    return torch.cond(predicate, true_branch, false_branch, (x,))


for x in (torch.tensor([2.0, -0.5]), torch.tensor([-2.0, 0.25])):
    eager = target(x)
    compiled = torch.compile(target, backend="eager")(x)
    torch.testing.assert_close(compiled, eager)
