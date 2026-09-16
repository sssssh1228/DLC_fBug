# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : List concatenation with constant tail
# sibling    : list.__add__ sequence concatenation

import torch


def fn(x):
    # Sibling construct: list.__add__ sequence concatenation.
    values = [x, x + 1]
    return values + [3, 4]


x = torch.tensor(2)
eager = fn(x)
compiled = torch.compile(fn, backend="eager")(x)
assert type(eager) is type(compiled)
assert len(eager) == len(compiled)
assert torch.equal(eager[0], compiled[0])
assert torch.equal(eager[1], compiled[1])
assert eager[2:] == compiled[2:]
