# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Starred tuple display expansion
# sibling    : starred sequence unpacking in a tuple display

import torch


CONSTANT_FIELDS = (11, 13)


def fn(x):
    # Sibling construct: starred sequence unpacking in a tuple display.
    dynamic_fields = (x.sin(),)
    return (*dynamic_fields, *CONSTANT_FIELDS)


x = torch.tensor(0.5)
eager = fn(x)
compiled = torch.compile(fn, backend="eager")(x)
assert type(eager) is type(compiled)
assert len(eager) == len(compiled)
assert torch.equal(eager[0], compiled[0])
assert eager[1:] == compiled[1:]
