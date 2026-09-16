# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Constant tuple operand breaks binary-add sequence unpack
# title      : Variadic call with constant tuple
# sibling    : starred positional argument unpacking

import torch


def collect(first, second, third):
    return first * 2, second, third


def fn(x, constants=(17, 19)):
    # Sibling construct: starred positional argument unpacking.
    dynamic_args = (x,)
    return collect(*dynamic_args, *constants)


x = torch.tensor(3)
eager = fn(x)
compiled = torch.compile(fn, backend="eager")(x)
assert type(eager) is type(compiled)
assert len(eager) == len(compiled)
assert torch.equal(eager[0], compiled[0])
assert eager[1:] == compiled[1:]
