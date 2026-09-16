# -*- pattern-testcase -*-
# root_cause : Semantic Modeling Fidelity-1
# pattern    : Iterator objects modeled as UserDefinedObjectVariable lacking iterator protocol
# title      : Reversed Iterator Identity and Advancement
# sibling    : reversed iterator __iter__ and __next__

import torch

# Sibling under test: reversed iterator __iter__ and __next__.
def fn(it, x):
    same_it = iter(it)
    assert same_it is it
    first = next(same_it)
    second = next(it)
    return x + first * 10 + second


def main():
    x = torch.tensor(1)
    eager = fn(reversed([3, 5, 7]), x)
    compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
    compiled = compiled_fn(reversed([3, 5, 7]), x)
    torch.testing.assert_close(compiled, eager)


if __name__ == "__main__":
    main()
