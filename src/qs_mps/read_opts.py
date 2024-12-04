import json
from itertools import product


def read_json(filename):
    with open(filename, 'r') as file:
        content = json.load(file)
    return content

def assure_list(x):
    return x if type(x) is list else [ x ]


def unpack_opts(input_opts: dict) -> dict:
    if 'L' not in input_opts:
        raise ValueError("Missing `L`, the horizontal dimension")
    if 'l' not in input_opts:
        raise ValueError("Missing `l`, the number of rungs")
    if 'chi' not in input_opts:
        raise ValueError("Missing `chi`, the bond dimension")

    lengths = assure_list(input_opts['L'])
    nrungs = assure_list(input_opts['l'])
    chis = assure_list(input_opts['chi'])

    other_opts = input_opts.copy()
    for k in ('L', 'l', 'chi'):
        other_opts.pop(k)

    opts_list = [
        dict(L=L, l=l, chi=chi, **other_opts)
        for L, l, chi in product(lengths, nrungs, chis)
    ]

    return opts_list

