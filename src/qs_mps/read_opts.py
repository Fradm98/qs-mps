import json
from itertools import product


def read_json(filename) -> dict:
    with open(filename, 'r') as file:
        content = json.load(file)
    return content

def assure_list(x):
    return x if type(x) is list else [ x ]


def unpack_opts(input_opts: dict) -> dict:
    # TODO unpack also for string length
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

    if "string_length" in input_opts:
        string_lengths = assure_list(input_opts['string_length'])
        other_opts.pop('string_length')
        opts_list = [
            dict(l=l, L=L, chi=chi, string_length=sl, **other_opts)
            for l, L, chi, sl in product(nrungs, lengths, chis, string_lengths)
        ]
    else:
        opts_list = [
            dict(l=l, L=L, chi=chi, **other_opts)
            for l, L, chi in product(nrungs, lengths, chis)
        ]

    return opts_list

