import os
import re
import json
import numpy as np
import argparse
import subprocess


def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def convert(path_dump_txt):

    raw = open(path_dump_txt, "r").read()

    # dump_to_txt prints actions like [idle] (not quoted), so make them strings.
    raw = re.sub(r"('action': )\[(\w+)\]", r"\1['\2']", raw)

    # allow `array(...)` and `uint8` used in the text
    safe_globals = {
        "__builtins__": {},  # keeps eval relatively contained
        "array": np.array,
        "uint8": np.uint8,
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
        "True": True,
        "False": False,
    }

    steps = eval(raw, safe_globals, {})  # steps is a list, length ≈ episode length

    with open(path_dump_txt + ".json", "w") as fout:
        json.dump(to_jsonable(steps), fp=fout, indent=4)


def dump_to_txt(path_dump):

    path_dump_txt = path_dump + ".txt"

    cmd = [
        "python3",
        "/gfootball/gfootball/dump_to_txt.py",
        "--trace_file",
        path_dump,
        "--output",
        path_dump_txt,
    ]

    subprocess.run(cmd)

    return path_dump_txt


def parse_arguments():

    parser = argparse.ArgumentParser(description="Convert dump file to JSON format")
    parser.add_argument(
        "path_dump", help="e.g. log/episode_done_20260101-085448548147.dump"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_arguments()

    path_dump_txt = dump_to_txt(args.path_dump)

    convert(path_dump_txt)
