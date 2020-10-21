import collections
import sys
from os import path
import os


def get_shell_command():
    cmd = " ".join([path.relpath(sys.argv[0])] + sys.argv[1:])
    return cmd


def dump_shell_command(dump_dir):
    with open(path.join(dump_dir, 'command'), "w") as f:
        f.write(get_shell_command() + "\n")


def get_filepaths(dirpath, extensions):
    files = []
    for r, _, f in os.walk(dirpath):
        for file in f:
            if any([file.endswith(ext) for ext in extensions]):
                files.append(os.path.join(r, file))
    return files


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
