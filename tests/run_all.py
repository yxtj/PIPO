# -*- coding: utf-8 -*-
"""Run every comparison script: all poc models + minionn + resnet-cifar + openpose.

Usage: python tests/run_all.py [protocol]  (default: scale)
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

SCRIPTS = [
    [str(HERE / 'poc.py')],
    [str(HERE / 'minionn.py')],
    [str(HERE / 'resnet.py'), 'resnet-cifar-20'],
    [str(HERE / 'openpose.py'), 'body'],
]


def main():
    protocol = sys.argv[1] if len(sys.argv) > 1 else 'scale'
    failures = []
    for args in SCRIPTS:
        print("$ python {} {} {}".format(Path(args[0]).name, ' '.join(args[1:]), protocol))
        proc = subprocess.run([PY, *args, protocol], cwd=str(HERE))
        if proc.returncode != 0:
            failures.append(args[0])
    if failures:
        print("Failed scripts: {}".format(failures))
        sys.exit(1)
    print("All checks passed.")


if __name__ == '__main__':
    main()
