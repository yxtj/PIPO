# -*- coding: utf-8 -*-
"""Compare system result vs local result for all small poc models.

Usage: python tests/poc.py [model_name] [protocol]
       (no model_name = run every poc-* model; protocol default: scale)
"""

import sys

from session import PROJECT_ROOT, run_compare, show_report, check_threshold


def main():
    from models import poc as poc_map

    protocol = 'scale'
    only = None
    argv = [a for a in sys.argv[1:] if not str(a).startswith('-')]
    if len(argv) >= 1 and not argv[0].startswith('poc-'):
        protocol = argv[0]
        argv = argv[1:]
    if len(argv) >= 1:
        only = argv[0]

    names = sorted(k for k in poc_map.map.keys() if only is None or k == only)
    failures = []
    for key in names:
        name = 'poc-{}'.format(key)
        try:
            show_report(check_threshold(run_compare(name, protocol)))
        except AssertionError as e:
            print("FAIL {}".format(name))
            print(e)
            failures.append(name)
    if failures:
        print("{} failing models: {}".format(len(failures), failures))
        sys.exit(1)
    print("All checks passed ({} models).".format(len(names)))


if __name__ == '__main__':
    main()
