# -*- coding: utf-8 -*-
"""Compare system result vs local result for all small poc models.

Usage: python tests/poc.py [protocol] [model_name]
       (no model_name = run every poc-* model; protocol default: scale)
"""

import sys

from session import PROJECT_ROOT, run_compare, show_report, check_threshold


def main():
    from models import poc as poc_map, registry

    protocol = 'scale'
    only = None
    argv = [a for a in sys.argv[1:] if not str(a).startswith('-')]
    if len(argv) >= 1 and not argv[0].startswith('poc-'):
        protocol = argv[0]
        argv = argv[1:]
    if len(argv) >= 1:
        only = argv[0]

    # scale-family experiments skip models whose client-side activations do
    # not commute with the multiplicative mask; an explicit model name still
    # runs (to reproduce the known failure).
    skip = registry.unsupported_models(protocol) if only is None else frozenset()
    names = sorted(k for k in poc_map.map.keys() if only is None or k == only)
    failures = []
    skipped = []
    for key in names:
        name = 'poc-{}'.format(key)
        if name in skip:
            skipped.append(name)
            print("SKIP {}: unsupported under {} (non-homogeneous activation)".format(name, protocol))
            continue
        try:
            show_report(check_threshold(run_compare(name, protocol)))
        except AssertionError as e:
            print("FAIL {}".format(name))
            print(e)
            failures.append(name)
    if skipped:
        print("Skipped {} unsupported {} model(s): {}".format(
            len(skipped), protocol, skipped))
    if failures:
        print("{} failing models: {}".format(len(failures), failures))
        sys.exit(1)
    print("All checks passed ({} models).".format(len(names) - len(skipped)))


if __name__ == '__main__':
    main()
