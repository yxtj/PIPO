# -*- coding: utf-8 -*-
"""Compare system result vs local result for ResNet-CIFAR and VGG models.

Usage: python tests/resnet.py [model_name] [protocol]
       model_name: resnet-cifar-20/32/44/... or vgg-16 etc. (default: resnet-cifar-20)
"""

import sys

from session import PROJECT_ROOT, run_compare, show_report, check_threshold


def main():
    argv = sys.argv[1:]
    model_name = argv[0] if len(argv) >= 1 else 'resnet-cifar-20'
    protocol = argv[1] if len(argv) >= 2 else 'scale'
    r = run_compare(model_name, protocol, n=1, seed=0)
    show_report(check_threshold(r))
    print("All checks passed.")


if __name__ == '__main__':
    main()
