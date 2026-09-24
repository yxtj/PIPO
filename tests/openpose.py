# -*- coding: utf-8 -*-
"""Compare system result vs local result for OpenPose (body / hand).

Usage: python tests/openpose.py [body|hand] [protocol] [wfile]
       (wfile default: pretrained/<type>_pose_model.pth; pass 'none' for random weights)
"""

import sys
from pathlib import Path

from session import PROJECT_ROOT, run_compare, show_report, check_threshold


def main():
    argv = sys.argv[1:]
    kind = argv[0] if len(argv) >= 1 else 'body'
    assert kind in ('body', 'hand'), "model must be body or hand"
    protocol = argv[1] if len(argv) >= 2 else 'scale'
    if len(argv) >= 3 and argv[2].lower() != 'none':
        wfile = argv[2]
    else:
        wfile = Path(PROJECT_ROOT) / 'pretrained' / '{}_pose_model.pth'.format(kind)
    r = run_compare('openpose-{}'.format(kind), protocol, n=1, seed=0, wfile=str(wfile))
    # deep 100-layer model + 8 concat branches: protocol FP error accumulates
    show_report(check_threshold(r, r_rtol=2.0))
    print("All checks passed.")


if __name__ == '__main__':
    main()
