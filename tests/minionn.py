# -*- coding: utf-8 -*-
"""Compare system result vs local result for MiniONN.

Usage: python tests/minionn.py [protocol] [wfile]
       (protocol default: scale; wfile default: pretrained/minionn.pt)
"""

import sys
from pathlib import Path

from session import PROJECT_ROOT, run_compare, show_report, check_threshold


def main():
    protocol = sys.argv[1] if len(sys.argv) > 1 else 'scale'
    wfile = Path(PROJECT_ROOT) / 'pretrained' / 'minionn.pt'
    wfile = sys.argv[2] if len(sys.argv) > 2 else wfile
    r = run_compare('minionn', protocol, n=1, seed=0, wfile=str(wfile))
    show_report(check_threshold(r))
    print("All checks passed.")


if __name__ == '__main__':
    main()
