import sys

sys.stdin.isatty = lambda: True
sys.stdout.isatty = lambda: True

import main

main.main()
