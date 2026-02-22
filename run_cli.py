import sys

sys.stdin.isatty = lambda: True
sys.stdin.fileno = lambda: 0
sys.stdout.isatty = lambda: True
sys.stdout.fileno = lambda: 1

import main

main.main()
