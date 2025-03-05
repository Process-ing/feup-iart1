#!/usr/bin/env python3

import sys

from src.controller import CommandResult, Controller
from src.view import Cli
from src.view.usage import print_usage

def main() -> None:
    cli = Cli()
    controller = Controller(cli)

    if len(sys.argv) > 1:
        if sys.argv[1].startswith('-'):
            if sys.argv[1] in ('-h', '--help'):
                print_usage()
                return

            print_usage()
            sys.exit(1)

        filename = sys.argv[1]
        result = controller.load_problem(filename)
        if result != CommandResult.SUCCESS:
            cli.print_error('Failed to load problem')

    controller.run_cli()

if __name__ == '__main__':
    main()
