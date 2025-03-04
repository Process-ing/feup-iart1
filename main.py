#!/usr/bin/env python3

import sys

from src.controller import CommandResult, Controller
from src.view import Cli

def main() -> None:
    cli = Cli()
    controller = Controller(cli)

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        result = controller.load_problem(filename)
        if result != CommandResult.SUCCESS:
            cli.print_error("Failed to load problem")

    controller.run_cli()

if __name__ == '__main__':
    main()
