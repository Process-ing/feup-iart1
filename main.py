#!/usr/bin/env python3

from src.controller import Controller
from src.view.cli import Cli

def main() -> None:
    cli = Cli()
    controller = Controller(cli=cli)

    controller.run_cli()

if __name__ == '__main__':
    main()
