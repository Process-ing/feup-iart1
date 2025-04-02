def print_solve_usage() -> None:
    print('''Usage: solve (algorithm) [parameters]...

Executes the solution for the current problem given the algorithm and optional parameters.

Options:
    random-walk <max-iterations>
        Solve using random walk.

    random-descent
        Solve using random descent.

    simulated-annealing <temperature> <cooling-schedule> <max-iterations>
        Solve using simulated annealing.

    tabu <tabu-tenure> <neighborhood-size> <max-iterations>
        Solve using tabu search.

    genetic <population-size> <max-generations>
        Solve using genetic algorithm.

    -h --help  
        Show this help and exit.
''')