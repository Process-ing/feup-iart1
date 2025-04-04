def print_solve_usage() -> None:
    print('''Usage: solve (algorithm) [parameters]...

Executes the solution for the current problem given the algorithm and optional parameters.

Options:
    random-walk [options]
        Solve using random walk.

        Options:
            --max-iterations <int>    Maximum number of iterations to run.

    random-descent [options]
        Solve using random descent.

        Options:
            --max-neighborhood <int>    Maximum number of better neighbors to explore.
            --max-iterations <int>      Maximum number of iterations to run.

    simulated-annealing [options]
        Solve using simulated annealing.

        Options:
            --init-temperature <float>  Initial temperature.
            --cooling-schedule <float>  Cooling rate/schedule coefficient.
            --max-iterations <int>      Maximum number of iterations to run.

    tabu [options]
        Solve using tabu search.

        Options:
            --tabu-tenure <int>         Tabu tenure.
            --max-iterations <int>      Maximum number of iterations to run.
            --max-neighborhood <int>    Maximum number of neighbors to explore.

    genetic [options]
        Solve using genetic algorithm.

        Options:
            --init-routers <int>        Maximum number of routers in each initial individual.
            --population-size <int>     Population size.
            --max-generations <int>     Maximum number of generations to run.
            --mutation-prob <float>     Mutation probability.
            --max-similarity <float>    Maximum similarity between individuals in each generation.
            --max-neighborhood <int>    Maximum number of better neighbors to explore.
            --mimetic <bool>            Whether to use mimetic crossover.

    -h --help
        Show this help and exit.
''')
