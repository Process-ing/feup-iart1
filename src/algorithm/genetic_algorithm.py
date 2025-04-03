import random
from typing import Iterator, List, override
from src.model import Building
from src.model import RouterProblem
from src.algorithm.algorithm import Algorithm

class GeneticAlgorithm(Algorithm):
    """
    Genetic Algorithm
    """
    def __init__(self, problem: RouterProblem, population_size: int = 10,
                 initial_routers: int | None = None, max_generations: int = 1000) -> None:
        self.__problem = problem
        self.__population_size = population_size
        self.__max_generations = max_generations

        if initial_routers is None:
            self.__initial_routers = self.__problem.budget \
                // (self.__problem.router_price + self.__problem.backbone_price)

    def placement_descent(self) -> Iterator[None]:
        found_neighbor = False
        current_score = self.__problem.building.score

        for _ in range(self.__initial_routers):
            operator = next(self.__problem.building.get_placement_neighborhood(), None)
            if not operator:
                yield "No neighbor found"
                continue

            neighbor = operator.apply(self.__problem.building)
            if not neighbor:
                yield "No neighbor found"
                continue

            neighbor_score = neighbor.score
            if neighbor_score > current_score:
                self.__problem.building = neighbor
                current_score = neighbor_score
                found_neighbor = True

            yield f"{'Placed' if operator.place else 'Removed'} router at ({operator.row}, {operator.col})"
            if not found_neighbor:
                break

    def sort_population(self, population: list[Building]) -> None:
        population.sort(key=lambda individual: individual.score)

    def get_best_individual(self, population: list[Building]) -> Building:
        return max(population, key=lambda individual: individual.score)

    def crossover(self, population: List[Building], offspring: List[Building]) -> Iterator[None]:
        min_score = min(individual.score for individual in population)
        fitness_scores = [individual.score - min_score + 1 for individual in population]

        while len(offspring) < self.__population_size:
            parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)

            children = parent1.crossover(parent2)
            if children is None:
                yield
                continue

            offspring.extend(children)
            yield

    def mutate(self, offspring: List[Building]) -> Iterator[None]:
        for i, child in enumerate(offspring):
            if random.random() < 0.5:
                for operator in child.get_neighborhood():
                    neighbor = operator.apply(child)
                    if neighbor is not None:
                        child = neighbor
                        yield
                        break
                    yield

            offspring[i] = child

    @override
    def run(self) -> Iterator[None]:
        original_building = self.__problem.building
        population = []
        best_score = -1
        best_neighbor = None

        for _ in range(self.__population_size):
            self.__problem.building = original_building
            yield from self.placement_descent()

            population.append(self.__problem.building)
            score = self.__problem.building.score
            if score > best_score:
                best_score = score
                best_neighbor = self.__problem.building

        assert best_neighbor is not None  # Can only happen if population size is 0
        self.__problem.building = best_neighbor

        for _ in range(self.__max_generations):
            offspring: List[Building] = []
            yield from self.crossover(population, offspring)
            yield from self.mutate(population)

            # Perform crossover to create offspring
            offspring: list[Building] = []
            while len(offspring) < self.__population_size:
                parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)

                children = parent1.crossover(parent2)
                if children is None:
                    yield "Crossover failed"
                    continue

                offspring.extend(children)
                yield f"Crossover successful, offspring size: {len(offspring)}"

            for i, child in enumerate(offspring):
                if random.random() < 0.5:
                    for operator in child.get_neighborhood():
                        neighbor = operator.apply(child)
                        if neighbor is not None:
                            child = neighbor
                            yield "Mutation successful"
                            break
                        yield "Mutation failed"

                offspring[i] = child

            self.sort_population(population)
            self.sort_population(offspring)

            # Check similarity of individuals
            filtered_offspring: List[Building] = []
            for i, child in enumerate(offspring):
                not_similar = True
                for j, individual in enumerate(population):
                    if child.is_similar(individual, max_similarity=0.001):
                        not_similar = False
                        break

                for j, other_child in enumerate(filtered_offspring):
                    if child.is_similar(other_child, max_similarity=0.001):
                        not_similar = False
                        break

                if not_similar:
                    filtered_offspring.append(child)

            # Replace the worst individuals in the population
            i = 0
            for j in range(len(filtered_offspring) - 1, -1, -1):
                if filtered_offspring[j].score < population[i].score:
                    break

                population[i] = filtered_offspring[j]
                i += 1
                yield "Individual replaced"

            best_individual = self.get_best_individual([population[0], population[-1]])
            # Best individual is either the first or last in the population
            best_gen_score = best_individual.score

            if best_gen_score > best_score:
                best_score = population[0].score
                self.__problem.building = population[0]

            yield "Best individual found"
