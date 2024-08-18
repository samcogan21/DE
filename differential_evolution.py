from population import Population
import numpy as np

class DifferentialEvolution:
    def __init__(self, objective_function, bounds, population_size=50, mutation_factor=0.8, crossover_prob=0.7, max_iter=1000):
        self.objective_function = objective_function
        self.bounds = bounds
        self.dim = len(bounds)
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.max_iter = max_iter
        self.population = Population(population_size, bounds)
        self.best_solution = None
        self.best_fitness = float('inf')

    def mutate(self, idx):
        candidates = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        x_a, x_b, x_c = self.population.get_individuals(a), self.population.get_individuals(b), self.population.get_individuals(c)
        mutant = x_a + self.mutation_factor * (x_b - x_c)
        bounds_array = np.array(self.bounds)
        print(f"Mutation: Original = {self.population.get_individuals(idx)}, Mutant = {mutant}")
        return np.clip(mutant, bounds_array[:, 0], bounds_array[:, 1])

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_prob
        trial = np.where(crossover_mask, mutant, target)
        print(f"Crossover: Target = {target}, Mutant = {mutant}, Trial = {trial}")
        return trial

    def select(self, idx, trial):
        target_fitness = self.objective_function(self.population.get_individuals(idx))
        trial_fitness = self.objective_function(trial)
        print(f"Select: Target Fitness = {target_fitness}, Trial Fitness = {trial_fitness}")
        if trial_fitness < target_fitness:
            self.population.update_individual(idx, trial)
        if trial_fitness < self.best_fitness:
            self.best_fitness = trial_fitness
            self.best_solution = trial
            print(f"Updated Best Fitness = {self.best_fitness}")

    def run(self):
        print("Initial Population:")
        for i in range(self.population_size):
            individual = self.population.get_individuals(i)
            print(f"Individual {i}: {individual}, Fitness = {self.objective_function(individual)}")
        for iteration in range(self.max_iter):
            print(f"Iteration: {iteration + 1}")
            for i in range(self.population_size):
                target = self.population.get_individuals(i)
                mutant = self.mutate(i)
                trial = self.crossover(target, mutant)
                self.select(i, trial)
                target_fitness = self.objective_function(target)
                trial_fitness = self.objective_function(trial)
                print(f"Individual {i}: Target Fitness = {target_fitness}, Trial Fitness = {trial_fitness}")
                print(f"Current Best Fitness = {self.best_fitness}")
        print("Final Best Solution:", self.best_solution)
        print("Final Best Fitness:", self.best_fitness)
        return self.best_solution, self.best_fitness


def objective_function(x):
    return np.sum(x**2) - 1
bounds = [(-10,-10), (10, 10)]
de = DifferentialEvolution(objective_function, bounds, population_size=50, mutation_factor=0.5, crossover_prob=0.7, max_iter=10)
best_solution, best_fitness = de.run()

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)

