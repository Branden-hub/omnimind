from typing import List, Dict, Any, Optional
from .mutation import MutationEngine
from .crossover import CrossoverEngine
from .fitness import FitnessEvaluator
from .population import PopulationManager

class EvolutionEngine:
    """Main engine for evolutionary algorithms and architecture evolution."""
    
    def __init__(self):
        self.mutation_engine = MutationEngine()
        self.crossover_engine = CrossoverEngine()
        self.fitness_evaluator = FitnessEvaluator()
        self.population_manager = PopulationManager()
        self.generation = 0
        self.best_fitness = float('-inf')
        self.best_individual = None
        
    def evolve(self, initial_architecture: Dict[str, Any],
               generations: int = 100,
               population_size: int = 50,
               mutation_rate: float = 0.1,
               crossover_rate: float = 0.7) -> Dict[str, Any]:
        """Evolve the architecture over multiple generations."""
        
        # Initialize population
        population = self.population_manager.initialize_population(
            initial_architecture, population_size)
        
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate fitness for all individuals
            fitness_scores = [
                self.fitness_evaluator.evaluate(individual)
                for individual in population
            ]
            
            # Update best individual
            max_fitness_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[max_fitness_idx]
                self.best_individual = population[max_fitness_idx].copy()
            
            # Selection
            selected = self.population_manager.select_parents(
                population, fitness_scores)
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individual
            new_population.append(self.best_individual)
            
            # Generate rest of new population
            while len(new_population) < population_size:
                # Select parents
                parent1, parent2 = self.population_manager.select_pair(selected)
                
                # Crossover
                if self._should_crossover(crossover_rate):
                    offspring1, offspring2 = self.crossover_engine.crossover(
                        parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if self._should_mutate(mutation_rate):
                    offspring1 = self.mutation_engine.mutate(offspring1)
                if self._should_mutate(mutation_rate):
                    offspring2 = self.mutation_engine.mutate(offspring2)
                
                new_population.extend([offspring1, offspring2])
            
            # Trim population to exact size
            population = new_population[:population_size]
            
            # Optional: Early stopping if fitness hasn't improved
            if self._should_stop_early():
                break
        
        return self.best_individual
    
    def _should_mutate(self, mutation_rate: float) -> bool:
        """Determine if mutation should occur."""
        return np.random.random() < mutation_rate
    
    def _should_crossover(self, crossover_rate: float) -> bool:
        """Determine if crossover should occur."""
        return np.random.random() < crossover_rate
    
    def _should_stop_early(self, patience: int = 10) -> bool:
        """Check if evolution should stop early."""
        # Implement early stopping logic based on fitness improvement
        return False
    
    def _evolution_step(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform one step of evolution."""
        # Evaluate fitness
        fitness_scores = [
            self.fitness_evaluator.evaluate(individual)
            for individual in population
        ]
        
        # Select parents
        parents = self.population_manager.select_parents(population, fitness_scores)
        
        # Create new population through crossover and mutation
        new_population = self.population_manager.create_new_generation(parents)
        
        return new_population
