# train.py
import random
import torch
from game_core import Snake, Apple, Agent # SnakeNet is implicitly part of Agent
from game_constants import COLS, ROWS

class Trainer:
    def __init__(self, population_size=50, mutation_rate=0.05,
                 mutation_strength=0.1, mutation_decay=0.97,
                 min_mutation_strength=0.02):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.population = [Agent() for _ in range(population_size)]
        self.mutation_decay = mutation_decay
        self.min_mutation_strength = min_mutation_strength

    def evaluate_population(self):
        fitness_scores = []
        for i, agent in enumerate(self.population):
            fitness = self.run_simulation(agent)
            fitness_scores.append((fitness, agent))
        return fitness_scores

    def run_simulation(self, agent):
        sim_snake = Snake()
        sim_apple = Apple()
        sim_apple.respawn(sim_snake.body)

        max_steps_per_game=ROWS*COLS*2
        moves_since_apple_limit = (ROWS * COLS) // 2 + 5 + len(sim_snake.body) * 4

        game_score = 0
        total_steps_survived = 0
        moves_since_last_apple = 0

        for _step in range(max_steps_per_game):
            inputs = agent.get_inputs(sim_snake, sim_apple, (COLS, ROWS))
            chosen_direction = agent.decide_direction(inputs)
            
            # Basic check to prevent immediate 180-degree turns, which are always fatal or suboptimal.
            # You might refine this logic if the agent should learn it.
            current_dir = sim_snake.direction
            if not ((chosen_direction == "UP" and current_dir == "DOWN") or \
                    (chosen_direction == "DOWN" and current_dir == "UP") or \
                    (chosen_direction == "LEFT" and current_dir == "RIGHT") or \
                    (chosen_direction == "RIGHT" and current_dir == "LEFT")):
                sim_snake.next_direction = chosen_direction
            # else: keep current direction if it tries to reverse (implicit)

            sim_snake.move()
            total_steps_survived += 1
            moves_since_last_apple += 1

            if (sim_snake.x, sim_snake.y) == sim_apple.coords:
                sim_snake.grow = True
                game_score += 1
                sim_apple.respawn(sim_snake.body)
                moves_since_last_apple = 0

            if sim_snake.check_collision():
                break
            
            if moves_since_last_apple > moves_since_apple_limit: # Penalize getting stuck
                break
        
        # Original fitness calculation
        fitness = total_steps_survived + (game_score * 100)
        return fitness

    def evolve_population(self):
        scored_population = self.evaluate_population()
        scored_population.sort(reverse=True, key=lambda x: x[0])

        num_elites = max(1, int(0.1 * self.population_size)) # Keep top 10%
        new_population = [agent for _, agent in scored_population[:num_elites]]
        
        potential_parents = [agent for _, agent in scored_population[:max(num_elites * 2, len(scored_population)//2)]]
        if not potential_parents and new_population: # Fallback if few distinct high scorers
             potential_parents = new_population
        elif not potential_parents: # Should not happen with >0 population
            return scored_population[0][0] if scored_population else -float('inf'), \
                   scored_population[0][1] if scored_population else Agent()


        num_offspring = self.population_size - num_elites
        for _ in range(num_offspring):
            parent = random.choice(potential_parents)
            child = Agent() # New model instance for child
            parent_weights = parent.get_weights().clone()
            mutated_weights = self.mutate_weights(parent_weights)
            child.set_weights(mutated_weights)
            new_population.append(child)

        self.population = new_population
        best_fitness_this_gen = scored_population[0][0] if scored_population else -float('inf')
        return best_fitness_this_gen, scored_population[0][1] if scored_population else None

    def mutate_weights(self, weights_tensor):
        mutated_weights = weights_tensor.clone()
        for i in range(len(mutated_weights)):
            if random.random() < self.mutation_rate:
                noise = torch.randn(1).item() * self.mutation_strength
                mutated_weights[i] += noise
        return mutated_weights

if __name__ == "__main__":
    NUM_GENERATIONS = 100  # Example: adjust as needed
    POPULATION_SIZE = 100
    MUTATION_RATE = 0.1
    MUTATION_STRENGTH = 0.2 # How much weights change during mutation

    trainer = Trainer(population_size=POPULATION_SIZE, mutation_rate=MUTATION_RATE, mutation_strength=MUTATION_STRENGTH)
    
    print(f"Starting training for {NUM_GENERATIONS} generations...")
    overall_best_agent_ever = None
    highest_fitness_ever = -float('inf')

    for gen in range(NUM_GENERATIONS):
        best_fitness_gen, best_agent_gen = trainer.evolve_population()
        print(f"Generation {gen + 1}/{NUM_GENERATIONS} - Best Fitness: {best_fitness_gen:.2f}")
        
        if best_agent_gen and best_fitness_gen > highest_fitness_ever:
            highest_fitness_ever = best_fitness_gen
            overall_best_agent_ever = best_agent_gen
            # Save model if it's the best one seen so far
            torch.save(overall_best_agent_ever.model.state_dict(), "best_snake_model.pt")
            print(f"🏆 New best model saved with fitness {highest_fitness_ever:.2f}")

    if overall_best_agent_ever:
        print("\n🎉 Training complete. Final best model saved as best_snake_model.pt")
    else:
        print("\nTraining complete. No agent was successfully trained or improved significantly.")