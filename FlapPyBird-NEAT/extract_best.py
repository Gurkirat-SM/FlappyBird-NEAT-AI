import neat
import pickle

checkpoint_file = "neat-checkpoint-67"

# Restore the saved population
population = neat.Checkpointer.restore_checkpoint(checkpoint_file)

# Find the genome with the highest fitness in that population
best_genome = max(
    population.population.values(),
    key=lambda genome: genome.fitness
    if genome.fitness is not None else float("-inf")
)

# Save it so the existing AI player can load it
with open("best_genome.pkl", "wb") as file:
    pickle.dump(best_genome, file)

print("Best genome extracted successfully!")
print("Generation:", population.generation)
print("Fitness:", best_genome.fitness)
print("Saved as best_genome.pkl")
