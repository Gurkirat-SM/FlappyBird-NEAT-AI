"""Train a NEAT agent on the original FlapPyBird game code."""
import argparse
import os
import pickle

import neat
import pygame

from src.flappy import Flappy
from src.entities import Background, Floor, Pipes, Player, PlayerMode, Score


def reset_episode(game):
    """Recreate the game's entities, preserving its original physics/assets."""
    game.background = Background(game.config)
    game.floor = Floor(game.config)
    game.player = Player(game.config)
    game.player.set_mode(PlayerMode.NORMAL)
    game.pipes = Pipes(game.config)
    game.score = Score(game.config)


def get_inputs(game):
    """Return five normalized features for the nearest upcoming pipe."""
    bird = game.player
    viewport_h = float(game.config.window.viewport_height)
    viewport_w = float(game.config.window.width)

    upcoming = None
    for upper, lower in zip(game.pipes.upper, game.pipes.lower):
        if upper.x + upper.w >= bird.x:
            upcoming = (upper, lower)
            break

    # A pipe should always be available, but keep the network input safe.
    if upcoming is None:
        return [
            bird.y / viewport_h,
            bird.vel_y / 10.0,
            1.0,
            0.5,
            0.0,
        ]

    upper, lower = upcoming
    gap_top = upper.y + upper.h
    gap_bottom = lower.y
    gap_center = (gap_top + gap_bottom) / 2.0
    distance = (upper.x - bird.x) / viewport_w

    return [
        max(-1.0, min(2.0, bird.y / viewport_h)),
        max(-2.0, min(2.0, bird.vel_y / 10.0)),
        max(-1.0, min(2.0, distance)),
        max(-1.0, min(2.0, gap_center / viewport_h)),
        max(-2.0, min(2.0, (gap_center - bird.cy) / viewport_h)),
    ]


def evaluate_one(game, genome, config, render=True, max_steps=1800):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    reset_episode(game)
    fitness = 0.0
    scored_pipes = set()

    for _ in range(max_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise KeyboardInterrupt("Training stopped by user.")
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                raise KeyboardInterrupt("Training stopped by user.")

        bird = game.player
        if bird.collided(game.pipes, game.floor):
            break

        # Reward surviving; passing a pipe gives a larger reward.
        fitness += 0.1
        for upper in game.pipes.upper:
            if upper not in scored_pipes and bird.crossed(upper):
                scored_pipes.add(upper)
                fitness += 5.0
                game.score.score += 1

        output = net.activate(get_inputs(game))[0]
        if output > 0.5:
            bird.flap()
            fitness -= 0.01  # tiny cost discourages pointless constant flapping

        game.background.tick()
        game.floor.tick()
        game.pipes.tick()
        game.score.tick()
        bird.tick()

        if render:
            # Keep the original game's score and graphics on screen.
            pygame.display.update()
            game.config.clock.tick(game.config.fps)
        else:
            # Avoid frame limiting while training quickly.
            pygame.display.update() if False else None

    genome.fitness = fitness
    return fitness


def main():
    parser = argparse.ArgumentParser(
        description="Train your own NEAT agent on FlapPyBird."
    )
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--fast", action="store_true",
                        help="run without drawing each training frame")
    parser.add_argument("--population", type=int, default=None,
                        help="override population size in neat-config.txt")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "neat-config.txt")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    if args.population:
        config.pop_size = args.population

    game = Flappy()
    render = not args.fast
    if args.fast:
        pygame.display.set_caption("Flappy Bird AI - fast training (rendering paused)")

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5, filename_prefix="neat-checkpoint-"))

    def evaluate_generation(genomes, neat_config):
        for genome_id, genome in genomes:
            fitness = evaluate_one(
                game, genome, neat_config, render=render
            )
            if render:
                pygame.display.set_caption(
                    f"Flappy Bird AI | genome {genome_id} | fitness {fitness:.1f}"
                )

    try:
        winner = population.run(evaluate_generation, args.generations)
        with open("best_genome.pkl", "wb") as file:
            pickle.dump(winner, file)
        with open("neat_stats.pkl", "wb") as file:
            pickle.dump(stats, file)
        print("\nTraining complete.")
        print("Saved best genome to best_genome.pkl")
        print("Saved generation statistics to neat_stats.pkl")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
