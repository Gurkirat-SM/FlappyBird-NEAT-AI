"""Watch a saved NEAT genome play the original FlapPyBird game."""
import argparse
import os
import pickle

import neat
import pygame

from src.flappy import Flappy
from src.entities import Background, Floor, Pipes, Player, PlayerMode, Score


def reset_episode(game):
    game.background = Background(game.config)
    game.floor = Floor(game.config)
    game.player = Player(game.config)
    game.player.set_mode(PlayerMode.NORMAL)
    game.pipes = Pipes(game.config)
    game.score = Score(game.config)


def get_inputs(game):
    bird = game.player
    vh = float(game.config.window.viewport_height)
    vw = float(game.config.window.width)
    upcoming = None
    for upper, lower in zip(game.pipes.upper, game.pipes.lower):
        if upper.x + upper.w >= bird.x:
            upcoming = (upper, lower)
            break
    if upcoming is None:
        return [bird.y / vh, bird.vel_y / 10.0, 1.0, 0.5, 0.0]
    upper, lower = upcoming
    gap_center = ((upper.y + upper.h) + lower.y) / 2.0
    return [
        max(-1.0, min(2.0, bird.y / vh)),
        max(-2.0, min(2.0, bird.vel_y / 10.0)),
        max(-1.0, min(2.0, (upper.x - bird.x) / vw)),
        max(-1.0, min(2.0, gap_center / vh)),
        max(-2.0, min(2.0, (gap_center - bird.cy) / vh)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", default="best_genome.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        os.path.join(os.path.dirname(__file__), "neat-config.txt"),
    )
    with open(args.genome, "rb") as file:
        genome = pickle.load(file)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = Flappy()

    try:
        for episode in range(1, args.episodes + 1):
            reset_episode(game)
            alive = True
            while alive:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return

                if game.player.collided(game.pipes, game.floor):
                    alive = False
                    break

                for upper in game.pipes.upper:
                    if game.player.crossed(upper):
                        game.score.add()

                if net.activate(get_inputs(game))[0] > 0.5:
                    game.player.flap()

                game.background.tick()
                game.floor.tick()
                game.pipes.tick()
                game.score.tick()
                game.player.tick()
                pygame.display.set_caption(
                    f"Flappy Bird AI | episode {episode}/{args.episodes} | score {game.score.score}"
                )
                pygame.display.update()
                game.config.tick()
            print(f"Episode {episode}: score = {game.score.score}")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
