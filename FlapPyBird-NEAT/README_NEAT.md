# Train your own NEAT AI on FlapPyBird

This project keeps the original FlapPyBird game, art, pipes, collision detection, and bird physics. The NEAT training and inference code in `train_ai.py` and `play_ai.py` is added separately so you can learn and modify it yourself.

## Windows setup

Use Python 3.10 or 3.11 if possible for the smoothest dependency setup.

1. Extract the ZIP.
2. Open a terminal in the extracted `FlapPyBird-NEAT` folder.
3. Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-ai.txt
```

If `py -3.11` is unavailable, use `python -m venv .venv` with an installed Python version.

## Run the original game manually

```powershell
python main.py
```

Press Space or Up to flap; press Escape to quit.

## Train your own model

```powershell
python train_ai.py --generations 30
```

The game window shows each genome being evaluated. Training runs genomes one after another, so it can take time. For faster training without drawing every frame:

```powershell
python train_ai.py --generations 30 --fast
```

To try a smaller population for a quick test:

```powershell
python train_ai.py --generations 5 --population 10
```

The best genome is saved to `best_genome.pkl`. NEAT checkpoints are saved every five generations.

## Watch your trained AI

After training completes:

```powershell
python play_ai.py --genome best_genome.pkl --episodes 5
```

This loads the saved genome and runs it in the original game. You must train first to create `best_genome.pkl`.

## How the agent learns

- **Inputs (5):** normalized bird height, vertical velocity, distance to the next pipe, next gap's center, and the vertical difference between bird and gap center.
- **Output (1):** if the output is greater than `0.5`, the bird flaps; otherwise it does nothing.
- **Fitness:** a small reward for each surviving frame, a larger reward for passing a pipe, and a tiny flap cost. Collision ends the current genome's episode.
- **Evolution:** NEAT selects and evolves genomes across generations, changing network weights and topology.

This is not a pretrained model. Running `train_ai.py` performs training on your machine from a newly initialized NEAT population.

## Notes

- Keep the assets folder and original `src` folder in place; the game uses them.
- `neat-config.txt` controls population size, mutation rates, species and network evolution.
- Training results vary because pipe layouts are random. A poor score in early generations is normal.
