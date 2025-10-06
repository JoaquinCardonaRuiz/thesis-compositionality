# Examining Algebraic Recombination for Compositional Generalisation

This repository contains the codebase for the paper Examining Algebraic Recombination for Compositional Generalisation.

It provides code to train the LeAR and DeAR models on COGS and SLOG, as well as to run web visualizer VizARD.

An example command to train the DeAR model on SLOG:

```
cd LEAR/CPU_DeAR
python main.py --mode train --checkpoint my_checkpoint_directory --task slog --random-seed 100
```

And to evaluate it on the generalisation set:

```
python main.py --mode test --checkpoint checkpoint/models/my_checkpoint_directory/epoch-1.mdl --task slog
```

To run the VizARD tool:

```
cd pred_visualiser
python pred_visualiser.py
```

This repository includes code taken from https://github.com/thousfeet/LEAR