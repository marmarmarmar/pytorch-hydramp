# pytorch-hydramp
Porting HydrAMP model to `pytorch`

## Requirements:
- compatible with `torch=='2.0.0+cu117`
- properly set up the environment of the HydrAMP model (for porting only),

## Models
Main HydrAMP encoder, decoder and porting of a GRU layer are implemented in `torch_hydramp_models.py`,

Weights to the main encoder and decoder are in the `weights` directory,

## Porting
Follow instructions from the `Porting to pytorch.ipynb` notebook.

## TODO:
- make encoder and decoder architectures fully customizable (they are now addapted to only main HydrAMP model),
- port `HydrAMP` classifiers to `pytorch`.
