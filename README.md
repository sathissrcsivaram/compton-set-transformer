# Compton Set Transformer

This repository contains a **Compton source localization model using a Set Transformer-style architecture**. It combines simulation-based event generation with a heatmap-based learning pipeline to estimate the two-dimensional source location from sets of Compton events.

## Project Overview

The objective of this work is to localize a radiation source from simulated detector events. For each source position, many events are generated. Each event contains physically meaningful measurements such as scattering position, absorption position, scattering angle, and energy. The model learns from the full collection of events associated with a source and predicts the most likely source location.

This problem is formulated as **heatmap prediction** rather than direct coordinate regression. Instead of predicting only one `(x, y)` point, the model predicts a two-dimensional probability map over possible source locations. This is useful because it preserves spatial uncertainty and supports multiple decoding methods after inference.

In practical terms, the model is best described as an **attention-based set model for Compton source localization**, structurally closer to a **Set Transformer / DeepSets hybrid** than to a standard sequence-based NLP Transformer.

## Why This Model Was Chosen

The input is naturally an **unordered set of events**, not a sequence. For that reason, a standard sequence Transformer is not the most appropriate architecture. This project instead uses a Set Transformer-style design in which:

- each event is encoded individually using shared dense residual layers
- cross-event relationships are learned using attention
- the full set is pooled into a source-level representation
- the final output is a spatial heatmap

This design was chosen because it matches the structure of the data and allows the model to learn both event-level features and cross-event relationships.

## Repository Layout

- `train_compton_set_transformer.py`
  - Main training, evaluation, plotting, and export script.
- `generate_events/gen_sim_events.py`
  - Simulation data generator.
- `generate_events/params.py`
  - Simulation constants and default parameters.
- `README.md`
  - Project documentation and workflow guidance.

## Environment

The project is designed to run in Python with TensorFlow/Keras and standard scientific Python packages.

Main dependencies:

- `tensorflow`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn` (optional, for colormap handling)

Typical installation:

```bash
pip install tensorflow pandas numpy matplotlib seaborn
```

The repository is suitable for local execution, but larger experiments are best run in **Google Colab with GPU support**.

## Data

For each event, the model uses the following input features:

- `Scatter_X`
- `Scatter_Y`
- `Absorb_X`
- `Absorb_Y`
- `Theta`
- `Energy`

The prediction target is the 2D source coordinate:

- `Source_X`
- `Source_Y`

So the learning problem is:

- input: a set of event-level measurements from one source
- output: a `50 x 50` probability heatmap over source positions

### Data Generation

The simulation generator supports two output modes.

**Merged mode**

Merged mode is the default and recommended mode for training. It produces a single combined CSV file.

```bash
python -m generate_events.gen_sim_events --events-per-source 2000 --output eventset_2000.csv
```

This command:

- generates `2000` events for each source position
- writes the complete dataset into `eventset_2000.csv`

**Separate mode**

Separate mode generates one CSV file per source position.

```bash
python -m generate_events.gen_sim_events --mode separate --events-per-source 2000
```

Typical output files:

- `events_0_0_-123.csv`
- `events_0_1_-123.csv`
- `events_10_21_-123.csv`

Separate mode is useful for inspection or debugging. Merged mode is generally preferable for training.

## Training

### Model Overview

The model operates as follows:

1. events belonging to a single source are grouped together
2. the set is padded or truncated to a fixed maximum number of events
3. event features are normalized
4. each event is passed through shared residual dense encoding blocks
5. an ISAB attention block models interactions across events
6. attention pooling produces a source-level representation
7. a prediction head outputs a `50 x 50` source-location heatmap

The final heatmap may then be decoded in two ways:

- expectation: `E[x, y]`, the weighted average center of the predicted heatmap
- MAP method: the single highest-probability location in the predicted heatmap

### Training Design Choices

The model uses the following core design choices:

- hidden activation: `GELU`
- output activation: `softmax`
- loss function: KL-divergence-style loss on heatmap distributions
- optimizer: `Adam`
- normalization: `LayerNormalization`
- regularization: dropout and small `L2` weight regularization

Why these choices were used:

- `GELU` is a smooth activation commonly used in Transformer-style models
- `softmax` is appropriate because the output is a probability heatmap
- KL-style loss is better aligned with distribution prediction than mean squared error
- `Adam` is a stable optimizer for attention-based deep learning models
- `LayerNormalization` helps stabilize hidden feature scales during training

### Running Training

Basic training run:

```bash
python train_compton_set_transformer.py
```

Training with a specific dataset:

```bash
COMPTON_CSV_PATH=eventset_2000.csv \
COMPTON_SAVE_DIR=Results_2000 \
COMPTON_MAX_EVENTS=2000 \
python train_compton_set_transformer.py
```

This command:

- loads `eventset_2000.csv`
- uses up to `2000` events per source
- saves outputs in `Results_2000`

### Configurable Training Parameters

The training script supports the following environment variables:

- `COMPTON_CSV_PATH`
  - dataset file to load
- `COMPTON_SAVE_DIR`
  - directory in which outputs are saved
- `COMPTON_MAX_EVENTS`
  - maximum number of events per source used by the model
- `COMPTON_EPOCHS`
  - maximum number of training epochs
- `COMPTON_MC_DROPOUT_PASSES`
  - number of Monte Carlo dropout inference passes
- `COMPTON_EARLY_STOPPING_PATIENCE`
  - early stopping patience
- `COMPTON_REDUCE_LR_PATIENCE`
  - patience for learning-rate reduction

## Evaluation

After training, the model is evaluated on the held-out test set. For each test source:

- the model predicts a heatmap
- the heatmap is decoded into a coordinate using either expectation or the MAP method
- the predicted coordinate is compared with the true coordinate

The following source-level metrics are reported:

- `MAE_x`
- `MAE_y`
- `MAE_euclid`
- `R^2_x`
- `R^2_y`
- `R^2_joint_xy`

This repository therefore evaluates not only the model’s final accuracy, but also the difference between two decoding strategies:

- expectation-based localization
- MAP-based localization

## Reporting

Each run now saves two text files that make later review easier:

- `training_log.txt`
  - full raw run output from the training script, including epoch progress, metric prints, and saved artifact messages
- `run_summary.txt`
  - short human-readable summary of the run, including best epoch, best validation loss, final losses, expectation metrics, MAP metrics, and key output file paths

This separation is useful because:

- `training_log.txt` preserves the complete training history
- `run_summary.txt` provides a clean final result for comparison across runs

## Outputs

After training, the script saves outputs such as:

- `best.weights.h5`
- `final.weights.h5`
- `history.json`
- `meta.json`
- `predictions_test_map.csv`
- `training_curves.png`
- `histograms_from_maps_test.png`
- `histograms_from_maps_test_zoom.png`
- `histograms_from_xy_test_map.png`
- `histograms_from_xy_test_map_zoom.png`
- `training_log.txt`
- `run_summary.txt`

Typical interpretation:

- `history.json`
  - epoch-by-epoch loss history
- `meta.json`
  - model and grid configuration used for the run
- `predictions_test_map.csv`
  - test-set predictions and decoded coordinates
- `training_curves.png`
  - training and validation loss curves
- histogram files
  - error distributions for expectation and MAP, in both full-scale and zoomed form

## Experimental Workflow

One important use of this repository is to study how performance changes when the number of events per source is reduced. A typical ablation study may include:

1. training with `3000` events per source as the reference configuration
2. training with `2000` events per source
3. training with `1500` events per source
4. training with `1000` events per source
5. training with `500` events per source
6. training with `100` events per source
7. comparing the resulting metrics and error distributions

Recommended dataset names:

- `eventset_3000.csv`
- `eventset_2000.csv`
- `eventset_1500.csv`
- `eventset_1000.csv`
- `eventset_500.csv`
- `eventset_100.csv`

Recommended result folders:

- `Results_3000`
- `Results_2000`
- `Results_1500`
- `Results_1000`
- `Results_500`
- `Results_100`

This structure makes comparisons easier and prevents accidental overwriting.

## Recommended GitHub Scope

The repository is intended primarily for:

- source code
- documentation
- lightweight configuration files

It is not intended to store:

- large generated datasets
- trained weights
- result folders
- large derived plots

Accordingly, the `.gitignore` file excludes generated data and outputs.

## Recommended GitHub + Colab Workflow

For larger experiments, the recommended workflow is:

1. keep the codebase in GitHub
2. clone or pull the repository in Google Colab
3. generate or upload the required dataset in Colab/Drive
4. train the model in Colab
5. store datasets and results in Google Drive
6. compare the resulting runs afterward using `run_summary.txt`, `predictions_test_map.csv`, and the histogram outputs

This approach keeps the repository lightweight while supporting GPU-based experiments outside the local system.

## Summary

This project implements a simulation-driven, attention-based set model for Compton source localization. It generates event data, trains a Set Transformer-style architecture on grouped event sets, predicts source position as a two-dimensional probability heatmap, and saves both detailed logs and clean run summaries for later analysis.
