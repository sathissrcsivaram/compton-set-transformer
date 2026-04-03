# Compton Set Transformer

This repository contains a **Compton source localization model using a Set Transformer-style architecture**. The project combines simulation-based data generation with a heatmap-based learning framework to estimate the two-dimensional source location from sets of Compton events.

## Project Purpose

The objective of this work is to localize a radiation source from simulated detector events. For each source position, many events are generated. Each event contains physically meaningful measurements, including scattering position, absorption position, scattering angle, and energy. The model learns from the collection of events associated with a source and predicts the most likely source location.

This problem is formulated as **heatmap prediction** rather than direct coordinate regression. Instead of producing only a single `(x, y)` point estimate, the model predicts a two-dimensional probability map over possible source locations. This formulation is useful because it preserves spatial uncertainty and provides a richer representation of the prediction.

## Motivation for the Modeling Approach

The input to the model is naturally an **unordered set of events**, not a sequence. For that reason, a standard sequence-based Transformer is not the most appropriate architecture. This project instead adopts a **Set Transformer-style approach**, in which:

- each event is encoded individually using shared dense residual layers
- cross-event relationships are learned using attention
- the full set is pooled into a source-level representation
- the final output is a spatial heatmap

In practical terms, the model is best described as an **attention-based set model for Compton source localization**, structurally closer to a **Set Transformer / DeepSets hybrid** than to a standard NLP Transformer.

## Repository Contents

- `train_compton_set_transformer.py`
  - Main training and evaluation script.
- `generate_events/gen_sim_events.py`
  - Simulation data generator.
- `generate_events/params.py`
  - Simulation constants and default parameters.
- `generate_architecture_workflow.py`
  - Utility for generating workflow documentation figures.

## Data Used by the Model

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

Thus, the learning problem can be summarized as:

- input: a set of event-level measurements from one source
- output: a 2D probability heatmap over source positions

## Model Overview

The model operates as follows:

1. events belonging to a single source are grouped together
2. the set is padded or truncated to a fixed maximum number of events
3. event features are normalized
4. each event is passed through shared residual dense encoding blocks
5. an ISAB attention block models interactions across events
6. attention pooling produces a source-level representation
7. a prediction head outputs a `50 x 50` source-location heatmap

The final heatmap may then be interpreted in multiple ways, including:

- the expected coordinate `E[x, y]`
- the peak location using **MAP (Maximum a Posteriori)**

## Training Design Choices

The model uses the following core training and architectural choices:

- hidden activation: `GELU`
- output activation: `softmax`
- loss function: KL-divergence-style loss on heatmap distributions
- optimizer: `Adam`
- normalization: `LayerNormalization`
- regularization: dropout and small `L2` weight regularization

### Why `GELU` Was Used

`GELU` is used in the hidden layers because it is a smooth nonlinear activation that is commonly effective in Transformer-style architectures. Compared with `ReLU`, it provides a softer activation profile and is often preferred in attention-based models.

### Why `softmax` Was Used at the Output

The model does not output a single coordinate directly. It outputs a `50 x 50` probability heatmap over possible source locations. `softmax` is therefore appropriate because it converts the output into a normalized probability distribution whose values sum to 1.

### Why KL-Divergence-Style Loss Was Used

The target is a Gaussian heatmap distribution centered at the true source location. Because both the target and the prediction are distributions, a KL-divergence-style loss is more appropriate than a direct regression loss. It measures how closely the predicted heatmap matches the target heatmap.

### Why Mean Squared Error Was Not Used

Mean squared error is more naturally suited to direct coordinate regression, where the model predicts only `(x, y)`. In this project, the model predicts a full probability map rather than a single point. For that reason, a distribution-matching loss is better aligned with the output formulation.

### Why `Adam` Was Used

`Adam` was used because it is a stable and widely adopted optimizer for deep learning models with attention, dense layers, and noisy gradients. It generally converges more reliably than plain stochastic gradient descent in this type of setting.

### Why `LayerNormalization` Was Used

`LayerNormalization` helps stabilize training by keeping internal feature scales more consistent across layers. This is a common and effective choice in Transformer-style architectures.

## Why Heatmap Prediction Was Chosen

Heatmap prediction was chosen because source localization is naturally spatial and may involve uncertainty. Compared with direct coordinate regression, this approach offers several advantages:

- it preserves uncertainty in the prediction
- it provides a smoother training target through Gaussian heatmap labels
- it allows multiple decoding strategies after inference

## Data Generation

The simulation generator supports two output modes.

### 1. Merged Mode

Merged mode is the default mode and is the recommended option for training. It produces a single combined CSV file containing all generated events.

Example:

```bash
python -m generate_events.gen_sim_events --events-per-source 2000 --output eventset_2000.csv
```

This command:

- generates 2000 events for each source position
- writes the complete dataset into one file named `eventset_2000.csv`

### 2. Separate Mode

Separate mode generates one CSV file per source position.

Example:

```bash
python -m generate_events.gen_sim_events --mode separate --events-per-source 2000
```

Typical output files include:

- `events_0_0_-123.csv`
- `events_0_1_-123.csv`
- `events_10_21_-123.csv`

This mode is useful for source-wise inspection, debugging, or intermediate analysis. For standard training workflows, merged mode is generally preferable.

## Training Workflow

### Basic Training Run

```bash
python train_compton_set_transformer.py
```

This command uses the default dataset path, output path, and configuration values.

### Training With a Specific Dataset

```bash
COMPTON_CSV_PATH=eventset_2000.csv \
COMPTON_SAVE_DIR=Results_2000 \
COMPTON_MAX_EVENTS=2000 \
python train_compton_set_transformer.py
```

This command:

- loads `eventset_2000.csv`
- uses up to 2000 events per source
- saves outputs in `Results_2000`

## Configurable Training Parameters

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

## Outputs Produced by Training

After training, the script saves outputs such as:

- best model weights
- final model weights
- training history
- metadata describing the run
- prediction CSV files
- training curves
- evaluation plots

Typical files include:

- `best.weights.h5`
- `final.weights.h5`
- `history.json`
- `meta.json`
- `predictions_test_map.csv`
- `training_curves.png`

## Suggested Experimental Workflow

One important use of this repository is to evaluate how model performance changes as the number of events per source is reduced. A typical study may include:

1. training with `3000` events per source as a baseline
2. training with `2000` events per source
3. training with `1500` events per source
4. training with `1000` events per source
5. comparing the resulting metrics and prediction behavior

This constitutes an **ablation study**, where the number of events is changed while the remainder of the modeling pipeline is kept fixed.

Recommended dataset names:

- `eventset_3000.csv`
- `eventset_2000.csv`
- `eventset_1500.csv`

Recommended result folders:

- `Results_3000`
- `Results_2000`
- `Results_1500`

This structure helps preserve experimental clarity and prevents accidental overwriting of prior runs.

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

For large-scale experiments, the recommended workflow is:

1. keep the codebase in GitHub
2. clone or pull the repository in Google Colab
3. generate the required dataset in Colab
4. train the model in Colab
5. store datasets and results in Google Drive
6. compare the resulting runs afterward

This approach keeps the repository lightweight while supporting computationally heavier experiments outside the local system.

## Additional Project Notes

### Repo Layout

The main repository structure can also be understood in the following way:

- code
  - model training and simulation scripts
- data
  - generated CSV datasets used for experiments
- results
  - run outputs such as weights, plots, and summaries
- documentation
  - README and supporting explanation files

### Environment

The project is intended to run with:

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn

Typical installation:

```bash
pip install tensorflow pandas numpy matplotlib seaborn
```

For large experiments, Google Colab with GPU support is recommended.

### Training Evaluation and Reporting

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

### Additional Output Files

Each run now also saves two text files that make later review easier:

- `training_log.txt`
  - full raw run output from the training script, including epoch progress, metric prints, and saved artifact messages
- `run_summary.txt`
  - short human-readable summary of the run, including best epoch, best validation loss, final losses, expectation metrics, MAP metrics, and key output file paths

Additional saved outputs may include:

- `histograms_from_maps_test.png`
- `histograms_from_maps_test_zoom.png`
- `histograms_from_xy_test_map.png`
- `histograms_from_xy_test_map_zoom.png`

These files are useful for comparing runs across different event-count settings.

## Summary

In summary, this project implements a simulation-driven, attention-based set model for Compton source localization. It generates event data, trains a Set Transformer-style architecture on grouped event sets, and predicts source position as a two-dimensional probability heatmap. The repository is structured to support both baseline experiments and controlled event-count ablation studies.
