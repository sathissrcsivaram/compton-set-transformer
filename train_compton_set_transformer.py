import os
import json
import math
import random
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

# ---- TF compat: Normalization layer location
try:
    from tensorflow.keras.layers import Normalization as KerasNormalization
except Exception:
    from tensorflow.keras.layers.experimental.preprocessing import Normalization as KerasNormalization

# Headless-friendly plotting on clusters
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

# =========================
# Smooth image rendering (global default)
# =========================
# This removes the "pixel boxes" look for all imshow() calls.
mpl.rcParams["image.interpolation"] = "bilinear"  # try "bicubic" if you want even smoother
mpl.rcParams["image.resample"] = True

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import csv

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = os.environ.get("COMPTON_CSV_PATH", str(BASE_DIR / "eventset.csv"))
SAVE_DIR = os.environ.get("COMPTON_SAVE_DIR", str(BASE_DIR / "Results"))
os.makedirs(SAVE_DIR, exist_ok=True)


BEST_WEIGHTS = os.path.join(SAVE_DIR, "best.weights.h5")
FINAL_WEIGHTS = os.path.join(SAVE_DIR, "final.weights.h5")

RANDOM_SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# Batches are sources (not events)
BATCH_SIZE = 16
EPOCHS = int(os.environ.get("COMPTON_EPOCHS", "300"))
LEARNING_RATE = 3e-4
DROPOUT = 0.10
L2 = 1e-6

# Per-event encoder sizes
EVENT_HIDDEN = [256, 256, 128]
USE_GELU = True
USE_LAYERNORM = True

# Pooling config
USE_ATTENTION_POOLING = True
ATTN_HIDDEN = 128

# Attention config
NUM_SAB_HEADS = 4
# PRE_SAB_GROUP is ignored in ISAB version (kept for reference)
PRE_SAB_GROUP = 10
# ISAB (Induced Set Attention) — number of inducing points
ISAB_NUM_INDUCING = 128

# MC Dropout
MC_DROPOUT_PASSES = int(os.environ.get("COMPTON_MC_DROPOUT_PASSES", "10"))

# Data specifics
CONVERT_THETA_TO_RADIANS = True
MAX_EVENTS = int(os.environ.get("COMPTON_MAX_EVENTS", "3000"))   # pad/truncate per source to this many events
EARLY_STOPPING_PATIENCE = int(os.environ.get("COMPTON_EARLY_STOPPING_PATIENCE", "7"))
REDUCE_LR_PATIENCE = int(os.environ.get("COMPTON_REDUCE_LR_PATIENCE", "10"))

# Heatmap specifics
HEATMAP_H = 50
HEATMAP_W = 50
GAUSS_SIGMA_PX = 1.5  # <-- changed from 2.0 to 0.5 (sharper labels)

# Visualization
HIST_EDGEWIDTH = 0.7
HIST_EDGE_COLOR = "#222222"
PRED_MARKER_SIZE = 80
TRUE_MARKER_SIZE = 90
DOT_EDGEWIDTH = 1.2
N_BINS_HIST = 30


def resolve_cmap(preferred: str = "mako"):
    try:
        import seaborn as sns
        return sns.color_palette(preferred, as_cmap=True)
    except Exception:
        return preferred if preferred in plt.colormaps() else "cividis"


CMAP = resolve_cmap("mako")

# =========================
# Reproducibility
# =========================
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# =========================
# Columns
# =========================
RAW_COLS = [
    "Source_X", "Source_Y", "Source_Z",
    "Scatter_X", "Scatter_Y", "Scatter_Z",
    "Absorb_X", "Absorb_Y", "Absorb_Z",
    "Theta", "Energy",
]
FEATURE_COLS = ["Scatter_X", "Scatter_Y", "Absorb_X", "Absorb_Y", "Theta", "Energy"]
TARGET_COLS = ["Source_X", "Source_Y"]

# =========================
# Split helpers (by source)
# =========================

def source_group_split(
    df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    unique_sources = df["source_id"].unique().tolist()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(unique_sources)
    n = len(unique_sources)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    train_ids = set(unique_sources[:n_train])
    val_ids = set(unique_sources[n_train : n_train + n_val])
    test_ids = set(unique_sources[n_train + n_val :])
    return (
        df[df.source_id.isin(train_ids)].copy(),
        df[df.source_id.isin(val_ids)].copy(),
        df[df.source_id.isin(test_ids)].copy(),
    )

# =========================
# Grid + Heatmap helpers
# =========================
GRID_X_CENTERS = None  # np.ndarray [W]
GRID_Y_CENTERS = None  # np.ndarray [H]
GRID_X_MIN = None
GRID_X_MAX = None
GRID_Y_MIN = None
GRID_Y_MAX = None


def compute_grid_from_train(df_train: pd.DataFrame):
    global GRID_X_CENTERS, GRID_Y_CENTERS, GRID_X_MIN, GRID_X_MAX, GRID_Y_MIN, GRID_Y_MAX
    x_min = float(df_train["Source_X"].min())
    x_max = float(df_train["Source_X"].max())
    y_min = float(df_train["Source_Y"].min())
    y_max = float(df_train["Source_Y"].max())

    # add 5% padding for robustness
    x_pad = 0.05 * (x_max - x_min + 1e-6)
    y_pad = 0.05 * (y_max - y_min + 1e-6)
    GRID_X_MIN, GRID_X_MAX = x_min - x_pad, x_max + x_pad
    GRID_Y_MIN, GRID_Y_MAX = y_min - y_pad, y_max + y_pad

    GRID_X_CENTERS = np.linspace(GRID_X_MIN, GRID_X_MAX, HEATMAP_W).astype(np.float32)
    GRID_Y_CENTERS = np.linspace(GRID_Y_MIN, GRID_Y_MAX, HEATMAP_H).astype(np.float32)


def coord_to_heatmap_flat(x: float, y: float) -> np.ndarray:
    """Return a normalized (sum=1) flattened Gaussian heatmap for coordinate (x,y)."""
    assert GRID_X_CENTERS is not None and GRID_Y_CENTERS is not None, "Grid not initialized"

    # pixel size in world units
    dx = float(GRID_X_CENTERS[1] - GRID_X_CENTERS[0]) if HEATMAP_W > 1 else 1.0
    dy = float(GRID_Y_CENTERS[1] - GRID_Y_CENTERS[0]) if HEATMAP_H > 1 else 1.0
    sx = max(GAUSS_SIGMA_PX * dx, 1e-6)
    sy = max(GAUSS_SIGMA_PX * dy, 1e-6)

    Xg, Yg = np.meshgrid(GRID_X_CENTERS, GRID_Y_CENTERS)
    g = np.exp(-0.5 * (((Xg - x) / sx) ** 2 + ((Yg - y) / sy) ** 2)).astype(np.float32)

    s = float(g.sum()) + 1e-8
    g /= s
    return g.reshape(-1)

# =========================
# Build per-source tensors (pad/truncate) + heatmaps
# =========================

def build_source_tensors(
    df: pd.DataFrame,
    max_events: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Returns:
      X: float32 [num_sources, max_events, F]
      mask: bool  [num_sources, max_events]  True for valid events
      y_heat: float32  [num_sources, H*W] (each row sums to 1)
      source_ids: list[str] aligned to X/y rows
      source_xy: float32 [num_sources, 2] (true coordinates for metrics)
    """
    groups = list(df.groupby("source_id"))
    num_sources = len(groups)
    F = len(FEATURE_COLS)
    X = np.zeros((num_sources, max_events, F), dtype=np.float32)
    mask = np.zeros((num_sources, max_events), dtype=bool)
    y_heat = np.zeros((num_sources, HEATMAP_H * HEATMAP_W), dtype=np.float32)
    source_ids: List[str] = []
    source_xy = np.zeros((num_sources, 2), dtype=np.float32)
    for i, (sid, g) in enumerate(groups):
        feats = g[FEATURE_COLS].to_numpy(dtype=np.float32)
        n = feats.shape[0]
        n_use = min(n, max_events)
        X[i, :n_use, :] = feats[:n_use]
        mask[i, :n_use] = True
        xy = g[TARGET_COLS].iloc[0].to_numpy(dtype=np.float32)
        source_xy[i] = xy
        y_heat[i] = coord_to_heatmap_flat(float(xy[0]), float(xy[1]))
        source_ids.append(str(sid))
    return X, mask, y_heat, source_ids, source_xy

# =========================
# TF Dataset
# =========================

def make_tf_dataset(X: np.ndarray, mask: np.ndarray, y: np.ndarray, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(({"events": X, "mask": mask}, y.astype("float32")))
    if training:
        ds = ds.shuffle(buffer_size=min(len(X), 10_000), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# =========================
# (Optional) Grouped masked mean (kept for reference; not used with ISAB)
# =========================

def grouped_masked_mean(x, m, group):
    T = int(x.shape[1])
    D = int(x.shape[-1])
    assert T % group == 0, f"MAX_EVENTS={T} must be divisible by PRE_SAB_GROUP={group}"
    G = T // group
    xg = layers.Reshape((G, group, D), name="group_reshape_x")(x)
    mg = layers.Reshape((G, group), name="group_reshape_m")(m)
    mf = layers.Lambda(lambda mm: tf.cast(mm, tf.float32), name="group_mask_float")(mg)  # [B,G,g]
    sum_x = layers.Lambda(
        lambda args: tf.reduce_sum(args[0] * args[1][..., None], axis=2), name="group_sum_x"
    )([xg, mf])  # [B,G,D]
    cnt = layers.Lambda(lambda mm: tf.reduce_sum(mm, axis=2, keepdims=True), name="group_cnt")(mf)  # [B,G,1]
    mean = layers.Lambda(lambda args: tf.math.divide_no_nan(args[0], args[1]), name="group_mean")(
        [sum_x, cnt]
    )  # [B,G,D]
    m_out = layers.Lambda(lambda mm: tf.greater(tf.reduce_sum(mm, axis=2), 0.0), name="group_mask_out")(mf)  # [B,G]
    return mean, m_out

# =========================
# Induced Set Attention Block (ISAB)
# =========================


class ISAB(layers.Layer):
    def __init__(self, d_model: int, num_heads: int = 4, m: int = 128, dropout: float = 0.1, name: str | None = None):
        super().__init__(name=name)
        key_dim = max(16, d_model // max(1, num_heads))
        self.m = m
        self.d = d_model
        self.mha1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout, name="isab_mha1")
        self.mha2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout, name="isab_mha2")
        self.ffn1 = keras.Sequential(
            [
                layers.Dense(4 * d_model, activation="gelu" if USE_GELU else "relu", kernel_regularizer=keras.regularizers.l2(L2)),
                layers.Dropout(dropout),
                layers.Dense(d_model, kernel_regularizer=keras.regularizers.l2(L2)),
            ],
            name="isab_ffn1",
        )
        self.ffn2 = keras.Sequential(
            [
                layers.Dense(4 * d_model, activation="gelu" if USE_GELU else "relu", kernel_regularizer=keras.regularizers.l2(L2)),
                layers.Dropout(dropout),
                layers.Dense(d_model, kernel_regularizer=keras.regularizers.l2(L2)),
            ],
            name="isab_ffn2",
        )
        self.ln1 = layers.LayerNormalization(name="isab_ln1")
        self.ln2 = layers.LayerNormalization(name="isab_ln2")
        self.ln3 = layers.LayerNormalization(name="isab_ln3")
        self.ln4 = layers.LayerNormalization(name="isab_ln4")

    def build(self, input_shape):
        # learnable inducing points: broadcast along batch
        self.I = self.add_weight(
            name="inducing", shape=(1, self.m, self.d), initializer="glorot_uniform", trainable=True
        )

    def call(self, X, mask=None, training=None):
        # X: [B, T, D], mask: [B, T] (bool)
        B = tf.shape(X)[0]
        I = tf.tile(self.I, [B, 1, 1])  # [B, m, d]

        # I attends to X  -> H (use key mask for X)
        attn_mask_ix = None
        if mask is not None:
            mask_bool = tf.cast(mask, tf.bool)  # [B, T]
            attn_mask_ix = tf.tile(tf.expand_dims(mask_bool, 1), [1, tf.shape(I)[1], 1])  # [B, m, T]
        H = self.mha1(I, X, attention_mask=attn_mask_ix, training=training)  # (B, m, d)
        H = self.ln1(I + H)
        H = self.ln2(H + self.ffn1(H, training=training))

        # X attends to H  -> Z (all inducing points valid)
        Z = self.mha2(X, H, training=training)  # (B, T, d)
        Z = self.ln3(X + Z)
        Z = self.ln4(Z + self.ffn2(Z, training=training))
        return Z

# =========================
# Model (DeepSets + ISAB + attention pooling; heatmap head)
# =========================

def build_set_model(max_events: int, feat_dim: int) -> keras.Model:
    events_in = keras.Input(shape=(max_events, feat_dim), name="events")  # [B, T, F]
    mask_in = keras.Input(shape=(max_events,), dtype="bool", name="mask")  # [B, T]

    # Per-event normalization (shared across time)
    norm = KerasNormalization(axis=-1, name="normalize")
    x = norm(events_in)  # [B, T, F]

    # Shared per-event encoder with residuals
    for i, units in enumerate(EVENT_HIDDEN):
        h = layers.Dense(units, kernel_regularizer=keras.regularizers.l2(L2), name=f"ev_d{i}_1")(x)
        h = layers.Activation("gelu" if USE_GELU else "relu", name=f"ev_act{i}_1")(h)
        h = layers.Dropout(DROPOUT, name=f"ev_drop{i}_1")(h)
        h = layers.Dense(units, kernel_regularizer=keras.regularizers.l2(L2), name=f"ev_d{i}_2")(h)
        if USE_LAYERNORM:
            h = layers.LayerNormalization(name=f"ev_ln{i}")(h)
        # residual (project if dims differ)
        if x.shape[-1] != units:
            x = layers.Dense(units, kernel_regularizer=keras.regularizers.l2(L2), name=f"ev_proj{i}")(x)
            if USE_LAYERNORM:
                x = layers.LayerNormalization(name=f"ev_proj_ln{i}")(x)
        x = layers.Add(name=f"ev_add{i}")([x, h])
        x = layers.Activation("gelu" if USE_GELU else "relu", name=f"ev_out{i}")(x)
        x = layers.Dropout(DROPOUT, name=f"ev_out_drop{i}")(x)

    # ISAB block (keeps original event length; scalable O(T·m))
    units = int(x.shape[-1])
    x = ISAB(d_model=units, num_heads=NUM_SAB_HEADS, m=ISAB_NUM_INDUCING, dropout=DROPOUT, name="isab")(x, mask_in)
    mask_sab = mask_in  # still used in pooling masks

    # Pool across events (permutation invariant)
    if USE_ATTENTION_POOLING:
        a = layers.Dense(ATTN_HIDDEN, activation="tanh", name="attn_h")(x)  # [B,T,H]
        a = layers.Dropout(DROPOUT, name="attn_drop")(a)
        logits = layers.Dense(1, use_bias=False, name="attn_score")(a)  # [B,T,1]

        mask_f = layers.Lambda(lambda m: tf.cast(m, tf.float32), name="mask_cast")(mask_sab)  # [B,T]
        neg_inf = layers.Lambda(
            lambda t: tf.fill(tf.shape(t[..., :1]), tf.constant(-1e9, tf.float32)),
            output_shape=lambda s: (s[0], s[1], 1),
            name="neg_inf",
        )(x)  # [B,T,1]
        mask_penalty = layers.Lambda(
            lambda args: (1.0 - args[0])[..., None] * args[1], name="mask_penalty"
        )([mask_f, neg_inf])

        logits_masked = layers.Add(name="logits_masked")([logits, mask_penalty])  # [B,T,1]
        w = layers.Softmax(axis=1, name="attn_softmax")(logits_masked)  # [B,T,1]
        pooled = layers.Lambda(
            lambda args: tf.reduce_sum(args[0] * args[1], axis=1), name="attn_pool"
        )([w, x])  # [B,D]
    else:
        m = layers.Lambda(
            lambda mm: tf.cast(mm, tf.float32)[..., None], name="mask_expand"
        )(mask_sab)  # [B,T,1]
        sum_x = layers.Lambda(
            lambda args: tf.reduce_sum(args[0] * args[1], axis=1), name="sum_x"
        )([x, m])
        count = layers.Lambda(lambda mm: tf.reduce_sum(mm, axis=1), name="count")(m)  # [B,1]
        pooled = layers.Lambda(
            lambda args: tf.math.divide_no_nan(args[0], args[1]), name="mean_pool"
        )([sum_x, count])

    # Event count as an extra scalar feature to the head (normalized by MAX_EVENTS)
    count_feat = layers.Lambda(
        lambda m: tf.reduce_sum(tf.cast(m, tf.float32), axis=1, keepdims=True) / float(MAX_EVENTS),
        name="event_count_feat",
    )(mask_in)  # use original event count
    z = layers.Concatenate(name="pooled_plus_count")([pooled, count_feat])

    # Head -> Heatmap distribution over H*W (softmax)
    z = layers.Dense(
        128, activation="gelu" if USE_GELU else "relu", kernel_regularizer=keras.regularizers.l2(L2), name="head_d1"
    )(z)
    z = layers.Dropout(DROPOUT, name="head_drop1")(z)
    z = layers.Dense(
        128, activation="gelu" if USE_GELU else "relu", kernel_regularizer=keras.regularizers.l2(L2), name="head_d2"
    )(z)
    if USE_LAYERNORM:
        z = layers.LayerNormalization(name="head_ln")(z)
    outputs = layers.Dense(HEATMAP_H * HEATMAP_W, activation="softmax", name="heatmap")(z)

    model = keras.Model(inputs={"events": events_in, "mask": mask_in}, outputs=outputs, name="compton_set_model_heatmap")

    # ---- KL divergence on distributions ----
    def kl_loss(y_true, y_pred):
        eps = 1e-7
        y_true = tf.clip_by_value(y_true, eps, 1.0)
        y_true = y_true / tf.reduce_sum(y_true, axis=-1, keepdims=True)
        y_pred = tf.clip_by_value(y_pred, eps, 1.0)
        return tf.reduce_mean(
            tf.reduce_sum(y_true * (tf.math.log(y_true) - tf.math.log(y_pred)), axis=-1)
        )

    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE, clipnorm=1.0), loss=kl_loss, metrics=["categorical_crossentropy"])
    return model


def fit_normalizer(model: keras.Model, X_train_events: np.ndarray):
    # Keras 3: adapt must match the layer's build rank (3D here). Use full (N, T, F) array.
    for l in model.layers:
        if isinstance(l, KerasNormalization):
            l.adapt(X_train_events.astype("float32"))
            return
    raise RuntimeError("Normalization layer not found.")

# =========================
# MC Dropout inference (per source)
# =========================

def predict_mc_dropout(
    model: keras.Model, X: np.ndarray, mask: np.ndarray, passes: int, batch_size: int = BATCH_SIZE
) -> np.ndarray:
    if passes <= 0:
        return model.predict({"events": X, "mask": mask}, batch_size=batch_size, verbose=0)
    preds = []
    @tf.function
    def fwd(e, m):
        return model({"events": e, "mask": m}, training=True)

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        xb = X[start:end].astype("float32")
        mb = mask[start:end]
        ys = []
        for _ in range(passes):
            ys.append(fwd(xb, mb))
        ys = tf.stack(ys, axis=0)  # (passes, B, H*W)
        y_mean = tf.reduce_mean(ys, axis=0)  # (B, H*W)
        preds.append(y_mean.numpy())
    return np.concatenate(preds, axis=0)

# =========================
# Metrics & Plots (per source)
# =========================

def expected_xy_from_map(prob_flat: np.ndarray) -> np.ndarray:
    """Compute E[x], E[y] from a flattened H*W probability map."""
    p = prob_flat.reshape(HEATMAP_H, HEATMAP_W)
    p = p / (p.sum() + 1e-8)
    x = (p.sum(axis=0) * GRID_X_CENTERS).sum()
    y = (p.sum(axis=1) * GRID_Y_CENTERS).sum()
    return np.array([x, y], dtype=np.float32)

# =========================
# MAP (argmax) utilities
# =========================

def _parabolic_peak_1d(y_minus: float, y0: float, y_plus: float) -> float:
    """Return subpixel offset in [-0.5, 0.5] for a 1D parabolic peak fit.
    Uses three samples at positions -1, 0, +1. If the denominator is tiny,
    fall back to 0.0 (no subpixel shift).
    """
    denom = (y_minus - 2.0 * y0 + y_plus)
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y_minus - y_plus) / denom


def map_xy_from_map(prob_flat: np.ndarray) -> np.ndarray:
    """Return the MAP (argmax) grid-center coordinate from a flattened heatmap."""
    p = prob_flat.reshape(HEATMAP_H, HEATMAP_W)
    idx = int(np.argmax(p))
    j = idx % HEATMAP_W  # x-index
    i = idx // HEATMAP_W # y-index
    return np.array([float(GRID_X_CENTERS[j]), float(GRID_Y_CENTERS[i])], dtype=np.float32)


def map_xy_subpixel_from_map(prob_flat: np.ndarray) -> np.ndarray:
    """Subpixel MAP: refine the argmax using a 3x3 parabolic fit around the peak.
    Assumes uniform grid spacing from linspace(). Falls back to exact cell center at borders.
    """
    p = prob_flat.reshape(HEATMAP_H, HEATMAP_W)
    idx = int(np.argmax(p))
    j = idx % HEATMAP_W
    i = idx // HEATMAP_W

    # grid spacing (world units per pixel)
    dx_world = float(GRID_X_CENTERS[1] - GRID_X_CENTERS[0]) if HEATMAP_W > 1 else 1.0
    dy_world = float(GRID_Y_CENTERS[1] - GRID_Y_CENTERS[0]) if HEATMAP_H > 1 else 1.0

    # Handle borders by returning exact center
    if i == 0 or i == HEATMAP_H - 1 or j == 0 or j == HEATMAP_W - 1:
        return np.array([float(GRID_X_CENTERS[j]), float(GRID_Y_CENTERS[i])], dtype=np.float32)

    # 1D parabolic peak in x using row i at columns j-1, j, j+1
    y_minus_x = float(p[i, j - 1])
    y0_x = float(p[i, j])
    y_plus_x = float(p[i, j + 1])
    dx = _parabolic_peak_1d(y_minus_x, y0_x, y_plus_x)
    dx = float(np.clip(dx, -0.5, 0.5))

    # 1D parabolic peak in y using column j at rows i-1, i, i+1
    y_minus_y = float(p[i - 1, j])
    y0_y = float(p[i, j])
    y_plus_y = float(p[i + 1, j])
    dy = _parabolic_peak_1d(y_minus_y, y0_y, y_plus_y)
    dy = float(np.clip(dy, -0.5, 0.5))

    x = float(GRID_X_CENTERS[j]) + dx * dx_world
    y = float(GRID_Y_CENTERS[i]) + dy * dy_world
    return np.array([x, y], dtype=np.float32)


def maps_to_xy(prob_maps: np.ndarray, subpixel: bool = True) -> np.ndarray:
    """Vectorized helper: convert a batch of probability maps [N, H*W] to [N, 2]."""
    func = map_xy_subpixel_from_map if subpixel else map_xy_from_map
    return np.vstack([func(p) for p in prob_maps]).astype(np.float32)


def map_xy_majority_vote(P_samples: np.ndarray) -> np.ndarray:
    """Majority-vote MAP across MC samples for a single example.
    P_samples: [N, H*W] from N MC passes for one source.
    Returns (x, y) at the mode of the per-pass argmax indices.
    """
    idxs = np.argmax(P_samples, axis=1)  # [N]
    # Find the most frequent index
    vals, counts = np.unique(idxs, return_counts=True)
    best_idx = int(vals[np.argmax(counts)])
    j = best_idx % HEATMAP_W
    i = best_idx // HEATMAP_W
    return np.array([float(GRID_X_CENTERS[j]), float(GRID_Y_CENTERS[i])], dtype=np.float32)


def maps_to_xy_majority(P_samples_batch: np.ndarray) -> np.ndarray:
    """Apply majority-vote MAP per source.
    P_samples_batch: [N, B, H*W] (N MC passes)
    Returns [B, 2]."""
    return np.vstack([map_xy_majority_vote(P_samples_batch[:, b, :]) for b in range(P_samples_batch.shape[1])]).astype(np.float32)


def r2_scores_xy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    r2 = {}
    for i, name in enumerate(["x", "y"]):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        sst = np.sum((yt - yt.mean()) ** 2)
        sse = np.sum((yt - yp) ** 2)
        r2[f"R2_{name}"] = float("nan") if sst == 0 else float(1.0 - sse / sst)
    yt_mean = y_true.mean(axis=0, keepdims=True)
    sst_joint = np.sum((y_true - yt_mean) ** 2)
    sse_joint = np.sum((y_true - y_pred) ** 2)
    r2["R2_joint_xy"] = float("nan") if sst_joint == 0 else float(1.0 - sse_joint / sst_joint)
    return r2


def print_source_metrics_xy(pred_xy: np.ndarray, true_xy: np.ndarray, label: str):
    dxdy = pred_xy - true_xy
    mae_x = float(np.mean(np.abs(dxdy[:, 0])))
    mae_y = float(np.mean(np.abs(dxdy[:, 1])))
    mae_e = float(np.mean(np.sqrt(np.sum(dxdy ** 2, axis=1))))
    r2 = r2_scores_xy(true_xy, pred_xy)
    print(f"=== Source-level Metrics [{label}] ===")
    print(f"MAE_x: {mae_x:.4f}")
    print(f"MAE_y: {mae_y:.4f}")
    print(f"MAE_euclid: {mae_e:.4f}")
    print(f"R^2_x: {r2['R2_x']:.4f} | R^2_y: {r2['R2_y']:.4f} | R^2_joint_xy: {r2['R2_joint_xy']:.4f}")


def print_source_metrics_from_maps(prob_maps: np.ndarray, true_xy: np.ndarray, label: str):
    pred_xy = np.vstack([expected_xy_from_map(p) for p in prob_maps])
    dxdy = pred_xy - true_xy
    mae_x = float(np.mean(np.abs(dxdy[:, 0])))
    mae_y = float(np.mean(np.abs(dxdy[:, 1])))
    mae_e = float(np.mean(np.sqrt(np.sum(dxdy ** 2, axis=1))))
    r2 = r2_scores_xy(true_xy, pred_xy)
    print(f"=== Source-level Metrics [{label}] ===")
    print(f"MAE_x: {mae_x:.4f}")
    print(f"MAE_y: {mae_y:.4f}")
    print(f"MAE_euclid: {mae_e:.4f}")
    print(f"R^2_x: {r2['R2_x']:.4f} | R^2_y: {r2['R2_y']:.4f} | R^2_joint_xy: {r2['R2_joint_xy']:.4f}")


def plot_per_source_heatmap(source_ids: List[str], true_xy: np.ndarray, prob_maps: np.ndarray, tag: str):
    # (Kept for reference; not used—replaced by side-by-side below)
    outdir = os.path.join(SAVE_DIR, "heatmaps", tag)
    os.makedirs(outdir, exist_ok=True)

    extent = [GRID_X_MIN, GRID_X_MAX, GRID_Y_MIN, GRID_Y_MAX]

    for sid, p, (tx, ty) in zip(source_ids, prob_maps, true_xy):
        P = p.reshape(HEATMAP_H, HEATMAP_W)
        px, py = expected_xy_from_map(p)

        fig, ax = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
        im = ax.imshow(P, origin="lower", extent=extent, cmap=CMAP, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="probability")

        ax.scatter([tx], [ty], marker="o", s=TRUE_MARKER_SIZE, facecolors="red", edgecolors="black", linewidths=DOT_EDGEWIDTH, label="ground truth")
        ax.scatter([px], [py], marker="x", s=PRED_MARKER_SIZE, color="white", linewidths=2.0, label="predicted E[x,y]")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Test Source {sid}")
        ax.legend(loc="upper right", framealpha=0.8)

        out = os.path.join(outdir, f"source_{sid}.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
    print(f"Saved per-source heatmap figures -> {outdir}")


def _plot_error_histograms(dx: np.ndarray, dy: np.ndarray, e: np.ndarray, xlim_max: float, euclid_max: float, bins: int, bar_width: float, title_suffix: str, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].hist(np.abs(dx), bins=bins, rwidth=bar_width, align="mid", color="#4C78A8", alpha=0.9, histtype="bar", edgecolor=HIST_EDGE_COLOR, linewidth=HIST_EDGEWIDTH)
    axes[0].set_xlim(0, xlim_max)
    axes[0].set_xticks(np.arange(0, xlim_max + 1, 10))
    axes[0].set_title(f"Per-Source |X Error| ({title_suffix})")
    axes[0].set_xlabel("|Δx| per source")
    axes[0].set_ylabel("number of sources")

    axes[1].hist(np.abs(dy), bins=bins, rwidth=bar_width, align="mid", color="#F58518", alpha=0.9, histtype="bar", edgecolor=HIST_EDGE_COLOR, linewidth=HIST_EDGEWIDTH)
    axes[1].set_xlim(0, xlim_max)
    axes[1].set_xticks(np.arange(0, xlim_max + 1, 10))
    axes[1].set_title(f"Per-Source |Y Error| ({title_suffix})")
    axes[1].set_xlabel("|Δy| per source")
    axes[1].set_ylabel("number of sources")

    axes[2].hist(e, bins=bins, rwidth=bar_width, align="mid", color="#54A24B", alpha=0.9, histtype="bar", edgecolor=HIST_EDGE_COLOR, linewidth=HIST_EDGEWIDTH)
    axes[2].set_xlim(0, euclid_max)
    axes[2].set_xticks(np.arange(0, int(euclid_max) + 1, 10))
    axes[2].set_title(f"Per-Source Euclidean Error ({title_suffix})")
    axes[2].set_xlabel("Euclidean error per source")
    axes[2].set_ylabel("number of sources")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved per-source histograms -> {out_path}")


def plot_histograms_from_maps(true_xy: np.ndarray, prob_maps: np.ndarray, tag: str, xlim_max: int = 50, euclid_max: float = 70.71, bins: int = 30, bar_width: float = 0.9):
    pred_xy = np.vstack([expected_xy_from_map(p) for p in prob_maps])
    dx = pred_xy[:, 0] - true_xy[:, 0]
    dy = pred_xy[:, 1] - true_xy[:, 1]
    e = np.sqrt(dx * dx + dy * dy)

    out = os.path.join(SAVE_DIR, f"histograms_from_maps_{tag}.png")
    _plot_error_histograms(dx, dy, e, xlim_max, euclid_max, bins, bar_width, "E[x,y]", out)

    zoom_xlim_max = 5
    zoom_euclid_max = 8
    out_zoom = os.path.join(SAVE_DIR, f"histograms_from_maps_{tag}_zoom.png")
    _plot_error_histograms(dx, dy, e, zoom_xlim_max, zoom_euclid_max, bins, bar_width, "E[x,y] - Zoomed", out_zoom)


def plot_histograms_from_xy(true_xy: np.ndarray, pred_xy: np.ndarray, tag: str, xlim_max: int = 50, euclid_max: float = 70.71, bins: int = 30, bar_width: float = 0.9):
    dx = pred_xy[:, 0] - true_xy[:, 0]
    dy = pred_xy[:, 1] - true_xy[:, 1]
    e = np.sqrt(dx * dx + dy * dy)

    out = os.path.join(SAVE_DIR, f"histograms_from_xy_{tag}.png")
    _plot_error_histograms(dx, dy, e, xlim_max, euclid_max, bins, bar_width, "MAP", out)

    zoom_xlim_max = 5
    zoom_euclid_max = 8
    out_zoom = os.path.join(SAVE_DIR, f"histograms_from_xy_{tag}_zoom.png")
    _plot_error_histograms(dx, dy, e, zoom_xlim_max, zoom_euclid_max, bins, bar_width, "MAP - Zoomed", out_zoom)


def plot_training_history(history: keras.callbacks.History, tag: str = "train_history"):
    hist = history.history
    loss_k = "loss"
    vloss_k = "val_loss" if "val_loss" in hist else None
    epochs = range(1, len(hist[loss_k]) + 1)

    plt.figure(figsize=(6, 4), constrained_layout=True)
    plt.plot(epochs, hist[loss_k], linewidth=2, label="train loss")
    if vloss_k:
        plt.plot(epochs, hist[vloss_k], linewidth=2, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss (KL)")
    plt.title("Training/Validation Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    out = os.path.join(SAVE_DIR, f"{tag}.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved training curves -> {out}")

# =========================
# New: Side-by-side GT vs Pred heatmaps (for each test source)
# =========================
def plot_gt_vs_pred_side_by_side(source_ids: List[str],
                                 true_xy: np.ndarray,
                                 prob_maps: np.ndarray,
                                 tag: str,
                                 limit: int | None = None):
    """
    For each source, save a figure with two panels:
      [Left]  Ground truth Gaussian label heatmap (no overlay markers)
      [Right] Predicted heatmap (no overlay markers)
    Titles show GT (x,y) and E[x,y] respectively.
    """
    outdir = os.path.join(SAVE_DIR, "heatmaps_compare", tag)
    os.makedirs(outdir, exist_ok=True)
    extent = [GRID_X_MIN, GRID_X_MAX, GRID_Y_MIN, GRID_Y_MAX]

    n = len(source_ids) if limit is None else min(limit, len(source_ids))
    for sid, p, (tx, ty) in zip(source_ids[:n], prob_maps[:n], true_xy[:n]):
        # Predicted probability map
        P_pred = p.reshape(HEATMAP_H, HEATMAP_W)
        P_pred = P_pred / (P_pred.sum() + 1e-8)

        # Ground truth probability map (Gaussian label)
        P_true = coord_to_heatmap_flat(float(tx), float(ty)).reshape(HEATMAP_H, HEATMAP_W)

        # Predicted expected coordinates (for title only)
        ex, ey = expected_xy_from_map(p)

        fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)

        # LEFT: Ground Truth heatmap (no dot)
        im0 = axs[0].imshow(P_true, origin="lower", extent=extent, cmap=CMAP, aspect="auto")
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label="probability")
        axs[0].set_xlabel("x"); axs[0].set_ylabel("y")
        axs[0].set_title(f"Ground Truth Heatmap (Gaussian label)\n"
                         f"GT: ({tx:.2f}, {ty:.2f})")

        # RIGHT: Predicted heatmap (no cross, no GT/MAP, no legend)
        im1 = axs[1].imshow(P_pred, origin="lower", extent=extent, cmap=CMAP, aspect="auto")
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label="probability")
        axs[1].set_xlabel("x"); axs[1].set_ylabel("y")
        axs[1].set_title(f"Predicted Heatmap\nE[x,y]: ({ex:.2f}, {ey:.2f})")

        fig.suptitle(f"Source {sid}", fontsize=12)
        out = os.path.join(outdir, f"source_{sid}.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)

    print(f"Saved GT vs Pred side-by-side figures -> {outdir}")

# =========================
# Main
# =========================

def main():
    # Load
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    if CONVERT_THETA_TO_RADIANS:
        df["Theta"] = np.deg2rad(df["Theta"].astype(float))

    df["source_id"] = df["Source_X"].astype(str) + "_" + df["Source_Y"].astype(str) + "_" + df["Source_Z"].astype(str)

    # Split by source
    df_train, df_val, df_test = source_group_split(df, TRAIN_FRAC, VAL_FRAC, TEST_FRAC)

    # Compute grid from TRAIN only (no leakage)
    compute_grid_from_train(df_train)

    # Per-source tensors with heatmaps
    X_train, M_train, y_train, sid_train, xy_train = build_source_tensors(df_train, MAX_EVENTS)
    X_val, M_val, y_val, sid_val, xy_val = build_source_tensors(df_val, MAX_EVENTS)
    X_test, M_test, y_test, sid_test, xy_test = build_source_tensors(df_test, MAX_EVENTS)

    # Model
    model = build_set_model(max_events=MAX_EVENTS, feat_dim=len(FEATURE_COLS))
    fit_normalizer(model, X_train)  # adapt on all train events (3D)

    train_ds = make_tf_dataset(X_train, M_train, y_train, training=True)
    val_ds = make_tf_dataset(X_val, M_val, y_val, training=False)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=BEST_WEIGHTS,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]
    try:
        swa = keras.callbacks.StochasticWeightAveraging(start_epoch=max(5, EPOCHS // 3))
        callbacks.append(swa)
    except Exception:
        pass

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2, callbacks=callbacks)

    with open(os.path.join(SAVE_DIR, "history.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)
    plot_training_history(history, tag="training_curves")

    # Re-load the best weights before evaluation/export
    if os.path.exists(BEST_WEIGHTS):
        model.load_weights(BEST_WEIGHTS)

    # Predictions (per source, with MC Dropout) -> distribution maps
    if MC_DROPOUT_PASSES > 0:
        print(f"MC Dropout: {MC_DROPOUT_PASSES} passes ...")
        p_val  = predict_mc_dropout(model, X_val,  M_val,  passes=MC_DROPOUT_PASSES, batch_size=BATCH_SIZE)
        p_test = predict_mc_dropout(model, X_test, M_test, passes=MC_DROPOUT_PASSES, batch_size=BATCH_SIZE)
    else:
        p_val  = model.predict({"events": X_val,  "mask": M_val},  batch_size=BATCH_SIZE, verbose=0)
        p_test = model.predict({"events": X_test, "mask": M_test}, batch_size=BATCH_SIZE, verbose=0)

    # Metrics (using expected coordinates from predicted distributions)
    print_source_metrics_from_maps(p_test, xy_test, label="heatmap_model (E[x,y] from map)")

    # === New: Side-by-side GT vs Predicted heatmaps per test source ===
    plot_gt_vs_pred_side_by_side(sid_test, xy_test, p_test, tag="test", limit=None)

    # Histograms
    plot_histograms_from_maps(xy_test, p_test, tag="test",
                              xlim_max=50, euclid_max=70.71,
                              bins=N_BINS_HIST, bar_width=0.9)

    # --- MAP estimates ---
    xy_test_map = maps_to_xy(p_test, subpixel=True)
    print_source_metrics_xy(xy_test_map, xy_test, label="heatmap_model (MAP, subpixel)")
    plot_histograms_from_xy(xy_test, xy_test_map, tag="test_map",
                            xlim_max=50, euclid_max=70.71,
                            bins=N_BINS_HIST, bar_width=0.9)

    # Save final weights
    model.save_weights(FINAL_WEIGHTS)

    # Save meta
    meta = {
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "convert_theta_to_radians": CONVERT_THETA_TO_RADIANS,
        "set_model": True,
        "event_hidden": EVENT_HIDDEN,
        "attention_pooling": USE_ATTENTION_POOLING,
        "attn_hidden": ATTN_HIDDEN,
        "layer_norm": USE_LAYERNORM,
        "gelu": USE_GELU,
        "mc_dropout_passes": MC_DROPOUT_PASSES,
        "max_events": MAX_EVENTS,
        "isab": {"num_heads": NUM_SAB_HEADS, "m": ISAB_NUM_INDUCING},
        "heatmap": {
            "H": HEATMAP_H, "W": HEATMAP_W, "gauss_sigma_px": GAUSS_SIGMA_PX,
            "grid": {
                "x_min": float(GRID_X_MIN), "x_max": float(GRID_X_MAX),
                "y_min": float(GRID_Y_MIN), "y_max": float(GRID_Y_MAX),
            },
        },
    }
    with open(os.path.join(SAVE_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Artifacts saved in: {SAVE_DIR}")

    # Optional: export MAP/Expectation CSV for TEST
    try:
        csv_path = os.path.join(SAVE_DIR, "predictions_test_map.csv")
        exy = np.vstack([expected_xy_from_map(p) for p in p_test])
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source_id", "true_x", "true_y", "ex", "ey", "map_x", "map_y"])
            for sid, (tx, ty), (mx, my), (ex, ey) in zip(sid_test, xy_test, xy_test_map, exy):
                w.writerow([sid, float(tx), float(ty), float(ex), float(ey), float(mx), float(my)])
        print(f"Saved predictions CSV -> {csv_path}")
    except Exception as e:
        print(f"Could not save predictions CSV: {e}")

    print("Done")


if __name__ == "__main__":
    main()
