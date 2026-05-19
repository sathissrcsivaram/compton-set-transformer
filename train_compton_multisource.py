import csv
import json
import itertools
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

import train_compton_set_transformer as t


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = os.environ.get("COMPTON_CSV_PATH", str(BASE_DIR / "eventset_multi_source.csv"))
SAVE_DIR = os.environ.get("COMPTON_SAVE_DIR", str(BASE_DIR / "Results_multisource"))
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_WEIGHTS = os.path.join(SAVE_DIR, "best.weights.h5")
FINAL_WEIGHTS = os.path.join(SAVE_DIR, "final.weights.h5")
TRAINING_LOG = os.path.join(SAVE_DIR, "training_log.txt")
RUN_SUMMARY = os.path.join(SAVE_DIR, "run_summary.txt")

# Keep imported helper module outputs aligned with this multi-source run directory.
t.SAVE_DIR = SAVE_DIR
t.BEST_WEIGHTS = BEST_WEIGHTS
t.FINAL_WEIGHTS = FINAL_WEIGHTS
t.TRAINING_LOG = TRAINING_LOG
t.RUN_SUMMARY = RUN_SUMMARY

RANDOM_SEED = t.RANDOM_SEED
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = t.TRAIN_FRAC, t.VAL_FRAC, t.TEST_FRAC

MAX_EVENTS = int(os.environ.get("COMPTON_MAX_EVENTS", "500"))
EVENTS_PER_IMAGE = int(os.environ.get("COMPTON_EVENTS_PER_IMAGE", str(MAX_EVENTS)))
HEATMAP_H = t.HEATMAP_H
HEATMAP_W = t.HEATMAP_W
FEATURE_COLS = t.FEATURE_COLS
VALID_PADDING_MODES = t.VALID_PADDING_MODES
PADDING_MODE = t.PADDING_MODE
MASK_PADDED_EVENTS = t.MASK_PADDED_EVENTS
CSV_REQUIRED_COLS = [
    "Image_ID",
    "Source_Instance_ID",
    "Source_X",
    "Source_Y",
    "Source_Z",
    "Scatter_X",
    "Scatter_Y",
    "Scatter_Z",
    "Absorb_X",
    "Absorb_Y",
    "Absorb_Z",
    "Theta",
    "Energy",
]
PEAK_MATCH_TOLERANCE_MM = float(os.environ.get("COMPTON_PEAK_MATCH_TOLERANCE_MM", "3.0"))
DEFAULT_PEAK_SUPPRESSION_RADIUS_PX = max(1, int(round(t.GAUSS_SIGMA_PX)))
PEAK_SUPPRESSION_RADIUS_PX = int(
    os.environ.get("COMPTON_PEAK_SUPPRESSION_RADIUS_PX", str(DEFAULT_PEAK_SUPPRESSION_RADIUS_PX))
)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


_ORIGINAL_STDOUT = t._ORIGINAL_STDOUT
_ORIGINAL_STDERR = t._ORIGINAL_STDERR
_LOG_HANDLE = None


def setup_log_capture():
    global _LOG_HANDLE
    _LOG_HANDLE = open(TRAINING_LOG, "w", encoding="utf-8")
    import sys

    sys.stdout = TeeStream(_ORIGINAL_STDOUT, _LOG_HANDLE)
    sys.stderr = TeeStream(_ORIGINAL_STDERR, _LOG_HANDLE)
    print(f"Training log -> {TRAINING_LOG}")


def restore_log_capture():
    global _LOG_HANDLE
    import sys

    sys.stdout = _ORIGINAL_STDOUT
    sys.stderr = _ORIGINAL_STDERR
    if _LOG_HANDLE is not None:
        _LOG_HANDLE.flush()
        _LOG_HANDLE.close()
        _LOG_HANDLE = None


def image_group_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    image_ids = sorted(df["Image_ID"].unique().tolist())
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(image_ids)
    n = len(image_ids)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    train_ids = set(image_ids[:n_train])
    val_ids = set(image_ids[n_train : n_train + n_val])
    test_ids = set(image_ids[n_train + n_val :])
    return (
        df[df["Image_ID"].isin(train_ids)].copy(),
        df[df["Image_ID"].isin(val_ids)].copy(),
        df[df["Image_ID"].isin(test_ids)].copy(),
    )


def coords_to_multisource_heatmap_flat(coords: Sequence[Tuple[float, float]]) -> np.ndarray:
    heat = np.zeros((HEATMAP_H * HEATMAP_W,), dtype=np.float32)
    for x, y in coords:
        heat += t.coord_to_heatmap_flat(float(x), float(y))
    heat /= float(heat.sum()) + 1e-8
    return heat


def build_image_tensors(
    df: pd.DataFrame,
    max_events: int,
    events_per_image: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[np.ndarray]]:
    groups = list(df.groupby("Image_ID"))
    num_images = len(groups)
    feat_dim = len(FEATURE_COLS)
    X = np.zeros((num_images, max_events, feat_dim), dtype=np.float32)
    mask = np.zeros((num_images, max_events), dtype=bool)
    y_heat = np.zeros((num_images, HEATMAP_H * HEATMAP_W), dtype=np.float32)
    image_ids: List[str] = []
    image_source_coords: List[np.ndarray] = []

    for i, (image_id, g) in enumerate(groups):
        feats = g[FEATURE_COLS].to_numpy(dtype=np.float32)
        n = feats.shape[0]
        n_real = min(n, events_per_image)
        n_use = min(n_real, max_events)
        X[i, :n_use, :] = feats[:n_use]
        mask[i, :n_use] = True

        if n_use < max_events and n_use > 0:
            pad_count = max_events - n_use
            if PADDING_MODE == "mean":
                X[i, n_use:, :] = feats[:n_use].mean(axis=0, keepdims=True)
            elif PADDING_MODE == "repeat":
                repeats = int(np.ceil(pad_count / n_use))
                X[i, n_use:, :] = np.tile(feats[:n_use], (repeats, 1))[:pad_count]
            elif PADDING_MODE == "sample":
                rng = np.random.default_rng(RANDOM_SEED + i)
                sample_idx = rng.choice(n_use, size=pad_count, replace=True)
                X[i, n_use:, :] = feats[sample_idx]
            if not MASK_PADDED_EVENTS:
                mask[i, n_use:] = True

        coords_df = (
            g[["Source_Instance_ID", "Source_X", "Source_Y"]]
            .drop_duplicates()
            .sort_values("Source_Instance_ID")
        )
        coords = coords_df[["Source_X", "Source_Y"]].to_numpy(dtype=np.float32)
        coord_pairs = [(float(x), float(y)) for x, y in coords]
        y_heat[i] = coords_to_multisource_heatmap_flat(coord_pairs)
        image_ids.append(str(image_id))
        image_source_coords.append(coords)

    return X, mask, y_heat, image_ids, image_source_coords


def plot_multisource_heatmap_pairs(
    image_ids: Sequence[str],
    true_heatmaps: np.ndarray,
    pred_heatmaps: np.ndarray,
    source_coords: Sequence[np.ndarray],
    tag: str,
    limit: int = 12,
):
    outdir = os.path.join(SAVE_DIR, "heatmaps_compare", tag)
    os.makedirs(outdir, exist_ok=True)
    extent = [t.GRID_X_MIN, t.GRID_X_MAX, t.GRID_Y_MIN, t.GRID_Y_MAX]

    n = min(limit, len(image_ids))
    for image_id, y_true, y_pred, coords in zip(
        image_ids[:n],
        true_heatmaps[:n],
        pred_heatmaps[:n],
        source_coords[:n],
    ):
        true_map = y_true.reshape(HEATMAP_H, HEATMAP_W)
        pred_map = y_pred.reshape(HEATMAP_H, HEATMAP_W)
        pred_map = pred_map / (pred_map.sum() + 1e-8)

        fig, axs = t.plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)

        im0 = axs[0].imshow(true_map, origin="lower", extent=extent, cmap=t.CMAP, aspect="auto")
        t.plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label="probability")
        axs[0].scatter(coords[:, 0], coords[:, 1], c="cyan", s=70)
        axs[0].set_title(f"Ground Truth Heatmap\nImage {image_id}")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")

        im1 = axs[1].imshow(pred_map, origin="lower", extent=extent, cmap=t.CMAP, aspect="auto")
        t.plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label="probability")
        axs[1].scatter(coords[:, 0], coords[:, 1], c="cyan", s=70)
        axs[1].set_title(f"Predicted Heatmap\nImage {image_id}")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")

        out = os.path.join(outdir, f"image_{image_id}.png")
        t.plt.savefig(out, dpi=150)
        t.plt.close(fig)

    print(f"Saved GT vs Pred side-by-side figures -> {outdir}")


def extract_topk_heatmap_peaks(
    heatmap_flat: np.ndarray,
    k: int,
    suppression_radius_px: int,
) -> np.ndarray:
    heat = np.asarray(heatmap_flat, dtype=np.float32).reshape(HEATMAP_H, HEATMAP_W).copy()
    peaks: List[Tuple[float, float]] = []

    for _ in range(k):
        flat_idx = int(np.argmax(heat))
        peak_value = float(heat.flat[flat_idx])
        if not np.isfinite(peak_value) or peak_value <= 0.0:
            break

        iy, ix = np.unravel_index(flat_idx, heat.shape)
        peaks.append((float(t.GRID_X_CENTERS[ix]), float(t.GRID_Y_CENTERS[iy])))

        y0 = max(0, iy - suppression_radius_px)
        y1 = min(HEATMAP_H, iy + suppression_radius_px + 1)
        x0 = max(0, ix - suppression_radius_px)
        x1 = min(HEATMAP_W, ix + suppression_radius_px + 1)
        heat[y0:y1, x0:x1] = -np.inf

    return np.asarray(peaks, dtype=np.float32)


def match_predicted_to_true_sources(
    true_coords: np.ndarray,
    pred_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    true_coords = np.asarray(true_coords, dtype=np.float32)
    pred_coords = np.asarray(pred_coords, dtype=np.float32)
    n_true = int(true_coords.shape[0])
    n_pred = int(pred_coords.shape[0])
    n = min(n_true, n_pred)

    if n == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    dmat = np.linalg.norm(true_coords[:, None, :] - pred_coords[None, :, :], axis=-1)
    best_cost = None
    best_perm = None
    for perm in itertools.permutations(range(n_pred), n):
        cost = float(sum(dmat[i, perm[i]] for i in range(n)))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm

    assert best_perm is not None
    matched_pred = pred_coords[list(best_perm)]
    matched_distances = np.asarray([dmat[i, best_perm[i]] for i in range(n)], dtype=np.float32)
    return matched_distances, matched_pred


def evaluate_multisource_predictions(
    image_ids: Sequence[str],
    true_source_coords: Sequence[np.ndarray],
    pred_heatmaps: np.ndarray,
    tolerance_mm: float,
    suppression_radius_px: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    mean_errors: List[float] = []
    rmse_errors: List[float] = []
    all_distances: List[float] = []
    exact_image_hits = 0
    peak_count_matches = 0

    for image_id, true_coords, pred_heat in zip(image_ids, true_source_coords, pred_heatmaps):
        true_coords = np.asarray(true_coords, dtype=np.float32)
        pred_peaks = extract_topk_heatmap_peaks(pred_heat, k=len(true_coords), suppression_radius_px=suppression_radius_px)
        distances, matched_pred = match_predicted_to_true_sources(true_coords, pred_peaks)

        if len(pred_peaks) == len(true_coords):
            peak_count_matches += 1

        if len(distances) > 0:
            mean_err = float(np.mean(distances))
            rmse_err = float(np.sqrt(np.mean(np.square(distances))))
            within_tol = int(np.sum(distances <= tolerance_mm))
            all_hit = bool(np.all(distances <= tolerance_mm))
            if all_hit:
                exact_image_hits += 1
            mean_errors.append(mean_err)
            rmse_errors.append(rmse_err)
            all_distances.extend(float(d) for d in distances.tolist())
        else:
            mean_err = float("nan")
            rmse_err = float("nan")
            within_tol = 0
            all_hit = False

        rows.append(
            {
                "image_id": image_id,
                "num_true_sources": int(len(true_coords)),
                "num_predicted_peaks": int(len(pred_peaks)),
                "mean_match_error_mm": mean_err,
                "rmse_match_error_mm": rmse_err,
                "within_tolerance_count": within_tol,
                "all_sources_within_tolerance": all_hit,
                "true_source_coords": json.dumps(true_coords.tolist()),
                "predicted_peak_coords": json.dumps(pred_peaks.tolist()),
                "matched_predicted_peak_coords": json.dumps(matched_pred.tolist()),
                "matched_distances_mm": json.dumps([float(x) for x in distances.tolist()]),
            }
        )

    num_images = max(1, len(rows))
    metrics = {
        "peak_match_tolerance_mm": float(tolerance_mm),
        "peak_suppression_radius_px": int(suppression_radius_px),
        "mean_image_match_error_mm": float(np.mean(mean_errors)) if mean_errors else float("nan"),
        "mean_image_rmse_mm": float(np.mean(rmse_errors)) if rmse_errors else float("nan"),
        "mean_source_match_error_mm": float(np.mean(all_distances)) if all_distances else float("nan"),
        "median_source_match_error_mm": float(np.median(all_distances)) if all_distances else float("nan"),
        "max_source_match_error_mm": float(np.max(all_distances)) if all_distances else float("nan"),
        "source_within_tolerance_rate": float(np.mean(np.asarray(all_distances) <= tolerance_mm)) if all_distances else 0.0,
        "exact_image_hit_rate": float(exact_image_hits / num_images),
        "peak_count_match_rate": float(peak_count_matches / num_images),
    }
    return metrics, pd.DataFrame(rows)


def save_test_heatmap_csv(
    image_ids: Sequence[str],
    pred_heatmaps: np.ndarray,
    source_coords: Sequence[np.ndarray],
) -> str:
    out_csv = os.path.join(SAVE_DIR, "predictions_test_heatmaps.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "true_source_coords", "predicted_heatmap_json"])
        for image_id, pred, coords in zip(image_ids, pred_heatmaps, source_coords):
            writer.writerow(
                [
                    image_id,
                    json.dumps(coords.tolist()),
                    json.dumps([float(x) for x in pred.tolist()]),
                ]
            )
    print(f"Saved heatmap predictions CSV -> {out_csv}")
    return out_csv


def save_multisource_eval_csv(eval_df: pd.DataFrame) -> str:
    out_csv = os.path.join(SAVE_DIR, "predictions_test_multisource_eval.csv")
    eval_df.to_csv(out_csv, index=False)
    print(f"Saved multi-source evaluation CSV -> {out_csv}")
    return out_csv


def write_run_summary(
    history_dict: Dict[str, List[float]],
    test_metrics: Dict[str, float],
    multisource_metrics: Dict[str, float],
    predictions_csv: str,
    eval_csv: str,
    num_train_images: int,
    num_val_images: int,
    num_test_images: int,
):
    best_epoch = t.best_epoch_from_history(history_dict)
    best_val_loss = min(history_dict["val_loss"]) if history_dict.get("val_loss") else None
    lines = [
        "Compton Set Transformer Multi-Source Run Summary",
        "===============================================",
        f"CSV_PATH: {CSV_PATH}",
        f"SAVE_DIR: {SAVE_DIR}",
        f"MAX_EVENTS: {MAX_EVENTS}",
        f"EVENTS_PER_IMAGE: {EVENTS_PER_IMAGE}",
        f"PADDING_MODE: {PADDING_MODE}",
        f"MASK_PADDED_EVENTS: {MASK_PADDED_EVENTS}",
        f"EPOCHS_REQUESTED: {t.EPOCHS}",
        f"EPOCHS_RUN: {len(history_dict.get('loss', []))}",
        f"BEST_EPOCH: {best_epoch if best_epoch is not None else 'N/A'}",
        f"BEST_VAL_LOSS: {best_val_loss:.6f}" if best_val_loss is not None else "BEST_VAL_LOSS: N/A",
        f"FINAL_TRAIN_LOSS: {history_dict['loss'][-1]:.6f}" if history_dict.get("loss") else "FINAL_TRAIN_LOSS: N/A",
        f"FINAL_VAL_LOSS: {history_dict['val_loss'][-1]:.6f}" if history_dict.get("val_loss") else "FINAL_VAL_LOSS: N/A",
        "",
        "Dataset",
        f"TRAIN_IMAGES: {num_train_images}",
        f"VAL_IMAGES: {num_val_images}",
        f"TEST_IMAGES: {num_test_images}",
        "",
        "Test Metrics",
        f"TEST_LOSS: {test_metrics['loss']:.6f}",
        f"TEST_CATEGORICAL_CROSSENTROPY: {test_metrics['categorical_crossentropy']:.6f}",
        "",
        "Multi-Source Peak Metrics",
        f"PEAK_MATCH_TOLERANCE_MM: {multisource_metrics['peak_match_tolerance_mm']:.6f}",
        f"PEAK_SUPPRESSION_RADIUS_PX: {int(multisource_metrics['peak_suppression_radius_px'])}",
        f"MEAN_IMAGE_MATCH_ERROR_MM: {multisource_metrics['mean_image_match_error_mm']:.6f}",
        f"MEAN_IMAGE_RMSE_MM: {multisource_metrics['mean_image_rmse_mm']:.6f}",
        f"MEAN_SOURCE_MATCH_ERROR_MM: {multisource_metrics['mean_source_match_error_mm']:.6f}",
        f"MEDIAN_SOURCE_MATCH_ERROR_MM: {multisource_metrics['median_source_match_error_mm']:.6f}",
        f"MAX_SOURCE_MATCH_ERROR_MM: {multisource_metrics['max_source_match_error_mm']:.6f}",
        f"SOURCE_WITHIN_TOLERANCE_RATE: {multisource_metrics['source_within_tolerance_rate']:.6f}",
        f"EXACT_IMAGE_HIT_RATE: {multisource_metrics['exact_image_hit_rate']:.6f}",
        f"PEAK_COUNT_MATCH_RATE: {multisource_metrics['peak_count_match_rate']:.6f}",
        "",
        "Artifacts",
        f"history.json: {os.path.join(SAVE_DIR, 'history.json')}",
        f"meta.json: {os.path.join(SAVE_DIR, 'meta.json')}",
        f"training_curves.png: {os.path.join(SAVE_DIR, 'training_curves.png')}",
        f"best.weights.h5: {BEST_WEIGHTS}",
        f"final.weights.h5: {FINAL_WEIGHTS}",
        f"predictions_test_heatmaps.csv: {predictions_csv}",
        f"predictions_test_multisource_eval.csv: {eval_csv}",
        f"training_log.txt: {TRAINING_LOG}",
    ]
    with open(RUN_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved run summary -> {RUN_SUMMARY}")


def main():
    setup_log_capture()

    df = pd.read_csv(CSV_PATH)
    missing = [c for c in CSV_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    if t.CONVERT_THETA_TO_RADIANS:
        df["Theta"] = np.deg2rad(df["Theta"].astype(float))

    df_train, df_val, df_test = image_group_split(df, TRAIN_FRAC, VAL_FRAC, TEST_FRAC)

    # Compute grid from all training source coordinates.
    t.compute_grid_from_train(df_train)

    X_train, M_train, y_train, iid_train, coords_train = build_image_tensors(df_train, MAX_EVENTS, EVENTS_PER_IMAGE)
    X_val, M_val, y_val, iid_val, coords_val = build_image_tensors(df_val, MAX_EVENTS, EVENTS_PER_IMAGE)
    X_test, M_test, y_test, iid_test, coords_test = build_image_tensors(df_test, MAX_EVENTS, EVENTS_PER_IMAGE)

    model = t.build_set_model(max_events=MAX_EVENTS, feat_dim=len(FEATURE_COLS))
    t.fit_normalizer(model, X_train)

    train_ds = t.make_tf_dataset(X_train, M_train, y_train, training=True)
    val_ds = t.make_tf_dataset(X_val, M_val, y_val, training=False)
    test_ds = t.make_tf_dataset(X_test, M_test, y_test, training=False)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=t.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=t.REDUCE_LR_PATIENCE,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_WEIGHTS,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        t.LearningRateLogger(),
    ]

    try:
        swa = tf.keras.callbacks.StochasticWeightAveraging(start_epoch=max(5, t.EPOCHS // 3))
        callbacks.append(swa)
    except Exception:
        pass

    history = model.fit(train_ds, validation_data=val_ds, epochs=t.EPOCHS, verbose=2, callbacks=callbacks)

    with open(os.path.join(SAVE_DIR, "history.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)
    t.plot_training_history(history, tag="training_curves")

    if os.path.exists(BEST_WEIGHTS):
        model.load_weights(BEST_WEIGHTS)

    if t.MC_DROPOUT_PASSES > 0:
        print(f"MC Dropout: {t.MC_DROPOUT_PASSES} passes ...")
        p_test = t.predict_mc_dropout(model, X_test, M_test, passes=t.MC_DROPOUT_PASSES, batch_size=t.BATCH_SIZE)
    else:
        p_test = model.predict({"events": X_test, "mask": M_test}, batch_size=t.BATCH_SIZE, verbose=0)

    eval_values = model.evaluate(test_ds, verbose=0, return_dict=True)
    test_metrics = {
        "loss": float(eval_values["loss"]),
        "categorical_crossentropy": float(eval_values.get("categorical_crossentropy", np.nan)),
    }
    print(f"=== Test Metrics ===")
    print(f"TEST_LOSS: {test_metrics['loss']:.6f}")
    print(f"TEST_CATEGORICAL_CROSSENTROPY: {test_metrics['categorical_crossentropy']:.6f}")

    plot_multisource_heatmap_pairs(iid_test, y_test, p_test, coords_test, tag="test", limit=len(iid_test))

    model.save_weights(FINAL_WEIGHTS)

    meta = {
        "task": "multisource_heatmap",
        "feature_cols": FEATURE_COLS,
        "convert_theta_to_radians": t.CONVERT_THETA_TO_RADIANS,
        "set_model": True,
        "event_hidden": t.EVENT_HIDDEN,
        "attention_pooling": t.USE_ATTENTION_POOLING,
        "attn_hidden": t.ATTN_HIDDEN,
        "layer_norm": t.USE_LAYERNORM,
        "gelu": t.USE_GELU,
        "mc_dropout_passes": t.MC_DROPOUT_PASSES,
        "max_events": MAX_EVENTS,
        "events_per_image": EVENTS_PER_IMAGE,
        "padding_mode": PADDING_MODE,
        "mask_padded_events": MASK_PADDED_EVENTS,
        "isab": {"num_heads": t.NUM_SAB_HEADS, "m": t.ISAB_NUM_INDUCING},
        "heatmap": {
            "H": HEATMAP_H,
            "W": HEATMAP_W,
            "gauss_sigma_px": t.GAUSS_SIGMA_PX,
            "grid": {
                "x_min": float(t.GRID_X_MIN),
                "x_max": float(t.GRID_X_MAX),
                "y_min": float(t.GRID_Y_MIN),
                "y_max": float(t.GRID_Y_MAX),
            },
        },
        "dataset": {
            "train_images": len(iid_train),
            "val_images": len(iid_val),
            "test_images": len(iid_test),
            "sources_per_image": 5,
        },
    }
    with open(os.path.join(SAVE_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    predictions_csv = save_test_heatmap_csv(iid_test, p_test, coords_test)
    multisource_metrics, eval_df = evaluate_multisource_predictions(
        iid_test,
        coords_test,
        p_test,
        tolerance_mm=PEAK_MATCH_TOLERANCE_MM,
        suppression_radius_px=PEAK_SUPPRESSION_RADIUS_PX,
    )
    eval_csv = save_multisource_eval_csv(eval_df)
    print(f"=== Multi-Source Peak Metrics ===")
    print(f"MEAN_IMAGE_MATCH_ERROR_MM: {multisource_metrics['mean_image_match_error_mm']:.6f}")
    print(f"MEAN_IMAGE_RMSE_MM: {multisource_metrics['mean_image_rmse_mm']:.6f}")
    print(f"MEAN_SOURCE_MATCH_ERROR_MM: {multisource_metrics['mean_source_match_error_mm']:.6f}")
    print(f"MEDIAN_SOURCE_MATCH_ERROR_MM: {multisource_metrics['median_source_match_error_mm']:.6f}")
    print(f"MAX_SOURCE_MATCH_ERROR_MM: {multisource_metrics['max_source_match_error_mm']:.6f}")
    print(f"SOURCE_WITHIN_TOLERANCE_RATE: {multisource_metrics['source_within_tolerance_rate']:.6f}")
    print(f"EXACT_IMAGE_HIT_RATE: {multisource_metrics['exact_image_hit_rate']:.6f}")
    print(f"PEAK_COUNT_MATCH_RATE: {multisource_metrics['peak_count_match_rate']:.6f}")
    write_run_summary(
        history.history,
        test_metrics,
        multisource_metrics,
        predictions_csv,
        eval_csv,
        num_train_images=len(iid_train),
        num_val_images=len(iid_val),
        num_test_images=len(iid_test),
    )
    print(f"Artifacts saved in: {SAVE_DIR}")
    print("Done")


if __name__ == "__main__":
    try:
        main()
    finally:
        restore_log_capture()
