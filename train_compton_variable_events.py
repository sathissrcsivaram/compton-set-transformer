import csv
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

import train_compton_set_transformer as t


TRAIN_EVENT_CHOICES_RAW = os.environ.get(
    "COMPTON_TRAIN_EVENT_CHOICES",
    "10,20,30,40,50,60,70,80,90,100",
)
VAL_EVENTS_PER_SOURCE = int(os.environ.get("COMPTON_VAL_EVENTS_PER_SOURCE", str(t.MAX_EVENTS)))
TEST_EVENTS_PER_SOURCE = int(os.environ.get("COMPTON_TEST_EVENTS_PER_SOURCE", str(t.MAX_EVENTS)))


def parse_train_event_choices(raw: str) -> List[int]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    if not vals:
        raise ValueError("COMPTON_TRAIN_EVENT_CHOICES must contain at least one integer")
    for v in vals:
        if v <= 0:
            raise ValueError(f"Training event choices must be positive, got {v}")
        if v > t.MAX_EVENTS:
            raise ValueError(
                f"Training event choice {v} cannot exceed COMPTON_MAX_EVENTS={t.MAX_EVENTS}"
            )
    return vals


TRAIN_EVENT_CHOICES = parse_train_event_choices(TRAIN_EVENT_CHOICES_RAW)


def build_variable_source_tensors(
    df: pd.DataFrame,
    max_events: int,
    train_event_choices: List[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    groups = list(df.groupby("source_id"))
    num_sources = len(groups)
    feat_dim = len(t.FEATURE_COLS)
    X = np.zeros((num_sources, max_events, feat_dim), dtype=np.float32)
    mask = np.zeros((num_sources, max_events), dtype=bool)
    y_heat = np.zeros((num_sources, t.HEATMAP_H * t.HEATMAP_W), dtype=np.float32)
    source_ids: List[str] = []
    source_xy = np.zeros((num_sources, 2), dtype=np.float32)

    choice_rng = np.random.default_rng(t.RANDOM_SEED)

    for i, (sid, g) in enumerate(groups):
        feats = g[t.FEATURE_COLS].to_numpy(dtype=np.float32)
        n_available = feats.shape[0]
        events_for_source = min(int(choice_rng.choice(train_event_choices)), n_available)
        n_use = min(events_for_source, max_events)
        X[i, :n_use, :] = feats[:n_use]
        mask[i, :n_use] = True

        if n_use < max_events and n_use > 0:
            pad_count = max_events - n_use
            if t.PADDING_MODE == "mean":
                X[i, n_use:, :] = feats[:n_use].mean(axis=0, keepdims=True)
            elif t.PADDING_MODE == "repeat":
                repeats = int(np.ceil(pad_count / n_use))
                X[i, n_use:, :] = np.tile(feats[:n_use], (repeats, 1))[:pad_count]
            elif t.PADDING_MODE == "sample":
                rng = np.random.default_rng(t.RANDOM_SEED + i)
                sample_idx = rng.choice(n_use, size=pad_count, replace=True)
                X[i, n_use:, :] = feats[sample_idx]
            if not t.MASK_PADDED_EVENTS:
                mask[i, n_use:] = True

        xy = g[t.TARGET_COLS].iloc[0].to_numpy(dtype=np.float32)
        source_xy[i] = xy
        y_heat[i] = t.coord_to_heatmap_flat(float(xy[0]), float(xy[1]))
        source_ids.append(str(sid))

    return X, mask, y_heat, source_ids, source_xy


def write_variable_run_summary(
    history_dict: Dict[str, List[float]],
    metrics_expectation: Dict[str, float],
    metrics_map: Dict[str, float],
    predictions_csv: str,
):
    best_epoch = t.best_epoch_from_history(history_dict)
    best_val_loss = min(history_dict["val_loss"]) if history_dict.get("val_loss") else None
    lines = [
        "Compton Variable-Event Training Run Summary",
        "===========================================",
        f"CSV_PATH: {t.CSV_PATH}",
        f"SAVE_DIR: {t.SAVE_DIR}",
        f"MAX_EVENTS: {t.MAX_EVENTS}",
        f"TRAIN_EVENT_CHOICES: {','.join(str(x) for x in TRAIN_EVENT_CHOICES)}",
        f"VAL_EVENTS_PER_SOURCE: {VAL_EVENTS_PER_SOURCE}",
        f"TEST_EVENTS_PER_SOURCE: {TEST_EVENTS_PER_SOURCE}",
        f"PADDING_MODE: {t.PADDING_MODE}",
        f"MASK_PADDED_EVENTS: {t.MASK_PADDED_EVENTS}",
        f"EPOCHS_REQUESTED: {t.EPOCHS}",
        f"EPOCHS_RUN: {len(history_dict.get('loss', []))}",
        f"BEST_EPOCH: {best_epoch if best_epoch is not None else 'N/A'}",
        f"BEST_VAL_LOSS: {best_val_loss:.6f}" if best_val_loss is not None else "BEST_VAL_LOSS: N/A",
        f"FINAL_TRAIN_LOSS: {history_dict['loss'][-1]:.6f}" if history_dict.get("loss") else "FINAL_TRAIN_LOSS: N/A",
        f"FINAL_VAL_LOSS: {history_dict['val_loss'][-1]:.6f}" if history_dict.get("val_loss") else "FINAL_VAL_LOSS: N/A",
        "",
        f"Expectation Metrics [{metrics_expectation['label']}]",
        f"MAE_x: {metrics_expectation['MAE_x']:.4f}",
        f"MAE_y: {metrics_expectation['MAE_y']:.4f}",
        f"MAE_euclid: {metrics_expectation['MAE_euclid']:.4f}",
        f"R2_x: {metrics_expectation['R2_x']:.4f}",
        f"R2_y: {metrics_expectation['R2_y']:.4f}",
        f"R2_joint_xy: {metrics_expectation['R2_joint_xy']:.4f}",
        "",
        f"MAP Metrics [{metrics_map['label']}]",
        f"MAE_x: {metrics_map['MAE_x']:.4f}",
        f"MAE_y: {metrics_map['MAE_y']:.4f}",
        f"MAE_euclid: {metrics_map['MAE_euclid']:.4f}",
        f"R2_x: {metrics_map['R2_x']:.4f}",
        f"R2_y: {metrics_map['R2_y']:.4f}",
        f"R2_joint_xy: {metrics_map['R2_joint_xy']:.4f}",
        "",
        "Artifacts",
        f"history.json: {os.path.join(t.SAVE_DIR, 'history.json')}",
        f"meta.json: {os.path.join(t.SAVE_DIR, 'meta.json')}",
        f"training_curves.png: {os.path.join(t.SAVE_DIR, 'training_curves.png')}",
        f"best.weights.h5: {t.BEST_WEIGHTS}",
        f"final.weights.h5: {t.FINAL_WEIGHTS}",
        f"predictions_test_map.csv: {predictions_csv}",
        f"histograms_from_maps_test.png: {os.path.join(t.SAVE_DIR, 'histograms_from_maps_test.png')}",
        f"histograms_from_maps_test_zoom.png: {os.path.join(t.SAVE_DIR, 'histograms_from_maps_test_zoom.png')}",
        f"histograms_from_xy_test_map.png: {os.path.join(t.SAVE_DIR, 'histograms_from_xy_test_map.png')}",
        f"histograms_from_xy_test_map_zoom.png: {os.path.join(t.SAVE_DIR, 'histograms_from_xy_test_map_zoom.png')}",
        f"training_log.txt: {t.TRAINING_LOG}",
    ]
    with open(t.RUN_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved run summary -> {t.RUN_SUMMARY}")


def main():
    t.setup_log_capture()
    try:
        print(f"Variable training event choices -> {TRAIN_EVENT_CHOICES}")
        print(f"Validation events per source -> {VAL_EVENTS_PER_SOURCE}")
        print(f"Test events per source -> {TEST_EVENTS_PER_SOURCE}")

        df = pd.read_csv(t.CSV_PATH)
        missing = [c for c in t.RAW_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        if t.CONVERT_THETA_TO_RADIANS:
            df["Theta"] = np.deg2rad(df["Theta"].astype(float))

        df["source_id"] = (
            df["Source_X"].astype(str) + "_" +
            df["Source_Y"].astype(str) + "_" +
            df["Source_Z"].astype(str)
        )

        df_train, df_val, df_test = t.source_group_split(df, t.TRAIN_FRAC, t.VAL_FRAC, t.TEST_FRAC)
        t.compute_grid_from_train(df_train)

        X_train, M_train, y_train, sid_train, xy_train = build_variable_source_tensors(
            df_train, t.MAX_EVENTS, TRAIN_EVENT_CHOICES
        )
        X_val, M_val, y_val, sid_val, xy_val = t.build_source_tensors(
            df_val, t.MAX_EVENTS, VAL_EVENTS_PER_SOURCE
        )
        X_test, M_test, y_test, sid_test, xy_test = t.build_source_tensors(
            df_test, t.MAX_EVENTS, TEST_EVENTS_PER_SOURCE
        )

        model = t.build_set_model(max_events=t.MAX_EVENTS, feat_dim=len(t.FEATURE_COLS))
        t.fit_normalizer(model, X_train)

        train_ds = t.make_tf_dataset(X_train, M_train, y_train, training=True)
        val_ds = t.make_tf_dataset(X_val, M_val, y_val, training=False)

        callbacks = [
            t.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=t.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            ),
            t.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=t.REDUCE_LR_PATIENCE,
                min_lr=1e-6,
            ),
            t.keras.callbacks.ModelCheckpoint(
                filepath=t.BEST_WEIGHTS,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            t.LearningRateLogger(),
        ]
        try:
            swa = t.keras.callbacks.StochasticWeightAveraging(start_epoch=max(5, t.EPOCHS // 3))
            callbacks.append(swa)
        except Exception:
            pass

        history = model.fit(train_ds, validation_data=val_ds, epochs=t.EPOCHS, verbose=2, callbacks=callbacks)

        with open(os.path.join(t.SAVE_DIR, "history.json"), "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)
        t.plot_training_history(history, tag="training_curves")

        if os.path.exists(t.BEST_WEIGHTS):
            model.load_weights(t.BEST_WEIGHTS)

        if t.MC_DROPOUT_PASSES > 0:
            print(f"MC Dropout: {t.MC_DROPOUT_PASSES} passes ...")
            p_test = t.predict_mc_dropout(
                model, X_test, M_test, passes=t.MC_DROPOUT_PASSES, batch_size=t.BATCH_SIZE
            )
        else:
            p_test = model.predict({"events": X_test, "mask": M_test}, batch_size=t.BATCH_SIZE, verbose=0)

        metrics_expectation = t.print_source_metrics_from_maps(
            p_test, xy_test, label=f"variable_train_model (E[x,y], test_events={TEST_EVENTS_PER_SOURCE})"
        )
        t.plot_gt_vs_pred_side_by_side(sid_test, xy_test, p_test, tag="test", limit=None)
        t.plot_histograms_from_maps(
            xy_test, p_test, tag="test",
            xlim_max=50, euclid_max=70.71, bins=t.N_BINS_HIST, bar_width=0.9
        )

        xy_test_map = t.maps_to_xy(p_test, subpixel=True)
        metrics_map = t.print_source_metrics_xy(
            xy_test_map, xy_test, label=f"variable_train_model (MAP, test_events={TEST_EVENTS_PER_SOURCE})"
        )
        t.plot_histograms_from_xy(
            xy_test, xy_test_map, tag="test_map",
            xlim_max=50, euclid_max=70.71, bins=t.N_BINS_HIST, bar_width=0.9
        )

        model.save_weights(t.FINAL_WEIGHTS)

        meta = {
            "feature_cols": t.FEATURE_COLS,
            "target_cols": t.TARGET_COLS,
            "convert_theta_to_radians": t.CONVERT_THETA_TO_RADIANS,
            "set_model": True,
            "event_hidden": t.EVENT_HIDDEN,
            "attention_pooling": t.USE_ATTENTION_POOLING,
            "attn_hidden": t.ATTN_HIDDEN,
            "layer_norm": t.USE_LAYERNORM,
            "gelu": t.USE_GELU,
            "mc_dropout_passes": t.MC_DROPOUT_PASSES,
            "max_events": t.MAX_EVENTS,
            "train_event_choices": TRAIN_EVENT_CHOICES,
            "val_events_per_source": VAL_EVENTS_PER_SOURCE,
            "test_events_per_source": TEST_EVENTS_PER_SOURCE,
            "padding_mode": t.PADDING_MODE,
            "mask_padded_events": t.MASK_PADDED_EVENTS,
            "isab": {"num_heads": t.NUM_SAB_HEADS, "m": t.ISAB_NUM_INDUCING},
            "heatmap": {
                "H": t.HEATMAP_H,
                "W": t.HEATMAP_W,
                "gauss_sigma_px": t.GAUSS_SIGMA_PX,
                "grid": {
                    "x_min": float(t.GRID_X_MIN),
                    "x_max": float(t.GRID_X_MAX),
                    "y_min": float(t.GRID_Y_MIN),
                    "y_max": float(t.GRID_Y_MAX),
                },
            },
        }
        with open(os.path.join(t.SAVE_DIR, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Artifacts saved in: {t.SAVE_DIR}")

        csv_path = os.path.join(t.SAVE_DIR, "predictions_test_map.csv")
        exy = np.vstack([t.expected_xy_from_map(p) for p in p_test])
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source_id", "true_x", "true_y", "ex", "ey", "map_x", "map_y"])
            for sid, (tx, ty), (mx, my), (ex, ey) in zip(sid_test, xy_test, xy_test_map, exy):
                w.writerow([sid, float(tx), float(ty), float(ex), float(ey), float(mx), float(my)])
        print(f"Saved predictions CSV -> {csv_path}")

        write_variable_run_summary(history.history, metrics_expectation, metrics_map, csv_path)
        print("Done")
    finally:
        t.restore_log_capture()


if __name__ == "__main__":
    main()
