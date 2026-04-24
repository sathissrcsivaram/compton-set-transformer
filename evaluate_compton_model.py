import json
import os
import csv
from pathlib import Path

import numpy as np
import pandas as pd

import train_compton_set_transformer as t


MODEL_DIR = Path(os.environ.get("COMPTON_MODEL_DIR", str(Path(t.SAVE_DIR).resolve())))
WEIGHTS_PATH = Path(os.environ.get("COMPTON_WEIGHTS_PATH", str(MODEL_DIR / "best.weights.h5")))
META_PATH = Path(os.environ.get("COMPTON_META_PATH", str(MODEL_DIR / "meta.json")))
HISTORY_PATH = Path(os.environ.get("COMPTON_HISTORY_PATH", str(MODEL_DIR / "history.json")))

EVAL_SUMMARY = os.path.join(t.SAVE_DIR, "eval_summary.txt")
EVAL_PREDICTIONS = os.path.join(t.SAVE_DIR, "eval_predictions_test_map.csv")


def _load_training_meta() -> dict:
    if not META_PATH.exists():
        raise FileNotFoundError(f"Training meta file not found: {META_PATH}")
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_history() -> dict:
    if not HISTORY_PATH.exists():
        return {}
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _best_epoch_from_history(history: dict) -> int | None:
    val_loss = history.get("val_loss")
    if not val_loss:
        return None
    return int(np.argmin(val_loss) + 1)


def write_eval_summary(
    training_meta: dict,
    training_history: dict,
    metrics_expectation: dict,
    metrics_map: dict,
    predictions_csv: str,
):
    best_epoch = _best_epoch_from_history(training_history)
    best_val_loss = min(training_history["val_loss"]) if training_history.get("val_loss") else None
    lines = [
        "Compton Set Transformer Evaluation Summary",
        "==========================================",
        f"CSV_PATH: {t.CSV_PATH}",
        f"MODEL_DIR: {MODEL_DIR}",
        f"WEIGHTS_PATH: {WEIGHTS_PATH}",
        f"EVAL_SAVE_DIR: {t.SAVE_DIR}",
        f"EVAL_MAX_EVENTS: {t.MAX_EVENTS}",
        f"EVAL_EVENTS_PER_SOURCE: {t.EVENTS_PER_SOURCE}",
        f"EVAL_PADDING_MODE: {t.PADDING_MODE}",
        f"EVAL_MASK_PADDED_EVENTS: {t.MASK_PADDED_EVENTS}",
        f"TRAIN_MAX_EVENTS: {training_meta.get('max_events')}",
        f"TRAIN_EVENTS_PER_SOURCE: {training_meta.get('events_per_source')}",
        f"TRAIN_PADDING_MODE: {training_meta.get('padding_mode')}",
        f"TRAIN_MASK_PADDED_EVENTS: {training_meta.get('mask_padded_events')}",
        f"TRAIN_BEST_EPOCH: {best_epoch if best_epoch is not None else 'N/A'}",
        f"TRAIN_BEST_VAL_LOSS: {best_val_loss:.6f}" if best_val_loss is not None else "TRAIN_BEST_VAL_LOSS: N/A",
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
        f"eval_predictions_test_map.csv: {predictions_csv}",
        f"histograms_from_maps_eval.png: {os.path.join(t.SAVE_DIR, 'histograms_from_maps_eval.png')}",
        f"histograms_from_maps_eval_zoom.png: {os.path.join(t.SAVE_DIR, 'histograms_from_maps_eval_zoom.png')}",
        f"histograms_from_xy_eval_map.png: {os.path.join(t.SAVE_DIR, 'histograms_from_xy_eval_map.png')}",
        f"histograms_from_xy_eval_map_zoom.png: {os.path.join(t.SAVE_DIR, 'histograms_from_xy_eval_map_zoom.png')}",
        f"training_log.txt: {t.TRAINING_LOG}",
    ]
    with open(EVAL_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved evaluation summary -> {EVAL_SUMMARY}")


def save_predictions_csv(source_ids, true_xy, prob_maps, map_xy):
    exy = np.vstack([t.expected_xy_from_map(p) for p in prob_maps])
    with open(EVAL_PREDICTIONS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "true_x", "true_y", "ex", "ey", "map_x", "map_y"])
        for sid, (tx, ty), (mx, my), (ex, ey) in zip(source_ids, true_xy, map_xy, exy):
            w.writerow([sid, float(tx), float(ty), float(ex), float(ey), float(mx), float(my)])
    print(f"Saved evaluation predictions CSV -> {EVAL_PREDICTIONS}")


def main():
    t.setup_log_capture()
    try:
        training_meta = _load_training_meta()
        training_history = _load_history()

        trained_max_events = int(training_meta.get("max_events", t.MAX_EVENTS))
        if t.MAX_EVENTS != trained_max_events:
            raise ValueError(
                f"Evaluation MAX_EVENTS={t.MAX_EVENTS} must match training max_events={trained_max_events} "
                f"for the saved model weights."
            )
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Trained weights file not found: {WEIGHTS_PATH}")

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

        df_train, _, df_test = t.source_group_split(df, t.TRAIN_FRAC, t.VAL_FRAC, t.TEST_FRAC)
        t.compute_grid_from_train(df_train)

        X_test, M_test, _, sid_test, xy_test = t.build_source_tensors(df_test, t.MAX_EVENTS, t.EVENTS_PER_SOURCE)

        model = t.build_set_model(max_events=t.MAX_EVENTS, feat_dim=len(t.FEATURE_COLS))
        model.load_weights(str(WEIGHTS_PATH))
        print(f"Loaded weights -> {WEIGHTS_PATH}")

        if t.MC_DROPOUT_PASSES > 0:
            print(f"MC Dropout: {t.MC_DROPOUT_PASSES} passes ...")
            p_test = t.predict_mc_dropout(
                model, X_test, M_test, passes=t.MC_DROPOUT_PASSES, batch_size=t.BATCH_SIZE
            )
        else:
            p_test = model.predict({"events": X_test, "mask": M_test}, batch_size=t.BATCH_SIZE, verbose=0)

        metrics_expectation = t.print_source_metrics_from_maps(
            p_test, xy_test, label=f"eval_heatmap_model (E[x,y], events={t.EVENTS_PER_SOURCE})"
        )
        t.plot_histograms_from_maps(
            xy_test, p_test, tag="eval",
            xlim_max=50, euclid_max=70.71, bins=t.N_BINS_HIST, bar_width=0.9
        )

        xy_test_map = t.maps_to_xy(p_test, subpixel=True)
        metrics_map = t.print_source_metrics_xy(
            xy_test_map, xy_test, label=f"eval_heatmap_model (MAP, subpixel, events={t.EVENTS_PER_SOURCE})"
        )
        t.plot_histograms_from_xy(
            xy_test, xy_test_map, tag="eval_map",
            xlim_max=50, euclid_max=70.71, bins=t.N_BINS_HIST, bar_width=0.9
        )

        save_predictions_csv(sid_test, xy_test, p_test, xy_test_map)
        write_eval_summary(training_meta, training_history, metrics_expectation, metrics_map, EVAL_PREDICTIONS)
    finally:
        t.restore_log_capture()


if __name__ == "__main__":
    main()
