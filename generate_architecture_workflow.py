from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path("/home/siv/Transformermodel")
PDF_PATH = OUT_DIR / "Transformermodel_Architecture_Workflow_Single_Page.pdf"
PNG_PATH = OUT_DIR / "Transformermodel_Architecture_Workflow_Single_Page.png"


ROWS = [
    [
        "1. Dataset Source",
        "/content/mydrive/MyDrive/Transformermodel/eventset.csv",
        "Raw CSV rows",
        "Loaded with pandas. Required columns include source, scatter, absorb, theta, and energy.",
    ],
    [
        "2. Feature Prep + Grouping",
        "Raw event rows",
        "6 features / event\nGrouped by source",
        "Uses Scatter_X, Scatter_Y, Absorb_X, Absorb_Y, Theta, Energy. Theta is converted to radians and source_id is built from source coordinates.",
    ],
    [
        "3. Train / Val / Test",
        "Grouped sources",
        "70% / 15% / 15%",
        "Split is by source_id to prevent leakage. Grid statistics are computed from train only.",
    ],
    [
        "4. Tensor Builder",
        "Per-source events",
        "events [B,3000,6]\nmask [B,3000]\ntarget [B,2500]",
        "Each source is padded or truncated to MAX_EVENTS=3000. Target is a flattened 50x50 heatmap.",
    ],
    [
        "5. Model Inputs",
        "events + mask",
        "[B,3000,6]\n[B,3000]",
        "Keras model receives event sequences and a boolean mask for valid events.",
    ],
    [
        "6. Shared Normalization",
        "events tensor",
        "[B,3000,6]",
        "KerasNormalization(axis=-1) is adapted on the full training tensor before fitting.",
    ],
    [
        "7. Event Encoder Block 1",
        "[B,3000,6]",
        "[B,3000,256]",
        "Dense -> GELU -> Dropout -> Dense -> LayerNorm with residual projection to 256 channels.",
    ],
    [
        "8. Event Encoder Block 2",
        "[B,3000,256]",
        "[B,3000,256]",
        "Second residual event block with shared per-event processing.",
    ],
    [
        "9. Event Encoder Block 3",
        "[B,3000,256]",
        "[B,3000,128]",
        "Third residual block reduces the event embedding width to 128.",
    ],
    [
        "10. ISAB Attention",
        "[B,3000,128] + mask",
        "[B,3000,128]",
        "Induced Set Attention Block with 4 heads, 128 inducing points, dropout 0.10, and learnable inducing tokens.",
    ],
    [
        "11. Attention Pooling",
        "[B,3000,128]",
        "[B,128]",
        "Masked attention scores over events create a permutation-invariant pooled source representation.",
    ],
    [
        "12. Count + Prediction Head",
        "Pooled vector + count",
        "[B,128]",
        "Normalized event count is concatenated, then Dense 128 -> Dropout -> Dense 128 -> LayerNorm.",
    ],
    [
        "13. Heatmap Output",
        "[B,128]",
        "[B,2500]",
        "Softmax over 50x50 cells. Output is a probability distribution over source location cells.",
    ],
    [
        "14. Training Setup",
        "Train/val datasets",
        "Epoch loop",
        "Adam, KL-style loss, batch size 16, max 300 epochs, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping(patience=7).",
    ],
    [
        "15. Inference + Uncertainty",
        "Best saved weights",
        "Mean heatmap",
        "MC Dropout uses 10 stochastic passes during evaluation.",
    ],
    [
        "16. Final Outputs",
        "Predicted heatmaps",
        "Metrics + files",
        "Saves best.weights.h5, final.weights.h5, training curves, histograms, side-by-side heatmaps, and predictions_test_map.csv.",
    ],
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(18, 10.5), dpi=200)
    fig.patch.set_facecolor("#111315")
    ax.set_facecolor("#111315")
    ax.axis("off")

    col_labels = ["Stage", "Input", "Output Shape", "Key Details"]
    col_widths = [0.21, 0.22, 0.16, 0.41]

    table = ax.table(
        cellText=ROWS,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1, 2.05)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1a1f26")
            cell.set_text_props(color="white", weight="bold")
            cell.set_edgecolor("#2f3640")
            cell.set_linewidth(0.8)
        else:
            cell.set_facecolor("#15181d" if row % 2 else "#14181e")
            cell.set_text_props(color="#f2f4f8")
            cell.set_edgecolor("#262c34")
            cell.set_linewidth(0.55)
            if col == 0:
                cell.set_text_props(color="#7cc4ff", weight="bold")
            elif col in (1, 2):
                cell.set_text_props(color="#d9e2ff")

    fig.text(
        0.02,
        0.975,
        "Transformermodel Architecture Workflow",
        color="white",
        fontsize=21,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.02,
        0.95,
        "Single-page workflow summary for the Compton source-localization heatmap model in train_compton_set_transformer.py",
        color="#b4bdc9",
        fontsize=10,
        ha="left",
        va="top",
    )
    fig.text(
        0.02,
        0.02,
        "Config: EVENT_HIDDEN=[256,256,128], MAX_EVENTS=3000, NUM_SAB_HEADS=4, ISAB_NUM_INDUCING=128, HEATMAP=50x50, BATCH_SIZE=16",
        color="#98a3b1",
        fontsize=9,
        ha="left",
        va="bottom",
    )

    plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.06)
    fig.savefig(PDF_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(PNG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")


if __name__ == "__main__":
    main()
