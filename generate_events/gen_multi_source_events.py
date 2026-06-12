import argparse
import math
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import params


REPO_ROOT = Path(__file__).resolve().parents[1]
MERGED_OUTPUT_PATH = REPO_ROOT / "eventset_multi_source.csv"
SEPARATE_OUTPUT_DIR = REPO_ROOT / "data" / "multiSourceData"

GRID_X = 50
GRID_Y = 50
HEATMAP_W = 50
HEATMAP_H = 50
GRID_PADDING_FRACTION = 0.05
DEFAULT_SOURCE_Z = -123
# Derived from the current single-source heatmap target.
# GAUSS_SIGMA_PX = 1.5 heatmap pixels.
# One heatmap pixel is approximately 1.1 mm, so:
# sigma ~= 1.5 * 1.1 = 1.65 mm.
# Using a practical outer blur radius of 3*sigma:
# radius ~= 3 * 1.65 = 4.95 mm.
# Therefore:
# min_dist ~= 2 * 4.95 + 1 = 10.9 mm,
# which is rounded up to 11 mm.
GAUSS_SIGMA_PX = 1.5
PRACTICAL_BLUR_SIGMA_MULTIPLIER = 3.0
DELTA_MM = 1.0


def compute_grid_spacing_mm(grid_size: int, heatmap_size: int, padding_fraction: float) -> float:
    coord_min = 0.0
    coord_max = float(grid_size - 1)
    coord_range = coord_max - coord_min
    pad = padding_fraction * coord_range
    padded_width = (coord_max + pad) - (coord_min - pad)
    return padded_width / float(heatmap_size - 1)


GRID_SPACING_X_MM = compute_grid_spacing_mm(GRID_X, HEATMAP_W, GRID_PADDING_FRACTION)
GRID_SPACING_Y_MM = compute_grid_spacing_mm(GRID_Y, HEATMAP_H, GRID_PADDING_FRACTION)
SIGMA_X_MM = GAUSS_SIGMA_PX * GRID_SPACING_X_MM
SIGMA_Y_MM = GAUSS_SIGMA_PX * GRID_SPACING_Y_MM
BLUR_RADIUS_X_MM = PRACTICAL_BLUR_SIGMA_MULTIPLIER * SIGMA_X_MM
BLUR_RADIUS_Y_MM = PRACTICAL_BLUR_SIGMA_MULTIPLIER * SIGMA_Y_MM
BLUR_RADIUS_MM = max(BLUR_RADIUS_X_MM, BLUR_RADIUS_Y_MM)
MIN_SOURCE_DISTANCE_MM = (2.0 * BLUR_RADIUS_MM) + DELTA_MM
# Use ceil so the enforced grid spacing is never smaller than the derived minimum.
DEFAULT_MIN_SOURCE_DISTANCE = float(math.ceil(MIN_SOURCE_DISTANCE_MM))
DEFAULT_RANDOM_SEED = 42

CSV_COLUMNS = [
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


def sample_num_sources_for_image(
    fixed_sources_per_image: Optional[int],
    min_sources_per_image: int,
    max_sources_per_image: int,
) -> int:
    if fixed_sources_per_image is not None:
        return int(fixed_sources_per_image)
    return int(random.randint(min_sources_per_image, max_sources_per_image))


def allocate_events_across_sources(
    num_sources: int,
    events_per_source: Optional[int],
    total_events_per_image: Optional[int],
    random_events_per_source: bool = False,
    min_events_per_source: int = 10,
    max_events_per_source: int = 100,
    events_per_source_step: int = 10,
) -> List[int]:
    if num_sources < 1:
        raise ValueError("num_sources must be at least 1.")

    if total_events_per_image is not None:
        if total_events_per_image < num_sources:
            raise ValueError(
                "total_events_per_image must be at least as large as num_sources "
                "so every source can contribute at least one event."
            )
        base = total_events_per_image // num_sources
        remainder = total_events_per_image % num_sources
        counts = [base] * num_sources
        for i in range(remainder):
            counts[i] += 1
        return counts

    if random_events_per_source:
        choices = list(range(min_events_per_source, max_events_per_source + 1, events_per_source_step))
        return [int(random.choice(choices)) for _ in range(num_sources)]

    if events_per_source is None:
        raise ValueError("Either events_per_source or total_events_per_image must be provided.")
    return [int(events_per_source)] * num_sources


def sample_source_positions(
    num_sources: int,
    min_distance: float,
    grid_x: int = GRID_X,
    grid_y: int = GRID_Y,
    max_attempts: int = 10_000,
) -> List[Tuple[int, int]]:
    positions: List[Tuple[int, int]] = []
    attempts = 0

    while len(positions) < num_sources:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Could not place {num_sources} sources with min distance {min_distance} "
                f"on a {grid_x}x{grid_y} grid after {max_attempts} attempts."
            )

        x = random.randint(0, grid_x - 1)
        y = random.randint(0, grid_y - 1)

        if all(math.dist((x, y), prev) >= min_distance for prev in positions):
            positions.append((x, y))

        attempts += 1

    return positions


def simulate_events_for_source(
    source_x: int,
    source_y: int,
    source_z: int,
    num_events: int,
) -> np.ndarray:
    events = np.zeros((num_events, 8), dtype=np.float32)

    for i in range(num_events):
        scatter_x = round(random.uniform(0, 49), 0)
        scatter_y = round(random.uniform(0, 49), 0)
        scatter_z = 0.0

        absorb_x = round(random.uniform(0, 45), 0)
        absorb_y = round(random.uniform(0, 45), 0)
        absorb_z = 100.0

        axis_vector = np.array(
            [scatter_x - absorb_x, scatter_y - absorb_y, scatter_z - absorb_z],
            dtype=np.float64,
        )
        source_vector = np.array(
            [source_x - scatter_x, source_y - scatter_y, source_z - scatter_z],
            dtype=np.float64,
        )

        cos_theta = np.dot(axis_vector, source_vector) / (
            np.linalg.norm(axis_vector, ord=2) * np.linalg.norm(source_vector, ord=2)
        )
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        theta = math.degrees(math.acos(cos_theta))

        energy = params.initialEnergy_eV * (
            1
            - (
                1
                / (
                    (
                        (params.initialEnergy_eV / params.eMass_eV)
                        * (1 - np.cos(theta * math.pi / 180.0))
                    )
                    + 1
                )
            )
        )

        events[i] = np.array(
            [scatter_x, scatter_y, scatter_z, absorb_x, absorb_y, absorb_z, theta, energy],
            dtype=np.float32,
        )

    return events


def build_image_dataframe(
    image_id: int,
    source_positions: Sequence[Tuple[int, int]],
    source_event_counts: Sequence[int],
    source_z: int = DEFAULT_SOURCE_Z,
) -> pd.DataFrame:
    rows: List[List[float]] = []

    if len(source_positions) != len(source_event_counts):
        raise ValueError("source_positions and source_event_counts must have the same length.")

    for source_instance_id, ((source_x, source_y), num_events) in enumerate(zip(source_positions, source_event_counts)):
        source_events = simulate_events_for_source(
            source_x=source_x,
            source_y=source_y,
            source_z=source_z,
            num_events=int(num_events),
        )

        for event in source_events:
            rows.append(
                [
                    image_id,
                    source_instance_id,
                    source_x,
                    source_y,
                    source_z,
                    event[0],
                    event[1],
                    event[2],
                    event[3],
                    event[4],
                    event[5],
                    event[6],
                    event[7],
                ]
            )

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)

    # Mix source events within the same image so the final event set is combined.
    return df.sample(frac=1.0, random_state=image_id).reset_index(drop=True)


def generate_dataset(
    num_images: int,
    sources_per_image: Optional[int],
    events_per_source: Optional[int],
    min_source_distance: float,
    min_sources_per_image: int = 1,
    max_sources_per_image: int = 5,
    total_events_per_image: Optional[int] = None,
    random_events_per_source: bool = False,
    min_events_per_source: int = 10,
    max_events_per_source: int = 100,
    events_per_source_step: int = 10,
    source_z: int = DEFAULT_SOURCE_Z,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for image_id in range(num_images):
        num_sources = sample_num_sources_for_image(
            fixed_sources_per_image=sources_per_image,
            min_sources_per_image=min_sources_per_image,
            max_sources_per_image=max_sources_per_image,
        )
        positions = sample_source_positions(
            num_sources=num_sources,
            min_distance=min_source_distance,
        )
        source_event_counts = allocate_events_across_sources(
            num_sources=num_sources,
            events_per_source=events_per_source,
            total_events_per_image=total_events_per_image,
            random_events_per_source=random_events_per_source,
            min_events_per_source=min_events_per_source,
            max_events_per_source=max_events_per_source,
            events_per_source_step=events_per_source_step,
        )
        frames.append(
            build_image_dataframe(
                image_id=image_id,
                source_positions=positions,
                source_event_counts=source_event_counts,
                source_z=source_z,
            )
        )

    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multi-source Compton event data with per-image source spacing constraints."
    )
    parser.add_argument(
        "--mode",
        choices=["merged", "separate"],
        default="merged",
        help="Output mode. 'merged' writes one CSV, 'separate' writes one CSV per image.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=2500,
        help="Number of multi-source images/samples to generate.",
    )
    parser.add_argument(
        "--sources-per-image",
        type=int,
        default=None,
        help="Fixed number of distinct sources in each image. If omitted, source count is sampled per image from --min-sources-per-image to --max-sources-per-image.",
    )
    parser.add_argument(
        "--min-sources-per-image",
        type=int,
        default=1,
        help="Minimum number of distinct sources per image in variable source-count mode.",
    )
    parser.add_argument(
        "--max-sources-per-image",
        type=int,
        default=5,
        help="Maximum number of distinct sources per image in variable source-count mode.",
    )
    parser.add_argument(
        "--events-per-source",
        type=int,
        default=100,
        help="Number of events generated from each source. Ignored if --total-events-per-image is provided.",
    )
    parser.add_argument(
        "--total-events-per-image",
        type=int,
        default=None,
        help="If provided, keeps the total events per image fixed and divides them approximately equally across that image's sources.",
    )
    parser.add_argument(
        "--random-events-per-source",
        action="store_true",
        help="Sample each source's event count from --min-events-per-source to --max-events-per-source using --events-per-source-step. Ignored if --total-events-per-image is provided.",
    )
    parser.add_argument(
        "--min-events-per-source",
        type=int,
        default=10,
        help="Minimum events per source when --random-events-per-source is used.",
    )
    parser.add_argument(
        "--max-events-per-source",
        type=int,
        default=100,
        help="Maximum events per source when --random-events-per-source is used.",
    )
    parser.add_argument(
        "--events-per-source-step",
        type=int,
        default=10,
        help="Step size for random per-source event counts.",
    )
    parser.add_argument(
        "--min-source-distance",
        type=float,
        default=DEFAULT_MIN_SOURCE_DISTANCE,
        help="Minimum Euclidean distance between source centers in grid units/mm.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducible source placement and event generation.",
    )
    parser.add_argument(
        "--source-z",
        type=int,
        default=DEFAULT_SOURCE_Z,
        help="Shared z coordinate for all sources.",
    )
    parser.add_argument(
        "--output",
        default=str(MERGED_OUTPUT_PATH),
        help="Merged CSV output path used when mode='merged'.",
    )
    return parser.parse_args()


def write_separate_files(
    dataset: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_id, image_df in dataset.groupby("Image_ID", sort=True):
        image_path = output_dir / f"multi_source_image_{int(image_id):04d}.csv"
        image_df.to_csv(image_path, index=False)


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.sources_per_image is not None:
        if args.sources_per_image < 1:
            raise ValueError("--sources-per-image must be at least 1.")
        fixed_sources_per_image: Optional[int] = int(args.sources_per_image)
        min_sources_per_image = fixed_sources_per_image
        max_sources_per_image = fixed_sources_per_image
    else:
        if args.min_sources_per_image < 1:
            raise ValueError("--min-sources-per-image must be at least 1.")
        if args.max_sources_per_image < args.min_sources_per_image:
            raise ValueError("--max-sources-per-image must be >= --min-sources-per-image.")
        fixed_sources_per_image = None
        min_sources_per_image = int(args.min_sources_per_image)
        max_sources_per_image = int(args.max_sources_per_image)

    if args.total_events_per_image is not None and args.total_events_per_image < 1:
        raise ValueError("--total-events-per-image must be at least 1.")
    if args.total_events_per_image is None and args.random_events_per_source:
        if args.min_events_per_source < 1:
            raise ValueError("--min-events-per-source must be at least 1.")
        if args.max_events_per_source < args.min_events_per_source:
            raise ValueError("--max-events-per-source must be >= --min-events-per-source.")
        if args.events_per_source_step < 1:
            raise ValueError("--events-per-source-step must be at least 1.")
    if args.total_events_per_image is None and not args.random_events_per_source and args.events_per_source < 1:
        raise ValueError("--events-per-source must be at least 1.")

    dataset = generate_dataset(
        num_images=args.num_images,
        sources_per_image=fixed_sources_per_image,
        events_per_source=args.events_per_source,
        min_source_distance=args.min_source_distance,
        min_sources_per_image=min_sources_per_image,
        max_sources_per_image=max_sources_per_image,
        total_events_per_image=args.total_events_per_image,
        random_events_per_source=args.random_events_per_source and args.total_events_per_image is None,
        min_events_per_source=args.min_events_per_source,
        max_events_per_source=args.max_events_per_source,
        events_per_source_step=args.events_per_source_step,
        source_z=args.source_z,
    )

    per_image_source_counts = (
        dataset[["Image_ID", "Source_Instance_ID"]]
        .drop_duplicates()
        .groupby("Image_ID")
        .size()
    )
    per_image_event_counts = dataset.groupby("Image_ID").size()

    if args.mode == "separate":
        write_separate_files(dataset, SEPARATE_OUTPUT_DIR)
        print("Generated separate multi-source CSV files:", args.num_images)
        print("Output directory:", SEPARATE_OUTPUT_DIR)
        if fixed_sources_per_image is not None:
            print("Sources per image:", fixed_sources_per_image)
        else:
            print("Sources per image:", f"variable [{min_sources_per_image}, {max_sources_per_image}]")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)
        print("Generated merged multi-source dataset")
        print("Images:", args.num_images)
        if fixed_sources_per_image is not None:
            print("Sources per image:", fixed_sources_per_image)
        else:
            print("Sources per image:", f"variable [{min_sources_per_image}, {max_sources_per_image}]")
            print(
                "Actual generated source-count range:",
                f"{int(per_image_source_counts.min())}..{int(per_image_source_counts.max())}",
            )
        if args.total_events_per_image is not None:
            print("Total events per image:", args.total_events_per_image)
            print(
                "Actual generated event-count range:",
                f"{int(per_image_event_counts.min())}..{int(per_image_event_counts.max())}",
            )
        else:
            if args.random_events_per_source:
                print(
                    "Random events per source:",
                    f"{args.min_events_per_source}..{args.max_events_per_source} step {args.events_per_source_step}",
                )
            else:
                print("Events per source:", args.events_per_source)
            print(
                "Actual generated event-count range:",
                f"{int(per_image_event_counts.min())}..{int(per_image_event_counts.max())}",
            )
        print("Minimum source distance:", args.min_source_distance)
        print("Derived blur radius (mm):", round(BLUR_RADIUS_MM, 4))
        print("Random seed:", args.seed)
        print("Total rows:", len(dataset))
        print("Output:", output_path)


if __name__ == "__main__":
    main()
