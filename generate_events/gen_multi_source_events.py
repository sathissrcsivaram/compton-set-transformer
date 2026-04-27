import argparse
import math
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import params


REPO_ROOT = Path(__file__).resolve().parents[1]
MERGED_OUTPUT_PATH = REPO_ROOT / "eventset_multi_source.csv"
SEPARATE_OUTPUT_DIR = REPO_ROOT / "data" / "multiSourceData"

GRID_X = 50
GRID_Y = 50
DEFAULT_SOURCE_Z = -123

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
        scatter_x = round(random.uniform(0, 50), 0)
        scatter_y = round(random.uniform(0, 50), 0)
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
    events_per_source: int,
    source_z: int = DEFAULT_SOURCE_Z,
) -> pd.DataFrame:
    rows: List[List[float]] = []

    for source_instance_id, (source_x, source_y) in enumerate(source_positions):
        source_events = simulate_events_for_source(
            source_x=source_x,
            source_y=source_y,
            source_z=source_z,
            num_events=events_per_source,
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
    sources_per_image: int,
    events_per_source: int,
    min_source_distance: float,
    source_z: int = DEFAULT_SOURCE_Z,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for image_id in range(num_images):
        positions = sample_source_positions(
            num_sources=sources_per_image,
            min_distance=min_source_distance,
        )
        frames.append(
            build_image_dataframe(
                image_id=image_id,
                source_positions=positions,
                events_per_source=events_per_source,
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
        default=5,
        help="Number of distinct sources in each image.",
    )
    parser.add_argument(
        "--events-per-source",
        type=int,
        default=100,
        help="Number of events generated from each source in one image.",
    )
    parser.add_argument(
        "--min-source-distance",
        type=float,
        default=10.0,
        help="Minimum Euclidean distance between source centers in grid units.",
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

    dataset = generate_dataset(
        num_images=args.num_images,
        sources_per_image=args.sources_per_image,
        events_per_source=args.events_per_source,
        min_source_distance=args.min_source_distance,
        source_z=args.source_z,
    )

    if args.mode == "separate":
        write_separate_files(dataset, SEPARATE_OUTPUT_DIR)
        print("Generated separate multi-source CSV files:", args.num_images)
        print("Output directory:", SEPARATE_OUTPUT_DIR)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)
        print("Generated merged multi-source dataset")
        print("Images:", args.num_images)
        print("Sources per image:", args.sources_per_image)
        print("Events per source:", args.events_per_source)
        print("Total rows:", len(dataset))
        print("Output:", output_path)


if __name__ == "__main__":
    main()
