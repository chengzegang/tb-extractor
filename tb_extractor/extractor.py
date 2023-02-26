import os

import click
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from PIL import Image
from functools import partial
from io import BytesIO
from tqdm.auto import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore # noqa: E402
    EventAccumulator,
    STORE_EVERYTHING_SIZE_GUIDANCE,
)


class TensorboardEventExtractor(object):
    def __init__(self, root: str, levels: int = 3, verbose: bool = False):
        self.root = root
        self.levels = levels
        self.verbose = verbose

    def extract(self, dest_root: str):
        os.makedirs(dest_root, exist_ok=True)
        event_dirs = self.find_event_dirs()
        self.extract_event_dirs(event_dirs, dest_root)
        if self.verbose:
            print("Done.")

    def find_event_dirs(self) -> list[str]:
        event_dirs = []
        if self.verbose:
            print("Searching for event dirs...")
        for dirpath, dirname, filenames in os.walk(self.root):
            for filename in filenames:
                if filename.startswith("events.out.tfevents"):
                    event_dirs.append(dirpath)
                    break
        if self.verbose:
            print(f"Found {len(event_dirs)} event dirs.")

        return event_dirs

    def get_levels(self, event_dir: str) -> list[str]:
        levels = event_dir.split("/")[-self.levels :]
        return levels

    def extract_event_dirs(self, event_dirs: list[str], dest_root: str):
        with ProcessPoolExecutor() as executor:
            list(
                tqdm(
                    executor.map(
                        self.extract_event_dir,
                        event_dirs,
                        [dest_root] * len(event_dirs),
                    ),
                    total=len(event_dirs),
                    desc="Extracting",
                )
            )

    def extract_event_dir(self, event_dir: str, dest_root: str):
        levels = self.get_levels(event_dir)
        dest_dir = os.path.join(dest_root, *levels)
        os.makedirs(dest_dir, exist_ok=True)

        accumulator = EventAccumulator(
            event_dir, STORE_EVERYTHING_SIZE_GUIDANCE
        ).Reload()
        imgs = self.get_images(accumulator)
        self.save_images(imgs, dest_dir)
        scalars = self.get_scalars(accumulator)
        self.save_scalars(scalars, dest_dir)

    def decode_img_string(self, img_string: bytes):
        img = Image.open(BytesIO(img_string))
        return img

    def get_images(self, summary: EventAccumulator) -> dict:
        tags = summary.Tags()["images"]
        results = defaultdict(partial(defaultdict, list))  # type: ignore
        for tag in tags:
            events = summary.Images(tag)
            results[tag]["img_string"] = [e.encoded_image_string for e in events]
            results[tag]["steps"] = [e.step for e in events]
            results[tag]["wall_times"] = [e.wall_time for e in events]
        return results

    def save_images(self, img_records: dict, dest_dir: str):
        for tag, record in img_records.items():
            tag_dir = os.path.join(dest_dir, tag)
            os.makedirs(tag_dir, exist_ok=True)

            for i, (img_string, step, wall_time) in enumerate(
                zip(record["img_string"], record["steps"], record["wall_times"])
            ):
                img = self.decode_img_string(img_string)
                img_fp = os.path.join(tag_dir, f"{step}_{wall_time}.jpg")
                img.save(img_fp)
            del record["img_string"]

    def get_scalars(self, summary: EventAccumulator) -> dict:
        tags = summary.Tags()["scalars"]
        results = defaultdict(partial(defaultdict, list))  # type: ignore
        for tag in tags:
            events = summary.Scalars(tag)
            results[tag]["values"] = [e.value for e in events]
            results[tag]["steps"] = [e.step for e in events]
            results[tag]["wall_times"] = [e.wall_time for e in events]
        return results

    def save_scalars(self, scalar_records: dict, dest_dir: str):
        for tag, record in scalar_records.items():
            tag_dir = os.path.join(dest_dir, tag)
            os.makedirs(tag_dir, exist_ok=True)
            scalar_df = pd.DataFrame.from_dict(record)
            scalar_fp = os.path.join(tag_dir, "scalars.csv")
            scalar_df.to_csv(scalar_fp)


@click.command()
@click.option(
    "--root",
    "-r",
    help="Root directory to search for event dirs.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    required=True,
)
@click.option(
    "--dest",
    "-d",
    "dest_root",
    default="data",
    help="Root directory to save extracted data.",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, readable=True, writable=True
    ),
    required=True,
)
@click.option(
    "--levels",
    "-l",
    default=3,
    help="Number of levels to use for directory structure.",
    type=int,
    show_default=True,
)
@click.option(
    "--quiet",
    "-q",
    default=False,
    help="Suppress output.",
    is_flag=True,
    show_default=True,
    type=bool,
)
def extract(root: str, dest_root: str, levels: int, quiet: bool):
    tbext = TensorboardEventExtractor(root, levels, not quiet)
    tbext.extract(dest_root)


if __name__ == "__main__":
    extract()
