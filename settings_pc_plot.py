import argparse
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from itertools import batched
from multiprocessing import Pool, cpu_count
from pathlib import Path

import hiplot
import pandas as pd
from loguru import logger

TAGS = ["-FileName", "-FocalLength", "-ShutterSpeed", "-Aperture", "-ISO"]


@dataclass
class ExifFields:
    file: str = "file"
    focal_length: str = "focal length (mm)"
    aperture: str = "aperture (f/)"
    shutter_speed: str = "shutter speed (s)"
    iso: str = "ISO"


F = ExifFields()
LOG_SCALE_FIELDS = {F.shutter_speed}


def _parse_record(cols: list[str]) -> dict | None:
    fname, fl, ss, ap, iso = (c.strip() for c in cols[: len(TAGS)])
    try:
        return {
            F.file: fname,
            F.focal_length: round(float(fl.split()[0])),
            F.shutter_speed: float(Fraction(ss)) if "/" in ss else float(ss),
            F.aperture: round(float(ap), 1),
            F.iso: int(iso),
        }
    except (ValueError, IndexError, ZeroDivisionError):
        logger.warning(
            f"Skipping {fname}: could not parse '{fl}', '{ss}', '{ap}', '{iso}'"
        )
        return None


def run_exiftool(paths: list[str]) -> list[dict]:
    """Run exiftool on a batch of files, return list of tag dicts."""
    if not paths:
        return []

    cmd = ["exiftool", "-q", "-T"] + TAGS + paths

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"exiftool error: {e.stderr.strip()}")
        return []

    records = [
        r
        for line in result.stdout.splitlines()
        if len(cols := line.split("\t")) >= len(TAGS)
        if (r := _parse_record(cols)) is not None
    ]

    assert len(records) == len(paths)
    return records


def export_hiplot(records: list[dict], hiplot_file: str) -> None:
    if not records:
        logger.warning("No records to export to HiPlot.")
        return

    exp = hiplot.Experiment.from_iterable(records)

    for col in LOG_SCALE_FIELDS:
        if col in exp.parameters_definition:
            exp.parameters_definition[col].type = hiplot.ValueType.NUMERIC_LOG

    exp.to_html(hiplot_file)
    logger.info(f"HiPlot saved to {hiplot_file}")


def collect_exif(args) -> list[dict]:
    if not args.input.is_dir():
        sys.exit(f"Error: {args.input} is not a directory")

    files = list({str(p) for p in args.input.rglob(f"*.{args.ext}")})

    if not files:
        sys.exit(f"No .{args.ext} files found in {args.input}")

    logger.info(f"Found {len(files)} files -- scanning with {args.jobs} workers...")

    batch_size = max(1, len(files) // args.jobs + 1)
    batches = [list(x) for x in batched(files, batch_size)]

    with Pool(processes=args.jobs) as pool:
        results = pool.map(run_exiftool, batches)

    records = [r for batch in results for r in batch]

    pd.DataFrame(records).to_csv(args.out_records, index=False, quoting=1)
    logger.info(
        f"Collected metadata for {len(records)} files; saved to {args.out_records}"
    )

    return records


def main(args):
    if not args.out_records.exists():
        records = collect_exif(args)
    else:
        records = pd.read_csv(args.out_records).to_dict(orient="records")
    logger.info(f"Read {len(records)=:,d}")

    export_hiplot(records, args.out_hiplot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Directory to scan")
    parser.add_argument("--ext", type=str, default="CR3")
    parser.add_argument(
        "--jobs",
        type=int,
        default=cpu_count(),
        help="Number of parallel exiftool workers",
    )
    parser.add_argument(
        "--out-records", type=Path, required=True, help="Output EXIF records CSV file"
    )
    parser.add_argument(
        "--out-hiplot", type=Path, required=True, help="Output HiPlot HTML file"
    )
    args = parser.parse_args()

    main(args)
