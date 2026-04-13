import argparse
import subprocess
import sys
from fractions import Fraction
from itertools import batched
from multiprocessing import Pool, cpu_count
from pathlib import Path

import hiplot
import pandas as pd
from loguru import logger

TAGS = ["-FileName", "-FocalLength", "-ShutterSpeed", "-Aperture", "-ISO"]


def run_exiftool(paths: list[str]) -> list[dict]:
    """Run exiftool on a batch of files, return list of tag dicts."""
    if not paths:
        return []

    cmd = ["exiftool", "-q", "-T"] + TAGS + paths

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"exiftool error: {e.stderr.strip()}", file=sys.stderr)
        return []

    records = []
    for line in result.stdout.splitlines():
        cols = line.split("\t")

        if len(cols) < len(TAGS):
            continue

        fname, fl, ss, ap, iso = (c.strip() for c in cols[: len(TAGS)])
        record = {"file": fname}  # full path from our input list

        try:
            record["focal_length_mm"] = round(float(fl.split()[0]))
        except (ValueError, IndexError):
            pass

        try:
            if "/" in ss:
                record["shutter_speed_s"] = float(Fraction(ss))
            else:
                record["shutter_speed_s"] = float(ss)
        except (ValueError, ZeroDivisionError):
            pass

        try:
            record["aperture_f"] = round(float(ap), 1)
        except ValueError:
            pass

        try:
            record["iso"] = int(iso)
        except ValueError:
            pass

        if record:
            records.append(record)

    assert len(records) == len(paths)
    return records


def shutter_label(v: float) -> str:
    if v >= 1:
        return f"{v:.0f}s"
    denom = round(1 / v)
    return f"1/{denom}"


def export_hiplot(records: list[dict], hiplot_file: str) -> None:
    if not records:
        logger.warning("No complete records to export to HiPlot.")
        return

    exp_records = []
    for r in records:
        row = {"file": r["file"]}
        row["focal length (mm)"] = r["focal_length_mm"]
        row["aperture (f/)"] = r["aperture_f"]
        v = r["shutter_speed_s"]
        row["shutter speed (s)"] = v
        row["shutter speed"] = shutter_label(v)
        row["ISO"] = r["iso"]
        exp_records.append(row)

    exp = hiplot.Experiment.from_iterable(exp_records)

    # Suggest log scale for axes with large dynamic range
    for col in ["shutter speed (s)", "ISO", "focal length (mm)"]:
        if col in exp.parameters_definition:
            exp.parameters_definition[col].type = hiplot.ValueType.NUMERIC_LOG
    exp.parameters_definition["shutter speed"].type = hiplot.ValueType.CATEGORICAL

    exp.to_html(hiplot_file)
    print(f"HiPlot saved to {hiplot_file}  --  open in any browser")


def collect_exif(args):
    if not args.input.is_dir():
        sys.exit(f"Error: {args.input} is not a args.input")

    files = list({str(p) for p in args.input.rglob(f"*.{args.ext}")})

    if not files:
        sys.exit(f"No files found in {args.input}")

    logger.info(f"Found {len(files)} files -- scanning with {args.jobs} workers...")

    batch_size = max(1, len(files) // args.jobs + 1)
    batches = [list(x) for x in batched(files, batch_size)]

    with Pool(processes=args.jobs) as pool:
        results = pool.map(run_exiftool, batches)

    records = [r for batch in results for r in batch]

    pd.DataFrame(records).to_csv(args.out_records, index=False, quoting=1)
    logger.info(f"Collected metadata for {len(records)} files; saved to {args.out_records}")

    return records


def main(args):
    if not args.out_records.exists():
        records = collect_exif(args)
    else:
        records = pd.read_csv(args.out_records).to_dict(orient="records")

    export_hiplot(records, args.out_hiplot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
        help="Directory to scan",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="CR3",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=cpu_count(),
        help="Number of parallel exiftool workers",
    )
    parser.add_argument(
        "--out-records",
        type=Path,
        required=True,
        help="Output EXIF records file.",
    )
    parser.add_argument(
        "--out-hiplot",
        type=Path,
        required=True,
        help="Output html file.",
    )
    args = parser.parse_args()

    main(args)
