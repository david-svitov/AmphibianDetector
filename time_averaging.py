import argparse
import struct
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Averaging times for all videos")
    parser.add_argument("folder", type=str,
                        help="Parent folder for search .time files")
    args = parser.parse_args()

    folder = Path(args.folder)
    time_measurements = []
    time_files = folder.glob("*.time")
    for time_file in time_files:
        with time_file.open("rb") as file:
            elapsed_time = struct.unpack("f", file.read(4))
            time_measurements.append(elapsed_time)

    average_time = np.mean(time_measurements)
    print(time_measurements)
    print(f"Average time per frame: {average_time} sec")


if __name__ == "__main__":
    main()
