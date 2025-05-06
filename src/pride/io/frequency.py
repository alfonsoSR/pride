from astropy import time
from .resources import internal_file


def load_three_way_ramping_data(
    source_file: str,
) -> dict[tuple[time.Time, time.Time], tuple[float, float, str]]:

    # Initialize output containers
    _t0: list[str] = []
    _t1: list[str] = []
    f0: list[float] = []
    df: list[float] = []
    name: list[str] = []

    # Read data from ramping file
    with internal_file(source_file).open() as f:
        for line in f:

            # Skip empty lines and comments
            if "#" in line or len(line.split()) == 0:
                continue

            # Determine time window
            content = line.strip().split()
            t0_str = "T".join(content[:2])
            t1_str = "T".join(content[2:4])

            # Filter out data after the end of the experiment

            # Read data from line
            content = line.strip().split()
            t0 = time.Time("T".join(content[:2]))
            t1 = time.Time("T".join(content[2:4]))
            f0 = float(content[4])
            df = float(content[5])
            name = content[6]

            # Add entry
            out[(t0, t1)] = (f0, df, name)

    return out


def load_one_way_ramping_data(
    source_file: str,
) -> dict[tuple[time.Time, time.Time], tuple[float, float]]:

    out = {}
    with internal_file(source_file).open() as f:
        for line in f:

            # Skip empty lines and comments
            if "#" in line or len(line.split()) == 0:
                continue

            # Read data from line
            content = line.strip().split()
            assert len(content) == 6  # Sanity
            t0 = time.Time("T".join(content[:2]))
            t1 = time.Time("T".join(content[2:4]))
            f0 = float(content[4])
            df = float(content[5])

            # Add entry
            out[(t0, t1)] = (f0, df)

    return out
